import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from .monitoring import ModelMonitor, PredictionStore, ReferenceProfile
from .monitoring import metrics as m
from .predictor import Predictor

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = os.getenv("MODEL_DIR", "artifacts")
    monitor_dir = Path(os.getenv("MONITOR_DIR", "monitoring_data"))
    predictor = Predictor(model_dir)
    store = PredictionStore(monitor_dir / "predictions.db")
    app.state.predictor = predictor
    app.state.monitor = ModelMonitor(
        store, predictor.class_names, predictor.version,
        reference=ReferenceProfile.load(model_dir),
        drift_window=int(os.getenv("DRIFT_WINDOW", "200")),
        min_drift_samples=int(os.getenv("DRIFT_MIN_SAMPLES", "30")),
        performance_window=int(os.getenv("PERFORMANCE_WINDOW", "200")),
    )
    m.MODEL_INFO.info({"version": predictor.version, "api_version": app.version,
                       "classes": str(len(predictor.class_names))})
    m.register_system_collector(str(monitor_dir))
    yield
    store.close()


app = FastAPI(
    title="ViT-LoRA EuroSAT classifier",
    description="Land-use classification of satellite images with a LoRA fine-tuned, "
                "INT8-quantized Vision Transformer.",
    version="1.2.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    m.HTTP_IN_PROGRESS.inc()
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        m.HTTP_IN_PROGRESS.dec()
        # Label by route template, not raw URL, to keep metric cardinality bounded.
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        m.HTTP_REQUESTS.labels(method=request.method, path=path, status=str(status)).inc()
        m.HTTP_LATENCY.labels(path=path).observe(time.perf_counter() - start)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "version": app.version}


@app.get("/labels")
def labels():
    return {"labels": app.state.predictor.class_names}


@app.post("/predict")
def predict(file: UploadFile = File(...), top_k: int = Query(3, ge=1, le=10)):
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB.")
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    predictor = app.state.predictor
    probs, inference_ms = predictor.predict_proba(image)
    request_id = app.state.monitor.record_prediction(image, probs, inference_ms)
    return {"request_id": request_id, "filename": file.filename,
            **predictor.format_result(probs, inference_ms, top_k)}


class Feedback(BaseModel):
    request_id: str
    label: str


@app.post("/feedback")
def feedback(body: Feedback):
    """Ground truth for an earlier prediction, e.g. from a human reviewer."""
    if body.label not in app.state.predictor.class_names:
        raise HTTPException(status_code=422, detail=f"Unknown label '{body.label}'.")
    result = app.state.monitor.record_feedback(body.request_id, body.label)
    if result is None:
        raise HTTPException(status_code=404, detail="Unknown request_id.")
    return result


@app.get("/monitoring/drift")
def drift():
    return app.state.monitor.drift_report()


@app.get("/monitoring/performance")
def performance():
    return app.state.monitor.performance_report()


@app.get("/metrics", include_in_schema=False)
def metrics():
    app.state.monitor.refresh()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
