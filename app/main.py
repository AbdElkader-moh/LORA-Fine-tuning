import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from .predictor import Predictor

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor(os.getenv("MODEL_DIR", "artifacts"))
    yield


app = FastAPI(
    title="ViT-LoRA EuroSAT classifier",
    description="Land-use classification of satellite images with a LoRA fine-tuned, "
                "INT8-quantized Vision Transformer.",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


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

    result = app.state.predictor.predict(image, top_k=top_k)
    return {"filename": file.filename, **result}
