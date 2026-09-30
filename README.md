# LoRA Fine-tuning: optimized ViT deployment

End-to-end image classification pipeline on EuroSAT (10 land-use classes) with a Vision Transformer:
LoRA fine-tuning, INT8 quantization, a FastAPI inference service, Docker, and CI with GitHub Actions.

## Pipeline

| Step | Command | Output |
|---|---|---|
| Fine-tune ViT with LoRA, keep the best epoch | `py train.py` | `checkpoints/best_lora_weights.pt` |
| Evaluate the best adapter on the test split | `py evaluate.py` | loss / accuracy / macro-F1 |
| Merge LoRA into the base weights and quantize to INT8 | `py quantize.py` | `artifacts/model_int8.pt`, `artifacts/model_meta.json`, size/accuracy report |
| Serve | `py -m uvicorn app.main:app --port 8000` | API on http://localhost:8000 |

Settings live in `configs/config.yaml` (CPU-friendly subset). `configs/config.kaggle.yaml` is the
full-dataset GPU run: `py train.py --config configs/config.kaggle.yaml`.

- **LoRA** (`vit_lora/lora.py`): rank-32 adapters on the query/value projections of the last three
  transformer blocks. Only the adapters and the classifier head train (~0.35% of the parameters).
- **Quantization** (`vit_lora/quantization.py`): the adapters are merged into the base weights, then
  every `nn.Linear` is dynamically quantized to int8, which cuts the model to roughly a quarter of its size.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `GET /labels` | The classes the model predicts |
| `POST /predict?top_k=3` | Multipart image upload, returns the label, confidence and top-k classes |

```bash
curl -X POST http://localhost:8000/predict -F "file=@image.jpg"
```

Interactive docs are at http://localhost:8000/docs.

## Docker

```bash
docker build -t vit-lora-api .
docker run -p 8000:8000 -v "$(pwd)/artifacts:/app/artifacts" vit-lora-api
```

The image holds only the inference code and its dependencies. The quantized model produced by
`quantize.py` is mounted at run time, so model binaries stay out of git and out of the image.

## Tests

```bash
py -m pip install -r requirements-dev.txt
py -m pytest tests -v
```

The tests use a tiny randomly-initialised ViT, so they need no dataset, download or GPU.

## Branches and CI

- `main`: stable, deployable code.
- `develop`: integration branch; work is merged into `main` from here.

`.github/workflows/ci.yml`:

| Job | Runs on | What it does |
|---|---|---|
| `test` | push to `main` or `develop`, PR into `main` | installs dependencies, runs pytest |
| `build` | same, after `test` passes | builds the Docker image and checks the app imports inside it |
| `deploy` | push to `main` only, after `build` passes | pushes the image to GitHub Container Registry |
