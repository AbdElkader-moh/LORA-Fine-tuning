FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/artifacts

WORKDIR /app

# Dependencies first so this layer is cached across code changes.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Only what inference needs: the API and the artifact loader.
COPY app/ app/
COPY vit_lora/__init__.py vit_lora/quantization.py vit_lora/

# The quantized model (output of quantize.py) is mounted here at run time:
#   docker run -p 8000:8000 -v "$(pwd)/artifacts:/app/artifacts" vit-lora-api
RUN mkdir -p artifacts && useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
