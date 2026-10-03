FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/artifacts \
    MONITOR_DIR=/app/monitoring_data

WORKDIR /app

# Dependencies first so this layer is cached across code changes.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Only what inference needs: the API (with its monitoring package) and the artifact loader.
COPY app/ app/
COPY vit_lora/__init__.py vit_lora/quantization.py vit_lora/

# The quantized model (output of quantize.py) and its drift reference
# (output of build_reference.py) are mounted here at run time:
#   docker run -p 8000:8000 -v "$(pwd)/artifacts:/app/artifacts" vit-lora-api
# The prediction log lives in monitoring_data/; mount a volume there to keep it.
RUN useradd --create-home appuser && mkdir -p artifacts monitoring_data && chown appuser monitoring_data
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
