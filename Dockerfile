# ---- Build stage (optional but keeps final image smaller)
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    # Where we will persist DB + FAISS on Railway volume
    DATA_ROOT=/data \
    FAISS_INDEX_DIR=/data/indexes \
    SQLITE_DB=/data/sidecar.db

# System deps (slim image needs compilers for some wheels; libgomp for faiss)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app
COPY . /app

# Create volume mount points (Railway will bind a persistent volume here)
RUN mkdir -p ${DATA_ROOT} ${FAISS_INDEX_DIR}

# Drop privileges
RUN useradd -m sidecar && chown -R sidecar:sidecar /app ${DATA_ROOT}
USER sidecar

# Gunicorn (Uvicorn workers) picks up $PORT from Railway
CMD gunicorn app.main:app \
  -k uvicorn.workers.UvicornWorker \
  -w 2 \
  -b 0.0.0.0:${PORT} \
  --timeout 120