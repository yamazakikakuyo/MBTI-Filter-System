FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/tmp/hf \
    HF_HOME=/tmp/hf \
    TOP_K=10 \
    IG_STEPS=120 \
    USE_GPU=false \
    MODEL_CHOICE=bert-base-uncased \
    HF_USER=yamazakikakuyo \
    PORT=8000

WORKDIR /app

# OS deps kept tiny; no compilers by default
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 1) install torch CPU wheel first (smaller, avoids CUDA)
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1

# 2) install the rest
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# 3) download tiny NLTK data
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

# 4) now copy the app code
COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD wget -qO- http://127.0.0.1:${PORT}/health || exit 1

# Pass HF_TOKEN at runtime (env), don't bake it in the image.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

