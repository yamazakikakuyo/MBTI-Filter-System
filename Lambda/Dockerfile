FROM public.ecr.aws/lambda/python:3.11

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/tmp/hf HF_HOME=/tmp/hf \
    TOP_K=10 IG_STEPS=120 USE_GPU=false MODEL_CHOICE=bert-base-uncased HF_USER=yamazakikakuyo

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

COPY . .

# Lambda looks for the module.function specified here
CMD ["app.handler"]