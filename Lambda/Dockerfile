FROM public.ecr.aws/lambda/python:3.11

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/tmp/hf HF_HOME=/tmp/hf \
    TOP_K=10 IG_STEPS=120 USE_GPU=false MODEL_CHOICE=bert-base-uncased HF_USER=yamazakikakuyo

# Always upgrade pip tooling first
RUN python -m pip install --upgrade pip setuptools wheel

# Install CPU-only torch from the PyTorch index (wheels provided)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1

# Pre-install binary wheels that commonly trigger source builds
# Pin NumPy to a wheel-known version; Tokenizers is a Rust ext → ensure wheel
RUN pip install --no-cache-dir --only-binary=:all: \
    numpy==1.26.4 tokenizers==0.20.3

# Now install the rest; wheel-only to prevent any build from source
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

# Your app code
COPY . .

# Lambda looks for module.function here
CMD ["app.handler"]