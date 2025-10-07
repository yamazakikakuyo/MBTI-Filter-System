FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
    TRANSFORMERS_CACHE=/tmp/hf HF_HOME=/tmp/hf \
    TOP_K=10 IG_STEPS=120 USE_GPU=false MODEL_CHOICE=bert-base-uncased HF_USER=yamazakikakuyo

COPY requirements.txt  .

RUN python -m pip install --upgrade pip && pip install -r requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1

COPY app.py .
COPY mbti.py .

CMD ["app.handler"]