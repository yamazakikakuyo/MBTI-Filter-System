import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr

from mbti import MBTIPipeline

TOP_K = int(os.getenv("TOP_K", "10"))
IG_STEPS = int(os.getenv("IG_STEPS", "120"))
USE_GPU = os.getenv("USE_GPU", "false").lower() in {"1", "true", "yes"}
MODEL_CHOICE = os.getenv("MODEL_CHOICE", "bert-base-uncased")
HF_USER = os.getenv("HF_USER", "yamazakikakuyo")

app = FastAPI(
    title="MBTI + Integrated Gradients XAI API",
    version="1.0.0",
    description="POST text → get MBTI classification + word attributions (score & percentage) via Integrated Gradients."
)

# Optional CORS (open by default; tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictIn(BaseModel):
    text: constr(min_length=1) = Field(..., description="Raw input text to analyze.")
    n_steps: int = Field(120, ge=1, le=1024)
    top_k: int = Field(10, ge=1, le=100)
    explain: bool = Field(True, description="Whether to compute Integrated Gradients")
    hf_token: str | None = Field(None, description="Optional HF token for private models")
    hf_user: str | None = Field(None, description="Optional HF user for private models")

class PredictOut(BaseModel):
    mbti_result: Dict[str, Any]
    explanation_result: Any

try:
    mbti_obj = MBTIPipeline(model_choice=MODEL_CHOICE, user=HF_USER, use_gpu=USE_GPU)
except Exception as e:
    # If models are private/restricted and token missing, you’ll get errors here.
    raise RuntimeError(f"Failed to initialize models: {e}") from e

@app.get("/health")
def health():
    return {"status": "ok"}

def _build_response(text, n_steps, top_k, explain) -> Dict[str, Any]:
    # Classification
    mbti_result = mbti_obj.predict_mbti_only(text)

    # XAI per category
    explanation_result = []
    if explain:
        for category in mbti_obj.categories:
            explanation_result.append(
                mbti_obj.report_ig(text, category, n_steps=n_steps, top_k=top_k)
            )

    return {
        "mbti_result": mbti_result,
        "explanation_result": explanation_result
    }

@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    try:
        if not os.getenv("HF_TOKEN"):
            os.environ["HF_TOKEN"] = body.hf_token
        if body.hf_user and body.hf_user != HF_USER:
            HF_USER = body.hf_user
            mbti_obj = MBTIPipeline(model_choice=MODEL_CHOICE, user=HF_USER, use_gpu=USE_GPU)
        return _build_response(body.text)
    except Exception as e:
        # Surface a clean error message
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e