import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr
from mangum import Mangum

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictIn(BaseModel):
    text: constr(min_length=1)
    n_steps: int = Field(IG_STEPS, ge=1, le=1024)
    top_k: int = Field(TOP_K, ge=1, le=100)
    explain: bool = True
    hf_token: str | None = None
    hf_user: str | None = None

class PredictOut(BaseModel):
    mbti_result: Dict[str, Any]
    explanation_result: Any

# --- Lazy model init ---
mbti_obj: MBTIPipeline | None = None
def get_mbti() -> MBTIPipeline:
    global mbti_obj
    if mbti_obj is None:
        # If your models are private, make sure HF_TOKEN is set on the Lambda env.
        mbti_obj = MBTIPipeline(model_choice=MODEL_CHOICE, user=HF_USER, use_gpu=USE_GPU)
    return mbti_obj

@app.get("/health")
def health():
    # Don't touch models; purely a liveness probe
    return {"status": "ok"}

def _build_response(text: str, n_steps: int, top_k: int, explain: bool) -> Dict[str, Any]:
    obj = get_mbti()
    mbti_result = obj.predict_mbti_only(text)
    explanation_result = []
    if explain:
        for category in obj.categories:
            explanation_result.append(
                obj.report_ig(text, category, n_steps=n_steps, top_k=top_k)
            )
    return {"mbti_result": mbti_result, "explanation_result": explanation_result}

@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    global HF_USER, mbti_obj
    try:
        # If token/user provided in body, set and force reinit (optional)
        if body.hf_token and not os.getenv("HF_TOKEN"):
            os.environ["HF_TOKEN"] = body.hf_token
            mbti_obj = None  # recreate with token

        if body.hf_user and body.hf_user != HF_USER:
            HF_USER = body.hf_user
            mbti_obj = None  # recreate with new owner

        return _build_response(body.text, body.n_steps, body.top_k, body.explain)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e

# --- Lambda entrypoint ---
handler = Mangum(app)