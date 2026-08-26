from fastapi import FastAPI, HTTPException

from .schemas import PredictRequest, PredictResponse
from .predictor import predict_url

app = FastAPI(
    title="Scam Detection API",
    description="Local ML API for malicious URL classification",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "name": "Scam Detection API",
        "author": "github@anhpham-dev",
        "status": "online",
        "model": "V5 scheme-free TF-IDF + V4 domain features"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        result = predict_url(request.url)
        return result

    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}"
        )

