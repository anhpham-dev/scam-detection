from fastapi import FastAPI, HTTPException

from schemas import PredictRequest, PredictResponse
from predictor import predict_url

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
        "model": "TF-IDF + V3"
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
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

