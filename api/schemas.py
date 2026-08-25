from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=8192)

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    phishing_probability: float
    risk: str
    risk_level: str
    trusted_domain: bool