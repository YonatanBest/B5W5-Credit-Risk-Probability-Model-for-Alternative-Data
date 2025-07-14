from fastapi import FastAPI
from src.api.pydantic_models import PredictRequest, PredictResponse
import pandas as pd
import joblib

app = FastAPI()

@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    # Convert request to DataFrame
    input_df = pd.DataFrame([request.dict()])
    # Load model and pipeline
    model = joblib.load('data/processed/best_model.joblib')
    pipeline = joblib.load('data/processed/feature_pipeline.joblib')
    X_trans = pipeline.transform(input_df)
    risk_prob = model.predict_proba(X_trans)[0, 1]
    return PredictResponse(risk_probability=float(risk_prob)) 