from pydantic import BaseModel
from typing import Optional

class PredictRequest(BaseModel):
    Amount_sum: float
    Amount_mean: float
    Amount_std: float
    Amount_count: int
    Value_sum: float
    Value_mean: float
    Value_std: float
    transaction_hour_mean: float
    transaction_day_mean: float
    transaction_month_mean: float
    transaction_year_mean: float
    FraudResult_mean: float
    # Add more features if you add more in data_processing.py

class PredictResponse(BaseModel):
    risk_probability: float 