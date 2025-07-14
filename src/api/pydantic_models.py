from pydantic import BaseModel

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

class PredictResponse(BaseModel): 
    risk_probability: float 