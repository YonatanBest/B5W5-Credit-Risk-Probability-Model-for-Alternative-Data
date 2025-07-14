import joblib

def predict(input_data): 
    # Load model and pipeline
    model = joblib.load('data/processed/best_model.joblib')
    pipeline = joblib.load('data/processed/feature_pipeline.joblib')
    # input_data: DataFrame with same columns as training features
    X_trans = pipeline.transform(input_data)
    return model.predict_proba(X_trans)[:, 1] 