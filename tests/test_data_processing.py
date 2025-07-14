import pandas as pd
from src.data_processing import build_feature_pipeline, extract_time_features, process_data

def test_build_feature_pipeline_numeric():
    pipeline = build_feature_pipeline(['num1'], [])
    assert pipeline is not None

def test_build_feature_pipeline_categorical():
    pipeline = build_feature_pipeline([], ['cat1'])
    assert pipeline is not None

def test_extract_time_features():
    df = pd.DataFrame({'TransactionStartTime': ['2020-01-01T10:00:00Z']})
    df2 = extract_time_features(df.copy())
    assert 'transaction_hour' in df2.columns
    assert 'transaction_day' in df2.columns
    assert 'transaction_month' in df2.columns
    assert 'transaction_year' in df2.columns

def test_process_data():
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C1'],
        'TransactionStartTime': ['2020-01-01T10:00:00Z', '2020-01-02T11:00:00Z'],
        'Amount': [100, 200],
        'Value': [100, 200],
        'FraudResult': [0, 1],
        'is_high_risk': [1, 1]
    })
    processed = process_data(df)
    assert 'Amount_sum' in processed.columns
    assert 'is_high_risk' in processed.columns 