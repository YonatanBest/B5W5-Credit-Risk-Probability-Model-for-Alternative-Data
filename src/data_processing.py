import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def extract_time_features(df): 
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_year'] = df['TransactionStartTime'].dt.year
    return df


def aggregate_customer_features(df): 
    agg_funcs = {
        'Amount': ['sum', 'mean', 'std', 'count'], 
        'Value': ['sum', 'mean', 'std'], 
        'transaction_hour': 'mean', 
        'transaction_day': 'mean', 
        'transaction_month': 'mean', 
        'transaction_year': 'mean', 
        'FraudResult': 'mean', 
    }
    customer_df = df.groupby('CustomerId').agg(agg_funcs)
    customer_df.columns = ['_'.join(col) for col in customer_df.columns]
    customer_df = customer_df.reset_index()
    return customer_df


def build_feature_pipeline(numeric_features, categorical_features): 
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features), 
        ('cat', categorical_transformer, categorical_features)
    ])
    return preprocessor


def process_data(raw_df): 
    df = extract_time_features(raw_df)
    customer_df = aggregate_customer_features(df)
    # Merge is_high_risk if present
    if 'is_high_risk' in df.columns:
        risk = df[['CustomerId', 'is_high_risk']].drop_duplicates()
        customer_df = customer_df.merge(risk, on='CustomerId', how='left')
    return customer_df

