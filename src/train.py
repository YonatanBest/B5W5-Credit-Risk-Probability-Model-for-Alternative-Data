import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import joblib
from data_processing import build_feature_pipeline, process_data

def main():
    # Load processed data
    df = pd.read_csv('data/processed/processed_data.csv')
    processed_df = process_data(df)
    # Drop CustomerId and rows with missing target
    processed_df = processed_df.dropna(subset=['is_high_risk'])
    X = processed_df.drop(['CustomerId', 'is_high_risk'], axis=1)
    y = processed_df['is_high_risk']
    # Identify feature types
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    # Build pipeline
    pipeline = build_feature_pipeline(numeric_features, categorical_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_trans = pipeline.fit_transform(X_train)
    X_test_trans = pipeline.transform(X_test)
    # Train models
    models = {
        'logreg': LogisticRegression(max_iter=1000),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    best_model = None
    best_auc = 0
    for name, model in models.items():
        model.fit(X_train_trans, y_train)
        y_pred = model.predict(X_test_trans)
        y_prob = model.predict_proba(X_test_trans)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f'[{name}] Accuracy:', accuracy_score(y_test, y_pred))
        print(f'[{name}] Precision:', precision_score(y_test, y_pred))
        print(f'[{name}] Recall:', recall_score(y_test, y_pred))
        print(f'[{name}] F1:', f1_score(y_test, y_pred))
        print(f'[{name}] ROC-AUC:', auc)
        mlflow.sklearn.log_model(model, f"model_{name}")
        if auc > best_auc:
            best_auc = auc
            best_model = model
    # Save best model and pipeline
    joblib.dump(best_model, 'data/processed/best_model.joblib')
    joblib.dump(pipeline, 'data/processed/feature_pipeline.joblib')
    print('Best model saved.')

if __name__ == '__main__':
    main()

    