import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from data_processing import extract_time_features


def calculate_rfm(df, snapshot_date):
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - pd.to_datetime(x).max()).days,
        'TransactionId': 'count',
        'Value': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    return rfm


def assign_high_risk(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(rfm_scaled)
    rfm['Cluster'] = clusters
    # High risk: cluster with highest Recency, lowest Frequency & Monetary
    cluster_stats = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    high_risk_cluster = cluster_stats.sort_values(
        ['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True]
    ).index[0]
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    return rfm[['CustomerId', 'is_high_risk']]


def main():
    df = pd.read_csv('data/raw/data.csv')
    df = extract_time_features(df)
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm = calculate_rfm(df, snapshot_date)
    risk_labels = assign_high_risk(rfm)
    # Merge back to main data
    df = df.merge(risk_labels, on='CustomerId', how='left')
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/processed_data.csv', index=False)

if __name__ == '__main__':
    main()


