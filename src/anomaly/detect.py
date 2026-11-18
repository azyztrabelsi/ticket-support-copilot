import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import mlflow

def load_and_prep_data():
    print("Loading data...")
    df = pd.read_parquet("data/processed/tickets.parquet")
    incidents = pd.read_csv("data/processed/incidents.csv")
    
    # We need to group by time (hourly) and cluster
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["hour"] = df["created_at"].dt.floor("h")
    return df, incidents

def model_kmeans(df):
    print("Running K-Means + Z-Score detection...")
    # 1. We use the intent as a proxy for cluster (since we know the topics)
    # In a real system, you would cluster the text embeddings.
    clusters = df["intent"]
    
    # 2. Create a pivot table: rows = intents, columns = hours, values = counts
    pivot = df.groupby(["intent", "hour"]).size().unstack(fill_value=0)
    
    alerts = []
    # 3. Detect spikes using Z-score (threshold = 3 standard deviations)
    for intent in pivot.index:
        series = pivot.loc[intent]
        mean, std = series.mean(), series.std()
        
        if std == 0: continue
        
        threshold = mean + 3 * std
        spikes = series[series > threshold]
        
        for hour, count in spikes.items():
            alerts.append({
                "model": "KMeans_ZScore",
                "intent": intent,
                "hour": hour,
                "count": int(count),
                "threshold": float(threshold)
            })
    return pd.DataFrame(alerts)

def model_isolation_forest(df):
    print("Running Isolation Forest detection...")
    pivot = df.groupby(["intent", "hour"]).size().unstack(fill_value=0)
    
    # Transpose so rows are hours, columns are intents (features)
    X = pivot.T
    
    # Train Isolation Forest
    iso = IsolationForest(contamination=0.01, random_state=42) # Expect ~1% anomalies
    preds = iso.fit_predict(X)
    
    alerts = []
    anomaly_hours = X[preds == -1]
    
    for hour in anomaly_hours.index:
        # Find which intent had the highest count during this anomaly hour
        highest_intent = anomaly_hours.loc[hour].idxmax()
        count = anomaly_hours.loc[hour, highest_intent]
        
        alerts.append({
            "model": "IsolationForest",
            "intent": highest_intent,
            "hour": hour,
            "count": int(count),
            "threshold": None
        })
    return pd.DataFrame(alerts)

def evaluate_detection(alerts_df, ground_truth_incidents):
    """
    Compares detected alerts with the synthetic incidents we injected in Step 3.
    """
    tp, fp = 0, 0
    detection_delays = []
    
    for _, gt in ground_truth_incidents.iterrows():
        gt_day = pd.Timestamp(gt["day"]).date()
        gt_intent = gt["intent"]
        
        # Check if our model detected an alert for this intent on this day
        matches = alerts_df[
            (alerts_df["intent"] == gt_intent) & 
            (pd.to_datetime(alerts_df["hour"]).dt.date == gt_day)
        ]
        
        if len(matches) > 0:
            tp += 1
            # Calculate delay (in hours) from the first hour of the day
            first_alert = pd.to_datetime(matches["hour"].min())
            start_of_day = pd.Timestamp(gt_day)
            delay = (first_alert - start_of_day).total_seconds() / 3600
            detection_delays.append(delay)
        else:
            # Missed incident
            pass 
            
    total_alerts = len(alerts_df)
    fp = total_alerts - tp # Rough estimate of False Positives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / len(ground_truth_incidents)
    avg_delay = np.mean(detection_delays) if detection_delays else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "true_positives": tp,
        "false_positives": fp,
        "avg_detection_delay_hours": avg_delay
    }

if __name__ == "__main__":
    df, incidents = load_and_prep_data()
    
    mlflow.set_experiment("Anomaly_Detection")
    
    # --- Model 1: KMeans + Z-Score ---
    alerts_kmeans = model_kmeans(df)
    metrics_kmeans = evaluate_detection(alerts_kmeans, incidents)
    
    with mlflow.start_run(run_name="KMeans_ZScore"):
        mlflow.log_metrics({
            "precision": metrics_kmeans["precision"],
            "recall": metrics_kmeans["recall"],
            "avg_delay_hours": metrics_kmeans["avg_detection_delay_hours"]
        })
        print("\n--- KMeans + Z-Score Results ---")
        print(metrics_kmeans)
        
    # --- Model 2: Isolation Forest ---
    alerts_iforest = model_isolation_forest(df)
    metrics_iforest = evaluate_detection(alerts_iforest, incidents)
    
    with mlflow.start_run(run_name="IsolationForest"):
        mlflow.log_metrics({
            "precision": metrics_iforest["precision"],
            "recall": metrics_iforest["recall"],
            "avg_delay_hours": metrics_iforest["avg_detection_delay_hours"]
        })
        print("\n--- Isolation Forest Results ---")
        print(metrics_iforest)