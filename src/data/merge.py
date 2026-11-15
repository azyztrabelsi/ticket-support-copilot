import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from src.data.load import load_bitext
import os

def synthesize_metadata(n_rows, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    # Date range: Sept 1, 2025 to Oct 31, 2025
    start = datetime(2025, 9, 1)
    end   = datetime(2025, 10, 31)
    total_seconds = int((end - start).total_seconds())

    operators = ["MTN", "Airtel", "Glo", "9mobile"]
    priorities = ["low", "medium", "high", "critical"]
    priority_p  = [0.25, 0.45, 0.20, 0.10]
    statuses    = ["open", "in_progress", "resolved", "closed"]
    status_p    = [0.10, 0.15, 0.20, 0.55]
    channels    = ["call_center", "app", "social_media", "walk_in", "email"]
    channel_p   = [0.40, 0.25, 0.20, 0.10, 0.05]

    rows = []
    for i in range(n_rows):
        created = start + timedelta(seconds=random.randint(0, total_seconds))
        status = np.random.choice(statuses, p=status_p)
        
        if status in ("resolved", "closed"):
            res_h = round(np.random.exponential(scale=50), 1)
            res_h = min(res_h, 168) # max 1 week
            csat  = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.20, 0.30, 0.35])
        else:
            res_h, csat = None, None

        rows.append({
            "ticket_id":             f"TKT{i+1:08d}",
            "created_at":            created,
            "customer_id":           f"CUST{random.randint(1000000,9999999)}",
            "phone_number":          f"+234{random.randint(7000000000, 9099999999)}",
            "operator":              np.random.choice(operators),
            "priority":              np.random.choice(priorities, p=priority_p),
            "status":                status,
            "channel":               np.random.choice(channels, p=channel_p),
            "assigned_to":           f"AGENT{random.randint(100,999)}",
            "resolution_time_hours": res_h,
            "customer_satisfaction": csat,
        })
    return pd.DataFrame(rows)

def inject_incidents(df, n_incidents=5, seed=42):
    """Injects artificial spikes for the anomaly detection component"""
    np.random.seed(seed)
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"])

    intents = df["intent"].unique()
    incident_log = []

    for k in range(n_incidents):
        intent = np.random.choice(intents)
        day = pd.Timestamp("2025-09-15") + pd.Timedelta(days=int(np.random.uniform(0, 40)))
        
        # Grab 20-40 tickets and move them to a specific day to create a spike
        pool = df[df["intent"] == intent].sample(n=np.random.randint(20, 40), random_state=seed + k)
        for idx in pool.index:
            new_time = day + pd.Timedelta(seconds=int(np.random.uniform(0, 86400)))
            df.at[idx, "created_at"] = new_time

        incident_log.append({"incident_id": f"INC{k:02d}", "intent": intent, "day": day.date(), "size": len(pool)})

    return df, pd.DataFrame(incident_log)

if __name__ == "__main__":
    # 1. Load Data
    bitext_df = load_bitext()
    
    # 2. Generate fake metadata
    print("Generating synthetic metadata...")
    meta_df = synthesize_metadata(len(bitext_df))
    
    # 3. Merge side-by-side
    print("Merging datasets...")
    merged = pd.concat([bitext_df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)
    
    # Rename columns to match our project structure
    merged = merged.rename(columns={
        "instruction": "description",
        "category":    "issue_type",
        "response":    "expected_response"
    })
    
    # 4. Inject incidents
    print("Injecting artificial incidents for anomaly detection...")
    merged, incidents = inject_incidents(merged)
    
    # 5. Save to parquet
    os.makedirs("data/processed", exist_ok=True)
    merged.to_parquet("data/processed/tickets.parquet")
    incidents.to_csv("data/processed/incidents.csv", index=False)
    
    print(f"Success! Saved {len(merged)} rows to data/processed/tickets.parquet")
    print("Sample columns:", merged.columns.tolist())