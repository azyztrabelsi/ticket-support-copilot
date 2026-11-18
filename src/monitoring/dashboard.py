import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def generate_dashboard():
    print("Generating Grafana-style Dashboard mockup...")
    
    # 1. Load data
    df = pd.read_parquet("data/processed/tickets.parquet")
    incidents = pd.read_csv("data/processed/incidents.csv")
    
    # 2. Create dummy metrics based on your project
    # (These represent what Grafana would pull from a database)
    metrics_data = {
        "Metric": [
            "Total Tickets Processed", "Classification F1-Macro", 
            "Human Review Rate (<0.7)", "RAG API Latency (sec)", 
            "Anomalies Detected (7 days)"
        ],
        "Value": [
            f"{len(df):,}", "99.72%", "9.43%", "1.4s", str(len(incidents))
        ]
    }
    metrics_df = pd.DataFrame(metrics_data)

    # 3. Create the dashboard layout (2x2 grid)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "📊 Ticket Volume Over Time", 
            "🚨 Anomaly Detection Alerts",
            "⏱️ RAG Generation Latency",
            "📈 Model Comparison (F1-Score)"
        ),
        specs=[
            [{"type": "xy"}, {"type": "domain"}],
            [{"type": "xy"}, {"type": "bar"}]
        ]
    )

    # --- Chart 1: Ticket Volume ---
    df["created_at"] = pd.to_datetime(df["created_at"])
    daily_counts = df.groupby(df["created_at"].dt.date).size().reset_index(name="count")
    fig.add_trace(
        go.Scatter(x=daily_counts["created_at"], y=daily_counts["count"], 
                   mode="lines", name="Tickets/Day", line=dict(color="#3498db", width=3)),
        row=1, col=1
    )

    # --- Chart 2: Anomaly Alerts (Pie Chart) ---
    fig.add_trace(
        go.Pie(labels=["Normal Traffic", "Anomalies Detected"], 
               values=[len(df) - len(incidents), len(incidents)], 
               hole=0.5, marker=dict(colors=["#2ecc71", "#e74c3c"])),
        row=1, col=2
    )

    # --- Chart 3: RAG Latency (Simulated) ---
    # Simulating latency spikes
    fig.add_trace(
        go.Bar(x=["P50", "P90", "P99"], y=[1.2, 1.4, 2.1], 
               name="Latency (sec)", marker_color="#f39c12"),
        row=2, col=1
    )

    # --- Chart 4: Model Comparison ---
    fig.add_trace(
        go.Bar(x=["Baseline (TF-IDF)", "Advanced (DistilBERT)"], 
               y=[99.39, 99.72], name="F1-Macro", 
               text=["99.39%", "99.72%"], textposition="auto",
               marker_color=["#95a5a6", "#9b59b6"]),
        row=2, col=2
    )

    # 4. Update Layout to look like Grafana
    fig.update_layout(
        title_text="🚀 Smart Support Ticket Copilot - System Dashboard",
        title_font_size=24,
        height=800,
        showlegend=False,
        template="plotly_white"
    )
    
    # 5. Save to HTML
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/grafana_dashboard.html"
    fig.write_html(output_path)
    
    print(f"Success! Dashboard saved to {output_path}")
    print("Open this file in your browser to see the interactive dashboard.")

if __name__ == "__main__":
    generate_dashboard()