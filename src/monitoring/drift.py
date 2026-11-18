import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
import os
import json

def generate_drift_report():
    print("Loading data...")
    df = pd.read_parquet("data/processed/tickets.parquet")
    
    split_idx = int(len(df) * 0.8)
    reference_data = df.iloc[:split_idx][["intent", "priority", "channel"]]
    current_data = df.iloc[split_idx:][["intent", "priority", "channel"]]
    
    print("Calculating Data Drift...")
    
    # 1. Initialize and run the report
    drift_report = Report([DataDriftPreset()])
    # In new Evidently versions, run() returns a Snapshot
    snapshot = drift_report.run(reference_data=reference_data, current_data=current_data)
    
    # 2. Extract the JSON dictionary from the snapshot
    # This is the most reliable way to get the data without API breaking
    report_dict = snapshot.dict()
    
    # 3. Create a clean HTML layout
    os.makedirs("reports", exist_ok=True)
    output_path = "reports/data_drift_report.html"
    
    html_content = """
    <html>
    <head>
        <title>Ticket Copilot - Data Drift Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f7f6; }
            h1 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; background: white; }
            th, td { padding: 12px; border: 1px solid #ddd; text-align: left; }
            th { background-color: #2c3e50; color: white; }
            .drift-detected { color: #e74c3c; font-weight: bold; }
            .no-drift { color: #27ae60; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>📊 EvidentlyAI - Data Drift Monitoring</h1>
        <p><strong>Reference Data:</strong> 80% of historical tickets</p>
        <p><strong>Current Data:</strong> Latest 20% of tickets</p>
        <hr>
        <h2>Drift Results</h2>
        <table>
            <tr>
                <th>Feature Column</th>
                <th>Drift Score</th>
                <th>Status</th>
            </tr>
    """
    
    # 4. Parse the metrics from the dictionary
    try:
        # Navigate the Evidently JSON structure to get the drift per column
        metrics = report_dict.get("metrics", [])
        for metric in metrics:
            if metric.get("metric") == "DataDriftTable":
                drift_by_columns = metric.get("result", {}).get("drift_by_columns", {})
                for col_name, col_data in drift_by_columns.items():
                    drift_score = round(col_data.get("drift_score", 0.0), 4)
                    is_drifted = col_data.get("drift_detected", False)
                    
                    status_text = "⚠️ Drift Detected" if is_drifted else "✅ Stable"
                    status_class = "drift-detected" if is_drifted else "no-drift"
                    
                    html_content += f"""
                    <tr>
                        <td><strong>{col_name}</strong></td>
                        <td>{drift_score}</td>
                        <td class="{status_class}">{status_text}</td>
                    </tr>
                    """
    except Exception as e:
        html_content += f"<tr><td colspan='3'>Error parsing metrics: {e}</td></tr>"
        
    html_content += """
        </table>
        <br>
        <p><em>Generated automatically by the Ticket Triage & Copilot MLOps Pipeline.</em></p>
    </body>
    </html>
    """
    
    # 5. Save it
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Success! Beautiful HTML report saved to {output_path}")
    print("Open this file in your web browser to view it.")

if __name__ == "__main__":
    generate_drift_report()