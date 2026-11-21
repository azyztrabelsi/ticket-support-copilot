from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Default arguments for the DAG
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'ticket_copilot_weekly_retraining',
    default_args=default_args,
    description='Weekly retraining pipeline for Ticket Triage Copilot',
    schedule_interval='@weekly', # Runs every week
    start_date=datetime(2025, 11, 1),
    catchup=False,
    tags=['mlops', 'nlp'],
) as dag:

    # Task 1: Run the DVC pipeline to prepare data and retrain models
    run_dvc_pipeline = BashOperator(
        task_id='run_dvc_pipeline',
        bash_command='cd /path/to/ticket-triage-copilot && dvc repro',
    )

    # Task 2: Re-index the FAISS vector database with new tickets
    reindex_faiss = BashOperator(
        task_id='reindex_vector_db',
        bash_command='cd /path/to/ticket-triage-copilot && python -m src.rag.index',
    )

    # Task 3: Generate the EvidentlyAI Drift Report
    check_drift = BashOperator(
        task_id='generate_drift_report',
        bash_command='cd /path/to/ticket-triage-copilot && python -m src.monitoring.drift',
    )

    # Define task dependencies (Task 1 -> Task 2 -> Task 3)
    run_dvc_pipeline >> reindex_faiss >> check_drift