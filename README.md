# 🎫 Smart Support Ticket Triage & Copilot

An end-to-end AI-powered system for technical support platforms. This project automates ticket classification, generates grounded responses using RAG, detects emerging incidents, and features a full Human-in-the-Loop web interface.

## 📖 Project Overview
Technical support platforms receive thousands of tickets daily. This system uses Natural Language Processing (NLP) and Machine Learning to augment human agents by:
1. **Classification & Routing:** Predicting ticket intent and priority using DistilBERT (with a confidence threshold for human review).
2. **RAG Response Generation:** Suggesting grounded responses based on similar historical tickets using FAISS and Llama-3.
3. **Anomaly Detection:** Detecting abnormal spikes in specific ticket categories before they become critical incidents.
4. **Human-in-the-Loop UI:** A Streamlit interface where agents can view AI suggestions and submit feedback for continuous model retraining.

## 🛠️ Tech Stack & Tools
- **Data Versioning:** DVC
- **Experiment Tracking:** MLflow
- **Version Control:** GitHub
- **Machine Learning:** Scikit-Learn, HuggingFace Transformers (DistilBERT)
- **Vector Database:** FAISS
- **LLM API:** Groq (Llama-3.3-70B)
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Monitoring:** EvidentlyAI, Plotly

## 📂 Project Structure
```text
ticket-triage-copilot/
├── app.py                 # Streamlit Frontend UI
├── data/                  # Raw and processed data (versioned via DVC)
├── models/                # Saved ML models (TF-IDF, DistilBERT)
├── src/
│   ├── data/              # Data ingestion and synthetic metadata generation
│   ├── classification/    # Baseline (LogReg) and DistilBERT models
│   ├── rag/               # FAISS index building and Groq LLM generation
│   ├── anomaly/           # K-Means and Isolation Forest detection
│   ├── api/               # FastAPI endpoints (Classification, RAG, Feedback)
│   └── monitoring/        # EvidentlyAI drift and Plotly Dashboard
├── reports/               # Generated HTML dashboards and drift reports
├── dvc.yaml               # Reproducible ML pipeline
├── requirements.txt       # Python dependencies
└── README.md
```

## 🚀 How to Run the Project

### 1. Setup Environment

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Data Pipeline & Training (DVC)

You can reproduce the entire data processing, training, and monitoring pipeline with a single command:

```bash
dvc repro
```

### 3. Start the Backend (FastAPI)

Create a .env file in the root directory and add your Groq API key:

```dotenv
GROQ_API_KEY=gsk_your_key_here
```

Start the API server:

```bash
uvicorn src.api.main:app --reload
```

(API documentation available at http://127.0.0.1:8000/docs)

### 4. Start the Frontend (Streamlit)

Open a new terminal tab and run:

```bash
streamlit run app.py
```

(The UI will open at http://localhost:8501)

### 5. View MLflow Experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 📊 Results Summary

| Component | Model | Metric | Result |
|---|---|---|---|
| Classification | TF-IDF + LogReg | F1-Macro | 99.39% |
| Classification | DistilBERT | F1-Macro | 99.72% |
| RAG Generation | Groq (Llama-3.3) | Latency | ~1.4s |
| Anomaly Detection | K-Means + Z-Score | Recall | 100% (6.6h lead) |

## 👥 Authors

- Aziz Trabelsi
- Mohamed Aziz Mejri

Project Module - Academic Year 2025-2026