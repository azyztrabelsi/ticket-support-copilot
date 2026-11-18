# 🎫 Smart Support Ticket Triage & Copilot

An end-to-end AI-powered system for technical support platforms. This project automates ticket classification, generates grounded responses using RAG, and detects emerging incidents in real-time.

## 📖 Project Overview
Technical support platforms receive thousands of tickets daily. This system uses Natural Language Processing (NLP) and Machine Learning to augment human agents by:
1. **Classification & Routing:** Predicting ticket intent and priority (with human-in-the-loop for low confidence).
2. **RAG Response Generation:** Suggesting grounded responses based on similar historical tickets.
3. **Anomaly Detection:** Detecting abnormal spikes in specific ticket categories before they become critical incidents.

## 🛠️ Tech Stack & Tools
- **Data Versioning:** DVC
- **Experiment Tracking:** MLflow
- **Version Control:** GitHub
- **Machine Learning:** Scikit-Learn, HuggingFace Transformers (DistilBERT)
- **Vector Database:** FAISS
- **LLM API:** Groq (Llama-3.3-70B)
- **Backend:** FastAPI
- **Monitoring:** EvidentlyAI

## 📂 Project Structure
```text
ticket-triage-copilot/
├── data/                  # Raw and processed data (versioned via DVC)
├── models/                # Saved ML models (TF-IDF, DistilBERT)
├── src/
│   ├── data/              # Data ingestion and synthetic metadata generation
│   ├── classification/    # Baseline (LogReg) and DistilBERT models
│   ├── rag/               # FAISS index building and Groq LLM generation
│   ├── anomaly/           # K-Means and Isolation Forest detection
│   ├── api/               # FastAPI endpoints
│   └── monitoring/        # EvidentlyAI drift reports
├── reports/               # Generated HTML drift reports
├── requirements.txt       # Python dependencies
└── README.md