from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import torch
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from src.rag.generate import RAGGenerator

app = FastAPI(title="Ticket Triage & Copilot API")

# --- Pydantic Models for Request Bodies ---
class TicketRequest(BaseModel):
    text: str

# --- Global Variables (loaded on startup) ---
models = {}

@app.on_event("startup")
def load_models():
    print("Loading models into memory...")
    
    # 1. Load Baseline Model
    models["baseline_clf"] = joblib.load("models/baseline_logreg.pkl")
    models["tfidf"] = joblib.load("models/tfidf_vectorizer.pkl")
    
    # 2. Load DistilBERT Model
    models["distilbert_tok"] = DistilBertTokenizer.from_pretrained("models/distilbert")
    models["distilbert_model"] = DistilBertForSequenceClassification.from_pretrained("models/distilbert")
    
    # 3. Load RAG Generator (Loads FAISS + Groq connection)
    models["rag"] = RAGGenerator()
    
    print("All models loaded successfully!")

# --- Endpoints ---

@app.post("/classify/baseline")
def classify_baseline(ticket: TicketRequest):
    vec = models["tfidf"].transform([ticket.text])
    clf = models["baseline_clf"]
    pred = clf.predict(vec)[0]
    proba = clf.predict_proba(vec).max()
    
    return {
        "model": "Baseline (TF-IDF + LogReg)",
        "predicted_intent": pred,
        "confidence": float(proba),
        "needs_human_review": bool(proba < 0.7)
    }

@app.post("/classify/distilbert")
def classify_distilbert(ticket: TicketRequest):
    tok = models["distilbert_tok"]
    model = models["distilbert_model"]
    
    inputs = tok(ticket.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        
    probas = torch.softmax(logits, dim=-1)
    confidence, predicted_idx = torch.max(probas, dim=-1)
    
    # The label mapping from your training (You can save/load this properly later)
    # For now, we just return the ID.
    return {
        "model": "DistilBERT",
        "predicted_class_id": predicted_idx.item(),
        "confidence": float(confidence.item()),
        "needs_human_review": bool(confidence.item() < 0.7)
    }

@app.post("/suggest_response")
def suggest_response(ticket: TicketRequest):
    rag = models["rag"]
    result = rag.generate_response(ticket.text)
    return result

@app.get("/anomalies")
def get_anomalies():
    # Load the latest alerts (in a real app, this would query a DB)
    try:
        df = pd.read_csv("data/processed/anomalies_latest.csv")
        return df.to_dict(orient="records")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No anomaly reports found.")

# Add this at the very bottom of src/api/main.py
import csv
from datetime import datetime

class FeedbackRequest(BaseModel):
    ticket_text: str
    predicted_intent: str
    correct_intent: str

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    # In a real app, this saves to a database. We will save to a CSV.
    file_path = "data/processed/human_feedback.csv"
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "ticket_text", "predicted_intent", "correct_intent"])
        writer.writerow([datetime.now(), feedback.ticket_text, feedback.predicted_intent, feedback.correct_intent])
        
    return {"status": "success", "message": "Feedback saved. Will be used for next retraining."}    