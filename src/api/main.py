from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import torch
import os
import csv
import re
from datetime import datetime
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from src.rag.generate import RAGGenerator

app = FastAPI(title="Ticket Triage & Copilot API")

# --- Pydantic Models for Request Bodies ---
class TicketRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    ticket_text: str
    predicted_intent: str
    correct_intent: str

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

# --- Helper Functions for Sentiment & Prioritization ---
def get_sentiment(text):
    angry_words = ["unacceptable", "terrible", "worst", "angry", "frustrated", "ridiculous", "hate", "useless", "furious", "scam"]
    text_lower = text.lower()
    for word in angry_words:
        if word in text_lower:
            return "High Frustration"
    return "Neutral"

def calculate_priority(intent, sentiment, workload=0.5):
    urgent_intents = ["payment_issue", "cancel_order", "complaint", "contact_human_agent"]
    urgency_score = 0.8 if intent in urgent_intents else 0.4
    sentiment_score = 0.9 if sentiment == "High Frustration" else 0.2
    
    # Formula: (Urgency * 0.4) + (Sentiment * 0.4) + (Workload * 0.2)
    priority_score = (urgency_score * 0.4) + (sentiment_score * 0.4) + (workload * 0.2)
    return round(priority_score, 2)

# --- Endpoints ---

@app.post("/classify/baseline")
def classify_baseline(ticket: TicketRequest):
    vec = models["tfidf"].transform([ticket.text])
    clf = models["baseline_clf"]
    pred = clf.predict(vec)[0]
    proba = clf.predict_proba(vec).max()
    
    # New: Calculate sentiment and priority
    sentiment = get_sentiment(ticket.text)
    priority = calculate_priority(pred, sentiment)
    
    return {
        "model": "Baseline (TF-IDF + LogReg)",
        "predicted_intent": pred,
        "confidence": float(proba),
        "sentiment": sentiment,
        "dynamic_priority_score": priority,
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
    try:
        df = pd.read_csv("data/processed/anomalies_latest.csv")
        return df.to_dict(orient="records")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No anomaly reports found.")

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    file_path = "data/processed/human_feedback.csv"
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "ticket_text", "predicted_intent", "correct_intent"])
        writer.writerow([datetime.now(), feedback.ticket_text, feedback.predicted_intent, feedback.correct_intent])
        
    return {"status": "success", "message": "Feedback saved. Will be used for next retraining."}