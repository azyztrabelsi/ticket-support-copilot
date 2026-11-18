import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, accuracy_score

def train_baseline():
    print("Loading data...")
    df = pd.read_parquet("data/processed/tickets.parquet")
    
    # We will predict the 'intent' (the specific issue type)
    X = df["description"]
    y = df["intent"]
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Start MLflow run to track this experiment
    mlflow.set_experiment("Ticket_Classification")
    
    with mlflow.start_run(run_name="Baseline_TFIDF_LogReg"):
        print("Training Logistic Regression...")
        # class_weight='balanced' helps with our imbalanced classes
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        clf.fit(X_train_vec, y_train)
        
        # Predict
        y_pred = clf.predict(X_test_vec)
        y_proba = clf.predict_proba(X_test_vec)
        
        # Calculate metrics
        f1_macro = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)
        
        # Human-in-the-loop logic (Confidence threshold < 0.7)
        max_probas = y_proba.max(axis=1)
        needs_review = (max_probas < 0.7).sum()
        review_percentage = (needs_review / len(y_test)) * 100
        
        # Log to MLflow
        mlflow.log_param("model_type", "TF-IDF + LogReg")
        mlflow.log_param("max_features", 10000)
        mlflow.log_param("confidence_threshold", 0.7)
        
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("needs_human_review_pct", review_percentage)
        
        # Print results to terminal
        print("\n--- Baseline Results ---")
        print(f"F1-Macro: {f1_macro:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Tickets flagged for Human Review (<0.7 confidence): {review_percentage:.2f}%")
        print("------------------------")
        
        # Save the models locally so we can use them in the API later
        os.makedirs("models", exist_ok=True)
        joblib.dump(clf, "models/baseline_logreg.pkl")
        joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
        print("Models saved to models/ folder.")
        
        # Save classification report for the dossier/presentation
        report = classification_report(y_test, y_pred, output_dict=False)
        with open("models/baseline_report.txt", "w") as f:
            f.write(report)
        print("Classification report saved to models/baseline_report.txt")

if __name__ == "__main__":
    train_baseline()