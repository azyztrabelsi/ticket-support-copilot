import pandas as pd
import numpy as np
import os
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
import torch
from torch.utils.data import Dataset

class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {"f1_macro": f1, "accuracy": acc}

def train_distilbert():
    print("Loading data...")
    df = pd.read_parquet("data/processed/tickets.parquet")
    
    # Encode labels (DistilBERT needs numbers, not strings)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['intent'])
    
    X_train, X_test, y_train, y_test = train_test_split(
    df["description"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42, stratify=df["label"]
    )
    
    print("Loading DistilBERT tokenizer and model...")
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(le.classes_)
    )
    
    train_dataset = TicketDataset(X_train, y_train, tokenizer)
    test_dataset = TicketDataset(X_test, y_test, tokenizer)
    
    # MLflow tracking
    mlflow.set_experiment("Ticket_Classification")
    
    with mlflow.start_run(run_name="Advanced_DistilBERT"):
        training_args = TrainingArguments(
            output_dir='./models/distilbert_results',
            num_train_epochs=2,              # Kept low for speed
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            eval_strategy="epoch",           # Evaluate at the end of each epoch
            save_strategy="no",              # Don't save checkpoints to save space
            report_to="none"                 # We handle logging manually
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        
        print("Training DistilBERT (this will take a few minutes)...")
        trainer.train()
        
        print("Evaluating...")
        eval_results = trainer.evaluate()
        
        # Log to MLflow
        mlflow.log_param("model_type", "DistilBERT")
        mlflow.log_param("epochs", 2)
        mlflow.log_metric("f1_macro", eval_results["eval_f1_macro"])
        mlflow.log_metric("accuracy", eval_results["eval_accuracy"])
        
        print("\n--- DistilBERT Results ---")
        print(f"F1-Macro: {eval_results['eval_f1_macro']:.4f}")
        print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
        print("---------------------------")
        
        # Save model and tokenizer for the API later
        os.makedirs("models/distilbert", exist_ok=True)
        model.save_pretrained("models/distilbert")
        tokenizer.save_pretrained("models/distilbert")
        print("DistilBERT model saved to models/distilbert/")

if __name__ == "__main__":
    train_distilbert()