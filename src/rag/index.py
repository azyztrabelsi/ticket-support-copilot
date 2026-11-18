import pandas as pd
import numpy as np
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer

def build_faiss_index():
    print("Loading data...")
    df = pd.read_parquet("data/processed/tickets.parquet")
    
    # We will embed the ticket descriptions
    # To save time during testing, you can uncomment the next line to use only 5000 rows
    # df = df.sample(n=5000, random_state=42).reset_index(drop=True)
    
    print(f"Embedding {len(df)} tickets. This might take a few minutes...")
    # Load a fast, lightweight embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Encode all tickets (batch size 64 to avoid memory issues)
    embeddings = model.encode(
        df["description"].tolist(), 
        batch_size=64, 
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Convert to float32 (required by FAISS)
    embeddings = embeddings.astype(np.float32)
    
    print("Building FAISS index...")
    # Create a FAISS index based on L2 distance (Euclidean)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save the FAISS index to disk
    os.makedirs("data/processed", exist_ok=True)
    faiss.write_index(index, "data/processed/tickets_faiss.index")
    
    # Save the embedding model and the dataframe mapping
    # (We need the dataframe later so when FAISS returns an ID, we know which ticket it is)
    with open("data/processed/faiss_metadata.pkl", "wb") as f:
        pickle.dump({
            "ticket_ids": df["ticket_id"].tolist(),
            "descriptions": df["description"].tolist(),
            "responses": df["expected_response"].tolist()
        }, f)
        
    print("Success! FAISS index saved to data/processed/tickets_faiss.index")
    print("Metadata saved to data/processed/faiss_metadata.pkl")

if __name__ == "__main__":
    build_faiss_index()