import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class TicketRetriever:
    def __init__(self):
        print("Loading FAISS index and metadata...")
        self.index = faiss.read_index("data/processed/tickets_faiss.index")
        
        with open("data/processed/faiss_metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
            
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query_text, top_k=3):
        # 1. Embed the new query
        query_vec = self.model.encode([query_text]).astype(np.float32)
        
        # 2. Search FAISS
        distances, indices = self.index.search(query_vec, top_k)
        
        # 3. Get results
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            dist = distances[0][i]
            results.append({
                "ticket_id": self.metadata["ticket_ids"][idx],
                "past_description": self.metadata["descriptions"][idx],
                "past_response": self.metadata["responses"][idx],
                "distance": float(dist)
            })
        return results

if __name__ == "__main__":
    retriever = TicketRetriever()
    
    # Let's test it with a fake new ticket!
    test_ticket = "I need to cancel my order. How can I do this?"
    print(f"\nTesting Query: '{test_ticket}'")
    
    results = retriever.retrieve(test_ticket, top_k=3)
    
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} (Distance: {res['distance']:.4f}) ---")
        print(f"Past Ticket: {res['past_description']}")
        print(f"Past Response: {res['past_response']}")