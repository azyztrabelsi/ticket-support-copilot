import os
from dotenv import load_dotenv
from groq import Groq
from src.rag.retrieve import TicketRetriever
import time

# Load environment variables (the GROQ_API_KEY)
load_dotenv()

class RAGGenerator:
    def __init__(self):
        print("Loading Retriever...")
        self.retriever = TicketRetriever()
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file!")
            
        print("Connecting to Groq API (Llama-3.3-70B)...")
        self.client = Groq(api_key=api_key)

    def generate_response(self, new_ticket_text):
        # 1. Retrieve top 3 similar past tickets
        retrieved = self.retriever.retrieve(new_ticket_text, top_k=3)
        
        # 2. Build the context string
        context = ""
        for i, res in enumerate(retrieved):
            context += f"\n--- Past Ticket {i+1} ---\n"
            context += f"Problem: {res['past_description']}\n"
            context += f"Solution: {res['past_response']}\n"
            
        print("Calling LLM API to generate response...")
        start_time = time.time()
        
        # 3. Build messages for the Chat API
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful customer support agent. Use the provided context to answer the new ticket. Keep the answer professional, concise, and directly address the user's problem."
            },
            {
                "role": "user", 
                "content": f"CONTEXT:\n{context}\n\nNEW TICKET:\n{new_ticket_text}\n\nWrite the suggested response."
            }
        ]
        
        # 4. Call the Groq API
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            max_tokens=800,
            temperature=0.3
        )       
        
        # 5. Extract the text
        generated_text = response.choices[0].message.content
        
        latency = time.time() - start_time
        sources = [res['ticket_id'] for res in retrieved]
        
        # --- NEW: RAG GROUNDING METRICS & FAQ GAP DETECTION ---
        # Calculate average retrieval distance to see if the FAISS search actually found good matches
        avg_distance = sum([res['distance'] for res in retrieved]) / len(retrieved) if retrieved else 0
        # If the distance is above a threshold (e.g., 1.0), the question is likely "uncovered"
        is_unanchored = avg_distance > 1.0 
        
        faq_suggestion = None
        if is_unanchored:
            print(f"⚠️ WARNING: Uncovered Question detected! Suggesting new FAQ article for: '{new_ticket_text}'")
            faq_suggestion = f"Create new FAQ for: {new_ticket_text}"
            
        return {
            "suggested_response": generated_text.strip(),
            "sources": sources,
            "latency_seconds": round(latency, 2),
            "grounding_metrics": {
                "avg_retrieval_distance": round(avg_distance, 4),
                "unanchored_response": is_unanchored,
                "faq_suggestion": faq_suggestion
            }
        }

if __name__ == "__main__":
    generator = RAGGenerator()
    
    test_ticket = "I need to cancel my order #12345. It hasn't shipped yet."
    print(f"\nTesting Query: '{test_ticket}'")
    
    result = generator.generate_response(test_ticket)
    
    print("\n" + "="*50)
    print("GENERATED RESPONSE:")
    print("="*50)
    print(result["suggested_response"])
    print("\n" + "-"*50)
    print(f"SOURCES CITED (Ticket IDs): {result['sources']}")
    print(f"LATENCY: {result['latency_seconds']} seconds")
    print(f"GROUNDING METRICS: {result['grounding_metrics']}")