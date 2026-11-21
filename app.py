import streamlit as st
import requests

# Set up the page configuration
st.set_page_config(page_title="Ticket Triage Copilot", page_icon="🎫", layout="wide")

st.title("🎫 Smart Support Ticket Copilot")
st.markdown("""
This is the **Human-in-the-Loop Interface** for the Support Agents. 
Paste a customer ticket below to get AI-powered classification and a suggested response.
""")

# Create two columns: Left for input, Right for output
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Incoming Ticket")
    ticket_text = st.text_area(
        "Paste the ticket text here:", 
        height=200,
        placeholder="e.g., My internet has been down since yesterday. I need a refund!"
    )
    
    analyze_btn = st.button("🚀 Analyze Ticket", type="primary", use_container_width=True)

with col2:
    st.subheader("🧠 AI Analysis & Copilot")
    
    if analyze_btn and ticket_text:
        with st.spinner("Analyzing with DistilBERT and RAG..."):
            try:
                # 1. Call the Classification Endpoint (DistilBERT)
                # Note: We are calling the baseline for simplicity here, but you can switch to /classify/distilbert
                class_response = requests.post(
                    "http://127.0.0.1:8000/classify/baseline", 
                    json={"text": ticket_text}
                )
                class_data = class_response.json()
                
                # 2. Call the RAG Endpoint
                rag_response = requests.post(
                    "http://127.0.0.1:8000/suggest_response", 
                    json={"text": ticket_text}
                )
                rag_data = rag_response.json()
                
                # --- Display Results ---
                st.success("Analysis Complete!")
                
                # Classification Card
                st.info(f"**Predicted Intent:** {class_data['predicted_intent']}")
                st.metric(label="Model Confidence", value=f"{class_data['confidence']*100:.2f}%")
                
                # Human-in-the-loop warning
                if class_data["needs_human_review"]:
                    st.warning("⚠️ Low Confidence! This ticket requires manual review.")
                else:
                    st.success("✅ High Confidence. Auto-routing approved.")
                
                st.divider()
                
                # RAG Response
                st.markdown("### ✍️ Suggested Response")
                st.write(rag_data["suggested_response"])
                
                # Sources
                st.markdown(f"**Sources Cited:** `{', '.join(rag_data['sources'])}`")
                st.caption(f"Generated in {rag_data['latency_seconds']} seconds")
                
            except Exception as e:
                st.error(f"Error connecting to API: {e}. Make sure the FastAPI server is running.")