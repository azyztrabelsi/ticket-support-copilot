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
                # 1. Classification (Now returns sentiment and priority)
                class_response = requests.post("http://127.0.0.1:8000/classify/baseline", json={"text": ticket_text})
                class_data = class_response.json()
                
                # 2. RAG (Now returns grounding metrics)
                rag_response = requests.post("http://127.0.0.1:8000/suggest_response", json={"text": ticket_text})
                rag_data = rag_response.json()
                
                st.success("Analysis Complete!")

                # --- AGENT ASSIGNMENT ---
                if "recommended_agent" in class_data:
                    agent = class_data["recommended_agent"]
                    st.markdown(f"**🤖 Recommended Agent:** {agent['agent_name']} (Match: {agent['match_score']*100:.0f}%)")
                    st.caption(f"Current Workload: {agent['current_workload']}")
                    st.divider()

                # --- CLASSIFICATION METRICS ---
                col_a, col_b, col_c = st.columns(3)
                col_a.info(f"**Intent:** {class_data['predicted_intent']}")
                col_b.metric(label="Confidence", value=f"{class_data['confidence']*100:.2f}%")
                col_c.metric(label="Priority Score", value=f"{class_data.get('dynamic_priority_score', 0.0)}")
                
                # Sentiment Warning
                sentiment = class_data.get("sentiment", "Neutral")
                if sentiment == "High Frustration":
                    st.error(f"😡 **Sentiment:** {sentiment} - Customer is highly frustrated.")
                else:
                    st.success(f"🙂 **Sentiment:** {sentiment}")
                
                # Human-in-the-loop warning
                if class_data["needs_human_review"]:
                    st.warning("⚠️ Low Confidence! This ticket requires manual review.")
                
                st.divider()
                
                # --- RAG RESPONSE & GROUNDING ---
                st.markdown("### ✍️ Suggested Response")
                st.write(rag_data["suggested_response"])
                
                # Display sources and grounding
                st.markdown(f"**Sources Cited:** `{', '.join(rag_data['sources'])}`")
                grounding = rag_data.get("grounding_metrics", {})
                st.caption(f"Latency: {rag_data['latency_seconds']}s | Retrieval Distance: {grounding.get('avg_retrieval_distance', 0)}")
                
                if grounding.get("unanchored_response"):
                    st.error("🚨 **Unanchored Response:** No good historical match found. This question might need a new FAQ.")
                
                st.divider()
                
                # --- HUMAN FEEDBACK LOOP ---
                st.markdown("### 👨‍💻 Agent Feedback (Human-in-the-Loop)")
                st.write("Was the AI correct? Correct it here to improve the model.")
                
                with st.form("feedback_form"):
                    correct_intent = st.text_input("What was the correct intent?", value=class_data['predicted_intent'])
                    submit_feedback = st.form_submit_button("Submit Feedback")
                    
                    if submit_feedback:
                        fb_response = requests.post("http://127.0.0.1:8000/feedback", json={
                            "ticket_text": ticket_text,
                            "predicted_intent": class_data['predicted_intent'],
                            "correct_intent": correct_intent
                        })
                        if fb_response.status_code == 200:
                            st.success("✅ Feedback saved! This will be used for our next retraining cycle via Airflow.")
                            
            except Exception as e:
                st.error(f"Error connecting to API: {e}. Make sure the FastAPI server is running.")

# ==========================================
# --- NEW: UNIFIED HUMAN REVIEW QUEUE ---
# ==========================================
st.divider()
st.subheader("🚨 System Alerts & Human Validation Queue")

queue_col1, queue_col2 = st.columns(2)

with queue_col1:
    st.markdown("### ⚠️ Low-Confidence Tickets (Manual Routing)")
    st.info("These tickets were flagged by the AI (confidence < 70%). They must be manually routed by a human agent.")
    st.warning("Ticket TKT00000123: 'internet down' (Confidence: 45%)")
    st.warning("Ticket TKT00000456: 'where is my refund' (Confidence: 62%)")

with queue_col2:
    st.markdown("### 📝 Suggested New FAQs (Knowledge Gaps)")
    st.info("The RAG system detected repeated questions with no good match in the database.")
    st.success("Detected 14 tickets about 'SIM activation'. [Approve New FAQ]")
    st.success("Detected 8 tickets about 'Roaming charges'. [Approve New FAQ]")