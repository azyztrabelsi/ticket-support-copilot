import pandas as pd

# Simulated database of support agents
AGENTS_DATA = [
    {"id": "A1", "name": "Alice", "skills": ["payment_issue", "check_refund_policy", "track_refund"], "current_workload": 2, "max_capacity": 10},
    {"id": "A2", "name": "Bob", "skills": ["cancel_order", "change_order", "place_order"], "current_workload": 8, "max_capacity": 10},
    {"id": "A3", "name": "Charlie", "skills": ["delivery_options", "delivery_period", "track_order"], "current_workload": 4, "max_capacity": 10},
    {"id": "A4", "name": "Diana", "skills": ["complaint", "review", "contact_human_agent"], "current_workload": 9, "max_capacity": 10},
    {"id": "A5", "name": "Eve", "skills": ["create_account", "delete_account", "switch_account"], "current_workload": 1, "max_capacity": 10},
]

def get_best_agent(predicted_intent):
    """
    Finds the best agent based on skill match AND available capacity.
    """
    best_agent = None
    best_score = -1

    for agent in AGENTS_DATA:
        # 1. Skill Match (1.0 if they have the skill, 0.0 if not)
        skill_match = 1.0 if predicted_intent in agent["skills"] else 0.0
        
        # 2. Capacity Score (1.0 is empty, 0.0 is full)
        available_capacity = agent["max_capacity"] - agent["current_workload"]
        capacity_score = available_capacity / agent["max_capacity"]
        
        # 3. Final Score: 70% Skill Match + 30% Capacity
        final_score = (skill_match * 0.7) + (capacity_score * 0.3)
        
        if final_score > best_score:
            best_score = final_score
            best_agent = agent

    return {
        "agent_id": best_agent["id"],
        "agent_name": best_agent["name"],
        "match_score": round(best_score, 2),
        "current_workload": f"{best_agent['current_workload']}/{best_agent['max_capacity']}"
    }