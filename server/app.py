import uvicorn
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# 1. Initialize the FastAPI app
app = FastAPI(title="OpenEnv Support Triage Environment")

# 2. Define the Data Models
class Ticket(BaseModel):
    id: int
    issue: str
    category: str  # e.g., "Billing", "Technical", "General"

class StepRequest(BaseModel):
    ticket_id: int
    action: str  # The category assigned by the AI

# 3. Your "Database" (In-memory for the hackathon)
# This mimics the tickets the AI needs to sort
initial_tickets = [
    {"id": 1, "issue": "I was charged twice for my subscription.", "category": "Billing"},
    {"id": 2, "issue": "The app crashes when I click the 'Save' button.", "category": "Technical"},
    {"id": 3, "issue": "How do I change my profile picture?", "category": "General"},
]

current_tickets = list(initial_tickets)

# 4. API Routes (The endpoints the validator and inference script talk to)

@app.get("/")
def read_root():
    return {"status": "Running", "message": "Support Triage Environment is live!"}

@app.post("/reset")
def reset_env():
    """Resets the environment to the starting state."""
    global current_tickets
    current_tickets = list(initial_tickets)
    return {"status": "success", "message": "Environment reset."}

@app.post("/step")
def step(request: StepRequest):
    """Processes one action from the AI and returns a reward."""
    global current_tickets
    
    # Find the ticket the AI is trying to categorize
    ticket = next((t for t in current_tickets if t["id"] == request.ticket_id), None)
    
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Check if the AI's action matches the correct category
    if request.action.lower() == ticket["category"].lower():
        reward = 1.0
    else:
        reward = 0.0
        
    return {
        "reward": reward,
        "done": True,  # In this task, one step finishes the ticket
        "info": {"correct_category": ticket["category"]}
    }

# 5. THE CRITICAL ADDITION: The main() function for the Validator
def main():
    """
    The entry point that the OpenEnv validator and pyproject.toml 
    will call to start the server.
    """
    # Get the port from environment variable (standard for Hugging Face)
    port = int(os.environ.get("PORT", 7860))
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)

# 6. The "standard" Python way to allow running the file directly
if __name__ == "__main__":
    main()