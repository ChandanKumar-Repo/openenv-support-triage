import uvicorn
import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from typing import Optional

# Set up logging to see errors in the Hugging Face logs tab
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenEnv Support Triage Environment")

# --- DATA MODELS ---
class StepRequest(BaseModel):
    ticket_id: str
    department: str

class ResetRequest(BaseModel):
    task_id: Optional[str] = None

# --- DATABASE ---
initial_tickets = [
    {"id": "T1", "issue": "I was charged twice for my subscription.", "category": "billing"},
    {"id": "T2", "issue": "The app crashes when I click the 'Save' button.", "category": "tech_support"},
    {"id": "T3", "issue": "I want to upgrade my plan to enterprise.", "category": "sales"},
]

current_tickets = list(initial_tickets)

# --- HELPER FUNCTION ---
def get_observation():
    return {
        "open_tickets": current_tickets,
        "system_message": "Route the tickets to 'billing', 'tech_support', or 'sales'."
    }

# --- GLOBAL EXCEPTION HANDLER ---
# Catch-all for any unexpected server-side crashes
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error caught: {exc}")
    return JSONResponse(
        status_code=500,
        content={"observation": get_observation(), "reward": 0.0, "done": False, "info": {"error": "Internal Server Error"}}
    )

# --- ENDPOINTS ---
@app.get("/")
def read_root():
    return {"status": "Running", "message": "Support Triage Environment is live!"}

@app.post("/reset")
def reset_env(req: Optional[ResetRequest] = None):
    try:
        global current_tickets
        current_tickets = list(initial_tickets)
        logger.info("Environment reset successfully.")
        return {"observation": get_observation()}
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset environment")

@app.post("/step")
def step(request: StepRequest):
    try:
        global current_tickets
        
        # 1. Check if we have any tickets left to process
        if not current_tickets:
            return {
                "observation": get_observation(),
                "reward": 0.0,
                "done": True,
                "info": {"error": "No tickets available to process"}
            }

        # 2. Find the specific ticket
        ticket = next((t for t in current_tickets if t["id"] == request.ticket_id), None)
        
        if not ticket:
            logger.warning(f"AI requested non-existent ticket: {request.ticket_id}")
            return {
                "observation": get_observation(),
                "reward": 0.0,
                "done": False,
                "info": {"error": f"Ticket {request.ticket_id} not found"}
            }
        
        # 3. Logic for reward
        if request.department.lower() == ticket["category"].lower():
            reward = 1.0
        else:
            reward = 0.0
            
        # 4. State update
        current_tickets = [t for t in current_tickets if t["id"] != request.ticket_id]
        done = len(current_tickets) == 0
            
        return {
            "observation": get_observation(),
            "reward": reward,
            "done": done,
            "info": {"correct_category": ticket["category"]}
        }

    except Exception as e:
        logger.error(f"Step failed: {e}")
        # Return a structured error so the inference script doesn't crash
        return {
            "observation": get_observation(),
            "reward": 0.0,
            "done": False,
            "info": {"error": str(e)}
        }

# --- VALIDATOR ENTRY POINT ---
def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()