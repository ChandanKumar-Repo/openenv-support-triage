from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class Ticket(BaseModel):
    id: str
    text: str
    status: str = "open" # open, routed

class Observation(BaseModel):
    open_tickets: List[Ticket]
    system_message: str
    
class Action(BaseModel):
    ticket_id: str = Field(description="The ID of the ticket to route.")
    department: str = Field(description="Must be: 'billing', 'tech_support', or 'sales'.")

class Reward(BaseModel):
    value: float
    reason: str

class State(BaseModel):
    current_task: str
    tickets: List[Ticket]
    correct_routes: Dict[str, str]
    steps_taken: int
    total_score: float
    is_done: bool