from typing import Dict, Tuple
from .models import Ticket

def get_task_data(task_id: str) -> Tuple[list[Ticket], Dict[str, str]]:
    if task_id == "triage-easy":
        tickets = [Ticket(id="T1", text="I need a refund for my last invoice.")]
        routes = {"T1": "billing"}
    elif task_id == "triage-medium":
        tickets = [
            Ticket(id="T1", text="My router keeps dropping connection."),
            Ticket(id="T2", text="How much does the enterprise plan cost?"),
            Ticket(id="T3", text="Update my credit card.")
        ]
        routes = {"T1": "tech_support", "T2": "sales", "T3": "billing"}
    else: # triage-hard
        tickets = [
            Ticket(id="T1", text="The app crashed and deleted my project!"),
            Ticket(id="T2", text="Can I talk to someone about volume discounts?"),
            Ticket(id="T3", text="Your pricing page is confusing, is it $10 or $20?"),
            Ticket(id="T4", text="I was double charged."),
            Ticket(id="T5", text="How do I reset my password?")
        ]
        routes = {
            "T1": "tech_support", "T2": "sales", 
            "T3": "sales", "T4": "billing", "T5": "tech_support"
        }
    return tickets, routes