from .models import Observation, Action, Reward, State, Ticket
from .tasks import get_task_data

class SupportEnv:
    def __init__(self):
        self.state: State = None

    def reset(self, task_id: str = "triage-easy") -> Observation:
        tickets, correct_routes = get_task_data(task_id)
        self.state = State(
            current_task=task_id,
            tickets=tickets,
            correct_routes=correct_routes,
            steps_taken=0,
            total_score=0.0,
            is_done=False
        )
        return self._get_obs("Environment reset. Ready for triage.")

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self.state.is_done:
            return self._get_obs("Task already completed."), Reward(value=0.0, reason="Done"), True, {}

        self.state.steps_taken += 1
        reward_val = 0.0
        reason = ""

        # Find ticket
        ticket = next((t for t in self.state.tickets if t.id == action.ticket_id and t.status == "open"), None)
        
        if not ticket:
            reward_val = -0.1
            reason = "Invalid ticket ID or ticket already routed."
        else:
            correct_dept = self.state.correct_routes.get(ticket.id)
            if action.department == correct_dept:
                ticket.status = "routed"
                # Partial reward based on total tickets to ensure max score = 1.0
                reward_val = 1.0 / len(self.state.correct_routes)
                self.state.total_score += reward_val
                reason = f"Correctly routed {ticket.id} to {action.department}."
            else:
                reward_val = -0.1
                reason = f"Incorrect routing for {ticket.id}."

        # Check completion
        open_tickets = [t for t in self.state.tickets if t.status == "open"]
        if not open_tickets or self.state.steps_taken >= len(self.state.correct_routes) + 3:
            self.state.is_done = True

        obs = self._get_obs(reason)
        reward = Reward(value=reward_val, reason=reason)
        info = {"error": None} if reward_val >= 0 else {"error": reason}
        
        return obs, reward, self.state.is_done, info

    def _get_obs(self, message: str) -> Observation:
        open_t = [t for t in self.state.tickets if t.status == "open"]
        return Observation(open_tickets=open_t, system_message=message)

    def get_state(self) -> State:
        return self.state