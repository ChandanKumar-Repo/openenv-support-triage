import asyncio
import os
import textwrap
import json
from typing import List, Optional
import requests
from openai import OpenAI

# Required Environment Variables
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "support-triage-env"
SPACE_URL = os.getenv("SPACE_URL", "http://127.0.0.1:7860") 
MAX_STEPS = 10
TEMPERATURE = 0.1

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Support Dispatcher. Your job is to route open tickets to the correct department.
    Valid departments: 'billing', 'tech_support', 'sales'.
    Reply in strictly valid JSON format matching this schema:
    {"ticket_id": "T1", "department": "billing"}
    Only pick ONE ticket to route right now.
    """
).strip()

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, obs: dict) -> dict:
    user_prompt = f"Current Open Tickets: {json.dumps(obs.get('open_tickets', []))}\nSystem: {obs.get('system_message')}"
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"}
        )
        response_text = completion.choices[0].message.content
        
        # Clean up any potential markdown formatting from the AI (e.g., ```json ... ```)
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        response = json.loads(response_text)
        
        if isinstance(response, list) and len(response) > 0:
            response = response[0]
            
        return {
            "ticket_id": str(response.get("ticket_id", "T1")),
            "department": str(response.get("department", "billing"))
        }
        
    except Exception as e:
        print(f"[DEBUG] AI Parsing/API Error: {e}")
        # Return a safe fallback to prevent the script from crashing
        return {"ticket_id": "T1", "department": "billing"}

async def run_task(task_name: str):
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    rewards = []
    steps_taken = 0
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # Reset Environment with a timeout (Hugging Face can be slow to wake up)
        reset_response = requests.post(f"{SPACE_URL}/reset", json={"task_id": task_name}, timeout=20)
        reset_response.raise_for_status()
        reset_res = reset_response.json()
        
        if "observation" not in reset_res:
            raise KeyError("The server reset response is missing the 'observation' key.")
            
        obs = reset_res["observation"]
        
        for step in range(1, MAX_STEPS + 1):
            action_payload = get_model_action(client, obs)
            action_str = json.dumps(action_payload).replace(" ", "")
            
            # Step Environment with timeout
            step_response = requests.post(f"{SPACE_URL}/step", json=action_payload, timeout=10)
            
            if step_response.status_code != 200:
                print(f"\n[DEBUG] Server Error: {step_response.text}\n", flush=True)
                log_step(step=step, action=action_str, reward=0.0, done=True, error=f"HTTP {step_response.status_code}")
                break
                
            step_res = step_response.json()
            
            # Use .get() to avoid KeyErrors if the server response is malformed
            obs = step_res.get("observation", {})
            reward = float(step_res.get("reward", 0.0))
            done = step_res.get("done", True)
            error = step_res.get("info", {}).get("error")
            
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            
            if done:
                break
                
        score = min(max(sum(rewards), 0.0), 1.0)
        log_end(success=(score >= 0.99), steps=steps_taken, score=score, rewards=rewards)

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Connection to Space failed: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error in run_task: {e}")

async def main():
    tasks = ["triage-easy", "triage-medium", "triage-hard"]
    for task in tasks:
        await run_task(task)

if __name__ == "__main__":
    asyncio.run(main())