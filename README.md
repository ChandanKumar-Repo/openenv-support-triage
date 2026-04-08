# OpenEnv: Support Triage Environment

## Motivation and Description
This environment simulates a real-world Tier-1 IT Support / Customer Service dispatcher. Humans spend countless hours reading vague emails and assigning them to the right queues. This environment tests an LLM's ability to analyze text context and apply business routing rules correctly over an API.

## Action & Observation Spaces
- **Observation Space**: JSON containing `open_tickets` (array of ticket objects with IDs and text) and a `system_message`.
- **Action Space**: JSON containing `ticket_id` (string) and `department` (enum: 'billing', 'tech_support', 'sales').

## Tasks
1. **triage-easy**: 1 clear ticket. Very high baseline success rate.
2. **triage-medium**: 3 mixed tickets. Tests context switching.
3. **triage-hard**: 5 complex tickets including vague prompts and dual-intent complaints. 

## Setup & Usage
1. Build the docker container: `docker build -t openenv-support .`
2. Run locally: `docker run -p 7860:7860 openenv-support`
3. Export environment variables (`HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`, `SPACE_URL=http://localhost:7860`).
4. Run inference: `python inference.py`

## Baseline Scores
- Qwen2.5-72B-Instruct achieves ~1.00 on Easy/Medium, and ~0.80 on Hard.