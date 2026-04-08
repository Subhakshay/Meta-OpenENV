---
title: Support Triage Env
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - nlp
  - multi-turn
  - real-world
---

# 🎫 Customer Support Triage Environment

A real world, stateful reinforcement learning environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

Agents act as a **customer support triage coordinators** reading incoming tickets, clarifying questions when needed, assigning priority and category to the tickets, drafting responses, and managing escalations.

[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://hub.docker.com/)
[![Python](https://img.shields.io/badge/python-3.11+-green)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/backend-fastapi-teal)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## 📋 Table of Contents

1. [Why Customer Support Triage?](#-why-customer-support-triage)
2. [Quick Start](#-quick-start)
3. [Environment Overview](#-environment-overview)
4. [Tasks & Difficulty Progression](#-tasks--difficulty-progression)
5. [Action Space](#-action-space)
6. [Observation Space](#-observation-space)
7. [Reward Structure](#-reward-structure)
8. [World State & MDP Mechanics](#-world-state--mdp-mechanics)
9. [Ticket Generation](#-ticket-generation)
10. [Multi-Turn Dialogue (Task 3)](#-multi-turn-dialogue-task-3)
11. [API Reference](#-api-reference)
12. [Examples](#-examples)
13. [Baseline Scores](#-baseline-scores)
14. [Installation & Usage](#-installation--usage)
15. [Troubleshooting](#-troubleshooting)

---

## 🧠 Why Customer Support Triage?

Customer support triage is a significant real world decision task. Many enterprises like SaaS companies, banks, hospitals, e-commerce platforms must continuously classify, prioritise, and respond to incoming issues under time pressure and resource constraints.

This environment is **not a classification benchmark**. It is a full **Markov Decision Process** where:

- Agent decisions have lasting consequences on business metrics that compound across the episode
- Some tickets require continous dialogue before they can be resolved
- Overescalating drains resources and underescalating causes SLA breaches
- Poor responses spike customer churn risk that persists for some time

### Research Motivation

| Research Theme | Role in This Environment |
|---|---|
| Multi turn dialogue management | `ASK` action triggers customer reply; agent must integrate new info before resolving |
| Resource-constrained decision making | Escalation queue has a hard capacity of 3; overflow triggers penalties |
| Reward shaping & partial credit | Priority grading gives partial credit for near miss classifications |
| LLM reasoning under business constraints | World state metrics must be managed while processing 10–20 tickets |
| Stateful MDP vs supervised classification | Actions mutate `company_balance`, `churn_risk`, and `sla_breach_count` |

---

## 🚀 Quick Start

### Using Docker

```bash
docker build -t customer-support-env:latest .
docker run -p 7860:7860 customer-support-env:latest
# Docs at http://localhost:7860/docs
```

### Basic Python Client

```python
import requests

BASE = "http://localhost:7860"

# Start a new episode
session = requests.post(f"{BASE}/reset", json={
    "task_id": "task_1_priority",
    "seed": 42
}).json()

session_id = session["session_id"]
obs = session["observation"]
print(f"Ticket: {obs['subject']}")
print(f"Tier: {obs['customer_tier']}  Sentiment: {obs['sentiment']}")

# Take a step
result = requests.post(f"{BASE}/step", json={
    "session_id":      session_id,
    "assign_priority": "critical",
    "assign_category": "technical",
    "response_text":   "Dear Valued Enterprise Customer, our engineering team has been alerted and is actively investigating. We will provide updates every 30 minutes. Best regards, Support Team",
    "escalate":        True,
}).json()

print(f"Reward: {result['reward']['value']}")
print(f"Breakdown: {result['reward']['breakdown']}")
print(f"Done: {result['done']}")
```

### Run Baseline Evaluation (No API Key Needed)

```bash
python inference.py
```

```bash
# With LLM agent
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="gsk_..."
python inference.py
```

---

## 🌍 Environment Overview

A queue of support tickets is generated **procedurally** at the start of each episode. The agent processes them one by one, and every decision it makes updates the **World State** business metrics that persist and compound across the full episode.

### Core Mechanics

- **Procedural Generation** - 8 ticket blueprint families × randomised fill-in pools yield millions of unique tickets. Memorisation is impossible.
- **Structural Difficulty Progression** - Each task changes what the environment requires of the agent, not just what the grader checks.
- **Stateful MDP** - Agent actions mutate `company_balance`, `customer_churn_risk`, `escalation_queue`, and `sla_breach_count`.
- **Ground Truth Withheld** - True labels are revealed only in `info["episode_summary"]` when `done=True`. The agent cannot look ahead.
- **Multi-Turn Dialogue** - In Task 3, ambiguous tickets require an `ASK` action first. The environment simulates a customer reply before the agent resolves.

### Episode Flow

```
reset()
  ├─ generates ticket queue
  ├─ initialises WorldState
  └─ returns first Observation

loop:
  agent reads Observation
  agent returns Action
  step(action)
    ├─ grades decision components
    ├─ applies world state effects
    ├─ computes weighted reward
    └─ advances to next ticket (or awaits customer reply on ASK)

done=True
  └─ episode_summary with full ground truth log revealed
```

### Episode Termination

- **All tickets processed** - queue is empty
- **Max steps reached** - 10 steps for task_1/task_2; 20 steps for task_3

---

## 🪜 Tasks & Difficulty Progression

The three tasks are **structurally different**, not just differently graded.

| ID | Name | Difficulty | What structurally changes |
|:---|:---|:---|:---|
| `task_1_priority` | Priority Assignment | **Easy** | Only unambiguous tickets. Only priority is scored. |
| `task_2_classification` | Ticket Classification | **Medium** | Ambiguous tickets included. Priority + category + response all scored. |
| `task_3_full_triage` | Full Ticket Triage | **Hard** | Multiple turns dialogue. World state scored. Ambiguous tickets at 2:1 ratio. |

### Task 1 - Priority Assignment (Easy)

Clear signal tickets only. Single scoring dimension. The agent needs to read explicit urgency signals and assign the correct priority.

**Example ticket:**
```
Subject: URGENT: Production API returning 500 errors
Body: Our enterprise integration has been down for 4 hours. We are losing
      $1,200 per minute. 2,000 users cannot access the platform.
```
**Correct action:** `assign_priority = "critical"`

### Task 2 - Ticket Classification (Medium)

Ambiguous tickets are introduced alongside clear ones. All three dimensions - priority, category, and response quality - are scored.

**Ambiguous ticket example:**
```
Subject: Something isn't working like it used to
Body: Hi, the thing I normally use for reporting isn't doing what it did
      last Tuesday. I haven't changed anything on my end.
```
The correct priority and category require reasoning from context, not keyword matching.

### Task 3 - Full Triage (Hard)

Ambiguous tickets appear at a 2:1 ratio. The agent must issue an `ASK` action on ambiguous tickets before classifying. World state health contributes 15% of every step's reward, so the agent must plan across the full episode.

---

## 🎮 Action Space

### `classify` - Resolve the Ticket (All Tasks)

```json
{
  "action_type":         "classify",
  "assign_priority":     "low | medium | high | critical",
  "assign_category":     "billing | technical | account | shipping | general | refund",
  "response_text":       "Hello, thank you for reaching out...",
  "escalate":            false
}
```

### `ask` - Request Clarification (Task 3 Only)

Used when the ticket is ambiguous. The environment simulates a customer reply. Use `classify` on the next step.

```json
{
  "action_type":         "ask",
  "clarifying_question": "Could you clarify which specific feature is affected and when the issue started?"
}
```

**Effects of ASK:**
- Small partial reward: `clarification_quality × 0.10`
- Customer reply appended to `clarification_history` in next observation
- `awaiting_clarification = true` - same ticket must be resolved next
- **Skipping ASK on an ambiguous ticket** reduces `priority_score` by 0.20

### Priority Values

| Value | When to use |
|---|---|
| `critical` | Production down, data loss, full outage, emergency |
| `high` | Cannot login, duplicate charges, overdue orders, enterprise blockers |
| `medium` | Bugs with workarounds, ambiguous billing, API limits |
| `low` | General questions, pricing inquiries, feedback |

### Category Values

| Value | When to use |
|---|---|
| `billing` | Charges, payments, invoices, subscriptions |
| `technical` | Bugs, API issues, outages, errors |
| `account` | Login, passwords, access issues |
| `shipping` | Orders, deliveries, tracking |
| `refund` | Refund or reimbursement requests |
| `general` | Anything else |

---

## 👁️ Observation Space

```python
{
  # Ticket content (no ground truth labels)
  "ticket_id":              "TKT-83421",
  "subject":                "URGENT: Production API returning 500 errors",
  "body":                   "Our enterprise integration has been down...",
  "customer_tier":          "free | pro | enterprise",
  "created_at":             "2025-04-01T06:30:00Z",
  "sentiment":              "angry | neutral | positive",

  # Multi-turn dialogue state (Task 3)
  "clarification_history":  [{"agent": "...", "customer": "..."}],
  "awaiting_clarification": false,
  "customer_reply":         null,

  # Episode context
  "queue_size":             7,
  "time_elapsed_seconds":   1.24,
  "agent_actions_taken":    2,
  "task_id":                "task_3_full_triage",
  "hint":                   "Multi-turn triage. Ambiguous tickets REQUIRE clarification...",

  # Live world state (observe business consequences of past actions)
  "world_state": {
    "company_balance":      9850.01,
    "escalation_queue":     1,
    "escalation_capacity":  3,
    "customer_churn_risk":  0.10,
    "sla_breach_count":     0,
    "tickets_resolved":     2,
    "avg_response_quality": 0.74
  }
}
```

---

## 💰 Reward Structure

All rewards are in `[0.0, 1.0]` - dense, per-step.

### Reward Weights by Task

| Component | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| `priority` | **100%** | 40% | 25% |
| `category` | - | 30% | 20% |
| `response` | - | 30% | 25% |
| `clarification` | - | - | 15% |
| `world_state` | - | - | 15% |

### Priority Grading (Partial Credit)

| Distance | Score |
|---|---|
| Exact match | `1.00` |
| Off by 1 (e.g. high → critical) | `0.60` |
| Off by 2 (e.g. low → high) | `0.20` |
| Off by 3 (e.g. low → critical) | `0.00` |

### Category Grading (Partial Credit)

Related categories receive partial credit:

| Assigned | True | Score |
|---|---|---|
| `billing` | `billing` | `1.00` |
| `refund` | `billing` | `0.40` ← related |
| `account` | `technical` | `0.40` ← related |
| `shipping` | `billing` | `0.00` |

### Response Quality Grading

| Factor | Max score |
|---|---|
| Greeting present | `+0.15` |
| References ticket subject | `+0.20` |
| Category-relevant vocabulary | `+0.25` |
| Commits to a next action | `+0.20` |
| Correct escalation decision | `+0.10` |
| Professional closing | `+0.10` |

### Clarification Quality Grading

| Factor | Max score |
|---|---|
| Uses question words | `+0.30` |
| Asks about specific feature/area | `+0.40` |
| Asks for reproduction steps or timing | `+0.30` |

### World State Score (Task 3)

```
world_state_score = (balance_score + churn_score + sla_score + queue_score) / 4

balance_score = min(1.0, company_balance / 10_000)
churn_score   = 1.0 - customer_churn_risk
sla_score     = max(0.0, 1.0 - sla_breach_count × 0.25)
queue_score   = max(0.0, 1.0 - queue_overflow × 0.20)
```

---

## ⚙️ World State & MDP Mechanics

The `WorldState` persists across the entire episode and is mutated by every `classify` action.

### Mutation Rules

| Agent action | World state effect |
|---|---|
| Escalates a refund/billing ticket | `company_balance -= $30–$99` |
| Response quality < 0.30 | `customer_churn_risk += 0.15` |
| Response quality > 0.70 | `customer_churn_risk -= 0.05` |
| Escalates any ticket | `escalation_queue += 1` |
| Escalation queue exceeds 3 | `customer_churn_risk += 0.10` |
| Misses a CRITICAL ticket | `sla_breach_count += 1` |

### Why It Is a Real MDP

In a pure classification task, ticket #3's classification has no effect on ticket #4. Here it does:

- A poor response on ticket #3 spikes `churn_risk`, reducing the `world_state_score` for all subsequent tickets
- Over-escalating early fills the queue, forcing churn penalties on later decisions
- Missing a critical ticket causes a permanent SLA breach that lowers `sla_score` for the rest of the episode

The agent must plan across the full episode, not optimise each ticket in isolation.

---

## 🎟️ Ticket Generation

Tickets are generated procedurally at `reset()` using 8 blueprint families with randomised fill-in pools.

### Blueprint Families

| Family | Priority | Category | Ambiguous? |
|---|---|---|---|
| Critical outage | CRITICAL | TECHNICAL | No |
| Login/access failure | HIGH | ACCOUNT | No |
| Duplicate billing | HIGH | BILLING | No |
| Vague technical issue | MEDIUM | TECHNICAL | **Yes** |
| Vague billing question | MEDIUM | BILLING | **Yes** |
| General inquiry | LOW | GENERAL | No |
| Late/wrong shipment | HIGH | SHIPPING | No |
| Partial refund request | MEDIUM | REFUND | No |

### Variable Pools

Body templates contain `{variable}` placeholders filled from randomised pools:

```
{duration}      → "2 hours" | "4 hours" | "since this morning" | ...
{user_count}    → "120" | "500" | "2,000" | "50,000" | ...
{charge_amount} → "49.99" | "99.00" | "299.00" | ...
{vague_feature} → "main panel" | "analytics tab" | "export tool" | ...
```

With 8 families × 4 subjects × 3 body templates × 30+ variable pools, the combinatorial space exceeds **10 million unique tickets**.

### Task-Specific Pools

```
task_1_priority       → only non-ambiguous blueprints
task_2_classification → full mix (clear + ambiguous)
task_3_full_triage    → ambiguous blueprints at 2:1 ratio
```

---

## 💬 Multi-Turn Dialogue (Task 3)

```
Step N:    Agent sends ASK
           ├─ Environment grades question relevance (0.0–1.0)
           ├─ Simulates customer reply
           ├─ Sets awaiting_clarification = True
           └─ Returns small partial reward (clarification_quality × 0.10)

Step N+1:  Agent sends CLASSIFY (same ticket)
           ├─ Clarification bonus applied to reward
           └─ Advances to next ticket
```

### Simulated Customer Reply

On ambiguous tickets after a good question:
```
"Sure! I'm referring to the analytics tab section.
My account ID is ACC-847291.
It started happening three times today."
```

On tickets that don't need clarification:
```
"I think I already explained everything in my original message."
```

### Observation After ASK

```json
{
  "clarification_history": [
    {
      "agent":    "Could you clarify which specific section is affected?",
      "customer": "Sure! I'm referring to the analytics tab. It started three times today."
    }
  ],
  "awaiting_clarification": true,
  "customer_reply": "Sure! I'm referring to the analytics tab. It started three times today."
}
```

---

## 📚 API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Status and links |
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/tasks/{task_id}` | Single task details |
| `GET` | `/action_space` | Action space spec |
| `GET` | `/observation_space` | Observation space spec |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Take one action |
| `GET` | `/state/{session_id}` | Episode state snapshot |
| `DELETE` | `/session/{session_id}` | Close and clean up session |

### POST /reset

```json
{ "task_id": "task_1_priority", "seed": 42 }
```

### POST /step

```json
{
  "session_id":      "uuid",
  "assign_priority": "high",
  "assign_category": "technical",
  "response_text":   "Hello, our team will investigate...",
  "escalate":        false
}
```

**Step response:**
```json
{
  "observation": { "...": "..." },
  "reward": {
    "value": 0.74,
    "breakdown": {
      "priority_raw": 1.0, "category_raw": 1.0, "response_raw": 0.80,
      "priority": 0.25,    "category": 0.20,    "response": 0.20,
      "clarification": 0.15, "world_state": 0.14
    },
    "done": false,
    "info": { "step": 3, "cumulative_reward": 2.14 }
  },
  "done": false,
  "info": { "step": 3, "mean_reward_so_far": 0.71 }
}
```

---

## 📖 Examples

### Example 1: Simple Task 1 Loop

```python
import requests

BASE = "http://localhost:7860"
session = requests.post(f"{BASE}/reset", json={"task_id": "task_1_priority", "seed": 42}).json()
session_id = session["session_id"]

total_reward, steps = 0, 0
result = session

while True:
    obs = result.get("observation", session["observation"])
    text = (obs["subject"] + " " + obs["body"]).lower()

    if any(w in text for w in ["urgent", "critical", "emergency", "outage", "down"]):
        priority = "critical"
    elif any(w in text for w in ["cannot", "locked", "charged twice", "not arrived"]):
        priority = "high"
    elif any(w in text for w in ["sometimes", "slow", "question", "feature"]):
        priority = "medium"
    else:
        priority = "low"

    result = requests.post(f"{BASE}/step", json={
        "session_id": session_id, "assign_priority": priority,
        "assign_category": "general", "response_text": "", "escalate": False,
    }).json()

    total_reward += result["reward"]["value"]
    steps += 1
    print(f"Step {steps}: priority={priority}  reward={result['reward']['value']:.4f}")

    if result["done"]:
        break

print(f"\nMean reward: {total_reward/steps:.4f}")
```

### Example 2: Multi-Turn Task 3 Agent

```python
import requests

BASE = "http://localhost:7860"
VAGUE = ["something isn't", "seems off", "acting weird", "not sure", "looks different"]

session = requests.post(f"{BASE}/reset", json={"task_id": "task_3_full_triage", "seed": 42}).json()
session_id = session["session_id"]

result = session
total_reward, steps = 0, 0

while True:
    obs = result.get("observation", session["observation"])
    text = (obs["subject"] + " " + obs["body"]).lower()
    is_vague = any(s in text for s in VAGUE)
    already_asked = bool(obs.get("clarification_history"))
    awaiting = obs.get("awaiting_clarification", False)

    if is_vague and not already_asked and not awaiting:
        # Ask first
        payload = {
            "session_id": session_id,
            "assign_priority": "medium", "assign_category": "general",
            "response_text": "", "escalate": False,
            "action_type": "ask",
            "clarifying_question": "Could you clarify which specific feature or section is affected, and when the issue started?",
        }
        print(f"Step {steps+1}: ASK")
    else:
        # Resolve
        full_text = text + " " + (obs.get("customer_reply") or "")
        priority = "critical" if any(w in full_text for w in ["urgent", "production", "outage"]) else "medium"
        escalate = priority == "critical"
        tier_greet = {"enterprise": "Dear Valued Enterprise Customer", "pro": "Hello", "free": "Hi there"}
        greeting = tier_greet.get(obs["customer_tier"], "Hello")
        response = (
            f"{greeting},\n\nThank you for contacting us regarding '{obs['subject']}'.\n\n"
            f"Our team will investigate and follow up with you shortly.\n\nBest regards, Support Team"
        )
        payload = {
            "session_id": session_id,
            "assign_priority": priority, "assign_category": "technical",
            "response_text": response, "escalate": escalate,
        }
        print(f"Step {steps+1}: CLASSIFY  priority={priority}")

    result = requests.post(f"{BASE}/step", json=payload).json()
    total_reward += result["reward"]["value"]
    steps += 1

    ws = result["observation"]["world_state"]
    print(f"  reward={result['reward']['value']:.4f}  churn={ws['customer_churn_risk']:.2f}  sla={ws['sla_breach_count']}")

    if result["done"]:
        s = result["info"].get("episode_summary", {})
        print(f"\n=== Done ===  mean_reward={s.get('mean_reward', 0):.4f}  sla_breaches={s.get('sla_breaches', 0)}")
        break
```

### Example 3: Evaluate All Tasks

```python
import requests

BASE = "http://localhost:7860"

for task_id in ["task_1_priority", "task_2_classification", "task_3_full_triage"]:
    scores = []
    for seed in [42, 123, 7]:
        session = requests.post(f"{BASE}/reset", json={"task_id": task_id, "seed": seed}).json()
        session_id = session["session_id"]
        total, steps, result = 0, 0, session

        while True:
            result = requests.post(f"{BASE}/step", json={
                "session_id": session_id,
                "assign_priority": "medium",
                "assign_category": "technical",
                "response_text": "Hello, thank you for reaching out. Our team will investigate and follow up. Best regards, Support Team",
                "escalate": False,
            }).json()
            total += result["reward"]["value"]
            steps += 1
            if result["done"]:
                break

        scores.append(total / steps)

    mean = sum(scores) / len(scores)
    print(f"{task_id:<30}  mean={mean:.4f}")
```

---

## 📊 Baseline Scores

Mean reward per step, averaged across seeds (42, 123, 7).

| Task | Rule-Based Agent | GPT-4o-mini |
|---|---|---|
| `task_1_priority` | ~0.68 | ~0.88 |
| `task_2_classification` | ~0.53 | ~0.74 |
| `task_3_full_triage` | ~0.45 | ~0.63 |

The gap is largest on Task 3 because the rule-based agent's clarifying question is generic. The LLM writes specific questions that reference the feature and ask for reproduction steps, scoring significantly higher on the `clarification` component.

---

## 🚀 Installation & Usage

### Docker

```bash
docker build -t customer-support-env:latest .
docker run -p 7860:7860 customer-support-env:latest

# With LLM credentials
docker run -p 7860:7860 \
  -e MODEL_NAME=llama-3.3-70b-versatile \
  -e HF_TOKEN=gsk_... \
  customer-support-env:latest
```

### Local Python

We natively utilize a `.env` file for API keys if present.

```bash
pip install uv
uv lock
pip install -r requirements.txt
python server/app.py    # starts server at localhost:7860
python inference.py     # runs baseline evaluation
```

### Submitting & Pre-Validation

To run the pre-validation script locally to verify your setup automatically passes the `openenv` spec checks:

```bash
# Standard Bash
bash pre_validation.sh https://subhakshay-support-triage-env.hf.space
```

If you are on Windows using PowerShell and the standard command fails, use this explicit command to route it through Git Bash:
```powershell
& "C:\Program Files\Git\bin\bash.exe" -c "export PATH=`"$PWD/venv/Scripts:`$PATH`" && ./pre_validation.sh https://subhakshay-support-triage-env.hf.space"
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | OpenAI-compatible API base URL | `https://api.groq.com/openai/v1` |
| `MODEL_NAME` | Model identifier | `llama-3.3-70b-versatile` |
| `HF_TOKEN` | API key | - |

---

## 🔧 Troubleshooting

**Session not found (404)**
The session expired (TTL = 1 hour) or the episode ended (`done=True` auto-cleans up). Call `/reset` again.

**Invalid enum value (400)**
All enum values must be lowercase. `"CRITICAL"` is invalid; `"critical"` is correct. Check valid values via `GET /action_space`.

**Rewards are always 0.0 on Task 3**
The agent is likely sending `response_text = ""`. Response quality contributes 25% of Task 3 reward. Always include a response of at least 30 characters on classify actions.

**LLM agent falls back to rule-based**
Check `HF_TOKEN` is exported before running. The agent logs `[LLM error: ...]` when falling back, check that output for the underlying cause.

**Docker container exits immediately**
```bash
docker logs <container-id>   # check the startup error
lsof -i :7860                # ensure port is free
```

---

## 📁 File Structure

```
.
├── environment.py   # Core OpenEnv environment (MDP, graders, world state)
├── main.py          # Original FastAPI app logic
├── server/app.py    # OpenEnv validate entrypoint (forwards to main.py)
├── pyproject.toml   # Project configuration and CLI entry points
├── uv.lock          # Exact dependency locks for Hackathon validation
├── inference.py     # Baseline agents + evaluation runner
├── openenv.yaml     # OpenEnv spec declaration
├── Dockerfile       # HuggingFace Spaces deployment
├── requirements.txt # Python dependencies
├── .env             # (Git-ignored) Local secrets like HF_TOKEN
└── README.md        # This file
```
