---
title: The Gauntlet + Shifting Sands
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - world-modeling
  - self-improvement
  - adversarial
  - customer-support
---

# The Gauntlet + Shifting Sands

An OpenEnv environment where support agents must make high-stakes decisions in a changing world.

- **Gauntlet**: an attacker generates deceptive tickets designed to mislead the defender.
- **Shifting Sands**: policy drift changes the rules mid-episode, and the agent must adapt.

This project is built for OpenEnv Hackathon themes around **World Modeling** and **Self-Improvement**.

---

## 1) The Story: Why This Exists

LLMs are good at one-shot text completion, but real support operations are not one-shot.
In production, agents face:

- partial information,
- adversarial or noisy inputs,
- changing policies,
- and delayed business consequences.

Most benchmarks do not capture this. They reward isolated classification, not durable decision-making under evolving constraints.

**This environment addresses that gap.**  
The agent is evaluated not just on "is this label correct now?" but also on "did this sequence of actions keep the business healthy across the whole episode?"

---

## 2) Problem Statement

We target a concrete capability gap:

> Train an LLM to triage customer-support tickets robustly when tickets can be deceptive and policies can change during the session.

The challenge is a partially observable, non-stationary MDP:

- observations are incomplete and sometimes intentionally misleading,
- hidden ground-truth labels are revealed only after decisions,
- policy version can drift mid-trajectory,
- bad early decisions damage later outcomes.

---

## 3) Why This Is Novel (Environment Innovation - 40%)

Compared with static ticket classification tasks, this environment introduces:

- **Adversarial ticket generation** via an attacker strategy component.
- **Mid-episode policy drift** that changes label expectations and schema surface.
- **Multi-component rewards** that combine local action quality with global world-state health.
- **Episode-level consequences** (e.g., churn risk, SLA breaches, escalation pressure) that persist across steps.

The result is a testbed for behaviors that matter in realistic deployments: adaptation, calibration, and long-horizon decision quality.

---

## 4) Environment Design

### Agent Roles

- **Defender (trainable)**: processes tickets, assigns priority/category, drafts response, decides escalation, optionally asks clarifying question.
- **Attacker (optional)**: proposes deceptive tickets based on current difficulty and policy context.

### Tasks

- `1` - **Priority Only** (easy)
- `2` - **Full Classification** (medium)
- `3` - **Multi-Turn + Drift** (hard)

### State and Dynamics

The environment tracks a persistent world state (exported every step), including business-health signals.
Actions mutate state, and those mutations affect future rewards.

### OpenEnv Contract

- `POST /reset`
- `POST /step`
- `GET /world_state/{session_id}`
- `GET /episodes`
- `GET /episodes/{episode_id}`
- `GET /health`

Manifest: `openenv.yaml`

---

## 5) Reward Logic (Reward + Pipeline Quality - 10%)

Reward is designed to teach behavior, not just score formatting:

- dense feedback at every step,
- component-level decomposition for diagnosis,
- penalties/reductions when behavior exploits shortcuts,
- world-state terms to prevent myopic optimization.

At each step, the server returns:

- scalar reward,
- reward breakdown,
- next observation,
- updated world state,
- optional drift notice.

This makes training and debugging measurable and reproducible.

---

## 6) Evidence of Learning (Improvement in Rewards - 20%)

Use this section to show before/after training performance and behavior changes.

### Current baseline snapshot

From local benchmark runs (`inference.py` / `run_baseline.py`):

| Task | Rule-Based | LLM Baseline |
|---|---:|---:|
| 1 (Priority) | ~0.68 | ~0.88 |
| 2 (Full Classification) | ~0.53 | ~0.74 |
| 3 (Multi-Turn + Drift) | ~0.45 | ~0.63 |

Interpretation: the largest gap appears on Task 3, where adaptation and better clarification behavior matter most.

### What judges should see here

- reward curves (trained vs untrained),
- baseline vs trained side-by-side metrics,
- short qualitative trajectory examples (before/after),
- links to plots committed under `plots/` (PNG/JPG).

---

## 7) Storytelling Checklist (Storytelling - 30%)

This README is structured to answer the four judge questions directly:

1. **Problem** - what capability gap is targeted?
2. **Environment** - what does the agent observe, do, and optimize?
3. **Results** - what improves after training?
4. **Why it matters** - who benefits and why this domain is valuable?

If a reviewer reads only this README for 3-5 minutes, they should understand both the motivation and the evidence plan.

---

## 8) Minimum Hackathon Requirements Mapping

From the attached judging criteria, here is how this repo maps:

- **OpenEnv usage (required)**  
  Uses OpenEnv-compatible API + `openenv.yaml` manifest.

- **Training script in Colab with Unsloth/HF TRL (required)**  
  Notebook: `train_colab.ipynb` (ensure final version is runnable and linked in submission).

- **Evidence of real training (required)**  
  Add reward/loss plots and trained-vs-baseline metrics in this README.

- **Mini-blog / short video (<2 min) (required)**  
  Add links below before final submission.

- **Hosted on Hugging Face Spaces (required)**  
  Space from manifest: [Subhakshay/support-triage-env](https://huggingface.co/spaces/Subhakshay/support-triage-env)

---

## 9) Quick Start

### Docker

```bash
docker build -t gauntlet-shifting-sands .
docker run -p 7860:7860 gauntlet-shifting-sands
```

Open docs: [http://localhost:7860/docs](http://localhost:7860/docs)

### Local

```bash
pip install -r requirements.txt
python main.py
```

### Smoke test

```bash
curl -X POST "http://localhost:7860/reset" -H "Content-Type: application/json" -d "{\"task_id\":1}"
```

---

## 10) API Example

### Reset

```json
{
  "task_id": 3,
  "attacker_enabled": true,
  "drift_enabled": true,
  "difficulty_init": 0.3
}
```

### Step

```json
{
  "session_id": "YOUR_SESSION_ID",
  "action": {
    "assign_priority": "High",
    "assign_category": "Technical",
    "draft_response": "Thanks for the report. We are investigating now and will update you shortly.",
    "escalate": true,
    "ask_clarification": false
  }
}
```

---

## 11) Repository Map

- `main.py` - FastAPI/OpenEnv server
- `environment.py` - environment dynamics and scoring orchestration
- `rewards.py` - reward components and aggregation
- `attacker.py` - adversarial ticket generation
- `policy.py` / `drift_scheduler.py` - policy versions and drift events
- `world_state.py` - persistent business state across an episode
- `db.py` - async persistence for episodes/steps/snapshots
- `openenv.yaml` - manifest for environment metadata and task schema
- `inference.py`, `run_baseline.py` - evaluation scripts
- `train_colab.ipynb` - training notebook entry point

---

## 12) Submission Links (Fill Before Final Submit)

- Hugging Face Space: [Subhakshay/support-triage-env](https://huggingface.co/spaces/Subhakshay/support-triage-env)
- Mini-blog / HF post: `ADD_LINK`
- <2 min demo video: `ADD_LINK`
- Training run dashboard (optional): `ADD_LINK`
- Key plot artifacts: `ADD_LINK_OR_RELATIVE_PATH`

---

## 13) Final Note

This environment is intended to push beyond static QA into **adaptive, long-horizon, world-aware LLM behavior**.
If the trained defender improves reward while maintaining world-state health under adversarial pressure and policy drift, it demonstrates meaningful progress on realistic agent capabilities.
