"""
inference.py — Local test runner for The Gauntlet + Shifting Sands

Bypasses the FastAPI server and runs episodes directly against the
CustomerSupportEnv. Supports both rule-based and LLM-powered agents.

Adapted from the existing root inference.py — reuses the same LLM client
setup, logging format, and rule-based heuristics.

Usage:
    python -m gauntlet_shifting_sands.inference
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from environment import CustomerSupportEnv
from policy import PolicyRegistry
import db

# ── LLM client setup ─────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")

_client = None
HAS_LLM = False
try:
    from openai import OpenAI
    if HF_TOKEN:
        _client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
        HAS_LLM = True
except Exception:
    pass

if not HAS_LLM:
    print("[INFO] No HF_TOKEN set — running rule-based agent only.")


# ── Priority / Category signals (reused from root inference.py) ───────────────

_PRIORITY_SIGNALS = {
    "Critical": ["production down", "completely down", "system down", "data loss",
                 "outage", "emergency", "critical", "urgent:", "all users locked",
                 "unauthorized access", "api keys leaked"],
    "High": ["cannot log in", "can't log in", "charged twice", "duplicate charge",
             "hasn't arrived", "wrong item", "locked out", "suspicious login",
             "bypass vulnerability"],
    "Medium": ["sometimes", "occasionally", "slow", "seems off", "not sure",
               "acting weird", "permission errors", "sessions I don't recognize"],
    "Low": ["question about", "interested in", "feedback", "how do i", "how to",
            "quick question", "no rush", "annual billing discount"],
}

_CATEGORY_SIGNALS = {
    "Billing": ["billing", "charge", "payment", "invoice", "plan", "refund", "subscription"],
    "Technical": ["api", "error", "production", "down", "bug", "dashboard", "webhook",
                  "integration", "sync", "500", "timeout"],
    "Shipping": ["order", "arrived", "delivery", "ship", "tracking", "package"],
    "Security": ["unauthorized", "access control", "data breach", "suspicious",
                 "api keys", "leaked", "vulnerability", "authentication bypass"],
}


def _classify_priority(text: str) -> str:
    t = text.lower()
    for p in ["Critical", "High", "Medium", "Low"]:
        if any(kw in t for kw in _PRIORITY_SIGNALS[p]):
            return p
    return "Medium"


def _classify_category(text: str) -> str:
    t = text.lower()
    for c in ["Security", "Billing", "Technical", "Shipping"]:
        if any(kw in t for kw in _CATEGORY_SIGNALS[c]):
            return c
    return "Technical"


# ── Rule-based agent ──────────────────────────────────────────────────────────

def rule_based_agent(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Keyword-based heuristic agent for The Gauntlet."""
    text = f"{observation.get('subject', '')} {observation.get('body', '')}"

    # Check for conversation history (multi-turn)
    if observation.get("conversation_history"):
        for turn in observation["conversation_history"]:
            text += f" {turn.get('customer', '')}"

    priority = _classify_priority(text)
    category = _classify_category(text)
    escalate = priority == "Critical"

    # Draft a response. We'll just use "Dear Customer" as a safe bet for rule-based.
    greeting = "Dear Customer"
    subject = observation.get("subject", "your issue")

    response = (
        f"{greeting},\n\n"
        f"Thank you for reaching out regarding '{subject}'.\n\n"
        f"We will investigate this matter and resolve it promptly. "
        f"Our team has been alerted and will look into this issue. "
        f"We will follow up with a detailed update shortly.\n\n"
        f"Best regards,\nCustomer Support Team"
    )

    action = {
        "assign_priority": priority,
        "assign_category": category,
        "draft_response": response,
        "escalate": escalate,
    }

    # Handle sentiment-aware response for v3
    sentiment = observation.get("sentiment_score")
    if sentiment is not None and sentiment < 0.3:
        action["draft_response"] = (
            f"{greeting},\n\n"
            f"I understand your frustration and I sincerely apologise for the inconvenience "
            f"regarding '{subject}'. We take this matter very seriously.\n\n"
            f"Our team will investigate and address this issue immediately. "
            f"We will follow up with a resolution as soon as possible.\n\n"
            f"Best regards,\nCustomer Support Team"
        )

    return action


# ── LLM agent ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert customer support triage agent in The Gauntlet environment.

You must classify tickets and respond to customers. Return ONLY a JSON object:
{
  "assign_priority": "Low" | "Medium" | "High" | "Critical",
  "assign_category": "Billing" | "Technical" | "Shipping" | "Security",
  "draft_response": "<professional reply to customer>",
  "escalate": true | false
}

RULES:
- Critical: production down, data loss, security breach → escalate: true
- High: login issues, billing errors, missing orders
- Medium: intermittent bugs, ambiguous billing, slow performance
- Low: general questions, feature requests, feedback
- After v2 drift: greeting is REQUIRED, refund window is 14 days not 30
- After v3 drift: Security category exists, use sentiment_score in responses
- If sentiment_score < 0.3: show empathy ("I understand your frustration")
- NEVER cite policy rules you haven't seen in a SYSTEM NOTICE

Return ONLY valid JSON."""


def llm_agent(observation: Dict[str, Any]) -> Dict[str, Any]:
    """LLM-powered agent using OpenAI-compatible API."""
    if not HAS_LLM or _client is None:
        return rule_based_agent(observation)

    parts = [
        f"Policy Version: {observation.get('active_policy_version', 'v1')}",
        f"Ticket ID: {observation.get('ticket_id', 'N/A')}",
        f"Subject: {observation.get('subject', '')}",
        f"Body:\n{observation.get('body', '')}",
        f"Tier: {observation.get('tier', 'unknown')}",
    ]

    if observation.get("system_notice"):
        parts.append(f"\nSYSTEM NOTICE:\n{observation['system_notice']}")
    if observation.get("sentiment_score") is not None:
        parts.append(f"Sentiment Score: {observation['sentiment_score']}")
    if observation.get("account_age_days") is not None:
        parts.append(f"Account Age (days): {observation['account_age_days']}")
    if observation.get("conversation_history"):
        parts.append("\nConversation History:")
        for turn in observation["conversation_history"]:
            parts.append(f"  Agent: {turn.get('agent', '')}")
            parts.append(f"  Customer: {turn.get('customer', '')}")

    ws = observation.get("world_state_summary", {})
    parts.append(f"\nWorld State: balance=${ws.get('company_balance',10000):.0f} "
                 f"churn={ws.get('churn_risk',0):.2f} "
                 f"sla_breaches={ws.get('sla_breaches',0)}")

    try:
        resp = _client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(parts)},
            ],
            temperature=0.1,
            max_tokens=600,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as exc:
        print(f"  [LLM error: {exc}] — falling back to rule-based")
        return rule_based_agent(observation)


# ── Logging helpers ───────────────────────────────────────────────────────────

def log_start(task_id, model):
    print(f"[START] task={task_id} env=gauntlet-shifting-sands model={model}", flush=True)

def log_step(step, action_str, reward, done, error=None):
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={r_str}", flush=True)


# ── Episode runner ────────────────────────────────────────────────────────────

async def run_episode(
    task_id: int = 2,
    attacker_enabled: bool = False,
    drift_enabled: bool = True,
    agent_fn: Callable = None,
    model_name: str = "Rule-Based",
) -> float:
    """Run one full episode and return mean reward."""
    if agent_fn is None:
        agent_fn = rule_based_agent

    env = CustomerSupportEnv()
    obs = env.reset(task_id=task_id, attacker_enabled=attacker_enabled, drift_enabled=drift_enabled)

    if attacker_enabled:
        from attacker import AttackerAgent
        attacker = AttackerAgent(policy_registry=env.policy_registry)
        try:
            adv_tickets = attacker.generate_batch(
                n=25,
                difficulty_level=env.world_state.difficulty_level,
                defender_error_history=[],
                active_policy=env.policy_registry.get_active()
            )
            env.set_attacker_tickets(adv_tickets)
            obs = env._build_observation(adv_tickets[0], None)
        except Exception as e:
            print(f"  [ATTACKER ERROR] {e} — falling back to clean tickets")

    episode_id = await db.create_episode(
        session_id=env.session_id,
        task_id=task_id,
        attacker_enabled=attacker_enabled,
        drift_enabled=drift_enabled,
        difficulty_init=env.world_state.difficulty_level
    )

    log_start(task_id, model_name)
    rewards = []
    steps = 0

    while True:
        try:
            action = agent_fn(obs)
            action_str = json.dumps({k: v for k, v in action.items() if k != "draft_response"})
        except Exception as e:
            action = rule_based_agent(obs)
            action_str = "fallback"

        current_ticket = env._ticket_queue[min(env.world_state.tickets_processed, len(env._ticket_queue)-1)] if env._ticket_queue else {}

        result = env.step(action)
        r_val = result["reward"]
        rewards.append(r_val)
        steps += 1
        done = result["done"]

        log_step(steps, action_str, r_val, done)

        await db.insert_step(
            episode_id=episode_id,
            step_number=steps,
            ticket_id=current_ticket.get("ticket_id", ""),
            action=action,
            defender_reward=r_val,
            attacker_reward=result.get("attacker_reward", 0.0),
            breakdown=result.get("reward_breakdown", {}),
            policy_version=env.policy_registry.active_version_id(),
            was_post_drift=result.get("drift_notice") is not None,
            deception_strategy=current_ticket.get("deception_strategy", "clean")
        )

        await db.insert_snapshot(episode_id, steps, env.world_state)
        await db.insert_ticket_log(episode_id, steps, current_ticket, env.world_state.difficulty_level)

        if result.get("drift_notice") and env._last_reconcile_record:
            rec = env._last_reconcile_record
            await db.insert_drift_event(
                episode_id, steps, rec["from_version"], rec["to_version"],
                ["dynamic_drift"], False, rec["tickets_replaced"]
            )

        if done:
            break
        obs = result["observation"]

    score = sum(rewards) / max(steps, 1)
    success = score >= 0.0
    log_end(success, steps, score, rewards)

    metrics = env.get_episode_metrics()
    await db.close_episode(episode_id, metrics)

    return score


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    await db.init_db()

    agent = llm_agent if HAS_LLM else rule_based_agent
    name = MODEL_NAME if HAS_LLM else "Rule-Based"

    print("=" * 60)
    print("The Gauntlet + Shifting Sands — Inference Runner (DB Linked)")
    print("=" * 60)

    # Task 1: Priority only (clean mode)
    print("\n--- Task 1: Priority Only (no attacker, no drift) ---")
    await run_episode(task_id=1, attacker_enabled=False, drift_enabled=False, agent_fn=agent, model_name=name)

    # Task 2: Full classification with drift
    print("\n--- Task 2: Full Classification + Drift ---")
    await run_episode(task_id=2, attacker_enabled=False, drift_enabled=True, agent_fn=agent, model_name=name)

    # Task 2: Full classification with attacker + drift
    print("\n--- Task 2: Full Classification + Attacker + Drift ---")
    await run_episode(task_id=2, attacker_enabled=True, drift_enabled=True, agent_fn=agent, model_name=name)

    # Task 3: Multi-turn
    print("\n--- Task 3: Multi-Turn + Drift ---")
    await run_episode(task_id=3, attacker_enabled=False, drift_enabled=True, agent_fn=agent, model_name=name)

    await db.close_db()

if __name__ == "__main__":
    asyncio.run(main())
