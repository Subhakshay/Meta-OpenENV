"""
inference.py — Baseline inference script for CustomerSupportEnv v2

Assignment requirements:
  - Named `inference.py`, placed in root directory
  - Uses OpenAI-compatible client for all LLM calls
  - Uses API_BASE_URL, MODEL, API_KEY environment variables
  - Must complete in <20 minutes on 2 vCPU / 8 GB machine

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL="gpt-4o-mini"
    export API_KEY="sk-..."
    python inference.py
"""

import os
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from environment import (
    CustomerSupportEnv, Action, Observation,
    Priority, Category, ActionType, run_episode,
)

# ── OpenAI-compatible client (assignment requirement) ─────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

try:
    from openai import OpenAI
    _client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    HAS_LLM = bool(HF_TOKEN)
except Exception:
    HAS_LLM = False
    _client = None  # type: ignore

if not HAS_LLM:
    print("[INFO] No HF_TOKEN set — running rule-based agent only.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule-based fallback agent
# ═══════════════════════════════════════════════════════════════════════════════

_PRIORITY_SIGNALS: Dict[Priority, List[str]] = {
    Priority.CRITICAL: [
        "production down", "production is down", "completely down", "system down",
        "data loss", "outage", "emergency", "critical", "urgent:",
        "50,000", "50000", "all users locked",
    ],
    Priority.HIGH: [
        "cannot log in", "can't log in", "cannot login", "can't login",
        "charged twice", "duplicate charge", "hasn't arrived", "not arrived",
        "wrong item", "locked out", "need access restored",
    ],
    Priority.MEDIUM: [
        "feature", "rate limit", "sometimes", "occasionally", "slow",
        "billing looks", "not sure", "partial refund", "service issues",
        "acting weird", "seems off",
    ],
    Priority.LOW: [
        "question about", "interested in", "feedback", "love the product",
        "how do i", "how to", "quick question", "no rush",
    ],
}

_CATEGORY_SIGNALS: Dict[Category, List[str]] = {
    Category.BILLING:   ["billing", "charge", "payment", "invoice", "plan", "upgrade", "subscription", "billed", "cost"],
    Category.TECHNICAL: ["api", "rate limit", "feature", "export", "production", "down", "bug", "error", "loading", "500", "dashboard", "webhook", "integration", "sync"],
    Category.ACCOUNT:   ["login", "log in", "credentials", "password", "account", "access", "2fa", "two-factor", "locked out"],
    Category.SHIPPING:  ["order", "arrived", "delivery", "ship", "tracking", "package", "item shipped"],
    Category.REFUND:    ["refund", "reimburse", "return", "credit", "compensation"],
    Category.GENERAL:   ["feedback", "question", "how do i", "how to", "quick question"],
}

_TIER_GREETING = {
    "enterprise": "Dear Valued Enterprise Customer",
    "pro":        "Hello",
    "free":       "Hi there",
}

_ACTION_COMMITMENT = {
    Priority.CRITICAL: "Our engineering team has been immediately alerted and is actively investigating. We will provide updates every 30 minutes.",
    Priority.HIGH:     "We are treating this as high priority and our team will investigate and respond within 2 hours.",
    Priority.MEDIUM:   "Our team will look into this and get back to you within 1–2 business days.",
    Priority.LOW:      "We are happy to help and will follow up with you shortly.",
}


def _classify_priority(text: str) -> Priority:
    t = text.lower()
    for p in [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
        if any(kw in t for kw in _PRIORITY_SIGNALS[p]):
            return p
    return Priority.MEDIUM


def _classify_category(text: str) -> Category:
    t = text.lower()
    for c in [Category.BILLING, Category.TECHNICAL, Category.ACCOUNT,
              Category.SHIPPING, Category.REFUND, Category.GENERAL]:
        if any(kw in t for kw in _CATEGORY_SIGNALS[c]):
            return c
    return Category.GENERAL


def rule_based_agent(obs: Observation) -> Action:
    """
    Keyword-based heuristic agent.
    Handles multi-turn task_3: asks a clarifying question when ticket is ambiguous,
    then resolves after receiving the customer reply.
    """
    # Multi-turn: if we're awaiting a reply, just classify now
    if obs.task_id == "task_3_full_triage":
        combined = f"{obs.subject} {obs.body}".lower()
        # Detect ambiguous ticket by vague language
        vague_signals = ["something isn't", "seems off", "acting weird",
                          "not sure", "looks different", "the thing"]
        is_vague = any(s in combined for s in vague_signals)

        if is_vague and not obs.awaiting_clarification:
            # First turn: ask a clarifying question
            return Action(
                action_type=ActionType.ASK,
                clarifying_question=(
                    "Could you please clarify which specific feature or section is affected? "
                    "Also, could you describe the steps to reproduce the issue and when it started?"
                ),
            )

    # Full text for classification
    full_text = f"{obs.subject} {obs.body}"
    if obs.customer_reply:
        full_text += f" {obs.customer_reply}"

    priority = _classify_priority(full_text)
    category = _classify_category(full_text)
    escalate = priority == Priority.CRITICAL

    greeting    = _TIER_GREETING.get(obs.customer_tier, "Hello")
    commitment  = _ACTION_COMMITMENT[priority]

    response = (
        f"{greeting},\n\n"
        f"Thank you for reaching out regarding '{obs.subject}'.\n\n"
        f"{commitment}\n\n"
        f"We have assigned this ticket {priority.value.upper()} priority "
        f"and routed it to our {category.value} team.\n\n"
        f"Best regards,\nCustomer Support Team"
    )

    return Action(
        action_type=ActionType.CLASSIFY,
        assign_priority=priority,
        assign_category=category,
        response_text=response,
        escalate=escalate,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM agent  (OpenAI-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are an expert customer support triage agent operating in a multi-turn environment.

You can take one of two action types per turn:

ACTION TYPE 1 — "ask": Request clarification when the ticket is ambiguous.
Use this when the customer's issue is vague and you cannot confidently classify it.
Return JSON:
{
  "action_type": "ask",
  "clarifying_question": "<your specific, relevant question>"
}

ACTION TYPE 2 — "classify": Classify and respond to the ticket.
Use this when you have enough information (either the ticket is clear, or the customer replied).
Return JSON:
{
  "action_type": "classify",
  "assign_priority": "low" | "medium" | "high" | "critical",
  "assign_category": "billing" | "technical" | "account" | "shipping" | "general" | "refund",
  "response_text": "<professional reply to the customer>",
  "escalate": true | false
}

PRIORITY RULES:
- critical: production down, data loss, full outage, emergency (→ escalate: true)
- high:     cannot login, duplicate charges, overdue orders, enterprise blockers
- medium:   bugs with workarounds, ambiguous billing questions, partial refunds
- low:      general questions, pricing inquiries, general feedback

WORLD STATE: Your decisions affect company_balance, customer_churn_risk, and sla_breach_count.
- Unnecessary escalations drain resources and raise churn risk
- Bad or short responses spike churn risk
- Missing a CRITICAL ticket causes an SLA breach

RESPONSE QUALITY:
- Greet the customer by tier (enterprise: "Dear Valued Enterprise Customer", pro: "Hello", free: "Hi there")
- Reference the ticket subject explicitly
- Use vocabulary relevant to the issue category
- Commit to a specific next action
- End with a professional closing

Return ONLY the JSON object. No markdown, no explanation."""


def llm_agent(obs: Observation) -> Action:
    """LLM-powered agent using OpenAI-compatible API."""
    if not HAS_LLM or _client is None:
        return rule_based_agent(obs)

    # Build the user message including world state and dialogue history
    world = obs.world_state
    context_parts = [
        f"Task: {obs.task_id}",
        f"Hint: {obs.hint or 'Full triage required.'}",
        "",
        f"Ticket ID:     {obs.ticket_id}",
        f"Customer Tier: {obs.customer_tier}",
        f"Sentiment:     {obs.sentiment}",
        f"Subject:       {obs.subject}",
        f"Body:\n{obs.body}",
    ]

    if obs.clarification_history:
        context_parts.append("\nDialogue History:")
        for turn in obs.clarification_history:
            context_parts.append(f"  Agent:    {turn.get('agent', '')}")
            context_parts.append(f"  Customer: {turn.get('customer', '')}")

    if obs.awaiting_clarification:
        context_parts.append("\nThe customer has replied to your question. Now classify and resolve.")

    context_parts += [
        "",
        "Current World State (your decisions affect these):",
        f"  company_balance:      ${world.get('company_balance', 10000):.2f}",
        f"  customer_churn_risk:  {world.get('customer_churn_risk', 0.0):.2f}",
        f"  escalation_queue:     {world.get('escalation_queue', 0)} / {world.get('escalation_capacity', 3)}",
        f"  sla_breach_count:     {world.get('sla_breach_count', 0)}",
        f"  queue_size:           {obs.queue_size} tickets remaining",
    ]

    user_msg = "\n".join(context_parts)

    try:
        resp = _client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=600,
        )
        raw  = resp.choices[0].message.content.strip()
        raw  = re.sub(r"^```(?:json)?\s*", "", raw)
        raw  = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)

        action_type = ActionType(data.get("action_type", "classify"))

        if action_type == ActionType.ASK:
            return Action(
                action_type=ActionType.ASK,
                clarifying_question=data.get("clarifying_question", "Could you clarify the issue?"),
            )
        else:
            return Action(
                action_type=ActionType.CLASSIFY,
                assign_priority=Priority(data["assign_priority"]),
                assign_category=Category(data["assign_category"]),
                response_text=data.get("response_text", "Thank you for contacting us. We will look into this shortly."),
                escalate=bool(data.get("escalate", False)),
            )
    except Exception as exc:
        print(f"  [LLM error: {exc}] — falling back to rule-based")
        return rule_based_agent(obs)


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation runner
# ═══════════════════════════════════════════════════════════════════════════════

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def _format_action(action: Action) -> str:
    d = {"type": action.action_type.value}
    if action.action_type == ActionType.ASK:
        d["question"] = action.clarifying_question
    else:
        d["priority"] = action.assign_priority.value
        d["category"] = action.assign_category.value
        d["escalate"] = action.escalate
    return json.dumps(d)

def run_episode_with_logging(env: CustomerSupportEnv, agent_fn: Callable, task_id: str, model_name: str) -> float:
    log_start(task=task_id, env="customer-support-triage", model=model_name)
    
    obs = env.reset()
    rewards = []
    steps = 0
    error = None
    
    while True:
        try:
            action = agent_fn(obs)
            action_str = _format_action(action)
        except Exception as e:
            action = Action()
            action_str = "error"
            error = str(e)
            
        try:
            obs, reward, done, info = env.step(action)
            r_val = reward.value
            error = None
        except Exception as e:
            r_val = 0.0
            done = True
            error = str(e)
            
        rewards.append(r_val)
        steps += 1
        
        log_step(step=steps, action=action_str, reward=r_val, done=done, error=error)
        
        if done:
            break
            
    score = sum(rewards) / max(steps, 1)
    score = min(max(score, 0.0), 1.0)
    success = score >= 0.5
    
    log_end(success=success, steps=steps, score=score, rewards=rewards)
    return score

def evaluate_agent(
    agent_fn: Callable[[Observation], Action],
    agent_name: str,
    seeds: tuple = (42, 123, 7),
) -> None:
    for task_id in CustomerSupportEnv.TASK_IDS:
        for seed in seeds:
            env = CustomerSupportEnv(task_id=task_id, seed=seed)
            run_episode_with_logging(env, agent_fn, task_id, agent_name)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if HAS_LLM:
        evaluate_agent(llm_agent, MODEL_NAME, seeds=(42, 123, 7))
    else:
        evaluate_agent(rule_based_agent, "Rule-Based", seeds=(42, 123, 7))
