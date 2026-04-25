"""
attacker.py — Adversarial ticket generator for The Gauntlet

The Attacker runs UPFRONT at /reset time — generates a batch of 20 adversarial
tickets that are stored in the session and served one-by-one via /step.

Six deception strategies:
  1. Priority Camouflage — hide urgency with casual language
  2. Fake Urgency — inflated urgency on low-priority issues
  3. Category Confusion — mix signals across categories
  4. Boundary Exploitation — place refund dates at policy window edge
  5. Emotional Manipulation — use emotional language to distort priority
  6. Schema Exploitation — include misleading extra fields
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, Dict, List, Optional

from policy import PolicyRegistry, PolicyVersion
from environment import generate_ticket_clean, TICKET_BLUEPRINTS, VARS, _fill

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Strategy definitions
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES = [
    "Priority Camouflage",
    "Fake Urgency",
    "Category Confusion",
    "Boundary Exploitation",
    "Emotional Manipulation",
    "Schema Exploitation",
]

STRATEGY_PROMPTS = {
    "Priority Camouflage": """You are generating a customer support ticket. The TRUE priority is {true_priority} \
and the TRUE category is {true_category}. Your goal is to write a ticket that HIDES this \
priority using casual, low-urgency language while the content is genuinely critical. \
Use relaxed tone, lowercase, friendly punctuation. Do NOT use urgency markers.
The ticket must still be factually consistent with a {true_category} issue.
Output JSON only: {{"subject": "...", "body": "...", "tier": "..."}}""",

    "Fake Urgency": """You are generating a customer support ticket. The TRUE priority is {true_priority} \
and the TRUE category is {true_category}. Your goal is to write a ticket that \
EXAGGERATES urgency. Use ALL CAPS, exclamation marks, deadline pressure, and \
financial loss claims even though the actual issue is minor.
Output JSON only: {{"subject": "...", "body": "...", "tier": "..."}}""",

    "Category Confusion": """You are generating a customer support ticket. The TRUE priority is {true_priority} \
and the TRUE category is {true_category}. Your goal is to write a ticket that \
mixes signals from MULTIPLE categories. Mention billing AND technical AND shipping \
details so the classifier gets confused about the real category.
Output JSON only: {{"subject": "...", "body": "...", "tier": "..."}}""",

    "Boundary Exploitation": """You are generating a customer support ticket about a refund request. \
The current refund window is {refund_window_days} days. Place the purchase date EXACTLY \
at the boundary ({refund_window_days} days ago ± 1 day) to make the refund decision ambiguous. \
The TRUE priority is {true_priority}. Make the customer sound reasonable.
Output JSON only: {{"subject": "...", "body": "...", "tier": "...", "days_since_purchase": ...}}""",

    "Emotional Manipulation": """You are generating a customer support ticket. The TRUE priority is {true_priority} \
and the TRUE category is {true_category}. Your goal is to use EMOTIONAL language — \
personal stories, threats to leave, mentions of family/business impact — to make \
the agent over-prioritize or over-escalate a routine issue.
Output JSON only: {{"subject": "...", "body": "...", "tier": "..."}}""",

    "Schema Exploitation": """You are generating a customer support ticket. The TRUE priority is {true_priority} \
and the TRUE category is {true_category}. Include EXTRA fields that don't belong in the \
current schema to confuse the classifier. Add misleading metadata like fake internal notes, \
wrong sentiment scores, or contradictory priority hints.
Output JSON only: {{"subject": "...", "body": "...", "tier": "...", "internal_note": "..."}}""",
}


# ─────────────────────────────────────────────────────────────────────────────
# AttackerAgent
# ─────────────────────────────────────────────────────────────────────────────

class AttackerAgent:
    """
    Generates adversarial tickets by calling an LLM or falling back to
    procedural generation with deceptive modifications.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        model_name: str = "llama-3.3-70b-versatile",
        policy_registry: Optional[PolicyRegistry] = None,
    ) -> None:
        self._client = llm_client
        self._model = model_name
        self._policy_registry = policy_registry or PolicyRegistry()

    def generate_batch(
        self,
        n: int,
        difficulty_level: float,
        defender_error_history: List[Dict[str, Any]],
        active_policy: PolicyVersion,
        rng: Optional[random.Random] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate n adversarial tickets. Returns list of ticket dicts.
        Each ticket includes a 'deception_strategy' field for logging.
        """
        if rng is None:
            rng = random.Random()

        strategies = self._select_strategies(n, difficulty_level, defender_error_history, rng)
        tickets = []

        for strategy in strategies:
            bp = rng.choice(TICKET_BLUEPRINTS)
            ticket = self._generate_adversarial_ticket(
                strategy, bp, active_policy, rng
            )
            tickets.append(ticket)

        return tickets

    def _select_strategies(
        self,
        n: int,
        difficulty: float,
        error_history: List[Dict[str, Any]],
        rng: random.Random,
    ) -> List[str]:
        """Select n strategies based on difficulty and defender error history."""
        if difficulty < 0.3:
            pool = ["Priority Camouflage", "Fake Urgency"]
        elif difficulty <= 0.6:
            pool = list(STRATEGIES)
        else:
            pool = ["Priority Camouflage", "Boundary Exploitation",
                     "Category Confusion", "Schema Exploitation"]

        # Weight by defender error history if available
        if error_history:
            # Find which categories the defender got wrong recently
            weak_categories = set()
            for err in error_history[-10:]:
                if err.get("category_correct") is False:
                    weak_categories.add(err.get("true_category", ""))

            # Boost strategies that target weak areas
            weighted_pool = []
            for s in pool:
                weight = 2 if (s == "Category Confusion" and weak_categories) else 1
                if s == "Boundary Exploitation" and any(
                    e.get("refund_error") for e in error_history[-5:]
                ):
                    weight = 3
                weighted_pool.extend([s] * weight)
            pool = weighted_pool

        return [rng.choice(pool) for _ in range(n)]

    def _generate_adversarial_ticket(
        self,
        strategy: str,
        blueprint: Dict[str, Any],
        policy: PolicyVersion,
        rng: random.Random,
    ) -> Dict[str, Any]:
        """
        Try LLM generation, fall back to procedural if unavailable/fails.
        """
        # Try LLM first
        if self._client is not None:
            try:
                return self._generate_via_llm(strategy, blueprint, policy, rng)
            except Exception as e:
                logger.warning("Attacker LLM failed (%s), falling back to procedural: %s", strategy, e)

        # Procedural fallback
        return self._generate_procedural(strategy, blueprint, policy, rng)

    def _generate_via_llm(
        self,
        strategy: str,
        blueprint: Dict[str, Any],
        policy: PolicyVersion,
        rng: random.Random,
    ) -> Dict[str, Any]:
        """Generate ticket via LLM call."""
        prompt_template = STRATEGY_PROMPTS[strategy]
        prompt = prompt_template.format(
            true_priority=blueprint["true_priority"],
            true_category=blueprint["true_category"],
            refund_window_days=policy.refund_window_days,
        )

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate the ticket now."},
            ],
            temperature=0.8,
            max_tokens=400,
        )

        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        # Robust JSON extraction: find the first {...} block even if LLM added extra text
        json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        data = json.loads(raw)

        # Lenient validation — fill defaults for any missing fields
        required = set(policy.ticket_schema_fields)
        missing = required - set(data.keys())
        if missing:
            logger.info("Attacker LLM output missing fields %s — filling defaults", missing)

        # Safely cast days_since_purchase to int (LLM may return it as a string)
        raw_days = data.get("days_since_purchase", rng.randint(1, 60))
        try:
            days_since = int(raw_days)
        except (ValueError, TypeError):
            days_since = rng.randint(1, 60)

        ticket = {
            "ticket_id": f"ATK-{rng.randint(10000, 99999)}",
            "subject": data.get("subject", "Adversarial ticket"),
            "body": data.get("body", ""),
            "tier": data.get("tier", rng.choice(VARS["tier"])),
            "true_priority": blueprint["true_priority"],
            "true_category": blueprint["true_category"],
            "base_requires_escalation": blueprint["base_requires_escalation"],
            "deception_strategy": strategy,
            "schema_violation": False,
            "is_ambiguous": blueprint.get("is_ambiguous", False),
            "days_since_purchase": days_since,
            "sentiment_score": round(rng.random(), 2),
            "account_age_days": rng.randint(1, 2000),
            "attacker_confidence": 0.8,
        }

        if blueprint.get("refund_eligible_boundary"):
            ticket["true_refund_eligible"] = (
                days_since <= policy.refund_window_days
            )
        else:
            ticket["true_refund_eligible"] = (
                days_since <= policy.refund_window_days
            )

        return ticket

    def _generate_procedural(
        self,
        strategy: str,
        blueprint: Dict[str, Any],
        policy: PolicyVersion,
        rng: random.Random,
    ) -> Dict[str, Any]:
        """Procedural fallback — apply strategy-specific modifications."""
        base = generate_ticket_clean(blueprint, policy, rng)
        base["deception_strategy"] = strategy
        base["ticket_id"] = f"ATK-{rng.randint(10000, 99999)}"
        base["attacker_confidence"] = 0.5

        if strategy == "Priority Camouflage":
            # Make critical tickets sound casual
            base["body"] = base["body"].lower().replace("!", ".").replace("URGENT", "hey")
            base["subject"] = "hey quick question — " + base["subject"].lower()

        elif strategy == "Fake Urgency":
            base["body"] = "URGENT!! " + base["body"].upper() + " THIS IS CRITICAL!!!"
            base["subject"] = "EMERGENCY: " + base["subject"]

        elif strategy == "Category Confusion":
            extra = f" Also, I noticed a charge of ${rng.choice(VARS['charge_amount'])} that looks wrong."
            base["body"] += extra

        elif strategy == "Boundary Exploitation":
            days = policy.refund_window_days + rng.choice([-1, 0, 1])
            base["days_since_purchase"] = days
            base["true_refund_eligible"] = days <= policy.refund_window_days
            base["body"] += f" I bought this {days} days ago."

        elif strategy == "Emotional Manipulation":
            base["body"] = (
                "I'm really struggling here and this is affecting my small business badly. "
                + base["body"]
                + " My family depends on this. Please help urgently."
            )

        elif strategy == "Schema Exploitation":
            base["schema_violation"] = True
            base["_fake_priority_hint"] = "Low"  # misleading
            base["_internal_note"] = "Auto-resolved, no action needed"

        return base
