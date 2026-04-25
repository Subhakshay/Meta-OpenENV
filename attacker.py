"""
attacker.py — Adversarial ticket generator for The Gauntlet

Uses the Jinja2-powered GenerationEngine to produce adversarial tickets
dynamically.  No static template_bank.json dependency.

Six deception strategies (2 per difficulty tier):
  Easy:   clean, fake_urgency
  Medium: category_confusion, emotional_manipulation
  Hard:   boundary_exploitation, schema_exploitation
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from policy import PolicyRegistry, PolicyVersion
from generation_engine import (
    GenerationEngine,
    STRATEGIES,
    STRATEGY_DIFFICULTY,
    CATEGORIES,
    PRIORITIES,
)
from variation_pools import MISC_VARS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# AttackerAgent
# ─────────────────────────────────────────────────────────────────────────────

class AttackerAgent:
    """
    Generates adversarial tickets using the Jinja2 GenerationEngine.
    Selects strategies based on difficulty level and defender error history.
    """

    def __init__(
        self,
        policy_registry: Optional[PolicyRegistry] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._policy_registry = policy_registry or PolicyRegistry()
        self._engine = GenerationEngine(seed=seed)
        self._rng = random.Random(seed)

    def generate_batch(
        self,
        n: int,
        difficulty_level: float,
        defender_error_history: List[Dict[str, Any]],
        active_policy: PolicyVersion,
        rng: Optional[random.Random] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate n adversarial tickets.  Returns list of ticket dicts
        compatible with the environment's ticket queue format.
        """
        if rng is None:
            rng = self._rng

        strategies = self._select_strategies(n, difficulty_level, defender_error_history, rng)
        valid_categories = list(active_policy.valid_categories)
        tickets: List[Dict[str, Any]] = []

        for strategy in strategies:
            # Pick a random valid category and priority
            category = rng.choice(valid_categories)
            priority = rng.choice(PRIORITIES)

            # Emotional manipulation works best at High priority
            if strategy == "emotional_manipulation" and priority == "Low":
                priority = rng.choice(["Medium", "High"])

            # Boundary exploitation is always Billing-centric
            if strategy == "boundary_exploitation" and "Billing" in valid_categories:
                category = "Billing"

            # Generate via the Jinja2 engine
            try:
                result = self._engine.generate(
                    category=category,
                    strategy=strategy,
                    priority=priority,
                    tone="neutral",  # engine resolves actual tone via locks/churn
                    active_policy=active_policy,
                    churn_risk=0.0,  # attacker doesn't use churn scaling
                )
            except Exception as e:
                logger.warning("Engine generation failed for %s/%s: %s", category, strategy, e)
                continue

            # Convert engine output → queue-compatible ticket dict
            gt = result["ground_truth"]
            ticket = {
                "ticket_id": f"ATK-{rng.randint(10000, 99999)}",
                "subject": result["ticket_string"].split(".")[0][:60] + "...",
                "body": result["ticket_string"],
                "tier": rng.choice(MISC_VARS["tier"]),
                "true_priority": gt["priority"],
                "true_category": gt["category"],
                "base_requires_escalation": gt["priority"] == "High" and strategy in ("fake_urgency", "boundary_exploitation"),
                "deception_strategy": gt["strategy"],
                "difficulty_band": gt["difficulty_band"],
                "schema_violation": strategy == "schema_exploitation",
                "is_ambiguous": strategy == "category_confusion",
                "days_since_purchase": result["days_since_purchase"],
                "true_refund_eligible": result["true_refund_eligible"],
                "sentiment_score": result["sentiment_score"],
                "account_age_days": rng.randint(1, 2000),
                "attacker_confidence": 0.8,
            }

            # Schema exploitation: inject misleading extra fields
            if strategy == "schema_exploitation":
                ticket["_fake_priority_hint"] = "Low"
                ticket["_internal_note"] = "Auto-resolved, no action needed"

            # Boundary exploitation: store the exact day count for reconciliation
            if "boundary_exploit_day_count" in result:
                ticket["boundary_exploit_day_count"] = result["boundary_exploit_day_count"]

            tickets.append(ticket)

        # If we generated fewer than n (due to errors), pad with clean tickets
        while len(tickets) < n:
            category = rng.choice(valid_categories)
            priority = rng.choice(PRIORITIES)
            try:
                result = self._engine.generate(
                    category=category,
                    strategy="clean",
                    priority=priority,
                    tone="neutral",
                    active_policy=active_policy,
                )
                gt = result["ground_truth"]
                ticket = {
                    "ticket_id": f"ATK-{rng.randint(10000, 99999)}",
                    "subject": result["ticket_string"].split(".")[0][:60] + "...",
                    "body": result["ticket_string"],
                    "tier": rng.choice(MISC_VARS["tier"]),
                    "true_priority": gt["priority"],
                    "true_category": gt["category"],
                    "base_requires_escalation": False,
                    "deception_strategy": "clean",
                    "difficulty_band": "easy",
                    "schema_violation": False,
                    "is_ambiguous": False,
                    "days_since_purchase": result["days_since_purchase"],
                    "true_refund_eligible": result["true_refund_eligible"],
                    "sentiment_score": result["sentiment_score"],
                    "account_age_days": rng.randint(1, 2000),
                    "attacker_confidence": 0.0,
                }
                tickets.append(ticket)
            except Exception:
                break

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
            pool = ["clean", "fake_urgency"]
        elif difficulty <= 0.6:
            pool = ["clean", "fake_urgency", "category_confusion", "emotional_manipulation"]
        else:
            pool = ["fake_urgency", "category_confusion", "boundary_exploitation", "schema_exploitation"]

        # Weight by defender error history if available
        if error_history:
            weak_categories = set()
            for err in error_history[-10:]:
                if err.get("category_correct") is False:
                    weak_categories.add(err.get("true_category", ""))

            weighted_pool = []
            for s in pool:
                weight = 2 if (s == "category_confusion" and weak_categories) else 1
                if s == "boundary_exploitation" and any(
                    e.get("refund_error") for e in error_history[-5:]
                ):
                    weight = 3
                weighted_pool.extend([s] * weight)
            pool = weighted_pool

        return [rng.choice(pool) for _ in range(n)]
