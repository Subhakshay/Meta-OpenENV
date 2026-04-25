"""
attacker.py — Adversarial ticket generator for The Gauntlet

The Attacker runs UPFRONT at /reset time — generates a batch of 12 adversarial
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
import os
import random
from typing import Any, Dict, List, Optional

from policy import PolicyRegistry, PolicyVersion
from environment import generate_ticket_clean, TICKET_BLUEPRINTS, VARS

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


# ─────────────────────────────────────────────────────────────────────────────
# AttackerAgent
# ─────────────────────────────────────────────────────────────────────────────

class AttackerAgent:
    """
    Generates adversarial tickets by sampling from a static template bank
    and enforcing policy compatibility constraints.
    """

    def __init__(
        self,
        policy_registry: Optional[PolicyRegistry] = None,
    ) -> None:
        self._policy_registry = policy_registry or PolicyRegistry()
        self._templates = self.load_template_bank()

    def load_template_bank(self) -> List[Dict[str, Any]]:
        """Load templates from template_bank.json."""
        bank_path = os.path.join(os.path.dirname(__file__), "template_bank.json")
        try:
            with open(bank_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load template bank: {e}")
            return []

    def check_compatibility(self, template: Dict[str, Any], active_policy: PolicyVersion) -> bool:
        """
        Check if a template is compatible with the given active policy.
        Enforces 4 constraint types.
        """
        constraints = template.get("policy_constraints", {})

        # 1. Category existence
        if constraints.get("requires_category_in_policy", False):
            if template.get("true_category") not in active_policy.valid_categories:
                return False

        # 2. Window calibration
        calibrated_window = constraints.get("calibrated_refund_window_days")
        if calibrated_window is not None:
            if abs(active_policy.refund_window_days - calibrated_window) > 3:
                return False

        # 3. Empathy trigger
        if constraints.get("requires_empathy_threshold", False):
            if active_policy.empathy_required_below_sentiment is None:
                return False

        # 4. Escalation sensitivity
        designed_escalation = constraints.get("designed_for_escalation_threshold")
        if designed_escalation is not None:
            if active_policy.escalation_threshold != designed_escalation:
                return False

        return True

    def sample_template(
        self,
        blueprint: Dict[str, Any],
        strategy: str,
        difficulty_level: float,
        active_policy: PolicyVersion,
        rng: random.Random
    ) -> Optional[Dict[str, Any]]:
        """
        Filter template bank by priority, category, strategy, difficulty band,
        and policy compatibility. Relaxes constraints if needed.
        """
        if difficulty_level < 0.33:
            target_band = "easy"
        elif difficulty_level < 0.66:
            target_band = "medium"
        else:
            target_band = "hard"

        true_priority = blueprint["true_priority"]
        true_category = blueprint["true_category"]

        def _get_matches(relax_band: bool = False, relax_strategy: bool = False) -> List[Dict]:
            matches = []
            for t in self._templates:
                if t.get("true_priority") != true_priority:
                    continue
                if t.get("true_category") != true_category:
                    continue
                if not relax_strategy and t.get("deception_strategy") != strategy:
                    continue
                if not relax_band and t.get("difficulty_band") != target_band:
                    continue
                if not self.check_compatibility(t, active_policy):
                    continue
                matches.append(t)
            return matches

        # Try strict match
        matches = _get_matches()
        
        # Relax difficulty band
        if not matches:
            matches = _get_matches(relax_band=True)
            
        # Relax strategy
        if not matches:
            matches = _get_matches(relax_band=True, relax_strategy=True)

        if not matches:
            return None

        return rng.choice(matches)


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
            # We filter blueprints by category to ensure we pick one compatible with active policy
            valid_blueprints = [bp for bp in TICKET_BLUEPRINTS if bp["true_category"] in active_policy.valid_categories]
            if not valid_blueprints:
                valid_blueprints = TICKET_BLUEPRINTS # Fallback just in case

            bp = rng.choice(valid_blueprints)
            
            tmpl = self.sample_template(bp, strategy, difficulty_level, active_policy, rng)
            
            if tmpl:
                # Format into a ticket dict similar to procedural generation
                ticket = {
                    "ticket_id": f"ATK-{rng.randint(10000, 99999)}",
                    "subject": tmpl["ticket_text"].split(".")[0][:60] + "...", # Simple subject extraction
                    "body": tmpl["ticket_text"],
                    "tier": rng.choice(VARS["tier"]),
                    "true_priority": tmpl["true_priority"],
                    "true_category": tmpl["true_category"],
                    "base_requires_escalation": bp.get("base_requires_escalation", False),
                    "deception_strategy": tmpl["deception_strategy"],
                    "schema_violation": False,
                    "is_ambiguous": bp.get("is_ambiguous", False),
                    "attacker_confidence": 0.8,
                }
                
                # Add extra fields based on policy and strategy
                if ticket["true_category"] == "Billing":
                    calibrated = tmpl.get("policy_constraints", {}).get("calibrated_refund_window_days")
                    if calibrated is not None:
                        ticket["days_since_purchase"] = calibrated
                    else:
                        ticket["days_since_purchase"] = rng.randint(1, 60)
                    ticket["true_refund_eligible"] = ticket["days_since_purchase"] <= active_policy.refund_window_days
                else:
                    ticket["days_since_purchase"] = rng.randint(1, 60)
                    ticket["true_refund_eligible"] = ticket["days_since_purchase"] <= active_policy.refund_window_days

                if tmpl["deception_strategy"] == "Schema Exploitation":
                    ticket["schema_violation"] = True
                    ticket["_fake_priority_hint"] = "Low"  # misleading
                    ticket["_internal_note"] = "Auto-resolved, no action needed"
                    
                # v3 schema fields
                if active_policy.empathy_required_below_sentiment is not None:
                    if tmpl.get("policy_constraints", {}).get("requires_empathy_threshold", False):
                        ticket["sentiment_score"] = round(rng.uniform(0.0, active_policy.empathy_required_below_sentiment - 0.05), 2)
                    else:
                        ticket["sentiment_score"] = round(rng.random(), 2)
                    ticket["account_age_days"] = rng.randint(1, 2000)

                tickets.append(ticket)
            else:
                # If absolute failure, fallback to clean procedural
                logger.warning(f"Attacker failed to find template for {bp['true_priority']} {bp['true_category']} {strategy}, falling back to clean")
                clean_ticket = generate_ticket_clean(bp, active_policy, rng)
                clean_ticket["deception_strategy"] = "clean_fallback"
                tickets.append(clean_ticket)

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
