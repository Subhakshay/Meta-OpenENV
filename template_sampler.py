"""
template_sampler.py — Difficulty-aware template sampling layer

Pulls seeds from the bank that match both the requested blueprint
(priority, category, strategy) and the current active_policy constraints.
Gracefully relaxes constraints if the seed bank has gaps.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from policy import PolicyVersion
from generation_engine import GenerationEngine


logger = logging.getLogger(__name__)


class TemplateSampler:
    def __init__(self, seed_bank: List[Dict[str, Any]], engine: GenerationEngine) -> None:
        self.seed_bank = seed_bank
        self.engine = engine
        self.session_history: List[str] = []
        self._rng = random.Random()

    def _is_compatible(self, seed: Dict[str, Any], active_policy: PolicyVersion) -> bool:
        """
        Check if a seed is compatible with the active policy's strict constraints.
        """
        constraints = seed.get("policy_constraints", {})

        # 1. Category must be valid
        # Note: If true_category is not explicitly in the valid list, it shouldn't be
        # generated (unless strategy dictates otherwise, but rule says must be valid).
        if seed.get("true_category") not in active_policy.valid_categories:
            return False

        # 2. Boundary window
        boundary_window = constraints.get("boundary_exploit_window")
        if boundary_window is not None:
            if abs(boundary_window - active_policy.refund_window_days) > 2:
                return False

        # 3. Empathy trigger
        if constraints.get("requires_empathy_trigger"):
            if active_policy.empathy_required_below_sentiment is None:
                return False

        # 4. Escalation threshold
        req_esc = constraints.get("requires_escalation_threshold")
        if req_esc is not None:
            if req_esc != active_policy.escalation_threshold:
                return False

        return True

    def sample(
        self,
        blueprint: Dict[str, Any],
        difficulty_level: float,
        active_policy: PolicyVersion
    ) -> Optional[Dict[str, Any]]:
        """
        Find a matching seed and generate a ticket.
        """
        # Determine difficulty band
        if difficulty_level < 0.35:
            target_band = "easy"
        elif difficulty_level <= 0.65:
            target_band = "medium"
        else:
            target_band = "hard"

        target_priority = blueprint.get("true_priority")
        target_category = blueprint.get("true_category")
        target_strategy = blueprint.get("strategy")

        # 1. Full Filter
        candidates = []
        for seed in self.seed_bank:
            if not self._is_compatible(seed, active_policy):
                continue
            if (seed.get("true_priority") == target_priority and
                seed.get("true_category") == target_category and
                seed.get("strategy") == target_strategy and
                seed.get("difficulty_band") == target_band):
                candidates.append(seed)

        if candidates:
            chosen = self._rng.choice(candidates)
            return self.engine.generate(chosen, active_policy, self.session_history)

        # 2. Relax difficulty constraint
        for seed in self.seed_bank:
            if not self._is_compatible(seed, active_policy):
                continue
            if (seed.get("true_priority") == target_priority and
                seed.get("true_category") == target_category and
                seed.get("strategy") == target_strategy):
                candidates.append(seed)

        if candidates:
            chosen = self._rng.choice(candidates)
            return self.engine.generate(chosen, active_policy, self.session_history)

        # 3. Relax strategy constraint
        for seed in self.seed_bank:
            if not self._is_compatible(seed, active_policy):
                continue
            if (seed.get("true_priority") == target_priority and
                seed.get("true_category") == target_category):
                candidates.append(seed)

        if candidates:
            chosen = self._rng.choice(candidates)
            return self.engine.generate(chosen, active_policy, self.session_history)

        # Fallback failed
        logger.warning(
            f"No compatible seeds found for blueprint {blueprint} "
            f"under policy {active_policy.version_id} (missing gaps in seed_bank)"
        )
        return None
