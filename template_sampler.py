"""
template_sampler.py — Difficulty-aware template sampling layer

Uses the Jinja2 GenerationEngine to produce tickets that match a requested
blueprint (priority, category, strategy) under the current active policy.

Compatibility is enforced by:
  1. Category must be in the active policy's valid_categories
  2. Boundary exploitation window must align with the policy refund window
  3. Emotional manipulation requires empathy threshold to be active
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from policy import PolicyVersion
from generation_engine import (
    GenerationEngine,
    STRATEGY_DIFFICULTY,
    CATEGORIES,
    PRIORITIES,
    TONES,
)

logger = logging.getLogger(__name__)


class TemplateSampler:
    """
    Bridge between the environment's reconcile logic and the GenerationEngine.
    Enforces policy compatibility before delegating to the engine.
    """

    def __init__(self, engine: Optional[GenerationEngine] = None) -> None:
        self.engine = engine or GenerationEngine()
        self._rng = random.Random()

    def _is_compatible(
        self,
        category: str,
        strategy: str,
        active_policy: PolicyVersion,
    ) -> bool:
        """Check if a category+strategy combo is valid under the active policy."""
        # 1. Category must be in valid set
        if category not in active_policy.valid_categories:
            return False

        # 2. Emotional manipulation requires empathy threshold
        if strategy == "emotional_manipulation":
            if active_policy.empathy_required_below_sentiment is None:
                return False

        return True

    def sample(
        self,
        blueprint: Dict[str, Any],
        difficulty_level: float,
        active_policy: PolicyVersion,
        churn_risk: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a ticket matching the blueprint under the active policy.

        Returns a dict with ticket_string, ground_truth, etc., or None if
        no compatible generation is possible.
        """
        category = blueprint.get("true_category", "Technical")
        strategy = blueprint.get("strategy", "clean")
        priority = blueprint.get("true_priority", "Medium")
        tone = blueprint.get("tone", "neutral")

        # Check compatibility
        if not self._is_compatible(category, strategy, active_policy):
            # Try to fix: if category is invalid, pick a valid one
            valid_cats = list(active_policy.valid_categories)
            if valid_cats:
                category = self._rng.choice(valid_cats)
            else:
                return None

            # Re-check with new category
            if not self._is_compatible(category, strategy, active_policy):
                # Fall back to clean strategy
                strategy = "clean"

        try:
            return self.engine.generate(
                category=category,
                strategy=strategy,
                priority=priority,
                tone=tone,
                active_policy=active_policy,
                churn_risk=churn_risk,
            )
        except Exception as e:
            logger.warning("Generation failed for %s/%s: %s", category, strategy, e)
            return None
