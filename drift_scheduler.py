"""
drift_scheduler.py — Randomized, difficulty-aware drift scheduler

Generates a fully resolved list of DriftEvents before the episode begins.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from policy import PolicyRegistry, PolicyVersion


@dataclass
class DriftEvent:
    fires_at_step: int
    new_policy: PolicyVersion


class DriftScheduler:
    def __init__(self, episode_length: int, registry: PolicyRegistry, seed: Optional[int] = None) -> None:
        self.episode_length = episode_length
        self.registry = registry
        self._rng = random.Random(seed)

    def schedule_episode(self, difficulty_level: float) -> List[DriftEvent]:
        """
        Return a fully resolved list of DriftEvent objects.
        No drift logic runs during episode execution.
        """
        if difficulty_level < 0.33:
            count = 1
        elif difficulty_level <= 0.66:
            count = 2
        else:
            count = 3

        earliest = max(2, int(self.episode_length * 0.2))
        latest = self.episode_length - 2
        
        candidates = list(range(earliest, latest + 1))
        selected_steps = []
        
        for _ in range(count):
            if not candidates:
                break
            step = self._rng.choice(candidates)
            selected_steps.append(step)
            # Remove step and anything within 2 steps
            candidates = [c for c in candidates if abs(c - step) > 2]
            
        selected_steps.sort()
        
        events = []
        for s in selected_steps:
            new_policy = self.registry.sample_policy(f"dyn-{s:03d}")
            events.append(DriftEvent(fires_at_step=s, new_policy=new_policy))
            
        return events


# ─────────────────────────────────────────────────────────────────────────────
# System notice generator
# ─────────────────────────────────────────────────────────────────────────────

_FIELD_LABELS = {
    "refund_window_days": "Refund eligibility window",
    "sla_critical_hours": "Critical ticket SLA",
    "sla_high_hours": "High ticket SLA",
    "valid_categories": "Valid ticket categories",
    "required_greeting": "Required greeting",
    "empathy_required_below_sentiment": "Empathy sentiment threshold",
    "escalation_threshold": "Escalation threshold",
    "refund_approval_authority": "Refund approval authority",
}


def _fmt_value(field: str, val) -> str:
    """Format a policy field value for human-readable display."""
    if val is None:
        return "None (not required)"
    if field == "refund_window_days":
        return f"{val} days"
    if field in ("sla_critical_hours", "sla_high_hours"):
        return f"{val} hours"
    if field == "valid_categories":
        cats = list(val) if isinstance(val, tuple) else val
        return ", ".join(cats)
    if field == "empathy_required_below_sentiment":
        return f"{val}"
    return str(val).replace("_", " ").title()


def build_drift_notice(
    old_policy: PolicyVersion,
    new_policy: PolicyVersion,
    step: int,
) -> str:
    """
    Build a human-readable system notice describing what changed
    between two policy versions.
    """
    changes = PolicyRegistry.get_changed_fields(old_policy, new_policy)
    if not changes:
        return ""

    lines = [f"=== SYSTEM NOTICE [Step {step}] ==="]
    lines.append("Effective immediately, the following policy changes are in effect:")

    for field_name, (old_val, new_val) in changes.items():
        label = _FIELD_LABELS.get(field_name, field_name)
        old_str = _fmt_value(field_name, old_val)
        new_str = _fmt_value(field_name, new_val)
        lines.append(f"- {label}: {old_str} → {new_str}")

    lines.append("Please update all future decisions accordingly.")
    lines.append("=== END NOTICE ===")
    return "\n".join(lines)
