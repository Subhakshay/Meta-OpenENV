"""
policy.py — Dynamic Policy Version & Registry

Defines the PolicyVersion dataclass mirroring the 8 keys in policy_matrix.py,
and a PolicyRegistry that samples versions with a >= 2 field change constraint.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from policy_matrix import POLICY_MATRIX


POLICY_FIELDS = list(POLICY_MATRIX.keys())

@dataclass(frozen=True)
class PolicyVersion:
    version_id: str
    refund_window_days: int
    sla_critical_hours: int
    sla_high_hours: int
    valid_categories: tuple  # frozen for hashability
    required_greeting: Optional[str]
    empathy_required_below_sentiment: Optional[float]
    escalation_threshold: str
    refund_approval_authority: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "version_id": self.version_id,
            "refund_window_days": self.refund_window_days,
            "sla_critical_hours": self.sla_critical_hours,
            "sla_high_hours": self.sla_high_hours,
            "valid_categories": list(self.valid_categories),
            "required_greeting": self.required_greeting,
            "empathy_required_below_sentiment": self.empathy_required_below_sentiment,
            "escalation_threshold": self.escalation_threshold,
            "refund_approval_authority": self.refund_approval_authority,
        }

    def get_field(self, name: str) -> Any:
        return getattr(self, name)


class PolicyRegistry:
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._history: List[PolicyVersion] = []

    def sample_policy(self, version_id: str) -> PolicyVersion:
        """
        Randomly picks one value from each field in the matrix.
        Checks _is_meaningfully_different against history. Resamples up to 10 times.
        """
        candidate = None
        last = self._history[-1] if self._history else None

        for _ in range(10):
            candidate = self._generate_candidate(version_id)
            if last is None:
                break
            if self._is_meaningfully_different(last, candidate):
                break
        
        # After 10 attempts (or if successful), append and return
        self._history.append(candidate)
        return candidate

    def _generate_candidate(self, version_id: str) -> PolicyVersion:
        return PolicyVersion(
            version_id=version_id,
            refund_window_days=self._rng.choice(POLICY_MATRIX["refund_window_days"]),
            sla_critical_hours=self._rng.choice(POLICY_MATRIX["sla_critical_hours"]),
            sla_high_hours=self._rng.choice(POLICY_MATRIX["sla_high_hours"]),
            valid_categories=tuple(self._rng.choice(POLICY_MATRIX["valid_categories"])),
            required_greeting=self._rng.choice(POLICY_MATRIX["required_greeting"]),
            empathy_required_below_sentiment=self._rng.choice(POLICY_MATRIX["empathy_required_below_sentiment"]),
            escalation_threshold=self._rng.choice(POLICY_MATRIX["escalation_threshold"]),
            refund_approval_authority=self._rng.choice(POLICY_MATRIX["refund_approval_authority"]),
        )

    def _is_meaningfully_different(self, p1: PolicyVersion, p2: PolicyVersion) -> bool:
        diffs = sum(1 for field_name in POLICY_FIELDS if getattr(p1, field_name) != getattr(p2, field_name))
        return diffs >= 2

    def get_current_policy(self) -> PolicyVersion:
        """Returns the most recent entry in history."""
        if not self._history:
            raise RuntimeError("No policies in history.")
        return self._history[-1]

    # --- Backward compatibility / Helper methods for other files ---
    
    def get_active(self) -> PolicyVersion:
        return self.get_current_policy()

    def get_previous(self) -> Optional[PolicyVersion]:
        return self._history[-2] if len(self._history) >= 2 else None

    def active_version_id(self) -> str:
        return self.get_current_policy().version_id

    def get_history(self) -> List[PolicyVersion]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()

    @staticmethod
    def get_changed_fields(old: PolicyVersion, new: PolicyVersion) -> Dict[str, tuple]:
        changes = {}
        for field_name in POLICY_FIELDS:
            val_old = old.get_field(field_name)
            val_new = new.get_field(field_name)
            if val_old != val_new:
                changes[field_name] = (val_old, val_new)
        return changes
