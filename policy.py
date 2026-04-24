"""
policy.py — Policy Registry for The Gauntlet + Shifting Sands

Defines versioned company policies (v1/v2/v3) that control:
  - Refund windows, SLA thresholds, ticket categories
  - Schema fields, escalation thresholds, greeting requirements

The PolicyRegistry is the single source of truth for all policy lookups.
Everything else (rewards, drift, environment) reads from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# PolicyVersion dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PolicyVersion:
    """Immutable snapshot of a company policy configuration."""
    version_id: str
    refund_window_days: int
    sla_critical_hours: int
    sla_high_hours: int
    valid_categories: list[str]
    auto_escalate_threshold_per_min: int
    response_greeting_required: bool
    ticket_schema_fields: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Named policy versions
# ─────────────────────────────────────────────────────────────────────────────

POLICY_V1 = PolicyVersion(
    version_id="v1",
    refund_window_days=30,
    sla_critical_hours=4,
    sla_high_hours=8,
    valid_categories=["Billing", "Technical", "Shipping"],
    auto_escalate_threshold_per_min=500,
    response_greeting_required=False,
    ticket_schema_fields=["subject", "body", "tier"],
)

POLICY_V2 = PolicyVersion(
    version_id="v2",
    refund_window_days=14,
    sla_critical_hours=2,
    sla_high_hours=4,
    valid_categories=["Billing", "Technical", "Shipping"],
    auto_escalate_threshold_per_min=250,
    response_greeting_required=True,
    ticket_schema_fields=["subject", "body", "tier"],
)

POLICY_V3 = PolicyVersion(
    version_id="v3",
    refund_window_days=14,
    sla_critical_hours=2,
    sla_high_hours=4,
    valid_categories=["Billing", "Technical", "Shipping", "Security"],
    auto_escalate_threshold_per_min=250,
    response_greeting_required=True,
    ticket_schema_fields=["subject", "body", "tier", "sentiment_score", "account_age_days"],
)


# ─────────────────────────────────────────────────────────────────────────────
# PolicyRegistry — runtime container
# ─────────────────────────────────────────────────────────────────────────────

class PolicyRegistry:
    """
    Manages the pool of policy versions and tracks which one is active.

    Usage:
        registry = PolicyRegistry()
        policy = registry.get_active()       # returns PolicyVersion for v1
        registry.set_active("v2")            # switch after drift event
        all_versions = registry.list_all_versions()  # used by hallucination checker
    """

    def __init__(self) -> None:
        self._versions: Dict[str, PolicyVersion] = {
            "v1": POLICY_V1,
            "v2": POLICY_V2,
            "v3": POLICY_V3,
        }
        self._active: str = "v1"

    def get_active(self) -> PolicyVersion:
        """Return the currently active policy version."""
        return self._versions[self._active]

    def get_version(self, vid: str) -> PolicyVersion:
        """Return a specific policy version by ID. Raises KeyError if not found."""
        if vid not in self._versions:
            raise KeyError(f"Unknown policy version '{vid}'. Known: {list(self._versions.keys())}")
        return self._versions[vid]

    def set_active(self, vid: str) -> None:
        """Set the active policy version. Raises KeyError if vid is invalid."""
        if vid not in self._versions:
            raise KeyError(f"Cannot activate unknown policy '{vid}'. Known: {list(self._versions.keys())}")
        self._active = vid

    def active_version_id(self) -> str:
        """Return the version ID string of the active policy."""
        return self._active

    def list_all_versions(self) -> Dict[str, PolicyVersion]:
        """
        Return the full dict of all registered policies.

        Used by the reward calculator's hallucination checker to verify
        whether a rule cited by the agent actually exists in any version.
        """
        return dict(self._versions)

    def reset(self) -> None:
        """Reset the active policy to v1 (called on episode reset)."""
        self._active = "v1"
