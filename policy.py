"""
policy.py — Policy Registry for The Gauntlet + Shifting Sands

Defines 6 versioned company policies (v1–v6) for a SaaS platform.
Controls refund windows, SLA thresholds, ticket categories, schema fields,
escalation thresholds, greeting requirements, PII handling, and compliance rules.

The PolicyRegistry is the single source of truth for all policy lookups.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PolicyVersion:
    """Immutable snapshot of a company policy configuration."""
    version_id: str
    refund_window_days: int
    sla_critical_hours: int
    sla_high_hours: int
    valid_categories: tuple
    auto_escalate_threshold_per_min: int
    response_greeting_required: bool
    ticket_schema_fields: tuple
    pii_redaction_required: bool = False
    tier_based_sla: bool = False
    mandatory_acknowledgment: bool = False
    max_escalation_queue: int = 5
    retention_offer_enabled: bool = False
    compliance_audit_enabled: bool = False


# ── SaaS-only policy versions ───────────────────────────────────────────────

POLICY_V1 = PolicyVersion(
    version_id="v1",
    refund_window_days=30,
    sla_critical_hours=4,
    sla_high_hours=8,
    valid_categories=("Billing", "Technical"),
    auto_escalate_threshold_per_min=500,
    response_greeting_required=False,
    ticket_schema_fields=("subject", "body", "tier"),
)

POLICY_V2 = PolicyVersion(
    version_id="v2",
    refund_window_days=14,
    sla_critical_hours=2,
    sla_high_hours=4,
    valid_categories=("Billing", "Technical"),
    auto_escalate_threshold_per_min=250,
    response_greeting_required=True,
    ticket_schema_fields=("subject", "body", "tier"),
)

POLICY_V3 = PolicyVersion(
    version_id="v3",
    refund_window_days=14,
    sla_critical_hours=2,
    sla_high_hours=4,
    valid_categories=("Billing", "Technical", "Security"),
    auto_escalate_threshold_per_min=250,
    response_greeting_required=True,
    ticket_schema_fields=("subject", "body", "tier", "sentiment_score", "account_age_days"),
)

POLICY_V4 = PolicyVersion(
    version_id="v4",
    refund_window_days=14,
    sla_critical_hours=1,
    sla_high_hours=3,
    valid_categories=("Billing", "Technical", "Security", "Compliance"),
    auto_escalate_threshold_per_min=150,
    response_greeting_required=True,
    ticket_schema_fields=("subject", "body", "tier", "sentiment_score", "account_age_days"),
    pii_redaction_required=True,
    compliance_audit_enabled=True,
    max_escalation_queue=3,
)

POLICY_V5 = PolicyVersion(
    version_id="v5",
    refund_window_days=10,
    sla_critical_hours=1,
    sla_high_hours=2,
    valid_categories=("Billing", "Technical", "Security", "Compliance", "Retention"),
    auto_escalate_threshold_per_min=100,
    response_greeting_required=True,
    ticket_schema_fields=("subject", "body", "tier", "sentiment_score", "account_age_days",
                          "loyalty_score", "lifetime_value"),
    pii_redaction_required=True,
    tier_based_sla=True,
    retention_offer_enabled=True,
    compliance_audit_enabled=True,
    max_escalation_queue=3,
)

POLICY_V6 = PolicyVersion(
    version_id="v6",
    refund_window_days=7,
    sla_critical_hours=1,
    sla_high_hours=2,
    valid_categories=("Billing", "Technical", "Security", "Compliance", "Retention"),
    auto_escalate_threshold_per_min=75,
    response_greeting_required=True,
    ticket_schema_fields=("subject", "body", "tier", "sentiment_score", "account_age_days",
                          "loyalty_score", "lifetime_value", "region", "contract_type"),
    pii_redaction_required=True,
    tier_based_sla=True,
    mandatory_acknowledgment=True,
    retention_offer_enabled=True,
    compliance_audit_enabled=True,
    max_escalation_queue=2,
)


class PolicyRegistry:
    """Manages the pool of policy versions and tracks which one is active."""

    def __init__(self) -> None:
        self._versions: Dict[str, PolicyVersion] = {
            "v1": POLICY_V1, "v2": POLICY_V2, "v3": POLICY_V3,
            "v4": POLICY_V4, "v5": POLICY_V5, "v6": POLICY_V6,
        }
        self._active: str = "v1"

    def get_active(self) -> PolicyVersion:
        return self._versions[self._active]

    def get_version(self, vid: str) -> PolicyVersion:
        if vid not in self._versions:
            raise KeyError(f"Unknown policy version '{vid}'.")
        return self._versions[vid]

    def set_active(self, vid: str) -> None:
        if vid not in self._versions:
            raise KeyError(f"Cannot activate unknown policy '{vid}'.")
        self._active = vid

    def active_version_id(self) -> str:
        return self._active

    def list_all_versions(self) -> Dict[str, PolicyVersion]:
        return dict(self._versions)

    def reset(self) -> None:
        self._active = "v1"
