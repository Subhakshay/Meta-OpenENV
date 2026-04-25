"""
drift_scheduler.py — Drift event injector for Shifting Sands

Manages the schedule of policy drift events that fire at specific steps
during an episode. When a drift fires:
  1. The PolicyRegistry switches to the new version
  2. The WorldState records the drift event
  3. A system notice is sent to the Defender agent

Default schedule (for 12-step episodes):
  - Step 3:  v1 → v2 (policy tightening)
  - Step 5:  v2 → v3 (schema + category expansion)
  - Step 7:  v3 → v4 (compliance + PII mode)
  - Step 9:  v4 → v5 (retention + tier-based SLA)
  - Step 11: v5 → v6 (full lockdown)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from policy import PolicyRegistry
from world_state import WorldState


# ─────────────────────────────────────────────────────────────────────────────
# DriftEvent dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DriftEvent:
    """A scheduled policy drift event."""
    fires_at_step: int
    from_version: str
    to_version: str
    drift_types: List[str]    # ["policy_drift", "schema_drift", "terminology_drift"]
    notice_text: str           # Pre-written system notice string


# ─────────────────────────────────────────────────────────────────────────────
# Default schedule — 5 drift events across 12 steps
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DRIFT_SCHEDULE = [
    DriftEvent(
        fires_at_step=3,
        from_version="v1",
        to_version="v2",
        drift_types=["policy_drift"],
        notice_text="""=== SYSTEM NOTICE [Step 3] ===
Effective immediately, the following policy changes are in effect:
- Refund eligibility window: 30 days → 14 days
- Critical SLA threshold: 4 hours → 2 hours
- High SLA threshold: 8 hours → 4 hours
- Auto-escalation threshold: $500/min → $250/min
- Customer responses must now begin with a formal greeting
Please update all future decisions accordingly.
=== END NOTICE ===""",
    ),
    DriftEvent(
        fires_at_step=5,
        from_version="v2",
        to_version="v3",
        drift_types=["policy_drift", "schema_drift", "terminology_drift"],
        notice_text="""=== SYSTEM NOTICE [Step 5] ===
Effective immediately, the following changes are in effect:
- New ticket category added: Security (previously classified as Technical)
- Any ticket involving authentication, access control, or data breach must now use category 'Security'
- Two new ticket fields are now available: sentiment_score (0.0–1.0) and account_age_days
- These fields should inform your escalation urgency and response tone
Please update all future decisions accordingly.
=== END NOTICE ===""",
    ),
    DriftEvent(
        fires_at_step=7,
        from_version="v3",
        to_version="v4",
        drift_types=["policy_drift", "compliance_drift"],
        notice_text="""=== SYSTEM NOTICE [Step 7] ===
Effective immediately, the following compliance changes are in effect:
- New ticket category added: Compliance (regulatory, audit, data privacy requests)
- Critical SLA threshold tightened: 2 hours → 1 hour
- High SLA threshold tightened: 4 hours → 3 hours
- Auto-escalation threshold: $250/min → $150/min
- PII redaction is now REQUIRED in all customer responses
- Any ticket containing personal data (SSN, credit card, DOB) must be flagged
- Maximum escalation queue reduced: 5 → 3
WARNING: Sending PII in a response is now a CATASTROPHIC FAILURE.
=== END NOTICE ===""",
    ),
    DriftEvent(
        fires_at_step=9,
        from_version="v4",
        to_version="v5",
        drift_types=["policy_drift", "schema_drift", "terminology_drift"],
        notice_text="""=== SYSTEM NOTICE [Step 9] ===
Effective immediately, the following changes are in effect:
- New ticket category added: Retention (customers threatening to leave or cancel)
- Refund eligibility window: 14 days → 10 days
- High SLA threshold tightened: 3 hours → 2 hours
- Auto-escalation threshold: $150/min → $100/min
- Two new ticket fields available: loyalty_score (0.0–1.0) and lifetime_value (USD)
- Tier-based SLA is now active: Enterprise/Premium get 50% faster response targets
- Retention offers are now enabled for high-loyalty customers threatening to leave
Please update all future decisions accordingly.
=== END NOTICE ===""",
    ),
    DriftEvent(
        fires_at_step=11,
        from_version="v5",
        to_version="v6",
        drift_types=["policy_drift", "schema_drift"],
        notice_text="""=== SYSTEM NOTICE [Step 11] ===
FINAL POLICY UPDATE — Full lockdown mode:
- Refund eligibility window: 10 days → 7 days
- Auto-escalation threshold: $100/min → $75/min
- Two new ticket fields: region and contract_type
- Maximum escalation queue reduced: 3 → 2
- Mandatory acknowledgment: Every response must explicitly acknowledge the customer's stated issue
- All previous rules remain in effect
This is the strictest policy tier. No further changes expected.
=== END NOTICE ===""",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# DriftScheduler class
# ─────────────────────────────────────────────────────────────────────────────

class DriftScheduler:
    """
    Checks each step for scheduled drift events and applies them.

    Usage:
        scheduler = DriftScheduler()
        event = scheduler.check_step(3)
        if event:
            scheduler.apply(event, world_state, policy_registry)
    """

    def __init__(self, schedule: Optional[List[DriftEvent]] = None) -> None:
        if schedule is None:
            schedule = DEFAULT_DRIFT_SCHEDULE
        self._schedule: Dict[int, DriftEvent] = {e.fires_at_step: e for e in schedule}

    def check_step(self, step_number: int) -> Optional[DriftEvent]:
        """Returns the DriftEvent if one fires at this step, else None."""
        return self._schedule.get(step_number)

    def apply(
        self,
        event: DriftEvent,
        world_state: WorldState,
        policy_registry: PolicyRegistry,
    ) -> None:
        """
        Apply a drift event:
        1. Switch PolicyRegistry to the new version
        2. Record the drift event in WorldState
        """
        policy_registry.set_active(event.to_version)
        world_state.record_drift_event(event.to_version)

    def get_all_events(self) -> List[DriftEvent]:
        """Return all scheduled drift events (for logging/display)."""
        return list(self._schedule.values())
