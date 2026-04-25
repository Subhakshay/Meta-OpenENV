"""
drift_scheduler.py — Drift event injector for Shifting Sands

Manages the schedule of policy drift events that fire at specific steps
during an episode. When a drift fires:
  1. The PolicyRegistry switches to the new version
  2. The WorldState records the drift event
  3. A system notice is sent to the Defender agent

Default schedule:
  - Step 4: v1 → v2 (policy tightening)
  - Step 9: v2 → v3 (schema + terminology expansion)
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
# Default schedule
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DRIFT_SCHEDULE = [
    DriftEvent(
        fires_at_step=4,
        from_version="v1",
        to_version="v2",
        drift_types=["policy_drift"],
        notice_text="""=== SYSTEM NOTICE [Step 4] ===
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
        fires_at_step=9,
        from_version="v2",
        to_version="v3",
        drift_types=["policy_drift", "schema_drift", "terminology_drift"],
        notice_text="""=== SYSTEM NOTICE [Step 9] ===
Effective immediately, the following changes are in effect:
- New ticket category added: Security (previously classified as Technical)
- Any ticket involving authentication, access control, or data breach must now use category 'Security'
- Two new ticket fields are now available: sentiment_score (0.0–1.0) and account_age_days
- These fields should inform your escalation urgency and response tone
Please update all future decisions accordingly.
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
        event = scheduler.check_step(4)
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
