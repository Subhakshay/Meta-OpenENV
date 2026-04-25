"""
drift_scheduler.py — Drift event injector for Shifting Sands

Manages the schedule of policy drift events that fire at specific steps
during an episode. When a drift fires:
  1. The PolicyRegistry switches to the new version
  2. The WorldState records the drift event
  3. A system notice is sent to the Defender agent

Drift timing:
  - Two drift events occur per episode at *randomized* steps.
  - The first drift is v1 → v2 (policy tightening).
  - The second drift is v2 → v3 (schema + terminology expansion).
  - Both steps are sampled uniformly without replacement from [1, max_step].
"""

from __future__ import annotations

import random
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

# ─────────────────────────────────────────────────────────────────────────────
# Per-event notice templates (step number filled in at runtime)
# ─────────────────────────────────────────────────────────────────────────────

_V2_NOTICE_TEMPLATE = """=== SYSTEM NOTICE [Step {step}] ===
Effective immediately, the following policy changes are in effect:
- Refund eligibility window: 30 days → 14 days
- Critical SLA threshold: 4 hours → 2 hours
- High SLA threshold: 8 hours → 4 hours
- Auto-escalation threshold: $500/min → $250/min
- Customer responses must now begin with a formal greeting
Please update all future decisions accordingly.
=== END NOTICE ==="""

_V3_NOTICE_TEMPLATE = """=== SYSTEM NOTICE [Step {step}] ===
Effective immediately, the following changes are in effect:
- New ticket category added: Security (previously classified as Technical)
- Any ticket involving authentication, access control, or data breach must now use category 'Security'
- Two new ticket fields are now available: sentiment_score (0.0–1.0) and account_age_days
- These fields should inform your escalation urgency and response tone
Please update all future decisions accordingly.
=== END NOTICE ==="""


def _make_default_schedule(step_v2: int = 4, step_v3: int = 9) -> List[DriftEvent]:
    """Build the two-event drift schedule with explicit step numbers."""
    return [
        DriftEvent(
            fires_at_step=step_v2,
            from_version="v1",
            to_version="v2",
            drift_types=["policy_drift"],
            notice_text=_V2_NOTICE_TEMPLATE.format(step=step_v2),
        ),
        DriftEvent(
            fires_at_step=step_v3,
            from_version="v2",
            to_version="v3",
            drift_types=["policy_drift", "schema_drift", "terminology_drift"],
            notice_text=_V3_NOTICE_TEMPLATE.format(step=step_v3),
        ),
    ]


DEFAULT_DRIFT_SCHEDULE = _make_default_schedule(step_v2=4, step_v3=9)


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

    @classmethod
    def randomize(
        cls,
        max_step: int,
        rng: Optional[random.Random] = None,
    ) -> "DriftScheduler":
        """
        Create a DriftScheduler with two drift steps drawn uniformly at random
        (without replacement) from [1, max_step].

        The earlier step always becomes the v1→v2 drift; the later step becomes
        the v2→v3 drift, preserving the logical ordering of the two drifts.

        Args:
            max_step: Inclusive upper bound for random step selection (e.g. 20).
            rng: Optional seeded Random instance for reproducibility.
        """
        if rng is None:
            rng = random.Random()
        if max_step < 2:
            raise ValueError("max_step must be >= 2 to fit two distinct drift events.")
        steps = rng.sample(range(1, max_step + 1), 2)
        step_v2, step_v3 = sorted(steps)  # earlier = v1→v2, later = v2→v3
        return cls(schedule=_make_default_schedule(step_v2=step_v2, step_v3=step_v3))

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
