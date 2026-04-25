"""
world_state.py — WorldState dataclass + mutation logic

All stateful consequences of agent decisions live here. The WorldState
object persists within an episode and tracks:
  - Financial impact (company_balance)
  - Customer satisfaction (churn_risk)
  - Operational metrics (escalation_queue, sla_breaches)
  - Policy drift tracking (drift_events_fired, agent_drift_accuracy)
  - Adversarial metrics (attacker_win_rate, difficulty_level 1-5)
  - Gauntlet metrics (current_round, catastrophic_failure)
  - Adaptation tracking (adaptation_speed)

Key design notes:
  - difficulty_level is an integer [1, 5] (ELO-like).
  - Rolling window of last 5 outcomes for difficulty adaptation.
  - Catastrophic failure immediately ends a Gauntlet episode.
  - _recent_outcomes deque persists ACROSS episodes for curriculum continuity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class WorldState:
    """All public fields are exported to the agent. _ prefixed fields are internal only."""

    # ── Public fields (exported via to_export_dict) ──────────────────────────
    company_balance: float = 10_000.0
    churn_risk: float = 0.0                    # clamp 0.0–1.0
    escalation_queue_size: int = 0
    sla_breaches: int = 0
    current_policy_version: str = "v1"
    drift_events_fired: int = 0
    agent_drift_accuracy: float = 0.0
    stale_decisions_made: int = 0
    hallucinations_caught: int = 0
    difficulty_level: int = 1                  # integer 1-5
    tickets_processed: int = 0
    multi_turn_active: bool = False
    catastrophic_failure: bool = False
    catastrophic_reason: str = ""
    current_round: int = 1                     # Gauntlet round counter
    rounds_survived: int = 0
    adaptation_speed: float = 0.0              # how fast defender recovers post-drift
    attacker_win_rate: float = 0.5             # rolling window

    # ── Internal tracking (not exported to agent) ────────────────────────────
    _post_drift_decisions_correct: int = 0
    _post_drift_decisions_total: int = 0
    _recent_outcomes: deque = field(default_factory=lambda: deque(maxlen=5))
    _drift_recovery_steps: List[int] = field(default_factory=list)
    _post_drift_step_counter: int = 0
    _post_drift_first_correct_step: int = -1

    # ── Mutation methods ─────────────────────────────────────────────────────

    def apply_wrong_refund(self, amount: float) -> None:
        """Decrement company_balance by amount. Floor at 0."""
        self.company_balance = max(0.0, self.company_balance - amount)

    def apply_churn_delta(self, delta: float) -> None:
        """Add delta to churn_risk. Clamp to [0.0, 1.0]."""
        self.churn_risk = max(0.0, min(1.0, self.churn_risk + delta))

    def apply_escalation(self) -> None:
        """Increment escalation_queue_size. If > max, trigger SLA breach."""
        self.escalation_queue_size += 1
        if self.escalation_queue_size > 5:
            self.trigger_sla_breach()

    def close_escalation(self) -> None:
        """Decrement escalation_queue_size. Floor at 0."""
        self.escalation_queue_size = max(0, self.escalation_queue_size - 1)

    def trigger_sla_breach(self) -> None:
        """Increment sla_breaches."""
        self.sla_breaches += 1

    def trigger_catastrophic_failure(self, reason: str) -> None:
        """Mark a catastrophic failure — immediately ends Gauntlet episode."""
        self.catastrophic_failure = True
        self.catastrophic_reason = reason

    def record_drift_event(self, new_version: str) -> None:
        """Set current_policy_version. Increment drift_events_fired."""
        self.current_policy_version = new_version
        self.drift_events_fired += 1
        # Reset adaptation tracking for this drift
        self._post_drift_step_counter = 0
        self._post_drift_first_correct_step = -1

    def record_post_drift_decision(self, correct: bool) -> None:
        """
        Update _post_drift_decisions_correct and _total.
        Recompute agent_drift_accuracy = correct / total.
        Track adaptation speed.
        """
        self._post_drift_decisions_total += 1
        self._post_drift_step_counter += 1
        if correct:
            self._post_drift_decisions_correct += 1
            if self._post_drift_first_correct_step == -1:
                self._post_drift_first_correct_step = self._post_drift_step_counter
                self._drift_recovery_steps.append(self._post_drift_step_counter)
        self.agent_drift_accuracy = (
            self._post_drift_decisions_correct / self._post_drift_decisions_total
        )
        # Compute adaptation speed as inverse of average recovery steps
        if self._drift_recovery_steps:
            avg_recovery = sum(self._drift_recovery_steps) / len(self._drift_recovery_steps)
            self.adaptation_speed = round(1.0 / max(avg_recovery, 0.1), 4)

    def record_stale_decision(self) -> None:
        """Increment stale_decisions_made."""
        self.stale_decisions_made += 1

    def record_hallucination(self) -> None:
        """Increment hallucinations_caught."""
        self.hallucinations_caught += 1

    def record_outcome(self, attacker_won: bool) -> None:
        """
        Append to _recent_outcomes deque (maxlen=5).
        Recompute attacker_win_rate = sum / len.
        """
        self._recent_outcomes.append(1 if attacker_won else 0)
        if len(self._recent_outcomes) > 0:
            self.attacker_win_rate = round(
                sum(self._recent_outcomes) / len(self._recent_outcomes), 4
            )

    def advance_round(self) -> None:
        """Move to the next Gauntlet round. Track survival."""
        self.current_round += 1
        self.rounds_survived += 1

    # ── Curriculum controller (ELO-like, integer 1-5) ────────────────────────

    def run_curriculum_step(self) -> None:
        """
        Called after every step. Adjusts difficulty_level based on
        rolling window of last 5 outcomes. Only activates after >= 3
        entries (cold start protection).

        Difficulty is clamped to [1, 5].
        """
        if len(self._recent_outcomes) < 3:
            return

        rate = self.attacker_win_rate

        if rate > 0.80:
            # Attacker dominating → make it easier for defender
            self.difficulty_level = max(1, self.difficulty_level - 1)
        elif rate > 0.60:
            # Balanced-ish → hold steady
            pass
        elif rate >= 0.40:
            # Defender doing well → increase difficulty by 1
            self.difficulty_level = min(5, self.difficulty_level + 1)
        else:
            # Defender dominating → ramp up by 2
            self.difficulty_level = min(5, self.difficulty_level + 2)

    # ── Export ────────────────────────────────────────────────────────────────

    def to_export_dict(self) -> Dict[str, Any]:
        """
        Return all public fields as a dict (exclude _ prefixed fields).
        Used for API responses and DB snapshots.
        """
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    def reset_episode(self, preserve_curriculum: bool = True) -> None:
        """
        Reset episode-level state. Optionally preserve curriculum data
        across episodes for continuity.
        """
        saved_deque = self._recent_outcomes if preserve_curriculum else deque(maxlen=5)
        saved_win_rate = self.attacker_win_rate if preserve_curriculum else 0.5
        saved_difficulty = self.difficulty_level if preserve_curriculum else 1

        self.company_balance = 10_000.0
        self.churn_risk = 0.0
        self.escalation_queue_size = 0
        self.sla_breaches = 0
        self.current_policy_version = "v1"
        self.drift_events_fired = 0
        self.agent_drift_accuracy = 0.0
        self.stale_decisions_made = 0
        self.hallucinations_caught = 0
        self.tickets_processed = 0
        self.multi_turn_active = False
        self.catastrophic_failure = False
        self.catastrophic_reason = ""
        self.current_round = 1
        self.rounds_survived = 0
        self.adaptation_speed = 0.0
        self._post_drift_decisions_correct = 0
        self._post_drift_decisions_total = 0
        self._post_drift_step_counter = 0
        self._post_drift_first_correct_step = -1
        self._drift_recovery_steps = []

        # Preserve cross-episode data
        self._recent_outcomes = saved_deque
        self.attacker_win_rate = saved_win_rate
        self.difficulty_level = saved_difficulty
