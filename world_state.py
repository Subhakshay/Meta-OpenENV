"""
world_state.py — WorldState dataclass + mutation logic

All stateful consequences of agent decisions live here. The WorldState
object persists within an episode and tracks:
  - Financial impact (company_balance)
  - Customer satisfaction (churn_risk)
  - Operational metrics (escalation_queue, sla_breaches)
  - Policy drift tracking (drift_events_fired, agent_drift_accuracy)
  - Adversarial metrics (attacker_win_rate_50, difficulty_level)

Key design notes:
  - attacker_win_rate_50 uses a deque(maxlen=50) that persists ACROSS
    episodes within a session. It is NOT reset on /reset so the curriculum
    controller has a meaningful signal.
  - Curriculum controller only activates after >= 10 entries (cold start protection).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict


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
    attacker_win_rate_50: float = 0.5          # rolling window, cross-episode
    difficulty_level: float = 0.3
    tickets_processed: int = 0
    multi_turn_active: bool = False

    # ── Internal tracking (not exported to agent) ────────────────────────────
    _post_drift_decisions_correct: int = 0
    _post_drift_decisions_total: int = 0
    _recent_attacker_results: deque = field(default_factory=lambda: deque(maxlen=50))

    # ── Mutation methods ─────────────────────────────────────────────────────

    def apply_wrong_refund(self, amount: float) -> None:
        """Decrement company_balance by amount. Floor at 0."""
        self.company_balance = max(0.0, self.company_balance - amount)

    def apply_churn_delta(self, delta: float) -> None:
        """Add delta to churn_risk. Clamp to [0.0, 1.0]."""
        self.churn_risk = max(0.0, min(1.0, self.churn_risk + delta))

    def apply_escalation(self) -> None:
        """Increment escalation_queue_size. If > 5, trigger SLA breach."""
        self.escalation_queue_size += 1
        if self.escalation_queue_size > 5:
            self.trigger_sla_breach()

    def close_escalation(self) -> None:
        """Decrement escalation_queue_size. Floor at 0."""
        self.escalation_queue_size = max(0, self.escalation_queue_size - 1)

    def trigger_sla_breach(self) -> None:
        """Increment sla_breaches."""
        self.sla_breaches += 1

    def record_drift_event(self, new_version: str) -> None:
        """Set current_policy_version. Increment drift_events_fired."""
        self.current_policy_version = new_version
        self.drift_events_fired += 1

    def record_post_drift_decision(self, correct: bool) -> None:
        """
        Update _post_drift_decisions_correct and _total.
        Recompute agent_drift_accuracy = correct / total.
        """
        self._post_drift_decisions_total += 1
        if correct:
            self._post_drift_decisions_correct += 1
        self.agent_drift_accuracy = (
            self._post_drift_decisions_correct / self._post_drift_decisions_total
        )

    def record_stale_decision(self) -> None:
        """Increment stale_decisions_made."""
        self.stale_decisions_made += 1

    def record_hallucination(self) -> None:
        """Increment hallucinations_caught."""
        self.hallucinations_caught += 1

    def record_attacker_result(self, attacker_won: bool) -> None:
        """
        Append to _recent_attacker_results deque.
        Recompute attacker_win_rate_50 = sum / len.
        """
        self._recent_attacker_results.append(1 if attacker_won else 0)
        if len(self._recent_attacker_results) > 0:
            self.attacker_win_rate_50 = (
                sum(self._recent_attacker_results) / len(self._recent_attacker_results)
            )

    # ── Curriculum controller ────────────────────────────────────────────────

    def run_curriculum_step(self) -> None:
        """
        Called after every step. Adjusts difficulty_level based on attacker_win_rate_50.
        Only activates after the deque has >= 10 entries (cold start protection).
        """
        if len(self._recent_attacker_results) < 10:
            return

        rate = self.attacker_win_rate_50
        if rate > 0.75:
            # Attacker dominating → make it easier for defender
            self.difficulty_level = max(0.0, self.difficulty_level - 0.10)
        elif 0.60 <= rate <= 0.75:
            # Balanced-ish → hold steady
            pass
        elif 0.40 <= rate < 0.60:
            # Defender doing well → increase difficulty slightly
            self.difficulty_level = min(1.0, self.difficulty_level + 0.05)
        else:  # < 0.40
            # Defender dominating → ramp up difficulty
            self.difficulty_level = min(1.0, self.difficulty_level + 0.15)

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

    def reset_episode(self, preserve_attacker_deque: bool = True) -> None:
        """
        Reset episode-level state. Optionally preserve attacker deque
        across episodes for curriculum continuity.
        """
        saved_deque = self._recent_attacker_results if preserve_attacker_deque else deque(maxlen=50)
        saved_win_rate = self.attacker_win_rate_50 if preserve_attacker_deque else 0.5
        saved_difficulty = self.difficulty_level if preserve_attacker_deque else 0.3

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
        self._post_drift_decisions_correct = 0
        self._post_drift_decisions_total = 0

        # Preserve cross-episode data
        self._recent_attacker_results = saved_deque
        self.attacker_win_rate_50 = saved_win_rate
        self.difficulty_level = saved_difficulty
