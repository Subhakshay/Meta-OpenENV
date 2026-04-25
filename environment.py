"""
environment.py — Core CustomerSupportEnv for The Gauntlet + Shifting Sands

Uses the Jinja2-powered GenerationEngine for all ticket generation.
Supports 25-ticket episodes with policy drift and reconciliation.

Implements:
  - Dynamic ticket generation via GenerationEngine (no static templates)
  - Observation schema enforcement (policy-gated fields)
  - Full step loop with drift integration
  - Customer reply simulation for multi-turn (Task 3)
  - Silent replacement reconciliation for policy drifts
"""

from __future__ import annotations

import random
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from policy import PolicyRegistry, PolicyVersion
from world_state import WorldState
from drift_scheduler import DriftScheduler, DriftEvent, build_drift_notice
from rewards import calculate_defender_reward, calculate_attacker_reward
from template_sampler import TemplateSampler
from generation_engine import GenerationEngine, CATEGORIES, PRIORITIES
from variation_pools import MISC_VARS


# ─────────────────────────────────────────────────────────────────────────────
# Customer reply simulation (Task 3 multi-turn)
# ─────────────────────────────────────────────────────────────────────────────

CLARIFICATION_REPLIES = {
    "Billing": [
        "I purchased it a couple weeks ago. My account is {tier}.",
        "The charge was for my subscription. I did not authorise it.",
        "I'm on the {plan} plan. The amount was ${charge_amount}.",
    ],
    "Technical": [
        "It's the main dashboard section. My account ID is ACC-{incident_id}.",
        "It started a few days ago. I'm using the {product}.",
        "The error happens when I try to export data. Browser is Chrome.",
    ],
    "Shipping": [
        "Order #{order_id}. I placed it last week.",
        "Tracking shows '{tracking_status}'. I need it urgently.",
    ],
    "Security": [
        "I noticed suspicious activity on my {tier} account recently.",
        "The unauthorized access was to the dashboard. I've changed my password.",
    ],
    "Fraud": [
        "I saw the fraudulent charge on my statement yesterday.",
        "Multiple charges appeared that I never authorized.",
    ],
    "Compliance": [
        "Our audit deadline is next week. We need the documentation urgently.",
        "The compliance module is not generating the required reports.",
    ],
}

# Variable pools for clarification reply fill
_REPLY_VARS = {
    "tier": MISC_VARS["tier"],
    "plan": ["Pro", "Enterprise", "Business", "Starter", "Growth"],
    "charge_amount": ["49.99", "99.00", "299.00", "19.99", "149.50", "599.00"],
    "incident_id": [f"{n:06d}" for n in range(100001, 100021)],
    "product": ["API Gateway", "Dashboard", "Analytics Suite", "Payment Module",
                "Auth Service", "Webhook Relay", "Storage Bucket"],
    "order_id": [f"ORD-{n}" for n in range(10001, 10021)],
    "tracking_status": ["In Transit", "Out for Delivery", "Delayed", "Pending"],
}


def _fill_reply(template: str, rng: random.Random) -> str:
    """Fill {placeholder} variables in clarification replies."""
    def replacer(m):
        key = m.group(1)
        pool = _REPLY_VARS.get(key, [f"[{key}]"])
        return rng.choice(pool)
    return re.sub(r"\{(\w+)\}", replacer, template)


def simulate_customer_reply(
    ticket: Dict[str, Any],
    rng: random.Random,
) -> str:
    """Pick a reply template matching the ticket category and fill variables."""
    category = ticket.get("true_category", "Technical")
    templates = CLARIFICATION_REPLIES.get(category, CLARIFICATION_REPLIES["Technical"])
    template = rng.choice(templates)
    return _fill_reply(template, rng)


# ─────────────────────────────────────────────────────────────────────────────
# CustomerSupportEnv
# ─────────────────────────────────────────────────────────────────────────────

class CustomerSupportEnv:
    """
    Core environment implementing Gym-style reset/step API.

    Supports three task modes:
      task_id=1: Priority only (easy)
      task_id=2: Full classification + response
      task_id=3: Multi-turn + full classification
    """

    EPISODE_LENGTH = 25

    def __init__(self) -> None:
        self.policy_registry = PolicyRegistry()
        self.drift_scheduler = DriftScheduler(episode_length=self.EPISODE_LENGTH, registry=self.policy_registry)
        self.world_state = WorldState()
        self._rng = random.Random(42)
        self._engine = GenerationEngine(seed=42)

        # Episode state
        self._session_id: str = ""
        self._task_id: int = 1
        self._attacker_enabled: bool = False
        self._drift_enabled: bool = False
        self._episode_id: Optional[int] = None
        self._ticket_queue: List[Dict[str, Any]] = []
        self._current_step: int = 0
        self._done: bool = False
        self._was_post_drift: bool = False
        self._pending_drift_notice: Optional[str] = None
        self._conversation_history: List[Dict[str, str]] = []
        self._attacker_tickets: List[Dict[str, Any]] = []
        self._attacker_agent: Any = None
        self._drift_events: List[DriftEvent] = []
        self._template_sampler: Optional[TemplateSampler] = None
        self._last_reconcile_record: Optional[Dict[str, Any]] = None

        # Step rewards log
        self._defender_rewards: List[float] = []
        self._attacker_rewards: List[float] = []

    def reset(
        self,
        task_id: int,
        attacker_enabled: bool = False,
        drift_enabled: bool = True,
        difficulty_init: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Start a new episode.
        Returns first observation dict.
        """
        self._session_id = str(uuid.uuid4())
        self._task_id = task_id
        self._attacker_enabled = attacker_enabled
        self._drift_enabled = drift_enabled
        self._current_step = 0
        self._done = False
        self._was_post_drift = False
        self._pending_drift_notice = None
        self._conversation_history = []
        self._defender_rewards = []
        self._attacker_rewards = []

        if seed is not None:
            self._rng = random.Random(seed)
            self._engine = GenerationEngine(seed=seed)

        # Preserve attacker deque across episodes, reset everything else
        self.world_state.reset_episode(preserve_attacker_deque=True)
        if difficulty_init is not None:
            self.world_state.difficulty_level = difficulty_init

        # Initialize policy registry and sample the first policy
        self.policy_registry.reset()
        self.policy_registry.sample_policy("dyn-init")

        # Task 1: cap difficulty at 0.2
        if task_id == 1:
            self.world_state.difficulty_level = min(0.2, difficulty_init)
            self._drift_enabled = False  # Drift disabled for task 1

        # Schedule drifts AFTER difficulty is set
        if self._drift_enabled:
            self._drift_events = self.drift_scheduler.schedule_episode(
                difficulty_level=self.world_state.difficulty_level
            )
        else:
            self._drift_events = []

        # Initialize template sampler for reconciliation
        self._template_sampler = TemplateSampler(engine=self._engine)
        self._last_reconcile_record = None

        # Generate ticket queue
        self._ticket_queue = self._generate_ticket_batch(n=self.EPISODE_LENGTH)

        # Build first observation
        return self._build_observation(
            ticket=self._ticket_queue[0],
            drift_notice=None,
        )

    def _reconcile_queue(self, new_policy: PolicyVersion, old_policy: PolicyVersion, step: int) -> int:
        """
        Silent Replacement method: check all unshown tickets against the new
        policy.  Replace incompatible tickets by re-rendering via the engine.

        For math/SLA drifts: re-render the same template with new policy vars.
        For category removals: template swap to a valid category.

        Returns the count of tickets replaced.
        """
        replaced = 0
        start_idx = self.world_state.tickets_processed

        for i in range(start_idx, len(self._ticket_queue)):
            ticket = self._ticket_queue[i]
            compatible = True

            # 1. Category must be in new policy's valid set
            if ticket.get("true_category") not in new_policy.valid_categories:
                compatible = False

            # 2. Boundary exploit window check
            if compatible and ticket.get("deception_strategy") == "boundary_exploitation":
                stored_days = ticket.get("boundary_exploit_day_count") or ticket.get("days_since_purchase")
                if stored_days is not None:
                    if abs(stored_days - new_policy.refund_window_days) > 2:
                        compatible = False

            # 3. Empathy trigger check
            if compatible and ticket.get("deception_strategy") == "emotional_manipulation":
                if new_policy.empathy_required_below_sentiment is None:
                    compatible = False

            if not compatible:
                # Determine replacement parameters
                category = ticket.get("true_category", "Technical")
                strategy = ticket.get("deception_strategy", "clean")
                priority = ticket.get("true_priority", "Medium")
                tone = ticket.get("ground_truth", {}).get("tone", "neutral") if isinstance(ticket.get("ground_truth"), dict) else "neutral"

                # If category is now invalid, swap to a valid one
                if category not in new_policy.valid_categories:
                    category = self._rng.choice(list(new_policy.valid_categories))

                # Re-generate via engine
                try:
                    result = self._engine.generate(
                        category=category,
                        strategy=strategy,
                        priority=priority,
                        tone=tone,
                        active_policy=new_policy,
                        churn_risk=self.world_state.churn_risk,
                    )

                    gt = result["ground_truth"]
                    new_ticket = {
                        "ticket_id": f"TKT-{self._rng.randint(10000, 99999)}",
                        "subject": result["ticket_string"].split(".")[0][:60],
                        "body": result["ticket_string"],
                        "tier": self._rng.choice(MISC_VARS["tier"]),
                        "true_priority": gt["priority"],
                        "true_category": gt["category"],
                        "base_requires_escalation": gt["priority"] == "High" and strategy in ("fake_urgency", "boundary_exploitation"),
                        "deception_strategy": gt["strategy"],
                        "schema_violation": gt["strategy"] == "schema_exploitation",
                        "is_ambiguous": gt["strategy"] == "category_confusion",
                        "days_since_purchase": result["days_since_purchase"],
                        "true_refund_eligible": result["true_refund_eligible"],
                        "sentiment_score": result["sentiment_score"],
                        "account_age_days": self._rng.randint(1, 2000),
                    }

                    if "boundary_exploit_day_count" in result:
                        new_ticket["boundary_exploit_day_count"] = result["boundary_exploit_day_count"]

                    self._ticket_queue[i] = new_ticket

                except Exception:
                    # Fallback: generate a clean ticket in a valid category
                    fallback_cat = self._rng.choice(list(new_policy.valid_categories))
                    fallback_pri = self._rng.choice(PRIORITIES)
                    try:
                        result = self._engine.generate(
                            category=fallback_cat,
                            strategy="clean",
                            priority=fallback_pri,
                            tone="neutral",
                            active_policy=new_policy,
                        )
                        gt = result["ground_truth"]
                        self._ticket_queue[i] = {
                            "ticket_id": f"TKT-{self._rng.randint(10000, 99999)}",
                            "subject": result["ticket_string"].split(".")[0][:60],
                            "body": result["ticket_string"],
                            "tier": self._rng.choice(MISC_VARS["tier"]),
                            "true_priority": gt["priority"],
                            "true_category": gt["category"],
                            "base_requires_escalation": False,
                            "deception_strategy": "clean",
                            "schema_violation": False,
                            "is_ambiguous": False,
                            "days_since_purchase": result["days_since_purchase"],
                            "true_refund_eligible": result["true_refund_eligible"],
                            "sentiment_score": result["sentiment_score"],
                            "account_age_days": self._rng.randint(1, 2000),
                        }
                    except Exception:
                        pass  # Leave original ticket if all else fails

                replaced += 1

        # Write reconciliation record
        self._last_reconcile_record = {
            "step": step,
            "from_version": old_policy.version_id,
            "to_version": new_policy.version_id,
            "tickets_replaced": replaced,
        }

        return replaced

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process one agent action. Returns observation dict with reward, done, etc.
        """
        if self._done:
            raise RuntimeError("Episode already closed. Call reset() first.")

        self._current_step += 1

        # ── Check for drift ──────────────────────────────────────────────
        drift_notice = None
        if self._drift_enabled:
            event = next((e for e in self._drift_events if e.fires_at_step == self._current_step), None)
            if event:
                old_policy = self.policy_registry.get_active()
                self.policy_registry._history.append(event.new_policy)
                self.world_state.record_drift_event(event.new_policy.version_id)
                self._was_post_drift = True

                drift_notice = build_drift_notice(old_policy, event.new_policy, self._current_step)
                self._pending_drift_notice = drift_notice

                # Reconcile the queue
                self._reconcile_queue(event.new_policy, old_policy, self._current_step)

        # Grab ticket AFTER drift reconciliation so we always score the right one
        ticket = self._ticket_queue[self.world_state.tickets_processed]

        # ── Handle multi-turn ASK (Task 3) ───────────────────────────────
        if (self._task_id == 3
                and action.get("ask_clarification", False)
                and not self.world_state.multi_turn_active):
            self.world_state.multi_turn_active = True
            reply = simulate_customer_reply(ticket, self._rng)
            self._conversation_history.append({
                "agent": action.get("clarification_text", "Could you clarify?"),
                "customer": reply,
            })

            # Clarification quality scoring
            clarif_reward = 0.0
            if ticket.get("is_ambiguous", False):
                clarif_reward = 1.0  # Genuinely ambiguous — good to ask
                q_text = action.get("clarification_text", "")
                if q_text and "?" in q_text:
                    # Check if references ticket content
                    subject_kws = set(re.split(r'\W+', ticket.get("subject", "").lower()))
                    if any(kw in q_text.lower() for kw in subject_kws if len(kw) > 3):
                        clarif_reward += 0.5
            else:
                clarif_reward = -0.5  # Unnecessary delay on obvious ticket

            obs = self._build_observation(ticket, drift_notice)
            return {
                "reward": round(clarif_reward, 4),
                "observation": obs,
                "world_state": self.world_state.to_export_dict(),
                "done": False,
                "drift_notice": drift_notice,
            }

        # ── Calculate rewards ────────────────────────────────────────────
        active_policy = self.policy_registry.get_active()
        previous_policy = self.policy_registry.get_previous() if self._was_post_drift else None

        defender_reward, breakdown = calculate_defender_reward(
            action=action,
            ticket=ticket,
            active_policy=active_policy,
            previous_policy=previous_policy,
            world_state=self.world_state,
            was_post_drift=self._was_post_drift,
            task_id=self._task_id,
            policy_registry=self.policy_registry,
        )

        attacker_reward = 0.0
        if self._attacker_enabled:
            attacker_reward = calculate_attacker_reward(action, ticket)

        self._defender_rewards.append(defender_reward)
        self._attacker_rewards.append(attacker_reward)

        # ── Advance state ────────────────────────────────────────────────
        self.world_state.tickets_processed += 1
        self.world_state.multi_turn_active = False
        self._conversation_history = []

        # Reset post-drift flag unconditionally after processing
        # (it gets set to True again at the top of step() if a new drift fires)
        self._was_post_drift = False

        # Check done
        done = self.world_state.tickets_processed >= self.EPISODE_LENGTH
        self._done = done

        # Build next observation
        next_obs = None
        if not done and self.world_state.tickets_processed < len(self._ticket_queue):
            next_ticket = self._ticket_queue[self.world_state.tickets_processed]
            next_obs = self._build_observation(next_ticket, drift_notice)

        return {
            "reward": defender_reward,
            "attacker_reward": attacker_reward,
            "reward_breakdown": breakdown,
            "observation": next_obs,
            "world_state": self.world_state.to_export_dict(),
            "done": done,
            "drift_notice": drift_notice,
        }

    def _build_observation(
        self,
        ticket: Dict[str, Any],
        drift_notice: Optional[str],
    ) -> Dict[str, Any]:
        """
        Assembles the observation the Defender sees.
        Schema fields are policy-gated.
        """
        active_policy = self.policy_registry.get_active()

        obs: Dict[str, Any] = {
            "ticket_id": ticket["ticket_id"],
            "active_policy_version": active_policy.version_id,
            "world_state_summary": {
                "company_balance": self.world_state.company_balance,
                "churn_risk": self.world_state.churn_risk,
                "escalation_queue_size": self.world_state.escalation_queue_size,
                "sla_breaches": self.world_state.sla_breaches,
                "tickets_processed": self.world_state.tickets_processed,
            },
        }

        # Include basic ticket fields
        for field_name in ["subject", "body", "tier"]:
            if field_name in ticket:
                obs[field_name] = ticket[field_name]

        # Conditionally expose sentiment/account age
        if active_policy.empathy_required_below_sentiment is not None:
             if "sentiment_score" in ticket:
                 obs["sentiment_score"] = ticket["sentiment_score"]
             if "account_age_days" in ticket:
                 obs["account_age_days"] = ticket["account_age_days"]

        # System notice (only if drift just fired)
        if drift_notice:
            obs["system_notice"] = drift_notice

        # Conversation history (only if multi-turn active)
        if self.world_state.multi_turn_active and self._conversation_history:
            obs["conversation_history"] = self._conversation_history

        return obs

    def _generate_ticket_batch(self, n: int = 25) -> List[Dict[str, Any]]:
        """Generate n clean procedural tickets via the Jinja2 engine."""
        policy = self.policy_registry.get_active()
        valid_categories = list(policy.valid_categories)

        tickets = []
        for _ in range(n):
            category = self._rng.choice(valid_categories)
            priority = self._rng.choice(PRIORITIES)

            result = self._engine.generate(
                category=category,
                strategy="clean",
                priority=priority,
                tone="neutral",
                active_policy=policy,
                churn_risk=self.world_state.churn_risk,
            )

            gt = result["ground_truth"]
            ticket = {
                "ticket_id": f"TKT-{self._rng.randint(10000, 99999)}",
                "subject": result["ticket_string"].split(".")[0][:60],
                "body": result["ticket_string"],
                "tier": self._rng.choice(MISC_VARS["tier"]),
                "true_priority": gt["priority"],
                "true_category": gt["category"],
                "base_requires_escalation": False,
                "refund_eligible_boundary": False,
                "is_ambiguous": False,
                "deception_strategy": "clean",
                "schema_violation": False,
                "days_since_purchase": result["days_since_purchase"],
                "true_refund_eligible": result["true_refund_eligible"],
                "sentiment_score": result["sentiment_score"],
                "account_age_days": self._rng.randint(1, 2000),
            }
            tickets.append(ticket)

        return tickets

    def set_attacker_tickets(self, tickets: List[Dict[str, Any]], attacker_agent: Any = None) -> None:
        """Replace the ticket queue with attacker-generated tickets."""
        self._ticket_queue = tickets
        self._attacker_tickets = tickets
        self._attacker_agent = attacker_agent

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def current_step(self) -> int:
        return self._current_step

    def get_episode_metrics(self) -> Dict[str, Any]:
        """Return episode summary metrics for DB logging."""
        ws = self.world_state
        return {
            "mean_defender_reward": (
                sum(self._defender_rewards) / max(len(self._defender_rewards), 1)
            ),
            "mean_attacker_reward": (
                sum(self._attacker_rewards) / max(len(self._attacker_rewards), 1)
            ),
            "final_balance": ws.company_balance,
            "sla_breaches": ws.sla_breaches,
            "drift_accuracy": ws.agent_drift_accuracy,
            "stale_decisions": ws.stale_decisions_made,
            "hallucinations": ws.hallucinations_caught,
            "attacker_win_rate_final": ws.attacker_win_rate_50,
            "difficulty_final": ws.difficulty_level,
        }
