"""
environment.py — Core CustomerSupportEnv for The Gauntlet + Shifting Sands

Implements:
  - 30+ ticket blueprints with variable pools
  - Procedural ticket generation (clean mode)
  - Observation schema enforcement (policy-gated fields)
  - Full step loop with drift integration
  - Customer reply simulation for multi-turn (Task 3)
"""

from __future__ import annotations

import random
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from policy import PolicyRegistry, PolicyVersion
from world_state import WorldState
from drift_scheduler import DriftScheduler, DriftEvent
from rewards import calculate_defender_reward, calculate_attacker_reward


# ─────────────────────────────────────────────────────────────────────────────
# Variable pools for procedural generation
# ─────────────────────────────────────────────────────────────────────────────

VARS = {
    "tier": ["Free", "Pro", "Enterprise", "Starter", "Growth", "Business",
             "Scale", "Premium", "Developer", "Team"],
    "symptom": ["down", "unresponsive", "erroring", "throttled", "timing out",
                "returning 500s", "dropping connections", "failing silently",
                "crashing intermittently", "rejecting requests", "leaking memory",
                "hanging on startup", "corrupting data", "looping infinitely",
                "throwing exceptions"],
    "duration": ["2 hours", "4 hours", "since this morning", "3 days",
                 "over 6 hours", "the last 3 hours", "since yesterday",
                 "about 45 minutes", "nearly a week", "since the last deploy",
                 "intermittently for 2 days", "30 minutes"],
    "loss": ["50", "100", "250", "500", "1000", "5000", "2500", "800"],
    "product": ["API Gateway", "Dashboard", "Analytics Suite", "Payment Module",
                "Auth Service", "CDN", "Webhook Relay", "Storage Bucket",
                "Search Index", "Notification Service"],
    "action": ["reset my password", "upgrade my plan", "export data",
               "revoke API keys", "enable 2FA", "transfer ownership",
               "merge accounts", "configure webhooks", "set up SSO",
               "generate invoice"],
    "days_since_purchase": ["13", "14", "15", "29", "30", "31"],
    "user_count": ["120", "500", "2000", "50000", "300", "15000"],
    "incident_id": [f"{n:06d}" for n in range(100001, 100021)],
    "time_ago": ["2:00 AM UTC", "08:30 EST", "14:15 PST", "09:00 GMT", "midnight"],
    "email": ["user@company.com", "admin@startup.io", "billing@corp.net",
              "ops@enterprise.co", "support@client.org"],
    "charge_amount": ["49.99", "99.00", "299.00", "19.99", "149.50", "599.00"],
    "card_last4": ["4242", "1234", "9876", "5555", "7890"],
    "plan": ["Pro", "Enterprise", "Business", "Starter", "Growth"],
    "cancel_date": ["March 1st", "February 15th", "last Tuesday", "January 31st"],
    "order_id": [f"ORD-{n}" for n in range(10001, 10021)],
    "tracking_status": ["In Transit", "Out for Delivery", "Delayed", "Pending"],
    "feature": ["API integration", "export tool", "real-time sync",
                "analytics dashboard", "webhook relay", "SSO module"],
    "vague_feature": ["main panel", "analytics tab", "export tool",
                      "webhook", "integration", "settings page"],
    "vague_behavior": ["update automatically", "show the right numbers",
                       "sync properly", "load fast", "save my changes"],
    "vague_frequency": ["twice this week", "three times today", "randomly",
                        "since the last update", "every few hours"],
}


def _fill(template: str, rng: random.Random) -> str:
    """Fill {placeholder} variables from pools."""
    def replacer(m):
        key = m.group(1)
        pool = VARS.get(key, [f"[{key}]"])
        return rng.choice(pool)
    return re.sub(r"\{(\w+)\}", replacer, template)


# ─────────────────────────────────────────────────────────────────────────────
# 30+ Ticket Blueprints
# ─────────────────────────────────────────────────────────────────────────────

TICKET_BLUEPRINTS = [
    # ── Critical / Technical ──
    {"template": "Our {tier} integration has been {symptom} for {duration}. We are losing ${loss} per minute.",
     "true_priority": "Critical", "true_category": "Technical",
     "base_requires_escalation": True, "refund_eligible_boundary": False, "is_ambiguous": False},
    {"template": "EMERGENCY: The {product} is completely {symptom}. {user_count} users affected since {time_ago}.",
     "true_priority": "Critical", "true_category": "Technical",
     "base_requires_escalation": True, "refund_eligible_boundary": False, "is_ambiguous": False},
    {"template": "Production database connection pool exhausted. All services {symptom} for {duration}.",
     "true_priority": "Critical", "true_category": "Technical",
     "base_requires_escalation": True, "refund_eligible_boundary": False, "is_ambiguous": False},
    {"template": "Complete system failure on {product}. Every request returns errors. Started {time_ago}.",
     "true_priority": "Critical", "true_category": "Technical",
     "base_requires_escalation": True, "refund_eligible_boundary": False, "is_ambiguous": False},
    # ── High / Technical ──
    {"template": "The {product} API is {symptom} intermittently. Our {tier} customers are complaining.",
     "true_priority": "High", "true_category": "Technical",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": False},
    {"template": "Webhook deliveries from {product} are failing with timeouts for {duration}.",
     "true_priority": "High", "true_category": "Technical",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": False},
    # ── High / Billing ──
    {"template": "I was charged ${charge_amount} twice this billing cycle. Card ending {card_last4}. Please refund.",
     "true_priority": "High", "true_category": "Billing",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": False},
    {"template": "Despite cancelling my {plan} plan on {cancel_date}, I was still charged ${charge_amount}.",
     "true_priority": "High", "true_category": "Billing",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": False},
    {"template": "Unexpected charge of ${charge_amount} on my account. I did not authorize this transaction.",
     "true_priority": "High", "true_category": "Billing",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": False},
    # ── High / Shipping ──
    {"template": "Order #{order_id} hasn't arrived after 3 weeks. Tracking says '{tracking_status}'.",
     "true_priority": "High", "true_category": "Shipping",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": False},
    {"template": "Package marked delivered but not received. Order #{order_id}. Checked with neighbours.",
     "true_priority": "High", "true_category": "Shipping",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": False},
    # ── Medium / Technical ──
    {"template": "The {vague_feature} seems off today. It used to {vague_behavior} but now it doesn't.",
     "true_priority": "Medium", "true_category": "Technical",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": True},
    {"template": "Something isn't working like it used to with the {product}. Happened {vague_frequency}.",
     "true_priority": "Medium", "true_category": "Technical",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": True},
    {"template": "Feature behaving unexpectedly on {product}. Sometimes it works, sometimes it doesn't.",
     "true_priority": "Medium", "true_category": "Technical",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": True},
    {"template": "Things are slow and I'm not sure why. The {vague_feature} is acting weird.",
     "true_priority": "Medium", "true_category": "Technical",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": True},
    # ── Medium / Billing ──
    {"template": "Question about my recent charges. My bill this month looks different from last month.",
     "true_priority": "Medium", "true_category": "Billing",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": True},
    {"template": "Not sure if I'm on the right plan. Can someone check if I'm being charged correctly?",
     "true_priority": "Medium", "true_category": "Billing",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": True},
    # ── Medium / Shipping ──
    {"template": "My order #{order_id} status says '{tracking_status}' for a week now. Any update?",
     "true_priority": "Medium", "true_category": "Shipping",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": False},
    # ── Low / Technical ──
    {"template": "Quick question — does the {plan} plan include {feature} access?",
     "true_priority": "Low", "true_category": "Technical",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": False},
    {"template": "How do I set up webhooks for {product}? I looked at docs but couldn't find it.",
     "true_priority": "Low", "true_category": "Technical",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": False},
    # ── Low / Billing ──
    {"template": "Interested in upgrading from {plan} plan. What are my options?",
     "true_priority": "Low", "true_category": "Billing",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": False},
    {"template": "Can I get an annual billing discount for {plan}?",
     "true_priority": "Low", "true_category": "Billing",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": False},
    # ── Low / Shipping ──
    {"template": "Do you ship to international addresses? Planning to order {product}.",
     "true_priority": "Low", "true_category": "Shipping",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": False},
    # ── Critical / Security (v3 only) ──
    {"template": "URGENT: Unauthorized access detected on our {tier} account. Data may be compromised.",
     "true_priority": "Critical", "true_category": "Security",
     "base_requires_escalation": True, "refund_eligible_boundary": False, "is_ambiguous": False},
    {"template": "Our API keys were leaked on GitHub. Need immediate revocation and audit of {product}.",
     "true_priority": "Critical", "true_category": "Security",
     "base_requires_escalation": True, "refund_eligible_boundary": False, "is_ambiguous": False},
    # ── High / Security ──
    {"template": "Suspicious login attempts on our {tier} account from unknown IPs for {duration}.",
     "true_priority": "High", "true_category": "Security",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": False},
    {"template": "Two-factor authentication bypass vulnerability found in {product}.",
     "true_priority": "High", "true_category": "Security",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": False},
    # ── Boundary Exploitation blueprints (refund window edge) ──
    {"template": "I purchased {product} {days_since_purchase} days ago and it doesn't work. Requesting refund.",
     "true_priority": "Medium", "true_category": "Billing",
     "base_requires_escalation": False, "refund_eligible_boundary": True, "is_ambiguous": False},
    {"template": "Bought the {plan} plan {days_since_purchase} days ago. Service has been {symptom}. Want my money back.",
     "true_priority": "High", "true_category": "Billing",
     "base_requires_escalation": False, "refund_eligible_boundary": True, "is_ambiguous": False},
    {"template": "It's been {days_since_purchase} days since I paid for {product}. Still broken. Refund please.",
     "true_priority": "Medium", "true_category": "Billing",
     "base_requires_escalation": False, "refund_eligible_boundary": True, "is_ambiguous": False},
    # ── Medium / Security (ambiguous — could be Technical) ──
    {"template": "I'm getting strange permission errors when accessing {product}. Not sure if it's a bug or security issue.",
     "true_priority": "Medium", "true_category": "Security",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": True},
    {"template": "Someone might have accessed my account. I see sessions I don't recognize. Or maybe I forgot.",
     "true_priority": "Medium", "true_category": "Security",
     "base_requires_escalation": False, "refund_eligible_boundary": False, "is_ambiguous": True},
]


# ─────────────────────────────────────────────────────────────────────────────
# Procedural generation (clean mode)
# ─────────────────────────────────────────────────────────────────────────────

def generate_ticket_clean(
    blueprint: Dict[str, Any],
    policy: PolicyVersion,
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Substitutes variables randomly. Returns full ticket dict.
    For v3 schema fields, generates sentiment_score and account_age_days.
    Only includes v3 fields if policy.ticket_schema_fields contains them.
    """
    body = _fill(blueprint["template"], rng)
    subject_words = body.split()[:8]
    subject = " ".join(subject_words)
    if len(subject) > 60:
        subject = subject[:57] + "..."

    ticket = {
        "ticket_id": f"TKT-{rng.randint(10000, 99999)}",
        "subject": subject,
        "body": body,
        "tier": rng.choice(VARS["tier"]),
        "true_priority": blueprint["true_priority"],
        "true_category": blueprint["true_category"],
        "base_requires_escalation": blueprint["base_requires_escalation"],
        "refund_eligible_boundary": blueprint.get("refund_eligible_boundary", False),
        "is_ambiguous": blueprint.get("is_ambiguous", False),
        "deception_strategy": "clean",
        "schema_violation": False,
    }

    # Add days_since_purchase for boundary blueprints
    if blueprint.get("refund_eligible_boundary"):
        ticket["days_since_purchase"] = int(rng.choice(VARS["days_since_purchase"]))
        ticket["true_refund_eligible"] = (
            ticket["days_since_purchase"] <= policy.refund_window_days
        )
    else:
        ticket["days_since_purchase"] = rng.randint(1, 60)
        ticket["true_refund_eligible"] = (
            ticket["days_since_purchase"] <= policy.refund_window_days
        )

    # v3 schema fields — always generated internally, but only exposed
    # in observations when policy schema includes them
    ticket["sentiment_score"] = round(rng.random(), 2)
    ticket["account_age_days"] = rng.randint(1, 2000)

    return ticket


# ─────────────────────────────────────────────────────────────────────────────
# Customer reply simulation (Task 3 multi-turn)
# ─────────────────────────────────────────────────────────────────────────────

CLARIFICATION_REPLIES = {
    "Billing": [
        "I purchased on {cancel_date}. My account is {tier}.",
        "The charge was for {product}. I did not authorise it.",
        "I'm on the {plan} plan. The amount was ${charge_amount}.",
    ],
    "Technical": [
        "It's the {vague_feature} section. My account ID is ACC-{incident_id}.",
        "It started {vague_frequency}. I'm using the {product}.",
        "The error happens when I try to {action}. Browser is Chrome.",
    ],
    "Shipping": [
        "Order #{order_id}. I placed it on {cancel_date}.",
        "Tracking shows '{tracking_status}'. I need it urgently.",
    ],
    "Security": [
        "I noticed suspicious activity on my {tier} account {time_ago}.",
        "The unauthorized access was to {product}. I've changed my password.",
    ],
}


def simulate_customer_reply(
    ticket: Dict[str, Any],
    rng: random.Random,
) -> str:
    """Pick a reply template matching the ticket category and fill variables."""
    category = ticket.get("true_category", "Technical")
    templates = CLARIFICATION_REPLIES.get(category, CLARIFICATION_REPLIES["Technical"])
    template = rng.choice(templates)
    return _fill(template, rng)


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

    EPISODE_LENGTH: int = 20  # Number of tickets per episode

    def __init__(self) -> None:
        self.policy_registry = PolicyRegistry()
        self.drift_scheduler = DriftScheduler()
        self.world_state = WorldState()
        self._rng = random.Random()

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
        self._last_drift_step: int = 0  # step at which the most-recent drift fired
        self._pending_drift_notice: Optional[str] = None
        self._conversation_history: List[Dict[str, str]] = []
        self._attacker_tickets: List[Dict[str, Any]] = []

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
        self._last_drift_step = 0
        self._pending_drift_notice = None
        self._conversation_history = []
        self._defender_rewards = []
        self._attacker_rewards = []

        if seed is not None:
            self._rng = random.Random(seed)

        # Preserve attacker deque across episodes, reset everything else
        self.world_state.reset_episode(preserve_attacker_deque=True)
        if difficulty_init is not None:
            self.world_state.difficulty_level = difficulty_init
        self.policy_registry.reset()

        # Task 1: cap difficulty at 0.2
        if task_id == 1:
            self.world_state.difficulty_level = min(0.2, difficulty_init)
            self._drift_enabled = False  # Drift disabled for task 1

        # Randomize drift schedule each episode (two events at random steps in [1, 20])
        if self._drift_enabled:
            self.drift_scheduler = DriftScheduler.randomize(max_step=20, rng=self._rng)
        else:
            self.drift_scheduler = DriftScheduler()  # default schedule (unused)

        # Generate ticket queue — 20 tickets per episode
        self._ticket_queue = self._generate_ticket_batch(n=20)

        # Build first observation
        return self._build_observation(
            ticket=self._ticket_queue[0],
            drift_notice=None,
        )

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process one agent action. Returns observation dict with reward, done, etc.
        """
        if self._done:
            raise RuntimeError("Episode already closed. Call reset() first.")

        self._current_step += 1
        ticket = self._ticket_queue[self.world_state.tickets_processed]

        # ── Check for drift ──────────────────────────────────────────────
        drift_notice = None
        if self._drift_enabled:
            event = self.drift_scheduler.check_step(self._current_step)
            if event:
                self.drift_scheduler.apply(event, self.world_state, self.policy_registry)
                self._was_post_drift = True
                self._last_drift_step = self._current_step
                drift_notice = event.notice_text
                self._pending_drift_notice = event.notice_text

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
        defender_reward, breakdown = calculate_defender_reward(
            action=action,
            ticket=ticket,
            active_policy=active_policy,
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

        # Reset post-drift flag one step after the drift fired
        if self._was_post_drift and self._current_step > self._last_drift_step:
            self._was_post_drift = False

        # Check done
        done = self.world_state.tickets_processed >= self.EPISODE_LENGTH
        self._done = done

        # Build next observation (queue is always padded to EPISODE_LENGTH)
        next_obs = None
        if not done:
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

        # Include ticket fields based on active schema
        for field_name in active_policy.ticket_schema_fields:
            if field_name in ticket:
                obs[field_name] = ticket[field_name]

        # System notice (only if drift just fired)
        if drift_notice:
            obs["system_notice"] = drift_notice

        # Conversation history (only if multi-turn active)
        if self.world_state.multi_turn_active and self._conversation_history:
            obs["conversation_history"] = self._conversation_history

        return obs

    def _generate_ticket_batch(self, n: int = 20) -> List[Dict[str, Any]]:
        """Generate n clean procedural tickets from blueprints (default: 20 per episode)."""
        policy = self.policy_registry.get_active()

        # Filter blueprints based on task
        if self._task_id == 1:
            pool = [b for b in TICKET_BLUEPRINTS if not b.get("is_ambiguous", False)]
        elif self._task_id == 3:
            # Weight ambiguous tickets higher for multi-turn
            ambig = [b for b in TICKET_BLUEPRINTS if b.get("is_ambiguous", False)]
            clear = [b for b in TICKET_BLUEPRINTS if not b.get("is_ambiguous", False)]
            pool = ambig * 2 + clear
        else:
            pool = list(TICKET_BLUEPRINTS)

        # Filter out Security tickets if not v3
        if self.world_state.current_policy_version != "v3":
            pool = [b for b in pool if b["true_category"] != "Security"]

        tickets = []
        for _ in range(n):
            bp = self._rng.choice(pool)
            ticket = generate_ticket_clean(bp, policy, self._rng)
            tickets.append(ticket)

        return tickets

    def set_attacker_tickets(self, tickets: List[Dict[str, Any]]) -> None:
        """
        Replace the ticket queue with attacker-generated tickets.
        If fewer than EPISODE_LENGTH tickets are provided, the queue is
        automatically padded with clean procedural tickets so the episode
        always runs to exactly EPISODE_LENGTH steps.
        """
        if len(tickets) < self.EPISODE_LENGTH:
            shortfall = self.EPISODE_LENGTH - len(tickets)
            tickets = list(tickets) + self._generate_ticket_batch(n=shortfall)
        self._ticket_queue = tickets[:self.EPISODE_LENGTH]
        self._attacker_tickets = self._ticket_queue

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
