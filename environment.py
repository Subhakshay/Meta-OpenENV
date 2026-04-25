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
from drift_scheduler import DriftScheduler, DriftEvent, build_drift_notice
from rewards import calculate_defender_reward, calculate_attacker_reward
from template_sampler import TemplateSampler
from generation_engine import GenerationEngine


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
    # ── Critical / Security ──
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
    # in observations when policy requires sentiment
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

    def __init__(self) -> None:
        self.policy_registry = PolicyRegistry()
        self.drift_scheduler = DriftScheduler(episode_length=12, registry=self.policy_registry)
        self.world_state = WorldState()
        self._rng = random.Random(42)

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
        self._attacker_agent: Any = None # Keep a reference for resampling
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
        difficulty_init: float = 0.3,
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

        # Preserve attacker deque across episodes, reset everything else
        self.world_state.reset_episode(preserve_attacker_deque=True)
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
        import json as _json
        import os as _os
        _seed_bank_path = _os.path.join(_os.path.dirname(__file__), "seed_bank.json")
        try:
            with open(_seed_bank_path, "r") as f:
                _seed_bank = _json.load(f)
        except (FileNotFoundError, _json.JSONDecodeError):
            _seed_bank = []
        self._template_sampler = TemplateSampler(_seed_bank, GenerationEngine())
        self._last_reconcile_record = None

        # Generate ticket queue
        self._ticket_queue = self._generate_ticket_batch(n=12)

        # Build first observation
        return self._build_observation(
            ticket=self._ticket_queue[0],
            drift_notice=None,
        )

    def _reconcile_queue(self, new_policy: PolicyVersion, old_policy: PolicyVersion, step: int) -> int:
        """
        Check all unshown tickets against the new policy using the sampler's
        four compatibility rules. Replace incompatible tickets via sampler.sample().
        Writes one drift_events record. Returns the count of tickets replaced.
        """
        replaced = 0
        start_idx = self.world_state.tickets_processed

        for i in range(start_idx, len(self._ticket_queue)):
            ticket = self._ticket_queue[i]

            # ── Four compatibility checks ────────────────────────────────
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
            if compatible and ticket.get("_requires_empathy_trigger", False):
                if new_policy.empathy_required_below_sentiment is None:
                    compatible = False

            if not compatible:
                # Build a blueprint from the ticket's ground truth
                blueprint = {
                    "true_priority": ticket.get("true_priority", "Medium"),
                    "true_category": ticket.get("true_category", "Technical"),
                    "strategy": ticket.get("deception_strategy", "priority_camouflage"),
                }

                # If category itself is now invalid, pick one that IS valid
                if blueprint["true_category"] not in new_policy.valid_categories:
                    blueprint["true_category"] = self._rng.choice(list(new_policy.valid_categories))

                replacement = None
                if self._template_sampler is not None:
                    replacement = self._template_sampler.sample(
                        blueprint=blueprint,
                        difficulty_level=self.world_state.difficulty_level,
                        active_policy=new_policy,
                    )

                if replacement is not None:
                    # Convert generation engine output into a queue ticket
                    gen_meta = replacement.get("ground_truth_metadata", {})
                    new_ticket = {
                        "ticket_id": f"TKT-{self._rng.randint(10000, 99999)}",
                        "subject": replacement["ticket_string"].split(".")[0][:60],
                        "body": replacement["ticket_string"],
                        "tier": self._rng.choice(VARS["tier"]),
                        "true_priority": gen_meta.get("priority", blueprint["true_priority"]),
                        "true_category": gen_meta.get("category", blueprint["true_category"]),
                        "base_requires_escalation": gen_meta.get("priority") == "Critical",
                        "deception_strategy": gen_meta.get("strategy", "clean"),
                        "schema_violation": gen_meta.get("strategy") == "schema_exploitation",
                        "is_ambiguous": False,
                        "days_since_purchase": replacement.get("boundary_exploit_day_count", self._rng.randint(1, 60)),
                    }
                    new_ticket["true_refund_eligible"] = new_ticket["days_since_purchase"] <= new_policy.refund_window_days

                    if replacement.get("committed_sentiment_score") is not None:
                        new_ticket["sentiment_score"] = replacement["committed_sentiment_score"]
                        new_ticket["_requires_empathy_trigger"] = True
                    else:
                        new_ticket["sentiment_score"] = round(self._rng.random(), 2)

                    new_ticket["account_age_days"] = self._rng.randint(1, 2000)
                    self._ticket_queue[i] = new_ticket
                else:
                    # Fallback: regenerate a clean procedural ticket
                    fallback_bps = [bp for bp in TICKET_BLUEPRINTS
                                    if bp["true_category"] in new_policy.valid_categories]
                    if fallback_bps:
                        bp = self._rng.choice(fallback_bps)
                        self._ticket_queue[i] = generate_ticket_clean(bp, new_policy, self._rng)

                replaced += 1

        # Write reconciliation record to DB (fire-and-forget via the caller)
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
        ticket = self._ticket_queue[self.world_state.tickets_processed]

        # ── Check for drift ──────────────────────────────────────────────
        drift_notice = None
        if self._drift_enabled:
            event = next((e for e in self._drift_events if e.fires_at_step == self._current_step), None)
            if event:
                old_policy = self.policy_registry.get_active()
                # Apply the event's new policy to the registry
                # (Since schedule generated them, we just append to history)
                # Note: actually schedule_episode already put them in schedule but didn't set active.
                # Let's just push it to registry history so get_active() returns it.
                self.policy_registry._history.append(event.new_policy)
                self.world_state.record_drift_event(event.new_policy.version_id)
                self._was_post_drift = True
                
                drift_notice = build_drift_notice(old_policy, event.new_policy, self._current_step)
                self._pending_drift_notice = drift_notice
                
                # Reconcile the queue
                self._reconcile_queue(event.new_policy, old_policy, self._current_step)

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

        # Reset post-drift flag after the drift step is processed
        if self._was_post_drift and self._current_step > (
            max(e.fires_at_step for e in self._drift_events)
            if self._drift_events else 0
        ):
            self._was_post_drift = False

        # Check done
        done = self.world_state.tickets_processed >= 12
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

    def _generate_ticket_batch(self, n: int = 12) -> List[Dict[str, Any]]:
        """Generate n clean procedural tickets from blueprints."""
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

        # Filter valid categories based on active policy
        pool = [b for b in pool if b["true_category"] in policy.valid_categories]

        tickets = []
        for _ in range(n):
            bp = self._rng.choice(pool)
            ticket = generate_ticket_clean(bp, policy, self._rng)
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
