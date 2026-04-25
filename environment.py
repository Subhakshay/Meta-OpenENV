"""
environment.py — CustomerSupportEnv v2

Five design principles that address all reviewer concerns:

1. TRUE difficulty progression — the environment changes structurally per task:
   - task_1 (easy):   clear-signal tickets, only priority scored
   - task_2 (medium): ambiguous tickets, priority + category + response scored
   - task_3 (hard):   multi-turn dialogue — agent must ASK a clarifying question
                       on ambiguous tickets, then RESOLVE; world state is scored

2. STATEFUL MDP — agent actions have real consequences on world state:
   - company_balance  decreases when refunds are processed
   - customer_churn_risk spikes on bad responses; drops on good ones
   - escalation_queue fills up (limited capacity; overflow triggers penalty)
   - sla_breach_count tracks missed critical SLAs; penalises final score

3. PROCEDURAL generation — 8 blueprint families × random fill-in variables
   give millions of unique ticket combinations; memorisation is impossible

4. Ground truth withheld until episode end — info dict reveals true labels
   only when done=True, preventing look-ahead cheating

5. Shaped rewards — per-step reward is partial; world_state score adds
   episode-level incentive for consistent, high-quality triage decisions
"""

import random
import re
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class Priority(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class Category(str, Enum):
    BILLING   = "billing"
    TECHNICAL = "technical"
    ACCOUNT   = "account"
    SHIPPING  = "shipping"
    GENERAL   = "general"
    REFUND    = "refund"


class ActionType(str, Enum):
    CLASSIFY = "classify"  # task_1 / task_2 / task_3 final resolution
    ASK      = "ask"       # task_3 only: request a clarifying question


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class Ticket(BaseModel):
    id:                     str
    subject:                str
    body:                   str
    customer_tier:          str        # free | pro | enterprise
    created_at:             str        # ISO-8601 UTC
    sentiment:              str        # angry | neutral | positive
    true_priority:          Priority   # NOT exposed in Observation
    true_category:          Category   # NOT exposed in Observation
    requires_clarification: bool       # True → agent should ASK before resolving
    hidden_account_id:      Optional[str] = None  # revealed after agent asks


class Observation(BaseModel):
    # Ticket fields visible to the agent (no ground truth)
    ticket_id:              str
    subject:                str
    body:                   str
    customer_tier:          str
    created_at:             str
    sentiment:              str
    # Multi-turn state (task_3)
    clarification_history:  List[Dict[str, str]] = Field(default_factory=list)
    awaiting_clarification: bool = False
    customer_reply:         Optional[str] = None
    # Episode context
    queue_size:             int
    time_elapsed_seconds:   float
    agent_actions_taken:    int
    task_id:                str
    hint:                   Optional[str] = None
    # Live world state (Flaw 2: agent can observe consequences of past actions)
    world_state:            Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    action_type:         ActionType = ActionType.CLASSIFY
    assign_priority:     Priority   = Priority.MEDIUM
    assign_category:     Category   = Category.GENERAL
    response_text:       str        = ""
    escalate:            bool       = False
    clarifying_question: str        = ""   # used with action_type=ASK


class Reward(BaseModel):
    value:     float = Field(ge=-1.0, le=1.0)
    breakdown: Dict[str, float]
    done:      bool
    info:      Dict[str, Any]


class WorldState(BaseModel):
    """Stateful consequences that persist across the entire episode."""
    company_balance:      float = 10_000.0
    escalation_queue:     int   = 0
    escalation_capacity:  int   = 3
    customer_churn_risk:  float = 0.0     # 0.0–1.0
    sla_breach_count:     int   = 0
    tickets_resolved:     int   = 0
    tickets_escalated:    int   = 0
    _quality_sum:         float = 0.0
    _quality_n:           int   = 0
    avg_response_quality: float = 0.0

    def record_quality(self, q: float):
        self._quality_sum += q
        self._quality_n   += 1
        self.avg_response_quality = round(self._quality_sum / self._quality_n, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Procedural ticket generator  (Flaw 3 fix)
# ─────────────────────────────────────────────────────────────────────────────

_TICKET_BLUEPRINTS = [
    # ── CRITICAL / TECHNICAL  (clear signal) ─────────────────────────────────
    {
        "subjects": [
            "URGENT: Production API returning 500 errors",
            "Critical outage — payment processing completely down",
            "EMERGENCY: Database connection pool exhausted",
            "Complete system failure — all users locked out",
        ],
        "bodies": [
            "Our {tier} integration has been down for {duration}. "
            "We are losing ${loss} per minute. {user_count} users cannot access the platform. "
            "Reference incident #{incident_id}.",
            "Since {time_ago}, every request to your API returns a 500. "
            "Our SLA demands 99.9% uptime. We need an engineer immediately. Incident #{incident_id}.",
            "The production environment is completely unreachable. "
            "{user_count} active sessions were dropped. This started at {time_ago}.",
        ],
        "true_priority":          Priority.CRITICAL,
        "true_category":          Category.TECHNICAL,
        "sentiment":              "angry",
        "requires_clarification": False,
        "tiers":                  ["enterprise", "pro"],
    },
    # ── HIGH / ACCOUNT  (clear signal) ───────────────────────────────────────
    {
        "subjects": [
            "Cannot log in to my account",
            "Account access lost after password reset",
            "Two-factor authentication locking me out",
            "Login credentials rejected — urgent",
        ],
        "bodies": [
            "I have been unable to log in for {duration}. I haven't changed my password. "
            "My account email is {email}. I need access restored immediately.",
            "After resetting my password I still cannot get in. "
            "The reset link expired. This is my {plan} account and I have a deadline today.",
            "The 2FA code is not accepted even though my phone time is synced. "
            "I've tried {attempts} times. Please help.",
        ],
        "true_priority":          Priority.HIGH,
        "true_category":          Category.ACCOUNT,
        "sentiment":              "neutral",
        "requires_clarification": False,
        "tiers":                  ["free", "pro", "enterprise"],
    },
    # ── HIGH / BILLING  (clear signal) ───────────────────────────────────────
    {
        "subjects": [
            "Duplicate charge on my account this month",
            "Unexpected charge of ${charge_amount}",
            "Billed for cancelled subscription",
        ],
        "bodies": [
            "I was charged ${charge_amount} twice this billing cycle. "
            "Card ending in {card_last4} shows two transactions. Please refund the duplicate.",
            "Despite cancelling my {plan} plan on {cancel_date}, I was still charged ${charge_amount}. "
            "I need an immediate refund and cancellation confirmation.",
            "Invoice #{invoice_id} shows ${charge_amount} but my plan is only ${expected_amount}/month. "
            "There seems to be an error.",
        ],
        "true_priority":          Priority.HIGH,
        "true_category":          Category.BILLING,
        "sentiment":              "neutral",
        "requires_clarification": False,
        "tiers":                  ["pro", "enterprise"],
    },
    # ── MEDIUM / TECHNICAL  (AMBIGUOUS — Flaw 1 fix) ─────────────────────────
    {
        "subjects": [
            "Something isn't working like it used to",
            "The dashboard seems off today",
            "Feature behaving unexpectedly",
            "Things are slow and I'm not sure why",
        ],
        "bodies": [
            "Hi, the thing I normally use for {vague_task} isn't doing what it did {time_ago}. "
            "I haven't changed anything. Can you look into it?",
            "Something's off with my {vague_feature}. It used to {vague_behavior} but now it doesn't. "
            "Not sure if it's on my side or yours.",
            "The {vague_feature} is acting weird. Sometimes it works, sometimes it doesn't. "
            "Happened {vague_frequency}.",
        ],
        "true_priority":          Priority.MEDIUM,
        "true_category":          Category.TECHNICAL,
        "sentiment":              "neutral",
        "requires_clarification": True,   # task_3: agent must ASK
        "tiers":                  ["free", "pro"],
    },
    # ── MEDIUM / BILLING  (AMBIGUOUS) ────────────────────────────────────────
    {
        "subjects": [
            "Question about my recent charges",
            "Not sure if I'm on the right plan",
            "Billing looks different this month",
        ],
        "bodies": [
            "Hey, my bill this month looks different from last month. "
            "I didn't change anything. Is there a price increase?",
            "I'm not totally sure what plan I'm on or if I'm being charged correctly. "
            "Can someone check? I don't want to be overcharged.",
            "The charges on my account seem off but I might be misreading the invoice. "
            "Can you help me understand each line item?",
        ],
        "true_priority":          Priority.MEDIUM,
        "true_category":          Category.BILLING,
        "sentiment":              "neutral",
        "requires_clarification": True,
        "tiers":                  ["free", "pro"],
    },
    # ── LOW / GENERAL  (clear signal) ────────────────────────────────────────
    {
        "subjects": [
            "Quick question about your features",
            "Interested in upgrading my plan",
            "Feedback on your product",
        ],
        "bodies": [
            "Hi! I was wondering if your {plan} plan includes {feature_question}. "
            "No rush, just comparing options.",
            "Love the product! One small thing — {minor_complaint}. "
            "Just wanted to flag it. Keep up the great work!",
            "Hello, I'm trying to figure out how to {how_to_task}. "
            "I looked at the docs but couldn't find it. Thanks!",
        ],
        "true_priority":          Priority.LOW,
        "true_category":          Category.GENERAL,
        "sentiment":              "positive",
        "requires_clarification": False,
        "tiers":                  ["free", "pro", "enterprise"],
    },
    # ── HIGH / SHIPPING  (clear signal) ──────────────────────────────────────
    {
        "subjects": [
            "Order #{order_id} hasn't arrived after {weeks} weeks",
            "Package marked delivered but not received",
            "Wrong item shipped — order #{order_id}",
        ],
        "bodies": [
            "I placed order #{order_id} on {order_date} and it still hasn't arrived. "
            "Tracking shows '{tracking_status}'. I need this for a client deadline.",
            "My tracking says delivered {time_ago} but there's nothing here. "
            "Checked with neighbours and building management. Order #{order_id}.",
            "I received the wrong item. Ordered {ordered_item} but got {received_item}. "
            "Order #{order_id}. Please send the correct item urgently.",
        ],
        "true_priority":          Priority.HIGH,
        "true_category":          Category.SHIPPING,
        "sentiment":              "angry",
        "requires_clarification": False,
        "tiers":                  ["pro", "enterprise"],
    },
    # ── MEDIUM / REFUND  (clear signal) ──────────────────────────────────────
    {
        "subjects": [
            "Requesting refund for service issues",
            "Partial refund request",
            "Refund for unused subscription period",
        ],
        "bodies": [
            "I'd like a refund for {refund_period} because the service {refund_reason}. "
            "I believe I'm entitled under your SLA.",
            "I cancelled on {cancel_date} but was charged for another cycle. "
            "Please refund ${refund_amount} for the unused period.",
            "The {feature} I paid for hasn't worked correctly for {duration}. "
            "I'd like a partial refund for the affected period.",
        ],
        "true_priority":          Priority.MEDIUM,
        "true_category":          Category.REFUND,
        "sentiment":              "neutral",
        "requires_clarification": False,
        "tiers":                  ["pro", "enterprise"],
    },
]

# Variable fill-in pools
_VARS: Dict[str, List[str]] = {
    "duration":         ["2 hours", "4 hours", "since this morning", "over 6 hours", "the last 3 hours"],
    "loss":             ["500", "1,200", "3,000", "800", "250"],
    "user_count":       ["120", "500", "2,000", "50,000", "300"],
    "incident_id":      [f"{n:06d}" for n in range(100001, 100021)],
    "time_ago":         ["2:00 AM UTC", "08:30 EST", "14:15 PST", "09:00 GMT", "midnight"],
    "email":            ["user@company.com", "admin@startup.io", "billing@corp.net"],
    "attempts":         ["5", "10", "over 20", "3"],
    "charge_amount":    ["49.99", "99.00", "299.00", "19.99", "149.50"],
    "card_last4":       ["4242", "1234", "9876", "5555"],
    "plan":             ["Pro", "Enterprise", "Business", "Starter"],
    "cancel_date":      ["March 1st", "February 15th", "last Tuesday", "January 31st"],
    "invoice_id":       [f"INV-{n}" for n in range(10001, 10021)],
    "expected_amount":  ["49.99", "29.99", "99.00", "199.00"],
    "vague_task":       ["reporting", "the exports", "data syncing", "notifications", "the dashboard"],
    "vague_feature":    ["main panel", "analytics tab", "export tool", "webhook", "integration"],
    "vague_behavior":   ["update automatically", "show the right numbers", "sync properly", "load fast"],
    "vague_frequency":  ["twice this week", "three times today", "randomly", "since the last update"],
    "feature_question": ["API access", "team collaboration", "custom reports", "SSO", "dedicated support"],
    "minor_complaint":  ["the export is slow", "dark mode would be great", "the UI is slightly confusing"],
    "how_to_task":      ["export data to CSV", "set up webhooks", "invite team members", "enable 2FA"],
    "order_id":         [f"ORD-{n}" for n in range(10001, 10021)],
    "weeks":            ["2", "3", "4", "nearly 3"],
    "order_date":       ["March 1st", "February 28th", "three weeks ago", "March 5th"],
    "tracking_status":  ["In Transit", "Out for Delivery", "Delayed", "Pending"],
    "ordered_item":     ["the Pro Kit bundle", "Model X adapter", "the annual license key"],
    "received_item":    ["a Starter Kit", "an unrelated item", "an empty box", "the wrong size"],
    "refund_period":    ["last month", "the past two months", "March", "Q1"],
    "refund_reason":    ["was down for 3 days", "had repeated outages", "did not meet the SLA"],
    "refund_amount":    ["49.99", "99.00", "29.99", "199.00"],
    "feature":          ["API integration", "export tool", "real-time sync", "analytics dashboard"],
    "tier":             ["enterprise", "Pro", "Business"],
}


def _fill(template: str, rng: random.Random) -> str:
    def replacer(m):
        key  = m.group(1)
        pool = _VARS.get(key, [f"[{key}]"])
        return rng.choice(pool)
    return re.sub(r"\{(\w+)\}", replacer, template)


def _generate_ticket(rng: random.Random, task_id: str) -> Ticket:
    """
    Procedurally generate a ticket appropriate for the given task.
    task_1: only unambiguous blueprints
    task_2: full mix (some ambiguous)
    task_3: weighted towards blueprints that require clarification
    """
    if task_id == "task_1_priority":
        pool = [b for b in _TICKET_BLUEPRINTS if not b["requires_clarification"]]
    elif task_id == "task_2_classification":
        pool = _TICKET_BLUEPRINTS
    else:
        clear = [b for b in _TICKET_BLUEPRINTS if not b["requires_clarification"]]
        ambig = [b for b in _TICKET_BLUEPRINTS if b["requires_clarification"]]
        pool  = ambig * 2 + clear  # 2:1 ratio

    bp        = rng.choice(pool)
    tier      = rng.choice(bp["tiers"])
    subject   = _fill(rng.choice(bp["subjects"]), rng)
    body      = _fill(rng.choice(bp["bodies"]), rng)
    ticket_id = f"TKT-{rng.randint(10000, 99999)}"
    created   = (datetime.utcnow() - timedelta(minutes=rng.randint(1, 480))).isoformat() + "Z"
    hidden    = f"ACC-{rng.randint(100000, 999999)}" if bp["requires_clarification"] else None

    return Ticket(
        id=ticket_id, subject=subject, body=body,
        customer_tier=tier, created_at=created, sentiment=bp["sentiment"],
        true_priority=bp["true_priority"], true_category=bp["true_category"],
        requires_clarification=bp["requires_clarification"],
        hidden_account_id=hidden,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Grading functions
# ─────────────────────────────────────────────────────────────────────────────

_PRIORITY_ORDER = [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.CRITICAL]


def grade_priority(assigned: Priority, true: Priority) -> float:
    diff = abs(_PRIORITY_ORDER.index(assigned) - _PRIORITY_ORDER.index(true))
    return {0: 1.0, 1: 0.6, 2: 0.2, 3: 0.0}[diff]


def grade_category(assigned: Category, true: Category) -> float:
    if assigned == true:
        return 1.0
    _related = {
        Category.BILLING:   {Category.REFUND},
        Category.REFUND:    {Category.BILLING},
        Category.ACCOUNT:   {Category.TECHNICAL},
        Category.TECHNICAL: {Category.ACCOUNT},
    }
    return 0.4 if assigned in _related.get(true, set()) else 0.0


def grade_response(response_text: str, ticket: Ticket, action: Action) -> float:
    if not response_text or len(response_text.strip()) < 30:
        return 0.0
    score = 0.0
    text  = response_text.lower()

    if any(w in text for w in ["hello", "hi ", "dear", "greetings", "thank you for"]):
        score += 0.15

    subject_words = {
        w for w in re.split(r'\W+', ticket.subject.lower())
        if len(w) > 3 and w not in {"with", "your", "this", "that", "have", "been", "urgent"}
    }
    if subject_words and any(w in text for w in subject_words):
        score += 0.20

    _cat_kw: Dict[Category, List[str]] = {
        Category.BILLING:   ["charge", "billing", "payment", "invoice", "plan", "subscription"],
        Category.TECHNICAL: ["issue", "bug", "error", "technical", "investigate", "fix", "resolve"],
        Category.ACCOUNT:   ["account", "login", "password", "access", "credentials", "reset"],
        Category.SHIPPING:  ["order", "ship", "delivery", "tracking", "package"],
        Category.REFUND:    ["refund", "reimburse", "credit", "return", "compensation"],
        Category.GENERAL:   ["help", "assist", "support", "question"],
    }
    if any(w in text for w in _cat_kw.get(ticket.true_category, [])):
        score += 0.25

    if any(p in text for p in ["will", "we'll", "resolve", "fix", "contact you",
                                 "follow up", "look into", "investigate", "get back"]):
        score += 0.20

    expected_escalate = ticket.true_priority == Priority.CRITICAL
    score += 0.10 if action.escalate == expected_escalate else -0.05

    if any(w in text for w in ["regards", "sincerely", "best", "support team", "kind regards"]):
        score += 0.10

    return round(max(0.0, min(1.0, score)), 4)


def grade_clarification(question: str, ticket: Ticket) -> float:
    """Grade whether the clarifying question is relevant and specific."""
    if not question or len(question.strip()) < 10:
        return 0.0
    score = 0.0
    text  = question.lower()
    if any(w in text for w in ["could you", "can you", "please", "which", "what", "when", "describe"]):
        score += 0.3
    if any(w in text for w in ["feature", "page", "section", "tool", "account", "specific", "area"]):
        score += 0.4
    if any(w in text for w in ["steps", "reproduce", "how long", "started", "browser", "error message"]):
        score += 0.3
    return round(min(1.0, score), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Task configuration
# ─────────────────────────────────────────────────────────────────────────────

TASK_CONFIG = {
    "task_1_priority": {
        "hint": (
            "Tickets have clear, unambiguous signals. "
            "Assign the correct priority. Only priority is scored."
        ),
        "weights":    {"priority": 1.0},
        "multi_turn": False,
        "max_steps":  10,
    },
    "task_2_classification": {
        "hint": (
            "Some tickets are ambiguous. Assign priority AND category "
            "and write a quality response. All three are scored."
        ),
        "weights":    {"priority": 0.4, "category": 0.3, "response": 0.3},
        "multi_turn": False,
        "max_steps":  10,
    },
    "task_3_full_triage": {
        "hint": (
            "Multi-turn triage. Ambiguous tickets REQUIRE clarification. "
            "Use action_type='ask' with a clarifying_question first when the ticket is unclear, "
            "then action_type='classify' with full classification and response. "
            "Your decisions affect world state: company_balance, customer_churn_risk, "
            "escalation_queue, and sla_breach_count. These impact your episode score."
        ),
        "weights":    {
            "priority":      0.25,
            "category":      0.20,
            "response":      0.25,
            "clarification": 0.15,
            "world_state":   0.15,
        },
        "multi_turn": True,
        "max_steps":  20,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class CustomerSupportEnv:
    TASK_IDS = list(TASK_CONFIG.keys())

    def __init__(self, task_id: str = "task_1_priority", seed: int = 42):
        if task_id not in self.TASK_IDS:
            raise ValueError(f"task_id must be one of {self.TASK_IDS}")
        self.task_id = task_id
        self.seed    = seed
        self._cfg    = TASK_CONFIG[task_id]
        self._rng    = random.Random(seed)
        self._init_state()

    def _init_state(self):
        self._start_time:             float            = 0.0
        self._step_count:             int              = 0
        self._episode_done:           bool             = False
        self._current_ticket:         Optional[Ticket] = None
        self._ticket_queue:           List[Ticket]     = []
        self._world:                  WorldState       = WorldState()
        self._episode_log:            List[Dict]       = []
        self._awaiting_clarification: bool             = False
        self._clarification_history:  List[Dict]       = []
        self._clarification_score:    float            = 0.0

    def _num_tickets(self) -> int:
        # multi-turn uses half the steps per ticket (ASK + CLASSIFY = 2 steps)
        return self._cfg["max_steps"] // 2 if self._cfg["multi_turn"] else self._cfg["max_steps"]

    def _make_obs(self, elapsed: float) -> Observation:
        t = self._current_ticket
        return Observation(
            ticket_id=t.id, subject=t.subject, body=t.body,
            customer_tier=t.customer_tier, created_at=t.created_at, sentiment=t.sentiment,
            clarification_history=list(self._clarification_history),
            awaiting_clarification=self._awaiting_clarification,
            customer_reply=(
                self._clarification_history[-1].get("customer")
                if self._clarification_history else None
            ),
            queue_size=len(self._ticket_queue),
            time_elapsed_seconds=round(elapsed, 3),
            agent_actions_taken=self._step_count,
            task_id=self.task_id,
            hint=self._cfg["hint"],
            world_state={
                "company_balance":      self._world.company_balance,
                "escalation_queue":     self._world.escalation_queue,
                "escalation_capacity":  self._world.escalation_capacity,
                "customer_churn_risk":  self._world.customer_churn_risk,
                "sla_breach_count":     self._world.sla_breach_count,
                "tickets_resolved":     self._world.tickets_resolved,
                "avg_response_quality": self._world.avg_response_quality,
            },
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        self._rng = random.Random(self.seed)
        self._init_state()
        self._start_time  = time.time()
        n = self._num_tickets()
        self._ticket_queue   = [_generate_ticket(self._rng, self.task_id) for _ in range(n)]
        self._current_ticket = self._ticket_queue.pop(0)
        return self._make_obs(0.0)

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self._episode_done:
            raise RuntimeError("Episode is done. Call reset() first.")

        elapsed = time.time() - self._start_time
        ticket  = self._current_ticket
        self._step_count += 1

        # ── Handle ASK action (task_3 only) ───────────────────────────────────
        if self.task_id == "task_3_full_triage" and action.action_type == ActionType.ASK:
            return self._handle_ask(action, elapsed)

        # ── Grade the classification / resolution action ───────────────────────
        p_score = grade_priority(action.assign_priority, ticket.true_priority)
        c_score = grade_category(action.assign_category, ticket.true_category)
        r_score = grade_response(action.response_text, ticket, action)

        # Clarification component (task_3)
        clarif_score = 0.0
        if self.task_id == "task_3_full_triage":
            if ticket.requires_clarification:
                # Reward if agent asked before resolving; penalise if they skipped
                if self._clarification_score > 0:
                    clarif_score = self._clarification_score
                else:
                    clarif_score = 0.0
                    p_score      = max(0.0, p_score - 0.2)  # penalty for skipping
            else:
                clarif_score = 1.0  # no clarification needed; full marks

        # ── Apply stateful world effects (Flaw 2 fix) ─────────────────────────
        self._apply_world_effects(action, ticket, r_score)

        # ── Compute weighted reward ───────────────────────────────────────────
        weights = self._cfg["weights"]

        if self.task_id == "task_1_priority":
            total = weights["priority"] * p_score
            breakdown = {
                "priority_raw": round(p_score, 4),
                "priority":     round(weights["priority"] * p_score, 4),
            }

        elif self.task_id == "task_2_classification":
            total = (weights["priority"] * p_score
                     + weights["category"] * c_score
                     + weights["response"] * r_score)
            breakdown = {
                "priority_raw":  round(p_score, 4),
                "category_raw":  round(c_score, 4),
                "response_raw":  round(r_score, 4),
                "priority":      round(weights["priority"] * p_score, 4),
                "category":      round(weights["category"] * c_score, 4),
                "response":      round(weights["response"] * r_score, 4),
            }

        else:  # task_3_full_triage
            ws_score = self._world_state_score()
            total = (weights["priority"]      * p_score
                     + weights["category"]      * c_score
                     + weights["response"]      * r_score
                     + weights["clarification"] * clarif_score
                     + weights["world_state"]   * ws_score)
            breakdown = {
                "priority_raw":      round(p_score, 4),
                "category_raw":      round(c_score, 4),
                "response_raw":      round(r_score, 4),
                "clarification_raw": round(clarif_score, 4),
                "world_state_raw":   round(ws_score, 4),
                "priority":          round(weights["priority"] * p_score, 4),
                "category":          round(weights["category"] * c_score, 4),
                "response":          round(weights["response"] * r_score, 4),
                "clarification":     round(weights["clarification"] * clarif_score, 4),
                "world_state":       round(weights["world_state"] * ws_score, 4),
            }

        total = round(max(0.0, min(1.0, total)), 4)

        self._world.record_quality(r_score)
        self._world.tickets_resolved += 1
        self._episode_log.append({
            "step": self._step_count, "ticket_id": ticket.id,
            "reward": total, "breakdown": breakdown,
        })

        # ── Termination ───────────────────────────────────────────────────────
        done = (len(self._ticket_queue) == 0) or (self._step_count >= self._cfg["max_steps"])
        self._episode_done = done

        info = self._build_info(ticket, action, breakdown, done)

        # Advance to next ticket
        self._clarification_history  = []
        self._clarification_score    = 0.0
        self._awaiting_clarification = False
        if not done:
            self._current_ticket = self._ticket_queue.pop(0)

        return self._make_obs(elapsed), Reward(value=total, breakdown=breakdown, done=done, info=info), done, info

    # ── ASK handler ───────────────────────────────────────────────────────────

    def _handle_ask(self, action: Action, elapsed: float) -> Tuple[Observation, Reward, bool, Dict]:
        ticket  = self._current_ticket
        q_score = grade_clarification(action.clarifying_question, ticket)
        self._clarification_score = q_score

        if ticket.requires_clarification:
            customer_reply = (
                f"Sure! I'm referring to the {self._rng.choice(_VARS['vague_feature'])} section. "
                f"My account ID is {ticket.hidden_account_id}. "
                f"It started {self._rng.choice(_VARS['vague_frequency'])}."
            )
        else:
            customer_reply = "I think I already explained everything in my original message."

        self._clarification_history.append({
            "agent":    action.clarifying_question,
            "customer": customer_reply,
        })
        self._awaiting_clarification = True

        breakdown = {"clarification_quality": round(q_score, 4)}
        reward    = Reward(
            value=round(q_score * 0.1, 4),
            breakdown=breakdown, done=False,
            info={"action": "ask", "clarification_score": q_score, "step": self._step_count},
        )
        return self._make_obs(elapsed), reward, False, reward.info

    # ── World effects ─────────────────────────────────────────────────────────

    def _apply_world_effects(self, action: Action, ticket: Ticket, r_score: float):
        """Real MDP consequences — agent actions change world state (Flaw 2 fix)."""
        w = self._world

        if action.assign_category in (Category.REFUND, Category.BILLING) and action.escalate:
            w.company_balance -= float(self._rng.choice([49.99, 99.00, 29.99]))

        if r_score < 0.3:
            w.customer_churn_risk = min(1.0, w.customer_churn_risk + 0.15)
        elif r_score > 0.7:
            w.customer_churn_risk = max(0.0, w.customer_churn_risk - 0.05)

        if action.escalate:
            w.escalation_queue  += 1
            w.tickets_escalated += 1
            if w.escalation_queue > w.escalation_capacity:
                w.customer_churn_risk = min(1.0, w.customer_churn_risk + 0.10)

        if ticket.true_priority == Priority.CRITICAL and action.assign_priority != Priority.CRITICAL:
            w.sla_breach_count += 1

    def _world_state_score(self) -> float:
        w             = self._world
        balance_score = min(1.0, w.company_balance / 10_000.0)
        churn_score   = 1.0 - w.customer_churn_risk
        sla_score     = max(0.0, 1.0 - w.sla_breach_count * 0.25)
        queue_score   = max(0.0, 1.0 - max(0, w.escalation_queue - w.escalation_capacity) * 0.2)
        return round((balance_score + churn_score + sla_score + queue_score) / 4, 4)

    # ── Info builder ──────────────────────────────────────────────────────────

    def _build_info(self, ticket: Ticket, action: Action, breakdown: Dict, done: bool) -> Dict:
        """Ground truth revealed ONLY when done=True (Flaw 4 fix)."""
        log       = self._episode_log
        cum_r     = round(sum(e["reward"] for e in log), 4)
        mean_r    = round(cum_r / max(len(log), 1), 4)

        info: Dict[str, Any] = {
            "step":               self._step_count,
            "ticket_id":          ticket.id,
            "cumulative_reward":  cum_r,
            "mean_reward_so_far": mean_r,
            "world_state": {
                "company_balance":     self._world.company_balance,
                "customer_churn_risk": self._world.customer_churn_risk,
                "escalation_queue":    self._world.escalation_queue,
                "sla_breach_count":    self._world.sla_breach_count,
            },
        }

        if done:
            info["episode_summary"] = {
                "total_reward":          cum_r,
                "mean_reward":           mean_r,
                "tickets_resolved":      self._world.tickets_resolved,
                "avg_response_quality":  self._world.avg_response_quality,
                "final_balance":         self._world.company_balance,
                "final_churn_risk":      self._world.customer_churn_risk,
                "sla_breaches":          self._world.sla_breach_count,
                "tickets_escalated":     self._world.tickets_escalated,
            }
            info["ground_truth_log"] = [
                {"step": e["step"], "ticket_id": e["ticket_id"], "reward": e["reward"]}
                for e in self._episode_log
            ]

        return info

    def state(self) -> Dict[str, Any]:
        log = self._episode_log
        return {
            "task_id":           self.task_id,
            "step":              self._step_count,
            "ticket_id":         self._current_ticket.id if self._current_ticket else None,
            "episode_done":      self._episode_done,
            "cumulative_reward": round(sum(e["reward"] for e in log), 4) if log else 0.0,
            "world_state": {
                "company_balance":     self._world.company_balance,
                "customer_churn_risk": self._world.customer_churn_risk,
                "escalation_queue":    self._world.escalation_queue,
                "sla_breach_count":    self._world.sla_breach_count,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner helper
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(env: CustomerSupportEnv, agent_fn: Callable) -> Dict[str, Any]:
    """agent_fn(obs: Observation) -> Action"""
    obs   = env.reset()
    total = 0.0
    steps = 0
    while True:
        action          = agent_fn(obs)
        obs, reward, done, info = env.step(action)
        total          += reward.value
        steps          += 1
        if done:
            break
    return {
        "total_reward":    round(total, 4),
        "mean_reward":     round(total / max(steps, 1), 4),
        "steps":           steps,
        "episode_summary": info.get("episode_summary", {}),
    }
