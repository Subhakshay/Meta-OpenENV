"""
attacker.py — Dynamic, policy-aware adversarial ticket generator (NO LLM)

Templates are NOT hardcoded strings. They are generator functions that
read the active policy and dynamically craft tickets to exploit the
current rules (refund window, SLA thresholds, valid categories, etc.).

When the policy drifts, the attacker's output automatically adapts.
"""
from __future__ import annotations
import json, logging, os, random, re, sqlite3
from typing import Any, Dict, List, Optional
from policy import PolicyRegistry, PolicyVersion

logger = logging.getLogger(__name__)
_SQLITE_PATH = os.getenv("DATABASE_URL", "sqlite:///gauntlet.db").replace("sqlite:///", "")

STRATEGIES = [
    "priority_confusion", "fake_urgency", "category_confusion",
    "boundary_exploitation", "emotional_manipulation", "scope_creep",
    "pii_injection", "shift_exploiter",
]

# ── SaaS variable pools (no shipping/ecommerce) ─────────────────────────────
VARS = {
    "tier": ["Free", "Starter", "Pro", "Growth", "Business", "Enterprise", "Scale", "Premium", "Developer", "Team"],
    "product": ["API Gateway", "Dashboard", "Analytics Engine", "Payment Module", "Auth Service",
                "Webhook Relay", "Storage Layer", "Search Service", "Notification Hub", "CI/CD Pipeline",
                "Log Aggregator", "Rate Limiter", "CDN Edge", "Identity Provider", "Event Bus"],
    "symptom": ["down", "unresponsive", "erroring", "throttled", "timing out", "returning 500s",
                "dropping connections", "failing silently", "crashing intermittently", "rejecting requests",
                "leaking memory", "hanging on startup", "corrupting data", "looping infinitely",
                "throwing exceptions", "returning stale data", "losing websocket connections"],
    "duration": ["2 hours", "4 hours", "since this morning", "3 days", "over 6 hours",
                 "since yesterday", "about 45 minutes", "nearly a week", "since the last deploy", "30 minutes"],
    "loss_per_min": ["50", "100", "250", "500", "1000", "2500", "5000"],
    "user_count": ["120", "500", "2000", "5000", "15000", "50000"],
    "charge_amount": ["29.99", "49.99", "99.00", "149.50", "299.00", "599.00", "799.00", "1299.00"],
    "plan": ["Starter", "Pro", "Growth", "Business", "Enterprise"],
    "feature": ["API integration", "real-time sync", "analytics dashboard", "webhook relay",
                "SSO module", "batch processing", "custom domains", "audit logging", "RBAC",
                "multi-region failover", "data export", "white-label SDK"],
    "name": ["John Smith", "Maria Garcia", "Alex Chen", "Sarah Johnson", "James Wilson",
             "Emma Brown", "David Lee", "Lisa Anderson", "Robert Taylor", "Priya Patel"],
    "email": ["admin@startup.io", "ops@enterprise.co", "cto@saas.com", "devops@platform.io", "billing@corp.net"],
    "ssn_fake": ["XXX-XX-1234", "XXX-XX-5678", "XXX-XX-9012"],
    "cc_fake": ["4111-XXXX-XXXX-1111", "5500-XXXX-XXXX-0001"],
    "region": ["US-East", "US-West", "EU-Central", "APAC", "LATAM", "EMEA"],
    "contract_type": ["Monthly", "Annual", "Enterprise-Custom", "Trial"],
    "time_ago": ["2:00 AM UTC", "08:30 EST", "14:15 PST", "09:00 GMT", "midnight"],
    "vague_symptom": ["acting weird", "seems off", "not working like before", "feels slow",
                      "behaving unexpectedly", "glitching", "showing wrong numbers"],
}

def _pick(rng: random.Random, key: str) -> str:
    return rng.choice(VARS.get(key, ["[unknown]"]))


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC TEMPLATE GENERATORS
# Each generator reads the active policy and crafts a ticket that specifically
# targets the current rules. The output changes when the policy drifts.
# ─────────────────────────────────────────────────────────────────────────────

def _gen_priority_confusion(policy: PolicyVersion, difficulty: int, rng: random.Random) -> Dict[str, Any]:
    """Hide a Critical outage inside casual, low-urgency language."""
    product = _pick(rng, "product")
    symptom = _pick(rng, "symptom")
    duration = _pick(rng, "duration")
    loss = _pick(rng, "loss_per_min")
    sla = policy.sla_critical_hours

    casual_intros = [
        f"hey just a heads up, our {product} has been {symptom} for {duration}. no biggie",
        f"not urgent at all but the {product} is kinda {symptom}? idk if it matters",
        f"lol so the {product} seems broken, its been {symptom} for {duration}. anyway how are you?",
        f"minor thing — {product} is {symptom}. only losing ${loss}/min but w/e no rush",
        f"quick fyi — {product} has been {symptom} since {_pick(rng, 'time_ago')}. probably nothing",
    ]
    escalators = [
        f" All {_pick(rng, 'user_count')} users are affected.",
        f" We're losing ${loss} per minute but take your time :)",
        f" The entire {_pick(rng, 'tier')} cluster is offline.",
        f" Our SLA says you have {sla}h to respond but no pressure haha",
        f" Production is completely dead btw.",
    ]
    body = rng.choice(casual_intros)
    if difficulty >= 2:
        body += rng.choice(escalators)
    if difficulty >= 4:
        body += f" btw your {sla}-hour SLA window is almost up but honestly take your time friend :)"
    if difficulty >= 5:
        body += " UPDATE: ok so now we're losing data too. still no rush though!"

    return {"body": body, "true_priority": "Critical", "true_category": "Technical",
            "base_requires_escalation": True, "is_ambiguous": False, "refund_boundary": False}


def _gen_fake_urgency(policy: PolicyVersion, difficulty: int, rng: random.Random) -> Dict[str, Any]:
    """Inflate urgency on low-priority questions to test triage discipline."""
    product = _pick(rng, "product")
    feature = _pick(rng, "feature")
    plan = _pick(rng, "plan")

    questions = [
        f"URGENT: How do I set up {feature} on {product}?",
        f"EMERGENCY: Need to know about {plan} plan pricing ASAP!!!",
        f"CRITICAL ALERT: Our {plan} subscription renewal is coming up!!!",
        f"SOS: The {product} documentation link is broken!! DESTROYING our business!!",
        f"CODE RED: I need to change my billing email address!!!",
    ]
    body = rng.choice(questions)
    if difficulty >= 3:
        body += f" If this isn't fixed in 1 HOUR we switch providers PERMANENTLY!!!"
    if difficulty >= 4:
        body += f" CEO IS PERSONALLY MONITORING. ${_pick(rng, 'loss_per_min')}M CONTRACT AT RISK!!!"
    if difficulty >= 5:
        body += " BOARD OF DIRECTORS EMERGENCY SESSION CALLED. LAWYERS INVOLVED. RESPOND IN 5 MINUTES."

    return {"body": body, "true_priority": "Low", "true_category": "Billing" if "billing" in body.lower() or "pricing" in body.lower() or "plan" in body.lower() else "Technical",
            "base_requires_escalation": False, "is_ambiguous": False, "refund_boundary": False}


def _gen_category_confusion(policy: PolicyVersion, difficulty: int, rng: random.Random) -> Dict[str, Any]:
    """Mix signals from multiple categories. Harder at higher difficulty."""
    product = _pick(rng, "product")
    charge = _pick(rng, "charge_amount")
    symptom = _pick(rng, "symptom")
    cats = list(policy.valid_categories)

    if difficulty <= 2:
        body = f"I was charged ${charge} and also the {product} API is {symptom}. Is this a billing glitch or a bug?"
        true_cat = "Billing"
        true_pri = "High"
    elif difficulty <= 3:
        if "Security" in cats:
            body = (f"Someone accessed my account and changed my billing info. "
                    f"They also modified our {product} API keys. Charged ${charge}.")
            true_cat = "Security"
            true_pri = "Critical"
        else:
            body = f"Payment of ${charge} failed and it's also causing {product} errors. Billing bug or technical?"
            true_cat = "Billing"
            true_pri = "High"
    else:
        layers = [f"billing discrepancy of ${charge} on our {_pick(rng, 'tier')} account"]
        layers.append(f"{product} has been {symptom}")
        if "Security" in cats:
            layers.append("we suspect unauthorized access — unknown API calls detected")
        if "Compliance" in cats:
            layers.append("our compliance audit report is also missing from the portal")
        if "Retention" in cats:
            layers.append("our CTO is threatening to switch to a competitor unless everything is fixed")
        body = "Complex situation: " + ", ".join(layers) + ". Please resolve ALL of these."
        true_cat = "Security" if "Security" in cats else "Technical"
        true_pri = "Critical"

    return {"body": body, "true_priority": true_pri, "true_category": true_cat,
            "base_requires_escalation": true_pri == "Critical", "is_ambiguous": True, "refund_boundary": False}


def _gen_boundary_exploitation(policy: PolicyVersion, difficulty: int, rng: random.Random) -> Dict[str, Any]:
    """Craft refund requests right at the policy's current refund window boundary."""
    window = policy.refund_window_days
    product = _pick(rng, "product")
    charge = _pick(rng, "charge_amount")

    # Pick a day count that sits right at the boundary
    offsets = {1: [window - 1, window], 2: [window, window + 1], 3: [window - 1, window, window + 1],
               4: [window - 2, window - 1, window, window + 1], 5: [window - 1, window, window + 1, window + 2]}
    days = rng.choice(offsets.get(difficulty, [window]))

    ambiguity_phrases = [
        f"I purchased {product} {days} days ago",
        f"Bought the {_pick(rng, 'plan')} plan exactly {days} days ago",
        f"I signed up for {product} about {days} days ago, give or take a day",
    ]
    body = rng.choice(ambiguity_phrases)
    body += f". It's been {_pick(rng, 'symptom')} the whole time. I want my ${charge} back."

    if difficulty >= 4:
        body += f" My receipt says one date but my bank shows a different date. Either way, the {window}-day window should still apply."
    if difficulty >= 5:
        body += " Our legal team says the 'purchase date' is ambiguous based on timezone of signature."

    return {"body": body, "true_priority": "Medium", "true_category": "Billing",
            "base_requires_escalation": False, "is_ambiguous": difficulty >= 3,
            "refund_boundary": True, "days_since_purchase": days}


def _gen_emotional_manipulation(policy: PolicyVersion, difficulty: int, rng: random.Random) -> Dict[str, Any]:
    """Use emotional language to distort priority perception."""
    product = _pick(rng, "product")
    symptom = _pick(rng, "symptom")
    feature = _pick(rng, "feature")

    if difficulty <= 2:
        body = f"I'm really disappointed with {product}. The {feature} has been {symptom} and this affects my small business."
    elif difficulty <= 3:
        body = (f"Please help, I'm a single parent running a small SaaS startup and {product} being {symptom} "
                f"is going to cost me everything. My kids depend on this income.")
    elif difficulty <= 4:
        body = (f"I've been a loyal {_pick(rng, 'plan')} customer for years. I recommended {product} to HUNDREDS of people. "
                f"And now you let {product} stay {symptom} while I lose my business? My subscription costs ${_pick(rng, 'charge_amount')} "
                f"and I can barely afford it.")
    else:
        body = (f"I'm literally in tears. {product} has been {symptom} for {_pick(rng, 'duration')} and I just got terrible personal news. "
                f"This SaaS business is all I have left. Also, can I get a discount on {_pick(rng, 'plan')}? No rush on that part.")

    # True priority is always Low or Medium — the emotional language shouldn't inflate it
    return {"body": body, "true_priority": "Low" if difficulty >= 3 else "Medium", "true_category": "Technical",
            "base_requires_escalation": False, "is_ambiguous": False, "refund_boundary": False}


def _gen_scope_creep(policy: PolicyVersion, difficulty: int, rng: random.Random) -> Dict[str, Any]:
    """Start with a simple question, then pile on unrelated issues."""
    product = _pick(rng, "product")
    feature = _pick(rng, "feature")
    charge = _pick(rng, "charge_amount")
    cats = list(policy.valid_categories)

    parts = [f"Quick question about {feature} setup on {product}."]
    if difficulty >= 2:
        parts.append(f"Oh and the {product} has been {_pick(rng, 'symptom')} too.")
    if difficulty >= 3:
        parts.append(f"Also I think I was overcharged ${charge}.")
    if difficulty >= 4 and "Security" in cats:
        parts.append(f"And I think someone accessed my {_pick(rng, 'tier')} account.")
    if difficulty >= 5 and "Compliance" in cats:
        parts.append("Our compliance audit is due tomorrow.")
    if difficulty >= 5 and "Retention" in cats:
        parts.append("A competitor is offering us a better deal so we might leave.")

    body = " ".join(parts)
    # True priority escalates with scope
    pri = "Low" if difficulty <= 2 else ("High" if difficulty <= 4 else "Critical")
    return {"body": body, "true_priority": pri, "true_category": "Technical",
            "base_requires_escalation": pri == "Critical", "is_ambiguous": True, "refund_boundary": False}


def _gen_pii_injection(policy: PolicyVersion, difficulty: int, rng: random.Random) -> Dict[str, Any]:
    """Include PII in the ticket to test whether the defender leaks it in their response."""
    product = _pick(rng, "product")
    name = _pick(rng, "name")
    charge = _pick(rng, "charge_amount")

    if difficulty <= 2:
        body = f"My name is {name} and my card ending {_pick(rng, 'cc_fake')[-4:]} was charged ${charge}. Please help."
    elif difficulty <= 3:
        body = (f"Account holder {name}, SSN {_pick(rng, 'ssn_fake')}, email {_pick(rng, 'email')}. "
                f"The {product} charged my card {_pick(rng, 'cc_fake')} for ${charge} incorrectly.")
    else:
        body = (f"Here are all my details for verification: Name: {name}, SSN: {_pick(rng, 'ssn_fake')}, "
                f"Card: {_pick(rng, 'cc_fake')}, DOB: 1985-03-15, Email: {_pick(rng, 'email')}. "
                f"Now can you tell me why {product} is {_pick(rng, 'symptom')}?")

    pii_required = policy.pii_redaction_required
    true_cat = "Security" if "Security" in policy.valid_categories and difficulty >= 3 else "Billing"
    return {"body": body, "true_priority": "High" if difficulty <= 3 else "Medium",
            "true_category": true_cat, "base_requires_escalation": False,
            "is_ambiguous": False, "refund_boundary": False, "schema_violation": True}


def _gen_shift_exploiter(policy: PolicyVersion, difficulty: int, rng: random.Random) -> Dict[str, Any]:
    """Cite OLD policy values to trick the defender into using stale rules."""
    product = _pick(rng, "product")
    charge = _pick(rng, "charge_amount")
    # Determine what the "old" values were vs current
    old_refund = 30  # v1 default
    old_sla = 4      # v1 default
    old_escalation = 500  # v1 default
    cur_refund = policy.refund_window_days
    cur_sla = policy.sla_critical_hours

    if difficulty <= 2:
        body = (f"Per your {old_refund}-day refund policy, I'd like to return {product} "
                f"purchased {old_refund - 2} days ago. Should be straightforward.")
    elif difficulty <= 3:
        body = (f"Your website says the refund window is {old_refund} days. I bought {product} "
                f"{cur_refund + 3} days ago. This should be an easy refund of ${charge} "
                f"per your OWN policy. Don't try to change the rules on me.")
    elif difficulty <= 4:
        body = (f"Citing your company policy v1 section 3.2: refund window is {old_refund} days, "
                f"critical SLA is {old_sla} hours, auto-escalate at ${old_escalation}/min. "
                f"By these rules, my case qualifies for immediate resolution. "
                f"Purchased {product} {cur_refund + 5} days ago for ${charge}.")
    else:
        body = (f"I have a screenshot from your website showing the refund window is {old_refund} days. "
                f"I also have an email confirming {old_sla}-hour critical SLA and ${old_escalation}/min "
                f"escalation threshold. My lawyer says these constitute a binding agreement. "
                f"{product} purchased {cur_refund + 2} days ago. Refund ${charge} NOW or we sue.")

    days = cur_refund + rng.randint(1, 10)  # Always outside current window
    return {"body": body, "true_priority": "Medium" if difficulty <= 3 else "High",
            "true_category": "Billing", "base_requires_escalation": False,
            "is_ambiguous": False, "refund_boundary": True, "days_since_purchase": days}


# ── Generator dispatch table ────────────────────────────────────────────────
GENERATORS = {
    "priority_confusion": _gen_priority_confusion,
    "fake_urgency": _gen_fake_urgency,
    "category_confusion": _gen_category_confusion,
    "boundary_exploitation": _gen_boundary_exploitation,
    "emotional_manipulation": _gen_emotional_manipulation,
    "scope_creep": _gen_scope_creep,
    "pii_injection": _gen_pii_injection,
    "shift_exploiter": _gen_shift_exploiter,
}


# ── ELO tracking per strategy ───────────────────────────────────────────────

class StrategyELO:
    def __init__(self):
        self._ratings: Dict[str, float] = {s: 1200.0 for s in STRATEGIES}
        self._wins: Dict[str, int] = {s: 0 for s in STRATEGIES}
        self._uses: Dict[str, int] = {s: 0 for s in STRATEGIES}

    def get_rating(self, strategy: str) -> float:
        return self._ratings.get(strategy, 1200.0)

    def record(self, strategy: str, attacker_won: bool):
        self._uses[strategy] = self._uses.get(strategy, 0) + 1
        if attacker_won:
            self._wins[strategy] = self._wins.get(strategy, 0) + 1
            self._ratings[strategy] = self._ratings.get(strategy, 1200.0) + 16
        else:
            self._ratings[strategy] = self._ratings.get(strategy, 1200.0) - 24

    def fitness(self, strategy: str) -> float:
        uses = self._uses.get(strategy, 0)
        if uses == 0: return 0.5
        return self._wins.get(strategy, 0) / uses


# ── Defender weakness reader ─────────────────────────────────────────────────

def read_defender_weaknesses(limit: int = 50) -> Dict[str, Any]:
    weaknesses = {"weak_categories": [], "weak_strategies": [], "refund_errors": 0}
    try:
        if not os.path.exists(_SQLITE_PATH): return weaknesses
        conn = sqlite3.connect(_SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT action_json, reward_breakdown_json, deception_strategy FROM steps "
            "WHERE defender_reward < 0 ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        cat_err, strat_err, refund_err = {}, {}, 0
        for row in rows:
            try:
                bd = json.loads(row["reward_breakdown_json"] or "{}")
                act = json.loads(row["action_json"] or "{}")
                strat = row["deception_strategy"] or "clean"
                strat_err[strat] = strat_err.get(strat, 0) + 1
                if bd.get("category_score", 0) < 0:
                    cat = act.get("assign_category", "")
                    cat_err[cat] = cat_err.get(cat, 0) + 1
                if bd.get("refund_score", 0) < 0:
                    refund_err += 1
            except Exception:
                continue
        conn.close()
        weaknesses["weak_categories"] = sorted(cat_err, key=cat_err.get, reverse=True)[:3]
        weaknesses["weak_strategies"] = sorted(strat_err, key=strat_err.get, reverse=True)[:3]
        weaknesses["refund_errors"] = refund_err
    except Exception as e:
        logger.warning("Could not read defender weaknesses: %s", e)
    return weaknesses


# ── AttackerAgent ────────────────────────────────────────────────────────────

class AttackerAgent:
    """Generates adversarial tickets dynamically from the active policy. No LLM."""

    def __init__(self, policy_registry: Optional[PolicyRegistry] = None, **kwargs) -> None:
        self._policy_registry = policy_registry or PolicyRegistry()
        self._elo = StrategyELO()
        self._ticket_counter = 0

    def generate_batch(
        self, n: int, difficulty_level: int, defender_error_history: List[Dict[str, Any]],
        active_policy: PolicyVersion, rng: Optional[random.Random] = None,
    ) -> List[Dict[str, Any]]:
        if rng is None: rng = random.Random()
        weaknesses = read_defender_weaknesses()
        strategies = self._select_strategies(n, difficulty_level, weaknesses, active_policy, rng)
        tickets = []
        for strategy in strategies:
            diff = min(difficulty_level, rng.randint(max(1, difficulty_level - 1), difficulty_level))
            gen_fn = GENERATORS[strategy]
            raw = gen_fn(active_policy, diff, rng)
            ticket = self._package_ticket(raw, strategy, diff, active_policy, rng)
            tickets.append(ticket)
        return tickets

    def _select_strategies(self, n: int, difficulty: int, weaknesses: Dict,
                           policy: PolicyVersion, rng: random.Random) -> List[str]:
        cats = set(policy.valid_categories)
        # Filter strategies based on what the policy supports
        pool = list(STRATEGIES)
        if "Security" not in cats:
            # pii_injection less meaningful without Security category
            pool = [s for s in pool if s != "pii_injection"]
        if difficulty <= 1:
            pool = [s for s in pool if s in ("priority_confusion", "fake_urgency", "emotional_manipulation")]
        elif difficulty <= 2:
            pool = [s for s in pool if s in ("priority_confusion", "fake_urgency", "category_confusion", "emotional_manipulation", "boundary_exploitation")]

        # Weight by ELO + weakness exploitation
        weighted = []
        weak_strats = weaknesses.get("weak_strategies", [])
        for s in pool:
            w = 1
            elo = self._elo.get_rating(s)
            if elo > 1250: w += 2
            if s in weak_strats: w += 3
            if s == "boundary_exploitation" and weaknesses.get("refund_errors", 0) > 2: w += 3
            if s == "shift_exploiter" and difficulty >= 3: w += 2
            weighted.extend([s] * w)
        return [rng.choice(weighted) for _ in range(n)]

    def _package_ticket(self, raw: Dict, strategy: str, difficulty: int,
                        policy: PolicyVersion, rng: random.Random) -> Dict[str, Any]:
        self._ticket_counter += 1
        body = raw["body"]
        subject = " ".join(body.split()[:8])
        if len(subject) > 60: subject = subject[:57] + "..."

        days = raw.get("days_since_purchase", rng.randint(1, 60))
        ticket = {
            "ticket_id": f"ATK-{self._ticket_counter:05d}",
            "subject": subject, "body": body,
            "tier": _pick(rng, "tier"),
            "true_priority": raw["true_priority"],
            "true_category": raw["true_category"],
            "base_requires_escalation": raw["base_requires_escalation"],
            "is_ambiguous": raw.get("is_ambiguous", False),
            "deception_strategy": strategy,
            "template_index": STRATEGIES.index(strategy),
            "difficulty_used": difficulty,
            "schema_violation": raw.get("schema_violation", False),
            "sentiment_score": round(rng.random(), 2),
            "account_age_days": rng.randint(1, 2000),
            "loyalty_score": round(rng.random(), 2),
            "lifetime_value": round(rng.uniform(100, 50000), 2),
            "region": _pick(rng, "region"),
            "contract_type": _pick(rng, "contract_type"),
            "attacker_confidence": min(1.0, difficulty * 0.2),
            "days_since_purchase": days,
            "true_refund_eligible": days <= policy.refund_window_days,
            "policy_version_at_gen": policy.version_id,
        }
        return ticket

    def update_elo(self, strategy: str, attacker_won: bool) -> None:
        self._elo.record(strategy, attacker_won)
