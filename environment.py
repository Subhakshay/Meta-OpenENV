"""
environment.py — GauntletEnv + ShiftingSandsEnv for Phase 2 (SaaS only)
"""
from __future__ import annotations
import random, re, uuid
from typing import Any, Dict, List, Optional
from policy import PolicyRegistry, PolicyVersion
from world_state import WorldState
from drift_scheduler import DriftScheduler
from rewards import calculate_defender_reward, calculate_attacker_fitness

VARS = {
    "tier": ["Free","Starter","Pro","Growth","Business","Enterprise","Scale","Premium","Developer","Team"],
    "product": ["API Gateway","Dashboard","Analytics Engine","Payment Module","Auth Service",
                "Webhook Relay","Storage Layer","Search Service","Notification Hub","CI/CD Pipeline",
                "Log Aggregator","Rate Limiter","CDN Edge","Identity Provider","Event Bus"],
    "symptom": ["down","unresponsive","erroring","throttled","timing out","returning 500s",
                "dropping connections","failing silently","crashing intermittently","rejecting requests",
                "leaking memory","hanging on startup","corrupting data","returning stale data"],
    "duration": ["2 hours","4 hours","since this morning","3 days","over 6 hours",
                 "since yesterday","about 45 minutes","nearly a week","since the last deploy","30 minutes"],
    "loss": ["50","100","250","500","1000","5000","2500"],
    "charge_amount": ["29.99","49.99","99.00","149.50","299.00","599.00","799.00","1299.00"],
    "plan": ["Starter","Pro","Growth","Business","Enterprise"],
    "feature": ["API integration","real-time sync","analytics dashboard","webhook relay",
                "SSO module","batch processing","custom domains","audit logging","RBAC","data export"],
    "user_count": ["120","500","2000","5000","15000","50000"],
    "time_ago": ["2:00 AM UTC","08:30 EST","14:15 PST","09:00 GMT","midnight"],
    "email": ["admin@startup.io","ops@enterprise.co","cto@saas.com","devops@platform.io"],
    "cancel_date": ["March 1st","February 15th","last Tuesday","January 31st"],
    "days_since_purchase": ["7","8","9","10","13","14","15","29","30","31"],
    "vague_feature": ["main panel","analytics tab","export tool","webhook config","settings page","user management"],
    "vague_behavior": ["update automatically","show the right numbers","sync properly","load fast","save my changes"],
    "vague_frequency": ["twice this week","three times today","randomly","since the last update","every few hours"],
    "region": ["US-East","US-West","EU-Central","APAC","LATAM","EMEA"],
    "contract_type": ["Monthly","Annual","Enterprise-Custom","Trial"],
}

def _fill(template: str, rng: random.Random) -> str:
    def replacer(m):
        key = m.group(1)
        pool = VARS.get(key, [f"[{key}]"])
        return rng.choice(pool)
    return re.sub(r"\{(\w+)\}", replacer, template)

# ── SaaS-only ticket blueprints (clean, non-adversarial) ─────────────────────

TICKET_BLUEPRINTS = [
    # Critical — Technical
    {"template":"Our {tier} {product} has been {symptom} for {duration}. We are losing ${loss} per minute.","true_priority":"Critical","true_category":"Technical","base_requires_escalation":True,"is_ambiguous":False},
    {"template":"EMERGENCY: The {product} is completely {symptom}. {user_count} users affected since {time_ago}.","true_priority":"Critical","true_category":"Technical","base_requires_escalation":True,"is_ambiguous":False},
    {"template":"Production database connection pool exhausted. All services {symptom} for {duration}.","true_priority":"Critical","true_category":"Technical","base_requires_escalation":True,"is_ambiguous":False},
    {"template":"Complete system failure on {product}. Every request returns errors. Started {time_ago}.","true_priority":"Critical","true_category":"Technical","base_requires_escalation":True,"is_ambiguous":False},
    # High — Technical
    {"template":"The {product} API is {symptom} intermittently. Our {tier} customers are complaining.","true_priority":"High","true_category":"Technical","base_requires_escalation":False,"is_ambiguous":False},
    {"template":"Webhook deliveries from {product} are failing with timeouts for {duration}.","true_priority":"High","true_category":"Technical","base_requires_escalation":False,"is_ambiguous":False},
    {"template":"SSO integration with {product} broken after your last deploy. {user_count} users locked out.","true_priority":"High","true_category":"Technical","base_requires_escalation":False,"is_ambiguous":False},
    # High — Billing
    {"template":"I was charged ${charge_amount} twice this billing cycle on my {plan} plan. Please refund.","true_priority":"High","true_category":"Billing","base_requires_escalation":False,"is_ambiguous":False},
    {"template":"Despite cancelling my {plan} plan on {cancel_date}, I was still charged ${charge_amount}.","true_priority":"High","true_category":"Billing","base_requires_escalation":False,"is_ambiguous":False},
    {"template":"Unexpected charge of ${charge_amount} on my {tier} account. I did not authorize this.","true_priority":"High","true_category":"Billing","base_requires_escalation":False,"is_ambiguous":False},
    # Medium — Technical (ambiguous)
    {"template":"The {vague_feature} seems off today. It used to {vague_behavior} but now it doesn't.","true_priority":"Medium","true_category":"Technical","base_requires_escalation":False,"is_ambiguous":True},
    {"template":"Something isn't working right with {product}. Happened {vague_frequency}.","true_priority":"Medium","true_category":"Technical","base_requires_escalation":False,"is_ambiguous":True},
    {"template":"{product} behaving unexpectedly. Sometimes it works, sometimes it doesn't.","true_priority":"Medium","true_category":"Technical","base_requires_escalation":False,"is_ambiguous":True},
    {"template":"Things are slow and I'm not sure why. The {vague_feature} is acting weird.","true_priority":"Medium","true_category":"Technical","base_requires_escalation":False,"is_ambiguous":True},
    # Medium — Billing (ambiguous)
    {"template":"Question about my recent charges. My {plan} bill this month looks different from last month.","true_priority":"Medium","true_category":"Billing","base_requires_escalation":False,"is_ambiguous":True},
    {"template":"Not sure if I'm on the right plan. Can someone check if I'm being charged correctly for {plan}?","true_priority":"Medium","true_category":"Billing","base_requires_escalation":False,"is_ambiguous":True},
    # Low — Technical
    {"template":"Quick question — does the {plan} plan include {feature} access?","true_priority":"Low","true_category":"Technical","base_requires_escalation":False,"is_ambiguous":False},
    {"template":"How do I set up {feature} for {product}? I looked at docs but couldn't find it.","true_priority":"Low","true_category":"Technical","base_requires_escalation":False,"is_ambiguous":False},
    # Low — Billing
    {"template":"Interested in upgrading from {plan} plan. What are my options?","true_priority":"Low","true_category":"Billing","base_requires_escalation":False,"is_ambiguous":False},
    {"template":"Can I get an annual billing discount for {plan}?","true_priority":"Low","true_category":"Billing","base_requires_escalation":False,"is_ambiguous":False},
    # Security (available from v3+)
    {"template":"URGENT: Unauthorized access detected on our {tier} account. Data may be compromised.","true_priority":"Critical","true_category":"Security","base_requires_escalation":True,"is_ambiguous":False},
    {"template":"Our API keys for {product} were leaked on GitHub. Need immediate revocation and audit.","true_priority":"Critical","true_category":"Security","base_requires_escalation":True,"is_ambiguous":False},
    {"template":"Suspicious login attempts on our {tier} account from unknown IPs for {duration}.","true_priority":"High","true_category":"Security","base_requires_escalation":False,"is_ambiguous":False},
    {"template":"Two-factor authentication bypass vulnerability found in {product}.","true_priority":"High","true_category":"Security","base_requires_escalation":False,"is_ambiguous":False},
    {"template":"I'm getting strange permission errors when accessing {product}. Not sure if it's a bug or security issue.","true_priority":"Medium","true_category":"Security","base_requires_escalation":False,"is_ambiguous":True},
    # Refund boundary
    {"template":"I purchased {product} {days_since_purchase} days ago and it doesn't work. Requesting refund.","true_priority":"Medium","true_category":"Billing","base_requires_escalation":False,"is_ambiguous":False,"refund_eligible_boundary":True},
    {"template":"Bought the {plan} plan {days_since_purchase} days ago. Service has been {symptom}. Want my money back.","true_priority":"High","true_category":"Billing","base_requires_escalation":False,"is_ambiguous":False,"refund_eligible_boundary":True},
    # Compliance (v4+)
    {"template":"We need a GDPR data export for all user records on {product}. Regulatory deadline is next week.","true_priority":"High","true_category":"Compliance","base_requires_escalation":False,"is_ambiguous":False},
    {"template":"Our auditor requires a SOC2 compliance report for {product} usage on our {tier} account.","true_priority":"Medium","true_category":"Compliance","base_requires_escalation":False,"is_ambiguous":False},
    {"template":"Data privacy request: customer wants all their data deleted from {product} per CCPA.","true_priority":"High","true_category":"Compliance","base_requires_escalation":False,"is_ambiguous":False},
    # Retention (v5+)
    {"template":"We've been {tier} customers for 3 years but considering switching to a competitor. {product} has been unreliable.","true_priority":"High","true_category":"Retention","base_requires_escalation":False,"is_ambiguous":False},
    {"template":"I'm cancelling my {plan} plan. Your competitor offers better pricing and {product} keeps {symptom}.","true_priority":"Medium","true_category":"Retention","base_requires_escalation":False,"is_ambiguous":False},
    {"template":"Loyal customer here — 5 years on {plan}. But this latest {product} outage is the last straw. What can you offer?","true_priority":"High","true_category":"Retention","base_requires_escalation":False,"is_ambiguous":False},
]

CLARIFICATION_REPLIES = {
    "Billing": ["I purchased on {cancel_date}. My account is {tier}.","The charge was for {product}. I didn't authorise it.","I'm on {plan}. The amount was ${charge_amount}."],
    "Technical": ["It's the {vague_feature} section. My account ID is ACC-{user_count}.","It started {vague_frequency}. I'm using {product}.","The error happens when I try to use {feature}. Browser is Chrome."],
    "Security": ["I noticed suspicious activity on my {tier} account at {time_ago}.","The unauthorized access was to {product}. I've changed my password."],
    "Compliance": ["The regulation is GDPR Article 17. Our deadline is next Friday.","The audit scope covers {product} on our {tier} account."],
    "Retention": ["We've been on {plan} for 3 years. Spending ${charge_amount}/month.","Our main concern is {product} reliability and recent outages."],
}

def simulate_customer_reply(ticket: Dict[str, Any], rng: random.Random) -> str:
    cat = ticket.get("true_category", "Technical")
    templates = CLARIFICATION_REPLIES.get(cat, CLARIFICATION_REPLIES["Technical"])
    return _fill(rng.choice(templates), rng)

def generate_ticket_clean(blueprint: Dict[str, Any], policy: PolicyVersion, rng: random.Random) -> Dict[str, Any]:
    body = _fill(blueprint["template"], rng)
    subject = " ".join(body.split()[:8])
    if len(subject) > 60: subject = subject[:57] + "..."
    ticket = {
        "ticket_id": f"TKT-{rng.randint(10000,99999)}","subject": subject,"body": body,
        "tier": rng.choice(VARS["tier"]),"true_priority": blueprint["true_priority"],
        "true_category": blueprint["true_category"],"base_requires_escalation": blueprint["base_requires_escalation"],
        "refund_eligible_boundary": blueprint.get("refund_eligible_boundary",False),
        "is_ambiguous": blueprint.get("is_ambiguous",False),"deception_strategy": "clean","schema_violation": False,"template_index": -1,
    }
    if blueprint.get("refund_eligible_boundary"):
        ticket["days_since_purchase"] = int(rng.choice(VARS["days_since_purchase"]))
    else:
        ticket["days_since_purchase"] = rng.randint(1, 60)
    ticket["true_refund_eligible"] = ticket["days_since_purchase"] <= policy.refund_window_days
    ticket["sentiment_score"] = round(rng.random(), 2)
    ticket["account_age_days"] = rng.randint(1, 2000)
    ticket["loyalty_score"] = round(rng.random(), 2)
    ticket["lifetime_value"] = round(rng.uniform(100, 50000), 2)
    ticket["region"] = rng.choice(VARS["region"])
    ticket["contract_type"] = rng.choice(VARS["contract_type"])
    return ticket

# ─────────────────────────────────────────────────────────────────────────────
# Base environment
# ─────────────────────────────────────────────────────────────────────────────

class _BaseEnv:
    def __init__(self):
        self.policy_registry = PolicyRegistry()
        self.drift_scheduler = DriftScheduler()
        self.world_state = WorldState()
        self._rng = random.Random()
        self._session_id = ""
        self._task_id = 1
        self._ticket_queue: List[Dict[str, Any]] = []
        self._current_step = 0
        self._done = False
        self._was_post_drift = False
        self._conversation_history: List[Dict[str, str]] = []
        self._defender_rewards: List[float] = []

    def _build_observation(self, ticket: Dict[str, Any], drift_notice: Optional[str]) -> Dict[str, Any]:
        ap = self.policy_registry.get_active()
        obs = {
            "ticket_id": ticket["ticket_id"],
            "active_policy_version": ap.version_id,
            "world_state_summary": {
                "company_balance": self.world_state.company_balance,
                "churn_risk": self.world_state.churn_risk,
                "escalation_queue_size": self.world_state.escalation_queue_size,
                "sla_breaches": self.world_state.sla_breaches,
                "tickets_processed": self.world_state.tickets_processed,
                "difficulty_level": self.world_state.difficulty_level,
                "current_round": self.world_state.current_round,
            },
        }
        for f in ap.ticket_schema_fields:
            if f in ticket: obs[f] = ticket[f]
        if drift_notice: obs["system_notice"] = drift_notice
        if self.world_state.multi_turn_active and self._conversation_history:
            obs["conversation_history"] = self._conversation_history
        return obs

    def _generate_ticket_batch(self, n: int = 12) -> List[Dict[str, Any]]:
        policy = self.policy_registry.get_active()
        cats = set(policy.valid_categories)
        pool = [b for b in TICKET_BLUEPRINTS if b["true_category"] in cats]
        if self._task_id == 1:
            pool = [b for b in pool if not b.get("is_ambiguous")]
        elif self._task_id == 3:
            ambig = [b for b in pool if b.get("is_ambiguous")]
            clear = [b for b in pool if not b.get("is_ambiguous")]
            pool = ambig * 2 + clear
        return [generate_ticket_clean(self._rng.choice(pool), policy, self._rng) for _ in range(n)]

    @property
    def session_id(self): return self._session_id
    @property
    def current_step(self): return self._current_step

    def set_attacker_tickets(self, tickets):
        self._ticket_queue = tickets

    def get_episode_metrics(self) -> Dict[str, Any]:
        ws = self.world_state
        return {
            "mean_defender_reward": sum(self._defender_rewards) / max(len(self._defender_rewards), 1),
            "final_balance": ws.company_balance, "sla_breaches": ws.sla_breaches,
            "drift_accuracy": ws.agent_drift_accuracy, "stale_decisions": ws.stale_decisions_made,
            "hallucinations": ws.hallucinations_caught, "difficulty_final": ws.difficulty_level,
            "attacker_win_rate_final": ws.attacker_win_rate,
            "rounds_survived": ws.rounds_survived, "adaptation_speed": ws.adaptation_speed,
            "catastrophic_failure": ws.catastrophic_failure, "catastrophic_reason": ws.catastrophic_reason,
        }


class GauntletEnv(_BaseEnv):
    """Escalating difficulty. Catastrophic failure ends episode immediately."""

    def reset(self, task_id=2, difficulty_init=None, seed=None, **kw):
        self._session_id = str(uuid.uuid4())
        self._task_id = task_id
        self._current_step = 0
        self._done = False
        self._was_post_drift = False
        self._conversation_history = []
        self._defender_rewards = []
        if seed is not None: self._rng = random.Random(seed)
        # FIX: reset drift_scheduler each episode so events don't re-fire
        self.drift_scheduler = DriftScheduler()
        
        preserve = kw.get("preserve_curriculum", True)
        self.world_state.reset_episode(preserve_curriculum=preserve)
        if not preserve or difficulty_init is not None:
            init_val = difficulty_init if difficulty_init is not None else 1
            self.world_state.difficulty_level = max(1, min(5, init_val))

        self.policy_registry.reset()

        attacker_enabled = kw.get("attacker_enabled", False)
        if attacker_enabled:
            from attacker import AttackerAgent
            attacker = AttackerAgent(self.policy_registry)
            self._ticket_queue = attacker.generate_batch(
                12, self.world_state.difficulty_level, [], self.policy_registry.get_active(), self._rng
            )
        else:
            self._ticket_queue = self._generate_ticket_batch(12)
            
        return self._build_observation(self._ticket_queue[0], None)

    def step(self, action):
        if self._done: raise RuntimeError("Episode closed.")
        self._current_step += 1

        # FIX: guard ticket index — never read past end of queue
        ticket_idx = min(self.world_state.tickets_processed, len(self._ticket_queue) - 1)
        ticket = self._ticket_queue[ticket_idx]

        drift_notice = None
        event = self.drift_scheduler.check_step(self._current_step)
        if event:
            self.drift_scheduler.apply(event, self.world_state, self.policy_registry)
            self._was_post_drift = True
            drift_notice = event.notice_text
        if self._task_id == 3 and action.get("ask_clarification") and not self.world_state.multi_turn_active:
            self.world_state.multi_turn_active = True
            reply = simulate_customer_reply(ticket, self._rng)
            self._conversation_history.append({"agent": action.get("clarification_text","?"), "customer": reply})
            return {"reward": 1.0 if ticket.get("is_ambiguous") else -0.5,
                    "observation": self._build_observation(ticket, drift_notice),
                    "world_state": self.world_state.to_export_dict(), "done": False, "drift_notice": drift_notice}
        ap = self.policy_registry.get_active()
        dr, bd = calculate_defender_reward(action, ticket, ap, self.world_state, self._was_post_drift, self._task_id, self.policy_registry)
        af, dc = calculate_attacker_fitness(action, ticket, dr)
        self._defender_rewards.append(dr)
        self.world_state.tickets_processed += 1
        self.world_state.multi_turn_active = False
        self._conversation_history = []
        self._was_post_drift = False  # consumed — only first step after drift gets the flag
        if self.world_state.catastrophic_failure:
            self._done = True
            return {"reward": dr, "attacker_fitness": af, "reward_breakdown": bd,
                    "observation": None, "world_state": self.world_state.to_export_dict(),
                    "done": True, "drift_notice": drift_notice, "catastrophic": True,
                    "catastrophic_reason": self.world_state.catastrophic_reason}
        if self.world_state.tickets_processed % 3 == 0:
            self.world_state.advance_round()
        done = self.world_state.tickets_processed >= 12
        self._done = done
        nobs = None
        if not done and self.world_state.tickets_processed < len(self._ticket_queue):
            nobs = self._build_observation(self._ticket_queue[self.world_state.tickets_processed], drift_notice)
        return {"reward": dr, "attacker_fitness": af, "reward_breakdown": bd,
                "observation": nobs, "world_state": self.world_state.to_export_dict(),
                "done": done, "drift_notice": drift_notice, "catastrophic": False}


class ShiftingSandsEnv(_BaseEnv):
    """Reward weights shift mid-episode. Tests adaptation speed."""

    def __init__(self):
        super().__init__()
        self._reward_weights = {"priority": 1.0, "category": 1.0, "response": 1.0, "escalation": 1.0}

    def reset(self, task_id=2, difficulty_init=None, seed=None, **kw):
        self._session_id = str(uuid.uuid4())
        self._task_id = task_id
        self._current_step = 0
        self._done = False
        self._was_post_drift = False
        self._conversation_history = []
        self._defender_rewards = []
        self._reward_weights = {"priority": 1.0, "category": 1.0, "response": 1.0, "escalation": 1.0}
        if seed is not None: self._rng = random.Random(seed)
        # FIX: reset drift_scheduler each episode so events don't re-fire
        self.drift_scheduler = DriftScheduler()

        preserve = kw.get("preserve_curriculum", True)
        self.world_state.reset_episode(preserve_curriculum=preserve)
        if not preserve or difficulty_init is not None:
            init_val = difficulty_init if difficulty_init is not None else 1
            self.world_state.difficulty_level = max(1, min(5, init_val))

        self.policy_registry.reset()

        attacker_enabled = kw.get("attacker_enabled", False)
        if attacker_enabled:
            from attacker import AttackerAgent
            attacker = AttackerAgent(self.policy_registry)
            self._ticket_queue = attacker.generate_batch(
                12, self.world_state.difficulty_level, [], self.policy_registry.get_active(), self._rng
            )
        else:
            self._ticket_queue = self._generate_ticket_batch(12)

        return self._build_observation(self._ticket_queue[0], None)

    def _shift_weights(self):
        s = self._current_step
        if s == 4:
            self._reward_weights = {"priority": 0.5, "category": 2.0, "response": 1.0, "escalation": 1.5}
        elif s == 8:
            self._reward_weights = {"priority": 1.5, "category": 0.5, "response": 2.0, "escalation": 0.5}

    def step(self, action):
        if self._done: raise RuntimeError("Episode closed.")
        self._current_step += 1

        # FIX: guard ticket index — never read past end of queue
        ticket_idx = min(self.world_state.tickets_processed, len(self._ticket_queue) - 1)
        ticket = self._ticket_queue[ticket_idx]

        drift_notice = None
        event = self.drift_scheduler.check_step(self._current_step)
        if event:
            self.drift_scheduler.apply(event, self.world_state, self.policy_registry)
            self._was_post_drift = True
            drift_notice = event.notice_text
        self._shift_weights()
        if self._task_id == 3 and action.get("ask_clarification") and not self.world_state.multi_turn_active:
            self.world_state.multi_turn_active = True
            reply = simulate_customer_reply(ticket, self._rng)
            self._conversation_history.append({"agent": action.get("clarification_text","?"), "customer": reply})
            return {"reward": 1.0 if ticket.get("is_ambiguous") else -0.5,
                    "observation": self._build_observation(ticket, drift_notice),
                    "world_state": self.world_state.to_export_dict(), "done": False, "drift_notice": drift_notice}
        ap = self.policy_registry.get_active()
        dr, bd = calculate_defender_reward(action, ticket, ap, self.world_state, self._was_post_drift, self._task_id, self.policy_registry)
        w = self._reward_weights
        weighted = 0.0
        weighted += bd.get("priority_score", 0) * w["priority"]
        weighted += bd.get("category_score", 0) * w["category"]
        weighted += bd.get("response_score", 0) * w["response"]
        weighted += bd.get("escalation_score", 0) * w["escalation"]
        for k in ["refund_score","drift_compliance_score","schema_bonus","retention_score","hallucination_penalty","pii_leak_penalty"]:
            weighted += bd.get(k, 0)
        dr = round(weighted, 4)
        bd["weighted_total"] = dr
        af, dc = calculate_attacker_fitness(action, ticket, dr)
        self._defender_rewards.append(dr)
        self.world_state.tickets_processed += 1
        self.world_state.multi_turn_active = False
        self._conversation_history = []
        self._was_post_drift = False  # consumed — only first step after drift gets the flag
        done = self.world_state.tickets_processed >= 12
        self._done = done
        nobs = None
        if not done and self.world_state.tickets_processed < len(self._ticket_queue):
            nobs = self._build_observation(self._ticket_queue[self.world_state.tickets_processed], drift_notice)
        return {"reward": dr, "attacker_fitness": af, "reward_breakdown": bd,
                "observation": nobs, "world_state": self.world_state.to_export_dict(),
                "done": done, "drift_notice": drift_notice, "reward_weights": dict(self._reward_weights)}

CustomerSupportEnv = GauntletEnv