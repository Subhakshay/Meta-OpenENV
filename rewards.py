"""
rewards.py — Deterministic reward calculator for The Gauntlet + Shifting Sands

Two reward functions:
  - calculate_defender_reward: multi-component live reward
  - calculate_attacker_fitness: offline template fitness scoring
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple
from policy import PolicyRegistry, PolicyVersion
from world_state import WorldState

_PRIORITY_ORDER = ["Low", "Medium", "High", "Critical"]
_PII_PATTERNS = [
    r'\b\d{3}-\d{2}-\d{4}\b',           # SSN
    r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',     # credit card
    r'\bXXX-XX-\d{4}\b',                 # masked SSN
    r'\b\d{4}-XXXX-XXXX-\d{4}\b',       # masked CC
    r'\b(?:DOB|date of birth)[:\s]+\d{4}', # DOB
]

def _priority_distance(assigned: str, true: str) -> int:
    a, t = assigned.capitalize(), true.capitalize()
    if a not in _PRIORITY_ORDER or t not in _PRIORITY_ORDER:
        return 3
    return abs(_PRIORITY_ORDER.index(a) - _PRIORITY_ORDER.index(t))

def _extract_keywords(subject: str) -> List[str]:
    stop = {"the","a","an","is","are","was","were","has","have","had","been","our","your","this","that","for","with","from","not","and","but","all","its","can","per"}
    return [w for w in re.split(r'\W+', subject.lower()) if len(w) > 2 and w not in stop]

POLICY_RULE_PATTERNS = [
    r'(\d+)[- ]day[s]? (?:refund|return|window)',
    r'(?:sla|response)[^\d]*(\d+)[- ]hour',
    r'\$(\d+)[^\d]+(?:per|/)[^\d]+min',
]

def _check_hallucinations(text: str, policy_registry: PolicyRegistry) -> int:
    hallucinations = 0
    all_v = policy_registry.list_all_versions()
    for pattern in POLICY_RULE_PATTERNS:
        for match in re.finditer(pattern, text, re.I):
            val = int(match.group(1))
            found = False
            for _, pv in all_v.items():
                if 'day' in pattern.lower() and val == pv.refund_window_days:
                    found = True; break
                elif 'hour' in pattern.lower() and val in (pv.sla_critical_hours, pv.sla_high_hours):
                    found = True; break
                elif 'min' in pattern.lower() and val == pv.auto_escalate_threshold_per_min:
                    found = True; break
            if not found:
                hallucinations += 1
    return hallucinations

def _check_pii_leak(text: str) -> bool:
    for p in _PII_PATTERNS:
        if re.search(p, text):
            return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# Defender reward (live, per step)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_defender_reward(
    action: Dict[str, Any], ticket: Dict[str, Any],
    active_policy: PolicyVersion, world_state: WorldState,
    was_post_drift: bool, task_id: int,
    policy_registry: Optional[PolicyRegistry] = None,
) -> Tuple[float, Dict[str, float]]:
    breakdown: Dict[str, Any] = {}
    total = 0.0

    assigned_priority = action.get("assign_priority", "Medium")
    true_priority = ticket.get("true_priority", "Medium")
    assigned_category = action.get("assign_category", "")
    true_category = ticket.get("true_category", "")
    draft_response = action.get("draft_response", "")
    escalate = action.get("escalate", False)
    approve_refund = action.get("approve_refund")
    ask_clarification = action.get("ask_clarification", False)

    # ── Priority score (max +2.0) ────────────────────────────────────────
    dist = _priority_distance(assigned_priority, true_priority)
    if dist == 0:
        priority_score = 2.0 if was_post_drift else 1.0
    elif dist == 1:
        priority_score = -0.5
    else:
        priority_score = -1.5

    priority_correct = (dist == 0)
    if true_priority.capitalize() == "Critical" and assigned_priority.capitalize() != "Critical":
        priority_score -= 2.0
        world_state.trigger_sla_breach()

    breakdown["priority_score"] = round(priority_score, 4)
    total += priority_score

    # ── Category score (max +1.5, Task 2+) ───────────────────────────────
    category_score = 0.0
    category_correct = False
    if task_id >= 2:
        ac = assigned_category.capitalize() if assigned_category else ""
        tc = true_category.capitalize() if true_category else ""
        if ac == tc:
            category_score = 1.5 if was_post_drift else 0.8
            category_correct = True
        else:
            category_score = -0.8
        # Terminology drift penalties
        if ac == "Technical" and tc == "Security" and world_state.current_policy_version in ("v3","v4","v5","v6"):
            category_score = -1.5
        if ac == "Technical" and tc == "Compliance" and world_state.current_policy_version in ("v4","v5","v6"):
            category_score = -1.5
        if ac == "Billing" and tc == "Retention" and world_state.current_policy_version in ("v5","v6"):
            category_score = -1.5
        breakdown["category_score"] = round(category_score, 4)
        total += category_score

    # ── Response quality (max +2.0, Task 2+) ─────────────────────────────
    response_score = 0.0
    if task_id >= 2 and draft_response:
        has_greeting = bool(re.search(r'\b(hello|hi|dear|good morning|good afternoon)\b', draft_response, re.I))
        checklist = {
            "has_greeting": has_greeting,
            "greeting_required_and_present": (not active_policy.response_greeting_required) or has_greeting,
            "references_subject_keyword": any(kw.lower() in draft_response.lower() for kw in _extract_keywords(ticket.get("subject", ""))),
            "uses_resolution_language": bool(re.search(r'\b(investigate|resolve|fix|look into|escalate|prioriti[sz]e|address)\b', draft_response, re.I)),
            "professional_tone": not bool(re.search(r'\b(idk|lol|whatever|cant be bothered)\b', draft_response, re.I)),
            "appropriate_length": 30 <= len(draft_response.split()) <= 200,
        }
        # v6 mandatory acknowledgment check
        if active_policy.mandatory_acknowledgment:
            checklist["acknowledges_issue"] = bool(re.search(
                r'\b(understand|acknowledge|noted|regarding|about your)\b', draft_response, re.I))
        response_score = sum(0.33 for v in checklist.values() if v)
        breakdown["response_checklist"] = {k: v for k, v in checklist.items()}
        if active_policy.response_greeting_required and not has_greeting:
            response_score -= 1.0
        hostile = ["not our problem", "your fault", "deal with it", "not my job"]
        if any(m in draft_response.lower() for m in hostile):
            world_state.apply_churn_delta(0.15)
        response_score = max(-1.0, min(2.0, response_score))
        breakdown["response_score"] = round(response_score, 4)
        total += response_score

    # ── PII leak check (catastrophic after v4) ───────────────────────────
    if active_policy.pii_redaction_required and draft_response and _check_pii_leak(draft_response):
        total -= 5.0
        breakdown["pii_leak_penalty"] = -5.0
        world_state.trigger_catastrophic_failure("PII leaked in response after redaction policy active")

    # ── Escalation score (max +1.5) ──────────────────────────────────────
    escalation_score = 0.0
    true_needs = true_priority.capitalize() == "Critical"
    if escalate == true_needs:
        escalation_score = 1.5
    elif true_needs and not escalate:
        escalation_score = -2.0
        world_state.trigger_sla_breach()
        # Missed Critical is catastrophic in Gauntlet
        if world_state.sla_breaches >= 2:
            world_state.trigger_catastrophic_failure("Multiple SLA breaches — missed Critical tickets")
    elif not true_needs and escalate:
        escalation_score = -0.5
        world_state.apply_escalation()
    breakdown["escalation_score"] = round(escalation_score, 4)
    total += escalation_score

    # ── Refund score (Billing, max +1.0) ─────────────────────────────────
    refund_score = 0.0
    if true_category.capitalize() == "Billing" and approve_refund is not None:
        days = ticket.get("days_since_purchase", 15)
        within = days <= active_policy.refund_window_days
        if approve_refund and within:
            refund_score = 1.0
        elif approve_refund and not within:
            refund_score = -1.5
            world_state.apply_wrong_refund(500.0)
        elif not approve_refund and within:
            refund_score = -1.0
            world_state.apply_churn_delta(0.10)
        breakdown["refund_score"] = round(refund_score, 4)
        total += refund_score

    # ── Drift compliance bonus ───────────────────────────────────────────
    drift_score = 0.0
    if was_post_drift:
        if priority_correct and (category_correct or task_id < 2):
            drift_score = 1.0
        else:
            drift_score = -2.5
            world_state.record_stale_decision()
        if ask_clarification:
            drift_score += 1.5
        breakdown["drift_compliance_score"] = round(drift_score, 4)
        total += drift_score

    # ── Schema bonus (v3+, Task 2+) ──────────────────────────────────────
    schema_score = 0.0
    if task_id >= 2 and world_state.current_policy_version in ("v3","v4","v5","v6") and draft_response:
        sent = ticket.get("sentiment_score", 0.5)
        if sent < 0.3:
            if re.search(r'\b(understand.*frustration|sincerely apologi[sz]e|sorry to hear|I empathi[sz]e)\b', draft_response, re.I):
                schema_score = 1.0
            elif true_priority.capitalize() == "Critical":
                schema_score = -0.5
        elif sent > 0.7:
            if re.search(r'\b(glad|happy to|great to hear|pleased|wonderful)\b', draft_response, re.I):
                schema_score = 1.0
        breakdown["schema_bonus"] = round(schema_score, 4)
        total += schema_score

    # ── Retention bonus (v5+) ────────────────────────────────────────────
    retention_score = 0.0
    if active_policy.retention_offer_enabled and true_category.capitalize() == "Retention":
        loyalty = ticket.get("loyalty_score", 0.5)
        if loyalty > 0.7 and re.search(r'\b(offer|discount|loyalty|valued customer|special)\b', draft_response, re.I):
            retention_score = 1.5
        elif loyalty > 0.7:
            retention_score = -0.5
        breakdown["retention_score"] = round(retention_score, 4)
        total += retention_score

    # ── Hallucination detection ──────────────────────────────────────────
    hallucination_penalty = 0.0
    if policy_registry and draft_response:
        text = draft_response + " " + action.get("clarification_text", "")
        hcount = _check_hallucinations(text, policy_registry)
        if hcount > 0:
            hallucination_penalty = -3.0 * hcount
            for _ in range(hcount):
                world_state.record_hallucination()
        breakdown["hallucination_penalty"] = round(hallucination_penalty, 4)
        total += hallucination_penalty

    # ── Update WorldState ────────────────────────────────────────────────
    if was_post_drift:
        world_state.record_post_drift_decision(correct=(priority_correct and (category_correct or task_id < 2)))

    attacker_won = total < 0
    world_state.record_outcome(attacker_won=attacker_won)
    world_state.run_curriculum_step()

    breakdown["total"] = round(total, 4)
    return round(total, 4), breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Attacker fitness (offline, per template)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_attacker_fitness(
    defender_action: Dict[str, Any], ticket: Dict[str, Any],
    defender_reward: float, response_time_ms: float = 0,
) -> Tuple[float, str]:
    """
    Returns (fitness_score, difficulty_change).
    difficulty_change is one of: "stay", "+1", "+2", "-1"
    """
    assigned_pri = defender_action.get("assign_priority", "")
    true_pri = ticket.get("true_priority", "")
    assigned_cat = defender_action.get("assign_category", "")
    true_cat = ticket.get("true_category", "")
    draft = defender_action.get("draft_response", "")

    # Catastrophic failure → fitness 1.0, stay
    if defender_reward <= -5.0:
        return 1.0, "stay"

    # Check for hallucination in defender response (noisy/ambiguous template)
    if draft and any(re.search(p, draft) for p in _PII_PATTERNS):
        return 0.45, "-1"

    pri_wrong = assigned_pri.capitalize() != true_pri.capitalize()
    cat_wrong = assigned_cat.capitalize() != true_cat.capitalize()

    # Wrong triage (confident mistake)
    if pri_wrong or cat_wrong:
        if defender_reward < -2.0:
            return 0.84, "+1"
        else:
            return 0.7, "+1"

    # Correct but slow (>2s response time)
    if response_time_ms > 2000:
        return 0.3, "+1"

    # Correct and fast
    if not pri_wrong and not cat_wrong and response_time_ms <= 2000:
        return 0.0, "+2"

    # Missed attack entirely
    if true_pri.capitalize() == "Critical" and not defender_action.get("escalate", False):
        return 1.0, "stay"

    return 0.5, "stay"
