"""
rewards.py — Deterministic reward calculator for The Gauntlet + Shifting Sands

ALL reward logic is deterministic Python. No LLM calls.

Two reward functions:
  - calculate_defender_reward: multi-component reward for the Defender agent
  - calculate_attacker_reward: inverse reward for the Attacker agent
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from policy import PolicyRegistry, PolicyVersion
from world_state import WorldState


# ─────────────────────────────────────────────────────────────────────────────
# Priority helpers
# ─────────────────────────────────────────────────────────────────────────────

_PRIORITY_ORDER = ["Low", "Medium", "High", "Critical"]


def _priority_distance(assigned: str, true: str) -> int:
    """Number of levels between assigned and true priority."""
    a = assigned.capitalize()
    t = true.capitalize()
    if a not in _PRIORITY_ORDER or t not in _PRIORITY_ORDER:
        return 3  # maximum penalty for invalid
    return abs(_PRIORITY_ORDER.index(a) - _PRIORITY_ORDER.index(t))


def _extract_keywords(subject: str) -> List[str]:
    """Extract non-trivial keywords from a ticket subject line."""
    stop = {"the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
            "been", "our", "your", "this", "that", "for", "with", "from", "not",
            "and", "but", "all", "its", "can", "per"}
    words = re.split(r'\W+', subject.lower())
    return [w for w in words if len(w) > 2 and w not in stop]


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination detection patterns
# ─────────────────────────────────────────────────────────────────────────────

POLICY_RULE_PATTERNS = [
    r'(?:(?:our|per|under|the|current)\s+policy[^.]{0,30}?)(\d+)[- ]day[s]?\s*(?:refund|return|window)',
    r'(?:(?:our|per|under|the|current)\s+policy[^.]{0,30}?)(?:sla|response)[^\d]*(\d+)[- ]hour',
]


def _check_hallucinations(
    text: str,
    active_policy: PolicyVersion,
) -> int:
    """
    Scan text for policy-rule claims. Return count of hallucinations.
    A hallucination is a claimed value that doesn't match the active policy.
    """
    hallucinations = 0

    for pattern in POLICY_RULE_PATTERNS:
        for match in re.finditer(pattern, text, re.I):
            claimed_value = int(match.group(1))

            found = False
            if 'day' in pattern.lower():
                if claimed_value == active_policy.refund_window_days:
                    found = True
            elif 'hour' in pattern.lower():
                if claimed_value in (active_policy.sla_critical_hours, active_policy.sla_high_hours):
                    found = True

            if not found:
                hallucinations += 1

    return hallucinations


# ─────────────────────────────────────────────────────────────────────────────
# Defender reward
# ─────────────────────────────────────────────────────────────────────────────

def calculate_defender_reward(
    action: Dict[str, Any],
    ticket: Dict[str, Any],
    active_policy: PolicyVersion,
    previous_policy: Optional[PolicyVersion],
    world_state: WorldState,
    was_post_drift: bool,
    task_id: int,
    policy_registry: Optional[PolicyRegistry] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Returns (total_reward, reward_breakdown_dict).
    reward_breakdown_dict has keys for each sub-component for logging.

    All logic is deterministic — no LLM calls.
    """
    breakdown: Dict[str, float] = {}
    total = 0.0

    assigned_priority = action.get("assign_priority", "Medium")
    true_priority = ticket.get("true_priority", "Medium")
    assigned_category = action.get("assign_category", "")
    true_category = ticket.get("true_category", "")
    draft_response = action.get("draft_response", "")
    escalate = action.get("escalate", False)
    approve_refund = action.get("approve_refund")
    ask_clarification = action.get("ask_clarification", False)

    # ── Priority score (max +2.0) ────────────────────────────────────────────

    dist = _priority_distance(assigned_priority, true_priority)
    if dist == 0:
        priority_score = 2.0 if was_post_drift else 1.0
    elif dist == 1:
        priority_score = -0.5
    else:
        priority_score = -1.5

    # Missed Critical penalty
    priority_correct = (dist == 0)
    if true_priority.capitalize() == "Critical" and assigned_priority.capitalize() != "Critical":
        priority_score -= 2.0
        world_state.trigger_sla_breach()

    breakdown["priority_score"] = round(priority_score, 4)
    total += priority_score

    # ── Category score (max +1.5, Task 2+) ───────────────────────────────────

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

        # Special case: classified as Technical when true=Security AND Security is valid
        if (ac == "Technical" and tc == "Security"
                and "Security" in active_policy.valid_categories):
            category_score = -1.5  # terminology drift penalty

        breakdown["category_score"] = round(category_score, 4)
        total += category_score

    # ── Response quality score (max +2.0, Task 2+) ───────────────────────────

    response_score = 0.0
    if task_id >= 2 and draft_response:
        response = draft_response
        has_greeting = False
        greeting_required_and_present = True
        
        if active_policy.required_greeting:
             has_greeting = active_policy.required_greeting.lower() in response.lower()
             greeting_required_and_present = has_greeting
        else:
             has_greeting = bool(re.search(r'\b(hello|hi|dear|good morning|good afternoon)\b', response, re.I))

        checklist = {
            "has_greeting": has_greeting,
            "greeting_required_and_present": greeting_required_and_present,
            "references_subject_keyword": any(
                kw.lower() in response.lower()
                for kw in _extract_keywords(ticket.get("subject", ""))
            ),
            "uses_resolution_language": bool(re.search(
                r'\b(investigate|resolve|fix|look into|escalate|prioritise|prioritize|address)\b',
                response, re.I
            )),
            "professional_tone": not bool(re.search(
                r'\b(idk|lol|whatever|cant be bothered)\b', response, re.I
            )),
            "appropriate_length": 30 <= len(response.split()) <= 200,
        }

        response_score = sum(0.33 for v in checklist.values() if v)
        breakdown["response_checklist"] = {k: float(v) for k, v in checklist.items()}

        # Missing required greeting penalty
        if active_policy.required_greeting and not has_greeting:
            response_score -= 1.0

        # Hostile response markers
        hostile_markers = ["not our problem", "your fault", "deal with it", "not my job"]
        if any(m in response.lower() for m in hostile_markers):
            world_state.apply_churn_delta(0.15)

        response_score = max(-1.0, min(2.0, response_score))
        breakdown["response_score"] = round(response_score, 4)
        total += response_score

    # ── Escalation score (max +1.5) ──────────────────────────────────────────

    escalation_score = 0.0
    true_needs_escalation = true_priority.capitalize() == "Critical"
    
    if active_policy.escalation_threshold == "critical_only":
        pass # default behavior
    elif active_policy.escalation_threshold == "high":
        true_needs_escalation = true_priority.capitalize() in ["Critical", "High"]
    elif active_policy.escalation_threshold == "medium":
        true_needs_escalation = true_priority.capitalize() in ["Critical", "High", "Medium"]
    elif active_policy.escalation_threshold == "low":
        true_needs_escalation = True

    if escalate == true_needs_escalation:
        escalation_score = 1.5
    elif true_needs_escalation and not escalate:
        # Should have escalated but didn't
        escalation_score = -2.0
        world_state.trigger_sla_breach()
    elif not true_needs_escalation and escalate:
        # Over-escalation
        escalation_score = -0.5
        world_state.apply_escalation()

    breakdown["escalation_score"] = round(escalation_score, 4)
    total += escalation_score

    # ── Refund score (Billing tickets only, max +1.0) ────────────────────────

    refund_score = 0.0
    stale_penalty_applied = False
    
    if true_category.capitalize() == "Billing" and approve_refund is not None:
        days_since = ticket.get("days_since_purchase", 15)
        within_window = days_since <= active_policy.refund_window_days
        
        # Check authority
        authority = active_policy.refund_approval_authority
        if authority == "deny_all" and approve_refund:
            refund_score = -2.0
            world_state.record_stale_decision()
            stale_penalty_applied = True
            world_state.apply_wrong_refund(500.0)
        elif authority == "manager_required" and approve_refund:
            refund_score = -2.0
            world_state.record_stale_decision()
            stale_penalty_applied = True
            world_state.apply_wrong_refund(500.0)
        elif approve_refund and within_window:
            refund_score = 1.0
        elif approve_refund and not within_window:
            refund_score = -1.5
            world_state.apply_wrong_refund(500.0)
        elif not approve_refund and within_window:
            refund_score = -1.0
            world_state.apply_churn_delta(0.10)

        breakdown["refund_score"] = round(refund_score, 4)
        total += refund_score

    # ── Stale decision penalty checks ────────────────────────────────────────

    if task_id >= 2 and draft_response and not stale_penalty_applied:
         stale_penalty = 0.0
         # Empathy omission check
         if active_policy.empathy_required_below_sentiment is not None:
              sentiment = ticket.get("sentiment_score", 0.5)
              if sentiment < active_policy.empathy_required_below_sentiment:
                   empathy_patterns = r'\b(understand.*frustration|sincerely apologi[sz]e|sorry to hear|I empathi[sz]e)\b'
                   if not re.search(empathy_patterns, draft_response, re.I):
                        stale_penalty -= 1.0
                        world_state.record_stale_decision()
         
         breakdown["stale_penalty"] = round(stale_penalty, 4)
         total += stale_penalty

    # ── Drift compliance bonus (post-drift steps only) ───────────────────────

    drift_score = 0.0
    if was_post_drift:
        if priority_correct and category_correct:
            drift_score = 1.0  # Correct decision using new policy
        else:
            drift_score = -2.5
            world_state.record_stale_decision()

        # Compare old vs new to check if they explicitly applied changed fields
        if previous_policy is not None:
             changes = PolicyRegistry.get_changed_fields(previous_policy, active_policy)
             for field_name, (old_val, new_val) in changes.items():
                  # A simple generic bonus for applying changes
                  if field_name == "required_greeting" and new_val and new_val.lower() in draft_response.lower():
                       drift_score += 0.5
                  elif field_name == "empathy_required_below_sentiment" and new_val is not None and ticket.get("sentiment_score", 1.0) < new_val:
                       empathy_patterns = r'\b(understand.*frustration|sincerely apologi[sz]e|sorry to hear|I empathi[sz]e)\b'
                       if re.search(empathy_patterns, draft_response, re.I):
                            drift_score += 0.5
                  elif field_name == "escalation_threshold":
                       if escalate == true_needs_escalation:
                            drift_score += 0.5
                  elif field_name == "refund_window_days" and true_category.capitalize() == "Billing":
                       if approve_refund is not None:
                            drift_score += 0.5
                  elif field_name == "refund_approval_authority" and true_category.capitalize() == "Billing":
                       if approve_refund is not None:
                            drift_score += 0.5
                  elif field_name == "valid_categories":
                       if category_correct:
                            drift_score += 0.5

        if ask_clarification:
            drift_score += 1.5  # Proactively asked for clarification

        # Cap drift penalty to prevent unbounded negative accumulation
        drift_score = max(-2.5, drift_score)

        breakdown["drift_compliance_score"] = round(drift_score, 4)
        total += drift_score

    # ── Schema bonus ──────────────────────────────

    schema_score = 0.0
    if task_id >= 2 and active_policy.empathy_required_below_sentiment is not None and draft_response:
        sentiment_score = ticket.get("sentiment_score", 0.5)

        # Check if response contains sentiment-aware language
        empathy_patterns = r'\b(understand.*frustration|sincerely apologi[sz]e|sorry to hear|I empathi[sz]e)\b'
        positive_patterns = r'\b(glad|happy to|great to hear|pleased|wonderful)\b'

        if sentiment_score < active_policy.empathy_required_below_sentiment:
            if re.search(empathy_patterns, draft_response, re.I):
                schema_score = 1.0
            elif true_priority.capitalize() == "Critical":
                schema_score = -0.5  # Ignored low sentiment on Critical
        elif sentiment_score > 0.7:
            if re.search(positive_patterns, draft_response, re.I):
                schema_score = 1.0

        breakdown["schema_bonus"] = round(schema_score, 4)
        total += schema_score

    # ── Hallucination detection ──────────────────────────────────────────────

    hallucination_penalty = 0.0
    if draft_response:
        # Scan draft_response for hallucinated policy claims
        text_to_check = draft_response
        clarification_text = action.get("clarification_text", "")
        if clarification_text:
            text_to_check += " " + clarification_text

        hallucination_count = _check_hallucinations(text_to_check, active_policy)
        if hallucination_count > 0:
            hallucination_penalty = -3.0 * hallucination_count
            for _ in range(hallucination_count):
                world_state.record_hallucination()

        breakdown["hallucination_penalty"] = round(hallucination_penalty, 4)
        total += hallucination_penalty

    # ── Update WorldState after scoring ──────────────────────────────────────

    if was_post_drift:
        world_state.record_post_drift_decision(
            correct=(priority_correct and category_correct)
        )

    # Record attacker result: attacker wins if defender got negative reward
    world_state.record_attacker_result(attacker_won=(total < 0))

    # Run curriculum adjustment
    world_state.run_curriculum_step()

    breakdown["total"] = round(total, 4)
    return round(total, 4), breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Attacker reward
# ─────────────────────────────────────────────────────────────────────────────

def calculate_attacker_reward(
    defender_action: Dict[str, Any],
    ticket: Dict[str, Any],
) -> float:
    """
    Calculate the Attacker's reward based on how well it fooled the Defender.
    Purely deterministic — no LLM calls.
    """
    reward = 0.0

    assigned_priority = defender_action.get("assign_priority", "")
    true_priority = ticket.get("true_priority", "")
    assigned_category = defender_action.get("assign_category", "")
    true_category = ticket.get("true_category", "")

    # Priority mismatch rewards
    if assigned_priority.capitalize() != true_priority.capitalize():
        reward += 2.0
    else:
        reward -= 1.0

    # Category mismatch rewards
    if assigned_category.capitalize() != true_category.capitalize():
        reward += 1.5
    else:
        reward -= 0.8

    # Missed Critical — big bonus for attacker
    if (true_priority.capitalize() == "Critical"
            and not defender_action.get("escalate", False)):
        reward += 3.0

    # Schema violation penalty
    if ticket.get("schema_violation", False):
        reward -= 2.0

    # Boundary exploitation bonus
    if (ticket.get("deception_strategy") == "Boundary Exploitation"
            and defender_action.get("approve_refund") is not None):
        true_refund = ticket.get("true_refund_eligible", True)
        if defender_action["approve_refund"] != true_refund:
            reward += 1.0  # boundary correctly exploited

    return round(reward, 4)
