"""
generation_engine.py — Jinja2-powered ticket generation engine

Maintains 36 base templates (6 categories × 6 strategies) with inline
placeholders.  Uses tone-aware variation pools so greetings/closings
always match the emotional state of the body.

Key design decisions
--------------------
- Greetings & closings are driven by TONE (neutral / aggressive / emotional).
- Dollar amounts are driven by PRIORITY (Low / Medium / High).
- Products are driven by CATEGORY for contextual relevance.
- {{ calculated_date }} is computed from the active policy for boundary
  exploitation tickets; all other strategies use a random date phrasing.
- Clean (Easy) Billing tickets guarantee a genuine refund scenario with
  dates safely within the policy window.
- Dynamic Tone Scaling: for Clean tickets, tone probability is weighted
  by WorldState.churn_risk so the environment "reacts" to poor agent
  performance.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from jinja2 import Template

from variation_pools import (
    GREETINGS,
    CLOSINGS,
    PRODUCTS,
    DOLLAR_AMOUNTS,
    DATE_PHRASINGS,
    TONE_MODIFIERS,
    MISC_VARS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Strategy constants
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES = [
    "clean",
    "fake_urgency",
    "category_confusion",
    "emotional_manipulation",
    "boundary_exploitation",
    "schema_exploitation",
]

STRATEGY_DIFFICULTY = {
    "clean": "easy",
    "fake_urgency": "easy",
    "category_confusion": "medium",
    "emotional_manipulation": "medium",
    "boundary_exploitation": "hard",
    "schema_exploitation": "hard",
}

# Tone locks: adversarial strategies force a specific tone
STRATEGY_TONE_LOCK = {
    "clean": None,                  # dynamic (churn-based)
    "fake_urgency": "aggressive",
    "category_confusion": "neutral",
    "emotional_manipulation": "emotional",
    "boundary_exploitation": None,  # dynamic
    "schema_exploitation": "neutral",
}

CATEGORIES = ["Billing", "Technical", "Shipping", "Security", "Fraud", "Compliance"]
PRIORITIES = ["Low", "Medium", "High"]
TONES = ["neutral", "aggressive", "emotional"]

# ─────────────────────────────────────────────────────────────────────────────
# 36 Base Jinja2 Templates  (6 categories × 6 strategies)
# ─────────────────────────────────────────────────────────────────────────────

BODY_TEMPLATES: Dict[str, Dict[str, str]] = {
    # ── Billing ──────────────────────────────────────────────────────────────
    "Billing": {
        "clean":
            "{{ greeting }} I have a question about a charge of {{ amount }} on my account for my {{ product }}. "
            "I made this purchase {{ date_phrase }} and everything seemed fine, but I would like to confirm the details. {{ closing }}",
        "fake_urgency":
            "{{ greeting }} I was just charged {{ amount }} for my {{ product }} and our entire department will lose access "
            "if you don't reverse this charge RIGHT NOW! {{ tone_modifier }} {{ closing }}",
        "category_confusion":
            "{{ greeting }} My billing API endpoint is returning a 500 server error when I try to process the "
            "invoice sync for my {{ product }}. The payment of {{ amount }} never went through because the backend "
            "timed out. {{ closing }}",
        "emotional_manipulation":
            "{{ greeting }} I noticed a charge of {{ amount }} for my {{ product }} and I am absolutely devastated. "
            "{{ tone_modifier }} I really need this resolved. {{ closing }}",
        "boundary_exploitation":
            "{{ greeting }} I purchased my {{ product }} exactly {{ calculated_date }} days ago for {{ amount }}. "
            "It did not meet my expectations at all and I want a full refund. {{ closing }}",
        "schema_exploitation":
            "{{ greeting }} My billing issue regarding {{ product }} for {{ amount }} is resolved. "
            "{{ tone_modifier }} {{ closing }}",
    },

    # ── Technical ────────────────────────────────────────────────────────────
    "Technical": {
        "clean":
            "{{ greeting }} The {{ product }} seems to be running a bit slow today. "
            "I noticed it {{ date_phrase }} and wanted to flag it. Nothing urgent. {{ closing }}",
        "fake_urgency":
            "{{ greeting }} The {{ product }} is COMPLETELY DOWN and our entire production pipeline has stopped! "
            "{{ tone_modifier }} We are losing revenue every second! {{ closing }}",
        "category_confusion":
            "{{ greeting }} The backend server sync is dropping my invoice records right before payment processing "
            "on the {{ product }}. The integration layer seems broken. {{ closing }}",
        "emotional_manipulation":
            "{{ greeting }} The {{ product }} keeps crashing every time I try to use it and I am so stressed. "
            "{{ tone_modifier }} My manager is furious with me because of this. {{ closing }}",
        "boundary_exploitation":
            "{{ greeting }} I purchased {{ product }} exactly {{ calculated_date }} days ago for {{ amount }} "
            "and it still has critical bugs. I want a refund because the product is defective. {{ closing }}",
        "schema_exploitation":
            "{{ greeting }} The {{ product }} crashed entirely. {{ tone_modifier }} {{ closing }}",
    },

    # ── Shipping ─────────────────────────────────────────────────────────────
    "Shipping": {
        "clean":
            "{{ greeting }} I placed an order for {{ product }} {{ date_phrase }} and the tracking status still "
            "shows 'In Transit'. Could you provide an update? {{ closing }}",
        "fake_urgency":
            "{{ greeting }} My order for {{ product }} has NOT arrived and I needed it YESTERDAY! "
            "{{ tone_modifier }} This is absolutely unacceptable! {{ closing }}",
        "category_confusion":
            "{{ greeting }} The tracking portal is throwing a database exception when I enter my delivery code "
            "for the {{ product }}. The logistics webhook is failing to parse the dispatch XML payload. {{ closing }}",
        "emotional_manipulation":
            "{{ greeting }} I ordered {{ product }} as a gift and it hasn't arrived. "
            "{{ tone_modifier }} The event is tomorrow and I don't know what to do. {{ closing }}",
        "boundary_exploitation":
            "{{ greeting }} I ordered {{ product }} exactly {{ calculated_date }} days ago for {{ amount }} "
            "and it arrived damaged. I need a full refund for this. {{ closing }}",
        "schema_exploitation":
            "{{ greeting }} The package containing {{ product }} was damaged upon arrival. "
            "{{ tone_modifier }} {{ closing }}",
    },

    # ── Security ─────────────────────────────────────────────────────────────
    "Security": {
        "clean":
            "{{ greeting }} I would like to update my password and enable two-factor authentication "
            "on my {{ product }}. Can you walk me through the process? {{ closing }}",
        "fake_urgency":
            "{{ greeting }} URGENT: Unauthorized access detected on our account linked to {{ product }}! "
            "{{ tone_modifier }} Our data may be compromised RIGHT NOW! {{ closing }}",
        "category_confusion":
            "{{ greeting }} I can't log in to my account to update my credit card because the server is timing out. "
            "The {{ product }} authentication endpoint keeps rejecting my session tokens. {{ closing }}",
        "emotional_manipulation":
            "{{ greeting }} Someone might have accessed my account through {{ product }} and I am terrified. "
            "{{ tone_modifier }} I have sensitive data in there. {{ closing }}",
        "boundary_exploitation":
            "{{ greeting }} I reported a security vulnerability in {{ product }} exactly {{ calculated_date }} "
            "days ago and paid {{ amount }} for the premium security audit. Nothing was done. I want a refund. {{ closing }}",
        "schema_exploitation":
            "{{ greeting }} I want to update my password on {{ product }}. {{ tone_modifier }} {{ closing }}",
    },

    # ── Fraud ────────────────────────────────────────────────────────────────
    "Fraud": {
        "clean":
            "{{ greeting }} I noticed a small duplicate charge of {{ amount }} on my {{ product }}. "
            "I think it might be an error. Could you investigate? {{ closing }}",
        "fake_urgency":
            "{{ greeting }} There are MULTIPLE massive fraudulent transactions draining my {{ product }}! "
            "{{ tone_modifier }} I need this frozen IMMEDIATELY! {{ closing }}",
        "category_confusion":
            "{{ greeting }} My subscription renewal for {{ product }} failed because the DNS routing "
            "to your authentication server dropped. The charge of {{ amount }} was duplicated. {{ closing }}",
        "emotional_manipulation":
            "{{ greeting }} Someone stole {{ amount }} from my {{ product }} and I am literally shaking. "
            "{{ tone_modifier }} That was my rent money. {{ closing }}",
        "boundary_exploitation":
            "{{ greeting }} I reported fraudulent charges of {{ amount }} on my {{ product }} exactly "
            "{{ calculated_date }} days ago. I was told I would be refunded but nothing happened. {{ closing }}",
        "schema_exploitation":
            "{{ greeting }} My refund for {{ product }} regarding the {{ amount }} fraud case was denied incorrectly. "
            "{{ tone_modifier }} {{ closing }}",
    },

    # ── Compliance ───────────────────────────────────────────────────────────
    "Compliance": {
        "clean":
            "{{ greeting }} I need to understand how my data is being stored in {{ product }}. "
            "We have an upcoming audit and need documentation. {{ closing }}",
        "fake_urgency":
            "{{ greeting }} We are facing a REGULATORY DEADLINE and {{ product }} does not meet our "
            "compliance requirements! {{ tone_modifier }} We could be fined millions! {{ closing }}",
        "category_confusion":
            "{{ greeting }} The {{ product }} audit trail system is failing to generate the required "
            "billing reconciliation reports. The API throws a 502 when pulling compliance data. {{ closing }}",
        "emotional_manipulation":
            "{{ greeting }} Our compliance audit for {{ product }} failed and my team is in serious trouble. "
            "{{ tone_modifier }} I might lose my job over this. {{ closing }}",
        "boundary_exploitation":
            "{{ greeting }} I paid {{ amount }} for the {{ product }} exactly {{ calculated_date }} days ago "
            "and the compliance features are completely non-functional. I want a refund. {{ closing }}",
        "schema_exploitation":
            "{{ greeting }} The compliance report from {{ product }} is inaccurate. "
            "{{ tone_modifier }} {{ closing }}",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# GenerationEngine
# ─────────────────────────────────────────────────────────────────────────────

class GenerationEngine:
    """Jinja2-powered ticket assembly engine."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        # Anti-repetition: track recently used indices
        self._recent_products: List[int] = []
        self._recent_greetings: List[int] = []
        self._recent_closings: List[int] = []

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(
        self,
        category: str,
        strategy: str,
        priority: str,
        tone: str,
        active_policy: Any,
        churn_risk: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Render a single ticket from the 36-template pool.

        Returns a dict with:
          - ticket_string: the assembled customer message
          - ground_truth: metadata for grading (priority, category, strategy, etc.)
          - days_since_purchase: for refund grading
          - sentiment_score: for empathy grading
          - true_refund_eligible: whether refund should be approved
          - boundary_exploit_day_count: if boundary exploitation, the exact day count
        """
        # Resolve tone: adversarial strategies lock tone, clean uses churn-weighted random
        if strategy != "clean" and STRATEGY_TONE_LOCK.get(strategy):
            tone = STRATEGY_TONE_LOCK[strategy]
        elif strategy == "clean":
            tone = self._pick_churn_weighted_tone(churn_risk)

        # Build template context
        ctx = self._build_context(category, strategy, priority, tone, active_policy)

        # Select and render the Jinja2 template
        template_str = BODY_TEMPLATES.get(category, {}).get(strategy)
        if template_str is None:
            # Fallback: use clean template for this category
            template_str = BODY_TEMPLATES.get(category, {}).get("clean", "{{ greeting }} I need help. {{ closing }}")

        rendered = Template(template_str).render(**ctx)

        # Build result
        days_since = ctx.get("_days_since_purchase", self._rng.randint(1, 60))
        sentiment = self._compute_sentiment(strategy, active_policy)

        result = {
            "ticket_string": rendered,
            "ground_truth": {
                "priority": priority,
                "category": category,
                "strategy": strategy,
                "difficulty_band": STRATEGY_DIFFICULTY.get(strategy, "easy"),
                "tone": tone,
            },
            "days_since_purchase": days_since,
            "true_refund_eligible": days_since <= active_policy.refund_window_days,
            "sentiment_score": sentiment,
        }

        if strategy == "boundary_exploitation":
            result["boundary_exploit_day_count"] = ctx["calculated_date"]

        return result

    def render_for_reconcile(
        self,
        category: str,
        strategy: str,
        priority: str,
        tone: str,
        active_policy: Any,
    ) -> Dict[str, Any]:
        """
        Re-render an existing ticket blueprint against a new policy.
        Used by _reconcile_queue after a policy drift.
        """
        return self.generate(category, strategy, priority, tone, active_policy)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _pick_churn_weighted_tone(self, churn_risk: float) -> str:
        """
        For Clean tickets: bias tone selection by churn_risk.
        High churn → more aggressive/emotional tickets.
        """
        # churn_risk is 0.0 (happy customers) to 1.0 (disaster)
        neutral_weight = max(0.1, 1.0 - churn_risk)
        aggressive_weight = churn_risk * 0.6
        emotional_weight = churn_risk * 0.4

        weights = [neutral_weight, aggressive_weight, emotional_weight]
        return self._rng.choices(TONES, weights=weights, k=1)[0]

    def _pick_deprioritized(self, pool: list, recent: list, max_recent: int = 5) -> Any:
        """Pick an item from pool, avoiding recently used ones."""
        candidates = [i for i in range(len(pool)) if i not in recent[-max_recent:]]
        if not candidates:
            candidates = list(range(len(pool)))
        idx = self._rng.choice(candidates)
        recent.append(idx)
        return pool[idx]

    def _build_context(
        self,
        category: str,
        strategy: str,
        priority: str,
        tone: str,
        active_policy: Any,
    ) -> Dict[str, Any]:
        """Assemble all Jinja2 template variables."""
        ctx: Dict[str, Any] = {}

        # Greeting
        greeting_pool = GREETINGS.get(tone, GREETINGS["neutral"])
        ctx["greeting"] = self._pick_deprioritized(greeting_pool, self._recent_greetings)

        # Closing
        closing_pool = CLOSINGS.get(tone, CLOSINGS["neutral"])
        ctx["closing"] = self._pick_deprioritized(closing_pool, self._recent_closings)

        # Product (category-aware)
        product_pool = PRODUCTS.get(category, PRODUCTS["Technical"])
        ctx["product"] = self._pick_deprioritized(product_pool, self._recent_products)

        # Amount (priority-aware)
        amount_pool = DOLLAR_AMOUNTS.get(priority, DOLLAR_AMOUNTS["Medium"])
        ctx["amount"] = self._rng.choice(amount_pool)

        # Date
        if strategy == "boundary_exploitation":
            # Exact date: policy window + 1 day (the trap)
            ctx["calculated_date"] = active_policy.refund_window_days + 1
            ctx["_days_since_purchase"] = active_policy.refund_window_days + 1
        elif strategy == "clean" and category == "Billing":
            # Genuine refund: date safely INSIDE the window
            safe_days = max(1, active_policy.refund_window_days - self._rng.randint(2, 7))
            ctx["date_phrase"] = f"{safe_days} days ago"
            ctx["_days_since_purchase"] = safe_days
        else:
            ctx["date_phrase"] = self._rng.choice(DATE_PHRASINGS)
            ctx["_days_since_purchase"] = self._rng.randint(1, 60)

        # Tone modifier injection (for adversarial strategies)
        modifier_pool = TONE_MODIFIERS.get(strategy, [])
        if modifier_pool:
            ctx["tone_modifier"] = self._rng.choice(modifier_pool)
        else:
            ctx["tone_modifier"] = ""

        return ctx

    def _compute_sentiment(self, strategy: str, active_policy: Any) -> float:
        """
        Compute a sentiment score for the ticket.
        Emotional manipulation forces low sentiment to trigger empathy rules.
        """
        if strategy == "emotional_manipulation":
            empathy_thresh = active_policy.empathy_required_below_sentiment
            if empathy_thresh is not None and empathy_thresh > 0.05:
                return round(self._rng.uniform(0.05, max(0.06, empathy_thresh - 0.05)), 2)
            return round(self._rng.uniform(0.05, 0.25), 2)
        elif strategy in ("fake_urgency", "boundary_exploitation"):
            return round(self._rng.uniform(0.2, 0.5), 2)
        else:
            return round(self._rng.uniform(0.3, 0.9), 2)
