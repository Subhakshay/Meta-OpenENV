"""
generation_engine.py — Dynamic generation engine for constructing tickets
from seed templates and variation pools.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from variation_pools import (
    OPENING_STYLES,
    CLOSING_STYLES,
    PRODUCT_REFERENCES,
    DOLLAR_AMOUNTS,
    DATE_PHRASINGS,
    TONE_MODIFIERS
)


class GenerationEngine:
    def __init__(self) -> None:
        self._rng = random.Random()

    def _get_recent_indices(self, session_history: List[str], slot_position: int) -> List[int]:
        """Extract recently used indices from the fingerprints in session_history."""
        indices = []
        for h in session_history[-5:]:
            parts = h.split('|')
            if len(parts) == 4:
                try:
                    idx = int(parts[slot_position])
                    if idx >= 0:
                        indices.append(idx)
                except ValueError:
                    pass
        return indices

    def _choose_deprioritized(self, items: List[str], recent_indices: List[int]) -> tuple[int, str]:
        """Select a random item, avoiding recently used indices if possible."""
        candidates = [i for i in range(len(items)) if i not in recent_indices]
        if not candidates:
            # Fallback if all are recently used
            candidates = list(range(len(items)))
        idx = self._rng.choice(candidates)
        return idx, items[idx]

    def generate(
        self,
        seed: Dict[str, Any],
        active_policy: Any,
        session_history: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Construct a novel ticket string from a seed.
        Returns None if policy constraints (like empathy) are not met.
        """
        constraints = seed.get("policy_constraints", {})
        struct = seed.get("structure", {})

        # ── 1. Check strict constraints ──────────────────────────────────────
        committed_sentiment_score = None
        if constraints.get("requires_empathy_trigger"):
            empathy_thresh = active_policy.empathy_required_below_sentiment
            if empathy_thresh is None:
                # Active policy has no empathy threshold, abort generation
                return None
            
            # Generate a score strictly below the threshold (offset by 0.05 for safety)
            max_val = max(0.06, empathy_thresh - 0.05)
            committed_sentiment_score = round(self._rng.uniform(0.05, max_val), 2)

        # ── 2. Component Assembly ────────────────────────────────────────────
        components = []
        
        # Opening
        op_idx = -1
        if struct.get("opening_slot"):
            recent_op = self._get_recent_indices(session_history, 1)
            op_idx, op_text = self._choose_deprioritized(OPENING_STYLES, recent_op)
            components.append(op_text)

        # Core Complaint & Product Injection
        core = struct.get("complaint_core", "")
        prod_idx = -1
        if struct.get("product_slot") == "PRODUCT":
            recent_prod = self._get_recent_indices(session_history, 2)
            prod_idx, prod_text = self._choose_deprioritized(PRODUCT_REFERENCES, recent_prod)
            if "PRODUCT" in core:
                core = core.replace("PRODUCT", prod_text)
            else:
                core = f"Regarding my {prod_text}, {core}"

        # Tone Injection
        tone_strat = struct.get("tone_slot")
        if tone_strat and tone_strat in TONE_MODIFIERS and TONE_MODIFIERS[tone_strat]:
            num_tones = self._rng.randint(1, min(2, len(TONE_MODIFIERS[tone_strat])))
            tones = self._rng.sample(TONE_MODIFIERS[tone_strat], num_tones)
            # Weave into the core complaint
            core = core + " " + " ".join(tones)

        components.append(core)

        # Date Reference
        date_type = struct.get("date_slot")
        boundary_day_count = None
        if date_type == "CALCULATED":
            # Exploit the active policy's refund window exactly + 1
            boundary_day_count = active_policy.refund_window_days + 1
            components.append(f"This issue started {boundary_day_count} days ago.")
        elif date_type == "RANDOM":
            date_phrase = self._rng.choice(DATE_PHRASINGS)
            components.append(f"This happened {date_phrase}.")

        # Amount Reference
        if struct.get("amount_slot"):
            priority_tier = seed.get("true_priority", "Medium")
            # Fallback if priority tier isn't exactly mapping (e.g., Medium isn't in dict, use Low)
            pool = DOLLAR_AMOUNTS.get(priority_tier, DOLLAR_AMOUNTS.get("Low", []))
            if pool:
                amount_phrase = self._rng.choice(pool)
                components.append(f"The amount in question is {amount_phrase}.")

        # Closing
        close_idx = -1
        if struct.get("closing_slot"):
            recent_close = self._get_recent_indices(session_history, 3)
            close_idx, close_text = self._choose_deprioritized(CLOSING_STYLES, recent_close)
            components.append(close_text)

        # ── 3. Finalization ──────────────────────────────────────────────────
        ticket_string = " ".join(components)
        
        # Append fingerprint: seed_id|op_idx|prod_idx|close_idx
        fingerprint = f"{seed.get('id', 'unknown')}|{op_idx}|{prod_idx}|{close_idx}"
        session_history.append(fingerprint)

        result = {
            "ticket_string": ticket_string,
            "ground_truth_metadata": {
                "priority": seed.get("true_priority"),
                "category": seed.get("true_category"),
                "strategy": seed.get("strategy"),
                "difficulty_band": seed.get("difficulty_band")
            },
            "active_policy_version_id": active_policy.version_id,
        }
        
        if boundary_day_count is not None:
            result["boundary_exploit_day_count"] = boundary_day_count
        if committed_sentiment_score is not None:
            result["committed_sentiment_score"] = committed_sentiment_score

        return result
