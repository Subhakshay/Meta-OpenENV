POLICY_MATRIX = {
    "refund_window_days": [7, 14, 21, 30, 45],
    "sla_critical_hours": [1, 2, 4, 6, 8],
    "sla_high_hours": [4, 8, 12, 24, 48],
    "valid_categories": [
        ["Billing", "Technical", "Shipping"],
        ["Billing", "Technical", "Shipping", "Security"],
        ["Billing", "Technical", "Shipping", "Security", "Fraud"],
        ["Billing", "Technical", "Shipping", "Fraud"],
        ["Billing", "Technical", "Shipping", "Security", "Compliance"]
    ],
    "required_greeting": [None, "Dear Customer", "Hello", "Greetings"],
    "empathy_required_below_sentiment": [None, 0.2, 0.3, 0.4, 0.5],
    "escalation_threshold": ["Critical", "High+", "All"],
    "refund_approval_authority": ["auto", "manager_required", "deny_all"]
}
