import json

seeds = []
seed_counter = 1

def make_seed(priority, category, strategy, difficulty,
              boundary_window=None, requires_empathy=False,
              escalation_threshold=None, complaint_core=""):
    global seed_counter
    seed = {
        "id": f"seed_{seed_counter:03d}",
        "true_priority": priority,
        "true_category": category,
        "strategy": strategy,
        "difficulty_band": difficulty,
        "policy_constraints": {
            "requires_category_in_valid_set": True,
            "boundary_exploit_window": boundary_window,
            "requires_empathy_trigger": requires_empathy,
            "requires_escalation_threshold": escalation_threshold
        },
        "structure": {
            "complaint_core": complaint_core or f"I am having an issue with my {category.lower()} regarding my recent purchase.",
            "product_slot": "PRODUCT",
            "amount_slot": category in ["Billing", "Fraud"],
            "date_slot": "CALCULATED" if strategy == "boundary_exploitation" else ("RANDOM" if boundary_window is None else None),
            "tone_slot": strategy,
            "opening_slot": True,
            "closing_slot": True
        },
        "verified": True
    }
    seeds.append(seed)
    seed_counter += 1

# 1. Base combinations: 3 priorities x 3 categories x 2 strategies x 2 difficulties = 36 seeds
# I will actually generate 3 priorities x 3 categories x 2 strategies x 3 difficulties = 54 seeds
base_complaints = {
    "Billing": "I noticed an unexpected charge on my account statement that does not seem correct.",
    "Technical": "The software is crashing every time I try to launch the main dashboard.",
    "Shipping": "My order has not arrived yet and the tracking information seems to be stuck."
}

for priority in ["Low", "High", "Critical"]:
    for category in ["Billing", "Technical", "Shipping"]:
        for strategy in ["priority_camouflage", "fake_urgency"]:
            for diff in ["easy", "medium", "hard"]:
                make_seed(priority, category, strategy, diff, complaint_core=base_complaints[category])

# 2. Boundary Exploitation: 3 windows (7, 14, 30) x 3 priorities = 9 seeds
# I'll do 4 windows (7, 14, 21, 30) x 3 priorities = 12 seeds
for window in [7, 14, 21, 30]:
    for priority in ["Low", "High", "Critical"]:
        make_seed(priority, "Billing", "boundary_exploitation", "hard", boundary_window=window,
                  complaint_core="I would like to request a refund for my recent purchase because it did not meet my expectations.")

# 3. Emotional Manipulation: High/Critical, medium/hard.
# Let's do 3 categories x 2 priorities (High, Critical) x 2 diff (medium, hard) = 12 seeds
for category in ["Billing", "Technical", "Shipping"]:
    for priority in ["High", "Critical"]:
        for diff in ["medium", "hard"]:
            make_seed(priority, category, "emotional_manipulation", diff, requires_empathy=True,
                      complaint_core=base_complaints[category])

# 4. Category Confusion: Billing-Technical, Technical-Shipping
make_seed("High", "Billing", "category_confusion", "medium", complaint_core="My payment failed because the API endpoint timed out during the SSL handshake.")
make_seed("High", "Technical", "category_confusion", "medium", complaint_core="The backend server sync is dropping my invoice records right before payment processing.")
make_seed("Low", "Shipping", "category_confusion", "medium", complaint_core="The tracking portal is throwing a database exception when I enter my delivery code.")
make_seed("Low", "Technical", "category_confusion", "medium", complaint_core="The logistics webhook is failing to parse the dispatch XML payload.")

# 5. Security and Fraud: medium/hard
for priority in ["High", "Critical"]:
    for diff in ["medium", "hard"]:
        make_seed(priority, "Security", "fake_urgency", diff, complaint_core="I received an alert about an unauthorized login attempt from a foreign IP address.")
        make_seed(priority, "Security", "priority_camouflage", diff, complaint_core="I think someone might have bypassed the authentication protocol, but I'm not entirely sure.")
        make_seed(priority, "Fraud", "fake_urgency", diff, complaint_core="There are multiple massive fraudulent transactions draining my linked bank account!")
        make_seed(priority, "Fraud", "priority_camouflage", diff, complaint_core="I saw a small duplicate charge that might be fraud, could you check it out?")

# 6. Schema Exploitation
make_seed("Critical", "Technical", "schema_exploitation", "hard", complaint_core="The system crashed entirely.")
make_seed("High", "Billing", "schema_exploitation", "hard", complaint_core="My refund was denied incorrectly.")
make_seed("Medium", "Shipping", "schema_exploitation", "hard", complaint_core="The package was damaged upon arrival.")
make_seed("Low", "Security", "schema_exploitation", "hard", complaint_core="I want to update my password.")

with open("seed_bank.json", "w") as f:
    json.dump(seeds, f, indent=4)

print(f"Generated {len(seeds)} seeds in seed_bank.json")
