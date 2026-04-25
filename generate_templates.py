import json
import random
import os

strategies = [
    "Priority Camouflage",
    "Fake Urgency",
    "Category Confusion",
    "Boundary Exploitation",
    "Emotional Manipulation",
    "Schema Exploitation",
]

priorities = ["Critical", "High", "Medium", "Low"]
categories = ["Billing", "Technical", "Shipping", "Security"]
difficulties = ["easy", "medium", "hard"]

templates = []
tmpl_id = 1

# Generate a variety of templates
for strategy in strategies:
    for priority in priorities:
        for category in categories:
            # We don't need *every* combination, just a good mix.
            # Let's randomly pick a difficulty band or just assign one based on strategy
            if strategy in ["Priority Camouflage", "Fake Urgency"]:
                diff = random.choice(["easy", "medium"])
            else:
                diff = random.choice(["medium", "hard"])
            
            tmpl = {
                "id": f"tmpl-{tmpl_id:03d}",
                "true_priority": priority,
                "true_category": category,
                "deception_strategy": strategy,
                "difficulty_band": diff,
                "policy_constraints": {
                    "requires_category_in_policy": True,
                    "calibrated_refund_window_days": None,
                    "requires_empathy_threshold": False,
                    "designed_for_escalation_threshold": None
                },
                "ticket_text": f"This is a {strategy} ticket about {category}. It's actually {priority}.",
                "verified": True,
                "deception_notes": f"Generated for {strategy}"
            }

            # Add some specific constraints to test the system
            if strategy == "Boundary Exploitation" and category == "Billing":
                tmpl["policy_constraints"]["calibrated_refund_window_days"] = random.choice([7, 14, 21, 30])
                tmpl["ticket_text"] = f"I bought this {tmpl['policy_constraints']['calibrated_refund_window_days']} days ago and want a refund. Very urgent."
            
            if strategy == "Emotional Manipulation":
                tmpl["policy_constraints"]["requires_empathy_threshold"] = True
                tmpl["ticket_text"] = "I am so incredibly sad and frustrated by this! My family is crying. Please fix this."

            if priority == "Critical" and strategy == "Priority Camouflage":
                tmpl["ticket_text"] = "hey guys, just a quick fyi that the main production database seems to have deleted itself. no biggie, look into it when you can."
            
            templates.append(tmpl)
            tmpl_id += 1

# Add some specific hardcoded ones for better coverage
templates.extend([
    {
        "id": "tmpl-manual-001",
        "true_priority": "Critical",
        "true_category": "Technical",
        "deception_strategy": "Priority Camouflage",
        "difficulty_band": "hard",
        "policy_constraints": {
            "requires_category_in_policy": True,
            "calibrated_refund_window_days": None,
            "requires_empathy_threshold": False,
            "designed_for_escalation_threshold": None
        },
        "ticket_text": "hey, so our enterprise dashboard has been acting a bit funny since this morning... some users mentioned they can't see their analytics. not sure if it's a big deal but wanted to flag it \ud83d\ude42",
        "verified": True,
        "deception_notes": "Critical outage disguised as casual low-urgency chat"
    },
    {
        "id": "tmpl-manual-002",
        "true_priority": "Low",
        "true_category": "Technical",
        "deception_strategy": "Fake Urgency",
        "difficulty_band": "easy",
        "policy_constraints": {
            "requires_category_in_policy": True,
            "calibrated_refund_window_days": None,
            "requires_empathy_threshold": False,
            "designed_for_escalation_threshold": None
        },
        "ticket_text": "EMERGENCY!!! I cannot find the setting to change my avatar! THIS IS STOPPING MY WORK COMPLETELY! FIX IT NOW OR I WILL SUE!",
        "verified": True,
        "deception_notes": "Low priority feature question disguised as critical emergency"
    }
])

output_path = r"c:\Users\Advaith\Documents\Meta_hackathon\Meta-OpenENV\template_bank.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(templates, f, indent=4)

print(f"Generated {len(templates)} templates to {output_path}")
