"""Patch train_colab.ipynb: n=12 -> n=20 in attacker batch, and fix markdown description."""
with open("train_colab.ipynb", "r", encoding="utf-8") as f:
    content = f.read()

# Patch 1: markdown cell description
content = content.replace(
    "Each episode has 12 steps.",
    "Each episode has 20 steps.",
)

# Patch 2: attacker generate_batch n=12 -> n=20 in the code cell
# The raw JSON stores Python newlines as literal \n inside the string
old = "                    n=12,\\n\","
new = "                    n=20,\\n\","
assert old in content, f"Pattern not found in notebook: {old!r}"
content = content.replace(old, new)

with open("train_colab.ipynb", "w", encoding="utf-8") as f:
    f.write(content)

import re
remaining = re.findall(r"n=12|12 steps", content)
found20   = re.findall(r"n=20|20 steps", content)
print("Remaining n=12/12steps:", remaining)
print("Found n=20/20steps:", found20)
print("Done.")
