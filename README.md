---
title: Support Triage Env
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🎫 CustomerSupportEnv v2

[cite_start]An **OpenEnv-compliant** reinforcement learning environment designed to train AI agents in real-world customer support triage. [cite_start]Agents must manage priority classification, category tagging, multi-turn dialogue, and stateful business consequences.

---

## 🌍 Environment Overview

[cite_start]A continuous queue of support tickets is generated procedurally using 8 blueprint families and random variables to ensure millions of unique combinations. [cite_start]The agent's goal is to resolve tickets while maintaining a healthy "World State".

### ⚙️ Core Mechanics
* [cite_start]**Multi-turn Dialogue**: In Task 3, agents must use the `ASK` action to clarify ambiguous tickets before resolving them.
* **Stateful MDP**: Actions have real consequences. [cite_start]Poor responses spike `customer_churn_risk`, and missed critical tickets increment `sla_breach_count`.
* [cite_start]**Business Constraints**: Over-escalating drains the `company_balance` and fills the `escalation_queue`.

---

## 🪜 Tasks & Difficulty

| ID | Name | Difficulty | Scoring Focus |
|:---|:---|:---|:---|
| `task_1_priority` | Priority Assignment | **Easy** | [cite_start]Clear signals; only priority is scored. |
| `task_2_classification` | Ticket Classification | **Medium** | Mix of signals; [cite_start]Priority, Category, and Response quality are scored. |
| `task_3_full_triage` | Full Triage | **Hard** | Multi-turn; requires `ASK` for ambiguity. [cite_start]World State is scored. |

---

## 🚀 Action & Observation Spaces

### Action Space (JSON)
[cite_start]The agent submits actions using the following schema:
```json
{
  "action_type": "classify | ask",
  "assign_priority": "low | medium | high | critical",
  "assign_category": "billing | technical | account | shipping | general | refund",
  "response_text": "string (Natural language reply)",
  "escalate": "boolean",
  "clarifying_question": "string (Used only with action_type: ask)"
}