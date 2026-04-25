# The Gauntlet + Shifting Sands — Project Summary

Technical reference document for the dynamic adversarial RL environment.

---

## 1. Architecture Overview

The Gauntlet + Shifting Sands is a dynamic, adversarial Reinforcement Learning environment for customer support ticket triage. It is designed to evaluate and train LLM agents (the "Defender") under two simultaneous pressures: an adversarial ticket generator (the "Attacker") that crafts deceptive tickets, and a policy drift system ("Shifting Sands") that changes the rules the Defender must follow mid-episode.

The system is **OpenEnv-compliant** and exposes a **Gym-style reset/step API** via a **FastAPI** server. All telemetry is persisted to an **async SQLite** database.

### Component Map

| Component                | File                    | Purpose                                                        |
|--------------------------|-------------------------|----------------------------------------------------------------|
| Policy Matrix            | `policy_matrix.py`      | Defines the universe of valid policy field values              |
| Policy Version/Registry  | `policy.py`             | Runtime policy sampling with meaningful-difference enforcement |
| Drift Scheduler          | `drift_scheduler.py`    | Pre-resolves all drift events at episode start                 |
| Variation Pools          | `variation_pools.py`    | Raw phrasing material for ticket generation (pure data)        |
| Seed Bank                | `seed_bank.json`        | 100+ structural ticket templates with ground truth metadata    |
| Generation Engine        | `generation_engine.py`  | Assembles novel tickets from seeds and variation pools         |
| Template Sampler         | `template_sampler.py`   | Policy-aware seed filtering with constraint relaxation         |
| Attacker Agent           | `attacker.py`           | Orchestrates adversarial ticket batch generation               |
| Core Environment         | `environment.py`        | Gym-style reset/step loop, queue management, observations      |
| Reward Calculator        | `rewards.py`            | Deterministic multi-objective reward for Defender and Attacker  |
| World State              | `world_state.py`        | Mutable episode state and curriculum controller                |
| FastAPI Server           | `main.py`               | HTTP endpoints for /reset, /step, /metrics                     |
| Database Layer           | `db.py`                 | Async SQLite persistence for all telemetry                     |
| Local Test Runner        | `inference.py`          | Bypasses HTTP, runs episodes directly for development          |

---

## 2. Policy System

### Policy Matrix

All valid policy configurations are defined in `policy_matrix.py` as a single dictionary with exactly 8 keys. This is the sole source of truth for what constitutes a valid policy. No policy values exist anywhere else in the codebase.

| Field                              | Options                                                                                                      |
|------------------------------------|--------------------------------------------------------------------------------------------------------------|
| `refund_window_days`               | 7, 14, 21, 30, 45                                                                                           |
| `sla_critical_hours`               | 1, 2, 4, 6, 8                                                                                               |
| `sla_high_hours`                   | 4, 8, 12, 24, 48                                                                                            |
| `valid_categories`                 | [Billing,Technical,Shipping], +Security, +Security+Fraud, +Fraud, +Security+Compliance                       |
| `required_greeting`                | None, "Dear Customer", "Hello", "Greetings"                                                                 |
| `empathy_required_below_sentiment` | None, 0.2, 0.3, 0.4, 0.5                                                                                   |
| `escalation_threshold`             | "Critical", "High+", "All"                                                                                  |
| `refund_approval_authority`        | "auto", "manager_required", "deny_all"                                                                      |

### PolicyVersion

A frozen dataclass whose fields mirror the 8 matrix keys plus a `version_id` string. Named versions (v1, v2, v3) no longer exist. All policies are runtime-sampled objects with IDs like `dyn-init`, `dyn-005`.

### PolicyRegistry

- `sample_policy(version_id)` randomly selects one value per matrix field.
- Before accepting, it calls `_is_meaningfully_different(last, candidate)` which requires at least **2 fields** to differ from the previous policy in history.
- If the candidate fails, it resamples up to **10 times**. After 10 failed attempts, the last candidate is accepted anyway.
- Every accepted policy is appended to an internal history list.
- `get_current_policy()` returns the most recent history entry.

---

## 3. Drift Scheduler

The `DriftScheduler` takes `episode_length` and a `PolicyRegistry` at initialization. It exposes a single public method `schedule_episode(difficulty_level)` that returns a fully resolved list of `DriftEvent` objects before the episode begins. No drift logic runs during episode execution.

### Drift Count (from difficulty)

| Difficulty Range   | Number of Drifts |
|--------------------|------------------|
| Below 0.33         | 1                |
| 0.33 – 0.66        | 2                |
| Above 0.66         | 3                |

### Earliest Firing Step

Calculated as: `max(2, int((1.0 - difficulty_level) * (episode_length / 2.0)))`

- At difficulty 0.0 → earliest step is half the episode length (step 6 for a 12-step episode).
- At difficulty 1.0 → earliest step is 2.

### Step Sampling

Firing steps are randomly selected from the valid range `[earliest, episode_length - 2]` with the constraint that no two selected steps are within 2 steps of each other. If the pool is exhausted before reaching the required drift count, the scheduler stops early.

For each firing step, `registry.sample_policy()` is called to generate the new policy. Each `DriftEvent` stores `fires_at_step` and `new_policy`.

---

## 4. Attacker System

The Attacker pipeline is a four-stage system that produces novel, policy-aware adversarial tickets without any LLM calls.

### Stage 1: Variation Pools (`variation_pools.py`)

Pure data file containing:
- **PRODUCT_REFERENCES** — 15 product/service descriptions spanning physical goods, software, and service plans.
- **DOLLAR_AMOUNTS** — Keyed by priority tier (Low, High, Critical), each with 10 calibrated dollar phrases.
- **DATE_PHRASINGS** — 10 natural time references for non-boundary tickets.
- **OPENING_STYLES** — 6 email openers from formal to frustrated.
- **CLOSING_STYLES** — 6 closers from polite to demanding.
- **TONE_MODIFIERS** — Keyed by deception strategy, 5 phrases each for priority_camouflage, fake_urgency, emotional_manipulation, category_confusion, and schema_exploitation. Empty for boundary_exploitation.

### Stage 2: Seed Bank (`seed_bank.json`)

102 structural seed templates. Each seed contains:
- Ground truth metadata (`true_priority`, `true_category`, `strategy`, `difficulty_band`).
- `policy_constraints` object with four subfields: `requires_category_in_valid_set`, `boundary_exploit_window`, `requires_empathy_trigger`, `requires_escalation_threshold`.
- `structure` object defining dynamic slots: `complaint_core`, `product_slot`, `amount_slot`, `date_slot`, `tone_slot`, `opening_slot`, `closing_slot`.

Seeds are not literal ticket strings. They are structural blueprints that the generation engine fills at runtime.

### Stage 3: Generation Engine (`generation_engine.py`)

The `GenerationEngine.generate(seed, active_policy, session_history)` method:
1. Checks strict constraints (empathy trigger requires non-None threshold; returns None otherwise).
2. Assembles components in order: opening → product-contextualized complaint core with tone injection → date reference → amount reference → closing.
3. For `date_slot == "CALCULATED"`: computes `active_policy.refund_window_days + 1` to place the ticket exactly outside the refund window.
4. Deprioritizes recently used openings, products, and closings by parsing the last 5 session history fingerprints.
5. Appends a fingerprint (`seed_id|opening_idx|product_idx|closing_idx`) to session history.
6. Returns the assembled ticket string, ground truth metadata, active policy version ID, boundary day count (if applicable), and committed sentiment score (if empathy-triggered).

### Stage 4: Template Sampler (`template_sampler.py`)

The `TemplateSampler.sample(blueprint, difficulty_level, active_policy)` method:
1. Runs the **compatibility filter** — a seed passes if all four conditions hold:
   - `true_category` is in `active_policy.valid_categories`
   - If `boundary_exploit_window` is not null, it is within ±2 of `active_policy.refund_window_days`
   - If `requires_empathy_trigger` is true, `active_policy.empathy_required_below_sentiment` is not None
   - If `requires_escalation_threshold` is not null, it matches `active_policy.escalation_threshold`
2. Filters further by priority, category, strategy, and difficulty band.
3. If no exact match: relaxes difficulty band first, then strategy. Never relaxes category or boundary window.
4. If all relaxations fail, returns None and logs a warning.

---

## 5. Reward System

All reward logic is 100% deterministic Python. No LLM calls. Two reward functions exist.

### Defender Reward Components

#### Priority Score (max +2.0)
- Exact match: +2.0 (post-drift) or +1.0 (normal).
- Off by 1 level: -0.5.
- Off by 2+ levels: -1.5.
- Missed Critical specifically: additional -2.0 and triggers an SLA breach.

#### Category Score (max +1.5, Task 2+)
- Exact match: +1.5 (post-drift) or +0.8 (normal).
- Mismatch: -0.8.
- Classifying as Technical when true is Security and Security is in `active_policy.valid_categories`: -1.5 (terminology drift penalty).

#### Response Quality Score (max +2.0, Task 2+)
Six-item checklist scored at +0.33 each:
- `has_greeting` — the `active_policy.required_greeting` string appears in the response (if required).
- `greeting_required_and_present` — greeting is present when policy mandates one.
- `references_subject_keyword` — response contains keywords from the ticket subject.
- `uses_resolution_language` — contains words like "investigate", "resolve", "fix".
- `professional_tone` — does not contain unprofessional markers ("idk", "lol", etc.).
- `appropriate_length` — 30–200 words.

Missing a required greeting incurs an additional -1.0. Hostile markers ("not our problem", etc.) add +0.15 to `churn_risk`.

#### Escalation Score (max +1.5)
The escalation threshold is dynamically interpreted from `active_policy.escalation_threshold`:
- `"Critical"` → only Critical tickets need escalation.
- `"High+"` → Critical and High need escalation.
- `"All"` → all tickets need escalation.

Correct escalation decision: +1.5. Missed required escalation: -2.0 + SLA breach. Over-escalation: -0.5 + escalation queue increment.

#### Refund Score (Billing only, max +1.0)
Evaluated against both `active_policy.refund_window_days` and `active_policy.refund_approval_authority`:
- If authority is `"deny_all"` and agent approves: -2.0, stale decision, $500 balance loss.
- If authority is `"manager_required"` and agent approves: -2.0, stale decision, $500 balance loss.
- If authority is `"auto"`, approved within window: +1.0.
- Approved outside window: -1.5 + $500 balance loss.
- Denied within window: -1.0 + churn risk increase.

#### Stale Decision Penalty (Task 2+)
If `active_policy.empathy_required_below_sentiment` is set and the ticket's sentiment falls below that threshold, the response must contain empathy language. Missing it: -1.0 and a stale decision record.

#### Drift Compliance Bonus (post-drift only)
- Correct priority + category on first post-drift step: +1.0.
- Incorrect: -2.5 + stale decision record.
- Per-field bonuses of +0.5 each when the agent correctly applies changed fields (greeting, empathy threshold, escalation threshold, refund window, refund authority, valid categories).
- Proactive clarification: +1.5.

#### Schema Bonus (Task 2+, when empathy threshold exists)
- Low sentiment + empathy language present: +1.0.
- Low sentiment + Critical ticket + no empathy: -0.5.
- High sentiment + positive language: +1.0.

#### Hallucination Penalty
Regex scans the draft response for claimed policy values (e.g., "30-day refund window", "SLA 4 hours"). Each claim is validated against the currently active policy's `refund_window_days`, `sla_critical_hours`, and `sla_high_hours`. Each unmatched claim: **-3.0** per hallucination.

### Attacker Reward
- Defender priority mismatch: +2.0 (match: -1.0).
- Defender category mismatch: +1.5 (match: -0.8).
- Missed Critical escalation: +3.0.
- Schema violation on the ticket: -2.0 (penalizes low-quality exploits).
- Successful boundary exploitation (defender made wrong refund call): +1.0.

### Curriculum Controller
After every step, `WorldState.run_curriculum_step()` adjusts `difficulty_level` based on `attacker_win_rate_50` (rolling 50-game deque, persists across episodes):
- Win rate > 0.75: difficulty -= 0.10 (ease up on defender).
- Win rate 0.60–0.75: hold steady.
- Win rate 0.40–0.60: difficulty += 0.05.
- Win rate < 0.40: difficulty += 0.15 (ramp up).

---

## 6. Episode Structure

An episode consists of exactly **12 tickets** processed sequentially.

### Reset Phase
1. `WorldState.reset_episode()` resets all episode-level counters while preserving the cross-episode attacker win rate deque and difficulty level.
2. `PolicyRegistry.sample_policy("dyn-init")` generates the starting policy.
3. `DriftScheduler.schedule_episode(difficulty_level)` pre-resolves all drift events.
4. `TemplateSampler` is initialized by loading `seed_bank.json` and creating a `GenerationEngine`.
5. 12 tickets are generated from `TICKET_BLUEPRINTS` filtered by `active_policy.valid_categories`.
6. If attacker is enabled, adversarial tickets replace the clean queue.

### Step Phase
1. Check if a pre-scheduled drift event fires at this step number.
2. If drift fires:
   - The new policy is pushed into registry history.
   - A human-readable system notice is generated listing exactly which fields changed.
   - `_reconcile_queue()` scans all unshown tickets and replaces any that are incompatible with the new policy.
3. The agent's action is scored against the current ticket using `calculate_defender_reward()`.
4. WorldState is mutated (balance, churn, SLA breaches, etc.).
5. `tickets_processed` advances. Episode ends when 12 tickets have been processed.

### Post-Drift Queue Reconciliation
When a drift fires, `_reconcile_queue(new_policy, old_policy, step)` iterates over all remaining tickets and applies the same four compatibility checks used by the `TemplateSampler`:
1. Category must be in `new_policy.valid_categories`.
2. Boundary exploit day count must be within ±2 of `new_policy.refund_window_days`.
3. Empathy-triggered tickets require a non-None empathy threshold.

Incompatible tickets are replaced via `template_sampler.sample()` with the same blueprint metadata. If no compatible seed exists, a clean procedural ticket is generated as fallback. The count of replaced tickets is recorded in the `drift_events` database table.

### Observation Payload
Each step, the agent receives:
- `ticket_id`, `subject`, `body`, `tier`
- `active_policy_version` — the version ID string
- `world_state_summary` — company balance, churn risk, escalation queue size, SLA breaches, tickets processed
- `sentiment_score` and `account_age_days` — only if `active_policy.empathy_required_below_sentiment` is not None
- `system_notice` — only on the step immediately after a drift event
- `conversation_history` — only during multi-turn (Task 3)

---

## 7. Telemetry

### Database Schema (SQLite via aiosqlite)

**episodes**
| Column                  | Type    | Notes                                    |
|-------------------------|---------|------------------------------------------|
| id                      | INTEGER | Primary key, auto-increment              |
| session_id              | TEXT    | UUID per session                         |
| task_id                 | INTEGER | 1, 2, or 3                               |
| attacker_enabled        | INTEGER | Boolean                                  |
| drift_enabled           | INTEGER | Boolean                                  |
| difficulty_init         | REAL    | Starting difficulty                      |
| created_at              | TEXT    | ISO timestamp                            |
| closed_at               | TEXT    | Set when episode ends                    |
| mean_defender_reward    | REAL    | Average reward across all steps          |
| mean_attacker_reward    | REAL    | Average attacker reward                  |
| final_balance           | REAL    | Company balance at episode end           |
| sla_breaches            | INTEGER | Total SLA breaches                       |
| drift_accuracy          | REAL    | Post-drift decision accuracy             |
| stale_decisions         | INTEGER | Count of stale decisions                 |
| hallucinations          | INTEGER | Count of hallucinated policy claims      |
| attacker_win_rate_final | REAL    | Rolling 50-game win rate at episode end  |
| difficulty_final        | REAL    | Difficulty after curriculum adjustment   |

**steps**
| Column                | Type    | Notes                                |
|-----------------------|---------|--------------------------------------|
| episode_id            | INTEGER | Foreign key to episodes              |
| step_number           | INTEGER | 1-indexed                            |
| ticket_id             | TEXT    | The ticket processed                 |
| action_json           | TEXT    | Full agent action                    |
| defender_reward       | REAL    | Reward for this step                 |
| attacker_reward       | REAL    | Attacker reward for this step        |
| reward_breakdown_json | TEXT    | Full component breakdown             |
| policy_version_at_step| TEXT    | Active policy version ID             |
| was_post_drift        | INTEGER | Boolean                              |
| deception_strategy    | TEXT    | Strategy of the ticket               |

**drift_events**
| Column            | Type    | Notes                                      |
|-------------------|---------|--------------------------------------------|
| episode_id        | INTEGER | Foreign key to episodes                    |
| step_number       | INTEGER | Step at which drift fired                  |
| from_version      | TEXT    | Previous policy version ID                 |
| to_version        | TEXT    | New policy version ID                      |
| drift_types_json  | TEXT    | JSON array of drift type labels            |
| agent_noticed     | INTEGER | Whether agent acknowledged the drift       |
| tickets_replaced  | INTEGER | Count of tickets replaced during reconciliation |

**world_state_snapshots**
| Column                 | Type    | Notes                            |
|------------------------|---------|----------------------------------|
| episode_id             | INTEGER | Foreign key to episodes          |
| step_number            | INTEGER | Step number                      |
| snapshot_json          | TEXT    | Full WorldState serialization    |
| current_policy_version | TEXT    | Active version at snapshot time  |
| drift_events_fired     | INTEGER | Cumulative drift count           |
| difficulty_level       | REAL    | Difficulty at snapshot time      |

**tickets_log**
| Column                 | Type    | Notes                            |
|------------------------|---------|----------------------------------|
| episode_id             | INTEGER | Foreign key to episodes          |
| step_number            | INTEGER | Step number                      |
| ticket_json            | TEXT    | Full ticket serialization        |
| true_priority          | TEXT    | Ground truth priority            |
| true_category          | TEXT    | Ground truth category            |
| deception_strategy     | TEXT    | Strategy label                   |
| difficulty_level_at_gen| REAL    | Difficulty when ticket generated |
| attacker_confidence    | REAL    | Attacker's confidence score      |

### Derived Metrics

- **drift_adaptation_lag**: Number of steps between a drift event firing and the first post-drift step where the agent achieves a positive drift compliance score. Computed post-hoc from the `steps` table by comparing `was_post_drift` flags against `drift_compliance_score` in `reward_breakdown_json`.

---

## 8. Training Setup

> **Stub**: Training configuration, model selection, hyperparameters, and reward curve targets will be documented here after the first successful training run against the environment.
