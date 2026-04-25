# Implementation Plan: The Gauntlet + Shifting Sands
### OpenEnv Hackathon India 2026 — Coding Agent Instructions

---

## Overview

Build an OpenEnv-compliant reinforcement learning environment called **The Gauntlet + Shifting Sands**. It is a customer support helpdesk simulation with two synergistic modules:

- **The Gauntlet**: An Attacker LLM generates adversarially deceptive support tickets. A Defender LLM must classify and respond correctly. Both agents improve through competition.
- **Shifting Sands**: Company policies (refund windows, SLA thresholds, ticket categories) drift mid-session via structured system notices. The Defender must detect drift and apply new policies immediately.

**Themes covered**: Theme #4 Self-Improvement (primary) + Theme #3.2 World Modeling (secondary).

**Stack**: Python 3.11, FastAPI, SQLite (local) / Postgres (prod), Groq API (LLM calls), OpenEnv base classes.

---

## File Structure to Create

```
gauntlet_shifting_sands/
├── environment.py          # Core env + procedural ticket engine
├── attacker.py             # Attacker agent logic
├── policy.py               # Policy registry (v1/v2/v3)
├── drift_scheduler.py      # Drift event injector
├── world_state.py          # WorldState dataclass + mutation logic
├── rewards.py              # Deterministic reward calculator
├── main.py                 # FastAPI server (OpenEnv-compliant)
├── db.py                   # SQLite/Postgres ORM layer
├── inference.py            # Local test runner (bypasses API)
├── train.ipynb             # Colab training notebook (HF TRL / Unsloth)
├── openenv.yaml            # OpenEnv manifest
├── requirements.txt
└── README.md
```

---

## Phase 1 — Policy Registry (`policy.py`)

Build this first. Everything else reads from it.

### 1.1 — Define the PolicyVersion dataclass

```python
@dataclass
class PolicyVersion:
    version_id: str
    refund_window_days: int
    sla_critical_hours: int
    sla_high_hours: int
    valid_categories: list[str]
    auto_escalate_threshold_per_min: int
    response_greeting_required: bool
    ticket_schema_fields: list[str]
```

### 1.2 — Instantiate three named versions

| Field | v1 (Baseline) | v2 (Tightened) | v3 (Expanded) |
|---|---|---|---|
| refund_window_days | 30 | 14 | 14 |
| sla_critical_hours | 4 | 2 | 2 |
| sla_high_hours | 8 | 4 | 4 |
| valid_categories | ["Billing","Technical","Shipping"] | same as v1 | ["Billing","Technical","Shipping","Security"] |
| auto_escalate_threshold_per_min | 500 | 250 | 250 |
| response_greeting_required | False | True | True |
| ticket_schema_fields | ["subject","body","tier"] | same as v1 | ["subject","body","tier","sentiment_score","account_age_days"] |

### 1.3 — PolicyRegistry class

```python
class PolicyRegistry:
    def __init__(self):
        self._versions = {"v1": V1, "v2": V2, "v3": V3}
        self._active = "v1"

    def get_active(self) -> PolicyVersion
    def get_version(self, vid: str) -> PolicyVersion
    def set_active(self, vid: str) -> None
    def list_all_versions(self) -> dict   # used by hallucination checker
```

The `list_all_versions` method returns the full dict of all registered policies. The reward calculator uses it to verify whether a rule cited by the agent actually exists in any version.

---

## Phase 2 — WorldState (`world_state.py`)

### 2.1 — WorldState dataclass (all fields, all types, all initial values)

```python
@dataclass
class WorldState:
    company_balance: float = 10_000.0
    churn_risk: float = 0.0             # clamp 0.0–1.0
    escalation_queue_size: int = 0
    sla_breaches: int = 0
    current_policy_version: str = "v1"
    drift_events_fired: int = 0
    agent_drift_accuracy: float = 0.0
    stale_decisions_made: int = 0
    hallucinations_caught: int = 0
    attacker_win_rate_50: float = 0.5   # rolling window, cross-episode
    difficulty_level: float = 0.3
    tickets_processed: int = 0
    multi_turn_active: bool = False

    # Internal tracking (not exported to agent)
    _post_drift_decisions_correct: int = 0
    _post_drift_decisions_total: int = 0
    _recent_attacker_results: deque = field(default_factory=lambda: deque(maxlen=50))
```

**Note**: `attacker_win_rate_50` uses a `deque(maxlen=50)` that persists across episodes in the session. It is NOT reset on `/reset` — it carries over to give the curriculum controller a meaningful signal. Document this clearly in the code.

### 2.2 — Mutation methods (one method per mutation type)

```python
def apply_wrong_refund(self, amount: float) -> None
    # Decrement company_balance by amount. Floor at 0.

def apply_churn_delta(self, delta: float) -> None
    # Add delta to churn_risk. Clamp to [0.0, 1.0].

def apply_escalation(self) -> None
    # Increment escalation_queue_size. If > 5, call trigger_sla_breach().

def close_escalation(self) -> None
    # Decrement escalation_queue_size. Floor at 0.

def trigger_sla_breach(self) -> None
    # Increment sla_breaches.

def record_drift_event(self, new_version: str) -> None
    # Set current_policy_version. Increment drift_events_fired.

def record_post_drift_decision(self, correct: bool) -> None
    # Update _post_drift_decisions_correct, _total.
    # Recompute agent_drift_accuracy = correct / total.

def record_stale_decision(self) -> None
    # Increment stale_decisions_made.

def record_hallucination(self) -> None
    # Increment hallucinations_caught.

def record_attacker_result(self, attacker_won: bool) -> None
    # Append to _recent_attacker_results deque.
    # Recompute attacker_win_rate_50 = sum / len.

def to_export_dict(self) -> dict
    # Return all public fields as a dict (exclude _ prefixed fields).
    # Used for API responses and DB snapshots.
```

### 2.3 — Curriculum controller (inside WorldState)

```python
def run_curriculum_step(self) -> None:
    """
    Called after every step. Adjusts difficulty_level based on attacker_win_rate_50.
    Only activates after the deque has >= 10 entries (cold start protection).
    """
    rate = self.attacker_win_rate_50
    if len(self._recent_attacker_results) < 10:
        return
    if rate > 0.75:
        self.difficulty_level = max(0.0, self.difficulty_level - 0.10)
    elif 0.60 <= rate <= 0.75:
        pass  # hold steady
    elif 0.40 <= rate < 0.60:
        self.difficulty_level = min(1.0, self.difficulty_level + 0.05)
    else:  # < 0.40
        self.difficulty_level = min(1.0, self.difficulty_level + 0.15)
```

---

## Phase 3 — Ticket Engine + Attacker (`environment.py` + `attacker.py`)

### 3.1 — Blueprint structure (in `environment.py`)

Each blueprint is a dict:

```python
{
    "template": "Our {tier} integration has been {symptom} for {duration}. We are losing ${loss} per minute.",
    "true_priority": "Critical",
    "true_category": "Technical",
    "base_requires_escalation": True,
    "refund_eligible_boundary": False,   # True = ticket sits near refund window edge
}
```

Create a minimum of **30 blueprints** covering all combinations of:
- Priority × Category matrix: 4 priorities × 4 categories (Billing, Technical, Shipping, Security) = 16 base combinations, multiple templates per slot
- At least 3 "boundary" blueprints where the refund date is exactly at the policy window edge (used by Attacker for Boundary Exploitation strategy)

Variable pools per placeholder — minimum entries:
- `{tier}`: 10 values (Free, Pro, Enterprise, Starter, Growth, etc.)
- `{symptom}`: 15 values (down, unresponsive, erroring, throttled, etc.)
- `{duration}`: 12 values (2 hours, since this morning, 3 days, etc.)
- `{loss}`: 8 values (50, 100, 250, 500, 1000, 5000, etc.)
- `{product}`: 10 values
- `{action}`: 10 values
- `{days_since_purchase}`: values that straddle refund window boundaries (e.g. 13, 14, 15, 29, 30, 31)

### 3.2 — Procedural generation (clean mode)

```python
def generate_ticket_clean(blueprint: dict, policy: PolicyVersion) -> dict:
    """
    Substitutes variables randomly. Returns full ticket dict.
    For v3 schema fields, generates sentiment_score (random float 0.0–1.0)
    and account_age_days (random int 1–2000).
    Only includes v3 fields if policy.ticket_schema_fields contains them.
    """
```

### 3.3 — Attacker agent (`attacker.py`)

**Timing model**: The Attacker runs UPFRONT at `/reset` time, not in real-time per step. At `/reset`, the server calls `AttackerAgent.generate_batch(n=12, ...)` which returns 12 pre-generated adversarial tickets. These are stored in the session and served one-by-one via `/step`. This avoids latency during the step loop.

```python
class AttackerAgent:
    def __init__(self, llm_client, policy_registry: PolicyRegistry):
        ...

    def generate_batch(
        self,
        n: int,
        difficulty_level: float,
        defender_error_history: list[dict],   # last 10 Defender decisions from DB, or [] if new session
        active_policy: PolicyVersion,
    ) -> list[dict]:
        """
        Generates n adversarial tickets. Returns list of ticket dicts.
        Each ticket includes a 'deception_strategy' field for logging.
        Strategy selection logic:
          - difficulty < 0.3: randomly pick from [Priority Camouflage, Fake Urgency]
          - 0.3–0.6: all 6 strategies available, weighted by defender_error_history
          - > 0.6: prefer strategies that exploited policy boundaries or suppressed urgency
        """
```

**Strategy selection**: Parse `defender_error_history` to find which ticket categories the Defender got wrong recently. Weight the strategy pool so those strategies are picked more often. If error history is empty (new session), pick uniformly at random.

**Prompt template per strategy** — construct a different system prompt for each of the 6 strategies:

```
STRATEGY: Priority Camouflage
System: You are generating a customer support ticket. The TRUE priority is {true_priority} 
and the TRUE category is {true_category}. Your goal is to write a ticket that HIDES this 
priority using casual, low-urgency language while the content is genuinely critical. 
Use relaxed tone, lowercase, friendly punctuation. Do NOT use urgency markers.
The ticket must still be factually consistent with a {true_category} issue.
Output JSON only: {"subject": "...", "body": "...", "tier": "..."}
```

Write equivalent prompts for all 6 strategies. Include `active_policy` context so the Attacker can place boundary-exploitation tickets accurately.

**Schema validation**: After receiving the LLM response, validate the output matches the current policy's `ticket_schema_fields`. If it fails schema validation, fall back to a clean procedural ticket and assign `deception_strategy = "fallback_procedural"`. Log the failure. This prevents the Attacker from getting stuck.

**Attacker reward logging**: The Attacker reward is calculated RETROACTIVELY after the Defender acts. In `rewards.py`, compute `attacker_reward` alongside `defender_reward` and log both to the `steps` table.

---

## Phase 4 — Drift Scheduler (`drift_scheduler.py`)

### 4.1 — DriftEvent dataclass

```python
@dataclass
class DriftEvent:
    fires_at_step: int
    from_version: str
    to_version: str
    drift_types: list[str]   # ["policy_drift", "schema_drift", "terminology_drift"]
    notice_text: str          # Pre-written system notice string
```

### 4.2 — Default schedule

```python
DEFAULT_DRIFT_SCHEDULE = [
    DriftEvent(
        fires_at_step=4,
        from_version="v1",
        to_version="v2",
        drift_types=["policy_drift"],
        notice_text="""=== SYSTEM NOTICE [Step 4] ===
Effective immediately, the following policy changes are in effect:
- Refund eligibility window: 30 days → 14 days
- Critical SLA threshold: 4 hours → 2 hours
- High SLA threshold: 8 hours → 4 hours
- Auto-escalation threshold: $500/min → $250/min
- Customer responses must now begin with a formal greeting
Please update all future decisions accordingly.
=== END NOTICE ===""",
    ),
    DriftEvent(
        fires_at_step=9,
        from_version="v2",
        to_version="v3",
        drift_types=["policy_drift", "schema_drift", "terminology_drift"],
        notice_text="""=== SYSTEM NOTICE [Step 9] ===
Effective immediately, the following changes are in effect:
- New ticket category added: Security (previously classified as Technical)
- Any ticket involving authentication, access control, or data breach must now use category 'Security'
- Two new ticket fields are now available: sentiment_score (0.0–1.0) and account_age_days
- These fields should inform your escalation urgency and response tone
Please update all future decisions accordingly.
=== END NOTICE ===""",
    ),
]
```

### 4.3 — DriftScheduler class

```python
class DriftScheduler:
    def __init__(self, schedule: list[DriftEvent] = DEFAULT_DRIFT_SCHEDULE):
        self._schedule = {e.fires_at_step: e for e in schedule}

    def check_step(self, step_number: int) -> DriftEvent | None:
        """Returns the DriftEvent if one fires at this step, else None."""

    def apply(self, event: DriftEvent, world_state: WorldState, policy_registry: PolicyRegistry) -> None:
        """
        1. Call policy_registry.set_active(event.to_version)
        2. Call world_state.record_drift_event(event.to_version)
        3. Log the event to the drift_events table via db.py
        """
```

---

## Phase 5 — Reward Calculator (`rewards.py`)

**All reward logic is deterministic Python. No LLM calls.**

### 5.1 — Defender reward function

```python
def calculate_defender_reward(
    action: dict,
    ticket: dict,
    active_policy: PolicyVersion,
    world_state: WorldState,
    was_post_drift: bool,
    task_id: int,
) -> tuple[float, dict]:
    """
    Returns (total_reward, reward_breakdown_dict).
    reward_breakdown_dict has keys for each sub-component for logging.
    """
```

**Sub-components**:

**Priority score** (max +2.0):
- Correct: +2.0 if `was_post_drift` else +1.0
- Off by one level: -0.5
- Off by two+ levels: -1.5
- Missed Critical (true=Critical, assigned!=Critical): additional -2.0 + trigger `world_state.trigger_sla_breach()`

**Category score** (max +1.5, Task 2+):
- Correct: +1.5 if `was_post_drift` else +0.8
- Wrong: -0.8
- Special case: classified as Technical when true=Security AND step >= 9 (post-v3): -1.5 (terminology drift penalty)

**Response quality score** (max +2.0, Task 2+):

Use a deterministic checklist. Score each item as +0.33 or 0:

```python
checklist = {
    "has_greeting": bool(re.search(r'\b(hello|hi|dear|good morning|good afternoon)\b', response, re.I)),
    "greeting_required_and_present": (not policy.response_greeting_required) or has_greeting,
    "references_subject_keyword": any(kw.lower() in response.lower() for kw in extract_keywords(ticket["subject"])),
    "uses_resolution_language": bool(re.search(r'\b(investigate|resolve|fix|look into|escalate|prioritise|address)\b', response, re.I)),
    "professional_tone": not bool(re.search(r'\b(idk|lol|whatever|cant be bothered)\b', response, re.I)),
    "appropriate_length": 30 <= len(response.split()) <= 200,
}
```

Missing required greeting when `policy.response_greeting_required=True`: -1.0

Hostile response markers (keywords like "not our problem", "your fault"): `world_state.apply_churn_delta(+0.15)`

**Escalation score** (max +1.5):
- Correct escalation decision: +1.5
- Should have escalated but didn't (missed Critical): -2.0 + `world_state.trigger_sla_breach()`
- Over-escalation (escalated Low/Medium unnecessarily): -0.5 + `world_state.apply_escalation()`

**Refund score** (Billing tickets only, max +1.0):
- Approved refund within `policy.refund_window_days`: +1.0
- Approved refund outside window: -1.5 + `world_state.apply_wrong_refund(500.0)`
- Denied refund within window: -1.0 + `world_state.apply_churn_delta(+0.10)`

**Drift compliance bonus** (post-drift steps only):
- Correct decision using new policy: +1.0 bonus (applied on top of base scores)
- Applied old policy after drift was broadcast: -2.5 + `world_state.record_stale_decision()`
- Proactively asked for clarification before deciding (ask_clarification=True on drift step): +1.5

**Schema bonus** (v3 tickets only, Task 2+):
- Correctly referenced `sentiment_score` in `draft_response` reasoning: +1.0 bonus

  Detection: check if response contains sentiment-aware language when `sentiment_score < 0.3` (e.g., "I understand your frustration", "I sincerely apologise") OR when `sentiment_score > 0.7` (positive acknowledgment language).

- Ignored `sentiment_score < 0.3` on a Critical ticket post-v3: -0.5

**Hallucination detection**:

After scoring, scan `draft_response` and any `clarification_text` for policy-rule claims.

```python
POLICY_RULE_PATTERNS = [
    r'(\d+)[- ]day[s]? (?:refund|return|window)',
    r'(?:sla|response)[^\d]*(\d+)[- ]hour',
    r'\$(\d+)[^\d]+(?:per|/)[^\d]+min',
]
```

For each match, extract the claimed value. Check if it matches any field in `policy_registry.list_all_versions()`. If the claimed value does not appear in any policy version for that field type: +1 hallucination, `world_state.record_hallucination()`, apply -3.0 penalty.

**Update WorldState after scoring**:
- If `was_post_drift`: call `world_state.record_post_drift_decision(correct=priority_correct and category_correct)`
- Always: call `world_state.record_attacker_result(attacker_won=(total_defender_reward < 0))`
- Always: call `world_state.run_curriculum_step()`

### 5.2 — Attacker reward function

```python
def calculate_attacker_reward(defender_action: dict, ticket: dict) -> float:
    reward = 0.0
    if defender_action["assign_priority"] != ticket["true_priority"]:
        reward += 2.0
    if defender_action.get("assign_category") != ticket["true_category"]:
        reward += 1.5
    if ticket["true_priority"] == "Critical" and not defender_action.get("escalate", False):
        reward += 3.0
    if defender_action["assign_priority"] == ticket["true_priority"]:
        reward -= 1.0
    if defender_action.get("assign_category") == ticket["true_category"]:
        reward -= 0.8
    if ticket.get("schema_violation", False):
        reward -= 2.0
    if ticket.get("deception_strategy") == "Boundary Exploitation" and defender_action.get("approve_refund") is not None:
        if defender_action["approve_refund"] != ticket["true_refund_eligible"]:
            reward += 1.0  # boundary correctly exploited
    return reward
```

---

## Phase 6 — Database Layer (`db.py`)

Use SQLAlchemy Core (not ORM) for portability between SQLite and Postgres. Connection string from environment variable `DATABASE_URL` — default to `sqlite:///gauntlet.db`.

### 6.1 — Table definitions

**episodes**
```
id (PK), session_id (UUID), task_id (int), attacker_enabled (bool),
drift_enabled (bool), difficulty_init (float), created_at, closed_at,
mean_defender_reward (float), mean_attacker_reward (float),
final_balance (float), sla_breaches (int), drift_accuracy (float),
stale_decisions (int), hallucinations (int), attacker_win_rate_final (float),
difficulty_final (float)
```

**steps**
```
id (PK), episode_id (FK), step_number (int), ticket_id (str),
action_json (text), defender_reward (float), attacker_reward (float),
reward_breakdown_json (text), policy_version_at_step (str),
was_post_drift (bool), deception_strategy (str), created_at
```

**world_state_snapshots**
```
id (PK), episode_id (FK), step_number (int), snapshot_json (text),
current_policy_version (str), drift_events_fired (int),
difficulty_level (float), created_at
```

**tickets_log**
```
id (PK), episode_id (FK), step_number (int), ticket_json (text),
true_priority (str), true_category (str), deception_strategy (str),
difficulty_level_at_gen (float), attacker_confidence (float), created_at
```

**drift_events**
```
id (PK), episode_id (FK), step_number (int), from_version (str),
to_version (str), drift_types_json (text), agent_noticed (bool), created_at
```

`agent_noticed` is True if the Defender's action on the drift step correctly applied the new policy (i.e. `was_post_drift=True` AND `priority_correct=True`).

### 6.2 — db.py functions (one per operation)

```python
def create_episode(session_id, task_id, attacker_enabled, drift_enabled, difficulty_init) -> int
def close_episode(episode_id, metrics: dict) -> None
def insert_step(episode_id, step_number, ticket_id, action, defender_reward, attacker_reward, breakdown, policy_version, was_post_drift, deception_strategy) -> None
def insert_snapshot(episode_id, step_number, world_state: WorldState) -> None
def insert_ticket_log(episode_id, step_number, ticket: dict, difficulty_level: float) -> None
def insert_drift_event(episode_id, step_number, event: DriftEvent, agent_noticed: bool) -> None
def get_episodes(limit=50, offset=0, closed_only=True) -> list[dict]
def get_episode_detail(episode_id: int) -> dict
```

---

## Phase 7 — Core Environment (`environment.py`)

### 7.1 — CustomerSupportEnv class

Inherit from OpenEnv base class. Implement standard Gym-style API.

```python
class CustomerSupportEnv:
    def reset(self, task_id: int, attacker_enabled: bool, drift_enabled: bool, difficulty_init: float = 0.3) -> dict:
        """
        1. Create new session_id (UUID4).
        2. Initialise fresh WorldState (but preserve attacker_win_rate_50 deque from previous session if exists).
        3. If attacker_enabled: call AttackerAgent.generate_batch(n=12, ...) to get 12 adversarial tickets upfront.
           Else: generate 12 clean procedural tickets.
        4. Store ticket queue internally.
        5. Load Policy v1 into PolicyRegistry.
        6. Create episode row in DB.
        7. Return first observation dict.
        """

    def step(self, action: dict) -> dict:
        """
        1. Get current ticket from queue.
        2. Check DriftScheduler.check_step(current_step) — if drift fires:
           a. Apply drift (update policy + world_state).
           b. Set was_post_drift = True for this step.
           c. Add system_notice to outgoing observation.
           d. Insert drift_event row in DB.
        3. Calculate defender_reward and attacker_reward.
        4. Mutate WorldState via reward side-effects.
        5. Log step, snapshot, ticket to DB.
        6. Increment tickets_processed.
        7. If tickets_processed == 12: set done=True, close_episode in DB.
        8. Else: advance to next ticket.
        9. Return observation dict.
        """

    def _build_observation(self, ticket: dict, drift_notice: str | None) -> dict:
        """
        Assembles the observation the Defender sees.
        Always includes: ticket fields for active schema, active_policy_version, world_state_summary.
        Only includes system_notice if drift_notice is not None.
        Only includes conversation_history if world_state.multi_turn_active is True.
        """
```

### 7.2 — Observation schema enforcement

When building the observation, filter ticket fields to `active_policy.ticket_schema_fields`. Do NOT include `sentiment_score` or `account_age_days` in observations when `current_policy_version` is v1 or v2. This is critical — the schema drift must be real, not cosmetic.

---

## Phase 8 — FastAPI Server (`main.py`)

### 8.1 — Session management

Use an in-memory dict `SESSIONS: dict[str, CustomerSupportEnv]` to hold active session objects. Sessions are keyed by `session_id`. Clean up sessions after `done=True`.

### 8.2 — Endpoints

**POST /reset**
```
Request body: { task_id: int, attacker_enabled: bool, drift_enabled: bool, difficulty_init: float }
Response: { session_id: str, observation: dict, world_state: dict, policy_version: str }
```

**POST /step**
```
Request body: { session_id: str, action: dict }
Response: { reward: float, observation: dict | null, world_state: dict, done: bool, drift_notice: str | null }
```

**GET /episodes**
```
Query params: limit (default 50), offset (default 0), closed_only (default true)
Response: list of episode summary dicts
```

**GET /episodes/{episode_id}**
```
Response: full episode dict including steps array and snapshots array
```

**GET /world_state/{session_id}**
```
Response: current WorldState export dict for an active session
```

**GET /health**
```
Response: { status: "ok", openenv_version: str, active_sessions: int }
```

### 8.3 — Error handling

- Invalid `session_id` on `/step`: 404 with message "Session not found or already closed"
- Action missing required field `assign_priority`: 422 with validation detail
- Step called on a done session: 400 with message "Episode already closed"

---

## Phase 9 — Task Definitions

Three task modes, selected via `task_id` in `/reset`:

**Task 1 (task_id=1) — Priority only**
- Required action fields: `assign_priority`
- Scored: priority only
- Tickets: obvious, no deception (force `difficulty_level` cap at 0.2 regardless of curriculum)
- Drift: disabled (ignore `drift_enabled` flag)

**Task 2 (task_id=2) — Full classification + response**
- Required action fields: `assign_priority`, `assign_category`, `draft_response`
- Scored: all components
- Tickets: some ambiguity
- Drift: enabled if flag set

**Task 3 (task_id=3) — Multi-turn + full**
- Required action fields: all Task 2 fields + optional `ask_clarification`, `clarification_text`
- Scored: all components + clarification quality
- Clarification quality score:
  - `ask_clarification=True` on a genuinely ambiguous ticket (marked in blueprint): +1.0
  - `ask_clarification=True` on an obvious ticket: -0.5 (unnecessary delay)
  - `clarification_text` is specific and relevant (contains a question mark + references ticket content): +0.5
- Drift: enabled if flag set
- Multi-turn: when `ask_clarification=True`, set `world_state.multi_turn_active=True`. Next `/step` must include the customer reply simulation before scoring the Defender's final decision.

**Customer reply simulation** (inside `environment.py`, no LLM needed):

```python
CLARIFICATION_REPLIES = {
    "billing": [
        "I purchased on {date}. My account is {tier}.",
        "The charge was for {product}. I did not authorise it.",
    ],
    "technical": [...],
    ...
}
```

Pick a reply template matching the ticket category. Fill variables from the original ticket. Return as `conversation_history` in the next observation.

---

## Phase 10 — Training Script (`train.ipynb`)

### 10.1 — Structure

The notebook must be Colab-compatible and runnable against the live `/step` endpoint. Use HuggingFace TRL's `PPOTrainer` or `GRPOTrainer` (GRPO preferred — simpler setup for single-model RL).

### 10.2 — Training loop

```python
for episode in range(NUM_EPISODES):
    obs = requests.post(f"{API_URL}/reset", json={
        "task_id": 2,
        "attacker_enabled": True,
        "drift_enabled": True,
        "difficulty_init": 0.3
    }).json()

    while True:
        # Format observation as prompt
        prompt = format_observation_as_prompt(obs["observation"])

        # Get model action
        output = model.generate(prompt, ...)
        action = parse_action_from_output(output)

        # Step environment
        result = requests.post(f"{API_URL}/step", json={
            "session_id": obs["session_id"],
            "action": action
        }).json()

        # Collect (prompt, output, reward) for trainer
        experience_buffer.append((prompt, output, result["reward"]))

        if result["done"]:
            break
        obs = result

    # Train on buffer
    trainer.step(experience_buffer)
```

### 10.3 — Plots to generate and save (commit all as .png to repo)

1. **Defender overall reward** — rolling mean over 50 episodes, x-axis = training step
2. **Drift accuracy** — `agent_drift_accuracy` rolling mean, same x-axis, same plot as (1) for comparison
3. **Stale decision rate** — `stale_decisions_made` per 200-step interval, bar chart
4. **Attacker win rate** — `attacker_win_rate_50` over time, should converge toward 0.5
5. **Difficulty level** — `difficulty_level` over time, should trend upward as Defender improves
6. **Company balance timeline** — `company_balance` per step for one representative trained episode vs one untrained episode

Label both axes on every plot. Save with `plt.savefig("plots/plot_name.png", dpi=150, bbox_inches="tight")`.

---

## Phase 11 — OpenEnv Manifest (`openenv.yaml`)

```yaml
name: gauntlet-shifting-sands
version: "1.0"
description: >
  Adversarial self-play customer support environment with policy drift.
  An Attacker agent generates deceptive tickets; a Defender agent must
  classify them correctly under evolving company policies.
themes:
  - self-improvement
  - world-modeling
tasks:
  - id: 1
    name: priority-only
    description: Classify ticket priority (Low/Medium/High/Critical)
  - id: 2
    name: full-classification
    description: Classify priority, category, draft response, handle escalation
  - id: 3
    name: multi-turn
    description: Full classification with clarification requests and policy drift
observation_space:
  type: dict
  fields: [ticket_id, subject, body, customer_tier, active_policy_version,
           world_state_summary, system_notice, conversation_history,
           sentiment_score, account_age_days]
action_space:
  type: dict
  fields: [assign_priority, assign_category, draft_response, escalate,
           ask_clarification, clarification_text, approve_refund]
reward_range: [-10.0, 10.0]
episode_length: 12
reset_endpoint: /reset
step_endpoint: /step
```

---

## Phase 12 — README.md

Structure the README to answer exactly four questions (judges read in 3–5 minutes):

1. **Problem**: What capability gap does this train? (LLMs fail when policies change mid-session and when tickets are adversarially deceptive)
2. **Environment**: What does the agent see, do, and get rewarded for? (Include a one-paragraph description of Gauntlet + Shifting Sands modules)
3. **Results**: What changed after training? (Embed the 6 plots with one-line captions each)
4. **Why it matters**: Who would care? (Enterprise AI, policy-following, adversarial robustness)

Include the following links prominently:
- HuggingFace Space URL
- YouTube demo video URL (< 2 minutes)
- Training notebook (Colab link)

Do not embed large video files in the repo. Use URL references only.

---

## Build Order (strict sequence)

1. `policy.py` — no dependencies
2. `world_state.py` — depends on policy.py
3. `rewards.py` — depends on policy.py + world_state.py
4. `drift_scheduler.py` — depends on policy.py + world_state.py
5. `db.py` — no code dependencies, just schema
6. `environment.py` (ticket engine only, no Attacker) — depends on policy.py
7. `main.py` (clean mode only, attacker_enabled=False) — wire up and test end-to-end
8. `attacker.py` — depends on environment.py + policy.py
9. `environment.py` (add Attacker integration) — extend existing file
10. `main.py` (add Attacker support) — extend existing file
11. `inference.py` — local runner, depends on environment.py
12. `train.ipynb` — depends on running main.py
13. `openenv.yaml` + `README.md` — final

Test after step 7 (clean end-to-end) and after step 10 (full adversarial mode) before writing the training notebook.

---

## Key Constraints and Gotchas

- **No LLM grading anywhere in rewards.py** — 100% deterministic Python. If a reward sub-component cannot be computed without an LLM, simplify the rule.
- **attacker_win_rate_50 persists across episodes** — do not reset this on `/reset`. It must span episodes to have signal.
- **Schema fields in observations are policy-gated** — sentiment_score and account_age_days must not appear in observations until step 9. This is enforced in `_build_observation`, not in the ticket generator.
- **Attacker runs at /reset time, not during /step** — all 12 tickets generated upfront. This is a deliberate simplification to keep step latency low.
- **Hallucination detection is rule-based** — regex + policy value lookup only. No semantic matching.
- **Curriculum cold start** — `run_curriculum_step()` does nothing until the deque has >= 10 entries. This prevents wild difficulty swings in the first episode.
- **Remove all Patronus AI bonus references** — do not reference schema drift as a bonus prize category anywhere in code comments, README, or manifest.
