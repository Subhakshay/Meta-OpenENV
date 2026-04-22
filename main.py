from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

import uuid
import time
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:     %(name)s - %(message)s')

try:
    from environment import CustomerSupportEnv, Action, Priority, Category, TASK_CONFIG
except Exception as e:
    print(f"ERROR importing environment: {e}")
    raise

from db import (
    init_db,
    close_db,
    save_episode,
    close_episode,
    save_step,
    save_world_snapshot,
    save_ticket_log,
    get_episodes,
    get_episode_steps,
    get_episode_by_id,
)

# ---------------------------------------------------------------------------
# Lifespan — DB init/teardown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CustomerSupportEnv",
    version="1.0.0",
    description=(
        "An OpenEnv-compliant environment for training AI agents on customer support "
        "ticket triage: priority classification, category tagging, response drafting, "
        "and escalation decisions. Implements the standard reset/step/state interface."
    ),
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

_sessions:     Dict[str, CustomerSupportEnv] = {}
_session_meta: Dict[str, Dict[str, Any]]     = {}

VALID_TASKS = CustomerSupportEnv.TASK_IDS
SESSION_TTL = 3600  # seconds before a stale session is dropped

TASK_DESCRIPTIONS = {
    "task_1_priority": {
        "id":            "task_1_priority",
        "name":          "Priority Assignment",
        "difficulty":    "easy",
        "description":   "Assign the correct urgency priority (low/medium/high/critical) to each ticket.",
        "scored_fields": ["assign_priority"],
        "weights":       {"priority": 1.0},
        "max_steps":     10,
    },
    "task_2_classification": {
        "id":            "task_2_classification",
        "name":          "Ticket Classification",
        "difficulty":    "medium",
        "description":   "Assign correct priority AND category to each ticket.",
        "scored_fields": ["assign_priority", "assign_category"],
        "weights":       {"priority": 0.6, "category": 0.4},
        "max_steps":     10,
    },
    "task_3_full_triage": {
        "id":            "task_3_full_triage",
        "name":          "Full Ticket Triage",
        "difficulty":    "hard",
        "description":   "Full triage: priority, category, quality response, and escalation decision.",
        "scored_fields": ["assign_priority", "assign_category", "response_text", "escalate"],
        "weights":       {"priority": 0.35, "category": 0.25, "response": 0.40},
        "max_steps":     10,
    },
}

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task_1_priority"
    seed:    int = 42

class StepRequest(BaseModel):
    session_id:          str
    assign_priority:     str           = "medium"
    assign_category:     str           = "general"
    response_text:       str           = ""
    escalate:            bool          = False
    action_type:         Optional[str] = None   # "classify" or "ask"
    clarifying_question: Optional[str] = None   # used with action_type="ask"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cleanup_stale():
    now   = time.time()
    stale = [sid for sid, m in _session_meta.items() if now - m["created_at"] > SESSION_TTL]
    for sid in stale:
        _sessions.pop(sid, None)
        _session_meta.pop(sid, None)
    return len(stale)


def _get_session(session_id: str) -> CustomerSupportEnv:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, detail="Session not found. Call POST /reset first.")
    return env

# ---------------------------------------------------------------------------
# Routes — Meta
# ---------------------------------------------------------------------------

@app.get("/", tags=["Meta"])
def root():
    return {
        "message":           "CustomerSupportEnv is running",
        "version":           "1.0.0",
        "docs":              "/docs",
        "tasks":             "/tasks",
        "action_space":      "/action_space",
        "observation_space": "/observation_space",
        "health":            "/health",
        "episodes":          "/episodes",
    }


@app.get("/health", tags=["Meta"])
def health():
    cleaned = _cleanup_stale()
    return {
        "status":                 "ok",
        "env":                    "CustomerSupportEnv",
        "version":                "1.0.0",
        "active_sessions":        len(_sessions),
        "stale_sessions_cleaned": cleaned,
    }

# ---------------------------------------------------------------------------
# Routes — Discovery
# ---------------------------------------------------------------------------

@app.get("/tasks", tags=["Discovery"])
def list_tasks():
    """List all tasks with metadata."""
    return {"tasks": list(TASK_DESCRIPTIONS.values())}


@app.get("/tasks/{task_id}", tags=["Discovery"])
def get_task(task_id: str):
    """Get details for a single task."""
    if task_id not in TASK_DESCRIPTIONS:
        raise HTTPException(404, f"task_id '{task_id}' not found. Valid: {VALID_TASKS}")
    return TASK_DESCRIPTIONS[task_id]


@app.get("/action_space", tags=["Discovery"])
def action_space():
    """Full action space specification."""
    return {
        "assign_priority": {
            "type":        "enum",
            "values":      [p.value for p in Priority],
            "description": "Urgency level of the ticket.",
        },
        "assign_category": {
            "type":        "enum",
            "values":      [c.value for c in Category],
            "description": "Department/type category of the ticket.",
        },
        "response_text": {
            "type":        "string",
            "description": "Natural language reply to send to the customer (scored in task_3).",
        },
        "escalate": {
            "type":        "boolean",
            "description": "Whether to escalate to a human agent (scored in task_3).",
        },
    }


@app.get("/observation_space", tags=["Discovery"])
def observation_space():
    """Full observation space specification."""
    return {
        "ticket": {
            "type": "object",
            "fields": {
                "id":            {"type": "string", "description": "Unique ticket identifier"},
                "subject":       {"type": "string", "description": "One-line ticket subject"},
                "body":          {"type": "string", "description": "Full customer message"},
                "customer_tier": {"type": "enum",   "values": ["free", "pro", "enterprise"]},
                "created_at":    {"type": "string", "description": "ISO-8601 UTC timestamp"},
                "sentiment":     {"type": "enum",   "values": ["angry", "neutral", "positive"]},
            },
        },
        "queue_size":           {"type": "integer", "description": "Remaining tickets in queue"},
        "time_elapsed_seconds": {"type": "float",   "description": "Wall-clock seconds since reset()"},
        "agent_actions_taken":  {"type": "integer", "description": "Steps completed so far"},
        "task_id":              {"type": "string",  "description": "Active task identifier"},
        "hint":                 {"type": "string",  "description": "Task-specific guidance for the agent"},
    }

# ---------------------------------------------------------------------------
# Routes — Environment (OpenEnv spec)
# ---------------------------------------------------------------------------

@app.post("/reset", tags=["Environment"])
async def reset(req: Optional[ResetRequest] = None):
    """
    Start a new episode.

    Returns a `session_id` — pass this to every subsequent /step call.
    Also returns the first observation and task metadata.
    """
    if req is None:
        req = ResetRequest()
    if req.task_id not in VALID_TASKS:
        raise HTTPException(400, f"task_id must be one of {VALID_TASKS}")

    _cleanup_stale()

    session_id = str(uuid.uuid4())
    env        = CustomerSupportEnv(task_id=req.task_id, seed=req.seed)
    obs        = env.reset()

    # --- DB: open episode row ---
    episode_id = await save_episode(req.task_id, req.seed)

    _sessions[session_id]     = env
    _session_meta[session_id] = {
        "task_id":      req.task_id,
        "seed":         req.seed,
        "created_at":   time.time(),
        "step_count":   0,
        "episode_id":   episode_id,
        "total_reward": 0.0,
    }

    return {
        "session_id":  session_id,
        "observation": obs.model_dump(),
        "task":        TASK_DESCRIPTIONS[req.task_id],
    }


@app.post("/step", tags=["Environment"])
async def step(req: StepRequest):
    """
    Take one action in the environment.

    Returns the next observation, a structured reward with component breakdown,
    a `done` flag, and an info dict with ground-truth labels for debugging.

    **Reward breakdown** (all weighted components sum to `overall` in [0.0, 1.0]):
    - `priority_raw`  — raw priority grade (0.0, 0.6, 0.8 or 1.0)
    - `category_raw`  — raw category grade (0.0, 0.4 or 1.0)
    - `response_raw`  — raw response quality (0.0–1.0, task_3 only)
    - `priority`, `category`, `response` — weighted contributions
    """
    env  = _get_session(req.session_id)
    meta = _session_meta.get(req.session_id, {})

    # Snapshot ticket BEFORE stepping — env advances _current_ticket after step()
    # Ticket model in environment.py uses field name `id` (not `ticket_id`)
    current_ticket = getattr(env, "_current_ticket", None)

    # Determine action type
    from environment import ActionType
    action_type = ActionType.CLASSIFY
    if req.action_type and req.action_type.lower() == "ask":
        action_type = ActionType.ASK

    if action_type == ActionType.ASK:
        action = Action(
            action_type=ActionType.ASK,
            clarifying_question=req.clarifying_question or "Could you clarify the issue?",
        )
    else:
        try:
            priority = Priority(req.assign_priority)
        except ValueError:
            raise HTTPException(
                400,
                f"Invalid assign_priority '{req.assign_priority}'. "
                f"Valid values: {[p.value for p in Priority]}"
            )
        try:
            category = Category(req.assign_category)
        except ValueError:
            raise HTTPException(
                400,
                f"Invalid assign_category '{req.assign_category}'. "
                f"Valid values: {[c.value for c in Category]}"
            )

        action = Action(
            action_type=ActionType.CLASSIFY,
            assign_priority=priority,
            assign_category=category,
            response_text=req.response_text,
            escalate=req.escalate,
        )

    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    # --- Update in-memory meta ---
    step_num = meta.get("step_count", 0) + 1
    if req.session_id in _session_meta:
        _session_meta[req.session_id]["step_count"]   = step_num
        _session_meta[req.session_id]["total_reward"] = (
            meta.get("total_reward", 0.0) + reward.value
        )

    episode_id   = meta.get("episode_id")
    total_reward = _session_meta.get(req.session_id, {}).get("total_reward", reward.value)

    # --- DB writes (fire-and-forget so HTTP response is not blocked) ---
    if episode_id:
        # reward.breakdown is a plain dict — no conversion needed
        reward_bd = reward.breakdown

        # FIX 1: obs.world_state is already a plain dict from _make_obs()
        # Do NOT call .model_dump() on it — it is not a pydantic model
        world_state_dict = obs.world_state  # already Dict[str, Any]

        asyncio.create_task(save_step(
            episode_id        = episode_id,
            step_num          = step_num,
            # FIX 2: Ticket model uses `.id`, not `.ticket_id`
            ticket_id         = getattr(current_ticket, "id", None),
            action_type       = action_type.value,
            assigned_priority = req.assign_priority if action_type != ActionType.ASK else None,
            assigned_category = req.assign_category if action_type != ActionType.ASK else None,
            reward_value      = reward.value,
            reward_breakdown  = reward_bd,
        ))

        asyncio.create_task(save_world_snapshot(
            episode_id  = episode_id,
            step_num    = step_num,
            world_state = world_state_dict,
        ))

        if current_ticket is not None:
            asyncio.create_task(save_ticket_log(
                episode_id = episode_id,
                step_num   = step_num,
                ticket     = current_ticket,
            ))

        if done:
            # FIX 3: episode_summary from _build_info() has these exact flat keys:
            #   mean_reward, total_reward, final_balance, final_churn_risk, sla_breaches
            # There is NO nested "final_world_state" dict — that was the bug
            episode_summary = info.get("episode_summary", {})
            asyncio.create_task(close_episode(
                episode_id       = episode_id,
                mean_reward      = episode_summary.get("mean_reward",      total_reward / max(step_num, 1)),
                total_reward     = episode_summary.get("total_reward",     total_reward),
                sla_breaches     = episode_summary.get("sla_breaches",     0),
                final_balance    = episode_summary.get("final_balance",    0.0),
                final_churn_risk = episode_summary.get("final_churn_risk", 0.0),
                step_count       = step_num,
            ))

    # Clean up finished session from memory
    if done:
        _sessions.pop(req.session_id, None)
        _session_meta.pop(req.session_id, None)

    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }


@app.get("/state/{session_id}", tags=["Environment"])
def get_state(session_id: str):
    """
    Return a lightweight state snapshot for the given session.
    Useful for checkpointing or debugging without consuming a step.
    """
    env  = _get_session(session_id)
    meta = _session_meta.get(session_id, {})
    return {
        "state":       env.state(),
        "task_id":     meta.get("task_id"),
        "step_count":  meta.get("step_count", 0),
        "episode_id":  meta.get("episode_id"),
        "age_seconds": round(time.time() - meta.get("created_at", time.time()), 1),
    }


@app.delete("/session/{session_id}", tags=["Environment"])
def delete_session(session_id: str):
    """Explicitly close and clean up a session."""
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found.")
    _sessions.pop(session_id, None)
    _session_meta.pop(session_id, None)
    return {"deleted": True, "session_id": session_id}

# ---------------------------------------------------------------------------
# Routes — Episodes (Phase 1 analytics, feeds dashboard + curriculum)
# ---------------------------------------------------------------------------

@app.get("/episodes", tags=["Analytics"])
async def list_episodes(
    task_id: Optional[str] = None,
    limit:   int           = 200,
):
    """
    Return completed episode summaries, newest first.

    Query params:
      - task_id : filter to a single task (optional)
      - limit   : max rows returned (default 200)
    """
    rows = await get_episodes(limit=limit, task_id=task_id, closed_only=True)
    return {"episodes": rows, "count": len(rows)}


@app.get("/episodes/{episode_id}", tags=["Analytics"])
async def get_episode(episode_id: str):
    """Return a single episode row by ID."""
    row = await get_episode_by_id(episode_id)
    if row is None:
        raise HTTPException(404, f"Episode '{episode_id}' not found.")
    return row


@app.get("/episodes/{episode_id}/steps", tags=["Analytics"])
async def list_episode_steps(episode_id: str):
    """
    Return all step rows for a given episode, ordered by step_num.
    Used by the dashboard replay view.
    """
    episode = await get_episode_by_id(episode_id)
    if episode is None:
        raise HTTPException(404, f"Episode '{episode_id}' not found.")
    steps = await get_episode_steps(episode_id)
    return {"episode_id": episode_id, "steps": steps, "count": len(steps)}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
