"""
main.py — FastAPI server for The Gauntlet + Shifting Sands

OpenEnv-compliant endpoints:
  POST /reset         — Start new episode
  POST /step          — Take one action
  GET  /episodes      — List completed episodes
  GET  /episodes/{id} — Episode detail
  GET  /world_state/{session_id} — Current world state
  GET  /health        — Health check
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from environment import CustomerSupportEnv
from attacker import AttackerAgent
from policy import PolicyRegistry
import db

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s - %(message)s")
logger = logging.getLogger(__name__)

# Try to load LLM client for attacker
_LLM_CLIENT = None
try:
    from openai import OpenAI
    _api_key = os.getenv("HF_TOKEN", "")
    _base_url = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    if _api_key:
        _LLM_CLIENT = OpenAI(api_key=_api_key, base_url=_base_url)
except Exception:
    pass

MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    yield
    await db.close_db()


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="The Gauntlet + Shifting Sands",
    version="2.0.0",
    description=(
        "Adversarial self-play customer support environment with policy drift. "
        "An Attacker agent generates deceptive tickets; a Defender agent must "
        "classify them correctly under evolving company policies."
    ),
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Session store
# ─────────────────────────────────────────────────────────────────────────────

SESSIONS: Dict[str, Dict[str, Any]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Request schemas
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = 1
    attacker_enabled: bool = False
    drift_enabled: bool = True
    difficulty_init: float = 0.3


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# POST /reset
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/reset", tags=["Environment"])
async def reset(req: Optional[ResetRequest] = None):
    """Start a new episode."""
    if req is None:
        req = ResetRequest()

    env = CustomerSupportEnv()
    observation = env.reset(
        task_id=req.task_id,
        attacker_enabled=req.attacker_enabled,
        drift_enabled=req.drift_enabled,
        difficulty_init=req.difficulty_init,
    )

    session_id = env.session_id

    # If attacker enabled, generate adversarial tickets
    if req.attacker_enabled:
        attacker = AttackerAgent(
            llm_client=_LLM_CLIENT,
            model_name=MODEL_NAME,
            policy_registry=env.policy_registry,
        )
        try:
            adversarial_tickets = attacker.generate_batch(
                n=20,
                difficulty_level=env.world_state.difficulty_level,
                defender_error_history=[],  # Empty for new session
                active_policy=env.policy_registry.get_active(),
            )
            env.set_attacker_tickets(adversarial_tickets)
            # Rebuild observation with new first ticket
            observation = env._build_observation(adversarial_tickets[0], None)
        except Exception as e:
            logger.warning("Attacker generation failed, using clean tickets: %s", e)

    # Create episode in DB
    episode_id = await db.create_episode(
        session_id=session_id,
        task_id=req.task_id,
        attacker_enabled=req.attacker_enabled,
        drift_enabled=req.drift_enabled,
        difficulty_init=req.difficulty_init,
    )

    SESSIONS[session_id] = {
        "env": env,
        "episode_id": episode_id,
        "created_at": time.time(),
    }

    return {
        "session_id": session_id,
        "observation": observation,
        "world_state": env.world_state.to_export_dict(),
        "policy_version": env.policy_registry.active_version_id(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /step
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/step", tags=["Environment"])
async def step(req: StepRequest):
    """Take one action in the environment."""
    session = SESSIONS.get(req.session_id)
    if session is None:
        raise HTTPException(404, "Session not found or already closed")

    env: CustomerSupportEnv = session["env"]
    episode_id = session["episode_id"]

    # Validate required field
    if "assign_priority" not in req.action and not req.action.get("ask_clarification"):
        raise HTTPException(422, "Action must include 'assign_priority' or 'ask_clarification'")

    try:
        result = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    # DB writes (fire-and-forget)
    step_num = env.current_step
    current_ticket = env._ticket_queue[
        min(env.world_state.tickets_processed - 1, len(env._ticket_queue) - 1)
    ] if env._ticket_queue else {}

    asyncio.create_task(db.insert_step(
        episode_id=episode_id,
        step_number=step_num,
        ticket_id=current_ticket.get("ticket_id", ""),
        action=req.action,
        defender_reward=result["reward"],
        attacker_reward=result.get("attacker_reward", 0.0),
        breakdown=result.get("reward_breakdown", {}),
        policy_version=env.policy_registry.active_version_id(),
        was_post_drift=result.get("drift_notice") is not None,
        deception_strategy=current_ticket.get("deception_strategy", "clean"),
    ))

    asyncio.create_task(db.insert_snapshot(
        episode_id=episode_id,
        step_number=step_num,
        world_state=env.world_state,
    ))

    asyncio.create_task(db.insert_ticket_log(
        episode_id=episode_id,
        step_number=step_num,
        ticket=current_ticket,
        difficulty_level=env.world_state.difficulty_level,
    ))

    # Handle drift event logging
    if result.get("drift_notice"):
        for evt in env.drift_scheduler.get_all_events():
            if evt.fires_at_step == step_num:
                asyncio.create_task(db.insert_drift_event(
                    episode_id=episode_id,
                    step_number=step_num,
                    from_version=evt.from_version,
                    to_version=evt.to_version,
                    drift_types=evt.drift_types,
                    agent_noticed=False,  # Will be updated after scoring
                ))

    # Close episode if done
    done = result["done"]
    if done:
        metrics = env.get_episode_metrics()
        asyncio.create_task(db.close_episode(episode_id, metrics))
        # Cleanup session
        del SESSIONS[req.session_id]

    return {
        "reward": result["reward"],
        "observation": result["observation"],
        "world_state": result["world_state"],
        "done": done,
        "drift_notice": result.get("drift_notice"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /episodes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/episodes", tags=["Analytics"])
async def list_episodes(limit: int = 50, offset: int = 0, closed_only: bool = True):
    rows = await db.get_episodes(limit=limit, offset=offset, closed_only=closed_only)
    return {"episodes": rows, "count": len(rows)}


# ─────────────────────────────────────────────────────────────────────────────
# GET /episodes/{episode_id}
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/episodes/{episode_id}", tags=["Analytics"])
async def get_episode(episode_id: int):
    detail = await db.get_episode_detail(episode_id)
    if detail is None:
        raise HTTPException(404, f"Episode {episode_id} not found")
    return detail


# ─────────────────────────────────────────────────────────────────────────────
# GET /world_state/{session_id}
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/world_state/{session_id}", tags=["Environment"])
def get_world_state(session_id: str):
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found or already closed")
    env: CustomerSupportEnv = session["env"]
    return env.world_state.to_export_dict()


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health():
    return {
        "status": "ok",
        "openenv_version": "2.0.0",
        "active_sessions": len(SESSIONS),
    }


@app.get("/", tags=["Meta"])
def root():
    return {
        "name": "The Gauntlet + Shifting Sands",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "themes": ["self-improvement", "world-modeling"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
