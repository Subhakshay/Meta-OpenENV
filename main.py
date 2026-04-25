"""
main.py — FastAPI server for The Gauntlet + Shifting Sands (SaaS only)
"""
from __future__ import annotations
import asyncio, logging, os, time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from environment import GauntletEnv, ShiftingSandsEnv
from attacker import AttackerAgent
import db

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s - %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    yield
    await db.close_db()

app = FastAPI(
    title="The Gauntlet + Shifting Sands",
    version="2.0.0",
    description="Adversarial template-based SaaS support environment with policy drift.",
    lifespan=lifespan,
)

SESSIONS: Dict[str, Dict[str, Any]] = {}

class ResetRequest(BaseModel):
    env_type: str = "gauntlet"
    task_id: int = 2
    attacker_enabled: bool = True
    drift_enabled: bool = True
    difficulty_init: int = 1

class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]

@app.post("/reset", tags=["Environment"])
async def reset(req: Optional[ResetRequest] = None):
    if req is None: req = ResetRequest()
    env = ShiftingSandsEnv() if req.env_type == "shifting_sands" else GauntletEnv()
    obs = env.reset(task_id=req.task_id, difficulty_init=req.difficulty_init)
    sid = env.session_id
    attacker = None
    if req.attacker_enabled:
        attacker = AttackerAgent(policy_registry=env.policy_registry)
        try:
            adv = attacker.generate_batch(n=12, difficulty_level=env.world_state.difficulty_level, defender_error_history=[], active_policy=env.policy_registry.get_active())
            env.set_attacker_tickets(adv)
            obs = env._build_observation(adv[0], None)
        except Exception as e:
            logger.warning("Attacker generation failed: %s", e)
    ep_id = await db.create_episode(session_id=sid, task_id=req.task_id, attacker_enabled=req.attacker_enabled, drift_enabled=req.drift_enabled, difficulty_init=req.difficulty_init, env_type=req.env_type)
    SESSIONS[sid] = {"env": env, "episode_id": ep_id, "attacker": attacker, "created_at": time.time()}
    return {"session_id": sid, "observation": obs, "world_state": env.world_state.to_export_dict(), "policy_version": env.policy_registry.active_version_id(), "env_type": req.env_type}

@app.post("/step", tags=["Environment"])
async def step(req: StepRequest):
    session = SESSIONS.get(req.session_id)
    if not session: raise HTTPException(404, "Session not found")
    env = session["env"]
    ep_id = session["episode_id"]
    if "assign_priority" not in req.action and not req.action.get("ask_clarification"):
        raise HTTPException(422, "Action must include 'assign_priority' or 'ask_clarification'")
    try:
        result = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    sn = env.current_step
    tkt = env._ticket_queue[min(env.world_state.tickets_processed - 1, len(env._ticket_queue) - 1)] if env._ticket_queue else {}
    af = result.get("attacker_fitness", 0.0)
    strategy = tkt.get("deception_strategy", "clean")
    asyncio.create_task(db.insert_step(episode_id=ep_id, step_number=sn, ticket_id=tkt.get("ticket_id",""), action=req.action, defender_reward=result["reward"], attacker_fitness=af, breakdown=result.get("reward_breakdown",{}), policy_version=env.policy_registry.active_version_id(), was_post_drift=result.get("drift_notice") is not None, deception_strategy=strategy, template_index=tkt.get("template_index",-1)))
    asyncio.create_task(db.insert_snapshot(ep_id, sn, env.world_state))
    asyncio.create_task(db.insert_ticket_log(ep_id, sn, tkt, env.world_state.difficulty_level))
    if strategy != "clean":
        asyncio.create_task(db.insert_template_fitness(ep_id, sn, tkt.get("template_index",-1), strategy, af, "stay", result["reward"]))
    if result.get("drift_notice"):
        for evt in env.drift_scheduler.get_all_events():
            if evt.fires_at_step == sn:
                asyncio.create_task(db.insert_drift_event(ep_id, sn, evt.from_version, evt.to_version, evt.drift_types, False))
    atk = session.get("attacker")
    if atk and strategy != "clean":
        atk.update_elo(strategy, result["reward"] < 0)
    done = result["done"]
    if done:
        asyncio.create_task(db.close_episode(ep_id, env.get_episode_metrics()))
        del SESSIONS[req.session_id]
    return {"reward": result["reward"], "observation": result["observation"], "world_state": result["world_state"], "done": done, "drift_notice": result.get("drift_notice"), "catastrophic": result.get("catastrophic", False)}

@app.get("/episodes", tags=["Analytics"])
async def list_episodes(limit: int = 50, offset: int = 0, closed_only: bool = True):
    return {"episodes": await db.get_episodes(limit=limit, offset=offset, closed_only=closed_only)}

@app.get("/episodes/{episode_id}", tags=["Analytics"])
async def get_episode(episode_id: int):
    detail = await db.get_episode_detail(episode_id)
    if not detail: raise HTTPException(404, f"Episode {episode_id} not found")
    return detail

@app.get("/world_state/{session_id}", tags=["Environment"])
def get_world_state(session_id: str):
    session = SESSIONS.get(session_id)
    if not session: raise HTTPException(404, "Session not found")
    return session["env"].world_state.to_export_dict()

@app.get("/defender_weaknesses", tags=["Analytics"])
async def defender_weaknesses():
    return {"weaknesses": await db.get_defender_weaknesses()}

@app.get("/template_leaderboard", tags=["Analytics"])
async def template_leaderboard():
    return {"templates": await db.get_template_leaderboard()}

@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "openenv_version": "2.0.0", "active_sessions": len(SESSIONS)}

@app.get("/", tags=["Meta"])
def root():
    return {"name": "The Gauntlet + Shifting Sands", "version": "2.0.0", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
