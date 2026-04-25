"""
db.py — Database layer for The Gauntlet + Shifting Sands
SQLite via aiosqlite. Includes template fitness tracking and defender weakness queries.
"""
from __future__ import annotations
import json, logging, os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///gauntlet.db")
_SQLITE_PATH = DATABASE_URL.replace("sqlite:///", "")

try:
    import aiosqlite
except ImportError:
    raise RuntimeError("aiosqlite required. pip install aiosqlite")

_DDL = [
    """CREATE TABLE IF NOT EXISTS episodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
        env_type TEXT DEFAULT 'gauntlet', task_id INTEGER NOT NULL,
        attacker_enabled INTEGER DEFAULT 0, drift_enabled INTEGER DEFAULT 1,
        difficulty_init INTEGER DEFAULT 1, created_at TEXT NOT NULL, closed_at TEXT,
        mean_defender_reward REAL, final_balance REAL, sla_breaches INTEGER,
        drift_accuracy REAL, stale_decisions INTEGER, hallucinations INTEGER,
        attacker_win_rate_final REAL, difficulty_final INTEGER,
        rounds_survived INTEGER, adaptation_speed REAL,
        catastrophic_failure INTEGER DEFAULT 0, catastrophic_reason TEXT)""",
    """CREATE TABLE IF NOT EXISTS steps (
        id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id INTEGER NOT NULL,
        step_number INTEGER NOT NULL, ticket_id TEXT, action_json TEXT,
        defender_reward REAL, attacker_fitness REAL, reward_breakdown_json TEXT,
        policy_version_at_step TEXT, was_post_drift INTEGER DEFAULT 0,
        deception_strategy TEXT, template_index INTEGER DEFAULT -1,
        difficulty_change TEXT, created_at TEXT NOT NULL,
        FOREIGN KEY (episode_id) REFERENCES episodes(id))""",
    """CREATE TABLE IF NOT EXISTS world_state_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id INTEGER NOT NULL,
        step_number INTEGER NOT NULL, snapshot_json TEXT NOT NULL,
        current_policy_version TEXT, drift_events_fired INTEGER,
        difficulty_level INTEGER, created_at TEXT NOT NULL,
        FOREIGN KEY (episode_id) REFERENCES episodes(id))""",
    """CREATE TABLE IF NOT EXISTS tickets_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id INTEGER NOT NULL,
        step_number INTEGER NOT NULL, ticket_json TEXT NOT NULL,
        true_priority TEXT, true_category TEXT, deception_strategy TEXT,
        template_index INTEGER DEFAULT -1, difficulty_level_at_gen INTEGER,
        attacker_confidence REAL, created_at TEXT NOT NULL,
        FOREIGN KEY (episode_id) REFERENCES episodes(id))""",
    """CREATE TABLE IF NOT EXISTS drift_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id INTEGER NOT NULL,
        step_number INTEGER NOT NULL, from_version TEXT NOT NULL,
        to_version TEXT NOT NULL, drift_types_json TEXT,
        agent_noticed INTEGER DEFAULT 0, created_at TEXT NOT NULL,
        FOREIGN KEY (episode_id) REFERENCES episodes(id))""",
    """CREATE TABLE IF NOT EXISTS template_fitness (
        id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id INTEGER NOT NULL,
        step_number INTEGER NOT NULL, template_index INTEGER NOT NULL,
        strategy TEXT, fitness_score REAL, difficulty_change TEXT,
        defender_reward REAL, created_at TEXT NOT NULL,
        FOREIGN KEY (episode_id) REFERENCES episodes(id))""",
    "CREATE INDEX IF NOT EXISTS idx_steps_ep ON steps(episode_id)",
    "CREATE INDEX IF NOT EXISTS idx_snap_ep ON world_state_snapshots(episode_id)",
    "CREATE INDEX IF NOT EXISTS idx_tlog_ep ON tickets_log(episode_id)",
    "CREATE INDEX IF NOT EXISTS idx_drift_ep ON drift_events(episode_id)",
    "CREATE INDEX IF NOT EXISTS idx_tf_ep ON template_fitness(episode_id)",
    "CREATE INDEX IF NOT EXISTS idx_tf_tmpl ON template_fitness(template_index)",
    "CREATE INDEX IF NOT EXISTS idx_steps_reward ON steps(defender_reward)",
]

def _now():
    return datetime.now(timezone.utc).isoformat()

async def init_db():
    async with aiosqlite.connect(_SQLITE_PATH) as conn:
        for ddl in _DDL:
            await conn.execute(ddl)
        await conn.commit()
    logger.info("DB initialised — %s", _SQLITE_PATH)

async def close_db():
    pass

async def create_episode(session_id, task_id, attacker_enabled, drift_enabled, difficulty_init, env_type="gauntlet") -> int:
    async with aiosqlite.connect(_SQLITE_PATH) as conn:
        cur = await conn.execute(
            "INSERT INTO episodes (session_id,env_type,task_id,attacker_enabled,drift_enabled,difficulty_init,created_at) VALUES (?,?,?,?,?,?,?)",
            (session_id, env_type, task_id, int(attacker_enabled), int(drift_enabled), difficulty_init, _now()))
        await conn.commit()
        return cur.lastrowid

async def close_episode(episode_id: int, metrics: Dict[str, Any]):
    async with aiosqlite.connect(_SQLITE_PATH) as conn:
        await conn.execute(
            "UPDATE episodes SET closed_at=?,mean_defender_reward=?,final_balance=?,sla_breaches=?,drift_accuracy=?,stale_decisions=?,hallucinations=?,attacker_win_rate_final=?,difficulty_final=?,rounds_survived=?,adaptation_speed=?,catastrophic_failure=?,catastrophic_reason=? WHERE id=?",
            (_now(), metrics.get("mean_defender_reward",0), metrics.get("final_balance",0),
             metrics.get("sla_breaches",0), metrics.get("drift_accuracy",0),
             metrics.get("stale_decisions",0), metrics.get("hallucinations",0),
             metrics.get("attacker_win_rate_final",0), metrics.get("difficulty_final",1),
             metrics.get("rounds_survived",0), metrics.get("adaptation_speed",0),
             int(metrics.get("catastrophic_failure",False)), metrics.get("catastrophic_reason",""),
             episode_id))
        await conn.commit()

async def insert_step(episode_id, step_number, ticket_id, action, defender_reward, attacker_fitness,
                      breakdown, policy_version, was_post_drift, deception_strategy, template_index=-1, difficulty_change="stay"):
    async with aiosqlite.connect(_SQLITE_PATH) as conn:
        await conn.execute(
            "INSERT INTO steps (episode_id,step_number,ticket_id,action_json,defender_reward,attacker_fitness,reward_breakdown_json,policy_version_at_step,was_post_drift,deception_strategy,template_index,difficulty_change,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (episode_id, step_number, ticket_id, json.dumps(action), defender_reward, attacker_fitness,
             json.dumps(breakdown), policy_version, int(was_post_drift), deception_strategy, template_index, difficulty_change, _now()))
        await conn.commit()

async def insert_snapshot(episode_id, step_number, world_state):
    ws = world_state.to_export_dict() if hasattr(world_state, "to_export_dict") else dict(world_state)
    async with aiosqlite.connect(_SQLITE_PATH) as conn:
        await conn.execute(
            "INSERT INTO world_state_snapshots (episode_id,step_number,snapshot_json,current_policy_version,drift_events_fired,difficulty_level,created_at) VALUES (?,?,?,?,?,?,?)",
            (episode_id, step_number, json.dumps(ws), ws.get("current_policy_version","v1"),
             ws.get("drift_events_fired",0), ws.get("difficulty_level",1), _now()))
        await conn.commit()

async def insert_ticket_log(episode_id, step_number, ticket, difficulty_level):
    async with aiosqlite.connect(_SQLITE_PATH) as conn:
        await conn.execute(
            "INSERT INTO tickets_log (episode_id,step_number,ticket_json,true_priority,true_category,deception_strategy,template_index,difficulty_level_at_gen,attacker_confidence,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (episode_id, step_number, json.dumps(ticket), ticket.get("true_priority",""),
             ticket.get("true_category",""), ticket.get("deception_strategy",""),
             ticket.get("template_index",-1), difficulty_level, ticket.get("attacker_confidence",0), _now()))
        await conn.commit()

async def insert_drift_event(episode_id, step_number, from_version, to_version, drift_types, agent_noticed):
    async with aiosqlite.connect(_SQLITE_PATH) as conn:
        await conn.execute(
            "INSERT INTO drift_events (episode_id,step_number,from_version,to_version,drift_types_json,agent_noticed,created_at) VALUES (?,?,?,?,?,?,?)",
            (episode_id, step_number, from_version, to_version, json.dumps(drift_types), int(agent_noticed), _now()))
        await conn.commit()

async def insert_template_fitness(episode_id, step_number, template_index, strategy, fitness_score, difficulty_change, defender_reward):
    async with aiosqlite.connect(_SQLITE_PATH) as conn:
        await conn.execute(
            "INSERT INTO template_fitness (episode_id,step_number,template_index,strategy,fitness_score,difficulty_change,defender_reward,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (episode_id, step_number, template_index, strategy, fitness_score, difficulty_change, defender_reward, _now()))
        await conn.commit()

async def get_episodes(limit=50, offset=0, closed_only=True):
    where = "WHERE closed_at IS NOT NULL" if closed_only else ""
    async with aiosqlite.connect(_SQLITE_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(f"SELECT * FROM episodes {where} ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)) as cur:
            return [dict(r) for r in await cur.fetchall()]

async def get_episode_detail(episode_id: int):
    async with aiosqlite.connect(_SQLITE_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)) as cur:
            ep = await cur.fetchone()
            if not ep: return None
            ep = dict(ep)
        async with conn.execute("SELECT * FROM steps WHERE episode_id=? ORDER BY step_number", (episode_id,)) as cur:
            ep["steps"] = [dict(r) for r in await cur.fetchall()]
        async with conn.execute("SELECT * FROM world_state_snapshots WHERE episode_id=? ORDER BY step_number", (episode_id,)) as cur:
            ep["snapshots"] = [dict(r) for r in await cur.fetchall()]
        async with conn.execute("SELECT * FROM drift_events WHERE episode_id=? ORDER BY step_number", (episode_id,)) as cur:
            ep["drift_events"] = [dict(r) for r in await cur.fetchall()]
        async with conn.execute("SELECT * FROM template_fitness WHERE episode_id=? ORDER BY step_number", (episode_id,)) as cur:
            ep["template_fitness"] = [dict(r) for r in await cur.fetchall()]
        return ep

async def get_defender_weaknesses(limit=100):
    """Query DB for defender's weak spots — used by attacker to target exploits."""
    async with aiosqlite.connect(_SQLITE_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT deception_strategy, COUNT(*) as cnt, AVG(defender_reward) as avg_reward "
            "FROM steps WHERE defender_reward < 0 GROUP BY deception_strategy ORDER BY cnt DESC LIMIT ?", (limit,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

async def get_template_leaderboard(limit=20):
    """Get top-performing attacker templates by average fitness."""
    async with aiosqlite.connect(_SQLITE_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT template_index, strategy, COUNT(*) as uses, AVG(fitness_score) as avg_fitness, AVG(defender_reward) as avg_def_reward "
            "FROM template_fitness GROUP BY template_index ORDER BY avg_fitness DESC LIMIT ?", (limit,)
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]
