"""
db.py — Phase 1: Persistent storage layer for Meta-OpenENV
==========================================================
Supports SQLite locally and PostgreSQL in production.

Auto-detects via DATABASE_URL env var:
  - unset / starts with "sqlite"  → SQLite at ./support_triage.db
  - starts with "postgresql"      → PostgreSQL (requires asyncpg)

Tables
------
  episodes              — one row per completed episode
  steps                 — one row per step within an episode
  world_state_snapshots — world state captured after every step
  tickets_log           — ticket metadata for every step

Usage in main.py
----------------
  from db import init_db, save_episode, save_step, save_world_snapshot, \
                 save_ticket_log, get_episodes, get_episode_steps

  # On startup:
  await init_db()

  # After reset():
  episode_id = await save_episode(task_id, seed)          # returns str UUID

  # After every step():
  await save_step(episode_id, step_num, ...)
  await save_world_snapshot(episode_id, step_num, world_state)
  await save_ticket_log(episode_id, step_num, ticket)

  # On done=True, update episode summary:
  await close_episode(episode_id, mean_reward, total_reward,
                      sla_breaches, final_balance, final_churn_risk)

  # GET /episodes, GET /episodes/{id}/steps:
  rows = await get_episodes(limit=100, task_id=None)
  rows = await get_episode_steps(episode_id)
"""

from __future__ import annotations

import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection setup — SQLite (aiosqlite) or PostgreSQL (asyncpg)
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./support_triage.db")

_USE_POSTGRES = DATABASE_URL.startswith("postgresql") or DATABASE_URL.startswith("postgres")

if _USE_POSTGRES:
    try:
        import asyncpg  # type: ignore
        _PG_POOL = None  # initialised in init_db()
    except ImportError:
        raise RuntimeError(
            "asyncpg is required for PostgreSQL. "
            "Add it to requirements.txt and reinstall."
        )
else:
    try:
        import aiosqlite  # type: ignore
    except ImportError:
        raise RuntimeError(
            "aiosqlite is required for SQLite. "
            "pip install aiosqlite"
        )
    _SQLITE_PATH = DATABASE_URL.replace("sqlite:///", "")

# ---------------------------------------------------------------------------
# DDL — table definitions (dialect-compatible)
# ---------------------------------------------------------------------------

_DDL_SQLITE = [
    """
    CREATE TABLE IF NOT EXISTS episodes (
        episode_id      TEXT PRIMARY KEY,
        task_id         TEXT NOT NULL,
        seed            INTEGER NOT NULL,
        created_at      TEXT NOT NULL,
        closed_at       TEXT,
        mean_reward     REAL,
        total_reward    REAL,
        sla_breaches    INTEGER,
        final_balance   REAL,
        final_churn_risk REAL,
        step_count      INTEGER DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS steps (
        step_id             TEXT PRIMARY KEY,
        episode_id          TEXT NOT NULL,
        step_num            INTEGER NOT NULL,
        ticket_id           TEXT,
        action_type         TEXT,
        assigned_priority   TEXT,
        assigned_category   TEXT,
        reward_value        REAL,
        priority_raw        REAL,
        category_raw        REAL,
        response_raw        REAL,
        clarification_raw   REAL,
        world_state_raw     REAL,
        created_at          TEXT NOT NULL,
        FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS world_state_snapshots (
        snapshot_id         TEXT PRIMARY KEY,
        episode_id          TEXT NOT NULL,
        step_num            INTEGER NOT NULL,
        company_balance     REAL,
        escalation_queue    INTEGER,
        customer_churn_risk REAL,
        sla_breach_count    INTEGER,
        tickets_resolved    INTEGER,
        avg_response_quality REAL,
        created_at          TEXT NOT NULL,
        FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tickets_log (
        log_id              TEXT PRIMARY KEY,
        episode_id          TEXT NOT NULL,
        step_num            INTEGER NOT NULL,
        ticket_id           TEXT,
        true_priority       TEXT,
        true_category       TEXT,
        requires_clarification INTEGER,
        subject             TEXT,
        created_at          TEXT NOT NULL,
        FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
    )
    """,
    # Indexes for common query patterns
    "CREATE INDEX IF NOT EXISTS idx_steps_episode ON steps(episode_id)",
    "CREATE INDEX IF NOT EXISTS idx_snapshots_episode ON world_state_snapshots(episode_id)",
    "CREATE INDEX IF NOT EXISTS idx_tickets_episode ON tickets_log(episode_id)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_task ON episodes(task_id)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at)",
]

_DDL_POSTGRES = [
    """
    CREATE TABLE IF NOT EXISTS episodes (
        episode_id          UUID PRIMARY KEY,
        task_id             TEXT NOT NULL,
        seed                INTEGER NOT NULL,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        closed_at           TIMESTAMPTZ,
        mean_reward         DOUBLE PRECISION,
        total_reward        DOUBLE PRECISION,
        sla_breaches        INTEGER,
        final_balance       DOUBLE PRECISION,
        final_churn_risk    DOUBLE PRECISION,
        step_count          INTEGER DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS steps (
        step_id             UUID PRIMARY KEY,
        episode_id          UUID NOT NULL REFERENCES episodes(episode_id),
        step_num            INTEGER NOT NULL,
        ticket_id           TEXT,
        action_type         TEXT,
        assigned_priority   TEXT,
        assigned_category   TEXT,
        reward_value        DOUBLE PRECISION,
        priority_raw        DOUBLE PRECISION,
        category_raw        DOUBLE PRECISION,
        response_raw        DOUBLE PRECISION,
        clarification_raw   DOUBLE PRECISION,
        world_state_raw     DOUBLE PRECISION,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS world_state_snapshots (
        snapshot_id         UUID PRIMARY KEY,
        episode_id          UUID NOT NULL REFERENCES episodes(episode_id),
        step_num            INTEGER NOT NULL,
        company_balance     DOUBLE PRECISION,
        escalation_queue    INTEGER,
        customer_churn_risk DOUBLE PRECISION,
        sla_breach_count    INTEGER,
        tickets_resolved    INTEGER,
        avg_response_quality DOUBLE PRECISION,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tickets_log (
        log_id              UUID PRIMARY KEY,
        episode_id          UUID NOT NULL REFERENCES episodes(episode_id),
        step_num            INTEGER NOT NULL,
        ticket_id           TEXT,
        true_priority       TEXT,
        true_category       TEXT,
        requires_clarification BOOLEAN,
        subject             TEXT,
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_steps_episode ON steps(episode_id)",
    "CREATE INDEX IF NOT EXISTS idx_snapshots_episode ON world_state_snapshots(episode_id)",
    "CREATE INDEX IF NOT EXISTS idx_tickets_episode ON tickets_log(episode_id)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_task ON episodes(task_id)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at)",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _new_id() -> str:
    return str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Low-level execute helpers (unified interface over aiosqlite / asyncpg)
# ---------------------------------------------------------------------------

async def _execute_sqlite(sql: str, params: tuple = ()) -> None:
    async with aiosqlite.connect(_SQLITE_PATH) as db:
        await db.execute(sql, params)
        await db.commit()

async def _fetchall_sqlite(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    async with aiosqlite.connect(_SQLITE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

async def _fetchone_sqlite(sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(_SQLITE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, params) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

async def _execute_pg(sql: str, params: tuple = ()) -> None:
    global _PG_POOL
    async with _PG_POOL.acquire() as conn:
        await conn.execute(sql, *params)

async def _fetchall_pg(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    global _PG_POOL
    async with _PG_POOL.acquire() as conn:
        rows = await conn.fetch(sql, *params)
        return [dict(r) for r in rows]

async def _fetchone_pg(sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    global _PG_POOL
    async with _PG_POOL.acquire() as conn:
        row = await conn.fetchrow(sql, *params)
        return dict(row) if row else None

# Dispatch to correct backend
async def _execute(sql: str, params: tuple = ()) -> None:
    if _USE_POSTGRES:
        # asyncpg uses $1 $2 placeholders — SQLite uses ?
        await _execute_pg(sql, params)
    else:
        await _execute_sqlite(sql, params)

async def _fetchall(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    if _USE_POSTGRES:
        return await _fetchall_pg(sql, params)
    return await _fetchall_sqlite(sql, params)

async def _fetchone(sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    if _USE_POSTGRES:
        return await _fetchone_pg(sql, params)
    return await _fetchone_sqlite(sql, params)

# ---------------------------------------------------------------------------
# SQL with dialect-aware placeholders
# ---------------------------------------------------------------------------

def _ph(n: int) -> str:
    """Return the nth positional placeholder for the active backend."""
    return f"${n}" if _USE_POSTGRES else "?"

def _phs(*ns) -> str:
    """Return comma-separated placeholders for the given column count."""
    return ", ".join(_ph(i) for i in ns)

# ---------------------------------------------------------------------------
# Public API — Lifecycle
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """
    Create all tables and indexes. Call once at application startup.

    In FastAPI add to lifespan:
        @asynccontextmanager
        async def lifespan(app):
            await init_db()
            yield

    Or simply call in an @app.on_event("startup") handler.
    """
    global _PG_POOL

    if _USE_POSTGRES:
        _PG_POOL = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
        async with _PG_POOL.acquire() as conn:
            for ddl in _DDL_POSTGRES:
                await conn.execute(ddl)
        logger.info("DB initialised — PostgreSQL")
    else:
        async with aiosqlite.connect(_SQLITE_PATH) as db:
            for ddl in _DDL_SQLITE:
                await db.execute(ddl)
            await db.commit()
        logger.info("DB initialised — SQLite @ %s", _SQLITE_PATH)


async def close_db() -> None:
    """Close connection pool. Call on application shutdown."""
    global _PG_POOL
    if _USE_POSTGRES and _PG_POOL:
        await _PG_POOL.close()
        logger.info("DB pool closed")

# ---------------------------------------------------------------------------
# Public API — Episodes
# ---------------------------------------------------------------------------

async def save_episode(task_id: str, seed: int) -> str:
    """
    Insert a new episode row and return its episode_id (UUID string).

    Call immediately after CustomerSupportEnv.reset().
    """
    episode_id = _new_id()
    now = _now_iso()

    if _USE_POSTGRES:
        sql = f"""
            INSERT INTO episodes
                (episode_id, task_id, seed, created_at)
            VALUES
                ($1, $2, $3, NOW())
        """
        await _execute_pg(sql, (episode_id, task_id, seed))
    else:
        sql = """
            INSERT INTO episodes
                (episode_id, task_id, seed, created_at)
            VALUES
                (?, ?, ?, ?)
        """
        await _execute_sqlite(sql, (episode_id, task_id, seed, now))

    logger.debug("episode created: %s  task=%s  seed=%s", episode_id, task_id, seed)
    return episode_id


async def close_episode(
    episode_id: str,
    mean_reward: float,
    total_reward: float,
    sla_breaches: int,
    final_balance: float,
    final_churn_risk: float,
    step_count: int,
) -> None:
    """
    Update episode row with summary metrics when done=True.

    Call at the end of every episode so that curriculum.py and the
    dashboard can query completed-episode stats.
    """
    now = _now_iso()

    if _USE_POSTGRES:
        sql = """
            UPDATE episodes SET
                closed_at        = NOW(),
                mean_reward      = $2,
                total_reward     = $3,
                sla_breaches     = $4,
                final_balance    = $5,
                final_churn_risk = $6,
                step_count       = $7
            WHERE episode_id = $1
        """
        await _execute_pg(
            sql,
            (episode_id, mean_reward, total_reward,
             sla_breaches, final_balance, final_churn_risk, step_count),
        )
    else:
        sql = """
            UPDATE episodes SET
                closed_at        = ?,
                mean_reward      = ?,
                total_reward     = ?,
                sla_breaches     = ?,
                final_balance    = ?,
                final_churn_risk = ?,
                step_count       = ?
            WHERE episode_id = ?
        """
        await _execute_sqlite(
            sql,
            (now, mean_reward, total_reward,
             sla_breaches, final_balance, final_churn_risk,
             step_count, episode_id),
        )
    logger.debug(
        "episode closed: %s  mean_reward=%.4f  sla=%d",
        episode_id, mean_reward, sla_breaches,
    )


async def get_episodes(
    limit: int = 200,
    task_id: Optional[str] = None,
    closed_only: bool = True,
) -> List[Dict[str, Any]]:
    """
    Return a list of episode summary dicts, newest first.

    Used by GET /episodes and the reward curve in the dashboard.

    Parameters
    ----------
    limit      : maximum rows returned (default 200)
    task_id    : filter to a single task if provided
    closed_only: if True (default) only return episodes where done=True
                 (i.e. closed_at IS NOT NULL)
    """
    conditions = []
    params: list = []
    idx = 1

    if closed_only:
        conditions.append("closed_at IS NOT NULL")

    if task_id:
        ph = _ph(idx) if _USE_POSTGRES else "?"
        conditions.append(f"task_id = {ph}")
        params.append(task_id)
        idx += 1

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    limit_ph = _ph(idx) if _USE_POSTGRES else "?"
    params.append(limit)

    sql = f"""
        SELECT
            episode_id, task_id, seed, created_at, closed_at,
            mean_reward, total_reward, sla_breaches,
            final_balance, final_churn_risk, step_count
        FROM episodes
        {where}
        ORDER BY created_at DESC
        LIMIT {limit_ph}
    """
    return await _fetchall(sql, tuple(params))


async def get_episode_by_id(episode_id: str) -> Optional[Dict[str, Any]]:
    """Return a single episode row or None."""
    if _USE_POSTGRES:
        sql = "SELECT * FROM episodes WHERE episode_id = $1"
    else:
        sql = "SELECT * FROM episodes WHERE episode_id = ?"
    return await _fetchone(sql, (episode_id,))

# ---------------------------------------------------------------------------
# Public API — Steps
# ---------------------------------------------------------------------------

async def save_step(
    episode_id: str,
    step_num: int,
    ticket_id: Optional[str],
    action_type: str,
    assigned_priority: Optional[str],
    assigned_category: Optional[str],
    reward_value: float,
    reward_breakdown: Optional[Dict[str, float]] = None,
) -> str:
    """
    Persist one agent step. Returns step_id (UUID string).

    reward_breakdown should be the dict returned by Reward.breakdown, e.g.:
        {
          "priority_raw": 1.0, "category_raw": 0.8, "response_raw": 0.72,
          "clarification_raw": 0.6, "world_state_raw": 0.85,
          ...weighted keys...
        }
    Only the *_raw keys are stored individually; the weighted sums are
    derivable from them + the task weights.
    """
    step_id = _new_id()
    now = _now_iso()
    bd = reward_breakdown or {}

    priority_raw      = bd.get("priority_raw")
    category_raw      = bd.get("category_raw")
    response_raw      = bd.get("response_raw")
    clarification_raw = bd.get("clarification_raw")
    world_state_raw   = bd.get("world_state_raw")

    if _USE_POSTGRES:
        sql = """
            INSERT INTO steps (
                step_id, episode_id, step_num, ticket_id,
                action_type, assigned_priority, assigned_category,
                reward_value, priority_raw, category_raw, response_raw,
                clarification_raw, world_state_raw, created_at
            ) VALUES (
                $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,NOW()
            )
        """
        await _execute_pg(
            sql,
            (step_id, episode_id, step_num, ticket_id,
             action_type, assigned_priority, assigned_category,
             reward_value, priority_raw, category_raw, response_raw,
             clarification_raw, world_state_raw),
        )
    else:
        sql = """
            INSERT INTO steps (
                step_id, episode_id, step_num, ticket_id,
                action_type, assigned_priority, assigned_category,
                reward_value, priority_raw, category_raw, response_raw,
                clarification_raw, world_state_raw, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        await _execute_sqlite(
            sql,
            (step_id, episode_id, step_num, ticket_id,
             action_type, assigned_priority, assigned_category,
             reward_value, priority_raw, category_raw, response_raw,
             clarification_raw, world_state_raw, now),
        )

    logger.debug(
        "step saved: ep=%s  step=%d  reward=%.4f",
        episode_id, step_num, reward_value,
    )
    return step_id


async def get_episode_steps(episode_id: str) -> List[Dict[str, Any]]:
    """
    Return all step rows for an episode, ordered by step_num.

    Used by GET /episodes/{id}/steps for the dashboard replay view
    and reward improvement charts.
    """
    if _USE_POSTGRES:
        sql = """
            SELECT * FROM steps
            WHERE episode_id = $1
            ORDER BY step_num ASC
        """
    else:
        sql = """
            SELECT * FROM steps
            WHERE episode_id = ?
            ORDER BY step_num ASC
        """
    return await _fetchall(sql, (episode_id,))

# ---------------------------------------------------------------------------
# Public API — World State Snapshots
# ---------------------------------------------------------------------------

async def save_world_snapshot(
    episode_id: str,
    step_num: int,
    world_state: Dict[str, Any],
) -> str:
    """
    Persist a world_state dict after each step.

    world_state should be the dict from Observation.world_state or
    WorldState.model_dump(). Expected keys:
        company_balance, escalation_queue, customer_churn_risk,
        sla_breach_count, tickets_resolved, avg_response_quality
    """
    snapshot_id = _new_id()
    now = _now_iso()

    balance      = world_state.get("company_balance")
    esc_queue    = world_state.get("escalation_queue")
    churn        = world_state.get("customer_churn_risk")
    sla          = world_state.get("sla_breach_count")
    resolved     = world_state.get("tickets_resolved")
    avg_quality  = world_state.get("avg_response_quality")

    if _USE_POSTGRES:
        sql = """
            INSERT INTO world_state_snapshots (
                snapshot_id, episode_id, step_num,
                company_balance, escalation_queue, customer_churn_risk,
                sla_breach_count, tickets_resolved, avg_response_quality,
                created_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,NOW())
        """
        await _execute_pg(
            sql,
            (snapshot_id, episode_id, step_num,
             balance, esc_queue, churn, sla, resolved, avg_quality),
        )
    else:
        sql = """
            INSERT INTO world_state_snapshots (
                snapshot_id, episode_id, step_num,
                company_balance, escalation_queue, customer_churn_risk,
                sla_breach_count, tickets_resolved, avg_response_quality,
                created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """
        await _execute_sqlite(
            sql,
            (snapshot_id, episode_id, step_num,
             balance, esc_queue, churn, sla, resolved, avg_quality, now),
        )

    return snapshot_id


async def get_world_snapshots(episode_id: str) -> List[Dict[str, Any]]:
    """Return all world state snapshots for an episode, ordered by step_num."""
    if _USE_POSTGRES:
        sql = """
            SELECT * FROM world_state_snapshots
            WHERE episode_id = $1
            ORDER BY step_num ASC
        """
    else:
        sql = """
            SELECT * FROM world_state_snapshots
            WHERE episode_id = ?
            ORDER BY step_num ASC
        """
    return await _fetchall(sql, (episode_id,))

# ---------------------------------------------------------------------------
# Public API — Tickets Log
# ---------------------------------------------------------------------------

async def save_ticket_log(
    episode_id: str,
    step_num: int,
    ticket: Any,  # Ticket pydantic model or plain dict
) -> str:
    """
    Log a ticket seen during the episode.

    Accepts either a Ticket pydantic model (uses .model_dump()) or a plain dict.
    Expected keys: ticket_id, true_priority, true_category,
                   requires_clarification, subject
    """
    log_id = _new_id()
    now = _now_iso()

    if hasattr(ticket, "model_dump"):
        td = ticket.model_dump()
    elif hasattr(ticket, "dict"):
        td = ticket.dict()
    else:
        td = dict(ticket)

    ticket_id             = td.get("ticket_id") or td.get("id")
    true_priority         = td.get("true_priority")
    true_category         = td.get("true_category")
    requires_clarification = int(bool(td.get("requires_clarification", False)))
    subject               = td.get("subject", "")[:500]  # truncate safety

    if _USE_POSTGRES:
        sql = """
            INSERT INTO tickets_log (
                log_id, episode_id, step_num,
                ticket_id, true_priority, true_category,
                requires_clarification, subject, created_at
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,NOW())
        """
        await _execute_pg(
            sql,
            (log_id, episode_id, step_num,
             ticket_id, true_priority, true_category,
             bool(requires_clarification), subject),
        )
    else:
        sql = """
            INSERT INTO tickets_log (
                log_id, episode_id, step_num,
                ticket_id, true_priority, true_category,
                requires_clarification, subject, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?)
        """
        await _execute_sqlite(
            sql,
            (log_id, episode_id, step_num,
             ticket_id, true_priority, true_category,
             requires_clarification, subject, now),
        )

    return log_id


async def get_episode_tickets(episode_id: str) -> List[Dict[str, Any]]:
    """Return all ticket log rows for an episode."""
    if _USE_POSTGRES:
        sql = "SELECT * FROM tickets_log WHERE episode_id = $1 ORDER BY step_num ASC"
    else:
        sql = "SELECT * FROM tickets_log WHERE episode_id = ? ORDER BY step_num ASC"
    return await _fetchall(sql, (episode_id,))

# ---------------------------------------------------------------------------
# Public API — Analytics helpers (used by curriculum.py)
# ---------------------------------------------------------------------------

async def get_rolling_mean_reward(
    task_id: str,
    last_n: int = 20,
) -> Optional[float]:
    """
    Return the rolling mean reward across the last `last_n` completed episodes
    for the given task_id.

    Returns None if there are no completed episodes yet.
    Used by CurriculumManager to decide when to unlock the next task level.
    """
    if _USE_POSTGRES:
        sql = """
            SELECT AVG(mean_reward) AS rolling_mean
            FROM (
                SELECT mean_reward
                FROM episodes
                WHERE task_id = $1 AND closed_at IS NOT NULL
                ORDER BY created_at DESC
                LIMIT $2
            ) sub
        """
    else:
        sql = """
            SELECT AVG(mean_reward) AS rolling_mean
            FROM (
                SELECT mean_reward
                FROM episodes
                WHERE task_id = ? AND closed_at IS NOT NULL
                ORDER BY created_at DESC
                LIMIT ?
            ) sub
        """
    row = await _fetchone(sql, (task_id, last_n))
    if row is None:
        return None
    val = row.get("rolling_mean")
    return float(val) if val is not None else None


async def get_low_reward_tickets(
    task_id: str,
    reward_threshold: float = 0.4,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Return ticket rows where the agent scored poorly (reward < threshold).

    Used by Phase 2 failure-driven ticket hardening in environment.py:
    after each episode, query this, generate harder variants of weak tickets,
    and inject them into the blueprint pool.
    """
    if _USE_POSTGRES:
        sql = """
            SELECT
                tl.ticket_id,
                tl.true_priority,
                tl.true_category,
                tl.requires_clarification,
                tl.subject,
                s.reward_value,
                s.priority_raw,
                s.category_raw
            FROM tickets_log tl
            JOIN steps s
                ON tl.episode_id = s.episode_id
               AND tl.step_num   = s.step_num
            JOIN episodes e
                ON tl.episode_id = e.episode_id
            WHERE e.task_id     = $1
              AND s.reward_value < $2
              AND e.closed_at IS NOT NULL
            ORDER BY s.reward_value ASC
            LIMIT $3
        """
    else:
        sql = """
            SELECT
                tl.ticket_id,
                tl.true_priority,
                tl.true_category,
                tl.requires_clarification,
                tl.subject,
                s.reward_value,
                s.priority_raw,
                s.category_raw
            FROM tickets_log tl
            JOIN steps s
                ON tl.episode_id = s.episode_id
               AND tl.step_num   = s.step_num
            JOIN episodes e
                ON tl.episode_id = e.episode_id
            WHERE e.task_id     = ?
              AND s.reward_value < ?
              AND e.closed_at IS NOT NULL
            ORDER BY s.reward_value ASC
            LIMIT ?
        """
    return await _fetchall(sql, (task_id, reward_threshold, limit))


async def get_task_episode_count(task_id: str) -> int:
    """Return the total number of completed episodes for a task."""
    if _USE_POSTGRES:
        sql = """
            SELECT COUNT(*) AS cnt FROM episodes
            WHERE task_id = $1 AND closed_at IS NOT NULL
        """
    else:
        sql = """
            SELECT COUNT(*) AS cnt FROM episodes
            WHERE task_id = ? AND closed_at IS NOT NULL
        """
    row = await _fetchone(sql, (task_id,))
    if row is None:
        return 0
    return int(row.get("cnt", 0))


async def get_reward_curve(
    task_id: Optional[str] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """
    Return (episode_id, task_id, created_at, mean_reward, step_count)
    for the reward curve chart in the dashboard.

    Ordered oldest-first so it can be plotted directly on an x-axis.
    """
    params: list = []
    idx = 1

    where_clause = "WHERE closed_at IS NOT NULL"
    if task_id:
        ph = _ph(idx) if _USE_POSTGRES else "?"
        where_clause += f" AND task_id = {ph}"
        params.append(task_id)
        idx += 1

    limit_ph = _ph(idx) if _USE_POSTGRES else "?"
    params.append(limit)

    sql = f"""
        SELECT episode_id, task_id, created_at, mean_reward, step_count
        FROM episodes
        {where_clause}
        ORDER BY created_at ASC
        LIMIT {limit_ph}
    """
    return await _fetchall(sql, tuple(params))
