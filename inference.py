"""
inference.py — Local test runner for GauntletEnv and ShiftingSandsEnv (SaaS only)

Agent functions (rule_based_agent, llm_agent) are plain synchronous functions
safe to import anywhere. The async run_episode() is only used by main() and
should not be called from synchronous plotting/eval scripts — use the env
directly instead (as generate_training_plots.py does).
"""
from __future__ import annotations
import asyncio, json, os, re
from typing import Any, Callable, Dict
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from environment import GauntletEnv, ShiftingSandsEnv
from attacker import AttackerAgent
import db

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")
_client = None
HAS_LLM = False
try:
    from openai import OpenAI
    if HF_TOKEN:
        _client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
        HAS_LLM = True
except Exception:
    pass
if not HAS_LLM:
    print("[INFO] No HF_TOKEN -- rule-based agent only.")

# ── SaaS-only keyword signals ────────────────────────────────────────────────
_PRIORITY_SIGNALS = {
    "Critical": ["production down","completely down","system down","data loss","outage","emergency",
                 "critical","urgent:","all users locked","unauthorized access","api keys leaked",
                 "connection pool exhausted","complete system failure"],
    "High": ["cannot log in","can't log in","charged twice","duplicate charge","locked out",
             "suspicious login","bypass vulnerability","gdpr","regulatory","data privacy",
             "cancelling","switching","competitor","unauthorized","leaked","sso broken"],
    "Medium": ["sometimes","occasionally","slow","seems off","not sure","acting weird",
               "permission errors","sessions I don't recognize","behaving unexpectedly"],
    "Low": ["question about","interested in","feedback","how do i","how to",
            "quick question","no rush","annual billing discount","does the","can i get"],
}
_CATEGORY_SIGNALS = {
    "Retention": ["cancelling","switching to","competitor","cancel","leaving","last straw","keep us","what can you offer"],
    "Compliance": ["gdpr","ccpa","audit","compliance","regulatory","data privacy","soc2","data export","data deleted"],
    "Security": ["unauthorized","access control","data breach","suspicious","api keys","leaked",
                 "vulnerability","authentication bypass","stolen","identity","unknown ips","2fa"],
    "Billing": ["billing","charge","payment","invoice","plan","refund","subscription","pricing",
                "discount","charged","upgrade","annual"],
    "Technical": ["api","error","production","down","bug","dashboard","webhook","integration",
                  "sync","500","timeout","deploy","connection","service","throttled"],
}

def _classify_priority(text):
    t = text.lower()
    for p in ["Critical","High","Medium","Low"]:
        if any(kw in t for kw in _PRIORITY_SIGNALS[p]): return p
    return "Medium"

def _classify_category(text):
    t = text.lower()
    for c in ["Retention","Compliance","Security","Billing","Technical"]:
        if any(kw in t for kw in _CATEGORY_SIGNALS[c]): return c
    return "Technical"

# ── Agent functions (synchronous — safe to import anywhere) ──────────────────

def rule_based_agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    text = f"{obs.get('subject','')} {obs.get('body','')}"
    if obs.get("conversation_history"):
        for turn in obs["conversation_history"]:
            text += f" {turn.get('customer','')}"
    pri = _classify_priority(text)
    cat = _classify_category(text)
    esc = pri == "Critical"
    pv = obs.get("active_policy_version","v1")
    greeting = "Dear Customer" if pv in ("v2","v3","v4","v5","v6") else "Hi"
    subject = obs.get("subject","your issue")
    resp = (f"{greeting},\n\nThank you for reaching out regarding '{subject}'.\n\n"
            f"I understand your frustration and I sincerely apologise for the inconvenience. "
            f"We acknowledge this issue and will investigate and resolve it promptly. "
            f"Our team has been alerted.\n\nBest regards,\nCustomer Support Team")
    sent = obs.get("sentiment_score")
    if sent is not None and sent < 0.3:
        resp = (f"{greeting},\n\nI understand your frustration and I sincerely apologise. "
                f"Regarding '{subject}', we take this very seriously and will investigate immediately.\n\n"
                f"Best regards,\nCustomer Support Team")
    loyalty = obs.get("loyalty_score")
    if loyalty is not None and loyalty > 0.7 and cat == "Retention":
        resp = (f"{greeting},\n\nAs a valued customer, we truly appreciate your loyalty. "
                f"Regarding '{subject}', I understand your frustration. I'd like to offer a special "
                f"discount to address your concerns. We will investigate and resolve this.\n\n"
                f"Best regards,\nCustomer Support Team")
    action = {"assign_priority": pri, "assign_category": cat, "draft_response": resp, "escalate": esc}
    if cat == "Billing":
        action["approve_refund"] = True
    return action

_SYS = """You are an expert SaaS customer support triage agent. Return ONLY JSON:
{"assign_priority":"Low|Medium|High|Critical","assign_category":"Billing|Technical|Security|Compliance|Retention","draft_response":"...","escalate":true|false}
RULES: Critical=production down/data loss/security breach -> escalate. After v2: greeting REQUIRED, refund 14d. After v3: Security category, use sentiment. After v4: Compliance, PII redaction REQUIRED. After v5: Retention, loyalty offers. After v6: 7d refund, mandatory acknowledgment. NEVER cite rules not in SYSTEM NOTICE. NEVER include PII in responses."""

def llm_agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    """LLM-powered agent. Falls back to rule_based_agent if no client available."""
    if not HAS_LLM:
        return rule_based_agent(obs)
    parts = [
        f"Policy: {obs.get('active_policy_version','v1')}",
        f"Ticket: {obs.get('ticket_id','')}",
        f"Subject: {obs.get('subject','')}",
        f"Body:\n{obs.get('body','')}",
    ]
    if obs.get("system_notice"):
        parts.append(f"\nSYSTEM NOTICE:\n{obs['system_notice']}")
    for f in ["sentiment_score","account_age_days","loyalty_score","lifetime_value","region","contract_type"]:
        if obs.get(f) is not None:
            parts.append(f"{f}: {obs[f]}")
    ws = obs.get("world_state_summary", {})
    parts.append(
        f"\nWorld: bal=${ws.get('company_balance',10000):.0f} "
        f"churn={ws.get('churn_risk',0):.2f} "
        f"sla={ws.get('sla_breaches',0)} "
        f"diff={ws.get('difficulty_level',1)}"
    )
    try:
        r = _client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"system","content":_SYS},{"role":"user","content":"\n".join(parts)}],
            temperature=0.1,
            max_tokens=600,
        )
        raw = r.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        print(f"  [LLM error: {e}] -- fallback")
        return rule_based_agent(obs)


# ── Logging helpers ───────────────────────────────────────────────────────────

def log_start(task, env_type, model):
    print(f"[START] task={task} env={env_type} model={model}", flush=True)

def log_step(s, a, r, d, cat=False):
    print(f"[STEP] step={s} action={a} reward={r:.2f} done={str(d).lower()} catastrophic={str(cat).lower()}", flush=True)

def log_end(ok, steps, score, rewards, cat=False):
    print(f"[END] success={str(ok).lower()} steps={steps} score={score:.3f} catastrophic={str(cat).lower()} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


# ── Async episode runner (used only by main(), requires db) ───────────────────

async def run_episode(
    task_id: int = 2,
    env_type: str = "gauntlet",
    attacker_enabled: bool = True,
    agent_fn: Callable = None,
    model_name: str = "Rule-Based",
) -> float:
    """
    Full async episode runner with DB logging and attacker integration.
    NOT intended for use in synchronous eval/plotting scripts.
    Use GauntletEnv directly for synchronous rollouts.
    """
    if agent_fn is None:
        agent_fn = rule_based_agent
    env = GauntletEnv() if env_type == "gauntlet" else ShiftingSandsEnv()
    obs = env.reset(task_id=task_id)
    atk = None
    if attacker_enabled:
        atk = AttackerAgent(policy_registry=env.policy_registry)
        try:
            tickets = atk.generate_batch(
                n=12,
                difficulty_level=env.world_state.difficulty_level,
                defender_error_history=[],
                active_policy=env.policy_registry.get_active(),
            )
            env.set_attacker_tickets(tickets)
            obs = env._build_observation(tickets[0], None)
        except Exception as e:
            print(f"  [ATK ERROR] {e}")
            atk = None

    ep_id = await db.create_episode(
        session_id=env.session_id, task_id=task_id,
        attacker_enabled=attacker_enabled, drift_enabled=True,
        difficulty_init=env.world_state.difficulty_level, env_type=env_type,
    )
    log_start(task_id, env_type, model_name)
    rewards, steps = [], 0

    while True:
        try:
            action = agent_fn(obs)
            astr = json.dumps({k: v for k, v in action.items() if k != "draft_response"})
        except Exception:
            action = rule_based_agent(obs)
            astr = "fallback"

        tkt = (env._ticket_queue[min(env.world_state.tickets_processed, len(env._ticket_queue) - 1)]
               if env._ticket_queue else {})
        result = env.step(action)
        rv = result["reward"]
        rewards.append(rv)
        steps += 1
        done = result["done"]
        cat = result.get("catastrophic", False)
        log_step(steps, astr, rv, done, cat)

        af = result.get("attacker_fitness", 0.0)
        strategy = tkt.get("deception_strategy", "clean")
        await db.insert_step(
            episode_id=ep_id, step_number=steps, ticket_id=tkt.get("ticket_id",""),
            action=action, defender_reward=rv, attacker_fitness=af,
            breakdown=result.get("reward_breakdown",{}),
            policy_version=env.policy_registry.active_version_id(),
            was_post_drift=result.get("drift_notice") is not None,
            deception_strategy=strategy, template_index=tkt.get("template_index",-1),
        )
        await db.insert_snapshot(ep_id, steps, env.world_state)
        await db.insert_ticket_log(ep_id, steps, tkt, env.world_state.difficulty_level)
        if strategy != "clean":
            await db.insert_template_fitness(ep_id, steps, tkt.get("template_index",-1), strategy, af, "stay", rv)
        if result.get("drift_notice"):
            for evt in env.drift_scheduler.get_all_events():
                if evt.fires_at_step == steps:
                    await db.insert_drift_event(ep_id, steps, evt.from_version, evt.to_version, evt.drift_types, False)
        if atk and strategy != "clean":
            atk.update_elo(strategy, rv < 0)
        if done:
            break
        obs = result["observation"]

    score = sum(rewards) / max(steps, 1)
    log_end(score >= 0, steps, score, rewards, env.world_state.catastrophic_failure)
    await db.close_episode(ep_id, env.get_episode_metrics())
    return score


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    await db.init_db()
    agent = llm_agent if HAS_LLM else rule_based_agent
    name = MODEL_NAME if HAS_LLM else "Rule-Based"
    print("=" * 60)
    print("The Gauntlet + Shifting Sands -- Phase 2 (SaaS)")
    print("=" * 60)
    print("\n--- Gauntlet: Task 2 + Attacker ---")
    await run_episode(task_id=2, env_type="gauntlet", attacker_enabled=True, agent_fn=agent, model_name=name)
    print("\n--- Gauntlet: Task 2 Clean ---")
    await run_episode(task_id=2, env_type="gauntlet", attacker_enabled=False, agent_fn=agent, model_name=name)
    print("\n--- Shifting Sands: Task 2 + Attacker ---")
    await run_episode(task_id=2, env_type="shifting_sands", attacker_enabled=True, agent_fn=agent, model_name=name)
    print("\n--- Gauntlet: Task 3 Multi-Turn ---")
    await run_episode(task_id=3, env_type="gauntlet", attacker_enabled=False, agent_fn=agent, model_name=name)
    await db.close_db()
    print("\n[DONE] All episodes complete. Results stored in gauntlet.db")


if __name__ == "__main__":
    asyncio.run(main())