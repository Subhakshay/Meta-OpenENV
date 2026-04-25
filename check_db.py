"""Quick DB inspector for gauntlet.db"""
import sqlite3, json

conn = sqlite3.connect("gauntlet.db")
conn.row_factory = sqlite3.Row

print("=" * 60)
print("EPISODES")
print("=" * 60)
for r in conn.execute("SELECT * FROM episodes").fetchall():
    d = dict(r)
    print(f"  Episode {d['id']}: env={d['env_type']} task={d['task_id']} attacker={d['attacker_enabled']}")
    print(f"    reward={d['mean_defender_reward']:.3f}  balance={d['final_balance']:.0f}  sla={d['sla_breaches']}")
    print(f"    difficulty={d['difficulty_final']}  rounds={d['rounds_survived']}  catastrophic={d['catastrophic_failure']}")
    if d['catastrophic_reason']:
        print(f"    reason: {d['catastrophic_reason']}")
    print()

print("=" * 60)
print("STEPS SUMMARY")
print("=" * 60)
total = conn.execute("SELECT COUNT(id) FROM steps").fetchone()[0]
print(f"  Total steps logged: {total}")
for r in conn.execute("SELECT episode_id, COUNT(id) as cnt, AVG(defender_reward) as avg_r FROM steps GROUP BY episode_id").fetchall():
    print(f"  Episode {r['episode_id']}: {r['cnt']} steps, avg_reward={r['avg_r']:.3f}")

print()
print("=" * 60)
print("TEMPLATE FITNESS (top attackers)")
print("=" * 60)
rows = conn.execute("SELECT template_index, strategy, fitness_score, defender_reward FROM template_fitness ORDER BY fitness_score DESC LIMIT 10").fetchall()
if rows:
    for r in rows:
        print(f"  Template #{r['template_index']} ({r['strategy']}): fitness={r['fitness_score']:.2f}  def_reward={r['defender_reward']:.2f}")
else:
    print("  No template fitness data yet.")

print()
print("=" * 60)
print("DEFENDER WEAKNESSES (strategies that beat the defender)")
print("=" * 60)
rows = conn.execute("SELECT deception_strategy, COUNT(id) as cnt, AVG(defender_reward) as avg_r FROM steps WHERE defender_reward < 0 GROUP BY deception_strategy ORDER BY cnt DESC").fetchall()
if rows:
    for r in rows:
        print(f"  {r['deception_strategy']}: {r['cnt']} failures, avg_reward={r['avg_r']:.2f}")
else:
    print("  No weaknesses recorded yet.")

print()
print("=" * 60)
print("DRIFT EVENTS")
print("=" * 60)
rows = conn.execute("SELECT * FROM drift_events").fetchall()
if rows:
    for r in rows:
        print(f"  Episode {r['episode_id']} Step {r['step_number']}: {r['from_version']} -> {r['to_version']}")
else:
    print("  No drift events recorded.")

print()
print("=" * 60)
print("TABLE ROW COUNTS")
print("=" * 60)
for t in ["episodes","steps","world_state_snapshots","tickets_log","drift_events","template_fitness"]:
    c = conn.execute(f"SELECT COUNT(id) FROM {t}").fetchone()[0]
    print(f"  {t}: {c} rows")

conn.close()
