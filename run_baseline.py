import asyncio
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

from inference import run_episode, rule_based_agent
import db

NUM_EPISODES = 60

async def run_tests():
    await db.init_db()
    print(f"Running {NUM_EPISODES} baseline episodes (Rule-Based Matcher)...")
    
    for i in range(NUM_EPISODES):
        score = await run_episode(
            task_id=2, 
            attacker_enabled=True, 
            drift_enabled=True, 
            agent_fn=rule_based_agent, 
            model_name="Baseline-Matcher"
        )
        if (i+1) % 10 == 0:
            print(f"Completed {i+1}/{NUM_EPISODES} episodes. Last score: {score:.2f}")

    await db.close_db()

def plot_results():
    print("Generating plots...")
    os.makedirs("plots", exist_ok=True)
    
    conn = sqlite3.connect("gauntlet.db")
    
    df_episodes = pd.read_sql_query("SELECT * FROM episodes ORDER BY id", conn)
    df_steps = pd.read_sql_query("SELECT * FROM steps ORDER BY id", conn)
    df_snapshots = pd.read_sql_query("SELECT * FROM world_state_snapshots ORDER BY id", conn)
    
    if len(df_episodes) == 0:
        print("No episodes found. Did the tests run?")
        return
        
    plt.figure(figsize=(10, 5))
    df_episodes['reward_rolling'] = df_episodes['mean_defender_reward'].rolling(window=10, min_periods=1).mean()
    plt.plot(df_episodes.index, df_episodes['reward_rolling'], label='Defender Reward (10-ep rolling)')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('Defender Overall Reward (Baseline)')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/defender_reward.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    df_episodes['drift_rolling'] = df_episodes['drift_accuracy'].rolling(window=10, min_periods=1).mean()
    plt.plot(df_episodes.index, df_episodes['drift_rolling'], color='orange', label='Drift Accuracy (10-ep rolling)')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.title('Agent Drift Accuracy (Baseline)')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/drift_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    df_snapshots['interval'] = df_snapshots.index // 200
    stale_counts = df_snapshots.groupby('interval')['snapshot_json'].apply(
        lambda x: x.apply(lambda s: pd.Series(json.loads(s)).get('stale_decisions_made', 0)).max() 
        if len(x) > 0 else 0
    )
    stale_per_interval = stale_counts.diff().fillna(stale_counts)
    stale_per_interval.plot(kind='bar', color='red')
    plt.xlabel('Interval (200 steps)')
    plt.ylabel('Stale Decisions')
    plt.title('Stale Decision Rate (Baseline)')
    plt.grid(True, axis='y')
    plt.savefig('plots/stale_decisions.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df_episodes.index, df_episodes['attacker_win_rate_final'], color='purple', label='Attacker Win Rate (rolling 50-step window)')
    plt.axhline(0.5, color='black', linestyle='--')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title('Attacker Win Rate (Baseline)')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/attacker_win_rate.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df_episodes.index, df_episodes['difficulty_final'], color='green', label='Difficulty Level')
    plt.xlabel('Episode')
    plt.ylabel('Difficulty')
    plt.title('Difficulty Level Over Time (Baseline)')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/difficulty_level.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    sample_ep = df_episodes['id'].iloc[-1] 
    df_ep_snaps = df_snapshots[df_snapshots['episode_id'] == sample_ep]
    
    balances = [json.loads(s).get('company_balance', 10000) for s in df_ep_snaps['snapshot_json']]
    plt.plot(range(len(balances)), balances, marker='o', label=f'Balance (Ep {sample_ep})')
    plt.xlabel('Step')
    plt.ylabel('Company Balance ($)')
    plt.title('Company Balance Timeline (Single Episode)')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/company_balance.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved 6 plots to the 'plots/' directory.")
    conn.close()

if __name__ == "__main__":
    if os.path.exists("gauntlet.db"):
        os.remove("gauntlet.db")
        
    asyncio.run(run_tests())
    plot_results()
