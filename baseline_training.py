"""
baseline_training.py — Rule-based agent baseline over N episodes

Runs the rule-based agent through many episodes across all task modes,
collects per-step and per-episode reward data, and plots reward curves.

Usage:
    python baseline_training.py
"""

from __future__ import annotations

import re
import sys
import random
from typing import Any, Dict, List

# ─── Rule-based agent (self-contained, no LLM needed) ────────────────────────

_PRIORITY_SIGNALS = {
    "Critical": [
        "production down",
        "completely down",
        "system down",
        "data loss",
        "outage",
        "emergency",
        "critical",
        "urgent:",
        "all users locked",
        "unauthorized access",
        "api keys leaked",
        "security breach",
    ],
    "High": [
        "cannot log in",
        "can't log in",
        "charged twice",
        "duplicate charge",
        "hasn't arrived",
        "wrong item",
        "locked out",
        "suspicious login",
        "bypass vulnerability",
        "not received",
        "unauthorized",
    ],
    "Medium": [
        "sometimes",
        "occasionally",
        "slow",
        "seems off",
        "not sure",
        "acting weird",
        "permission errors",
        "sessions I don't recognize",
        "intermittent",
    ],
    "Low": [
        "question about",
        "interested in",
        "feedback",
        "how do i",
        "how to",
        "quick question",
        "no rush",
        "annual billing discount",
    ],
}

_CATEGORY_SIGNALS = {
    "Billing": [
        "billing",
        "charge",
        "payment",
        "invoice",
        "plan",
        "refund",
        "subscription",
        "pricing",
        "discount",
        "money",
    ],
    "Technical": [
        "api",
        "error",
        "production",
        "down",
        "bug",
        "dashboard",
        "webhook",
        "integration",
        "sync",
        "500",
        "timeout",
        "crash",
        "server",
    ],
    "Shipping": [
        "order",
        "arrived",
        "delivery",
        "ship",
        "tracking",
        "package",
        "dispatch",
        "received",
    ],
    "Security": [
        "unauthorized",
        "access control",
        "data breach",
        "suspicious",
        "api keys",
        "leaked",
        "vulnerability",
        "authentication bypass",
        "security",
    ],
    "Fraud": ["fraud", "fraudulent", "scam"],
    "Compliance": ["compliance", "regulation", "audit"],
}


def _classify_priority(text: str) -> str:
    t = text.lower()
    for p in ["Critical", "High", "Medium", "Low"]:
        if any(kw in t for kw in _PRIORITY_SIGNALS[p]):
            return p
    return "Medium"


def _classify_category(text: str, valid_categories) -> str:
    t = text.lower()
    for c in ["Security", "Fraud", "Compliance", "Billing", "Technical", "Shipping"]:
        if c in valid_categories and any(
            kw in t for kw in _CATEGORY_SIGNALS.get(c, [])
        ):
            return c
    # Fallback to a valid category
    return valid_categories[0] if valid_categories else "Technical"


def rule_based_agent(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Keyword-based heuristic agent."""
    text = f"{observation.get('subject', '')} {observation.get('body', '')}"
    if observation.get("conversation_history"):
        for turn in observation["conversation_history"]:
            text += f" {turn.get('customer', '')}"

    # Try to extract valid categories from the policy notice if available
    valid_cats = ["Billing", "Technical", "Shipping"]  # safe default

    priority = _classify_priority(text)
    category = _classify_category(text, valid_cats)
    escalate = priority == "Critical"

    subject = observation.get("subject", "your issue")
    greeting = "Dear Customer"

    response = (
        f"{greeting},\n\n"
        f"Thank you for reaching out regarding '{subject}'.\n\n"
        f"We will investigate this matter and resolve it promptly. "
        f"Our team has been alerted and will look into this issue. "
        f"We will follow up with a detailed update shortly.\n\n"
        f"Best regards,\nCustomer Support Team"
    )

    action = {
        "assign_priority": priority,
        "assign_category": category,
        "draft_response": response,
        "escalate": escalate,
    }

    sentiment = observation.get("sentiment_score")
    if sentiment is not None and sentiment < 0.3:
        action["draft_response"] = (
            f"{greeting},\n\n"
            f"I understand your frustration and I sincerely apologise for the inconvenience "
            f"regarding '{subject}'. We take this matter very seriously.\n\n"
            f"Our team will investigate and address this issue immediately. "
            f"We will follow up with a resolution as soon as possible.\n\n"
            f"Best regards,\nCustomer Support Team"
        )

    return action


# ─── Training loop ────────────────────────────────────────────────────────────


def run_baseline(
    num_episodes: int = 3,
    task_id: int = 2,
    drift_enabled: bool = True,
    attacker_enabled: bool = False,
):
    """Run N episodes and collect reward data."""
    from environment import CustomerSupportEnv

    env = CustomerSupportEnv()
    episode_rewards = []  # mean reward per episode
    episode_step_rewards = []  # all step rewards across all episodes
    difficulty_trace = []  # difficulty after each episode

    for ep in range(num_episodes):
        obs = env.reset(
            task_id=task_id,
            attacker_enabled=attacker_enabled,
            drift_enabled=drift_enabled,
            seed=ep,  # deterministic per episode
        )

        step_rewards = []
        done = False

        while not done:
            action = rule_based_agent(obs)
            result = env.step(action)

            step_rewards.append(result["reward"])
            done = result["done"]

            if not done:
                obs = result["observation"]

        mean_r = sum(step_rewards) / max(len(step_rewards), 1)
        episode_rewards.append(mean_r)
        episode_step_rewards.extend(step_rewards)
        difficulty_trace.append(env.world_state.difficulty_level)

        status = f"Episode {ep + 1:3d}/{num_episodes} | Mean Reward: {mean_r:+.3f} | Difficulty: {env.world_state.difficulty_level:.3f}"
        print(status, flush=True)

    return episode_rewards, episode_step_rewards, difficulty_trace


def plot_results(episode_rewards, step_rewards, difficulty_trace, title_suffix=""):
    """Generate and save reward curve plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ERROR] matplotlib not installed. Run: pip install matplotlib")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), facecolor="#1a1a2e")
    fig.suptitle(
        f"Rule-Based Agent Baseline{title_suffix}",
        fontsize=16,
        color="white",
        fontweight="bold",
    )

    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#444")

    # 1. Episode mean reward
    ax1 = axes[0]
    ax1.plot(
        range(1, len(episode_rewards) + 1),
        episode_rewards,
        color="#e94560",
        linewidth=1.5,
        alpha=0.7,
        label="Episode Reward",
    )
    # Rolling average
    window = min(10, len(episode_rewards))
    if window > 1:
        rolling = [
            sum(episode_rewards[max(0, i - window) : i + 1])
            / len(episode_rewards[max(0, i - window) : i + 1])
            for i in range(len(episode_rewards))
        ]
        ax1.plot(
            range(1, len(rolling) + 1),
            rolling,
            color="#0f3460",
            linewidth=2.5,
            label=f"{window}-Episode Rolling Avg",
        )
    ax1.axhline(y=0, color="#444", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("Per-Episode Mean Reward")
    ax1.legend(facecolor="#16213e", edgecolor="#444", labelcolor="white")

    # 2. Step-level rewards (scatter)
    ax2 = axes[1]
    ax2.scatter(range(len(step_rewards)), step_rewards, s=2, alpha=0.3, color="#e94560")
    # Rolling step average
    step_window = min(50, len(step_rewards))
    if step_window > 1:
        step_rolling = [
            sum(step_rewards[max(0, i - step_window) : i + 1])
            / len(step_rewards[max(0, i - step_window) : i + 1])
            for i in range(len(step_rewards))
        ]
        ax2.plot(
            range(len(step_rolling)),
            step_rolling,
            color="#0f3460",
            linewidth=1.5,
            label=f"{step_window}-Step Rolling Avg",
        )
    ax2.axhline(y=0, color="#444", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Step (global)")
    ax2.set_ylabel("Reward")
    ax2.set_title("Per-Step Reward Distribution")
    ax2.legend(facecolor="#16213e", edgecolor="#444", labelcolor="white")

    # 3. Difficulty trace
    ax3 = axes[2]
    ax3.plot(
        range(1, len(difficulty_trace) + 1),
        difficulty_trace,
        color="#533483",
        linewidth=2,
    )
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Difficulty Level")
    ax3.set_title("Curriculum Difficulty Over Time")
    ax3.set_ylim(-0.05, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = "baseline_reward_curve.png"
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[SAVED] Reward curves saved to {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    NUM_EPISODES = 3

    print("=" * 60)
    print("Baseline Training — Rule-Based Agent")
    print("=" * 60)

    # Run 1: Task 2, drift enabled, no attacker
    print("\n--- Run 1: Task 2 + Drift (no attacker) ---")
    ep_r, step_r, diff_t = run_baseline(
        num_episodes=NUM_EPISODES,
        task_id=2,
        drift_enabled=True,
        attacker_enabled=False,
    )
    plot_results(ep_r, step_r, diff_t, " — Task 2 + Drift")

    # Print summary stats
    print(f"\n{'=' * 40}")
    print(f"  Episodes:         {NUM_EPISODES}")
    print(f"  Mean Reward:      {sum(ep_r) / len(ep_r):+.4f}")
    print(f"  Best Episode:     {max(ep_r):+.4f}")
    print(f"  Worst Episode:    {min(ep_r):+.4f}")
    print(f"  Final Difficulty: {diff_t[-1]:.4f}")
    print(f"{'=' * 40}")
