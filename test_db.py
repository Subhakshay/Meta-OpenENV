import requests
import time

BASE = "http://localhost:7860"

# Start episode
print(f"Starting new episode...")
session = requests.post(
    f"{BASE}/reset", json={"task_id": "task_1_priority", "seed": 42}
).json()

session_id = session["session_id"]
print("Session ID:", session_id)

done = False
step = 0
while not done:
    step += 1
    result = requests.post(
        f"{BASE}/step",
        json={
            "session_id": session_id,
            "assign_priority": "critical",
            "assign_category": "technical",
            "response_text": "Hello, our team is investigating this urgent issue immediately. We will follow up shortly. Best regards, Support Team",
            "escalate": True,
        },
    ).json()

    done = result["done"]
    print(f"Step {step} -> Reward: {result['reward']['value']:.2f} | Done: {done}")

print("\nEpisode finished! Fetching /episodes endpoint:")
# The DB save_episode/close_episode logic in main.py is asynchronous (fire-and-forget),
# so we wait a tiny bit to make sure the row is written before fetching.
time.sleep(0.5)

episodes = requests.get(f"{BASE}/episodes").json()
print("Episodes count:", episodes["count"])
if episodes["count"] > 0:
    for ep in episodes["episodes"]:
        print(
            f" - ID: {ep['episode_id']} | Total Reward: {ep.get('total_reward'):.2f} | Steps: {ep.get('step_count')} | Closed At: {ep.get('closed_at')}"
        )
