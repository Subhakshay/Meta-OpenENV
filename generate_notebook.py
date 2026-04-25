import json

cells = []

def add_markdown(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [source]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\\n" if i < len(source.split("\\n")) - 1 else line for i, line in enumerate(source.split("\\n"))]
    })

# 1. Install
add_markdown("## 1. Install")
install_code = """import warnings
import transformers.logging

# Suppress specified warnings
warnings.filterwarnings("ignore", message=".*Both `max_new_tokens` and `max_length`.*")
warnings.filterwarnings("ignore", message=".*attention mask.*")
transformers.logging.set_verbosity_error()

!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
!pip install trl peft accelerate bitsandbytes matplotlib nest_asyncio -q
"""
add_code(install_code)

# 2. Load Model
add_markdown("## 2. Load Model\\nLoad unsloth/Qwen2.5-1.5B-Instruct in 4-bit. Do NOT call FastLanguageModel.for_inference() before training.")
load_model_code = """from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None # None for auto detection
load_in_4bit = True # Use 4bit quantization

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
"""
add_code(load_model_code)

# 3. Add LoRA
add_markdown("## 3. Add LoRA")
lora_code = """model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
"""
add_code(lora_code)

# 4. Reward Function
add_markdown("## 4. Reward Function\\nSynchronous reward function parsing JSON and running CustomerSupportEnv.")
reward_fn_code = """import asyncio
import json
import re
from environment import CustomerSupportEnv

import nest_asyncio
nest_asyncio.apply()

def _score_one(completion_text: str) -> float:
    env = CustomerSupportEnv()
    env.reset(task_id=2, attacker_enabled=True, drift_enabled=True)
    
    # Parse JSON using regex
    match = re.search(r'\\{.*?\\}', completion_text, re.DOTALL)
    action = None
    if match:
        try:
            action = json.loads(match.group(0))
        except:
            pass
            
    if not action:
        # Fallback action
        action = {
            "assign_priority": "Medium",
            "assign_category": "Technical",
            "draft_response": "placeholder",
            "escalate": False,
            "approve_refund": None
        }
        
    result = env.step(action)
    return float(result["reward"])

def gauntlet_reward_fn(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        try:
            loop = asyncio.new_event_loop()
            r = loop.run_until_complete(asyncio.to_thread(_score_one, completion[0]['content'] if isinstance(completion, list) else completion))
            loop.close()
            rewards.append(r)
        except Exception as e:
            print(f"Exception in reward function: {e}")
            rewards.append(-1.0)
    return rewards
"""
add_code(reward_fn_code)

# 5. Agent Logic
add_markdown("## 5. Agent Logic\\nEvaluation loop agent using unsloth FastLanguageModel.")
agent_logic_code = """SYSTEM_PROMPT = \"\"\"You are an expert customer support triage agent in The Gauntlet environment.
You must classify tickets and respond to customers. Return ONLY a valid JSON object:
{
  "assign_priority": "Low" | "Medium" | "High" | "Critical",
  "assign_category": "Billing" | "Technical" | "Shipping" | "Security" | "Fraud" | "Compliance",
  "draft_response": "<professional reply to customer>",
  "escalate": true | false,
  "approve_refund": true | false | null
}
Read the system notice and ticket carefully and follow all active policy rules.\"\"\"

def unsloth_agent(observation):
    obs_str = json.dumps(observation, indent=2)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Observation:\\n{obs_str}\\n\\nAction JSON:"}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(input_ids=inputs, max_new_tokens=256, use_cache=True)
    response_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    
    match = re.search(r'\\{.*?\\}', response_text, re.DOTALL)
    action = None
    if match:
        try:
            action = json.loads(match.group(0))
        except:
            pass
            
    if not action:
        action = {
            "assign_priority": "Medium",
            "assign_category": "Technical",
            "draft_response": "Thank you for your message. We are looking into this.",
            "escalate": False,
            "approve_refund": None
        }
    return action
"""
add_code(agent_logic_code)

# 6. Build Dataset
add_markdown("## 6. Build Dataset\\n100 prompts using dummy actions to advance the environment.")
dataset_code = """from datasets import Dataset

env = CustomerSupportEnv()
obs = env.reset(task_id=2, attacker_enabled=True, drift_enabled=True)

prompts_data = []
dummy_action = {
    "assign_priority": "Medium",
    "assign_category": "Technical",
    "draft_response": "placeholder",
    "escalate": False,
    "approve_refund": None
}

for _ in range(100):
    prompt_text = f"Observation:\\n{json.dumps(obs, indent=2)}\\n\\nAction JSON:"
    prompts_data.append({"prompt": prompt_text})
    
    result = env.step(dummy_action)
    obs = result["observation"]
    
    if result["done"]:
        obs = env.reset(task_id=2, attacker_enabled=True, drift_enabled=True)

dataset = Dataset.from_list(prompts_data)
"""
add_code(dataset_code)

# 7. GRPO Training
add_markdown("## 7. GRPO Training")
training_code = """from trl import GRPOConfig, GRPOTrainer

# TRL formatting function for GRPO
def format_prompts(example):
    return [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]}
        ]
    ]

dataset = dataset.map(lambda x: {"prompt": format_prompts(x)[0]})

training_args = GRPOConfig(
    output_dir="gauntlet_grpo_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_steps=1,
    save_steps=100,
    max_completion_length=256,
    report_to="none"
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[gauntlet_reward_fn],
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()

# Extract and print reward logs
print("\\n--- Reward Logs ---")
reward_logs = []
available_keys = set()
for log in trainer.state.log_history:
    available_keys.update(log.keys())
    for key in ['reward', 'train/reward', 'rewards', 'mean_reward']:
        if key in log:
            reward_logs.append(log[key])
            break

print(f"Total reward data points found: {len(reward_logs)}")
print(f"Available log keys: {available_keys}")
"""
add_code(training_code)

# 8. Post-Training Evaluation
add_markdown("## 8. Post-Training Evaluation\\nRe-enable fast inference and run 30 episodes.")
eval_code = """import os
FastLanguageModel.for_inference(model)

episode_rewards = []
step_rewards = []
difficulty_trace = []

for ep in range(30):
    env = CustomerSupportEnv()
    obs = env.reset(task_id=2, attacker_enabled=True, drift_enabled=True)
    done = False
    
    ep_reward_sum = 0
    steps = 0
    
    while not done:
        action = unsloth_agent(obs)
        result = env.step(action)
        
        r = float(result["reward"])
        ep_reward_sum += r
        step_rewards.append(r)
        
        obs = result["observation"]
        done = result["done"]
        steps += 1
        
    mean_r = ep_reward_sum / steps if steps > 0 else 0
    episode_rewards.append(mean_r)
    
    try:
        diff = float(env.world_state.difficulty_level)
        assert 0.0 <= diff <= 1.0
    except Exception:
        diff = difficulty_trace[-1] if difficulty_trace else 0.5
    difficulty_trace.append(diff)
    
    print(f"Ep {ep+1}/30 | Reward: {mean_r:+.3f} | Difficulty: {diff:.3f}")
    
    # Save progress iteratively
    with open("eval_progress.json", "w") as f:
        json.dump({
            "episode_rewards": episode_rewards,
            "step_rewards": step_rewards,
            "difficulty_trace": difficulty_trace
        }, f)
"""
add_code(eval_code)

# 9. Plot Results
add_markdown("## 9. Plot Results\\nDark theme 3-subplot figure.")
plot_code = """import matplotlib.pyplot as plt
import numpy as np

# Load progress just in case this cell is run independently
try:
    with open("eval_progress.json", "r") as f:
        data = json.load(f)
        episode_rewards = data.get("episode_rewards", [])
        step_rewards = data.get("step_rewards", [])
        difficulty_trace = data.get("difficulty_trace", [])
except Exception:
    pass

fig, axes = plt.subplots(3, 1, figsize=(10, 15), facecolor="#1a1a2e")

for ax in axes:
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#444")

# Plot 1: Per-Episode Mean Reward
ax1 = axes[0]
episodes = range(1, len(episode_rewards) + 1)
ax1.plot(episodes, episode_rewards, color="#00d2ff", alpha=0.7, linewidth=1.5, label="Episode Reward")

# 5-episode rolling average
if len(episode_rewards) >= 5:
    rolling_5 = np.convolve(episode_rewards, np.ones(5)/5, mode='valid')
    ax1.plot(range(5, len(episode_rewards) + 1), rolling_5, color="#3a7bd5", linewidth=2.5)

ax1.axhline(y=0, color="#444", linestyle="--")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Mean Reward")
ax1.set_title("Per-Episode Mean Reward")

# Plot 2: Per-Step Reward Distribution
ax2 = axes[1]
steps = range(1, len(step_rewards) + 1)
ax2.scatter(steps, step_rewards, s=2, alpha=0.5, color="#00d2ff")

# 20-step rolling average
if len(step_rewards) >= 20:
    rolling_20 = np.convolve(step_rewards, np.ones(20)/20, mode='valid')
    ax2.plot(range(20, len(step_rewards) + 1), rolling_20, color="#3a7bd5", linewidth=1.5)

ax2.axhline(y=0, color="#444", linestyle="--")
ax2.set_xlabel("Step (global)")
ax2.set_ylabel("Reward")
ax2.set_title("Per-Step Reward Distribution")

# Plot 3: Curriculum Difficulty Over Time
ax3 = axes[2]
valid_diff = [d for d in difficulty_trace if isinstance(d, (int, float))]
if len(valid_diff) > 0:
    ax3.plot(range(1, len(valid_diff) + 1), valid_diff, color="#b620e0", linewidth=2)
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Difficulty Level")
    ax3.set_title("Curriculum Difficulty Over Time")
else:
    ax3.set_title("Curriculum Difficulty Over Time")
    ax3.text(0.5, 0.5, "Difficulty data unavailable", color="white", ha="center", va="center", transform=ax3.transAxes)

plt.tight_layout()
plt.savefig("gauntlet_training_results.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
"""
add_code(plot_code)

notebook = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("train.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("train.ipynb successfully generated!")
