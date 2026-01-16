#!/usr/bin/env python3
"""
Debug Training Script for DTDE Spoiled Broth - Episode-by-Episode Data Capture

This script runs training for a small number of epochs while capturing complete
simulation data (actions, observations, state, etc.) for each episode. This allows
detailed debugging of what happens during training.

Usage: python training-debug-DTDE-spoiled_broth.py <input_path> <map_nr> <lr> <game_version> [<num_agents>] [<checkpoint_id>] [<pretraining>] [<seed>]

Example:
nohup python training-debug-DTDE-spoiled_broth.py cuenca cuenca/input_0_0.txt baseline_division_of_labor_v2 0.0003 classic 2 5 0 none > debug_training.log 2>&1 &

This will create a debug training folder with:
- Regular training checkpoints and stats
- Episode-by-episode simulation data in episodes/ subfolder
  - Each episode gets: simulation.csv, actions.csv, observations.csv, counters.csv, config.txt
- Summary analysis of all episodes

Key differences from regular training:
1. Limited epochs (default 10) for quick debugging
2. Captures full simulation data for every episode
3. Saves detailed episode analysis
4. More frequent progress reporting
"""

import os
import sys
from spoiled_broth.rl.make_train_rllib_debug import make_train_rllib_debug
import ray
import torch
from datetime import datetime
import pandas as pd
from pathlib import Path

# PyTorch, NumPy, MKL, etc. not creating more threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

##### Cluster config ##################
NUM_GPUS = 0.2
NUM_CPUS = 10

# DEBUG SETTINGS
DEBUG_MAX_EPOCHS = 100 # Maximum number of training epochs for debugging
DEBUG_CAPTURE_ALL_EPISODES = True  # Capture simulation data for every episode
DEBUG_SAVE_EVERY_N_EPOCHS = 100  # Save checkpoint more frequently for debugging

print("=" * 60)
print("DEBUG TRAINING MODE - EPISODE DATA CAPTURE")
print("=" * 60)
print(f"This debug script will:")
print(f"- Train for a maximum of {DEBUG_MAX_EPOCHS} epochs")
print(f"- Capture complete simulation data for {'ALL' if DEBUG_CAPTURE_ALL_EPISODES else 'SELECTED'} episodes")
print(f"- Save checkpoints every {DEBUG_SAVE_EVERY_N_EPOCHS} epochs")
print(f"- Create detailed episode analysis files")
print("=" * 60)

# Read input file
CLUSTER = str(sys.argv[1]).lower()
input_path = sys.argv[2]
MAP_NR = str(sys.argv[3]).lower()
LR = float(sys.argv[4])
GAME_VERSION = str(sys.argv[5]).lower() ## If game_version = classic, one type of food (tomato); if game_version = competition, two types of food (tomato and pumpkin)
if len(sys.argv) > 6:
    NUM_AGENTS = int(sys.argv[6])
    if NUM_AGENTS not in [1, 2]:
        raise ValueError("NUM_AGENTS must be 1 or 2")
else:
    NUM_AGENTS = 2  # Default to 2 agents for backward compatibility

NUM_EPOCHS = int(sys.argv[7]) if len(sys.argv) > 7 else 500
SEED = int(sys.argv[8]) if len(sys.argv) > 8 else 0

# Optional checkpoint paths for loading pretrained policies (CHECKPOINT_PATHS should be a file with two lines per agent)
# Line 1: policy_id_to_be_loaded (policy_ai_rl_1, policy_ai_rl_2, etc.)
# Line 2: path_to_checkpoint
CHECKPOINT_PATHS = sys.argv[9] if len(sys.argv) > 9 else "none"

with open(input_path, "r") as f:
    lines = f.readlines()
    alpha_1, beta_1 = [round(float(x), 4) for x in lines[0].strip().split()]
    walking_speed_1, cutting_speed_1 = [round(float(x), 4) for x in lines[1].strip().split()]
    if NUM_AGENTS == 2:
        alpha_2, beta_2 = [round(float(x), 4) for x in lines[2].strip().split()]
        walking_speed_2, cutting_speed_2 = [round(float(x), 4) for x in lines[3].strip().split()]    

if CLUSTER == 'brigit':
    local = '/mnt/lustre/home/samuloza'
elif CLUSTER == 'cuenca':
    local = ''
elif CLUSTER == 'local':
    local = 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'
else:
    raise ValueError("Invalid cluster specified. Choose from 'brigit', 'cuenca', or 'local'.")

# Hyperparameters
NUM_ENVS = 1
INNER_SECONDS = 180  # Shorter episodes for debugging
NUM_EPOCHS = DEBUG_MAX_EPOCHS  # Limited epochs for debugging
TRAIN_BATCH_SIZE = 200 # Smaller batch size for debugging
NUM_MINIBATCHES = 10  # Fewer minibatches for faster debugging
SHOW_EVERY_N_EPOCHS = 100
SAVE_EVERY_N_EPOCHS = DEBUG_SAVE_EVERY_N_EPOCHS
PAYOFF_MATRIX = [1,1,-2]

# Neural network architecture
MLP_LAYERS = [512, 512, 256]

# Game characteristics
PENALTIES_CFG = {
    "busy": 0.01, # Penalty per second spent busy
    "useless_action": 2.0, # Penalty for useless actions
    "destructive_action": 10.0, # Penalty for destructive actions
    "inaccessible_tile": 5.0, # Penalty for trying to access an inaccessible tile
    "not_available": 2.0, # Penalty for trying to perform an action that is not available
}

REWARDS_CFG = {
    "raw_food": 2.0,
    "plate": 2.0,
    "counter": 0.5,
    "cut": 5.0,
    "salad": 7.0,
    "deliver": 10.0,
}

# Dynamic rewards configuration - exponential decay [rewards_cfg = original_rewards_cfg * exp(-decay_rate * (episode - decay_start_episode))]
DYNAMIC_REWARDS_CFG = {
    "enabled": True,  # Disabled for debugging to keep things simple
    "decay_rate": 0.0001,
    "min_reward_multiplier": 0.0,
    "decay_start_episode": 5000,
    "affected_rewards": ["raw_food", "plate", "counter", "cut", "salad"],
}

# Dynamic PPO parameters configuration
DYNAMIC_PPO_PARAMS_CFG = {
    "enabled": False,  # Disabled for debugging to keep things simple
    "decay_rate": 0.0001,
    "min_param_multiplier": 0.1,
    "decay_start_episode": 100,
    "affected_params": ["ent_coef"],
}

WAIT_FOR_ACTION_COMPLETION = True  # Flag to ensure actions complete before next step

# Path definitions - use debug_training for all scenarios
save_dir = f'{local}/data/samuel_lozano/cooked/debug_training/{GAME_VERSION}/map_{MAP_NR}'

# Path definitions
if NUM_AGENTS == 1:
    reward_weights = {"ai_rl_1": (globals()[f"alpha_1"], globals()[f"beta_1"])}
    walking_speeds = {"ai_rl_1": globals()[f"walking_speed_1"]}
    cutting_speeds = {"ai_rl_1": globals()[f"cutting_speed_1"]}
else: 
    for i in range(1, NUM_AGENTS + 1):
        reward_weights = {f"ai_rl_{i}": (globals()[f"alpha_{i}"], globals()[f"beta_{i}"])}
        walking_speeds = {f"ai_rl_{i}": globals()[f"walking_speed_{i}"]}
        cutting_speeds = {f"ai_rl_{i}": globals()[f"cutting_speed_{i}"]}

os.makedirs(save_dir, exist_ok=True)

pretrained_policies = None
# Load pretrained policies if specified
if CHECKPOINT_PATHS != "none":
    pretrained_policies = {}
    with open(input_path, "r") as f:
        lines = f.readlines()
        # Note: Debug script doesn't support agent_to_train parameter yet
        # This assumes standard agent numbering for now
        for i in range(NUM_AGENTS):
            policy_id = str(lines[i]).strip()
            CHECKPOINT_PATH = str(lines[i+1]).strip()
            if policy_id.lower() != "none" and CHECKPOINT_PATH.lower() != "none":
                pretrained_policies[f"ai_rl_{i+1}"] = {"loaded_agent_id": policy_id, "path": CHECKPOINT_PATH}
            else:
                pretrained_policies[f"ai_rl_{i+1}"] = None

# Determine grid size from map file (text format)
map_txt_path = os.path.join(os.path.dirname(__file__), 'spoiled_broth', 'maps', f'{MAP_NR}.txt')
if not os.path.exists(map_txt_path):
    raise FileNotFoundError(f"Map file {map_txt_path} not found.")
with open(map_txt_path, 'r') as f:
    map_lines = [line.rstrip('\n') for line in f.readlines()]
rows = len(map_lines)
cols = len(map_lines[0]) if rows > 0 else 0
if rows != cols:
    raise ValueError(f"Map must be square, but got {rows} rows and {cols} columns.")
GRID_SIZE = (rows, cols)

# RLlib specific configuration
config = {
    "NUM_ENVS": NUM_ENVS,
    "INNER_SECONDS": INNER_SECONDS,
    "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
    "NUM_MINIBATCHES": NUM_MINIBATCHES,
    "NUM_EPOCHS": NUM_EPOCHS,
    "NUM_AGENTS": NUM_AGENTS,
    "SHOW_EVERY_N_EPOCHS": SHOW_EVERY_N_EPOCHS,
    "SAVE_EVERY_N_EPOCHS": SAVE_EVERY_N_EPOCHS,
    "LR": LR,
    "MAP_NR": MAP_NR,
    "REWARD_WEIGHTS": reward_weights,
    "GAME_VERSION": GAME_VERSION,
    "PAYOFF_MATRIX": PAYOFF_MATRIX,
    "WALKING_SPEEDS": walking_speeds,
    "CUTTING_SPEEDS": cutting_speeds,
    "INITIAL_SEED": SEED,
    "PENALTIES_CFG": PENALTIES_CFG,
    "REWARDS_CFG": REWARDS_CFG,
    "DYNAMIC_REWARDS_CFG": DYNAMIC_REWARDS_CFG,
    "DYNAMIC_PPO_PARAMS_CFG": DYNAMIC_PPO_PARAMS_CFG,
    "WAIT_FOR_COMPLETION": WAIT_FOR_ACTION_COMPLETION,
    "SAVE_DIR": save_dir,
    "CHECKPOINTS": pretrained_policies,
    # RLlib specific parameters
    "NUM_UPDATES": 10,
    "GAMMA": 0.975,
    "GAE_LAMBDA": 0.95,
    "ENT_COEF": 0.07,
    "CLIP_EPS": 0.2,
    "VF_COEF": 0.5,
    "GRID_SIZE": GRID_SIZE,
    "FCNET_HIDDENS": MLP_LAYERS,
    "FCNET_ACTIVATION": "tanh",
    "NUM_CPUS": NUM_CPUS,
    "NUM_GPUS": NUM_GPUS,
    
    # DEBUG-SPECIFIC CONFIG
    "DEBUG_MODE": True,
    "DEBUG_CAPTURE_EPISODES": DEBUG_CAPTURE_ALL_EPISODES,
    "DEBUG_MAX_EPISODES_TO_CAPTURE": 100,  # Limit to avoid too many files
}

if ray.is_initialized():
    ray.shutdown()
ray.init(num_cpus=NUM_CPUS)

torch.set_num_threads(NUM_CPUS)

print("Starting debug training with episode data capture...")
print(f"Configuration:")
print(f"  Map: {MAP_NR}")
print(f"  Agents: {NUM_AGENTS}")
print(f"  Game Version: {GAME_VERSION}")
print(f"  Episode Duration: {INNER_SECONDS} seconds")
print(f"  Max Epochs: {NUM_EPOCHS}")
print(f"  Batch Size: {TRAIN_BATCH_SIZE}")
print(f"  Walking Speeds: {walking_speeds}")
print(f"  Cutting Speeds: {cutting_speeds}")
print(f"  Reward Weights: {reward_weights}")
print(f"  Agent IDs: {list(reward_weights.keys())}")

# Run training
trainer, current_date, final_episode_count, episode_data_summary = make_train_rllib_debug(config)

# Save the final model
path = os.path.join(config["SAVE_DIR"], f"DebugTraining_{current_date}")
os.makedirs(path, exist_ok=True)

# Update config with final episode count
if final_episode_count is not None:
    config["NUM_EPISODES"] = final_episode_count
    
    # Append the final episode count to config.txt
    config_path = os.path.join(path, "config.txt")
    with open(config_path, "a") as f:
        f.write(f"NUM_EPISODES: {final_episode_count}\n")

# Save the final policy
final_checkpoint_result = trainer.save(os.path.join(path, f"checkpoint_final"))
final_checkpoint_path = final_checkpoint_result.checkpoint.path        
print(f"Final checkpoint saved at {final_checkpoint_path}")
if final_episode_count is not None:
    print(f"Debug training completed after {final_episode_count} episodes")

# Create summary analysis of all captured episodes
if episode_data_summary:
    print("=" * 60)
    print("EPISODE DATA SUMMARY")
    print("=" * 60)
    
    episodes_dir = Path(path) / "episodes"
    summary_file = episodes_dir / "episodes_summary.csv"
    
    # Convert summary data to DataFrame and save
    if episode_data_summary:
        summary_df = pd.DataFrame(episode_data_summary)
        summary_df.to_csv(summary_file, index=False)
        print(f"Episode summary saved to: {summary_file}")
        
        # Print key statistics
        print(f"\nCaptured {len(episode_data_summary)} episodes")
        if len(episode_data_summary) > 0:
            print("Episode Statistics:")
            # Use actual agent IDs from reward_weights to avoid hardcoded numbering
            agent_ids = list(reward_weights.keys()) if reward_weights else [f"ai_rl_{i+1}" for i in range(NUM_AGENTS)]
            for agent_id in agent_ids:
                if f"total_actions_{agent_id}" in summary_df.columns:
                    agent_actions = summary_df[f"total_actions_{agent_id}"]
                    print(f"  {agent_id} actions per episode: avg={agent_actions.mean():.1f}, min={agent_actions.min()}, max={agent_actions.max()}")
                
                if f"total_rewards_{agent_id}" in summary_df.columns:
                    agent_rewards = summary_df[f"total_rewards_{agent_id}"]
                    print(f"  {agent_id} rewards per episode: avg={agent_rewards.mean():.2f}, min={agent_rewards.min():.2f}, max={agent_rewards.max():.2f}")

print(f"\nDebug training completed!")
print(f"Main training directory: {path}")
print(f"Use these files to debug what happens during training episodes.")