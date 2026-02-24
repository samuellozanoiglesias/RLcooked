# USE:   <cluster> <input_path> <map_nr> <lr> <game_version> [<num_agents>] [<num_epochs>] [<seed>] [<checkpoints>] [<rewards_on_delivery_only>] [<random_initial_state>] [<synergy_scaling_factor>] [<specialization_penalty_scale>] [<collision_penalty>] [<agent_to_train>] [<allow_blocked>] [<kappa>] [<activate_synergy_positive>] [<adaptive_cooperation_scale>] [<collision_harshness>] > log_training.log 2>&1 &
# Example: nohup python training-DTDE-spoiled_broth.py cuenca ./cuenca/input_0_0.txt baseline_division_of_labor_v2 0.0003 classic 2 1000 0 none true false 0.5 5.0 10.0 false 1.0 true 0.5 2.0 > log_training.log 2>&1 &
#   synergy_scaling_factor=0: Standard rewards (no team synergy shaping)
#   synergy_scaling_factor>0: Team synergy-based reward shaping enabled with given sensitivity
#   specialization_penalty_scale=0: No specialization penalty
#   specialization_penalty_scale>0: Specialization penalty with given scale
#   kappa: Competence transformation parameter for team synergy distribution (default=1.0)
#   activate_synergy_positive: Whether to apply positive synergy signals (true/false, default=false)
#   adaptive_cooperation_scale: Path-length-dependent cooperation penalty for slow walkers (0=disabled, >0=enabled, multiplied by collision_harshness when collisions enabled)
#   collision_harshness: Multiplier for specialization and adaptive cooperation when collisions enabled (1.0=same, 2.0=double, default=2.0)

import os
import sys
from spoiled_broth.rl.make_train_rllib import make_train_rllib
import ray
import torch
import pandas as pd

# PyTorch, NumPy, MKL, etc. not creating more threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Read input file
CLUSTER = str(sys.argv[1]).lower()
INPUT_PATH = sys.argv[2]
MAP_NR = str(sys.argv[3]).lower()
LR = float(sys.argv[4])
GAME_VERSION = str(sys.argv[5]).lower() ## If game_version = classic, one type of food (tomato); if game_version = competition, two types of food (tomato and pumpkin); if game_version ends with '_collision', enables collision detection
NUM_AGENTS = int(sys.argv[6])
if NUM_AGENTS not in [1, 2]:
    raise ValueError("NUM_AGENTS must be 1 or 2")
NUM_EPOCHS = int(sys.argv[7]) if len(sys.argv) > 7 else 500
SEED = int(sys.argv[8]) if len(sys.argv) > 8 else 0

# Optional checkpoint paths for loading pretrained policies (CHECKPOINT_PATHS should be a file with three lines per agent)
# Line 1: policy_id_to_be_loaded (policy_ai_rl_1, policy_ai_rl_2, etc.)
# Line 2: checkpoint_number
# Line 3: path_to_checkpoint
CHECKPOINT_PATHS = str(sys.argv[9]).lower() if len(sys.argv) > 9 else "none"
REWARDS_ON_DELIVERY_ONLY = str(sys.argv[10]).lower() if len(sys.argv) > 10 else "true"
RANDOM_INITIAL_STATE = str(sys.argv[11]).lower() if len(sys.argv) > 11 else "false"  # Flag to randomize initial game state (items on counters and in hands)
SYNERGY_SCALING_FACTOR = float(sys.argv[12]) if len(sys.argv) > 12 else 0.0  # Team synergy sensitivity: 0=no shaping, >0=team synergy-based shaping enabled
SPECIALIZATION_PENALTY_SCALE = float(sys.argv[13]) if len(sys.argv) > 13 else 0.0  # Specialization penalty scale (lambda): 0=no penalty, >0=penalty scale
COUNTER_REWARD = float(sys.argv[14]) if len(sys.argv) > 14 else 0.0  # Penalty for collisions (set to 0 to disable)
COLLISION_PENALTY = float(sys.argv[15]) if len(sys.argv) > 15 else 0.0  # Penalty for collisions (set to 0 to disable)
ALLOW_BLOCKED_ARG = str(sys.argv[16]).lower() if len(sys.argv) > 16 else "false"
KAPPA = float(sys.argv[17]) if len(sys.argv) > 17 else 1.0  # Competence transformation parameter for team synergy distribution
ACTIVATE_SYNERGY_POSITIVE_ARG = str(sys.argv[18]).lower() if len(sys.argv) > 18 else "false"
ADAPTIVE_COOPERATION_SCALE = float(sys.argv[19]) if len(sys.argv) > 19 else 0.0  # Path-length-dependent cooperation penalty scale
COLLISION_HARSHNESS = float(sys.argv[20]) if len(sys.argv) > 20 else 2.0  # Multiplier for specialization and adaptive cooperation when collisions enabled

# Optional when number of agents = 1:
# Decide which agent to train (1 or 2)
if NUM_AGENTS == 1:
    agent_to_train = 1  # Default to agent 1
    if len(sys.argv) > 21:  # agent_to_train is now the 21st argument (sys.argv[21])
        agent_to_train = int(sys.argv[21])
        if agent_to_train not in [1, 2]:
            raise ValueError("When NUM_AGENTS=1, agent_to_train must be 1 or 2")

######### ----------------------------------------------------------------- #########
######### -------------- Code below this line is automatic ---------------- #########

with open(INPUT_PATH, "r") as f:
    lines = f.readlines()
    for i in range(lines.__len__() // 2):
        globals()[f"alpha_{i+1}"], globals()[f"beta_{i+1}"] = [round(float(x), 4) for x in lines[2*i].strip().split()]
        globals()[f"walking_speed_{i+1}"], globals()[f"cutting_speed_{i+1}"] = [round(float(x), 4) for x in lines[2*i + 1].strip().split()]

##### Cluster config ##################
NUM_ENV_WORKERS = 8  # Parallel environment workers for efficient training
NUM_LEARNER_WORKERS = 1  # GPU learner workers
if CLUSTER == 'brigit':
    local = '/mnt/lustre/home/samuloza'
    # Resource allocation optimized for RL training
    NUM_GPUS = 1.0  # Full GPU for neural network training
    NUM_CPUS = 24   # Increased CPU cores for parallel environments
elif CLUSTER == 'cuenca':
    local = ''
    # Resource allocation optimized for RL training
    NUM_GPUS = 0.1  # Full GPU for neural network training
    NUM_CPUS = 12   # Increased CPU cores for parallel environments
elif CLUSTER == 'local':
    local = 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'
    # Resource allocation optimized for RL training
    NUM_GPUS = 0.0  # Full GPU for neural network training
    NUM_CPUS = 1   # Increased CPU cores for parallel environments
else:
    raise ValueError("Invalid cluster specified. Choose from 'brigit', 'cuenca', or 'local'.")

# Hyperparameters - Optimized for parallel training
NUM_ENVS = NUM_ENV_WORKERS  # Use all environment workers
INNER_SECONDS = 180  # Full episode length for proper learning
TRAIN_BATCH_SIZE = 4000  # Increased for better GPU utilization (NUM_ENVS * rollout_fragment_length * num_timesteps)
SGD_MINIBATCH_SIZE = 500  # Optimized minibatch size for GPU
NUM_SGD_ITER = 10  # Number of SGD iterations per training batch
SHOW_EVERY_N_EPOCHS = 1
SAVE_EVERY_N_EPOCHS = 1000
PAYOFF_MATRIX = [1,1,-2]

# Neural network architecture
MLP_LAYERS = [1024, 512, 256]

# Game characteristics
# Override ALLOW_BLOCKED with command line argument if provided
ALLOW_BLOCKED = (ALLOW_BLOCKED_ARG == "true")  # Whether to allow agents to attempt blocked actions

PENALTIES_CFG = {
    "do_nothing": 1.0, # Penalty for do_nothing action
    "useless_action": 5.0, # Penalty for useless actions
    "destructive_action": 10.0, # Penalty for destructive actions
    "inaccessible_tile": 10.0, # Penalty for trying to access an inaccessible tile (no path exists)
    "blocked": 5.0, # Penalty when path is blocked (by agents or collision)
    "collision": COLLISION_PENALTY, # Penalty when a collision occurs (if collision_enabled=True)
    "specialization_penalty_scale": SPECIALIZATION_PENALTY_SCALE,  # Specialization penalty scale (lambda): 0=no penalty, >0=penalty scale
    "adaptive_cooperation_scale": ADAPTIVE_COOPERATION_SCALE,  # Path-length penalty for slow walkers: 0=disabled, >0=enabled (multiplied by collision_harshness when collisions enabled)
    "collision_harshness": COLLISION_HARSHNESS,  # Multiplier for specialization and adaptive cooperation when collisions enabled (1.0=same, 2.0=double)
}

if REWARDS_ON_DELIVERY_ONLY == "true":
    REWARDS_CFG = {
        "raw_food": 0.0,
        "plate": 0.0,
        "counter": 0.0,
        "cut": 0.0,
        "salad": 0.0,
        "deliver": 10.0,
    }
else:
    REWARDS_CFG = {
        "raw_food": 1.0,
        "plate": 1.0,
        "counter": COUNTER_REWARD,
        "cut": 4.0,
        "salad": 5.0,
        "deliver": 10.0,
    }

# Team Synergy-Based Reward Shaping
# 1. Solo baselines are looked up from BASELINE_LOOKUP based on map and agent competences
# 2. Team Synergy: S(τ) = tanh(R^cum_team(τ) - τ * R̄^solo_team) where τ is episode progress
# 3. Asymmetric distribution: Ψ_i(τ) = S(τ) * ρ_i (if S<0) or S(τ) * (1-ρ_i) (if S≥0)
# 4. Final reward: R^ref_i = R^env_i - P^spec_i + η * Ψ_i(τ)
REFERENCE_REWARD_CFG = {
    "enabled": (SYNERGY_SCALING_FACTOR > 0),
    "synergy_scaling_factor": SYNERGY_SCALING_FACTOR,  # Fixed: was "eta", must match key used in game_env.py
    "kappa": KAPPA,  # Competence transformation parameter for team synergy distribution
    "activate_synergy_positive": (ACTIVATE_SYNERGY_POSITIVE_ARG == "true"),  # Whether to apply positive synergy signals
}

WAIT_FOR_ACTION_COMPLETION = True  # Flag to ensure actions complete before next step

# Solo Baseline Lookup Dictionary
# To add new baselines:
# 1. Add map name as key if not already present
# 2. Add speed configuration tuple (walk1, cut1, walk2, cut2) with number of deliveries
BASELINE_LOOKUP = {
    "baseline_division_of_labor_large": {
        (1.0, 1.0, 1.0, 1.0): 14.0,
        (0.8, 1.0, 1.0, 0.8): 12.0,
        (0.6, 1.0, 1.0, 0.6): 8.0,
        (0.4, 1.0, 1.0, 0.4): 5.0,
        (0.4, 1.0, 1.0, 0.2): 4.0,
        (0.2, 1.0, 1.0, 0.2): 2.0,
    },
    "semiencouraged_division_of_labor_large": {
        (1.0, 1.0, 1.0, 1.0): 10.0,
        (0.8, 1.0, 1.0, 0.8): 8.0,
        (0.6, 1.0, 1.0, 0.6): 6.0,
        (0.4, 1.0, 1.0, 0.4): 4.0,
        (0.4, 1.0, 1.0, 0.2): 3.0,
        (0.2, 1.0, 1.0, 0.2): 2.0,
    },
    "1-semiencouraged_division_of_labor_large": {
        (1.0, 1.0, 1.0, 1.0): 10.0,
        (0.8, 1.0, 1.0, 0.8): 8.0,
        (0.6, 1.0, 1.0, 0.6): 6.0,
        (0.4, 1.0, 1.0, 0.4): 4.0,
        (0.4, 1.0, 1.0, 0.2): 3.0,
        (0.2, 1.0, 1.0, 0.2): 2.0,
    },
    "2-semiencouraged_division_of_labor_large": {
        (1.0, 1.0, 1.0, 1.0): 10.0,
        (0.8, 1.0, 1.0, 0.8): 8.0,
        (0.6, 1.0, 1.0, 0.6): 6.0,
        (0.4, 1.0, 1.0, 0.4): 4.0,
        (0.4, 1.0, 1.0, 0.2): 3.0,
        (0.2, 1.0, 1.0, 0.2): 2.0,
    },
    "encouraged_division_of_labor_large": {
        (1.0, 1.0, 1.0, 1.0): 10.0,
        (0.8, 1.0, 1.0, 0.8): 8.0,
        (0.6, 1.0, 1.0, 0.6): 6.0,
        (0.4, 1.0, 1.0, 0.4): 4.0,
        (0.4, 1.0, 1.0, 0.2): 3.0,
        (0.2, 1.0, 1.0, 0.2): 2.0,
    },
    "1-encouraged_division_of_labor_large": {
        (1.0, 1.0, 1.0, 1.0): 8.0,
        (0.8, 1.0, 1.0, 0.8): 6.0,
        (0.6, 1.0, 1.0, 0.6): 5.0,
        (0.4, 1.0, 1.0, 0.4): 3.0,
        (0.4, 1.0, 1.0, 0.2): 2.0,
        (0.2, 1.0, 1.0, 0.2): 1.0,
    },
    "2-encouraged_division_of_labor_large": {
        (1.0, 1.0, 1.0, 1.0): 6.0,
        (0.8, 1.0, 1.0, 0.8): 4.0,
        (0.6, 1.0, 1.0, 0.6): 3.0,
        (0.4, 1.0, 1.0, 0.4): 2.0,
        (0.4, 1.0, 1.0, 0.2): 1.0,
        (0.2, 1.0, 1.0, 0.2): 0.0,
    },
    "3-encouraged_division_of_labor_large": {
        (1.0, 1.0, 1.0, 1.0): 4.0,
        (0.8, 1.0, 1.0, 0.8): 3.0,
        (0.6, 1.0, 1.0, 0.6): 2.0,
        (0.4, 1.0, 1.0, 0.4): 1.0,
        (0.4, 1.0, 1.0, 0.2): 0.0,
        (0.2, 1.0, 1.0, 0.2): 0.0,
    },
}

# Validate reference reward configuration
if SYNERGY_SCALING_FACTOR > 0:
    if NUM_AGENTS != 2:
        raise ValueError("Reference-based reward shaping is currently only supported for 2-agent teams.")

reward_weights, walking_speeds, cutting_speeds = {}, {}, {}

pretrained_policies = None
# Load pretrained policies if specified
if CHECKPOINT_PATHS != "none":
    pretrained_policies = {}
    with open(CHECKPOINT_PATHS, "r") as f:
        lines = f.readlines()
        if NUM_AGENTS == 1:
            # For single agent training, use the specific agent to train
            policy_id = str(lines[0]).strip()
            checkpoint_number = str(lines[1]).strip()
            checkpoint_path = str(lines[2]).strip()
            if policy_id.lower() != "none" and checkpoint_number.lower() != "none" and checkpoint_path.lower() != "none":
                pretrained_policies[f"ai_rl_{agent_to_train}"] = {"source_policy_id": policy_id, "checkpoint_number": checkpoint_number, "path": checkpoint_path}
            else:
                pretrained_policies[f"ai_rl_{agent_to_train}"] = None
        else:
            # For multi-agent training, use the standard loop
            for i in range(NUM_AGENTS):
                policy_id = str(lines[3*i]).strip()
                checkpoint_number = str(lines[3*i + 1]).strip()
                checkpoint_path = str(lines[3*i + 2]).strip()
                if policy_id.lower() != "none" and checkpoint_number.lower() != "none" and checkpoint_path.lower() != "none":
                    pretrained_policies[f"ai_rl_{i+1}"] = {"source_policy_id": policy_id, "checkpoint_number": checkpoint_number, "path": checkpoint_path}
                else:
                    pretrained_policies[f"ai_rl_{i+1}"] = None

# Determine collision mode from game version
COLLISION_ENABLED = GAME_VERSION.endswith('_collision')
BASE_GAME_VERSION = GAME_VERSION.replace('_collision', '') if COLLISION_ENABLED else GAME_VERSION

# Update save directory to reflect collision mode
save_dir_base = GAME_VERSION  # Use full game version (including _collision suffix if present)

# Determine initialization folder based on random_initial_state flag
init_folder = "random_init" if RANDOM_INITIAL_STATE == "true" else "empty_init"

# Add synergy subfolder if reference reward is enabled
synergy_folder = f"synergy_{SYNERGY_SCALING_FACTOR:.2f}" if SYNERGY_SCALING_FACTOR > 0 else "synergy_0"

# Add specialization penalty subfolder based on lambda value
if SPECIALIZATION_PENALTY_SCALE == 0:
    spec_folder = "specialized_0"
else:
    # Format lambda value, preserving 2 decimal places
    lambda_str = f"{SPECIALIZATION_PENALTY_SCALE:.2f}"
    spec_folder = f"specialized_{lambda_str}"

# Path definitions
if NUM_AGENTS == 1:
    if synergy_folder:
        save_dir = f'{local}/data/samuel_lozano/cooked/pretraining/{save_dir_base}/{init_folder}/map_{MAP_NR}/{synergy_folder}/{spec_folder}'
    else:
        save_dir = f'{local}/data/samuel_lozano/cooked/pretraining/{save_dir_base}/{init_folder}/map_{MAP_NR}/{spec_folder}'
    reward_weights[f"ai_rl_{agent_to_train}"] = (globals()[f"alpha_{agent_to_train}"], globals()[f"beta_{agent_to_train}"])
    walking_speeds[f"ai_rl_{agent_to_train}"] = globals()[f"walking_speed_{agent_to_train}"]
    cutting_speeds[f"ai_rl_{agent_to_train}"] = globals()[f"cutting_speed_{agent_to_train}"]
else:
    if synergy_folder:
        save_dir = f'{local}/data/samuel_lozano/cooked/{save_dir_base}/{init_folder}/map_{MAP_NR}/{synergy_folder}/{spec_folder}'
    else:
        save_dir = f'{local}/data/samuel_lozano/cooked/{save_dir_base}/{init_folder}/map_{MAP_NR}/{spec_folder}'
    for i in range(1, NUM_AGENTS + 1):
        reward_weights[f"ai_rl_{i}"] = (globals()[f"alpha_{i}"], globals()[f"beta_{i}"])
        walking_speeds[f"ai_rl_{i}"] = globals()[f"walking_speed_{i}"]
        cutting_speeds[f"ai_rl_{i}"] = globals()[f"cutting_speed_{i}"]

os.makedirs(save_dir, exist_ok=True)

# Determine grid size from map file (text format)
map_txt_path = os.path.join(os.path.dirname(__file__), 'spoiled_broth', 'maps', f'{MAP_NR}.txt')
if not os.path.exists(map_txt_path):
    raise FileNotFoundError(f"Map file {map_txt_path} not found.")
with open(map_txt_path, 'r') as f:
    map_lines = [line.rstrip('\n') for line in f.readlines()]
rows = len(map_lines)
cols = len(map_lines[0]) if rows > 0 else 0
if rows != cols:
    print(f"WARNING: Map is not square, this could cause errors in the future (got {rows} rows and {cols} columns).")
GRID_SIZE = (cols, rows)

# Function to lookup solo baseline from predefined dictionary
def lookup_solo_baseline(map_nr, walking_speeds, cutting_speeds, delivery_reward, num_agents=2):
    """
    Look up team baseline performance from BASELINE_LOOKUP dictionary.
    
    Args:
        map_nr: Map identifier (e.g., 'baseline_division_of_labor_v2')
        walking_speeds: Dict of {agent_id: walk_speed}
        cutting_speeds: Dict of {agent_id: cut_speed}
        delivery_reward: Reward per delivery (from REWARDS_CFG["deliver"])
        num_agents: Number of agents
    
    Returns:
        tuple: (solo_baselines_dict, team_baseline)
            solo_baselines_dict: {agent_id: individual_baseline_reward}
            team_baseline: sum of individual baseline rewards
    """
    if num_agents != 2:
        raise ValueError("Baseline lookup currently only supports 2 agents")
    
    # Extract agent speeds in sorted order
    agent_ids = sorted(walking_speeds.keys())
    speeds_tuple = tuple([
        walking_speeds[agent_ids[0]],
        cutting_speeds[agent_ids[0]],
        walking_speeds[agent_ids[1]],
        cutting_speeds[agent_ids[1]]
    ])
    
    # Round speeds to avoid floating point precision issues
    speeds_tuple = tuple(round(s, 2) for s in speeds_tuple)
    
    # Lookup baseline
    if map_nr not in BASELINE_LOOKUP:
        raise KeyError(f"Map '{map_nr}' not found in BASELINE_LOOKUP. Available maps: {list(BASELINE_LOOKUP.keys())}")
    
    map_baselines = BASELINE_LOOKUP[map_nr]
    if speeds_tuple not in map_baselines:
        raise KeyError(
            f"Speed configuration {speeds_tuple} not found for map '{map_nr}'.\n"
            f"Available configurations: {list(map_baselines.keys())}"
        )
    
    # Get number of deliveries and convert to reward
    team_deliveries = map_baselines[speeds_tuple]
    team_baseline = team_deliveries * delivery_reward
    
    # Split baseline among agents proportionally to their competence
    # Competence = average of walking and cutting speed
    competences = {}
    total_competence = 0.0
    for agent_id in agent_ids:
        comp = (walking_speeds[agent_id] + cutting_speeds[agent_id]) / 2.0
        competences[agent_id] = comp
        total_competence += comp
    
    # Distribute team baseline proportionally
    solo_baselines = {}
    if total_competence > 0:
        for agent_id in agent_ids:
            solo_baselines[agent_id] = team_baseline * (competences[agent_id] / total_competence)
    else:
        # Equal split if all competences are zero (shouldn't happen)
        for agent_id in agent_ids:
            solo_baselines[agent_id] = team_baseline / num_agents
    
    return solo_baselines, team_baseline

# Load solo baseline for reference-based reward shaping from BASELINE_LOOKUP
solo_baselines = None
if REFERENCE_REWARD_CFG["enabled"]:
    print(f"\n=== Reference-Based Opportunity Cost Shaping ===")
    print(f"Synergy scaling factor (opportunity cost sensitivity): {SYNERGY_SCALING_FACTOR}")
    print(f"Loading solo baselines from BASELINE_LOOKUP dictionary...")
    
    # First, populate walking_speeds and cutting_speeds (needed for lookup)
    if NUM_AGENTS == 1:
        walking_speeds_temp = {f"ai_rl_{agent_to_train}": globals()[f"walking_speed_{agent_to_train}"]}
        cutting_speeds_temp = {f"ai_rl_{agent_to_train}": globals()[f"cutting_speed_{agent_to_train}"]}
    else:
        walking_speeds_temp = {f"ai_rl_{i}": globals()[f"walking_speed_{i}"] for i in range(1, NUM_AGENTS + 1)}
        cutting_speeds_temp = {f"ai_rl_{i}": globals()[f"cutting_speed_{i}"] for i in range(1, NUM_AGENTS + 1)}
    
    # Lookup baseline from dictionary
    try:
        solo_baselines, solo_baseline_team = lookup_solo_baseline(
            MAP_NR,
            walking_speeds_temp,
            cutting_speeds_temp,
            REWARDS_CFG["deliver"],  # Pass delivery reward for conversion
            NUM_AGENTS
        )
        
        print(f"  Map: {MAP_NR}")
        print(f"  Delivery reward: {REWARDS_CFG['deliver']:.1f}")
        print(f"  Team deliveries: {solo_baseline_team / REWARDS_CFG['deliver']:.2f}")
        print(f"  Team baseline reward: {solo_baseline_team:.2f}")
        print(f"  Agent speeds and baselines:")
        for agent_id in sorted(solo_baselines.keys()):
            walk = walking_speeds_temp[agent_id]
            cut = cutting_speeds_temp[agent_id]
            baseline = solo_baselines[agent_id]
            deliveries = baseline / REWARDS_CFG['deliver']
            print(f"    {agent_id}: walk={walk:.2f}, cut={cut:.2f} → {deliveries:.2f} deliveries → baseline={baseline:.2f}")
        
    except (KeyError, ValueError) as e:
        print(f"  WARNING: {e}")
        print(f"  Baseline lookup failed. Setting baselines to None (synergy shaping disabled).")
        solo_baselines = None
        REFERENCE_REWARD_CFG["enabled"] = False
    
    print(f"===================================\n")

# RLlib specific configuration - Optimized for GPU training
config = {
    "NUM_ENVS": NUM_ENVS,
    "INNER_SECONDS": INNER_SECONDS,
    "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
    "SGD_MINIBATCH_SIZE": SGD_MINIBATCH_SIZE,
    "NUM_SGD_ITER": NUM_SGD_ITER,
    "NUM_EPOCHS": NUM_EPOCHS,
    "NUM_AGENTS": NUM_AGENTS,
    "AGENT_TO_TRAIN": agent_to_train if NUM_AGENTS == 1 else None,
    "SHOW_EVERY_N_EPOCHS": SHOW_EVERY_N_EPOCHS,
    "SAVE_EVERY_N_EPOCHS": SAVE_EVERY_N_EPOCHS,
    "LR": LR,
    "MAP_NR": MAP_NR,
    "REWARD_WEIGHTS": reward_weights,
    "GAME_VERSION": BASE_GAME_VERSION,  # Use base version without collision suffix
    "COLLISION_ENABLED": COLLISION_ENABLED,  # Add collision flag
    "GRID_SIZE": GRID_SIZE,
    "PAYOFF_MATRIX": PAYOFF_MATRIX,
    "WALKING_SPEEDS": walking_speeds,
    "CUTTING_SPEEDS": cutting_speeds,
    "INITIAL_SEED": SEED,
    "WAIT_FOR_COMPLETION": WAIT_FOR_ACTION_COMPLETION,
    "RANDOM_INITIAL_STATE": RANDOM_INITIAL_STATE,
    "SAVE_DIR": save_dir,
    "CHECKPOINTS": pretrained_policies,  # Add pretrained policies configuration
    # Reward and penalty configurations
    "PENALTIES_CFG": PENALTIES_CFG,
    "REWARDS_CFG": REWARDS_CFG,
    "REFERENCE_REWARD_CFG": REFERENCE_REWARD_CFG,  # Reference-based opportunity cost shaping
    "SOLO_BASELINES": solo_baselines,  # Individual solo baselines for reference reward (dict: agent_id -> baseline)
    "ALLOW_BLOCKED": ALLOW_BLOCKED,  # Whether to allow blocked actions to be attempted
    # Hyperparameters
    "NUM_UPDATES": NUM_SGD_ITER,  # Number of SGD iterations per batch
    "GAMMA": 0.9,     # Discount factor for future rewards (close to 1 = long-term, lower = short-term)
    "GAE_LAMBDA": 0.95,  # GAE lambda for advantage estimation
    "ENT_COEF": 0.01,    # Entropy coefficient for exploration
    "CLIP_EPS": 0.3,     # PPO clip parameter (limits how much the policy can change at each update; stabilizes training)
    "VF_COEF": 1.0,      # Value function coefficient
    "GRAD_CLIP": 0.5,    # Gradient clipping threshold
    "FCNET_HIDDENS": MLP_LAYERS,  # Hidden layer sizes for MLP
    "FCNET_ACTIVATION": "tanh",  # Activation function for MLP ("tanh", "relu", etc.)
    # Resource allocation
    "NUM_CPUS": NUM_CPUS,
    "NUM_GPUS": NUM_GPUS,
    # RLlib specific parameters - Optimized for GPU
    "NUM_ENV_WORKERS": NUM_ENV_WORKERS,  # Parallel environment workers (CPU)
    "NUM_LEARNER_WORKERS": NUM_LEARNER_WORKERS,  # GPU learner workers
    # Performance optimizations
    "ROLLOUT_FRAGMENT_LENGTH": "auto",  # Steps per rollout fragment
    "BATCH_MODE": "complete_episodes",  # Collect complete episodes for better learning
    "COMPRESS_OBSERVATIONS": False,  # Disable compression for speed
    "NUM_CPUS_PER_WORKER": 1,  # CPU cores per environment worker
    "NUM_GPUS_PER_WORKER": 0,  # Environment workers run on CPU only
    "NUM_CPUS_FOR_DRIVER": 1,  # Driver CPU usage
}

if ray.is_initialized():
    ray.shutdown()

# Initialize Ray with optimized resource allocation
ray.init(
    num_cpus=NUM_CPUS,
    num_gpus=1,  # Ensure GPU is available
    object_store_memory=2000000000,  # 2GB object store for efficient data transfer
    _plasma_directory="/tmp",  # Use fast storage for plasma store
)

# Optimize PyTorch for GPU training
torch.set_num_threads(2)  # Limit CPU threads per process to avoid oversubscription

# GPU optimization settings
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # GPU memory optimizations
    torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of GPU memory
    torch.cuda.empty_cache()  # Clear cache
    
    # Performance optimizations
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
    
    print("GPU optimizations enabled")
else:
    print("CUDA not available, falling back to CPU training")

# Environment optimization
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Enable async CUDA operations
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow GPU memory growth

# Run training with performance monitoring
print(f"Starting training with configuration:")
print(f"  Environment workers: {NUM_ENV_WORKERS} (CPU)")
print(f"  Learner workers: {NUM_LEARNER_WORKERS} (GPU)")
print(f"  Train batch size: {TRAIN_BATCH_SIZE}")
print(f"  SGD minibatch size: {SGD_MINIBATCH_SIZE}")
print(f"  Total CPU cores: {NUM_CPUS}")
print(f"  GPU allocation: {NUM_GPUS}")

# Monitor GPU memory before training
if torch.cuda.is_available():
    print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")

try:
    trainer, current_date, final_episode_count = make_train_rllib(config)

    # Save the final model
    path = os.path.join(config["SAVE_DIR"], f"Training_{current_date}")
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
        print(f"Training completed after {final_episode_count} episodes")

except Exception as e:
    print(f"Error during training: {e}")
    raise
finally:
    # Proper cleanup to avoid Ray shutdown warnings
    try:
        if 'trainer' in locals() and trainer is not None:
            # Stop the trainer properly
            trainer.stop()
            print("Trainer stopped successfully")
    except Exception as cleanup_error:
        print(f"Warning: Error during trainer cleanup: {cleanup_error}")
    
    try:
        # Shutdown Ray
        ray.shutdown()
        print("Ray shutdown completed")
    except Exception as ray_error:
        print(f"Warning: Error during Ray shutdown: {ray_error}")