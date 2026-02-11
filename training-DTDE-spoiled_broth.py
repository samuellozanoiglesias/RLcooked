# USE:   <cluster> <input_path> <map_nr> <lr> <game_version> [<num_agents>] [<num_epochs>] [<seed>] [<checkpoints>] [<rewards_on_delivery_only>] [<random_initial_state>] [<ability_risk_enabled>] [<synergy_scaling_factor>] [<specialization_penalty_scale>] [<agent_to_train>] > log_training.log 2>&1 &
# Example: nohup python training-DTDE-spoiled_broth.py cuenca ./cuenca/input_0_0.txt baseline_division_of_labor_v2 0.0003 classic 2 1000 0 none true false true 0.5 5.0 > log_training.log 2>&1 &
#   synergy_scaling_factor=0: Standard rewards (no team synergy shaping)
#   synergy_scaling_factor>0: Team synergy-based reward shaping enabled with given sensitivity
#   specialization_penalty_scale=0: No specialization penalty
#   specialization_penalty_scale>0: Specialization penalty with given scale

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
if len(sys.argv) > 6:
    NUM_AGENTS = int(sys.argv[6])
    if NUM_AGENTS not in [1, 2]:
        raise ValueError("NUM_AGENTS must be 1 or 2")
else:
    NUM_AGENTS = 2  # Default to 2 agents for backward compatibility

NUM_EPOCHS = int(sys.argv[7]) if len(sys.argv) > 7 else 500
SEED = int(sys.argv[8]) if len(sys.argv) > 8 else 0

# Optional checkpoint paths for loading pretrained policies (CHECKPOINT_PATHS should be a file with three lines per agent)
# Line 1: policy_id_to_be_loaded (policy_ai_rl_1, policy_ai_rl_2, etc.)
# Line 2: checkpoint_number
# Line 3: path_to_checkpoint
CHECKPOINT_PATHS = str(sys.argv[9]).lower() if len(sys.argv) > 9 else "none"
REWARDS_ON_DELIVERY_ONLY = str(sys.argv[10]).lower() if len(sys.argv) > 10 else "true"
RANDOM_INITIAL_STATE = str(sys.argv[11]).lower() if len(sys.argv) > 11 else "false"  # Flag to randomize initial game state (items on counters and in hands)
ABILITY_RISK_ENABLED = str(sys.argv[12]).lower() if len(sys.argv) > 12 else "false"  # Flag to enable ability-based risk modeling
SYNERGY_SCALING_FACTOR = float(sys.argv[13]) if len(sys.argv) > 13 else 0.0  # Team synergy sensitivity: 0=no shaping, >0=team synergy-based shaping enabled
SPECIALIZATION_PENALTY_SCALE = float(sys.argv[14]) if len(sys.argv) > 14 else 0.0  # Specialization penalty scale (lambda): 0=no penalty, >0=penalty scale

# Optional when number of agents = 1:
# Decide which agent to train (1 or 2)
if NUM_AGENTS == 1:
    agent_to_train = 1  # Default to agent 1
    if len(sys.argv) > 15:  # agent_to_train is the 15th argument (sys.argv[15])
        agent_to_train = int(sys.argv[15])
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
SAVE_EVERY_N_EPOCHS = 50
PAYOFF_MATRIX = [1,1,-2]

# Neural network architecture
MLP_LAYERS = [1024, 512, 256]

# Game characteristics
PENALTIES_CFG = {
    "busy": 0.01, # Penalty per second spent busy
    "useless_action": 5.0, # Penalty for useless actions
    "destructive_action": 10.0, # Penalty for destructive actions
    "inaccessible_tile": 10.0, # Penalty for trying to access an inaccessible tile (no path exists)
    "not_available": 1.0, # Penalty for trying to perform an action that is not available (runtime block)
    "collision": 5.0, # Penalty when collision cannot be rerouted (only with collision_enabled=True)
    "specialization_penalty_scale": SPECIALIZATION_PENALTY_SCALE,  # Specialization penalty scale (lambda): 0=no penalty, >0=penalty scale
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
        "raw_food": 0.0,
        "plate": 0.0,
        "counter": 0.0,
        "cut": 2.0,
        "salad": 5.0,
        "deliver": 10.0,
    }

# Dynamic rewards configuration - exponential decay [rewards_cfg = original_rewards_cfg * exp(-decay_rate * (episode - decay_start_episode))]
DYNAMIC_REWARDS_CFG = {
    "enabled": False,  # Set to False to disable dynamic rewards
    "decay_rate": 0.005,  # Decay rate for exponential function (higher = faster decay)
    "min_reward_multiplier": 0.00,  # Minimum multiplier (e.g., 0.1 = 10% of initial reward)
    "decay_start_episode": 100,  # Episode to start applying decay (0 = from beginning)
    "affected_rewards": ["raw_food", "plate", "counter", "cut", "salad"],  # Which reward types to apply decay to
}

# Dynamic PPO parameters configuration - exponential decay for exploration and policy change control
# This gradually reduces clip_eps (policy change constraint) and ent_coef (exploration) during training
# Starting with high values for exploration, then reducing them for more stable exploitation
DYNAMIC_PPO_PARAMS_CFG = {
    "enabled": False,  # Set to False to disable dynamic PPO parameters
    "decay_rate": 0.0001,  # Decay rate for exponential function (higher = faster decay)
    "min_param_multiplier": 0.1,  # Minimum multiplier (e.g., 0.1 = 10% of initial value)
    "decay_start_episode": 100,  # Episode to start applying decay (0 = from beginning)
    "affected_params": ["ent_coef"],  # Which PPO parameters to apply decay to
}

# Ability-based training dynamics configuration
# Models "risk of cooperation" through learning stability and exploration
# Hyperparameters scale continuously with collective ability (sum of all agents' abilities)
# Low collective ability = stable, conservative learning (cooperation safer)
# High collective ability = risky, exploratory learning (independence viable)
ABILITY_RISK_CFG = {
    "enabled": ABILITY_RISK_ENABLED == "true",
    
    # Reference ability values for scaling (collective ability = sum of all agents' walking_speed + cutting_speed)
    "reference_low_ability": 2.0,   # Reference point for low ability teams
    "reference_high_ability": 4.0,  # Reference point for high ability teams
    
    # Learning rate scaling (low ability = more stable gradients)
    "low_ability_lr_multiplier": 1.0,    # Conservative learning for weak teams (lower)
    "high_ability_lr_multiplier": 1.0,   # Aggressive learning for strong teams (higher)
    
    # Entropy coefficient (exploration vs exploitation)
    # Low ability agents: low entropy = stick to working strategies (cooperation)
    # High ability agents: high entropy = explore independence
    "low_ability_ent_multiplier": 4.0,   # Low exploration, find cooperation quickly (lower)
    "high_ability_ent_multiplier": 0.2,  # High exploration, find solo strategies (higher)
    
    # Value function coefficient (how much to trust value estimates)
    # Low ability: high VF weight = trust learned cooperation value
    # High ability: low VF weight = explore beyond current value estimates
    "low_ability_vf_multiplier": 1.0,    # Trust cooperation values (higher)
    "high_ability_vf_multiplier": 1.0,   # Question cooperation necessity (lower)
    
    # GAE Lambda (temporal credit assignment)
    # Low ability: high lambda = long-term thinking (cooperation pays off later)
    # High ability: low lambda = short-term rewards (solo actions pay immediately)
    "low_ability_gae_multiplier": 1.0,   # Value long-term cooperation (higher)
    "high_ability_gae_multiplier": 1.0,  # Value immediate solo rewards (lower)
    
    # Gradient clipping (training stability)
    # Low ability: aggressive clipping = avoid destabilizing updates
    # High ability: loose clipping = allow big policy shifts
    "low_ability_grad_clip": 1.0,    # Stable learning for weak teams (lower)
    "high_ability_grad_clip": 1.0,  # Flexible learning for strong teams (higher)
}

# Team Synergy-Based Reward Shaping
# Models cooperation through bounded synergy signals relative to individual pre-trained performance
# Enabled automatically when pretrained agents are provided AND eta > 0
# Solo baselines are loaded from training_stats.csv of pretrained checkpoints (no re-evaluation)
# Uses hyperbolic tangent to provide bounded, stable reward signals
# High-ability agents get smaller cooperation incentives, larger risk aversion penalties
# Low-ability agents get larger cooperation incentives, smaller risk aversion penalties
# 
# Implementation notes:
# 1. Solo baselines (R_solo_1, R_solo_2) are loaded from last episode of pretraining (training_stats.csv)
# 2. Team Synergy: S(τ) = tanh(R^cum_team(τ) - τ * R̄^solo_team) where τ is episode progress
# 3. Asymmetric distribution: Ψ_i(τ) = S(τ) * ρ_i (if S<0) or S(τ) * (1-ρ_i) (if S≥0)
# 4. Final reward: R^ref_i = R^env_i - P^spec_i + η * Ψ_i(τ)
REFERENCE_REWARD_CFG = {
    "enabled": (CHECKPOINT_PATHS != "none" and SYNERGY_SCALING_FACTOR > 0),
    
    # Opportunity cost sensitivity parameter η ∈ [0,∞)
    # Controls how much agents are penalized for underperforming solo baseline
    # η=0: No penalty (standard reward)
    # η>0: Reference-based shaping enabled
    "eta": SYNERGY_SCALING_FACTOR,
    
    # Solo baselines loaded from training_stats.csv (not re-evaluated)
    "load_from_training_stats": True,
}

WAIT_FOR_ACTION_COMPLETION = True  # Flag to ensure actions complete before next step

# Validate reference reward configuration
if SYNERGY_SCALING_FACTOR > 0:
    if CHECKPOINT_PATHS == "none":
        raise ValueError(f"Reference-based reward shaping (synergy_scaling_factor={SYNERGY_SCALING_FACTOR}) requires pretrained agents. Please provide checkpoint paths or set synergy_scaling_factor=0.")
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

# Helper functions for ability-based parameter scaling
def calculate_collective_ability(walking_speeds, cutting_speeds):
    """
    Calculate collective ability as sum of all agents' abilities.
    Each agent's ability = walking_speed + cutting_speed
    """
    total_ability = 0.0
    for agent_id in walking_speeds.keys():
        agent_ability = walking_speeds[agent_id] + cutting_speeds[agent_id]
        total_ability += agent_ability
    return total_ability

def linear_interpolate(value, low_ref, high_ref, low_mult, high_mult):
    """
    Linearly interpolate multiplier based on value between low and high references.
    If value < low_ref, return low_mult.
    If value > high_ref, return high_mult.
    Otherwise, interpolate linearly between low_mult and high_mult.
    """
    if value <= low_ref:
        return low_mult
    elif value >= high_ref:
        return high_mult
    else:
        # Linear interpolation
        ratio = (value - low_ref) / (high_ref - low_ref)
        return low_mult + ratio * (high_mult - low_mult)

def get_ability_based_params(walking_speeds, cutting_speeds, risk_cfg, base_lr, base_ent, base_vf, base_gae):
    """
    Calculate training parameters based on collective ability.
    Scales parameters continuously (no thresholds) proportional to total team ability.
    Returns dict with scaled parameters that model cooperation risk.
    """
    if not risk_cfg["enabled"]:
        return {
            "lr": base_lr,
            "ent_coef": base_ent,
            "vf_coef": base_vf,
            "gae_lambda": base_gae,
            "grad_clip": 0.5,
        }
    
    # Calculate collective ability (sum of all agents' abilities)
    collective_ability = calculate_collective_ability(walking_speeds, cutting_speeds)
    
    # Get reference points
    low_ref = risk_cfg["reference_low_ability"]
    high_ref = risk_cfg["reference_high_ability"]
    
    # Interpolate multipliers based on collective ability
    lr_mult = linear_interpolate(
        collective_ability, low_ref, high_ref,
        risk_cfg["low_ability_lr_multiplier"],
        risk_cfg["high_ability_lr_multiplier"]
    )
    
    ent_mult = linear_interpolate(
        collective_ability, low_ref, high_ref,
        risk_cfg["low_ability_ent_multiplier"],
        risk_cfg["high_ability_ent_multiplier"]
    )
    
    vf_mult = linear_interpolate(
        collective_ability, low_ref, high_ref,
        risk_cfg["low_ability_vf_multiplier"],
        risk_cfg["high_ability_vf_multiplier"]
    )
    
    gae_mult = linear_interpolate(
        collective_ability, low_ref, high_ref,
        risk_cfg["low_ability_gae_multiplier"],
        risk_cfg["high_ability_gae_multiplier"]
    )
    
    grad_clip = linear_interpolate(
        collective_ability, low_ref, high_ref,
        risk_cfg["low_ability_grad_clip"],
        risk_cfg["high_ability_grad_clip"]
    )
    
    # Apply multipliers to base parameters
    return {
        "lr": base_lr * lr_mult,
        "ent_coef": base_ent * ent_mult,
        "vf_coef": base_vf * vf_mult,
        "gae_lambda": base_gae * gae_mult,
        "grad_clip": grad_clip,
        "collective_ability": collective_ability,  # For logging
    }

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

# Calculate ability-based parameters
ability_params = get_ability_based_params(
    walking_speeds, 
    cutting_speeds, 
    ABILITY_RISK_CFG,
    base_lr=LR,
    base_ent=0.01,
    base_vf=1.0,
    base_gae=0.95
)

# Print ability-based scaling info
if ABILITY_RISK_CFG["enabled"]:
    print(f"\n=== Ability-Based Risk Modeling ===")
    print(f"Collective ability: {ability_params['collective_ability']:.3f}")
    print(f"Scaled learning rate: {ability_params['lr']:.6f} (base: {LR:.6f})")
    print(f"Scaled entropy coef: {ability_params['ent_coef']:.4f} (base: 0.01)")
    print(f"Scaled value coef: {ability_params['vf_coef']:.3f} (base: 1.0)")
    print(f"Scaled GAE lambda: {ability_params['gae_lambda']:.3f} (base: 0.95)")
    print(f"Gradient clip: {ability_params['grad_clip']:.3f}")
    print(f"===================================\n")

# Load solo baseline for reference-based reward shaping from training_stats.csv
solo_baselines = None
if REFERENCE_REWARD_CFG["enabled"]:
    print(f"\n=== Reference-Based Opportunity Cost Shaping ===")
    print(f"Synergy scaling factor (opportunity cost sensitivity): {SYNERGY_SCALING_FACTOR}")
    print(f"Loading solo baselines from pretrained checkpoints...")
    
    # Load solo baselines from training_stats.csv of each pretrained agent
    solo_baselines = {}
    
    for i in range(1, NUM_AGENTS + 1):
        agent_id = f"ai_rl_{i}"
        checkpoint_info = pretrained_policies.get(agent_id)
        
        if checkpoint_info is None:
            raise ValueError(f"No pretrained checkpoint for {agent_id}")
        
        # Extract checkpoint directory from path
        checkpoint_path = checkpoint_info["path"]
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        # Look for training_stats.csv in checkpoint directory or parent Training_* directory
        training_dir = checkpoint_dir
        while not os.path.exists(os.path.join(training_dir, "training_stats.csv")):
            parent = os.path.dirname(training_dir)
            if parent == training_dir:  # Reached root
                raise FileNotFoundError(f"Could not find training_stats.csv for {agent_id} starting from {checkpoint_path}")
            training_dir = parent
        
        stats_file = os.path.join(training_dir, "training_stats.csv")
        print(f"  Loading {agent_id} from: {stats_file}")
        
        # Read training stats and get pure reward from last episode
        df = pd.read_csv(stats_file)
        if df.empty:
            raise ValueError(f"Empty training_stats.csv for {agent_id}")
        
        # Remove duplicate header rows (where 'episode' column contains the string 'episode')
        df = df[df['episode'] != 'episode']
        
        # Convert episode column to numeric (it may have been read as string due to duplicate headers)
        df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
        df = df.dropna(subset=['episode'])  # Remove any rows where episode couldn't be converted
        
        # Get pure reward from last episode - use agent-specific column
        # Average across all environments in the last episode
        last_episode = df['episode'].max()
        last_episode_rows = df[df['episode'] == last_episode].copy()
        
        pure_reward_col = f'pure_reward_{agent_id}'
        if pure_reward_col not in df.columns:
            raise KeyError(f"Column '{pure_reward_col}' not found in {stats_file}. Available columns: {list(df.columns)}")
        
        # Convert pure reward column to numeric
        last_episode_rows.loc[:, pure_reward_col] = pd.to_numeric(last_episode_rows[pure_reward_col], errors='coerce')
        
        R_solo = last_episode_rows[pure_reward_col].mean()
        solo_baselines[agent_id] = R_solo
            
    solo_baseline_team = sum(solo_baselines.values())

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
    "LR": ability_params["lr"],  # Ability-based learning rate
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
    "DYNAMIC_REWARDS_CFG": DYNAMIC_REWARDS_CFG,
    "DYNAMIC_PPO_PARAMS_CFG": DYNAMIC_PPO_PARAMS_CFG,
    "ABILITY_RISK_CFG": ABILITY_RISK_CFG,  # Ability-based risk modeling config
    "REFERENCE_REWARD_CFG": REFERENCE_REWARD_CFG,  # Reference-based opportunity cost shaping
    "SOLO_BASELINES": solo_baselines,  # Individual solo baselines for reference reward (dict: agent_id -> baseline)
    # Hyperparameters - Ability-dependent
    "NUM_UPDATES": NUM_SGD_ITER,  # Number of SGD iterations per batch
    "GAMMA": 0.9,     # Discount factor for future rewards (close to 1 = long-term, lower = short-term)
    "GAE_LAMBDA": ability_params["gae_lambda"],  # Ability-based GAE lambda
    "ENT_COEF": ability_params["ent_coef"],      # Ability-based entropy coefficient
    "CLIP_EPS": 0.3,    # PPO clip parameter (limits how much the policy can change at each update; stabilizes training)
    "VF_COEF": ability_params["vf_coef"],        # Ability-based value function coefficient
    "GRAD_CLIP": ability_params["grad_clip"],    # Ability-based gradient clipping
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