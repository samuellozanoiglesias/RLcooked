# USE: python training-DTDE-gridsearch.py <cluster> <input_path> <map_nr> <lr> <game_version> <num_agents> <num_epochs> <seed> <checkpoint_paths> <rewards_only_on_delivery> <agent_to_train> <gamma> <gae_lambda> <ent_coef> <clip_eps> <vf_coef> <busy_penalty> <useless_action_penalty> <destructive_action_penalty> <cut_reward> <salad_reward> <deliver_reward> <train_batch_size> <sgd_minibatch_size> <num_sgd_iter> <mlp_hidden1> <mlp_hidden2> <mlp_hidden3> <experiment_name>
# Example: python training-DTDE-gridsearch.py cuenca ./cuenca/input_0_0.txt baseline_division_of_labor_v2 0.0003 classic 1 500 0 none true 1 0.99 0.95 0.01 0.2 0.5 0.01 2.0 10.0 5.0 7.0 10.0 4000 500 10 512 512 256 exp1

import os
import sys
from spoiled_broth.maps.map_paths import get_map_txt_path
from spoiled_broth.rl.make_train_rllib import make_train_rllib
import ray
import torch

# PyTorch, NumPy, MKL, etc. not creating more threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Parse command line arguments
if len(sys.argv) < 29:
    print("Usage: python training-DTDE-gridsearch.py <cluster> <input_path> <map_nr> <lr> <game_version> <num_agents> <num_epochs> <seed> <checkpoint_paths> <rewards_only_on_delivery> <agent_to_train> <gamma> <gae_lambda> <ent_coef> <clip_eps> <vf_coef> <busy_penalty> <useless_action_penalty> <destructive_action_penalty> <cut_reward> <salad_reward> <deliver_reward> <train_batch_size> <sgd_minibatch_size> <num_sgd_iter> <mlp_hidden1> <mlp_hidden2> <mlp_hidden3> <experiment_name>")
    sys.exit(1)

CLUSTER = str(sys.argv[1]).lower()
INPUT_PATH = sys.argv[2]
MAP_NR = str(sys.argv[3]).lower()
LR = float(sys.argv[4])
GAME_VERSION = str(sys.argv[5]).lower()
NUM_AGENTS = int(sys.argv[6])
NUM_EPOCHS = int(sys.argv[7])
SEED = int(sys.argv[8])
CHECKPOINT_PATHS = str(sys.argv[9]).lower()
REWARDS_ON_DELIVERY_ONLY = str(sys.argv[10]).lower()
AGENT_TO_TRAIN = int(sys.argv[11]) if NUM_AGENTS == 1 else None

# Hyperparameters from command line
GAMMA = float(sys.argv[12])
GAE_LAMBDA = float(sys.argv[13])
ENT_COEF = float(sys.argv[14])
CLIP_EPS = float(sys.argv[15])
VF_COEF = float(sys.argv[16])

# Penalties from command line
BUSY_PENALTY = float(sys.argv[17])
USELESS_ACTION_PENALTY = float(sys.argv[18])
DESTRUCTIVE_ACTION_PENALTY = float(sys.argv[19])

# Rewards from command line
CUT_REWARD = float(sys.argv[20])
SALAD_REWARD = float(sys.argv[21])
DELIVER_REWARD = float(sys.argv[22])

# Training hyperparameters from command line
TRAIN_BATCH_SIZE = int(sys.argv[23])
SGD_MINIBATCH_SIZE = int(sys.argv[24])
NUM_SGD_ITER = int(sys.argv[25])

# MLP architecture from command line
MLP_HIDDEN1 = int(sys.argv[26])
MLP_HIDDEN2 = int(sys.argv[27])
MLP_HIDDEN3 = int(sys.argv[28])
MLP_LAYERS = [MLP_HIDDEN1, MLP_HIDDEN2, MLP_HIDDEN3]

# Experiment name for organizing results
EXPERIMENT_NAME = sys.argv[29] if len(sys.argv) > 29 else "default"

# Validation
if NUM_AGENTS not in [1, 2]:
    raise ValueError("NUM_AGENTS must be 1 or 2")

if NUM_AGENTS == 1 and AGENT_TO_TRAIN not in [1, 2]:
    raise ValueError("When NUM_AGENTS=1, agent_to_train must be 1 or 2")

# Read input file for agent characteristics
with open(INPUT_PATH, "r") as f:
    lines = f.readlines()
    for i in range(lines.__len__() // 2):
        globals()[f"alpha_{i+1}"], globals()[f"beta_{i+1}"] = [round(float(x), 4) for x in lines[2*i].strip().split()]
        globals()[f"walking_speed_{i+1}"], globals()[f"cutting_speed_{i+1}"] = [round(float(x), 4) for x in lines[2*i + 1].strip().split()]

##### Cluster config ##################
NUM_ENV_WORKERS = 8  # Parallel environment workers
NUM_LEARNER_WORKERS = 1  # GPU learner workers
if CLUSTER == 'brigit':
    local = '/mnt/lustre/home/samuloza'
    NUM_GPUS = 1.0
    NUM_CPUS = 24
elif CLUSTER == 'cuenca':
    local = ''
    NUM_GPUS = 0.1
    NUM_CPUS = 12
elif CLUSTER == 'local':
    local = 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'
    NUM_GPUS = 0.0
    NUM_CPUS = 1
else:
    raise ValueError("Invalid cluster specified. Choose from 'brigit', 'cuenca', or 'local'.")

# Training settings
NUM_ENVS = NUM_ENV_WORKERS
INNER_SECONDS = 180
SHOW_EVERY_N_EPOCHS = 1
SAVE_EVERY_N_EPOCHS = 100
PAYOFF_MATRIX = [1,1,-2]

# Game characteristics - penalties from command line arguments
PENALTIES_CFG = {
    "busy": BUSY_PENALTY,
    "useless_action": USELESS_ACTION_PENALTY,
    "destructive_action": DESTRUCTIVE_ACTION_PENALTY,
    "inaccessible_tile": 5.0,  # Keep default
    "not_available": 2.0,  # Keep default
}

# Rewards configuration from command line arguments
if REWARDS_ON_DELIVERY_ONLY == "true":
    REWARDS_CFG = {
        "raw_food": 0.0,
        "plate": 0.0,
        "counter": 0.0,
        "cut": 0.0,
        "salad": 0.0,
        "deliver": DELIVER_REWARD,
    }
else:
    REWARDS_CFG = {
        "raw_food": 0.0,
        "plate": 0.0,
        "counter": 0.0,
        "cut": CUT_REWARD,
        "salad": SALAD_REWARD,
        "deliver": DELIVER_REWARD,
    }

# Dynamic configurations (replaced with new intermediate reward decay system)
from training_configuration.reward_penalties import get_intermediate_reward_decay_config
INTERMEDIATE_REWARD_DECAY_CFG = get_intermediate_reward_decay_config(enabled=False)  # Disabled for grid search to ensure clean comparisons

DYNAMIC_PPO_PARAMS_CFG = {
    "enabled": False,
    "decay_rate": 0.0001,
    "min_param_multiplier": 0.1,
    "decay_start_episode": 100,
    "affected_params": ["ent_coef"],
}

WAIT_FOR_ACTION_COMPLETION = True

reward_weights, walking_speeds, cutting_speeds = {}, {}, {}

# Path definitions - include experiment name for organization
if NUM_AGENTS == 1:
    save_dir = f'{local}/data/samuel_lozano/cooked/gridsearch/{EXPERIMENT_NAME}/pretraining/{GAME_VERSION}/map_{MAP_NR}'
    reward_weights[f"ai_rl_{AGENT_TO_TRAIN}"] = (globals()[f"alpha_{AGENT_TO_TRAIN}"], globals()[f"beta_{AGENT_TO_TRAIN}"])
    walking_speeds[f"ai_rl_{AGENT_TO_TRAIN}"] = globals()[f"walking_speed_{AGENT_TO_TRAIN}"]
    cutting_speeds[f"ai_rl_{AGENT_TO_TRAIN}"] = globals()[f"cutting_speed_{AGENT_TO_TRAIN}"]
else: 
    save_dir = f'{local}/data/samuel_lozano/cooked/gridsearch/{EXPERIMENT_NAME}/{GAME_VERSION}/map_{MAP_NR}'
    for i in range(1, NUM_AGENTS + 1):
        reward_weights[f"ai_rl_{i}"] = (globals()[f"alpha_{i}"], globals()[f"beta_{i}"])
        walking_speeds[f"ai_rl_{i}"] = globals()[f"walking_speed_{i}"]
        cutting_speeds[f"ai_rl_{i}"] = globals()[f"cutting_speed_{i}"]

os.makedirs(save_dir, exist_ok=True)

# Handle pretrained policies
pretrained_policies = None
if CHECKPOINT_PATHS != "none":
    pretrained_policies = {}
    with open(CHECKPOINT_PATHS, "r") as f:
        lines = f.readlines()
        if NUM_AGENTS == 1:
            policy_id = str(lines[0]).strip()
            checkpoint_number = str(lines[1]).strip()
            checkpoint_path = str(lines[2]).strip()
            if policy_id.lower() != "none" and checkpoint_number.lower() != "none" and checkpoint_path.lower() != "none":
                pretrained_policies[f"ai_rl_{AGENT_TO_TRAIN}"] = {"source_policy_id": policy_id, "checkpoint_number": checkpoint_number, "path": checkpoint_path}
            else:
                pretrained_policies[f"ai_rl_{AGENT_TO_TRAIN}"] = None
        else:
            for i in range(NUM_AGENTS):
                policy_id = str(lines[3*i]).strip()
                checkpoint_number = str(lines[3*i + 1]).strip()
                checkpoint_path = str(lines[3*i + 2]).strip()
                if policy_id.lower() != "none" and checkpoint_number.lower() != "none" and checkpoint_path.lower() != "none":
                    pretrained_policies[f"ai_rl_{i+1}"] = {"source_policy_id": policy_id, "checkpoint_number": checkpoint_number, "path": checkpoint_path}
                else:
                    pretrained_policies[f"ai_rl_{i+1}"] = None

# Determine grid size from map file
map_txt_path = get_map_txt_path(MAP_NR, game_version=GAME_VERSION)
if not os.path.exists(map_txt_path):
    raise FileNotFoundError(f"Map file {map_txt_path} not found.")
with open(map_txt_path, 'r') as f:
    map_lines = [line.rstrip('\n') for line in f.readlines()]
rows = len(map_lines)
cols = len(map_lines[0]) if rows > 0 else 0
if rows != cols:
    print(f"WARNING: Map is not square, this could cause errors in the future (got {rows} rows and {cols} columns).")
GRID_SIZE = (cols, rows)

# Create configuration with hyperparameters from command line
config = {
    "NUM_ENVS": NUM_ENVS,
    "INNER_SECONDS": INNER_SECONDS,
    "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
    "SGD_MINIBATCH_SIZE": SGD_MINIBATCH_SIZE,
    "NUM_SGD_ITER": NUM_SGD_ITER,
    "NUM_EPOCHS": NUM_EPOCHS,
    "NUM_AGENTS": NUM_AGENTS,
    "AGENT_TO_TRAIN": AGENT_TO_TRAIN,
    "SHOW_EVERY_N_EPOCHS": SHOW_EVERY_N_EPOCHS,
    "SAVE_EVERY_N_EPOCHS": SAVE_EVERY_N_EPOCHS,
    "LR": LR,
    "MAP_NR": MAP_NR,
    "REWARD_WEIGHTS": reward_weights,
    "GAME_VERSION": GAME_VERSION,
    "GRID_SIZE": GRID_SIZE,
    "PAYOFF_MATRIX": PAYOFF_MATRIX,
    "WALKING_SPEEDS": walking_speeds,
    "CUTTING_SPEEDS": cutting_speeds,
    "INITIAL_SEED": SEED,
    "WAIT_FOR_COMPLETION": WAIT_FOR_ACTION_COMPLETION,
    "SAVE_DIR": save_dir,
    "CHECKPOINTS": pretrained_policies,
    # Reward and penalty configurations
    "PENALTIES_CFG": PENALTIES_CFG,
    "REWARDS_CFG": REWARDS_CFG,
    "INTERMEDIATE_REWARD_DECAY_CFG": INTERMEDIATE_REWARD_DECAY_CFG,
    "DYNAMIC_PPO_PARAMS_CFG": DYNAMIC_PPO_PARAMS_CFG,
    # Hyperparameters from command line
    "NUM_UPDATES": NUM_SGD_ITER,
    "GAMMA": GAMMA,
    "GAE_LAMBDA": GAE_LAMBDA,
    "ENT_COEF": ENT_COEF,
    "CLIP_EPS": CLIP_EPS,
    "VF_COEF": VF_COEF,
    # Neural network architecture
    "FCNET_HIDDENS": MLP_LAYERS,
    "FCNET_ACTIVATION": "tanh",
    # Training batch parameters
    "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
    "SGD_MINIBATCH_SIZE": SGD_MINIBATCH_SIZE,
    "NUM_SGD_ITER": NUM_SGD_ITER,
    # Resource allocation
    "NUM_CPUS": NUM_CPUS,
    "NUM_GPUS": NUM_GPUS,
    "NUM_ENV_WORKERS": NUM_ENV_WORKERS,
    "NUM_LEARNER_WORKERS": NUM_LEARNER_WORKERS,
    # Performance optimizations
    "ROLLOUT_FRAGMENT_LENGTH": 200,
    "BATCH_MODE": "complete_episodes",
    "COMPRESS_OBSERVATIONS": False,
    "NUM_CPUS_PER_WORKER": 1,
    "NUM_GPUS_PER_WORKER": 0,
    "NUM_CPUS_FOR_DRIVER": 1,
}

# Ray initialization
if ray.is_initialized():
    ray.shutdown()

ray.init(
    num_cpus=NUM_CPUS,
    num_gpus=1,
    object_store_memory=2000000000,
    _plasma_directory="/tmp",
)

# PyTorch optimizations
torch.set_num_threads(2)

if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    torch.cuda.set_per_process_memory_fraction(0.85)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    print("GPU optimizations enabled")
else:
    print("CUDA not available, falling back to CPU training")

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Print experiment configuration
print(f"Starting grid search experiment: {EXPERIMENT_NAME}")
print(f"PPO Hyperparameters: GAMMA={GAMMA}, GAE_LAMBDA={GAE_LAMBDA}, ENT_COEF={ENT_COEF}, CLIP_EPS={CLIP_EPS}, VF_COEF={VF_COEF}")
print(f"Training Hyperparameters: BATCH_SIZE={TRAIN_BATCH_SIZE}, MINIBATCH_SIZE={SGD_MINIBATCH_SIZE}, SGD_ITER={NUM_SGD_ITER}")
print(f"Network Architecture: MLP_LAYERS={MLP_LAYERS}")
print(f"Penalties: busy={BUSY_PENALTY}, useless_action={USELESS_ACTION_PENALTY}, destructive_action={DESTRUCTIVE_ACTION_PENALTY}")
print(f"Rewards: cut={CUT_REWARD}, salad={SALAD_REWARD}, deliver={DELIVER_REWARD}")

# Monitor GPU memory before training
if torch.cuda.is_available():
    print(f"GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")

try:
    trainer, current_date, final_episode_count = make_train_rllib(config)

    # Save the final model with experiment details
    path = os.path.join(config["SAVE_DIR"], f"Training_{current_date}")
    os.makedirs(path, exist_ok=True)

    # Save experiment configuration
    config_path = os.path.join(path, "experiment_config.txt")
    with open(config_path, "w") as f:
        f.write(f"EXPERIMENT_NAME: {EXPERIMENT_NAME}\n")
        f.write(f"GAMMA: {GAMMA}\n")
        f.write(f"GAE_LAMBDA: {GAE_LAMBDA}\n")
        f.write(f"ENT_COEF: {ENT_COEF}\n")
        f.write(f"CLIP_EPS: {CLIP_EPS}\n")
        f.write(f"VF_COEF: {VF_COEF}\n")
        f.write(f"BUSY_PENALTY: {BUSY_PENALTY}\n")
        f.write(f"USELESS_ACTION_PENALTY: {USELESS_ACTION_PENALTY}\n")
        f.write(f"DESTRUCTIVE_ACTION_PENALTY: {DESTRUCTIVE_ACTION_PENALTY}\n")
        f.write(f"CUT_REWARD: {CUT_REWARD}\n")
        f.write(f"SALAD_REWARD: {SALAD_REWARD}\n")
        f.write(f"DELIVER_REWARD: {DELIVER_REWARD}\n")
        f.write(f"TRAIN_BATCH_SIZE: {TRAIN_BATCH_SIZE}\n")
        f.write(f"SGD_MINIBATCH_SIZE: {SGD_MINIBATCH_SIZE}\n")
        f.write(f"NUM_SGD_ITER: {NUM_SGD_ITER}\n")
        f.write(f"MLP_LAYERS: {MLP_LAYERS}\n")
        f.write(f"LR: {LR}\n")
        f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
        f.write(f"SEED: {SEED}\n")
        if final_episode_count is not None:
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
    try:
        if 'trainer' in locals() and trainer is not None:
            trainer.stop()
            print("Trainer stopped successfully")
    except Exception as cleanup_error:
        print(f"Warning: Error during trainer cleanup: {cleanup_error}")
    
    try:
        ray.shutdown()
        print("Ray shutdown completed")
    except Exception as ray_error:
        print(f"Warning: Error during Ray shutdown: {ray_error}")