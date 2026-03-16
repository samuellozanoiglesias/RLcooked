# USE:   <cluster> <input_path> <map_nr> <lr> <game_version> [<num_agents>] [<num_epochs>] [<seed>] [<checkpoints>] [<rewards_on_delivery_only>] [<random_initial_state>] [<synergy_scaling_factor>] [<specialization_penalty_scale>] [<collision_penalty>] [<agent_to_train>] [<allow_blocked>] [<kappa>] [<activate_synergy_positive>] [<enable_reward_decay>] [<collision_harshness>] > log_training.log 2>&1 &
# Example: nohup python training-DTDE-spoiled_broth.py cuenca ./cuenca/input_0_0.txt baseline_division_of_labor_v2 0.0003 classic 2 1000 0 none true false 0.5 5.0 10.0 false 1.0 true false true 2.0 > log_training.log 2>&1 &
#   synergy_scaling_factor=0: Standard rewards (no team synergy shaping)
#   synergy_scaling_factor>0: Team synergy-based reward shaping enabled with given sensitivity
#   specialization_penalty_scale=0: No specialization penalty
#   specialization_penalty_scale>0: Specialization penalty with given scale
#   kappa: Competence transformation parameter for team synergy distribution (default=1.0)
#   activate_synergy_positive: Whether to apply positive synergy signals (true/false, default=false)
#   enable_reward_decay: Exponentially decay intermediate rewards after episode 200 (true/false, default=false)
#   collision_harshness: Multiplier for specialization penalty when collisions enabled (1.0=same, 2.0=double, default=2.0)

import os
import sys
from spoiled_broth.rl.make_train_rllib import make_train_rllib
import ray
import torch

# Import training configuration modules
from training_configuration.config_utils import (
    parse_input_file, setup_agent_configurations, parse_pretrained_policies,
    parse_game_version, get_hyperparameters, validate_configuration
)
from training_configuration.cluster_config import get_cluster_config
from training_configuration.reward_penalties import (
    get_penalties_config, get_rewards_config, get_reference_reward_config,
    get_intermediate_reward_decay_config
)
from training_configuration.path_utils import generate_save_directory, get_map_grid_size
from training_configuration.cooperation_factor import get_cooperation_factor
from training_configuration.baseline_lookup import lookup_solo_baseline

# PyTorch, NumPy, MKL, etc. not creating more threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Parse command line arguments
CLUSTER = str(sys.argv[1]).lower()
INPUT_PATH = sys.argv[2]
MAP_NR = str(sys.argv[3]).lower()
LR = float(sys.argv[4])
GAME_VERSION = str(sys.argv[5]).lower()
NUM_AGENTS = int(sys.argv[6])
NUM_EPOCHS = int(sys.argv[7]) if len(sys.argv) > 7 else 500
SEED = int(sys.argv[8]) if len(sys.argv) > 8 else 0
CHECKPOINT_PATHS = str(sys.argv[9]).lower() if len(sys.argv) > 9 else "none"
REWARDS_ON_DELIVERY_ONLY = str(sys.argv[10]).lower() if len(sys.argv) > 10 else "true"
RANDOM_INITIAL_STATE = str(sys.argv[11]).lower() if len(sys.argv) > 11 else "false"
SYNERGY_SCALING_FACTOR = float(sys.argv[12]) if len(sys.argv) > 12 else 0.0
SPECIALIZATION_PENALTY_SCALE = float(sys.argv[13]) if len(sys.argv) > 13 else 0.0
COUNTER_REWARD = float(sys.argv[14]) if len(sys.argv) > 14 else 0.0
COLLISION_PENALTY = float(sys.argv[15]) if len(sys.argv) > 15 else 0.0
ALLOW_BLOCKED_ARG = str(sys.argv[16]).lower() if len(sys.argv) > 16 else "false"
KAPPA = float(sys.argv[17]) if len(sys.argv) > 17 else 1.0
ACTIVATE_SYNERGY_POSITIVE_ARG = str(sys.argv[18]).lower() if len(sys.argv) > 18 else "false"
ENABLE_REWARD_DECAY_ARG = str(sys.argv[19]).lower() if len(sys.argv) > 19 else "false"
COLLISION_HARSHNESS = float(sys.argv[20]) if len(sys.argv) > 20 else 2.0

# Entropy coefficient
ENTROPY_COEF = float(sys.argv[21]) if len(sys.argv) > 21 else 0.01

# Handle single agent training
agent_to_train = 1
if NUM_AGENTS == 1 and len(sys.argv) > 22:
    agent_to_train = int(sys.argv[22])

######### ----------------------------------------------------------------- #########
######### -------------- Configuration Processing ------------------------- #########

# Validate configuration
validate_configuration(NUM_AGENTS, SYNERGY_SCALING_FACTOR, agent_to_train)

# Parse input file for agent parameters
agent_params = parse_input_file(INPUT_PATH)

# Get cluster configuration
cluster_config = get_cluster_config(CLUSTER)

# Parse game version
BASE_GAME_VERSION, COLLISION_ENABLED = parse_game_version(GAME_VERSION)

# Setup boolean flags
ALLOW_BLOCKED = (ALLOW_BLOCKED_ARG == "true")
ACTIVATE_SYNERGY_POSITIVE = (ACTIVATE_SYNERGY_POSITIVE_ARG == "true")
REWARDS_ON_DELIVERY = (REWARDS_ON_DELIVERY_ONLY == "true")
ENABLE_REWARD_DECAY = (ENABLE_REWARD_DECAY_ARG == "true")

# Get hyperparameters
hyperparams = get_hyperparameters()

# Setup agent configurations
reward_weights, walking_speeds, cutting_speeds = setup_agent_configurations(
    agent_params, NUM_AGENTS, agent_to_train
)

# Parse pretrained policies
pretrained_policies = parse_pretrained_policies(CHECKPOINT_PATHS, NUM_AGENTS, agent_to_train)

# Get map grid size
GRID_SIZE = get_map_grid_size(MAP_NR)

# Calculate cooperation factor 
cooperation_factor = get_cooperation_factor(MAP_NR)

# Configure rewards and penalties
PENALTIES_CFG = get_penalties_config(
    COLLISION_PENALTY, SPECIALIZATION_PENALTY_SCALE, COLLISION_HARSHNESS, cooperation_factor
)
REWARDS_CFG = get_rewards_config(REWARDS_ON_DELIVERY, COUNTER_REWARD)
REFERENCE_REWARD_CFG = get_reference_reward_config(
    SYNERGY_SCALING_FACTOR, cooperation_factor, KAPPA, ACTIVATE_SYNERGY_POSITIVE
)
INTERMEDIATE_REWARD_DECAY_CFG = get_intermediate_reward_decay_config(
    enable_decay=ENABLE_REWARD_DECAY, decay_start_episode=200, alpha=0.05
)

# Display cooperation factor effects (use values from configurations)
if SPECIALIZATION_PENALTY_SCALE > 0:
    effective_specialization_penalty = PENALTIES_CFG["specialization_penalty_scale"]
    print(f"\n=== Cooperation-Adjusted Specialization ===")
    print(f"Base specialization penalty scale: {SPECIALIZATION_PENALTY_SCALE}")
    print(f"Map cooperation factor: {cooperation_factor:.3f}")
    print(f"Effective specialization penalty: {effective_specialization_penalty:.3f} (base × cooperation)")
    print(f"Reasoning: Maps requiring more cooperation need stronger specialization incentives")
    print(f"=========================================\n")

# Generate save directory
init_folder = "random_init" if RANDOM_INITIAL_STATE == "true" else "empty_init"
save_dir = generate_save_directory(
    cluster_config['local_path'], GAME_VERSION, NUM_AGENTS, MAP_NR, 
    init_folder, SYNERGY_SCALING_FACTOR, SPECIALIZATION_PENALTY_SCALE, agent_to_train
)
os.makedirs(save_dir, exist_ok=True)

# Load solo baseline for reference-based reward shaping from BASELINE_LOOKUP
solo_baselines = None
if REFERENCE_REWARD_CFG["enabled"]:
    effective_synergy_scaling = REFERENCE_REWARD_CFG["synergy_scaling_factor"]
    print(f"\n=== Reference-Based Opportunity Cost Shaping ===")
    print(f"Base synergy scaling factor: {SYNERGY_SCALING_FACTOR}")
    print(f"Map cooperation factor: {cooperation_factor:.2f} (mathematically calculated)")
    print(f"Effective synergy scaling factor: {effective_synergy_scaling:.3f} (base × cooperation)")
    print(f"Mathematical cooperation analysis:")
    print(f"  - Analyzes spatial constraints, bottlenecks, path diversity")
    print(f"  - Considers workspace overlap and critical dependencies")
    print(f"  - Higher factor = more coordination structurally required")
    print(f"Loading solo baselines from BASELINE_LOOKUP dictionary...")
    
    # First, populate walking_speeds and cutting_speeds (needed for lookup)
    if NUM_AGENTS == 1:
        walking_speeds_temp = {f"ai_rl_{agent_to_train}": agent_params[f"walking_speed_{agent_to_train}"]}
        cutting_speeds_temp = {f"ai_rl_{agent_to_train}": agent_params[f"cutting_speed_{agent_to_train}"]}
    else:
        walking_speeds_temp = {f"ai_rl_{i}": agent_params[f"walking_speed_{i}"] for i in range(1, NUM_AGENTS + 1)}
        cutting_speeds_temp = {f"ai_rl_{i}": agent_params[f"cutting_speed_{i}"] for i in range(1, NUM_AGENTS + 1)}
    
    # Lookup baseline from dictionary
    try:
        solo_baselines, solo_baseline_team = lookup_solo_baseline(
            MAP_NR,
            walking_speeds_temp,
            cutting_speeds_temp,
            REWARDS_CFG["deliver"],  # Pass delivery reward for conversion
            NUM_AGENTS
        )
        
        print(f"  Map: {MAP_NR} (cooperation factor: {cooperation_factor:.2f})")
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
        raise RuntimeError(
            f"Synergy shaping is enabled (synergy_scaling_factor={SYNERGY_SCALING_FACTOR}) "
            f"but baseline lookup failed for map '{MAP_NR}'.\n"
            f"Add an entry for this map to BASELINE_LOOKUP in training_configuration/baseline_lookup.py "
            f"before training.\nOriginal error: {e}"
        ) from e
    
    print(f"===================================\n")

# RLlib training configuration
config = {
    "NUM_ENVS": cluster_config['num_env_workers'],
    "INNER_SECONDS": hyperparams["inner_seconds"],
    "TRAIN_BATCH_SIZE": hyperparams["train_batch_size"],
    "SGD_MINIBATCH_SIZE": hyperparams["sgd_minibatch_size"],
    "NUM_SGD_ITER": hyperparams["num_sgd_iter"],
    "NUM_EPOCHS": NUM_EPOCHS,
    "NUM_AGENTS": NUM_AGENTS,
    "AGENT_TO_TRAIN": agent_to_train if NUM_AGENTS == 1 else None,
    "SHOW_EVERY_N_EPOCHS": hyperparams["show_every_n_epochs"],
    "SAVE_EVERY_N_EPOCHS": hyperparams["save_every_n_epochs"],
    "LR": LR,
    "MAP_NR": MAP_NR,
    "REWARD_WEIGHTS": reward_weights,
    "GAME_VERSION": BASE_GAME_VERSION,
    "COLLISION_ENABLED": COLLISION_ENABLED,
    "GRID_SIZE": GRID_SIZE,
    "PAYOFF_MATRIX": hyperparams["payoff_matrix"],
    "WALKING_SPEEDS": walking_speeds,
    "CUTTING_SPEEDS": cutting_speeds,
    "INITIAL_SEED": SEED,
    "WAIT_FOR_COMPLETION": True,
    "RANDOM_INITIAL_STATE": RANDOM_INITIAL_STATE,
    "SAVE_DIR": save_dir,
    "CHECKPOINTS": pretrained_policies,
    # Configurations from modules
    "PENALTIES_CFG": PENALTIES_CFG,
    "REWARDS_CFG": REWARDS_CFG,
    "REFERENCE_REWARD_CFG": REFERENCE_REWARD_CFG,
    "INTERMEDIATE_REWARD_DECAY_CFG": INTERMEDIATE_REWARD_DECAY_CFG,
    "SOLO_BASELINES": solo_baselines,
    "ALLOW_BLOCKED": ALLOW_BLOCKED,
    # Hyperparameters
    "NUM_UPDATES": hyperparams["num_sgd_iter"],
    "GAMMA": hyperparams["gamma"],
        "GAE_LAMBDA": hyperparams["gae_lambda"],
        "ENT_COEF": ENTROPY_COEF,
    "CLIP_EPS": hyperparams["clip_eps"],
    "VF_COEF": hyperparams["vf_coef"],
    "GRAD_CLIP": hyperparams["grad_clip"],
    "FCNET_HIDDENS": hyperparams["mlp_layers"],
    "FCNET_ACTIVATION": hyperparams["fcnet_activation"],
    # Resource allocation
    "NUM_CPUS": cluster_config['num_cpus'],
    "NUM_GPUS": cluster_config['num_gpus'],
    "NUM_ENV_WORKERS": cluster_config['num_env_workers'],
    "NUM_LEARNER_WORKERS": cluster_config['num_learner_workers'],
    # Performance optimizations
    "ROLLOUT_FRAGMENT_LENGTH": hyperparams["rollout_fragment_length"],
    "BATCH_MODE": hyperparams["batch_mode"],
    "COMPRESS_OBSERVATIONS": hyperparams["compress_observations"],
    "NUM_CPUS_PER_WORKER": hyperparams["num_cpus_per_worker"],
    "NUM_GPUS_PER_WORKER": hyperparams["num_gpus_per_worker"],
    "NUM_CPUS_FOR_DRIVER": hyperparams["num_cpus_for_driver"],
}

if ray.is_initialized():
    ray.shutdown()

# Initialize Ray with optimized resource allocation
ray.init(
    num_cpus=cluster_config['num_cpus'],
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
print(f"  Environment workers: {cluster_config['num_env_workers']} (CPU)")
print(f"  Learner workers: {cluster_config['num_learner_workers']} (GPU)")
print(f"  Train batch size: {hyperparams['train_batch_size']}")
print(f"  SGD minibatch size: {hyperparams['sgd_minibatch_size']}")
print(f"  Total CPU cores: {cluster_config['num_cpus']}")
print(f"  GPU allocation: {cluster_config['num_gpus']}")

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