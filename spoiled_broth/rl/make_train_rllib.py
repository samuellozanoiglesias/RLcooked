import os
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.core import (COMPONENT_LEARNER, COMPONENT_LEARNER_GROUP, COMPONENT_RL_MODULE)
from spoiled_broth.rl.game_env import GameEnv
from spoiled_broth.rl.dynamic_ppo_params import calculate_dynamic_ppo_params
import warnings
from datetime import datetime
import torch  # For direct state loading fallback
warnings.filterwarnings("ignore", category=DeprecationWarning)

def env_creator(config):
    # Your existing GameEnv class goes here
    return GameEnv(**config)

# Define separate policies for each agent
def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    return f"policy_{agent_id}"

def betas_tensor_to_float(learner):
    for param_grp_key in learner._optimizer_parameters.keys():
        param_grp = param_grp_key.param_groups[0]
        param_grp["betas"] = tuple(beta.item() if hasattr(beta, 'item') else beta for beta in param_grp["betas"])

def make_train_rllib(config):
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(config["SAVE_DIR"], f"Training_{current_date}")
    os.makedirs(path, exist_ok=True)
    config["PATH"] = path

    # Save config to file
    with open(os.path.join(path, "config.txt"), "w") as f:
        for key, val in config.items():
            f.write(f"{key}: {val}\n")

    # --- Load distance_map cache based on map_nr ---
    map_nr = config["MAP_NR"]
    cache_dir = os.path.join(os.path.dirname(__file__), "../maps/distance_cache")
    # Prefer the precomputed .npz cache file in the repo cache
    cache_filename = f"distance_map_{str(map_nr)}.npz"
    cache_path = os.path.join(cache_dir, cache_filename)
    distance_map_path = None
    if os.path.exists(cache_path):
        distance_map_path = cache_path
    else:
        print(f"[WARNING] .npz Distance map cache not found for map_nr '{map_nr}'. Expected path: {cache_path}")

    register_env("spoiled_broth", lambda cfg: ParallelPettingZooEnv(env_creator(cfg)))

    # --- Dynamic policy setup ---
    num_agents = config.get("NUM_AGENTS", 2)
    agent_to_train = config.get("AGENT_TO_TRAIN", None)
    
    # Create agent IDs - for single agent training, use the specific agent to train
    if num_agents == 1 and agent_to_train is not None:
        agent_ids = [f"ai_rl_{agent_to_train}"]
    else:
        agent_ids = [f"ai_rl_{i+1}" for i in range(num_agents)]
    
    policies = {}
    policies_to_train = []
    for agent_id in agent_ids:
        env_cfg = {
            "map_nr": config["MAP_NR"],
            "grid_size": config["GRID_SIZE"],
            "game_mode": config["GAME_VERSION"],
            "collision_enabled": config.get("COLLISION_ENABLED", False),  # Add collision support
            "path": path,  # Add path for collision debug logging
            "distance_map": distance_map_path,
            "walking_speeds": config.get("WALKING_SPEEDS", None),
            "cutting_speeds": config.get("CUTTING_SPEEDS", None),
            "penalties_cfg": config.get("PENALTIES_CFG", None),
            "rewards_cfg": config.get("REWARDS_CFG", None),
            "dynamic_rewards_cfg": config.get("DYNAMIC_REWARDS_CFG", None),
        }
        policies[f"policy_{agent_id}"] = (
            None,  # Use default PPO policy
            env_creator(env_cfg).observation_space(agent_id),
            env_creator(env_cfg).action_space(agent_id),
            {}
        )
        policies_to_train.append(f"policy_{agent_id}")

    start_episode = 0
    checkpoint_episodes = []
    checkpoint_log_lines = []  # Initialize early for error handling
    if config["CHECKPOINTS"] is not None: 
        for agent_id, checkpoint_info in config["CHECKPOINTS"].items():
            if checkpoint_info and "path" in checkpoint_info:
                try:
                    stats_path = os.path.join(checkpoint_info["path"], "training_stats.csv")
                    if os.path.exists(stats_path):
                        with open(stats_path, "r") as f:
                            lines = f.readlines()
                            if lines:
                                last_line = lines[-1].strip()
                                if last_line:
                                    episode_str = last_line.split(",")[0]
                                    try:
                                        checkpoint_episodes.append(int(episode_str))
                                    except Exception:
                                        pass

                except Exception as e:
                    msg = f"\nError loading {agent_id} policy: {e}"
                    checkpoint_log_lines.append(msg)
                    raise

        if checkpoint_episodes:
            start_episode = min(checkpoint_episodes) + 1

    def dynamic_policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
        return f"policy_{agent_id}"

    # --- Model config for MLP ---
    model_config = {
        "fcnet_hiddens": config.get("FCNET_HIDDENS", [256, 256]),
        "fcnet_activation": config.get("FCNET_ACTIVATION", "tanh"),
    }
    
    # Configuration for multi-agent training (time-based episodes)
    # Ensure train_batch_size is a fixed number of transitions, not tied to episode/step count
    ppo_config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .rl_module(
            model_config=model_config
        )
        .environment(
            env="spoiled_broth",
            env_config={
                "reward_weights": config["REWARD_WEIGHTS"],
                "map_nr": config["MAP_NR"],
                "game_mode": config["GAME_VERSION"],
                "inner_seconds": config["INNER_SECONDS"],
                "path": config["PATH"],
                "grid_size": config.get("GRID_SIZE", (8, 8)),
                "payoff_matrix": config.get("PAYOFF_MATRIX", [1,1,-2]),
                "initial_seed": config.get("INITIAL_SEED", 0),
                "wait_for_completion": config.get("WAIT_FOR_COMPLETION", True),
                "start_episode": start_episode,
                "distance_map": distance_map_path,
                "walking_speeds": config.get("WALKING_SPEEDS", None),
                "cutting_speeds": config.get("CUTTING_SPEEDS", None),
                "penalties_cfg": config.get("PENALTIES_CFG", None),
                "rewards_cfg": config.get("REWARDS_CFG", None),
                "dynamic_rewards_cfg": config.get("DYNAMIC_REWARDS_CFG", None),
                "collision_enabled": config.get("COLLISION_ENABLED", False),
            },
            clip_actions=True,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=dynamic_policy_mapping_fn,
            policies_to_train=policies_to_train
        )
        .resources(
            num_gpus=config["NUM_GPUS"],  # Full GPU for learner
            num_cpus_for_main_process=config.get("NUM_CPUS_FOR_DRIVER", 1)
        )
        .env_runners(
            num_env_runners=config.get("NUM_ENV_WORKERS", 4),  # Multiple environment workers
            num_gpus_per_env_runner=config.get("NUM_GPUS_PER_WORKER", 0),  # Environment workers use CPU only
            num_cpus_per_env_runner=config.get("NUM_CPUS_PER_WORKER", 1),  # CPU cores per env worker
            rollout_fragment_length=500,  # Steps per rollout (adjusted for batch size validation)
            batch_mode=config.get("BATCH_MODE", "complete_episodes"),  # Collection mode
            compress_observations=config.get("COMPRESS_OBSERVATIONS", False)  # Performance optimization
        )
        .learners(
            num_learners=config.get("NUM_LEARNER_WORKERS", 1),  # Number of learner workers
            num_gpus_per_learner=config["NUM_GPUS"],  # GPU allocation per learner
            num_cpus_per_learner=2,  # CPU cores per learner worker
        )
        .training(
            train_batch_size=config.get("TRAIN_BATCH_SIZE", 4000),  # Large batch for GPU efficiency
            minibatch_size=config.get("SGD_MINIBATCH_SIZE", 500),  # GPU-optimized minibatch
            num_epochs=config.get("NUM_SGD_ITER", 10),  # Training epochs per batch (renamed from num_sgd_iter)
            lr=config["LR"],
            gamma=config["GAMMA"],
            lambda_=config["GAE_LAMBDA"],
            entropy_coeff=config["ENT_COEF"],
            clip_param=config["CLIP_EPS"],
            vf_loss_coeff=config["VF_COEF"]
        )
    )

    # Build algorithm with error handling
    trainer = ppo_config.build_algo()
    print("Algorithm and environment successfully initialized.")

    # Initialize dynamic PPO parameters if configured
    dynamic_ppo_params_cfg = config.get("DYNAMIC_PPO_PARAMS_CFG", None)
    initial_ppo_params = {
        "clip_eps": config["CLIP_EPS"],
        "ent_coef": config["ENT_COEF"]
    }
    
    if dynamic_ppo_params_cfg and dynamic_ppo_params_cfg.get("enabled", False):
        print(f"Dynamic PPO parameters enabled with config: {dynamic_ppo_params_cfg}")
        print(f"Initial PPO params: clip_eps={initial_ppo_params['clip_eps']}, ent_coef={initial_ppo_params['ent_coef']}")

    # Load pretrained policies individually if specified in config
    if config["CHECKPOINTS"] is not None:        
        checkpoint_log_lines.append(f"Loading individual policies from checkpoints:\n{config['CHECKPOINTS']}")
        
        loaded_policies = set()
        
        # Process each policy individually - this naturally supports policy swapping
        for target_agent_id, checkpoint_info in config["CHECKPOINTS"].items():
            if checkpoint_info is None:
                checkpoint_log_lines.append(f"✓ Training from scratch {target_agent_id} (checkpoint_info = None)")
                continue
            elif not checkpoint_info or "path" not in checkpoint_info:
                checkpoint_log_lines.append(f"Skipping {target_agent_id}: invalid checkpoint info")
                continue
                
            try:
                checkpoint_path = checkpoint_info["path"]
                checkpoint_number = checkpoint_info.get("checkpoint_number", "final")
                source_policy_id = checkpoint_info.get("source_policy_id", f"policy_{target_agent_id}")
                
                full_ckpt_path = os.path.join(checkpoint_path, f"checkpoint_{checkpoint_number}")
                target_policy_id = f"policy_{target_agent_id}"
                
                if source_policy_id != target_policy_id:
                    checkpoint_log_lines.append(f"Policy swapping: Loading {target_policy_id} from {source_policy_id} in {full_ckpt_path}")
                else:
                    checkpoint_log_lines.append(f"No policy swapping: Loading {target_policy_id} from {source_policy_id} in {full_ckpt_path}")
                
                # Construct the source path to the specific policy's RLModule state
                source_policy_path = (
                    Path(full_ckpt_path) 
                    / COMPONENT_LEARNER_GROUP  # learner_group
                    / COMPONENT_LEARNER        # learner  
                    / COMPONENT_RL_MODULE      # rl_module
                    / source_policy_id         # policy_ai_rl_X
                )
                
                if source_policy_path.exists():
                    # Target component path in current trainer
                    target_component_path = (
                        COMPONENT_LEARNER_GROUP + "/" + 
                        COMPONENT_LEARNER + "/" + 
                        COMPONENT_RL_MODULE + "/" + 
                        target_policy_id
                    )
                    
                    trainer.restore_from_path(
                        source_policy_path,
                        component=target_component_path
                    )
                    
                    loaded_policies.add(target_policy_id)
                    checkpoint_log_lines.append(f"✓ Successfully loaded {target_policy_id} using restore_from_path")
                    
                else:
                    checkpoint_log_lines.append(f"✗ Source policy path does not exist: {source_policy_path}")
                    raise FileNotFoundError(f"Policy path not found: {source_policy_path}")
                        
            except Exception as e:
                checkpoint_log_lines.append(f"✗ Error processing {target_agent_id}: {e}")
        
        # Apply beta tensor fix to main trainer after all loading is complete
        try:
            trainer.learner_group.foreach_learner(betas_tensor_to_float)
            checkpoint_log_lines.append("Applied beta tensor fix to loaded policies")
        except Exception as beta_error:
            checkpoint_log_lines.append(f"Warning: Beta tensor fix failed: {beta_error}")
        
        checkpoint_log_lines.append(f"Individual policy loading completed. Successfully loaded: {loaded_policies}")


    else:
        checkpoint_log_lines.append("No checkpoint specified, training from scratch.")
    
    # Write checkpoint log to file in the training directory
    log_path = os.path.join(path, "checkpoint_load_log.txt")
    with open(log_path, "w") as logf:
        for line in checkpoint_log_lines:
            logf.write(line + "\n")        

    # Save initial checkpoint at episode 0 (or start_episode if resuming)
    initial_checkpoint_path = trainer.save(os.path.join(path, f"checkpoint_{start_episode}"))
    print(f"Initial checkpoint saved at {initial_checkpoint_path} (episode {start_episode})")

    for epoch in range(config["NUM_EPOCHS"]):
        # Update dynamic PPO parameters if configured
        if dynamic_ppo_params_cfg and dynamic_ppo_params_cfg.get("enabled", False):
            # Get current episode count from training_stats.csv for parameter updates
            current_episode = epoch  # fallback to epoch if CSV reading fails
            stats_csv_path = os.path.join(path, "training_stats.csv")
            if os.path.exists(stats_csv_path):
                try:
                    with open(stats_csv_path, "r") as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if last_line:
                                current_episode = int(last_line.split(",")[0])
                except Exception:
                    pass  # Use fallback value
            
            # Calculate new PPO parameters using the dynamic_ppo_params module
            updated_ppo_params = calculate_dynamic_ppo_params(
                current_episode,
                initial_ppo_params,
                dynamic_ppo_params_cfg
            )
            
            # Update the algorithm's configuration 
            # Note: For RLlib, we'll update the config dict and try to apply changes
            try:
                if "clip_eps" in updated_ppo_params:
                    trainer.config["clip_param"] = updated_ppo_params["clip_eps"]
                if "ent_coef" in updated_ppo_params:
                    trainer.config["entropy_coeff"] = updated_ppo_params["ent_coef"]
                
                # Some RLlib versions might require updating the learner config as well
                if hasattr(trainer, 'learner_group') and trainer.learner_group:
                    def update_learner_config(learner):
                        if "clip_eps" in updated_ppo_params:
                            learner.config["clip_param"] = updated_ppo_params["clip_eps"]
                        if "ent_coef" in updated_ppo_params:
                            learner.config["entropy_coeff"] = updated_ppo_params["ent_coef"]
                    trainer.learner_group.foreach_learner(update_learner_config)
                    
            except Exception as e:
                # If direct config update fails, log it but continue training
                print(f"Warning: Could not update PPO parameters dynamically: {e}")
            
            # Log parameter changes periodically
            if current_episode % 100 == 0 and current_episode > 0:
                affected_params = dynamic_ppo_params_cfg.get("affected_params", [])
                print(f"[Episode {current_episode}] Updated PPO parameters:")
                for param_name in affected_params:
                    if param_name in updated_ppo_params:
                        initial_val = initial_ppo_params[param_name]
                        current_val = updated_ppo_params[param_name]
                        print(f"  {param_name}: {initial_val:.4f} -> {current_val:.4f}")

        result = trainer.train()

        # Log metrics
        if (epoch - 1) % config["SHOW_EVERY_N_EPOCHS"] == 0:
            print(f"Epoch {epoch} complete.")

        # Save checkpoint
        if (epoch - 1) % config["SAVE_EVERY_N_EPOCHS"] == 0:
            # Get current episode count from training_stats.csv for checkpoint naming
            current_episode = epoch  # fallback to epoch if CSV reading fails
            stats_csv_path = os.path.join(path, "training_stats.csv")
            if os.path.exists(stats_csv_path):
                try:
                    with open(stats_csv_path, "r") as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if last_line:
                                current_episode = int(last_line.split(",")[0])
                except Exception:
                    pass  # Use fallback value
            
            checkpoint_path = trainer.save(os.path.join(path, f"checkpoint_{current_episode}"))
            print(f"Checkpoint saved at {checkpoint_path} (episode {current_episode})")

    # Get the final episode count from training_stats.csv
    final_episode_count = None
    stats_csv_path = os.path.join(path, "training_stats.csv")
    if os.path.exists(stats_csv_path):
        try:
            with open(stats_csv_path, "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        final_episode_count = int(last_line.split(",")[0])
        except Exception as e:
            print(f"Warning: Could not read final episode count from CSV: {e}")

    return trainer, current_date, final_episode_count