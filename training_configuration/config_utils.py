"""
Configuration Utilities Module

Handles argument parsing, validation, and configuration setup.
"""

import os


def parse_input_file(input_path):
    """
    Parse the input file to extract agent parameters.
    
    Args:
        input_path: Path to the input file
        
    Returns:
        dict: Dictionary containing agent parameters
    """
    agent_params = {}
    
    with open(input_path, "r") as f:
        lines = f.readlines()
        for i in range(len(lines) // 2):
            alpha, beta = [round(float(x), 4) for x in lines[2*i].strip().split()]
            walking_speed, cutting_speed = [round(float(x), 4) for x in lines[2*i + 1].strip().split()]
            
            agent_params[f"alpha_{i+1}"] = alpha
            agent_params[f"beta_{i+1}"] = beta
            agent_params[f"walking_speed_{i+1}"] = walking_speed
            agent_params[f"cutting_speed_{i+1}"] = cutting_speed
    
    return agent_params


def setup_agent_configurations(agent_params, num_agents, agent_to_train=None):
    """
    Setup agent-specific configurations.
    
    Args:
        agent_params: Dictionary of agent parameters
        num_agents: Number of agents
        agent_to_train: Specific agent to train (for single agent mode)
        
    Returns:
        tuple: (reward_weights, walking_speeds, cutting_speeds)
    """
    reward_weights = {}
    walking_speeds = {}
    cutting_speeds = {}
    
    if num_agents == 1 and agent_to_train is not None:
        reward_weights[f"ai_rl_{agent_to_train}"] = (
            agent_params[f"alpha_{agent_to_train}"], 
            agent_params[f"beta_{agent_to_train}"]
        )
        walking_speeds[f"ai_rl_{agent_to_train}"] = agent_params[f"walking_speed_{agent_to_train}"]
        cutting_speeds[f"ai_rl_{agent_to_train}"] = agent_params[f"cutting_speed_{agent_to_train}"]
    else:
        for i in range(1, num_agents + 1):
            reward_weights[f"ai_rl_{i}"] = (agent_params[f"alpha_{i}"], agent_params[f"beta_{i}"])
            walking_speeds[f"ai_rl_{i}"] = agent_params[f"walking_speed_{i}"]
            cutting_speeds[f"ai_rl_{i}"] = agent_params[f"cutting_speed_{i}"]
    
    return reward_weights, walking_speeds, cutting_speeds


def parse_pretrained_policies(checkpoint_paths, num_agents, agent_to_train=None):
    """
    Parse pretrained policy configuration from file.
    
    Args:
        checkpoint_paths: Path to checkpoint configuration file
        num_agents: Number of agents
        agent_to_train: Specific agent to train (for single agent mode)
        
    Returns:
        dict: Pretrained policies configuration
    """
    if checkpoint_paths.lower() == "none":
        return None
    
    pretrained_policies = {}
    
    with open(checkpoint_paths, "r") as f:
        lines = f.readlines()
        
        if num_agents == 1 and agent_to_train is not None:
            policy_id = str(lines[0]).strip()
            checkpoint_number = str(lines[1]).strip()
            checkpoint_path = str(lines[2]).strip()
            
            if (policy_id.lower() != "none" and 
                checkpoint_number.lower() != "none" and 
                checkpoint_path.lower() != "none"):
                pretrained_policies[f"ai_rl_{agent_to_train}"] = {
                    "source_policy_id": policy_id,
                    "checkpoint_number": checkpoint_number,
                    "path": checkpoint_path
                }
            else:
                pretrained_policies[f"ai_rl_{agent_to_train}"] = None
        else:
            for i in range(num_agents):
                policy_id = str(lines[3*i]).strip()
                checkpoint_number = str(lines[3*i + 1]).strip()
                checkpoint_path = str(lines[3*i + 2]).strip()
                
                if (policy_id.lower() != "none" and 
                    checkpoint_number.lower() != "none" and 
                    checkpoint_path.lower() != "none"):
                    pretrained_policies[f"ai_rl_{i+1}"] = {
                        "source_policy_id": policy_id,
                        "checkpoint_number": checkpoint_number,
                        "path": checkpoint_path
                    }
                else:
                    pretrained_policies[f"ai_rl_{i+1}"] = None
    
    return pretrained_policies


def parse_game_version(game_version):
    """
    Parse game version to extract collision mode.
    
    Args:
        game_version: Game version string
        
    Returns:
        tuple: (base_game_version, collision_enabled)
    """
    collision_enabled = game_version.endswith('_collision')
    base_game_version = game_version.replace('_collision', '') if collision_enabled else game_version
    
    return base_game_version, collision_enabled


def get_hyperparameters():
    """
    Get default hyperparameters for training.
    
    Returns:
        dict: Hyperparameters configuration
    """
    return {
        "inner_seconds": 180,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 500,
        "num_sgd_iter": 10,
        "show_every_n_epochs": 1,
        "save_every_n_epochs": 1000,
        "payoff_matrix": [1, 1, -2],
        "mlp_layers": [1024, 512, 256],
        "gamma": 0.9,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "clip_eps": 0.3,
        "vf_coef": 1.0,
        "grad_clip": 0.5,
        "fcnet_activation": "tanh",
        "rollout_fragment_length": "auto",
        "batch_mode": "complete_episodes",
        "compress_observations": False,
        "num_cpus_per_worker": 1,
        "num_gpus_per_worker": 0,
        "num_cpus_for_driver": 1,
    }


def validate_configuration(num_agents, synergy_scaling_factor, agent_to_train=None):
    """
    Validate configuration parameters.
    
    Args:
        num_agents: Number of agents
        synergy_scaling_factor: Synergy scaling factor
        agent_to_train: Agent to train (for single agent mode)
    """
    if num_agents not in [1, 2]:
        raise ValueError("NUM_AGENTS must be 1 or 2")
    
    if num_agents == 1 and agent_to_train is not None and agent_to_train not in [1, 2]:
        raise ValueError("When NUM_AGENTS=1, agent_to_train must be 1 or 2")
    
    if synergy_scaling_factor > 0 and num_agents != 2:
        raise ValueError("Reference-based reward shaping is currently only supported for 2-agent teams.")