"""
Reward and Penalty Configuration Module

Defines reward structures and penalty systems for training.
"""

def get_penalties_config(collision_penalty, specialization_penalty_scale, collision_harshness,
                         cooperation_factor=1.0, specialization_theta=1.0):
    """
    Get penalty configuration dictionary.
    
    Args:
        collision_penalty: Penalty for collisions
        specialization_penalty_scale: Base specialization penalty scale
        collision_harshness: Collision harshness multiplier
        cooperation_factor: Map cooperation factor (higher = more cooperation needed)
        specialization_theta: Exponential sensitivity for specialization penalties
        
    Returns:
        dict: Penalty configuration
    """
    # Scale specialization penalty by cooperation factor
    # Maps requiring more cooperation get stronger specialization penalties
    effective_specialization_penalty = specialization_penalty_scale * cooperation_factor
    
    return {
        "do_nothing": 1.0,
        "useless_action": 5.0,
        "destructive_action": 10.0,
        "inaccessible_tile": 10.0,
        "blocked": 5.0,
        "collision": collision_penalty,
        "specialization_penalty_scale": effective_specialization_penalty,
        "specialization_theta": specialization_theta,
        "collision_harshness": collision_harshness,
    }


def get_rewards_config(rewards_on_delivery_only, counter_reward=0.0):
    """
    Get reward configuration dictionary.
    
    Args:
        rewards_on_delivery_only: Whether to only give rewards on delivery
        counter_reward: Reward for counter interactions
        
    Returns:
        dict: Reward configuration
    """
    if rewards_on_delivery_only:
        return {
            "raw_food": 0.0,
            "plate": 0.0,
            "counter": 0.0,
            "cut": 0.0,
            "salad": 0.0,
            "deliver": 10.0,
        }
    else:
        return {
            "raw_food": 1.0,
            "plate": 1.0,
            "counter": counter_reward,
            "cut": 4.0,
            "salad": 5.0,
            "deliver": 10.0,
        }


def get_intermediate_reward_decay_config(enable_decay=True, decay_start_episode=200, alpha=0.05):
    """
    Get intermediate reward decay configuration.
    
    Decays all intermediate rewards (everything except 'deliver') exponentially after a certain episode.
    Decay formula: reward * exp(-alpha * max(0, episode - decay_start_episode))
    
    Args:
        enable_decay: Whether to enable intermediate reward decay
        decay_start_episode: Episode number to start decaying (default: 200)
        alpha: Decay rate parameter (default: 0.05)
        
    Returns:
        dict: Intermediate reward decay configuration
    """
    return {
        "enabled": enable_decay,
        "decay_start_episode": decay_start_episode,
        "alpha": alpha,
        "intermediate_rewards": ["raw_food", "plate", "counter", "cut", "salad"],  # All except 'deliver'
    }


def apply_reward_decay(rewards_dict, episode_num, decay_config):
    """
    Apply exponential decay to intermediate rewards based on episode number.
    
    Args:
        rewards_dict: Original rewards dictionary
        episode_num: Current episode number
        decay_config: Decay configuration from get_intermediate_reward_decay_config
        
    Returns:
        dict: Rewards dictionary with decay applied
    """
    import math
    
    if not decay_config["enabled"] or episode_num < decay_config["decay_start_episode"]:
        return rewards_dict.copy()
    
    # Calculate decay factor
    episodes_since_start = episode_num - decay_config["decay_start_episode"]
    decay_factor = math.exp(-decay_config["alpha"] * episodes_since_start)
    
    # Apply decay to intermediate rewards only
    decayed_rewards = rewards_dict.copy()
    for reward_type in decay_config["intermediate_rewards"]:
        if reward_type in decayed_rewards:
            decayed_rewards[reward_type] = decayed_rewards[reward_type] * decay_factor
    
    return decayed_rewards


def get_reference_reward_config(synergy_scaling_factor, cooperation_factor, kappa, activate_synergy_positive):
    """
    Get reference reward configuration for synergy-based reward shaping.
    
    Args:
        synergy_scaling_factor: Base synergy scaling factor
        cooperation_factor: Map-specific cooperation factor
        kappa: Competence transformation parameter
        activate_synergy_positive: Whether to activate positive synergy
        
    Returns:
        dict: Reference reward configuration
    """
    effective_synergy_scaling = synergy_scaling_factor #* cooperation_factor
    
    return {
        "enabled": (synergy_scaling_factor > 0),
        "synergy_scaling_factor": effective_synergy_scaling,
        "base_synergy_scaling_factor": synergy_scaling_factor,
        "cooperation_factor": cooperation_factor,
        "kappa": kappa,
        "activate_synergy_positive": activate_synergy_positive,
    }
