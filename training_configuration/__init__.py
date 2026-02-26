# Training Configuration Package
# Contains modular components for RL training setup

"""
Training Configuration Package

This package provides modular components for configuring reinforcement learning training:

- cooperation_factor: Mathematical calculation of cooperation necessity based on map analysis
- baseline_lookup: Performance baseline data and lookup functions 
- cluster_config: Cluster-specific resource allocation configurations
- reward_penalties: Reward structures and penalty system configurations
- path_utils: Path generation and directory management utilities
- config_utils: Configuration parsing, validation, and setup utilities

Usage:
    from training_configuration.cooperation_factor import get_cooperation_factor
    from training_configuration.cluster_config import get_cluster_config
    # ... etc
"""

from .cooperation_factor import get_cooperation_factor, calculate_cooperation_factor
from .baseline_lookup import lookup_solo_baseline, BASELINE_LOOKUP
from .cluster_config import get_cluster_config
from .reward_penalties import get_penalties_config, get_rewards_config, get_reference_reward_config
from .path_utils import generate_save_directory, get_map_grid_size
from .config_utils import (
    parse_input_file, setup_agent_configurations, parse_pretrained_policies,
    parse_game_version, get_hyperparameters, validate_configuration
)

__all__ = [
    'get_cooperation_factor',
    'calculate_cooperation_factor', 
    'lookup_solo_baseline',
    'BASELINE_LOOKUP',
    'get_cluster_config',
    'get_penalties_config',
    'get_rewards_config', 
    'get_reference_reward_config',
    'generate_save_directory',
    'get_map_grid_size',
    'parse_input_file',
    'setup_agent_configurations',
    'parse_pretrained_policies',
    'parse_game_version',
    'get_hyperparameters',
    'validate_configuration'
]
