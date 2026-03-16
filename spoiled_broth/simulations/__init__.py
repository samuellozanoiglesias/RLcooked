"""
Simulation utilities package for running reinforcement learning simulations.

This package provides modular components for:
- Configuration management
- Path management
- Data logging (actions, positions, counters)
- Video recording
- Policy loading from RLlib checkpoints
- Ray cluster management
- Complete simulation execution using GameEnv (same dynamics as RL training)

Author: Samuel Lozano
"""

# Configuration
from .simulation_config import SimulationConfig

# Core managers
from .path_manager import PathManager
from .ray_manager import RayManager

# Data handling
from .data_logger import DataLogger
from .video_recorder import VideoRecorder

# Policy loading
from .policy_loader import PolicyInferrer, load_policies

# Main simulation runner
from .simulation_runner import SimulationRunner

# Utility functions
from .simulation_utils import (
    setup_simulation_argument_parser,
    main_simulation_pipeline
)

# Video replay from CSV
from .replay_simulation_video import replay_from_directory

__all__ = [
    # Configuration
    'SimulationConfig',

    # Core managers
    'PathManager',
    'RayManager',

    # Data handling
    'DataLogger',
    'VideoRecorder',

    # Policy loading
    'PolicyInferrer',
    'load_policies',

    # Main simulation runner
    'SimulationRunner',

    # Utility functions
    'setup_simulation_argument_parser',
    'main_simulation_pipeline',
    
    # Video replay
    'replay_from_directory',
]
