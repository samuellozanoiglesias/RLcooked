"""
Utility functions for simulation pipeline.

Author: Samuel Lozano
"""

import time
import argparse
from pathlib import Path
from typing import Dict

from .simulation_config import SimulationConfig
from .simulation_runner import SimulationRunner
from .path_manager import PathManager


def setup_simulation_argument_parser() -> argparse.ArgumentParser:
    """
    Set up command line argument parser for simulations.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Run reinforcement learning simulations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'map_nr',
        type=str,
        help='Map name identifier'
    )
    
    parser.add_argument(
        'game_version',
        type=str,
        help='Game version identifier'
    )
    
    parser.add_argument(
        'training_id',
        type=str,
        help='Training identifier'
    )
    
    parser.add_argument(
        'checkpoint_number',
        type=str,
        help='Checkpoint number to load (integer) or "final" for the latest checkpoint'
    )

    parser.add_argument(
        '--num_agents',
        type=int,
        default=2,
        help='Number of agents in the simulation'
    )
    
    parser.add_argument(
        '--enable_video',
        type=str,
        choices=['true', 'false'],
        default='true',
        help='Enable video recording'
    )
    
    parser.add_argument(
        '--cluster',
        type=str,
        choices=['brigit', 'cuenca', 'local'],
        default='cuenca',
        help='Cluster type for path configuration'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=180,
        help='Simulation duration in seconds'
    )

    parser.add_argument(
        '--agent_initialization_period',
        type=float,
        default=15.0,
        help='Agent initialization period in seconds (no actions taken during this time)'
    )
    
    parser.add_argument(
        '--tick-rate',
        type=int,
        default=24,
        help='Engine tick rate (frames per second)'
    )
    
    parser.add_argument(
        '--video-fps',
        type=int,
        default=24,
        help='Video recording frame rate'
    )
    
    parser.add_argument(
        '--custom_checkpoints',
        type=str,
        default='none',
        help='Path to checkpoint configuration file with policy IDs and paths for each agent (format: policy_id\npath_to_checkpoint per agent), or "none" to use default checkpoint loading'
    )
    
    parser.add_argument(
        '--study_name',
        type=str,
        default='default',
        help='Study name for organizing simulations (e.g., "speeds", "collision")'
    )
    
    parser.add_argument(
        '--game_type',
        type=str,
        default='',
        help='Game type for folder organization (e.g., "classic", "classic_collision"). If empty, no game_type subfolder is used.'
    )
    
    return parser


def main_simulation_pipeline(map_nr: str, num_agents: int,
                           game_version: str, training_id: str,
                           checkpoint_number: str, enable_video: bool = True,
                           cluster: str = 'cuenca', duration: int = 180,
                           tick_rate: int = 24, video_fps: int = 24,
                           agent_initialization_period: float = 15.0,
                           custom_checkpoints: str = 'none',
                           study_name: str = 'default',
                           game_type: str = '') -> Dict[str, Path]:
    """
    Main simulation pipeline that can be used by different simulation scripts.
    
    Args:
        MAP_NR: Map name identifier
        NUM_AGENTS: Number of agents
        GAME_VERSION: Game version identifier
        TRAINING_ID: Training identifier
        CHECKPOINT_NUMBER: Checkpoint number (integer or "final")
        ENABLE_VIDEO: Whether to enable video recording
        CLUSTER: Cluster type
        DURATION: Simulation duration in seconds
        TICK_RATE: Engine tick rate
        VIDEO_FPS: Video frame rate
        AGENT_INITIALIZATION_PERIOD: Time period for agent initialization in seconds
        CHECKPOINTS: Path to checkpoint configuration file or "none" for default behavior
    Returns:
        Dictionary containing output file paths
    """
    # Create a temporary config to get paths for loading speeds
    temp_config = SimulationConfig(cluster=cluster)
    temp_path_manager = PathManager(temp_config)
    temp_paths = temp_path_manager.setup_paths(
        map_nr, num_agents, game_version, training_id, checkpoint_number, study_name, game_type
    )
    
    # Load agent speeds from training configuration
    walking_speeds, cutting_speeds = temp_path_manager.load_agent_speeds_from_training(
        temp_paths['training_path'], num_agents
    )
    
    # Parse checkpoint configuration if provided
    pretrained_checkpoints = None
    if custom_checkpoints is not None and custom_checkpoints.lower() != 'none':
        pretrained_checkpoints = {}
        try:
            with open(custom_checkpoints, "r") as f:
                lines = f.readlines()
                for i in range(num_agents):
                    policy_id = str(lines[3*i]).strip()
                    checkpoint_number = str(lines[3*i + 1]).strip()
                    checkpoint_path = str(lines[3*i + 2]).strip()
                    if policy_id.lower() != "none" and checkpoint_number.lower() != "none" and checkpoint_path.lower() != "none":
                        pretrained_checkpoints[f"ai_rl_{i+1}"] = {"loaded_agent_id": policy_id, "checkpoint_number": checkpoint_number, "path": checkpoint_path}
                    else:
                        pretrained_checkpoints[f"ai_rl_{i+1}"] = None
        except Exception as e:
            print(f"Warning: Could not parse checkpoint configuration file '{custom_checkpoints}': {e}")
            print("Using default checkpoint loading behavior")
            pretrained_checkpoints = None
    
    # Create configuration with loaded speeds
    config = SimulationConfig(
        cluster=cluster,
        engine_tick_rate=tick_rate,
        duration_seconds=duration,
        enable_video=enable_video,
        video_fps=video_fps,
        agent_initialization_period=agent_initialization_period,
        walking_speeds=walking_speeds,
        cutting_speeds=cutting_speeds,
        custom_checkpoints=pretrained_checkpoints
    )
    
    # Create timestamp
    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    
    # Create and run simulation
    runner = SimulationRunner(config)
    
    output_paths = runner.run_simulation(
        map_nr=map_nr,
        num_agents=num_agents,
        game_version=game_version,
        training_id=training_id,
        checkpoint_number=checkpoint_number,
        timestamp=timestamp,
        study_name=study_name,
        game_type=game_type
    )
    
    return output_paths, walking_speeds, cutting_speeds