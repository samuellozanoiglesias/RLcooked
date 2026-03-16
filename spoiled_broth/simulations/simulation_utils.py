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
        nargs='?',
        default='custom',
        help='Training identifier (default: "custom" when using --custom_checkpoints)'
    )
    
    parser.add_argument(
        'checkpoint_number',
        type=str,
        nargs='?',
        default='custom',
        help='Checkpoint number to load (integer) or "final" for the latest checkpoint (default: "custom" when using --custom_checkpoints)'
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
        default='false',
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
        default='',
        help='Study name for organizing simulations (e.g., "speeds", "collision"). If empty, simulations are saved directly under /simulations/ without study subfolder.'
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

    Returns:
        Dictionary containing output file paths.
    """
    # Parse checkpoint configuration if provided
    pretrained_checkpoints = None
    using_custom_checkpoints = custom_checkpoints is not None and custom_checkpoints.lower() != 'none'
    
    if using_custom_checkpoints:
        pretrained_checkpoints = {}
        try:
            with open(custom_checkpoints, "r") as f:
                lines = f.readlines()
            for i in range(num_agents):
                policy_id = str(lines[3 * i]).strip()
                chk_number = str(lines[3 * i + 1]).strip()
                chk_path = str(lines[3 * i + 2]).strip()
                agent_key = f"ai_rl_{i + 1}"
                if (policy_id.lower() != "none"
                        and chk_number.lower() != "none"
                        and chk_path.lower() != "none"):
                    pretrained_checkpoints[agent_key] = {
                        "loaded_agent_id": policy_id,
                        "checkpoint_number": chk_number,
                        "path": chk_path,
                    }
                else:
                    pretrained_checkpoints[agent_key] = None
        except Exception as e:
            print(f"Warning: Could not parse checkpoint config '{custom_checkpoints}': {e}")
            print("Using default checkpoint loading.")
            pretrained_checkpoints = None
            using_custom_checkpoints = False
    
    # If using custom checkpoints and training_id/checkpoint_number are defaults,
    # extract them from the first agent's checkpoint path
    synergy_folder = None
    specialization_folder = None
    
    if using_custom_checkpoints and training_id == "custom" and checkpoint_number == "custom":
        first_agent_info = pretrained_checkpoints.get("ai_rl_1")
        if first_agent_info:
            # Extract training_id from path (e.g., "/path/Training_2025-11-15_13-23-45" -> "2025-11-15_13-23-45")
            training_path = Path(first_agent_info["path"])
            training_folder_name = training_path.name
            if training_folder_name.startswith("Training_"):
                training_id = training_folder_name.replace("Training_", "")
                print(f"Extracted training_id from custom checkpoint: {training_id}")
            
            # Use the first agent's checkpoint number
            checkpoint_number = first_agent_info["checkpoint_number"]
            print(f"Using checkpoint_number from custom checkpoint: {checkpoint_number}")
            
            # Extract synergy and specialization folders from training path
            # Path format: .../map_name/synergy_X.XX/specialized_X.XX/Training_ID/
            path_parts = training_path.parts
            for i, part in enumerate(path_parts):
                if part.startswith("synergy_"):
                    synergy_folder = part
                    print(f"Extracted synergy folder: {synergy_folder}")
                if part.startswith("specialized_"):
                    specialization_folder = part
                    print(f"Extracted specialization folder: {specialization_folder}")
    
    # Create a temporary config to get paths for loading speeds
    temp_config = SimulationConfig(cluster=cluster)
    temp_path_manager = PathManager(temp_config)
    
    # Load agent speeds - use custom checkpoint paths if available, otherwise use training path
    walking_speeds = {}
    cutting_speeds = {}
    
    if using_custom_checkpoints:
        # Load speeds from each agent's respective training config
        print("Loading agent speeds from custom checkpoint training configs...")
        for i in range(1, num_agents + 1):
            agent_key = f"ai_rl_{i}"
            agent_info = pretrained_checkpoints.get(agent_key)
            
            if agent_info and agent_info is not None:
                agent_training_path = Path(agent_info["path"])
                # Load all speeds from the training config
                agent_walking_speeds, agent_cutting_speeds = temp_path_manager.load_agent_speeds_from_training(
                    agent_training_path, num_agents=num_agents
                )
                
                # Extract the speed for the specific policy being loaded
                # Convert policy_ai_rl_1 -> ai_rl_1
                loaded_policy_id = agent_info["loaded_agent_id"]
                if loaded_policy_id.startswith("policy_"):
                    loaded_agent_id = loaded_policy_id.replace("policy_", "")
                else:
                    loaded_agent_id = "ai_rl_1"  # fallback
                
                if agent_walking_speeds and agent_cutting_speeds:
                    # Use the speed for the specific agent being loaded
                    walking_speed = agent_walking_speeds.get(loaded_agent_id, 1.0)
                    cutting_speed = agent_cutting_speeds.get(loaded_agent_id, 1.0)
                    walking_speeds[agent_key] = walking_speed
                    cutting_speeds[agent_key] = cutting_speed
                    print(f"  {agent_key}: walking={walking_speed}, cutting={cutting_speed} (policy={loaded_policy_id} from {agent_training_path.name})")
                else:
                    # Fallback to defaults
                    walking_speeds[agent_key] = 1.0
                    cutting_speeds[agent_key] = 1.0
                    print(f"  {agent_key}: using default speeds (config not found)")
            else:
                # Fallback to defaults
                walking_speeds[agent_key] = 1.0
                cutting_speeds[agent_key] = 1.0
                print(f"  {agent_key}: using default speeds (no custom checkpoint)")
    else:
        # Load speeds from the specified training path (original behavior)
        temp_paths = temp_path_manager.setup_paths(
            map_nr, num_agents, game_version, training_id, checkpoint_number, study_name, game_type,
            synergy_folder, specialization_folder
        )
        walking_speeds, cutting_speeds = temp_path_manager.load_agent_speeds_from_training(
            temp_paths['training_path'], num_agents
        )

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
        custom_checkpoints=pretrained_checkpoints,
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
        game_type=game_type,
        synergy_folder=synergy_folder,
        specialization_folder=specialization_folder,
    )

    return output_paths
