"""
Path management utilities for simulation runs.

Author: Samuel Lozano
"""

import os
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional

from .simulation_config import SimulationConfig


class PathManager:
    """Manages paths and directories for simulation runs."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config

    def _find_last_episode_in_training_stats(self, training_path: Path) -> Optional[int]:
        """
        Find the last episode number in the training_stats folder.
        
        Args:
            training_path: Path to the training folder
            training_id: Training identifier
            
        Returns:
            Last episode number as string, or None if not found
        """
        training_stats_path = training_path / "training_stats.csv"
        
        if not training_stats_path.exists():
            return None
    
        max_episode = -1

        try:
            with open(training_stats_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)

                if 'episode' not in reader.fieldnames:
                    print(f"Warning: 'episode' column not found in {training_stats_path}")
                    return None

                for row in reader:
                    try:
                        episode_value = int(row['episode'])
                        if episode_value > max_episode:
                            max_episode = episode_value
                    except ValueError:
                        continue
                    
            if max_episode >= 0:
                return max_episode

        except Exception as e:
            print(f"Warning: Could not read or process training_stats file {training_stats_path}: {e}")

        return None

    def setup_paths(self, map_nr: str, num_agents: int,
                   game_version: str, training_id: str,
                   checkpoint_number: str, study_name: str = 'default',
                   game_type: str = '') -> Dict[str, Path]:
        """
        Set up all necessary paths for a simulation run.
        
        Args:
            map_nr: Name of the map
            num_agents: Number of agents
            game_version: Game version identifier
            training_id: Training identifier
            checkpoint_number: Checkpoint number to load (integer or "final")
            
        Returns:
            Dictionary containing all relevant paths
        """
        # Updated path structure: 
        # If game_type is empty: /data/samuel_lozano/cooked/map_{map_nr}/simulations/{study_name}/Training_{training_id}/checkpoint_{checkpoint_number}/
        # If game_type is specified: /data/samuel_lozano/cooked/{game_type}/map_{map_nr}/simulations/{study_name}/Training_{training_id}/checkpoint_{checkpoint_number}/
        # Note: game_type (e.g., 'classic', 'classic_collision') determines folder, while game_version affects game logic
        if num_agents == 1:
            if game_type:
                base_path = Path(f"{self.config.local_path}/data/samuel_lozano/cooked/pretraining/{game_type}/map_{map_nr}")
            else:
                base_path = Path(f"{self.config.local_path}/data/samuel_lozano/cooked/pretraining/map_{map_nr}")
        else:
            if game_type:
                base_path = Path(f"{self.config.local_path}/data/samuel_lozano/cooked/{game_type}/map_{map_nr}")
            else:
                base_path = Path(f"{self.config.local_path}/data/samuel_lozano/cooked/map_{map_nr}")

        # New structure: all simulations go under /simulations/{study_name}/Training_{training_id}/checkpoint_{checkpoint_number}/
        simulations_base = base_path / "simulations" / study_name
        training_simulations_path = simulations_base / f"Training_{training_id}"

        # Training path for model checkpoints - now organized by study
        # If study_name is 'default', keep backward compatibility with old structure
        if study_name == 'default':
            training_path = base_path / f"Training_{training_id}"
        else:
            # For named studies, look for training data within the study folder
            training_path = base_path / study_name / f"Training_{training_id}"
        
        checkpoint_dir = training_path / f"checkpoint_{checkpoint_number}"
        
        # Handle "final" checkpoint by finding the last episode
        checkpoint_number_for_folder = checkpoint_number
        if checkpoint_number.lower() == "final":            
            last_episode = self._find_last_episode_in_training_stats(training_path)
            if last_episode is not None:
                checkpoint_number_for_folder = str(last_episode + 1)
                print(f"Found last episode {checkpoint_number_for_folder} for final checkpoint")
            else:
                print(f"Warning: Could not find last episode in training_stats, using 'final' for folder name")
                checkpoint_number_for_folder = "final"
        
        # Validate checkpoint_number (only if it's not "final")
        if checkpoint_number.lower() != "final":
            try:
                int(checkpoint_number)  # Validate it's a number
            except ValueError:
                raise ValueError(f"Invalid checkpoint number: '{checkpoint_number}'. Must be an integer or 'final'")
        
        # Construct paths based on checkpoint type - use checkpoint_number_for_folder for the folder name
        checkpoint_simulations_dir = training_simulations_path / f"checkpoint_{checkpoint_number_for_folder}"
        
        # The saving path is now the checkpoint simulations directory where individual simulation folders will be created
        saving_path = checkpoint_simulations_dir
        
        # Find project root by looking for spoiled_broth directory
        project_root = Path(__file__).parent.parent.parent  # path_manager.py is in spoiled_broth/simulations/
        
        paths = {
            'base_path': base_path,
            'training_path': training_path,
            'checkpoint_dir': checkpoint_dir,
            'saving_path': saving_path,
            'simulations_base': simulations_base,
            'training_simulations_path': training_simulations_path,
            'checkpoint_simulations_dir': checkpoint_simulations_dir,
            'path_root': project_root / "spoiled_broth",
            'map_txt_path': project_root / "spoiled_broth" / "maps" / f"{map_nr}.txt",
            'config_path': training_path / "config.txt",
            'checkpoint_number': checkpoint_number,  # Keep original checkpoint_number
            'checkpoint_number_for_folder': checkpoint_number_for_folder  # Actual number used for folder name
        }
        
        # Create directories
        os.makedirs(paths['saving_path'], exist_ok=True)
        
        return paths
    
    def get_grid_size_from_map(self, map_txt_path: Path) -> Tuple[int, int]:
        """
        Determine grid size from map file.
        
        Args:
            map_txt_path: Path to map text file
            
        Returns:
            Tuple of (width, height) for the grid
        """
        try:
            with open(map_txt_path) as f:
                lines = [l.rstrip("\n") for l in f.readlines()]
            if lines:
                return (len(lines[0]), len(lines))
        except Exception as e:
            print(f"Warning: Could not read map file {map_txt_path}: {e}")
        
        return self.config.default_grid_size
    
    def load_agent_speeds_from_training(self, training_path: Path, num_agents: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Load agent walking and cutting speeds from training config file.
        
        Args:
            training_path: Path to the training directory
            num_agents: Number of agents
            
        Returns:
            Tuple of (walking_speeds, cutting_speeds) dictionaries
        """
        walking_speeds = {}
        cutting_speeds = {}
        
        # Look for config.txt file in training directory
        config_file = training_path / "config.txt"
        
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    lines = f.readlines()
                
                # Parse config file to extract speeds
                for line in lines:
                    line = line.strip()
                    if line.startswith("WALKING_SPEEDS:"):
                        # Extract dictionary from line like "WALKING_SPEEDS: {'ai_rl_1': 1.0, 'ai_rl_2': 1.0}"
                        dict_str = line.split(":", 1)[1].strip()
                        walking_speeds = eval(dict_str)  # Safe since this is our own config file
                        print(f"Loaded walking speeds from {config_file}: {walking_speeds}")
                    elif line.startswith("CUTTING_SPEEDS:"):
                        # Extract dictionary from line like "CUTTING_SPEEDS: {'ai_rl_1': 1.0, 'ai_rl_2': 1.0}"
                        dict_str = line.split(":", 1)[1].strip()
                        cutting_speeds = eval(dict_str)  # Safe since this is our own config file
                        print(f"Loaded cutting speeds from {config_file}: {cutting_speeds}")
                        
            except Exception as e:
                print(f"Warning: Could not read speeds from {config_file}: {e}")
        else:
            print(f"Warning: Config file not found at {config_file}")
        
        # Set defaults if not loaded
        for i in range(1, num_agents + 1):
            agent_id = f"ai_rl_{i}"
            if agent_id not in walking_speeds:
                walking_speeds[agent_id] = 1.0
                print(f"Using default walking speed 1.0 for {agent_id}")
            if agent_id not in cutting_speeds:
                cutting_speeds[agent_id] = 1.0
                print(f"Using default cutting speed 1.0 for {agent_id}")
        
        return walking_speeds, cutting_speeds