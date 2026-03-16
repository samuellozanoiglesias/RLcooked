"""
Utility functions for analyzing training results from reinforcement learning experiments.

This module provides common functionality for:
- Data loading and preprocessing
- Configuration parsing
- Directory management
- Plotting utilities
- Statistical analysis

Author: Samuel Lozano
"""

import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from io import StringIO


class AnalysisConfig:
    """Configuration class for analysis parameters."""
    
    def __init__(self):
        # Graph settings
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        
        # Default smoothing factor
        self.smoothing_factor = 15
        
        # Cluster configurations
        self.cluster_paths = {
            'brigit': '/mnt/lustre/home/samuloza',
            'cuenca': '',
            'local': 'D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado'
        }


class DataProcessor:
    """Handles data loading, preprocessing and CSV operations."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.reward_pattern = re.compile(
            r"'([^']+)':\s*\(\s*([-\d\.eE+]+),\s*([-\d\.eE+]+)\)"
        )
        # Cache: (raw_dir, num_agents, study_name) -> DataFrame
        # Avoids re-reading the same CSV directory multiple times (e.g. for different x_val filters)
        self._dir_cache: Dict = {}
    
    def setup_directories(self, experiment_type: str, map_name: str, cluster: str = 'cuenca', study_name: str = None) -> Dict[str, str]:
        """
        Set up directory structure for analysis.
        
        Args:
            experiment_type: Type of experiment ('classic', 'competition', 'pretrained')
            map_name: Map name identifier  
            cluster: Cluster type ('brigit', 'cuenca', 'local')
            study_name: Study name for specific study folders (optional)
            
        Returns:
            Dictionary containing all relevant paths
        """
        if cluster not in self.config.cluster_paths:
            raise ValueError(f"Invalid cluster '{cluster}'. Choose from {list(self.config.cluster_paths.keys())}")
        
        local_path = self.config.cluster_paths[cluster]
        
        if study_name:
            # Use study_name folder structure: /data/samuel_lozano/cooked/{experiment_type}/map_{map_name}/{study_name}/
            raw_dir = f"{local_path}/data/samuel_lozano/cooked/{experiment_type}/map_{map_name}/{study_name}"
        else:
            # Default structure: /data/samuel_lozano/cooked/{experiment_type}/map_{map_name}/
            raw_dir = f"{local_path}/data/samuel_lozano/cooked/{experiment_type}/map_{map_name}"
        
        paths = {
            'raw_dir': raw_dir,
            'output_path': f"{raw_dir}/training_results.csv",
            'figures_dir': f"{raw_dir}/training_figures/",
            'smoothed_figures_dir': f"{raw_dir}/training_figures/smoothed_{self.config.smoothing_factor}/",
            'study_name': study_name
        }
        
        # Create directories
        base_dirs = [paths['raw_dir']]
        for dir_path in base_dirs + [paths['figures_dir'], paths['smoothed_figures_dir']]:
            os.makedirs(dir_path, exist_ok=True)
        
        return paths
    
    def parse_training_folder(self, folder_path: str, num_agents: int = 1,
                              keep_env_rows: bool = False) -> Optional[pd.DataFrame]:
        """
        Parse a single training folder and extract data.
        
        Args:
            folder_path: Path to training folder
            num_agents: Number of agents (1 for classic, 2 for competition)
            keep_env_rows: If True, keep one row per environment per episode instead of
                           averaging across environments (default: False)
            
        Returns:
            DataFrame with processed data or None if parsing fails
        """
        config_path = os.path.join(folder_path, "config.txt")
        csv_path = os.path.join(folder_path, "training_stats.csv")
        
        if not (os.path.exists(config_path) and os.path.exists(csv_path)):
            print(f"Missing config or CSV in {folder_path}")
            return None
        
        # Parse config file
        with open(config_path, "r") as f:
            config_contents = f.read()
        
        matches = self.reward_pattern.findall(config_contents)
        if len(matches) != num_agents:
            print(f"Expected {num_agents} agents, found {len(matches)} in {folder_path}")
            return None
        
        # Extract learning rate
        lr_match = re.search(r"LR:\s*([0-9.eE+-]+)", config_contents)
        if not lr_match:
            print(f"Learning rate not found in {folder_path}")
            return None
        lr = float(lr_match.group(1))
        
        # Extract seed if present
        seed_match = re.search(r"INITIAL_SEED:\s*([0-9]+)", config_contents)
        seed = int(seed_match.group(1)) if seed_match else None
        
        # Extract game_type (classic or classic_collision)
        game_type_match = re.search(r"GAME_VERSION:\s*(\S+)", config_contents)
        game_type = game_type_match.group(1) if game_type_match else 'classic'
        
        # Extract walking and cutting speeds for each agent
        walking_speeds = {}
        cutting_speeds = {}
        
        # Parse WALKING_SPEEDS dictionary format
        walking_speeds_match = re.search(r"WALKING_SPEEDS:\s*({[^}]+})", config_contents)
        if walking_speeds_match:
            try:
                speeds_dict_str = walking_speeds_match.group(1)
                for i in range(1, num_agents + 1):
                    agent_key = f"'ai_rl_{i}'"
                    speed_match = re.search(rf"{agent_key}:\s*([0-9.eE+-]+)", speeds_dict_str)
                    if speed_match:
                        walking_speeds[i] = float(speed_match.group(1))
            except Exception:
                pass
            
        # Parse CUTTING_SPEEDS dictionary format  
        cutting_speeds_match = re.search(r"CUTTING_SPEEDS:\s*({[^}]+})", config_contents)
        if cutting_speeds_match:
            try:
                speeds_dict_str = cutting_speeds_match.group(1)
                for i in range(1, num_agents + 1):
                    agent_key = f"'ai_rl_{i}'"
                    speed_match = re.search(rf"{agent_key}:\s*([0-9.eE+-]+)", speeds_dict_str)
                    if speed_match:
                        cutting_speeds[i] = float(speed_match.group(1))
            except Exception:
                pass
        
        # Load and clean CSV data
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        # Check if NUM_ENVS is > 1 in config to determine if we need to aggregate
        num_envs_match = re.search(r"NUM_ENVS:\s*([0-9]+)", config_contents)
        num_envs = int(num_envs_match.group(1)) if num_envs_match else 1
        
        if num_envs > 1:
            # Strip duplicate headers, then either average across envs (default) or keep each
            # environment row as a separate data point (keep_env_rows=True).
            data_lines = [lines[0].strip()]  # keep the single header
            for line in lines[1:]:
                stripped = line.strip()
                if stripped and not stripped.startswith('episode,'):
                    data_lines.append(stripped)
            df_raw = pd.read_csv(StringIO('\n'.join(data_lines)))
            df_raw['episode'] = pd.to_numeric(df_raw['episode'], errors='coerce')
            df_raw = df_raw.dropna(subset=['episode'])
            df_raw['episode'] = df_raw['episode'].astype(int)
            if keep_env_rows:
                df = df_raw  # Keep all individual environment rows
            else:
                # Fast path: average across environments per episode (original behaviour)
                numeric_cols = [c for c in df_raw.select_dtypes(include=[np.number]).columns if c != 'episode']
                df = df_raw.groupby('episode')[numeric_cols].mean().reset_index()
        else:
            # Single environment - use original logic but clean header duplication
            filtered_lines = [lines[0]] + [line for line in lines[1:] if not line.startswith("episode,")]
            df = pd.read_csv(StringIO("".join(filtered_lines)))
        # Check if 'episode' column exists and use it, otherwise use line numbers
        if 'episode' in df.columns:
            # Ensure episode column is numeric - convert from string if necessary
            df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
            # Drop rows where episode conversion failed (NaN values)
            df = df.dropna(subset=['episode'])
            # Convert to integer
            df['episode'] = df['episode'].astype(int)
        else:
            print(f"Warning: 'episode' column not found in {folder_path}. Using line numbers as episode values.")
            # If no episode column exists, create one with line numbers
            if df.shape[1] > 0:
                df.iloc[:, 0] = range(1, len(df) + 1)
                # Rename the first column to 'episode' if it's not already named
                if df.columns[0] != 'episode':
                    df.rename(columns={df.columns[0]: 'episode'}, inplace=True)
            else:
                # If dataframe is empty or has no columns, create an episode column
                df['episode'] = range(1, len(df) + 1)
        
        # Extract folder timestamp
        folder_name = os.path.basename(folder_path)
        date_time_str = folder_name.replace("Training_", "")
        
        # Add metadata columns
        df.insert(0, "timestamp", date_time_str)
        
        # Add agent parameters
        for i, (name, alpha, beta) in enumerate(matches, 1):
            df.insert(0 + i, f"alpha_{i}", float(alpha))
            df.insert(1 + i, f"beta_{i}", float(beta))
        
        df.insert(len(matches) * 2 + 1, "lr", lr)
        
        # Add seed if found in config
        if seed is not None:
            df.insert(len(matches) * 2 + 2, "seed", seed)
        
        # Add game_type
        insert_pos = len(matches) * 2 + 2 + (1 if seed is not None else 0)
        df.insert(insert_pos, "game_type", game_type)
        
        # Add walking and cutting speeds for each agent
        current_pos = insert_pos + 1  # Start after game_type column
        for i in range(1, num_agents + 1):
            if i in walking_speeds:
                df.insert(current_pos, f"walking_speed_{i}", walking_speeds[i])
                current_pos += 1
            if i in cutting_speeds:
                df.insert(current_pos, f"cutting_speed_{i}", cutting_speeds[i])
                current_pos += 1
        
        return df
        
    def _average_across_seeds(self, df: pd.DataFrame, num_agents: int) -> pd.DataFrame:
        """
        Average data across different seeds for the same experimental conditions.
        
        Args:
            df: DataFrame containing data from multiple seeds
            num_agents: Number of agents
            
        Returns:
            DataFrame with data averaged across seeds
        """
        # Define grouping columns (experimental conditions that should be the same across seeds)
        group_cols = ['episode']
        
        # Add agent parameters to grouping
        for i in range(1, num_agents + 1):
            if f'alpha_{i}' in df.columns:
                group_cols.append(f'alpha_{i}')
            if f'beta_{i}' in df.columns:
                group_cols.append(f'beta_{i}')
            if f'walking_speed_{i}' in df.columns:
                group_cols.append(f'walking_speed_{i}')
            if f'cutting_speed_{i}' in df.columns:
                group_cols.append(f'cutting_speed_{i}')
        
        # Add learning rate if present
        if 'lr' in df.columns:
            group_cols.append('lr')
        
        # Add game_type if present
        if 'game_type' in df.columns:
            group_cols.append('game_type')
        
        # Get columns to average (numeric columns except grouping columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_average = [col for col in numeric_cols if col not in group_cols and col != 'seed']
        
        # Check how many seeds we have
        unique_seeds = df['seed'].nunique()
        
        if unique_seeds == 1:
            return df
        
        # Perform grouping and averaging
        grouped = df.groupby(group_cols)
        
        # Calculate means for numeric columns
        averaged_data = grouped[cols_to_average].mean().reset_index()
        
        # Add back non-numeric columns (take first value from each group)
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols and col not in group_cols and col != 'seed']
        if non_numeric_cols:
            first_values = grouped[non_numeric_cols].first().reset_index()
            averaged_data = averaged_data.merge(first_values, on=group_cols, how='left')
        
        # Add seed information
        seed_counts = grouped['seed'].agg(['count', 'nunique']).reset_index()
        seed_counts.rename(columns={'count': 'total_episodes', 'nunique': 'seeds_averaged'}, inplace=True)
        averaged_data = averaged_data.merge(seed_counts[group_cols + ['seeds_averaged']], on=group_cols, how='left')
        
        # Create a representative timestamp (use first timestamp but mark as averaged)
        if 'timestamp' in df.columns:
            first_timestamp = df['timestamp'].iloc[0]
            averaged_data['timestamp'] = f"{first_timestamp}_averaged_{unique_seeds}_seeds"
        
        print(f"Averaging complete: {len(df)} rows -> {len(averaged_data)} rows ({unique_seeds} seeds)")
        
        return averaged_data
    
    def load_experiment_data(self, paths: Dict[str, str], 
                           num_agents: int = 1,
                           keep_env_rows: bool = False) -> pd.DataFrame:
        """
        Load all experiment data from the raw directory.
        Supports both direct training folders and study_name folder structure.
        When study_name is used, averages across different seeds for the same experiment.
        
        Args:
            paths: Dictionary containing directory paths
            num_agents: Number of agents in the experiment
            keep_env_rows: If True, keep one row per environment per episode instead of
                           averaging across environments (default: False)
            
        Returns:
            Combined DataFrame with all experiment data
        """
        raw_dir = paths['raw_dir']
        study_name = paths.get('study_name')

        # Return cached result if this directory has already been loaded
        cache_key = (raw_dir, num_agents, study_name, keep_env_rows)
        if cache_key in self._dir_cache:
            return self._dir_cache[cache_key]
        
        if not os.path.exists(raw_dir):
            # If study_name is specified but directory doesn't exist, try to find available study names
            if study_name:
                parent_dir = os.path.dirname(raw_dir)
                if os.path.exists(parent_dir):
                    available_studies = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
                    print(f"Directory {raw_dir} does not exist.")
                    print(f"Available study names in {parent_dir}: {available_studies}")
                    if available_studies:
                        print(f"Try using one of these study names: {', '.join(available_studies)}")
            raise ValueError(f"Data directory does not exist: {raw_dir}")
            
        all_dfs = []
        folders_found = []
        folders_processed = []
        
        # Check if we have study_name folders or direct training folders
        items_in_dir = os.listdir(raw_dir)
        training_folders = [item for item in items_in_dir if item.startswith('Training_') and os.path.isdir(os.path.join(raw_dir, item))]
        
        if study_name is None and training_folders:
            # Direct training folders mode (legacy)
            for folder in training_folders:
                folder_path = os.path.join(raw_dir, folder)
                folders_found.append(folder)
                df = self.parse_training_folder(folder_path, num_agents, keep_env_rows)
                if df is not None:
                    all_dfs.append(df)
                    folders_processed.append(folder)
        
        elif study_name is None:
            # Auto-detect study names and ask user to specify
            study_dirs = [item for item in items_in_dir if os.path.isdir(os.path.join(raw_dir, item)) and not item.startswith('training_')]
            if study_dirs:
                print(f"Found multiple study directories: {study_dirs}")
                print("Please specify a study_name parameter. Example usage:")
                print(f"  python analysis_pretrained.py {os.path.basename(raw_dir).replace('map_', '')} --study_name {study_dirs[0]}")
                raise ValueError(f"Multiple study directories found. Please specify --study_name parameter from: {study_dirs}")
            else:
                # No training folders and no study folders - try direct processing
                print("No training folders or study folders found, trying to process directory directly")
                for folder in items_in_dir:
                    folder_path = os.path.join(raw_dir, folder)
                    if not os.path.isdir(folder_path):
                        continue
                    folders_found.append(folder)
                    df = self.parse_training_folder(folder_path, num_agents, keep_env_rows)
                    if df is not None:
                        all_dfs.append(df)
                        folders_processed.append(folder)
        
        else:
            # Study name mode - process all training folders and average seeds
            # Find all training folders
            for folder in items_in_dir:
                folder_path = os.path.join(raw_dir, folder)
                if not os.path.isdir(folder_path) or not folder.startswith('Training_'):
                    continue
                    
                folders_found.append(folder)
                df = self.parse_training_folder(folder_path, num_agents, keep_env_rows)
                if df is not None:
                    # Add study_name information
                    df['study_name'] = study_name
                    
                    # Seed should already be extracted from config.txt during parse_training_folder
                    # If not present, use timestamp as a unique identifier
                    if 'seed' not in df.columns:
                        df['seed'] = df['timestamp'].iloc[0] if len(df) > 0 else 'unknown'
                    
                    all_dfs.append(df)
                    folders_processed.append(folder)
        
        if not all_dfs:
            error_msg = f"No valid training data found in {raw_dir}. "
            if study_name:
                error_msg += f"Check that Training_* folders exist in the study directory '{study_name}' and contain config.txt and training_stats.csv files."
            else:
                error_msg += "Check that config.txt and training_stats.csv files exist in training folders."
            raise ValueError(error_msg)
        
        # Combine all dataframes
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # If we have seeds, perform averaging across seeds for same experimental conditions
        if 'seed' in final_df.columns and study_name:
            final_df = self._average_across_seeds(final_df, num_agents)
        
        # Cache the result so repeated calls for the same directory are instant
        self._dir_cache[cache_key] = final_df
        return final_df
    
    def prepare_dataframe(self, df: pd.DataFrame, num_agents: int = 1) -> pd.DataFrame:
        """
        Prepare dataframe with computed columns and proper types.
        
        Args:
            df: Raw dataframe
            num_agents: Number of agents
            
        Returns:
            Processed dataframe
        """
        # Set data types
        dtype_dict = {
            "timestamp": str,
        }
        
        for i in range(1, num_agents + 1):
            dtype_dict[f"alpha_{i}"] = float
            dtype_dict[f"beta_{i}"] = float
        
        # Convert numeric columns
        for col in df.columns[6:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by alpha values
        alpha_cols = [f"alpha_{i}" for i in range(1, num_agents + 1)]
        df = df.sort_values(by=alpha_cols, ascending=[False] * len(alpha_cols))
        
        # Create attitude key
        if num_agents == 1:
            df["attitude_key"] = df.apply(lambda row: f"{row['alpha_1']}_{row['beta_1']}", axis=1)
            
            # Create speed key
            if "walking_speed_1" in df.columns and "cutting_speed_1" in df.columns:
                df["speed_key"] = df.apply(lambda row: f"{row['walking_speed_1']}_{row['cutting_speed_1']}", axis=1)
            
            # Create total columns for single agent (check if source columns exist)
            if "pure_reward_ai_rl_1" in df.columns:
                df["pure_reward_total"] = df["pure_reward_ai_rl_1"]
                
            if "deliver_ai_rl_1" in df.columns:
                df["total_deliveries"] = df["deliver_ai_rl_1"]
        else:
            df["attitude_key"] = df.apply(
                lambda row: f"{row['alpha_1']}_{row['beta_1']}_{row['alpha_2']}_{row['beta_2']}", 
                axis=1
            )
            
            # Create speed key for both agents (more robust handling)
            speed_cols_1 = ["walking_speed_1", "cutting_speed_1"]
            speed_cols_2 = ["walking_speed_2", "cutting_speed_2"]
            
            has_speed_1 = all(col in df.columns for col in speed_cols_1)
            has_speed_2 = all(col in df.columns for col in speed_cols_2)
            
            if has_speed_1 and has_speed_2:
                df["speed_key"] = df.apply(
                    lambda row: f"{row['walking_speed_1']}_{row['cutting_speed_1']}_{row['walking_speed_2']}_{row['cutting_speed_2']}", 
                    axis=1
                )
            elif has_speed_1:
                df["speed_key"] = df.apply(
                    lambda row: f"{row['walking_speed_1']}_{row['cutting_speed_1']}", 
                    axis=1
                )
            elif has_speed_2:
                df["speed_key"] = df.apply(
                    lambda row: f"{row['walking_speed_2']}_{row['cutting_speed_2']}", 
                    axis=1
                )
            
            # Create total columns for two agents
            if "pure_reward_ai_rl_1" in df.columns and "pure_reward_ai_rl_2" in df.columns:
                df["pure_reward_total"] = df["pure_reward_ai_rl_1"] + df["pure_reward_ai_rl_2"]
            if "deliver_ai_rl_1" in df.columns and "deliver_ai_rl_2" in df.columns:
                df["total_deliveries"] = df["deliver_ai_rl_1"] + df["deliver_ai_rl_2"]
        
        return df


class MetricDefinitions:
    """Defines metrics and their visual properties for different experiment types."""
    
    @staticmethod
    def get_base_classic_metrics() -> Dict:
        """Get base metric definitions for classic experiments (without agent suffix)."""
        return {
            'rewarded_metrics': [
                "deliver",
                "cut",
                "salad",
                "plate",
                "raw_food",
                "counter"
            ],
            'movement_metrics': [
                "do_nothing",
                "useless_floor",
                "useless_wall",
                "useless_counter",
                "useful_counter",
                "salad_assembly",
                "destructive_food_dispenser",
                "useful_food_dispenser",
                "useless_cutting_board",
                "useful_cutting_board",
                "destructive_plate_dispenser",
                "useful_plate_dispenser",
                "useless_delivery",
                "useful_delivery",
                "inaccessible_tile"
            ]
        }
    
    @staticmethod
    def get_classic_metrics() -> Dict:
        """Get metric definitions for classic experiments (for backward compatibility)."""
        base_metrics = MetricDefinitions.get_base_classic_metrics()
        return {
            'rewarded_metrics_1': [f"{metric}_ai_rl_1" for metric in base_metrics['rewarded_metrics']],
            'movement_metrics_1': [f"{metric}_ai_rl_1" for metric in base_metrics['movement_metrics']]
        }
    
    @staticmethod
    def get_agent_metrics(base_metrics: List[str], agent_num: int) -> List[str]:
        """Generate agent-specific metrics from base metric names."""
        return [f"{metric}_ai_rl_{agent_num}" for metric in base_metrics]
    
    @staticmethod
    def get_base_competition_metrics() -> Dict:
        """Get base metric definitions for competition experiments (without agent suffix)."""
        return {
            'result_events': [
                "deliver_own",
                "deliver_other", 
                "salad_own",
                "salad_other",
                "cut_own",
                "cut_other",
            ],
            'action_types': [
                "do_nothing",
                "floor_actions",
                "wall_actions",
                "useless_counter_actions",
                "useful_counter_actions",
                "useless_own_food_dispenser_actions",
                "useful_own_food_dispenser_actions",
                "useless_other_food_dispenser_actions",
                "useful_other_food_dispenser_actions",
                "useless_cutting_board_actions",
                "useful_own_cutting_board_actions",
                "useful_other_cutting_board_actions",
                "useless_plate_dispenser_actions",
                "useful_plate_dispenser_actions",
                "useless_delivery_actions",
                "useful_own_delivery_actions",
                "useful_other_delivery_actions",
            ]
        }
    
    @staticmethod
    def get_competition_metrics() -> Dict:
        """Get metric definitions for competition experiments (for backward compatibility)."""
        base_metrics = MetricDefinitions.get_base_competition_metrics()
        return {
            'result_events_1': [f"{metric}_ai_rl_1" for metric in base_metrics['result_events']],
            'result_events_2': [f"{metric}_ai_rl_2" for metric in base_metrics['result_events']],
            'action_types_1': [f"{metric}_ai_rl_1" for metric in base_metrics['action_types']],
            'action_types_2': [f"{metric}_ai_rl_2" for metric in base_metrics['action_types']]
        }
    
    @staticmethod
    def get_metric_labels() -> Dict:
        """Get human-readable labels for metrics."""
        return {
            # Classic metrics
            "deliver": "Delivered",
            "cut": "Cut", 
            "salad": "Salad",
            "plate": "Plate",
            "raw_food": "Raw Food",
            "counter": "Counter",
            "do_nothing": "No action",
            "useless_floor": "Useless Floor",
            "useless_wall": "Useless Wall",
            "useless_counter": "Useless Counter",
            "useful_counter": "Useful Counter",
            "salad_assembly": "Salad Assembly",
            "destructive_food_dispenser": "Destructive Food Dispenser",
            "useful_food_dispenser": "Useful Food Dispenser",
            "useless_cutting_board": "Useless Cutting Board",
            "useful_cutting_board": "Useful Cutting Board",
            "destructive_plate_dispenser": "Destructive Plate Dispenser",
            "useful_plate_dispenser": "Useful Plate Dispenser",
            "useless_delivery": "Useless Delivery",
            "useful_delivery": "Useful Delivery",
            "inaccessible_tile": "Inaccessible Tile",
            
            # Legacy classic metrics for compatibility
            "delivered": "Delivered",
            "floor_actions": "Floor",
            "wall_actions": "Wall",
            "useless_counter_actions": "Useless Counter",
            "useful_counter_actions": "Useful Counter",
            "useless_food_dispenser_actions": "Useless Food Dispenser",
            "useful_food_dispenser_actions": "Useful Food Dispenser",
            "useless_cutting_board_actions": "Useless Cutting Board",
            "useful_cutting_board_actions": "Useful Cutting Board",
            "useless_plate_dispenser_actions": "Useless Plate Dispenser",
            "useful_plate_dispenser_actions": "Useful Plate Dispenser",
            "useless_delivery_actions": "Useless Delivery",
            "useful_delivery_actions": "Useful Delivery",
            
            # Competition metrics
            "deliver_own": "Delivered Own",
            "deliver_other": "Delivered Other",
            "delivered_own": "Delivered Own",
            "delivered_other": "Delivered Other",
            "salad_own": "Salad Own",
            "salad_other": "Salad Other",
            "cut_own": "Cut Own",
            "cut_other": "Cut Other",
            "useless_own_food_dispenser_actions": "Useless Own Food Dispenser",
            "useful_own_food_dispenser_actions": "Useful Own Food Dispenser",
            "useless_other_food_dispenser_actions": "Useless Other Food Dispenser",
            "useful_other_food_dispenser_actions": "Useful Other Food Dispenser",
            "useful_own_cutting_board_actions": "Useful Own Cutting Board",
            "useful_other_cutting_board_actions": "Useful Other Cutting Board",
            "useful_own_delivery_actions": "Useful Own Delivery",
            "useful_other_delivery_actions": "Useful Other Delivery",
        }
    
    @staticmethod
    def get_metric_colors() -> Dict:
        """Get color scheme for metrics."""
        return {
            # Classic colors
            "deliver": "#27AE60",
            "cut": "#2980B9",
            "salad": "#E67E22",
            "plate": "#F39C12",
            "raw_food": "#8E44AD",
            "counter": "#34495E",
            "do_nothing": "#000000",
            "useless_floor": "#9B59B6",
            "useless_wall": "#59351F",
            "useless_counter": "#D5D8DC",
            "useful_counter": "#7B7D7D",
            "salad_assembly": "#A569BD",
            "destructive_food_dispenser": "#E74C3C",
            "useful_food_dispenser": "#C0392B",
            "useless_cutting_board": "#AED6F1",
            "useful_cutting_board": "#2980B9",
            "destructive_plate_dispenser": "#FF6B35",
            "useful_plate_dispenser": "#E67E22",
            "useless_delivery": "#A9DFBF",
            "useful_delivery": "#27AE60",
            "inaccessible_tile": "#95A5A6",
            
            # Legacy classic colors for compatibility
            "delivered": "#27AE60",
            "floor_actions": "#9B59B6",
            "wall_actions": "#59351F",
            "useless_counter_actions": "#D5D8DC",
            "useful_counter_actions": "#7B7D7D",
            "useless_food_dispenser_actions": "#F5B7B1",
            "useful_food_dispenser_actions": "#C0392B",
            "useless_cutting_board_actions": "#AED6F1",
            "useful_cutting_board_actions": "#2980B9",
            "useless_plate_dispenser_actions": "#FAD7A0",
            "useful_plate_dispenser_actions": "#E67E22",
            "useless_delivery_actions": "#A9DFBF",
            "useful_delivery_actions": "#27AE60",
            
            # Competition colors
            "deliver_own": "#A9DFBF",
            "deliver_other": "#27AE60",
            "delivered_own": "#A9DFBF",
            "delivered_other": "#27AE60",
            "salad_own": "#F5CBA7",
            "salad_other": "#E67E22",
            "cut_own": "#AED6F1",
            "cut_other": "#2980B9",
            "useless_own_food_dispenser_actions": "#F5B7B1",
            "useful_own_food_dispenser_actions": "#C0392B",
            "useless_other_food_dispenser_actions": "#D7BDE2",
            "useful_other_food_dispenser_actions": "#8E44AD",
            "useful_own_cutting_board_actions": "#2980B9",
            "useful_other_cutting_board_actions": "#1ABC9C",
            "useful_own_delivery_actions": "#27AE60",
            "useful_other_delivery_actions": "#145A32",
        }


class PlotGenerator:
    """Generates various types of plots for analysis."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.metric_labels = MetricDefinitions.get_metric_labels()
        self.metric_colors = MetricDefinitions.get_metric_colors()
    
    def _sanitize_filename(self, attitude: str) -> str:
        """Sanitize attitude string for filename use."""
        return attitude.replace('.', 'p')
    
    def _get_metric_info(self, metric: str) -> Tuple[str, str]:
        """Get label and color for a metric."""
        # Extract base metric name
        base_metric = '_'.join(metric.split('_')[:-3])  # Remove _ai_rl_X suffix
        
        label = self.metric_labels.get(base_metric, base_metric.replace('_', ' ').title())
        color = self.metric_colors.get(base_metric, "#000000")
        
        return label, color
    
    def plot_basic_metrics(self, df: pd.DataFrame, figures_dir: str,
                          metric_col: str, metric_name: str, by_attitude: bool = True,
                          xlim=None, ylim=None):
        """
        Generate basic metric plots (score, reward, etc.).
        
        Args:
            df: Data DataFrame
            figures_dir: Directory to save figures
            metric_col: Column name for the metric
            metric_name: Human-readable metric name
            by_attitude: Whether to plot by attitude or overall
        """
        unique_attitudes = df["attitude_key"].unique()
        unique_lr = df["lr"].unique()
        
        if by_attitude:
            for attitude in unique_attitudes:
                subset = df[df["attitude_key"] == attitude]
                plt.figure(figsize=(10, 6))
                for lr in unique_lr:
                    lr_filtered = subset[subset["lr"] == lr]
                    # Data is already averaged per episode during loading
                    label = f"LR {lr}"
                    plt.plot(lr_filtered["episode"], lr_filtered[metric_col], label=label)
                if xlim:
                    plt.xlim(xlim)
                if ylim:
                    plt.ylim(ylim)
                plt.title(f"{metric_name} vs Episode\nAttitude {attitude}")
                plt.xlabel("Episode")
                plt.ylabel(metric_name)
                plt.legend()
                plt.tight_layout()
                sanitized_attitude = self._sanitize_filename(attitude)
                filename = f"{metric_name.lower().replace(' ', '_')}_attitude_{sanitized_attitude}.png"
                plt.savefig(os.path.join(figures_dir, filename))
                plt.close()
        else:
            plt.figure(figsize=(10, 6))
            for lr in unique_lr:
                lr_filtered = df[df["lr"] == lr]
                # Data is already averaged per episode during loading
                label = f"LR {lr}"
                plt.plot(lr_filtered["episode"], lr_filtered[metric_col], label=label)
            if xlim:
                plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim)
            plt.xlabel("Episodes", fontsize=20)
            plt.ylabel(f"Mean {metric_name.lower()}", fontsize=20)
            plt.legend(fontsize=20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            filename = f"{metric_name.lower().replace(' ', '_')}.png"
            plt.savefig(os.path.join(figures_dir, filename))
            plt.close()
    
    def plot_smoothed_metrics(self, df: pd.DataFrame, figures_dir: str,
                             metric_col: str, metric_name: str, by_attitude: bool = True,
                             xlim=None, ylim=None):
        """Generate smoothed versions of metric plots."""
        N = self.config.smoothing_factor
        unique_attitudes = df["attitude_key"].unique()
        unique_lr = df["lr"].unique()
        
        if by_attitude:
            for attitude in unique_attitudes:
                subset = df[df["attitude_key"] == attitude]
                plt.figure(figsize=(10, 6))
                for lr in unique_lr:
                    lr_filtered = subset[subset["lr"] == lr]
                    lr_filtered = lr_filtered.copy()
                    lr_filtered["episode_block"] = (lr_filtered["episode"] // N)
                    block_means = lr_filtered.groupby("episode_block")[metric_col].mean()
                    middle_episodes = lr_filtered.groupby("episode_block")["episode"].median()
                    label = f"LR {lr}"
                    plt.plot(middle_episodes, block_means, label=label)
                if xlim:
                    plt.xlim(xlim)
                if ylim:
                    plt.ylim(ylim)
                plt.title(f"{metric_name} vs Episode\nAttitude {attitude}")
                plt.xlabel("Episode")
                plt.ylabel(metric_name)
                plt.legend()
                plt.tight_layout()
                sanitized_attitude = self._sanitize_filename(attitude)
                filename = f"{metric_name.lower().replace(' ', '_')}_attitude_{sanitized_attitude}_smoothed_{N}.png"
                plt.savefig(os.path.join(figures_dir, filename))
                plt.close()
        else:
            plt.figure(figsize=(10, 6))
            for lr in unique_lr:
                lr_filtered = df[df["lr"] == lr]
                lr_filtered = lr_filtered.copy()
                lr_filtered["episode_block"] = (lr_filtered["episode"] // N)
                block_means = lr_filtered.groupby("episode_block")[metric_col].mean()
                middle_episodes = lr_filtered.groupby("episode_block")["episode"].median()
                label = f"LR {lr}"
                plt.plot(middle_episodes, block_means, label=label)
            if xlim:
                plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim)
            plt.xlabel("Episodes", fontsize=20)
            plt.ylabel(f"Mean {metric_name.lower()}", fontsize=20)
            plt.legend(fontsize=20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            filename = f"{metric_name.lower().replace(' ', '_')}_smoothed_{N}.png"
            plt.savefig(os.path.join(figures_dir, filename))
            plt.close()
    
    def plot_agent_metrics(self, df: pd.DataFrame, figures_dir: str, metrics: List[str],
                          agent_num: int, smoothed: bool = False):
        """Plot individual agent metrics."""
        N = self.config.smoothing_factor if smoothed else 1
        unique_lr = df["lr"].unique()
        
        for lr in unique_lr:
            filtered_subset = df[df["lr"] == lr]
            
            if smoothed:
                filtered_subset = filtered_subset.copy()
                filtered_subset["episode_block"] = (filtered_subset["episode"] // N)
            
            plt.figure(figsize=(12, 6))
            
            for metric in metrics:
                label, color = self._get_metric_info(metric)
                
                if smoothed:
                    block_means = filtered_subset.groupby("episode_block")[metric].mean()
                    middle_episodes = filtered_subset.groupby("episode_block")["episode"].median()
                    plt.plot(middle_episodes, block_means, label=label, color=color)
                else:
                    # Data is already averaged per episode during loading
                    plt.plot(filtered_subset["episode"], filtered_subset[metric], label=label, color=color)
            
            title = f"Metrics per Episode - LR {lr}"
            if smoothed:
                title += f" (Smoothed {N})"
            
            plt.title(title)
            plt.xlabel("Episode")
            plt.ylabel("Mean value")
            plt.legend()
            plt.tight_layout()
            
            suffix = f"_smoothed_{N}" if smoothed else ""
            filename = f"metrics_agent{agent_num}_lr{str(lr).replace('.', 'p')}{suffix}.png"
            plt.savefig(os.path.join(figures_dir, filename))
            plt.close()


def setup_argument_parser(experiment_type: str) -> argparse.ArgumentParser:
    """
    Set up command line argument parser.
    
    Args:
        experiment_type: Type of experiment ('classic', 'competition', 'pretrained')
        
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description=f'Analysis script for {experiment_type} experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'map_name',
        type=str,
        help='Map name identifier (e.g., simple_kitchen_circular)'
    )
    
    parser.add_argument(
        '--cluster',
        type=str,
        choices=['brigit', 'cuenca', 'local'],
        default='cuenca',
        help='Cluster type for path configuration'
    )
    
    parser.add_argument(
        '--smoothing_factor',
        type=int,
        default=15,
        help='Smoothing factor for curve generation'
    )
    
    parser.add_argument(
        '--output_format',
        type=str,
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Output format for figures'
    )
    
    parser.add_argument(
        '--individual_trainings',
        type=str,
        choices=['yes', 'no', 'Yes', 'No', 'YES', 'NO'],
        default='no',
        help='Generate individual figures for each training ID (yes/no)'
    )

    parser.add_argument(
        '--study_name',
        type=str,
        default=None,
        help='Study name for specific study folders (optional)'
    )
    
    parser.add_argument(
        '--game_type',
        type=str,
        choices=['classic', 'classic_collision'],
        default='classic',
        help='Game type (classic or classic_collision)'
    )
    
    parser.add_argument(
        '--init_type',
        type=str,
        choices=['random_init', 'empty_init'],
        default="",
        help='Initialization type subdirectory to analyze (default: process all available init types)'
    )
    
    parser.add_argument(
        '--synergy_scaling_factor',
        type=float,
        default=0.0,
        help='Synergy scaling factor for reference-based reward shaping (default: 0.0)'
    )
    
    return parser


def main_analysis_pipeline(experiment_type: str, map_name: str,
                          cluster: str = 'cuenca', smoothing_factor: int = 15, 
                          num_agents: Optional[int] = None, study_name: Optional[str] = None,
                          game_type: str = 'classic', init_type: str = 'random_init',
                          synergy_scaling_factor: float = 0.0, synergy_provided: bool = False,
                          specialization_lambda: Optional[float] = None) -> Union[Dict, Dict[str, Dict]]:
    """
    Main analysis pipeline that can be used by all experiment types.
    
    Args:
        experiment_type: Type of experiment (e.g., 'classic', 'competition', 'pretrained')
        map_name: Map name
        cluster: Cluster type
        smoothing_factor: Smoothing factor for plots
        num_agents: Number of agents (default: 2 for classic/competition, 1 for pretrained)
        study_name: Study name for specific study folders (optional)
        game_type: Game type (classic or classic_collision) - used to construct path
        init_type: Initialization type (random_init or empty_init) - used to construct path
        synergy_scaling_factor: Team synergy scaling factor (default: 0.0)
        synergy_provided: Whether synergy was explicitly provided (to determine folder structure)
        specialization_lambda: Specialization penalty scale (None = auto-detect all available lambdas)
        
    Returns:
        If analyzing single lambda: Dict containing processed data and paths
        If analyzing multiple lambdas: Dict[str, Dict] with keys 'specialized_{lambda}'
    """
    # Initialize components
    config = AnalysisConfig()
    config.smoothing_factor = smoothing_factor
    
    processor = DataProcessor(config)
    plotter = PlotGenerator(config)
    
    # Custom path setup that includes init_type and synergy in the correct order
    local_path = config.cluster_paths[cluster]
    
    # Determine which synergy values to analyze
    if not synergy_provided:
        # Auto-detect available synergy folders when synergy was not explicitly provided
        synergy_values_to_analyze = _detect_available_synergy_folders(
            local_path, experiment_type, game_type, init_type, 
            map_name, study_name
        )
        analyze_multiple_synergies = len([s for s in synergy_values_to_analyze if s is not None]) > 1
        print(f"Auto-detected synergy values: {synergy_values_to_analyze}")
    else:
        # Use provided synergy value
        synergy_values_to_analyze = [synergy_scaling_factor]
        analyze_multiple_synergies = False
        print(f"Using provided synergy value: {synergy_scaling_factor}")
    
    # Determine which specialization lambdas to analyze
    if specialization_lambda is None:
        # Will auto-detect for each synergy value
        spec_folders_to_analyze = None  # Detect per synergy
    else:
        # Use specified lambda - format with 2 decimal places to match folder names
        if specialization_lambda == 0:
            spec_folders_to_analyze = ["specialized_0"]
        else:
            spec_folders_to_analyze = [f"specialized_{specialization_lambda:.2f}"]
    
    # If analyzing multiple synergies and/or specializations, process each combination
    all_results = {}
    
    for synergy_val in synergy_values_to_analyze:
        # Determine if we're using synergy subfolders for this value
        if synergy_provided:
            # User explicitly provided synergy value
            current_synergy_provided = True
        elif not synergy_provided and synergy_val is not None:
            # Auto-detected synergy folders exist, so treat as provided for path construction
            current_synergy_provided = True
        else:
            # No synergy subfolders
            current_synergy_provided = False
        
        # Auto-detect specialization folders for this synergy value if needed
        if spec_folders_to_analyze is None:
            current_spec_folders = _detect_available_specialization_folders(
                local_path, experiment_type, game_type, init_type, 
                map_name, current_synergy_provided, synergy_val, study_name
            )
            if not current_spec_folders:
                print(f"No specialization folders found for synergy={synergy_val}")
                continue
            print(f"Detected specialization folders for synergy={synergy_val}: {current_spec_folders}")
        else:
            current_spec_folders = spec_folders_to_analyze
        
        analyze_multiple_specs = len(current_spec_folders) > 1
    
        # Determine number of agents based on experiment type (needed for data loading)
        if num_agents is None:
            # Pretrained experiments have 1 agent, others have 2
            num_agents = 1 if 'pretrain' in experiment_type.lower() else 2
        
        # Process each specialization mode for this synergy value
        for spec_folder in current_spec_folders:
            # Create a unique key for this combination
            if analyze_multiple_synergies and synergy_val is not None:
                if synergy_val == 0:
                    result_key = f"synergy_0_{spec_folder}"
                else:
                    result_key = f"synergy_{synergy_val:.2f}_{spec_folder}"
            elif analyze_multiple_specs:
                result_key = spec_folder
            else:
                result_key = 'single_result'
            
            print(f"\n{'='*60}")
            if synergy_val is not None:
                if synergy_val == 0:
                    print(f"PROCESSING SYNERGY=0, {spec_folder.upper()} DATA")
                else:
                    print(f"PROCESSING SYNERGY={synergy_val:.2f}, {spec_folder.upper()} DATA")
            else:
                print(f"PROCESSING {spec_folder.upper()} DATA")
            print(f"{'='*60}")
            
            # Build path for this synergy/specialization combination
            raw_dir = _build_experiment_path(
                local_path, experiment_type, game_type, init_type, 
                map_name, current_synergy_provided, synergy_val, spec_folder, study_name
            )
            
            # Check if path exists
            if not os.path.exists(raw_dir):
                print(f"  Warning: Path not found: {raw_dir}")
                print(f"  Skipping...")
                continue
            
            print(f"  Found data at: {raw_dir}")
            
            # Set up paths for this combination
            paths = {
                'raw_dir': raw_dir,
                'output_path': f"{raw_dir}/training_results.csv",
                'figures_dir': f"{raw_dir}/training_figures/",
                'smoothed_figures_dir': f"{raw_dir}/training_figures/smoothed_{config.smoothing_factor}/",
                'study_name': study_name,
                'init_type': init_type,
                'synergy': synergy_val,
                'specialization': spec_folder
            }
            
            # Create directories
            for dir_path in [paths['raw_dir'], paths['figures_dir'], paths['smoothed_figures_dir']]:
                os.makedirs(dir_path, exist_ok=True)
            
            # Load and process data
            raw_df = processor.load_experiment_data(paths, num_agents)
            df = processor.prepare_dataframe(raw_df, num_agents)
            
            print(f"  Loaded {len(df)} training records")
            print(f"  Unique attitudes: {len(df['attitude_key'].unique())}")
            print(f"  Figures will be saved to: {paths['figures_dir']}")
            
            all_results[result_key] = {
                'df': df,
                'paths': paths,
                'config': config,
                'processor': processor,
                'plotter': plotter,
                'num_agents': num_agents
            }
    
    if not all_results:
        raise ValueError("No data could be loaded from any synergy/specialization combination")
    
    # Return results based on how many combinations we found
    if len(all_results) == 1 and 'single_result' in all_results:
        # Single combination - return the dict directly
        return all_results['single_result']
    else:
        # Multiple combinations - return the full dictionary
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMBINATIONS FOUND: {sorted(all_results.keys())}")
        print(f"{'='*60}\n")
        return all_results

    # Note: Code below this point is no longer reachable but kept for reference
    # Single specialization mode - use original logic
    spec_folder = current_spec_folders[0]
    
    # Build path
    raw_dir = _build_experiment_path(
        local_path, experiment_type, game_type, init_type, 
        map_name, synergy_provided, synergy_scaling_factor, spec_folder, study_name
    )
    
    paths = {
        'raw_dir': raw_dir,
        'output_path': f"{raw_dir}/training_results.csv",
        'figures_dir': f"{raw_dir}/training_figures/",
        'smoothed_figures_dir': f"{raw_dir}/training_figures/smoothed_{config.smoothing_factor}/",
        'study_name': study_name,
        'init_type': init_type,
        'synergy': synergy_scaling_factor,
        'specialization': spec_folder
    }
    
    # Create directories
    for dir_path in [paths['raw_dir'], paths['figures_dir'], paths['smoothed_figures_dir']]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Load and process data
    raw_df = processor.load_experiment_data(paths, num_agents)
    df = processor.prepare_dataframe(raw_df, num_agents)
    
    print(f"Loaded {len(df)} training records")
    print(f"Unique attitudes: {len(df['attitude_key'].unique())}")
    print(f"Figures will be saved to: {paths['figures_dir']}")
    
    return {
        'df': df,
        'paths': paths,
        'config': config,
        'processor': processor,
        'plotter': plotter,
        'num_agents': num_agents
    }


def _build_experiment_path(local_path: str, experiment_type: str, game_type: str, 
                          init_type: str, map_name: str, synergy_provided: bool, 
                          synergy_scaling_factor: float, spec_folder: str, study_name: Optional[str]) -> str:
    """Build experiment path based on parameters."""
    
    # Only create synergy subfolder if synergy was explicitly provided (even if it's 0)
    if synergy_provided:
        # Format synergy folder name with 2 decimal places to match actual folder names
        if synergy_scaling_factor == 0:
            synergy_folder = "synergy_0"
        else:
            synergy_folder = f"synergy_{synergy_scaling_factor:.2f}"
        
        # Build the experiment path based on experiment type with synergy folder and specialization
        if 'pretrain' in experiment_type.lower():
            # For pretraining: /data/samuel_lozano/cooked/pretraining/{game_type}/{init_type}/map_{map_name}/{synergy_folder}/{spec_folder}/
            if study_name:
                if spec_folder:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{study_name}/pretraining/{game_type}/{init_type}/map_{map_name}/{synergy_folder}/{spec_folder}/"
                else:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{study_name}/pretraining/{game_type}/{init_type}/map_{map_name}/{synergy_folder}"
            else:
                if spec_folder:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/pretraining/{game_type}/{init_type}/map_{map_name}/{synergy_folder}/{spec_folder}"
                else:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/pretraining/{game_type}/{init_type}/map_{map_name}/{synergy_folder}"
        elif 'classic' in experiment_type.lower() or 'competition' in experiment_type.lower():
            # For classic/competition: /data/samuel_lozano/cooked/{game_type}/{init_type}/map_{map_name}/{synergy_folder}/{spec_folder}/
            if study_name:
                if spec_folder:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{study_name}/{game_type}/{init_type}/map_{map_name}/{synergy_folder}/{spec_folder}/"
                else:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{study_name}/{game_type}/{init_type}/map_{map_name}/{synergy_folder}/"
            else:
                if spec_folder:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{game_type}/{init_type}/map_{map_name}/{synergy_folder}/{spec_folder}"
                else:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{game_type}/{init_type}/map_{map_name}/{synergy_folder}"
        else:
            # For other experiment types: /data/samuel_lozano/cooked/{experiment_type}/{init_type}/map_{map_name}/{synergy_folder}/{spec_folder}/
            if study_name:
                if spec_folder:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{study_name}/{experiment_type}/{init_type}/map_{map_name}/{synergy_folder}/{spec_folder}/"
                else:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{study_name}/{experiment_type}/{init_type}/map_{map_name}/{synergy_folder}/"
            else:
                if spec_folder:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{experiment_type}/{init_type}/map_{map_name}/{synergy_folder}/{spec_folder}"
                else:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{experiment_type}/{init_type}/map_{map_name}/{synergy_folder}"
    else:
        # No synergy folder - use original structure when synergy is not provided (with specialization folder)
        if 'pretrain' in experiment_type.lower():
            # For pretraining: /data/samuel_lozano/cooked/pretraining/{game_type}/{init_type}/map_{map_name}/{spec_folder}/
            if study_name:
                if spec_folder:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{study_name}/pretraining/{game_type}/{init_type}/map_{map_name}/{spec_folder}/"
                else:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{study_name}/pretraining/{game_type}/{init_type}/map_{map_name}/"
            else:
                if spec_folder:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/pretraining/{game_type}/{init_type}/map_{map_name}/{spec_folder}"
                else:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/pretraining/{game_type}/{init_type}/map_{map_name}"
        elif 'classic' in experiment_type.lower() or 'competition' in experiment_type.lower():
            # For classic/competition: /data/samuel_lozano/cooked/{game_type}/{init_type}/map_{map_name}/{spec_folder}/
            if study_name:
                if spec_folder:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{study_name}/{game_type}/{init_type}/map_{map_name}/{spec_folder}/"
                else:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{study_name}/{game_type}/{init_type}/map_{map_name}/"
            else:
                if spec_folder:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{game_type}/{init_type}/map_{map_name}/{spec_folder}"
                else:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{game_type}/{init_type}/map_{map_name}"
        else:
            # For other experiment types: /data/samuel_lozano/cooked/{experiment_type}/{init_type}/map_{map_name}/{spec_folder}/
            if study_name:
                if spec_folder:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{study_name}/{experiment_type}/{init_type}/map_{map_name}/{spec_folder}/"
                else:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{study_name}/{experiment_type}/{init_type}/map_{map_name}/"
            else:
                if spec_folder:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{experiment_type}/{init_type}/map_{map_name}/{spec_folder}"
                else:
                    raw_dir = f"{local_path}/data/samuel_lozano/cooked/{experiment_type}/{init_type}/map_{map_name}"
    
    return raw_dir


def _detect_available_synergy_folders(local_path: str, experiment_type: str, 
                                     game_type: str, init_type: str, map_name: str, 
                                     study_name: Optional[str]) -> List[float]:
    """
    Detect available synergy folders (synergy_0, synergy_0.5, synergy_1, etc.).
    
    Returns a list of synergy values that actually exist in the filesystem.
    If no synergy folders exist, returns [None] to indicate no synergy subfolders.
    """
    import os
    import glob
    
    # Build base path without synergy folder (synergy_provided=False)
    base_path = _build_experiment_path(
        local_path, experiment_type, game_type, init_type, 
        map_name, False, None, None, study_name
    )
    
    # Look for synergy_* folders
    if os.path.exists(base_path):
        synergy_pattern = os.path.join(base_path, "synergy_*")
        synergy_paths = glob.glob(synergy_pattern)
        
        # Extract synergy values and sort them
        synergy_values = []
        for path in synergy_paths:
            folder_name = os.path.basename(path)
            if os.path.isdir(path):
                try:
                    synergy_str = folder_name.replace("synergy_", "")
                    synergy_val = float(synergy_str)
                    synergy_values.append(synergy_val)
                except ValueError:
                    continue  # Skip invalid folder names
        
        # Sort by synergy value for consistent ordering
        synergy_values.sort()
        
        # If synergy folders found, return them; otherwise return [None]
        return synergy_values if synergy_values else [None]
    
    # If base path doesn't exist, return [None]
    return [None]


def _detect_available_specialization_folders(local_path: str, experiment_type: str, 
                                           game_type: str, init_type: str, map_name: str, 
                                           synergy_provided: bool, synergy_scaling_factor: float, 
                                           study_name: Optional[str]) -> List[str]:
    """
    Detect available specialization folders (specialized_0, specialized_5, etc.).
    
    Returns a list of folder names that actually exist in the filesystem.
    """
    import os
    import glob
    
    # Build base path without specialization folder
    base_path = _build_experiment_path(
        local_path, experiment_type, game_type, init_type, 
        map_name, synergy_provided, synergy_scaling_factor, None, study_name
    )
    
    # Look for specialized_* folders
    if os.path.exists(base_path):
        specialized_pattern = os.path.join(base_path, "specialized_*")
        specialized_paths = glob.glob(specialized_pattern)
        
        # Extract folder names and sort them
        folder_names = []
        for path in specialized_paths:
            folder_name = os.path.basename(path)
            if os.path.isdir(path):
                folder_names.append(folder_name)
        
        # Sort by lambda value for consistent ordering
        def lambda_sort_key(folder_name):
            try:
                lambda_str = folder_name.replace("specialized_", "")
                return float(lambda_str)
            except ValueError:
                return float('inf')  # Put invalid names at the end
        
        folder_names.sort(key=lambda_sort_key)
        return folder_names
    
    return []
