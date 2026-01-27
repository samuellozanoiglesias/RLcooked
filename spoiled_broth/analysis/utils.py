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
    
    def parse_training_folder(self, folder_path: str, num_agents: int = 1) -> Optional[pd.DataFrame]:
        """
        Parse a single training folder and extract data.
        
        Args:
            folder_path: Path to training folder
            num_agents: Number of agents (1 for classic, 2 for competition)
            
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
        
        # Debug: Show REWARD_WEIGHTS line for troubleshooting
        reward_weights_match = re.search(r"REWARD_WEIGHTS:\s*(.+)", config_contents)
        if reward_weights_match:
            print(f"  Found REWARD_WEIGHTS: {reward_weights_match.group(1)}")
        
        matches = self.reward_pattern.findall(config_contents)
        print(f"  Reward pattern matches: {matches}")
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
        print(f"  GAME_VERSION extracted: {game_type}")
        
        # Extract walking and cutting speeds for each agent
        walking_speeds = {}
        cutting_speeds = {}
        
        # Parse WALKING_SPEEDS dictionary format
        walking_speeds_match = re.search(r"WALKING_SPEEDS:\s*({[^}]+})", config_contents)
        if walking_speeds_match:
            try:
                # Extract the dictionary string and evaluate it safely
                speeds_dict_str = walking_speeds_match.group(1)
                print(f"  Found WALKING_SPEEDS: {speeds_dict_str}")
                
                # Parse the dictionary manually to be safe
                for i in range(1, num_agents + 1):
                    agent_key = f"'ai_rl_{i}'"
                    speed_match = re.search(rf"{agent_key}:\s*([0-9.eE+-]+)", speeds_dict_str)
                    if speed_match:
                        walking_speeds[i] = float(speed_match.group(1))
                        print(f"    Extracted walking_speed_{i}: {walking_speeds[i]}")
            except Exception as e:
                print(f"  Error parsing WALKING_SPEEDS: {e}")
        else:
            print(f"  No WALKING_SPEEDS dictionary found in config")
            
        # Parse CUTTING_SPEEDS dictionary format  
        cutting_speeds_match = re.search(r"CUTTING_SPEEDS:\s*({[^}]+})", config_contents)
        if cutting_speeds_match:
            try:
                # Extract the dictionary string and evaluate it safely
                speeds_dict_str = cutting_speeds_match.group(1)
                print(f"  Found CUTTING_SPEEDS: {speeds_dict_str}")
                
                # Parse the dictionary manually to be safe
                for i in range(1, num_agents + 1):
                    agent_key = f"'ai_rl_{i}'"
                    speed_match = re.search(rf"{agent_key}:\s*([0-9.eE+-]+)", speeds_dict_str)
                    if speed_match:
                        cutting_speeds[i] = float(speed_match.group(1))
                        print(f"    Extracted cutting_speed_{i}: {cutting_speeds[i]}")
            except Exception as e:
                print(f"  Error parsing CUTTING_SPEEDS: {e}")
        else:
            print(f"  No CUTTING_SPEEDS dictionary found in config")
        
        # Load and clean CSV data
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        # Check if NUM_ENVS is > 1 in config to determine if we need to aggregate
        num_envs_match = re.search(r"NUM_ENVS:\s*([0-9]+)", config_contents)
        num_envs = int(num_envs_match.group(1)) if num_envs_match else 1
        
        print(f"Processing {folder_path} with NUM_ENVS = {num_envs}")
        
        if num_envs > 1:
            # Parse the multi-environment format where each data line is preceded by a header
            # Structure: header, episode_X_env_1, header, episode_X_env_2, ..., header, episode_X_env_N, 
            #           header, episode_X+1_env_1, header, episode_X+1_env_2, ...
            
            print(f"Processing multi-environment CSV with {num_envs} environments")
            
            header = None
            episode_groups = {}  # Dictionary to group data by episode number
            
            # Process the file alternating between headers and data
            i = 0
            data_lines_processed = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip empty lines
                if not line:
                    i += 1
                    continue
                
                # Check if this line is a header
                if line.startswith('episode,'):
                    if header is None:
                        header = line
                        print(f"Found header: {header[:50]}...")
                    i += 1
                    continue
                
                # This should be a data line
                values = line.split(',')
                if len(values) >= 2:  # At least episode and one metric
                    try:
                        episode_num = int(values[0])
                        
                        # Initialize episode group if not exists
                        if episode_num not in episode_groups:
                            episode_groups[episode_num] = []
                        
                        # Add this environment's data for this episode
                        episode_groups[episode_num].append(line)
                        data_lines_processed += 1
                        
                        # Print progress every 1000 lines
                        if data_lines_processed % 1000 == 0:
                            print(f"Processed {data_lines_processed} data lines, current episode: {episode_num}")
                            
                    except ValueError:
                        print(f"Warning: Could not parse episode number from line: {line[:50]}...")
                
                i += 1
            
            print(f"Finished parsing. Found {len(episode_groups)} unique episodes, {data_lines_processed} total data lines")
            
            if header and episode_groups:
                # Create cleaned CSV with single header and averaged data
                cleaned_lines = [header]
                
                # Process episodes in order - sample first 10 and last 10 for debugging
                episode_nums = sorted(episode_groups.keys())
                print(f"Episode range: {episode_nums[0]} to {episode_nums[-1]}")
                print(f"First 10 episodes: {episode_nums[:10]}")
                print(f"Last 10 episodes: {episode_nums[-10:]}")
                
                # Check a sample of episodes for environment count
                sample_episodes = episode_nums[:5] + episode_nums[-5:]
                for ep in sample_episodes:
                    env_count = len(episode_groups[ep])
                    if env_count != num_envs:
                        print(f"Warning: Episode {ep} has {env_count} environments, expected {num_envs}")
                
                # Process all episodes
                episodes_processed = 0
                for episode_num in episode_nums:
                    episode_data_lines = episode_groups[episode_num]
                    
                    # Parse each environment's data for this episode
                    episode_values = []
                    
                    for env_data in episode_data_lines:
                        values = env_data.split(',')
                        if len(values) >= 2:  # At least episode and one metric
                            # Convert numeric values (skip episode column)
                            numeric_values = []
                            for val in values[1:]:  # Skip episode column
                                try:
                                    numeric_values.append(float(val))
                                except ValueError:
                                    numeric_values.append(0.0)  # Default for non-numeric
                            episode_values.append(numeric_values)
                    
                    # Calculate mean across environments for this episode
                    if episode_values:
                        mean_values = [str(episode_num)]  # Start with episode number
                        
                        # Calculate means for each metric across environments
                        num_metrics = len(episode_values[0])
                        for metric_idx in range(num_metrics):
                            metric_sum = sum(env_vals[metric_idx] for env_vals in episode_values if len(env_vals) > metric_idx)
                            valid_envs = len([env_vals for env_vals in episode_values if len(env_vals) > metric_idx])
                            if valid_envs > 0:
                                mean_value = metric_sum / valid_envs
                                mean_values.append(str(mean_value))
                            else:
                                mean_values.append('0.0')
                        
                        cleaned_lines.append(','.join(mean_values))
                        episodes_processed += 1
                
                print(f"Successfully processed {episodes_processed} episodes")
                
                # Create DataFrame from cleaned data
                df = pd.read_csv(StringIO('\n'.join(cleaned_lines)))
                print(f"Final DataFrame: {len(df)} episodes aggregated across {num_envs} environments")
                print(f"  Each episode row now contains the MEAN values across {num_envs} parallel environments")
            else:
                print("Warning: Could not parse multi-environment format, falling back to single environment parsing")
                # Fallback to single environment parsing
                filtered_lines = [lines[0]] + [line for line in lines[1:] if not line.startswith("episode,")]
                df = pd.read_csv(StringIO("".join(filtered_lines)))
        else:
            # Single environment - use original logic but clean header duplication
            print(f"Single environment detected (NUM_ENVS = {num_envs})")
            filtered_lines = [lines[0]] + [line for line in lines[1:] if not line.startswith("episode,")]
            df = pd.read_csv(StringIO("".join(filtered_lines)))
            print(f"Loaded {len(df)} episodes from single environment")
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
        
        # Debug: Print DataFrame info
        print(f"Parsed training folder {folder_path}:")
        print(f"  NUM_ENVS detected: {num_envs}")
        print(f"  SEED detected: {seed if seed is not None else 'None'}")
        print(f"  GAME_TYPE detected: {game_type}")
        print(f"  Walking speeds: {walking_speeds}")
        print(f"  Cutting speeds: {cutting_speeds}")
        print(f"  DataFrame shape: {df.shape}")
        print(f"  All columns: {list(df.columns)}")
        
        # Check specifically for speed columns
        speed_cols_found = []
        for col in df.columns:
            if 'speed' in col.lower():
                speed_cols_found.append(col)
        print(f"  Speed columns found: {speed_cols_found}")
        
        if len(df) > 0:
            print(f"  Episodes range: {df['episode'].min()} to {df['episode'].max()}")
            print(f"  Sample data for first episode:")
            print(f"    Episode: {df.iloc[0]['episode']}")
            if 'seed' in df.columns:
                print(f"    Seed: {df.iloc[0]['seed']}")
            if 'game_type' in df.columns:
                print(f"    Game type: {df.iloc[0]['game_type']}")
            # Show speed values if they exist
            for col in speed_cols_found:
                print(f"    {col}: {df.iloc[0][col]}")
            # Show some key metrics if they exist
            key_metrics = ['pure_reward_ai_rl_1', 'deliver_ai_rl_1', 'cut_ai_rl_1', 'salad_ai_rl_1']
            for metric in key_metrics:
                if metric in df.columns:
                    print(f"    {metric}: {df.iloc[0][metric]}")
        else:
            print("  WARNING: DataFrame is empty!")
        
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
        print("Averaging data across seeds...")
        
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
            
        print(f"Grouping by: {group_cols}")
        
        # Get columns to average (numeric columns except grouping columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_average = [col for col in numeric_cols if col not in group_cols and col != 'seed']
        
        print(f"Columns to average: {len(cols_to_average)} columns")
        print(f"Example columns: {cols_to_average[:10]}..." if len(cols_to_average) > 10 else f"Columns: {cols_to_average}")
        
        # Check how many seeds we have
        unique_seeds = df['seed'].nunique()
        print(f"Found {unique_seeds} different seeds")
        
        if unique_seeds == 1:
            print("Only one seed found, no averaging needed")
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
        
        print(f"Averaging complete:")
        print(f"  Original data: {len(df)} rows")
        print(f"  Averaged data: {len(averaged_data)} rows")
        print(f"  Seeds per condition: {averaged_data['seeds_averaged'].iloc[0] if len(averaged_data) > 0 else 'N/A'}")
        
        return averaged_data
    
    def load_experiment_data(self, paths: Dict[str, str], 
                           num_agents: int = 1) -> pd.DataFrame:
        """
        Load all experiment data from the raw directory.
        Supports both direct training folders and study_name folder structure.
        When study_name is used, averages across different seeds for the same experiment.
        
        Args:
            paths: Dictionary containing directory paths
            num_agents: Number of agents in the experiment
            
        Returns:
            Combined DataFrame with all experiment data
        """
        raw_dir = paths['raw_dir']
        study_name = paths.get('study_name')
        
        print(f"Loading experiment data from directory: {raw_dir}")
        if study_name:
            print(f"Using study name: {study_name}")

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
            print("Direct training folders detected")
            for folder in training_folders:
                folder_path = os.path.join(raw_dir, folder)
                folders_found.append(folder)
                df = self.parse_training_folder(folder_path, num_agents)
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
                    df = self.parse_training_folder(folder_path, num_agents)
                    if df is not None:
                        all_dfs.append(df)
                        folders_processed.append(folder)
        
        else:
            # Study name mode - process all training folders and average seeds
            print(f"Processing study: {study_name}")
            print(f"Looking for Training_ folders in: {raw_dir}")
            
            # Find all training folders
            for folder in items_in_dir:
                folder_path = os.path.join(raw_dir, folder)
                if not os.path.isdir(folder_path) or not folder.startswith('Training_'):
                    continue
                    
                folders_found.append(folder)
                df = self.parse_training_folder(folder_path, num_agents)
                if df is not None:
                    # Add study_name information
                    df['study_name'] = study_name
                    
                    # Seed should already be extracted from config.txt during parse_training_folder
                    # If not present, use timestamp as a unique identifier
                    if 'seed' not in df.columns:
                        df['seed'] = df['timestamp'].iloc[0] if len(df) > 0 else 'unknown'
                    
                    all_dfs.append(df)
                    folders_processed.append(folder)
        
        print(f"Found {len(folders_found)} folders, successfully processed {len(folders_processed)}")
        if folders_found:
            print(f"Folders found: {folders_found}")
        if folders_processed:
            print(f"Folders processed: {folders_processed}")
        
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
        
        # Remove existing output file and save new one
        if os.path.exists(paths['output_path']):
            os.remove(paths['output_path'])
        
        final_df.to_csv(paths['output_path'], index=False)
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
                print(f"Created pure_reward_total from pure_reward_ai_rl_1")
            else:
                print(f"WARNING: pure_reward_ai_rl_1 column not found!")
                
            if "deliver_ai_rl_1" in df.columns:
                df["total_deliveries"] = df["deliver_ai_rl_1"]
                print(f"Created total_deliveries from deliver_ai_rl_1")
            else:
                print(f"WARNING: deliver_ai_rl_1 column not found!")
                
        else:  # num_agents == 2
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
                print(f"Created speed_key with all 4 speed values")
            elif has_speed_1:
                df["speed_key"] = df.apply(
                    lambda row: f"{row['walking_speed_1']}_{row['cutting_speed_1']}", 
                    axis=1
                )
                print(f"Created speed_key with agent 1 speeds only")
            elif has_speed_2:
                df["speed_key"] = df.apply(
                    lambda row: f"{row['walking_speed_2']}_{row['cutting_speed_2']}", 
                    axis=1
                )
                print(f"Created speed_key with agent 2 speeds only")
            else:
                print(f"WARNING: No speed columns found for speed_key creation")
                print(f"  Available columns containing 'speed': {[col for col in df.columns if 'speed' in col.lower()]}")
            
            # Create total columns for two agents
            if "pure_reward_ai_rl_1" in df.columns and "pure_reward_ai_rl_2" in df.columns:
                df["pure_reward_total"] = df["pure_reward_ai_rl_1"] + df["pure_reward_ai_rl_2"]
            if "deliver_ai_rl_1" in df.columns and "deliver_ai_rl_2" in df.columns:
                df["total_deliveries"] = df["deliver_ai_rl_1"] + df["deliver_ai_rl_2"]
        
        # Debug: Print final DataFrame info after processing
        print(f"After prepare_dataframe processing:")
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Key columns present:")
        key_cols = ['episode', 'pure_reward_total', 'total_deliveries', 'pure_reward_ai_rl_1', 'deliver_ai_rl_1']
        for col in key_cols:
            if col in df.columns:
                print(f"    {col}: YES (sample value: {df[col].iloc[0] if len(df) > 0 else 'N/A'})")
            else:
                print(f"    {col}: NO")
        
        # Show speed-related columns
        speed_cols = [col for col in df.columns if 'speed' in col.lower()]
        print(f"  Speed-related columns: {speed_cols}")
        if speed_cols and len(df) > 0:
            for col in speed_cols:
                print(f"    {col}: {df[col].iloc[0]}")
        
        # Show if game_type column exists
        if 'game_type' in df.columns:
            print(f"  game_type: {df['game_type'].iloc[0] if len(df) > 0 else 'N/A'}")
        else:
            print(f"  game_type: NO")
        
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
        default=None,
        help='Initialization type subdirectory to analyze (default: process all available init types)'
    )
    
    parser.add_argument(
        '--eta',
        type=float,
        default=0.0,
        help='Eta parameter for reference-based reward shaping (default: 0.0)'
    )
    
    return parser


def main_analysis_pipeline(experiment_type: str, map_name: str,
                          cluster: str = 'cuenca', smoothing_factor: int = 15, 
                          num_agents: Optional[int] = None, study_name: Optional[str] = None,
                          game_type: str = 'classic', init_type: str = 'random_init',
                          eta: float = 0.0, eta_provided: bool = False,
                          specialization_enabled: Optional[bool] = None) -> Dict:
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
        eta: Eta parameter for reference-based reward shaping (default: 0.0)
        eta_provided: Whether eta was explicitly provided (to determine folder structure)
        specialization_enabled: Whether specialization penalty is enabled (None = auto-detect from folders)
        
    Returns:
        Dictionary containing processed data and paths
    """
    # Initialize components
    config = AnalysisConfig()
    config.smoothing_factor = smoothing_factor
    
    processor = DataProcessor(config)
    plotter = PlotGenerator(config)
    
    # Custom path setup that includes init_type and eta in the correct order
    local_path = config.cluster_paths[cluster]
    
    # Auto-detect specialization from folder structure if not provided
    spec_folders_to_try = []
    if specialization_enabled is None:
        # Try both folders, prioritize specialized
        spec_folders_to_try = ["specialized", "non_specialized", ""]  # Empty string for backwards compatibility
    elif specialization_enabled:
        spec_folders_to_try = ["specialized"]
    else:
        spec_folders_to_try = ["non_specialized"]
    
    # Only create eta subfolder if eta was explicitly provided (even if it's 0)
    if eta_provided:
        eta_folder = f"eta_{eta}"
        
        # Build the experiment path based on experiment type with eta folder and specialization
        for spec_folder in spec_folders_to_try:
            if 'pretrain' in experiment_type.lower():
                # For pretraining: /data/samuel_lozano/cooked/pretraining/{game_type}/{init_type}/map_{map_name}/{eta_folder}/{spec_folder}/
                if study_name:
                    if spec_folder:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/pretraining/{game_type}/{init_type}/map_{map_name}/{eta_folder}/{spec_folder}/{study_name}"
                    else:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/pretraining/{game_type}/{init_type}/map_{map_name}/{eta_folder}/{study_name}"
                else:
                    if spec_folder:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/pretraining/{game_type}/{init_type}/map_{map_name}/{eta_folder}/{spec_folder}"
                    else:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/pretraining/{game_type}/{init_type}/map_{map_name}/{eta_folder}"
            elif 'classic' in experiment_type.lower() or 'competition' in experiment_type.lower():
                # For classic/competition: /data/samuel_lozano/cooked/{game_type}/{init_type}/map_{map_name}/{eta_folder}/{spec_folder}/
                if study_name:
                    if spec_folder:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/{game_type}/{init_type}/map_{map_name}/{eta_folder}/{spec_folder}/{study_name}"
                    else:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/{game_type}/{init_type}/map_{map_name}/{eta_folder}/{study_name}"
                else:
                    if spec_folder:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/{game_type}/{init_type}/map_{map_name}/{eta_folder}/{spec_folder}"
                    else:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/{game_type}/{init_type}/map_{map_name}/{eta_folder}"
            else:
                # For other experiment types: /data/samuel_lozano/cooked/{experiment_type}/{init_type}/map_{map_name}/{eta_folder}/{spec_folder}/
                if study_name:
                    if spec_folder:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/{experiment_type}/{init_type}/map_{map_name}/{eta_folder}/{spec_folder}/{study_name}"
                    else:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/{experiment_type}/{init_type}/map_{map_name}/{eta_folder}/{study_name}"
                else:
                    if spec_folder:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/{experiment_type}/{init_type}/map_{map_name}/{eta_folder}/{spec_folder}"
                    else:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/{experiment_type}/{init_type}/map_{map_name}/{eta_folder}"
            
            # Check if this path exists (for auto-detection)
            if os.path.exists(raw_dir):
                break
    else:
        # No eta folder - use original structure when eta is not provided (with specialization folder)
        for spec_folder in spec_folders_to_try:
            if 'pretrain' in experiment_type.lower():
                # For pretraining: /data/samuel_lozano/cooked/pretraining/{game_type}/{init_type}/map_{map_name}/{spec_folder}/
                if study_name:
                    if spec_folder:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/pretraining/{game_type}/{init_type}/map_{map_name}/{spec_folder}/{study_name}"
                    else:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/pretraining/{game_type}/{init_type}/map_{map_name}/{study_name}"
                else:
                    if spec_folder:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/pretraining/{game_type}/{init_type}/map_{map_name}/{spec_folder}"
                    else:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/pretraining/{game_type}/{init_type}/map_{map_name}"
            
            # Check if this path exists (for auto-detection)
            if os.path.exists(raw_dir):
                break
            
            elif 'classic' in experiment_type.lower() or 'competition' in experiment_type.lower():
                # For classic/competition: /data/samuel_lozano/cooked/{game_type}/{init_type}/map_{map_name}/{spec_folder}/
                if study_name:
                    if spec_folder:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/{game_type}/{init_type}/map_{map_name}/{spec_folder}/{study_name}"
                    else:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/{game_type}/{init_type}/map_{map_name}/{study_name}"
                else:
                    if spec_folder:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/{game_type}/{init_type}/map_{map_name}/{spec_folder}"
                    else:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/{game_type}/{init_type}/map_{map_name}"
            else:
                # For other experiment types: /data/samuel_lozano/cooked/{experiment_type}/{init_type}/map_{map_name}/{spec_folder}/
                if study_name:
                    if spec_folder:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/{experiment_type}/{init_type}/map_{map_name}/{spec_folder}/{study_name}"
                    else:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/{experiment_type}/{init_type}/map_{map_name}/{study_name}"
                else:
                    if spec_folder:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/{experiment_type}/{init_type}/map_{map_name}/{spec_folder}"
                    else:
                        raw_dir = f"{local_path}/data/samuel_lozano/cooked/{experiment_type}/{init_type}/map_{map_name}"
            
            # Check if this path exists (for auto-detection)
            if os.path.exists(raw_dir):
                # Detect which specialization folder was found
                detected_spec = spec_folder if spec_folder else None
                break
    
    # If no specialization was detected, set to None
    if 'detected_spec' not in locals():
        detected_spec = None
    
    paths = {
        'raw_dir': raw_dir,
        'output_path': f"{raw_dir}/training_results.csv",
        'figures_dir': f"{raw_dir}/training_figures/",
        'smoothed_figures_dir': f"{raw_dir}/training_figures/smoothed_{config.smoothing_factor}/",
        'study_name': study_name,
        'init_type': init_type,
        'eta': eta,
        'specialization': detected_spec  # 'specialized', 'non_specialized', or None
    }
    
    # Create directories
    for dir_path in [paths['raw_dir'], paths['figures_dir'], paths['smoothed_figures_dir']]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Determine number of agents based on experiment type
    if num_agents is None:
        # Pretrained experiments have 1 agent, others have 2
        num_agents = 1 if 'pretrain' in experiment_type.lower() else 2
    
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