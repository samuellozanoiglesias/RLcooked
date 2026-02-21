#!/usr/bin/env python3
"""
Map Configuration Analysis Figure Generator

This script creates color grid plots comparing different map configurations
across performance metrics. The visualization shows how different maps perform
relative to the baseline map configuration.

Map configurations tested (hardcoded list):
- baseline_division_of_labor_large (baseline)
- encouraged_division_of_labor_large  
- forced_division_of_labor_large
- baseline_competition
- baseline_division_of_labor
- encouraged_division_of_labor
- forced_division_of_labor

All agents use fixed abilities: Agent 1 (1.0, 1.0), Agent 2 (1.0, 1.0)

Color coding:
- White: Baseline map (difference=0)
- Blue: Better performance than baseline
- Red: Worse performance than baseline

The script supports different initialization types (random_init, empty_init) that
are located in subdirectories under the main experiment path.

Usage:
    python grid_map_cooperative_analysis.py [options]

Examples:
    # Default analysis
    nohup python grid_map_cooperative_analysis.py --episode_range final --num_episodes 100 > grid_map_analysis.log 2>&1 &
    
    # Analyze specific lambda data
    nohup python grid_map_cooperative_analysis.py --episode_range final --specialization 0 --num_episodes 100 > map_grid_lambda_0.log 2>&1 &
    
    # Analyze empty_init data
    nohup python grid_map_cooperative_analysis.py --episode_range final --init_type empty_init --num_episodes 100 > map_grid_empty_init.log 2>&1 &
    
    # Analyze with synergy parameter
    nohup python grid_map_cooperative_analysis.py --episode_range final --init_type random_init --synergy 0.5 --num_episodes 100 > map_grid_synergy_0.5.log 2>&1 &
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spoiled_broth.analysis.utils import DataProcessor, AnalysisConfig


class CooperativeAnalyzer:
    """Handles data loading and processing for cooperative analysis."""
    
    def __init__(self, study_name: Optional[str] = None, 
                 map_name_1: str = 'baseline_division_of_labor_large',
                 map_name_2: str = 'encouraged_division_of_labor_large',
                 init_type: str = 'empty_init',
                 synergy: float = 0.0,
                 synergy_provided: bool = False,
                 specialization: Optional[float] = None,
                 game_type: str = 'classic',
                 cluster: str = 'cuenca'):
        self.config = AnalysisConfig()
        self.data_processor = DataProcessor(self.config)
        self.study_name = study_name
        self.map_name_1 = map_name_1
        self.map_name_2 = map_name_2
        self.init_type = init_type
        self.synergy = synergy
        self.synergy_provided = synergy_provided
        self.specialization = specialization  # None=auto-detect, float=specific lambda
        self.game_type = game_type
        self.cluster = cluster
        self.detected_specializations = []  # Track which specializations were found
        
        # Set up cluster path prefix
        if cluster not in self.config.cluster_paths:
            raise ValueError(f"Invalid cluster '{cluster}'. Choose from {list(self.config.cluster_paths.keys())}")
        self.local_path = self.config.cluster_paths[cluster]
        
        # Define hardcoded map list for grid analysis
        # All agents use fixed abilities: Agent 1: (1.0, 1.0), Agent 2: (1.0, 1.0)
        self.map_names = [
            'baseline_division_of_labor_large',
            'semiencouraged_division_of_labor_large',
            '1-semiencouraged_division_of_labor_large',
            '2-semiencouraged_division_of_labor_large',
            'encouraged_division_of_labor_large',
            '1-encouraged_division_of_labor_large', 
            '2-encouraged_division_of_labor_large',
            '3-encouraged_division_of_labor_large',
        ]
        
        # Fixed ability configuration - always use baseline (1.0, 1.0) for both agents
        self.baseline_abilities = (1.0, 1.0)
        
        # Define experimental conditions mapping for map configurations
        self.base_condition_mapping = {}
        for map_name in self.map_names:
            map_short = self._extract_map_short_name(map_name)
            collision_suffix = '_collision' if 'collision' in game_type else ''
            condition_key = (map_name, game_type)
            condition_name = f'{map_short}{collision_suffix}'
            self.base_condition_mapping[condition_key] = condition_name
        
        # Color palette will be generated dynamically based on performance
        # White for X=1.0, blue for better performance, red for worse
        self.color_palette = {}
        
        # Define baseline map for comparison (first in the list)
        self.baseline_map = self.map_names[0] if self.map_names else 'baseline_division_of_labor_large'
        
        # Performance metrics to analyze (updated for multi-agent cooperative data)
        # These will be created from individual agent metrics
        self.performance_metrics = {
            'total_deliveries': 'Total Deliveries (Combined)',
            'total_counters': 'Total Counters (Combined)', 
            'useful_delivery_ai_rl_1': 'Deliveries Agent 1',
            'useful_delivery_ai_rl_2': 'Deliveries Agent 2',
            'cut_ai_rl_1': 'Cuts Agent 1',
            'cut_ai_rl_2': 'Cuts Agent 2'
        }
    
    def _extract_map_short_name(self, map_name: str) -> str:
        """Extract a short identifier from the full map name.
        
        Examples:
            'baseline_division_of_labor_large' -> 'baseline'
            'encouraged_division_of_labor_large' -> 'encouraged'
            'forced_division_of_labor_large' -> 'forced'
        """
        # Common pattern: {prefix}_division_of_labor_large
        if '_division_of_labor' in map_name:
            return map_name.split('_division_of_labor')[0]
        # Fallback: use first word
        return map_name.split('_')[0]
    
    def _build_data_path(self, *path_parts) -> str:
        """Build a data path handling empty local_path (cuenca cluster).
        
        Args:
            *path_parts: Path components to join
            
        Returns:
            Properly constructed path
        """
        if self.local_path:
            return os.path.join(self.local_path, *path_parts)
        else:
            # For cuenca cluster (empty local_path), use absolute path
            return os.path.join('/', *path_parts)
    
    def _detect_available_synergy_values(self) -> List[float]:
        """Detect available synergy values from directory structure.
        
        Returns:
            List of synergy values found in the directory structure
        """
        synergy_values = []
        
        # Build path to check for synergy directories
        base_path = self._build_data_path('data', 'samuel_lozano', 'cooked', self.game_type, self.init_type, f'map_{self.map_name_1}')
        
        if os.path.exists(base_path):
            try:
                # List all directories in the base path
                subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                
                for subdir in subdirs:
                    if subdir.startswith('synergy_'):
                        try:
                            # Extract synergy value from directory name
                            synergy_str = subdir.replace('synergy_', '')
                            if synergy_str == '0':
                                synergy_val = 0.0
                            else:
                                synergy_val = float(synergy_str)
                            synergy_values.append(synergy_val)
                        except ValueError:
                            print(f"    Warning: Could not parse synergy value from directory {subdir}")
                            continue
                            
                if synergy_values:
                    synergy_values.sort()
                    print(f"    Found synergy values: {synergy_values}")
                else:
                    print(f"    No synergy directories found, using default synergy=0.0")
                    synergy_values = [0.0]
                    
            except Exception as e:
                print(f"    Warning: Error detecting synergy values: {e}")
                synergy_values = [0.0]
        else:
            print(f"    Warning: Base path {base_path} not found, using default synergy=0.0")
            synergy_values = [0.0]
            
        return synergy_values
    
    def _detect_available_specialization_values(self, synergy_val: float) -> List[str]:
        """Detect available specialization values for a given synergy value.
        
        Args:
            synergy_val: Synergy value to check
            
        Returns:
            List of specialization folder names found
        """
        specialization_values = []
        
        # Build path to check for specialization directories
        # Use 2 decimal places to match actual folder names on disk
        synergy_str = 'synergy_0' if synergy_val == 0.0 else f'synergy_{synergy_val:.2f}'
        base_path = self._build_data_path('data', 'samuel_lozano', 'cooked', self.game_type, self.init_type, f'map_{self.map_name_1}', synergy_str)
        
        if os.path.exists(base_path):
            try:
                # List all directories in the synergy path
                subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                
                for subdir in subdirs:
                    if subdir.startswith('specialized_'):
                        specialization_values.append(subdir)
                        
                if specialization_values:
                    specialization_values.sort()
                    print(f"    Found specialization values for synergy {synergy_val}: {specialization_values}")
                else:
                    print(f"    No specialization directories found for synergy {synergy_val}")
                    
            except Exception as e:
                print(f"    Warning: Error detecting specialization values for synergy {synergy_val}: {e}")
        else:
            print(f"    Warning: Synergy path {base_path} not found")
            
        return specialization_values
    
    def load_experimental_data(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load and combine data from all experimental conditions.
        
        Returns:
            If analyzing single synergy/specialization: pd.DataFrame
            If analyzing multiple combinations: Dict[str, pd.DataFrame] with keys like:
                - 'synergy_0.50__specialized_0.25': Data for that specific combination
                - 'synergy_1.00__specialized_5.00': Data for that specific combination
                - etc.
        """
        print("Loading experimental data from all conditions...")
        if self.study_name:
            print(f"Using study name: {self.study_name}")
        if self.specialization is None:
            print("Specialization lambda: Auto-detecting all available lambda values")
        else:
            print(f"Specialization lambda: {self.specialization}")
        
        # Auto-detect synergy values if not explicitly specified
        synergy_values_to_analyze = []
        if not self.synergy_provided:
            print("Synergy: Auto-detecting all available synergy values")
            synergy_values_to_analyze = self._detect_available_synergy_values()
        else:
            synergy_values_to_analyze = [self.synergy]
            print(f"Synergy: Using specified value {self.synergy}")
        
        # Store data separately for each synergy/specialization combination
        all_combinations = {}
        
        # Iterate over all synergy values
        for synergy_val in synergy_values_to_analyze:
            print(f"\nProcessing synergy value: {synergy_val}")
            
            # Determine which specialization lambdas to try for this synergy value
            if self.specialization is None:
                # Auto-detect available specialization values for this synergy
                spec_modes_to_analyze = self._detect_available_specialization_values(synergy_val)
                if not spec_modes_to_analyze:
                    # Fallback to common values
                    spec_modes_to_analyze = ['specialized_0', 'specialized_5']
                    print(f"    Using fallback specialization values: {spec_modes_to_analyze}")
            else:
                # Use specified lambda - format with 2 decimal places to match folder names
                if self.specialization == 0:
                    spec_folder_name = 'specialized_0'
                else:
                    spec_folder_name = f'specialized_{self.specialization:.2f}'
                spec_modes_to_analyze = [spec_folder_name]
            
            # Load data for each specialization mode for this synergy value
            for spec_mode in spec_modes_to_analyze:
                print(f"  Processing specialization: {spec_mode}")
                
                # Temporarily update synergy for this iteration
                original_synergy = self.synergy
                self.synergy = synergy_val
                condition_data = []  # Reset for each specialization mode
                
                # Load all conditions for this specialization mode
                for (map_name, game_type), condition_name in self.base_condition_mapping.items():
                    # Keep original condition name (no suffix)
                    print(f"\nProcessing condition: {condition_name}")
                    print(f"  Map: {map_name}, Game type: {game_type}")
                    if spec_mode:
                        print(f"  Specialization: {spec_mode}")
                    
                    try:
                        # Set up paths for this condition with init_type and synergy in the correct order
                        # Always use synergy folder when we have a specific synergy value (either provided or auto-detected)
                        
                        # Build path with synergy folder
                        if self.synergy == 0.0:
                            synergy_folder = "synergy_0"
                        else:
                            synergy_folder = f"synergy_{self.synergy:.2f}"
                        
                        # Use the specific specialization mode
                        if spec_mode:
                            spec_folder = spec_mode
                        else:
                            spec_folder = ""  # No specialization folder
                        
                        # Build directory path
                        if self.study_name:
                            # Use study_name folder structure
                            if spec_folder:
                                raw_dir = self._build_data_path('data', 'samuel_lozano', 'cooked', game_type, self.init_type, f'map_{map_name}', synergy_folder, spec_folder, self.study_name)
                            else:
                                raw_dir = self._build_data_path('data', 'samuel_lozano', 'cooked', game_type, self.init_type, f'map_{map_name}', synergy_folder, self.study_name)
                        else:
                            # Default structure
                            if spec_folder:
                                raw_dir = self._build_data_path('data', 'samuel_lozano', 'cooked', game_type, self.init_type, f'map_{map_name}', synergy_folder, spec_folder)
                            else:
                                raw_dir = self._build_data_path('data', 'samuel_lozano', 'cooked', game_type, self.init_type, f'map_{map_name}', synergy_folder)
                        
                        print(f"  Loading from: {raw_dir}")
                    
                    except Exception as e:
                        print(f"  Error loading condition {condition_name}: {e}")
                        continue
                    
                    paths = {
                        'raw_dir': raw_dir,
                        'output_path': f"{raw_dir}/training_results.csv",
                        'figures_dir': f"{raw_dir}/training_figures/",
                        'smoothed_figures_dir': f"{raw_dir}/training_figures/smoothed_{self.config.smoothing_factor}/",
                        'study_name': self.study_name
                    }
                    
                    # Create the directories if they don't exist
                    for dir_path in [paths['raw_dir'], paths['figures_dir'], paths['smoothed_figures_dir']]:
                        os.makedirs(dir_path, exist_ok=True)
                    
                    # Load the data (using 2 agents since this is multi-agent cooperative data)
                    df = self.data_processor.load_experiment_data(paths, num_agents=2)
                    
                    if df is not None and len(df) > 0:
                        print(f"  Loaded initial data shape: {df.shape}")
                        print(f"  Available columns: {list(df.columns)[:10]}...")  # Show first 10 columns
                        
                        # Check if the training has the expected ability configuration
                        # Ability configuration: Agent1 (1.0, 1.0), Agent2 (1.0, 1.0)
                        df_filtered = self._filter_by_ability_config(df)
                        
                        if len(df_filtered) == 0:
                            print(f"  Warning: No data found matching baseline ability configuration (1.0, 1.0)")
                            print(f"  Available ability configurations in this training:")
                            self._print_available_abilities(df)
                            continue
                        
                        print(f"  Filtered to {len(df_filtered)} episodes matching baseline abilities (1.0, 1.0)")
                        
                        # Create combined metrics from individual agent metrics
                        self._create_combined_metrics(df_filtered)
                        
                        # Add condition metadata
                        df_filtered['condition'] = condition_name
                        df_filtered['map_name'] = map_name 
                        df_filtered['game_type_clean'] = game_type
                        df_filtered['map_name_clean'] = map_name
                        if spec_mode:
                            df_filtered['specialization'] = spec_mode
                        
                        # Add color palette entry for this specific condition
                        if condition_name not in self.color_palette:
                            # No color palette needed for grid plots
                            pass
                        
                        # Store data for this condition
                        condition_data.append(df_filtered)
                        
                        print(f"  Loaded {len(df_filtered)} episodes")
                        
                        # Debug: Check if our target metrics exist
                        print(f"  Loaded {len(df_filtered)} episodes")
                        
                        # Debug: Check if our target metrics exist
                        missing_metrics = [metric for metric in self.performance_metrics.keys() if metric not in df_filtered.columns]
                        if missing_metrics:
                            print(f"  Warning: Missing metrics: {missing_metrics}")
                            available_metrics = [metric for metric in self.performance_metrics.keys() if metric in df_filtered.columns]
                            print(f"  Available metrics: {available_metrics}")
                        
                    else:
                        print(f"  Warning: No data loaded for condition {condition_name}")
                
                # Store data for this combination if we have any
                if condition_data:
                    combined_df = pd.concat(condition_data, ignore_index=True)
                    
                    # Add metadata about this combination
                    combined_df['synergy_value'] = synergy_val
                    combined_df['specialization_mode'] = spec_mode
                    
                    # Track this combination
                    combination_key = f"synergy_{synergy_val}__{spec_mode}"
                    
                    print(f"    Loaded {len(combined_df)} episodes for {combination_key}")
                    print(f"    Conditions in this combination: {sorted(combined_df['condition'].unique())}")
                    
                    all_combinations[combination_key] = combined_df
                else:
                    print(f"    No data found for synergy {synergy_val}, specialization {spec_mode}")
                
                # Restore original synergy
                self.synergy = original_synergy
        
        # Return based on how many combinations we found
        if not all_combinations:
            raise ValueError("No data could be loaded from any synergy/specialization combination")
        
        if len(all_combinations) == 1:
            # Single combination - return as DataFrame
            combination_key = list(all_combinations.keys())[0]
            print(f"\nReturning single combination: {combination_key}")
            return all_combinations[combination_key]
        else:
            # Multiple combinations - return as Dict
            print(f"\nReturning {len(all_combinations)} combinations:")
            for combination_key, df in all_combinations.items():
                print(f"  {combination_key}: {len(df)} episodes")
            return all_combinations
        
        return final_combined_df
    
    def prepare_episode_data(self, df: pd.DataFrame, episode_selection: str = 'all', 
                           num_episodes: int = 100, target_episode: Optional[int] = None) -> pd.DataFrame:
        """
        Prepare data based on episode selection criteria.
        
        Args:
            df: Combined dataframe with all experimental data
            episode_selection: 'all', 'final', 'average', or 'specific'
            num_episodes: Number of episodes to use (for 'final' or 'specific' modes)
            target_episode: Target episode number (required for 'specific' mode)
            
        Returns:
            Processed dataframe ready for plotting
        """
        print(f"Preparing data with episode selection: {episode_selection}")
        
        if episode_selection == 'all':
            # Use all episodes as individual data points
            return df.copy()
            
        elif episode_selection == 'final':
            # Use only the final N episodes from each training
            final_data = []
            
            for condition in df['condition'].unique():
                condition_df = df[df['condition'] == condition].copy()
                
                # Group by training session (timestamp)
                for timestamp in condition_df['timestamp'].unique():
                    training_df = condition_df[condition_df['timestamp'] == timestamp].copy()
                    training_df = training_df.sort_values('episode')
                    
                    # Filter out incomplete episodes first (episodes with fewer environments than expected)
                    # This happens when training is interrupted and final episodes don't have all environments
                    complete_episodes = self._filter_complete_episodes(training_df)
                    
                    if len(complete_episodes) == 0:
                        print(f"    Warning: No complete episodes found for training {timestamp}")
                        continue
                    
                    # Take final episodes from complete episodes only
                    final_episodes = complete_episodes.tail(num_episodes)
                    final_data.append(final_episodes)
            
            final_df = pd.concat(final_data, ignore_index=True)
            print(f"Using final {num_episodes} episodes: {len(final_df)} total episodes")
            return final_df
            
        elif episode_selection == 'specific':
            # Use N episodes around a specific target episode
            if target_episode is None:
                raise ValueError("target_episode must be provided when using episode_selection='specific'")
            
            specific_data = []
            half_window = num_episodes // 2
            
            for condition in df['condition'].unique():
                condition_df = df[df['condition'] == condition].copy()
                
                # Group by training session (timestamp)
                for timestamp in condition_df['timestamp'].unique():
                    training_df = condition_df[condition_df['timestamp'] == timestamp].copy()
                    training_df = training_df.sort_values('episode')
                    
                    # Filter out incomplete episodes
                    complete_episodes = self._filter_complete_episodes(training_df)
                    
                    if len(complete_episodes) == 0:
                        print(f"    Warning: No complete episodes found for training {timestamp}")
                        continue
                    
                    # Convert target_episode (1-based index within training) to actual episode number
                    # target_episode refers to the Nth episode within this training session
                    if target_episode > len(complete_episodes):
                        episode_range = f"{complete_episodes['episode'].min()}-{complete_episodes['episode'].max()}"
                        print(f"    Warning: target_episode {target_episode} exceeds training length ({len(complete_episodes)} episodes) "
                              f"for training {timestamp}. Available episode range: {episode_range}")
                        continue
                    
                    # Get the actual episode number for the target_episode index (1-based)
                    target_episode_actual = complete_episodes.iloc[target_episode - 1]['episode']
                    
                    # Calculate the window around this target episode (using actual episode numbers)
                    start_episode = target_episode_actual - half_window
                    end_episode = target_episode_actual + half_window
                    
                    # Filter episodes within the range
                    specific_episodes = complete_episodes[
                        (complete_episodes['episode'] >= start_episode) & 
                        (complete_episodes['episode'] <= end_episode)
                    ]
                    
                    if len(specific_episodes) > 0:
                        specific_data.append(specific_episodes)
                        print(f"    Found {len(specific_episodes)} episodes around episode {target_episode} "
                              f"(actual episode {target_episode_actual}, range: {start_episode}-{end_episode}) "
                              f"for training {timestamp}")
                    else:
                        episode_range = f"{complete_episodes['episode'].min()}-{complete_episodes['episode'].max()}"
                        print(f"    Warning: No episodes found around target episode {target_episode} "
                              f"(actual episode {target_episode_actual}) for training {timestamp}. "
                              f"Available episode range: {episode_range}")
            
            if not specific_data:
                raise ValueError(f"No episodes found around target episode {target_episode}")
            
            specific_df = pd.concat(specific_data, ignore_index=True)
            print(f"Using {num_episodes} episodes around episode {target_episode}: {len(specific_df)} total episodes")
            return specific_df
            
        elif episode_selection == 'average':
            # Average across episodes for each training, then use training averages as data points
            averaged_data = []
            
            for condition in df['condition'].unique():
                condition_df = df[df['condition'] == condition].copy()
                
                # Group by training session and average
                for timestamp in condition_df['timestamp'].unique():
                    training_df = condition_df[condition_df['timestamp'] == timestamp].copy()
                    
                    # Calculate averages for performance metrics
                    avg_row = {
                        'condition': condition,
                        'timestamp': timestamp,
                        'map_name': training_df['map_name'].iloc[0],
                        'game_type_clean': training_df['game_type_clean'].iloc[0],
                        'speed_condition': training_df['speed_condition'].iloc[0]
                    }
                    
                    # Average the performance metrics
                    for metric in self.performance_metrics.keys():
                        if metric in training_df.columns:
                            avg_row[metric] = training_df[metric].mean()
                        else:
                            print(f"Warning: Metric {metric} not found in data")
                            avg_row[metric] = 0
                    
                    averaged_data.append(avg_row)
            
            avg_df = pd.DataFrame(averaged_data)
            print(f"Using training averages: {len(avg_df)} training sessions")
            return avg_df
        
        else:
            raise ValueError(f"Invalid episode_selection: {episode_selection}. Choose from 'all', 'final', 'average', 'specific'")
    
    def _create_combined_metrics(self, df: pd.DataFrame):
        """Create combined metrics from individual agent metrics."""
        
        # Create total_deliveries from sum of both agents' deliveries
        if 'deliver_ai_rl_1' in df.columns and 'deliver_ai_rl_2' in df.columns:
            df['total_deliveries'] = df['deliver_ai_rl_1'] + df['deliver_ai_rl_2']
            print(f"    Created total_deliveries from deliver_ai_rl_1 + deliver_ai_rl_2")
        elif 'deliver_ai_rl_1' in df.columns:
            df['total_deliveries'] = df['deliver_ai_rl_1']
            print(f"    Created total_deliveries from deliver_ai_rl_1 only")
        else:
            print(f"    Warning: Could not create total_deliveries - no deliver columns found")
            df['total_deliveries'] = 0
        
        # Create pure_reward_total from sum of both agents' pure rewards
        if 'pure_reward_ai_rl_1' in df.columns and 'pure_reward_ai_rl_2' in df.columns:
            df['pure_reward_total'] = df['pure_reward_ai_rl_1'] + df['pure_reward_ai_rl_2']
            print(f"    Created pure_reward_total from pure_reward_ai_rl_1 + pure_reward_ai_rl_2")
        elif 'pure_reward_ai_rl_1' in df.columns:
            df['pure_reward_total'] = df['pure_reward_ai_rl_1']
            print(f"    Created pure_reward_total from pure_reward_ai_rl_1 only")
        else:
            print(f"    Warning: Could not create pure_reward_total - no pure_reward columns found")
            df['pure_reward_total'] = 0
        
        # Create total_counters from sum of both agents' counter uses
        if 'counter_ai_rl_1' in df.columns and 'counter_ai_rl_2' in df.columns:
            df['total_counters'] = df['counter_ai_rl_1'] + df['counter_ai_rl_2']
            print(f"    Created total_counters from counter_ai_rl_1 + counter_ai_rl_2")
        elif 'counter_ai_rl_1' in df.columns:
            df['total_counters'] = df['counter_ai_rl_1']
            print(f"    Created total_counters from counter_ai_rl_1 only")
        else:
            print(f"    Warning: Could not create total_counters - no counter columns found")
            df['total_counters'] = 0
    
    def _filter_complete_episodes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out episodes that don't have the expected number of environments.
        
        Incomplete episodes typically occur at the end of training when the process
        is interrupted before all environments finish an episode.
        """
        if len(df) == 0:
            return df
            
        # For multi-environment training, episodes should have consistent structure
        # We can detect incomplete episodes by checking if the last few episodes 
        # have significantly different patterns or if there are obvious gaps
        
        # Simple approach: remove the last episode if it seems incomplete
        # by checking if the last episode has very different values than previous ones
        # or if there are obvious data quality issues
        
        # Sort by episode to ensure proper order
        df_sorted = df.sort_values('episode').copy()
        
        if len(df_sorted) < 2:
            return df_sorted
        
        # Check for obvious data quality issues in the last episode
        last_episode = df_sorted.iloc[-1]
        second_last_episode = df_sorted.iloc[-2] if len(df_sorted) > 1 else None
        
        # Remove last episode if it has NaN values in key metrics where previous episode doesn't
        key_metrics = ['deliver_ai_rl_1', 'pure_reward_ai_rl_1']
        
        if second_last_episode is not None:
            for metric in key_metrics:
                if metric in df_sorted.columns:
                    if (pd.isna(last_episode[metric]) and not pd.isna(second_last_episode[metric])):
                        print(f"      Removing incomplete last episode {last_episode['episode']} due to NaN in {metric}")
                        return df_sorted.iloc[:-1]
        
        # If no obvious issues, return all episodes
        return df_sorted
    
    def _filter_by_ability_config(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter training data to match the expected baseline ability configuration.
        
        Args:
            df: DataFrame containing training data with speed columns
            
        Returns:
            Filtered DataFrame containing only data matching the baseline ability configuration (1.0, 1.0) for both agents
        """
        if len(df) == 0:
            return df
            
        # Baseline ability configuration: Agent1 (1.0, 1.0), Agent2 (1.0, 1.0)
        if all(col in df.columns for col in ['walking_speed_1', 'cutting_speed_1', 'walking_speed_2', 'cutting_speed_2']):
            filter_condition = (
                (abs(df['walking_speed_1'] - 1.0) < 0.01) &
                (abs(df['cutting_speed_1'] - 1.0) < 0.01) &
                (abs(df['walking_speed_2'] - 1.0) < 0.01) &
                (abs(df['cutting_speed_2'] - 1.0) < 0.01)
            )
            return df[filter_condition].copy()
        else:
            print(f"    Warning: Missing speed columns for ability filtering")
            return pd.DataFrame()
    
    def _print_available_abilities(self, df: pd.DataFrame):
        """Print the available ability configurations in the training data."""
        if len(df) == 0:
            print("    No data available")
            return
            
        # Check what ability configurations are present
        ability_configs = set()
        
        for _, row in df.head(10).iterrows():  # Check first few rows to get unique configs
            abilities = []
            
            # Collect agent 1 abilities
            if 'walking_speed_1' in row and 'cutting_speed_1' in row:
                if not pd.isna(row['walking_speed_1']) and not pd.isna(row['cutting_speed_1']):
                    abilities.append(f"Agent1: {row['walking_speed_1']:.1f}_{row['cutting_speed_1']:.1f}")
            
            # Collect agent 2 abilities
            if 'walking_speed_2' in row and 'cutting_speed_2' in row:
                if not pd.isna(row['walking_speed_2']) and not pd.isna(row['cutting_speed_2']):
                    abilities.append(f"Agent2: {row['walking_speed_2']:.1f}_{row['cutting_speed_2']:.1f}")
            
            if abilities:
                ability_configs.add(" | ".join(abilities))
        
        for config in sorted(ability_configs):
            print(f"    {config}")
    



class ColorGridPlotter:
    """Creates color grid plots for map analysis."""
    
    def __init__(self, map_names: List[str], game_type: str = 'classic'):
        self.map_names = map_names
        self.game_type = game_type
        
        # Set up matplotlib style
        plt.style.use('default')
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['grid.alpha'] = 0.3
        
    def calculate_performance_differences(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance differences relative to baseline map.
        
        Returns:
            Dictionary with structure: {map_name: difference}
        """
        performance_diffs = {}
        
        collision_suffix = '_collision' if 'collision' in self.game_type else ''
        
        # Find baseline performance (first map in the list)
        baseline_map_name = self.map_names[0]
        baseline_map_short = self._extract_map_short_name(baseline_map_name)
        baseline_condition = f"{baseline_map_short}{collision_suffix}"
        baseline_data = data[data['condition'] == baseline_condition]
            
        if len(baseline_data) == 0:
            print(f"    Warning: No baseline data found for {baseline_condition}")
            return performance_diffs
        
        baseline_deliveries = baseline_data['total_deliveries'].mean()
        
        # Calculate differences for each map
        for map_name in self.map_names:
            map_short = self._extract_map_short_name(map_name)
            condition_name = f"{map_short}{collision_suffix}"
            condition_data = data[data['condition'] == condition_name]
            
            if len(condition_data) == 0:
                performance_diffs[map_name] = 0.0
                continue
            
            current_deliveries = condition_data['total_deliveries'].mean()
            
            if map_name == baseline_map_name:
                # Baseline vs itself = 0
                performance_diffs[map_name] = 0.0
            else:
                # Difference from baseline
                performance_diffs[map_name] = current_deliveries - baseline_deliveries
        
        return performance_diffs
    
    def _extract_map_short_name(self, map_name: str) -> str:
        """Extract a short identifier from the full map name.
        
        Examples:
            'baseline_division_of_labor_large' -> 'baseline'
            'encouraged_division_of_labor_large' -> 'encouraged'
            'forced_division_of_labor_large' -> 'forced'
        """
        # Common pattern: {prefix}_division_of_labor_large
        if '_division_of_labor' in map_name:
            return map_name.split('_division_of_labor')[0]
        # Fallback: use first word
        return map_name.split('_')[0]
    
    def calculate_specialization_differences(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate specialization coefficient differences relative to baseline map.
        
        The specialization coefficient S = (N1C/(N1C + N1D)) - (N2C/(N2C + N2D))
        where:
        - N1C = times agent 1 chose cutting actions
        - N1D = times agent 1 chose delivery/other actions  
        - N2C = times agent 2 chose cutting actions
        - N2D = times agent 2 chose delivery/other actions
        
        Returns:
            Dictionary with structure: {map_name: specialization_difference}
        """
        specialization_diffs = {}
        
        collision_suffix = '_collision' if 'collision' in self.game_type else ''
        
        # Find baseline specialization (first map in the list)
        baseline_map_name = self.map_names[0]
        baseline_map_short = self._extract_map_short_name(baseline_map_name)
        baseline_condition = f"{baseline_map_short}{collision_suffix}"
        baseline_data = data[data['condition'] == baseline_condition]
            
        if len(baseline_data) == 0:
            print(f"    Warning: No baseline data found for specialization {baseline_condition}")
            return specialization_diffs
        
        baseline_spec = self._calculate_specialization_index(baseline_data)
        
        # Calculate differences for each map
        for map_name in self.map_names:
            map_short = self._extract_map_short_name(map_name)
            condition_name = f"{map_short}{collision_suffix}"
            condition_data = data[data['condition'] == condition_name]
            
            if len(condition_data) == 0:
                specialization_diffs[map_name] = 0.0
                continue
            
            current_spec = self._calculate_specialization_index(condition_data)
            
            if map_name == baseline_map_name:
                # Baseline vs itself = 0
                specialization_diffs[map_name] = 0.0
            else:
                # Difference from baseline
                specialization_diffs[map_name] = current_spec - baseline_spec
        
        return specialization_diffs
    
    def _calculate_specialization_index(self, data: pd.DataFrame) -> float:
        """Calculate the specialization index S for a given dataset.
        
        S = (N1C/(N1C + N1D)) - (N2C/(N2C + N2D))
        
        Args:
            data: DataFrame containing episode data
            
        Returns:
            Specialization index (float)
        """
        # Get cutting actions for each agent
        if 'cut_ai_rl_1' in data.columns and 'cut_ai_rl_2' in data.columns:
            n1c = data['cut_ai_rl_1'].sum()  # Agent 1 cutting actions
            n2c = data['cut_ai_rl_2'].sum()  # Agent 2 cutting actions
        else:
            print("    Warning: Missing cut columns for specialization calculation")
            return 0.0
        
        # Get delivery actions for each agent (representing "other" actions)
        if 'deliver_ai_rl_1' in data.columns and 'deliver_ai_rl_2' in data.columns:
            n1d = data['deliver_ai_rl_1'].sum()  # Agent 1 delivery actions
            n2d = data['deliver_ai_rl_2'].sum()  # Agent 2 delivery actions
        else:
            print("    Warning: Missing deliver columns for specialization calculation")
            return 0.0
        
        # Calculate totals for each agent
        t1 = n1c + n1d  # Total actions for agent 1
        t2 = n2c + n2d  # Total actions for agent 2
        
        # Avoid division by zero
        if t1 == 0 or t2 == 0:
            return 0.0
        
        # Calculate specialization index
        p1 = n1c / t1  # Fraction of cutting for agent 1
        p2 = n2c / t2  # Fraction of cutting for agent 2
        
        specialization = np.abs(p1 - p2)
        
        return specialization
    
    def create_color_grid_figure(self, data: pd.DataFrame, output_path: str = None) -> plt.Figure:
        """Create color grid plot showing map performance and specialization.
        
        Args:
            data: DataFrame with processed data
            output_path: Path to save figure
        """
        
        print("Creating color grid figure...")
        
        # Calculate performance and specialization differences
        performance_diffs = self.calculate_performance_differences(data)
        specialization_diffs = self.calculate_specialization_differences(data)
        
        if not performance_diffs:
            raise ValueError("No performance differences calculated")
        
        # Create figure with subplots: 1 row, 2 columns (performance + specialization)
        fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(self.map_names) * 1.5)))
        
        fig.suptitle('Map Comparison Analysis\n(Relative to Baseline Map)', 
                    fontsize=16, fontweight='bold')
        
        # Find the maximum absolute differences for color scaling
        # Performance differences
        all_perf_diffs = list(performance_diffs.values())
        
        if all_perf_diffs:
            max_abs_perf_diff = max(abs(d) for d in all_perf_diffs if d != 0)  # Exclude zeros (baseline)
        else:
            max_abs_perf_diff = 1.0
        
        # Specialization differences  
        all_spec_diffs = list(specialization_diffs.values())
        
        if all_spec_diffs:
            max_abs_spec_diff = max(abs(d) for d in all_spec_diffs if d != 0)  # Exclude zeros (baseline)
        else:
            max_abs_spec_diff = 0.1  # Default small value
        
        # Create color grids
        # Performance subplot (left)
        perf_ax = axes[0]
        self._create_single_color_grid(perf_ax, performance_diffs, 
                                     max_abs_perf_diff, 'Performance Difference (Deliveries)', 
                                     'Performance vs Baseline')
        
        # Specialization subplot (right)
        spec_ax = axes[1]
        self._create_single_color_grid(spec_ax, specialization_diffs, 
                                     max_abs_spec_diff, 'Specialization Difference', 
                                     'Specialization vs Baseline')
        
        # Add colorbars
        # Performance colorbar (below left plot)
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['red', 'white', 'blue']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('performance', colors_list, N=n_bins)
        
        # Create dummy mappable for colorbar
        import matplotlib.cm as cm
        norm = plt.Normalize(vmin=-1, vmax=1)
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        
        # Performance colorbar
        perf_cbar_ax = fig.add_axes([0.15, 0.02, 0.25, 0.03])  # [left, bottom, width, height] 
        perf_cbar = fig.colorbar(mappable, cax=perf_cbar_ax, orientation='horizontal')
        perf_cbar.set_label('Performance Difference (Deliveries)', fontsize=10, fontweight='bold')
        perf_cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        perf_cbar.set_ticklabels([f'{-max_abs_perf_diff:.1f}', f'{-max_abs_perf_diff/2:.1f}', '0', 
                                 f'{max_abs_perf_diff/2:.1f}', f'{max_abs_perf_diff:.1f}'])
        
        # Specialization colorbar (below right plot)
        spec_cbar_ax = fig.add_axes([0.6, 0.02, 0.25, 0.03])  # [left, bottom, width, height]
        spec_cbar = fig.colorbar(mappable, cax=spec_cbar_ax, orientation='horizontal')
        spec_cbar.set_label('Specialization Difference (Index)', fontsize=10, fontweight='bold')
        spec_cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        spec_cbar.set_ticklabels([f'{-max_abs_spec_diff:.2f}', f'{-max_abs_spec_diff/2:.2f}', '0', 
                                 f'{max_abs_spec_diff/2:.2f}', f'{max_abs_spec_diff:.2f}'])
        
        # Add explanation text
        fig.text(0.5, 0.08, 
                'Blue: Better/Higher than baseline | Red: Worse/Lower than baseline | White: Baseline (reference)',
                ha='center', fontsize=10, style='italic')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Save figure if output path provided
        if output_path:
            print(f"Saving color grid figure to: {output_path}")
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def _create_single_color_grid(self, ax, map_diffs: Dict[str, float], 
                                max_abs_diff: float, metric_label: str, title: str):
        """Create a single vertical color grid subplot.
        
        Args:
            ax: Matplotlib axis to plot on
            map_diffs: Dictionary of map names to difference values
            max_abs_diff: Maximum absolute difference for normalization
            metric_label: Label for the metric being plotted
            title: Title for the subplot
        """
        # Check if this is a specialization metric
        is_specialization = 'Specialization' in metric_label
        
        # Prepare data for the vertical grid
        grid_data = np.zeros((len(self.map_names), 1))  # Y rows, single column
        colors = np.zeros((len(self.map_names), 1))
        
        for i, map_name in enumerate(self.map_names):
            diff = map_diffs.get(map_name, 0.0)
            grid_data[i, 0] = diff
            
            if map_name == self.map_names[0]:  # Baseline map
                # Baseline is always 0 (white)
                colors[i, 0] = 0.0
            else:
                if is_specialization:
                    # Normalize for white-to-green mapping (0 to 1)
                    colors[i, 0] = diff / max_abs_diff if max_abs_diff > 0 else 0.0
                else:
                    # Normalize difference for color mapping (-1 to 1)
                    colors[i, 0] = diff / max_abs_diff if max_abs_diff > 0 else 0.0
        
        # Create the color grid
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.cm as cm
        
        if is_specialization:
            # Use white-to-green colormap for specialization
            cmap = cm.Greens
            vmin, vmax = 0, 1
        else:
            # Create custom colormap: red (-1) -> white (0) -> blue (1)
            colors_list = ['red', 'white', 'blue']
            n_bins = 100
            cmap = LinearSegmentedColormap.from_list('performance', colors_list, N=n_bins)
            vmin, vmax = -1, 1
        
        # Plot the vertical grid
        im = ax.imshow(colors, cmap=cmap, vmin=vmin, vmax=vmax, 
                      aspect='auto', interpolation='nearest')
        
        # Set labels
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_yticks(range(len(self.map_names)))
        
        # Create short labels for the y-axis
        short_labels = [self._extract_map_short_name(map_name) for map_name in self.map_names]
        ax.set_yticklabels(short_labels, fontsize=10)
        ax.set_xticks([])
        
        # Add text annotations with the actual values
        for i, map_name in enumerate(self.map_names):
            diff = map_diffs.get(map_name, 0.0)
            
            if is_specialization:
                # For specialization: white text on dark green, black text on light green/white
                normalized_value = abs(colors[i, 0])
                text_color = 'white' if normalized_value > 0.5 else 'black'
                text_val = f'{diff:.3f}'  # More decimal places for specialization
            else:
                # For performance: white text on dark colors, black text on light colors
                text_color = 'black' if abs(colors[i, 0]) < 0.5 else 'white'
                text_val = f'{diff:.1f}'   # Fewer decimal places for deliveries
            
            ax.text(0, i, text_val, ha='center', va='center', 
                   color=text_color, fontsize=9, fontweight='bold')
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Generate map configuration analysis color grid plots',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--episode_range',
        choices=['all', 'final', 'average', 'specific'],
        default='final',
        help='Episode selection method:\n'
             '  all: Use all episodes as individual points\n'
             '  final: Use final N episodes from each training\n'
             '  average: Average episodes per training, use training averages as points\n'
             '  specific: Use N episodes around a specific target episode'
    )
    
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=100,
        help='Number of episodes to use (default: 100). For final: last N episodes. For specific: N episodes total (N/2 before and N/2 after target)'
    )
    
    parser.add_argument(
        '--target_episode',
        type=int,
        default=None,
        help='Target episode index within each training (required when episode_range=specific). '
             'This refers to the Nth episode within each training session. For example, target_episode=10 '
             'means the 10th episode of each training, regardless of the actual episode numbers in the data.'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save output figures (default: {cluster_path}/data/samuel_lozano/cooked/grid_map_analysis_figures)'
    )
    
    parser.add_argument(
        '--filename_suffix',
        type=str,
        default='',
        help='Suffix to add to output filename'
    )
    
    parser.add_argument(
        '--study_name',
        type=str,
        default=None,
        help='Study name for specific study folders (optional)'
    )
    
    parser.add_argument(
        '--map_name_1',
        type=str,
        default='baseline_division_of_labor_large',
        help='First map name (legacy parameter - script now uses hardcoded map list)'
    )
    
    parser.add_argument(
        '--map_name_2',
        type=str,
        default='encouraged_division_of_labor_large',
        help='Second map name (legacy parameter - script now uses hardcoded map list)'
    )
    

    
    parser.add_argument(
        '--game_type',
        type=str,
        choices=['classic', 'classic_collision'],
        default='classic',
        help='Game type to analyze (default: classic)'
    )
    
    parser.add_argument(
        '--init_type',
        type=str,
        choices=['random_init', 'empty_init'],
        default='empty_init',
        help='Initialization type subdirectory to analyze (default: empty_init)'
    )
    
    parser.add_argument(
        '--synergy',
        type=float,
        default=0.0,
        help='Synergy scaling factor for reference-based reward shaping (default: 0.0)'
    )
    
    parser.add_argument(
        '--specialization',
        type=float,
        default=None,
        help='Which specialization lambda value to analyze (e.g., 0, 5.0). If not specified, auto-detects all available lambdas.'
    )
    
    parser.add_argument(
        '--cluster',
        type=str,
        default=None,
        help='Cluster name (optional, for reference only)'
    )
    
    return parser


def main():
    """Main function to run the cooperative analysis."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        # Check if synergy was explicitly provided
        synergy_provided = '--synergy' in sys.argv
        
        # Use specialization lambda directly
        specialization = args.specialization
        
        # Set up output directory based on cluster if not explicitly provided
        cluster = args.cluster if args.cluster else 'cuenca'
        config = AnalysisConfig()
        local_path = config.cluster_paths[cluster]
        
        if args.output_dir is None:
            output_dir_base = f"{local_path}/data/samuel_lozano/cooked/grid_map_analysis_figures"
        else:
            output_dir_base = args.output_dir
        
        # Initialize analyzer
        analyzer = CooperativeAnalyzer(
            study_name=args.study_name,
            map_name_1=args.map_name_1,
            map_name_2=args.map_name_2,
            init_type=args.init_type,
            synergy=args.synergy,
            synergy_provided=synergy_provided,
            specialization=specialization,
            game_type=args.game_type,
            cluster=cluster
        )
        
        # Load experimental data
        print("=" * 60)
        print("MAP CONFIGURATION ANALYSIS - COLOR GRID PLOTS")
        print("=" * 60)
        print(f"Cluster: {args.cluster if args.cluster else 'cuenca'}")
        print(f"Game type: {args.game_type}")
        print(f"Init type: {args.init_type}")
        print(f"Synergy scaling factor: {args.synergy}")
        print(f"Specialization: {args.specialization}")
        print(f"Map 1: {args.map_name_1}")
        print(f"Map 2: {args.map_name_2}")
        print(f"Hardcoded maps for analysis: {analyzer.map_names}")
        if args.study_name:
            print(f"Study: {args.study_name}")
        
        df_or_dict = analyzer.load_experimental_data()
        
        # Check if we got a dict (multiple combinations) or single DataFrame
        if isinstance(df_or_dict, dict):
            # Multiple synergy/specialization combinations - create separate figure for each
            print("\n" + "=" * 60)
            print(f"CREATING {len(df_or_dict)} SEPARATE FIGURES FOR EACH COMBINATION")
            print("=" * 60)
            
            for combination_key in sorted(df_or_dict.keys()):
                print(f"\nProcessing {combination_key}...")
                df = df_or_dict[combination_key]
                
                # Extract synergy and specialization from combination key
                # Format: "synergy_{value}__specialized_{value}"
                parts = combination_key.split('__')
                synergy_part = parts[0]  # e.g., "synergy_0.50"
                spec_part = parts[1]      # e.g., "specialized_0.25"
                
                # Prepare data based on episode selection
                processed_df = analyzer.prepare_episode_data(
                    df, 
                    episode_selection=args.episode_range,
                    num_episodes=args.num_episodes,
                    target_episode=args.target_episode
                )
                
                # Create output directory
                output_dir = Path(output_dir_base)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate filename with synergy and specialization
                filename_parts = ["grid_map_analysis_maps"]
                filename_parts.append(args.init_type)
                
                # Add synergy part
                filename_parts.append(synergy_part)
                
                # Add specialization part
                filename_parts.append(spec_part)
                
                # Add map info to filename
                filename_parts.append("multiple_maps")
                
                if args.study_name:
                    filename_parts.append(args.study_name)
                
                filename_parts.append(args.episode_range)
                
                if args.episode_range == 'final':
                    filename_parts.append(f"{args.num_episodes}eps")
                
                if args.episode_range == 'specific':
                    filename_parts.append(f"ep{args.target_episode}_{args.num_episodes}eps")
                
                if args.filename_suffix:
                    filename_parts.append(args.filename_suffix)
                
                filename = "_".join(filename_parts) + ".png"
                output_path = output_dir / filename
                
                # Create color grid plots for this combination
                plotter = ColorGridPlotter(analyzer.map_names, analyzer.game_type)
                fig = plotter.create_color_grid_figure(processed_df, str(output_path))
                
                # Display results
                print(f"\nFigure for {combination_key} completed!")
                print(f"  Saved to: {output_path}")
                print(f"  Total episodes: {len(processed_df)}")
                print(f"  Conditions: {sorted(processed_df['condition'].unique())}")
                
                plt.close(fig)  # Close figure to free memory
            
            print("\n" + "=" * 60)
            print(f"ALL {len(df_or_dict)} FIGURES COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
        else:
            # Single specialization mode - process normally
            df = df_or_dict
            
            # Prepare data based on episode selection
            processed_df = analyzer.prepare_episode_data(
                df, 
                episode_selection=args.episode_range,
                num_episodes=args.num_episodes,
                target_episode=args.target_episode
            )
            
            # Create output directory
            output_dir = Path(output_dir_base)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with all relevant parameters
            filename_parts = ["grid_map_analysis_maps"]
            
            # Add init type
            filename_parts.append(args.init_type)
            
            # Add synergy if explicitly provided (even if 0)
            if analyzer.synergy_provided:
                filename_parts.append(f"synergy_{args.synergy}" if args.synergy != 0 else "synergy_0")
            
            # Add specialization information
            if args.specialization is not None and args.specialization != 'both':
                # When analyzing specific mode, add it to filename
                filename_parts.append(f"specialized_{args.specialization:.2f}" if args.specialization != 0 else "specialized_0")
            
            # Add map info to filename  
            filename_parts.append("multiple_maps")
            
            # Add study name if specified
            if args.study_name:
                filename_parts.append(args.study_name)
            
            # Add episode selection type
            filename_parts.append(args.episode_range)
            
            # Add number of episodes if using 'final' or 'specific' selection
            if args.episode_range == 'final':
                filename_parts.append(f"{args.num_episodes}eps")
            
            if args.episode_range == 'specific':
                filename_parts.append(f"ep{args.target_episode}_{args.num_episodes}eps")
            
            # Add custom suffix if provided
            if args.filename_suffix:
                filename_parts.append(args.filename_suffix)
            
            filename = "_".join(filename_parts) + ".png"
            output_path = output_dir / filename
            
            # Create color grid plots
            plotter = ColorGridPlotter(analyzer.map_names, analyzer.game_type)
            fig = plotter.create_color_grid_figure(processed_df, str(output_path))
            
            # Display results
            print(f"\nAnalysis completed successfully!")
            print(f"Figure saved to: {output_path}")
            print(f"Data summary:")
            print(f"  Init type: {args.init_type}")
            print(f"  Total episodes: {len(processed_df)}")
            print(f"  Conditions: {sorted(processed_df['condition'].unique())}")
            print(f"  Episode selection: {args.episode_range}")
            if args.study_name:
                print(f"  Study name: {args.study_name}")
            
            if args.episode_range == 'final':
                print(f"  Final episodes used: {args.num_episodes}")
            
            if args.episode_range == 'specific':
                print(f"  Episodes around episode {args.target_episode}: {args.num_episodes} total")
            
            # Note: Skipping plt.show() since we're running in headless mode
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()