#!/usr/bin/env python3
"""
Full Grid Cooperative Analysis Figure Generator

This script creates 2D color grid plots combining both map configurations (Y-axis) 
and agent ability configurations (X-axis). The visualization shows how different 
combinations perform across performance metrics.

Map configurations tested (Y-axis):
- baseline_division_of_labor_large (baseline)
- semiencouraged_division_of_labor_large
- 1-semiencouraged_division_of_labor_large
- 2-semiencouraged_division_of_labor_large
- encouraged_division_of_labor_large
- 1-encouraged_division_of_labor_large
- 2-encouraged_division_of_labor_large
- 3-encouraged_division_of_labor_large

Agent ability configurations tested (X-axis):
- X=1.0: Agent 1 (1.0, 1.0), Agent 2 (1.0, 1.0) - baseline abilities
- X=0.8: Agent 1 (0.8, 1.0), Agent 2 (1.0, 0.8)
- X=0.6: Agent 1 (0.6, 1.0), Agent 2 (1.0, 0.6)
- X=0.4: Agent 1 (0.4, 1.0), Agent 2 (1.0, 0.4)
- X=0.2: Agent 1 (0.2, 1.0), Agent 2 (1.0, 0.2)

Generates 6 color grid plots with different comparison baselines:
1. Global baseline comparison (baseline map + X=1.0 abilities)
2. Row-wise comparison (each map vs X=1.0 abilities for that map)
3. Column-wise comparison (baseline map vs each ability configuration)

Color coding:
- White: Baseline condition (difference=0)
- Blue: Better performance than baseline
- Red: Worse performance than baseline

Usage:
    python grid_full_cooperative_analysis.py [options]

Examples:
    # Default analysis
    nohup python grid_full_cooperative_analysis.py --episode_range final --num_episodes 100 > grid_full_analysis.log 2>&1 &
    
    # Analyze specific lambda data
    nohup python grid_full_cooperative_analysis.py --episode_range final --specialization 0 --num_episodes 100 > full_grid_lambda_0.log 2>&1 &
    
    # Analyze empty_init data
    nohup python grid_full_cooperative_analysis.py --episode_range final --init_type empty_init --num_episodes 100 > full_grid_empty_init.log 2>&1 &

    # Analyze specific episode range around episode 500
    nohup python grid_full_cooperative_analysis.py --episode_range specific --init_type empty_init --num_episodes 20 --cluster brigit --specialization 0.05 --synergy 0.40 --target_episode 100 > full_grid_empty_init.log 2>&1 
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
        
        # Define hardcoded map list for 2D grid analysis (Y-axis)
        self.map_names = [
            #'baseline_division_of_labor_large',
            'encouraged_division_of_labor_large',
        ]
        
        # Define ability configurations for 2D grid analysis (X-axis) manually,
        # analogous to how map names are manually introduced for Y-axis.
        # Each entry is explicit: (id, label, (walk1, cut1, walk2, cut2))
        ### SPEEDS
        self.ability_config_definitions = [
            ('1.0', '1.0', (1.0, 1.0, 1.0, 1.0)),
            #('0.9', '0.9', (0.9, 1.0, 1.0, 0.3)),
            #('0.8', '0.8', (0.8, 1.0, 1.0, 0.3)),
            #('0.7', '0.7', (0.7, 1.0, 1.0, 0.3)),
            ('0.6', '0.6', (0.6, 1.0, 1.0, 0.3)),
            ('0.58', '0.58', (0.58, 1.0, 1.0, 0.3)),
            ('0.56', '0.56', (0.56, 1.0, 1.0, 0.3)),
            ('0.54', '0.54', (0.54, 1.0, 1.0, 0.3)),
            ('0.52', '0.52', (0.52, 1.0, 1.0, 0.3)),
            ('0.5', '0.5', (0.5, 1.0, 1.0, 0.3)),
            ('0.48', '0.48', (0.48, 1.0, 1.0, 0.3)),
            ('0.46', '0.46', (0.46, 1.0, 1.0, 0.3)),
            ('0.44', '0.44', (0.44, 1.0, 1.0, 0.3)),
            ('0.42', '0.42', (0.42, 1.0, 1.0, 0.3)),
            ('0.4', '0.4', (0.4, 1.0, 1.0, 0.3)),
            #('0.3', '0.3', (0.3, 1.0, 1.0, 0.3)),
            #('0.2', '0.2', (0.2, 1.0, 1.0, 0.3)),
            #('0.1', '0.1', (0.1, 1.0, 1.0, 0.3)),
        ]

        self.ability_configs = [config_id for config_id, _, _ in self.ability_config_definitions]
        self.ability_config_labels = {config_id: label for config_id, label, _ in self.ability_config_definitions}
        self.ability_config_speeds = {config_id: speeds for config_id, _, speeds in self.ability_config_definitions}
        
        # Define experimental conditions mapping for all map/ability combinations
        self.base_condition_mapping = {}
        for map_name in self.map_names:
            map_short = self._extract_map_short_name(map_name)
            
            # Add symmetric configurations
            for ability_config in self.ability_configs:
                collision_suffix = '_collision' if 'collision' in game_type else ''
                condition_key = (map_name, game_type, ability_config)
                condition_name = f'{map_short}_{ability_config}{collision_suffix}'
                self.base_condition_mapping[condition_key] = condition_name
        
        # Color palette will be generated dynamically based on performance
        # White for X=1.0, blue for better performance, red for worse
        self.color_palette = {}
        
        # Define baseline conditions for different comparison types
        self.baseline_x = 1.0  # Baseline ability configuration
        self.baseline_map = self.map_names[0]  # First map as baseline
        
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
            'encouraged_division_of_labor_large_random_positions' -> 'encouraged_random_positions'
            'forced_division_of_labor_large' -> 'forced'
        """
        short_name = map_name
        if '_division_of_labor_large' in short_name:
            short_name = short_name.replace('_division_of_labor_large', '')
        elif '_division_of_labor' in short_name:
            short_name = short_name.replace('_division_of_labor', '')

        short_name = short_name.strip('_')
        return short_name if short_name else map_name

    def _format_ability_config(self, ability_config: str) -> str:
        """Format an ability config for logging and labels."""
        return self.ability_config_labels.get(ability_config, str(ability_config))
    
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
                for (map_name, game_type, ability_config), condition_name in self.base_condition_mapping.items():
                    # Keep original condition name (no suffix)
                    print(f"Processing condition: {condition_name} (map={map_name}, ability={self._format_ability_config(ability_config)})")
                    
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
                                raw_dir = self._build_data_path('data', 'samuel_lozano', 'cooked', self.study_name, game_type, self.init_type, f'map_{map_name}', synergy_folder, spec_folder)
                            else:
                                raw_dir = self._build_data_path('data', 'samuel_lozano', 'cooked', self.study_name, game_type, self.init_type, f'map_{map_name}', synergy_folder)
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
                        # Filter to the expected ability configuration for this ability config
                        df_filtered = self._filter_by_ability_config(df, ability_config)
                        
                        if len(df_filtered) == 0:
                            print(f"  Warning: No data matching ability {self._format_ability_config(ability_config)} in {condition_name}")
                            continue
                        
                        # Create combined metrics and attach metadata
                        self._create_combined_metrics(df_filtered)
                        df_filtered['condition'] = condition_name
                        df_filtered['map_name'] = map_name
                        df_filtered['game_type_clean'] = game_type
                        df_filtered['ability_config'] = [ability_config] * len(df_filtered)
                        df_filtered['ability_label'] = self._format_ability_config(ability_config)
                        df_filtered['x_value'] = [ability_config] * len(df_filtered)
                        if spec_mode:
                            df_filtered['specialization'] = spec_mode
                        
                        condition_data.append(df_filtered)
                        print(f"  OK: {len(df_filtered)} episodes loaded for {condition_name}")
                        
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
    
    def _filter_by_ability_config(self, df: pd.DataFrame, ability_config_id: str) -> pd.DataFrame:
        """Filter training data to match the expected ability configuration.
        
        Args:
            df: DataFrame containing training data with speed columns
            ability_config_id: Ability config identifier defined in ability_config_definitions
            
        Returns:
            Filtered DataFrame containing only data matching the ability configuration
        """
        if len(df) == 0:
            return df
            
        if not all(col in df.columns for col in ['walking_speed_1', 'cutting_speed_1', 'walking_speed_2', 'cutting_speed_2']):
            print(f"    Warning: Missing speed columns for ability filtering")
            return pd.DataFrame()

        if ability_config_id not in self.ability_config_speeds:
            print(f"    Warning: Unknown ability configuration '{ability_config_id}'")
            return pd.DataFrame()

        walk1, cut1, walk2, cut2 = self.ability_config_speeds[ability_config_id]
        filter_condition = (
            (abs(df['walking_speed_1'] - walk1) < 0.01) &
            (abs(df['cutting_speed_1'] - cut1) < 0.01) &
            (abs(df['walking_speed_2'] - walk2) < 0.01) &
            (abs(df['cutting_speed_2'] - cut2) < 0.01)
        )
        
        return df[filter_condition].copy()
    
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
    """Creates 2D color grid plots for map × ability configuration analysis."""
    
    def __init__(self, map_names: List[str], ability_configs: List[str], game_type: str = 'classic',
                 ability_config_labels: Optional[Dict[str, str]] = None):
        self.map_names = map_names
        self.ability_configs = ability_configs
        self.game_type = game_type
        self.ability_config_labels = ability_config_labels or {}
        
        # Set up matplotlib style
        plt.style.use('default')
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['grid.alpha'] = 0.3
        
    def calculate_performance_differences_global(self, data: pd.DataFrame) -> Dict[str, Dict[float, float]]:
        """Calculate performance differences relative to global baseline (baseline map + X=1.0).
        
        Returns:
            Dictionary with structure: {map_short: {x_value: difference}}
        """
        performance_diffs = {}
        
        # Find global baseline: baseline map + X=1.0
        baseline_map_short = self._extract_map_short_name(self.map_names[0])
        collision_suffix = '_collision' if 'collision' in self.game_type else ''
        baseline_condition = f"{baseline_map_short}_1.0{collision_suffix}"
        baseline_data = data[data['condition'] == baseline_condition]
        
        if len(baseline_data) == 0:
            print(f"Warning: No global baseline data found for condition {baseline_condition}. Using 0.0 baseline.")
            baseline_deliveries = 0.0
        else:
            baseline_deliveries = baseline_data['total_deliveries'].mean()
            print(f"Global baseline ({baseline_condition}): {baseline_deliveries:.2f} deliveries")
        
        # Calculate differences for all map/ability combinations
        for map_name in self.map_names:
            map_short = self._extract_map_short_name(map_name)
            performance_diffs[map_short] = {}
            
            for ability_config in self.ability_configs:
                condition_name = self._ability_condition_name(map_short, ability_config, collision_suffix)
                condition_data = data[data['condition'] == condition_name]
                
                if len(condition_data) == 0:
                    print(f"Warning: No data found for condition {condition_name}")
                    performance_diffs[map_short][ability_config] = 0.0
                    continue
                    
                condition_deliveries = condition_data['total_deliveries'].mean()
                difference = condition_deliveries - baseline_deliveries
                performance_diffs[map_short][ability_config] = difference
        
        return performance_diffs
    
    def calculate_performance_differences_row(self, data: pd.DataFrame) -> Dict[str, Dict[float, float]]:
        """Calculate performance differences relative to each map's X=1.0 baseline (row-wise).
        
        Returns:
            Dictionary with structure: {map_short: {x_value: difference}}
        """
        performance_diffs = {}
        collision_suffix = '_collision' if 'collision' in self.game_type else ''
        
        # Calculate differences for each map relative to its own X=1.0 baseline
        for map_name in self.map_names:
            map_short = self._extract_map_short_name(map_name)
            
            # Find row baseline: this map + X=1.0
            baseline_condition = f"{map_short}_1.0{collision_suffix}"
            baseline_data = data[data['condition'] == baseline_condition]
            
            if len(baseline_data) == 0:
                print(f"Warning: No row baseline data found for condition {baseline_condition}. Using 0.0 baseline.")
                baseline_deliveries = 0.0
            else:
                baseline_deliveries = baseline_data['total_deliveries'].mean()

            performance_diffs[map_short] = {}
            
            for ability_config in self.ability_configs:
                condition_name = self._ability_condition_name(map_short, ability_config, collision_suffix)
                condition_data = data[data['condition'] == condition_name]
                
                if len(condition_data) == 0:
                    print(f"Warning: No data found for condition {condition_name}")
                    performance_diffs[map_short][ability_config] = 0.0
                    continue
                    
                condition_deliveries = condition_data['total_deliveries'].mean()
                difference = condition_deliveries - baseline_deliveries
                performance_diffs[map_short][ability_config] = difference
        
        return performance_diffs
    
    def calculate_performance_differences_column(self, data: pd.DataFrame) -> Dict[str, Dict[float, float]]:
        """Calculate performance differences relative to baseline map for each ability (column-wise).
        
        Returns:
            Dictionary with structure: {map_short: {x_value: difference}}
        """
        performance_diffs = {}
        collision_suffix = '_collision' if 'collision' in self.game_type else ''
        
        # Calculate differences for each ability configuration relative to baseline map
        baseline_map_short = self._extract_map_short_name(self.map_names[0])
        
        for ability_config in self.ability_configs:
            # Find column baseline: baseline map + this X value
            baseline_condition = self._ability_condition_name(baseline_map_short, ability_config, collision_suffix)
            baseline_data = data[data['condition'] == baseline_condition]
            
            if len(baseline_data) == 0:
                print(f"Warning: No column baseline data found for condition {baseline_condition}")
                baseline_deliveries = 0.0
            else:
                baseline_deliveries = baseline_data['total_deliveries'].mean()
            
            # Calculate differences for all maps at this ability level
            for map_name in self.map_names:
                map_short = self._extract_map_short_name(map_name)
                
                if map_short not in performance_diffs:
                    performance_diffs[map_short] = {}
                
                condition_name = self._ability_condition_name(map_short, ability_config, collision_suffix)
                condition_data = data[data['condition'] == condition_name]
                
                if len(condition_data) == 0:
                    print(f"Warning: No data found for condition {condition_name}")
                    performance_diffs[map_short][ability_config] = 0.0
                    continue
                    
                condition_deliveries = condition_data['total_deliveries'].mean()
                difference = condition_deliveries - baseline_deliveries
                performance_diffs[map_short][ability_config] = difference
        
        return performance_diffs
    
    def _extract_map_short_name(self, map_name: str) -> str:
        """Extract a short identifier from the full map name."""
        short_name = map_name
        if '_division_of_labor_large' in short_name:
            short_name = short_name.replace('_division_of_labor_large', '')
        elif '_division_of_labor' in short_name:
            short_name = short_name.replace('_division_of_labor', '')

        short_name = short_name.strip('_')
        return short_name if short_name else map_name

    def _ability_condition_name(self, map_short: str, ability_config: str, collision_suffix: str) -> str:
        return f"{map_short}_{ability_config}{collision_suffix}"

    def _ability_label(self, ability_config: str) -> str:
        return self.ability_config_labels.get(ability_config, str(ability_config))
    
    def calculate_specialization_differences_global(self, data: pd.DataFrame) -> Dict[str, Dict[float, float]]:
        """Calculate specialization differences relative to global baseline (baseline map + X=1.0).
        
        Returns:
            Dictionary with structure: {map_short: {x_value: difference}}
        """
        specialization_diffs = {}
        
        # Find global baseline: baseline map + X=1.0
        baseline_map_short = self._extract_map_short_name(self.map_names[0])
        collision_suffix = '_collision' if 'collision' in self.game_type else ''
        baseline_condition = f"{baseline_map_short}_1.0{collision_suffix}"
        baseline_data = data[data['condition'] == baseline_condition]
        
        if len(baseline_data) == 0:
            print(f"Warning: No global baseline data found for condition {baseline_condition}. Using 0.0 baseline.")
            baseline_specialization = 0.0
        else:
            baseline_specialization = self._calculate_specialization_index(baseline_data)
            print(f"Global baseline ({baseline_condition}): {baseline_specialization:.3f} specialization")
        
        # Calculate differences for all map/ability combinations
        for map_name in self.map_names:
            map_short = self._extract_map_short_name(map_name)
            specialization_diffs[map_short] = {}
            
            for ability_config in self.ability_configs:
                condition_name = self._ability_condition_name(map_short, ability_config, collision_suffix)
                condition_data = data[data['condition'] == condition_name]
                
                if len(condition_data) == 0:
                    print(f"Warning: No data found for condition {condition_name}")
                    specialization_diffs[map_short][ability_config] = 0.0
                    continue
                    
                condition_specialization = self._calculate_specialization_index(condition_data)
                difference = condition_specialization - baseline_specialization
                specialization_diffs[map_short][ability_config] = difference
        
        return specialization_diffs
    
    def calculate_specialization_differences_row(self, data: pd.DataFrame) -> Dict[str, Dict[float, float]]:
        """Calculate specialization differences relative to each map's X=1.0 baseline (row-wise).
        
        Returns:
            Dictionary with structure: {map_short: {x_value: difference}}
        """
        specialization_diffs = {}
        collision_suffix = '_collision' if 'collision' in self.game_type else ''
        
        # Calculate differences for each map relative to its own X=1.0 baseline
        for map_name in self.map_names:
            map_short = self._extract_map_short_name(map_name)
            
            # Find row baseline: this map + X=1.0
            baseline_condition = f"{map_short}_1.0{collision_suffix}"
            baseline_data = data[data['condition'] == baseline_condition]
            
            if len(baseline_data) == 0:
                print(f"Warning: No row baseline data found for condition {baseline_condition}. Using 0.0 baseline.")
                baseline_specialization = 0.0
            else:
                baseline_specialization = self._calculate_specialization_index(baseline_data)

            specialization_diffs[map_short] = {}
            
            for ability_config in self.ability_configs:
                condition_name = self._ability_condition_name(map_short, ability_config, collision_suffix)
                condition_data = data[data['condition'] == condition_name]
                
                if len(condition_data) == 0:
                    print(f"Warning: No data found for condition {condition_name}")
                    specialization_diffs[map_short][ability_config] = 0.0
                    continue
                    
                condition_specialization = self._calculate_specialization_index(condition_data)
                difference = condition_specialization - baseline_specialization
                specialization_diffs[map_short][ability_config] = difference
        
        return specialization_diffs
    
    def calculate_specialization_differences_column(self, data: pd.DataFrame) -> Dict[str, Dict[float, float]]:
        """Calculate specialization differences relative to baseline map for each ability (column-wise).
        
        Returns:
            Dictionary with structure: {map_short: {x_value: difference}}
        """
        specialization_diffs = {}
        collision_suffix = '_collision' if 'collision' in self.game_type else ''
        
        # Calculate differences for each ability configuration relative to baseline map
        baseline_map_short = self._extract_map_short_name(self.map_names[0])
        
        for ability_config in self.ability_configs:
            # Find column baseline: baseline map + this X value
            baseline_condition = self._ability_condition_name(baseline_map_short, ability_config, collision_suffix)
            baseline_data = data[data['condition'] == baseline_condition]
            
            if len(baseline_data) == 0:
                print(f"Warning: No column baseline data found for condition {baseline_condition}")
                baseline_specialization = 0.0
            else:
                baseline_specialization = self._calculate_specialization_index(baseline_data)
            
            # Calculate differences for all maps at this ability level
            for map_name in self.map_names:
                map_short = self._extract_map_short_name(map_name)
                
                if map_short not in specialization_diffs:
                    specialization_diffs[map_short] = {}
                
                condition_name = self._ability_condition_name(map_short, ability_config, collision_suffix)
                condition_data = data[data['condition'] == condition_name]
                
                if len(condition_data) == 0:
                    print(f"Warning: No data found for condition {condition_name}")
                    specialization_diffs[map_short][ability_config] = 0.0
                    continue
                    
                condition_specialization = self._calculate_specialization_index(condition_data)
                difference = condition_specialization - baseline_specialization
                specialization_diffs[map_short][ability_config] = difference
        
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

    def _calculate_action_differentiation_index(self, data: pd.DataFrame) -> float:
        """Calculate the action differentiation index AD for a given dataset.

        AD is the total variation distance between the two agents' action-type
        distributions over five action types: deliver, cut, salad, plate, raw_food.

        AD = (1/2) * Σ_a |p1_a - p2_a|

        where p_agent_a = count_a / (sum of all five action counts for that agent).
        AD ∈ [0, 1]: 0 means identical distributions, 1 means fully non-overlapping.

        Args:
            data: DataFrame containing episode data

        Returns:
            Action differentiation index (float)
        """
        action_types = ['deliver', 'cut', 'salad', 'plate', 'raw_food']
        counts_1 = []
        counts_2 = []

        for action in action_types:
            col_1 = f'{action}_ai_rl_1'
            col_2 = f'{action}_ai_rl_2'
            if col_1 in data.columns and col_2 in data.columns:
                counts_1.append(data[col_1].sum())
                counts_2.append(data[col_2].sum())
            else:
                print(f"    Warning: Missing columns {col_1}/{col_2} for action differentiation")
                counts_1.append(0.0)
                counts_2.append(0.0)

        t1 = sum(counts_1)
        t2 = sum(counts_2)

        if t1 == 0 or t2 == 0:
            return 0.0

        p1 = [c / t1 for c in counts_1]
        p2 = [c / t2 for c in counts_2]

        ad = 0.5 * sum(abs(p1[i] - p2[i]) for i in range(len(action_types)))
        return ad

    def calculate_action_differentiation_differences_global(self, data: pd.DataFrame) -> Dict[str, Dict[float, float]]:
        """Calculate action differentiation differences relative to global baseline (baseline map + X=1.0).

        Returns:
            Dictionary with structure: {map_short: {x_value: difference}}
        """
        ad_diffs = {}
        baseline_map_short = self._extract_map_short_name(self.map_names[0])
        collision_suffix = '_collision' if 'collision' in self.game_type else ''
        baseline_condition = f"{baseline_map_short}_1.0{collision_suffix}"
        baseline_data = data[data['condition'] == baseline_condition]

        if len(baseline_data) == 0:
            print(f"Warning: No global baseline data found for condition {baseline_condition}. Using 0.0 baseline.")
            baseline_ad = 0.0
        else:
            baseline_ad = self._calculate_action_differentiation_index(baseline_data)
            print(f"Global baseline ({baseline_condition}): {baseline_ad:.3f} action differentiation")

        for map_name in self.map_names:
            map_short = self._extract_map_short_name(map_name)
            ad_diffs[map_short] = {}
            for ability_config in self.ability_configs:
                condition_name = self._ability_condition_name(map_short, ability_config, collision_suffix)
                condition_data = data[data['condition'] == condition_name]
                if len(condition_data) == 0:
                    print(f"Warning: No data found for condition {condition_name}")
                    ad_diffs[map_short][ability_config] = 0.0
                    continue
                ad_diffs[map_short][ability_config] = self._calculate_action_differentiation_index(condition_data) - baseline_ad

        return ad_diffs

    def calculate_action_differentiation_differences_row(self, data: pd.DataFrame) -> Dict[str, Dict[float, float]]:
        """Calculate action differentiation differences relative to each map's X=1.0 baseline (row-wise).

        Returns:
            Dictionary with structure: {map_short: {x_value: difference}}
        """
        ad_diffs = {}
        collision_suffix = '_collision' if 'collision' in self.game_type else ''

        for map_name in self.map_names:
            map_short = self._extract_map_short_name(map_name)
            baseline_condition = f"{map_short}_1.0{collision_suffix}"
            baseline_data = data[data['condition'] == baseline_condition]

            if len(baseline_data) == 0:
                print(f"Warning: No row baseline data found for condition {baseline_condition}. Using 0.0 baseline.")
                baseline_ad = 0.0
            else:
                baseline_ad = self._calculate_action_differentiation_index(baseline_data)

            ad_diffs[map_short] = {}

            for ability_config in self.ability_configs:
                condition_name = self._ability_condition_name(map_short, ability_config, collision_suffix)
                condition_data = data[data['condition'] == condition_name]
                if len(condition_data) == 0:
                    print(f"Warning: No data found for condition {condition_name}")
                    ad_diffs[map_short][ability_config] = 0.0
                    continue
                ad_diffs[map_short][ability_config] = self._calculate_action_differentiation_index(condition_data) - baseline_ad

        return ad_diffs

    def calculate_action_differentiation_differences_column(self, data: pd.DataFrame) -> Dict[str, Dict[float, float]]:
        """Calculate action differentiation differences relative to baseline map for each ability (column-wise).

        Returns:
            Dictionary with structure: {map_short: {x_value: difference}}
        """
        ad_diffs = {}
        collision_suffix = '_collision' if 'collision' in self.game_type else ''
        baseline_map_short = self._extract_map_short_name(self.map_names[0])

        for ability_config in self.ability_configs:
            baseline_condition = self._ability_condition_name(baseline_map_short, ability_config, collision_suffix)
            baseline_data = data[data['condition'] == baseline_condition]

            if len(baseline_data) == 0:
                print(f"Warning: No column baseline data found for condition {baseline_condition}")
                baseline_ad = 0.0
            else:
                baseline_ad = self._calculate_action_differentiation_index(baseline_data)

            for map_name in self.map_names:
                map_short = self._extract_map_short_name(map_name)
                if map_short not in ad_diffs:
                    ad_diffs[map_short] = {}
                condition_name = self._ability_condition_name(map_short, ability_config, collision_suffix)
                condition_data = data[data['condition'] == condition_name]
                if len(condition_data) == 0:
                    print(f"Warning: No data found for condition {condition_name}")
                    ad_diffs[map_short][ability_config] = 0.0
                    continue
                ad_diffs[map_short][ability_config] = self._calculate_action_differentiation_index(condition_data) - baseline_ad

        return ad_diffs
    
    def create_color_grid_figure(self, data: pd.DataFrame, comparison_type: str = 'global', output_path: str = None) -> plt.Figure:
        """Create a 2D color grid figure showing map × ability analysis.
        
        Args:
            data: DataFrame with all experimental data
            comparison_type: Type of comparison ('global', 'row', 'column')
            output_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Calculate performance, specialization, and action differentiation differences
        if comparison_type == 'global':
            perf_diffs = self.calculate_performance_differences_global(data)
            spec_diffs = self.calculate_specialization_differences_global(data)
            ad_diffs = self.calculate_action_differentiation_differences_global(data)
            subtitle = "vs Global Baseline (baseline map + X=1.0)"
        elif comparison_type == 'row':
            perf_diffs = self.calculate_performance_differences_row(data)
            spec_diffs = self.calculate_specialization_differences_row(data)
            ad_diffs = self.calculate_action_differentiation_differences_row(data)
            subtitle = "vs Row Baseline (each map + X=1.0)"
        elif comparison_type == 'column':
            perf_diffs = self.calculate_performance_differences_column(data)
            spec_diffs = self.calculate_specialization_differences_column(data)
            ad_diffs = self.calculate_action_differentiation_differences_column(data)
            subtitle = "vs Column Baseline (baseline map + each X)"
        else:
            raise ValueError(f"Unknown comparison_type: {comparison_type}")
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f'Map × Ability Configuration Analysis\n{subtitle}', 
                    fontsize=16, fontweight='bold')
        
        # Create 2D grids for plotting
        perf_grid = self._create_2d_grid(perf_diffs)
        spec_grid = self._create_2d_grid(spec_diffs)
        ad_grid = self._create_2d_grid(ad_diffs)
        
        # Plot performance differences
        self._plot_2d_grid(ax1, perf_grid, 'Total Deliveries Difference', 
                          f'Performance Differences ({comparison_type.title()})')
        
        # Plot specialization differences  
        self._plot_2d_grid(ax2, spec_grid, 'Specialization Index Difference',
                          f'Specialization Differences ({comparison_type.title()})')

        # Plot action differentiation differences
        self._plot_2d_grid(ax3, ad_grid, 'Action Differentiation Difference',
                          f'Action Differentiation ({comparison_type.title()})')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved color grid figure to: {output_path}")
        
        return fig
    
    def _create_2d_grid(self, diffs_dict: Dict[str, Dict[float, float]]) -> np.ndarray:
        """Create a 2D numpy array from the differences dictionary.
        
        Args:
            diffs_dict: Dictionary with structure {map_short: {x_value: difference}}
            
        Returns:
            2D numpy array with maps on rows and abilities on columns
        """
        n_maps = len(self.map_names)
        n_abilities = len(self.ability_configs)
        grid = np.zeros((n_maps, n_abilities))
        
        for i, map_name in enumerate(self.map_names):
            map_short = self._extract_map_short_name(map_name)
            if map_short in diffs_dict:
                for j, ability_config in enumerate(self.ability_configs):
                    if ability_config in diffs_dict[map_short]:
                        grid[i, j] = diffs_dict[map_short][ability_config]
        
        return grid
    
    def _plot_2d_grid(self, ax, grid: np.ndarray, metric_label: str, title: str):
        """Plot a 2D color grid on the given axis.
        
        Args:
            ax: matplotlib axis
            grid: 2D numpy array to plot
            metric_label: Label for the colorbar
            title: Title for the subplot
        """
        # Determine colormap based on metric type
        is_specialization = 'Specialization' in metric_label
        is_action_diff = 'Action Differentiation' in metric_label

        max_abs_diff = np.max(np.abs(grid))
        if is_specialization or is_action_diff:
            # Orange-white-green: negative = less specialised/differentiated, positive = more
            cmap = plt.cm.RdYlGn
        else:
            # Red-white-blue: negative = worse performance, positive = better
            cmap = plt.cm.RdBu_r
        vmin, vmax = -max_abs_diff, max_abs_diff
        
        # Plot the grid
        im = ax.imshow(grid, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # Set ticks and labels
        ax.set_xticks(range(len(self.ability_configs)))
        ax.set_xticklabels([self._ability_label(ability_config) for ability_config in self.ability_configs])
        ax.set_yticks(range(len(self.map_names)))
        ax.set_yticklabels([self._extract_map_short_name(name) for name in self.map_names])
        
        # Labels and title
        ax.set_xlabel('Ability Configuration (X)', fontsize=12)
        ax.set_ylabel('Map Configuration', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(metric_label, fontsize=10)
        
        # Add text annotations
        for i in range(len(self.map_names)):
            for j in range(len(self.ability_configs)):
                value = grid[i, j]
                
                normalized_value = abs(value) / max_abs_diff if max_abs_diff > 0 else 0
                color = 'white' if normalized_value > 0.5 else 'black'
                if is_specialization or is_action_diff:
                    text_val = f'{value:.3f}'
                else:
                    text_val = f'{value:.1f}'
                    
                ax.text(j, i, text_val, ha='center', va='center', 
                       color=color, fontsize=8, fontweight='bold')
        
        # Add grid lines
        ax.set_xticks(np.arange(len(self.ability_configs) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(self.map_names) + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(which='minor', size=0)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Generate ability configuration analysis color grid plots',
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
        help='Directory to save output figures (default: {cluster_path}/data/samuel_lozano/cooked/grid_full_analysis_figures)'
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
        help='First map name (default: baseline_division_of_labor_large)'
    )
    
    parser.add_argument(
        '--map_name_2',
        type=str,
        default='encouraged_division_of_labor_large',
        help='Second map name (default: encouraged_division_of_labor_large)'
    )
    

    
    parser.add_argument(
        '--game_type',
        type=str,
        choices=['classic', 'classic_collision'],
        default='classic_collision',
        help='Game type to analyze (default: classic_collision)'
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
    """Main function to run the full 2D grid cooperative analysis."""
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
            output_dir_base = f"{local_path}/data/samuel_lozano/cooked/grid_full_analysis_figures"
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
        print("FULL 2D GRID ANALYSIS - MAPS × ABILITIES")
        print("=" * 60)
        print(f"Cluster: {cluster}")
        print(f"Game type: {args.game_type}")
        print(f"Init type: {args.init_type}")
        print(f"Synergy scaling factor: {args.synergy}")
        print(f"Specialization: {args.specialization}")
        print(f"Maps analyzed: {len(analyzer.map_names)} maps")
        print(f"Ability configurations: {len(analyzer.ability_configs)} columns")
        if args.study_name:
            print(f"Study: {args.study_name}")
        
        df_or_dict = analyzer.load_experimental_data()
        
        # Check if we got a dict (multiple combinations) or single DataFrame
        if isinstance(df_or_dict, dict):
            # Multiple synergy/specialization combinations - create separate figures for each
            print("\\n" + "=" * 60)
            print(f"CREATING FIGURES FOR {len(df_or_dict)} SYNERGY/SPECIALIZATION COMBINATIONS")
            print("=" * 60)
            
            for combination_key in sorted(df_or_dict.keys()):
                print(f"\\nProcessing {combination_key}...")
                df = df_or_dict[combination_key]
                
                # Prepare data based on episode selection
                prepared_data = analyzer.prepare_episode_data(
                    df, 
                    episode_selection=args.episode_range,
                    num_episodes=args.num_episodes,
                    target_episode=args.target_episode
                )
                
                print(f"\\n=== Creating 2D Grid Analysis (Maps × Abilities) ===\\n")
                
                # Initialize the plotter
                plotter = ColorGridPlotter(
                    analyzer.map_names,
                    analyzer.ability_configs,
                    args.game_type,
                    ability_config_labels=analyzer.ability_config_labels
                )
                
                # Set up output directory
                output_dir = Path(output_dir_base)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Build base filename parts
                filename_parts = []
                filename_parts.append('speeds_full_grid')
                
                # Extract synergy and specialization from combination key
                parts = combination_key.split('__')
                synergy_part = parts[0]  # e.g., "synergy_0.50"
                spec_part = parts[1]      # e.g., "specialized_0.25"
                
                filename_parts.append(synergy_part)
                filename_parts.append(spec_part)
                
                if args.episode_range != 'final':
                    filename_parts.append(args.episode_range)
                
                if args.num_episodes != 100:
                    filename_parts.append(f'{args.num_episodes}ep')
                
                if args.target_episode is not None:
                    filename_parts.append(f'target{args.target_episode}')
                
                if args.init_type != 'empty_init':
                    filename_parts.append(args.init_type)
                
                if args.game_type != 'classic':
                    filename_parts.append(args.game_type)
                
                if args.filename_suffix:
                    filename_parts.append(args.filename_suffix)
                
                base_filename = '_'.join(filename_parts)
                
                # Generate the 6 different comparison plots
                comparison_types = [
                    ('global', 'Global Baseline'),
                    ('row', 'Row Baseline'), 
                    ('column', 'Column Baseline')
                ]
                
                output_paths = []
                
                for comparison_type, type_name in comparison_types:
                    print(f"\\nGenerating {type_name} comparison plots...")
                    
                    # Determine subfolder based on comparison type
                    if comparison_type == 'row':
                        subfolder = output_dir / 'grid_per_row'
                    elif comparison_type == 'column':
                        subfolder = output_dir / 'grid_per_column'
                    else:  # global
                        subfolder = output_dir
                    
                    # Create subfolder if it doesn't exist
                    subfolder.mkdir(parents=True, exist_ok=True)
                    
                    # Create filename for this comparison type
                    filename = f"{base_filename}_{comparison_type}.png"
                    output_path = subfolder / filename
                    
                    # Generate the figure
                    fig = plotter.create_color_grid_figure(prepared_data, comparison_type, str(output_path))
                    output_paths.append(output_path)
                    
                    plt.close(fig)  # Close the figure to free memory
                
                print(f"\\n=== Analysis Complete for {combination_key} ===\\n")
                print("Generated 2D grid plots:")
                for path in output_paths:
                    print(f"  - {path}")
                
            print("\\n" + "=" * 60)
            print(f"ALL {len(df_or_dict)} COMBINATION SETS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
        else:
            # Single specialization mode - process normally
            df = df_or_dict
            
            # Prepare data based on episode selection
            prepared_data = analyzer.prepare_episode_data(
                df, 
                episode_selection=args.episode_range,
                num_episodes=args.num_episodes,
                target_episode=args.target_episode
            )
            
            print(f"\\n=== Creating 2D Grid Analysis (Maps × Abilities) ===\\n")
            
            # Initialize the plotter
            plotter = ColorGridPlotter(
                analyzer.map_names,
                analyzer.ability_configs,
                args.game_type,
                ability_config_labels=analyzer.ability_config_labels
            )
            
            # Set up output directory
            output_dir = Path(output_dir_base)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Build base filename parts
            filename_parts = []
            filename_parts.append('speeds_full_grid')
            
            if args.episode_range != 'final':
                filename_parts.append(args.episode_range)
            
            if args.num_episodes != 100:
                filename_parts.append(f'{args.num_episodes}ep')
            
            if args.target_episode is not None:
                filename_parts.append(f'target{args.target_episode}')
            
            if args.init_type != 'empty_init':
                filename_parts.append(args.init_type)
            
            if args.synergy != 0.0:
                filename_parts.append(f'synergy{args.synergy}')
            
            if args.specialization is not None:
                filename_parts.append(f'lambda{args.specialization}')
            
            if args.game_type != 'classic':
                filename_parts.append(args.game_type)
            
            if args.filename_suffix:
                filename_parts.append(args.filename_suffix)
            
            base_filename = '_'.join(filename_parts)
            
            # Generate the 6 different comparison plots
            comparison_types = [
                ('global', 'Global Baseline'),
                ('row', 'Row Baseline'), 
                ('column', 'Column Baseline')
            ]
            
            output_paths = []
            
            for comparison_type, type_name in comparison_types:
                print(f"\\nGenerating {type_name} comparison plots...")
                
                # Determine subfolder based on comparison type
                if comparison_type == 'row':
                    subfolder = output_dir / 'grid_per_row'
                elif comparison_type == 'column':
                    subfolder = output_dir / 'grid_per_column'
                else:  # global
                    subfolder = output_dir
                
                # Create subfolder if it doesn't exist
                subfolder.mkdir(parents=True, exist_ok=True)
                
                # Create filename for this comparison type
                filename = f"{base_filename}_{comparison_type}.png"
                output_path = subfolder / filename
                
                # Generate the figure
                fig = plotter.create_color_grid_figure(prepared_data, comparison_type, str(output_path))
                output_paths.append(output_path)
                
                plt.close(fig)  # Close the figure to free memory
            
            print(f"\\n=== Analysis Complete ===\\n")
            print("Generated 2D grid plots:")
            for path in output_paths:
                print(f"  - {path}")
            
            # Print summary statistics for global comparison
            print("\\n=== Summary Statistics (Global Baseline) ===\\n")
            
            perf_diffs = plotter.calculate_performance_differences_global(prepared_data)
            spec_diffs = plotter.calculate_specialization_differences_global(prepared_data)

            def ability_sort_key(value):
                try:
                    return float(value)
                except ValueError:
                    return float('inf')
            
            print("Performance differences (vs baseline map + X=1.0):")
            for map_short in [plotter._extract_map_short_name(name) for name in analyzer.map_names]:
                if map_short in perf_diffs:
                    print(f"\\n{map_short}:")
                    for ability_config in sorted(perf_diffs[map_short].keys(), key=ability_sort_key):
                        diff = perf_diffs[map_short][ability_config]
                        print(f"  {plotter._ability_label(ability_config)}: {diff:+.2f} deliveries")
            
            print("\\nSpecialization differences (vs baseline map + X=1.0):")
            for map_short in [plotter._extract_map_short_name(name) for name in analyzer.map_names]:
                if map_short in spec_diffs:
                    print(f"\\n{map_short}:")
                    for ability_config in sorted(spec_diffs[map_short].keys(), key=ability_sort_key):
                        diff = spec_diffs[map_short][ability_config] 
                        print(f"  {plotter._ability_label(ability_config)}: {diff:+.3f} specialization")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()