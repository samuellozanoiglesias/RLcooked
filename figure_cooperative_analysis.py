#!/usr/bin/env python3
"""
Cooperative Analysis Figure Generator

This script creates raincloud plots comparing 8 different experimental conditions
across four performance metrics. The visualization style includes:
- Half-violin density plots on the left
- Raw data dot plots on the right  
- Diamond markers with error bars for central tendency

The script now supports different initialization types (random_init, empty_init) that
are located in subdirectories under the main experiment path.

Usage:
    python figure_cooperative_analysis.py [options]

Examples:
    # Default (baseline vs encouraged, random_init, both specializations)
    nohup python figure_cooperative_analysis.py --episode_range all --output_dir ./figures > figure_cooperative_analysis_all.log 2>&1 &
    nohup python figure_cooperative_analysis.py --episode_range final --num_episodes 100 > figure_cooperative_analysis_final.log 2>&1 &
    
    # Analyze only specialized data
    nohup python figure_cooperative_analysis.py --episode_range final --specialization specialized --num_episodes 100 > figure_cooperative_analysis_specialized.log 2>&1 &
    
    # Analyze only non-specialized data
    nohup python figure_cooperative_analysis.py --episode_range final --specialization non_specialized --num_episodes 100 > figure_cooperative_analysis_non_specialized.log 2>&1 &
    
    # Analyze empty_init data (both specializations)
    nohup python figure_cooperative_analysis.py --episode_range final --init_type empty_init --num_episodes 100 > figure_cooperative_analysis_empty_init.log 2>&1 &
    
    # Analyze speeds study with specific initialization
    nohup python figure_cooperative_analysis.py --episode_range final --study_name speeds --init_type random_init --num_episodes 100 > figure_cooperative_analysis_speeds.log 2>&1 &
    
    # Analyze with eta parameter (reference-based reward shaping)
    nohup python figure_cooperative_analysis.py --episode_range final --init_type random_init --eta 0.5 --num_episodes 100 > figure_cooperative_analysis_eta_0.5.log 2>&1 &
    
    # Custom maps (e.g., baseline vs forced) with empty_init
    nohup python figure_cooperative_analysis.py --episode_range final --map_name_1 baseline_division_of_labor_large --map_name_2 collision_division_of_labor_large --init_type empty_init --num_episodes 100 > figure_cooperative_analysis_baseline_forced.log 2>&1 &
    
    # Extended mode with additional metrics
    nohup python figure_cooperative_analysis.py --episode_range final --extended --init_type random_init --num_episodes 100 > figure_cooperative_analysis_extended.log 2>&1 &
    
    # Analyze specific episodes around episode 500
    nohup python figure_cooperative_analysis.py --episode_range specific --target_episode 500 --num_episodes 20 > figure_cooperative_analysis_ep500.log 2>&1 &    
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
                 init_type: str = 'random_init',
                 eta: float = 0.0,
                 eta_provided: bool = False,
                 specialization_mode: Optional[str] = None,
                 cluster: str = 'cuenca'):
        self.config = AnalysisConfig()
        self.data_processor = DataProcessor(self.config)
        self.study_name = study_name
        self.map_name_1 = map_name_1
        self.map_name_2 = map_name_2
        self.init_type = init_type
        self.eta = eta
        self.eta_provided = eta_provided
        self.specialization_mode = specialization_mode  # None=both, 'specialized', or 'non_specialized'
        self.cluster = cluster
        self.detected_specializations = []  # Track which specializations were found
        
        # Set up cluster path prefix
        if cluster not in self.config.cluster_paths:
            raise ValueError(f"Invalid cluster '{cluster}'. Choose from {list(self.config.cluster_paths.keys())}")
        self.local_path = self.config.cluster_paths[cluster]
        
        # Extract short names for condition labels
        # E.g., 'baseline_division_of_labor_large' -> 'baseline'
        # E.g., 'encouraged_division_of_labor_large' -> 'encouraged'
        self.map_1_short = self._extract_map_short_name(map_name_1)
        self.map_2_short = self._extract_map_short_name(map_name_2)
        
        # Define experimental conditions mapping
        # Updated based on actual training configurations:
        # - Superstar: Both agents have 1.0_1.0 (walking=1.0, cutting=1.0)
        # - Mixed: Agent1 has 0.4_1.0 (walking=0.4, cutting=1.0), Agent2 has 1.0_0.2 (walking=1.0, cutting=0.2)
        # Base condition mapping (will be expanded with specialization if analyzing both)
        self.base_condition_mapping = {
            # Map 1 conditions  
            (map_name_1, 'classic', 'mixed'): f'{self.map_1_short}_mixed',
            (map_name_1, 'classic_collision', 'mixed'): f'{self.map_1_short}_mixed_collision', 
            (map_name_1, 'classic', 'superstar'): f'{self.map_1_short}_superstar',
            (map_name_1, 'classic_collision', 'superstar'): f'{self.map_1_short}_superstar_collision',
            # Map 2 conditions
            (map_name_2, 'classic', 'mixed'): f'{self.map_2_short}_mixed',
            (map_name_2, 'classic_collision', 'mixed'): f'{self.map_2_short}_mixed_collision',
            (map_name_2, 'classic', 'superstar'): f'{self.map_2_short}_superstar', 
            (map_name_2, 'classic_collision', 'superstar'): f'{self.map_2_short}_superstar_collision'
        }
        
        # Color palette mapping - dynamically generated based on map names
        # Base colors for specialized (or when not distinguishing)
        self.base_color_palette = {
            f'{self.map_1_short}_mixed': '#FF6B6B',              # Red
            f'{self.map_1_short}_mixed_collision': '#FF9F40',    # Orange  
            f'{self.map_1_short}_superstar': '#90EE90',          # Light Green
            f'{self.map_1_short}_superstar_collision': '#228B22', # Dark Green
            f'{self.map_2_short}_mixed': '#40E0D0',              # Teal
            f'{self.map_2_short}_mixed_collision': '#4169E1',    # Blue
            f'{self.map_2_short}_superstar': '#8A2BE2',          # Purple
            f'{self.map_2_short}_superstar_collision': '#FF69B4' # Pink
        }
        
        # Will be expanded in load_experimental_data if analyzing both specializations
        self.color_palette = self.base_color_palette.copy()
        
        # Performance metrics to analyze (updated for multi-agent cooperative data)
        # These will be created from individual agent metrics
        # Arranged for 3x2 layout: (total_deliveries, total_counters), (deliveries_agent1, deliveries_agent2), (cuts_agent1, cuts_agent2)
        self.performance_metrics = {
            'total_deliveries': 'Total Deliveries (Combined)',
            'total_counters': 'Total Counters (Combined)', 
            'useful_delivery_ai_rl_1': 'Deliveries Agent 1',
            'useful_delivery_ai_rl_2': 'Deliveries Agent 2',
            'cut_ai_rl_1': 'Cuts Agent 1',
            'cut_ai_rl_2': 'Cuts Agent 2'
        }
        
        # Extended metrics for detailed analysis (4x2 layout)
        self.extended_metrics = {
            'salad_ai_rl_1': 'Salad Agent 1',
            'salad_ai_rl_2': 'Salad Agent 2',
            'plate_ai_rl_1': 'Plate Agent 1',
            'plate_ai_rl_2': 'Plate Agent 2',
            'raw_food_ai_rl_1': 'Raw Food Agent 1',
            'raw_food_ai_rl_2': 'Raw Food Agent 2',
            'counter_ai_rl_1': 'Counter Agent 1',
            'counter_ai_rl_2': 'Counter Agent 2'
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
    
    def load_experimental_data(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load and combine data from all experimental conditions.
        
        Returns:
            If analyzing single specialization: pd.DataFrame
            If analyzing both specializations: Dict[str, pd.DataFrame] with keys:
                - 'non_specialized': All conditions using non_specialized data
                - 'mixed': Mixed conditions with specialized data, stars conditions with non_specialized data
                - 'specialized': All conditions using specialized data
        """
        print("Loading experimental data from all conditions...")
        if self.study_name:
            print(f"Using study name: {self.study_name}")
        if self.specialization_mode is None:
            print("Specialization mode: Creating three figures (non_specialized, mixed, specialized)")
        else:
            print(f"Specialization mode: {self.specialization_mode}")
        
        # Determine which specialization modes to try
        if self.specialization_mode is None:
            # When analyzing both, we'll create three datasets
            spec_modes_to_analyze = ['specialized', 'non_specialized']
            create_three_datasets = True
        elif self.specialization_mode:
            spec_modes_to_analyze = [self.specialization_mode]
            create_three_datasets = False
        else:
            spec_modes_to_analyze = [None]  # No specialization folder
            create_three_datasets = False
        
        # Store data separately for each specialization mode
        # When create_three_datasets is True, we'll store data organized by (spec_mode, speed_config)
        data_by_spec_and_speed = {} if create_three_datasets else None
        data_by_specialization = {}
        
        # Load data for each specialization mode
        for spec_mode in spec_modes_to_analyze:
            all_data = []  # Reset for each specialization mode
            
            # Load all conditions for this specialization mode
            for (map_name, game_type, speed_config), condition_name in self.base_condition_mapping.items():
                # Keep original condition name (no suffix)
                print(f"\nProcessing condition: {condition_name}")
                print(f"  Map: {map_name}, Game type: {game_type}, Speed config: {speed_config}")
                if spec_mode:
                    print(f"  Specialization: {spec_mode}")
                
                try:
                    # Set up paths for this condition with init_type and eta in the correct order
                    # Only include eta folder if eta was explicitly provided (even if it's 0)
                    
                    if self.eta_provided:
                        # Build path with eta folder
                        eta_folder = f"eta_{self.eta}" if self.eta != 0 else "eta_0"
                        
                        # Use the specific specialization mode (no fallback)
                        if spec_mode:
                            spec_folder = spec_mode
                        else:
                            spec_folder = ""  # No specialization folder
                        
                        # Build directory path
                        if self.study_name:
                            # Use study_name folder structure
                            if spec_folder:
                                raw_dir = f"{self.local_path}/data/samuel_lozano/cooked/{game_type}/{self.init_type}/map_{map_name}/{eta_folder}/{spec_folder}/{self.study_name}"
                            else:
                                raw_dir = f"{self.local_path}/data/samuel_lozano/cooked/{game_type}/{self.init_type}/map_{map_name}/{eta_folder}/{self.study_name}"
                        else:
                            # Default structure
                            if spec_folder:
                                raw_dir = f"{self.local_path}/data/samuel_lozano/cooked/{game_type}/{self.init_type}/map_{map_name}/{eta_folder}/{spec_folder}"
                            else:
                                raw_dir = f"{self.local_path}/data/samuel_lozano/cooked/{game_type}/{self.init_type}/map_{map_name}/{eta_folder}"
                    
                    else:
                        # No eta folder - use original structure when eta is not provided
                        if self.study_name:
                            # Use study_name folder structure
                            raw_dir = f"{self.local_path}/data/samuel_lozano/cooked/{game_type}/{self.init_type}/map_{map_name}/{self.study_name}"
                        else:
                            # Default structure
                            raw_dir = f"{self.local_path}/data/samuel_lozano/cooked/{game_type}/{self.init_type}/map_{map_name}"
                
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
                    
                    # Check if the training has the expected speed configuration
                    # Speed information is stored in walking_speed_1, cutting_speed_1, etc. columns
                    df_filtered = self._filter_by_speed_config(df, speed_config)
                    
                    if len(df_filtered) == 0:
                        print(f"  Warning: No data found matching speed configuration {speed_config}")
                        print(f"  Available speed configurations in this training:")
                        self._print_available_speeds(df)
                        continue
                    
                    print(f"  Filtered to {len(df_filtered)} episodes matching speed {speed_config}")
                    
                    # Create combined metrics from individual agent metrics
                    self._create_combined_metrics(df_filtered)
                    
                    # Add condition metadata
                    df_filtered['condition'] = condition_name
                    df_filtered['map_name'] = map_name 
                    df_filtered['game_type_clean'] = game_type
                    df_filtered['speed_condition'] = speed_config
                    if spec_mode:
                        df_filtered['specialization'] = spec_mode
                    
                    # Add color palette entry for this specific condition
                    if condition_name not in self.color_palette:
                        # Get base color
                        base_color = self.base_color_palette.get(condition_name, '#666666')
                        self.color_palette[condition_name] = base_color
                    
                    # Store data appropriately
                    if create_three_datasets:
                        # Store by (spec_mode, speed_config) tuple for later combining
                        key = (spec_mode, speed_config)
                        if key not in data_by_spec_and_speed:
                            data_by_spec_and_speed[key] = []
                        data_by_spec_and_speed[key].append(df_filtered)
                    else:
                        # Normal storage for single specialization mode
                        all_data.append(df_filtered)
                    
                    print(f"  Loaded {len(df_filtered)} episodes")
                    
                    # Debug: Check if our target metrics exist
                    missing_metrics = [metric for metric in self.performance_metrics.keys() if metric not in df_filtered.columns]
                    if missing_metrics:
                        print(f"  Warning: Missing metrics: {missing_metrics}")
                        available_metrics = [metric for metric in self.performance_metrics.keys() if metric in df_filtered.columns]
                        print(f"  Available metrics: {available_metrics}")
                    
                else:
                    print(f"  Warning: No data loaded for condition {condition_name}")
            
            # Store data for this specialization mode if we have any (only for non-create_three_datasets mode)
            if not create_three_datasets and all_data:
                if spec_mode:
                    data_by_specialization[spec_mode] = pd.concat(all_data, ignore_index=True)
                else:
                    # No specialization mode - just return single DataFrame
                    data_by_specialization['default'] = pd.concat(all_data, ignore_index=True)
        
        # Return based on what we collected
        if not data_by_specialization and not create_three_datasets:
            raise ValueError("No data could be loaded from any experimental condition")
        
        # If creating three datasets, combine data accordingly
        if create_three_datasets:
            print("\n" + "=" * 60)
            print("CREATING THREE DATASETS")
            print("=" * 60)
            
            # Dataset 1: All non_specialized (both stars and mixed)
            dataset_1_data = []
            for (spec_mode, speed_config), data_list in data_by_spec_and_speed.items():
                if spec_mode == 'non_specialized':
                    dataset_1_data.extend(data_list)
            
            if dataset_1_data:
                data_by_specialization['non_specialized'] = pd.concat(dataset_1_data, ignore_index=True)
                print(f"Dataset 1 (non_specialized): {len(data_by_specialization['non_specialized'])} episodes")
            
            # Dataset 2: Mixed from specialized, Stars from non_specialized
            dataset_2_data = []
            for (spec_mode, speed_config), data_list in data_by_spec_and_speed.items():
                if spec_mode == 'specialized' and speed_config == 'mixed':
                    dataset_2_data.extend(data_list)
                elif spec_mode == 'non_specialized' and speed_config == 'superstar':
                    dataset_2_data.extend(data_list)
            
            if dataset_2_data:
                data_by_specialization['mixed'] = pd.concat(dataset_2_data, ignore_index=True)
                print(f"Dataset 2 (mixed): {len(data_by_specialization['mixed'])} episodes")
            
            # Dataset 3: All specialized (both stars and mixed)
            dataset_3_data = []
            for (spec_mode, speed_config), data_list in data_by_spec_and_speed.items():
                if spec_mode == 'specialized':
                    dataset_3_data.extend(data_list)
            
            if dataset_3_data:
                data_by_specialization['specialized'] = pd.concat(dataset_3_data, ignore_index=True)
                print(f"Dataset 3 (specialized): {len(data_by_specialization['specialized'])} episodes")
            
            if not data_by_specialization:
                raise ValueError("No data could be loaded from any experimental condition")
            
            print("\n" + "=" * 60)
            for spec_mode, df in data_by_specialization.items():
                print(f"\n{spec_mode.upper()} data: {len(df)} episodes across {df['condition'].nunique()} conditions")
                print(f"  Conditions: {sorted(df['condition'].unique())}")
            print(f"Datasets created: {sorted(data_by_specialization.keys())}")
            return data_by_specialization
        
        # If analyzing both specializations (old logic - shouldn't happen now), return dict with separate DataFrames
        if self.specialization_mode is None and len(data_by_specialization) > 1:
            for spec_mode, df in data_by_specialization.items():
                print(f"\n{spec_mode.upper()} data: {len(df)} episodes across {df['condition'].nunique()} conditions")
                print(f"  Conditions: {sorted(df['condition'].unique())}")
            print(f"Specialization modes found: {sorted(data_by_specialization.keys())}")
            return data_by_specialization
        
        # Otherwise, return single DataFrame
        combined_df = list(data_by_specialization.values())[0]
        print(f"\nTotal loaded data: {len(combined_df)} episodes across {combined_df['condition'].nunique()} conditions")
        print(f"Conditions found: {sorted(combined_df['condition'].unique())}")
        if self.detected_specializations:
            print(f"Specialization modes found: {sorted(self.detected_specializations)}")
        
        return combined_df
    
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
    
    def _filter_by_speed_config(self, df: pd.DataFrame, expected_config: str) -> pd.DataFrame:
        """Filter training data to match the expected speed configuration.
        
        Args:
            df: DataFrame containing training data with speed columns
            expected_config: Expected configuration - either "superstar" or "mixed"
            
        Returns:
            Filtered DataFrame containing only data matching the speed configuration
        """
        if len(df) == 0:
            return df
            
        if expected_config == "superstar":
            # Superstar: Both agents have 1.0_1.0
            if all(col in df.columns for col in ['walking_speed_1', 'cutting_speed_1', 'walking_speed_2', 'cutting_speed_2']):
                filter_condition = (
                    (abs(df['walking_speed_1'] - 1.0) < 0.01) &
                    (abs(df['cutting_speed_1'] - 1.0) < 0.01) &
                    (abs(df['walking_speed_2'] - 1.0) < 0.01) &
                    (abs(df['cutting_speed_2'] - 1.0) < 0.01)
                )
                return df[filter_condition].copy()
            else:
                print(f"    Warning: Missing speed columns for superstar filtering")
                return pd.DataFrame()
                
        elif expected_config == "mixed":
            # Mixed: Agent1 has 0.4_1.0, Agent2 has 1.0_0.2
            if all(col in df.columns for col in ['walking_speed_1', 'cutting_speed_1', 'walking_speed_2', 'cutting_speed_2']):
                filter_condition = (
                    (abs(df['walking_speed_1'] - 0.4) < 0.01) &
                    (abs(df['cutting_speed_1'] - 1.0) < 0.01) &
                    (abs(df['walking_speed_2'] - 1.0) < 0.01) &
                    (abs(df['cutting_speed_2'] - 0.2) < 0.01)
                )
                return df[filter_condition].copy()
            else:
                print(f"    Warning: Missing speed columns for mixed filtering")
                return pd.DataFrame()
        else:
            print(f"    Warning: Unknown speed configuration {expected_config}")
            return df
    
    def _print_available_speeds(self, df: pd.DataFrame):
        """Print the available speed configurations in the training data."""
        if len(df) == 0:
            print("    No data available")
            return
            
        # Check what speed configurations are present
        speed_configs = set()
        
        for _, row in df.head(10).iterrows():  # Check first few rows to get unique configs
            speeds = []
            
            # Collect agent 1 speeds
            if 'walking_speed_1' in row and 'cutting_speed_1' in row:
                if not pd.isna(row['walking_speed_1']) and not pd.isna(row['cutting_speed_1']):
                    speeds.append(f"Agent1: {row['walking_speed_1']:.1f}_{row['cutting_speed_1']:.1f}")
            
            # Collect agent 2 speeds
            if 'walking_speed_2' in row and 'cutting_speed_2' in row:
                if not pd.isna(row['walking_speed_2']) and not pd.isna(row['cutting_speed_2']):
                    speeds.append(f"Agent2: {row['walking_speed_2']:.1f}_{row['cutting_speed_2']:.1f}")
            
            if speeds:
                speed_configs.add(" | ".join(speeds))
        
        for config in sorted(speed_configs):
            print(f"    {config}")


class RaincloudPlotter:
    """Creates raincloud plots for cooperative analysis."""
    
    def __init__(self, color_palette: Dict[str, str], performance_metrics: Dict[str, str], 
                 extended_metrics: Optional[Dict[str, str]] = None):
        self.color_palette = color_palette
        self.performance_metrics = performance_metrics
        self.extended_metrics = extended_metrics or {}
        
        # Set up matplotlib style
        plt.style.use('default')
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['grid.alpha'] = 0.3
        
    def create_raincloud_subplot(self, ax, data: pd.DataFrame, metric: str, metric_label: str, speed_info: str = ""):
        """Create a single raincloud plot on the given axis."""
        
        # Get unique conditions and sort them for consistent ordering
        conditions = sorted(data['condition'].unique())
        
        # Set up positions for each condition
        positions = np.arange(len(conditions))
        width_offset = 0.4
        
        # Clear the axis
        ax.clear()
        
        for i, condition in enumerate(conditions):
            condition_data = data[data['condition'] == condition][metric].dropna()
            
            if len(condition_data) == 0:
                print(f"    Warning: No data for condition {condition} in metric {metric}")
                continue
                
            color = self.color_palette.get(condition, '#666666')
            position = positions[i]
            
            try:
                # 1. Left side: Half violin (density) - simplified approach
                if len(condition_data) > 1:  # Need at least 2 points for violin plot
                    try:
                        # Create violin plot
                        parts = ax.violinplot([condition_data], positions=[position - width_offset], 
                                            widths=[0.6], showmeans=False, showmedians=False, 
                                            showextrema=False)
                        
                        for pc in parts['bodies']:
                            pc.set_facecolor('none')
                            pc.set_edgecolor('black')
                            pc.set_linewidth(1.0)
                            pc.set_alpha(0.8)
                    except Exception as violin_error:
                        print(f"    Warning: Could not create violin plot for {condition}: {violin_error}")
                        # Fall back to a simple density representation using histogram
                        try:
                            y_hist, bin_edges = np.histogram(condition_data, bins=20, density=True)
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                            # Scale the histogram to fit in the left area
                            x_hist = position - width_offset - (y_hist / y_hist.max()) * width_offset * 0.8
                            ax.plot(x_hist, bin_centers, color='black', linewidth=1.0, alpha=0.8)
                        except Exception as hist_error:
                            print(f"    Warning: Could not create histogram fallback for {condition}: {hist_error}")
                
            except Exception as e:
                print(f"    Warning: Could not create density plot for {condition}: {e}")
                # Skip density plot for this condition
            
            # 2. Right side: Dot plot (simplified beeswarm)
            try:
                y_values = np.array(condition_data)
                # Simple jittering instead of complex beeswarm
                np.random.seed(42)  # For reproducible jittering
                x_jitter = np.random.normal(position + width_offset, 0.03, len(y_values))
                
                ax.scatter(x_jitter, y_values, c=color, s=8, alpha=0.7, 
                          edgecolors='black', linewidths=0.3, zorder=5)
                
            except Exception as e:
                print(f"    Warning: Could not create scatter plot for {condition}: {e}")
                # Fallback: simple vertical line plot
                try:
                    for j, y_val in enumerate(y_values[:min(100, len(y_values))]):  # Limit to 100 points
                        x_pos = position + width_offset + (j % 5 - 2) * 0.01  # Simple spread
                        ax.plot([x_pos, x_pos], [y_val, y_val], 'o', color=color, markersize=2, alpha=0.7)
                except Exception as fallback_error:
                    print(f"    Warning: Could not create fallback scatter for {condition}: {fallback_error}")
            
            # 3. Center: Diamond with error bar (mean ± std)
            try:
                mean_val = condition_data.mean()
                std_val = condition_data.std()
                
                # Diamond marker for mean
                ax.scatter(position, mean_val, marker='D', s=100, c=color, 
                          edgecolors='black', linewidths=1.0, zorder=10)
                
                # Error bar for standard deviation
                if not np.isnan(std_val) and std_val > 0:
                    ax.errorbar(position, mean_val, yerr=std_val, fmt='none', 
                               color='black', linewidth=1.5, capsize=4, zorder=9)
                
            except Exception as e:
                print(f"    Warning: Could not create mean/std markers for {condition}: {e}")
        
        # Styling
        ax.set_xticks(positions)
        ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
        
        # Set ylabel with speed info if provided
        if speed_info:
            ax.set_ylabel(f"{metric_label}\n{speed_info}", fontsize=10)
        else:
            ax.set_ylabel(metric_label, fontsize=10)
            
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Add border around subplot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('black')
    
    def create_composite_figure(self, data: pd.DataFrame, output_path: str = None, 
                                 extended: bool = False) -> plt.Figure:
        """Create the complete raincloud plot figure.
        
        Args:
            data: DataFrame with processed data
            output_path: Path to save figure
            extended: If True, create 7x2 layout with extended metrics; if False, create 3x2 layout
        """
        
        print(f"Creating {'extended' if extended else 'standard'} composite raincloud figure...")
        
        if extended:
            # Extended mode: 7x2 layout (3 main + 4 extended rows)
            fig, axes = plt.subplots(7, 2, figsize=(16, 32))
            fig.suptitle('Cooperative Performance Analysis (Extended)', fontsize=16, fontweight='bold', y=0.98)
        else:
            # Standard mode: 3x2 layout
            fig, axes = plt.subplots(3, 2, figsize=(16, 18))
            fig.suptitle('Cooperative Performance Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        # Flatten axes for easy iteration
        axes_flat = axes.flatten()
        
        # Create subplot for each metric
        metrics_list = list(self.performance_metrics.items())
        
        # Add extended metrics if in extended mode
        if extended and self.extended_metrics:
            metrics_list.extend(list(self.extended_metrics.items()))
        
        # Generate subplot labels
        subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
        
        # Define speed information for agent-specific metrics  
        speed_info_mapping = {
            'useful_delivery_ai_rl_1': '(Superstar: walk=1.0, cut=1.0 | Mixed: walk=0.4, cut=1.0)',
            'useful_delivery_ai_rl_2': '(Superstar: walk=1.0, cut=1.0 | Mixed: walk=1.0, cut=0.2)',
            'cut_ai_rl_1': '(Superstar: walk=1.0, cut=1.0 | Mixed: walk=0.4, cut=1.0)',
            'cut_ai_rl_2': '(Superstar: walk=1.0, cut=1.0 | Mixed: walk=1.0, cut=0.2)'
        }
        
        for i, (metric, metric_label) in enumerate(metrics_list):
            ax = axes_flat[i]
            
            # Add subplot label in top-left corner
            ax.text(0.02, 0.98, subplot_labels[i], transform=ax.transAxes, 
                   fontsize=14, fontweight='bold', va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Get speed info for this metric if available
            speed_info = speed_info_mapping.get(metric, "")
            
            # Create raincloud plot for this metric
            self.create_raincloud_subplot(ax, data, metric, metric_label, speed_info)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create legend at the bottom
        handles = []
        labels = []
        
        for condition, color in self.color_palette.items():
            if condition in data['condition'].unique():
                handle = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black')
                handles.append(handle)
                labels.append(condition)
        
        # Place legend at the bottom
        if extended:
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), 
                      ncol=4, frameon=True, fancybox=True, shadow=True, fontsize=10)
            plt.subplots_adjust(bottom=0.03)
        else:
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                      ncol=4, frameon=True, fancybox=True, shadow=True, fontsize=10)
            plt.subplots_adjust(bottom=0.06)
        
        # Save figure if output path provided
        if output_path:
            print(f"Saving figure to: {output_path}")
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Generate cooperative analysis raincloud plots',
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
        help='Directory to save output figures (default: {cluster_path}/data/samuel_lozano/cooked/cooperative_analysis_figures)'
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
        '--extended',
        action='store_true',
        help='Create extended plot with additional metrics (salad, plate, raw_food, counter) for both agents'
    )
    
    parser.add_argument(
        '--init_type',
        type=str,
        choices=['random_init', 'empty_init'],
        default='random_init',
        help='Initialization type subdirectory to analyze (default: random_init)'
    )
    
    parser.add_argument(
        '--eta',
        type=float,
        default=0.0,
        help='Eta parameter for reference-based reward shaping (default: 0.0)'
    )
    
    parser.add_argument(
        '--specialization',
        type=str,
        choices=['specialized', 'non_specialized', 'both'],
        default='both',
        help='Which specialization mode to analyze (default: both - analyzes both specialized and non_specialized if available)'
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
        # Check if eta was explicitly provided
        eta_provided = '--eta' in sys.argv
        
        # Determine specialization mode
        specialization_mode = None if args.specialization == 'both' else args.specialization
        
        # Set up output directory based on cluster if not explicitly provided
        cluster = args.cluster if args.cluster else 'cuenca'
        config = AnalysisConfig()
        local_path = config.cluster_paths[cluster]
        
        if args.output_dir is None:
            output_dir_base = f"{local_path}/data/samuel_lozano/cooked/cooperative_analysis_figures"
        else:
            output_dir_base = args.output_dir
        
        # Initialize analyzer
        analyzer = CooperativeAnalyzer(
            study_name=args.study_name,
            map_name_1=args.map_name_1,
            map_name_2=args.map_name_2,
            init_type=args.init_type,
            eta=args.eta,
            eta_provided=eta_provided,
            specialization_mode=specialization_mode,
            cluster=cluster
        )
        
        # Load experimental data
        print("=" * 60)
        print("COOPERATIVE ANALYSIS - RAINCLOUD PLOTS")
        print("=" * 60)
        print(f"Cluster: {args.cluster if args.cluster else 'cuenca'}")
        print(f"Init type: {args.init_type}")
        print(f"Eta: {args.eta}")
        print(f"Specialization: {args.specialization}")
        print(f"Map 1: {args.map_name_1}")
        print(f"Map 2: {args.map_name_2}")
        if args.study_name:
            print(f"Study: {args.study_name}")
        
        df_or_dict = analyzer.load_experimental_data()
        
        # Check if we got a dict (multiple specializations) or single DataFrame
        if isinstance(df_or_dict, dict):
            # Process each specialization separately and create separate graphs
            print("\n" + "=" * 60)
            print("CREATING SEPARATE GRAPHS FOR EACH SPECIALIZATION")
            print("=" * 60)
            
            for spec_mode in sorted(df_or_dict.keys()):
                print(f"\nProcessing {spec_mode} data...")
                df = df_or_dict[spec_mode]
                
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
                
                # Generate filename with specialization mode
                filename_parts = ["cooperative_analysis"]
                filename_parts.append(args.init_type)
                
                if analyzer.eta_provided:
                    filename_parts.append(f"eta_{args.eta}" if args.eta != 0 else "eta_0")
                
                # Add the specific specialization mode
                filename_parts.append(spec_mode)
                
                # Add map names (short versions)
                filename_parts.append(f"{analyzer.map_1_short}_vs_{analyzer.map_2_short}")
                
                if args.study_name:
                    filename_parts.append(args.study_name)
                
                filename_parts.append(args.episode_range)
                
                if args.extended:
                    filename_parts.append('extended')
                
                if args.episode_range == 'final':
                    filename_parts.append(f"{args.num_episodes}eps")
                
                if args.episode_range == 'specific':
                    filename_parts.append(f"ep{args.target_episode}_{args.num_episodes}eps")
                
                if args.filename_suffix:
                    filename_parts.append(args.filename_suffix)
                
                filename = "_".join(filename_parts) + ".png"
                output_path = output_dir / filename
                
                # Create raincloud plots for this specialization
                plotter = RaincloudPlotter(
                    analyzer.color_palette, 
                    analyzer.performance_metrics,
                    analyzer.extended_metrics
                )
                fig = plotter.create_composite_figure(processed_df, str(output_path), extended=args.extended)
                
                # Display results
                print(f"\nGraph for {spec_mode} completed!")
                print(f"  Figure saved to: {output_path}")
                print(f"  Total episodes: {len(processed_df)}")
                print(f"  Conditions: {sorted(processed_df['condition'].unique())}")
                
                plt.close(fig)  # Close figure to free memory
            
            print("\n" + "=" * 60)
            print("ALL GRAPHS COMPLETED SUCCESSFULLY!")
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
            filename_parts = ["cooperative_analysis"]
            
            # Add init type
            filename_parts.append(args.init_type)
            
            # Add eta if explicitly provided (even if 0)
            if analyzer.eta_provided:
                filename_parts.append(f"eta_{args.eta}" if args.eta != 0 else "eta_0")
            
            # Add specialization information
            if args.specialization != 'both':
                # When analyzing specific mode, add it to filename
                filename_parts.append(args.specialization)
            
            # Add map names (short versions)
            map_1_short = analyzer.map_1_short
            map_2_short = analyzer.map_2_short
            filename_parts.append(f"{map_1_short}_vs_{map_2_short}")
            
            # Add study name if specified
            if args.study_name:
                filename_parts.append(args.study_name)
            
            # Add episode selection type
            filename_parts.append(args.episode_range)
            
            # Add 'extended' marker if using extended mode
            if args.extended:
                filename_parts.append('extended')
            
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
            
            # Create raincloud plots
            plotter = RaincloudPlotter(
                analyzer.color_palette, 
                analyzer.performance_metrics,
                analyzer.extended_metrics
            )
            fig = plotter.create_composite_figure(processed_df, str(output_path), extended=args.extended)
            
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
            # plt.show()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()