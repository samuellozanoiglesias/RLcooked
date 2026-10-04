#!/usr/bin/env python3
"""
Figure 5 -- MARL theoretical solutions: flow maps (a) and grid map (b),
plus the supplementary MARL-simulations grid.

Default output (--version v2):
  fig5.pdf   a  flow maps of P1 (Open map) and P2 (PB map), Mixed / High abilities
             b  theoretical score-difference grid map with the points 1 (P1) and 2 (P2)
  figS_marl_simulations_grid.pdf   (with --sm) the MARL-simulations grid (old panel c)
--version original rebuilds the 3-panel Fig5.pdf described below.

Original 3-panel layout (all elements keep these sizes/fonts in every version):

  a  "MARL Theoretical Solutions": score-difference heat map (rho x switch cost),
     taken from ./figures/figure5a_reward_heatmap_with_lines_and_text.pdf, with
     axes, ticks, colour bar and the numbered (rho, s) points 1-4 of panel b.
  b  "MARL Theoretical Solutions": 2 x 4 vector fields (Mixed / High abilities,
     P1-P4), taken from ./figures/vector_field_{MA,HA}_P{1..4}_no_axes_no_mini_legend.pdf,
     with axes, tick labels and the 'Local reward' / 'Final trajectory
     specialization' colour bars.
  c  "MARL Simulations": map x ability grid of the score difference with respect
     to rho = 1 of each map (the 'row' comparison of
     BLUR-grid_full_cooperative_analysis.py, nearest interpolation), computed
     from the training data, with the map thumbnails ./figures/map_thumb_{1..6}.png.

The panel PDFs are merged into the page as VECTOR graphics (requires `pypdf`);
everything else (axes, colour bars, text) is drawn with matplotlib. All
positions/sizes were measured from Fig5.pdf (page 3174.96 x 637.80 pt).

Usage:
nohup python fig5_marl_theory_and_simulations.py --sm --episode_range specific --init_type empty_init \
    --num_episodes 20 --specialization 0.25 --synergy 1.35 --target_episode 750 \
    > fig5.log 2>&1 &

    # main figure only (needs no training data)
    python fig5_marl_theory_and_simulations.py
    # re-plot the SM figure from the cached MARL-simulations grid
    python fig5_marl_theory_and_simulations.py --sm --grid_csv ./figures/fig5c_grid.csv
    # layout test (synthetic panel c)
    python fig5_marl_theory_and_simulations.py --sm --demo
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.transforms import Affine2D
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# 1. DATA LOADING (CooperativeAnalyzer from BLUR-grid_full_cooperative_analysis.py)
# ===========================================================================
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
        from spoiled_broth.analysis.utils import DataProcessor, AnalysisConfig  # project modules
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
            #'m7_encouraged_division_of_labor_large',
            #'m6_encouraged_division_of_labor_large',
            #'m5_encouraged_division_of_labor_large',
            #'m4_encouraged_division_of_labor_large',
            #'m3_encouraged_division_of_labor_large',
            #'m2_encouraged_division_of_labor_large',
            #'m1_encouraged_division_of_labor_large',
            'a5_encouraged_division_of_labor_large',
            'a4_encouraged_division_of_labor_large',
            'a3_encouraged_division_of_labor_large',
            'a2_encouraged_division_of_labor_large',
            'a1_encouraged_division_of_labor_large',
            'encouraged_division_of_labor_large',
            'd1_encouraged_division_of_labor_large',
            'd2_encouraged_division_of_labor_large',
            'd3_encouraged_division_of_labor_large',
            'd4_encouraged_division_of_labor_large',
            'd5_encouraged_division_of_labor_large',
            'baseline_division_of_labor_large',
        ]
        
        # Define ability configurations for 2D grid analysis (X-axis) manually,
        # analogous to how map names are manually introduced for Y-axis.
        # Each entry is explicit: (id, label, (walk1, cut1, walk2, cut2))
        self.ability_config_definitions = [
            ('1.0', '1.0', (1.0, 1.0, 1.0, 1.0)),
            #('1.0-0.2', '0.2|1.0', (0.2, 1.0, 1.0, 1.0)),
            ('0.9', '0.9', (0.9, 1.0, 1.0, 0.9)),
            ('0.8', '0.8', (0.8, 1.0, 1.0, 0.8)),
            #('0.7-0.4', '0.7|0.4', (0.7, 1.0, 1.0, 0.4)),
            ('0.7', '0.7', (0.7, 1.0, 1.0, 0.7)),
            ('0.6', '0.6', (0.6, 1.0, 1.0, 0.6)),
            #('0.55', '0.55', (0.55, 1.0, 1.0, 0.55)),
            ('0.5', '0.5', (0.5, 1.0, 1.0, 0.5)),
            #('0.45', '0.45', (0.45, 1.0, 1.0, 0.45)),
            ('0.4', '0.4', (0.4, 1.0, 1.0, 0.4)),
            #('0.4-0.2', '0.4|0.2', (0.4, 1.0, 1.0, 0.2)),
            ('0.3', '0.3', (0.3, 1.0, 1.0, 0.3)),
            ('0.2', '0.2', (0.2, 1.0, 1.0, 0.2)),
            ('0.1', '0.1', (0.1, 1.0, 1.0, 0.1)),
        ]

        ### SPEEDS
        #self.ability_config_definitions = [
        #    ('0.6', '0.6', (0.6, 1.0, 1.0, 0.3)),
        #    ('0.58', '0.58', (0.58, 1.0, 1.0, 0.3)),
        #    ('0.56', '0.56', (0.56, 1.0, 1.0, 0.3)),
        #    ('0.54', '0.54', (0.54, 1.0, 1.0, 0.3)),
        #    ('0.52', '0.52', (0.52, 1.0, 1.0, 0.3)),
        #    ('0.5', '0.5', (0.5, 1.0, 1.0, 0.3)),
        #    ('0.48', '0.48', (0.48, 1.0, 1.0, 0.3)),
        #    ('0.46', '0.46', (0.46, 1.0, 1.0, 0.3)),
        #    ('0.44', '0.44', (0.44, 1.0, 1.0, 0.3)),
        #    ('0.42', '0.42', (0.42, 1.0, 1.0, 0.3)),
        #    ('0.4', '0.4', (0.4, 1.0, 1.0, 0.3)),
        #]

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
    




# ===========================================================================
# 2. PANEL c DATA  (row-wise score difference, from ColorGridPlotter)
# ===========================================================================
def extract_map_short_name(map_name: str) -> str:
    short_name = map_name
    if '_division_of_labor_large' in short_name:
        short_name = short_name.replace('_division_of_labor_large', '')
    elif '_division_of_labor' in short_name:
        short_name = short_name.replace('_division_of_labor', '')
    short_name = short_name.strip('_')
    return short_name if short_name else map_name


def row_score_difference_grid(data: pd.DataFrame, map_names: List[str], ability_configs: List[str],
                              game_type: str) -> np.ndarray:
    """Mean total deliveries of every (map, ability) cell minus the rho = 1.0 cell of the
    same map (the 'row' comparison of the original script). Rows follow map_names,
    columns follow ability_configs."""
    collision_suffix = '_collision' if 'collision' in game_type else ''
    grid = np.zeros((len(map_names), len(ability_configs)))
    for i, map_name in enumerate(map_names):
        map_short = extract_map_short_name(map_name)
        base = data[data['condition'] == f"{map_short}_1.0{collision_suffix}"]
        if len(base) == 0:
            print(f"Warning: no row baseline for {map_short}_1.0{collision_suffix}; using 0.0")
            base_val = 0.0
        else:
            base_val = base['total_deliveries'].mean()
        for j, ability in enumerate(ability_configs):
            cond = data[data['condition'] == f"{map_short}_{ability}{collision_suffix}"]
            if len(cond) == 0:
                print(f"Warning: no data for condition {map_short}_{ability}{collision_suffix}")
                continue
            grid[i, j] = cond['total_deliveries'].mean() - base_val
    return grid


# ===========================================================================
# 3. FIGURE 5 LAYOUT   (points, y measured from the TOP of the page;
#                       all values measured from Fig5.pdf)
# ===========================================================================
PAGE_W, PAGE_H = 3174.96, 637.80

# ---- panel a ------------------------------------------------------------
AX_A = (141.15, 596.24, 104.01, 522.52)            # axes frame (left, right, top, bottom)
A_HEATMAP_RECT = (141.65, 595.74, 104.51, 522.02)  # where the heat-map PDF is placed
A_XLIM, A_YLIM = (0.05, 1.00), (0.00, 1.00)
A_XTICKS = [0.05, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00]
A_YTICKS = [0.05, 0.25, 0.50, 0.75, 1.00]
A_FRAME_LW = 1.0
# numbered (rho, s) points of panel b, drawn on panel a
A_POINTS = {  # label: (rho, s, label centre x, label centre y  [page pt])
    '1': (0.851, 0.041, 539.0, 473.7),
    '2': (0.852, 0.500, 504.5, 285.5),
    '3': (0.706, 0.949, 435.0, 142.0),
    '4': (0.092, 0.799, 202.5, 171.2),
}
CB_A = (613.34, 644.90, 107.66, 517.30)
CB_A_VMAX = (312.58 - 107.66) / 228.3               # 0 at y=312.58, 0.2 per 45.66 pt
CB_A_TICKS = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
# (top y of band, colour) -- banded colour scale of the panel-a heat map
CB_A_BANDS = [(107.66, '#f10905'), (121.69, '#fd1f0f'), (139.66, '#fe3422'), (160.62, '#ff5151'),
              (184.58, '#ff6e6e'), (207.03, '#ff8686'), (224.25, '#ff9c9c'), (238.47, '#f5aaa8'),
              (248.21, '#ffb6b6'), (257.94, '#ffbfbf'), (262.43, '#ffc3c3'), (269.17, '#ffcbcb'),
              (273.66, '#ffd1d1'), (280.39, '#ffd9d9'), (283.39, '#ffdddd'), (290.13, '#ffe5e5'),
              (294.62, '#ffecec'), (299.86, '#fff2f2'), (304.35, '#fff7f7'), (307.34, '#fffdff'),
              (315.58, '#f8f8ff'), (320.82, '#f2f2ff'), (325.31, '#ededff'), (328.30, '#e8e8ff'),
              (333.54, '#e1e1ff'), (339.53, '#dadaff'), (344.77, '#d3d4ff'), (350.76, '#cacaff'),
              (360.49, '#babbff'), (373.22, '#acadff'), (384.45, '#9d9dff'), (398.67, '#8a8aff'),
              (413.64, '#7979ff'), (424.87, '#6e6fff'), (430.86, '#6666ff'), (439.10, '#5c5cff'),
              (447.33, '#5152ff'), (454.82, '#4a4aff'), (460.06, '#4343ff'), (466.05, '#3c3cff'),
              (471.29, '#3434ff'), (478.77, '#2829ff'), (488.50, '#181cff'), (501.23, '#080afe')]

# ---- panel b ------------------------------------------------------------
VF_RECTS = {  # (row, P) -> axes frame (left, right, top, bottom)
    ('MA', 1): (1090.04, 1301.70, 100.89, 312.55), ('MA', 2): (1351.28, 1562.94, 100.89, 312.55),
    ('MA', 3): (1612.97, 1824.63, 98.48, 310.14),  ('MA', 4): (1879.36, 2091.02, 98.48, 310.14),
    ('HA', 1): (1090.04, 1301.70, 331.67, 543.33), ('HA', 2): (1351.28, 1562.94, 331.67, 543.33),
    ('HA', 3): (1613.67, 1825.33, 331.67, 543.33), ('HA', 4): (1879.36, 2091.02, 331.67, 543.33),
}
VF_FILE = 'vector_field_{row}_P{p}_no_axes_no_mini_legend.pdf'
VF_LIM = (-4.0, 4.0)
VF_TICKS = [-4, 0, 4]
CB_GRAY = (2122.20, 2135.00, 122.25, 536.07)
CB_GRAY_BANDS = [(122.25, '#000000'), (128.37, '#070607'), (138.18, '#0f0e0d'), (149.07, '#181817'),
                 (160.50, '#1f1f1f'), (173.57, '#292928'), (183.37, '#303030'), (191.54, '#363636'),
                 (198.07, '#3c3c3c'), (204.06, '#434343'), (213.86, '#4b4b4b'), (220.39, '#504f4e'),
                 (223.66, '#535353'), (231.83, '#565656'), (236.73, '#5c5c5c'), (246.53, '#606060'),
                 (251.43, '#656565'), (257.97, '#686868'), (263.95, '#6c6c6c'), (268.86, '#6f6f6f'),
                 (275.39, '#747474'), (283.56, '#797979'), (288.46, '#7c7c7c'), (293.36, '#7f7f7f'),
                 (296.62, '#838383'), (306.43, '#888888'), (311.33, '#8c8c8c'), (317.86, '#909090'),
                 (322.22, '#939393'), (327.12, '#979797'), (333.65, '#9b9b9b'), (338.55, '#a0a0a0'),
                 (346.72, '#a5a5a5'), (351.62, '#a9a9a9'), (358.15, '#aeaeae'), (364.69, '#b2b2b2'),
                 (367.95, '#b4b5b4'), (371.22, '#b7b7b7'), (377.21, '#bebebe'), (385.38, '#c0c0c0'),
                 (391.91, '#c6c6c6'), (403.35, '#cbcbcb'), (411.51, '#cfcfcf'), (416.42, '#d4d4d4'),
                 (427.85, '#d9d9d9'), (437.11, '#dddddd'), (446.91, '#e1e1e1'), (453.44, '#e4e4e4'),
                 (464.88, '#e8e8e8'), (471.41, '#ececec'), (477.94, '#f0f0f0'), (489.38, '#f2f2f2'),
                 (498.63, '#f6f6f6'), (510.07, '#f9f9f9'), (523.14, '#fcfcfc'), (526.40, '#fffeff')]
CB_JET = (2221.10, 2236.85, 125.80, 534.30)
CB_JET_RANGE = (0.0, 4.0)
CB_JET_BANDS = [(125.80, '#9a020c'), (141.82, '#ba0a17'), (155.81, '#f10905'), (170.34, '#fd1f0f'),
                (181.65, '#fe3422'), (191.33, '#fd5308'), (206.40, '#fd7502'), (219.59, '#f89008'),
                (229.28, '#faa50f'), (238.97, '#f8bc0b'), (248.38, '#fbce0d'), (256.73, '#f8e50b'),
                (267.22, '#ecfa11'), (278.79, '#dafc26'), (290.10, '#c2fc3e'), (303.01, '#a9fb54'),
                (316.20, '#8bfc75'), (326.70, '#78ff80'), (338.00, '#5fff98'), (350.92, '#50fab2'),
                (362.49, '#2bfcd4'), (384.55, '#0fe6ee'), (395.59, '#09cff9'), (405.28, '#09b6fb'),
                (416.58, '#0397fe'), (429.50, '#0776fe'), (441.61, '#0158ff'), (454.52, '#0638fd'),
                (466.36, '#181cff'), (474.17, '#080afe'), (497.58, '#0101ce'), (515.61, '#04059a')]

# ---- panel c ------------------------------------------------------------
AX_C = (2498.47, 2965.10, 115.23, 533.30)
CB_C = (2989.60, 3010.70, 115.80, 537.00)
C_VMAX = 6.0
C_CBAR_TICKS = [-4, -2, 0, 2, 4]
THUMB_RECTS = [(2416.47, 2481.67, 110.65, 175.85), (2416.47, 2481.67, 183.30, 248.50),
               (2416.52, 2481.62, 255.94, 321.04), (2416.50, 2481.63, 328.49, 393.62),
               (2416.52, 2481.62, 401.07, 466.17), (2416.52, 2481.62, 473.61, 538.71)]
THUMB_FILES = [f'map_thumb_{k}.jpg' for k in range(1, 7)]
A_HEATMAP_FILE = 'figure5a_reward_heatmap_plain.pdf'     # heat map only (no lines / text)

# Guide lines drawn on the grid map (data coordinates of panel a/b axes)
GUIDE_LINE_STYLE = dict(color='black', linewidth=2.0, linestyle=(0, (6, 4)), alpha=0.9, zorder=4)
GUIDE_FONT_SIZE = 28
GUIDE_BOX = dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8)
GUIDE_HLINES = [   # (s value, label, label rho position)
    (0.05, 'Open map', 0.10),
    (0.50, 'PB map', 0.10),
]
GUIDE_VLINES = [   # (rho value, label, label s position = bottom end of the rotated label)
    (0.85, 'Mixed-abilities', 0.54),
]

# ---- free text: key -> (LaTeX, size pt, x, y, rotation); (x, y) = left/baseline anchor
RHO = r'$\boldsymbol{\rho}$'
TEXTS = {
    # panel a
    'a': (r'\textbf{a}', 60, 22.20, 64.09, 0),
    #'title_a': (r'\textbf{MARL Theoretical Solutions}', 46, 103.18, 40.88, 0),
    'xt_a_0.05': ('0.05', 34, 121.02, 560.00, 0), 'xt_a_0.25': ('0.25', 34, 197.12, 560.00, 0),
    'xt_a_0.40': ('0.40', 34, 273.14, 560.00, 0), 'xt_a_0.55': ('0.55', 34, 349.18, 560.00, 0),
    'xt_a_0.70': ('0.70', 34, 425.29, 560.00, 0), 'xt_a_0.85': ('0.85', 34, 500.61, 560.00, 0),
    'xt_a_1.00': ('1.00', 34, 575.05, 560.12, 0),
    'yt_a_1.00': ('1.00', 34, 71.97, 115.63, 0), 'yt_a_0.75': ('0.75', 34, 71.89, 221.35, 0),
    'yt_a_0.50': ('0.50', 34, 71.89, 325.75, 0), 'yt_a_0.25': ('0.25', 34, 71.89, 430.98, 0),
    'yt_a_0.05': ('0.05', 34, 72.03, 513.78, 0),
    'xlab_a': (rf'Mixed-ability parameter ({RHO})', 38, 141.87, 619.03, 0),
    'ylab_a': ('Switch cost (s)', 38.3, 48.92, 438.72, 90),
    'cbt_a_+0.8': ('+0.8', 30, 653.12, 138.48, 0), 'cbt_a_+0.4': ('+0.4', 30, 653.00, 229.52, 0),
    'cbt_a_0': ('0', 30, 662.76, 325.56, 0), 'cbt_a_-0.4': ('-0.4', 30, 652.86, 412.39, 0),
    'cbt_a_-0.8': ('-0.8', 30, 653.10, 501.18, 0),
    'cbl_a1': ('Score difference with respect', 38.25, 748.72, 576.84, 90),
    'cbl_a2': (rf'to {RHO} $\boldsymbol{{=1}}$ of that switch cost', 38, 794.24, 569.28, 90),
    # panel b
    'b': (r'\textbf{b}', 60, 917.34, 64.09, 0),
    #'title_b': (r'\textbf{MARL Theoretical Solutions}', 46, 1292.06, 40.88, 0),
    'P1': (r'\textbf{P1}', 34, 1068.73, 89.70, 0), 'P2': (r'\textbf{P2}', 34, 1326.04, 89.70, 0),
    'P3': (r'\textbf{P3}', 34, 1591.75, 89.82, 0), 'P4': (r'\textbf{P4}', 34, 1857.37, 89.58, 0),
    'row_MA1': (r'\textbf{Mixed}', 38, 921.88, 261.96, 90),
    'row_MA2': (r'\textbf{abilities}', 38, 965.28, 275.04, 90),
    #'row_MA3': (r'\textbf{(}$\boldsymbol{\rho = 0.85}$\textbf{)}', 39.83, 970.24, 266.04, 90),
    'row_HA1': (r'\textbf{High}', 38, 921.88, 486.60, 90),
    'row_HA2': (r'\textbf{abilities}', 38, 965.28, 513.24, 90),
    #'row_HA3': (r'\textbf{(}$\boldsymbol{\rho = 1}$\textbf{)}', 39.83, 970.24, 504.12, 90),
    'sw_MA': (r'Switching ($a$)', 38.62, 1020.00, 311.52, 90),
    'sw_HA': (r'Switching ($a$)', 38.58, 1020.00, 549.48, 90),
    'yt_MA_+4': ('+4', 34, 1038.04, 122.94, 0), 'yt_MA_0': ('0', 34, 1058.94, 216.61, 0),
    'yt_MA_-4': ('-4', 34, 1050.42, 311.70, 0),
    'yt_HA_+4': ('+4', 34, 1038.58, 356.38, 0), 'yt_HA_0': ('0', 34, 1058.91, 449.77, 0),
    'yt_HA_-4': ('-4', 34, 1052.70, 543.90, 0),
    'xt_P1_-4': ('-4', 34, 1078.38, 578.62, 0), 'xt_P1_0': ('0', 34, 1188.63, 578.74, 0),
    'xt_P1_+4': ('+4', 34, 1268.44, 578.62, 0),
    'xt_P2_-4': ('-4', 34, 1337.46, 578.62, 0), 'xt_P2_0': ('0', 34, 1447.56, 578.74, 0),
    'xt_P2_+4': ('+4', 34, 1527.37, 578.62, 0),
    'xt_P3_-4': ('-4', 34, 1600.89, 578.62, 0), 'xt_P3_0': ('0', 34, 1711.14, 578.74, 0),
    'xt_P3_+4': ('+4', 34, 1790.98, 578.62, 0),
    'xt_P4_-4': ('-4', 34, 1867.14, 578.62, 0), 'xt_P4_0': ('0', 34, 1977.18, 578.74, 0),
    'xt_P4_+4': ('+4', 34, 2057.11, 578.62, 0),
    'xlab_b': (r'Specialization ($\boldsymbol{x}$)', 38, 1477.16, 622.08, 0),
    'local': ('Local reward', 38, 2179.24, 428.52, 90),
    'high': ('high', 34, 2145.33, 141.64, 0), 'low': ('low', 34, 2144.85, 548.21, 0),
    'final': (r'Final trajectory specialization ($x_{10}$)', 38, 2329.16, 598.44, 90),
    'jt_0.0': ('0.0', 34, 2246.32, 544.92, 0), 'jt_1.0': ('1.0', 34, 2246.32, 442.38, 0),
    'jt_2.0': ('2.0', 34, 2246.43, 339.87, 0), 'jt_3.0': ('3.0', 34, 2246.32, 237.25, 0),
    'jt_4.0': ('4.0', 34, 2246.32, 134.61, 0),
    # panel c
    'c': (r'\textbf{c}', 60, 2438.46, 84.09, 0),
    'title_c': (r'\textbf{MARL Simulations}', 48, 2551.66, 49.90, 0),
    'xt_c_0.2': ('0.2', 34, 2548.56, 569.94, 0), 'xt_c_0.4': ('0.4', 34, 2639.64, 569.82, 0),
    'xt_c_0.6': ('0.6', 34, 2734.50, 569.94, 0), 'xt_c_0.8': ('0.8', 34, 2827.64, 569.94, 0),
    'xt_c_1.0': ('1.0', 34, 2921.40, 569.94, 0),
    'xlab_c': (rf'Mixed-ability parameter ({RHO})', 38, 2497.98, 619.03, 0),
    'cbt_c_+4': ('+4', 34, 3021.19, 194.08, 0), 'cbt_c_+2': ('+2', 34, 3022.57, 267.09, 0),
    'cbt_c_0': ('0', 34, 3030.39, 339.49, 0), 'cbt_c_-2': ('-2', 34, 3026.07, 405.75, 0),
    'cbt_c_-4': ('-4', 34, 3024.93, 476.58, 0),
    'cbl_c1': ('Score difference with respect', 38.29, 3108.24, 578.04, 90),
    'cbl_c2': (rf'to {RHO} $\boldsymbol{{=1}}$ of that column', 38, 3154.00, 539.04, 90),
}


# ===========================================================================
# 4. STYLE
# ===========================================================================
def configure_style():
    """Computer Modern scaled from the 10 pt design size (as in the LaTeX labels of Fig5)."""
    plt.style.use('default')
    mpl.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'text.latex.preamble': r"""
            \usepackage{amsmath}\usepackage{bm}
            \DeclareFontShape{OT1}{cmr}{m}{n}{<->cmr10}{}
            \DeclareFontShape{OT1}{cmr}{bx}{n}{<->cmbx10}{}
            \DeclareFontShape{OT1}{cmr}{m}{it}{<->cmti10}{}
            \DeclareFontShape{OT1}{cmss}{bx}{n}{<->cmssbx10}{}
            \DeclareFontShape{OML}{cmm}{m}{it}{<->cmmi10}{}
            \DeclareFontShape{OML}{cmm}{b}{it}{<->cmmib10}{}
            \DeclareFontShape{OMS}{cmsy}{m}{n}{<->cmsy10}{}
            \DeclareFontShape{OMS}{cmsy}{b}{n}{<->cmbsy10}{}
        """,
        'axes.unicode_minus': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
    })


# ===========================================================================
# 5. LAYOUT VERSIONS
# ===========================================================================
# The coordinates above are those of the original 3-panel Fig5.pdf. The other
# layouts are obtained by moving whole groups of elements horizontally, so
# every element keeps exactly its original size, font and relative position.
#
#   'v2'  (default) main figure:  a = flow maps P1 (Open map) & P2 (PB map),
#                                 b = theoretical grid map with points 1 & 2
#   'sm'  supplementary figure:   MARL-simulations grid (old panel c)
#   'original'                    the 3-panel Fig5 as submitted
GRID_KEYS = ['a', 'title_a', 'xlab_a', 'ylab_a', 'cbl_a1', 'cbl_a2']       # + xt_a_*, yt_a_*, cbt_a_*
FLOW_CB_KEYS = ['local', 'high', 'low', 'final']                            # + jt_*
SIM_KEYS = ['c', 'title_c', 'xlab_c', 'cbl_c1', 'cbl_c2']                   # + xt_c_*, cbt_c_*


def _group_of(key):
    if key in GRID_KEYS or key.startswith(('xt_a_', 'yt_a_', 'cbt_a_')):
        return 'grid'
    if key in SIM_KEYS or key.startswith(('xt_c_', 'cbt_c_')):
        return 'sim'
    if key in FLOW_CB_KEYS or key.startswith('jt_'):
        return 'flow_cb'
    return 'flow'


# --- v2 -------------------------------------------------------------------
V2_COLUMNS = [1, 2]                          # P1 = Open map, P2 = PB map
V2_POINTS = ['1', '2']                       # numbered points kept on the grid map
V2_COLUMN_TITLES = {1: r'\textbf{Open map}', 2: r'\textbf{PB map}'}
V2_DX_FLOW = -837.5                          # flow-map group -> left margin (20 pt)
V2_DX_FLOW_CB = V2_DX_FLOW - (VF_RECTS[('MA', 4)][0] - VF_RECTS[('MA', 2)][0])   # colour bars next to P2
V2_DX_GRID = 996.4                           # grid-map group to the right of the flow maps
V2_PAGE_W = 1818.5
_v2_axes_centre = (VF_RECTS[('MA', 1)][0] + VF_RECTS[('MA', 2)][1]) / 2 + V2_DX_FLOW
V2_OVERRIDES = {          # key -> (text or None, x or None[, ha]) ; x = new anchor (left unless ha given)
    'b': (r'\textbf{a}', None),                                    # flow maps become panel a
    'a': (r'\textbf{b}', None),                                    # grid map becomes panel b
    'title_b': (None, _v2_axes_centre - 667.1 / 2 - 0.34),         # centred over P1-P2
    'xlab_b': (None, _v2_axes_centre - 286.8 / 2 - 3.14),
    # column titles centred over their column of flow maps
    'P1': (V2_COLUMN_TITLES[1], (VF_RECTS[('MA', 1)][0] + VF_RECTS[('MA', 1)][1]) / 2 + V2_DX_FLOW, 'center'),
    'P2': (V2_COLUMN_TITLES[2], (VF_RECTS[('MA', 2)][0] + VF_RECTS[('MA', 2)][1]) / 2 + V2_DX_FLOW, 'center'),
}

# --- sm -------------------------------------------------------------------
SM_DX = -(THUMB_RECTS[0][0] - 20.0)
SM_PAGE_W = 3161.76 + SM_DX + 20.0


# ===========================================================================
# 6. FIGURE ASSEMBLY
# ===========================================================================
def _shift(rect, dx):
    L, R, T, B = rect
    return (L + dx, R + dx, T, B)


def _fig_rect(rect, page_w):
    L, R, T, B = rect
    return [L / page_w, 1.0 - B / PAGE_H, (R - L) / page_w, (B - T) / PAGE_H]


def _page_transform(fig):
    return Affine2D().scale(1.0, -1.0).translate(0.0, PAGE_H).scale(1.0 / 72.0) + fig.dpi_scale_trans


def _banded_colorbar(fig, rect, page_w, bands, vmin, vmax, ticks, tick_len, tick_w=0.8, frame_lw=0.8):
    """Colour bar made of the measured bands (top y, colour), mapped linearly y <-> value."""
    L, R, T, B = rect
    y2v = lambda y: vmax - (y - T) / (B - T) * (vmax - vmin)
    bounds = [y2v(y) for y, _ in bands] + [vmin]          # descending
    colors = [c for _, c in bands]
    cmap = mcolors.ListedColormap(colors[::-1])
    norm = mcolors.BoundaryNorm(bounds[::-1], cmap.N)
    cax = fig.add_axes(_fig_rect(rect, page_w))
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=bounds[::-1],
                                   orientation='vertical', ticks=ticks)
    cb.ax.set_ylim(vmin, vmax)
    cb.ax.set_yticklabels([])
    cb.ax.tick_params(length=tick_len, width=tick_w, color='black')
    cb.outline.set_linewidth(frame_lw)
    return cb


def _clean_axes(ax, xticks, yticks, frame_lw=0.8):
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(labelbottom=False, labelleft=False, length=3.5, width=0.8, color='black')
    ax.set_facecolor('none')
    for sp in ax.spines.values():
        sp.set_linewidth(frame_lw)
        sp.set_edgecolor('black')
        sp.set_zorder(10)


def _draw_guide_lines(ax):
    """Dashed lines at the Open (s = 0.05) and PB (s = 0.5) switch costs and at the
    mixed-ability rho = 0.85, labelled in the figure font (Computer Modern)."""
    for s_val, label, x_lab in GUIDE_HLINES:
        ax.axhline(s_val, **GUIDE_LINE_STYLE)
        ax.annotate(label, xy=(x_lab, s_val), xytext=(0, 8), textcoords='offset points',
                    fontsize=GUIDE_FONT_SIZE, ha='left', va='bottom', zorder=6,
                    bbox=GUIDE_BOX, annotation_clip=False)
    for rho_val, label, y_lab in GUIDE_VLINES:
        ax.axvline(rho_val, **GUIDE_LINE_STYLE)
        ax.annotate(label, xy=(rho_val, y_lab), xytext=(8, 0), textcoords='offset points',
                    fontsize=GUIDE_FONT_SIZE, rotation=90, ha='left', va='top',   # text right of the line
                    rotation_mode='anchor', zorder=6, bbox=GUIDE_BOX, annotation_clip=False)


def _draw_grid_map(fig, page, page_w, dx, figures_dir, embeds, points, guide_lines=False):
    """Theoretical score-difference grid map (heat-map PDF + axes, colour bar, guide lines)."""
    ax = fig.add_axes(_fig_rect(_shift(AX_A, dx), page_w))
    ax.set_xlim(*A_XLIM)
    ax.set_ylim(*A_YLIM)
    _clean_axes(ax, A_XTICKS, A_YTICKS, frame_lw=A_FRAME_LW)
    embeds.append((figures_dir / 'subfigures' / A_HEATMAP_FILE, _shift(A_HEATMAP_RECT, dx)))
    if guide_lines:
        _draw_guide_lines(ax)
    #for lab in points:
    #    rho, s, lx, ly = A_POINTS[lab]
    #    ax.scatter([rho], [s], s=120, c='black', edgecolors='white', linewidths=2.2,
    #               zorder=5, clip_on=False)
    #    fig.text(lx + dx, ly, rf'\textsf{{\textbf{{{lab}}}}}', transform=page, fontsize=52,
    #             ha='center', va='center', zorder=6,
    #             bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.8))
    _banded_colorbar(fig, _shift(CB_A, dx), page_w, CB_A_BANDS, -CB_A_VMAX, CB_A_VMAX,
                     CB_A_TICKS, tick_len=3.5)


def _draw_flow_maps(fig, page_w, dx, dx_cb, figures_dir, embeds, columns):
    """Vector-field flow maps (PDFs + axes) and their two colour bars."""
    for (row, p), rect in VF_RECTS.items():
        if p not in columns:
            continue
        r = _shift(rect, dx)
        embeds.append((figures_dir / 'subfigures' / VF_FILE.format(row=row, p=p), r))
        ax = fig.add_axes(_fig_rect(r, page_w))
        ax.set_xlim(*VF_LIM)
        ax.set_ylim(*VF_LIM)
        _clean_axes(ax, VF_TICKS, VF_TICKS)
    _banded_colorbar(fig, _shift(CB_GRAY, dx_cb), page_w, CB_GRAY_BANDS, 0.0, 1.0, [0.0, 1.0], tick_len=2.0)
    _banded_colorbar(fig, _shift(CB_JET, dx_cb), page_w, CB_JET_BANDS, *CB_JET_RANGE,
                     [0, 1, 2, 3, 4], tick_len=3.0)


def _draw_sim_grid(fig, page_w, dx, grid_c, figures_dir):
    """MARL-simulations map x ability grid (old panel c) with map thumbnails."""
    ax = fig.add_axes(_fig_rect(_shift(AX_C, dx), page_w))
    im = ax.imshow(grid_c, cmap=plt.cm.RdBu_r, vmin=-C_VMAX, vmax=C_VMAX,
                   interpolation='nearest', aspect='auto')
    ax.invert_xaxis()                      # rho = 0.1 on the left, 1.0 on the right
    _clean_axes(ax, range(grid_c.shape[1]), range(grid_c.shape[0]))
    ax.set_facecolor('white')
    cax = fig.add_axes(_fig_rect(_shift(CB_C, dx), page_w))
    cb = fig.colorbar(im, cax=cax, ticks=C_CBAR_TICKS)
    cb.ax.set_yticklabels([])
    cb.ax.tick_params(length=3.5, width=0.8)
    cb.outline.set_linewidth(0.8)
    for rect, fn in zip(THUMB_RECTS, THUMB_FILES):
        p = figures_dir / 'subfigures' / fn
        if p.exists():
            tax = fig.add_axes(_fig_rect(_shift(rect, dx), page_w))
            tax.imshow(plt.imread(str(p)), aspect='auto', interpolation='nearest')
            tax.set_axis_off()
        else:
            print(f'[fig5] map thumbnail not found: {p}')


def _draw_texts(fig, page, keep, shifts, overrides=None):
    """Free text: keep(key) -> bool, shifts[group] -> dx, overrides[key] -> (text|None, x|None)."""
    overrides = overrides or {}
    for key, (tex, size, x, y, rot) in TEXTS.items():
        if not keep(key):
            continue
        x = x + shifts[_group_of(key)]
        ov = overrides.get(key, (None, None))
        new_tex, new_x = ov[0], ov[1]
        ha = ov[2] if len(ov) > 2 else 'left'
        tex = new_tex if new_tex is not None else tex
        x = new_x if new_x is not None else x
        fig.text(x, y, tex, transform=page, fontsize=size, ha=ha, va='baseline',
                 rotation=rot, rotation_mode='anchor', zorder=20)


def build_figure(version: str, figures_dir: Path, grid_c: Optional[np.ndarray] = None,
                 draw_points: bool = True):
    """Returns (figure, list of (pdf path, page rect) to embed underneath, page width)."""
    configure_style()
    embeds = []

    if version == 'v2':
        page_w = V2_PAGE_W
        fig = plt.figure(figsize=(page_w / 72.0, PAGE_H / 72.0))
        page = _page_transform(fig)
        _draw_flow_maps(fig, page_w, V2_DX_FLOW, V2_DX_FLOW_CB, figures_dir, embeds, V2_COLUMNS)
        _draw_grid_map(fig, page, page_w, V2_DX_GRID, figures_dir, embeds,
                       V2_POINTS if draw_points else [], guide_lines=True)
        dropped = [f'P{p}' for p in (1, 2, 3, 4) if p not in V2_COLUMNS]

        def keep(key):
            g = _group_of(key)
            if g == 'sim':
                return False
            return not any(key == d or key.startswith(f'xt_{d}_') for d in dropped)
        _draw_texts(fig, page, keep, {'grid': V2_DX_GRID, 'flow': V2_DX_FLOW,
                                      'flow_cb': V2_DX_FLOW_CB, 'sim': 0.0}, V2_OVERRIDES)

    elif version == 'sm':
        page_w = SM_PAGE_W
        fig = plt.figure(figsize=(page_w / 72.0, PAGE_H / 72.0))
        page = _page_transform(fig)
        _draw_sim_grid(fig, page_w, SM_DX, grid_c, figures_dir)
        _draw_texts(fig, page, lambda k: _group_of(k) == 'sim' and k != 'c',
                    {'grid': 0.0, 'flow': 0.0, 'flow_cb': 0.0, 'sim': SM_DX})

    elif version == 'original':
        page_w = PAGE_W
        fig = plt.figure(figsize=(page_w / 72.0, PAGE_H / 72.0))
        page = _page_transform(fig)
        _draw_grid_map(fig, page, page_w, 0.0, figures_dir, embeds,
                       list(A_POINTS) if draw_points else [])
        _draw_flow_maps(fig, page_w, 0.0, 0.0, figures_dir, embeds, [1, 2, 3, 4])
        _draw_sim_grid(fig, page_w, 0.0, grid_c, figures_dir)
        _draw_texts(fig, page, lambda k: True, {'grid': 0.0, 'flow': 0.0, 'flow_cb': 0.0, 'sim': 0.0})
    else:
        raise ValueError(f'unknown version {version!r}')
    return fig, embeds, page_w


def save_figure(fig, embeds, output_pdf: Path, page_w: float, png_dpi: int = 300):
    """Save the matplotlib layer and merge the vector panel PDFs underneath (keeps them vector)."""
    try:
        from pypdf import PdfReader, PdfWriter, PageObject, Transformation
    except ImportError:
        raise SystemExit("pypdf is required to embed the panel PDFs:  pip install pypdf")

    tmp = output_pdf.with_suffix('.layer.pdf')
    fig.savefig(tmp, transparent=True)           # NB: no bbox_inches='tight'
    plt.close(fig)

    base = PageObject.create_blank_page(width=page_w, height=PAGE_H)
    for path, (L, R, T, B) in embeds:
        if not Path(path).exists():
            print(f'[fig5] WARNING: panel PDF not found: {path}')
            continue
        src = PdfReader(str(path)).pages[0]
        mb = src.mediabox
        sx = (R - L) / float(mb.width)
        sy = (B - T) / float(mb.height)
        tx = L - float(mb.left) * sx
        ty = (PAGE_H - B) - float(mb.bottom) * sy
        base.merge_transformed_page(src, Transformation().scale(sx, sy).translate(tx, ty))
    base.merge_page(PdfReader(str(tmp)).pages[0])

    writer = PdfWriter()
    writer.add_page(base)
    with open(output_pdf, 'wb') as fh:
        writer.write(fh)
    tmp.unlink()
    print(f'[fig5] saved {output_pdf}')

    png = output_pdf.with_suffix('.png')
    try:
        import subprocess
        subprocess.run(['pdftoppm', '-r', str(png_dpi), '-png', '-singlefile', str(output_pdf),
                        str(png.with_suffix(''))], check=True, capture_output=True)
        print(f'[fig5] saved {png}')
    except Exception:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(output_pdf))
            doc[0].get_pixmap(dpi=png_dpi).save(str(png))
            print(f'[fig5] saved {png}')
        except Exception:
            print('[fig5] PNG not written (needs poppler "pdftoppm" or PyMuPDF)')


# ===========================================================================
# 7. CLI
# ===========================================================================
def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Figure 5 (flow maps + theoretical grid map) and its SM figure (MARL simulations grid)',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--version', choices=['v2', 'original'], default='v2',
                        help="v2: a = flow maps P1/P2, b = grid map (default); original: 3-panel Fig5")
    parser.add_argument('--sm', action='store_true',
                        help='Also build the SM figure with the MARL-simulations grid (needs training data, '
                             '--grid_csv or --demo)')
    parser.add_argument('--episode_range', choices=['all', 'final', 'average', 'specific'], default='final')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--target_episode', type=int, default=None)
    parser.add_argument('--study_name', type=str, default=None)
    parser.add_argument('--map_name_1', type=str, default='baseline_division_of_labor_large')
    parser.add_argument('--map_name_2', type=str, default='encouraged_division_of_labor_large')
    parser.add_argument('--game_type', type=str, choices=['classic', 'classic_collision'], default='classic_collision')
    parser.add_argument('--init_type', type=str, choices=['random_init', 'empty_init'], default='empty_init')
    parser.add_argument('--synergy', type=float, default=0.0)
    parser.add_argument('--specialization', type=float, default=None)
    parser.add_argument('--cluster', type=str, default=None)
    parser.add_argument('--figures_dir', type=str, default='./figures',
                        help='Folder with the panel PDFs (heat map, vector fields) and map thumbnails')
    parser.add_argument('--output_dir', type=str, default='./figures')
    parser.add_argument('--output_name', type=str, default='fig5_theory.pdf')
    parser.add_argument('--sm_output_name', type=str, default='figS_marl_simulations_grid.pdf')
    parser.add_argument('--grid_csv', type=str, default=None,
                        help='Load the MARL-simulations grid from this CSV instead of the training data')
    parser.add_argument('--no_points', action='store_true', help='Do not draw the numbered points on the grid map')
    parser.add_argument('--demo', action='store_true', help='Synthetic MARL-simulations grid (layout test)')
    return parser


def load_sim_grid(args, output_dir: Path) -> np.ndarray:
    """MARL-simulations grid: synthetic (--demo), cached CSV (--grid_csv) or from the training data."""
    if args.demo:
        rng = np.random.default_rng(0)
        grid = np.clip(rng.normal(0.5, 1.5, (12, 10)), -6, 6)
        grid[:, 0] = 0.0
        grid[:, -1] = -5.5
        return grid
    if args.grid_csv:
        return np.loadtxt(args.grid_csv, delimiter=',')

    analyzer = CooperativeAnalyzer(
        study_name=args.study_name, map_name_1=args.map_name_1, map_name_2=args.map_name_2,
        init_type=args.init_type, synergy=args.synergy, synergy_provided='--synergy' in sys.argv,
        specialization=args.specialization, game_type=args.game_type,
        cluster=args.cluster if args.cluster else 'cuenca')
    df_or_dict = analyzer.load_experimental_data()
    if isinstance(df_or_dict, dict):
        key = sorted(df_or_dict)[0]
        print(f'[fig5] several synergy/specialization combinations found; using {key} '
              f'(select one with --synergy / --specialization)')
        df = df_or_dict[key]
    else:
        df = df_or_dict
    prepared = analyzer.prepare_episode_data(df, episode_selection=args.episode_range,
                                             num_episodes=args.num_episodes,
                                             target_episode=args.target_episode)
    grid = row_score_difference_grid(prepared, analyzer.map_names, analyzer.ability_configs,
                                     args.game_type)
    grid_csv = output_dir / 'fig5c_grid.csv'
    np.savetxt(grid_csv, grid, delimiter=',')
    print(f'[fig5] MARL-simulations grid cached -> {grid_csv}  (re-use with --grid_csv)')
    return grid


def main():
    args = setup_argument_parser().parse_args()
    figures_dir = Path(args.figures_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    need_grid = args.version == 'original' or args.sm
    grid = load_sim_grid(args, output_dir) if need_grid else None

    fig, embeds, page_w = build_figure(args.version, figures_dir, grid, draw_points=not args.no_points)
    save_figure(fig, embeds, output_dir / args.output_name, page_w)

    if args.sm and args.version != 'original':
        fig, embeds, page_w = build_figure('sm', figures_dir, grid)
        save_figure(fig, embeds, output_dir / args.sm_output_name, page_w)


if __name__ == '__main__':
    main()