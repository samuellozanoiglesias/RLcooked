#!/usr/bin/env python3
"""
Grid Search Analysis Figure Generator

This script creates comprehensive visualizations for a large-scale grid search across:
- 4 maps: baseline, encouraged, semiencouraged, corridor
- 2 collision settings: Yes/No
- 25 speed pair combinations

The visualization approaches include:
- Heatmaps for performance across speed combinations
- Faceted plots by map and collision setting
- Statistical summaries and comparisons

Usage:
    python figure_gridsearch_analysis.py [options]

Examples:
    # Full grid search analysis
    nohup python figure_gridsearch_analysis.py --episode_range final --num_final_episodes 100 > gridsearch_analysis.log 2>&1 &
    
    # Focus on specific maps
    nohup python figure_gridsearch_analysis.py --maps baseline encouraged --episode_range final > gridsearch_subset.log 2>&1 &
    
    # Collision comparison only
    nohup python figure_gridsearch_analysis.py --collision_only --episode_range final > gridsearch_collision.log 2>&1 &
    
    # Analyze only empty initialization
    nohup python figure_gridsearch_analysis.py --inits empty_init --episode_range final > gridsearch_empty_init.log 2>&1 &
    
    # Compare initialization types for baseline map
    nohup python figure_gridsearch_analysis.py --maps baseline --inits empty_init random_init --episode_range final > gridsearch_baseline_init.log 2>&1 &
    
    # Speed analysis for random initialization only
    nohup python figure_gridsearch_analysis.py --inits random_init --speed_analysis_only --episode_range final > gridsearch_random_speed.log 2>&1 &
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
from itertools import product

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spoiled_broth.analysis.utils import DataProcessor, AnalysisConfig


class GridSearchAnalyzer:
    """Handles data loading and processing for grid search analysis."""
    
    def __init__(self, study_name: Optional[str] = None, cluster: str = 'cuenca'):
        self.config = AnalysisConfig()
        self.data_processor = DataProcessor(self.config)
        self.study_name = study_name
        self.cluster = cluster
        
        # Define grid search parameters
        self.maps = ['baseline', 'encouraged', 'semiencouraged', 'corridor']
        self.collision_settings = ['classic', 'classic_collision']  # No collision / With collision
        self.init_types = ['empty_init', 'random_init']  # Initialization types
        
        # Parse speed pairs from the actual grid search specification
        # Format: "cutting_speed1,walking_speed1 cutting_speed2,walking_speed2"
        # Agent1: cutting_speed = 1.0 (fixed), walking_speed varies (0.1-1.0)
        # Agent2: walking_speed = 1.0 (fixed), cutting_speed varies (0.1-1.0)
        
        walking_speeds_1 = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]  # Agent 1 walking speeds
        cutting_speeds_2 = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]   # Agent 2 cutting speeds
        
        self.speed_pairs = []
        for walk_1 in walking_speeds_1:
            for cut_2 in cutting_speeds_2:
                # (walking_speed_1, cutting_speed_1, walking_speed_2, cutting_speed_2)
                self.speed_pairs.append((walk_1, 1.0, 1.0, cut_2))
        
        # Create speed pair labels for plotting
        self.speed_labels = [f"{w1:.2f},{c1:.2f}_{w2:.2f},{c2:.2f}" for w1, c1, w2, c2 in self.speed_pairs]
        
        # Create full map names for directory lookup
        self.map_full_names = {
            'baseline': 'baseline_division_of_labor_large',
            'encouraged': 'encouraged_division_of_labor_large', 
            'semiencouraged': 'semiencouraged_division_of_labor_large',
            'corridor': 'corridor_division_of_labor_large_v2'
        }
        
        # Performance metrics to analyze
        self.performance_metrics = {
            'total_deliveries': 'Total Deliveries',
            'total_cuts': 'Total Cuts',
            'pure_reward_total': 'Total Reward',
            'deliver_ai_rl_1': 'Deliveries Agent 1',
            'deliver_ai_rl_2': 'Deliveries Agent 2',
            'cut_ai_rl_1': 'Cuts Agent 1',
            'cut_ai_rl_2': 'Cuts Agent 2'
        }
        
        # Extended metrics for detailed analysis
        self.extended_metrics = {
            'salad_ai_rl_1': 'Salad Agent 1',
            'salad_ai_rl_2': 'Salad Agent 2',
            'plate_ai_rl_1': 'Plate Agent 1',
            'plate_ai_rl_2': 'Plate Agent 2',
            'raw_food_ai_rl_1': 'Raw Food Agent 1',
            'raw_food_ai_rl_2': 'Raw Food Agent 2'
        }
    
    def load_grid_search_data(self, selected_maps: Optional[List[str]] = None, 
                             selected_collisions: Optional[List[str]] = None,
                             selected_inits: Optional[List[str]] = None) -> pd.DataFrame:
        """Load and combine data from all grid search conditions."""
        print("Loading grid search data from all conditions...")
        if self.study_name:
            print(f"Using study name: {self.study_name}")
        
        all_data = []
        maps_to_process = selected_maps if selected_maps else self.maps
        collisions_to_process = selected_collisions if selected_collisions else self.collision_settings
        inits_to_process = selected_inits if selected_inits else self.init_types
        
        total_conditions = len(maps_to_process) * len(collisions_to_process) * len(inits_to_process) * len(self.speed_pairs)
        print(f"Processing {total_conditions} total experimental conditions...")
        
        condition_count = 0
        
        # Load data for each experimental condition
        for map_name, collision_setting, init_type in product(maps_to_process, collisions_to_process, inits_to_process):
            full_map_name = self.map_full_names[map_name]
            
            print(f"\nProcessing map: {map_name} ({full_map_name}), collision: {collision_setting}, init: {init_type}")
            
            try:
                # Set up paths for this map/collision/init combination
                experiment_type_with_init = f"{collision_setting}/{init_type}"
                paths = self.data_processor.setup_directories(
                    experiment_type=experiment_type_with_init,
                    map_name=full_map_name,
                    cluster=self.cluster,
                    study_name=self.study_name
                )
                
                # Load the base data (using 2 agents since this is multi-agent cooperative data)
                base_df = self.data_processor.load_experiment_data(paths, num_agents=2)
                
                if base_df is not None and len(base_df) > 0:
                    print(f"  Loaded base data shape: {base_df.shape}")
                    print(f"  All columns: {list(base_df.columns)}")
                    
                    # Debug: Show what speed columns exist
                    speed_related_cols = [col for col in base_df.columns if 'speed' in col.lower()]
                    print(f"  Speed-related columns found: {speed_related_cols}")
                    
                    # Debug: Show what speed combinations are actually in the data
                    speed_cols = ['walking_speed_1', 'cutting_speed_1', 'walking_speed_2', 'cutting_speed_2']
                    if all(col in base_df.columns for col in speed_cols):
                        unique_speeds = base_df[speed_cols].drop_duplicates()
                        print(f"  Found {len(unique_speeds)} unique speed combinations in data:")
                        for idx, row in unique_speeds.head(10).iterrows():  # Show first 10
                            print(f"    walk1={row['walking_speed_1']:.3f}, cut1={row['cutting_speed_1']:.3f}, walk2={row['walking_speed_2']:.3f}, cut2={row['cutting_speed_2']:.3f}")
                        if len(unique_speeds) > 10:
                            print(f"    ... and {len(unique_speeds) - 10} more")
                    else:
                        missing_cols = [col for col in speed_cols if col not in base_df.columns]
                        print(f"  Missing speed columns: {missing_cols}")
                        print(f"  Available speed-related columns: {speed_related_cols}")
                        
                        # Try alternative column names
                        alt_speed_patterns = ['speed', 'walk', 'cut']
                        for pattern in alt_speed_patterns:
                            matching_cols = [col for col in base_df.columns if pattern in col.lower()]
                            if matching_cols:
                                print(f"  Columns containing '{pattern}': {matching_cols}")
                        
                        # Skip this map/collision combination if no speed data
                        print(f"  Skipping {map_name}+{collision_setting} - no speed data found")
                        continue
                    
                    # Process each speed pair for this map/collision combination
                    for i, (walk_1, cut_1, walk_2, cut_2) in enumerate(self.speed_pairs):
                        condition_count += 1
                        speed_label = self.speed_labels[i]
                        
                        print(f"    Processing speed pair {condition_count}/{total_conditions}: {speed_label}")
                        
                        # Filter data for this specific speed configuration
                        df_filtered = self._filter_by_speed_exact(base_df, walk_1, cut_1, walk_2, cut_2)
                        
                        if len(df_filtered) == 0:
                            print(f"      Warning: No data found for speed {speed_label}")
                            continue
                        
                        print(f"      Found {len(df_filtered)} episodes")
                        
                        # Create combined metrics
                        self._create_combined_metrics(df_filtered)
                        
                        # Add condition metadata
                        df_filtered = df_filtered.copy()
                        df_filtered['map_name'] = map_name
                        df_filtered['map_full_name'] = full_map_name
                        df_filtered['collision_setting'] = collision_setting
                        df_filtered['collision_enabled'] = collision_setting == 'classic_collision'
                        df_filtered['init_type'] = init_type
                        df_filtered['speed_pair'] = speed_label
                        df_filtered['walk_speed_1'] = walk_1
                        df_filtered['cut_speed_1'] = cut_1
                        df_filtered['walk_speed_2'] = walk_2
                        df_filtered['cut_speed_2'] = cut_2
                        
                        # Create comprehensive condition identifier
                        df_filtered['condition'] = f"{map_name}_{collision_setting}_{init_type}_{speed_label}"
                        
                        all_data.append(df_filtered)
                        print(f"      ✓ Added {len(df_filtered)} episodes for condition: {map_name}_{collision_setting}_{speed_label}")
                        print(f"      ✓ Added {len(df_filtered)} episodes for condition: {map_name}_{collision_setting}_{speed_label}")
                
                else:
                    print(f"  Warning: No base data loaded for {map_name} + {collision_setting}")
                    
            except Exception as e:
                print(f"  Error loading data for {map_name} + {collision_setting}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data could be loaded from any experimental condition")
            
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal loaded data: {len(combined_df)} episodes across {combined_df['condition'].nunique()} conditions")
        print(f"Maps found: {sorted(combined_df['map_name'].unique())}")
        print(f"Collision settings: {sorted(combined_df['collision_setting'].unique())}")
        print(f"Initialization types: {sorted(combined_df['init_type'].unique())}")
        print(f"Speed pairs: {len(combined_df['speed_pair'].unique())}")
        
        return combined_df
    
    def _filter_by_speed_exact(self, df: pd.DataFrame, walk_1: float, cut_1: float, 
                               walk_2: float, cut_2: float) -> pd.DataFrame:
        """Filter data to exact speed configuration with tolerance."""
        if len(df) == 0:
            return df
            
        required_cols = ['walking_speed_1', 'cutting_speed_1', 'walking_speed_2', 'cutting_speed_2']
        if not all(col in df.columns for col in required_cols):
            print(f"      Missing speed columns. Required: {required_cols}")
            print(f"      Available: {[col for col in df.columns if 'speed' in col.lower()]}")
            return pd.DataFrame()
        
        tolerance = 0.05  # Increased tolerance
        filter_condition = (
            (abs(df['walking_speed_1'] - walk_1) < tolerance) &
            (abs(df['cutting_speed_1'] - cut_1) < tolerance) &
            (abs(df['walking_speed_2'] - walk_2) < tolerance) &
            (abs(df['cutting_speed_2'] - cut_2) < tolerance)
        )
        
        filtered_df = df[filter_condition].copy()
        
        # Debug: If no match, show what speeds are actually present
        if len(filtered_df) == 0:
            print(f"        No match for target: walk1={walk_1:.3f}, cut1={cut_1:.3f}, walk2={walk_2:.3f}, cut2={cut_2:.3f}")
            unique_speeds = df[required_cols].drop_duplicates()
            print(f"        Available {len(unique_speeds)} speed combinations:")
            for idx, row in unique_speeds.head(5).iterrows():
                print(f"          walk1={row['walking_speed_1']:.3f}, cut1={row['cutting_speed_1']:.3f}, walk2={row['walking_speed_2']:.3f}, cut2={row['cutting_speed_2']:.3f}")
            if len(unique_speeds) > 5:
                print(f"          ... and {len(unique_speeds) - 5} more")
        else:
            print(f"        ✓ Found {len(filtered_df)} episodes for target speeds")

        return filtered_df
    
    def _create_combined_metrics(self, df: pd.DataFrame):
        """Create combined metrics from individual agent metrics."""
        
        # Total deliveries
        if 'deliver_ai_rl_1' in df.columns and 'deliver_ai_rl_2' in df.columns:
            df['total_deliveries'] = df['deliver_ai_rl_1'] + df['deliver_ai_rl_2']
        elif 'deliver_ai_rl_1' in df.columns:
            df['total_deliveries'] = df['deliver_ai_rl_1']
        else:
            df['total_deliveries'] = 0
        
        # Total cuts
        if 'cut_ai_rl_1' in df.columns and 'cut_ai_rl_2' in df.columns:
            df['total_cuts'] = df['cut_ai_rl_1'] + df['cut_ai_rl_2']
        elif 'cut_ai_rl_1' in df.columns:
            df['total_cuts'] = df['cut_ai_rl_1']
        else:
            df['total_cuts'] = 0
        
        # Total reward
        if 'pure_reward_ai_rl_1' in df.columns and 'pure_reward_ai_rl_2' in df.columns:
            df['pure_reward_total'] = df['pure_reward_ai_rl_1'] + df['pure_reward_ai_rl_2']
        elif 'pure_reward_ai_rl_1' in df.columns:
            df['pure_reward_total'] = df['pure_reward_ai_rl_1']
        else:
            df['pure_reward_total'] = 0
    
    def prepare_episode_data(self, df: pd.DataFrame, episode_selection: str = 'final', 
                           num_final_episodes: int = 100) -> pd.DataFrame:
        """Prepare data based on episode selection criteria."""
        print(f"Preparing data with episode selection: {episode_selection}")
        
        if episode_selection == 'all':
            return df.copy()
            
        elif episode_selection == 'final':
            final_data = []
            
            for condition in df['condition'].unique():
                condition_df = df[df['condition'] == condition].copy()
                
                # Group by training session (timestamp)
                for timestamp in condition_df['timestamp'].unique():
                    training_df = condition_df[condition_df['timestamp'] == timestamp].copy()
                    training_df = training_df.sort_values('episode')
                    
                    # Take final episodes
                    final_episodes = training_df.tail(num_final_episodes)
                    final_data.append(final_episodes)
            
            final_df = pd.concat(final_data, ignore_index=True)
            print(f"Using final {num_final_episodes} episodes: {len(final_df)} total episodes")
            return final_df
            
        elif episode_selection == 'average':
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
                        'collision_setting': training_df['collision_setting'].iloc[0],
                        'collision_enabled': training_df['collision_enabled'].iloc[0],
                        'speed_pair': training_df['speed_pair'].iloc[0],
                        'walk_speed_1': training_df['walk_speed_1'].iloc[0],
                        'cut_speed_1': training_df['cut_speed_1'].iloc[0],
                        'walk_speed_2': training_df['walk_speed_2'].iloc[0],
                        'cut_speed_2': training_df['cut_speed_2'].iloc[0]
                    }
                    
                    # Average the performance metrics
                    for metric in list(self.performance_metrics.keys()) + list(self.extended_metrics.keys()):
                        if metric in training_df.columns:
                            avg_row[metric] = training_df[metric].mean()
                        else:
                            avg_row[metric] = 0
                    
                    averaged_data.append(avg_row)
            
            avg_df = pd.DataFrame(averaged_data)
            print(f"Using training averages: {len(avg_df)} training sessions")
            return avg_df
        
        else:
            raise ValueError(f"Invalid episode_selection: {episode_selection}")


class GridSearchVisualizer:
    """Creates comprehensive visualizations for grid search analysis."""
    
    def __init__(self, performance_metrics: Dict[str, str], extended_metrics: Dict[str, str]):
        self.performance_metrics = performance_metrics
        self.extended_metrics = extended_metrics
        
        # Set up matplotlib style
        plt.style.use('default')
        plt.rcParams['font.size'] = 8
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['grid.alpha'] = 0.3
        
        # Color palettes
        self.map_colors = {
            'baseline': '#FF6B6B',
            'encouraged': '#4ECDC4', 
            'semiencouraged': '#45B7D1',
            'corridor': '#96CEB4'
        }
        
        self.collision_colors = {
            'classic': '#FFB6C1',           # Light pink - no collision
            'classic_collision': '#FF69B4'  # Hot pink - with collision
        }
        
        self.init_colors = {
            'empty_init': '#87CEEB',        # Sky blue - empty initialization
            'random_init': '#DDA0DD'       # Plum - random initialization
        }
    
    def create_performance_heatmaps(self, data: pd.DataFrame, output_dir: str):
        """Create heatmaps showing performance across all speed combinations."""
        print("Creating performance heatmaps...")
        
        # Create summary statistics for heatmaps
        summary_data = self._create_summary_for_heatmaps(data)
        
        for metric, metric_label in self.performance_metrics.items():
            if metric in summary_data.columns:
                self._create_single_heatmap(summary_data, metric, metric_label, output_dir)
    
    def _create_summary_for_heatmaps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics grouped by experimental factors."""
        
        # Group by all experimental factors and calculate mean performance
        summary = data.groupby(['map_name', 'collision_setting', 'walk_speed_1', 'cut_speed_1', 
                               'walk_speed_2', 'cut_speed_2']).agg({
            **{metric: 'mean' for metric in self.performance_metrics.keys()},
            'episode': 'count'  # Count of episodes per condition
        }).reset_index()
        
        # Rename episode count column
        summary.rename(columns={'episode': 'episode_count'}, inplace=True)
        
        return summary
    
    def _create_single_heatmap(self, summary_data: pd.DataFrame, metric: str, 
                              metric_label: str, output_dir: str):
        """Create a single heatmap for one performance metric."""
        
        # Create subplot grid: maps x collision settings
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'{metric_label} - Grid Search Results', fontsize=16, fontweight='bold')
        
        maps = ['baseline', 'encouraged', 'semiencouraged', 'corridor']
        collision_settings = ['classic', 'classic_collision']
        
        for row, collision in enumerate(collision_settings):
            for col, map_name in enumerate(maps):
                ax = axes[row, col]
                
                # Filter data for this map and collision setting
                subset = summary_data[
                    (summary_data['map_name'] == map_name) & 
                    (summary_data['collision_setting'] == collision)
                ]
                
                if len(subset) == 0:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{map_name}\n{collision}')
                    continue
                
                # Create pivot table for heatmap (Agent 1 walking vs Agent 2 cutting speeds)
                # Agent 1 cutting is always 1.0, Agent 2 walking is always 1.0 based on the grid
                pivot = subset.pivot_table(
                    values=metric, 
                    index='walk_speed_1',  # Agent 1 walking speed (y-axis)
                    columns='cut_speed_2',  # Agent 2 cutting speed (x-axis)
                    aggfunc='mean'
                )
                
                # Create heatmap
                sns.heatmap(pivot, ax=ax, cmap='viridis', annot=True, fmt='.2f', 
                           cbar=False, square=True, linewidths=0.5)
                
                ax.set_title(f'{map_name}\n{collision}')
                ax.set_xlabel('Agent 2 Cutting Speed')
                ax.set_ylabel('Agent 1 Walking Speed')
        
        plt.tight_layout()
        
        # Save figure
        filename = f"heatmap_{metric}.png"
        filepath = Path(output_dir) / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved heatmap: {filepath}")
    
    def create_collision_comparison(self, data: pd.DataFrame, output_dir: str):
        """Create plots comparing collision vs non-collision conditions."""
        print("Creating collision comparison plots...")
        
        # Group data for comparison
        comparison_data = self._prepare_collision_comparison_data(data)
        
        # Create faceted plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Collision vs Non-Collision Performance Comparison', fontsize=16, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        for i, (metric, metric_label) in enumerate(list(self.performance_metrics.items())[:6]):
            ax = axes_flat[i]
            
            if metric in comparison_data.columns:
                # Box plot comparing collision settings
                sns.boxplot(data=comparison_data, x='map_name', y=metric, hue='collision_setting', ax=ax)
                ax.set_title(metric_label)
                ax.set_xlabel('Map')
                ax.set_ylabel(metric_label)
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        filename = "collision_comparison.png"
        filepath = Path(output_dir) / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved collision comparison: {filepath}")
    
    def _prepare_collision_comparison_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for collision comparison analysis."""
        
        # Aggregate across speed pairs to focus on collision effect
        comparison = data.groupby(['map_name', 'collision_setting', 'timestamp']).agg({
            **{metric: 'mean' for metric in self.performance_metrics.keys()}
        }).reset_index()
        
        return comparison
    
    def create_speed_effectiveness_analysis(self, data: pd.DataFrame, output_dir: str):
        """Analyze which speed combinations are most effective."""
        print("Creating speed effectiveness analysis...")
        
        # Calculate speed pair effectiveness
        effectiveness = self._calculate_speed_effectiveness(data)
        
        # Create effectiveness ranking plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Speed Pair Effectiveness Analysis', fontsize=16, fontweight='bold')
        
        metrics_subset = ['total_deliveries', 'total_cuts', 'pure_reward_total', 'total_deliveries'][:4]
        
        for i, metric in enumerate(metrics_subset):
            if metric in self.performance_metrics:
                ax = axes[i//2, i%2]
                metric_label = self.performance_metrics[metric]
                
                # Get top 10 speed pairs for this metric
                top_speeds = effectiveness.nlargest(10, f'{metric}_mean')[['speed_pair', f'{metric}_mean']]
                
                # Horizontal bar plot
                bars = ax.barh(range(len(top_speeds)), top_speeds[f'{metric}_mean'])
                ax.set_yticks(range(len(top_speeds)))
                ax.set_yticklabels(top_speeds['speed_pair'], fontsize=8)
                ax.set_xlabel(metric_label)
                ax.set_title(f'Top 10 Speed Pairs - {metric_label}')
                
                # Color bars by performance
                norm = plt.Normalize(top_speeds[f'{metric}_mean'].min(), top_speeds[f'{metric}_mean'].max())
                for j, bar in enumerate(bars):
                    bar.set_color(plt.cm.viridis(norm(top_speeds[f'{metric}_mean'].iloc[j])))
        
        plt.tight_layout()
        
        # Save figure  
        filename = "speed_effectiveness.png"
        filepath = Path(output_dir) / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved speed effectiveness: {filepath}")
        
        # Also save the effectiveness data as CSV
        csv_filepath = Path(output_dir) / "speed_effectiveness.csv"
        effectiveness.to_csv(csv_filepath, index=False)
        print(f"  Saved effectiveness data: {csv_filepath}")
    
    def _calculate_speed_effectiveness(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate effectiveness metrics for each speed pair."""
        
        # Group by speed pair and calculate statistics
        effectiveness = data.groupby('speed_pair').agg({
            **{f'{metric}': ['mean', 'std', 'count'] for metric in self.performance_metrics.keys()}
        }).round(3)
        
        # Flatten column names
        effectiveness.columns = ['_'.join(col).strip() for col in effectiveness.columns.values]
        effectiveness = effectiveness.reset_index()
        
        return effectiveness
    
    def create_map_comparison(self, data: pd.DataFrame, output_dir: str):
        """Create comprehensive map comparison visualizations."""
        print("Creating map comparison plots...")
        
        # Aggregate data by map for comparison
        map_comparison = self._prepare_map_comparison_data(data)
        
        # Create violin plots comparing maps
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Performance Comparison Across Maps', fontsize=16, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        for i, (metric, metric_label) in enumerate(list(self.performance_metrics.items())[:6]):
            ax = axes_flat[i]
            
            if metric in map_comparison.columns:
                # Violin plot with overlaid strip plot
                sns.violinplot(data=map_comparison, x='map_name', y=metric, ax=ax, 
                             palette=self.map_colors, inner=None)
                sns.stripplot(data=map_comparison, x='map_name', y=metric, ax=ax, 
                             size=3, alpha=0.6, color='black')
                
                ax.set_title(metric_label)
                ax.set_xlabel('Map')
                ax.set_ylabel(metric_label)
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        filename = "map_comparison.png"
        filepath = Path(output_dir) / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved map comparison: {filepath}")
    
    def _prepare_map_comparison_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for map comparison analysis."""
        
        # Aggregate across collision settings and speed pairs to focus on map differences
        map_comparison = data.groupby(['map_name', 'timestamp']).agg({
            **{metric: 'mean' for metric in self.performance_metrics.keys()}
        }).reset_index()
        
        return map_comparison


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive grid search analysis plots',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--episode_range',
        choices=['all', 'final', 'average'],
        default='final',
        help='Episode selection method (default: final)'
    )
    
    parser.add_argument(
        '--num_final_episodes',
        type=int,
        default=100,
        help='Number of final episodes to use when episode_range=final (default: 100)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,  # Will be set based on cluster
        help='Directory to save output figures (default: cluster-specific path)'
    )
    
    parser.add_argument(
        '--study_name',
        type=str,
        default=None,
        help='Study name for specific study folders (optional)'
    )
    
    parser.add_argument(
        '--maps',
        nargs='+',
        choices=['baseline', 'encouraged', 'semiencouraged', 'corridor'],
        default=None,
        help='Specific maps to analyze (default: all)'
    )
    
    parser.add_argument(
        '--inits',
        nargs='+',
        choices=['empty_init', 'random_init'],
        default=None,
        help='Specific initialization types to analyze (default: all)'
    )
    
    parser.add_argument(
        '--collision_only',
        action='store_true',
        help='Focus only on collision vs non-collision comparison'
    )
    
    parser.add_argument(
        '--speed_analysis_only',
        action='store_true',
        help='Focus only on speed combination effectiveness analysis'
    )
    
    parser.add_argument(
        '--cluster',
        type=str,
        default='cuenca',
        help='Cluster name for data directories (default: cuenca)'
    )
    
    return parser


def main():
    """Main function to run the grid search analysis."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = GridSearchAnalyzer(study_name=args.study_name, cluster=args.cluster)
        
        # Set cluster-aware output directory if not specified
        if args.output_dir is None:
            cluster_base = analyzer.config.cluster_paths[args.cluster]
            args.output_dir = f"{cluster_base}/data/samuel_lozano/cooked/gridsearch_analysis_figures"
        
        print("=" * 60)
        print("GRID SEARCH ANALYSIS")
        print("=" * 60)
        print(f"Cluster: {args.cluster}")
        print(f"Maps to analyze: {args.maps if args.maps else 'all'}")
        if args.study_name:
            print(f"Study: {args.study_name}")
        
        # Load grid search data
        df = analyzer.load_grid_search_data(
            selected_maps=args.maps,
            selected_collisions=None,
            selected_inits=args.inits
        )
        
        # Prepare data based on episode selection
        processed_df = analyzer.prepare_episode_data(
            df, 
            episode_selection=args.episode_range,
            num_final_episodes=args.num_final_episodes
        )
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        visualizer = GridSearchVisualizer(analyzer.performance_metrics, analyzer.extended_metrics)
        
        # Create visualizations based on user preferences
        if args.collision_only:
            print("Creating collision-focused analysis...")
            visualizer.create_collision_comparison(processed_df, str(output_dir))
        elif args.speed_analysis_only:
            print("Creating speed-focused analysis...")
            visualizer.create_speed_effectiveness_analysis(processed_df, str(output_dir))
        else:
            print("Creating comprehensive grid search analysis...")
            
            # Create all visualization types
            visualizer.create_performance_heatmaps(processed_df, str(output_dir))
            visualizer.create_collision_comparison(processed_df, str(output_dir))
            visualizer.create_speed_effectiveness_analysis(processed_df, str(output_dir))
            visualizer.create_map_comparison(processed_df, str(output_dir))
        
        # Display results
        print(f"\nAnalysis completed successfully!")
        print(f"Figures saved to: {output_dir}")
        print(f"Data summary:")
        print(f"  Total episodes: {len(processed_df)}")
        print(f"  Unique conditions: {processed_df['condition'].nunique()}")
        print(f"  Maps: {sorted(processed_df['map_name'].unique())}")
        print(f"  Collision settings: {sorted(processed_df['collision_setting'].unique())}")
        print(f"  Speed pairs: {len(processed_df['speed_pair'].unique())}")
        print(f"  Episode selection: {args.episode_range}")
        
        if args.episode_range == 'final':
            print(f"  Final episodes used: {args.num_final_episodes}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()