#!/usr/bin/env python3
"""
Cooperative Analysis Figure Generator - Simulation Data Version

This script creates raincloud plots comparing experimental conditions from simulation data
instead of training data. It reads simulation results from checkpoint directories.

The simulation path structure is:
/data/samuel_lozano/cooked/{game_type}/map_{map}/simulations/{study_name}/Training_{id}/checkpoint_{num}/simulation_{timestamp}/

Usage:
    python figure_cooperative_analysis_simulations.py [options]

Examples:
    # Default (baseline vs encouraged)
    nohup python figure_cooperative_analysis_simulations.py --checkpoint final --output_dir ./figures > figure_sim_analysis_all.log 2>&1 &
    
    # Specific checkpoint number
    nohup python figure_cooperative_analysis_simulations.py --checkpoint 33278 --study_name first_study > figure_sim_analysis.log 2>&1 &
    
    # Custom maps (e.g., other maps)
    nohup python figure_cooperative_analysis_simulations.py --checkpoint final --map_name_1 baseline_division_of_labor_large --map_name_2 collision_division_of_labor_large > figure_sim_baseline_forced.log 2>&1 &
    
    # Extended mode with additional metrics
    nohup python figure_cooperative_analysis_simulations.py --checkpoint final --extended > figure_sim_analysis_extended.log 2>&1 &    
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
import re
from collections import defaultdict


class SimulationDataLoader:
    """Handles loading and processing simulation data."""
    
    def __init__(self, study_name: str = 'default',
                 map_name_1: str = 'baseline_division_of_labor_large',
                 map_name_2: str = 'encouraged_division_of_labor_large'):
        self.study_name = study_name
        self.map_name_1 = map_name_1
        self.map_name_2 = map_name_2
        
        # Extract short names for condition labels
        self.map_1_short = self._extract_map_short_name(map_name_1)
        self.map_2_short = self._extract_map_short_name(map_name_2)
        
        # Define experimental conditions mapping
        self.condition_mapping = {
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
        
        # Color palette mapping
        self.color_palette = {
            f'{self.map_1_short}_mixed': '#FF6B6B',              # Red
            f'{self.map_1_short}_mixed_collision': '#FF9F40',    # Orange  
            f'{self.map_1_short}_superstar': '#90EE90',          # Light Green
            f'{self.map_1_short}_superstar_collision': '#228B22', # Dark Green
            f'{self.map_2_short}_mixed': '#40E0D0',              # Teal
            f'{self.map_2_short}_mixed_collision': '#4169E1',    # Blue
            f'{self.map_2_short}_superstar': '#8A2BE2',          # Purple
            f'{self.map_2_short}_superstar_collision': '#FF69B4' # Pink
        }
        
        # Performance metrics
        self.performance_metrics = {
            'total_deliveries': 'Total Deliveries (Combined)',
            'total_cuts': 'Total Cuts (Combined)', 
            'deliveries_agent_1': 'Deliveries Agent 1',
            'deliveries_agent_2': 'Deliveries Agent 2',
            'cuts_agent_1': 'Cuts Agent 1',
            'cuts_agent_2': 'Cuts Agent 2'
        }
        
        # Extended metrics
        self.extended_metrics = {
            'pickup_salad_agent_1': 'Pickup Salad Agent 1',
            'pickup_salad_agent_2': 'Pickup Salad Agent 2',
            'pickup_plate_agent_1': 'Pickup Plate Agent 1',
            'pickup_plate_agent_2': 'Pickup Plate Agent 2',
            'pickup_raw_food_agent_1': 'Pickup Raw Food Agent 1',
            'pickup_raw_food_agent_2': 'Pickup Raw Food Agent 2',
            'counter_drops_agent_1': 'Counter Drops Agent 1',
            'counter_drops_agent_2': 'Counter Drops Agent 2'
        }
    
    def _extract_map_short_name(self, map_name: str) -> str:
        """Extract a short identifier from the full map name."""
        if '_division_of_labor' in map_name:
            return map_name.split('_division_of_labor')[0]
        return map_name.split('_')[0]
    
    def find_simulation_directories(self, base_path: str, checkpoint: str) -> List[Path]:
        """Find all simulation directories for a given checkpoint.
        
        Args:
            base_path: Base path like /data/samuel_lozano/cooked/{game_type}/map_{map}/simulations/{study_name}
            checkpoint: Checkpoint number or 'final'
            
        Returns:
            List of simulation directory paths
        """
        base = Path(base_path)
        if not base.exists():
            print(f"  Warning: Base path does not exist: {base}")
            return []
        
        simulation_dirs = []
        
        # Find all Training_* directories
        for training_dir in base.glob("Training_*"):
            if not training_dir.is_dir():
                continue
            
            # Look for checkpoint directories
            if checkpoint == 'final':
                # Find the highest numbered checkpoint
                checkpoint_dirs = list(training_dir.glob("checkpoint_*"))
                if checkpoint_dirs:
                    # Extract numbers from checkpoint names
                    def get_checkpoint_num(p):
                        match = re.search(r'checkpoint_(\d+)', p.name)
                        if match:
                            return int(match.group(1))
                        elif 'final' in p.name:
                            return float('inf')
                        return -1
                    
                    checkpoint_dir = max(checkpoint_dirs, key=get_checkpoint_num)
                else:
                    # Try checkpoint_final
                    checkpoint_dir = training_dir / "checkpoint_final"
                    if not checkpoint_dir.exists():
                        continue
            else:
                checkpoint_dir = training_dir / f"checkpoint_{checkpoint}"
                if not checkpoint_dir.exists():
                    continue
            
            # Find all simulation_* directories in this checkpoint
            for sim_dir in checkpoint_dir.glob("simulation_*"):
                if sim_dir.is_dir():
                    simulation_dirs.append(sim_dir)
        
        return simulation_dirs
    
    def parse_config_file(self, config_path: Path) -> Dict:
        """Parse config.txt to extract metadata."""
        config = {}
        
        try:
            with open(config_path, 'r') as f:
                current_section = None
                for line in f:
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Section headers
                    if line.startswith('[') and line.endswith(']'):
                        current_section = line[1:-1]
                        continue
                    
                    # Key-value pairs
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Parse specific keys
                        if key == 'WALKING_SPEEDS':
                            # Parse dictionary string
                            try:
                                config['walking_speeds'] = eval(value)
                            except:
                                pass
                        elif key == 'CUTTING_SPEEDS':
                            try:
                                config['cutting_speeds'] = eval(value)
                            except:
                                pass
                        elif key == 'SIMULATION_ID':
                            config['simulation_id'] = value
                        elif key == 'TRAINING_ID':
                            config['training_id'] = value
                        elif key == 'MAP_NR':
                            config['map_name'] = value
                        elif key == 'GAME_VERSION':
                            config['game_version'] = value
                        elif key == 'CHECKPOINT_NUMBER':
                            config['checkpoint_number'] = value
        
        except Exception as e:
            print(f"  Error parsing config file {config_path}: {e}")
        
        return config
    
    def analyze_meaningful_actions(self, actions_path: Path) -> Dict:
        """Analyze meaningful_actions.csv to compute metrics."""
        metrics = {
            'deliveries_agent_1': 0,
            'deliveries_agent_2': 0,
            'cuts_agent_1': 0,
            'cuts_agent_2': 0,
            'pickup_salad_agent_1': 0,
            'pickup_salad_agent_2': 0,
            'pickup_plate_agent_1': 0,
            'pickup_plate_agent_2': 0,
            'pickup_raw_food_agent_1': 0,
            'pickup_raw_food_agent_2': 0,
            'counter_drops_agent_1': 0,
            'counter_drops_agent_2': 0
        }
        
        try:
            df = pd.read_csv(actions_path)
            
            for _, row in df.iterrows():
                agent_id = row['agent_id']
                action_type = row['processed_action_type']
                action_category = row.get('action_category_name', '')
                
                agent_suffix = '_1' if agent_id == 'ai_rl_1' else '_2'
                
                # Count deliveries
                if action_type == 'use_delivery' or 'deliver' in str(action_category).lower():
                    metrics[f'deliveries_agent{agent_suffix}'] += 1
                
                # Count cuts (using cutting board)
                if action_type == 'use_cutting_board':
                    # Only count the "drop" action (putting item on cutting board)
                    # The cutting action has compound_action_part=1
                    if row.get('compound_action_part', 0) == 1:
                        metrics[f'cuts_agent{agent_suffix}'] += 1
                
                # Extended metrics
                # Pickup salad
                if 'salad' in str(action_type).lower() and row.get('item_change_type') == 'pickup':
                    metrics[f'pickup_salad_agent{agent_suffix}'] += 1
                
                # Pickup plate
                if 'plate' in str(action_type).lower() and row.get('item_change_type') == 'pickup':
                    metrics[f'pickup_plate_agent{agent_suffix}'] += 1
                
                # Pickup raw food (tomato, lettuce, etc.)
                if row.get('item_change_type') == 'pickup':
                    current_item = str(row.get('current_item', '')).lower()
                    if any(food in current_item for food in ['tomato', 'lettuce', 'onion']) and 'cut' not in current_item:
                        metrics[f'pickup_raw_food_agent{agent_suffix}'] += 1
                
                # Counter drops
                if 'counter' in str(action_type).lower() and row.get('item_change_type') == 'drop':
                    metrics[f'counter_drops_agent{agent_suffix}'] += 1
        
        except Exception as e:
            print(f"  Error analyzing actions file {actions_path}: {e}")
        
        return metrics
    
    def load_simulation_data(self, sim_dir: Path) -> Optional[Dict]:
        """Load data from a single simulation directory."""
        config_path = sim_dir / "config.txt"
        actions_path = sim_dir / "meaningful_actions.csv"
        
        if not config_path.exists():
            print(f"  Warning: No config.txt in {sim_dir}")
            return None
        
        if not actions_path.exists():
            print(f"  Warning: No meaningful_actions.csv in {sim_dir}")
            return None
        
        # Parse config
        config = self.parse_config_file(config_path)
        
        # Analyze actions
        metrics = self.analyze_meaningful_actions(actions_path)
        
        # Combine config and metrics
        result = {**config, **metrics}
        
        # Compute total metrics
        result['total_deliveries'] = metrics['deliveries_agent_1'] + metrics['deliveries_agent_2']
        result['total_cuts'] = metrics['cuts_agent_1'] + metrics['cuts_agent_2']
        
        return result
    
    def determine_speed_config(self, walking_speeds: Dict, cutting_speeds: Dict) -> str:
        """Determine if simulation is 'superstar' or 'mixed' based on speeds."""
        if not walking_speeds or not cutting_speeds:
            return 'unknown'
        
        # Superstar: all 1.0
        if all(abs(v - 1.0) < 0.01 for v in walking_speeds.values()) and \
           all(abs(v - 1.0) < 0.01 for v in cutting_speeds.values()):
            return 'superstar'
        
        # Mixed: check for the specific mixed pattern
        # Agent1: walk=0.4, cut=1.0; Agent2: walk=1.0, cut=0.2
        walk1 = walking_speeds.get('ai_rl_1', 0)
        walk2 = walking_speeds.get('ai_rl_2', 0)
        cut1 = cutting_speeds.get('ai_rl_1', 0)
        cut2 = cutting_speeds.get('ai_rl_2', 0)
        
        if (abs(walk1 - 0.4) < 0.01 and abs(cut1 - 1.0) < 0.01 and
            abs(walk2 - 1.0) < 0.01 and abs(cut2 - 0.2) < 0.01):
            return 'mixed'
        
        return 'unknown'
    
    def load_experimental_data(self, checkpoint: str) -> pd.DataFrame:
        """Load simulation data from all experimental conditions."""
        print("Loading simulation data from all conditions...")
        print(f"Study name: {self.study_name}")
        print(f"Checkpoint: {checkpoint}")
        
        all_data = []
        
        # Load data for each experimental condition
        for (map_name, game_type, speed_config), condition_name in self.condition_mapping.items():
            print(f"\nProcessing condition: {condition_name}")
            print(f"  Map: {map_name}, Game type: {game_type}, Speed config: {speed_config}")
            
            # Build base path
            base_path = f"/data/samuel_lozano/cooked/{game_type}/map_{map_name}/simulations/{self.study_name}"
            
            # Find simulation directories
            sim_dirs = self.find_simulation_directories(base_path, checkpoint)
            print(f"  Found {len(sim_dirs)} simulation directories")
            
            if not sim_dirs:
                print(f"  Warning: No simulations found for condition {condition_name}")
                continue
            
            # Load each simulation
            condition_data = []
            for sim_dir in sim_dirs:
                sim_data = self.load_simulation_data(sim_dir)
                
                if sim_data is None:
                    continue
                
                # Check speed configuration
                walking_speeds = sim_data.get('walking_speeds', {})
                cutting_speeds = sim_data.get('cutting_speeds', {})
                detected_speed = self.determine_speed_config(walking_speeds, cutting_speeds)
                
                if detected_speed != speed_config:
                    # Skip simulations that don't match expected speed config
                    continue
                
                # Add metadata
                sim_data['condition'] = condition_name
                sim_data['map_name'] = map_name
                sim_data['game_type_clean'] = game_type
                sim_data['speed_condition'] = speed_config
                sim_data['simulation_dir'] = str(sim_dir)
                
                condition_data.append(sim_data)
            
            print(f"  Loaded {len(condition_data)} simulations matching speed {speed_config}")
            all_data.extend(condition_data)
        
        if not all_data:
            raise ValueError("No simulation data could be loaded from any experimental condition")
        
        # Convert to DataFrame
        combined_df = pd.DataFrame(all_data)
        print(f"\nTotal loaded simulations: {len(combined_df)} across {combined_df['condition'].nunique()} conditions")
        print(f"Conditions found: {sorted(combined_df['condition'].unique())}")
        
        return combined_df


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
                # 1. Left side: Half violin (density)
                if len(condition_data) > 1:
                    try:
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
                
            except Exception as e:
                print(f"    Warning: Could not create density plot for {condition}: {e}")
            
            # 2. Right side: Dot plot
            try:
                y_values = np.array(condition_data)
                np.random.seed(42)
                x_jitter = np.random.normal(position + width_offset, 0.03, len(y_values))
                
                ax.scatter(x_jitter, y_values, c=color, s=8, alpha=0.7, 
                          edgecolors='black', linewidths=0.3, zorder=5)
                
            except Exception as e:
                print(f"    Warning: Could not create scatter plot for {condition}: {e}")
            
            # 3. Center: Diamond with error bar (mean ± std)
            try:
                mean_val = condition_data.mean()
                std_val = condition_data.std()
                
                ax.scatter(position, mean_val, marker='D', s=100, c=color, 
                          edgecolors='black', linewidths=1.0, zorder=10)
                
                if not np.isnan(std_val) and std_val > 0:
                    ax.errorbar(position, mean_val, yerr=std_val, fmt='none', 
                               color='black', linewidth=1.5, capsize=4, zorder=9)
                
            except Exception as e:
                print(f"    Warning: Could not create mean/std markers for {condition}: {e}")
        
        # Styling
        ax.set_xticks(positions)
        ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
        
        if speed_info:
            ax.set_ylabel(f"{metric_label}\n{speed_info}", fontsize=10)
        else:
            ax.set_ylabel(metric_label, fontsize=10)
            
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('black')
    
    def create_composite_figure(self, data: pd.DataFrame, output_path: str = None, 
                                 extended: bool = False) -> plt.Figure:
        """Create the complete raincloud plot figure."""
        
        print(f"Creating {'extended' if extended else 'standard'} composite raincloud figure...")
        
        if extended:
            fig, axes = plt.subplots(7, 2, figsize=(16, 32))
            fig.suptitle('Cooperative Simulation Analysis (Extended)', fontsize=16, fontweight='bold', y=0.98)
        else:
            fig, axes = plt.subplots(3, 2, figsize=(16, 18))
            fig.suptitle('Cooperative Simulation Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        axes_flat = axes.flatten()
        
        metrics_list = list(self.performance_metrics.items())
        
        if extended and self.extended_metrics:
            metrics_list.extend(list(self.extended_metrics.items()))
        
        subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
        
        speed_info_mapping = {
            'deliveries_agent_1': '(Superstar: walk=1.0, cut=1.0 | Mixed: walk=0.4, cut=1.0)',
            'deliveries_agent_2': '(Superstar: walk=1.0, cut=1.0 | Mixed: walk=1.0, cut=0.2)',
            'cuts_agent_1': '(Superstar: walk=1.0, cut=1.0 | Mixed: walk=0.4, cut=1.0)',
            'cuts_agent_2': '(Superstar: walk=1.0, cut=1.0 | Mixed: walk=1.0, cut=0.2)'
        }
        
        for i, (metric, metric_label) in enumerate(metrics_list):
            ax = axes_flat[i]
            
            ax.text(0.02, 0.98, subplot_labels[i], transform=ax.transAxes, 
                   fontsize=14, fontweight='bold', va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            speed_info = speed_info_mapping.get(metric, "")
            
            self.create_raincloud_subplot(ax, data, metric, metric_label, speed_info)
        
        plt.tight_layout()
        
        # Create legend
        handles = []
        labels = []
        
        for condition, color in self.color_palette.items():
            if condition in data['condition'].unique():
                handle = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black')
                handles.append(handle)
                labels.append(condition)
        
        if extended:
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), 
                      ncol=4, frameon=True, fancybox=True, shadow=True, fontsize=10)
            plt.subplots_adjust(bottom=0.03)
        else:
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                      ncol=4, frameon=True, fancybox=True, shadow=True, fontsize=10)
            plt.subplots_adjust(bottom=0.06)
        
        if output_path:
            print(f"Saving figure to: {output_path}")
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Generate cooperative analysis raincloud plots from simulation data',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Checkpoint number or "final" to use highest numbered checkpoint'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/data/samuel_lozano/cooked/cooperative_analysis_figures',
        help='Directory to save output figures (default: /data/samuel_lozano/cooked/cooperative_analysis_figures)'
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
        default='default',
        help='Study name for simulation folders (default: default)'
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
        help='Create extended plot with additional metrics'
    )
    
    return parser


def main():
    """Main function to run the simulation-based cooperative analysis."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        # Initialize data loader
        loader = SimulationDataLoader(
            study_name=args.study_name,
            map_name_1=args.map_name_1,
            map_name_2=args.map_name_2
        )
        
        # Load simulation data
        print("=" * 60)
        print("COOPERATIVE SIMULATION ANALYSIS - RAINCLOUD PLOTS")
        print("=" * 60)
        print(f"Map 1: {args.map_name_1}")
        print(f"Map 2: {args.map_name_2}")
        print(f"Study: {args.study_name}")
        print(f"Checkpoint: {args.checkpoint}")
        
        df = loader.load_experimental_data(args.checkpoint)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename_parts = ["cooperative_analysis_simulations"]
        
        map_1_short = loader.map_1_short
        map_2_short = loader.map_2_short
        filename_parts.append(f"{map_1_short}_vs_{map_2_short}")
        
        if args.study_name != 'default':
            filename_parts.append(args.study_name)
        
        filename_parts.append(f"checkpoint_{args.checkpoint}")
        
        if args.extended:
            filename_parts.append('extended')
        
        if args.filename_suffix:
            filename_parts.append(args.filename_suffix)
        
        filename = "_".join(filename_parts) + ".png"
        output_path = output_dir / filename
        
        # Create raincloud plots
        plotter = RaincloudPlotter(
            loader.color_palette, 
            loader.performance_metrics,
            loader.extended_metrics
        )
        fig = plotter.create_composite_figure(df, str(output_path), extended=args.extended)
        
        # Display results
        print(f"\nAnalysis completed successfully!")
        print(f"Figure saved to: {output_path}")
        print(f"Data summary:")
        print(f"  Total simulations: {len(df)}")
        print(f"  Conditions: {sorted(df['condition'].unique())}")
        print(f"  Study name: {args.study_name}")
        print(f"  Checkpoint: {args.checkpoint}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
