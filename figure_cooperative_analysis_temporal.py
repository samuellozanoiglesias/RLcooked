#!/usr/bin/env python3
"""
Cooperative Analysis Temporal Figure Generator

This script creates line plots showing temporal evolution of performance metrics
across episodes for different experimental conditions. Each plot shows:
- Four lines (one per configuration: baseline/encouraged × mixed/superstar)
- Only classic_collision conditions
- Shaded areas representing standard deviation
- Data points averaged over windows of episodes

Usage:
    python figure_cooperative_analysis_temporal.py [options]

Examples:

nohup python figure_cooperative_analysis_temporal.py --map_name_1 baseline --map_name_2 encouraged --init_type empty_init --extended --synergy 0.4 --specialization 0.05 --initial_episode 0 --final_episode 1000 --window_size 50 --step_size 25 > temporal_analysis_extended.log 2>&1 &

"""


import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spoiled_broth.analysis.utils import DataProcessor, AnalysisConfig
from figure_cooperative_analysis import CooperativeAnalyzer


class TemporalPlotter:
    """Creates temporal line plots showing evolution over episodes."""
    
    def __init__(self, color_palette: Dict[str, str], performance_metrics: Dict[str, str],
                 extended_metrics: Dict[str, str] = None):
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
    
    def prepare_temporal_data(self, data: pd.DataFrame, window_size: int = 20, 
                             step_size: int = 10) -> pd.DataFrame:
        """
        Prepare data for temporal plotting by averaging over episode windows.
        
        Args:
            data: Raw episode data
            window_size: Total window size (M in the description - uses M/2 before and after)
            step_size: Step between consecutive data points (N in the description)
            
        Returns:
            DataFrame with averaged data points
        """
        print(f"Preparing temporal data (window_size={window_size}, step_size={step_size})...")
        
        temporal_data = []
        half_window = window_size // 2
        
        # Process each condition separately
        for condition in sorted(data['condition'].unique()):
            condition_df = data[data['condition'] == condition].copy()
            
            # Process each training session separately
            for timestamp in condition_df['timestamp'].unique():
                training_df = condition_df[condition_df['timestamp'] == timestamp].copy()
                training_df = training_df.sort_values('episode')
                
                # Find the range of episodes
                min_episode = training_df['episode'].min()
                max_episode = training_df['episode'].max()
                
                print(f"  Processing {condition} - training {timestamp}")
                print(f"    Episode range: {min_episode} to {max_episode}")
                
                # Create data points every step_size episodes
                center_episodes = range(int(min_episode) + half_window, 
                                       int(max_episode) - half_window + 1, 
                                       step_size)
                
                for center_ep in center_episodes:
                    # Get window of episodes around this center
                    window_start = center_ep - half_window
                    window_end = center_ep + half_window
                    
                    window_df = training_df[
                        (training_df['episode'] >= window_start) & 
                        (training_df['episode'] <= window_end)
                    ]
                    
                    if len(window_df) == 0:
                        continue
                    
                    # Calculate mean and std for each metric in this window
                    row = {
                        'condition': condition,
                        'timestamp': timestamp,
                        'episode': center_ep,
                        'map_name': training_df['map_name'].iloc[0],
                        'game_type_clean': training_df['game_type_clean'].iloc[0],
                        'speed_condition': training_df['speed_condition'].iloc[0]
                    }
                    
                    all_metrics = list(self.performance_metrics.keys()) + list(self.extended_metrics.keys())
                    for metric in all_metrics:
                        if metric in window_df.columns:
                            row[f'{metric}_mean'] = window_df[metric].mean()
                            row[f'{metric}_std'] = window_df[metric].std()
                        else:
                            row[f'{metric}_mean'] = np.nan
                            row[f'{metric}_std'] = np.nan
                    
                    temporal_data.append(row)
                
                print(f"    Created {len([d for d in temporal_data if d['timestamp'] == timestamp])} data points")
        
        temporal_df = pd.DataFrame(temporal_data)
        print(f"Total temporal data points: {len(temporal_df)}")
        
        return temporal_df
    
    def create_temporal_subplot(self, ax, data: pd.DataFrame, metric: str, 
                               metric_label: str, speed_info: str = ""):
        """Create a single temporal line plot on the given axis."""
        
        # Get unique conditions and sort them
        conditions = sorted(data['condition'].unique())
        
        # Clear the axis
        ax.clear()
        
        for condition in conditions:
            condition_data = data[data['condition'] == condition].copy()
            
            if len(condition_data) == 0:
                continue
            
            # Group by episode and aggregate across training sessions
            grouped = condition_data.groupby('episode').agg({
                f'{metric}_mean': ['mean', 'std'],
                f'{metric}_std': 'mean'
            }).reset_index()
            
            episodes = grouped['episode'].values
            
            # Mean of means across training sessions
            mean_values = grouped[(f'{metric}_mean', 'mean')].values
            
            # Std aggregated across training sessions
            # Combine within-window std and between-training std
            within_std = grouped[(f'{metric}_std', 'mean')].values
            between_std = grouped[(f'{metric}_mean', 'std')].values
            
            # Use between-training std if available, otherwise within-window std
            std_values = np.where(
                pd.notna(between_std) & (between_std > 0), 
                between_std, 
                within_std
            )
            
            color = self.color_palette.get(condition, '#666666')
            
            # Plot line
            ax.plot(episodes, mean_values, color=color, linewidth=2, 
                   label=condition, marker='o', markersize=4, alpha=0.8)
            
            # Plot shaded std area
            ax.fill_between(episodes, 
                           mean_values - std_values, 
                           mean_values + std_values,
                           color=color, alpha=0.2)
        
        # Styling
        ax.set_xlabel('Episode', fontsize=10)
        
        if speed_info:
            ax.set_ylabel(f"{metric_label}\n{speed_info}", fontsize=10)
        else:
            ax.set_ylabel(metric_label, fontsize=10)
        
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        
        # Add border around subplot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('black')
    
    def create_temporal_figure(self, data: pd.DataFrame, window_size: int = 20, 
                              step_size: int = 10, output_path: str = None,
                              extended: bool = False) -> plt.Figure:
        """
        Create the complete temporal evolution figure.
        
        Args:
            data: Raw episode data
            window_size: Window size for averaging
            step_size: Step between data points
            output_path: Path to save figure
            extended: If True, create 7x2 layout with extended metrics; if False, create 3x2 layout
        """
        
        print(f"Creating {'extended' if extended else 'standard'} temporal evolution figure...")
        
        # Prepare temporal data
        temporal_df = self.prepare_temporal_data(data, window_size, step_size)
        
        if extended:
            # Extended mode: 7x2 layout (3 main + 4 extended rows)
            fig, axes = plt.subplots(7, 2, figsize=(16, 32))
            fig.suptitle('Cooperative Performance Evolution Over Episodes (Extended)', 
                        fontsize=16, fontweight='bold', y=0.98)
        else:
            # Standard mode: 3x2 layout
            fig, axes = plt.subplots(3, 2, figsize=(16, 18))
            fig.suptitle('Cooperative Performance Evolution Over Episodes', 
                        fontsize=16, fontweight='bold', y=0.95)
        
        # Flatten axes for easy iteration
        axes_flat = axes.flatten()
        
        # Metrics list
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
            
            # Create temporal plot for this metric
            self.create_temporal_subplot(ax, temporal_df, metric, metric_label, speed_info)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if output path provided
        if output_path:
            print(f"Saving figure to: {output_path}")
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        return fig


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Generate temporal evolution line plots for cooperative analysis',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--window_size',
        type=int,
        default=20,
        help='Window size for averaging (M in description - uses M/2 before and after center episode)'
    )
    
    parser.add_argument(
        '--step_size',
        type=int,
        default=10,
        help='Step between consecutive data points (N in description - creates points every N episodes)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save output figures'
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
        help='Which specialization lambda value to analyze (e.g., 0, 5.0). If not specified, auto-detects first available.'
    )
    
    parser.add_argument(
        '--cluster',
        type=str,
        default='brigit',
        help='Cluster name (optional, for reference only)'
    )
    
    parser.add_argument(
        '--initial_episode',
        type=int,
        default=None,
        help='Initial episode number to start analysis (inclusive). If not specified, starts from first available episode.'
    )
    
    parser.add_argument(
        '--final_episode',
        type=int,
        default=None,
        help='Final episode number to end analysis (inclusive). If not specified, goes to last available episode.'
    )
    
    return parser


def main():
    """Main function to run the temporal analysis."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        # Check if synergy was explicitly provided
        synergy_provided = '--synergy' in sys.argv
        
        # Set up output directory
        cluster = args.cluster if args.cluster else 'brigit'
        config = AnalysisConfig()
        local_path = config.cluster_paths[cluster]
        
        if args.output_dir is None:
            output_dir_base = f"{local_path}/data/samuel_lozano/cooked/temporal_analysis_figures"
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
            specialization=args.specialization,
            cluster=cluster
        )
        
        # Load experimental data
        print("=" * 60)
        print("COOPERATIVE ANALYSIS - TEMPORAL EVOLUTION PLOTS")
        print("=" * 60)
        print(f"Cluster: {cluster}")
        print(f"Init type: {args.init_type}")
        print(f"Synergy scaling factor: {args.synergy}")
        print(f"Specialization: {args.specialization}")
        print(f"Map 1: {args.map_name_1}")
        print(f"Map 2: {args.map_name_2}")
        print(f"Window size: {args.window_size}")
        print(f"Step size: {args.step_size}")
        print(f"Extended mode: {args.extended}")
        if args.initial_episode is not None or args.final_episode is not None:
            print(f"Episode range filter: {args.initial_episode or 'start'} to {args.final_episode or 'end'}")
        if args.study_name:
            print(f"Study: {args.study_name}")
        
        df_or_dict = analyzer.load_experimental_data()
        
        # Check if we got a dict (multiple combinations) or single DataFrame
        if isinstance(df_or_dict, dict):
            # Use first combination or specific one if only one exists
            if len(df_or_dict) == 1:
                combination_key = list(df_or_dict.keys())[0]
                print(f"\nUsing single combination: {combination_key}")
                df = df_or_dict[combination_key]
            else:
                # Multiple combinations - let user know we're using the first one
                combination_key = sorted(df_or_dict.keys())[0]
                print(f"\nMultiple combinations found. Using: {combination_key}")
                print(f"Available combinations: {sorted(df_or_dict.keys())}")
                df = df_or_dict[combination_key]
        else:
            df = df_or_dict
        
        # Filter to only classic_collision conditions
        print("\nFiltering to classic_collision conditions only...")
        collision_conditions = [cond for cond in df['condition'].unique() 
                               if 'collision' in cond]
        df_collision = df[df['condition'].isin(collision_conditions)].copy()
        
        print(f"Filtered conditions: {sorted(df_collision['condition'].unique())}")
        print(f"Total episodes before episode filtering: {len(df_collision)}")
        
        # Apply episode range filtering if specified
        if args.initial_episode is not None or args.final_episode is not None:
            initial_ep = args.initial_episode if args.initial_episode is not None else df_collision['episode'].min()
            final_ep = args.final_episode if args.final_episode is not None else df_collision['episode'].max()
            
            print(f"\nApplying episode range filter: {initial_ep} to {final_ep}")
            df_collision = df_collision[
                (df_collision['episode'] >= initial_ep) & 
                (df_collision['episode'] <= final_ep)
            ].copy()
            
            print(f"Total episodes after episode filtering: {len(df_collision)}")
        
        if len(df_collision) == 0:
            raise ValueError("No collision conditions found in the data (or no episodes in the specified range)!")
        
        # Create output directory
        output_dir = Path(output_dir_base)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename_parts = ["extended_temporal_evolution"] if args.extended else ["temporal_evolution"]
        filename_parts.append(args.init_type)
        
        if synergy_provided:
            filename_parts.append(f"synergy_{args.synergy}" if args.synergy != 0 else "synergy_0")
        
        if args.specialization is not None:
            filename_parts.append(f"specialized_{args.specialization:.2f}" if args.specialization != 0 else "specialized_0")
        
        filename_parts.append(f"{analyzer.map_1_short}_vs_{analyzer.map_2_short}")
        
        if args.study_name:
            filename_parts.append(args.study_name)
        
        filename_parts.append(f"window{args.window_size}_step{args.step_size}")
        
        # Add episode range to filename if specified
        if args.initial_episode is not None or args.final_episode is not None:
            initial_ep = args.initial_episode if args.initial_episode is not None else int(df_collision['episode'].min())
            final_ep = args.final_episode if args.final_episode is not None else int(df_collision['episode'].max())
            filename_parts.append(f"ep{initial_ep}-{final_ep}")
        
        if args.filename_suffix:
            filename_parts.append(args.filename_suffix)
        
        filename = "_".join(filename_parts) + ".png"
        output_path = output_dir / filename
        
        # Create temporal plots
        plotter = TemporalPlotter(
            analyzer.color_palette,
            analyzer.performance_metrics,
            analyzer.extended_metrics
        )
        fig = plotter.create_temporal_figure(
            df_collision, 
            window_size=args.window_size,
            step_size=args.step_size,
            output_path=str(output_path),
            extended=args.extended
        )
        
        # Display results
        print(f"\nTemporal analysis completed successfully!")
        print(f"Figure saved to: {output_path}")
        
        plt.close(fig)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
