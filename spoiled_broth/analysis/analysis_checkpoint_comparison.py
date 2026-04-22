#!/usr/bin/env python3
"""
Checkpoint Delivery Comparison Analysis for spoiled_broth experiments.

This script analyzes simulation data across all training_ids and checkpoints for a given map_nr
to generate a figure comparing averaged deliveries in the last frame over checkpoint numbers.

The script expects the following directory structure:
/data/samuel_lozano/cooked/map_{map_nr}/.../simulations/Training_{training_id}/checkpoint_{checkpoint}/

Supported simulation files:
- Legacy: simulation.csv
- Current: positions_<agent>.csv (or human_like_positions_<agent>.csv)

Usage:
nohup python3 analysis_checkpoint_comparison.py --cluster cuenca --map_nr baseline_division_of_labor --game_version classic > log_analysis_checkpoint_comparison.out 2>&1 &

Author: Samuel Lozano
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import re
from collections import defaultdict
import warnings
import argparse
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class CheckpointDeliveryAnalyzer:
    """Main class for analyzing deliveries across checkpoints and training_ids."""
    
    def __init__(self, base_cluster_dir="", map_nr=None, game_version="classic", num_agents=2, study_name='', game_type='classic', base_dir=None):
        """
        Initialize the analyzer with specific parameters.
        
        Args:
            base_cluster_dir: Base cluster directory 
            map_nr: Map number/name (e.g., "baseline_division_of_labor")
            game_version: Game version ("classic" or "competition")
            num_agents: Number of agents in the simulation (1 or 2)
            study_name: Study name for simulation folders (default: empty, use /simulations/ directly)
            game_type: Game type for folder organization (default: 'classic')
            base_dir: Optional explicit map base directory containing simulations/
        """
        self.base_cluster_dir = base_cluster_dir
        self.map_nr = map_nr
        self.game_version = game_version
        self.num_agents = num_agents
        self.study_name = (study_name or "").strip()
        self.game_type = game_type
        
        # Construct the base map directory path - updated for new structure with study_name
        if base_dir:
            self.base_map_dir = Path(base_dir) / "simulations"
        else:
            candidate_base_dirs = []
            if self.num_agents == 1:
                candidate_base_dirs.append(
                    Path(
                        f"{base_cluster_dir}/data/samuel_lozano/cooked/pretraining/{game_version}/map_{map_nr}/simulations"
                    )
                )
                candidate_base_dirs.append(
                    Path(
                        f"{base_cluster_dir}/data/samuel_lozano/cooked/pretraining/map_{map_nr}/simulations"
                    )
                )
            else:
                candidate_base_dirs.append(
                    Path(
                        f"{base_cluster_dir}/data/samuel_lozano/cooked/{game_version}/map_{map_nr}/simulations"
                    )
                )
                candidate_base_dirs.append(
                    Path(
                        f"{base_cluster_dir}/data/samuel_lozano/cooked/map_{map_nr}/simulations"
                    )
                )

            self.base_map_dir = candidate_base_dirs[0]
            for candidate in candidate_base_dirs:
                if candidate.exists():
                    self.base_map_dir = candidate
                    break

        if self.study_name:
            self.base_map_dir = self.base_map_dir / self.study_name

        self.checkpoint_data = {}  # {training_id: {checkpoint: delivery_data}}
        
    def find_all_training_directories(self):
        """Find all training directories for the specified map."""
        training_dirs = []
        
        if not self.base_map_dir.exists():
            print(f"Error: Map directory does not exist: {self.base_map_dir}")
            return training_dirs
        
        print(f"Searching for training directories in: {self.base_map_dir}")
        
        # Search for Training_* directories
        for training_dir in self.base_map_dir.glob("Training_*"):
            if training_dir.is_dir():
                training_id = training_dir.name.replace("Training_", "")
                training_dirs.append((training_dir, training_id))
                
        print(f"Found {len(training_dirs)} training directories")
        return training_dirs
    
    def find_checkpoints_for_training(self, training_dir):
        """Find all checkpoint directories for a specific training."""
        checkpoints = []
        
        # Search for checkpoint_* directories (updated for new structure)
        for checkpoint_dir in training_dir.glob("checkpoint_*"):
            if checkpoint_dir.is_dir():
                checkpoint_name = checkpoint_dir.name.replace("checkpoint_", "")
                checkpoints.append((checkpoint_dir, checkpoint_name))
        
        return checkpoints

    def _extract_agent_id_from_filename(self, file_path, prefixes):
        """Extract agent id from filename prefixes like positions_*.csv."""
        stem = file_path.stem
        for prefix in prefixes:
            if stem.startswith(prefix):
                return stem[len(prefix):]
        return stem

    def _normalize_positions_dataframe(self, sim_data):
        """Normalize position columns across legacy and current schemas."""
        sim_data = sim_data.copy()

        if 'x' not in sim_data.columns:
            if 'tile_x' in sim_data.columns:
                sim_data['x'] = sim_data['tile_x']
            elif 'pixel_x' in sim_data.columns:
                sim_data['x'] = sim_data['pixel_x']

        if 'y' not in sim_data.columns:
            if 'tile_y' in sim_data.columns:
                sim_data['y'] = sim_data['tile_y']
            elif 'pixel_y' in sim_data.columns:
                sim_data['y'] = sim_data['pixel_y']

        if 'frame' not in sim_data.columns and 'tick' in sim_data.columns:
            sim_data['frame'] = sim_data['tick']

        return sim_data

    def _load_simulation_dataframe(self, sim_dir):
        """Load simulation dataframe from legacy or current file layout."""
        sim_csv = sim_dir / "simulation.csv"
        if sim_csv.exists():
            return self._normalize_positions_dataframe(pd.read_csv(sim_csv))

        position_files = sorted(sim_dir.glob("positions_*.csv"))
        if not position_files:
            position_files = sorted(sim_dir.glob("human_like_positions_*.csv"))

        if not position_files:
            return None

        all_positions = []
        for position_file in position_files:
            try:
                position_data = pd.read_csv(position_file)
            except Exception as e:
                print(f"Error reading {position_file}: {e}")
                continue

            if 'agent_id' not in position_data.columns:
                position_data['agent_id'] = self._extract_agent_id_from_filename(
                    position_file,
                    prefixes=['positions_', 'human_like_positions_'],
                )

            all_positions.append(self._normalize_positions_dataframe(position_data))

        if not all_positions:
            return None

        return pd.concat(all_positions, ignore_index=True, sort=False)
    
    def read_config_file(self, sim_dir):
        """Read config.txt file and extract AGENT_INITIALIZATION_PERIOD."""
        config_path = sim_dir / "config.txt"
        initialization_period = 0.0  # Default if not found
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_contents = f.read()
                
                # Look for AGENT_INITIALIZATION_PERIOD
                init_match = re.search(r"AGENT_INITIALIZATION_PERIOD:\s*([0-9.]+)", config_contents)
                if init_match:
                    initialization_period = float(init_match.group(1))
                else:
                    # Silent fallback for checkpoint comparison - don't spam logs
                    pass
                    
            except Exception as e:
                # Silent fallback for checkpoint comparison - don't spam logs
                pass
            
        return initialization_period
    
    def convert_frames_to_seconds(self, sim_data, initialization_period):
        """Convert frame numbers to adjusted seconds (subtracting initialization time)."""
        if 'second' in sim_data.columns:
            # If seconds column exists, use it and adjust by subtracting initialization period
            sim_data = sim_data.copy()
            sim_data['adjusted_second'] = sim_data['second'] - initialization_period
            # Ensure we don't have negative seconds (clamp to 0)
            sim_data['adjusted_second'] = sim_data['adjusted_second'].clip(lower=0)
            return sim_data
        elif 'frame' in sim_data.columns:
            # If no seconds column, create one assuming some frame rate (e.g., 30 FPS)
            sim_data = sim_data.copy()
            assumed_fps = 30
            sim_data['adjusted_second'] = (sim_data['frame'] / assumed_fps) - initialization_period
            sim_data['adjusted_second'] = sim_data['adjusted_second'].clip(lower=0)
            return sim_data
        else:
            sim_data = sim_data.copy()
            sim_data['adjusted_second'] = 0.0
            return sim_data
    
    def compute_deliveries(self, sim_data):
        """Compute delivery counts over time from simulation data."""
        # Use adjusted_second instead of frame for time axis
        time_column = 'adjusted_second' if 'adjusted_second' in sim_data.columns else 'frame'
        
        # Deliveries are typically tracked through score increases
        deliveries = sim_data.groupby(['agent_id', time_column])['score'].max().reset_index()
        
        # Calculate delivery events (score increases)
        delivery_events = []
        for agent_id in deliveries['agent_id'].unique():
            agent_data = deliveries[deliveries['agent_id'] == agent_id].sort_values(time_column)
            agent_data['score_diff'] = agent_data['score'].diff().fillna(0)
            agent_data['deliveries'] = (agent_data['score_diff'] > 0).astype(int)
            agent_data['cumulative_deliveries'] = agent_data['deliveries'].cumsum()
            delivery_events.append(agent_data)
        
        return pd.concat(delivery_events, ignore_index=True) if delivery_events else pd.DataFrame()
    
    def get_final_deliveries_for_checkpoint(self, checkpoint_dir):
        """Get the final delivery count for all simulations in a checkpoint directory."""
        final_deliveries = []
        
        # Find all simulation directories within this checkpoint
        for sim_dir in checkpoint_dir.glob("simulation_*"):
            if sim_dir.is_dir():
                try:
                    sim_data = self._load_simulation_dataframe(sim_dir)
                    if sim_data is None or sim_data.empty or 'score' not in sim_data.columns:
                        continue

                    # Read config file to get initialization period
                    initialization_period = self.read_config_file(sim_dir)

                    # Convert frames to adjusted seconds
                    sim_data = self.convert_frames_to_seconds(sim_data, initialization_period)

                    time_column = 'adjusted_second' if 'adjusted_second' in sim_data.columns else 'frame'
                    score_over_time = sim_data.groupby(time_column)['score'].max().sort_index()

                    if score_over_time.empty:
                        continue

                    score_diff = score_over_time.diff().fillna(0)
                    total_final_deliveries = int((score_diff > 0).sum())
                    final_time = score_over_time.index.max()

                    final_deliveries.append({
                        'simulation': sim_dir.name,
                        'final_deliveries': total_final_deliveries,
                        'final_time': final_time
                    })

                except Exception as e:
                    print(f"Error processing {sim_dir}: {e}")
        
        return final_deliveries
    
    def load_all_data(self):
        """Load delivery data for all training_ids and checkpoints."""
        training_dirs = self.find_all_training_directories()
        
        if not training_dirs:
            print("No training directories found!")
            return
        
        for training_dir, training_id in training_dirs:
            print(f"\nProcessing training: {training_id}")
            
            checkpoints = self.find_checkpoints_for_training(training_dir)
            
            if not checkpoints:
                print(f"  No checkpoints found for {training_id}")
                continue
            
            self.checkpoint_data[training_id] = {}
            
            for checkpoint_dir, checkpoint_name in checkpoints:
                print(f"  Processing checkpoint: {checkpoint_name}")
                
                final_deliveries = self.get_final_deliveries_for_checkpoint(checkpoint_dir)
                
                if final_deliveries:
                    # Get real checkpoint number - crucial for final checkpoints
                    real_checkpoint_num = self.get_real_checkpoint_number(checkpoint_name, training_dir)
                    
                    self.checkpoint_data[training_id][checkpoint_name] = {
                        'simulations': final_deliveries,
                        'real_checkpoint_numeric': real_checkpoint_num,
                        'training_dir': training_dir
                    }
                    
                    print(f"    Found {len(final_deliveries)} simulations, real episode: {real_checkpoint_num}")
                else:
                    print(f"    No valid simulations found")
    
    def extract_numeric_checkpoint(self, checkpoint_name):
        """Extract numeric value from checkpoint name for sorting."""
        if checkpoint_name == "final":
            return float('inf')  # Put 'final' at the end
        
        # Try to extract number from checkpoint name
        numbers = re.findall(r'\d+', checkpoint_name)
        if numbers:
            return int(numbers[0])
        else:
            return 0
    
    def get_final_episode_from_training_stats(self, training_dir):
        """Get the final episode number from training_stats.csv."""
        training_stats_file = training_dir / "training_stats.csv"
        
        if training_stats_file.exists():
            try:
                # Read the training stats file
                stats_df = pd.read_csv(training_stats_file)
                
                # Get the last episode number and add 1
                if 'episode' in stats_df.columns and len(stats_df) > 0:
                    final_episode = stats_df['episode'].iloc[-1] + 1
                    return final_episode
                else:
                    print(f"Warning: No 'episode' column found in {training_stats_file}")
                    return None
                    
            except Exception as e:
                print(f"Error reading {training_stats_file}: {e}")
                return None
        else:
            print(f"Warning: training_stats.csv not found in {training_dir}")
            return None
    
    def get_real_checkpoint_number(self, checkpoint_name, training_dir):
        """Get the real checkpoint number, reading from training_stats.csv for 'final'."""
        if checkpoint_name == "final":
            final_episode = self.get_final_episode_from_training_stats(training_dir)
            if final_episode is not None:
                return final_episode
            else:
                # Fallback to original logic if we can't read the stats
                return float('inf')
        else:
            # Try to extract number from checkpoint name
            numbers = re.findall(r'\d+', checkpoint_name)
            if numbers:
                return int(numbers[0])
            else:
                return 0
    
    def plot_deliveries_vs_checkpoints(self, output_dir):
        """Create the main plot comparing deliveries across checkpoints and training_ids."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.checkpoint_data:
            print("No data loaded. Cannot create plot.")
            return
        
        # Prepare data for plotting
        plot_data = []
        
        for training_id, checkpoints in self.checkpoint_data.items():
            for checkpoint_name, checkpoint_info in checkpoints.items():
                # Use the consistent structure
                simulations = checkpoint_info['simulations']
                real_checkpoint_num = checkpoint_info['real_checkpoint_numeric']
                
                if simulations:
                    # Calculate statistics for this checkpoint
                    deliveries = [sim['final_deliveries'] for sim in simulations]
                    mean_deliveries = np.mean(deliveries)
                    std_deliveries = np.std(deliveries) if len(deliveries) > 1 else 0
                    min_deliveries = np.min(deliveries)
                    max_deliveries = np.max(deliveries)
                    
                    plot_data.append({
                        'training_id': training_id,
                        'checkpoint': checkpoint_name,
                        'checkpoint_numeric': self.extract_numeric_checkpoint(checkpoint_name),  # For sorting
                        'real_checkpoint_numeric': real_checkpoint_num,  # For plotting
                        'mean_deliveries': mean_deliveries,
                        'std_deliveries': std_deliveries,
                        'min_deliveries': min_deliveries,
                        'max_deliveries': max_deliveries,
                        'n_simulations': len(deliveries),
                        'all_deliveries': deliveries
                    })
        
        if not plot_data:
            print("No valid data for plotting.")
            return
        
        plot_df = pd.DataFrame(plot_data)
        
        # Sort by checkpoint number
        plot_df = plot_df.sort_values('checkpoint_numeric')
        
        # Create the main plot
        plt.figure(figsize=(14, 10))
        
        # Get unique training_ids for color mapping - use stronger colors
        unique_training_ids = plot_df['training_id'].unique()
        # Use more vibrant colors that are easier to see
        strong_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
        colors = strong_colors[:len(unique_training_ids)]
        if len(unique_training_ids) > len(strong_colors):
            # If more training_ids than predefined colors, generate additional ones
            import matplotlib.cm as cm
            extra_colors = cm.Dark2(np.linspace(0, 1, len(unique_training_ids) - len(strong_colors)))
            colors.extend([tuple(c) for c in extra_colors])
        color_map = dict(zip(unique_training_ids, colors))
        
        # Plot 1: Mean deliveries with error bars
        plt.subplot(2, 2, 1)
        for training_id in unique_training_ids:
            training_data = plot_df[plot_df['training_id'] == training_id]
            
            # Use real checkpoint numbers for x-axis
            x_values = []
            for _, row in training_data.iterrows():
                real_checkpoint = row['real_checkpoint_numeric']
                if real_checkpoint == float('inf'):
                    # Fallback to old logic if we couldn't read the real checkpoint
                    numeric_checkpoints = training_data['real_checkpoint_numeric'][training_data['real_checkpoint_numeric'] != float('inf')]
                    if len(numeric_checkpoints) > 0:
                        x_val = max(numeric_checkpoints) + 50  # Add some gap
                    else:
                        x_val = 1
                else:
                    x_val = real_checkpoint
                x_values.append(x_val)
            
            plt.errorbar(x_values, 
                        training_data['mean_deliveries'],
                        yerr=training_data['std_deliveries'],
                        label=f'Training {training_id}', 
                        marker='o', 
                        linewidth=2,
                        capsize=5,
                        color=color_map[training_id])
        
        plt.xlabel('Checkpoint Number')
        plt.ylabel('Mean Final Deliveries')
        plt.title('Mean Final Deliveries vs Checkpoint Number')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Box plot for each checkpoint (aggregated across training_ids)
        plt.subplot(2, 2, 2)
        
        # Prepare data for box plot using real checkpoint numbers
        checkpoint_groups = defaultdict(list)
        checkpoint_labels = {}
        for _, row in plot_df.iterrows():
            real_cp = row['real_checkpoint_numeric']
            checkpoint_groups[real_cp].extend(row['all_deliveries'])
            checkpoint_labels[real_cp] = real_cp
        
        # Sort checkpoints by real numbers
        sorted_checkpoints = sorted(checkpoint_groups.keys())
        
        box_data = [checkpoint_groups[cp] for cp in sorted_checkpoints]
        
        plt.boxplot(box_data, labels=[checkpoint_labels[cp] for cp in sorted_checkpoints])
        plt.xlabel('Checkpoint')
        plt.ylabel('Final Deliveries')
        plt.title('Distribution of Final Deliveries by Checkpoint\n(All Training IDs Combined)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Heatmap of mean deliveries
        plt.subplot(2, 2, 3)
        
        # Create pivot table for heatmap using real checkpoint numbers
        heatmap_data = plot_df.pivot(index='training_id', 
                                    columns='real_checkpoint_numeric', 
                                    values='mean_deliveries')
        
        # Reorder columns by real checkpoint numbers
        column_order = sorted(heatmap_data.columns)
        heatmap_data = heatmap_data[column_order]
        
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.1f')
        plt.title('Mean Final Deliveries Heatmap\n(Training ID vs Checkpoint)')
        plt.xlabel('Checkpoint')
        plt.ylabel('Training ID')
        
        # Plot 4: Number of simulations per checkpoint
        plt.subplot(2, 2, 4)
        
        sim_counts = plot_df.groupby('real_checkpoint_numeric')['n_simulations'].sum().reset_index()
        sim_counts = sim_counts.sort_values('real_checkpoint_numeric')
        
        plt.bar(range(len(sim_counts)), sim_counts['n_simulations'])
        plt.xlabel('Checkpoint (Episode)')
        plt.ylabel('Total Number of Simulations')
        plt.title('Number of Simulations per Checkpoint')
        plt.xticks(range(len(sim_counts)), sim_counts['real_checkpoint_numeric'], rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'deliveries_vs_checkpoints_{self.map_nr}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a focused line plot
        plt.figure(figsize=(12, 8))
        
        # Find the maximum real checkpoint number for proper x-axis scaling
        all_real_checkpoints = plot_df['real_checkpoint_numeric'][plot_df['real_checkpoint_numeric'] != float('inf')]
        max_real_checkpoint = max(all_real_checkpoints) if len(all_real_checkpoints) > 0 else 0
        
        for training_id in unique_training_ids:
            training_data = plot_df[plot_df['training_id'] == training_id]
            
            # Use real checkpoint numbers for x-axis
            x_values = []
            x_labels = []
            for _, row in training_data.iterrows():
                real_checkpoint = row['real_checkpoint_numeric']
                if real_checkpoint == float('inf'):
                    # Fallback if we couldn't read the real checkpoint
                    x_val = max_real_checkpoint + 50
                else:
                    x_val = real_checkpoint
                x_values.append(x_val)
                x_labels.append(row['checkpoint'])
            
            plt.plot(x_values, training_data['mean_deliveries'],
                    label=f'Training {training_id}', 
                    marker='o', 
                    linewidth=3,
                    markersize=8,
                    color=color_map[training_id])
            
            # Add error bars
            plt.errorbar(x_values, training_data['mean_deliveries'],
                        yerr=training_data['std_deliveries'],
                        color=color_map[training_id],
                        alpha=0.5,
                        capsize=3)
        
        # Add grey trend line showing overall tendency across all training_ids
        overall_trend = plot_df.groupby('real_checkpoint_numeric')['mean_deliveries'].mean().reset_index()
        overall_trend = overall_trend.sort_values('real_checkpoint_numeric')
        
        # Handle inf values for plotting
        trend_x = []
        trend_y = []
        for _, row in overall_trend.iterrows():
            real_checkpoint = row['real_checkpoint_numeric']
            if real_checkpoint == float('inf'):
                x_val = max_real_checkpoint + 50
            else:
                x_val = real_checkpoint
            trend_x.append(x_val)
            trend_y.append(row['mean_deliveries'])
        
        plt.plot(trend_x, trend_y, color='grey', linewidth=2, linestyle='--', 
                alpha=0.8, label='Overall Trend', marker='s', markersize=4)
        
        plt.xlabel('Episode Number', fontsize=12)
        plt.ylabel('Mean Final Deliveries', fontsize=12)
        plt.title(f'Delivery Performance vs Training Episode\nMap: {self.map_nr}', 
                 fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Custom x-axis labeling with real checkpoint positions
        unique_checkpoints = plot_df['real_checkpoint_numeric'].unique()
        checkpoint_positions = [cp for cp in unique_checkpoints if cp != float('inf')]
        
        # Sort by position and use real numbers as labels
        if checkpoint_positions:
            checkpoint_positions = sorted(checkpoint_positions)
            plt.xticks(checkpoint_positions, checkpoint_positions)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'deliveries_vs_checkpoints_focused_{self.map_nr}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a trajectory plot showing individual simulation paths
        plt.figure(figsize=(14, 10))
        
        # Find the maximum real checkpoint number for proper scaling
        all_real_checkpoints = []
        for training_id, checkpoints in self.checkpoint_data.items():
            for checkpoint_name, checkpoint_info in checkpoints.items():
                real_checkpoint_num = checkpoint_info['real_checkpoint_numeric']
                if real_checkpoint_num != float('inf'):
                    all_real_checkpoints.append(real_checkpoint_num)
        
        max_real_checkpoint = max(all_real_checkpoints) if len(all_real_checkpoints) > 0 else 0
        
        # Prepare data for trajectory plotting
        trajectory_data = []
        
        # Debug: Let's see what simulation data we have
        print("Debug: Checking simulation data structure...")
        sim_count_by_training = {}
        
        for training_id, checkpoints in self.checkpoint_data.items():
            print(f"Training {training_id}:")
            sim_count_by_training[training_id] = {}
            
            for checkpoint_name, checkpoint_info in checkpoints.items():
                # Use consistent structure
                simulations = checkpoint_info['simulations']
                real_checkpoint_num = checkpoint_info['real_checkpoint_numeric']
                
                # Use real checkpoint number, with fallback
                if real_checkpoint_num == float('inf'):
                    checkpoint_num = max_real_checkpoint + 50  # Add gap for final
                else:
                    checkpoint_num = real_checkpoint_num
                
                print(f"  Checkpoint {checkpoint_name} (episode {checkpoint_num}): {len(simulations)} simulations")
                sim_count_by_training[training_id][checkpoint_name] = len(simulations)
                
                for sim in simulations:
                    sim_id = sim['simulation']
                    trajectory_data.append({
                        'checkpoint': checkpoint_num,
                        'deliveries': sim['final_deliveries'],
                        'training_id': training_id,
                        'simulation': sim_id,
                        'checkpoint_name': checkpoint_name
                    })
        
        # Convert to DataFrame for easier manipulation
        traj_df = pd.DataFrame(trajectory_data)
        
        print(f"Debug: Total trajectory data points: {len(traj_df)}")
        if not traj_df.empty:
            print(f"Debug: Unique training_ids: {traj_df['training_id'].unique()}")
            print(f"Debug: Unique checkpoints: {sorted(traj_df['checkpoint'].unique())}")
            print(f"Debug: Total unique simulations: {len(traj_df['simulation'].unique())}")
            
            # Check how many simulations appear in multiple checkpoints
            sim_checkpoint_counts = traj_df.groupby('simulation')['checkpoint'].count()
            multi_checkpoint_sims = sim_checkpoint_counts[sim_checkpoint_counts > 1]
            print(f"Debug: Simulations appearing in multiple checkpoints: {len(multi_checkpoint_sims)}")
            
        if not traj_df.empty:
            # Plot 1: All trajectories with training_id color coding
            plt.subplot(1, 3, 1)
            
            trajectories_plotted = 0
            
            # Group trajectories by training_id and simulation name patterns
            for training_id in unique_training_ids:
                training_traj = traj_df[traj_df['training_id'] == training_id]
                
                # Try to match simulations across checkpoints by name similarity
                # Extract simulation numbers from names like "simulation_0", "simulation_1", etc.
                training_traj['sim_number'] = training_traj['simulation'].str.extract(r'simulation_?(\d+)')[0]
                
                # Group by simulation number and plot trajectories
                for sim_num in training_traj['sim_number'].dropna().unique():
                    sim_data = training_traj[training_traj['sim_number'] == sim_num].sort_values('checkpoint')
                    if len(sim_data) > 1:  # Only plot if there are multiple checkpoints
                        plt.plot(sim_data['checkpoint'], sim_data['deliveries'], 
                               color=color_map[training_id], alpha=0.3, linewidth=1)
                        trajectories_plotted += 1
                
                # Plot mean trajectory for this training_id
                mean_traj = training_traj.groupby('checkpoint')['deliveries'].mean().reset_index()
                if len(mean_traj) > 1:
                    plt.plot(mean_traj['checkpoint'], mean_traj['deliveries'], 
                           color=color_map[training_id], linewidth=3, 
                           label=f'Training {training_id} (mean)', marker='o', markersize=6)
            
            print(f"Debug: Individual trajectories plotted: {trajectories_plotted}")
            
            plt.xlabel('Checkpoint (Episode)')
            plt.ylabel('Final Deliveries')
            plt.title('Individual Simulation Trajectories\n(Thin lines = individual, thick lines = mean)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Show all data points (scatter plot) to verify data exists
            plt.subplot(1, 3, 2)
            
            for training_id in unique_training_ids:
                training_traj = traj_df[traj_df['training_id'] == training_id]
                plt.scatter(training_traj['checkpoint'], training_traj['deliveries'], 
                           color=color_map[training_id], alpha=0.6, s=20, 
                           label=f'Training {training_id}')
            
            plt.xlabel('Checkpoint (Episode)')
            plt.ylabel('Final Deliveries')
            plt.title('All Data Points\n(Scatter plot for verification)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Show simulation count per checkpoint per training
            plt.subplot(1, 3, 3)
            
            # Create a bar plot showing simulation counts using real checkpoint numbers
            checkpoint_numbers = sorted(traj_df['checkpoint'].unique())
            training_ids = sorted(traj_df['training_id'].unique())
            
            x_pos = np.arange(len(checkpoint_numbers))
            width = 0.8 / len(training_ids)
            
            for i, training_id in enumerate(training_ids):
                counts = []
                for cp_num in checkpoint_numbers:
                    count = len(traj_df[(traj_df['training_id'] == training_id) & 
                                       (traj_df['checkpoint'] == cp_num)])
                    counts.append(count)
                
                plt.bar(x_pos + i * width, counts, width, 
                       label=f'Training {training_id}', 
                       color=color_map[training_id], alpha=0.7)
            
            plt.xlabel('Checkpoint (Episode)')
            plt.ylabel('Number of Simulations')
            plt.title('Simulation Count per Checkpoint')
            plt.xticks(x_pos + width * (len(training_ids) - 1) / 2, checkpoint_numbers, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        else:
            # If no trajectory data, show a message
            plt.subplot(1, 3, 1)
            plt.text(0.5, 0.5, 'No trajectory data available\nCheck simulation naming consistency', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('Trajectory Plot - No Data')
            
            plt.subplot(1, 3, 2)
            plt.text(0.5, 0.5, 'Debug: Check console output\nfor data structure information', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('Debug Information')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'simulation_trajectories_{self.map_nr}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {output_dir}")
        
        return plot_df
    
    def generate_summary_report(self, plot_df, output_dir):
        """Generate a summary report of the checkpoint analysis."""
        output_dir = Path(output_dir)
        
        summary = []
        summary.append("# Checkpoint Delivery Comparison Analysis Report\n")
        summary.append(f"Generated on: {pd.Timestamp.now()}\n")
        
        # Analysis parameters
        summary.append("## Analysis Parameters")
        summary.append(f"Map: {self.map_nr}")
        summary.append(f"Game version: {self.game_version}")
        summary.append(f"Base directory: {self.base_map_dir}\n")
        
        # Summary statistics
        summary.append("## Summary Statistics")
        
        total_training_ids = len(self.checkpoint_data)
        total_checkpoints = sum(len(checkpoints) for checkpoints in self.checkpoint_data.values())
        total_simulations = plot_df['n_simulations'].sum()
        
        summary.append(f"Total training IDs analyzed: {total_training_ids}")
        summary.append(f"Total checkpoints analyzed: {total_checkpoints}")
        summary.append(f"Total simulations analyzed: {total_simulations}\n")
        
        # Training ID breakdown
        summary.append("## Training ID Breakdown")
        for training_id in sorted(self.checkpoint_data.keys()):
            checkpoints = list(self.checkpoint_data[training_id].keys())
            n_checkpoints = len(checkpoints)
            n_simulations = sum(len(checkpoint_info['simulations']) for checkpoint_info in self.checkpoint_data[training_id].values())
            summary.append(f"- Training {training_id}: {n_checkpoints} checkpoints, {n_simulations} simulations")
            summary.append(f"  Checkpoints: {', '.join(sorted(checkpoints, key=self.extract_numeric_checkpoint))}")
        summary.append("")
        
        # Performance analysis
        summary.append("## Performance Analysis")
        
        # Best performing checkpoint overall
        best_checkpoint = plot_df.loc[plot_df['mean_deliveries'].idxmax()]
        summary.append(f"Best performing checkpoint overall:")
        summary.append(f"- Training {best_checkpoint['training_id']}, Checkpoint {best_checkpoint['checkpoint']}")
        summary.append(f"- Mean deliveries: {best_checkpoint['mean_deliveries']:.2f} ± {best_checkpoint['std_deliveries']:.2f}")
        summary.append(f"- Based on {best_checkpoint['n_simulations']} simulations\n")
        
        # Best performing training ID (average across checkpoints)
        training_performance = plot_df.groupby('training_id')['mean_deliveries'].mean().sort_values(ascending=False)
        summary.append(f"Training ID performance ranking (average across checkpoints):")
        for i, (training_id, avg_performance) in enumerate(training_performance.items(), 1):
            summary.append(f"{i}. Training {training_id}: {avg_performance:.2f} average deliveries")
        summary.append("")
        
        # Checkpoint progression analysis
        summary.append("## Checkpoint Progression Analysis")
        
        # Check if there's a general trend across training IDs
        checkpoint_avg = plot_df.groupby('checkpoint_numeric')['mean_deliveries'].mean().sort_index()
        
        if len(checkpoint_avg) > 1:
            # Calculate correlation between checkpoint number and performance
            numeric_checkpoints = [cp for cp in checkpoint_avg.index if cp != float('inf')]
            if len(numeric_checkpoints) > 1:
                checkpoint_values = [checkpoint_avg[cp] for cp in numeric_checkpoints]
                correlation = np.corrcoef(numeric_checkpoints, checkpoint_values)[0, 1]
                summary.append(f"Correlation between checkpoint number and delivery performance: {correlation:.3f}")
                
                if correlation > 0.5:
                    summary.append("→ Strong positive trend: Performance improves with training")
                elif correlation > 0.2:
                    summary.append("→ Moderate positive trend: Some improvement with training")
                elif correlation < -0.5:
                    summary.append("→ Strong negative trend: Performance decreases with training")
                elif correlation < -0.2:
                    summary.append("→ Moderate negative trend: Some performance decline with training")
                else:
                    summary.append("→ No clear trend: Performance varies without clear pattern")
        
        summary.append("")
        
        # Generated files
        summary.append("## Generated Files")
        summary.append(f"- deliveries_vs_checkpoints_{self.map_nr}.png")
        summary.append(f"- deliveries_vs_checkpoints_focused_{self.map_nr}.png")
        summary.append(f"- simulation_trajectories_{self.map_nr}.png")
        summary.append("- checkpoint_analysis_summary.md")
        
        # Save summary
        with open(output_dir / 'checkpoint_analysis_summary.md', 'w') as f:
            f.write('\n'.join(summary))
        
        print("Summary report generated!")
    
    def run_analysis(self, output_dir=None):
        """Run the complete checkpoint comparison analysis."""
        if not self.map_nr:
            print("Error: Missing required parameter map_nr.")
            return
        
        # Use map directory as output if not specified
        if output_dir is None:
            output_dir = self.base_map_dir
        else:
            output_dir = Path(output_dir)
            
        print("Starting checkpoint delivery comparison analysis...")
        print(f"Map: {self.map_nr}")
        print(f"Game version: {self.game_version}")
        print(f"Search path: {self.base_map_dir}")
        print(f"Output directory: {output_dir}")
        print()
        
        # Load all data
        print("1. Loading data from all training directories and checkpoints...")
        self.load_all_data()
        
        if not self.checkpoint_data:
            print("No valid data found!")
            return
        
        # Create plots
        print("2. Generating comparison plots...")
        plot_df = self.plot_deliveries_vs_checkpoints(output_dir)
        
        if plot_df is not None:
            print("3. Generating summary report...")
            self.generate_summary_report(plot_df, output_dir)
            
            print(f"\nAnalysis complete! Results saved to: {output_dir}")
            print(f"Open {output_dir}/checkpoint_analysis_summary.md for a summary of results.")
        else:
            print("Analysis failed - no plots generated.")


def main():
    """Main function to run the checkpoint comparison analysis."""
    parser = argparse.ArgumentParser(description='Compare deliveries across checkpoints and training_ids for a specific map')
    parser.add_argument('--map_nr', type=str, required=True,
                       help='Map number/name (e.g., "baseline_division_of_labor")')
    parser.add_argument('--cluster', type=str, default='cuenca',
                       help='Base cluster (default: cuenca)')
    parser.add_argument('--game_version', type=str, default='classic', 
                       choices=['classic', 'competition', 'classic_collision'],
                       help='Game version (default: classic)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: map directory)')
    parser.add_argument('--num_agents', type=int, default=2,
                       help='Number of agents in the simulation (default: 2)')
    parser.add_argument('--base_dir', type=str, default=None,
                       help='Optional explicit map base directory containing simulations/ and simulation_figures/')
    parser.add_argument('--study_name', type=str, default='',
                       help='Study name for simulation folders (default: empty, use /simulations/ directly)')
    parser.add_argument('--game_type', type=str, default='classic',
                       help='Game type for folder organization (default: classic)')

    args = parser.parse_args()

    # Set base cluster directory
    if args.cluster.lower() == 'cuenca':
        base_cluster_dir = ""
    elif args.cluster.lower() == 'brigit':
        base_cluster_dir = "/mnt/lustre/home/samuloza/"
    elif args.cluster.lower() == 'local':
        base_cluster_dir = "C:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
    else:
        print(f"Unknown cluster: {args.cluster}")
        return

    # Create analyzer and run analysis
    study_name = (args.study_name or '').strip()

    analyzer = CheckpointDeliveryAnalyzer(
        base_cluster_dir=base_cluster_dir,
        map_nr=args.map_nr,
        game_version=args.game_version,
        num_agents=args.num_agents,
        study_name=study_name,
        game_type=args.game_type,
        base_dir=args.base_dir
    )

    analyzer.run_analysis(args.output_dir)


if __name__ == "__main__":
    main()