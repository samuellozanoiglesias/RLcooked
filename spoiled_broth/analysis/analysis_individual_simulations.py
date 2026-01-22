#!/usr/bin/env python3
"""
Comprehensive simulation analysis script for spoiled_broth experiments.

This script analyzes simulation.csv and meaningful_actions.csv files from all simulations
to generate aggregated and non-aggregated plots for understanding agent behavior and performance.

The script expects the following data structure:
- simulation.csv: agent_id, frame, second, x, y, tile_x, tile_y, item, score
- meaningful_actions.csv: Contains action_category_name and other action metadata

Usage:
nohup python3 analysis_individual_simulations.py --cluster cuenca --map_nr baseline_division_of_labor --training_id 2025-09-13_13-27-52 --checkpoint_number final > log_analysis_simulations.out 2>&1 &

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
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class SimulationAnalyzer:
    """Main class for analyzing simulation data."""
    
    def __init__(self, base_dir="/data", map_nr=None, training_id=None, 
                 checkpoint_number=None, game_version="classic"):
        """
        Initialize the analyzer with specific parameters.
        
        Args:
            base_dir: Base directory (e.g., "/data")
            map_nr: Map number/name (e.g., "baseline_division_of_labor")
            training_id: Training ID (e.g., "2025-09-13_13-27-52")
            checkpoint_number: Checkpoint number (e.g., "final", "50", etc.)
            game_version: Game version ("classic" or "competition")
        """
        self.base_dir = Path(base_dir)
        self.map_nr = map_nr
        self.training_id = training_id
        self.checkpoint_number = checkpoint_number
        self.game_version = game_version
        
        # Construct the specific simulation directory path
        if all([map_nr, training_id, checkpoint_number]):
            self.simulation_base_path = self.base_dir
        else:
            self.simulation_base_path = None
            
        self.simulation_data = {}
        self.aggregated_data = {}
        
    def resolve_checkpoint_number(self, training_dir):
        """
        Resolve checkpoint number when it's "final" by reading training_stats.csv.
        
        Args:
            training_dir: Path to the training directory
            
        Returns:
            Resolved checkpoint number as string
        """
        if self.checkpoint_number != "final":
            return self.checkpoint_number
            
        training_stats_path = Path(training_dir) / "training_stats.csv"
        
        if not training_stats_path.exists():
            print(f"Warning: training_stats.csv not found at {training_stats_path}")
            print("Using 'final' as checkpoint_number")
            return "final"
            
        try:
            # Read the CSV and get the last episode number
            df = pd.read_csv(training_stats_path)
            
            if df.empty:
                print("Warning: training_stats.csv is empty")
                return "final"
                
            # Get the first column (episode) from the last row and add 1
            last_episode = df.iloc[-1, 0]  # First column of last row
            resolved_checkpoint = str(int(last_episode) + 1)
            
            print(f"Resolved checkpoint_number 'final' to '{resolved_checkpoint}' based on training_stats.csv")
            return resolved_checkpoint
            
        except Exception as e:
            print(f"Error reading training_stats.csv: {e}")
            print("Using 'final' as checkpoint_number")
            return "final"
        
    def find_simulation_directories(self):
        """Find all simulation directories in the specified path."""
        simulation_dirs = []
        
        if not self.simulation_base_path:
            print("Error: No simulation base path specified. Please provide map_nr, training_id, and checkpoint_number.")
            return simulation_dirs
            
        if not self.simulation_base_path.exists():
            print(f"Error: Simulation directory does not exist: {self.simulation_base_path}")
            return simulation_dirs
        
        print(f"Searching for simulations in: {self.simulation_base_path}")
        
        # Search pattern for simulation directories within the specific path
        pattern = str(self.simulation_base_path / "simulation_*")
        for path in glob.glob(pattern):
            path_obj = Path(path)
            if path_obj.is_dir():
                # Create metadata from the known parameters
                metadata = {
                    'training_id': f"Training_{self.training_id}",
                    'map': self.map_nr,
                    'game_type': self.game_version,
                    'checkpoint': self.checkpoint_number,
                    'simulation_timestamp': path_obj.name.replace('simulation_', '')
                }
                simulation_dirs.append((path_obj, metadata))
                    
        print(f"Found {len(simulation_dirs)} simulation directories")
        return simulation_dirs
    
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
                    print(f"  Found AGENT_INITIALIZATION_PERIOD: {initialization_period} seconds")
                else:
                    print(f"  Warning: AGENT_INITIALIZATION_PERIOD not found in config.txt, using default: {initialization_period}")
                    
            except Exception as e:
                print(f"  Error reading config.txt: {e}, using default initialization period: {initialization_period}")
        else:
            print(f"  Warning: config.txt not found, using default initialization period: {initialization_period}")
            
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
        else:
            print("  Warning: 'second' column not found in simulation data, cannot convert to seconds")
            # If no seconds column, create one assuming some frame rate (e.g., 30 FPS)
            # This is a fallback, but we should warn the user
            sim_data = sim_data.copy()
            assumed_fps = 30
            sim_data['adjusted_second'] = (sim_data['frame'] / assumed_fps) - initialization_period
            sim_data['adjusted_second'] = sim_data['adjusted_second'].clip(lower=0)
            print(f"  Using assumed frame rate of {assumed_fps} FPS for time conversion")
            return sim_data
    
    def load_simulation_data(self, simulation_dirs):
        """Load all simulation and meaningful actions data."""
        for sim_dir, metadata in simulation_dirs:
            try:
                # Load simulation.csv
                sim_csv = sim_dir / "simulation.csv"
                actions_csv = sim_dir / "meaningful_actions.csv"
                
                if sim_csv.exists() and actions_csv.exists():
                    sim_data = pd.read_csv(sim_csv)
                    actions_data = pd.read_csv(actions_csv)
                    
                    # Read config file to get initialization period
                    initialization_period = self.read_config_file(sim_dir)
                    
                    # Convert frames to adjusted seconds
                    sim_data = self.convert_frames_to_seconds(sim_data, initialization_period)
                    
                    # Create a unique key for this simulation
                    sim_key = f"{metadata.get('simulation_timestamp', 'unknown')}"
                    
                    self.simulation_data[sim_key] = {
                        'simulation': sim_data,
                        'actions': actions_data,
                        'metadata': metadata,
                        'path': sim_dir,
                        'initialization_period': initialization_period
                    }
                    
                    print(f"Loaded data for {sim_key}")
                    print(f"  - Simulation data: {len(sim_data)} rows")
                    print(f"  - Actions data: {len(actions_data)} rows")
                else:
                    missing_files = []
                    if not sim_csv.exists():
                        missing_files.append("simulation.csv")
                    if not actions_csv.exists():
                        missing_files.append("meaningful_actions.csv")
                    print(f"Missing files in {sim_dir}: {', '.join(missing_files)}")
                    
            except Exception as e:
                print(f"Error loading data from {sim_dir}: {e}")
    
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
        
        return pd.concat(delivery_events, ignore_index=True)
    
    def compute_agent_distance(self, sim_data):
        """Compute distance between agents over time."""
        distances = []
        time_column = 'adjusted_second' if 'adjusted_second' in sim_data.columns else 'frame'
        
        for time_point in sim_data[time_column].unique():
            time_data = sim_data[sim_data[time_column] == time_point]
            if len(time_data) >= 2:
                agents = time_data.groupby('agent_id')[['x', 'y']].first()
                if len(agents) >= 2:
                    agent_positions = agents.values
                    # Calculate Euclidean distance between first two agents
                    if len(agent_positions) >= 2:
                        dist = np.sqrt((agent_positions[0][0] - agent_positions[1][0])**2 + 
                                     (agent_positions[0][1] - agent_positions[1][1])**2)
                        distances.append({time_column: time_point, 'distance': dist})
        
        return pd.DataFrame(distances)
    
    def plot_deliveries_over_time(self, save_dir):
        """Plot 1: Number of deliveries over time for each simulation."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Individual simulations - create separate plot for each
        for sim_key in self.simulation_data.keys():
            sim_data = self.simulation_data[sim_key]['simulation']
            deliveries = self.compute_deliveries(sim_data)
            
            # Determine time column to use
            time_column = 'adjusted_second' if 'adjusted_second' in deliveries.columns else 'frame'
            time_label = 'Time (seconds)' if time_column == 'adjusted_second' else 'Frame'
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot individual agent deliveries and total
            for agent_id in deliveries['agent_id'].unique():
                agent_data = deliveries[deliveries['agent_id'] == agent_id]
                ax.plot(agent_data[time_column], agent_data['cumulative_deliveries'], 
                       label=f'Agent {agent_id}', linewidth=2)
            
            # Plot total deliveries
            total_deliveries = deliveries.groupby(time_column)['cumulative_deliveries'].sum().reset_index()
            ax.plot(total_deliveries[time_column], total_deliveries['cumulative_deliveries'], 
                   label='Total', linewidth=3, linestyle='--', color='black')
            
            ax.set_title(f'Deliveries Over Time\n{sim_key}')
            ax.set_xlabel(time_label)
            ax.set_ylabel('Cumulative Deliveries')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save individual plot
            safe_sim_key = sim_key.replace('/', '_').replace(':', '_')
            plt.tight_layout()
            plt.savefig(save_dir / f'deliveries_over_time_{safe_sim_key}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Aggregated plot by training_id and map
        fig, ax = plt.subplots(figsize=(14, 8))
        
        aggregated_deliveries = defaultdict(list)
        time_column = None
        
        for sim_key, data in self.simulation_data.items():
            metadata = data['metadata']
            key = f"{metadata['training_id']}_{metadata['map']}"
            
            sim_data = data['simulation']
            deliveries = self.compute_deliveries(sim_data)
            
            # Determine time column (should be consistent across simulations)
            if time_column is None:
                time_column = 'adjusted_second' if 'adjusted_second' in deliveries.columns else 'frame'
            
            total_deliveries = deliveries.groupby(time_column)['cumulative_deliveries'].sum()
            
            aggregated_deliveries[key].append(total_deliveries)
        
        time_label = 'Time (seconds)' if time_column == 'adjusted_second' else 'Frame'
        
        # Plot mean and std for each training_id + map combination
        for key, delivery_series in aggregated_deliveries.items():
            if delivery_series:
                # Align all series to same time range
                all_times = set()
                for series in delivery_series:
                    all_times.update(series.index)
                all_times = sorted(all_times)
                
                aligned_data = []
                for series in delivery_series:
                    aligned_series = series.reindex(all_times, fill_value=series.iloc[-1] if len(series) > 0 else 0)
                    aligned_data.append(aligned_series.values)
                
                if aligned_data:
                    mean_deliveries = np.mean(aligned_data, axis=0)
                    std_deliveries = np.std(aligned_data, axis=0)
                    
                    ax.plot(all_times, mean_deliveries, label=key, linewidth=2)
                    ax.fill_between(all_times, 
                                  mean_deliveries - std_deliveries,
                                  mean_deliveries + std_deliveries, 
                                  alpha=0.2)
        
        ax.set_title('Aggregated Deliveries Over Time by Training ID and Map')
        ax.set_xlabel(time_label)
        ax.set_ylabel('Cumulative Deliveries')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(save_dir / 'deliveries_over_time_aggregated.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_action_histograms(self, save_dir):
        """Plot 2: Histogram of action categories."""
        save_dir = Path(save_dir)
        
        # Collect all action categories
        all_actions = []
        for sim_key, data in self.simulation_data.items():
            if 'action_category_name' in data['actions'].columns:
                actions_with_sim = data['actions'].copy()
                actions_with_sim['sim_key'] = sim_key
                actions_with_sim['training_map'] = f"{data['metadata']['training_id']}_{data['metadata']['map']}"
                all_actions.append(actions_with_sim)
        
        if not all_actions:
            print("No action_category_name column found in meaningful_actions.csv files")
            return
        
        all_actions_df = pd.concat(all_actions, ignore_index=True)
        
        # Individual simulation histograms - create separate plot for each
        for sim_key in all_actions_df['sim_key'].unique():
            sim_actions = all_actions_df[all_actions_df['sim_key'] == sim_key]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Count actions by category (vertical bars with actions on x-axis)
            if 'agent_id' in sim_actions.columns:
                # Create grouped bar chart with agents as different bars for each action
                action_counts = sim_actions.groupby(['action_category_name', 'agent_id']).size().unstack(fill_value=0)
                action_counts.plot(kind='bar', ax=ax)
                ax.set_title(f'Action Categories\n{sim_key}')
                ax.set_xlabel('Action Category')
                ax.legend(title='Agent ID', bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                action_counts = sim_actions['action_category_name'].value_counts()
                action_counts.plot(kind='bar', ax=ax)
                ax.set_title(f'Action Categories\n{sim_key}')
                ax.set_xlabel('Action Category')
            
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=90)
            
            # Save individual plot
            safe_sim_key = sim_key.replace('/', '_').replace(':', '_')
            plt.tight_layout()
            plt.savefig(save_dir / f'action_histograms_{safe_sim_key}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Aggregated histogram by training_id and map
        fig, ax = plt.subplots(figsize=(14, 8))
        
        training_map_actions = all_actions_df.groupby(['action_category_name', 'training_map']).size().unstack(fill_value=0)
        training_map_actions.plot(kind='bar', ax=ax)
        
        ax.set_title('Aggregated Action Categories by Training ID and Map')
        ax.set_xlabel('Action Category')
        ax.set_ylabel('Total Count')
        ax.tick_params(axis='x', rotation=90)
        ax.legend(title='Training ID + Map', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'action_histograms_aggregated.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_deliveries_vs_actions_correlation(self, save_dir):
        """Plot 3: Correlation between deliveries and action categories."""
        save_dir = Path(save_dir)
        
        # Prepare data for correlation analysis
        correlation_data = []
        
        for sim_key, data in self.simulation_data.items():
            sim_data = data['simulation']
            actions_data = data['actions']
            
            # Calculate total deliveries
            deliveries = self.compute_deliveries(sim_data)
            total_deliveries = deliveries['cumulative_deliveries'].max()
            
            # Calculate action counts
            if 'action_category_name' in actions_data.columns:
                action_counts = actions_data['action_category_name'].value_counts().to_dict()
                
                # Also calculate per agent if agent_id exists
                if 'agent_id' in actions_data.columns:
                    agent_deliveries = deliveries.groupby('agent_id')['cumulative_deliveries'].max()
                    agent_actions = actions_data.groupby(['agent_id', 'action_category_name']).size().unstack(fill_value=0)
                    
                    for agent_id in agent_deliveries.index:
                        row_data = {
                            'sim_key': sim_key,
                            'agent_id': agent_id,
                            'deliveries': agent_deliveries[agent_id],
                            'training_map': f"{data['metadata']['training_id']}_{data['metadata']['map']}"
                        }
                        
                        if agent_id in agent_actions.index:
                            for action_cat in agent_actions.columns:
                                row_data[f'action_{action_cat}'] = agent_actions.loc[agent_id, action_cat]
                        
                        correlation_data.append(row_data)
                
                # Overall simulation data
                row_data = {
                    'sim_key': sim_key,
                    'agent_id': 'total',
                    'deliveries': total_deliveries,
                    'training_map': f"{data['metadata']['training_id']}_{data['metadata']['map']}"
                }
                
                for action_cat, count in action_counts.items():
                    row_data[f'action_{action_cat}'] = count
                
                correlation_data.append(row_data)
        
        if not correlation_data:
            print("No correlation data could be prepared")
            return
        
        corr_df = pd.DataFrame(correlation_data)
        
        # Plot correlation matrix
        action_cols = [col for col in corr_df.columns if col.startswith('action_')]
        if action_cols and 'deliveries' in corr_df.columns:
            
            # Individual agent analysis - create separate plot for each agent
            agent_data = corr_df[corr_df['agent_id'] != 'total']
            if not agent_data.empty:
                # Get unique agent IDs
                unique_agents = agent_data['agent_id'].unique()
                
                for agent_id in unique_agents:
                    agent_subset = agent_data[agent_data['agent_id'] == agent_id]
                    
                    if len(agent_subset) > 1:  # Need at least 2 data points for correlation
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                        
                        # Agent correlation matrix
                        corr_matrix = agent_subset[['deliveries'] + action_cols].corr()
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1)
                        ax1.set_title(f'Correlation: Agent {agent_id} Deliveries vs Actions')
                        
                        # Scatter plots for strongest correlations
                        deliveries_vs_actions = agent_subset[['deliveries'] + action_cols].corr()['deliveries'].abs().sort_values(ascending=False)
                        if len(deliveries_vs_actions) > 1:
                            strongest_action = deliveries_vs_actions.index[1]  # Skip 'deliveries' itself
                            
                            ax2.scatter(agent_subset[strongest_action], agent_subset['deliveries'], alpha=0.6)
                            ax2.set_xlabel(strongest_action)
                            ax2.set_ylabel('Deliveries')
                            ax2.set_title(f'Strongest Correlation: {strongest_action}')
                        
                        plt.tight_layout()
                        plt.savefig(save_dir / f'deliveries_vs_actions_correlation_agent_{agent_id}.png', dpi=300, bbox_inches='tight')
                        plt.close()
            
            # Total simulation analysis
            total_data = corr_df[corr_df['agent_id'] == 'total']
            if not total_data.empty:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                corr_matrix = total_data[['deliveries'] + action_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Correlation: Total Deliveries vs Actions')
                
                plt.tight_layout()
                plt.savefig(save_dir / 'deliveries_vs_actions_correlation_total.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def plot_markov_matrices(self, save_dir):
        """Plot 4: Markov transition matrices for action sequences."""
        save_dir = Path(save_dir)
        
        # Collect action sequences
        all_sequences = {'total': [], 'by_agent': defaultdict(list)}
        
        for sim_key, data in self.simulation_data.items():
            actions_data = data['actions']
            
            if 'action_category_name' in actions_data.columns:
                # Sort by timestamp or frame if available
                if 'frame' in actions_data.columns:
                    actions_data = actions_data.sort_values('frame')
                elif 'timestamp' in actions_data.columns:
                    actions_data = actions_data.sort_values('timestamp')
                
                # Overall sequence
                sequence = actions_data['action_category_name'].tolist()
                all_sequences['total'].extend(sequence)
                
                # By agent sequences
                if 'agent_id' in actions_data.columns:
                    for agent_id in actions_data['agent_id'].unique():
                        agent_actions = actions_data[actions_data['agent_id'] == agent_id]
                        agent_sequence = agent_actions['action_category_name'].tolist()
                        all_sequences['by_agent'][agent_id].extend(agent_sequence)
        
        def create_transition_matrix(sequence):
            """Create transition matrix from action sequence."""
            if len(sequence) < 2:
                return pd.DataFrame()
            
            unique_actions = list(set(sequence))
            transition_counts = defaultdict(lambda: defaultdict(int))
            
            for i in range(len(sequence) - 1):
                from_action = sequence[i]
                to_action = sequence[i + 1]
                transition_counts[from_action][to_action] += 1
            
            # Convert to matrix
            matrix = pd.DataFrame(0, index=unique_actions, columns=unique_actions)
            for from_action in transition_counts:
                total_from = sum(transition_counts[from_action].values())
                for to_action in transition_counts[from_action]:
                    if total_from > 0:
                        matrix.loc[from_action, to_action] = transition_counts[from_action][to_action] / total_from
            
            return matrix
        
        # Plot overall Markov matrix
        if all_sequences['total']:
            overall_matrix = create_transition_matrix(all_sequences['total'])
            if not overall_matrix.empty:
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(overall_matrix, annot=True, cmap='viridis', ax=ax, fmt='.3f')
                ax.set_title('Markov Transition Matrix - All Agents')
                ax.set_xlabel('To Action')
                ax.set_ylabel('From Action')
                plt.tight_layout()
                plt.savefig(save_dir / 'markov_matrix_all_agents.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Plot by-agent Markov matrices - create separate plot for each agent
        agent_ids = list(all_sequences['by_agent'].keys())
        for agent_id in agent_ids:
            agent_matrix = create_transition_matrix(all_sequences['by_agent'][agent_id])
            if not agent_matrix.empty:
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(agent_matrix, annot=True, cmap='viridis', ax=ax, fmt='.3f')
                ax.set_title(f'Markov Transition Matrix - Agent {agent_id}')
                ax.set_xlabel('To Action')
                ax.set_ylabel('From Action')
                
                plt.tight_layout()
                plt.savefig(save_dir / f'markov_matrix_agent_{agent_id}.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def plot_distance_and_deliveries(self, save_dir):
        """Plot 5: Agent distance over time with deliveries."""
        save_dir = Path(save_dir)
        
        # Individual simulations - create separate plot for each
        for sim_key in self.simulation_data.keys():
            sim_data = self.simulation_data[sim_key]['simulation']
            distances = self.compute_agent_distance(sim_data)
            deliveries = self.compute_deliveries(sim_data)
            
            if not distances.empty and not deliveries.empty:
                fig, ax1 = plt.subplots(figsize=(12, 8))
                
                # Calculate deliveries per frame
                frame_deliveries = deliveries.groupby('frame')['deliveries'].sum().reset_index()
                
                # Determine time column to use
                time_column = 'adjusted_second' if 'adjusted_second' in distances.columns else 'frame'
                time_label = 'Time (seconds)' if time_column == 'adjusted_second' else 'Frame'
                delivery_label = 'Deliveries per second' if time_column == 'adjusted_second' else 'Deliveries per frame'
                
                # Plot distance
                color = 'tab:blue'
                ax1.set_xlabel(time_label)
                ax1.set_ylabel('Distance between agents', color=color)
                line1 = ax1.plot(distances[time_column], distances['distance'], color=color, linewidth=2, label='Distance')
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.grid(True, alpha=0.3)
                
                # Plot deliveries on second y-axis
                ax2 = ax1.twinx()
                color = 'tab:red'
                ax2.set_ylabel(delivery_label, color=color)
                
                # Use same time column for deliveries
                frame_deliveries_time_col = time_column if time_column in frame_deliveries.columns else 'frame'
                bars = ax2.bar(frame_deliveries[frame_deliveries_time_col], frame_deliveries['deliveries'], 
                             alpha=0.6, color=color, label=delivery_label)
                ax2.tick_params(axis='y', labelcolor=color)
                
                # Add legends
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + [bars], labels1 + labels2, loc='upper left')
                
                ax1.set_title(f'Distance & Deliveries Over Time\n{sim_key}')
                
                # Save individual plot
                safe_sim_key = sim_key.replace('/', '_').replace(':', '_')
                plt.tight_layout()
                plt.savefig(save_dir / f'distance_and_deliveries_{safe_sim_key}.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Aggregated plot - average over frames
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        aggregated_distances = defaultdict(list)
        aggregated_deliveries = defaultdict(list)
        
        for sim_key, data in self.simulation_data.items():
            metadata = data['metadata']
            key = f"{metadata['training_id']}_{metadata['map']}"
            
            sim_data = data['simulation']
            distances = self.compute_agent_distance(sim_data)
            deliveries = self.compute_deliveries(sim_data)
            
            if not distances.empty and not deliveries.empty:
                # Determine time column to use
                time_column = 'adjusted_second' if 'adjusted_second' in distances.columns else 'frame'
                
                # Get time-based deliveries
                time_deliveries = deliveries.groupby(time_column)['deliveries'].sum().reset_index()
                
                # Align distances and deliveries by time
                merged_data = distances.merge(time_deliveries, on=time_column, how='inner')
                
                aggregated_distances[key].append(merged_data)
        
        # Plot mean distances and deliveries over frames
        for key, data_list in aggregated_distances.items():
            if data_list:
                # Combine all simulations for this training_map
                time_column = 'adjusted_second' if 'adjusted_second' in data_list[0].columns else 'frame'
                all_frames = set()
                for df in data_list:
                    all_frames.update(df[time_column])
                all_frames = sorted(all_frames)
                
                # Determine time column from first dataset
                time_column = 'adjusted_second' if 'adjusted_second' in data_list[0].columns else 'frame'
                
                # Average distances and deliveries across simulations
                avg_distances = []
                avg_deliveries = []
                
                for time_point in all_frames:
                    time_distances = []
                    time_deliveries = []
                    
                    for df in data_list:
                        time_data = df[df[time_column] == time_point]
                        if not time_data.empty:
                            time_distances.append(time_data['distance'].iloc[0])
                            time_deliveries.append(time_data['deliveries'].iloc[0])
                    
                    if time_distances:
                        avg_distances.append(np.mean(time_distances))
                        avg_deliveries.append(np.mean(time_deliveries))
                    else:
                        avg_distances.append(np.nan)
                        avg_deliveries.append(np.nan)
                
                # Plot distance
                ax1.plot(all_frames, avg_distances, label=f'{key} - Distance', linewidth=2)
        
                # Plot deliveries on second y-axis\n        ax2 = ax1.twinx()\n        for key, data_list in aggregated_distances.items():\n            if data_list:\n                # Same calculation as above for deliveries\n                time_column = 'adjusted_second' if 'adjusted_second' in data_list[0].columns else 'frame'\n                all_frames = set()\n                for df in data_list:\n                    all_frames.update(df[time_column])\n                all_frames = sorted(all_frames)
                
                avg_deliveries = []
                for time_point in all_frames:
                    time_deliveries = []
                    for df in data_list:
                        time_data = df[df[time_column] == time_point]
                        if not time_data.empty:
                            time_deliveries.append(time_data['deliveries'].iloc[0])
                    
                    if time_deliveries:
                        avg_deliveries.append(np.mean(time_deliveries))
                    else:
                        avg_deliveries.append(np.nan)
                
                ax2.plot(all_frames, avg_deliveries, '--', label=f'{key} - Deliveries', linewidth=2)
        
        # Determine label based on time column used
        time_label = 'Time (seconds)' if time_column == 'adjusted_second' else 'Frame'
        delivery_label = 'Deliveries per Second' if time_column == 'adjusted_second' else 'Deliveries per Frame'
        
        ax1.set_xlabel(time_label)
        ax1.set_ylabel('Average Distance', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_ylabel(delivery_label, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax1.set_title('Aggregated Distance & Deliveries Over Time by Training & Map')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'distance_and_deliveries_aggregated.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_distance_vs_deliveries_scatter(self, save_dir):
        """Plot 6: 2D scatter plot of average distance vs total deliveries."""
        save_dir = Path(save_dir)
        
        scatter_data = []
        
        for sim_key, data in self.simulation_data.items():
            sim_data = data['simulation']
            distances = self.compute_agent_distance(sim_data)
            deliveries = self.compute_deliveries(sim_data)
            
            if not distances.empty and not deliveries.empty:
                avg_distance = distances['distance'].mean()
                total_deliveries = deliveries['deliveries'].sum()
                
                scatter_data.append({
                    'sim_key': sim_key,
                    'avg_distance': avg_distance,
                    'total_deliveries': total_deliveries,
                    'training_id': data['metadata']['training_id'],
                    'map': data['metadata']['map'],
                    'training_map': f"{data['metadata']['training_id']}_{data['metadata']['map']}"
                })
        
        if not scatter_data:
            print("No scatter data could be prepared")
            return
        
        scatter_df = pd.DataFrame(scatter_data)
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color by training_map combination
        unique_training_maps = scatter_df['training_map'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_training_maps)))
        
        for i, training_map in enumerate(unique_training_maps):
            data_subset = scatter_df[scatter_df['training_map'] == training_map]
            ax.scatter(data_subset['avg_distance'], data_subset['total_deliveries'], 
                      c=[colors[i]], label=training_map, s=100, alpha=0.7)
        
        ax.set_xlabel('Average Distance Between Agents')
        ax.set_ylabel('Total Deliveries')
        ax.set_title('Average Distance vs Total Deliveries\n(Each point is one simulation)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(scatter_df) > 1:
            correlation = scatter_df['avg_distance'].corr(scatter_df['total_deliveries'])
            ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                   transform=ax.transAxes, fontsize=12, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_dir / 'distance_vs_deliveries_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional analysis: grouped by training_map
        fig, ax = plt.subplots(figsize=(12, 8))
        
        grouped_data = scatter_df.groupby('training_map').agg({
            'avg_distance': ['mean', 'std'],
            'total_deliveries': ['mean', 'std']
        }).reset_index()
        
        grouped_data.columns = ['training_map', 'distance_mean', 'distance_std', 'deliveries_mean', 'deliveries_std']
        
        ax.errorbar(grouped_data['distance_mean'], grouped_data['deliveries_mean'],
                   xerr=grouped_data['distance_std'], yerr=grouped_data['deliveries_std'],
                   fmt='o', markersize=10, capsize=5, capthick=2)
        
        # Label points
        for _, row in grouped_data.iterrows():
            ax.annotate(row['training_map'], 
                       (row['distance_mean'], row['deliveries_mean']),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Average Distance Between Agents (Mean ± Std)')
        ax.set_ylabel('Total Deliveries (Mean ± Std)')
        ax.set_title('Average Distance vs Total Deliveries\n(Grouped by Training ID and Map)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'distance_vs_deliveries_grouped.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, save_dir):
        """Generate a summary report of the analysis."""
        save_dir = Path(save_dir)
        
        summary = []
        summary.append("# Simulation Analysis Summary Report\n")
        summary.append(f"Generated on: {pd.Timestamp.now()}\n")
        
        # Analysis parameters
        summary.append("## Analysis Parameters")
        summary.append(f"Map: {self.map_nr}")
        summary.append(f"Training ID: {self.training_id}")
        summary.append(f"Checkpoint: {self.checkpoint_number}")
        summary.append(f"Game version: {self.game_version}")
        summary.append(f"Analysis path: {self.simulation_base_path}\n")
        
        summary.append(f"Total simulations analyzed: {len(self.simulation_data)}\n")
        
        # Basic statistics
        total_deliveries = []
        avg_distances = []
        action_counts = []
        
        for sim_key, data in self.simulation_data.items():
            sim_data = data['simulation']
            actions_data = data['actions']
            
            # Deliveries
            deliveries = self.compute_deliveries(sim_data)
            if not deliveries.empty:
                total_deliveries.append(deliveries['deliveries'].sum())
            
            # Distances
            distances = self.compute_agent_distance(sim_data)
            if not distances.empty:
                avg_distances.append(distances['distance'].mean())
            
            # Actions
            if 'action_category_name' in actions_data.columns:
                action_counts.append(len(actions_data))
        
        if total_deliveries:
            summary.append(f"## Delivery Statistics")
            summary.append(f"Mean total deliveries per simulation: {np.mean(total_deliveries):.2f}")
            summary.append(f"Std total deliveries per simulation: {np.std(total_deliveries):.2f}")
            summary.append(f"Min/Max total deliveries: {np.min(total_deliveries)}/{np.max(total_deliveries)}\n")
        
        if avg_distances:
            summary.append(f"## Distance Statistics")
            summary.append(f"Mean average distance between agents: {np.mean(avg_distances):.2f}")
            summary.append(f"Std average distance between agents: {np.std(avg_distances):.2f}")
            summary.append(f"Min/Max average distance: {np.min(avg_distances):.2f}/{np.max(avg_distances):.2f}\n")
        
        if action_counts:
            summary.append(f"## Action Statistics")
            summary.append(f"Mean actions per simulation: {np.mean(action_counts):.2f}")
            summary.append(f"Std actions per simulation: {np.std(action_counts):.2f}")
            summary.append(f"Min/Max actions per simulation: {np.min(action_counts)}/{np.max(action_counts)}\n")
        
        # List all simulations analyzed
        summary.append("## Simulations Analyzed")
        for sim_key in sorted(self.simulation_data.keys()):
            summary.append(f"- {sim_key}")
        summary.append("")
        
        # Generated plots
        summary.append("## Generated Plots")
        
        # Count different types of plots
        plot_files = list(save_dir.glob('*.png'))
        plot_counts = {
            'deliveries_over_time': len([f for f in plot_files if 'deliveries_over_time_' in f.name and 'aggregated' not in f.name]),
            'action_histograms': len([f for f in plot_files if 'action_histograms_' in f.name and 'aggregated' not in f.name]),
            'distance_and_deliveries': len([f for f in plot_files if 'distance_and_deliveries_' in f.name and 'aggregated' not in f.name]),
            'markov_matrices': len([f for f in plot_files if 'markov_matrix_agent_' in f.name]),
        }
        
        summary.append(f"Individual simulation plots:")
        summary.append(f"- Deliveries over time plots: {plot_counts['deliveries_over_time']}")
        summary.append(f"- Action histogram plots: {plot_counts['action_histograms']}")
        summary.append(f"- Distance and deliveries plots: {plot_counts['distance_and_deliveries']}")
        summary.append(f"- Markov matrix plots (by agent): {plot_counts['markov_matrices']}")
        summary.append("")
        
        # List aggregated plots
        aggregated_plots = [
            "deliveries_over_time_aggregated.png",
            "action_histograms_aggregated.png",
            "deliveries_vs_actions_correlation_total.png",
            "markov_matrix_all_agents.png",
            "distance_and_deliveries_aggregated.png",
            "distance_vs_deliveries_scatter.png",
            "distance_vs_deliveries_grouped.png"
        ]
        
        # Count agent-specific correlation plots
        agent_correlation_plots = len([f for f in plot_files if 'deliveries_vs_actions_correlation_agent_' in f.name])
        if agent_correlation_plots > 0:
            summary.append(f"- Agent-specific correlation plots: {agent_correlation_plots}")
        
        summary.append("")
        
        summary.append("Aggregated plots:")
        for plot_file in aggregated_plots:
            if (save_dir / plot_file).exists():
                summary.append(f"- {plot_file}")
        
        # Save summary
        with open(save_dir / 'analysis_summary.md', 'w') as f:
            f.write('\n'.join(summary))
        
        print("Summary report generated!")
    
    def run_full_analysis(self, output_dir="analysis_output"):
        """Run the complete analysis pipeline."""
        if not all([self.map_nr, self.training_id, self.checkpoint_number]):
            print("Error: Missing required parameters. Please provide map_nr, training_id, and checkpoint_number.")
            return
            
        # Create output directory with descriptive name
        analysis_name = f"figures_simulations_{self.checkpoint_number}"
        output_dir = Path(output_dir) / analysis_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Starting comprehensive simulation analysis...")
        print(f"Map: {self.map_nr}")
        print(f"Training ID: {self.training_id}")
        print(f"Checkpoint: {self.checkpoint_number}")
        print(f"Game version: {self.game_version}")
        print(f"Search path: {self.simulation_base_path}")
        print()
        
        # Find and load all simulation data
        print("1. Finding simulation directories...")
        simulation_dirs = self.find_simulation_directories()
        
        if not simulation_dirs:
            print("No simulation directories found!")
            print(f"Please verify that the path exists: {self.simulation_base_path}")
            return
        
        print("2. Loading simulation data...")
        self.load_simulation_data(simulation_dirs)
        
        if not self.simulation_data:
            print("No valid simulation data loaded!")
            print("Please verify that simulation.csv and meaningful_actions.csv files exist in the simulation directories.")
            return
        
        print("3. Generating plots...")
        
        print("   - Plot 1: Deliveries over time...")
        self.plot_deliveries_over_time(output_dir)
        
        print("   - Plot 2: Action histograms...")
        self.plot_action_histograms(output_dir)
        
        print("   - Plot 3: Deliveries vs actions correlation...")
        self.plot_deliveries_vs_actions_correlation(output_dir)
        
        print("   - Plot 4: Markov transition matrices...")
        self.plot_markov_matrices(output_dir)
        
        print("   - Plot 5: Distance and deliveries over time...")
        self.plot_distance_and_deliveries(output_dir)
        
        print("   - Plot 6: Distance vs deliveries scatter plot...")
        self.plot_distance_vs_deliveries_scatter(output_dir)
        
        print("4. Generating summary report...")
        self.generate_summary_report(output_dir)
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        print(f"Open {output_dir}/analysis_summary.md for a summary of results.")


def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze simulation data from spoiled_broth experiments')
    parser.add_argument('--map_nr', type=str, required=True,
                       help='Map number/name (e.g., "baseline_division_of_labor")')
    parser.add_argument('--training_id', type=str, required=True,
                       help='Training ID (e.g., "2025-09-13_13-27-52")')
    parser.add_argument('--checkpoint_number', type=str, required=True,
                       help='Checkpoint number (e.g., "final", "50", etc.)')
    parser.add_argument('--cluster', type=str, default='cuenca',
                       help='Base cluster (default: cuenca)')
    parser.add_argument('--game_version', type=str, default='classic', 
                       choices=['classic', 'competition'],
                       help='Game version (default: classic)')
    parser.add_argument('--num_agents', type=int, default=2,
                       help='Number of agents in the simulation (default: 2)')
    parser.add_argument('--study_name', type=str, default='default',
                       help='Study name for simulation folders (default: default)')
    parser.add_argument('--game_type', type=str, default='classic',
                       help='Game type for folder organization (default: classic)')
    
    args = parser.parse_args()

    if args.cluster.lower() == 'cuenca':
        base_cluster_dir = ""
    elif args.cluster.lower() == 'brigit':
        base_cluster_dir = "/mnt/lustre/home/samuloza/"
    elif args.cluster.lower() == 'local':
        base_cluster_dir = "C:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"

    # Updated for new folder structure: /data/.../map_{map_nr}/simulations/{study_name}/Training_{training_id}/checkpoint_{checkpoint_number}/
    if args.num_agents == 1:
        base_dir = f"{base_cluster_dir}/data/samuel_lozano/cooked/pretraining/{args.game_type}/map_{args.map_nr}"
    else:
        base_dir = f"{base_cluster_dir}/data/samuel_lozano/cooked/{args.game_type}/map_{args.map_nr}"
    training_dir = f"{base_dir}/simulations/{args.study_name}/Training_{args.training_id}/"
    simulation_dir = f"{training_dir}/checkpoint_{args.checkpoint_number}/"
    
    # Create a temporary analyzer to resolve checkpoint number if needed
    temp_analyzer = SimulationAnalyzer(
        map_nr=args.map_nr,
        training_id=args.training_id,
        checkpoint_number=args.checkpoint_number,
        game_version=args.game_version,
    )
    
    # Resolve checkpoint number (handles "final" case)
    resolved_checkpoint = temp_analyzer.resolve_checkpoint_number(training_dir)
    
    output_dir = f"{base_dir}/simulation_figures/Training_{args.training_id}"

    # Create analyzer and run analysis with resolved checkpoint number
    analyzer = SimulationAnalyzer(
        base_dir=simulation_dir,
        map_nr=args.map_nr,
        training_id=args.training_id,
        checkpoint_number=resolved_checkpoint,
        game_version=args.game_version
    )

    analyzer.run_full_analysis(output_dir)

if __name__ == "__main__":
    main()
