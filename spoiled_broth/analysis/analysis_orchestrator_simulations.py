import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import time
import json
import glob
import re

class ComprehensiveAnalysisOrchestrator:
    """Main class that orchestrates both simulation and checkpoint analyses."""
    
    def __init__(self, base_cluster_dir="", map_nr=None, game_version="classic", num_agents=2,
                 training_id=None, checkpoint_number=None, output_dir=None, study_name='default', game_type='classic'):
        """
        Initialize the comprehensive analysis orchestrator.
        
        Args:
            base_cluster_dir: Base cluster directory 
            map_nr: Map number/name (e.g., "baseline_division_of_labor")
            game_version: Game version ("classic" or "competition")
            num_agents: Number of agents (1 or 2)
            training_id: Optional specific training ID for detailed analysis
            checkpoint_number: Optional specific checkpoint for detailed analysis
            output_dir: Custom output directory
            study_name: Study name for simulation folders (default: 'default')
            game_type: Game type for folder organization (default: 'classic')
        """
        self.base_cluster_dir = base_cluster_dir
        self.map_nr = map_nr
        self.game_version = game_version
        self.num_agents = num_agents
        self.training_id = training_id
        self.checkpoint_number = checkpoint_number
        self.output_dir = output_dir
        self.study_name = study_name
        self.game_type = game_type
        
        # Determine the output directory
        if output_dir is None:
            if num_agents == 1:
                self.map_base_dir = Path(f"{base_cluster_dir}/data/samuel_lozano/cooked/pretraining/{game_type}/map_{map_nr}/")
            else:
                self.map_base_dir = Path(f"{base_cluster_dir}/data/samuel_lozano/cooked/{game_type}/map_{map_nr}/")
            # Output figures should be one level up from the simulations directory
            self.output_dir = self.map_base_dir / "simulation_figures"
        else:
            self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get paths to the analysis scripts
        self.script_dir = Path(__file__).parent
        self.simulations_script = self.script_dir / "analysis_individual_simulations.py"
        self.checkpoint_script = self.script_dir / "analysis_checkpoint_comparison.py"
        
        # Initialize simulation results data storage
        self.simulation_results = []
    
    def find_all_training_directories(self):
        """Find all training directories for the specified map."""
        base_map_dir = Path(f"{self.map_base_dir}/simulations/{self.study_name}")
        
        training_dirs = []
        
        if not base_map_dir.exists():
            print(f"Error: Map directory does not exist: {base_map_dir}")
            return training_dirs
        
        print(f"Searching for training directories in: {base_map_dir}")
        
        # Search for Training_* directories
        for training_dir in base_map_dir.glob("Training_*"):
            if training_dir.is_dir():
                training_id = training_dir.name.replace("Training_", "")
                training_dirs.append(training_id)
                print(f"  Found training: {training_id}")
        
        print(f"Found {len(training_dirs)} training directories")
        return training_dirs
    
    def find_all_checkpoints_for_training(self, training_id):
        """Find all checkpoint directories for a specific training."""
        training_dir = Path(f"{self.map_base_dir}/simulations/{self.study_name}/Training_{training_id}")
        
        checkpoints = []
        
        if not training_dir.exists():
            print(f"Error: Training directory does not exist: {training_dir}")
            return checkpoints
        
        print(f"  Searching for checkpoints in: {training_dir}")
        
        # Search for checkpoint_* directories
        for checkpoint_dir in training_dir.glob("checkpoint_*"):
            if checkpoint_dir.is_dir():
                checkpoint_number = checkpoint_dir.name.replace("checkpoint_", "")
                checkpoints.append(checkpoint_number)
                print(f"    Found checkpoint: {checkpoint_number}")
        
        # Sort checkpoints numerically if they are numbers, otherwise alphabetically
        try:
            checkpoints.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
        except:
            checkpoints.sort()
        
        print(f"  Found {len(checkpoints)} checkpoints for Training {training_id}: {checkpoints}")
        return checkpoints
    
    def extract_simulation_metrics(self, simulation_path, training_id, checkpoint_number):
        """
        Extract metrics from a single simulation directory.
        
        Args:
            simulation_path: Path to simulation directory
            training_id: Training ID
            checkpoint_number: Checkpoint number
            
        Returns:
            Dictionary with simulation metrics or None if extraction fails
        """
        try:
            sim_csv_path = simulation_path / "simulation.csv"
            actions_csv_path = simulation_path / "meaningful_actions.csv"
            config_path = simulation_path / "config.txt"
            
            if not (sim_csv_path.exists() and actions_csv_path.exists()):
                print(f"  Missing required files in {simulation_path}")
                return None
            
            # Load simulation data
            sim_data = pd.read_csv(sim_csv_path)
            actions_data = pd.read_csv(actions_csv_path)
            
            # Extract simulation ID from path
            simulation_id = simulation_path.name.replace('simulation_', '')
            
            # Read initialization period from config
            initialization_period = 0.0
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_contents = f.read()
                    init_match = re.search(r"AGENT_INITIALIZATION_PERIOD:\s*([0-9.]+)", config_contents)
                    if init_match:
                        initialization_period = float(init_match.group(1))
                except Exception as e:
                    print(f"    Warning: Could not read config.txt: {e}")
            
            # Convert frames to adjusted seconds
            if 'second' in sim_data.columns:
                sim_data['adjusted_second'] = (sim_data['second'] - initialization_period).clip(lower=0)
            else:
                # Assume 30 FPS if no seconds column
                sim_data['adjusted_second'] = (sim_data['frame'] / 30.0 - initialization_period).clip(lower=0)
            
            # Compute deliveries (score increases)
            deliveries_data = []
            for agent_id in sim_data['agent_id'].unique():
                agent_data = sim_data[sim_data['agent_id'] == agent_id].sort_values('adjusted_second')
                agent_data['score_diff'] = agent_data['score'].diff().fillna(0)
                agent_data['deliveries'] = (agent_data['score_diff'] > 0).astype(int)
                agent_data['cumulative_deliveries'] = agent_data['deliveries'].cumsum()
                deliveries_data.append(agent_data)
            
            all_deliveries = pd.concat(deliveries_data, ignore_index=True)
            
            # Compute agent distances (for multi-agent scenarios)
            distances = []
            for time_point in sim_data['adjusted_second'].unique():
                time_data = sim_data[sim_data['adjusted_second'] == time_point]
                if len(time_data) >= 2:
                    agents = time_data.groupby('agent_id')[['x', 'y']].first()
                    if len(agents) >= 2:
                        agent_positions = agents.values
                        if len(agent_positions) >= 2:
                            dist = np.sqrt((agent_positions[0][0] - agent_positions[1][0])**2 + 
                                         (agent_positions[0][1] - agent_positions[1][1])**2)
                            distances.append(dist)
            
            # Calculate metrics
            metrics = {
                'simulation_id': simulation_id,
                'training_id': training_id,
                'checkpoint_number': checkpoint_number,
                'map_name': self.map_nr,
                'game_version': self.game_version,
                'num_agents': len(sim_data['agent_id'].unique()),
                'simulation_duration_seconds': sim_data['adjusted_second'].max(),
                'initialization_period': initialization_period,
                
                # Delivery metrics
                'total_deliveries': all_deliveries['deliveries'].sum(),
                'final_score': sim_data['score'].max(),
                'deliveries_per_agent': all_deliveries.groupby('agent_id')['deliveries'].sum().tolist(),
                'deliveries_per_second': all_deliveries['deliveries'].sum() / max(sim_data['adjusted_second'].max(), 1),
                
                # Action metrics
                'total_actions': len(actions_data),
                'actions_per_agent': actions_data.groupby('agent_id').size().tolist() if 'agent_id' in actions_data.columns else [len(actions_data)],
                'unique_action_types': len(actions_data['action_category_name'].unique()) if 'action_category_name' in actions_data.columns else 0,
                'action_types_distribution': actions_data['action_category_name'].value_counts().to_dict() if 'action_category_name' in actions_data.columns else {},
                
                # Distance metrics (for multi-agent)
                'avg_agent_distance': np.mean(distances) if distances else 0,
                'max_agent_distance': np.max(distances) if distances else 0,
                'min_agent_distance': np.min(distances) if distances else 0,
                'std_agent_distance': np.std(distances) if distances else 0,
                
                # Efficiency metrics
                'actions_per_delivery': len(actions_data) / max(all_deliveries['deliveries'].sum(), 1),
                'score_per_action': sim_data['score'].max() / max(len(actions_data), 1),
                'time_to_first_delivery': all_deliveries[all_deliveries['deliveries'] > 0]['adjusted_second'].min() if any(all_deliveries['deliveries'] > 0) else None,
            }
            
            # Add per-agent metrics
            for agent_id in sim_data['agent_id'].unique():
                agent_sim_data = sim_data[sim_data['agent_id'] == agent_id]
                agent_deliveries = all_deliveries[all_deliveries['agent_id'] == agent_id]
                agent_actions = actions_data[actions_data['agent_id'] == agent_id] if 'agent_id' in actions_data.columns else pd.DataFrame()
                
                metrics.update({
                    f'agent_{agent_id}_final_score': agent_sim_data['score'].max(),
                    f'agent_{agent_id}_deliveries': agent_deliveries['deliveries'].sum(),
                    f'agent_{agent_id}_actions': len(agent_actions),
                    f'agent_{agent_id}_avg_x': agent_sim_data['x'].mean(),
                    f'agent_{agent_id}_avg_y': agent_sim_data['y'].mean(),
                    f'agent_{agent_id}_distance_traveled': self.calculate_distance_traveled(agent_sim_data),
                })
            
            return metrics
            
        except Exception as e:
            print(f"  Error extracting metrics from {simulation_path}: {e}")
            return None
    
    def calculate_distance_traveled(self, agent_data):
        """Calculate total distance traveled by an agent."""
        if len(agent_data) <= 1:
            return 0.0
        
        agent_sorted = agent_data.sort_values('adjusted_second')
        x_diff = agent_sorted['x'].diff().fillna(0)
        y_diff = agent_sorted['y'].diff().fillna(0)
        distances = np.sqrt(x_diff**2 + y_diff**2)
        return distances.sum()
    
    def collect_simulation_results(self, training_id, checkpoint_number):
        """
        Collect simulation results from all simulations for a specific training and checkpoint.
        
        Args:
            training_id: Training ID
            checkpoint_number: Checkpoint number (resolved)
        """
        checkpoint_dir = Path(f"{self.map_base_dir}/simulations/{self.study_name}/Training_{training_id}/checkpoint_{checkpoint_number}")
        
        if not checkpoint_dir.exists():
            print(f"  Checkpoint directory does not exist: {checkpoint_dir}")
            return
        
        print(f"  Collecting simulation results from: {checkpoint_dir}")
        
        # Find all simulation directories
        simulation_dirs = list(checkpoint_dir.glob("simulation_*"))
        
        if not simulation_dirs:
            print(f"  No simulation directories found in {checkpoint_dir}")
            return
        
        print(f"  Found {len(simulation_dirs)} simulation directories")
        
        # Extract metrics from each simulation
        for sim_dir in simulation_dirs:
            if sim_dir.is_dir():
                print(f"    Processing {sim_dir.name}")
                metrics = self.extract_simulation_metrics(sim_dir, training_id, checkpoint_number)
                if metrics:
                    self.simulation_results.append(metrics)
                    print(f"      Extracted metrics: {metrics['total_deliveries']} deliveries, {metrics['total_actions']} actions")
    
    def create_simulation_results_csv(self):
        """
        Create a comprehensive CSV file with all simulation results and mean values.
        """
        if not self.simulation_results:
            print("No simulation results to save.")
            return
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.simulation_results)
        
        # Flatten complex columns (lists and dicts) for CSV compatibility
        for col in results_df.columns:
            if results_df[col].dtype == 'object':
                # Handle lists by converting to string representation
                if any(isinstance(x, list) for x in results_df[col] if x is not None):
                    results_df[col] = results_df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
                # Handle dicts by converting to JSON string
                elif any(isinstance(x, dict) for x in results_df[col] if x is not None):
                    results_df[col] = results_df[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
        
        # Calculate mean values for numerical columns
        numerical_columns = results_df.select_dtypes(include=[np.number]).columns
        mean_values = results_df[numerical_columns].mean()
        
        # Calculate additional aggregate statistics
        aggregate_stats = {
            'simulation_id': 'MEAN_VALUES',
            'training_id': 'AGGREGATE',
            'checkpoint_number': 'ALL',
            'map_name': self.map_nr,
            'game_version': self.game_version,
        }
        
        # Add mean values
        for col in numerical_columns:
            if col not in ['simulation_id', 'training_id', 'checkpoint_number']:  # Skip non-numeric identifiers
                aggregate_stats[col] = mean_values[col]
        
        # Add additional aggregate statistics
        if len(results_df) > 0:
            aggregate_stats.update({
                'total_simulations_analyzed': len(results_df),
                'std_total_deliveries': results_df['total_deliveries'].std() if 'total_deliveries' in results_df.columns else 0,
                'std_final_score': results_df['final_score'].std() if 'final_score' in results_df.columns else 0,
                'std_total_actions': results_df['total_actions'].std() if 'total_actions' in results_df.columns else 0,
                'min_deliveries': results_df['total_deliveries'].min() if 'total_deliveries' in results_df.columns else 0,
                'max_deliveries': results_df['total_deliveries'].max() if 'total_deliveries' in results_df.columns else 0,
                'min_final_score': results_df['final_score'].min() if 'final_score' in results_df.columns else 0,
                'max_final_score': results_df['final_score'].max() if 'final_score' in results_df.columns else 0,
                'success_rate': (results_df['total_deliveries'] > 0).mean() if 'total_deliveries' in results_df.columns else 0,
            })
        
        # Add aggregate row to the DataFrame
        aggregate_df = pd.DataFrame([aggregate_stats])
        
        # Ensure all columns exist in both DataFrames
        all_columns = set(results_df.columns) | set(aggregate_df.columns)
        for col in all_columns:
            if col not in results_df.columns:
                results_df[col] = None
            if col not in aggregate_df.columns:
                aggregate_df[col] = None
        
        # Reorder columns to match
        results_df = results_df.reindex(columns=sorted(all_columns))
        aggregate_df = aggregate_df.reindex(columns=sorted(all_columns))
        
        # Combine individual results with aggregate statistics
        final_df = pd.concat([results_df, aggregate_df], ignore_index=True)
        
        # Save to CSV
        csv_path = self.output_dir / 'simulation_results.csv'
        final_df.to_csv(csv_path, index=False)
        
        print(f"\n📊 Simulation results CSV created: {csv_path}")
        print(f"   - Total simulations: {len(results_df)}")
        print(f"   - Mean deliveries per simulation: {mean_values.get('total_deliveries', 0):.2f}")
        print(f"   - Mean final score per simulation: {mean_values.get('final_score', 0):.2f}")
        print(f"   - Mean actions per simulation: {mean_values.get('total_actions', 0):.2f}")
        
        # Create a summary report
        self.create_simulation_summary_report(results_df, mean_values, aggregate_stats)
        
        return csv_path
    
    def create_simulation_summary_report(self, results_df, mean_values, aggregate_stats):
        """
        Create a markdown summary report of the simulation results.
        """
        report_path = self.output_dir / 'simulation_results_summary.md'
        
        summary = []
        summary.append("# Simulation Results Summary")
        summary.append(f"Generated on: {pd.Timestamp.now()}\n")
        
        summary.append("## Dataset Overview")
        summary.append(f"- **Map**: {self.map_nr}")
        summary.append(f"- **Game Version**: {self.game_version}")
        summary.append(f"- **Total Simulations Analyzed**: {len(results_df)}")
        if self.training_id:
            summary.append(f"- **Training ID**: {self.training_id}")
        if self.checkpoint_number:
            summary.append(f"- **Checkpoint**: {self.checkpoint_number}")
        summary.append("")
        
        summary.append("## Key Performance Metrics")
        summary.append("### Deliveries")
        summary.append(f"- **Mean Deliveries per Simulation**: {mean_values.get('total_deliveries', 0):.2f}")
        summary.append(f"- **Standard Deviation**: {aggregate_stats.get('std_total_deliveries', 0):.2f}")
        summary.append(f"- **Min Deliveries**: {aggregate_stats.get('min_deliveries', 0):.0f}")
        summary.append(f"- **Max Deliveries**: {aggregate_stats.get('max_deliveries', 0):.0f}")
        summary.append(f"- **Success Rate**: {aggregate_stats.get('success_rate', 0):.1%} (simulations with >0 deliveries)")
        summary.append("")
        
        summary.append("### Scores")
        summary.append(f"- **Mean Final Score**: {mean_values.get('final_score', 0):.2f}")
        summary.append(f"- **Standard Deviation**: {aggregate_stats.get('std_final_score', 0):.2f}")
        summary.append(f"- **Min Score**: {aggregate_stats.get('min_final_score', 0):.0f}")
        summary.append(f"- **Max Score**: {aggregate_stats.get('max_final_score', 0):.0f}")
        summary.append("")
        
        summary.append("### Actions")
        summary.append(f"- **Mean Actions per Simulation**: {mean_values.get('total_actions', 0):.2f}")
        summary.append(f"- **Standard Deviation**: {aggregate_stats.get('std_total_actions', 0):.2f}")
        summary.append(f"- **Mean Actions per Delivery**: {mean_values.get('actions_per_delivery', 0):.2f}")
        summary.append("")
        
        summary.append("### Efficiency Metrics")
        summary.append(f"- **Mean Deliveries per Second**: {mean_values.get('deliveries_per_second', 0):.4f}")
        summary.append(f"- **Mean Score per Action**: {mean_values.get('score_per_action', 0):.2f}")
        if mean_values.get('time_to_first_delivery'):
            summary.append(f"- **Mean Time to First Delivery**: {mean_values.get('time_to_first_delivery', 0):.2f} seconds")
        summary.append("")
        
        if mean_values.get('avg_agent_distance', 0) > 0:
            summary.append("### Agent Coordination (Multi-Agent)")
            summary.append(f"- **Mean Agent Distance**: {mean_values.get('avg_agent_distance', 0):.2f}")
            summary.append(f"- **Max Agent Distance**: {mean_values.get('max_agent_distance', 0):.2f}")
            summary.append(f"- **Min Agent Distance**: {mean_values.get('min_agent_distance', 0):.2f}")
            summary.append("")
        
        summary.append("## Files Generated")
        summary.append("- `simulation_results.csv` - Complete dataset with individual simulation metrics")
        summary.append("- `simulation_results_summary.md` - This summary report")
        summary.append("")
        
        summary.append("## CSV Structure")
        summary.append("The CSV file contains the following key columns:")
        summary.append("- **simulation_id**: Unique identifier for each simulation")
        summary.append("- **training_id, checkpoint_number, map_name**: Experiment identifiers")
        summary.append("- **total_deliveries, final_score**: Performance metrics")
        summary.append("- **total_actions, unique_action_types**: Behavioral metrics")
        summary.append("- **avg_agent_distance**: Coordination metrics (multi-agent)")
        summary.append("- **agent_X_***: Per-agent detailed metrics")
        summary.append("- **MEAN_VALUES row**: Aggregate statistics across all simulations")
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(summary))
        
        print(f"📋 Summary report created: {report_path}")
    
    def resolve_checkpoint_number(self, training_id, checkpoint_number):
        """
        Resolve checkpoint number when it's "final" by reading training_stats.csv.
        
        Args:
            training_id: Training ID 
            checkpoint_number: Checkpoint number (e.g., "final", "50", etc.)
            
        Returns:
            Resolved checkpoint number as string
        """
        if checkpoint_number != "final":
            return checkpoint_number
            
        training_dir = Path(f"{self.map_base_dir}/Training_{training_id}")
        training_stats_path = training_dir / "training_stats.csv"
        
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
            
            print(f"Resolved checkpoint_number 'final' to '{resolved_checkpoint}' for Training {training_id} based on training_stats.csv")
            return resolved_checkpoint
            
        except Exception as e:
            print(f"Error reading training_stats.csv: {e}")
            print("Using 'final' as checkpoint_number")
            return "final"
    
    def run_single_simulation_analysis(self, training_id, checkpoint_number="final"):
        """Run simulation analysis for a single training_id and checkpoint."""
        
        # Resolve "final" checkpoint if needed
        resolved_checkpoint = self.resolve_checkpoint_number(training_id, checkpoint_number)
        
        if checkpoint_number == "final" and resolved_checkpoint != "final":
            print(f"✅ Resolved 'final' checkpoint to '{resolved_checkpoint}' for Training {training_id}")
        
        print(f"Running analysis_simulations.py for Training {training_id}, Checkpoint {resolved_checkpoint}")
        
        # Collect simulation results for CSV creation
        print(f"Collecting simulation results for CSV...")
        self.collect_simulation_results(training_id, resolved_checkpoint)
        
        # Build command for analysis_simulations.py
        cmd = [
            "python3", str(self.simulations_script),
            "--map_nr", self.map_nr,
            "--training_id", training_id,
            "--checkpoint_number", resolved_checkpoint,
            "--game_version", self.game_version,
            "--num_agents", str(self.num_agents),
            "--study_name", self.study_name,
            "--game_type", self.game_type,
        ]
        
        # Determine cluster argument
        if self.base_cluster_dir == "":
            cmd.extend(["--cluster", "cuenca"])
        elif "brigit" in self.base_cluster_dir.lower():
            cmd.extend(["--cluster", "brigit"])
        elif "onedrive" in self.base_cluster_dir.lower():
            cmd.extend(["--cluster", "local"])
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Run the simulations analysis
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                checkpoint_display = f"{resolved_checkpoint}" + (f" (resolved from '{checkpoint_number}')" if checkpoint_number == "final" and resolved_checkpoint != "final" else "")
                print(f"✅ Simulation analysis for Training {training_id}, Checkpoint {checkpoint_display} completed successfully!")
                print("Output:")
                print(result.stdout)
                
                # Create CSV file with simulation results
                if self.simulation_results:
                    print("Creating simulation results CSV...")
                    self.create_simulation_results_csv()
                
                return True
            else:
                checkpoint_display = f"{resolved_checkpoint}" + (f" (resolved from '{checkpoint_number}')" if checkpoint_number == "final" and resolved_checkpoint != "final" else "")
                print(f"❌ Simulation analysis for Training {training_id}, Checkpoint {checkpoint_display} failed!")
                print("Error output:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            checkpoint_display = f"{resolved_checkpoint}" + (f" (resolved from '{checkpoint_number}')" if checkpoint_number == "final" and resolved_checkpoint != "final" else "")
            print(f"❌ Simulation analysis for Training {training_id}, Checkpoint {checkpoint_display} timed out (>1 hour)")
            return False
        except Exception as e:
            checkpoint_display = f"{resolved_checkpoint}" + (f" (resolved from '{checkpoint_number}')" if checkpoint_number == "final" and resolved_checkpoint != "final" else "")
            print(f"❌ Error running simulation analysis for Training {training_id}, Checkpoint {checkpoint_display}: {e}")
            return False
    
    def run_simulations_analysis(self):
        """Run the detailed simulations analysis."""
        print("PHASE 1: DETAILED SIMULATION ANALYSIS")
        print("-" * 40)
        
        if self.training_id:
            # Specific training_id provided
            if self.checkpoint_number:
                # Both training_id and checkpoint_number provided - run only for that specific combination
                print(f"Running simulation analysis for specific Training {self.training_id}, Checkpoint {self.checkpoint_number}")
                return self.run_single_simulation_analysis(self.training_id, self.checkpoint_number)
            else:
                # Only training_id provided - run for all checkpoints of that training
                print(f"Running simulation analysis for Training {self.training_id}, all checkpoints")
                checkpoints = self.find_all_checkpoints_for_training(self.training_id)
                
                if not checkpoints:
                    print(f"❌ No checkpoints found for Training {self.training_id}!")
                    return False
                
                print(f"Found {len(checkpoints)} checkpoints for Training {self.training_id}: {checkpoints}")
                
                success_count = 0
                for checkpoint in checkpoints:
                    print(f"\n--- Processing Training {self.training_id}, Checkpoint {checkpoint} ---")
                    if self.run_single_simulation_analysis(self.training_id, checkpoint):
                        success_count += 1
                    else:
                        print(f"⚠️  Training {self.training_id}, Checkpoint {checkpoint} failed, continuing with next...")
                
                print(f"\n📊 Training {self.training_id} Analysis Summary: {success_count}/{len(checkpoints)} checkpoints completed successfully")
                
                # Create comprehensive CSV for all checkpoints of this training
                if self.simulation_results:
                    print("Creating comprehensive simulation results CSV for all checkpoints...")
                    self.create_simulation_results_csv()
                
                return success_count >= len(checkpoints) // 2
        else:
            # No specific training_id - run for all training_ids and all their checkpoints
            training_ids = self.find_all_training_directories()
            
            if not training_ids:
                print("❌ No training directories found!")
                return False
            
            print(f"Running simulation analysis for all {len(training_ids)} training directories and all their checkpoints")
            
            total_combinations = 0
            success_count = 0
            
            for training_id in training_ids:
                print(f"\n=== Processing Training {training_id} ===")
                
                # Get all checkpoints for this training
                checkpoints = self.find_all_checkpoints_for_training(training_id)
                
                if not checkpoints:
                    print(f"⚠️  No checkpoints found for Training {training_id}, skipping...")
                    continue
                
                print(f"Found {len(checkpoints)} checkpoints for Training {training_id}: {checkpoints}")
                total_combinations += len(checkpoints)
                
                # Process each checkpoint
                for checkpoint in checkpoints:
                    print(f"\n--- Processing Training {training_id}, Checkpoint {checkpoint} ---")
                    if self.run_single_simulation_analysis(training_id, checkpoint):
                        success_count += 1
                    else:
                        print(f"⚠️  Training {training_id}, Checkpoint {checkpoint} failed, continuing with next...")
            
            print(f"\n📊 Overall Analysis Summary: {success_count}/{total_combinations} training-checkpoint combinations completed successfully")
            
            # Create comprehensive CSV for all trainings and checkpoints
            if self.simulation_results:
                print("Creating comprehensive simulation results CSV for all trainings and checkpoints...")
                self.create_simulation_results_csv()
            
            # Consider it successful if at least half completed successfully
            return success_count >= total_combinations // 2 if total_combinations > 0 else False
    
    def run_checkpoint_analysis(self):
        """Run the checkpoint comparison analysis."""
        print("\nPHASE 2: CHECKPOINT COMPARISON ANALYSIS")
        print("-" * 40)
        print(f"Running analysis_checkpoint_comparison.py for map {self.map_nr}")
        
        # Build command for analysis_checkpoint_comparison.py
        cmd = [
            "python3", str(self.checkpoint_script),
            "--map_nr", self.map_nr,
            "--game_version", self.game_version,
            "--output_dir", str(self.output_dir),
            "--num_agents", str(self.num_agents),
            "--study_name", self.study_name,
            "--game_type", self.game_type,
        ]
        
        # Determine cluster argument
        if self.base_cluster_dir == "":
            cmd.extend(["--cluster", "cuenca"])
        elif "brigit" in self.base_cluster_dir.lower():
            cmd.extend(["--cluster", "brigit"])
        elif "onedrive" in self.base_cluster_dir.lower():
            cmd.extend(["--cluster", "local"])
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Run the checkpoint analysis
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                print("✅ Checkpoint comparison analysis completed successfully!")
                print("Output:")
                print(result.stdout)
                return True
            else:
                print("❌ Checkpoint comparison analysis failed!")
                print("Error output:")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Checkpoint comparison analysis timed out (>1 hour)")
            return False
        except Exception as e:
            print(f"❌ Error running checkpoint comparison analysis: {e}")
            return False
    
    def generate_comprehensive_summary(self, simulations_success, checkpoint_success):
        """Generate a comprehensive summary combining both analyses."""
        print("\nPHASE 3: COMPREHENSIVE SUMMARY")
        print("-" * 40)
        
        summary = []
        summary.append("# Comprehensive Analysis Summary Report")
        summary.append("## Spoiled Broth Experiments\n")
        summary.append(f"Generated on: {pd.Timestamp.now()}\n")
        
        # Analysis parameters
        summary.append("## Analysis Configuration")
        summary.append(f"- **Map**: {self.map_nr}")
        summary.append(f"- **Game version**: {self.game_version}")
        summary.append(f"- **Simulations directory**: {self.map_base_dir}/simulations/")
        summary.append(f"- **Figures output directory**: {self.output_dir}")
        if self.training_id and self.checkpoint_number:
            summary.append(f"- **Detailed analysis**: Training {self.training_id}, Checkpoint {self.checkpoint_number}")
        summary.append("")
        
        # Analysis phases completed
        summary.append("## Analysis Phases Completed")
        
        if simulations_success:
            summary.append("### ✅ Phase 1: Detailed Simulation Analysis")
            if self.training_id:
                if self.checkpoint_number:
                    summary.append(f"- Analyzed individual simulations for Training {self.training_id}, Checkpoint {self.checkpoint_number}")
                else:
                    summary.append(f"- Analyzed individual simulations for Training {self.training_id}, all checkpoints")
            else:
                summary.append("- Analyzed individual simulations for all training directories and all checkpoints")
                summary.append("- Processed all available training_ids with all their checkpoint directories")
            summary.append("- Generated detailed behavioral plots (deliveries, actions, distances)")
            summary.append("- Created individual and aggregated visualizations")
            summary.append("- **Created comprehensive simulation_results.csv with all metrics and mean values**")
            summary.append("- Results saved in subdirectories of simulation_figures/")
        else:
            summary.append("### ❌ Phase 1: Detailed Simulation Analysis")
            summary.append("- Failed to complete (check logs for details)")
        
        if checkpoint_success:
            summary.append("\n### ✅ Phase 2: Checkpoint Comparison Analysis")
            summary.append("- Analyzed delivery performance across all training IDs and checkpoints")
            summary.append("- Generated comparative plots showing training progression")
            summary.append("- Created statistical summaries of checkpoint performance")
            summary.append("- Results saved directly in simulation_figures/")
        else:
            summary.append("\n### ❌ Phase 2: Checkpoint Comparison Analysis")
            summary.append("- Failed to complete (check logs for details)")
        
        summary.append("")
        
        # Generated files overview
        summary.append("## Generated Files Structure")
        summary.append(f"All analysis outputs (figures and reports) are organized in: `{self.output_dir}`")
        summary.append(f"This is separate from the simulation data located in: `{self.map_base_dir}/simulations/`")
        summary.append("")
        
        if checkpoint_success:
            summary.append("### Checkpoint Comparison Analysis")
            summary.append("- `deliveries_vs_checkpoints_{}.png` - Multi-panel checkpoint comparison".format(self.map_nr))
            summary.append("- `deliveries_vs_checkpoints_focused_{}.png` - Focused line plot".format(self.map_nr))
            summary.append("- `checkpoint_analysis_summary.md` - Detailed checkpoint analysis report")
        
        if simulations_success:
            summary.append("\n### Detailed Simulation Analysis")
            summary.append("- **`simulation_results.csv`** - Comprehensive dataset with all simulation metrics and aggregate statistics")
            summary.append("- **`simulation_results_summary.md`** - Summary report with key performance indicators")
            if self.training_id and self.checkpoint_number:
                summary.append(f"- `detailed_simulations_Training_{self.training_id}_checkpoint_{self.checkpoint_number}/` - Subdirectory containing:")
            elif self.training_id:
                summary.append(f"- `detailed_simulations_Training_{self.training_id}_all_checkpoints/` - Subdirectories containing:")
            else:
                summary.append("- `detailed_simulations_all_trainings_all_checkpoints/` - Subdirectories containing:")
            summary.append("  - Individual simulation plots (deliveries, actions, distances)")
            summary.append("  - Aggregated simulation visualizations")
            summary.append("  - `detailed_simulation_analysis_summary.md` - Detailed simulation report")
        
        summary.append("\n### This Report")
        summary.append("- `comprehensive_analysis_summary.md` - This comprehensive overview")
        
        # Usage recommendations
        summary.append("\n## Usage Recommendations")
        summary.append("### For Research Analysis:")
        
        if checkpoint_success:
            summary.append("1. **Start with checkpoint comparison plots** to identify best-performing training setups")
            summary.append("2. **Use focused line plots** to understand training progression trends")
            summary.append("3. **Examine heatmaps** to compare training IDs across different checkpoints")
        
        if simulations_success:
            summary.append("4. **Analyze simulation_results.csv** for quantitative performance metrics and statistical analysis")
            summary.append("5. **Use the MEAN_VALUES row** in the CSV for quick overview of average performance")
            summary.append("6. **Dive into detailed simulation analysis plots** for behavioral insights")
            summary.append("7. **Compare individual vs aggregated plots** to understand variability")
            summary.append("8. **Import CSV into statistical software** (R, Python, etc.) for advanced analysis")
        
        summary.append("\n### For Future Experiments:")
        summary.append("- **Monitor early checkpoint performance** for early stopping decisions")
        summary.append("- **Use this analysis framework** for systematic experiment evaluation")
        summary.append("- **Run comprehensive analysis regularly** to track training progress")
        
        # Analysis execution details
        summary.append("\n## Execution Details")
        summary.append("This comprehensive analysis was executed by calling:")
        if simulations_success:
            summary.append("1. `analysis_simulations.py` - for detailed simulation analysis")
            summary.append("   - Extracts metrics from each simulation.csv and meaningful_actions.csv")
            summary.append("   - Computes aggregate statistics and creates simulation_results.csv")
            summary.append("   - Generates individual and comparative visualization plots")
        if checkpoint_success:
            summary.append("2. `analysis_checkpoint_comparison.py` - for checkpoint comparison")
        summary.append("\nAll scripts include automatic CSV generation with comprehensive metrics and mean values.")
        
        # Save comprehensive summary
        summary_file = self.output_dir / 'comprehensive_analysis_summary.md'
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary))
        
        print(f"Comprehensive summary report generated: {summary_file}")
    
    def run_comprehensive_analysis(self):
        """Run the complete comprehensive analysis by orchestrating both sub-analyses."""
        print("="*60)
        print("COMPREHENSIVE ANALYSIS - SPOILED BROTH EXPERIMENTS")
        print("="*60)
        print(f"Map: {self.map_nr}")
        print(f"Game version: {self.game_version}")
        simulations_base_dir = f"{self.map_base_dir}/simulations/"
        print(f"Simulations directory: {simulations_base_dir}")
        print(f"Figures output directory: {self.output_dir}")
        if self.training_id and self.checkpoint_number:
            print(f"Detailed analysis for: Training {self.training_id}, Checkpoint {self.checkpoint_number}")
        print("="*60)
        print()
        
        # Track start time
        start_time = time.time()
        
        if self.training_id:
            # Specific training_id provided - only run simulation analysis
            if self.checkpoint_number:
                print(f"🎯 Specific training_id and checkpoint provided - running detailed simulation analysis for Training {self.training_id}, Checkpoint {self.checkpoint_number}")
            else:
                print(f"🎯 Specific training_id provided - running detailed simulation analysis for Training {self.training_id}, all checkpoints")
            simulations_success = self.run_simulations_analysis()
            checkpoint_success = False  # Skip checkpoint analysis
        else:
            # No specific training_id - run simulations for all trainings and checkpoints, then checkpoint comparison
            print("🔄 No specific training_id provided - running simulation analysis for all training_ids and all checkpoints, then checkpoint comparison")
            
            # Phase 1: Detailed Simulation Analysis for all training_ids
            simulations_success = self.run_simulations_analysis()
            
            # Phase 2: Checkpoint Comparison Analysis
            checkpoint_success = self.run_checkpoint_analysis()
        
        # Phase 3: Generate comprehensive summary
        self.generate_comprehensive_summary(simulations_success, checkpoint_success)
        
        # Calculate total execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS COMPLETED!")
        print("="*60)
        print(f"Total execution time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")
        print(f"All analysis results (figures and reports) saved to: {self.output_dir}")
        print(f"Simulation data remains in: {self.map_base_dir}/simulations/")
        print(f"Open {self.output_dir}/comprehensive_analysis_summary.md for complete results overview.")
        if simulations_success:
            print(f"📊 Key results CSV: {self.output_dir}/simulation_results.csv (includes individual metrics + means)")
        print()
        
        # Summary of what was completed
        if self.training_id:
            # Specific training_id mode
            if simulations_success:
                if self.checkpoint_number:
                    print(f"✅ Detailed simulation analysis for Training {self.training_id}, Checkpoint {self.checkpoint_number} completed successfully!")
                    print(f"    → CSV file with simulation results available at: {self.output_dir}/simulation_results.csv")
                else:
                    print(f"✅ Detailed simulation analysis for Training {self.training_id}, all checkpoints completed successfully!")
                    print(f"    → Comprehensive CSV file with all results available at: {self.output_dir}/simulation_results.csv")
            else:
                if self.checkpoint_number:
                    print(f"❌ Detailed simulation analysis for Training {self.training_id}, Checkpoint {self.checkpoint_number} failed. Check error messages above.")
                else:
                    print(f"❌ Detailed simulation analysis for Training {self.training_id}, all checkpoints failed. Check error messages above.")
        else:
            # All training_ids mode
            if simulations_success and checkpoint_success:
                print("✅ All analyses completed successfully!")
                print(f"    → Comprehensive simulation results CSV: {self.output_dir}/simulation_results.csv")
            elif checkpoint_success:
                print("⚠️  Checkpoint analysis completed, but some simulation analyses failed.")
            elif simulations_success:
                print("⚠️  Simulation analyses completed, but checkpoint analysis failed.")
                print(f"    → Simulation results CSV still available: {self.output_dir}/simulation_results.csv")
            else:
                print("❌ Both simulation and checkpoint analyses failed. Check error messages above.")
        
        return simulations_success, checkpoint_success
