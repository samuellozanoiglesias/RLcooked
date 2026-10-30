#!/usr/bin/env python3
"""
Check and Clean Corrupted Simulations Script

This script scans simulation directories for corrupted simulations based on log file content.
If a simulation's experimental_simulation_*.log file contains the word "Could" (case-insensitive),
the entire simulation folder is deleted as it indicates a corrupted/failed simulation.

At the end, it provides a summary of remaining simulations per training and checkpoint.

Folder structure:
/data/samuel_lozano/cooked/classic/map_{map_nr}/simulations/Training_{training_id}/checkpoint_{checkpoint_number}/simulation_{simulation_id}/

Usage:
nohup python3 check_corrupted_simulations.py --map_nr baseline_division_of_labor --max-num-simulations 100 > check_corrupted_simulations_baseline.log 2>&1 &

nohup python3 check_corrupted_simulations.py --map_nr baseline_division_of_labor_v2 --num_agents 1 > check_corrupted_simulations_baseline.log 2>&1 &

Author: Samuel Lozano
"""

import argparse
import shutil
import sys
from pathlib import Path
import re
from collections import defaultdict
import random


class CorruptedSimulationCleaner:
    """Main class that handles corrupted simulation detection and cleanup."""
    
    def __init__(self, base_cluster_dir="", map_nr=None, dry_run=False, max_num_simulations: int = None, num_agents=2):
        """
        Initialize the corrupted simulation cleaner.
        
        Args:
            base_cluster_dir: Base cluster directory 
            map_nr: Map number/name (e.g., "baseline_division_of_labor")
            dry_run: If True, only report what would be deleted without actually deleting
        """
        self.base_cluster_dir = base_cluster_dir
        self.map_nr = map_nr
        self.dry_run = dry_run
        # Maximum number of simulations to keep per checkpoint. If None, keep all.
        self.max_num_simulations = max_num_simulations

        # Base simulation directory
        if num_agents == 1:
            self.simulations_dir = Path(f"{base_cluster_dir}/data/samuel_lozano/cooked/pretraining/classic/map_{map_nr}/simulations")
        else:
            self.simulations_dir = Path(f"{base_cluster_dir}/data/samuel_lozano/cooked/classic/map_{map_nr}/simulations")

        # Statistics tracking
        self.total_simulations_found = 0
        self.corrupted_simulations_found = 0
        self.corrupted_simulations_deleted = 0
        self.simulation_counts = defaultdict(lambda: defaultdict(int))  # training_id -> checkpoint -> count
        # Track pruned deletions (deletions due to exceeding max_num_simulations)
        self.pruned_simulations_deleted = 0
    
    def find_all_simulations(self):
        """Find all simulation directories and return them organized by training and checkpoint."""
        simulations = defaultdict(lambda: defaultdict(list))  # training_id -> checkpoint -> [sim_ids]
        
        if not self.simulations_dir.exists():
            print(f"❌ Error: Simulations directory does not exist: {self.simulations_dir}")
            return simulations
        
        print(f"🔍 Scanning for simulations in: {self.simulations_dir}")
        
        # Search for Training_* directories
        for training_dir in self.simulations_dir.glob("Training_*"):
            if not training_dir.is_dir():
                continue
                
            training_id = training_dir.name.replace("Training_", "")
            print(f"  📁 Found training: {training_id}")
            
            # Search for checkpoint_* directories within each training
            for checkpoint_dir in training_dir.glob("checkpoint_*"):
                if not checkpoint_dir.is_dir():
                    continue
                    
                checkpoint_number = checkpoint_dir.name.replace("checkpoint_", "")
                print(f"    📁 Found checkpoint: {checkpoint_number}")
                
                # Search for simulation_* directories within each checkpoint
                simulation_dirs = []
                for simulation_dir in checkpoint_dir.glob("simulation_*"):
                    if simulation_dir.is_dir():
                        simulation_id = simulation_dir.name.replace("simulation_", "")
                        simulation_dirs.append(simulation_id)
                        self.total_simulations_found += 1
                
                simulations[training_id][checkpoint_number] = simulation_dirs
                self.simulation_counts[training_id][checkpoint_number] = len(simulation_dirs)
                
                if simulation_dirs:
                    print(f"      🎮 Found {len(simulation_dirs)} simulations: {simulation_dirs[:5]}{'...' if len(simulation_dirs) > 5 else ''}")
        
        print(f"\n📊 Total simulations found: {self.total_simulations_found}")
        return simulations
    
    def check_simulation_for_corruption(self, training_id, checkpoint_number, simulation_id):
        """
        Check if a simulation is corrupted by looking for 'Could' in its log file.
        
        Returns:
            tuple: (is_corrupted: bool, log_file_path: str, reason: str)
        """
        simulation_path = self.simulations_dir / f"Training_{training_id}" / f"checkpoint_{checkpoint_number}" / f"simulation_{simulation_id}"
        
        # Look for experimental_simulation_*.log files
        log_files = list(simulation_path.glob("experimental_simulation_*.log"))
        
        if not log_files:
            return True, "No log file found", "Missing log file"
        
        if len(log_files) > 1:
            print(f"    ⚠️  Multiple log files found in {simulation_path}, using first one")
        
        log_file = log_files[0]
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Check for "Could" (case-insensitive)
                if re.search(r'\bCould\b', content, re.IGNORECASE):
                    return True, str(log_file), "Contains 'Could' in log file"
                
        except Exception as e:
            return True, str(log_file), f"Error reading log file: {e}"
        
        return False, str(log_file), "Clean simulation"
    
    def delete_simulation_directory(self, training_id, checkpoint_number, simulation_id):
        """Delete a corrupted simulation directory."""
        simulation_path = self.simulations_dir / f"Training_{training_id}" / f"checkpoint_{checkpoint_number}" / f"simulation_{simulation_id}"
        
        if self.dry_run:
            print(f"    🧪 [DRY RUN] Would delete: {simulation_path}")
            return True
        
        try:
            shutil.rmtree(simulation_path)
            print(f"    🗑️  Deleted corrupted simulation: {simulation_path}")
            return True
        except Exception as e:
            print(f"    ❌ Failed to delete {simulation_path}: {e}")
            return False
    
    def process_all_simulations(self):
        """Process all simulations, checking for corruption and deleting if necessary."""
        print("🧹 CORRUPTION CHECK AND CLEANUP")
        print("=" * 50)
        
        simulations = self.find_all_simulations()
        
        if not simulations:
            print("❌ No simulations found!")
            return
        
        # Track final counts after cleanup
        final_counts = defaultdict(lambda: defaultdict(int))
        
        for training_id in sorted(simulations.keys()):
            print(f"\n🔍 Processing Training {training_id}")
            
            for checkpoint_number in sorted(simulations[training_id].keys()):
                simulation_ids = simulations[training_id][checkpoint_number]
                print(f"  📋 Checkpoint {checkpoint_number}: {len(simulation_ids)} simulations")
                
                corrupted_in_checkpoint = 0
                remaining_in_checkpoint = 0
                
                for simulation_id in simulation_ids:
                    is_corrupted, log_file, reason = self.check_simulation_for_corruption(
                        training_id, checkpoint_number, simulation_id
                    )
                    
                    if is_corrupted:
                        self.corrupted_simulations_found += 1
                        corrupted_in_checkpoint += 1
                        print(f"    ❌ Simulation {simulation_id}: {reason}")
                        
                        if self.delete_simulation_directory(training_id, checkpoint_number, simulation_id):
                            self.corrupted_simulations_deleted += 1
                    else:
                        remaining_in_checkpoint += 1
                        print(f"    ✅ Simulation {simulation_id}: Clean")
                
                final_counts[training_id][checkpoint_number] = remaining_in_checkpoint
                
                # If a max_num_simulations cap is provided, prune randomly to meet it
                if self.max_num_simulations is not None and remaining_in_checkpoint > self.max_num_simulations:
                    # Find the remaining (clean) simulation ids in the checkpoint by re-listing dir
                    checkpoint_path = self.simulations_dir / f"Training_{training_id}" / f"checkpoint_{checkpoint_number}"
                    clean_sims = [d.name.replace('simulation_', '') for d in checkpoint_path.glob('simulation_*') if d.is_dir()]
                    # Filter out any that may have been marked corrupted earlier (we only want the ones still present)
                    clean_sims = [s for s in clean_sims if s in simulation_ids]

                    # Number to remove
                    num_to_remove = len(clean_sims) - self.max_num_simulations
                    if num_to_remove > 0:
                        print(f"    🧾 Pruning {num_to_remove} random simulations to meet cap of {self.max_num_simulations} for checkpoint {checkpoint_number}")
                        sims_to_remove = random.sample(clean_sims, num_to_remove)

                        for sim_id in sims_to_remove:
                            if self.dry_run:
                                print(f"    🧪 [DRY RUN] Would prune simulation: {sim_id}")
                                self.pruned_simulations_deleted += 1
                            else:
                                if self.delete_simulation_directory(training_id, checkpoint_number, sim_id):
                                    self.pruned_simulations_deleted += 1

                        # Update remaining count after pruning
                        remaining_in_checkpoint = len(clean_sims) - num_to_remove

                final_counts[training_id][checkpoint_number] = remaining_in_checkpoint

                print(f"    📊 Checkpoint {checkpoint_number} summary: {corrupted_in_checkpoint} corrupted, {remaining_in_checkpoint} remaining")
        
        # Update final simulation counts
        self.simulation_counts = final_counts
    
    def generate_final_report(self):
        """Generate a comprehensive final report."""
        print("\n" + "=" * 60)
        print("📊 FINAL CLEANUP REPORT")
        print("=" * 60)
        
        print(f"🎮 Total simulations found: {self.total_simulations_found}")
        print(f"❌ Corrupted simulations found: {self.corrupted_simulations_found}")
        
        if self.dry_run:
            print(f"🧪 [DRY RUN] Simulations that would be deleted: {self.corrupted_simulations_found}")
        else:
            print(f"🗑️  Corrupted simulations deleted: {self.corrupted_simulations_deleted}")
            if self.corrupted_simulations_deleted != self.corrupted_simulations_found:
                print(f"⚠️  Warning: {self.corrupted_simulations_found - self.corrupted_simulations_deleted} simulations could not be deleted")
        
        if self.max_num_simulations is not None:
            print(f"🗑️  Pruned simulations deleted to meet cap: {self.pruned_simulations_deleted}")
        
        remaining_total = sum(
            sum(checkpoints.values()) 
            for checkpoints in self.simulation_counts.values()
        )
        print(f"✅ Remaining clean simulations: {remaining_total}")
        
        print(f"\n📋 DETAILED BREAKDOWN BY TRAINING AND CHECKPOINT:")
        print("-" * 60)
        
        if not self.simulation_counts:
            print("❌ No simulations remaining after cleanup!")
            return
        
        # Create summary table
        for training_id in sorted(self.simulation_counts.keys()):
            checkpoints = self.simulation_counts[training_id]
            training_total = sum(checkpoints.values())
            
            print(f"\n🔬 Training {training_id} (Total: {training_total} simulations)")
            
            for checkpoint_number in sorted(checkpoints.keys()):
                count = checkpoints[checkpoint_number]
                print(f"  📌 Checkpoint {checkpoint_number}: {count} simulations")
        
        # Summary statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"  • Total trainings: {len(self.simulation_counts)}")
        total_checkpoints = sum(len(checkpoints) for checkpoints in self.simulation_counts.values())
        print(f"  • Total checkpoints: {total_checkpoints}")
        print(f"  • Average simulations per checkpoint: {remaining_total / total_checkpoints:.1f}" if total_checkpoints > 0 else "  • Average simulations per checkpoint: 0")
        
        # Save report to file
        self.save_report_to_file(remaining_total)
    
    def save_report_to_file(self, remaining_total):
        """Save the cleanup report to a file."""
        report_file = Path(f"corruption_cleanup_report_map_{self.map_nr}.txt")
        
        with open(report_file, 'w') as f:
            f.write(f"Corruption Cleanup Report - Map {self.map_nr}\n")
            f.write(f"Generated on: {Path(__file__).stat().st_mtime}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total simulations found: {self.total_simulations_found}\n")
            f.write(f"Corrupted simulations found: {self.corrupted_simulations_found}\n")
            
            if self.dry_run:
                f.write(f"[DRY RUN] Simulations that would be deleted: {self.corrupted_simulations_found}\n")
            else:
                f.write(f"Corrupted simulations deleted: {self.corrupted_simulations_deleted}\n")
            if self.max_num_simulations is not None:
                f.write(f"Pruned simulations deleted to meet cap ({self.max_num_simulations}): {self.pruned_simulations_deleted}\n")
            
            f.write(f"Remaining clean simulations: {remaining_total}\n\n")
            
            f.write("Detailed breakdown:\n")
            f.write("-" * 30 + "\n")
            
            for training_id in sorted(self.simulation_counts.keys()):
                checkpoints = self.simulation_counts[training_id]
                training_total = sum(checkpoints.values())
                f.write(f"\nTraining {training_id} (Total: {training_total})\n")
                
                for checkpoint_number in sorted(checkpoints.keys()):
                    count = checkpoints[checkpoint_number]
                    f.write(f"  Checkpoint {checkpoint_number}: {count} simulations\n")
        
        print(f"📝 Detailed report saved to: {report_file}")
    
    def run_cleanup(self):
        """Run the complete corruption check and cleanup process."""
        print("🧹 CORRUPTED SIMULATION CLEANUP TOOL")
        print("=" * 60)
        print(f"📍 Map: {self.map_nr}")
        print(f"📁 Base directory: {self.simulations_dir}")
        
        if self.dry_run:
            print("🧪 DRY RUN MODE: No files will actually be deleted")
        else:
            print("⚠️  LIVE MODE: Corrupted simulations will be permanently deleted!")
        
        print("=" * 60)
        
        # Process all simulations
        self.process_all_simulations()
        
        # Generate final report
        self.generate_final_report()
        
        return self.simulation_counts


def main():
    """Main function to run the corrupted simulation cleanup."""
    parser = argparse.ArgumentParser(
        description='Check and clean corrupted simulations based on log file content'
    )
    
    # Required arguments
    parser.add_argument('--map_nr', type=str, required=True,
                       help='Map number/name (e.g., "baseline_division_of_labor")')
    
    # Optional arguments
    parser.add_argument('--cluster', type=str, default='cuenca',
                       choices=['cuenca', 'brigit', 'local'],
                       help='Base cluster (default: cuenca)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Dry run mode: show what would be deleted without actually deleting')
    parser.add_argument('--max-num-simulations', type=int, default=None,
                       help='Maximum number of simulations to keep per checkpoint (randomly prune extras)')
    parser.add_argument('--num_agents', type=int, default=2,
                       help='Number of agents in the simulation (default: 2)')

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
        return 1

    # Create cleaner and run cleanup
    cleaner = CorruptedSimulationCleaner(
        base_cluster_dir=base_cluster_dir,
        map_nr=args.map_nr,
        dry_run=args.dry_run        ,
        max_num_simulations=args.max_num_simulations,
        num_agents=args.num_agents
    )

    final_counts = cleaner.run_cleanup()
    
    # Return success if we have remaining simulations
    remaining_total = sum(
        sum(checkpoints.values()) 
        for checkpoints in final_counts.values()
    )
    
    if remaining_total > 0:
        print(f"\n✅ Cleanup completed successfully! {remaining_total} clean simulations remaining.")
        return 0
    else:
        print(f"\n⚠️  Warning: No simulations remaining after cleanup!")
        return 1


if __name__ == "__main__":
    sys.exit(main())