#!/usr/bin/env python3
"""
Script to copy experiment data from local directories.
Copies CSV files from /data/samuel_lozano/cooked/classic/v3.1/map_{map_identifier}/simulations/Training_{training_id}/checkpoint_{checkpoint_number}/simulation_{simulation_id}/
while maintaining the folder structure.

Usage:
    python3 copy_experiment_data.py <map_identifier> <experiment_name> [--source-path /path/to/data] [--dry-run]
    
Examples:
    python3 copy_experiment_data.py baseline_division_of_labor experiment_1

    python3 copy_experiment_data.py map experiment_2 --source-path /custom/path/to/data
"""

import os
import shutil
import argparse
from pathlib import Path
import logging
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_experiment_folder(base_path="/data/samuel_lozano/cooked/classic/v3.1", experiment_name="experiment"):
    """Create the experiment folder if it doesn't exist."""
    experiment_path = Path(base_path) / experiment_name
    experiment_path.mkdir(exist_ok=True)
    return experiment_path

def extract_ids_from_path(path, map_nr):
    """Extract training_id, checkpoint_number, and simulation_id from path."""
    path_parts = Path(path).parts
    
    # Extract IDs
    training_id = None
    checkpoint_number = None
    simulation_id = None
    
    for part in path_parts:
        if part.startswith('Training_'):
            training_id = part.replace('Training_', '')
        elif part.startswith('checkpoint_'):
            checkpoint_number = part.replace('checkpoint_', '')
        elif part.startswith('simulation_'):
            simulation_id = part.replace('simulation_', '')
    
    return training_id, checkpoint_number, simulation_id

def check_csv_files_exist(source_path):
    """Check if the required CSV files exist in the source directory."""
    csv_files = [
        'actions.csv',
        'counters.csv',
        'items.csv',
        'actions_ai_rl_1.csv',
        'ai_rl_1_positions.csv',
        'actions_ai_rl_2.csv',
        'ai_rl_2_positions.csv'
    ]
    
    existing_files = []
    source_dir = Path(source_path)
    
    for csv_file in csv_files:
        file_path = source_dir / csv_file
        if file_path.exists():
            existing_files.append(csv_file)
    
    return existing_files

def copy_files(source_path, target_path, csv_files):
    """Copy CSV files from source to target directory."""
    target_path.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    source_dir = Path(source_path)
    
    for csv_file in csv_files:
        source_file = source_dir / csv_file
        target_file = target_path / csv_file
        
        try:
            shutil.copy2(source_file, target_file)
            logger.info(f"Copied: {csv_file} to {target_file}")
            copied_files.append(csv_file)
        except Exception as e:
            logger.error(f"Failed to copy {csv_file}: {e}")
    
    return copied_files

def scan_and_copy_data(map_nr, source_base="/data/samuel_lozano/cooked/classic/v3.1", experiment_name="experiment", dry_run=False):
    """Main function to scan local directories and copy data."""
    # Setup local experiment folder
    experiment_path = setup_experiment_folder(source_base, experiment_name=experiment_name)
    
    # Source base path
    source_base_path = Path(source_base) / f"map_{map_nr}" / "simulations"
    
    logger.info(f"Scanning local directories for {map_nr} data...")
    logger.info(f"Source base path: {source_base_path}")
    
    if not source_base_path.exists():
        logger.error(f"Source path does not exist: {source_base_path}")
        return False
    
    # Find all simulation directories
    simulation_pattern = str(source_base_path / "**/simulation_*")
    simulation_dirs = glob.glob(simulation_pattern, recursive=True)
    
    logger.info(f"Found {len(simulation_dirs)} simulation directories")
    
    total_copied = 0
    total_files = 0
    
    for sim_dir in simulation_dirs:
        sim_path = Path(sim_dir)
        
        # Extract IDs from path
        training_id, checkpoint_number, simulation_id = extract_ids_from_path(sim_path, map_nr)
        
        if not all([training_id, checkpoint_number, simulation_id]):
            logger.warning(f"Could not extract IDs from path: {sim_path}")
            continue
        
        logger.info(f"Processing Training_{training_id}/checkpoint_{checkpoint_number}/simulation_{simulation_id}")
        
        # Check which CSV files exist
        existing_files = check_csv_files_exist(sim_path)
        
        if not existing_files:
            logger.warning(f"No CSV files found in {sim_path}")
            continue
        
        logger.info(f"Found CSV files: {existing_files}")
        
        # Create local directory structure
        local_dir = experiment_path / f"map_{map_nr}" / "simulations" / f"Training_{training_id}" / f"checkpoint_{checkpoint_number}" / f"simulation_{simulation_id}"
        
        if dry_run:
            logger.info(f"[DRY RUN] Would copy {len(existing_files)} files to {local_dir}")
            total_files += len(existing_files)
            continue
        
        # Copy files
        copied = copy_files(sim_path, local_dir, existing_files)
        total_copied += len(copied)
        total_files += len(existing_files)
        
        logger.info(f"Copied {len(copied)}/{len(existing_files)} files for this simulation")
    
    if dry_run:
        logger.info(f"[DRY RUN] Would copy {total_files} files total")
    else:
        logger.info(f"Copy complete! Copied {total_copied}/{total_files} files total")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Copy experiment data from local directories')
    parser.add_argument('map_nr', type=str, help='Map identifier (can be number like "1" or string like "baseline_division_of_labor")')
    parser.add_argument('experiment_name', type=str, help='Experiment name (e.g., "experiment_1")')
    parser.add_argument('--source-path', default='/data/samuel_lozano/cooked/classic/v3.1', 
                        help='Source base path (default: /data/samuel_lozano/cooked/classic/v3.1)')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be copied without actually copying')
    
    args = parser.parse_args()
    
    logger.info(f"Starting copy for {args.map_nr}")
    
    if args.dry_run:
        logger.info("Running in DRY RUN mode - no files will be copied")

    success = scan_and_copy_data(args.map_nr, args.source_path, args.experiment_name, args.dry_run)

    if success:
        logger.info("Script completed successfully")
    else:
        logger.error("Script completed with errors")
        exit(1)

if __name__ == "__main__":
    main()