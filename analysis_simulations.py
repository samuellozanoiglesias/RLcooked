#!/usr/bin/env python3
"""
Comprehensive Analysis Script for spoiled_broth experiments.

This script orchestrates the complete analysis by calling:
1. analysis_simulations.py - for detailed individual simulation analysis
2. analysis_checkpoint_comparison.py - for checkpoint comparison across training_ids

Both analyses save their outputs to the organized directory: map_{map_nr}/simulation_figures/

Folder structure:
- Simulations: /data/samuel_lozano/cooked/classic/map_{map_nr}/simulations/Training_{training_id}/checkpoint_{checkpoint_number}/simulation_{simulation_id}
- Figures: /data/samuel_lozano/cooked/classic/map_{map_nr}/simulation_figures/

Usage examples:
# Run analysis for all trainings and all checkpoints:
nohup python3 analysis_simulations.py --cluster cuenca --map_nr baseline_division_of_labor --game_type classic --study_name default > log_analysis_simulations.out 2>&1 &

# Run analysis for specific training, all checkpoints:
nohup python3 analysis_simulations.py --cluster cuenca --map_nr baseline_division_of_labor --training_id 12345 --num_agents 1 --game_type classic > log_analysis_simulations.out 2>&1 &

# Run analysis for specific training and checkpoint:
nohup python3 analysis_simulations.py --cluster cuenca --map_nr baseline_division_of_labor --training_id 12345 --checkpoint_number 50 --game_version classic > log_analysis_simulations.out 2>&1 &

Optional arguments:
--training_id: If provided, run analysis only for that training (all checkpoints unless checkpoint_number also specified)
--checkpoint_number: If provided along with training_id, run analysis for that specific combination only
--output_dir: Custom output directory (default: map_{map_nr}/simulation_figures/)

Author: Samuel Lozano
"""

import argparse
import sys
from spoiled_broth.analysis.analysis_orchestrator_simulations import ComprehensiveAnalysisOrchestrator

def main():
    """Main function to run the comprehensive analysis."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive analysis of spoiled_broth experiments by orchestrating individual analysis scripts'
    )
    
    # Required arguments
    parser.add_argument('--map_nr', type=str, required=True,
                       help='Map number/name (e.g., "baseline_division_of_labor")')
    
    # Optional arguments with defaults
    parser.add_argument('--cluster', type=str, default='cuenca',
                       help='Base cluster (default: cuenca)')
    parser.add_argument('--game_version', type=str, default='classic', 
                       choices=['classic', 'competition', 'classic_collision'],
                       help='Game version (default: classic)')
    parser.add_argument('--num_agents', type=int, default=2, choices=[1, 2],
                       help='Number of agents (default: 2)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: map_{map_nr}/simulation_figures/)')
    
    # Optional arguments for detailed simulation analysis
    parser.add_argument('--training_id', type=str, default=None,
                       help='Optional: specific training ID for detailed simulation analysis')
    parser.add_argument('--checkpoint_number', type=str, default=None,
                       help='Optional: specific checkpoint for detailed simulation analysis (e.g., "final", "50")')
    parser.add_argument('--study_name', type=str, default='',
                       help='Study name for simulation folders (default: empty, use /simulations/ directly)')
    parser.add_argument('--game_type', type=str, default='classic',
                       help='Game type for folder organization (default: classic)')
    parser.add_argument('--synergy', type=str, default=None,
                       help='Optional synergy folder value (e.g., 1.70 -> synergy_1.70)')
    parser.add_argument('--specialization', type=str, default=None,
                       help='Optional specialization folder value (e.g., 0.05 -> specialized_0.05)')
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

    # Create orchestrator and run comprehensive analysis
    study_name = (args.study_name or '').strip()

    orchestrator = ComprehensiveAnalysisOrchestrator(
        base_cluster_dir=base_cluster_dir,
        map_nr=args.map_nr,
        game_version=args.game_version,
        num_agents=args.num_agents,
        training_id=args.training_id,
        checkpoint_number=args.checkpoint_number,
        output_dir=args.output_dir,
        study_name=study_name,
        game_type=args.game_type,
        synergy=args.synergy,
        specialization=args.specialization
    )

    simulations_success, checkpoint_success = orchestrator.run_comprehensive_analysis()
    
    # Return appropriate exit code
    if args.training_id:
        # Specific training_id mode - only simulation analysis matters
        return 0 if simulations_success else 1
    else:
        # All training_ids mode - both analyses matter
        if simulations_success and checkpoint_success:
            return 0  # All good
        elif checkpoint_success:
            return 0  # Checkpoint analysis completed, some simulations may have failed but that's acceptable
        else:
            return 1  # Major failure


if __name__ == "__main__":
    sys.exit(main())
