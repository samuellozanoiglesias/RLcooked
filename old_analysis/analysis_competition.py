#!/usr/bin/env python3
"""
Analysis script for competition reinforcement learning experiments.

This script analyzes training results from two-agent competitive experiments,
generating comprehensive visualizations and statistics.

Usage:
nohup python analysis_competition.py <map_nr> [<optional>] > analysis_competition.log 2>&1 &

Example:
nohup python analysis_competition.py baseline_competition --game_type competition_collision --cluster brigit --init_type empty_init --smoothing_factor 15 > analysis_competition.log 2>&1 &
"""

import sys
import os
from spoiled_broth.analysis.utils import (
    setup_argument_parser, main_analysis_pipeline
)
from spoiled_broth.analysis.competition_plots import (
    generate_individual_training_plots,
    generate_individual_training_competition_specific_plots,
    generate_multi_training_competition_comparison_plots,
    generate_individual_attitude_plots,
    generate_combined_attitude_plots
)

# Add the project root to the path to import utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_competition_plots(analysis_results):
    """Generate all plots specific to competition experiments.
    
    Note: All metrics in df are already averaged across NUM_ENVS during data loading.
    Each row represents the mean of NUM_ENVS parallel environments for that episode.
    """
    paths = analysis_results['paths']
    
    print("Generating competition experiment plots...")
    
    # Generate individual training plots first
    generate_individual_training_plots(analysis_results)
    print("Individual training plots generated.")

     # Generate attitude-specific analysis (competition-specific)
    generate_attitude_analysis(analysis_results)
    print("Attitude-specific analysis generated.")

    # Generate specific requested plots
    generate_individual_training_competition_specific_plots(analysis_results)
    print("Individual training specific plots generated.")

    generate_multi_training_competition_comparison_plots(analysis_results)
    print("Multi-training comparison plots generated.")

    print(f"Competition analysis completed. Figures saved to {paths['figures_dir']}")


def generate_attitude_analysis(analysis_results):
    """Generate analysis plots grouped by individual attitudes."""
    df = analysis_results['df']
    paths = analysis_results['paths']
    config = analysis_results['config']
    
    print("Generating attitude-specific analysis...")
    
    # Create individual attitude keys for each agent
    df['attitude_agent_1'] = df['alpha_1'].astype(str) + '_' + df['beta_1'].astype(str)
    df['attitude_agent_2'] = df['alpha_2'].astype(str) + '_' + df['beta_2'].astype(str)
    
    # Get unique individual attitudes
    unique_attitudes = df["attitude_key"].unique()
    unique_individual_attitudes = set()
    
    for attitude in unique_attitudes:
        att_parts = attitude.split('_')
        unique_individual_attitudes.add(f"{att_parts[0]}_{att_parts[1]}")  # agent 1 attitude
        unique_individual_attitudes.add(f"{att_parts[2]}_{att_parts[3]}")  # agent 2 attitude
    
    print(f"Individual attitudes found: {sorted(unique_individual_attitudes)}")
    
    # Generate plots for individual attitudes
    generate_individual_attitude_plots(df, paths, unique_individual_attitudes, config)
    
    # Generate combined attitude plots
    generate_combined_attitude_plots(df, paths, unique_attitudes, config)


def main():
    """Main execution function."""
    parser = setup_argument_parser('competition')
    args = parser.parse_args()

    allowed_game_types = {'competition', 'competition_collision'}
    if args.game_type not in allowed_game_types:
        parser.error(
            f"Invalid --game_type '{args.game_type}' for analysis_competition.py. "
            f"Allowed values: {', '.join(sorted(allowed_game_types))}."
        )
    
    print(f"Starting competition experiment analysis...")
    print(f"Map: {args.map_name}")
    print(f"Cluster: {args.cluster}")
    print(f"Smoothing factor: {args.smoothing_factor}")
    print(f"Study name: {args.study_name}")
    print(f"Game type: {args.game_type}")
    print(f"Init type: {args.init_type}")
    print(f"Synergy scaling factor: {args.synergy_scaling_factor}")
    
    try:
        # Check if synergy_scaling_factor was explicitly provided
        synergy_provided = '--synergy_scaling_factor' in sys.argv
        
        # Run main analysis pipeline
        analysis_results = main_analysis_pipeline(
            experiment_type='competition',
            map_name=args.map_name,
            cluster=args.cluster,
            smoothing_factor=args.smoothing_factor,
            study_name=args.study_name,
            game_type=args.game_type,
            init_type=args.init_type,
            synergy_scaling_factor=args.synergy_scaling_factor,
            synergy_provided=synergy_provided
        )
        
        # Generate competition-specific plots
        generate_competition_plots(analysis_results)
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()