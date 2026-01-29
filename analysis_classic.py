#!/usr/bin/env python3
"""
Analysis script for classic reinforcement learning experiments.

This script analyzes training results from single-agent experiments,
generating comprehensive visualizations and statistics.

Usage:
nohup python analysis_classic.py <map_name> [options] > analysis_classic.log 2>&1 &

Examples:
nohup python analysis_classic.py baseline_division_of_labor_large --study_name speeds --game_type classic > analysis_classic.log 2>&1 &
nohup python analysis_classic.py baseline_division_of_labor_large --study_name speeds --game_type classic_collision > analysis_classic_collision.log 2>&1 &
nohup python analysis_classic.py baseline_division_of_labor_large --init_type empty_init --game_type classic > analysis_classic_empty_init.log 2>&1 &
nohup python analysis_classic.py baseline_division_of_labor_large --synergy 0.5 --game_type classic > analysis_classic_synergy_0.5.log 2>&1 &
nohup python analysis_classic.py baseline_division_of_labor_large --specialization 0 --game_type classic > analysis_classic_lambda_0.log 2>&1 &
nohup python analysis_classic.py baseline_division_of_labor_large --specialization 5.0 --game_type classic > analysis_classic_lambda_5.log 2>&1 &
"""

import sys
import os
import pandas as pd
from spoiled_broth.analysis.utils import (
    setup_argument_parser, main_analysis_pipeline, MetricDefinitions
)
from spoiled_broth.analysis.individual_training_classic_plots import (
    generate_individual_basic_metrics_plots,
    generate_individual_combined_reward_plots,
    generate_individual_combined_delivery_cut_plots,
    generate_individual_meaningful_actions_combined,
    generate_individual_combined_plots
)
from spoiled_broth.analysis.multi_training_classic_plots import (
    generate_combined_plots,
    generate_meaningful_actions_combined,
    generate_combined_reward_plots,
    generate_combined_delivery_cut_plots
)

# Add the project root to the path to import utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_classic_plots(analysis_results):
    """Generate all plots specific to classic experiments.
    
    Note: All metrics in df are already averaged across NUM_ENVS during data loading.
    Each row represents the mean of NUM_ENVS parallel environments for that episode.
    """
    df = analysis_results['df']
    paths = analysis_results['paths']
    plotter = analysis_results['plotter']
    
    print("Generating classic experiment plots...")
    
    # Log specialization mode if detected
    if paths.get('specialization'):
        print(f"Specialization mode: {paths['specialization']}")
    
    # Generate individual training plots first
    generate_individual_training_plots(analysis_results)
    
    # Get classic-specific metrics
    metrics = MetricDefinitions.get_classic_metrics()
    
    # Define base metric names (without agent suffix)
    base_rewarded_metrics = [
        "deliver",
        "cut",
        "salad",
        "plate",
        "raw_food",
        "counter"
    ]
    
    base_movement_metrics = [
        "do_nothing",
        "useless_floor",
        "useless_wall",
        "useless_counter",
        "useful_counter",
        "salad_assembly",
        "destructive_food_dispenser",
        "useful_food_dispenser",
        "useless_cutting_board",
        "useful_cutting_board",
        "destructive_plate_dispenser",
        "useful_plate_dispenser",
        "useless_delivery",
        "useful_delivery",
        "inaccessible_tile"
    ]
    
    # Generate agent-specific metrics dynamically
    def get_agent_metrics(base_metrics, agent_num):
        return [f"{metric}_ai_rl_{agent_num}" for metric in base_metrics]
    
    rewarded_metrics_1 = get_agent_metrics(base_rewarded_metrics, 1)
    rewarded_metrics_2 = get_agent_metrics(base_rewarded_metrics, 2)
    movement_metrics_1 = get_agent_metrics(base_movement_metrics, 1)
    movement_metrics_2 = get_agent_metrics(base_movement_metrics, 2)
    
    # Basic plots
    plotter.plot_basic_metrics(df, paths['figures_dir'], "total_deliveries", "Score (deliveries)")
    plotter.plot_basic_metrics(df, paths['figures_dir'], "pure_reward_total", "Pure Reward")
    
    # Agent-specific metrics plots for both agents
    plotter.plot_agent_metrics(df, paths['figures_dir'], rewarded_metrics_1, 1)
    plotter.plot_agent_metrics(df, paths['figures_dir'], rewarded_metrics_2, 2)
    
    # Smoothed plots
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "total_deliveries", "Score", by_attitude=False)
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "pure_reward_total", "Pure Reward", by_attitude=False)

    # Smoothed agent metrics for both agents - rewarded metrics
    plotter.plot_agent_metrics(df, paths['smoothed_figures_dir'], rewarded_metrics_1, 1, smoothed=True)
    plotter.plot_agent_metrics(df, paths['smoothed_figures_dir'], rewarded_metrics_2, 2, smoothed=True)
    
    # Smoothed agent metrics for both agents - movement metrics
    plotter.plot_agent_metrics(df, paths['smoothed_figures_dir'], movement_metrics_1, 1, smoothed=True)
    plotter.plot_agent_metrics(df, paths['smoothed_figures_dir'], movement_metrics_2, 2, smoothed=True)
    
    # Generate combined plots (use smoothing factor from config)
    # Note: Smoothing groups episodes into blocks and averages within blocks
    # This is separate from NUM_ENVS averaging which happens during data loading
    smoothing_factor = analysis_results.get('config').smoothing_factor if analysis_results.get('config') else 15
    generate_combined_plots(df, paths, rewarded_metrics_1, rewarded_metrics_2, 
                           movement_metrics_1, movement_metrics_2, smoothing_factor)    # Generate meaningful actions combined plots (excluding counter)
    generate_meaningful_actions_combined(df, paths, base_rewarded_metrics, base_movement_metrics, smoothing_factor)
    
    # Combined reward plots for both agents
    generate_combined_reward_plots(df, paths, smoothing_factor)
    
    # Combined delivery and cut plots for both agents  
    generate_combined_delivery_cut_plots(df, paths, smoothing_factor)
    
    print(f"Classic analysis completed. Figures saved to {paths['figures_dir']}")


def generate_individual_training_plots(analysis_results):
    """Generate plots for each individual training session."""
    df = analysis_results['df']
    paths = analysis_results['paths']
    config = analysis_results.get('config')
    smoothing_factor = config.smoothing_factor if config else 15
    
    print("Generating individual training plots...")
    
    # Create smoothed individual training directory
    individual_smoothed_dir = os.path.join(paths['smoothed_figures_dir'], 'individual_training')
    os.makedirs(individual_smoothed_dir, exist_ok=True)
    
    # Define base metric names (without agent suffix)
    base_rewarded_metrics = [
        "deliver",
        "cut", 
        "salad",
        "plate",
        "raw_food",
        "counter"
    ]
    
    base_movement_metrics = [
        "do_nothing",
        "useless_floor",
        "useless_wall",
        "useless_counter",
        "useful_counter",
        "salad_assembly",
        "destructive_food_dispenser",
        "useful_food_dispenser",
        "useless_cutting_board",
        "useful_cutting_board",
        "destructive_plate_dispenser",
        "useful_plate_dispenser",
        "useless_delivery",
        "useful_delivery",
        "inaccessible_tile"
    ]
    
    # Get unique training sessions (based on timestamp)
    unique_trainings = df['timestamp'].unique()
    print(f"Found {len(unique_trainings)} individual training sessions")
    
    # Generate individual paths for each training
    individual_paths = {
        'smoothed_figures_dir': individual_smoothed_dir
    }
    
    # Generate plots for each training session
    for training_id in unique_trainings:
        training_df = df[df['timestamp'] == training_id].copy()
        
        if len(training_df) == 0:
            continue
            
        print(f"  Processing training {training_id} with {len(training_df)} episodes")
        
        # Get training metadata
        lr = training_df['lr'].iloc[0]
        attitude_key = training_df['attitude_key'].iloc[0]
        speed_key = training_df['speed_key'].iloc[0] if 'speed_key' in training_df.columns else None
        
        # Handle game_type with proper NaN check
        if 'game_type' in training_df.columns:
            game_type_raw = training_df['game_type'].iloc[0]
            # Check if it's NaN or 'nan' string and provide fallback
            if pd.isna(game_type_raw) or str(game_type_raw).lower() == 'nan':
                game_type = 'classic'
            else:
                game_type = str(game_type_raw)
        else:
            game_type = 'classic'
        
        # Extract individual speed values for each agent with proper NaN handling
        walking_speed_1 = None
        cutting_speed_1 = None
        walking_speed_2 = None
        cutting_speed_2 = None
        
        if 'walking_speed_1' in training_df.columns:
            raw_val = training_df['walking_speed_1'].iloc[0]
            walking_speed_1 = raw_val if not pd.isna(raw_val) else None
            
        if 'cutting_speed_1' in training_df.columns:
            raw_val = training_df['cutting_speed_1'].iloc[0]
            cutting_speed_1 = raw_val if not pd.isna(raw_val) else None
            
        if 'walking_speed_2' in training_df.columns:
            raw_val = training_df['walking_speed_2'].iloc[0]
            walking_speed_2 = raw_val if not pd.isna(raw_val) else None
            
        if 'cutting_speed_2' in training_df.columns:
            raw_val = training_df['cutting_speed_2'].iloc[0]
            cutting_speed_2 = raw_val if not pd.isna(raw_val) else None
        
        print(f"  Metadata: game_type={game_type}, speed_key={speed_key}")
        print(f"  Agent 1 speeds: walk={walking_speed_1}, cut={cutting_speed_1}")
        print(f"  Agent 2 speeds: walk={walking_speed_2}, cut={cutting_speed_2}")
        
        # Generate agent-specific metrics dynamically
        def get_agent_metrics(base_metrics, agent_num):
            return [f"{metric}_ai_rl_{agent_num}" for metric in base_metrics]
        
        rewarded_metrics_1 = get_agent_metrics(base_rewarded_metrics, 1)
        rewarded_metrics_2 = get_agent_metrics(base_rewarded_metrics, 2)
        movement_metrics_1 = get_agent_metrics(base_movement_metrics, 1)
        movement_metrics_2 = get_agent_metrics(base_movement_metrics, 2)
        
        # Generate individual training plots
        generate_individual_basic_metrics_plots(training_df, individual_paths, training_id, lr, attitude_key, speed_key, game_type, walking_speed_1, cutting_speed_1, walking_speed_2, cutting_speed_2, smoothing_factor)
        generate_individual_combined_reward_plots(training_df, individual_paths, training_id, lr, attitude_key, speed_key, game_type, walking_speed_1, cutting_speed_1, walking_speed_2, cutting_speed_2, smoothing_factor)
        generate_individual_combined_delivery_cut_plots(training_df, individual_paths, training_id, lr, attitude_key, speed_key, game_type, walking_speed_1, cutting_speed_1, walking_speed_2, cutting_speed_2, smoothing_factor)
        generate_individual_meaningful_actions_combined(training_df, individual_paths, training_id, lr, attitude_key, speed_key, game_type, walking_speed_1, cutting_speed_1, walking_speed_2, cutting_speed_2, base_rewarded_metrics, smoothing_factor)
        generate_individual_combined_plots(training_df, individual_paths, training_id, lr, attitude_key, speed_key, game_type, walking_speed_1, cutting_speed_1, walking_speed_2, cutting_speed_2, rewarded_metrics_1, rewarded_metrics_2, movement_metrics_1, movement_metrics_2, smoothing_factor)
    
    print(f"Individual training plots saved to: {individual_smoothed_dir}")

def main():
    """Main execution function."""
    parser = setup_argument_parser('classic')
    parser.add_argument('--specialization', type=float, 
                       help='Analyze only specific specialization lambda value (e.g., 0, 5.0). If not specified, analyzes all available lambdas.')
    args = parser.parse_args()

    print(f"Starting classic experiment analysis...")
    print(f"Map: {args.map_name}")
    print(f"Cluster: {args.cluster}")
    print(f"Smoothing factor: {args.smoothing_factor}")
    print(f"Study name: {args.study_name}")
    print(f"Game type: {args.game_type}")
    print(f"Init type: {args.init_type}")
    print(f"Synergy scaling factor: {args.synergy}")
    if hasattr(args, 'specialization') and args.specialization is not None:
        print(f"Specialization lambda: {args.specialization}")

    try:
        # Check if synergy was explicitly provided
        synergy_provided = '--synergy' in sys.argv
        
        # Set specialization based on command line argument
        specialization = getattr(args, 'specialization', None)
        
        # Run main analysis pipeline
        analysis_results = main_analysis_pipeline(
            experiment_type='classic',
            map_name=args.map_name,
            cluster=args.cluster,
            smoothing_factor=args.smoothing_factor,
            study_name=args.study_name,
            game_type=args.game_type,
            init_type=args.init_type,
            synergy_scaling_factor=args.synergy,
            synergy_provided=synergy_provided,
            specialization_lambda=specialization
        )

        # Check if we got a dict of results (multiple specialization modes) or single result
        if isinstance(analysis_results, dict) and any(key.startswith('specialized_') for key in analysis_results.keys()):
            # Multiple specialization modes - process each separately
            print("\n" + "=" * 60)
            print("GENERATING PLOTS FOR EACH SPECIALIZATION MODE")
            print("=" * 60)
            
            for spec_mode in sorted(analysis_results.keys()):
                print(f"\n{'='*60}")
                print(f"CREATING PLOTS FOR {spec_mode.upper()}")
                print(f"{'='*60}")
                
                # Generate classic-specific plots for this specialization mode
                generate_classic_plots(analysis_results[spec_mode])
                
                print(f"\n{spec_mode.upper()} plots completed!")
            
            print("\n" + "=" * 60)
            print("ALL SPECIALIZATION MODES COMPLETED SUCCESSFULLY!")
            print("=" * 60)
        else:
            # Single specialization mode
            generate_classic_plots(analysis_results)

        print("Analysis completed successfully!")

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()