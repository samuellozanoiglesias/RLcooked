#!/usr/bin/env python3
"""
Analysis script for pretrained reinforcement learning experiments.

This script analyzes training results from experiments using pretrained models,
generating comprehensive visualizations and statistics.

Usage:
nohup python analysis_pretrained.py <map_name> [options] > analysis_pretrained.log 2>&1 &

Examples:
nohup python analysis_pretrained.py baseline_division_of_labor_v2 --cluster cuenca --smoothing_factor 15 > analysis_pretrained.log 2>&1 &
nohup python analysis_pretrained.py baseline_division_of_labor_v2 --cluster cuenca --smoothing_factor 15 --individual_trainings yes > analysis_pretrained.log 2>&1 &
nohup python analysis_pretrained.py baseline_division_of_labor_v2 --cluster cuenca --init_type random_init > analysis_pretrained.log 2>&1 &
nohup python analysis_pretrained.py baseline_division_of_labor_v2 --cluster cuenca --smoothing_factor 15 --init_type empty_init --individual_trainings yes > analysis_pretrained.log 2>&1 &
nohup python analysis_pretrained.py baseline_division_of_labor_v2 --cluster cuenca --game_type classic_collision --init_type random_init > analysis_pretrained.log 2>&1 &
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the path to import utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spoiled_broth.analysis.utils import (
    setup_argument_parser, main_analysis_pipeline, MetricDefinitions
)


def detect_training_agent(df):
    """Detect which agent is being trained based on column names.
    
    Returns:
        int or 'both': Agent number (1 or 2) if only one is present, 'both' if both are present
    """
    # Check for agent 1 metrics
    has_agent_1 = any('ai_rl_1' in col for col in df.columns)
    # Check for agent 2 metrics
    has_agent_2 = any('ai_rl_2' in col for col in df.columns)
    
    if has_agent_1 and not has_agent_2:
        return 1
    elif has_agent_2 and not has_agent_1:
        return 2
    elif has_agent_1 and has_agent_2:
        # Both present - mixed training sessions
        print("Info: Both agent 1 and agent 2 training sessions found in dataset.")
        return 'both'
    else:
        # Neither found, default to 1
        print("Warning: No agent-specific metrics found. Defaulting to agent 1.")
        return 1


def detect_agent_for_training(training_df):
    """Detect which agent a specific training session is for.
    
    Args:
        training_df: DataFrame for a single training session
        
    Returns:
        int: Agent number (1 or 2)
    """
    # Check which agent has non-null data
    has_agent_1_data = False
    has_agent_2_data = False
    
    for col in training_df.columns:
        if 'ai_rl_1' in col:
            if training_df[col].notna().any():
                has_agent_1_data = True
                break
    
    for col in training_df.columns:
        if 'ai_rl_2' in col:
            if training_df[col].notna().any():
                has_agent_2_data = True
                break
    
    if has_agent_1_data and not has_agent_2_data:
        return 1
    elif has_agent_2_data and not has_agent_1_data:
        return 2
    else:
        # If both or neither, try to infer from columns that exist
        agent_1_cols = [col for col in training_df.columns if 'ai_rl_1' in col]
        agent_2_cols = [col for col in training_df.columns if 'ai_rl_2' in col]
        
        if len(agent_1_cols) > len(agent_2_cols):
            return 1
        elif len(agent_2_cols) > len(agent_1_cols):
            return 2
        else:
            return 1  # Default


def generate_pretrained_plots(analysis_results):
    """Generate all plots specific to pretrained experiments."""
    df = analysis_results['df']
    paths = analysis_results['paths']
    plotter = analysis_results['plotter']
    num_agents = analysis_results['num_agents']
    individual_trainings = analysis_results.get('individual_trainings', False)
    
    # Detect which agent(s) are being trained
    trained_agent = detect_training_agent(df)
    analysis_results['trained_agent'] = trained_agent
    
    if trained_agent == 'both':
        print(f"Generating pretrained experiment plots for mixed agent trainings (Agent 1 and Agent 2)...")
        agents_to_plot = [1, 2]
    else:
        print(f"Generating pretrained experiment plots for Agent {trained_agent}...")
        agents_to_plot = [trained_agent]
    
    # Get metrics appropriate for the number of agents
    metrics = MetricDefinitions.get_classic_metrics()
    
    # Generate plots for each agent that has data
    for agent_num in agents_to_plot:
        agent_suffix = f"ai_rl_{agent_num}"
        
        # Filter data for this agent (only rows where this agent has non-null data)
        agent_rows = df[[col for col in df.columns if f'ai_rl_{agent_num}' in col]].notna().any(axis=1)
        agent_df = df[agent_rows].copy()
        
        if len(agent_df) == 0:
            print(f"  No data found for Agent {agent_num}, skipping...")
            continue
            
        print(f"  Generating plots for Agent {agent_num} ({len(agent_df)} episodes)")
        
        # Basic plots - agent-specific
        if f"pure_reward_{agent_suffix}" in agent_df.columns:
            plotter.plot_basic_metrics(agent_df, paths['figures_dir'], f"pure_reward_{agent_suffix}", f"Pure Reward Agent {agent_num}")
        if f"modified_reward_{agent_suffix}" in agent_df.columns:
            plotter.plot_basic_metrics(agent_df, paths['figures_dir'], f"modified_reward_{agent_suffix}", f"Modified Reward Agent {agent_num}")
        
        # Deliveries for this agent
        if f"deliver_{agent_suffix}" in agent_df.columns:
            plotter.plot_basic_metrics(agent_df, paths['figures_dir'], f"deliver_{agent_suffix}", f"Deliveries Agent {agent_num}")
        
        # Agent-specific metrics plots
        metrics_key = f'rewarded_metrics_{agent_num}'
        if metrics_key in metrics:
            plotter.plot_agent_metrics(agent_df, paths['figures_dir'], metrics[metrics_key], agent_num)
        
        # Smoothed plots for this agent
        if f"pure_reward_{agent_suffix}" in agent_df.columns:
            plotter.plot_smoothed_metrics(agent_df, paths['smoothed_figures_dir'], f"pure_reward_{agent_suffix}", f"Pure Reward Agent {agent_num}")
        if f"modified_reward_{agent_suffix}" in agent_df.columns:
            plotter.plot_smoothed_metrics(agent_df, paths['smoothed_figures_dir'], f"modified_reward_{agent_suffix}", f"Modified Reward Agent {agent_num}")
        if f"deliver_{agent_suffix}" in agent_df.columns:
            plotter.plot_smoothed_metrics(agent_df, paths['smoothed_figures_dir'], f"deliver_{agent_suffix}", f"Deliveries Agent {agent_num}")
        
        # Smoothed agent metrics
        rewarded_key = f'rewarded_metrics_{agent_num}'
        movement_key = f'movement_metrics_{agent_num}'
        if rewarded_key in metrics:
            plotter.plot_agent_metrics(agent_df, paths['smoothed_figures_dir'], metrics[rewarded_key], agent_num, smoothed=True)
        if movement_key in metrics:
            plotter.plot_agent_metrics(agent_df, paths['smoothed_figures_dir'], metrics[movement_key], agent_num, smoothed=True)
    
    # Generate individual training plots if requested
    if individual_trainings:
        generate_individual_training_plots(analysis_results)
    
    # Generate attitude-specific analysis
    generate_attitude_analysis(analysis_results)
    
    print(f"Pretrained analysis completed. Figures saved to {paths['figures_dir']}")


def generate_individual_training_plots(analysis_results):
    """Generate plots for each individual training session."""
    
    df = analysis_results['df']
    paths = analysis_results['paths']
    config = analysis_results['config']
    trained_agent = analysis_results['trained_agent']
    
    print(f"Generating individual training plots...")
    
    # Create individual training directory
    individual_dir = os.path.join(paths['figures_dir'], 'individual_trainings')
    os.makedirs(individual_dir, exist_ok=True)
    
    # Get unique training sessions (based on timestamp)
    unique_trainings = df['timestamp'].unique()
    
    print(f"Found {len(unique_trainings)} individual training sessions")
    
    # Apply smoothing factor
    N = config.smoothing_factor
    
    # Generate plots for each training session
    for training_id in unique_trainings:
        training_df = df[df['timestamp'] == training_id].copy()
        
        if len(training_df) == 0:
            continue
        
        # Detect which agent this specific training is for
        training_agent = detect_agent_for_training(training_df)
        print(f"  Training {training_id}: Agent {training_agent}")
        
        # Get available reward metrics for this training's agent
        agent_suffix = f"ai_rl_{training_agent}"
        available_rewarded_metrics = []
        potential_metrics = [
            (f"delivered_{agent_suffix}", "#27AE60", "Delivered"),
            (f"cut_{agent_suffix}", "#2980B9", "Cut"), 
            (f"salad_{agent_suffix}", "#E67E22", "Salad"),
            (f"deliver_{agent_suffix}", "#27AE60", "Deliver"),  # Alternative name
            (f"plate_{agent_suffix}", "#9B59B6", "Plate"),
            (f"raw_food_{agent_suffix}", "#E74C3C", "Raw Food"),
            (f"counter_{agent_suffix}", "#34495E", "Counter")
        ]
        
        for metric_name, color, label in potential_metrics:
            if metric_name in training_df.columns and training_df[metric_name].notna().any():
                available_rewarded_metrics.append((metric_name, color, label))
        
        # Get training metadata
        lr = training_df['lr'].iloc[0]
        attitude = training_df['attitude_key'].iloc[0]
        
        # Apply smoothing
        training_df["episode_block"] = (training_df["episode"] // N)
        
        # Plot 1: Basic reward metrics
        plt.figure(figsize=(12, 8))
        
        for metric, color, label in available_rewarded_metrics:
            if metric in training_df.columns:
                block_means = training_df.groupby("episode_block")[metric].mean()
                middle_episodes = training_df.groupby("episode_block")["episode"].median()
                plt.plot(middle_episodes, block_means, label=label, color=color, linewidth=2)
        
        plt.title(f"Training {training_id} - Agent {training_agent} (Smoothed {N})\nAttitude: {attitude}, LR: {lr}")
        plt.xlabel("Episode")
        plt.ylabel("Reward Values")
        if available_rewarded_metrics:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save with sanitized filename
        safe_training_id = training_id.replace(':', '_').replace(' ', '_')
        filename = f"individual_training_{safe_training_id}_rewards_smoothed_{N}.png"
        plt.savefig(os.path.join(individual_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Pure reward and deliveries for this specific agent
        plt.figure(figsize=(12, 6))
        
        # Use agent-specific metrics
        pure_reward_col = f"pure_reward_ai_rl_{training_agent}"
        deliveries_col = f"deliver_ai_rl_{training_agent}"
        
        if pure_reward_col in training_df.columns and training_df[pure_reward_col].notna().any():
            plt.subplot(1, 2, 1)
            block_means = training_df.groupby("episode_block")[pure_reward_col].mean()
            middle_episodes = training_df.groupby("episode_block")["episode"].median()
            plt.plot(middle_episodes, block_means, color='#3498DB', linewidth=2)
            plt.title(f"Pure Reward Agent {training_agent}")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True, alpha=0.3)
        
        if deliveries_col in training_df.columns and training_df[deliveries_col].notna().any():
            plt.subplot(1, 2, 2)
            block_means = training_df.groupby("episode_block")[deliveries_col].mean()
            middle_episodes = training_df.groupby("episode_block")["episode"].median()
            plt.plot(middle_episodes, block_means, color='#E74C3C', linewidth=2)
            plt.title(f"Deliveries Agent {training_agent}")
            plt.xlabel("Episode")
            plt.ylabel("Deliveries")
            plt.grid(True, alpha=0.3)
        
        plt.suptitle(f"Training {training_id} - Agent {training_agent} - Score Metrics (Smoothed {N})\nAttitude: {attitude}, LR: {lr}")
        plt.tight_layout()
        
        filename = f"individual_training_{safe_training_id}_score_smoothed_{N}.png"
        plt.savefig(os.path.join(individual_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Individual training plots saved to: {individual_dir}")


def generate_attitude_analysis(analysis_results):
    """Generate analysis plots grouped by individual attitudes."""
    df = analysis_results['df']
    paths = analysis_results['paths']
    config = analysis_results['config']
    num_agents = analysis_results['num_agents']
    individual_trainings = analysis_results.get('individual_trainings', False)
    trained_agent = analysis_results['trained_agent']
    
    # Determine which agents to analyze
    if trained_agent == 'both':
        agents_to_analyze = [1, 2]
        print(f"Generating attitude-specific analysis for both Agent 1 and Agent 2...")
    else:
        agents_to_analyze = [trained_agent]
        print(f"Generating attitude-specific analysis for Agent {trained_agent}...")
    
    # Process each agent separately
    for agent_num in agents_to_analyze:
        # For single agent experiments, attitude_key is already just the agent's attitude
        if num_agents == 1:
            df[f'attitude_agent_{agent_num}'] = df['attitude_key']  # attitude_key is already "alpha_beta"
            unique_attitudes = df["attitude_key"].unique()
            unique_individual_attitudes = set(unique_attitudes)
        else:
            # Create individual attitude keys for each agent (legacy 2-agent code)
            df[f'attitude_agent_{agent_num}'] = df[f'alpha_{agent_num}'].astype(str) + '_' + df[f'beta_{agent_num}'].astype(str)
            
            # Get unique individual attitudes
            unique_attitudes = df["attitude_key"].unique()
            unique_individual_attitudes = set()
            
            for attitude in unique_attitudes:
                att_parts = attitude.split('_')
                # Extract attitude for this agent
                agent_offset = (agent_num - 1) * 2
                unique_individual_attitudes.add(f"{att_parts[agent_offset]}_{att_parts[agent_offset + 1]}")
        
        print(f"Individual attitudes found for Agent {agent_num}: {sorted(unique_individual_attitudes)}")
        
        # Generate plots for individual attitudes
        generate_individual_attitude_plots(df, paths, unique_individual_attitudes, config, individual_trainings, agent_num)
        
        # Generate combined attitude plots (only meaningful for multi-agent)
        if num_agents > 1:
            generate_combined_attitude_plots(df, paths, unique_attitudes, config, individual_trainings, agent_num)


def generate_individual_attitude_plots(df, paths, unique_individual_attitudes, config, individual_trainings=False, trained_agent=1):
    """Generate plots for individual attitudes with averaged metrics."""    

    N = config.smoothing_factor
    unique_lr = df["lr"].unique()
    agent_suffix = f"ai_rl_{trained_agent}"
    
    # Check which reward metrics are actually available in the dataframe
    available_rewarded_metrics = []
    potential_metrics = [
        (f"delivered_{agent_suffix}", "#27AE60", "Delivered"),
        (f"cut_{agent_suffix}", "#2980B9", "Cut"), 
        (f"salad_{agent_suffix}", "#E67E22", "Salad"),
        (f"deliver_{agent_suffix}", "#27AE60", "Deliver"),  # Alternative name
        (f"plate_{agent_suffix}", "#9B59B6", "Plate"),
        (f"raw_food_{agent_suffix}", "#E74C3C", "Raw Food"),
        (f"counter_{agent_suffix}", "#34495E", "Counter")
    ]
    
    for metric_name, color, label in potential_metrics:
        if metric_name in df.columns and df[metric_name].notna().any():
            available_rewarded_metrics.append((metric_name, color, label))
    
    print(f"Available rewarded metrics for Agent {trained_agent}: {[m[0] for m in available_rewarded_metrics]}")
    
    for individual_attitude in unique_individual_attitudes:
        att_parts = individual_attitude.split('_')
        alpha = float(att_parts[0])
        beta = float(att_parts[1])
        
        # Calculate degree for title
        if alpha == 0 and beta == 0:
            degree = 0
        else:
            degree = np.degrees(np.arctan2(beta, alpha)) % 360
        
        for lr in unique_lr:
            # Filter data where agent has this attitude
            mask_agent = (df[f'attitude_agent_{trained_agent}'] == individual_attitude)
            filtered_subset = df[mask_agent].copy()

            if len(filtered_subset) > 0:
                # Generate individual training plots if requested
                if individual_trainings:
                    generate_individual_training_attitude_plots(
                        filtered_subset, paths, individual_attitude, lr, degree, N, available_rewarded_metrics, trained_agent
                    )
                
                # Generate averaged plots (always generated)
                filtered_subset["episode_block"] = (filtered_subset["episode"] // N)
                
                # Plot rewarded metrics
                plt.figure(figsize=(12, 6))
                
                for metric, color, label in available_rewarded_metrics:
                    if metric in filtered_subset.columns:
                        block_means = filtered_subset.groupby("episode_block")[metric].mean()
                        middle_episodes = filtered_subset.groupby("episode_block")["episode"].median()
                        plt.plot(middle_episodes, block_means, label=label, color=color)
                
                plt.title(f"Rewarded Metrics (Averaged) - Agent {trained_agent} - Attitude {individual_attitude} ({degree:.1f}°)\n"
                         f"LR {lr} (Smoothed {N})")
                plt.xlabel("Episode")
                plt.ylabel("Mean value")
                if available_rewarded_metrics:  # Only add legend if we have metrics
                    plt.legend()
                plt.tight_layout()
                
                sanitized_attitude = individual_attitude.replace('.', 'p')
                filename = f"rewarded_individual_attitude_{sanitized_attitude}_lr{str(lr).replace('.', 'p')}_smoothed_{N}.png"
                plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
                plt.close()


def generate_individual_training_attitude_plots(filtered_subset, paths, individual_attitude, lr, degree, N, available_rewarded_metrics, trained_agent=1):
    """Generate individual training plots for a specific attitude."""
    
    # Create directory for individual training attitude plots
    individual_attitude_dir = os.path.join(paths['smoothed_figures_dir'], 'individual_trainings_by_attitude')
    os.makedirs(individual_attitude_dir, exist_ok=True)
    
    # Get unique training sessions
    unique_trainings = filtered_subset['timestamp'].unique()
    
    for training_id in unique_trainings:
        training_df = filtered_subset[filtered_subset['timestamp'] == training_id].copy()
        
        if len(training_df) == 0:
            continue
        
        # Apply smoothing
        training_df["episode_block"] = (training_df["episode"] // N)
            
        # Plot rewarded metrics for this specific training
        plt.figure(figsize=(12, 6))
        
        for metric, color, label in available_rewarded_metrics:
            if metric in training_df.columns:
                block_means = training_df.groupby("episode_block")[metric].mean()
                middle_episodes = training_df.groupby("episode_block")["episode"].median()
                plt.plot(middle_episodes, block_means, label=label, color=color, linewidth=2)
        
        plt.title(f"Training {training_id} - Agent {trained_agent} - Attitude {individual_attitude} ({degree:.1f}°) (Smoothed {N})\n"
                 f"LR {lr}")
        plt.xlabel("Episode")
        plt.ylabel("Reward Values")
        if available_rewarded_metrics:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save with sanitized filename
        safe_training_id = training_id.replace(':', '_').replace(' ', '_')
        sanitized_attitude = individual_attitude.replace('.', 'p')
        filename = f"individual_training_{safe_training_id}_attitude_{sanitized_attitude}_lr{str(lr).replace('.', 'p')}_smoothed_{N}.png"
        plt.savefig(os.path.join(individual_attitude_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()


def generate_combined_attitude_plots(df, paths, unique_attitudes, config, individual_trainings=False, trained_agent=1):
    """Generate plots showing metrics averaged over all other agent attitudes."""
    
    N = config.smoothing_factor
    unique_lr = df["lr"].unique()
    agent_suffix = f"ai_rl_{trained_agent}"
    
    # Check which reward metrics are actually available in the dataframe
    available_rewarded_metrics = []
    potential_metrics = [
        (f"delivered_{agent_suffix}", "#27AE60", "Delivered"),
        (f"cut_{agent_suffix}", "#2980B9", "Cut"), 
        (f"salad_{agent_suffix}", "#E67E22", "Salad"),
        (f"deliver_{agent_suffix}", "#27AE60", "Deliver"),  # Alternative name
        (f"plate_{agent_suffix}", "#9B59B6", "Plate"),
        (f"raw_food_{agent_suffix}", "#E74C3C", "Raw Food"),
        (f"counter_{agent_suffix}", "#34495E", "Counter")
    ]
    
    for metric_name, color, label in potential_metrics:
        if metric_name in df.columns and df[metric_name].notna().any():
            available_rewarded_metrics.append((metric_name, color, label))
    
    for attitude in unique_attitudes:
        subset = df[df["attitude_key"] == attitude]
        att_parts = attitude.split('_')
        att1_title = f"{att_parts[0]}_{att_parts[1]}"
        
        for lr in unique_lr:
            game_lr_filtered = subset[(subset["lr"] == lr)]

            if len(game_lr_filtered) > 0:
                game_lr_filtered = game_lr_filtered.copy()
                game_lr_filtered["episode_block"] = (game_lr_filtered["episode"] // N)
                
                # Plot rewarded metrics averaged
                plt.figure(figsize=(12, 6))
                
                for metric, color, label in available_rewarded_metrics:
                    if metric in game_lr_filtered.columns:
                        block_means = game_lr_filtered.groupby("episode_block")[metric].mean()
                        middle_episodes = game_lr_filtered.groupby("episode_block")["episode"].median()
                        plt.plot(middle_episodes, block_means, label=label, color=color)
                
                plt.title(f"Rewarded Metrics (Averaged) - Agent {trained_agent} - Attitude {att1_title}\n"
                         f"LR {lr} (Smoothed {N})")
                plt.xlabel("Episode")
                plt.ylabel("Mean value")
                if available_rewarded_metrics:  # Only add legend if we have metrics
                    plt.legend()
                plt.tight_layout()
                
                sanitized_attitude = attitude.replace('.', 'p')
                filename = f"rewarded_avg_attitude_{sanitized_attitude}_lr{str(lr).replace('.', 'p')}_smoothed_{N}.png"
                plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
                plt.close()


def main():
    """Main execution function."""
    parser = setup_argument_parser('pretrained')
    args = parser.parse_args()
    
    # Parse individual_trainings flag
    individual_trainings = args.individual_trainings.lower() in ['yes', 'y']
    
    print(f"Starting pretraining experiment analysis...")
    print(f"Map: {args.map_name}")
    print(f"Cluster: {args.cluster}")
    print(f"Smoothing factor: {args.smoothing_factor}")
    print(f"Study name: {args.study_name}")
    print(f"Game type: {args.game_type}")
    print(f"Eta: {args.eta}")
    print(f"Individual trainings: {individual_trainings}")
    
    # Determine which inits to process
    if args.init_type:
        init_types = [args.init_type]
        print(f"Processing single init type: {args.init_type}")
    else:
        # Process all available init types
        init_types = ['random_init', 'empty_init']
        print(f"Processing all init types: {init_types}")
    
    try:
        # Check if eta was explicitly provided
        eta_provided = '--eta' in sys.argv
        
        for init_type in init_types:
            print(f"\n{'='*80}")
            print(f"Processing init type: {init_type}")
            print(f"{'='*80}\n")
            
            # Run main analysis pipeline
            analysis_results = main_analysis_pipeline(
                experiment_type='pretraining',
                map_name=args.map_name,
                cluster=args.cluster,
                smoothing_factor=args.smoothing_factor,
                num_agents=1,  # Pretrained experiments use only 1 agent
                study_name=args.study_name,
                game_type=args.game_type,
                init_type=init_type,
                eta=args.eta,
                eta_provided=eta_provided
            )
            
            # Add individual_trainings flag to results
            analysis_results['individual_trainings'] = individual_trainings
            
            # Generate pretrained-specific plots
            generate_pretrained_plots(analysis_results)
            
            print(f"\nCompleted analysis for init type: {init_type}")
        
        print(f"\n{'='*80}")
        print("All analysis completed successfully!")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()