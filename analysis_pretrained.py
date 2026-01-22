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


def generate_pretrained_plots(analysis_results):
    """Generate all plots specific to pretrained experiments."""
    df = analysis_results['df']
    paths = analysis_results['paths']
    plotter = analysis_results['plotter']
    num_agents = analysis_results['num_agents']
    individual_trainings = analysis_results.get('individual_trainings', False)
    
    print("Generating pretrained experiment plots...")
    
    # Get metrics appropriate for the number of agents
    if num_agents == 1:
        # For single agent, use agent 1 metrics
        metrics = MetricDefinitions.get_classic_metrics()
    else:
        # For multi-agent, use the full classic metrics
        metrics = MetricDefinitions.get_classic_metrics()
    
    # Basic plots - use total metrics that work for both single and multi-agent
    plotter.plot_basic_metrics(df, paths['figures_dir'], "total_deliveries", "Score (deliveries)")
    plotter.plot_basic_metrics(df, paths['figures_dir'], "pure_reward_total", "Pure Reward")
    
    # Individual agent plots - these should always work since we have agent 1
    if "pure_reward_ai_rl_1" in df.columns:
        plotter.plot_basic_metrics(df, paths['figures_dir'], "pure_reward_ai_rl_1", "Pure Reward Agent 1")
    if "modified_reward_ai_rl_1" in df.columns:
        plotter.plot_basic_metrics(df, paths['figures_dir'], "modified_reward_ai_rl_1", "Modified Reward Agent 1")
    
    # Agent-specific metrics plots (only for agent 1 in pretrained)
    plotter.plot_agent_metrics(df, paths['figures_dir'], metrics['rewarded_metrics_1'], 1)
    
    # Smoothed plots
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "total_deliveries", "Score", by_attitude=False)
    plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "pure_reward_total", "Pure Reward", by_attitude=False)
    
    if "pure_reward_ai_rl_1" in df.columns:
        plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "pure_reward_ai_rl_1", "Pure Reward Agent 1")
    if "modified_reward_ai_rl_1" in df.columns:
        plotter.plot_smoothed_metrics(df, paths['smoothed_figures_dir'], "modified_reward_ai_rl_1", "Modified Reward Agent 1")
    
    # Smoothed agent metrics
    plotter.plot_agent_metrics(df, paths['smoothed_figures_dir'], metrics['rewarded_metrics_1'], 1, smoothed=True)
    plotter.plot_agent_metrics(df, paths['smoothed_figures_dir'], metrics['movement_metrics_1'], 1, smoothed=True)
    
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
    
    print("Generating individual training plots...")
    
    # Create individual training directory
    individual_dir = os.path.join(paths['figures_dir'], 'individual_trainings')
    os.makedirs(individual_dir, exist_ok=True)
    
    # Get available reward metrics
    available_rewarded_metrics = []
    potential_metrics = [
        ("delivered_ai_rl_1", "#27AE60", "Delivered"),
        ("cut_ai_rl_1", "#2980B9", "Cut"), 
        ("salad_ai_rl_1", "#E67E22", "Salad"),
        ("deliver_ai_rl_1", "#27AE60", "Deliver"),  # Alternative name
        ("plate_ai_rl_1", "#9B59B6", "Plate"),
        ("raw_food_ai_rl_1", "#E74C3C", "Raw Food"),
        ("counter_ai_rl_1", "#34495E", "Counter")
    ]
    
    for metric_name, color, label in potential_metrics:
        if metric_name in df.columns:
            available_rewarded_metrics.append((metric_name, color, label))
    
    # Get unique training sessions (based on timestamp)
    unique_trainings = df['timestamp'].unique()
    unique_lr = df["lr"].unique()
    unique_attitudes = df["attitude_key"].unique()
    
    print(f"Found {len(unique_trainings)} individual training sessions")
    
    # Apply smoothing factor
    N = config.smoothing_factor
    
    # Generate plots for each training session
    for training_id in unique_trainings:
        training_df = df[df['timestamp'] == training_id].copy()
        
        if len(training_df) == 0:
            continue
            
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
        
        plt.title(f"Training {training_id} (Smoothed {N})\nAttitude: {attitude}, LR: {lr}")
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
        
        # Plot 2: Pure reward and total deliveries
        plt.figure(figsize=(12, 6))
        
        if "pure_reward_total" in training_df.columns:
            plt.subplot(1, 2, 1)
            block_means = training_df.groupby("episode_block")["pure_reward_total"].mean()
            middle_episodes = training_df.groupby("episode_block")["episode"].median()
            plt.plot(middle_episodes, block_means, color='#3498DB', linewidth=2)
            plt.title("Pure Reward Total")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True, alpha=0.3)
        
        if "total_deliveries" in training_df.columns:
            plt.subplot(1, 2, 2)
            block_means = training_df.groupby("episode_block")["total_deliveries"].mean()
            middle_episodes = training_df.groupby("episode_block")["episode"].median()
            plt.plot(middle_episodes, block_means, color='#E74C3C', linewidth=2)
            plt.title("Total Deliveries")
            plt.xlabel("Episode")
            plt.ylabel("Deliveries")
            plt.grid(True, alpha=0.3)
        
        plt.suptitle(f"Training {training_id} - Core Metrics (Smoothed {N})\nAttitude: {attitude}, LR: {lr}")
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
    
    print("Generating attitude-specific analysis...")
    
    # For single agent experiments, attitude_key is already just agent 1's attitude
    if num_agents == 1:
        df['attitude_agent_1'] = df['attitude_key']  # attitude_key is already "alpha_1_beta_1"
        unique_attitudes = df["attitude_key"].unique()
        unique_individual_attitudes = set(unique_attitudes)
    else:
        # Create individual attitude keys for each agent (legacy 2-agent code)
        df['attitude_agent_1'] = df['alpha_1'].astype(str) + '_' + df['beta_1'].astype(str)
        
        # Get unique individual attitudes
        unique_attitudes = df["attitude_key"].unique()
        unique_individual_attitudes = set()
        
        for attitude in unique_attitudes:
            att_parts = attitude.split('_')
            unique_individual_attitudes.add(f"{att_parts[0]}_{att_parts[1]}")  # agent 1 attitude
    
    print(f"Individual attitudes found: {sorted(unique_individual_attitudes)}")
    
    # Generate plots for individual attitudes
    generate_individual_attitude_plots(df, paths, unique_individual_attitudes, config, individual_trainings)
    
    # Generate combined attitude plots (only meaningful for multi-agent)
    if num_agents > 1:
        generate_combined_attitude_plots(df, paths, unique_attitudes, config, individual_trainings)


def generate_individual_attitude_plots(df, paths, unique_individual_attitudes, config, individual_trainings=False):
    """Generate plots for individual attitudes with averaged metrics."""    

    N = config.smoothing_factor
    unique_lr = df["lr"].unique()
    
    # Check which reward metrics are actually available in the dataframe
    available_rewarded_metrics = []
    potential_metrics = [
        ("delivered_ai_rl_1", "#27AE60", "Delivered"),
        ("cut_ai_rl_1", "#2980B9", "Cut"), 
        ("salad_ai_rl_1", "#E67E22", "Salad"),
        ("deliver_ai_rl_1", "#27AE60", "Deliver"),  # Alternative name
        ("plate_ai_rl_1", "#9B59B6", "Plate"),
        ("raw_food_ai_rl_1", "#E74C3C", "Raw Food"),
        ("counter_ai_rl_1", "#34495E", "Counter")
    ]
    
    for metric_name, color, label in potential_metrics:
        if metric_name in df.columns:
            available_rewarded_metrics.append((metric_name, color, label))
    
    print(f"Available rewarded metrics for plotting: {[m[0] for m in available_rewarded_metrics]}")
    
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
            mask_agent_1 = (df['attitude_agent_1'] == individual_attitude)
            filtered_subset = df[mask_agent_1].copy()

            if len(filtered_subset) > 0:
                # Generate individual training plots if requested
                if individual_trainings:
                    generate_individual_training_attitude_plots(
                        filtered_subset, paths, individual_attitude, lr, degree, N, available_rewarded_metrics
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
                
                plt.title(f"Rewarded Metrics (Averaged) - Individual Attitude {individual_attitude} ({degree:.1f}°)\n"
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


def generate_individual_training_attitude_plots(filtered_subset, paths, individual_attitude, lr, degree, N, available_rewarded_metrics):
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
        
        plt.title(f"Training {training_id} - Attitude {individual_attitude} ({degree:.1f}°) (Smoothed {N})\n"
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


def generate_combined_attitude_plots(df, paths, unique_attitudes, config, individual_trainings=False):
    """Generate plots showing metrics averaged over all other agent attitudes."""
    
    N = config.smoothing_factor
    unique_lr = df["lr"].unique()
    
    # Check which reward metrics are actually available in the dataframe
    available_rewarded_metrics = []
    potential_metrics = [
        ("delivered_ai_rl_1", "#27AE60", "Delivered"),
        ("cut_ai_rl_1", "#2980B9", "Cut"), 
        ("salad_ai_rl_1", "#E67E22", "Salad"),
        ("deliver_ai_rl_1", "#27AE60", "Deliver"),  # Alternative name
        ("plate_ai_rl_1", "#9B59B6", "Plate"),
        ("raw_food_ai_rl_1", "#E74C3C", "Raw Food"),
        ("counter_ai_rl_1", "#34495E", "Counter")
    ]
    
    for metric_name, color, label in potential_metrics:
        if metric_name in df.columns:
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
                
                plt.title(f"Rewarded Metrics (Averaged) - Attitude {att1_title}\n"
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
    
    print(f"Starting pretrained experiment analysis...")
    print(f"Map: {args.map_name}")
    print(f"Cluster: {args.cluster}")
    print(f"Smoothing factor: {args.smoothing_factor}")
    print(f"Study name: {args.study_name}")
    print(f"Individual trainings: {individual_trainings}")
    
    try:
        # Run main analysis pipeline
        analysis_results = main_analysis_pipeline(
            experiment_type='pretraining/classic',  # Note: using 'pretraining' to match directory structure
            map_name=args.map_name,
            cluster=args.cluster,
            smoothing_factor=args.smoothing_factor,
            num_agents=1,  # Pretrained experiments use only 1 agent
            study_name=args.study_name
        )
        
        # Add individual_trainings flag to results
        analysis_results['individual_trainings'] = individual_trainings
        
        # Generate pretrained-specific plots
        generate_pretrained_plots(analysis_results)
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()