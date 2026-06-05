import os
import numpy as np
import matplotlib.pyplot as plt

from spoiled_broth.analysis.individual_training_classic_plots import (
    generate_individual_basic_metrics_plots,
    generate_individual_combined_reward_plots,
    generate_individual_combined_plots
)

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
    
    # Define base metric names (without agent suffix) - competition format with own/other
    base_rewarded_metrics_own = [
        "deliver_own",
        "cut_own", 
        "salad_own",
        "raw_food_own"
    ]
    
    base_rewarded_metrics_other = [
        "deliver_other",
        "cut_other", 
        "salad_other",
        "raw_food_other"
    ]
    
    base_rewarded_metrics_single = [
        "plate",
        "counter"
    ]
    
    base_movement_metrics_own = [
        "salad_assembly_own",
        "destructive_food_dispenser_own",
        "useful_food_dispenser_own",
        "useful_cutting_board_own",
        "useful_delivery_own"
    ]
    
    base_movement_metrics_other = [
        "salad_assembly_other",
        "destructive_food_dispenser_other",
        "useful_food_dispenser_other",
        "useful_cutting_board_other",
        "useful_delivery_other"
    ]
    
    base_movement_metrics_single = [
        "do_nothing",
        "useless_floor",
        "useless_wall",
        "useless_counter",
        "useful_counter",
        "useless_cutting_board",
        "destructive_plate_dispenser",
        "useful_plate_dispenser",
        "useless_delivery",
        "inaccessible_tile",
        "not_available"
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
        
        # Generate agent-specific metrics dynamically
        def get_agent_metrics(base_metrics, agent_num):
            return [f"{metric}_ai_rl_{agent_num}" for metric in base_metrics]
        
        # Combine all metric types for each agent
        all_rewarded_metrics = base_rewarded_metrics_own + base_rewarded_metrics_other + base_rewarded_metrics_single
        all_movement_metrics = base_movement_metrics_own + base_movement_metrics_other + base_movement_metrics_single
        
        rewarded_metrics_1 = get_agent_metrics(all_rewarded_metrics, 1)
        rewarded_metrics_2 = get_agent_metrics(all_rewarded_metrics, 2)
        movement_metrics_1 = get_agent_metrics(all_movement_metrics, 1)
        movement_metrics_2 = get_agent_metrics(all_movement_metrics, 2)
        
        # Generate individual training plots
        generate_individual_basic_metrics_plots(training_df, individual_paths, training_id, lr, attitude_key, smoothing_factor)
        generate_individual_combined_reward_plots(training_df, individual_paths, training_id, lr, attitude_key, smoothing_factor)
        generate_individual_combined_plots(
            training_df,
            individual_paths,
            training_id,
            lr,
            attitude_key,
            rewarded_metrics_1=rewarded_metrics_1,
            rewarded_metrics_2=rewarded_metrics_2,
            movement_metrics_1=movement_metrics_1,
            movement_metrics_2=movement_metrics_2,
            smoothing_factor=smoothing_factor,
        )
    
    print(f"Individual training plots saved to: {individual_smoothed_dir}")


def generate_individual_training_competition_specific_plots(analysis_results):
    """Generate specific plots requested for individual training sessions."""
    df = analysis_results['df']
    paths = analysis_results['paths']
    config = analysis_results.get('config')
    smoothing_factor = config.smoothing_factor if config else 15
    
    print("Generating individual training specific plots...")
    
    # Create specific plots directory
    specific_plots_dir = os.path.join(paths['smoothed_figures_dir'], 'individual_training_competition_specific')
    os.makedirs(specific_plots_dir, exist_ok=True)
    
    # Get unique training sessions
    unique_trainings = df['timestamp'].unique()

    # y_lim should be the maximum across all trainings
    y_lim_pure_rewards_max = max(df["pure_reward_ai_rl_1"].max(), df["pure_reward_ai_rl_2"].max())
    y_lim_pure_rewards_min = min(df["pure_reward_ai_rl_1"].min(), df["pure_reward_ai_rl_2"].min())
    y_lim_modified_rewards_max = max(df["modified_reward_ai_rl_1"].max(), df["modified_reward_ai_rl_2"].max())
    y_lim_modified_rewards_min = min(df["modified_reward_ai_rl_1"].min(), df["modified_reward_ai_rl_2"].min())
    y_lim_deliveries = max(df["deliver_own_ai_rl_1"].max(), df["deliver_other_ai_rl_1"].max(),
                df["deliver_own_ai_rl_2"].max(), df["deliver_other_ai_rl_2"].max())
    
    y_lim_own = max(df["deliver_own_ai_rl_1"].max(), df["deliver_own_ai_rl_2"].max(), df["cut_own_ai_rl_1"].max(), df["cut_own_ai_rl_2"].max(), df["salad_own_ai_rl_1"].max(), df["salad_own_ai_rl_2"].max())
    y_lim_other = max(df["deliver_other_ai_rl_1"].max(), df["deliver_other_ai_rl_2"].max(), df["cut_other_ai_rl_1"].max(), df["cut_other_ai_rl_2"].max(), df["salad_other_ai_rl_1"].max(), df["salad_other_ai_rl_2"].max())
    y_lim_own_other = max(y_lim_own, y_lim_other)

    y_lim_total_deliveries = max(df["deliver_own_ai_rl_1"].max() + df["deliver_other_ai_rl_1"].max(),
                df["deliver_own_ai_rl_2"].max() + df["deliver_other_ai_rl_2"].max())
    y_lim_total_salads = max(df["salad_own_ai_rl_1"].max() + df["salad_other_ai_rl_1"].max(),
                df["salad_own_ai_rl_2"].max() + df["salad_other_ai_rl_2"].max())
    y_lim_total_cuts = max(df["cut_own_ai_rl_1"].max() + df["cut_other_ai_rl_1"].max(),
                df["cut_own_ai_rl_2"].max() + df["cut_other_ai_rl_2"].max())

    for training_id in unique_trainings:
        training_df = df[df['timestamp'] == training_id].copy()
        
        if len(training_df) == 0:
            continue
            
        print(f"  Processing specific plots for training {training_id}")
        
        # Get training metadata for filename
        lr = training_df['lr'].iloc[0]
        attitude_key = training_df['attitude_key'].iloc[0]
        
        # Create episode blocks for smoothing
        training_df["episode_block"] = (training_df["episode"] // smoothing_factor)

        training_df["total_salads_ai_rl_1"] = training_df["salad_own_ai_rl_1"] + training_df["salad_other_ai_rl_1"]
        training_df["total_salads_ai_rl_2"] = training_df["salad_own_ai_rl_2"] + training_df["salad_other_ai_rl_2"]
        training_df['total_cuts_ai_rl_1'] = training_df['cut_own_ai_rl_1'] + training_df['cut_other_ai_rl_1']
        training_df['total_cuts_ai_rl_2'] = training_df['cut_own_ai_rl_2'] + training_df['cut_other_ai_rl_2']
        training_df["total_deliveries_ai_rl_1"] = training_df["deliver_own_ai_rl_1"] + training_df["deliver_other_ai_rl_1"]
        training_df["total_deliveries_ai_rl_2"] = training_df["deliver_own_ai_rl_2"] + training_df["deliver_other_ai_rl_2"]
        training_df["total_deliveries"] = training_df["total_deliveries_ai_rl_1"] + training_df["total_deliveries_ai_rl_2"]
        
        # 1. Deliveries and cuts (sum of all) for each agent - deliveries above, cuts below
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
        
        # Calculate total deliveries and cuts for each agent
        block_means_total_deliveries_1 = training_df.groupby("episode_block")["total_deliveries_ai_rl_1"].mean()
        block_means_total_deliveries_2 = training_df.groupby("episode_block")["total_deliveries_ai_rl_2"].mean()
        block_means_total_salads_1 = training_df.groupby("episode_block")["total_salads_ai_rl_1"].mean()
        block_means_total_salads_2 = training_df.groupby("episode_block")["total_salads_ai_rl_2"].mean()
        block_means_total_cuts_1 = training_df.groupby("episode_block")["total_cuts_ai_rl_1"].mean()
        block_means_total_cuts_2 = training_df.groupby("episode_block")["total_cuts_ai_rl_2"].mean()
        
        middle_episodes = training_df.groupby("episode_block")["episode"].median()
        
        # Deliveries subplot
        ax1.plot(middle_episodes, block_means_total_deliveries_1, label="Agent 1", color="#27AE60", linewidth=2)
        ax1.plot(middle_episodes, block_means_total_deliveries_2, label="Agent 2", color="#2980B9", linewidth=2)
        ax1.set_title(f"Total Deliveries per Agent | Training {training_id} (Smoothed {smoothing_factor})")
        ax1.set_ylabel("Deliveries")
        ax1.set_ylim(0, y_lim_total_deliveries)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Salads subplot
        ax2.plot(middle_episodes, block_means_total_salads_1, label="Agent 1", color="#DA8A00", linewidth=2)
        ax2.plot(middle_episodes, block_means_total_salads_2, label="Agent 2", color="#DFBF3F", linewidth=2)
        ax2.set_title(f"Total Salads per Agent | Training {training_id} (Smoothed {smoothing_factor})")
        ax2.set_ylabel("Salads")
        ax2.set_ylim(0, y_lim_total_salads)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Cuts subplot
        ax3.plot(middle_episodes, block_means_total_cuts_1, label="Agent 1", color="#E74C3C", linewidth=2)
        ax3.plot(middle_episodes, block_means_total_cuts_2, label="Agent 2", color="#8E44AD", linewidth=2)
        ax3.set_title(f"Total Cuts per Agent | Training {training_id} (Smoothed {smoothing_factor})")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Cuts")
        ax3.set_ylim(0, y_lim_total_cuts)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f"Total Deliveries, Salads and Cuts per Agent\nAttitude: {attitude_key} | LR: {lr}", fontsize=14)
        plt.tight_layout()
        
        sanitized_attitude = attitude_key.replace('.', 'p')
        filename = f"deliveries_salads_cuts_combined_{training_id}_{sanitized_attitude}_lr{str(lr).replace('.', 'p')}.png"
        plt.savefig(os.path.join(specific_plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Deliveries own, other, cuts own, other for each agent - owns above, others below
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
        
        block_means_deliver_own_1 = training_df.groupby("episode_block")["deliver_own_ai_rl_1"].mean()
        block_means_deliver_other_1 = training_df.groupby("episode_block")["deliver_other_ai_rl_1"].mean()
        block_means_cut_own_1 = training_df.groupby("episode_block")["cut_own_ai_rl_1"].mean()
        block_means_cut_other_1 = training_df.groupby("episode_block")["cut_other_ai_rl_1"].mean()
        block_means_salad_own_1 = training_df.groupby("episode_block")["salad_own_ai_rl_1"].mean()
        block_means_salad_other_1 = training_df.groupby("episode_block")["salad_other_ai_rl_1"].mean()
        
        block_means_deliver_own_2 = training_df.groupby("episode_block")["deliver_own_ai_rl_2"].mean()
        block_means_deliver_other_2 = training_df.groupby("episode_block")["deliver_other_ai_rl_2"].mean()
        block_means_cut_own_2 = training_df.groupby("episode_block")["cut_own_ai_rl_2"].mean()
        block_means_cut_other_2 = training_df.groupby("episode_block")["cut_other_ai_rl_2"].mean()
        block_means_salad_own_2 = training_df.groupby("episode_block")["salad_own_ai_rl_2"].mean()
        block_means_salad_other_2 = training_df.groupby("episode_block")["salad_other_ai_rl_2"].mean()
        
        # Deliveries
        ax1.plot(middle_episodes, block_means_deliver_own_1, label="Agent 1 Deliver Own", color="#27AE60", linewidth=2)
        ax1.plot(middle_episodes, block_means_deliver_own_2, label="Agent 2 Deliver Own", color="#2980B9", linewidth=2)
        ax1.plot(middle_episodes, block_means_deliver_other_1, label="Agent 1 Deliver Other", color="#58D68D", linewidth=2)
        ax1.plot(middle_episodes, block_means_deliver_other_2, label="Agent 2 Deliver Other", color="#85C1E9", linewidth=2)
        ax1.set_title(f"Delivery actions | Training {training_id} (Smoothed {smoothing_factor})")
        ax1.set_ylabel("Count")
        ax1.set_ylim(0, y_lim_own_other)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Salads
        ax2.plot(middle_episodes, block_means_salad_own_1, label="Agent 1 Salad Own", color="#DA8A00", linewidth=2, linestyle='-.')
        ax2.plot(middle_episodes, block_means_salad_own_2, label="Agent 2 Salad Own", color="#DFBF3F", linewidth=2, linestyle='-.')
        ax2.plot(middle_episodes, block_means_salad_other_1, label="Agent 1 Salad Other", color="#FFB600", linewidth=2, linestyle='-.')
        ax2.plot(middle_episodes, block_means_salad_other_2, label="Agent 2 Salad Other", color="#F9E79F", linewidth=2, linestyle='-.')
        ax2.set_title(f"Salad actions | Training {training_id} (Smoothed {smoothing_factor})")
        ax2.set_ylabel("Count")
        ax2.set_ylim(0, y_lim_own_other)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Cuts
        ax3.plot(middle_episodes, block_means_cut_own_1, label="Agent 1 Cut Own", color="#E74C3C", linewidth=2, linestyle='--')
        ax3.plot(middle_episodes, block_means_cut_own_2, label="Agent 2 Cut Own", color="#8E44AD", linewidth=2, linestyle='--')
        ax3.plot(middle_episodes, block_means_cut_other_1, label="Agent 1 Cut Other", color="#F1948A", linewidth=2, linestyle='--')
        ax3.plot(middle_episodes, block_means_cut_other_2, label="Agent 2 Cut Other", color="#BB8FCE", linewidth=2, linestyle='--')
        ax3.set_title(f"Cut actions | Training {training_id} (Smoothed {smoothing_factor})")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Count")
        ax3.set_ylim(0, y_lim_own_other)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.suptitle(f"Deliveries, Salads and Cuts Actions per Agent\nAttitude: {attitude_key} | LR: {lr}", fontsize=14)
        plt.tight_layout()
        
        filename = f"deliveries_salads_cuts_detailed_{training_id}_{sanitized_attitude}_lr{str(lr).replace('.', 'p')}.png"
        plt.savefig(os.path.join(specific_plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Deliveries own, other, cuts own, other for each agent - owns above, others below
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Own actions subplot (deliveries and cuts)
        ax1.plot(middle_episodes, block_means_deliver_own_1, label="Agent 1 Deliver Own", color="#27AE60", linewidth=2)
        ax1.plot(middle_episodes, block_means_deliver_own_2, label="Agent 2 Deliver Own", color="#2980B9", linewidth=2)
        ax1.plot(middle_episodes, block_means_cut_own_1, label="Agent 1 Cut Own", color="#E74C3C", linewidth=2, linestyle='--')
        ax1.plot(middle_episodes, block_means_cut_own_2, label="Agent 2 Cut Own", color="#8E44AD", linewidth=2, linestyle='--')
        ax1.plot(middle_episodes, block_means_salad_own_1, label="Agent 1 Salad Own", color="#DA8A00", linewidth=2, linestyle='-.')
        ax1.plot(middle_episodes, block_means_salad_own_2, label="Agent 2 Salad Own", color="#DFBF3F", linewidth=2, linestyle='-.')
        ax1.set_title(f"Own Actions (Deliveries, Salads and Cuts) | Training {training_id} (Smoothed {smoothing_factor})")
        ax1.set_ylabel("Count")
        ax1.set_ylim(0, y_lim_own_other)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Other actions subplot (deliveries and cuts)
        ax2.plot(middle_episodes, block_means_deliver_other_1, label="Agent 1 Deliver Other", color="#58D68D", linewidth=2)
        ax2.plot(middle_episodes, block_means_deliver_other_2, label="Agent 2 Deliver Other", color="#85C1E9", linewidth=2)
        ax2.plot(middle_episodes, block_means_cut_other_1, label="Agent 1 Cut Other", color="#F1948A", linewidth=2, linestyle='--')
        ax2.plot(middle_episodes, block_means_cut_other_2, label="Agent 2 Cut Other", color="#BB8FCE", linewidth=2, linestyle='--')
        ax2.plot(middle_episodes, block_means_salad_other_1, label="Agent 1 Salad Other", color="#FFB600", linewidth=2, linestyle='-.')
        ax2.plot(middle_episodes, block_means_salad_other_2, label="Agent 2 Salad Other", color="#F7DC6F", linewidth=2, linestyle='-.')
        ax2.set_title(f"Other Actions (Deliveries, Salads and Cuts) | Training {training_id} (Smoothed {smoothing_factor})")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Count")
        ax2.set_ylim(0, y_lim_own_other)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"Own vs Other Actions per Agent\nAttitude: {attitude_key} | LR: {lr}", fontsize=14)
        plt.tight_layout()
        
        filename = f"own_others_detailed_{training_id}_{sanitized_attitude}_lr{str(lr).replace('.', 'p')}.png"
        plt.savefig(os.path.join(specific_plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Deliveries own and other for each agent (agent 1 above, agent 2 below)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Agent 1 subplot
        ax1.plot(middle_episodes, block_means_deliver_own_1, label="Deliver Own", color="#27AE60", linewidth=2)
        ax1.plot(middle_episodes, block_means_deliver_other_1, label="Deliver Other", color="#58D68D", linewidth=2, linestyle='--')
        ax1.set_title(f"Agent 1 Deliveries | Training {training_id} (Smoothed {smoothing_factor})")
        ax1.set_ylabel("Deliveries")
        ax1.legend()
        ax1.set_ylim(0, y_lim_deliveries)
        ax1.grid(True, alpha=0.3)
        
        # Agent 2 subplot
        ax2.plot(middle_episodes, block_means_deliver_own_2, label="Deliver Own", color="#2980B9", linewidth=2)
        ax2.plot(middle_episodes, block_means_deliver_other_2, label="Deliver Other", color="#85C1E9", linewidth=2, linestyle='--')
        ax2.set_title(f"Agent 2 Deliveries | Training {training_id} (Smoothed {smoothing_factor})")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Deliveries")
        ax2.legend()
        ax2.set_ylim(0, y_lim_deliveries)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"Deliveries Own vs Other per Agent\nAttitude: {attitude_key} | LR: {lr}", fontsize=14)
        plt.tight_layout()
        
        filename = f"deliveries_own_other_{training_id}_{sanitized_attitude}_lr{str(lr).replace('.', 'p')}.png"
        plt.savefig(os.path.join(specific_plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Pure reward above, modified reward below for each agent
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        block_means_pure_1 = training_df.groupby("episode_block")["pure_reward_ai_rl_1"].mean()
        block_means_pure_2 = training_df.groupby("episode_block")["pure_reward_ai_rl_2"].mean()
        block_means_modified_1 = training_df.groupby("episode_block")["modified_reward_ai_rl_1"].mean()
        block_means_modified_2 = training_df.groupby("episode_block")["modified_reward_ai_rl_2"].mean()
        
        # Pure reward subplot
        ax1.plot(middle_episodes, block_means_pure_1, label="Agent 1", color="#27AE60", linewidth=2)
        ax1.plot(middle_episodes, block_means_pure_2, label="Agent 2", color="#2980B9", linewidth=2)
        ax1.set_title(f"Pure Reward per Agent | Training {training_id} (Smoothed {smoothing_factor})")
        ax1.set_ylabel("Pure Reward")
        ax1.set_ylim(y_lim_pure_rewards_min, y_lim_pure_rewards_max)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Modified reward subplot
        ax2.plot(middle_episodes, block_means_modified_1, label="Agent 1", color="#E74C3C", linewidth=2)
        ax2.plot(middle_episodes, block_means_modified_2, label="Agent 2", color="#8E44AD", linewidth=2)
        ax2.set_title(f"Modified Reward per Agent | Training {training_id} (Smoothed {smoothing_factor})")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Modified Reward")
        ax2.set_ylim(y_lim_modified_rewards_min, y_lim_modified_rewards_max)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"Pure vs Modified Rewards\nAttitude: {attitude_key} | LR: {lr}", fontsize=14)
        plt.tight_layout()
        
        filename = f"rewards_comparison_{training_id}_{sanitized_attitude}_lr{str(lr).replace('.', 'p')}.png"
        plt.savefig(os.path.join(specific_plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Individual training specific plots saved to: {specific_plots_dir}")


def generate_multi_training_competition_comparison_plots(analysis_results):
    """Generate comparison plots across multiple training sessions."""
    df = analysis_results['df']
    paths = analysis_results['paths']
    config = analysis_results.get('config')
    smoothing_factor = config.smoothing_factor if config else 15
    
    print("Generating multi-training comparison plots...")
    
    # Create comparison plots directory
    comparison_plots_dir = os.path.join(paths['smoothed_figures_dir'], 'multi_training_competition_comparison')
    os.makedirs(comparison_plots_dir, exist_ok=True)
    
    # Get unique training sessions
    unique_trainings = df['timestamp'].unique()
    
    # 5. Plot with one line for each training showing total deliveries
    plt.figure(figsize=(15, 8))
    
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']
    
    for i, training_id in enumerate(unique_trainings):
        training_df = df[df['timestamp'] == training_id].copy()
        
        if len(training_df) == 0:
            continue

        training_df["total_deliveries_ai_rl_1"] = training_df["deliver_own_ai_rl_1"] + training_df["deliver_other_ai_rl_1"]
        training_df["total_deliveries_ai_rl_2"] = training_df["deliver_own_ai_rl_2"] + training_df["deliver_other_ai_rl_2"]
        training_df["total_deliveries"] = training_df["total_deliveries_ai_rl_1"] + training_df["total_deliveries_ai_rl_2"]
        
            
        # Get attitude for legend
        attitude_key = training_df['attitude_key'].iloc[0]
        
        # Create episode blocks and calculate total deliveries
        training_df["episode_block"] = (training_df["episode"] // smoothing_factor)
        block_means_total = training_df.groupby("episode_block")["total_deliveries"].mean()
        middle_episodes = training_df.groupby("episode_block")["episode"].median()
        
        color = colors[i % len(colors)]
        plt.plot(middle_episodes, block_means_total, label=f"{attitude_key}", color=color, linewidth=2)
    
    plt.title(f"Total Deliveries Comparison Across Trainings (Smoothed {smoothing_factor})")
    plt.xlabel("Episode")
    plt.ylabel("Total Deliveries")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"total_deliveries_comparison_smoothed_{smoothing_factor}.png"
    plt.savefig(os.path.join(comparison_plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Plot with two lines for each training (own and other deliveries)
    plt.figure(figsize=(15, 10))
    
    for i, training_id in enumerate(unique_trainings):
        training_df = df[df['timestamp'] == training_id].copy()
        
        if len(training_df) == 0:
            continue
            
        # Get attitude for legend
        attitude_key = training_df['attitude_key'].iloc[0]
        
        # Create episode blocks and calculate own/other deliveries
        training_df["episode_block"] = (training_df["episode"] // smoothing_factor)
        
        # Total own and other deliveries across both agents
        training_df['total_own'] = training_df['deliver_own_ai_rl_1'] + training_df['deliver_own_ai_rl_2']
        training_df['total_other'] = training_df['deliver_other_ai_rl_1'] + training_df['deliver_other_ai_rl_2']
        
        block_means_own = training_df.groupby("episode_block")["total_own"].mean()
        block_means_other = training_df.groupby("episode_block")["total_other"].mean()
        middle_episodes = training_df.groupby("episode_block")["episode"].median()
        
        color = colors[i % len(colors)]
        plt.plot(middle_episodes, block_means_own, label=f"{attitude_key} (Own)", color=color, linewidth=2, linestyle='-')
        plt.plot(middle_episodes, block_means_other, label=f"{attitude_key} (Other)", color=color, linewidth=2, linestyle='--')
    
    plt.title(f"Own vs Other Deliveries Comparison Across Trainings (Smoothed {smoothing_factor})")
    plt.xlabel("Episode")
    plt.ylabel("Deliveries")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"own_other_deliveries_comparison_smoothed_{smoothing_factor}.png"
    plt.savefig(os.path.join(comparison_plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Multi-training comparison plots saved to: {comparison_plots_dir}")

def generate_individual_attitude_plots(df, paths, unique_individual_attitudes, config):
    """Generate plots for individual attitudes regardless of partner."""    
    N = config.smoothing_factor
    unique_lr = df["lr"].unique()
    
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
            # Filter data where either agent has this attitude
            mask_agent_1 = (df['attitude_agent_1'] == individual_attitude)
            mask_agent_2 = (df['attitude_agent_2'] == individual_attitude) 
            mask_conditions = (df["lr"] == lr)
            
            filtered_subset = df[mask_conditions & (mask_agent_1 | mask_agent_2)].copy()
            
            if len(filtered_subset) > 0:
                filtered_subset["episode_block"] = (filtered_subset["episode"] // N)
                
                # Plot delivery metrics
                plt.figure(figsize=(12, 6))
                
                # Combine metrics from both agents when they have this attitude
                delivery_own_combined = []
                delivery_other_combined = []
                middle_episodes = []
                
                for episode_block in filtered_subset["episode_block"].unique():
                    block_data = filtered_subset[filtered_subset["episode_block"] == episode_block]
                    
                    own_total = 0
                    other_total = 0
                    count = 0
                    
                    for _, row in block_data.iterrows():
                        if row['attitude_agent_1'] == individual_attitude:
                            own_total += row['deliver_own_ai_rl_1']
                            other_total += row['deliver_other_ai_rl_1']
                            count += 1
                        if row['attitude_agent_2'] == individual_attitude:
                            own_total += row['deliver_own_ai_rl_2'] 
                            other_total += row['deliver_other_ai_rl_2']
                            count += 1
                    
                    if count > 0:
                        delivery_own_combined.append(own_total / count)
                        delivery_other_combined.append(other_total / count)
                        middle_episodes.append(block_data["episode"].median())
                
                plt.plot(middle_episodes, delivery_own_combined, label="Delivered Own", color="#A9DFBF")
                plt.plot(middle_episodes, delivery_other_combined, label="Delivered Other", color="#27AE60")
                
                plt.title(f"Delivery Metrics - Individual Attitude {individual_attitude} ({degree:.1f}°)\n"
                         f"LR {lr} (Smoothed {N})")
                plt.xlabel("Episode")
                plt.ylabel("Mean value")
                plt.legend()
                plt.tight_layout()
                
                sanitized_attitude = individual_attitude.replace('.', 'p')
                filename = f"delivery_individual_attitude_{sanitized_attitude}_lr{str(lr).replace('.', 'p')}_smoothed_{N}.png"
                plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
                plt.close()


def generate_combined_attitude_plots(df, paths, unique_attitudes, config):
    """Generate plots showing both agents together for each attitude combination."""
    
    N = config.smoothing_factor
    
    for attitude in unique_attitudes:
        game_filtered = df[df["attitude_key"] == attitude]
        att_parts = attitude.split('_')
        att1_title = f"{att_parts[0]}_{att_parts[1]}" 
        att2_title = f"{att_parts[2]}_{att_parts[3]}"
        
        
        if len(game_filtered) > 0:
            game_filtered = game_filtered.copy()
            game_filtered["episode_block"] = (game_filtered["episode"] // N)
            
            # Plot combined delivery metrics
            plt.figure(figsize=(12, 6))
            
            # Agent 1 metrics
            block_means_own_1 = game_filtered.groupby("episode_block")["deliver_own_ai_rl_1"].mean()
            block_means_other_1 = game_filtered.groupby("episode_block")["deliver_other_ai_rl_1"].mean()
            middle_episodes = game_filtered.groupby("episode_block")["episode"].median()
            
            plt.plot(middle_episodes, block_means_own_1, label=f"Agent 1 Own ({att1_title})", 
                    color="#A9DFBF", linestyle='-')
            plt.plot(middle_episodes, block_means_other_1, label=f"Agent 1 Other ({att1_title})", 
                    color="#27AE60", linestyle='-')
            
            # Agent 2 metrics
            block_means_own_2 = game_filtered.groupby("episode_block")["deliver_own_ai_rl_2"].mean()
            block_means_other_2 = game_filtered.groupby("episode_block")["deliver_other_ai_rl_2"].mean()
            
            plt.plot(middle_episodes, block_means_own_2, label=f"Agent 2 Own ({att2_title})", 
                    color="#AED6F1", linestyle='--')
            plt.plot(middle_episodes, block_means_other_2, label=f"Agent 2 Other ({att2_title})", 
                    color="#2980B9", linestyle='--')
            
            plt.title(f"Combined Delivery Metrics - Attitude {attitude} (Smoothed {N})")
            plt.xlabel("Episode")
            plt.ylabel("Mean deliveries")
            plt.legend()
            plt.tight_layout()
            
            sanitized_attitude = attitude.replace('.', 'p')
            filename = f"combined_delivery_attitude_{sanitized_attitude}_smoothed_{N}.png"
            plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
            plt.close()