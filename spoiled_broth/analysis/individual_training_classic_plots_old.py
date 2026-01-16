import os
import matplotlib.pyplot as plt
from spoiled_broth.analysis.utils import MetricDefinitions

def generate_individual_basic_metrics_plots(training_df, paths, training_id, lr, attitude_key, smoothing_factor=15):
    """Generate basic metrics plots (total deliveries and pure reward total) for a single training session."""
    N = smoothing_factor
    training_df = training_df.copy()
    training_df["episode_block"] = (training_df["episode"] // N)
    
    att_parts = attitude_key.split('_')
    att1_title = f"{att_parts[0]}_{att_parts[1]}"
    att2_title = f"{att_parts[2]}_{att_parts[3]}"
    
    # Total deliveries plot
    plt.figure(figsize=(10, 6))
    
    if "total_deliveries" in training_df.columns:
        block_means = training_df.groupby("episode_block")["total_deliveries"].mean()
        middle_episodes = training_df.groupby("episode_block")["episode"].median()
        plt.plot(middle_episodes, block_means, color="#27AE60", linewidth=2)
        
        plt.title(f"Total Deliveries - Training {training_id} - LR {lr} (Smoothed {N})\nAttitudes: Agent1={att1_title}, Agent2={att2_title}", fontsize=14)
        plt.xlabel("Episodes", fontsize=12)
        plt.ylabel("Score (deliveries)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        safe_training_id = training_id.replace(':', '_').replace(' ', '_').replace('-', '_')
        filename = f"individual_total_deliveries_{safe_training_id}_lr{str(lr).replace('.', 'p')}_smoothed_{N}.png"
        plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
        plt.close()
    
    # Pure reward total plot
    plt.figure(figsize=(10, 6))
    
    if "pure_reward_total" in training_df.columns:
        block_means = training_df.groupby("episode_block")["pure_reward_total"].mean()
        middle_episodes = training_df.groupby("episode_block")["episode"].median()
        plt.plot(middle_episodes, block_means, color="#3498DB", linewidth=2)
        
        plt.title(f"Pure Reward Total - Training {training_id} - LR {lr} (Smoothed {N})\nAttitudes: Agent1={att1_title}, Agent2={att2_title}", fontsize=14)
        plt.xlabel("Episodes", fontsize=12)
        plt.ylabel("Pure Reward", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        safe_training_id = training_id.replace(':', '_').replace(' ', '_').replace('-', '_')
        filename = f"individual_pure_reward_total_{safe_training_id}_lr{str(lr).replace('.', 'p')}_smoothed_{N}.png"
        plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
        plt.close()


def generate_individual_combined_reward_plots(training_df, paths, training_id, lr, attitude_key, smoothing_factor=15):
    """Generate combined reward plots for a single training session."""
    N = smoothing_factor
    training_df = training_df.copy()
    training_df["episode_block"] = (training_df["episode"] // N)
    
    # Combined pure rewards plot
    plt.figure(figsize=(10, 6))
    
    # Agent 1 pure reward
    if "pure_reward_ai_rl_1" in training_df.columns:
        block_means_1 = training_df.groupby("episode_block")["pure_reward_ai_rl_1"].mean()
        middle_episodes = training_df.groupby("episode_block")["episode"].median()
        plt.plot(middle_episodes, block_means_1, label="Agent 1", color="#2980B9", linewidth=2)
    
    # Agent 2 pure reward
    if "pure_reward_ai_rl_2" in training_df.columns:
        block_means_2 = training_df.groupby("episode_block")["pure_reward_ai_rl_2"].mean()
        middle_episodes = training_df.groupby("episode_block")["episode"].median()
        plt.plot(middle_episodes, block_means_2, label="Agent 2", color="#E74C3C", linewidth=2)
    
    att_parts = attitude_key.split('_')
    att1_title = f"{att_parts[0]}_{att_parts[1]}"
    att2_title = f"{att_parts[2]}_{att_parts[3]}"
    
    plt.title(f"Pure Rewards - Training {training_id} - LR {lr} (Smoothed {N})\nAttitudes: Agent1={att1_title}, Agent2={att2_title}", fontsize=14)
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Pure Reward", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    safe_training_id = training_id.replace(':', '_').replace(' ', '_').replace('-', '_')
    filename = f"individual_pure_rewards_{safe_training_id}_lr{str(lr).replace('.', 'p')}_smoothed_{N}.png"
    plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
    plt.close()
    
    # Combined modified rewards plot
    plt.figure(figsize=(10, 6))
    
    # Agent 1 modified reward
    if "modified_reward_ai_rl_1" in training_df.columns:
        block_means_1 = training_df.groupby("episode_block")["modified_reward_ai_rl_1"].mean()
        plt.plot(middle_episodes, block_means_1, label="Agent 1", color="#2980B9", linewidth=2)
    
    # Agent 2 modified reward
    if "modified_reward_ai_rl_2" in training_df.columns:
        block_means_2 = training_df.groupby("episode_block")["modified_reward_ai_rl_2"].mean()
        plt.plot(middle_episodes, block_means_2, label="Agent 2", color="#E74C3C", linewidth=2)
    
    plt.title(f"Modified Rewards - Training {training_id} - LR {lr} (Smoothed {N})\nAttitudes: Agent1={att1_title}, Agent2={att2_title}", fontsize=14)
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Modified Reward", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"individual_modified_rewards_{safe_training_id}_lr{str(lr).replace('.', 'p')}_smoothed_{N}.png"
    plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
    plt.close()


def generate_individual_combined_delivery_cut_plots(training_df, paths, training_id, lr, attitude_key, smoothing_factor=15):
    """Generate combined delivery and cut plots for a single training session."""
    N = smoothing_factor
    training_df = training_df.copy()
    training_df["episode_block"] = (training_df["episode"] // N)
    middle_episodes = training_df.groupby("episode_block")["episode"].median()
    
    # Combined deliveries and cuts plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    att_parts = attitude_key.split('_')
    att1_title = f"{att_parts[0]}_{att_parts[1]}"
    att2_title = f"{att_parts[2]}_{att_parts[3]}"
    
    # Agent 1 subplot
    if "deliver_ai_rl_1" in training_df.columns and "cut_ai_rl_1" in training_df.columns:
        deliver_means_1 = training_df.groupby("episode_block")["deliver_ai_rl_1"].mean()
        cut_means_1 = training_df.groupby("episode_block")["cut_ai_rl_1"].mean()
        
        ax1.plot(middle_episodes, deliver_means_1, label="Deliveries", color="#27AE60", linewidth=2)
        ax1.plot(middle_episodes, cut_means_1, label="Cuts", color="#2980B9", linewidth=2)
        ax1.set_title(f"Agent 1 ({att1_title}) - Deliveries and Cuts - Training {training_id}", fontsize=12)
        ax1.set_ylabel("Count", fontsize=10)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
    
    # Agent 2 subplot
    if "deliver_ai_rl_2" in training_df.columns and "cut_ai_rl_2" in training_df.columns:
        deliver_means_2 = training_df.groupby("episode_block")["deliver_ai_rl_2"].mean()
        cut_means_2 = training_df.groupby("episode_block")["cut_ai_rl_2"].mean()
        
        ax2.plot(middle_episodes, deliver_means_2, label="Deliveries", color="#27AE60", linewidth=2)
        ax2.plot(middle_episodes, cut_means_2, label="Cuts", color="#2980B9", linewidth=2)
        ax2.set_title(f"Agent 2 ({att2_title}) - Deliveries and Cuts - Training {training_id}", fontsize=12)
        ax2.set_xlabel("Episodes", fontsize=10)
        ax2.set_ylabel("Count", fontsize=10)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Training {training_id} - LR {lr} (Smoothed {N})", fontsize=14)
    plt.tight_layout()
    
    safe_training_id = training_id.replace(':', '_').replace(' ', '_').replace('-', '_')
    filename = f"individual_delivery_cut_{safe_training_id}_lr{str(lr).replace('.', 'p')}_smoothed_{N}.png"
    plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
    plt.close()


def generate_individual_meaningful_actions_combined(training_df, paths, training_id, lr, attitude_key, base_rewarded_metrics, smoothing_factor=15):
    """Generate meaningful actions combined plots for a single training session (deliver, cut, salad, plate, raw_food only)."""
    N = smoothing_factor
    training_df = training_df.copy()
    training_df["episode_block"] = (training_df["episode"] // N)
    
    # Define meaningful actions: deliver, cut, salad, plate, raw_food
    meaningful_actions = ["deliver", "cut", "salad", "plate", "raw_food"]
    
    metric_labels = MetricDefinitions.get_metric_labels()
    metric_colors = MetricDefinitions.get_metric_colors()
    
    def get_metric_info(metric):
        """Get label and color for a metric."""
        label = metric_labels.get(metric, metric.replace('_', ' ').title())
        color = metric_colors.get(metric, "#000000")
        return label, color
    
    att_parts = attitude_key.split('_')
    att1_title = f"{att_parts[0]}_{att_parts[1]}"
    att2_title = f"{att_parts[2]}_{att_parts[3]}"
    
    # Create combined plot for meaningful actions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Agent 1 metrics
    for metric in meaningful_actions:
        metric_col_1 = f"{metric}_ai_rl_1"
        if metric_col_1 in training_df.columns:
            label, color = get_metric_info(metric)
            block_means = training_df.groupby("episode_block")[metric_col_1].mean()
            middle_episodes = training_df.groupby("episode_block")["episode"].median()
            ax1.plot(middle_episodes, block_means, label=label, color=color, linewidth=1.5)
    
    ax1.set_title(f"Agent 1 ({att1_title}) - Meaningful Actions - Training {training_id}", fontsize=12)
    ax1.set_ylabel("Number of times the action was taken", fontsize=10)
    ax1.legend(
        fontsize=9,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.9,
        edgecolor='black'
    )
    ax1.grid(True, alpha=0.3)
    
    # Agent 2 metrics
    for metric in meaningful_actions:
        metric_col_2 = f"{metric}_ai_rl_2"
        if metric_col_2 in training_df.columns:
            label, color = get_metric_info(metric)
            block_means = training_df.groupby("episode_block")[metric_col_2].mean()
            middle_episodes = training_df.groupby("episode_block")["episode"].median()
            ax2.plot(middle_episodes, block_means, label=label, color=color, linewidth=1.5)
    
    ax2.set_title(f"Agent 2 ({att2_title}) - Meaningful Actions - Training {training_id}", fontsize=12)
    ax2.set_xlabel("Episodes", fontsize=10)
    ax2.set_ylabel("Number of times the action was taken", fontsize=10)
    ax2.legend(
        fontsize=9,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.9,
        edgecolor='black'
    )
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"LR {lr} (Smoothed {N})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    safe_training_id = training_id.replace(':', '_').replace(' ', '_').replace('-', '_')
    filename = f"individual_meaningful_actions_{safe_training_id}_lr{str(lr).replace('.', 'p')}_smoothed_{N}.png"
    plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename), dpi=300, bbox_inches='tight')
    plt.close()


def generate_individual_combined_plots(training_df, paths, training_id, lr, attitude_key, rewarded_metrics_1, rewarded_metrics_2, movement_metrics_1, movement_metrics_2, smoothing_factor=15):
    """Generate combined plots for a single training session showing both agents together."""
    N = smoothing_factor
    training_df = training_df.copy()
    training_df["episode_block"] = (training_df["episode"] // N)
    
    metric_labels = MetricDefinitions.get_metric_labels()
    metric_colors = MetricDefinitions.get_metric_colors()
    
    def get_metric_info(metric):
        """Get label and color for a metric."""
        base_metric = '_'.join(metric.split('_')[:-3])
        label = metric_labels.get(base_metric, base_metric.replace('_', ' ').title())
        color = metric_colors.get(base_metric, "#000000")
        return label, color
    
    att_parts = attitude_key.split('_')
    att1_title = f"{att_parts[0]}_{att_parts[1]}"
    att2_title = f"{att_parts[2]}_{att_parts[3]}"
    
    safe_training_id = training_id.replace(':', '_').replace(' ', '_').replace('-', '_')
    
    # Create combined plot for rewarded metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Agent 1 metrics
    for metric in rewarded_metrics_1:
        if metric in training_df.columns:
            label, color = get_metric_info(metric)
            block_means = training_df.groupby("episode_block")[metric].mean()
            middle_episodes = training_df.groupby("episode_block")["episode"].median()
            ax1.plot(middle_episodes, block_means, label=label, color=color)
    
    ax1.set_title(f"Agent 1 ({att1_title}) - Training {training_id}", fontsize=12)
    ax1.set_ylabel("Number of times the action was taken", fontsize=11)
    ax1.legend(fontsize=10, loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')
    ax1.grid(True, alpha=0.3)
    
    # Agent 2 metrics
    for metric in rewarded_metrics_2:
        if metric in training_df.columns:
            label, color = get_metric_info(metric)
            block_means = training_df.groupby("episode_block")[metric].mean()
            middle_episodes = training_df.groupby("episode_block")["episode"].median()
            ax2.plot(middle_episodes, block_means, label=label, color=color)
    
    ax2.set_title(f"Agent 2 ({att2_title}) - Training {training_id}", fontsize=12)
    ax2.set_xlabel("Episodes", fontsize=11)
    ax2.set_ylabel("Number of times the action was taken", fontsize=11)
    ax2.legend(fontsize=10, loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Rewarded Metrics - LR {lr} (Smoothed {N})", fontsize=14)
    plt.tight_layout()
    
    filename = f"individual_rewarded_metrics_{safe_training_id}_lr{str(lr).replace('.', 'p')}_smoothed_{N}.png"
    plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
    plt.close()
    
    # Create combined plot for movement metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Agent 1 metrics
    for metric in movement_metrics_1:
        if metric in training_df.columns:
            label, color = get_metric_info(metric)
            block_means = training_df.groupby("episode_block")[metric].mean()
            middle_episodes = training_df.groupby("episode_block")["episode"].median()
            ax1.plot(middle_episodes, block_means, label=label, color=color)
    
    ax1.set_title(f"Agent 1 ({att1_title}) - Training {training_id}", fontsize=12)
    ax1.set_ylabel("Number of times the action was taken", fontsize=11)
    ax1.legend(
        fontsize=9,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.9,
        edgecolor='black'
    )
    ax1.grid(True, alpha=0.3)
    
    # Agent 2 metrics
    for metric in movement_metrics_2:
        if metric in training_df.columns:
            label, color = get_metric_info(metric)
            block_means = training_df.groupby("episode_block")[metric].mean()
            middle_episodes = training_df.groupby("episode_block")["episode"].median()
            ax2.plot(middle_episodes, block_means, label=label, color=color)
    
    ax2.set_title(f"Agent 2 ({att2_title}) - Training {training_id}", fontsize=12)
    ax2.set_xlabel("Episodes", fontsize=11)
    ax2.set_ylabel("Number of times the action was taken", fontsize=11)
    ax2.legend(
        fontsize=9,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        framealpha=0.9,
        edgecolor='black'
    )
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Movement Metrics - LR {lr} (Smoothed {N})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.99, 1])
    
    filename = f"individual_movement_metrics_{safe_training_id}_lr{str(lr).replace('.', 'p')}_smoothed_{N}.png"
    plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
    plt.close()