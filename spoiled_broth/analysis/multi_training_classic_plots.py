import os
import matplotlib.pyplot as plt
from spoiled_broth.analysis.utils import MetricDefinitions

def generate_combined_reward_plots(df, paths, smoothing_factor=15):
    """Generate combined reward plots showing both agents together, grouped by speeds and game_type."""
    N = smoothing_factor
    unique_lr = df["lr"].unique()
    unique_speeds = df["speed_key"].unique() if "speed_key" in df.columns else [None]
    unique_game_types = df["game_type"].unique() if "game_type" in df.columns else ["classic"]
    
    for game_type in unique_game_types:
        game_type_filtered = df[df["game_type"] == game_type] if "game_type" in df.columns else df
        
        for speed_key in unique_speeds:
            speed_filtered = game_type_filtered[game_type_filtered["speed_key"] == speed_key] if speed_key else game_type_filtered
            
            if len(speed_filtered) == 0:
                continue
            
            # Extract speed values for title
            speed_title = ""
            if speed_key and "_" in str(speed_key):
                speed_parts = str(speed_key).split("_")
                if len(speed_parts) >= 4:
                    speed_title = f" - Speeds: W1={speed_parts[0]} C1={speed_parts[1]} W2={speed_parts[2]} C2={speed_parts[3]}"
                elif len(speed_parts) >= 2:
                    speed_title = f" - Speeds: W={speed_parts[0]} C={speed_parts[1]}"
            
            for lr in unique_lr:
                lr_filtered = speed_filtered[speed_filtered["lr"] == lr]
                
                if len(lr_filtered) == 0:
                    continue
                    
                lr_filtered = lr_filtered.copy()
                lr_filtered["episode_block"] = (lr_filtered["episode"] // N)
                
                # Combined pure rewards plot
                plt.figure(figsize=(10, 6))
                
                # Agent 1 pure reward
                # Note: This groupby averages N consecutive episodes for smoothing (not NUM_ENVS averaging)
                block_means_1 = lr_filtered.groupby("episode_block")["pure_reward_ai_rl_1"].mean()
                middle_episodes = lr_filtered.groupby("episode_block")["episode"].median()
                plt.plot(middle_episodes, block_means_1, label="Agent 1", color="#2980B9", linewidth=2)
                
                # Agent 2 pure reward
                block_means_2 = lr_filtered.groupby("episode_block")["pure_reward_ai_rl_2"].mean()
                plt.plot(middle_episodes, block_means_2, label="Agent 2", color="#E74C3C", linewidth=2)
                
                plt.title(f"Pure Rewards - {game_type} - LR {lr}{speed_title} (Smoothed {N})", fontsize=16)
                plt.xlabel("Episodes", fontsize=14)
                plt.ylabel("Pure Reward", fontsize=14)
                plt.legend(fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                speed_suffix = f"_speed_{str(speed_key).replace('.', 'p')}" if speed_key else ""
                filename = f"pure_rewards_combined_{game_type}_lr{str(lr).replace('.', 'p')}{speed_suffix}_smoothed_{N}.png"
                plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
                plt.close()
                
                # Combined modified rewards plot
                plt.figure(figsize=(10, 6))
                
                # Agent 1 modified reward
                block_means_1 = lr_filtered.groupby("episode_block")["modified_reward_ai_rl_1"].mean()
                plt.plot(middle_episodes, block_means_1, label="Agent 1", color="#2980B9", linewidth=2)
                
                # Agent 2 modified reward
                block_means_2 = lr_filtered.groupby("episode_block")["modified_reward_ai_rl_2"].mean()
                plt.plot(middle_episodes, block_means_2, label="Agent 2", color="#E74C3C", linewidth=2)
                
                plt.title(f"Modified Rewards - {game_type} - LR {lr}{speed_title} (Smoothed {N})", fontsize=16)
                plt.xlabel("Episodes", fontsize=14)
                plt.ylabel("Modified Reward", fontsize=14)
                plt.legend(fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                filename = f"modified_rewards_combined_{game_type}_lr{str(lr).replace('.', 'p')}{speed_suffix}_smoothed_{N}.png"
                plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
                plt.close()
    
    print("Combined reward plots generated successfully.")


def generate_combined_delivery_cut_plots(df, paths, smoothing_factor=15):
    """Generate combined delivery and cut plots for both agents, grouped by speeds and game_type."""
    N = smoothing_factor
    unique_lr = df["lr"].unique()
    unique_speeds = df["speed_key"].unique() if "speed_key" in df.columns else [None]
    unique_game_types = df["game_type"].unique() if "game_type" in df.columns else ["classic"]
    
    for game_type in unique_game_types:
        game_type_filtered = df[df["game_type"] == game_type] if "game_type" in df.columns else df
        
        for speed_key in unique_speeds:
            speed_filtered = game_type_filtered[game_type_filtered["speed_key"] == speed_key] if speed_key else game_type_filtered
            
            if len(speed_filtered) == 0:
                continue
            
            # Extract speed values for title
            speed_title = ""
            if speed_key and "_" in str(speed_key):
                speed_parts = str(speed_key).split("_")
                if len(speed_parts) >= 4:
                    speed_title = f" - Speeds: W1={speed_parts[0]} C1={speed_parts[1]} W2={speed_parts[2]} C2={speed_parts[3]}"
                elif len(speed_parts) >= 2:
                    speed_title = f" - Speeds: W={speed_parts[0]} C={speed_parts[1]}"
            
            for lr in unique_lr:
                lr_filtered = speed_filtered[speed_filtered["lr"] == lr]
                
                if len(lr_filtered) == 0:
                    continue
                    
                lr_filtered = lr_filtered.copy()
                lr_filtered["episode_block"] = (lr_filtered["episode"] // N)
                middle_episodes = lr_filtered.groupby("episode_block")["episode"].median()
                
                # Combined deliveries and cuts plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Agent 1 subplot
                deliver_means_1 = lr_filtered.groupby("episode_block")["deliver_ai_rl_1"].mean()
                cut_means_1 = lr_filtered.groupby("episode_block")["cut_ai_rl_1"].mean()
                
                ax1.plot(middle_episodes, deliver_means_1, label="Deliveries", color="#27AE60", linewidth=2)
                ax1.plot(middle_episodes, cut_means_1, label="Cuts", color="#2980B9", linewidth=2)
                ax1.set_title(f"Agent 1 - Deliveries and Cuts - {game_type} - LR {lr}{speed_title}", fontsize=14)
                ax1.set_ylabel("Count", fontsize=12)
                ax1.legend(fontsize=12)
                ax1.grid(True, alpha=0.3)
                
                # Agent 2 subplot
                deliver_means_2 = lr_filtered.groupby("episode_block")["deliver_ai_rl_2"].mean()
                cut_means_2 = lr_filtered.groupby("episode_block")["cut_ai_rl_2"].mean()
                
                ax2.plot(middle_episodes, deliver_means_2, label="Deliveries", color="#27AE60", linewidth=2)
                ax2.plot(middle_episodes, cut_means_2, label="Cuts", color="#2980B9", linewidth=2)
                ax2.set_title(f"Agent 2 - Deliveries and Cuts - {game_type} - LR {lr}{speed_title}", fontsize=14)
                ax2.set_xlabel("Episodes", fontsize=12)
                ax2.set_ylabel("Count", fontsize=12)
                ax2.legend(fontsize=12)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                speed_suffix = f"_speed_{str(speed_key).replace('.', 'p')}" if speed_key else ""
                filename = f"delivery_cut_combined_{game_type}_lr{str(lr).replace('.', 'p')}{speed_suffix}_smoothed_{N}.png"
                plt.savefig(os.path.join(paths['smoothed_figures_dir'], filename))
                plt.close()
    
    print("Combined delivery and cut plots generated successfully.")


def generate_meaningful_actions_combined(df, paths, base_rewarded_metrics, base_movement_metrics, smoothing_factor=15):
    """Generate meaningful actions combined plots for both agents (deliver, cut, salad, plate, raw_food only), grouped by speeds and game_type."""
    N = smoothing_factor
    unique_attitudes = df["attitude_key"].unique()
    unique_lr = df["lr"].unique()
    unique_speeds = df["speed_key"].unique() if "speed_key" in df.columns else [None]
    unique_game_types = df["game_type"].unique() if "game_type" in df.columns else ["classic"]
    
    # Define meaningful actions: deliver, cut, salad, plate, raw_food
    meaningful_actions = ["deliver", "cut", "salad", "plate", "raw_food"]
    
    metric_labels = MetricDefinitions.get_metric_labels()
    metric_colors = MetricDefinitions.get_metric_colors()
    
    def get_metric_info(metric):
        """Get label and color for a metric."""
        label = metric_labels.get(metric, metric.replace('_', ' ').title())
        color = metric_colors.get(metric, "#000000")
        return label, color
    
    # Combined meaningful actions plots by attitude, speed, and game_type
    for game_type in unique_game_types:
        game_type_filtered = df[df["game_type"] == game_type] if "game_type" in df.columns else df
        
        for speed_key in unique_speeds:
            speed_filtered = game_type_filtered[game_type_filtered["speed_key"] == speed_key] if speed_key else game_type_filtered
            
            if len(speed_filtered) == 0:
                continue
            
            # Extract speed values for title
            speed_title = ""
            speed_suffix = ""
            if speed_key and "_" in str(speed_key):
                speed_parts = str(speed_key).split("_")
                if len(speed_parts) >= 4:
                    speed_title = f" - Speeds: W1={speed_parts[0]} C1={speed_parts[1]} W2={speed_parts[2]} C2={speed_parts[3]}"
                elif len(speed_parts) >= 2:
                    speed_title = f" - Speeds: W={speed_parts[0]} C={speed_parts[1]}"
                speed_suffix = f"_speed_{str(speed_key).replace('.', 'p')}"
            
            for attitude in unique_attitudes:
                subset = speed_filtered[speed_filtered["attitude_key"] == attitude]
                att_parts = attitude.split('_')
                att1_title = f"{att_parts[0]}_{att_parts[1]}"
                att2_title = f"{att_parts[2]}_{att_parts[3]}"
                
                for lr in unique_lr:
                    filtered_subset = subset[subset["lr"] == lr]
                    
                    if len(filtered_subset) == 0:
                        continue
                        
                    filtered_subset = filtered_subset.copy()
                    filtered_subset["episode_block"] = (filtered_subset["episode"] // N)
                    
                    # Create combined plot for meaningful actions
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
                    
                    # Agent 1 metrics
                    for metric in meaningful_actions:
                        metric_col_1 = f"{metric}_ai_rl_1"
                        if metric_col_1 in filtered_subset.columns:
                            label, color = get_metric_info(metric)
                            block_means = filtered_subset.groupby("episode_block")[metric_col_1].mean()
                            middle_episodes = filtered_subset.groupby("episode_block")["episode"].median()
                            ax1.plot(middle_episodes, block_means, label=label, color=color, linewidth=1.5)
                    
                    ax1.set_title(f"Agent 1 - Meaningful Actions - {game_type} - Attitude {att1_title}{speed_title}", fontsize=14)
                    ax1.set_ylabel("Number of times the action was taken", fontsize=12)
                    ax1.legend(
                        fontsize=10,
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
                        if metric_col_2 in filtered_subset.columns:
                            label, color = get_metric_info(metric)
                            block_means = filtered_subset.groupby("episode_block")[metric_col_2].mean()
                            middle_episodes = filtered_subset.groupby("episode_block")["episode"].median()
                            ax2.plot(middle_episodes, block_means, label=label, color=color, linewidth=1.5)
                    
                    ax2.set_title(f"Agent 2 - Meaningful Actions - {game_type} - Attitude {att2_title}{speed_title}", fontsize=14)
                    ax2.set_xlabel("Episodes", fontsize=12)
                    ax2.set_ylabel("Number of times the action was taken", fontsize=12)
                    ax2.legend(
                        fontsize=10,
                        loc='center left',
                        bbox_to_anchor=(1.02, 0.5),
                        frameon=True,
                        framealpha=0.9,
                        edgecolor='black'
                    )
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout(rect=[0, 0, 0.85, 1])
                    
                    sanitized_attitude = attitude.replace('.', 'p')
                    filename_combined = f"meaningful_actions_combined_{game_type}_lr{str(lr).replace('.', 'p')}_attitude_{sanitized_attitude}{speed_suffix}_smoothed_{N}.png"
                    filepath_combined = os.path.join(paths['smoothed_figures_dir'], filename_combined)
                    plt.savefig(filepath_combined, dpi=300, bbox_inches='tight')
                    plt.close()
    
    print("Meaningful actions combined plots generated successfully.")


def generate_combined_plots(df, paths, rewarded_metrics_1, rewarded_metrics_2, 
                          movement_metrics_1, movement_metrics_2, smoothing_factor=15):
    """Generate combined plots showing both agents together, grouped by speeds and game_type."""

    
    N = smoothing_factor  # smoothing factor
    unique_attitudes = df["attitude_key"].unique()
    unique_lr = df["lr"].unique()
    unique_speeds = df["speed_key"].unique() if "speed_key" in df.columns else [None]
    unique_game_types = df["game_type"].unique() if "game_type" in df.columns else ["classic"]
    
    metric_labels = MetricDefinitions.get_metric_labels()
    metric_colors = MetricDefinitions.get_metric_colors()
    
    def get_metric_info(metric):
        """Get label and color for a metric."""
        base_metric = '_'.join(metric.split('_')[:-3])
        label = metric_labels.get(base_metric, base_metric.replace('_', ' ').title())
        color = metric_colors.get(base_metric, "#000000")
        return label, color
    
    # Combined rewarded and movement metrics plots by attitude, speed, and game_type
    for game_type in unique_game_types:
        game_type_filtered = df[df["game_type"] == game_type] if "game_type" in df.columns else df
        
        for speed_key in unique_speeds:
            speed_filtered = game_type_filtered[game_type_filtered["speed_key"] == speed_key] if speed_key else game_type_filtered
            
            if len(speed_filtered) == 0:
                continue
            
            # Extract speed values for title
            speed_title = ""
            speed_suffix = ""
            if speed_key and "_" in str(speed_key):
                speed_parts = str(speed_key).split("_")
                if len(speed_parts) >= 4:
                    speed_title = f" - Speeds: W1={speed_parts[0]} C1={speed_parts[1]} W2={speed_parts[2]} C2={speed_parts[3]}"
                elif len(speed_parts) >= 2:
                    speed_title = f" - Speeds: W={speed_parts[0]} C={speed_parts[1]}"
                speed_suffix = f"_speed_{str(speed_key).replace('.', 'p')}"
            
            for attitude in unique_attitudes:
                subset = speed_filtered[speed_filtered["attitude_key"] == attitude]
                att_parts = attitude.split('_')
                att1_title = f"{att_parts[0]}_{att_parts[1]}"
                att2_title = f"{att_parts[2]}_{att_parts[3]}"
                
                for lr in unique_lr:
                    filtered_subset = subset[subset["lr"] == lr]
                    
                    if len(filtered_subset) == 0:
                        continue
                        
                    filtered_subset = filtered_subset.copy()
                    filtered_subset["episode_block"] = (filtered_subset["episode"] // N)
                    
                    # Create combined plot for rewarded metrics
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Agent 1 metrics
                    for metric in rewarded_metrics_1:
                        label, color = get_metric_info(metric)
                        block_means = filtered_subset.groupby("episode_block")[metric].mean()
                        middle_episodes = filtered_subset.groupby("episode_block")["episode"].median()
                        ax1.plot(middle_episodes, block_means, label=label, color=color)
                    
                    ax1.set_title(f"Agent 1 - {game_type} - Attitude {att1_title}{speed_title}")
                    ax1.set_ylabel("Number of times the action was taken", fontsize=18)
                    ax1.legend(fontsize=20, loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')
                    
                    # Agent 2 metrics
                    for metric in rewarded_metrics_2:
                        label, color = get_metric_info(metric)
                        block_means = filtered_subset.groupby("episode_block")[metric].mean()
                        middle_episodes = filtered_subset.groupby("episode_block")["episode"].median()
                        ax2.plot(middle_episodes, block_means, label=label, color=color)
                    
                    ax2.set_title(f"Agent 2 - {game_type} - Attitude {att2_title}{speed_title}")
                    ax2.set_xlabel("Episodes", fontsize=20)
                    ax2.set_ylabel("Number of times the action was taken", fontsize=18)
                    ax2.legend(fontsize=20, loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')
                    
                    plt.tight_layout()
                    
                    sanitized_attitude = attitude.replace('.', 'p')
                    filename_combined = f"rewarded_metrics_combined_avg_{game_type}_lr{str(lr).replace('.', 'p')}_attitude_{sanitized_attitude}{speed_suffix}_smoothed_{N}.png"
                    filepath_combined = os.path.join(paths['smoothed_figures_dir'], filename_combined)
                    plt.savefig(filepath_combined)
                    plt.close()
                    
                    # Create combined plot for movement metrics
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Agent 1 metrics
                    for metric in movement_metrics_1:
                        label, color = get_metric_info(metric)
                        block_means = filtered_subset.groupby("episode_block")[metric].mean()
                        middle_episodes = filtered_subset.groupby("episode_block")["episode"].median()
                        ax1.plot(middle_episodes, block_means, label=label, color=color)
                    
                    ax1.set_title(f"Agent 1 - {game_type} - Attitude {att1_title}{speed_title}")
                    ax1.set_ylabel("Number of times the action was taken", fontsize=18)
                    ax1.legend(
                        fontsize=16,
                        loc='center left',
                        bbox_to_anchor=(1.02, 0.5),
                        frameon=True,
                        framealpha=0.9,
                        edgecolor='black'
                    )
                    
                    # Agent 2 metrics
                    for metric in movement_metrics_2:
                        label, color = get_metric_info(metric)
                        block_means = filtered_subset.groupby("episode_block")[metric].mean()
                        middle_episodes = filtered_subset.groupby("episode_block")["episode"].median()
                        ax2.plot(middle_episodes, block_means, label=label, color=color)
                    
                    ax2.set_title(f"Agent 2 - {game_type} - Attitude {att2_title}{speed_title}")
                    ax2.set_xlabel("Episodes", fontsize=20)
                    ax2.set_ylabel("Number of times the action was taken", fontsize=18)
                    ax2.legend(
                        fontsize=16,
                        loc='center left',
                        bbox_to_anchor=(1.02, 0.5),
                        frameon=True,
                        framealpha=0.9,
                        edgecolor='black'
                    )
                    
                    plt.tight_layout(rect=[0, 0, 0.99, 1])
                    
                    filename_combined = f"action_types_combined_avg_{game_type}_lr{str(lr).replace('.', 'p')}_attitude_{sanitized_attitude}{speed_suffix}_smoothed_{N}.png"
                    filepath_combined = os.path.join(paths['smoothed_figures_dir'], filename_combined)
                    plt.savefig(filepath_combined)
                    plt.close()
    
    print("Combined plots generated successfully.")
