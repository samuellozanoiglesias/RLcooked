#!/usr/bin/env python3
"""
Fast Grid Search Results Macro Analyzer
========================================

Super-fast analysis tool to identify the best hyperparameters from DTDE grid search.
This script quickly scans output files and extracts key performance metrics without
heavy dependencies or complex processing.

Usage:
    python fast_results_analyzer.py <results_directory>
    python fast_results_analyzer.py /data/samuel_lozano/cooked/gridsearch/dtde_FIRST_gridsearch_2025-12-02_03-13-28

Features:
- Lightning-fast parsing of output files
- Top-k best configurations identification
- Hyperparameter impact analysis
- Simple visualizations
- CSV export of results

Author: Fast Analysis Tool
"""

import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict
import csv
import glob
import warnings
warnings.filterwarnings('ignore')

class FastGridSearchAnalyzer:
    """Ultra-fast grid search results analyzer"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.experiment_scan_dir = self.results_dir  # Directory where experiments are located
        self.results = []
        self.best_configs = []
        
    def extract_config_from_dirname(self, dirname: str) -> Dict:
        """Extract hyperparameters from experiment directory name"""
        config = {}
        
        # Parse the structured dirname: exp<N>_lr<LR>_seed<SEED>_batch<BATCH>_<ARCH>_<PENALTY>_<REWARD>
        pattern = r'exp(\d+)_lr([0-9.]+)_seed(\d+)_batch(\d+)_(\w+)_(\w+)_(\w+)'
        match = re.search(pattern, dirname)
        
        if match:
            config['exp_id'] = int(match.group(1))
            config['lr'] = float(match.group(2))
            config['seed'] = int(match.group(3))
            config['batch_size'] = int(match.group(4))
            config['architecture'] = match.group(5)
            config['penalty_config'] = match.group(6)
            config['reward_config'] = match.group(7)
            
        return config
    
    def find_training_stats_csv(self, exp_dir: Path) -> Optional[Path]:
        """Find the training_stats.csv file in the experiment directory structure"""
        # Look for training_stats.csv in subdirectories
        csv_patterns = [
            exp_dir / "**/training_stats.csv",
            exp_dir / "pretraining/**/training_stats.csv",
            exp_dir / "**/Training_*/training_stats.csv"
        ]
        
        for pattern in csv_patterns:
            csv_files = list(exp_dir.glob(str(pattern).replace(str(exp_dir) + "/", "")))
            if csv_files:
                return csv_files[0]  # Return the first found
        
        return None
    
    def extract_metrics_from_training_stats(self, exp_dir: Path) -> Dict:
        """Extract performance metrics from training_stats.csv with proper column handling"""
        metrics = {
            'final_deliveries': None,
            'avg_deliveries': None,
            'max_deliveries': None,
            'final_pure_reward': None,
            'avg_pure_reward': None,
            'max_pure_reward': None,
            'final_modified_reward': None,
            'avg_modified_reward': None,
            'max_modified_reward': None,
            'convergence_episodes': None,
            'convergence_rate': None,
            'total_episodes': 0,
            'training_completed': False,
            'deliveries_std': None,
            'reward_stability': None,
            'final_salad_count': None,
            'avg_salad_count': None,
            'final_cut_count': None,
            'avg_cut_count': None,
            'final_plate_count': None,
            'avg_plate_count': None,
            'efficiency_ratio': None,  # useful actions / total actions
            'useless_action_rate': None,
            'destructive_action_rate': None
        }
        
        try:
            csv_file = self.find_training_stats_csv(exp_dir)
            if not csv_file or not csv_file.exists():
                return metrics
                
            # Read the CSV file with error handling
            df = pd.read_csv(csv_file, on_bad_lines='skip', encoding='utf-8')
            
            if df.empty:
                return metrics
            
            # Convert all metric columns to numeric, coercing errors to NaN
            numeric_cols = [
                'episode', 'deliver_ai_rl_1', 'salad_ai_rl_1', 'cut_ai_rl_1', 'plate_ai_rl_1',
                'pure_reward_ai_rl_1', 'modified_reward_ai_rl_1', 'actions_asked_ai_rl_1',
                'useless_floor_ai_rl_1', 'useless_wall_ai_rl_1', 'useless_counter_ai_rl_1',
                'useful_counter_ai_rl_1', 'destructive_food_dispenser_ai_rl_1',
                'useful_food_dispenser_ai_rl_1', 'useless_cutting_board_ai_rl_1',
                'useful_cutting_board_ai_rl_1', 'destructive_plate_dispenser_ai_rl_1',
                'useful_plate_dispenser_ai_rl_1', 'useless_delivery_ai_rl_1',
                'useful_delivery_ai_rl_1'
            ]
            
            for col in numeric_cols:
                if col in df.columns:
                    original_dtype = df[col].dtype
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            metrics['training_completed'] = True
            
            # Group by episode and average across environments
            if 'episode' in df.columns:
                # Only aggregate columns that exist in the DataFrame
                desired_cols = [
                    # Delivery metrics
                    'deliver_ai_rl_1', 'salad_ai_rl_1', 'cut_ai_rl_1', 'plate_ai_rl_1',
                    # Reward metrics
                    'pure_reward_ai_rl_1', 'modified_reward_ai_rl_1',
                    # Action efficiency metrics
                    'actions_asked_ai_rl_1', 'useless_floor_ai_rl_1', 'useless_wall_ai_rl_1',
                    'useless_counter_ai_rl_1', 'useful_counter_ai_rl_1',
                    'destructive_food_dispenser_ai_rl_1', 'useful_food_dispenser_ai_rl_1',
                    'useless_cutting_board_ai_rl_1', 'useful_cutting_board_ai_rl_1',
                    'destructive_plate_dispenser_ai_rl_1', 'useful_plate_dispenser_ai_rl_1',
                    'useless_delivery_ai_rl_1', 'useful_delivery_ai_rl_1'
                ]
                
                # Only include columns that exist AND are numeric
                agg_dict = {}
                for col in desired_cols:
                    if col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            agg_dict[col] = 'mean'
                        else:
                            print(f"      WARNING: Skipping non-numeric column '{col}' (dtype: {df[col].dtype})")
                
                if not agg_dict:
                    # No columns to aggregate, use original df
                    episode_stats = df
                else:
                    try:
                        episode_stats = df.groupby('episode').agg(agg_dict).reset_index()
                    except Exception as agg_error:
                        print(f"      ERROR: Aggregation failed - {agg_error}")
                        first_col = list(agg_dict.keys())[0]
                        print(f"             {first_col}: {df[first_col].head(5).tolist()}")
                        print(f"             dtype: {df[first_col].dtype}")
                        raise
                
                metrics['total_episodes'] = len(episode_stats)
                
                # Extract delivery metrics
                if 'deliver_ai_rl_1' in episode_stats.columns:
                    deliveries = episode_stats['deliver_ai_rl_1'].fillna(0)
                    
                    metrics['final_deliveries'] = float(deliveries.iloc[-1]) if len(deliveries) > 0 else 0
                    metrics['avg_deliveries'] = float(deliveries.mean())
                    metrics['max_deliveries'] = float(deliveries.max())
                    metrics['deliveries_std'] = float(deliveries.std())
                    
                    # Convergence analysis: find when deliveries reach 80% of max
                    if metrics['max_deliveries'] > 0:
                        convergence_threshold = 0.8 * metrics['max_deliveries']
                        convergence_mask = deliveries >= convergence_threshold
                        if convergence_mask.any():
                            convergence_idx = convergence_mask.idxmax()
                            metrics['convergence_episodes'] = int(episode_stats.iloc[convergence_idx]['episode'])
                            metrics['convergence_rate'] = metrics['convergence_episodes'] / metrics['total_episodes']
                
                # Extract reward metrics
                if 'pure_reward_ai_rl_1' in episode_stats.columns:
                    pure_rewards = episode_stats['pure_reward_ai_rl_1'].fillna(0)
                    
                    metrics['final_pure_reward'] = float(pure_rewards.iloc[-1]) if len(pure_rewards) > 0 else 0
                    metrics['avg_pure_reward'] = float(pure_rewards.mean())
                    metrics['max_pure_reward'] = float(pure_rewards.max())
                    
                    # Reward stability (inverse of coefficient of variation)
                    if pure_rewards.std() > 0 and pure_rewards.mean() != 0:
                        metrics['reward_stability'] = 1.0 / (pure_rewards.std() / abs(pure_rewards.mean()))
                    else:
                        metrics['reward_stability'] = float('inf') if pure_rewards.std() == 0 else 0
                        
                if 'modified_reward_ai_rl_1' in episode_stats.columns:
                    modified_rewards = episode_stats['modified_reward_ai_rl_1'].fillna(0)
                    
                    metrics['final_modified_reward'] = float(modified_rewards.iloc[-1]) if len(modified_rewards) > 0 else 0
                    metrics['avg_modified_reward'] = float(modified_rewards.mean())
                    metrics['max_modified_reward'] = float(modified_rewards.max())
                
                # Extract task-specific metrics
                if 'salad_ai_rl_1' in episode_stats.columns:
                    salads = episode_stats['salad_ai_rl_1'].fillna(0)
                    metrics['final_salad_count'] = float(salads.iloc[-1]) if len(salads) > 0 else 0
                    metrics['avg_salad_count'] = float(salads.mean())
                    
                if 'cut_ai_rl_1' in episode_stats.columns:
                    cuts = episode_stats['cut_ai_rl_1'].fillna(0)
                    metrics['final_cut_count'] = float(cuts.iloc[-1]) if len(cuts) > 0 else 0
                    metrics['avg_cut_count'] = float(cuts.mean())
                    
                if 'plate_ai_rl_1' in episode_stats.columns:
                    plates = episode_stats['plate_ai_rl_1'].fillna(0)
                    metrics['final_plate_count'] = float(plates.iloc[-1]) if len(plates) > 0 else 0
                    metrics['avg_plate_count'] = float(plates.mean())
                
                # Calculate efficiency metrics
                if all(col in episode_stats.columns for col in ['actions_asked_ai_rl_1', 'useful_counter_ai_rl_1', 
                       'useful_food_dispenser_ai_rl_1', 'useful_cutting_board_ai_rl_1', 'useful_plate_dispenser_ai_rl_1', 'useful_delivery_ai_rl_1']):
                    
                    total_actions = episode_stats['actions_asked_ai_rl_1']
                    useful_actions = (episode_stats['useful_counter_ai_rl_1'] + 
                                    episode_stats['useful_food_dispenser_ai_rl_1'] + 
                                    episode_stats['useful_cutting_board_ai_rl_1'] + 
                                    episode_stats['useful_plate_dispenser_ai_rl_1'] + 
                                    episode_stats['useful_delivery_ai_rl_1'])
                    
                    # Efficiency ratio (final episode)
                    if total_actions.iloc[-1] > 0:
                        metrics['efficiency_ratio'] = float(useful_actions.iloc[-1] / total_actions.iloc[-1])
                    else:
                        metrics['efficiency_ratio'] = 0.0
                
                # Calculate useless action rate
                if all(col in episode_stats.columns for col in ['actions_asked_ai_rl_1', 'useless_floor_ai_rl_1',
                       'useless_wall_ai_rl_1', 'useless_counter_ai_rl_1', 'useless_cutting_board_ai_rl_1', 'useless_delivery_ai_rl_1']):
                    
                    total_actions = episode_stats['actions_asked_ai_rl_1']
                    useless_actions = (episode_stats['useless_floor_ai_rl_1'] + 
                                     episode_stats['useless_wall_ai_rl_1'] + 
                                     episode_stats['useless_counter_ai_rl_1'] + 
                                     episode_stats['useless_cutting_board_ai_rl_1'] + 
                                     episode_stats['useless_delivery_ai_rl_1'])
                    
                    # Useless action rate (final episode)
                    if total_actions.iloc[-1] > 0:
                        metrics['useless_action_rate'] = float(useless_actions.iloc[-1] / total_actions.iloc[-1])
                    else:
                        metrics['useless_action_rate'] = 0.0
                
                # Calculate destructive action rate
                if all(col in episode_stats.columns for col in ['actions_asked_ai_rl_1', 'destructive_food_dispenser_ai_rl_1', 'destructive_plate_dispenser_ai_rl_1']):
                    
                    total_actions = episode_stats['actions_asked_ai_rl_1']
                    destructive_actions = (episode_stats['destructive_food_dispenser_ai_rl_1'] + 
                                         episode_stats['destructive_plate_dispenser_ai_rl_1'])
                    
                    # Destructive action rate (final episode)
                    if total_actions.iloc[-1] > 0:
                        metrics['destructive_action_rate'] = float(destructive_actions.iloc[-1] / total_actions.iloc[-1])
                    else:
                        metrics['destructive_action_rate'] = 0.0
            else:
                # Fallback: if no episode column, assume already aggregated
                metrics['total_episodes'] = len(df)
                
                if 'deliver_ai_rl_1' in df.columns:
                    deliveries = df['deliver_ai_rl_1'].fillna(0)
                    metrics['final_deliveries'] = float(deliveries.iloc[-1]) if len(deliveries) > 0 else 0
                    metrics['avg_deliveries'] = float(deliveries.mean())
                    metrics['max_deliveries'] = float(deliveries.max())
                    
        except Exception as e:
            print(f"Warning: Error parsing training stats from {exp_dir}: {e}")
            
        return metrics
    
    def scan_results_directory(self) -> None:
        """Quickly scan all experiment results from directories"""
        print(f"Scanning results in: {self.results_dir}")
        
        # Check if experiment directories are in the current directory or parent directory
        exp_dirs = [d for d in self.results_dir.iterdir() if d.is_dir() and d.name.startswith('exp')]
        
        # If no experiment directories found, try parent directory
        if not exp_dirs:
            parent_dir = self.results_dir.parent
            print(f"No experiment directories found in {self.results_dir}")
            print(f"Checking parent directory: {parent_dir}")
            
            if parent_dir.exists():
                exp_dirs = [d for d in parent_dir.iterdir() if d.is_dir() and d.name.startswith('exp')]
                if exp_dirs:
                    print(f"Found experiment directories in parent directory!")
                    # Update the results_dir to save outputs in the timestamped directory
                    # but scan experiments from the parent directory
                    self.experiment_scan_dir = parent_dir
                else:
                    print(f"No experiment directories found in parent directory either")
            else:
                print(f"Parent directory does not exist")
        else:
            self.experiment_scan_dir = self.results_dir
        
        print(f"Found {len(exp_dirs)} experiment directories")
        
        for exp_dir in exp_dirs:
            try:
                # Extract experiment config from directory name
                config = self.extract_config_from_dirname(exp_dir.name)
                
                if not config:
                    print(f"Could not parse config from: {exp_dir.name}")
                    continue
                    
                # Extract performance metrics from training_stats.csv
                metrics = self.extract_metrics_from_training_stats(exp_dir)
                
                # Combine config and metrics
                result = {**config, **metrics}
                result['exp_dir'] = exp_dir.name
                
                self.results.append(result)
                
            except Exception as e:
                print(f"Error processing {exp_dir}: {e}")
                
        print(f"Successfully parsed {len(self.results)} experiments")
    
    def analyze_results(self, top_k: int = 5) -> None:
        """Fast analysis of results focusing on delivery and efficiency metrics"""
        if not self.results:
            print("No results to analyze!")
            return
            
        df = pd.DataFrame(self.results)
        
        # Filter out failed experiments
        completed_df = df[df['training_completed'] == True].copy()
        
        if len(completed_df) == 0:
            print("No completed experiments found!")
            print("Showing all available results...")
            completed_df = df.copy()
            
        print(f"\nAnalyzing {len(completed_df)} completed experiments...")
        
        # Focus on key performance metrics
        performance_metrics = [
            ('final_deliveries', 'Final Deliveries'),
            ('avg_deliveries', 'Average Deliveries'),
            ('max_deliveries', 'Maximum Deliveries'),
            ('convergence_rate', 'Convergence Rate'),
            ('efficiency_ratio', 'Action Efficiency'),
            ('final_pure_reward', 'Final Pure Reward'),
            ('final_modified_reward', 'Final Modified Reward'),
            ('reward_stability', 'Reward Stability'),
            ('useless_action_rate', 'Useless Action Rate'),
            ('destructive_action_rate', 'Destructive Action Rate')
        ]
        
        all_best_configs = {}
        
        for metric_key, metric_name in performance_metrics:
            if metric_key in completed_df.columns and completed_df[metric_key].notna().any():
                print(f"\n🏆 TOP {top_k} by {metric_name.upper()}:")
                
                # Handle metrics where lower is better
                if metric_key in ['convergence_rate', 'useless_action_rate', 'destructive_action_rate']:
                    top_configs = completed_df[completed_df[metric_key].notna()].nsmallest(top_k, metric_key)
                else:
                    top_configs = completed_df[completed_df[metric_key].notna()].nlargest(top_k, metric_key)
                
                all_best_configs[metric_key] = top_configs.to_dict('records')
                
                for i, (_, row) in enumerate(top_configs.iterrows(), 1):
                    value = row[metric_key]
                    if pd.isna(value):
                        continue
                        
                    # Format value based on metric type
                    if metric_key in ['convergence_rate', 'efficiency_ratio', 'useless_action_rate', 'destructive_action_rate']:
                        value_str = f"{value:.4f}"
                    elif metric_key in ['final_deliveries', 'avg_deliveries', 'max_deliveries', 'final_salad_count', 'final_cut_count', 'final_plate_count']:
                        value_str = f"{value:.2f}"
                    elif metric_key == 'convergence_episodes':
                        value_str = f"{int(value)}"
                    else:
                        value_str = f"{value:.4f}"
                        
                    # Add extra context for some metrics
                    extra_info = ""
                    if 'final_deliveries' in row and not pd.isna(row['final_deliveries']):
                        extra_info = f" | Del: {row['final_deliveries']:.2f}"
                    if 'efficiency_ratio' in row and not pd.isna(row['efficiency_ratio']):
                        extra_info += f" | Eff: {row['efficiency_ratio']:.3f}"
                        
                    print(f"  {i}. {metric_name}: {value_str} | "
                          f"LR: {row['lr']:.6f} | Arch: {row['architecture']} | "
                          f"Penalty: {row['penalty_config']} | Reward: {row['reward_config']} | "
                          f"Seed: {row['seed']} | Episodes: {row['total_episodes']}{extra_info}")
        
        # Store best configs for final deliveries as primary metric
        if 'final_deliveries' in all_best_configs:
            self.best_configs = all_best_configs['final_deliveries']
        
        # Quick hyperparameter analysis
        self._analyze_hyperparameters(completed_df)
        
        # Save detailed results
        self._save_results(df, completed_df, all_best_configs)
    
    def _analyze_hyperparameters(self, df: pd.DataFrame) -> None:
        """Quick hyperparameter impact analysis focusing on deliveries"""
        print(f"\n📊 HYPERPARAMETER IMPACT ANALYSIS:")
        
        # Primary metric for analysis
        primary_metric = 'final_deliveries'
        if primary_metric not in df.columns or df[primary_metric].notna().sum() == 0:
            primary_metric = 'final_reward'
            
        print(f"   (Analysis based on {primary_metric})")
        
        # Analyze categorical hyperparameters
        categorical_params = ['architecture', 'penalty_config', 'reward_config']
        numerical_params = ['lr', 'seed', 'batch_size']
        
        for param in categorical_params:
            if param in df.columns and df[param].notna().any():
                grouped = df[df[primary_metric].notna()].groupby(param)[primary_metric].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
                print(f"\n  {param.upper()}:")
                for idx, row in grouped.iterrows():
                    if not pd.isna(row['mean']):
                        print(f"    {idx}: {row['mean']:.4f} ± {row['std']:.4f} (n={row['count']})")
        
        # Analyze numerical hyperparameters
        for param in numerical_params:
            if param in df.columns and df[param].notna().any() and df[primary_metric].notna().any():
                valid_data = df[df[primary_metric].notna() & df[param].notna()]
                if len(valid_data) > 1:
                    corr = valid_data[param].corr(valid_data[primary_metric])
                    if not pd.isna(corr):
                        print(f"\n  {param.upper()}: correlation with {primary_metric} = {corr:.4f}")
    
    def _save_results(self, full_df: pd.DataFrame, completed_df: pd.DataFrame, all_best_configs: Dict) -> None:
        """Save analysis results to files"""
        output_dir = self.results_dir
        
        # Save full results CSV
        full_df.to_csv(output_dir / 'grid_search_results_full.csv', index=False)
        print(f"\n💾 Full results saved to: {output_dir / 'grid_search_results_full.csv'}")
        
        # Save completed results CSV (main analysis file)
        completed_df.to_csv(output_dir / 'grid_search_results_completed.csv', index=False)
        print(f"💾 Completed results saved to: {output_dir / 'grid_search_results_completed.csv'}")
        
        # Save best configs for each metric
        if all_best_configs:
            with open(output_dir / 'best_configs_by_metric.json', 'w') as f:
                json.dump(all_best_configs, f, indent=2, default=str)
            print(f"💾 Best configs by metric saved to: {output_dir / 'best_configs_by_metric.json'}")
        
        # Save legacy best configs JSON for compatibility
        if self.best_configs:
            with open(output_dir / 'best_configs.json', 'w') as f:
                json.dump(self.best_configs, f, indent=2, default=str)
            print(f"💾 Best configs saved to: {output_dir / 'best_configs.json'}")
            
        # Create summary report
        self._create_summary_report(full_df, completed_df, output_dir)
    
    def _create_summary_report(self, full_df: pd.DataFrame, completed_df: pd.DataFrame, output_dir: Path) -> None:
        """Create a text summary report focusing on delivery and efficiency performance"""
        with open(output_dir / 'fast_analysis_summary.txt', 'w') as f:
            f.write("FAST GRID SEARCH ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
            f.write(f"Results Directory: {self.results_dir}\n\n")
            
            f.write(f"EXPERIMENT OVERVIEW:\n")
            f.write(f"- Total experiments: {len(full_df)}\n")
            f.write(f"- Completed experiments: {len(completed_df)}\n")
            f.write(f"- Success rate: {len(completed_df)/len(full_df)*100:.1f}%\n\n")
            
            if not completed_df.empty:
                # Delivery performance summary
                if 'final_deliveries' in completed_df.columns and completed_df['final_deliveries'].notna().any():
                    f.write(f"DELIVERY PERFORMANCE SUMMARY:\n")
                    f.write(f"- Best final deliveries: {completed_df['final_deliveries'].max():.2f}\n")
                    f.write(f"- Average final deliveries: {completed_df['final_deliveries'].mean():.2f}\n")
                    f.write(f"- Std final deliveries: {completed_df['final_deliveries'].std():.2f}\n\n")
                    
                    # Best configuration by deliveries
                    best_config = completed_df.loc[completed_df['final_deliveries'].idxmax()]
                    f.write(f"BEST CONFIGURATION (by final deliveries):\n")
                    f.write(f"- Experiment ID: {best_config['exp_id']}\n")
                    f.write(f"- Learning Rate: {best_config['lr']}\n")
                    f.write(f"- Architecture: {best_config['architecture']}\n")
                    f.write(f"- Penalty Config: {best_config['penalty_config']}\n")
                    f.write(f"- Reward Config: {best_config['reward_config']}\n")
                    f.write(f"- Seed: {best_config['seed']}\n")
                    f.write(f"- Final Deliveries: {best_config['final_deliveries']:.2f}\n")
                    if 'convergence_episodes' in best_config and not pd.isna(best_config['convergence_episodes']):
                        f.write(f"- Convergence Episodes: {int(best_config['convergence_episodes'])}\n")
                    if 'efficiency_ratio' in best_config and not pd.isna(best_config['efficiency_ratio']):
                        f.write(f"- Action Efficiency: {best_config['efficiency_ratio']:.4f}\n")
                    if 'useless_action_rate' in best_config and not pd.isna(best_config['useless_action_rate']):
                        f.write(f"- Useless Action Rate: {best_config['useless_action_rate']:.4f}\n")
                    f.write(f"- Total Episodes: {best_config['total_episodes']}\n\n")
                    
                # Efficiency analysis
                if 'efficiency_ratio' in completed_df.columns and completed_df['efficiency_ratio'].notna().any():
                    f.write(f"ACTION EFFICIENCY SUMMARY:\n")
                    f.write(f"- Best efficiency ratio: {completed_df['efficiency_ratio'].max():.4f}\n")
                    f.write(f"- Average efficiency ratio: {completed_df['efficiency_ratio'].mean():.4f}\n")
                    f.write(f"- Std efficiency ratio: {completed_df['efficiency_ratio'].std():.4f}\n\n")
                
                # Reward performance summary (fallback)
                elif 'final_pure_reward' in completed_df.columns and completed_df['final_pure_reward'].notna().any():
                    f.write(f"REWARD PERFORMANCE SUMMARY:\n")
                    f.write(f"- Best final pure reward: {completed_df['final_pure_reward'].max():.4f}\n")
                    f.write(f"- Average final pure reward: {completed_df['final_pure_reward'].mean():.4f}\n")
                    f.write(f"- Std final pure reward: {completed_df['final_pure_reward'].std():.4f}\n\n")
                    
                    # Best configuration by reward
                    best_config = completed_df.loc[completed_df['final_pure_reward'].idxmax()]
                    f.write(f"BEST CONFIGURATION (by final pure reward):\n")
                    f.write(f"- Experiment ID: {best_config['exp_id']}\n")
                    f.write(f"- Learning Rate: {best_config['lr']}\n")
                    f.write(f"- Architecture: {best_config['architecture']}\n")
                    f.write(f"- Penalty Config: {best_config['penalty_config']}\n")
                    f.write(f"- Reward Config: {best_config['reward_config']}\n")
                    f.write(f"- Seed: {best_config['seed']}\n")
                    f.write(f"- Final Pure Reward: {best_config['final_pure_reward']:.4f}\n\n")
        
        print(f"📄 Summary report saved to: {output_dir / 'fast_analysis_summary.txt'}")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Fast Grid Search Results Analyzer')
    parser.add_argument('results_dir', help='Directory containing grid search results')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top configurations to show')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    print("🚀 Starting fast grid search analysis...")
    print(f"Results directory: {args.results_dir}")
    print(f"Top-k configurations: {args.top_k}")
    print("-" * 50)
    
    # Run analysis
    analyzer = FastGridSearchAnalyzer(args.results_dir)
    analyzer.scan_results_directory()
    analyzer.analyze_results(top_k=args.top_k)
    
    print("\n✅ Fast analysis completed!")


if __name__ == "__main__":
    main()