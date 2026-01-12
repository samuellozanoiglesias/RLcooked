#!/usr/bin/env python3
"""
Simple Grid Search Analyzer
============================

Simplified analyzer that scans the correct directory structure for your grid search results.
This version directly targets the parent directory where experiment folders are located.

Usage:
    python simple_analyzer.py
"""

import os
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

def scan_experiment_directories(base_dir="/data/samuel_lozano/cooked/gridsearch"):
    """Scan for experiment directories in the correct location"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ Base directory not found: {base_dir}")
        return []
    
    # Find all experiment directories
    exp_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('exp')]
    
    print(f"✅ Found {len(exp_dirs)} experiment directories in {base_dir}")
    
    # Show sample directories
    if exp_dirs:
        print("📁 Sample experiment directories:")
        for exp_dir in exp_dirs[:5]:
            print(f"   - {exp_dir.name}")
        if len(exp_dirs) > 5:
            print(f"   ... and {len(exp_dirs) - 5} more")
    
    return exp_dirs

def parse_experiment_name(exp_name):
    """Parse experiment configuration from directory name"""
    import re
    
    # Pattern: exp<N>_lr<LR>_seed<SEED>_batch<BATCH>_<ARCH>_<PENALTY>_<REWARD>
    pattern = r'exp(\d+)_lr([0-9.]+)_seed(\d+)_batch(\d+)_(\w+)_(\w+)_(\w+)'
    match = re.search(pattern, exp_name)
    
    if match:
        return {
            'exp_id': int(match.group(1)),
            'lr': float(match.group(2)),
            'seed': int(match.group(3)),
            'batch_size': int(match.group(4)),
            'architecture': match.group(5),
            'penalty_config': match.group(6),
            'reward_config': match.group(7)
        }
    return {}

def find_training_csv(exp_dir):
    """Find training_stats.csv in experiment directory"""
    csv_patterns = [
        "training_stats.csv",
        "**/training_stats.csv",
        "pretraining/**/training_stats.csv",
        "**/Training_*/training_stats.csv"
    ]
    
    for pattern in csv_patterns:
        csv_files = list(exp_dir.glob(pattern))
        if csv_files:
            return csv_files[0]
    return None

def analyze_training_csv(csv_file):
    """Extract metrics from training_stats.csv"""
    try:
        # Read CSV with error handling for malformed data
        df = pd.read_csv(csv_file, on_bad_lines='skip', encoding='utf-8')
        
        if df.empty:
            return {}
        
        print(f"   📋 CSV shape: {df.shape}")
        print(f"   📋 Columns: {list(df.columns)[:10]}..." if len(df.columns) > 10 else f"   📋 Columns: {list(df.columns)}")
        
        # Convert numeric columns that are read as objects
        numeric_cols = ['deliver_ai_rl_1', 'pure_reward_ai_rl_1', 'modified_reward_ai_rl_1', 
                       'salad_ai_rl_1', 'cut_ai_rl_1', 'actions_asked_ai_rl_1', 'plate_ai_rl_1',
                       'episode']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"   🔧 Converted columns to numeric")
        
        metrics = {
            'total_rows': len(df),
            'csv_found': True
        }
        
        # Check for episode-based data and aggregate
        if 'episode' in df.columns:
            # Only aggregate columns that exist
            agg_dict = {}
            for col in ['deliver_ai_rl_1', 'pure_reward_ai_rl_1', 'modified_reward_ai_rl_1', 
                       'salad_ai_rl_1', 'cut_ai_rl_1', 'actions_asked_ai_rl_1']:
                if col in df.columns:
                    # Check data type before adding
                    dtype = df[col].dtype
                    print(f"   🔍 Column '{col}': dtype={dtype}, sample={df[col].iloc[0] if len(df) > 0 else 'N/A'}")
                    if pd.api.types.is_numeric_dtype(df[col]):
                        agg_dict[col] = 'mean'
                    else:
                        print(f"   ⚠️  Skipping non-numeric column: {col} (dtype: {dtype})")
            
            print(f"   📊 Aggregating {len(agg_dict)} columns: {list(agg_dict.keys())}")
            
            if not agg_dict:
                # No columns to aggregate, skip grouping
                episode_stats = df
            else:
                try:
                    episode_stats = df.groupby('episode').agg(agg_dict).reset_index()
                    print(f"   ✅ Successfully aggregated into {len(episode_stats)} episodes")
                except Exception as agg_error:
                    print(f"   ❌ Aggregation failed: {agg_error}")
                    print(f"   📋 First few rows of problematic columns:")
                    for col in list(agg_dict.keys())[:3]:
                        print(f"      {col}: {df[col].head(3).tolist()}")
                    raise
            
            metrics['total_episodes'] = len(episode_stats)
            
            # Delivery metrics
            if 'deliver_ai_rl_1' in episode_stats.columns:
                deliveries = episode_stats['deliver_ai_rl_1'].fillna(0)
                metrics['final_deliveries'] = float(deliveries.iloc[-1])
                metrics['avg_deliveries'] = float(deliveries.mean())
                metrics['max_deliveries'] = float(deliveries.max())
            
            # Reward metrics
            if 'pure_reward_ai_rl_1' in episode_stats.columns:
                rewards = episode_stats['pure_reward_ai_rl_1'].fillna(0)
                metrics['final_pure_reward'] = float(rewards.iloc[-1])
                metrics['avg_pure_reward'] = float(rewards.mean())
                
        else:
            # No episode grouping needed
            metrics['total_episodes'] = len(df)
            
            if 'deliver_ai_rl_1' in df.columns:
                deliveries = df['deliver_ai_rl_1'].fillna(0)
                metrics['final_deliveries'] = float(deliveries.iloc[-1])
                metrics['avg_deliveries'] = float(deliveries.mean())
                metrics['max_deliveries'] = float(deliveries.max())
        
        return metrics
        
    except Exception as e:
        print(f"   ⚠️ Error analyzing CSV: {e}")
        return {'csv_found': False, 'error': str(e)}

def main():
    """Main analysis function"""
    print("🔬 SIMPLE GRID SEARCH ANALYZER")
    print("=" * 50)
    
    # Scan for experiment directories
    exp_dirs = scan_experiment_directories()
    
    if not exp_dirs:
        print("❌ No experiment directories found")
        return
    
    print(f"\n📊 Analyzing experiments...")
    
    results = []
    successful_analyses = 0
    
    for exp_dir in exp_dirs:
        print(f"\n🗂️ Processing: {exp_dir.name}")
        
        # Parse configuration
        config = parse_experiment_name(exp_dir.name)
        if not config:
            print(f"   ⚠️ Could not parse experiment name")
            continue
        
        # Find training CSV
        csv_file = find_training_csv(exp_dir)
        if not csv_file:
            print(f"   ❌ No training_stats.csv found")
            continue
        
        print(f"   ✅ Found CSV: {csv_file.relative_to(exp_dir)}")
        
        # Analyze CSV
        metrics = analyze_training_csv(csv_file)
        
        if metrics.get('csv_found', False):
            print(f"   📈 Episodes: {metrics.get('total_episodes', 'N/A')}")
            if 'final_deliveries' in metrics:
                print(f"   📦 Final deliveries: {metrics['final_deliveries']:.2f}")
            if 'final_pure_reward' in metrics:
                print(f"   🎯 Final reward: {metrics['final_pure_reward']:.4f}")
            successful_analyses += 1
        
        # Combine config and metrics
        result = {**config, **metrics, 'exp_dir': exp_dir.name}
        results.append(result)
    
    print(f"\n📈 ANALYSIS SUMMARY:")
    print(f"   • Total experiments: {len(exp_dirs)}")
    print(f"   • Successful analyses: {successful_analyses}")
    print(f"   • Success rate: {successful_analyses/len(exp_dirs)*100:.1f}%")
    
    # Save results
    if results:
        output_dir = Path("/data/samuel_lozano/cooked/gridsearch/dtde_FIRST_gridsearch_2025-12-02_03-13-28")
        output_dir.mkdir(exist_ok=True)
        
        # Save to CSV
        df = pd.DataFrame(results)
        csv_file = output_dir / "simple_analysis_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n💾 Results saved to: {csv_file}")
        
        # Show top performers
        completed_df = df[df['csv_found'] == True]
        if not completed_df.empty and 'final_deliveries' in completed_df.columns:
            print(f"\n🏆 TOP 5 BY FINAL DELIVERIES:")
            top_deliveries = completed_df.nlargest(5, 'final_deliveries')
            
            for i, (_, row) in enumerate(top_deliveries.iterrows(), 1):
                print(f"   {i}. {row['final_deliveries']:.2f} deliveries | "
                      f"LR: {row['lr']:.6f} | Arch: {row['architecture']} | "
                      f"Penalty: {row['penalty_config']} | Reward: {row['reward_config']}")
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(exp_dirs),
            'successful_analyses': successful_analyses,
            'success_rate': successful_analyses/len(exp_dirs),
            'output_csv': str(csv_file)
        }
        
        with open(output_dir / "simple_analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Summary saved to: {output_dir / 'simple_analysis_summary.json'}")

if __name__ == "__main__":
    main()