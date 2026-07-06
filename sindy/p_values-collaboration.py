"""
Script to compute p-values for collaboration measures across different
experimental maps using Welch's T-test based on summary statistics.

Usage examples:
nohup python3 p_values-collaboration.py --map_names baseline_division_of_labor,encouraged_division_of_labor,forced_division_of_labor > log_p_values.out 2>&1 &

[Optional Arguments]
--base_path: Base path for data storage (default: /data/samuel_lozano/cooked/classic/v3.1/experiment)

Author: Samuel Lozano
"""

import os
import glob
import re
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import argparse

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Compute p-values for collaboration measures across maps.")
    parser.add_argument('--map_names', type=str, required=True,
                        help='Comma-separated list of map names (no spaces). E.g. baseline,encouraged,forced')
    parser.add_argument('--base_path', type=str, default='/data/samuel_lozano/cooked/classic/v3.1/experiment',
                        help='Base path for data storage (default: /data/samuel_lozano/cooked/classic/v3.1/experiment)')
    args = parser.parse_args()
    map_names = [m.strip() for m in args.map_names.split(',') if m.strip()]
    if not map_names:
        parser.error('You must provide at least one map name via --map_names')
    return map_names, args.base_path

# --- Helper Function for Data Extraction ---
def _safe_extract(stats_list, key):
    """
    Extracts a list of values for a given key, skipping entries where the 
    value is NaN or the key is missing.
    """
    return [stat[key] for stat in stats_list if not np.isnan(stat.get(key, np.nan))]

# --- Custom T-Test Function from Summary Statistics ---
def welch_ttest_from_stats(mean1, std1, n1, mean2, std2, n2):
    """
    Calculates the T-statistic and two-sided p-value for Welch's T-test 
    using only summary statistics (mean, std, N) for two independent groups.
    
    This function implements the statistical calculation based on the constraint 
    that only summary statistics are available for the groups being compared.
    """
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    
    # Calculate terms used in both T-statistic and degrees of freedom (DF)
    s1_sq_n1 = (std1**2) / n1
    s2_sq_n2 = (std2**2) / n2
    
    # 1. Calculate T-statistic
    t_stat = (mean1 - mean2) / np.sqrt(s1_sq_n1 + s2_sq_n2)
    
    # 2. Calculate Degrees of Freedom (Welch–Satterthwaite equation)
    nu = (s1_sq_n1 + s2_sq_n2)**2 / \
         ((s1_sq_n1**2 / (n1 - 1)) + (s2_sq_n2**2 / (n2 - 1)))
         
    # 3. Calculate Two-Sided P-value
    # t.sf is the survival function (1 - CDF) which gives the p-value for a one-sided test.
    # Multiply by 2 for the two-sided p-value.
    p_val_two_sided = t.sf(np.abs(t_stat), nu) * 2
    
    return t_stat, p_val_two_sided, nu

def get_param_folders(map_name):
    sindy_path = os.path.join(base_path, f'map_{map_name}', 'sindy_analysis')
    if not os.path.isdir(sindy_path):
        return []
    return [f for f in os.listdir(sindy_path) if f.startswith('threshold_')]

# --- Main Execution ---
map_names, base_path = parse_args()

# --- Main Analysis Logic ---
comparison_results = []
coeff_stats = {} # This will store all data across all runs/maps/params

# 0. Find shared parameter folders
param_sets = [set(get_param_folders(m)) for m in map_names]
shared_params = set.intersection(*param_sets)

print(f"Shared parameters across maps: {shared_params}")

# 1. Data Collection Loop
for param in shared_params:
    for map_idx, map_name in enumerate(map_names):
        collab_dir = os.path.join(base_path, f'map_{map_name}', 'sindy_analysis', param)
        json_files = glob.glob(os.path.join(collab_dir, 'collaboration_*.json'))
        
        for jf in json_files:
            match = re.match(r'collaboration_([0-9\-]+)_([0-9\-]+)_([0-9]+)\.json', os.path.basename(jf))
            if not match:
                continue
            
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error reading or parsing {jf}: {e}")
                continue

            for metric, values in data.get('collaboration_measures', {}).items():
                coeff_stats.setdefault((map_idx, metric), []).append({
                    'norm_mean_abs_coeff': values.get('norm_mean_abs_coeff', np.nan),
                    'norm_std_abs_coeff': values.get('norm_std_abs_coeff', np.nan),
                    'n_coeff': values.get('n_coeff', np.nan)
                })

print("Data collection complete. Starting statistical tests...")

# 2. Statistical Testing Loop
metrics = set(k[1] for k in coeff_stats if isinstance(k, tuple))

for metric in metrics:
    # Pairwise t-tests
    for i in range(len(map_names)):
        for j in range(len(map_names)):
            if i == j:
                continue

            stats_i = coeff_stats.get((i, metric), [])
            stats_j = coeff_stats.get((j, metric), [])

            # --- EXTRACT SAMPLES ---
            mean_i_val = _safe_extract(stats_i, 'norm_mean_abs_coeff')[0]
            mean_j_val = _safe_extract(stats_j, 'norm_mean_abs_coeff')[0]
            std_i_val = _safe_extract(stats_i, 'norm_std_abs_coeff')[0]
            std_j_val = _safe_extract(stats_j, 'norm_std_abs_coeff')[0]
            n_i_val = _safe_extract(stats_i, 'n_coeff')[0]
            n_j_val = _safe_extract(stats_j, 'n_coeff')[0]

            # Only compare if mean_i_val > mean_j_val
            if mean_i_val > mean_j_val:
                print(f'Comparing {map_names[i]} > {map_names[j]} for metric {metric}:')
                print(f'   mean_i_val: {mean_i_val}')
                print(f'   mean_j_val : {mean_j_val}')
                print(f'   std_i_val: {std_i_val}')
                print(f'   std_j_val: {std_j_val}')
                print(f'   n_i_val: {n_i_val}')
                print(f'   n_j_val: {n_j_val}')

                # --- PERFORM T-TEST ---
                t_stat, p_value, degs_freedom = welch_ttest_from_stats(
                    mean_i_val, std_i_val, n_i_val,
                    mean_j_val, std_j_val, n_j_val
                )

                # Define 'veredict' based on the one-sided test (Map i > Map j)
                veredict = True if (not np.isnan(p_value) and p_value < 0.005) else False

                comparison_results.append({
                    'param': param,
                    'metric': metric,
                    'comparison': f"{map_names[i]} > {map_names[j]}",
                    'comparison_means': [round(mean_i_val, 4), round(mean_j_val, 4)],
                    'comparison_stddevs': [round(std_i_val, 4), round(std_j_val, 4)],
                    'comparison_n_coeffs_raw': [n_i_val, n_j_val],
                    't_statistic': t_stat,
                    'degrees_of_freedom': degs_freedom,
                    'p_value': p_value,
                    'veredict': veredict,
                })

print("Statistical tests complete. Saving results...")

df = pd.DataFrame(comparison_results)
for param in shared_params:
    param_df = df[df['param'] == param]
    csv_name = f"p_values-{param}.csv"
    csv_path = os.path.join(base_path, csv_name)
    param_df.to_csv(csv_path, index=False)
    print(f"Saved results for {param} to {csv_path}")