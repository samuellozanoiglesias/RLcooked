#!/usr/bin/env python3
"""
Optimize STLSQ sparsity threshold for SINDy by maximizing average R² across simulations.

Usage:
    nohup python3 sindy_optimization.py <map_identifier> [--base-path /path/to/experiment] [--thresholds 1e-5,1e-4,1e-3,1e-2] [--poly-degrees 2] [--fourier-n 0] [--output-path /path/to/output] > log_sindy_optimization.out 2>&1 &

Example:
    nohup python3 sindy_optimization.py baseline_division_of_labor > log_sindy_optimization.out 2>&1 &

    nohup python3 sindy_optimization.py baseline_division_of_labor --thresholds 1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1 --poly-degrees 1,2,3,4,5,6,7 --fourier-n 0,1,2,3,4 > log_sindy_optimization.out 2>&1 &

This script treats each simulation as an independent trajectory (list of trajectories) and fits SINDy using each threshold candidate.
Saves results JSON and a plot of threshold vs score to the output directory.

Requires:
    pip install pysindy numpy pandas matplotlib scikit-learn
"""

import argparse
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import pysindy as ps
import json
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

from spoiled_broth.sindy.sindy_helpers import load_positions, calculate_velocities, compute_scaler, apply_scaler, inverse_scaler
from spoiled_broth.sindy.postprocess_sindy_results import postprocess_results


def load_all_simulations(map_identifier, base_path):
    experiment_base = Path(base_path) / f"map_{map_identifier}" / "simulations"
    sim_pattern = str(experiment_base / "Training_*" / "checkpoint_*" / "simulation_*")
    simulation_dirs = glob.glob(sim_pattern)
    Xs = []
    sim_ids = []
    for sim_dir in sorted(simulation_dirs):
        pos1_path = Path(sim_dir) / "ai_rl_1_positions.csv"
        pos2_path = Path(sim_dir) / "ai_rl_2_positions.csv"
        if not pos1_path.exists() or not pos2_path.exists():
            continue
        pos1 = load_positions(pos1_path)
        pos2 = load_positions(pos2_path)
        vel1 = calculate_velocities(pos1)
        vel2 = calculate_velocities(pos2)
        # include relative features to match the main analysis pipeline
        rel_pos = pos1 - pos2
        rel_vel = vel1 - vel2
        data = np.hstack([pos1, vel1, pos2, vel2, rel_pos, rel_vel])
        Xs.append(data)
        sim_ids.append(Path(sim_dir).name)
    return Xs, sim_ids


def score_threshold(Xs, threshold, poly_degree=2, fourier_n=0):
    """Fit SINDy with given threshold and library params on scaled trajectories Xs and return mean R2 and mean MSE across sims and features.

    Returns: mean_r2 (float or None), mean_mse (float or None), err (None or str)
    """
    if not Xs:
        return None, None, 'no_data'
    X_concat = np.vstack(Xs)
    mu, sigma = compute_scaler(X_concat)
    # compute true derivatives per trajectory and their global scaler (targets)
    Xdots = [np.diff(Xi, axis=0, prepend=Xi[0:1]) for Xi in Xs]
    Xdot_concat = np.vstack(Xdots)
    mu_dot, sigma_dot = compute_scaler(Xdot_concat)

    Xs_scaled = [apply_scaler(Xi, mu, sigma) for Xi in Xs]
    Xdots_scaled = [apply_scaler(Xd, mu_dot, sigma_dot) for Xd in Xdots]
    try:
        # build feature library consistent with main analysis
        poly_lib = ps.PolynomialLibrary(degree=poly_degree)
        if fourier_n and fourier_n > 0:
            fourier_lib = ps.FourierLibrary(n_frequencies=fourier_n)
            feature_library = poly_lib + fourier_lib
        else:
            feature_library = poly_lib
        optimizer = ps.STLSQ(threshold=threshold)
        model = ps.SINDy(optimizer=optimizer, feature_library=feature_library)
        # fit with x_dot (scaled) so predictions are in target-scaled units
        model.fit(Xs_scaled, t=1, x_dot=Xdots_scaled)
        Xs_pred_scaled = [model.predict(Xs) for Xs in Xs_scaled]
    except Exception as e:
        return None, None, str(e)
    # Inverse-scale and compute R2 per-sim per-feature
    r2s = []
    mses = []
    for Xi, Xi_pred_scaled in zip(Xs, Xs_pred_scaled):
        # predicted derivatives (scaled -> original units)
        Xi_pred = inverse_scaler(Xi_pred_scaled, mu_dot, sigma_dot)
        # true derivatives
        Xi_dot = np.diff(Xi, axis=0, prepend=Xi[0:1])
        # align lengths
        min_len = min(Xi_dot.shape[0], Xi_pred.shape[0])
        if min_len < 2:
            continue
        y_true_all = Xi_dot[:min_len]
        y_pred_all = Xi_pred[:min_len]
        for i in range(Xi.shape[1]):
            y_true = y_true_all[:, i]
            y_pred = y_pred_all[:, i]
            # skip constant true arrays for R2 (undefined)
            if np.allclose(y_true, y_true[0]):
                # record MSE but skip R2
                mses.append(float(mean_squared_error(y_true, y_pred)))
                continue
            r2 = r2_score(y_true, y_pred)
            r2s.append(r2)
            mses.append(float(mean_squared_error(y_true, y_pred)))
    if not r2s:
        # still may have MSEs
        mean_r2 = None
    else:
        mean_r2 = float(np.mean(r2s))
    mean_mse = float(np.mean(mses)) if mses else None
    return mean_r2, mean_mse, None


def run_optimization(map_identifier, base_path, thresholds, output_path, poly_degrees=[2], fourier_ns=[0], objective='r2'):
    Xs, sim_ids = load_all_simulations(map_identifier, base_path)
    if not Xs:
        print(f"No simulations found for map {map_identifier} in {base_path}")
        return
    results = []
    out_dir = Path(output_path) / f"map_{map_identifier}" / 'hyperparameter_optimization'
    out_dir.mkdir(parents=True, exist_ok=True)
    # We'll iterate over grid: thresholds x poly_degrees x fourier_ns
    for poly_deg in poly_degrees:
        for fourier_n in fourier_ns:
            scores_for_combo = []
            print(f"Testing poly_degree={poly_deg}, fourier_n={fourier_n} ...")
            for thr in thresholds:
                print(f"  Testing threshold={thr}...")
                mean_r2, mean_mse, err = score_threshold(Xs, thr, poly_degree=poly_deg, fourier_n=fourier_n)
                print(f"   -> mean_r2={mean_r2}, mean_mse={mean_mse}, err={err}")
                results.append({'threshold': thr, 'poly_degree': poly_deg, 'fourier_n': fourier_n, 'mean_r2': mean_r2, 'mean_mse': mean_mse, 'error': err})
                if mean_r2 is not None or mean_mse is not None:
                    # compute scalar score according to objective
                    if objective == 'r2':
                        score_val = mean_r2 if mean_r2 is not None else -np.inf
                    elif objective == 'mse':
                        # smaller mse is better, invert sign so larger is better
                        score_val = -mean_mse if mean_mse is not None else -np.inf
                    else:  # combined (50% R2, 50% MSE)
                        # Normalize R2 to [0,1] via (r2+1)/2 (R2 can be negative)
                        normalized_r2 = ((mean_r2 + 1.0) / 2.0) if mean_r2 is not None else 0.0
                        # Convert MSE to a [0,1] score where smaller MSE -> closer to 1
                        normalized_mse = (1.0 / (1.0 + mean_mse)) if mean_mse is not None else 0.0
                        # 50/50 weighting
                        score_val = 0.5 * normalized_r2 + 0.5 * normalized_mse
                    scores_for_combo.append((thr, score_val))
            # plot thresholds vs score for this combo
            thr_vals = [s for s, _ in scores_for_combo]
            scores_vals = [sc for _, sc in scores_for_combo]
            if thr_vals:
                plt.figure()
                plt.semilogx(thr_vals, scores_vals, marker='o')
                plt.xlabel('STLSQ threshold')
                ylabel = 'Objective score'
                if objective == 'r2':
                    ylabel = 'Mean R2 across sims and features'
                elif objective == 'mse':
                    ylabel = '-Mean MSE (lower MSE is better)'
                elif objective == 'combined':
                    ylabel = f'Combined score (50% R2, 50% MSE)'
                plt.ylabel(ylabel)
                plt.title(f'poly={poly_deg} fourier={fourier_n} map {map_identifier} objective={objective}')
                plt.grid(True)
                plt.savefig(out_dir / f'threshold_vs_score_poly{poly_deg}_fourier{fourier_n}.png')
                plt.close()


    # Save full results
    results_path = out_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Post-process: use postprocess_sindy_results.py for normalization, best config, and plots
    if postprocess_results:
        postprocess_results(str(out_dir), weight_norm_mse=0.5, weight_norm_r2=0.5)
    else:
        print("postprocess_sindy_results.py not found or import failed. Please run it manually on", str(out_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize SINDy STLSQ threshold')
    parser.add_argument('map_identifier', type=str)
    parser.add_argument('--base-path', type=str, default='/data/samuel_lozano/cooked/classic/v3.1/experiment')
    parser.add_argument('--thresholds', type=str, default='1e-6, 1e-5, 1e-4, 1e-3, 1e-2', help='Comma-separated thresholds')
    parser.add_argument('--poly-degrees', type=str, default='1, 2, 3, 4', help='Comma-separated polynomial degrees to try, e.g. 1,2,3')
    parser.add_argument('--fourier-ns', type=str, default='0, 1, 2', help='Comma-separated Fourier n values to try, e.g. 0,1,2')
    parser.add_argument('--objective', type=str, default='combined', choices=['r2', 'mse', 'combined'], help='Optimization objective: maximize r2, minimize mse, or combined')
    parser.add_argument('--output-path', type=str, default='/data/samuel_lozano/cooked/classic/v3.1/experiment', help='Output base path')
    args = parser.parse_args()
    thr_list = [float(x) for x in args.thresholds.split(',')]
    # sensible default grid if user provided single value
    if len(thr_list) == 1:
        thr_list = [1e-6, 1e-4, 1e-2]
    poly_degrees = [int(x) for x in args.poly_degrees.split(',')]
    if len(poly_degrees) == 1:
        poly_degrees = [1, 2, 3]
    fourier_ns = [int(x) for x in args.fourier_ns.split(',')]
    if len(fourier_ns) == 1:
        fourier_ns = [0, 1, 2]
    run_optimization(args.map_identifier, args.base_path, thr_list, args.output_path, poly_degrees=poly_degrees, fourier_ns=fourier_ns, objective=args.objective)
