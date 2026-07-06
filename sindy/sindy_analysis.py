#!/usr/bin/env python3
"""
Detailed SINDy analysis over experiment data in /data/samuel_lozano/cooked/classic/v3.1/experiment/

- Runs SINDy for each (training_id, checkpoint_number)
- Saves equations and analysis to /experiments/map_{map_identifier}/
- Analyzes collaboration/influence via cross-agent terms
- Validates equations (accuracy, RMSE, R²)
- Plots average prediction vs. true and error distributions

Usage:
    nohup python3 sindy_analysis.py <map_identifier> [--base-path /path/to/experiment] [--output-path /path/to/output] > log_sindy_analysis.txt 2>&1 &

Example:
    nohup python3 sindy_analysis.py encouraged_division_of_labor --threshold 1e-05 --poly-degree 4 --fourier-n 2 > log_sindy_analysis.out 2>&1 &

    nohup python3 sindy_analysis.py baseline_division_of_labor > log_sindy_analysis.out 2>&1 &

Requires:
    pip install pysindy numpy pandas matplotlib
"""

import argparse
from pathlib import Path
from spoiled_broth.sindy.extract_equations import prepare_standardized_data, extract_equations_from_standardized
from spoiled_broth.sindy.analyze_collaboration import analyze_collaboration
from spoiled_broth.sindy.plot_and_summarize import plot_and_summarize

def main():
    parser = argparse.ArgumentParser(description='Orchestrate SINDy analysis pipeline')
    parser.add_argument('map_identifier', type=str, help='Map identifier (e.g. baseline_division_of_labor)')
    parser.add_argument('--base-path', type=str, default='/data/samuel_lozano/cooked/classic/v3.1/experiment', help='Base path to experiment folder')
    parser.add_argument('--threshold', type=float, default=1e-4, help='STLSQ threshold')
    parser.add_argument('--poly-degree', type=int, default=2, help='Polynomial library degree')
    parser.add_argument('--fourier-n', type=int, default=0, help='Number of Fourier components to add (0 = none)')
    args = parser.parse_args() 

    # Define paths
    experiment_path = Path(args.base_path) / f"map_{args.map_identifier}" / "simulations"
    coeff_dir = Path(args.base_path) / f"map_{args.map_identifier}" / "sindy_analysis" / f"threshold_{args.threshold}-poly_{args.poly_degree}-fourier_{args.fourier_n}"
    coeff_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.base_path) / f"map_{args.map_identifier}" / "sindy_analysis" / 'standardized_data'
    data_path.mkdir(parents=True, exist_ok=True)
    summary_path = Path(coeff_dir) / f'summary_{args.map_identifier}.json'

    ## Step 0: Data iterator
    print("Step 0: Preparing data...")
    data_path = prepare_standardized_data(
        experiment_path=experiment_path,
        data_path=data_path
    )

    ## Step 1: Extract equations and coefficients
    print("Step 1: Extracting equations and coefficients...")
    extract_equations_from_standardized(
        data_path=data_path,
        coeff_dir=coeff_dir,
        threshold=args.threshold,
        poly_degree=args.poly_degree,
        fourier_n=args.fourier_n
    )

    ## Step 2: Analyze collaboration
    print("Step 2: Analyzing collaboration...")
    analyze_collaboration(map_identifier=args.map_identifier, coeff_dir=coeff_dir)

    ## Step 3: Plot and summarize
    print("Step 3: Plotting and summarizing...")
    plot_and_summarize(
        coeff_dir=coeff_dir,
        data_path=data_path,
        summary_path=summary_path,
    )
    print(f"Pipeline complete. See {summary_path} for results.")

if __name__ == "__main__":
    main()
