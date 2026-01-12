# Grid Search Analysis Tools

This directory contains powerful tools for analyzing your DTDE grid search results. The tools are designed to extract performance metrics from `training_stats.csv` files and identify the best hyperparameter configurations.

## Overview

The analysis focuses on key performance metrics:
- **Final deliveries**: Number of deliveries in the final episode
- **Average deliveries**: Mean deliveries across all episodes
- **Maximum deliveries**: Peak delivery performance
- **Convergence rate**: How quickly the model reaches good performance
- **Convergence episodes**: Number of episodes needed to reach 80% of max deliveries
- **Reward metrics**: Final, average, and maximum rewards
- **Reward stability**: Consistency of reward performance

## Quick Start

### 1. Check Directory Access
```bash
python check_directory.py
```
This verifies that you can access the results directory and shows the structure.

### 2. Test CSV Analysis (Recommended)
```bash
python test_csv_analysis.py
```
This tests the CSV parsing on a few experiments to ensure everything works.

### 3. Run Full Analysis
```bash
# Option A: Use the launcher script
./run_fast_analysis.sh

# Option B: Run directly
python fast_results_analyzer.py /data/samuel_lozano/cooked/gridsearch/dtde_FIRST_gridsearch_2025-12-02_03-13-28 --top-k 5
```

## Output Files

The analysis generates several output files:

### CSV Files
- `grid_search_results_full.csv`: All experiments (completed and failed)
- `grid_search_results_completed.csv`: Only successfully completed experiments
  - **This is the main file for analysis**
  - Contains all performance metrics and hyperparameters
  - Ready for Excel/Python/R analysis

### JSON Files
- `best_configs_by_metric.json`: Top-k configurations for each performance metric
- `best_configs.json`: Top configurations by primary metric (final deliveries)

### Reports
- `fast_analysis_summary.txt`: Human-readable summary report
- Quick overview of best configurations and performance

## Analysis Features

### Top-K Analysis
For each metric, the tool finds the top-k (default k=5) best performing configurations:
- Final Deliveries (higher is better)
- Average Deliveries (higher is better) 
- Maximum Deliveries (higher is better)
- Convergence Rate (lower is better - faster convergence)
- Final Reward (higher is better)
- Reward Stability (higher is better)

### Hyperparameter Impact Analysis
The tool analyzes how different hyperparameters affect performance:
- **Categorical parameters**: Architecture size, penalty config, reward config
- **Numerical parameters**: Learning rate, seed, batch size
- Shows mean ± std performance for each category
- Correlation analysis for numerical parameters

## Example Usage

```bash
# Basic analysis with default settings
python fast_results_analyzer.py /path/to/results

# Show top 10 configurations for each metric
python fast_results_analyzer.py /path/to/results --top-k 10

# Check current experiment status
python local_analyzer.py

# Quick status check
python quick_results_checker.py
```

## Understanding the Results

### Key Metrics Interpretation

1. **Final Deliveries**: Most important metric - how many deliveries the agent makes in the final episode
2. **Convergence Episodes**: How quickly the agent learns (lower is better)
3. **Reward Stability**: How consistent the performance is (higher is better)

### Best Configuration Selection

The tool identifies the best configurations for each metric. Consider:
- **Final deliveries** for overall performance
- **Convergence rate** for training efficiency
- **Reward stability** for reliability

### Hyperparameter Insights

Look for patterns in the hyperparameter analysis:
- Which architecture sizes perform best?
- Which learning rates converge fastest?
- How do penalty and reward configurations affect performance?

## Troubleshooting

### Common Issues

1. **Directory not found**: Check if the results path is correct
2. **No CSV files found**: Experiments might still be running or failed
3. **Empty results**: Check if experiments have completed successfully

### Solutions

```bash
# Check directory structure
python check_directory.py

# Test CSV parsing
python test_csv_analysis.py

# Check experiment status
python local_analyzer.py
```

## File Structure Expected

The tool expects this directory structure:
```
results_directory/
├── exp1_lr0.0001_seed0_batch4000_small_low_only_final/
│   └── pretraining/classic/map_*/Training_*/training_stats.csv
├── exp2_lr0.0001_seed0_batch4000_small_low_guided/
│   └── pretraining/classic/map_*/Training_*/training_stats.csv
└── ...
```

## Advanced Usage

### Custom Analysis
You can modify `fast_results_analyzer.py` to:
- Add new performance metrics
- Change convergence criteria
- Adjust analysis parameters

### Data Export
The CSV files can be imported into:
- Excel for visualization
- Python/Pandas for custom analysis
- R for statistical analysis
- Any data analysis tool

## Tips for Best Results

1. **Wait for completion**: Let more experiments finish for better statistics
2. **Check data quality**: Use test scripts to verify CSV parsing
3. **Multiple seeds**: Look for consistent patterns across different seeds
4. **Domain knowledge**: Consider your specific problem constraints when interpreting results

## Contact

For issues or questions about the analysis tools, check:
1. Console output for error messages
2. Generated log files
3. Test scripts for debugging