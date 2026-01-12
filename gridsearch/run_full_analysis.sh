#!/bin/bash

# Full Grid Search Analysis Launcher
# ===================================
# This script runs the complete/detailed analysis on your current grid search results

# Configuration
GRIDSEARCH_BASE_DIR="/data/samuel_lozano/cooked/gridsearch"
RESULTS_DIR="/data/samuel_lozano/cooked/gridsearch/dtde_FIRST_gridsearch_2025-12-02_03-13-28"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZER_SCRIPT="$SCRIPT_DIR/full_results_analyzer.py"

echo "🚀 Full Grid Search Analysis"
echo "============================"
echo "Gridsearch base directory: $GRIDSEARCH_BASE_DIR"
echo "Results output directory: $RESULTS_DIR"
echo "Analyzer script: $ANALYZER_SCRIPT"
echo ""

# Check if base gridsearch directory exists
if [ ! -d "$GRIDSEARCH_BASE_DIR" ]; then
    echo "❌ Error: Gridsearch base directory not found: $GRIDSEARCH_BASE_DIR"
    exit 1
fi

# Check if results directory exists (create if needed)
if [ ! -d "$RESULTS_DIR" ]; then
    echo "📁 Creating results directory: $RESULTS_DIR"
    mkdir -p "$RESULTS_DIR"
fi

# Check if analyzer script exists
if [ ! -f "$ANALYZER_SCRIPT" ]; then
    echo "❌ Error: Analyzer script not found: $ANALYZER_SCRIPT"
    exit 1
fi

# Make script executable
chmod +x "$ANALYZER_SCRIPT"

# Run the full analysis
echo "📊 Running full analysis..."
python "$ANALYZER_SCRIPT" "$RESULTS_DIR" --top-k 5

echo ""
echo "📁 Check these files for results:"
echo "   - $RESULTS_DIR/grid_search_results_full.csv"
echo "   - $RESULTS_DIR/grid_search_results_completed.csv" 
echo "   - $RESULTS_DIR/best_configs_by_metric.json"
echo "   - $RESULTS_DIR/fast_analysis_summary.txt"
echo ""
echo "✅ Full analysis complete!"
