#!/bin/bash

# Simple Grid Search Analysis Launcher
# =====================================
# This script runs the simple/quick analysis on your current grid search results

# Configuration
GRIDSEARCH_BASE_DIR="/data/samuel_lozano/cooked/gridsearch"
RESULTS_DIR="/data/samuel_lozano/cooked/gridsearch/dtde_FIRST_gridsearch_2025-12-02_03-13-28"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMPLE_ANALYZER_SCRIPT="$SCRIPT_DIR/simple_analyzer.py"

echo "🚀 Simple Grid Search Analysis"
echo "=============================="
echo "Gridsearch base directory: $GRIDSEARCH_BASE_DIR"
echo "Results output directory: $RESULTS_DIR"
echo "Analyzer script: $SIMPLE_ANALYZER_SCRIPT"
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
if [ ! -f "$SIMPLE_ANALYZER_SCRIPT" ]; then
    echo "❌ Error: Simple analyzer script not found: $SIMPLE_ANALYZER_SCRIPT"
    exit 1
fi

# Run the simple analysis
echo "📊 Running simple analysis..."
python "$SIMPLE_ANALYZER_SCRIPT"

echo ""
echo "📁 Check this file for results:"
echo "   - $RESULTS_DIR/simple_analysis_results.csv"
echo ""
echo "✅ Simple analysis complete!"
