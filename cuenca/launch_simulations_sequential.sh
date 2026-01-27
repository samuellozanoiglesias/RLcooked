#!/bin/bash

# =============================================================================
# Simple Sequential Simulation Launcher
# =============================================================================
# Basic script to launch multiple simulations sequentially with the same parameters
#
# Usage: nohup ./launch_simulations_sequential.sh MAP_NR TRAINING_ID CHECKPOINT_NUMBER NUM_SIMULATIONS [OPTIONAL_ARGS] > log_file 2>&1 &
#
# Example:
#   nohup ./launch_simulations_sequential.sh baseline_division_of_labor 2025-10-07_16-37-58 final 100 > log_sequential_launcher.out 2>&1 &
#   nohup ./launch_simulations_sequential.sh baseline_division_of_labor training_001 final 5 --enable-video true --duration 600 > log_sequential_launcher.out 2>&1 &
# =============================================================================

# Check minimum arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 MAP_NR TRAINING_ID CHECKPOINT_NUMBER NUM_SIMULATIONS [OPTIONAL_ARGS]"
    echo ""
    echo "Arguments:"
    echo "  MAP_NR              Map name"
    echo "  TRAINING_ID         Training identifier"
    echo "  CHECKPOINT_NUMBER   Checkpoint number or 'final'"
    echo "  NUM_SIMULATIONS     Number of simulations to run"
    echo "  OPTIONAL_ARGS       Additional arguments passed to experimental_simulation.py"
    echo ""
    echo "Examples:"
    echo "  $0 training_001 50 3"
    echo "  $0 training_001 final 5 --enable-video true --duration 600"
    echo "  $0 myrun 25 2 --cluster local --tick-rate 30"
    exit 1
fi

# Parse arguments
MAP_NR=$1
TRAINING_ID=$2
CHECKPOINT_NUMBER=$3
NUM_SIMULATIONS=$4
shift 4  # Remove first 4 arguments, rest are optional args

# Default simulation parameters
GAME_VERSION="classic"

# Additional arguments (passed as-is to the Python script)
OPTIONAL_ARGS="$@"

# Validate num_simulations is a positive integer
if ! [[ "$NUM_SIMULATIONS" =~ ^[0-9]+$ ]] || [[ "$NUM_SIMULATIONS" -lt 1 ]]; then
    echo "Error: NUM_SIMULATIONS must be a positive integer"
    exit 1
fi

# Create logs directory
LOG_DIR="simulation_logs"
mkdir -p "$LOG_DIR"

# Redirect stdout and stderr of this script to a log file
SCRIPT_LOG_FILE="$LOG_DIR/sequential_launcher_$(date +%Y-%m-%d_%H-%M-%S).log"
exec > >(tee -a "$SCRIPT_LOG_FILE") 2>&1

echo "=================================================="
echo "Launching $NUM_SIMULATIONS simulations sequentially"
echo "=================================================="
echo "Training ID: $TRAINING_ID"
echo "Checkpoint: $CHECKPOINT_NUMBER"
echo "Map: $MAP_NR"
echo "Game Version: $GAME_VERSION"
echo "Optional Args: $OPTIONAL_ARGS"
echo "Log Directory: $LOG_DIR"
echo "This script's output is being logged to: $SCRIPT_LOG_FILE"
echo "To kill all simulations: pkill -f \"experimental_simulation.py\""
echo "=================================================="

# Launch simulations sequentially
for i in $(seq 1 $NUM_SIMULATIONS); do
    SIM_LOG_FILE="$LOG_DIR/sim_${MAP_NR}_${TRAINING_ID}_${CHECKPOINT_NUMBER}_${i}.log"
    
    echo "--------------------------------------------------"
    echo "Running simulation $i/$NUM_SIMULATIONS..."
    
    # Build the command
    CMD="python3 ../experimental_simulation.py $MAP_NR $GAME_VERSION $TRAINING_ID $CHECKPOINT_NUMBER $OPTIONAL_ARGS"
    
    # Launch simulation and wait for it to complete
    bash -c "$CMD" > "$SIM_LOG_FILE" 2>&1
    
    echo "Simulation $i finished."
    echo "Log file: $SIM_LOG_FILE"
done

echo ""
echo "=================================================="
echo "All $NUM_SIMULATIONS simulations have completed."
echo "Individual simulation logs are in: $LOG_DIR/"
echo "This launcher script's log is: $SCRIPT_LOG_FILE"
echo "=================================================="
echo ""
