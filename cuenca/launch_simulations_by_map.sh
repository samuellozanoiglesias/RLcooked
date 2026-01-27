#!/bin/bash

# =============================================================================
# Map-Wide Simulation Launcher (Max 5 Training IDs in Parallel)
# =============================================================================
# This script launches a maximum of MAX_PARALLEL_TRAINING_IDS concurrently.
# It calls the 'launch_simulations_sequential.sh' script for each training ID.
#
# Usage: nohup ./launch_simulations_by_map.sh MAP_NR CHECKPOINT_NUMBER NUM_SIMULATIONS [OPTIONAL_ARGS] > map_launcher_log.out 2>&1 &
#
# Example:
#   nohup ./launch_simulations_by_map.sh baseline_division_of_labor final 100 > log_baseline_launcher.out 2>&1 &
# =============================================================================

# --- Configuration ---
BASE_COOKED_DATA_PATH="/data/samuel_lozano/cooked/classic"
# El número máximo de Training IDs que se ejecutarán en paralelo:
MAX_PARALLEL_TRAINING_IDS=5 
SEQUENTIAL_LAUNCHER="./launch_simulations_sequential.sh"
# ---------------------

# --- Main Script Execution ---

# Check minimum arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 MAP_NR CHECKPOINT_NUMBER NUM_SIMULATIONS [OPTIONAL_ARGS]"
    echo ""
    echo "Arguments:"
    echo "  MAP_NR              Map name (e.g., baseline_division_of_labor)"
    echo "  CHECKPOINT_NUMBER   Checkpoint number or 'final'"
    echo "  NUM_SIMULATIONS     Number of simulations to run per training ID"
    echo "  OPTIONAL_ARGS       Additional arguments passed to experimental_simulation.py"
    echo ""
    exit 1
fi

# Check if the required sequential launcher script exists
if [ ! -x "$SEQUENTIAL_LAUNCHER" ]; then
    echo "Error: Required script '$SEQUENTIAL_LAUNCHER' not found or is not executable."
    echo "Please ensure both launch_simulations_by_map.sh and launch_simulations_sequential.sh are in the same directory and have execution permissions (chmod +x)."
    exit 1
fi


# Parse arguments
MAP_NR=$1
CHECKPOINT_NUMBER=$2
NUM_SIMULATIONS=$3
shift 3  # Remove first 3 arguments, rest are optional args

# Additional arguments
OPTIONAL_ARGS="$@"

# Validate num_simulations is a positive integer
if ! [[ "$NUM_SIMULATIONS" =~ ^[0-9]+$ ]] || [[ "$NUM_SIMULATIONS" -lt 1 ]]; then
    echo "Error: NUM_SIMULATIONS must be a positive integer"
    exit 1
fi

# Determine the directory to search for training IDs
MAP_COOP_PATH="${BASE_COOKED_DATA_PATH}/map_${MAP_NR}"

echo "=================================================="
echo "Map-Wide Simulation Launcher Initializing (Controlled Parallelism)"
echo "=================================================="
echo "Target Map: $MAP_NR"
echo "Checkpoint: $CHECKPOINT_NUMBER"
echo "Simulations per Training ID: $NUM_SIMULATIONS (Sequential, via $SEQUENTIAL_LAUNCHER)"
echo "MAX Parallel Training IDs: $MAX_PARALLEL_TRAINING_IDS"
echo "Optional Args: $OPTIONAL_ARGS"
echo "=================================================="

# Check if the search path exists
if [ ! -d "$MAP_COOP_PATH" ]; then
    echo "Error: Base path not found for map '$MAP_NR': $MAP_COOP_PATH"
    exit 1
fi

# Find all training directories
echo "Discovering Training IDs..."
TRAINING_DIRS=$(find "$MAP_COOP_PATH" -maxdepth 1 -type d -name "Training_*" -printf "%f\n")

if [ -z "$TRAINING_DIRS" ]; then
    echo "No 'Training_*' directories found in $MAP_COOP_PATH. Exiting."
    exit 0
fi

# Create logs directory
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="simulation_logs_${MAP_NR}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
echo "All training ID batch logs will be stored in: $LOG_DIR"
echo "=================================================="

# --- Main Iteration Loop (Parallel Launch of Training IDs) ---
for FULL_DIR_NAME in $TRAINING_DIRS; do
    TRAINING_ID="${FULL_DIR_NAME#Training_}"

    if [ "$TRAINING_ID" = "$FULL_DIR_NAME" ] || [ -z "$TRAINING_ID" ]; then
        continue
    fi

    # --- Job Control: Wait if MAX_PARALLEL_TRAINING_IDS are already running ---
    # Checks the number of running background jobs (`jobs -r | wc -l`). If the limit is reached,
    # it sleeps until a slot opens up.
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL_TRAINING_IDS ]; do
        echo "[MAIN] Max parallel training IDs ($MAX_PARALLEL_TRAINING_IDS) reached. Waiting 300s (5 minutes)..."
        # Wait for any job to finish before checking again
        sleep 300
    done
    # ------------------------------------------------------------------
    
    # Define the log file for this specific training ID's entire batch
    TRAINING_LOG_FILE="$LOG_DIR/batch_${TRAINING_ID}_${CHECKPOINT_NUMBER}.log"

    echo "[MAIN] Launching Training ID: $TRAINING_ID in background. Log: $TRAINING_LOG_FILE"
    
    # Launch the sequential launcher script in the background ('&').
    # The output (which includes the sequential run of N simulations) is redirected to its own log file.
    ( 
        "$SEQUENTIAL_LAUNCHER" "$MAP_NR" "$TRAINING_ID" "$CHECKPOINT_NUMBER" "$NUM_SIMULATIONS" "$OPTIONAL_ARGS"
    ) > "$TRAINING_LOG_FILE" 2>&1 &
done

echo ""
echo "=================================================="
echo "All training IDs launched. Waiting for ALL background jobs (Training IDs) to complete..."
# Wait for all background processes to finish
wait
echo "=================================================="
echo "All workloads have completed."
echo "Final batch logs are in: $LOG_DIR/"
echo "=================================================="
echo ""
