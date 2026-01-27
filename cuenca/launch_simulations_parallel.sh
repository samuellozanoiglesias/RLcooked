#!/bin/bash

# =============================================================================
# Simple Multiple Simulation Launcher with Optional Parallel Limit
# =============================================================================
# Launches multiple simulations, limiting the maximum number of N running concurrently.
# If TRAINING_ID is "all", runs simulations for all Training_* folders in the study.
#
# Usage (Default N=5): ./launch_simulations_parallel.sh MAP_NR TRAINING_ID CHECKPOINT_NUMBER NUM_SIMULATIONS [OPTIONAL_ARGS]
# Usage (Custom N):   ./launch_simulations_parallel.sh MAP_NR TRAINING_ID CHECKPOINT_NUMBER NUM_SIMULATIONS MAX_PARALLEL [OPTIONAL_ARGS]
# Usage (All trainings): ./launch_simulations_parallel.sh MAP_NR all CHECKPOINT_NUMBER NUM_SIMULATIONS [--study_name NAME] [OPTIONAL_ARGS]
#
# Example (N=5 default):
#   nohup ./launch_simulations_parallel.sh baseline_division_of_labor 2025-09-13_13-27-52 final 20 > log_simulation_parallel_baseline.out 2>&1 &
#   nohup ./launch_simulations_parallel.sh baseline_division_of_labor_v2 2025-11-12_13-52-30 final 20 --num_agents 1 --duration 120 > log_simulation_parallel_baseline.out 2>&1 &
#   nohup ./launch_simulations_parallel.sh baseline_division_of_labor 2025-09-13_13-27-52 final 20 --study_name speeds --duration 180 > log_simulation_parallel_baseline_speeds.out 2>&1 &
#   nohup ./launch_simulations_parallel.sh baseline_division_of_labor all final 20 --study_name speeds --game_type classic --duration 300 > log_simulation_parallel_all.out 2>&1 &
#   
# Example (Custom N=3):
#   nohup ./launch_simulations_parallel.sh my_map myrun final 10 3 --enable_video true --duration 240 > log_simulation_parallel_my_map.out 2>&1 &
#   nohup ./launch_simulations_parallel.sh my_map myrun final 10 3 --study_name speeds --enable_video true --duration 180 > log_simulation_parallel_my_map.out 2>&1 &
#   nohup ./launch_simulations_parallel.sh my_map all final 10 3 --game_type classic_collision --study_name collision_study --duration 360 > log_simulation_parallel_my_map.out 2>&1 &
# =============================================================================

# Define the default maximum parallel processes
DEFAULT_MAX_PARALLEL=5

# Check minimum arguments (4 are required)
if [ $# -lt 4 ]; then
    echo "Usage: $0 MAP_NR TRAINING_ID CHECKPOINT_NUMBER NUM_SIMULATIONS [MAX_PARALLEL] [OPTIONAL_ARGS]"
    echo ""
    echo "Arguments:"
    echo "  MAP_NR              Map name"
    echo "  TRAINING_ID         Training identifier or 'all' to run all trainings in study"
    echo "  CHECKPOINT_NUMBER   Checkpoint number or 'final'"
    echo "  NUM_SIMULATIONS     Total number of simulations to run per training"
    echo "  MAX_PARALLEL        Maximum parallel processes (optional, default is $DEFAULT_MAX_PARALLEL)"
    echo "  OPTIONAL_ARGS       Additional arguments passed to experimental_simulation.py"
    echo "                      Common options: --game_type (classic, classic_collision), --study_name, --num_agents, --enable_video, --duration"
    echo ""
    echo "Note: When TRAINING_ID is 'all', --study_name must be specified to locate training folders"
    echo "      If --game_type is not specified, the script looks directly in /data/samuel_lozano/cooked/map_<MAP_NR>/<STUDY_NAME>"
    echo "      If --game_type is specified, the script looks in /data/samuel_lozano/cooked/<GAME_TYPE>/map_<MAP_NR>/<STUDY_NAME>"
    echo ""
    exit 1
fi

# Parse arguments
MAP_NR=$1
TRAINING_ID=$2
CHECKPOINT_NUMBER=$3
NUM_SIMULATIONS=$4

# Check if MAX_PARALLEL is provided (5th argument) AND it is a valid positive integer
if [ $# -ge 5 ]; then
    CANDIDATE_MAX_PARALLEL=$5
    if [[ "$CANDIDATE_MAX_PARALLEL" =~ ^[0-9]+$ ]] && [[ "$CANDIDATE_MAX_PARALLEL" -ge 1 ]]; then
        MAX_PARALLEL=$CANDIDATE_MAX_PARALLEL
        shift 5 # Remove first 5 arguments (4 required + MAX_PARALLEL)
    else
        # If the 5th argument is present but NOT a valid number, assume it's the start of OPTIONAL_ARGS
        # and fall back to the default MAX_PARALLEL.
        MAX_PARALLEL=$DEFAULT_MAX_PARALLEL
        shift 4 # Remove first 4 arguments, leaving $5 and onwards as optional args
        echo "Warning: 5th argument '$CANDIDATE_MAX_PARALLEL' is not a valid positive integer for MAX_PARALLEL. Using default ($DEFAULT_MAX_PARALLEL)."
    fi
else
    # Only 4 arguments provided, use the default MAX_PARALLEL
    MAX_PARALLEL=$DEFAULT_MAX_PARALLEL
    shift 4 # Remove first 4 arguments, leaving nothing for optional args
fi

# Default simulation parameters
GAME_VERSION="classic"

# Additional arguments (passed as-is to the Python script)
OPTIONAL_ARGS="$@"

# Validate NUM_SIMULATIONS is a positive integer
if ! [[ "$NUM_SIMULATIONS" =~ ^[0-9]+$ ]] || [[ "$NUM_SIMULATIONS" -lt 1 ]]; then
    echo "Error: NUM_SIMULATIONS must be a positive integer"
    exit 1
fi
# MAX_PARALLEL is already validated or set to a valid default/provided value

# Extract study_name and game_type from optional args
STUDY_NAME="default"
GAME_TYPE=""  # Empty by default - no game_type subfolder unless specified

# Parse optional args to extract study_name and game_type
# Convert OPTIONAL_ARGS to array for easier parsing
ARGS_ARRAY=($OPTIONAL_ARGS)
for ((i=0; i<${#ARGS_ARRAY[@]}; i++)); do
    arg="${ARGS_ARRAY[$i]}"
    if [[ "$arg" == "--study_name" ]]; then
        next_i=$((i+1))
        STUDY_NAME="${ARGS_ARRAY[$next_i]}"
    elif [[ "$arg" == "--game_type" ]]; then
        next_i=$((i+1))
        GAME_TYPE="${ARGS_ARRAY[$next_i]}"
    fi
done

# If TRAINING_ID is "all", find all Training_* folders in the map
if [[ "$TRAINING_ID" == "all" ]]; then
    # Construct path to search for Training folders
    # Logic: If study_name is provided (via --study_name), look inside that study folder
    #        Otherwise, look directly in the map folder
    
    # First, determine if we have a study_name from the arguments (not just the default)
    STUDY_NAME_PROVIDED=false
    for arg in "${ARGS_ARRAY[@]}"; do
        if [[ "$arg" == "--study_name" ]]; then
            STUDY_NAME_PROVIDED=true
            break
        fi
    done
    
    if [[ -z "$GAME_TYPE" ]]; then
        # No game_type specified - look directly in map folder
        MAP_PATH="/data/samuel_lozano/cooked/map_${MAP_NR}"
    else
        # game_type specified - use game_type subfolder
        MAP_PATH="/data/samuel_lozano/cooked/${GAME_TYPE}/map_${MAP_NR}"
    fi
    
    # If study_name was explicitly provided, look in the study subfolder for Training directories
    if [[ "$STUDY_NAME_PROVIDED" == true ]]; then
        SEARCH_PATH="${MAP_PATH}/${STUDY_NAME}"
    else
        SEARCH_PATH="${MAP_PATH}"
    fi
    
    if [[ ! -d "$SEARCH_PATH" ]]; then
        echo "Error: Search path does not exist: $SEARCH_PATH"
        if [[ "$STUDY_NAME_PROVIDED" == true ]]; then
            echo "Make sure --study_name is specified correctly"
        fi
        if [[ -n "$GAME_TYPE" ]]; then
            echo "and --game_type is specified correctly"
        fi
        exit 1
    fi
    
    # Find all Training_* directories
    TRAINING_DIRS=($(find "$SEARCH_PATH" -maxdepth 1 -type d -name "Training_*" | sort))
    
    if [[ ${#TRAINING_DIRS[@]} -eq 0 ]]; then
        echo "Error: No Training_* folders found in $SEARCH_PATH"
        exit 1
    fi
    
    # Extract just the training IDs (remove path and Training_ prefix)
    TRAINING_IDS=()
    for dir in "${TRAINING_DIRS[@]}"; do
        training_id=$(basename "$dir" | sed 's/^Training_//')
        TRAINING_IDS+=("$training_id")
    done
    
    echo "=================================================="
    echo "Found ${#TRAINING_IDS[@]} training(s) in study '$STUDY_NAME':"
    for tid in "${TRAINING_IDS[@]}"; do
        echo "  - $tid"
    done
    echo "=================================================="
else
    # Single training ID provided
    TRAINING_IDS=("$TRAINING_ID")
fi

# Create logs directory
LOG_DIR="simulation_logs"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "Launching $NUM_SIMULATIONS simulations per training"
echo "Total trainings: ${#TRAINING_IDS[@]}"
echo "Total simulations: $((NUM_SIMULATIONS * ${#TRAINING_IDS[@]}))"
echo "Max Parallel: $MAX_PARALLEL (Default is $DEFAULT_MAX_PARALLEL)"
echo "=================================================="
echo "Checkpoint: $CHECKPOINT_NUMBER"
echo "Map: $MAP_NR"
echo "Study: $STUDY_NAME"
echo "Game Type: $GAME_TYPE"
echo "Optional Args: $OPTIONAL_ARGS"
echo "Log Directory: $LOG_DIR"
echo "To kill all simulations: pkill -f \"experimental_simulation.py\""
echo "================================================="

# Array to store process IDs
PIDS=()
RUNNING_COUNT=0

# Launch simulations for each training
for CURRENT_TRAINING_ID in "${TRAINING_IDS[@]}"; do
    echo ""
    echo "Starting simulations for training: $CURRENT_TRAINING_ID"
    echo "--------------------------------------------------"
    
    # Launch simulations for this training
    for i in $(seq 1 $NUM_SIMULATIONS); do
        # Create log filename - handle empty GAME_TYPE
        if [[ -z "$GAME_TYPE" ]]; then
            LOG_FILE="$LOG_DIR/sim_${STUDY_NAME}_${MAP_NR}_${CURRENT_TRAINING_ID}_${CHECKPOINT_NUMBER}_${i}.log"
        else
            LOG_FILE="$LOG_DIR/sim_${GAME_TYPE}_${STUDY_NAME}_${MAP_NR}_${CURRENT_TRAINING_ID}_${CHECKPOINT_NUMBER}_${i}.log"
        fi
        
        # === CONCURRENCY CONTROL LOGIC ===
        # If the number of currently running jobs ($RUNNING_COUNT) equals MAX_PARALLEL,
        # wait for one to finish before launching the next.
        while [ "$RUNNING_COUNT" -ge "$MAX_PARALLEL" ]; do
            echo "Max parallel limit ($MAX_PARALLEL) reached. Waiting for a job to finish..."
            # Wait for any background job to finish
            wait -n
            # Update the running count by checking the job table for running jobs.
            RUNNING_COUNT=$(jobs -p | wc -l)
            # Note: jobs -p counts jobs *in the current shell's job table*
            echo "A job completed. $RUNNING_COUNT jobs still running. Resuming launch..."
        done
        # ==================================
        
        echo "Starting simulation $i/$NUM_SIMULATIONS for $CURRENT_TRAINING_ID..."
        
        # Build the command
        CMD="python3 ../experimental_simulation.py $MAP_NR $GAME_VERSION $CURRENT_TRAINING_ID $CHECKPOINT_NUMBER $OPTIONAL_ARGS"
        
        # Launch simulation in background
        nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
        PID=$!
        PIDS+=($PID)
        
        echo "  Simulation $i started with PID: $PID"
        echo "  Log file: $LOG_FILE"
        
        # Update the running count
        RUNNING_COUNT=$(jobs -p | wc -l)
        
        sleep 5 # Small delay
    done
done

# === FINAL WAIT ===
echo ""
echo "=================================================="
echo "All simulations have been launched."
echo "Total: $((NUM_SIMULATIONS * ${#TRAINING_IDS[@]})) simulations across ${#TRAINING_IDS[@]} training(s)"
echo "Waiting for all remaining simulations to complete..."
echo "=================================================="

# Wait for ALL remaining background processes to finish
wait

echo ""
echo "=================================================="
echo "All simulations completed successfully."
echo "Process IDs: ${PIDS[*]}"
echo "=================================================="
echo "Log files are in: $LOG_DIR/"