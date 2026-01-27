#!/bin/bash

# =============================================================================
# Multiple Training Analysis Launcher
# =============================================================================
# Script to launch analysis_simulations.py for all training_ids in a given 
# map_nr and checkpoint_number combination
#
# Usage: ./launch_simulations_analysis.sh MAP_NR CHECKPOINT_NUMBER [OPTIONS]
#
# Example:
#   nohup ./launch_simulations_analysis.sh baseline_division_of_labor final > log_simulations_analysis.out 2>&1 &
#   nohup ./launch_simulations_analysis.sh baseline_division_of_labor 50 --cluster cuenca --game-version classic --num_agents 1 > log_simulations_analysis.out 2>&1 &
# =============================================================================

# Check minimum arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 MAP_NR CHECKPOINT_NUMBER [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  MAP_NR              Map name (e.g., 'baseline_division_of_labor')"
    echo "  CHECKPOINT_NUMBER   Checkpoint number (e.g., 'final', '50', etc.)"
    echo ""
    echo "Options:"
    echo "  --cluster           Cluster name (default: cuenca)"
    echo "  --game-version      Game version (default: classic)"
    echo "  --num_agents        Number of agents (default: 2)"
    echo "  --dry-run           Show what would be executed without running"
    echo "  --parallel          Run analyses in parallel (default: sequential)"
    echo ""
    echo "Examples:"
    echo "  $0 baseline_division_of_labor final"
    echo "  $0 baseline_division_of_labor 50 --cluster cuenca --parallel"
    echo "  $0 baseline_division_of_labor final --dry-run"
    exit 1
fi

# Parse required arguments
MAP_NR=$1
CHECKPOINT_NUMBER=$2
shift 2

# Default parameters
CLUSTER="cuenca"
GAME_VERSION="classic"
NUM_AGENTS=2
DRY_RUN=false
PARALLEL=false

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        --game-version)
            GAME_VERSION="$2"
            shift 2
            ;;
        --num_agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine base directory based on cluster
if [[ "$CLUSTER" == "cuenca" ]]; then
    BASE_CLUSTER_DIR=""
elif [[ "$CLUSTER" == "brigit" ]]; then
    BASE_CLUSTER_DIR="/mnt/lustre/home/samuloza/"
elif [[ "$CLUSTER" == "local" ]]; then
    BASE_CLUSTER_DIR="C:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
else
    echo "Error: Unknown cluster '$CLUSTER'. Use 'cuenca', 'brigit', or 'local'"
    exit 1
fi

# Construct data directory path
if [[ "$NUM_AGENTS" == 1 ]]; then
    DATA_DIR="${BASE_CLUSTER_DIR}/data/samuel_lozano/cooked/pretraining/${GAME_VERSION}/map_${MAP_NR}"
else
    DATA_DIR="${BASE_CLUSTER_DIR}/data/samuel_lozano/cooked/${GAME_VERSION}/map_${MAP_NR}"
fi

# Check if data directory exists
if [[ ! -d "$DATA_DIR" ]]; then
    echo "Error: Data directory does not exist: $DATA_DIR"
    echo "Please verify the following parameters:"
    echo "  Cluster: $CLUSTER"
    echo "  Game version: $GAME_VERSION"
    echo "  Map: $MAP_NR"
    exit 1
fi

echo "=================================================="
echo "Training Analysis Launcher"
echo "=================================================="
echo "Map: $MAP_NR"
echo "Checkpoint: $CHECKPOINT_NUMBER"
echo "Cluster: $CLUSTER"
echo "Game version: $GAME_VERSION"
echo "Data directory: $DATA_DIR"
echo "Parallel execution: $PARALLEL"
echo "Dry run: $DRY_RUN"
echo "=================================================="

# Find all training directories
echo "Searching for training directories in: $DATA_DIR"
TRAINING_DIRS=($(find "$DATA_DIR" -maxdepth 1 -type d -name "Training_*" | sort))

if [[ ${#TRAINING_DIRS[@]} -eq 0 ]]; then
    echo "Error: No training directories found in $DATA_DIR"
    echo "Expected directories with pattern: Training_*"
    exit 1
fi

echo "Found ${#TRAINING_DIRS[@]} training directories:"
for training_dir in "${TRAINING_DIRS[@]}"; do
    training_id=$(basename "$training_dir" | sed 's/Training_//')
    echo "  - $training_id"
done
echo ""

# Check which training directories have the required checkpoint
VALID_TRAINING_IDS=()
for training_dir in "${TRAINING_DIRS[@]}"; do
    training_id=$(basename "$training_dir" | sed 's/Training_//')
    simulations_dir="${training_dir}/checkpoint_${CHECKPOINT_NUMBER}"
    
    if [[ -d "$simulations_dir" ]]; then
        VALID_TRAINING_IDS+=("$training_id")
        echo "✓ $training_id has checkpoint $CHECKPOINT_NUMBER"
    else
        echo "✗ $training_id missing checkpoint $CHECKPOINT_NUMBER (no directory: $simulations_dir)"
    fi
done

if [[ ${#VALID_TRAINING_IDS[@]} -eq 0 ]]; then
    echo ""
    echo "Error: No training directories have checkpoint '$CHECKPOINT_NUMBER'"
    echo "Please verify the checkpoint number and ensure simulations have been run."
    exit 1
fi

echo ""
echo "Will analyze ${#VALID_TRAINING_IDS[@]} training runs with checkpoint $CHECKPOINT_NUMBER"
echo ""

# Create logs directory
LOG_DIR="analysis_logs"
mkdir -p "$LOG_DIR"

# Array to store process IDs (for parallel execution)
PIDS=()

# Launch analysis for each valid training ID
for i in "${!VALID_TRAINING_IDS[@]}"; do
    training_id="${VALID_TRAINING_IDS[$i]}"
    log_file="$LOG_DIR/analysis_${MAP_NR}_${training_id}_${CHECKPOINT_NUMBER}.log"
    
    # Build the command
    cmd="python3 ../analysis_simulations.py --map_nr $MAP_NR --training_id $training_id --checkpoint_number $CHECKPOINT_NUMBER --cluster $CLUSTER --game_version $GAME_VERSION --num_agents $NUM_AGENTS"
    
    echo "[$((i+1))/${#VALID_TRAINING_IDS[@]}] Analyzing training_id: $training_id"
    echo "  Command: $cmd"
    echo "  Log file: $log_file"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would execute command above"
    else
        if [[ "$PARALLEL" == "true" ]]; then
            # Run in parallel
            nohup bash -c "$cmd" > "$log_file" 2>&1 &
            PID=$!
            PIDS+=($PID)
            echo "  Started with PID: $PID"
        else
            # Run sequentially
            echo "  Executing..."
            if bash -c "$cmd" > "$log_file" 2>&1; then
                echo "  ✓ Completed successfully"
            else
                echo "  ✗ Failed (check log: $log_file)"
            fi
        fi
    fi
    
    echo ""
    
    # Small delay between launches when running in parallel
    if [[ "$PARALLEL" == "true" && "$DRY_RUN" == "false" ]]; then
        sleep 2
    fi
done

if [[ "$DRY_RUN" == "true" ]]; then
    echo "=================================================="
    echo "DRY RUN COMPLETE"
    echo "No analyses were actually executed."
    echo "Remove --dry-run flag to run the analyses."
    echo "=================================================="
elif [[ "$PARALLEL" == "true" ]]; then
    echo "=================================================="
    echo "All ${#VALID_TRAINING_IDS[@]} analyses started in parallel!"
    echo "Process IDs: ${PIDS[*]}"
    echo "=================================================="
    echo ""
    echo "To monitor progress:"
    echo "  tail -f $LOG_DIR/analysis_${MAP_NR}_*_${CHECKPOINT_NUMBER}.log"
    echo ""
    echo "To check running processes:"
    echo "  ps aux | grep analysis_simulations"
    echo ""
    echo "To kill all analyses:"
    echo "  kill ${PIDS[*]}"
    echo ""
    echo "Log files are in: $LOG_DIR/"
    
    # Wait for all processes to complete
    echo ""
    echo "Waiting for all analyses to complete..."
    for pid in "${PIDS[@]}"; do
        wait $pid
        echo "Process $pid completed"
    done
    echo "All analyses completed!"
    
else
    echo "=================================================="
    echo "All ${#VALID_TRAINING_IDS[@]} analyses completed sequentially!"
    echo "=================================================="
    echo ""
    echo "Log files are in: $LOG_DIR/"
    echo ""
    echo "Results should be available in the respective training directories:"
    for training_id in "${VALID_TRAINING_IDS[@]}"; do
        result_dir="${DATA_DIR}/Training_${training_id}/figures_simulations_${CHECKPOINT_NUMBER}"
        echo "  - $result_dir"
    done
fi

echo ""
echo "Analysis launcher finished!"
