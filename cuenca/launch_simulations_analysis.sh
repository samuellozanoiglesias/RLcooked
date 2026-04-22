#!/bin/bash

# =============================================================================
# Multiple Training Analysis Launcher
# =============================================================================
# Script to launch analysis_simulations.py for all training_ids in a given 
# folder, optionally filtered by checkpoint_number.
#
# Default behavior launches analyses in parallel with a concurrency cap.
#
# Usage: ./launch_simulations_analysis.sh MAP_NR [CHECKPOINT_NUMBER] [OPTIONS]
#
# Example:
#   nohup ./launch_simulations_analysis.sh baseline_division_of_labor > log_simulations_analysis.out 2>&1 &
#   nohup ./launch_simulations_analysis.sh baseline_division_of_labor 50 --max-parallel 8 > log_simulations_analysis.out 2>&1 &
#   nohup ./launch_simulations_analysis.sh baseline_division_of_labor --checkpoint_number final --trainings-dir /data/samuel_lozano/cooked/map_baseline_division_of_labor/simulations --max-parallel 4 > log_simulations_analysis.out 2>&1 &
# =============================================================================

# Check minimum arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 MAP_NR [CHECKPOINT_NUMBER] [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  MAP_NR              Map name (e.g., 'baseline_division_of_labor')"
    echo "  CHECKPOINT_NUMBER   Optional positional checkpoint filter"
    echo ""
    echo "Options:"
    echo "  --checkpoint_number Optional checkpoint filter (e.g., final, 50)"
    echo "  --cluster           Cluster name (default: cuenca)"
    echo "  --game-version      Game version (default: classic)"
    echo "  --num_agents        Number of agents (default: 2)"
    echo "  --trainings-dir     Folder containing Training_* directories"
    echo "  --study-name        Optional subfolder under simulations/"
    echo "  --synergy           Optional synergy segment (e.g., 1.70 or synergy_1.70)"
    echo "  --specialization    Optional specialization segment (e.g., 0.05 or specialized_0.05)"
    echo "  --max-parallel      Max simultaneous nohup python processes (default: 4)"
    echo "  --dry-run           Show what would be executed without running"
    echo "  --parallel          Force parallel mode"
    echo "  --sequential        Run one training at a time"
    echo ""
    echo "Examples:"
    echo "  $0 baseline_division_of_labor --max-parallel 6"
    echo "  $0 baseline_division_of_labor 50 --cluster cuenca --max-parallel 8"
    echo "  $0 baseline_division_of_labor --checkpoint_number final --dry-run"
    exit 1
fi

# Parse required argument
MAP_NR=$1
shift 1

# Optional positional checkpoint filter (backward-compatible)
CHECKPOINT_NUMBER=""
if [[ $# -gt 0 && "$1" != --* ]]; then
    CHECKPOINT_NUMBER=$1
    shift 1
fi

# Default parameters
CLUSTER="cuenca"
GAME_VERSION="classic_collision"
NUM_AGENTS=2
DRY_RUN=false
PARALLEL=true
MAX_PARALLEL=12
TRAININGS_DIR_OVERRIDE=""
STUDY_NAME=""
SYNERGY=1.70
SPECIALIZATION=0.05

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_number)
            CHECKPOINT_NUMBER="$2"
            shift 2
            ;;
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
        --trainings-dir)
            TRAININGS_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --study-name)
            STUDY_NAME="$2"
            shift 2
            ;;
        --synergy)
            SYNERGY="$2"
            shift 2
            ;;
        --specialization)
            SPECIALIZATION="$2"
            shift 2
            ;;
        --max-parallel)
            MAX_PARALLEL="$2"
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
        --sequential)
            PARALLEL=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --max-parallel must be a positive integer. Got: $MAX_PARALLEL"
    exit 1
fi

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

# Resolve folder that contains Training_* directories
if [[ -n "$TRAININGS_DIR_OVERRIDE" ]]; then
    TRAININGS_DIR="$TRAININGS_DIR_OVERRIDE"
else
    # Construct map base directory path
    if [[ "$NUM_AGENTS" == 1 ]]; then
        MAP_BASE_DIR="${BASE_CLUSTER_DIR}/data/samuel_lozano/cooked/pretraining/${GAME_VERSION}/map_${MAP_NR}"
    else
        MAP_BASE_DIR="${BASE_CLUSTER_DIR}/data/samuel_lozano/cooked/${GAME_VERSION}/map_${MAP_NR}"
    fi

    if [[ -n "$SYNERGY" ]]; then
        if [[ "$SYNERGY" == synergy_* ]]; then
            MAP_BASE_DIR="${MAP_BASE_DIR}/${SYNERGY}"
        else
            MAP_BASE_DIR="${MAP_BASE_DIR}/synergy_${SYNERGY}"
        fi
    fi

    if [[ -n "$SPECIALIZATION" ]]; then
        if [[ "$SPECIALIZATION" == specialized_* ]]; then
            MAP_BASE_DIR="${MAP_BASE_DIR}/${SPECIALIZATION}"
        else
            MAP_BASE_DIR="${MAP_BASE_DIR}/specialized_${SPECIALIZATION}"
        fi
    fi

    TRAININGS_DIR="${MAP_BASE_DIR}/simulations"
    if [[ -n "$STUDY_NAME" ]]; then
        TRAININGS_DIR="${TRAININGS_DIR}/${STUDY_NAME}"
    fi
fi

# Check if trainings directory exists
if [[ ! -d "$TRAININGS_DIR" ]]; then
    echo "Error: Trainings directory does not exist: $TRAININGS_DIR"
    echo "Please verify the following parameters:"
    echo "  Cluster: $CLUSTER"
    echo "  Game version: $GAME_VERSION"
    echo "  Map: $MAP_NR"
    echo "  Study name: ${STUDY_NAME:-<none>}"
    echo "  Synergy: ${SYNERGY:-<none>}"
    echo "  Specialization: ${SPECIALIZATION:-<none>}"
    exit 1
fi

echo "=================================================="
echo "Training Analysis Launcher"
echo "=================================================="
echo "Map: $MAP_NR"
if [[ -n "$CHECKPOINT_NUMBER" ]]; then
    echo "Checkpoint filter: $CHECKPOINT_NUMBER"
else
    echo "Checkpoint filter: <all checkpoints>"
fi
echo "Cluster: $CLUSTER"
echo "Game version: $GAME_VERSION"
echo "Trainings directory: $TRAININGS_DIR"
echo "Parallel execution: $PARALLEL"
echo "Max simultaneous jobs: $MAX_PARALLEL"
echo "Dry run: $DRY_RUN"
echo "=================================================="

# Find all training directories
echo "Searching for training directories in: $TRAININGS_DIR"
mapfile -t TRAINING_DIRS < <(find "$TRAININGS_DIR" -maxdepth 1 -type d -name "Training_*" | sort)

if [[ ${#TRAINING_DIRS[@]} -eq 0 ]]; then
    echo "Error: No training directories found in $TRAININGS_DIR"
    echo "Expected directories with pattern: Training_*"
    exit 1
fi

echo "Found ${#TRAINING_DIRS[@]} training directories:"
for training_dir in "${TRAINING_DIRS[@]}"; do
    training_id=$(basename "$training_dir" | sed 's/Training_//')
    echo "  - $training_id"
done
echo ""

# Check which training directories are valid targets
VALID_TRAINING_IDS=()
for training_dir in "${TRAINING_DIRS[@]}"; do
    training_id=$(basename "$training_dir" | sed 's/Training_//')

    if [[ -n "$CHECKPOINT_NUMBER" ]]; then
        checkpoint_dir="${training_dir}/checkpoint_${CHECKPOINT_NUMBER}"

        if [[ -d "$checkpoint_dir" ]]; then
            VALID_TRAINING_IDS+=("$training_id")
            echo "✓ $training_id has checkpoint $CHECKPOINT_NUMBER"
        else
            echo "✗ $training_id missing checkpoint $CHECKPOINT_NUMBER (no directory: $checkpoint_dir)"
        fi
    else
        mapfile -t CHECKPOINT_DIRS < <(find "$training_dir" -maxdepth 1 -type d -name "checkpoint_*" | sort)
        if [[ ${#CHECKPOINT_DIRS[@]} -gt 0 ]]; then
            VALID_TRAINING_IDS+=("$training_id")
            echo "✓ $training_id has ${#CHECKPOINT_DIRS[@]} checkpoints"
        else
            echo "✗ $training_id has no checkpoint_* directories"
        fi
    fi
done

if [[ ${#VALID_TRAINING_IDS[@]} -eq 0 ]]; then
    echo ""
    if [[ -n "$CHECKPOINT_NUMBER" ]]; then
        echo "Error: No training directories have checkpoint '$CHECKPOINT_NUMBER'"
        echo "Please verify the checkpoint number and ensure simulations have been run."
    else
        echo "Error: No training directories with checkpoint_* content were found"
        echo "Please verify the target trainings folder and simulation outputs."
    fi
    exit 1
fi

echo ""
if [[ -n "$CHECKPOINT_NUMBER" ]]; then
    echo "Will analyze ${#VALID_TRAINING_IDS[@]} training runs with checkpoint $CHECKPOINT_NUMBER"
else
    echo "Will analyze ${#VALID_TRAINING_IDS[@]} training runs across ALL checkpoints"
fi
echo ""

# Create logs directory
LOG_DIR="analysis_logs"
mkdir -p "$LOG_DIR"

# Array to store process IDs (for parallel execution)
PIDS=()
CHECKPOINT_LABEL="${CHECKPOINT_NUMBER:-all_checkpoints}"

# Launch analysis for each valid training ID
for i in "${!VALID_TRAINING_IDS[@]}"; do
    training_id="${VALID_TRAINING_IDS[$i]}"
    log_file="$LOG_DIR/analysis_${MAP_NR}_${training_id}_${CHECKPOINT_LABEL}.log"
    
    # Build the command
    cmd=(
        python3 ../analysis_simulations.py
        --map_nr "$MAP_NR"
        --training_id "$training_id"
        --cluster "$CLUSTER"
        --game_version "$GAME_VERSION"
        --num_agents "$NUM_AGENTS"
    )

    if [[ -n "$CHECKPOINT_NUMBER" ]]; then
        cmd+=(--checkpoint_number "$CHECKPOINT_NUMBER")
    fi

    if [[ -n "$STUDY_NAME" ]]; then
        cmd+=(--study_name "$STUDY_NAME")
    fi
    if [[ -n "$SYNERGY" ]]; then
        cmd+=(--synergy "$SYNERGY")
    fi
    if [[ -n "$SPECIALIZATION" ]]; then
        cmd+=(--specialization "$SPECIALIZATION")
    fi

    cmd_display=$(printf '%q ' "${cmd[@]}")
    
    echo "[$((i+1))/${#VALID_TRAINING_IDS[@]}] Analyzing training_id: $training_id"
    echo "  Command: $cmd_display"
    echo "  Log file: $log_file"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would execute command above"
    else
        if [[ "$PARALLEL" == "true" ]]; then
            # Respect max concurrency limit.
            while [[ $(jobs -pr | wc -l) -ge $MAX_PARALLEL ]]; do
                sleep 2
            done

            # Run in parallel with nohup.
            nohup "${cmd[@]}" > "$log_file" 2>&1 &
            PID=$!
            PIDS+=($PID)
            echo "  Started with PID: $PID (running $(jobs -pr | wc -l)/$MAX_PARALLEL)"
        else
            # Run sequentially
            echo "  Executing..."
            if "${cmd[@]}" > "$log_file" 2>&1; then
                echo "  ✓ Completed successfully"
            else
                echo "  ✗ Failed (check log: $log_file)"
            fi
        fi
    fi
    
    echo ""
done

if [[ "$DRY_RUN" == "true" ]]; then
    echo "=================================================="
    echo "DRY RUN COMPLETE"
    echo "No analyses were actually executed."
    echo "Remove --dry-run flag to run the analyses."
    echo "=================================================="
elif [[ "$PARALLEL" == "true" ]]; then
    echo "=================================================="
    echo "All ${#VALID_TRAINING_IDS[@]} analyses started in parallel (max $MAX_PARALLEL at once)!"
    echo "Process IDs: ${PIDS[*]}"
    echo "=================================================="
    echo ""
    echo "To monitor progress:"
    echo "  tail -f $LOG_DIR/analysis_${MAP_NR}_*_${CHECKPOINT_LABEL}.log"
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
    echo "Results are written by analysis_simulations.py under map-level simulation_figures/."
fi

echo ""
echo "Analysis launcher finished!"
