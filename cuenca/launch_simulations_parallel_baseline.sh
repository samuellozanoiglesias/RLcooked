#!/bin/bash

# =============================================================================
# Parallel Experimental Simulation Launcher (Custom Checkpoint Files Only)
# =============================================================================
# This script only runs simulations using --custom_checkpoints.
#
# For each hardcoded configuration, it auto-generates checkpoint files in:
#   cuenca/experimental_checkpoints/checkpoint_<config_name>_<checkpoint>.txt
#
# Then it launches each configuration/checkpoint combination NUM_SIMULATIONS times,
# with a maximum of MAX_PARALLEL jobs running concurrently.
#
# Usage:
#   ./launch_simulations_parallel.sh MAP_NR GAME_VERSION NUM_SIMULATIONS [MAX_PARALLEL] [--checkpoint_mode MODE] [OPTIONAL_SIM_ARGS]
#
# Arguments:
#   MAP_NR            Map name used by experimental_simulation.py
#   GAME_VERSION      Game version (classic, classic_collision, competition, competition_collision)
#   NUM_SIMULATIONS   Number of runs per (configuration, checkpoint)
#   MAX_PARALLEL      Optional max parallel jobs (default: 5)
#
# Options:
#   --checkpoint_mode found|hardcoded
#       found:      discover every checkpoint_* folder inside each Training path
#       hardcoded:  use HARDCODED_CHECKPOINTS array from this script
#
# Examples:
#   nohup ./launch_simulations_parallel.sh baseline_division_of_labor_large classic_collision 20 5 --checkpoint_mode hardcoded --enable_video true > log_simulation_parallel.out 2>&1 &
#
#   nohup ./launch_simulations_parallel.sh baseline_division_of_labor_large classic_collision 20 --checkpoint_mode found --enable_video true --duration 300 > log_simulation_parallel_found.out 2>&1 &
# =============================================================================

DEFAULT_MAX_PARALLEL=5
DEFAULT_CHECKPOINT_MODE="found"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMULATION_SCRIPT="${SCRIPT_DIR}/../experimental_simulation.py"
CHECKPOINT_CONFIG_DIR="${SCRIPT_DIR}/experimental_checkpoints"
LOG_DIR="${SCRIPT_DIR}/simulation_logs"

# -----------------------------------------------------------------------------
# Hardcoded configurations
# -----------------------------------------------------------------------------
# Format per row:
#   CONFIG_NAME|FULL_PATH_TO_TRAINING_DIRECTORY
#
# Example training path:
#   /data/.../map_<MAP_NR>/synergy_X.XX/specialized_Y.YY/Training_<TRAINING_ID>
CONFIGURATIONS=(
    "baseline_collision_MA_1|/data/samuel_lozano/RLcooked/classic_collision/empty_init/map_baseline_division_of_labor_large/synergy_1.70/specialized_0.05/Training_2026-04-16_00-45-15"
    "baseline_collision_MA_2|/data/samuel_lozano/RLcooked/classic_collision/empty_init/map_baseline_division_of_labor_large/synergy_1.70/specialized_0.05/Training_2026-04-16_00-55-20"
    "baseline_collision_MA_3|/data/samuel_lozano/RLcooked/classic_collision/empty_init/map_baseline_division_of_labor_large/synergy_1.70/specialized_0.05/Training_2026-04-16_01-05-24"
    "baseline_collision_MA_4|/data/samuel_lozano/RLcooked/classic_collision/empty_init/map_baseline_division_of_labor_large/synergy_1.70/specialized_0.05/Training_2026-04-16_01-15-28"
    "baseline_collision_HA_1|/data/samuel_lozano/RLcooked/classic_collision/empty_init/map_baseline_division_of_labor_large/synergy_1.70/specialized_0.05/Training_2026-04-16_00-39-42"
    "baseline_collision_HA_2|/data/samuel_lozano/RLcooked/classic_collision/empty_init/map_baseline_division_of_labor_large/synergy_1.70/specialized_0.05/Training_2026-04-16_00-48-37"
    "baseline_collision_HA_3|/data/samuel_lozano/RLcooked/classic_collision/empty_init/map_baseline_division_of_labor_large/synergy_1.70/specialized_0.05/Training_2026-04-16_00-58-43"
    "baseline_collision_HA_4|/data/samuel_lozano/RLcooked/classic_collision/empty_init/map_baseline_division_of_labor_large/synergy_1.70/specialized_0.05/Training_2026-04-16_01-08-43"
)

# Used only when --checkpoint_mode hardcoded
HARDCODED_CHECKPOINTS=(
    "final"
)

print_usage() {
    echo "Usage: $0 MAP_NR GAME_VERSION NUM_SIMULATIONS [MAX_PARALLEL] [--checkpoint_mode MODE] [OPTIONAL_SIM_ARGS]"
    echo ""
    echo "Arguments:"
    echo "  MAP_NR            Map name"
    echo "  GAME_VERSION      classic | classic_collision | competition | competition_collision"
    echo "  NUM_SIMULATIONS   Positive integer"
    echo "  MAX_PARALLEL      Optional positive integer (default: $DEFAULT_MAX_PARALLEL)"
    echo ""
    echo "Options:"
    echo "  --checkpoint_mode found|hardcoded  (default: $DEFAULT_CHECKPOINT_MODE)"
    echo ""
    echo "All remaining arguments are forwarded to experimental_simulation.py."
}

is_positive_integer() {
    [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -ge 1 ]]
}

extract_training_id() {
    local training_path="$1"
    local base_name
    base_name="$(basename "$training_path")"

    if [[ "$base_name" == Training_* ]]; then
        echo "${base_name#Training_}"
    else
        echo "$base_name"
    fi
}

sanitize_for_filename() {
    local value="$1"
    value="${value//\//_}"
    value="${value// /_}"
    echo "$value"
}

create_custom_checkpoint_file() {
    local checkpoint_file="$1"
    local checkpoint_number="$2"
    local training_path="$3"

    if [[ "$training_path" != */ ]]; then
        training_path="${training_path}/"
    fi

    cat > "$checkpoint_file" <<EOF
policy_ai_rl_1
${checkpoint_number}
${training_path}
policy_ai_rl_2
${checkpoint_number}
${training_path}
EOF
}

discover_checkpoints_in_training() {
    local training_path="$1"

    find "$training_path" -maxdepth 1 -mindepth 1 -type d -name "checkpoint_*" -printf "%f\n" \
        | sed 's/^checkpoint_//' \
        | sort -V
}

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
if [[ $# -lt 3 ]]; then
    print_usage
    exit 1
fi

MAP_NR="$1"
GAME_VERSION="$2"
NUM_SIMULATIONS="$3"
shift 3

if ! is_positive_integer "$NUM_SIMULATIONS"; then
    echo "Error: NUM_SIMULATIONS must be a positive integer"
    exit 1
fi

MAX_PARALLEL="$DEFAULT_MAX_PARALLEL"
if [[ $# -gt 0 ]] && is_positive_integer "$1"; then
    MAX_PARALLEL="$1"
    shift
fi

CHECKPOINT_MODE="$DEFAULT_CHECKPOINT_MODE"
SIMULATION_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint_mode)
            if [[ $# -lt 2 ]]; then
                echo "Error: --checkpoint_mode requires a value (found|hardcoded)"
                exit 1
            fi
            CHECKPOINT_MODE="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            SIMULATION_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$CHECKPOINT_MODE" != "found" && "$CHECKPOINT_MODE" != "hardcoded" ]]; then
    echo "Error: invalid --checkpoint_mode '$CHECKPOINT_MODE'. Use 'found' or 'hardcoded'."
    exit 1
fi

if [[ ! -f "$SIMULATION_SCRIPT" ]]; then
    echo "Error: simulation script not found: $SIMULATION_SCRIPT"
    exit 1
fi

mkdir -p "$CHECKPOINT_CONFIG_DIR"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "Custom-checkpoint parallel launcher"
echo "Map: $MAP_NR"
echo "Game version: $GAME_VERSION"
echo "Simulations per configuration/checkpoint: $NUM_SIMULATIONS"
echo "Max parallel: $MAX_PARALLEL"
echo "Checkpoint mode: $CHECKPOINT_MODE"
echo "Checkpoint files dir: $CHECKPOINT_CONFIG_DIR"
echo "Log dir: $LOG_DIR"
echo "Forwarded simulation args: ${SIMULATION_ARGS[*]}"
echo "=================================================="

PIDS=()
RUNNING_COUNT=0
TOTAL_LAUNCHED=0

for config_row in "${CONFIGURATIONS[@]}"; do
    IFS='|' read -r CONFIG_NAME TRAINING_PATH <<< "$config_row"

    if [[ -z "$CONFIG_NAME" || -z "$TRAINING_PATH" ]]; then
        echo "Error: invalid CONFIGURATIONS row: '$config_row'"
        exit 1
    fi

    if [[ ! -d "$TRAINING_PATH" ]]; then
        echo "Error: training path does not exist for config '$CONFIG_NAME': $TRAINING_PATH"
        exit 1
    fi

    TRAINING_ID="$(extract_training_id "$TRAINING_PATH")"
    CHECKPOINTS=()

    if [[ "$CHECKPOINT_MODE" == "found" ]]; then
        while IFS= read -r checkpoint_value; do
            if [[ -n "$checkpoint_value" ]]; then
                CHECKPOINTS+=("$checkpoint_value")
            fi
        done < <(discover_checkpoints_in_training "$TRAINING_PATH")
    else
        CHECKPOINTS=("${HARDCODED_CHECKPOINTS[@]}")
    fi

    if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
        echo "Warning: no checkpoints found for config '$CONFIG_NAME' in $TRAINING_PATH"
        continue
    fi

    echo ""
    echo "Configuration: $CONFIG_NAME"
    echo "Training path: $TRAINING_PATH"
    echo "Training ID: $TRAINING_ID"
    echo "Checkpoints: ${CHECKPOINTS[*]}"
    echo "--------------------------------------------------"

    for checkpoint in "${CHECKPOINTS[@]}"; do
        SAFE_CHECKPOINT="$(sanitize_for_filename "$checkpoint")"
        CHECKPOINT_FILE="${CHECKPOINT_CONFIG_DIR}/checkpoint_${CONFIG_NAME}_${SAFE_CHECKPOINT}.txt"
        create_custom_checkpoint_file "$CHECKPOINT_FILE" "$checkpoint" "$TRAINING_PATH"

        for run_idx in $(seq 1 "$NUM_SIMULATIONS"); do
            while [[ "$RUNNING_COUNT" -ge "$MAX_PARALLEL" ]]; do
                echo "Max parallel limit ($MAX_PARALLEL) reached. Waiting for a job to finish..."
                wait -n
                RUNNING_COUNT="$(jobs -p | wc -l)"
                echo "A job completed. $RUNNING_COUNT jobs still running."
            done

            LOG_FILE="${LOG_DIR}/sim_${CONFIG_NAME}_${TRAINING_ID}_${SAFE_CHECKPOINT}_${run_idx}.log"
            CMD=(python3 "$SIMULATION_SCRIPT" "$MAP_NR" "$GAME_VERSION" --custom_checkpoints "$CHECKPOINT_FILE")

            if [[ ${#SIMULATION_ARGS[@]} -gt 0 ]]; then
                CMD+=("${SIMULATION_ARGS[@]}")
            fi

            nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
            PID=$!
            PIDS+=("$PID")
            TOTAL_LAUNCHED=$((TOTAL_LAUNCHED + 1))

            echo "Launched run $run_idx/$NUM_SIMULATIONS for config '$CONFIG_NAME' checkpoint '$checkpoint' (PID: $PID)"
            echo "  checkpoint_file: $CHECKPOINT_FILE"
            echo "  log_file: $LOG_FILE"

            RUNNING_COUNT="$(jobs -p | wc -l)"
            sleep 2
        done
    done
done

if [[ "$TOTAL_LAUNCHED" -eq 0 ]]; then
    echo "Error: no simulations were launched."
    exit 1
fi

echo ""
echo "=================================================="
echo "All simulations launched. Total jobs: $TOTAL_LAUNCHED"
echo "Waiting for remaining jobs to finish..."
echo "=================================================="

wait

echo ""
echo "=================================================="
echo "All simulations completed."
echo "Process IDs: ${PIDS[*]}"
echo "Checkpoint files: $CHECKPOINT_CONFIG_DIR"
echo "Logs: $LOG_DIR"
echo "=================================================="
