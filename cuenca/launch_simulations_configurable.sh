#!/bin/bash

# =============================================================================
# Parallel Experimental Simulation Launcher (sweeps a whole trainings tree)
# =============================================================================
# Finds every Training_* folder under ROOT (ROOT can be the whole empty_init tree,
# one map_*/synergy_*/specialized_* folder, or a single Training_* folder)
# and launches NUM_SIMULATIONS independent simulations per (training, checkpoint),
# with at most MAX_PARALLEL jobs at once.
#
# The map name passed to experimental_simulation.py is taken from the folder
# (map_<MAP_NR>). Each simulation ends up in
#     <SIMULATIONS_ROOT>/<init>/map_*/synergy_*/specialized_*/abilities_<W>_<C>/Training_*/simulation_<timestamp>/
# (handled by experimental_simulation.py, abilities read from each config.txt).
#
# Usage:
#   ./launch_simulations_parallel.sh NUM_SIMULATIONS [MAX_PARALLEL] [options] [OPTIONAL_SIM_ARGS]
#
# Options:
#   --root DIR                 Where to look for Training_* folders (default: $DEFAULT_ROOT).
#                              Any level works: empty_init, a map/synergy/specialized folder,
#                              or one Training_* folder.
#   --game_version V           classic | classic_collision | competition | competition_collision
#                              (default: $DEFAULT_GAME_VERSION)
#   --simulations_root DIR     Output root (default: <parent of ROOT>/simulations)
#   --checkpoint_mode MODE     hardcoded: use HARDCODED_CHECKPOINTS (default, "final")
#                              found:     every checkpoint_* folder inside each training
#   --map_filter GLOB          Only maps whose folder name matches GLOB (e.g. 'map_a1_*')
#   --abilities W_C            Only trainings whose config.txt has WALKING_SPEEDS['ai_rl_1'] = W
#                              and CUTTING_SPEEDS['ai_rl_2'] = C (e.g. 0.4_0.2)
#   --dry_run                  Print the job list without launching anything
#   -h | --help
#
# Every other argument is forwarded to experimental_simulation.py.
#
# Examples:
#   nohup ./launch_simulations_parallel.sh 20 5 --enable_video false > log_simulation_parallel.out 2>&1 &
#   ./launch_simulations_parallel.sh 20 5 --map_filter 'map_a1_*' --dry_run
#   ./launch_simulations_parallel.sh 20 5 --abilities 0.4_0.2 \
#       --root /data/samuel_lozano/cooked/classic_collision/empty_init/map_a1_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25 --dry_run
#   nohup ./launch_simulations_parallel.sh 10 8 --checkpoint_mode found --duration 300 > log_found.out 2>&1 &
# =============================================================================

DEFAULT_ROOT="/data/samuel_lozano/cooked/classic_collision/empty_init"
DEFAULT_GAME_VERSION="classic_collision"
DEFAULT_MAX_PARALLEL=5
DEFAULT_CHECKPOINT_MODE="hardcoded"

# Used only when --checkpoint_mode hardcoded
HARDCODED_CHECKPOINTS=(
    "final"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMULATION_SCRIPT="${SCRIPT_DIR}/../experimental_simulation.py"
CHECKPOINT_CONFIG_DIR="${SCRIPT_DIR}/experimental_checkpoints"
LOG_DIR="${SCRIPT_DIR}/simulation_logs"

print_usage() {
    sed -n '/^# Usage:/,/^# =====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | sed '$d'
}

is_positive_integer() {
    [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -ge 1 ]]
}

sanitize_for_filename() {
    local value="$1"
    value="${value//\//__}"
    value="${value// /_}"
    echo "$value"
}

create_custom_checkpoint_file() {
    local checkpoint_file="$1"
    local checkpoint_number="$2"
    local training_path="$3"

    [[ "$training_path" != */ ]] && training_path="${training_path}/"

    cat > "$checkpoint_file" <<CKPT
policy_ai_rl_1
${checkpoint_number}
${training_path}
policy_ai_rl_2
${checkpoint_number}
${training_path}
CKPT
}

# Prints "W_C" (walking speed of ai_rl_1, cutting speed of ai_rl_2) from a config.txt,
# formatted exactly like the abilities_<W>_<C> output folder.
read_abilities() {
    python3 - "$1" <<'PY'
import ast, sys
speeds = {}
with open(sys.argv[1], encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        for key in ('WALKING_SPEEDS', 'CUTTING_SPEEDS'):
            if line.startswith(key + ':'):
                speeds[key] = ast.literal_eval(line.split(':', 1)[1].strip())
print(f"{float(speeds['WALKING_SPEEDS']['ai_rl_1'])!r}_{float(speeds['CUTTING_SPEEDS']['ai_rl_2'])!r}")
PY
}

# Normalizes a user-given "W_C" (e.g. 0.40_0.20 -> 0.4_0.2)
normalize_abilities() {
    python3 -c "import sys; w, c = sys.argv[1].split('_'); print(f'{float(w)!r}_{float(c)!r}')" "$1"
}

# Prints the last map_* component of a path
find_map_dir() {
    local part map_dir=""
    IFS='/' read -ra _parts <<< "$1"
    for part in "${_parts[@]}"; do
        [[ "$part" == map_* ]] && map_dir="$part"
    done
    echo "$map_dir"
}

discover_checkpoints_in_training() {
    find "$1" -maxdepth 1 -mindepth 1 -type d -name "checkpoint_*" -printf "%f\n" \
        | sed 's/^checkpoint_//' \
        | sort -V
}

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    print_usage
    [[ $# -lt 1 ]] && exit 1 || exit 0
fi

NUM_SIMULATIONS="$1"
shift
if ! is_positive_integer "$NUM_SIMULATIONS"; then
    echo "Error: NUM_SIMULATIONS must be a positive integer"
    exit 1
fi

MAX_PARALLEL="$DEFAULT_MAX_PARALLEL"
if [[ $# -gt 0 ]] && is_positive_integer "$1"; then
    MAX_PARALLEL="$1"
    shift
fi

ROOT="$DEFAULT_ROOT"
GAME_VERSION="$DEFAULT_GAME_VERSION"
SIMULATIONS_ROOT=""
CHECKPOINT_MODE="$DEFAULT_CHECKPOINT_MODE"
MAP_FILTER="map_*"
ABILITIES_FILTER=""
DRY_RUN=false
SIMULATION_ARGS=()

require_value() {
    if [[ $2 -lt 2 ]]; then
        echo "Error: $1 requires a value"
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)             require_value "$1" $#; ROOT="$2"; shift 2 ;;
        --game_version)     require_value "$1" $#; GAME_VERSION="$2"; shift 2 ;;
        --simulations_root) require_value "$1" $#; SIMULATIONS_ROOT="$2"; shift 2 ;;
        --checkpoint_mode)  require_value "$1" $#; CHECKPOINT_MODE="$2"; shift 2 ;;
        --map_filter)       require_value "$1" $#; MAP_FILTER="$2"; shift 2 ;;
        --abilities)        require_value "$1" $#; ABILITIES_FILTER="$2"; shift 2 ;;
        --dry_run)          DRY_RUN=true; shift ;;
        -h|--help)          print_usage; exit 0 ;;
        *)                  SIMULATION_ARGS+=("$1"); shift ;;
    esac
done

ROOT="${ROOT%/}"
if [[ -z "$SIMULATIONS_ROOT" ]]; then
    # Default: <folder above the init folder>/simulations, e.g. .../classic_collision/simulations
    INIT_DIR="$ROOT"
    while [[ "$INIT_DIR" == */* && "$(basename "$INIT_DIR")" != *_init && "$INIT_DIR" != "/" ]]; do
        INIT_DIR="$(dirname "$INIT_DIR")"
    done
    if [[ "$(basename "$INIT_DIR")" == *_init ]]; then
        SIMULATIONS_ROOT="$(dirname "$INIT_DIR")/simulations"
    else
        SIMULATIONS_ROOT="$(dirname "$ROOT")/simulations"
    fi
fi

if [[ "$CHECKPOINT_MODE" != "found" && "$CHECKPOINT_MODE" != "hardcoded" ]]; then
    echo "Error: invalid --checkpoint_mode '$CHECKPOINT_MODE'. Use 'found' or 'hardcoded'."
    exit 1
fi
if [[ -n "$ABILITIES_FILTER" ]]; then
    if ! ABILITIES_FILTER="$(normalize_abilities "$ABILITIES_FILTER" 2>/dev/null)"; then
        echo "Error: --abilities must look like W_C, e.g. 0.4_0.2"
        exit 1
    fi
fi
if [[ ! -d "$ROOT" ]]; then
    echo "Error: root folder does not exist: $ROOT"
    exit 1
fi
if [[ ! -f "$SIMULATION_SCRIPT" ]]; then
    echo "Error: simulation script not found: $SIMULATION_SCRIPT"
    exit 1
fi

mkdir -p "$CHECKPOINT_CONFIG_DIR" "$LOG_DIR"

# -----------------------------------------------------------------------------
# Build the job list
# -----------------------------------------------------------------------------
JOB_MAPS=()          # MAP_NR per job
JOB_CKPT_FILES=()    # checkpoint file per job
JOB_LOGS=()          # log file per job
JOB_DESC=()          # human-readable description per job
NUM_TRAININGS=0
NUM_SKIPPED=0

while IFS= read -r -d '' TRAINING_PATH; do
    TRAINING_PATH="${TRAINING_PATH%/}"
    MAP_DIR="$(find_map_dir "$TRAINING_PATH")"
    if [[ -z "$MAP_DIR" ]]; then
        echo "Warning: skipping (no map_* folder in path): $TRAINING_PATH"
        NUM_SKIPPED=$((NUM_SKIPPED + 1))
        continue
    fi
    MAP_NR="${MAP_DIR#map_}"
    REL_PATH="${MAP_DIR}${TRAINING_PATH##*/"$MAP_DIR"}"   # map_*/.../Training_*
    LABEL="$(sanitize_for_filename "$REL_PATH")"

    # shellcheck disable=SC2053
    if [[ "$MAP_DIR" != $MAP_FILTER ]]; then
        continue
    fi

    if [[ ! -f "$TRAINING_PATH/config.txt" ]]; then
        echo "Warning: skipping (no config.txt): $TRAINING_PATH"
        NUM_SKIPPED=$((NUM_SKIPPED + 1))
        continue
    fi

    if ! ABILITIES="$(read_abilities "$TRAINING_PATH/config.txt" 2>/dev/null)"; then
        echo "Warning: skipping (could not read WALKING_SPEEDS/CUTTING_SPEEDS): $TRAINING_PATH"
        NUM_SKIPPED=$((NUM_SKIPPED + 1))
        continue
    fi
    if [[ -n "$ABILITIES_FILTER" && "$ABILITIES" != "$ABILITIES_FILTER" ]]; then
        continue
    fi

    CHECKPOINTS=()
    if [[ "$CHECKPOINT_MODE" == "found" ]]; then
        while IFS= read -r checkpoint_value; do
            [[ -n "$checkpoint_value" ]] && CHECKPOINTS+=("$checkpoint_value")
        done < <(discover_checkpoints_in_training "$TRAINING_PATH")
    else
        for checkpoint_value in "${HARDCODED_CHECKPOINTS[@]}"; do
            if [[ -d "$TRAINING_PATH/checkpoint_${checkpoint_value}" ]]; then
                CHECKPOINTS+=("$checkpoint_value")
            else
                echo "Warning: checkpoint_${checkpoint_value} missing in $TRAINING_PATH (skipped)"
            fi
        done
    fi

    if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
        echo "Warning: skipping (no usable checkpoints): $TRAINING_PATH"
        NUM_SKIPPED=$((NUM_SKIPPED + 1))
        continue
    fi

    NUM_TRAININGS=$((NUM_TRAININGS + 1))

    for checkpoint in "${CHECKPOINTS[@]}"; do
        SAFE_CHECKPOINT="$(sanitize_for_filename "$checkpoint")"
        CHECKPOINT_FILE="${CHECKPOINT_CONFIG_DIR}/checkpoint_${LABEL}_${SAFE_CHECKPOINT}.txt"
        if [[ "$DRY_RUN" == false ]]; then
            create_custom_checkpoint_file "$CHECKPOINT_FILE" "$checkpoint" "$TRAINING_PATH"
        fi

        for run_idx in $(seq 1 "$NUM_SIMULATIONS"); do
            JOB_MAPS+=("$MAP_NR")
            JOB_CKPT_FILES+=("$CHECKPOINT_FILE")
            JOB_LOGS+=("${LOG_DIR}/sim_${LABEL}_${SAFE_CHECKPOINT}_${run_idx}.log")
            JOB_DESC+=("${REL_PATH} | abilities ${ABILITIES} | checkpoint ${checkpoint} | run ${run_idx}/${NUM_SIMULATIONS}")
        done
    done
done < <(find "$ROOT" -type d -name 'simulations' -prune -o \
              -type d -name 'Training_*' -prune -print0 | sort -z -V)

TOTAL_JOBS=${#JOB_MAPS[@]}

echo "=================================================="
echo "Trainings-tree parallel launcher"
echo "Root: $ROOT"
echo "Map filter: $MAP_FILTER"
echo "Abilities filter: ${ABILITIES_FILTER:-none}"
echo "Game version: $GAME_VERSION"
echo "Simulations root: $SIMULATIONS_ROOT"
echo "Checkpoint mode: $CHECKPOINT_MODE"
echo "Simulations per training/checkpoint: $NUM_SIMULATIONS"
echo "Max parallel: $MAX_PARALLEL"
echo "Trainings used: $NUM_TRAININGS (skipped: $NUM_SKIPPED)"
echo "Total jobs: $TOTAL_JOBS"
echo "Checkpoint files dir: $CHECKPOINT_CONFIG_DIR"
echo "Log dir: $LOG_DIR"
echo "Forwarded simulation args: ${SIMULATION_ARGS[*]}"
echo "=================================================="

if [[ "$TOTAL_JOBS" -eq 0 ]]; then
    echo "Error: no simulations to launch."
    exit 1
fi

if [[ "$DRY_RUN" == true ]]; then
    for i in "${!JOB_DESC[@]}"; do
        echo "[dry run] map_nr=${JOB_MAPS[$i]} | ${JOB_DESC[$i]}"
    done
    exit 0
fi

# -----------------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------------
PIDS=()
for i in "${!JOB_MAPS[@]}"; do
    while [[ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]]; do
        wait -n
    done

    CMD=(python3 "$SIMULATION_SCRIPT" "${JOB_MAPS[$i]}" "$GAME_VERSION"
         --custom_checkpoints "${JOB_CKPT_FILES[$i]}"
         --simulations_root "$SIMULATIONS_ROOT")
    [[ ${#SIMULATION_ARGS[@]} -gt 0 ]] && CMD+=("${SIMULATION_ARGS[@]}")

    nohup "${CMD[@]}" > "${JOB_LOGS[$i]}" 2>&1 &
    PID=$!
    PIDS+=("$PID")

    echo "[$((i + 1))/$TOTAL_JOBS] Launched ${JOB_DESC[$i]} (PID: $PID)"
    echo "  log_file: ${JOB_LOGS[$i]}"

    # Small gap so runs of the same training don't get the same timestamp
    sleep 2
done

echo ""
echo "All $TOTAL_JOBS simulations launched. Waiting for remaining jobs to finish..."
wait

echo ""
echo "=================================================="
echo "All simulations completed."
echo "Simulations: $SIMULATIONS_ROOT"
echo "Logs: $LOG_DIR"
echo "Failed runs (logs containing 'Error during simulation'):"
grep -l "Error during simulation" "${JOB_LOGS[@]}" 2>/dev/null || echo "  none"
echo "=================================================="
