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
#   --trainings FILE           Only the Training_* folders named in FILE, one name per line.
#                              A line [GLOB] starts a section: the names below it only match
#                              trainings whose map folder matches GLOB (e.g. [*encouraged*],
#                              [map_a1_baseline_division_of_labor_large]). If GLOB contains '/',
#                              it is matched against map_*/synergy_*/specialized_* instead.
#                              Blank lines, # comments and other text are ignored.
#                              Aborts if an entry is not found or matches more than one folder.
#   --override_abilities W_C[,W_C...]
#                              Load the agents from each training, but SIMULATE them with
#                              walking speed W for ai_rl_1 and cutting speed C for ai_rl_2,
#                              whatever they were trained with. Several values separated by commas
#                              run NUM_SIMULATIONS for each. Results go to abilities_<W>_<C>/,
#                              and each simulation gets a speed_override.txt with the trained values.
#   --checkpoint N             Use only checkpoint_N (e.g. 753 or final); same as editing
#                              HARDCODED_CHECKPOINTS. Non-final checkpoints add checkpoint_N/ to the output.
#   --output_subfolder NAME    Extra folder after the init folder in the output, e.g. with_pretraining:
#                              simulations/empty_init/with_pretraining/map_*/...
#   --dry_run                  Print the job list without launching anything
#   -h | --help
#
# Every other argument is forwarded to experimental_simulation.py.
#
# Examples:
#   nohup ./launch_simulations_parallel.sh 20 5 --enable_video false > log_simulation_parallel.out 2>&1 &
#   ./launch_simulations_parallel.sh 20 5 --map_filter 'map_a1_*' --dry_run
#   ./launch_simulations_parallel.sh 20 5 --override_abilities 0.4_0.2 \
#       --root /data/samuel_lozano/cooked/classic_collision/empty_init/map_a1_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/Training_2026-05-24_11-37-03 --dry_run
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

# Builds a training folder that symlinks everything from the real one except config.txt,
# which is copied with WALKING_SPEEDS['ai_rl_1'] and CUTTING_SPEEDS['ai_rl_2'] replaced.
create_override_training() {
    local src="$1" dst="$2" walk="$3" cut="$4"
    rm -rf "$dst"
    mkdir -p "$dst"
    find "$src" -mindepth 1 -maxdepth 1 ! -name config.txt -exec ln -s {} "$dst"/ \;
    python3 - "$src/config.txt" "$dst/config.txt" "$walk" "$cut" <<'PY'
import ast, sys
src, dst, walk, cut = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
targets = {'WALKING_SPEEDS': ('ai_rl_1', walk), 'CUTTING_SPEEDS': ('ai_rl_2', cut)}
done = set()
out = []
with open(src, encoding='utf-8') as f:
    for line in f:
        key = line.strip().split(':', 1)[0]
        if key in targets and line.strip().startswith(key + ':'):
            speeds = ast.literal_eval(line.split(':', 1)[1].strip())
            agent, value = targets[key]
            speeds[agent] = value
            line = f"{key}: {speeds!r}\n"
            done.add(key)
        out.append(line)
missing = set(targets) - done
if missing:
    sys.exit(f"missing {missing} in {src}")
with open(dst, 'w', encoding='utf-8') as f:
    f.writelines(out)
PY
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
OVERRIDE_ABILITIES=""
TRAININGS_FILE=""
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
        --trainings)        require_value "$1" $#; TRAININGS_FILE="$2"; shift 2 ;;
        --checkpoint)       require_value "$1" $#; CHECKPOINT_MODE="hardcoded"; HARDCODED_CHECKPOINTS=("$2"); shift 2 ;;
        --override_abilities) require_value "$1" $#; OVERRIDE_ABILITIES="$2"; shift 2 ;;
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
OVERRIDE_LIST=()
if [[ -n "$OVERRIDE_ABILITIES" ]]; then
    IFS=',' read -ra _overrides <<< "$OVERRIDE_ABILITIES"
    for value in "${_overrides[@]}"; do
        if ! value="$(normalize_abilities "$value" 2>/dev/null)"; then
            echo "Error: --override_abilities must look like W_C or W_C,W_C,... e.g. 0.1_0.1,0.2_1.0"
            exit 1
        fi
        OVERRIDE_LIST+=("$value")
    done
    OVERRIDE_ABILITIES="$(IFS=','; echo "${OVERRIDE_LIST[*]}")"
    # Shadow trainings for this launch only (symlinks + edited config.txt)
    OVERRIDE_DIR="${SCRIPT_DIR}/speed_override_trainings/launch_$(date +%Y%m%d_%H%M%S)_$$"
fi
# Selected trainings: one entry per (map glob, training name)
SEL_GLOBS=()
SEL_NAMES=()
SEL_FOUND=()     # newline-separated folders matched by each entry
if [[ -n "$TRAININGS_FILE" ]]; then
    if [[ ! -f "$TRAININGS_FILE" ]]; then
        echo "Error: trainings file not found: $TRAININGS_FILE"
        exit 1
    fi
    current_glob="*"
    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(echo "$line" | tr -d '[:space:]')"
        if [[ "$line" =~ ^\[(.+)\]$ ]]; then
            current_glob="${BASH_REMATCH[1]}"
        elif [[ "$line" == Training_* ]]; then
            SEL_GLOBS+=("$current_glob")
            SEL_NAMES+=("$line")
            SEL_FOUND+=("")
        fi
    done < "$TRAININGS_FILE"
    if [[ ${#SEL_NAMES[@]} -eq 0 ]]; then
        echo "Error: no Training_* names in $TRAININGS_FILE"
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
JOB_ORIGINALS=()     # real training folder per job (used with --override_abilities)
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

    if [[ ${#SEL_NAMES[@]} -gt 0 ]]; then
        TRAINING_NAME="${REL_PATH##*/}"
        REL_DIR="${REL_PATH%/*}"
        SELECTED=false
        for k in "${!SEL_NAMES[@]}"; do
            [[ "${SEL_NAMES[$k]}" != "$TRAINING_NAME" ]] && continue
            glob="${SEL_GLOBS[$k]}"
            if [[ "$glob" == */* ]]; then match_target="$REL_DIR"; else match_target="$MAP_DIR"; fi
            # shellcheck disable=SC2053
            if [[ "$match_target" == $glob ]]; then
                SEL_FOUND[$k]+="${TRAINING_PATH}"$'\n'
                SELECTED=true
            fi
        done
        [[ "$SELECTED" == false ]] && continue
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

    # One pass per simulated abilities value ("" = the training's own speeds)
    if [[ ${#OVERRIDE_LIST[@]} -gt 0 ]]; then
        PASSES=("${OVERRIDE_LIST[@]}")
    else
        PASSES=("")
    fi

    for OVERRIDE in "${PASSES[@]}"; do
        # Folder the simulation loads the agents from (a shadow copy when overriding speeds)
        SIM_TRAINING_PATH="$TRAINING_PATH"
        SIM_ABILITIES="$ABILITIES"
        LABEL_SUFFIX=""
        if [[ -n "$OVERRIDE" ]]; then
            INIT_NAME="$(basename "${TRAINING_PATH%/"$REL_PATH"}")"
            SIM_TRAINING_PATH="${OVERRIDE_DIR}/abilities_${OVERRIDE}/${INIT_NAME}/${REL_PATH}"
            SIM_ABILITIES="$OVERRIDE"
            LABEL_SUFFIX="_sim${OVERRIDE}"
            if [[ "$DRY_RUN" == false ]]; then
                if ! create_override_training "$TRAINING_PATH" "$SIM_TRAINING_PATH" \
                        "${OVERRIDE%_*}" "${OVERRIDE#*_}"; then
                    echo "Warning: skipping abilities ${OVERRIDE} (could not build speed override): $TRAINING_PATH"
                    NUM_SKIPPED=$((NUM_SKIPPED + 1))
                    continue
                fi
            fi
        fi

        for checkpoint in "${CHECKPOINTS[@]}"; do
            SAFE_CHECKPOINT="$(sanitize_for_filename "$checkpoint")"
            CHECKPOINT_FILE="${CHECKPOINT_CONFIG_DIR}/checkpoint_${LABEL}_${SAFE_CHECKPOINT}${LABEL_SUFFIX}.txt"
            if [[ "$DRY_RUN" == false ]]; then
                create_custom_checkpoint_file "$CHECKPOINT_FILE" "$checkpoint" "$SIM_TRAINING_PATH"
            fi

            for run_idx in $(seq 1 "$NUM_SIMULATIONS"); do
                JOB_MAPS+=("$MAP_NR")
                JOB_CKPT_FILES+=("$CHECKPOINT_FILE")
                JOB_LOGS+=("${LOG_DIR}/sim_${LABEL}_${SAFE_CHECKPOINT}${LABEL_SUFFIX}_${run_idx}.log")
                JOB_ORIGINALS+=("$([[ -n "$OVERRIDE" ]] && echo "$TRAINING_PATH")")
                if [[ -n "$OVERRIDE" ]]; then
                    JOB_DESC+=("${REL_PATH} | trained ${ABILITIES} -> simulated ${SIM_ABILITIES} | checkpoint ${checkpoint} | run ${run_idx}/${NUM_SIMULATIONS}")
                else
                    JOB_DESC+=("${REL_PATH} | abilities ${ABILITIES} | checkpoint ${checkpoint} | run ${run_idx}/${NUM_SIMULATIONS}")
                fi
            done
        done
    done
done < <(find "$ROOT" -type d -name 'simulations' -prune -o \
              -type d -name 'Training_*' -prune -print0 | sort -z -V)

TOTAL_JOBS=${#JOB_MAPS[@]}

if [[ ${#SEL_NAMES[@]} -gt 0 ]]; then
    SELECTION_OK=true
    for k in "${!SEL_NAMES[@]}"; do
        n_found="$(printf '%s' "${SEL_FOUND[$k]}" | grep -c .)"
        if [[ "$n_found" -eq 0 ]]; then
            echo "Error: [${SEL_GLOBS[$k]}] ${SEL_NAMES[$k]} not found under $ROOT"
            SELECTION_OK=false
        elif [[ "$n_found" -gt 1 ]]; then
            echo "Error: [${SEL_GLOBS[$k]}] ${SEL_NAMES[$k]} matches $n_found folders:"
            printf '%s' "${SEL_FOUND[$k]}" | sed 's/^/    /'
            SELECTION_OK=false
        fi
    done
    if [[ "$SELECTION_OK" == false ]]; then
        echo "Aborting: fix the trainings list (or --root / --map_filter) first."
        exit 1
    fi
fi

echo "=================================================="
echo "Trainings-tree parallel launcher"
echo "Root: $ROOT"
echo "Map filter: $MAP_FILTER"
echo "Trainings list: ${TRAININGS_FILE:-all} (${#SEL_NAMES[@]} entries)"
echo "Abilities filter (trained): ${ABILITIES_FILTER:-none}"
echo "Simulated abilities: ${OVERRIDE_ABILITIES:-same as training}"
echo "Game version: $GAME_VERSION"
echo "Simulations root: $SIMULATIONS_ROOT"
if [[ "$CHECKPOINT_MODE" == "hardcoded" ]]; then
    echo "Checkpoints: ${HARDCODED_CHECKPOINTS[*]}"
else
    echo "Checkpoint mode: found (every checkpoint_* folder)"
fi
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
    [[ -n "${JOB_ORIGINALS[$i]}" ]] && CMD+=(--original_training_path "${JOB_ORIGINALS[$i]}")
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

# Remove the shadow trainings (only symlinks + config.txt) unless a simulation got stuck there
if [[ -n "$OVERRIDE_ABILITIES" && -d "$OVERRIDE_DIR" ]]; then
    if [[ -z "$(find "$OVERRIDE_DIR" -type d -name 'simulation_*' -print -quit)" ]]; then
        rm -rf "$OVERRIDE_DIR"
        rmdir "$(dirname "$OVERRIDE_DIR")" 2>/dev/null
        echo "Removed temporary speed-override trainings."
    else
        echo "Some simulations were not relocated; kept $OVERRIDE_DIR"
    fi
fi
echo "Logs: $LOG_DIR"
echo "Failed runs (logs containing 'Error during simulation'):"
grep -l "Error during simulation" "${JOB_LOGS[@]}" 2>/dev/null || echo "  none"
echo "=================================================="