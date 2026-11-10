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
#                              Repeat it to sweep several folders in one launch
#                              (--root DIR1 --root DIR2); every Training_* below each is used.
#                              Any level works: empty_init, a map/synergy/specialized folder,
#                              or one Training_* folder.
#   --game_version V           classic | classic_collision | competition | competition_collision
#                              (default: $DEFAULT_GAME_VERSION)
#   --simulations_root DIR     Output root (default: <folder above the *_init folder>/simulations;
#                              with several roots they must agree, or pass this explicitly)
#   --checkpoint_mode MODE     hardcoded: use HARDCODED_CHECKPOINTS (default, "final")
#                              found:     every checkpoint_* folder inside each training
#   --map_filter GLOB          Only maps whose folder name matches GLOB (e.g. 'map_a1_*')
#   --synergy S                Only trainings inside a synergy_<S> folder (compared as numbers,
#                              so 0 == 0.0 and 1.35 == 1.350)
#   --specialization L         Only trainings inside a specialized_<L> folder (same matching)
#   --training_glob GLOB       Only Training_* folders whose name matches GLOB
#                              (e.g. 'Training_2026-10-30_*'), in every map
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
#   --checkpoint N             Use only checkpoint_N (e.g. 750 or final); same as editing
#                              HARDCODED_CHECKPOINTS. Non-final checkpoints add checkpoint_N/ to the output.
#   --output_subfolder NAME    Extra folder after the init folder in the output, e.g. with_pretraining:
#                              simulations/empty_init/with_pretraining/map_*/...
#   --log_dir DIR              Keep the per-simulation logs in DIR. By default they are written to a
#                              temporary folder (in $TMPDIR or /tmp, NOT in this folder) and deleted
#                              at the end, except the logs of failed runs, which are kept and listed.
#   --keep_logs                Keep every per-simulation log (in the temporary folder or --log_dir)
#   --dry_run                  Launch nothing; only list the Training_* folders (after the filters)
#                              that do NOT have a usable checkpoint (missing or empty
#                              checkpoint_<N> folder; also trainings without a readable config.txt)
#   -h | --help
#
# Every other argument is forwarded to experimental_simulation.py.
#
# Examples:
#   nohup ./launch_simulations_parallel.sh 20 5 --enable_video false > log_simulation_parallel.out 2>&1 &
#   ./launch_simulations_parallel.sh 20 5 --map_filter 'map_a1_*' --dry_run
#   ./launch_simulations_parallel.sh 20 5 --synergy 0 --specialization 0.25 --dry_run
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
    "750"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMULATION_SCRIPT="${SCRIPT_DIR}/../experimental_simulation.py"
# Per-launch scratch folder (checkpoint files + per-simulation logs): outside this folder,
# removed at the end (only the logs of failed runs are kept). Set in the argument parsing.
CHECKPOINT_CONFIG_DIR=""
LOG_DIR=""

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

# Prints the value of the last <prefix>_<value> component of a path
# (e.g. path_level_value .../synergy_1.35/... synergy -> 1.35), or nothing
path_level_value() {
    local part value=""
    IFS='/' read -ra _parts <<< "$1"
    for part in "${_parts[@]}"; do
        [[ "$part" == "$2"_* ]] && value="${part#"$2"_}"
    done
    echo "$value"
}

is_number() {
    [[ "$1" =~ ^[0-9]*\.?[0-9]+$ ]]
}

# True if two numeric strings are the same number (0 == 0.0, 0.25 == .250)
same_number() {
    is_number "$1" && is_number "$2" && awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 == b + 0) }'
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

ROOTS=()
GAME_VERSION="$DEFAULT_GAME_VERSION"
SIMULATIONS_ROOT=""
CHECKPOINT_MODE="$DEFAULT_CHECKPOINT_MODE"
MAP_FILTER="map_*"
TRAINING_GLOB="Training_*"
SYNERGY_FILTER=""
SPECIALIZATION_FILTER=""
ABILITIES_FILTER=""
OVERRIDE_ABILITIES=""
TRAININGS_FILE=""
DRY_RUN=false
USER_LOG_DIR=""
KEEP_LOGS=false
OUTPUT_SUBFOLDER=""
SIMULATION_ARGS=()

require_value() {
    if [[ $2 -lt 2 ]]; then
        echo "Error: $1 requires a value"
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)             require_value "$1" $#; ROOTS+=("${2%/}"); shift 2 ;;
        --game_version)     require_value "$1" $#; GAME_VERSION="$2"; shift 2 ;;
        --simulations_root) require_value "$1" $#; SIMULATIONS_ROOT="$2"; shift 2 ;;
        --checkpoint_mode)  require_value "$1" $#; CHECKPOINT_MODE="$2"; shift 2 ;;
        --map_filter)       require_value "$1" $#; MAP_FILTER="$2"; shift 2 ;;
        --training_glob)    require_value "$1" $#; TRAINING_GLOB="$2"; shift 2 ;;
        --synergy)          require_value "$1" $#; SYNERGY_FILTER="$2"; shift 2 ;;
        --specialization)   require_value "$1" $#; SPECIALIZATION_FILTER="$2"; shift 2 ;;
        --abilities)        require_value "$1" $#; ABILITIES_FILTER="$2"; shift 2 ;;
        --trainings)        require_value "$1" $#; TRAININGS_FILE="$2"; shift 2 ;;
        --checkpoint)       require_value "$1" $#; CHECKPOINT_MODE="hardcoded"; HARDCODED_CHECKPOINTS=("$2"); shift 2 ;;
        --override_abilities) require_value "$1" $#; OVERRIDE_ABILITIES="$2"; shift 2 ;;
        --output_subfolder) require_value "$1" $#; OUTPUT_SUBFOLDER="${2%/}"
                            SIMULATION_ARGS+=("$1" "$OUTPUT_SUBFOLDER"); shift 2 ;;   # also forwarded
        --log_dir)          require_value "$1" $#; USER_LOG_DIR="${2%/}"; shift 2 ;;
        --keep_logs)        KEEP_LOGS=true; shift ;;
        --dry_run)          DRY_RUN=true; shift ;;
        -h|--help)          print_usage; exit 0 ;;
        *)                  SIMULATION_ARGS+=("$1"); shift ;;
    esac
done

[[ ${#ROOTS[@]} -eq 0 ]] && ROOTS=("$DEFAULT_ROOT")

# Default output root of one search root: <folder above the init folder>/simulations,
# e.g. .../classic_collision/simulations
default_simulations_root() {
    local init_dir="$1"
    while [[ "$init_dir" == */* && "$(basename "$init_dir")" != *_init && "$init_dir" != "/" ]]; do
        init_dir="$(dirname "$init_dir")"
    done
    if [[ "$(basename "$init_dir")" == *_init ]]; then
        echo "$(dirname "$init_dir")/simulations"
    else
        echo "$(dirname "$1")/simulations"
    fi
}
if [[ -z "$SIMULATIONS_ROOT" ]]; then
    SIMULATIONS_ROOT="$(default_simulations_root "${ROOTS[0]}")"
    for root in "${ROOTS[@]}"; do
        if [[ "$(default_simulations_root "$root")" != "$SIMULATIONS_ROOT" ]]; then
            echo "Error: the roots would write to different simulations folders; pass --simulations_root."
            exit 1
        fi
    done
fi

if [[ "$CHECKPOINT_MODE" != "found" && "$CHECKPOINT_MODE" != "hardcoded" ]]; then
    echo "Error: invalid --checkpoint_mode '$CHECKPOINT_MODE'. Use 'found' or 'hardcoded'."
    exit 1
fi
for _pair in "--synergy:$SYNERGY_FILTER" "--specialization:$SPECIALIZATION_FILTER"; do
    if [[ -n "${_pair#*:}" ]] && ! is_number "${_pair#*:}"; then
        echo "Error: ${_pair%%:*} must be a number, e.g. 0 or 1.35 (got '${_pair#*:}')"
        exit 1
    fi
done
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
# Where the simulations are expected to land: <SIMULATIONS_ROOT>/<init>[/<subfolder>]
INIT_NAME=""
IFS='/' read -ra _parts <<< "${ROOTS[0]}"
for part in "${_parts[@]}"; do
    [[ "$part" == *_init ]] && INIT_NAME="$part"
done
EXPECTED_OUTPUT="${SIMULATIONS_ROOT%/}${INIT_NAME:+/$INIT_NAME}${OUTPUT_SUBFOLDER:+/$OUTPUT_SUBFOLDER}"

for root in "${ROOTS[@]}"; do
    if [[ ! -d "$root" ]]; then
        echo "Error: root folder does not exist: $root"
        exit 1
    fi
done
if [[ ! -f "$SIMULATION_SCRIPT" ]]; then
    echo "Error: simulation script not found: $SIMULATION_SCRIPT"
    exit 1
fi

RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sim_launch_$(date +%Y%m%d_%H%M%S)_XXXXXX")" || {
    echo "Error: could not create a temporary folder"; exit 1; }
CHECKPOINT_CONFIG_DIR="${RUN_DIR}/checkpoints"
LOG_DIR="${USER_LOG_DIR:-${RUN_DIR}/logs}"
mkdir -p "$CHECKPOINT_CONFIG_DIR" "$LOG_DIR" || exit 1

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
PROBLEM_PATHS=()     # trainings that cannot be simulated (shown by --dry_run)
PROBLEM_REASONS=()
NUM_CHECKED=0        # trainings that passed the filters

# Records a training that cannot be simulated; prints a warning only in a real launch
report_problem() {
    PROBLEM_PATHS+=("$1")
    PROBLEM_REASONS+=("$2")
    NUM_SKIPPED=$((NUM_SKIPPED + 1))
    [[ "$DRY_RUN" == false ]] && echo "Warning: skipping ($2): $1"
}

while IFS= read -r -d '' TRAINING_PATH; do
    TRAINING_PATH="${TRAINING_PATH%/}"
    MAP_DIR="$(find_map_dir "$TRAINING_PATH")"
    if [[ -z "$MAP_DIR" ]]; then
        NUM_CHECKED=$((NUM_CHECKED + 1))
        report_problem "$TRAINING_PATH" "no map_* folder in path"
        continue
    fi
    MAP_NR="${MAP_DIR#map_}"
    REL_PATH="${MAP_DIR}${TRAINING_PATH##*/"$MAP_DIR"}"   # map_*/.../Training_*
    LABEL="$(sanitize_for_filename "$REL_PATH")"

    # shellcheck disable=SC2053
    if [[ "$MAP_DIR" != $MAP_FILTER ]]; then
        continue
    fi
    # shellcheck disable=SC2053
    if [[ "${REL_PATH##*/}" != $TRAINING_GLOB ]]; then
        continue
    fi
    if [[ -n "$SYNERGY_FILTER" ]] && \
       ! same_number "$(path_level_value "$REL_PATH" synergy)" "$SYNERGY_FILTER"; then
        continue
    fi
    if [[ -n "$SPECIALIZATION_FILTER" ]] && \
       ! same_number "$(path_level_value "$REL_PATH" specialized)" "$SPECIALIZATION_FILTER"; then
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

    NUM_CHECKED=$((NUM_CHECKED + 1))

    if [[ ! -f "$TRAINING_PATH/config.txt" ]]; then
        report_problem "$REL_PATH" "no config.txt"
        continue
    fi

    if ! ABILITIES="$(read_abilities "$TRAINING_PATH/config.txt" 2>/dev/null)"; then
        report_problem "$REL_PATH" "could not read WALKING_SPEEDS/CUTTING_SPEEDS from config.txt"
        continue
    fi
    if [[ -n "$ABILITIES_FILTER" && "$ABILITIES" != "$ABILITIES_FILTER" ]]; then
        NUM_CHECKED=$((NUM_CHECKED - 1))
        continue
    fi

    CHECKPOINTS=()
    CKPT_PROBLEMS=()
    if [[ "$CHECKPOINT_MODE" == "found" ]]; then
        while IFS= read -r checkpoint_value; do
            [[ -z "$checkpoint_value" ]] && continue
            if [[ -n "$(ls -A "$TRAINING_PATH/checkpoint_${checkpoint_value}" 2>/dev/null)" ]]; then
                CHECKPOINTS+=("$checkpoint_value")
            else
                CKPT_PROBLEMS+=("checkpoint_${checkpoint_value} empty")
            fi
        done < <(discover_checkpoints_in_training "$TRAINING_PATH")
        [[ ${#CHECKPOINTS[@]} -eq 0 && ${#CKPT_PROBLEMS[@]} -eq 0 ]] && CKPT_PROBLEMS+=("no checkpoint_* folders")
    else
        for checkpoint_value in "${HARDCODED_CHECKPOINTS[@]}"; do
            ckpt_dir="$TRAINING_PATH/checkpoint_${checkpoint_value}"
            if [[ ! -d "$ckpt_dir" ]]; then
                CKPT_PROBLEMS+=("checkpoint_${checkpoint_value} missing")
            elif [[ -z "$(ls -A "$ckpt_dir" 2>/dev/null)" ]]; then
                CKPT_PROBLEMS+=("checkpoint_${checkpoint_value} empty")
            else
                CHECKPOINTS+=("$checkpoint_value")
            fi
        done
    fi

    if [[ ${#CKPT_PROBLEMS[@]} -gt 0 ]]; then
        # a training missing ANY requested checkpoint is listed (and, if none is usable, skipped)
        if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
            report_problem "$REL_PATH" "$(IFS=';'; echo "${CKPT_PROBLEMS[*]}" | sed 's/;/, /g')"
            continue
        fi
        PROBLEM_PATHS+=("$REL_PATH")
        PROBLEM_REASONS+=("$(IFS=';'; echo "${CKPT_PROBLEMS[*]}" | sed 's/;/, /g') (other checkpoints OK)")
        [[ "$DRY_RUN" == false ]] && echo "Warning: ${PROBLEM_REASONS[-1]}: $REL_PATH"
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
done < <(find "${ROOTS[@]}" -type d -name 'simulations' -prune -o \
              -type d -name 'Training_*' -prune -print0 | sort -z -V -u)

TOTAL_JOBS=${#JOB_MAPS[@]}

if [[ ${#SEL_NAMES[@]} -gt 0 ]]; then
    SELECTION_OK=true
    for k in "${!SEL_NAMES[@]}"; do
        n_found="$(printf '%s' "${SEL_FOUND[$k]}" | grep -c .)"
        if [[ "$n_found" -eq 0 ]]; then
            echo "Error: [${SEL_GLOBS[$k]}] ${SEL_NAMES[$k]} not found under ${ROOTS[*]}"
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

if [[ "$DRY_RUN" == true ]]; then
    rm -rf "$RUN_DIR"
    [[ -n "$USER_LOG_DIR" ]] && rmdir "$USER_LOG_DIR" 2>/dev/null
    if [[ "$CHECKPOINT_MODE" == "hardcoded" ]]; then
        ckpt_text="checkpoint_${HARDCODED_CHECKPOINTS[*]}"
    else
        ckpt_text="any checkpoint_* folder"
    fi
    echo "=================================================="
    echo "Dry run: trainings without a usable ${ckpt_text}"
    printf 'Root: %s\n' "${ROOTS[@]}"
    echo "Trainings checked (after filters): $NUM_CHECKED"
    echo "Trainings OK: $((NUM_CHECKED - ${#PROBLEM_PATHS[@]}))"
    echo "Trainings with problems: ${#PROBLEM_PATHS[@]}"
    echo "=================================================="
    if [[ ${#PROBLEM_PATHS[@]} -eq 0 ]]; then
        echo "All trainings have a usable ${ckpt_text}."
    else
        for i in "${!PROBLEM_PATHS[@]}"; do
            echo "${PROBLEM_PATHS[$i]}  ->  ${PROBLEM_REASONS[$i]}"
        done
    fi
    exit 0
fi

echo "=================================================="
echo "Trainings-tree parallel launcher"
printf 'Root: %s\n' "${ROOTS[@]}"
echo "Map filter: $MAP_FILTER"
echo "Training filter: $TRAINING_GLOB"
echo "Synergy filter: ${SYNERGY_FILTER:-none}"
echo "Specialization filter: ${SPECIALIZATION_FILTER:-none}"
echo "Trainings list: ${TRAININGS_FILE:-all} (${#SEL_NAMES[@]} entries)"
echo "Abilities filter (trained): ${ABILITIES_FILTER:-none}"
echo "Simulated abilities: ${OVERRIDE_ABILITIES:-same as training}"
echo "Game version: $GAME_VERSION"
echo "Simulations root: $SIMULATIONS_ROOT"
echo "Expected output folder: $EXPECTED_OUTPUT/map_*/..."
if [[ "$CHECKPOINT_MODE" == "hardcoded" ]]; then
    echo "Checkpoints: ${HARDCODED_CHECKPOINTS[*]}"
else
    echo "Checkpoint mode: found (every checkpoint_* folder)"
fi
echo "Simulations per training/checkpoint: $NUM_SIMULATIONS"
echo "Max parallel: $MAX_PARALLEL"
echo "Trainings used: $NUM_TRAININGS (skipped: $NUM_SKIPPED)"
echo "Total jobs: $TOTAL_JOBS"
echo "Checkpoint files dir: $CHECKPOINT_CONFIG_DIR (temporary)"
if [[ -n "$USER_LOG_DIR" || "$KEEP_LOGS" == true ]]; then
    echo "Log dir: $LOG_DIR (logs kept)"
else
    echo "Log dir: $LOG_DIR (temporary; only logs of failed runs are kept)"
fi
echo "Forwarded simulation args: ${SIMULATION_ARGS[*]}"
echo "=================================================="

if [[ "$TOTAL_JOBS" -eq 0 ]]; then
    echo "Error: no simulations to launch."
    rm -rf "$RUN_DIR"
    exit 1
fi


# -----------------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------------
LAUNCH_MARKER="$(mktemp "${TMPDIR:-/tmp}/launch_marker.XXXXXX")"
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

    # the exit status is written next to the log so that failed runs can be found at the end
    ( nohup "${CMD[@]}" > "${JOB_LOGS[$i]}" 2>&1; echo "$?" > "${JOB_LOGS[$i]}.status" ) &
    PID=$!
    PIDS+=("$PID")

    echo "[$((i + 1))/$TOTAL_JOBS] Launched ${JOB_DESC[$i]} (PID: $PID)"

    # Small gap so runs of the same training don't get the same timestamp
    sleep 2
done

echo ""
echo "All $TOTAL_JOBS simulations launched. Waiting for remaining jobs to finish..."
wait

echo ""
echo "=================================================="
echo "All simulations completed."
echo "Simulations: $EXPECTED_OUTPUT"

# Check that the new simulation_* folders landed where expected
NEW_SIMS="$(find "$SIMULATIONS_ROOT" -type d -name 'simulation_*' -newer "$LAUNCH_MARKER" 2>/dev/null)"
MISPLACED="$(printf '%s\n' "$NEW_SIMS" | grep . | grep -v "^${EXPECTED_OUTPUT}/map_")"
echo "New simulation folders: $(printf '%s\n' "$NEW_SIMS" | grep -c .)"
if [[ -n "$MISPLACED" ]]; then
    echo "WARNING: $(printf '%s\n' "$MISPLACED" | grep -c .) simulation folders are NOT under $EXPECTED_OUTPUT, e.g.:"
    printf '%s\n' "$MISPLACED" | head -5 | sed 's/^/    /'
fi
rm -f "$LAUNCH_MARKER"

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
# Failed runs: non-zero exit status or 'Error during simulation' in the log
FAILED_LOGS=()
for log in "${JOB_LOGS[@]}"; do
    status="$(cat "${log}.status" 2>/dev/null)"
    if [[ "$status" != "0" ]] || grep -q "Error during simulation" "$log" 2>/dev/null; then
        FAILED_LOGS+=("$log")
    fi
done
echo "Failed runs: ${#FAILED_LOGS[@]} of $TOTAL_JOBS"
if [[ ${#FAILED_LOGS[@]} -gt 0 ]]; then
    printf '    %s\n' "${FAILED_LOGS[@]}"
fi

# Clean up: checkpoint files and status files always; logs of successful runs unless kept
rm -rf "$CHECKPOINT_CONFIG_DIR"
rm -f "${LOG_DIR}"/*.status
if [[ -z "$USER_LOG_DIR" && "$KEEP_LOGS" == false ]]; then
    for log in "${JOB_LOGS[@]}"; do
        printf '%s\n' "${FAILED_LOGS[@]}" | grep -qxF "$log" || rm -f "$log"
    done
    rmdir "$LOG_DIR" "$RUN_DIR" 2>/dev/null
    if [[ ${#FAILED_LOGS[@]} -gt 0 ]]; then
        echo "Logs of the failed runs kept in: $LOG_DIR"
    else
        echo "No failed runs: per-simulation logs deleted."
    fi
else
    rmdir "$RUN_DIR" 2>/dev/null
    echo "Logs: $LOG_DIR"
fi
echo "=================================================="