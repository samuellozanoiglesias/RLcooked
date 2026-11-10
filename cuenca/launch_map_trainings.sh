#!/bin/bash
# =============================================================================
# Launches NUM_SIMULATIONS (default 50) simulations at checkpoint 750 for EVERY
# Training_* folder of ONE map (all synergy_* / specialized_* folders below it),
# with at most MAX_PARALLEL (default 20) simulations running at once.
# Each training is simulated with its own config.txt (its own abilities).
#
# Results go to ${SIMULATIONS_ROOT}/empty_init/[${OUTPUT_SUBFOLDER}/]map_<MAP>/..., i.e.
#     /data/samuel_lozano/cooked/classic_collision/simulations/empty_init/map_<MAP>/...
#
# Usage:
#   ./launch_map_trainings.sh MAP [--with_pretraining] [extra launcher/simulation args]
#     MAP: map name with or without the 'map_' prefix, e.g.
#          baseline_division_of_labor_large  or  map_baseline_division_of_labor_large
#     --with_pretraining: use the pretrained trainings instead, i.e. read from
#          ${BASE}/with_pretraining/map_<MAP>/...  and write to
#          ${SIMULATIONS_ROOT}/empty_init/with_pretraining/map_<MAP>/...
#          (same as WITH_PRETRAINING=1)
#
# Optional settings (environment variables):
#   NUM_SIMULATIONS=50      simulations per training
#   MAX_PARALLEL=20         simulations running at the same time
#   CHECKPOINT=750          checkpoint_<N> folder used
#   SYNERGY=1.35            only trainings inside synergy_1.35        (empty = all)
#   SPECIALIZATION=0.25     only trainings inside specialized_0.25    (empty = all)
#   ABILITIES=0.4_0.2       only trainings whose config.txt has these speeds (empty = all)
#   WITH_PRETRAINING=1      same as --with_pretraining
#   PRETRAINING_SUBFOLDER=with_pretraining   name of that subfolder (input and output)
#   OUTPUT_SUBFOLDER=NAME   extra folder after empty_init in the output (empty = none;
#                           with --with_pretraining it defaults to PRETRAINING_SUBFOLDER)
#   BASE, SIMULATIONS_ROOT  input / output roots (defaults below)
#
# Examples:
#   ./launch_map_trainings.sh baseline_division_of_labor_large --dry_run     # check the checkpoints first
#   ./launch_map_trainings.sh baseline_division_of_labor_large --with_pretraining --dry_run
#   nohup ./launch_map_trainings.sh baseline_division_of_labor_large --enable_video false > log_map_trainings.out 2>&1 &
#   SYNERGY=1.35 SPECIALIZATION=0.25 nohup ./launch_map_trainings.sh encouraged_division_of_labor_large \
#       --enable_video false > log_map_trainings.out 2>&1 &
# =============================================================================

NUM_SIMULATIONS="${NUM_SIMULATIONS:-30}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
CHECKPOINT="${CHECKPOINT:-750}"
BASE="${BASE:-/data/samuel_lozano/cooked/classic_collision/empty_init}"
SIMULATIONS_ROOT="${SIMULATIONS_ROOT:-/data/samuel_lozano/cooked/classic_collision/simulations}"

SYNERGY="${SYNERGY:-}"
SPECIALIZATION="${SPECIALIZATION:-}"
ABILITIES="${ABILITIES:-}"
OUTPUT_SUBFOLDER="${OUTPUT_SUBFOLDER:-}"
WITH_PRETRAINING="${WITH_PRETRAINING:-1}"
PRETRAINING_SUBFOLDER="${PRETRAINING_SUBFOLDER:-with_pretraining}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/launch_simulations_parallel.sh"

# ---- map ---------------------------------------------------------------------
if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" || "$1" == --* ]]; then
    sed -n '/^# Usage:/,/^# Optional/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | sed '$d'
    echo "Available maps in ${BASE}:"
    find "${BASE%/}" -maxdepth 1 -mindepth 1 -type d -name 'map_*' -printf '    %f\n' 2>/dev/null | sort
    [[ $# -lt 1 || "$1" == --* ]] && exit 1 || exit 0
fi
MAP="${1#map_}"
shift

# --with_pretraining can be given anywhere after MAP; it is not forwarded to the launcher
FORWARD=()
for a in "$@"; do
    if [[ "$a" == "--with_pretraining" ]]; then
        WITH_PRETRAINING=1
    else
        FORWARD+=("$a")
    fi
done
set -- "${FORWARD[@]}"

if [[ "$WITH_PRETRAINING" == 1 || "$WITH_PRETRAINING" == true ]]; then
    TRAININGS_BASE="${BASE%/}/${PRETRAINING_SUBFOLDER}"
    OUTPUT_SUBFOLDER="${OUTPUT_SUBFOLDER:-$PRETRAINING_SUBFOLDER}"
    MODE="WITH pretraining"
else
    TRAININGS_BASE="${BASE%/}"
    MODE="from scratch"
fi
MAP_ROOT="${TRAININGS_BASE}/map_${MAP}"

if [[ ! -d "$MAP_ROOT" ]]; then
    echo "Error: map folder not found: $MAP_ROOT"
    echo "Available maps in ${TRAININGS_BASE}:"
    find "$TRAININGS_BASE" -maxdepth 1 -mindepth 1 -type d -name 'map_*' -printf '    %f\n' 2>/dev/null | sort
    exit 1
fi
if [[ ! -f "$LAUNCHER" ]]; then
    echo "Error: launcher not found: $LAUNCHER"
    exit 1
fi

N_TRAININGS="$(find "$MAP_ROOT" -type d -name 'simulations' -prune -o \
                    -type d -name 'Training_*' -prune -print | wc -l)"
echo "Map: map_${MAP} (${MODE})"
echo "Trainings from: $MAP_ROOT"
echo "Simulations to: ${SIMULATIONS_ROOT%/}/$(basename "${BASE%/}")${OUTPUT_SUBFOLDER:+/$OUTPUT_SUBFOLDER}/map_${MAP}/..."
echo "Training_* folders found under $MAP_ROOT: $N_TRAININGS (before the optional filters)"
echo "Simulations per training: $NUM_SIMULATIONS | max parallel: $MAX_PARALLEL | checkpoint: $CHECKPOINT"

# ---- launcher arguments --------------------------------------------------------
ARGS=(--root "$MAP_ROOT" --checkpoint "$CHECKPOINT" --simulations_root "$SIMULATIONS_ROOT")
[[ -n "$SYNERGY" ]]          && ARGS+=(--synergy "$SYNERGY")
[[ -n "$SPECIALIZATION" ]]   && ARGS+=(--specialization "$SPECIALIZATION")
[[ -n "$ABILITIES" ]]        && ARGS+=(--abilities "$ABILITIES")
[[ -n "$OUTPUT_SUBFOLDER" ]] && ARGS+=(--output_subfolder "$OUTPUT_SUBFOLDER")

# launch_simulations_parallel.sh NUM_SIMULATIONS MAX_PARALLEL [options] [simulation args]
bash "$LAUNCHER" "$NUM_SIMULATIONS" "$MAX_PARALLEL" "${ARGS[@]}" "$@"