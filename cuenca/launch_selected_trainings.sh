#!/bin/bash
# =============================================================================
# Launches NUM_SIMULATIONS (default 20) simulations at checkpoint 750 for every
# Training_* folder found under each folder in ROOTS (at any depth, e.g.
# map_*/synergy_*/specialized_*/Training_*), using each training's own config.txt.
# Results go to ${SIMULATIONS_ROOT}/empty_init/${OUTPUT_SUBFOLDER}/map_*/..., i.e.
#     /data/samuel_lozano/cooked/classic_collision/simulations/empty_init/
#
# Optional filters (environment variables, empty = no filter):
#   SYNERGY=0            only trainings inside synergy_0
#   SPECIALIZATION=0.25  only trainings inside specialized_0.25
#   ABILITIES=0.4_0.2    only trainings whose config.txt has these speeds
# To run only some trainings, put their names in the LIST block below
# (empty list = all trainings).
#
# Usage:
#   ./launch_selected_trainings.sh [MAX_PARALLEL] [extra launcher/simulation args]
#
# Examples:
#   ./launch_selected_trainings.sh 5 --dry_run          # check the job list first
#   nohup ./launch_selected_trainings.sh 5 --enable_video false > log_selected_trainings.out 2>&1 &
# =============================================================================

NUM_SIMULATIONS="${NUM_SIMULATIONS:-3}"
CHECKPOINT="${CHECKPOINT:-750}"
BASE="${BASE:-/data/samuel_lozano/cooked/classic_collision/empty_init}"  # -> map_*/synergy_*/specialized_*/Training_*/config.txt
SIMULATIONS_ROOT="${SIMULATIONS_ROOT:-/data/samuel_lozano/cooked/classic_collision/simulations}"
#OUTPUT_SUBFOLDER="${OUTPUT_SUBFOLDER:-with_pretraining}"   # -> simulations/empty_init/with_pretraining/map_*/...

ROOTS=(
    "${BASE}/map_d5_encouraged_division_of_labor_large"
    "${BASE}/map_d4_encouraged_division_of_labor_large"
    "${BASE}/map_d3_encouraged_division_of_labor_large"
    "${BASE}/map_d2_encouraged_division_of_labor_large"
    "${BASE}/map_d1_encouraged_division_of_labor_large"
    "${BASE}/map_a1_encouraged_division_of_labor_large"
    "${BASE}/map_a2_encouraged_division_of_labor_large"
    "${BASE}/map_a3_encouraged_division_of_labor_large"
    "${BASE}/map_a4_encouraged_division_of_labor_large"
    "${BASE}/map_a5_encouraged_division_of_labor_large"
    "${BASE}/map_baseline_division_of_labor_large"
    "${BASE}/map_encouraged_division_of_labor_large"
)

SYNERGY="${SYNERGY:-1.35}"
SPECIALIZATION="${SPECIALIZATION:-0.25}"
ABILITIES="${ABILITIES:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/launch_simulations_parallel.sh"

ARGS=(--checkpoint "$CHECKPOINT" --simulations_root "$SIMULATIONS_ROOT")
for root in "${ROOTS[@]}"; do
    ARGS+=(--root "$root")
done
[[ -n "$SYNERGY" ]]          && ARGS+=(--synergy "$SYNERGY")
[[ -n "$SPECIALIZATION" ]]   && ARGS+=(--specialization "$SPECIALIZATION")
[[ -n "$ABILITIES" ]]        && ARGS+=(--abilities "$ABILITIES")
[[ -n "$OUTPUT_SUBFOLDER" ]] && ARGS+=(--output_subfolder "$OUTPUT_SUBFOLDER")

TRAININGS_FILE="$(mktemp "${TMPDIR:-/tmp}/selected_trainings.XXXXXX")"
trap 'rm -f "$TRAININGS_FILE"' EXIT

# Optional: [GLOB] lines and Training_* names to restrict the run (see launcher --trainings).
cat > "$TRAININGS_FILE" <<'LIST'
[map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25]
Training_2026-05-18_07-18-07
Training_2026-05-19_14-09-31
[map_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25]
Training_2026-05-13_09-49-05
Training_2026-05-20_01-03-54
Training_2026-05-12_23-01-18
Training_2026-05-20_00-44-42
LIST

if grep -q '^[[:space:]]*Training_' "$TRAININGS_FILE"; then
    ARGS+=(--trainings "$TRAININGS_FILE")
fi

bash "$LAUNCHER" "$NUM_SIMULATIONS" "$@" "${ARGS[@]}"