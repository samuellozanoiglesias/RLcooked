#!/bin/bash
# =============================================================================
# Launches NUM_SIMULATIONS (default 20) simulations for each training listed below
# (maps d4, d3, d2), using each training's own speeds from its config.txt.
#
# Each [GLOB] line restricts the trainings below it to map folders matching GLOB,
# because the same Training_* name can exist in several maps. The launcher aborts
# if an entry is missing or matches more than one folder (and prints the paths);
# then write the exact map name in the brackets, e.g. [map_d3_xxx].
#
# Usage:
#   bash launch_d_maps.sh [MAX_PARALLEL] [extra launcher/simulation args]
#
# Examples:
#   bash launch_d_maps.sh 5 --dry_run          # check the job list first
#   nohup bash launch_d_maps.sh 5 --enable_video false > log_d_maps.out 2>&1 &
# =============================================================================

NUM_SIMULATIONS=20
CHECKPOINT="753"            # checkpoint_753 (same as the figures); use "final" for checkpoint_final
ROOT="/data/samuel_lozano/cooked/classic_collision/empty_init"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/launch_simulations_parallel.sh"

TRAININGS_FILE="$(mktemp "${TMPDIR:-/tmp}/d_maps_trainings.XXXXXX")"
trap 'rm -f "$TRAININGS_FILE"' EXIT

cat > "$TRAININGS_FILE" <<'LIST'
[map_d4_*]
Training_2026-05-22_08-20-40

[map_d3_*]
Training_2026-05-23_03-31-15
Training_2026-05-22_19-58-53
Training_2026-05-22_20-08-23
Training_2026-05-22_20-10-01
Training_2026-05-22_20-25-43
Training_2026-05-22_20-50-24
Training_2026-05-22_21-24-28
Training_2026-05-22_21-42-08
Training_2026-05-22_22-36-43

[map_d2_*]
Training_2026-05-23_07-52-55
LIST

bash "$LAUNCHER" "$NUM_SIMULATIONS" "$@" \
    --root "$ROOT" \
    --trainings "$TRAININGS_FILE" \
    --checkpoint "$CHECKPOINT"
