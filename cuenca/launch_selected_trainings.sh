#!/bin/bash
# =============================================================================
# Launches NUM_SIMULATIONS (default 20) simulations for each training listed below,
# using each training's own speeds from its config.txt.
#
# Each [GLOB] line restricts the trainings below it to map folders matching GLOB,
# because the same Training_* name can exist in several maps. The launcher aborts
# if an entry is missing or matches more than one folder (and prints the paths),
# so tighten the GLOB (e.g. [map_encouraged_division_of_labor_large]) if needed.
#
# Usage:
#   ./launch_selected_trainings.sh [MAX_PARALLEL] [extra launcher/simulation args]
#
# Examples:
#   ./launch_selected_trainings.sh 5 --dry_run          # check the job list first
#   nohup ./launch_selected_trainings.sh 5 --enable_video false > log_selected_trainings.out 2>&1 &
# =============================================================================

NUM_SIMULATIONS=20
ROOT="/data/samuel_lozano/cooked/classic_collision/empty_init"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/launch_simulations_parallel.sh"

TRAININGS_FILE="$(mktemp "${TMPDIR:-/tmp}/selected_trainings.XXXXXX")"
trap 'rm -f "$TRAININGS_FILE"' EXIT

cat > "$TRAININGS_FILE" <<'LIST'
[*encouraged*]
Training_2026-06-06_09-55-50
Training_2026-06-06_07-16-02
Training_2026-06-06_00-53-50
Training_2026-06-05_18-37-24
Training_2026-06-05_12-15-31
Training_2026-05-21_06-04-43
Training_2026-05-21_01-33-23
Training_2026-05-20_10-14-10

[*baseline*]
Training_2026-06-05_12-16-42
Training_2026-06-05_12-16-24
Training_2026-06-05_12-16-07
Training_2026-06-05_12-15-48
Training_2026-06-05_12-15-32
Training_2026-06-05_12-15-16

Training_2026-05-21_08-32-34

Training_2026-05-21_08-29-34
Training_2026-05-21_08-24-32
Training_2026-05-21_08-23-59
Training_2026-05-21_08-10-26
Training_2026-05-21_08-06-53
Training_2026-05-21_07-57-23
Training_2026-05-21_07-56-21
Training_2026-05-21_07-55-41
LIST

bash "$LAUNCHER" "$NUM_SIMULATIONS" "$@" --root "$ROOT" --trainings "$TRAININGS_FILE"