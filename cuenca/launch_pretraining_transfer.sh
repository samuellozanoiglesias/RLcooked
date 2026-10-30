#!/bin/bash
# =============================================================================
# Simulates pretrained agents (checkpoint_753) with other abilities.
#
# Agents are loaded from the trainings listed below; each one is simulated with
# every value in SIMULATED_ABILITIES (walking speed of ai_rl_1 _ cutting speed of ai_rl_2).
# Output:
#   /data/samuel_lozano/cooked/classic_collision/simulations/empty_init/with_pretraining/
#       map_*/synergy_*/specialized_*/abilities_<W>_<C>/Training_*/checkpoint_753/simulation_<timestamp>/
#
# Usage:
#   bash launch_pretraining_transfer.sh [MAX_PARALLEL] [extra launcher/simulation args]
#
# Examples:
#   bash launch_pretraining_transfer.sh 5 --dry_run
#   nohup bash launch_pretraining_transfer.sh 5 --enable_video false > log_pretraining_transfer.out 2>&1 &
# =============================================================================

NUM_SIMULATIONS=20
CHECKPOINT="753"
SIMULATED_ABILITIES="0.1_0.1,0.2_1.0"
OUTPUT_SUBFOLDER="with_pretraining"
ROOT="/data/samuel_lozano/cooked/classic_collision/empty_init"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SCRIPT_DIR}/launch_simulations_parallel.sh"

TRAININGS_FILE="$(mktemp "${TMPDIR:-/tmp}/pretraining_trainings.XXXXXX")"
trap 'rm -f "$TRAININGS_FILE"' EXIT

cat > "$TRAININGS_FILE" <<'LIST'
[map_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25]
Training_2026-06-06_00-56-58

[map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25]
Training_2026-05-18_23-04-38
LIST

bash "$LAUNCHER" "$NUM_SIMULATIONS" "$@" \
    --root "$ROOT" \
    --trainings "$TRAININGS_FILE" \
    --checkpoint "$CHECKPOINT" \
    --override_abilities "$SIMULATED_ABILITIES" \
    --output_subfolder "$OUTPUT_SUBFOLDER"
