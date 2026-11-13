#!/usr/bin/env bash
# Delete the 58 problematic simulation folders.
# Usage:
#   bash delete_problematic_simulations.sh          # dry run: only shows what would be deleted
#   bash delete_problematic_simulations.sh --yes    # actually deletes

ROOT="/data/samuel_lozano/cooked/classic_collision/simulations/empty_init"
DELETE=false
[[ "$1" == "--yes" ]] && DELETE=true

# Folder names for zero values may be written "0" or "0.0" on disk: try both.
resolve() {
    local rel="$1" cand
    for cand in "$rel" \
        "$(sed -E 's#/synergy_0/#/synergy_0.0/#' <<< "$rel")" \
        "$(sed -E 's#/specialized_0/#/specialized_0.0/#' <<< "$rel")" \
        "$(sed -E 's#/synergy_0/#/synergy_0.0/#; s#/specialized_0/#/specialized_0.0/#' <<< "$rel")"; do
        [[ -d "$ROOT/$cand" ]] && { echo "$ROOT/$cand"; return 0; }
    done
    return 1
}

SIMS=(
  "map_a1_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.4_0.4/Training_2026-05-24_11-38-17/checkpoint_750/simulation_2026_11_06-02_32_07"
  "map_a1_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.8_0.8/Training_2026-05-25_03-54-40/checkpoint_750/simulation_2026_11_06-06_27_10"
  "map_a2_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.6_0.6/Training_2026-05-24_17-37-33/checkpoint_750/simulation_2026_11_06-01_34_47"
  "map_a2_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_1.0_1.0/Training_2026-05-26_16-15-20/checkpoint_750/simulation_2026_11_06-08_25_53"
  "map_a3_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.3_0.3/Training_2026-05-25_04-06-05/checkpoint_750/simulation_2026_11_06-02_29_46"
  "map_a3_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.5_0.5/Training_2026-05-25_13-28-39/checkpoint_750/simulation_2026_11_06-04_48_06"
  "map_a4_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.1_0.1/Training_2026-05-26_10-22-22/checkpoint_750/simulation_2026_11_06-07_53_32"
  "map_a4_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.3_0.3/Training_2026-05-25_20-19-24/checkpoint_750/simulation_2026_11_06-04_58_19"
  "map_a4_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.4_0.4/Training_2026-05-25_09-19-21/checkpoint_750/simulation_2026_11_06-02_14_07"
  "map_a4_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.7_0.7/Training_2026-05-25_19-49-11/checkpoint_750/simulation_2026_11_06-03_58_53"
  "map_a5_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.2_0.2/Training_2026-05-26_12-30-30/checkpoint_750/simulation_2026_11_06-07_49_55"
  "map_a5_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.9_0.9/Training_2026-05-26_08-51-47/checkpoint_750/simulation_2026_11_06-06_06_23"
  "map_baseline_division_of_labor_large/synergy_0/specialized_0.25/abilities_0.4_0.2/Training_2026-06-25_14-03-54/checkpoint_750/simulation_2026_11_05-22_35_09"
  "map_baseline_division_of_labor_large/synergy_0/specialized_0.25/abilities_0.4_0.2/Training_2026-06-25_14-03-54/checkpoint_750/simulation_2026_11_05-22_37_59"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0/abilities_0.4_0.2/Training_2026-06-25_14-01-28/checkpoint_750/simulation_2026_11_07-05_27_44"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.1_0.5/Training_2026-05-17_00-34-20/checkpoint_750/simulation_2026_11_05-23_40_24"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.1_0.7/Training_2026-05-21_08-23-59/checkpoint_750/simulation_2026_11_07-01_26_42"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.1_0.9/Training_2026-05-17_01-50-52/checkpoint_750/simulation_2026_11_05-23_52_29"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.2_0.9/Training_2026-05-19_06-38-51/checkpoint_750/simulation_2026_11_06-11_12_24"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.3_0.1/Training_2026-05-17_06-57-53/checkpoint_750/simulation_2026_11_06-00_13_56"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.3_0.7/Training_2026-06-05_12-16-07/checkpoint_750/simulation_2026_11_07-02_11_41"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.4_0.1/Training_2026-06-05_12-14-42/checkpoint_750/simulation_2026_11_07-01_47_30"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.4_0.2/Training_2026-05-19_14-11-42/checkpoint_750/simulation_2026_11_06-12_11_05"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.4_0.6/Training_2026-05-17_13-50-51/checkpoint_750/simulation_2026_11_06-01_08_43"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.4_1.0/Training_2026-05-20_17-43-41/checkpoint_750/simulation_2026_11_07-00_10_31"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.5_0.4/Training_2026-05-17_16-40-15/checkpoint_750/simulation_2026_11_06-02_34_07"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.6_0.9/Training_2026-05-18_00-43-17/checkpoint_750/simulation_2026_11_06-08_14_28"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.7_0.2/Training_2026-05-18_01-43-49/checkpoint_750/simulation_2026_11_09-01_49_57"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.7_0.5/Training_2026-05-20_04-34-51/checkpoint_750/simulation_2026_11_06-18_55_34"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.7_0.6/Training_2026-06-06_04-34-45/checkpoint_750/simulation_2026_11_07-03_30_18"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.7_0.8/Training_2026-05-20_06-06-03/checkpoint_750/simulation_2026_11_06-19_33_51"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.8_1.0/Training_2026-05-20_11-09-40/checkpoint_750/simulation_2026_11_06-22_43_46"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_1.0_0.3/Training_2026-06-06_16-25-03/checkpoint_750/simulation_2026_11_07-04_41_02"
  "map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_1.0_0.6/Training_2026-05-18_19-33-41/checkpoint_750/simulation_2026_11_06-10_22_51"
  "map_d1_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.3_0.3/Training_2026-05-22_13-21-18/checkpoint_750/simulation_2026_11_06-14_58_10"
  "map_d1_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.8_0.8/Training_2026-05-23_07-56-36/checkpoint_750/simulation_2026_11_06-20_05_58"
  "map_d2_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.4_0.4/Training_2026-05-23_04-09-24/checkpoint_750/simulation_2026_11_06-21_39_25"
  "map_d2_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.5_0.5/Training_2026-05-23_03-58-14/checkpoint_750/simulation_2026_11_06-21_23_22"
  "map_d3_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.1_0.1/Training_2026-05-22_22-36-43/checkpoint_750/simulation_2026_11_06-21_47_23"
  "map_d3_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.2_0.2/Training_2026-05-22_09-43-32/checkpoint_750/simulation_2026_11_06-18_48_54"
  "map_d4_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.2_0.2/Training_2026-05-23_03-20-06/checkpoint_750/simulation_2026_11_06-22_48_04"
  "map_d4_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.2_0.2/Training_2026-05-23_03-20-06/checkpoint_750/simulation_2026_11_06-22_56_00"
  "map_d4_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.5_0.5/Training_2026-05-23_03-07-04/checkpoint_750/simulation_2026_11_06-22_06_55"
  "map_d4_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_1.0_1.0/Training_2026-05-22_19-30-10/checkpoint_750/simulation_2026_11_06-17_11_54"
  "map_d5_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.1_0.1/Training_2026-05-22_22-12-44/checkpoint_750/simulation_2026_11_06-20_37_38"
  "map_encouraged_division_of_labor_large/synergy_0/specialized_0/abilities_0.4_0.2/Training_2026-06-25_14-01-11/checkpoint_750/simulation_2026_11_05-22_51_29"
  "map_encouraged_division_of_labor_large/synergy_0/specialized_0/abilities_0.4_0.2/Training_2026-06-25_14-03-37/checkpoint_750/simulation_2026_11_05-23_18_23"
  "map_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.1_0.4/Training_2026-05-17_13-58-05/checkpoint_750/simulation_2026_11_06-21_50_32"
  "map_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.1_0.7/Training_2026-05-15_03-03-50/checkpoint_750/simulation_2026_11_06-09_56_44"
  "map_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.2_0.1/Training_2026-05-17_18-42-40/checkpoint_750/simulation_2026_11_06-23_12_41"
  "map_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.2_0.2/Training_2026-05-15_03-04-52/checkpoint_750/simulation_2026_11_06-10_15_28"
  "map_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.3_0.9/Training_2026-05-18_06-21-47/checkpoint_750/simulation_2026_11_07-00_09_33"
  "map_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.5_0.4/Training_2026-05-16_02-05-32/checkpoint_750/simulation_2026_11_06-11_31_38"
  "map_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.7_1.0/Training_2026-05-14_04-38-30/checkpoint_750/simulation_2026_11_06-08_41_03"
  "map_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.8_0.4/Training_2026-05-16_18-43-16/checkpoint_750/simulation_2026_11_06-13_58_13"
  "map_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.9_0.5/Training_2026-05-17_02-00-07/checkpoint_750/simulation_2026_11_06-16_38_11"
  "map_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.9_0.6/Training_2026-05-19_15-57-02/checkpoint_750/simulation_2026_11_07-03_24_42"
  "map_encouraged_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.9_1.0/Training_2026-05-14_04-48-32/checkpoint_750/simulation_2026_11_06-08_51_46"
)

found=0; missing=0; deleted=0
for rel in "${SIMS[@]}"; do
    if dir=$(resolve "$rel"); then
        found=$((found + 1))
        if $DELETE; then
            rm -rf -- "$dir" && { echo "DELETED  $dir"; deleted=$((deleted + 1)); } \
                             || echo "FAILED   $dir"
        else
            echo "WOULD DELETE  $dir"
        fi
    else
        echo "NOT FOUND     $ROOT/$rel"
        missing=$((missing + 1))
    fi
done

echo
echo "Listed: ${#SIMS[@]} | Found: $found | Not found: $missing | Deleted: $deleted"
$DELETE || echo "Dry run only. Re-run with --yes to delete."
