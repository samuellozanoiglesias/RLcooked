#!/usr/bin/env python3
"""
Equalize simulation folder counts across all training/checkpoint configurations.

Given a directory containing Training_* folders and checkpoint_* folders inside
each training, the script:

1. Counts simulation_* folders for every (training, checkpoint) configuration.
2. Finds the lowest simulation count across all configurations.
3. Optionally caps that target with --max_simulations.
3. Randomly deletes extra simulation_* folders in configurations above that
   minimum so all configurations end with exactly the same count.

By default, it performs deletions. Use --dry_run to only print what would be
deleted and obtain all counts without modifying files.

Example:
    nohup python3 equalize_number_simulations.py --simulations_dir /data/samuel_lozano/cooked/map_baseline/simulations --dry_run > log_equalize_simulations_dry_run.out 2>&1 &
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List


def iter_training_dirs(simulations_dir: Path) -> List[Path]:
    """Return sorted Training_* directories directly inside simulations root."""
    return sorted(
        [
            path
            for path in simulations_dir.iterdir()
            if path.is_dir() and path.name.startswith("Training_")
        ],
        key=lambda path: path.name,
    )


def iter_checkpoint_dirs(training_dir: Path) -> List[Path]:
    """Return sorted checkpoint_* directories directly inside a training directory."""
    return sorted(
        [
            path
            for path in training_dir.iterdir()
            if path.is_dir() and path.name.startswith("checkpoint_")
        ],
        key=lambda path: path.name,
    )


def iter_simulation_dirs(checkpoint_dir: Path) -> List[Path]:
    """Return sorted simulation_* directories directly inside a checkpoint directory."""
    return sorted(
        [
            path
            for path in checkpoint_dir.iterdir()
            if path.is_dir() and path.name.startswith("simulation_")
        ],
        key=lambda path: path.name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Equalize the number of simulation_* folders across all "
            "(Training_*, checkpoint_*) configurations under --simulations_dir."
        )
    )

    parser.add_argument(
        "--simulations_dir",
        type=str,
        required=True,
        help="Directory containing Training_* folders.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print what would be deleted, without deleting folders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible deletion selection.",
    )
    parser.add_argument(
        "--max_simulations",
        type=int,
        default=None,
        help=(
            "Optional hard cap for final simulations per configuration. "
            "If the natural minimum across configurations is greater than this value, "
            "the script will delete down to this cap instead."
        ),
    )

    return parser.parse_args()


def resolve_configurations(simulations_dir: Path) -> List[dict]:
    """
    Collect all (training, checkpoint) configurations and simulation directories.

    Returns a list of dictionaries with:
        training_dir, checkpoint_dir, simulation_dirs, simulation_count
    """
    if not simulations_dir.exists() or not simulations_dir.is_dir():
        raise ValueError(
            f"--simulations_dir does not exist or is not a directory: {simulations_dir}"
        )

    training_dirs = iter_training_dirs(simulations_dir)
    if not training_dirs:
        raise ValueError(f"No Training_* directories found in: {simulations_dir}")

    configurations: List[dict] = []

    for training_dir in training_dirs:
        checkpoint_dirs = iter_checkpoint_dirs(training_dir)
        if not checkpoint_dirs:
            print(f"[WARN] No checkpoint_* directories in {training_dir}")
            continue

        for checkpoint_dir in checkpoint_dirs:
            simulation_dirs = iter_simulation_dirs(checkpoint_dir)
            configurations.append(
                {
                    "training_dir": training_dir,
                    "checkpoint_dir": checkpoint_dir,
                    "simulation_dirs": simulation_dirs,
                    "simulation_count": len(simulation_dirs),
                }
            )

    if not configurations:
        raise ValueError(
            "No checkpoint configurations found (no Training_*/checkpoint_* combinations)."
        )

    return configurations


def main() -> int:
    args = parse_args()

    if args.max_simulations is not None and args.max_simulations < 0:
        print("Argument error: --max_simulations must be >= 0")
        return 2

    if args.seed is not None:
        random.seed(args.seed)

    try:
        simulations_dir = Path(args.simulations_dir)
        configurations = resolve_configurations(simulations_dir)
    except ValueError as exc:
        print(f"Argument error: {exc}")
        return 2

    natural_min_simulation_count = min(cfg["simulation_count"] for cfg in configurations)
    target_simulation_count = natural_min_simulation_count
    cap_applied = False
    if (
        args.max_simulations is not None
        and natural_min_simulation_count > args.max_simulations
    ):
        target_simulation_count = args.max_simulations
        cap_applied = True

    training_count = len({cfg["training_dir"].name for cfg in configurations})
    checkpoint_count = len(configurations)

    print("Equalize simulation counts across configurations")
    print("=" * 60)
    print(f"Simulations root: {simulations_dir}")
    print(f"Training directories found: {training_count}")
    print(f"Checkpoint configurations found: {checkpoint_count}")
    print(f"Natural minimum simulation count found: {natural_min_simulation_count}")
    if args.max_simulations is not None:
        print(f"Max simulations cap requested: {args.max_simulations}")
        if cap_applied:
            print(f"Target simulation count per configuration (cap applied): {target_simulation_count}")
        else:
            print(f"Target simulation count per configuration (cap not applied): {target_simulation_count}")
    else:
        print(f"Target simulation count per configuration: {target_simulation_count}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'LIVE DELETE'}")
    print("=" * 60)

    total_simulations_before = 0
    total_to_delete = 0
    total_deleted = 0
    total_deletion_errors = 0
    per_configuration_summary: List[dict] = []

    for cfg in configurations:
        training_dir: Path = cfg["training_dir"]
        checkpoint_dir: Path = cfg["checkpoint_dir"]
        simulation_dirs: List[Path] = cfg["simulation_dirs"]
        simulation_count: int = cfg["simulation_count"]
        deleted_for_config = 0
        deletion_errors_for_config = 0

        total_simulations_before += simulation_count
        to_delete_count = max(0, simulation_count - target_simulation_count)
        total_to_delete += to_delete_count

        print(
            f"\n[CONFIG] {training_dir.name}/{checkpoint_dir.name} "
            f"| current={simulation_count} | to_delete={to_delete_count} "
            f"| final_target={target_simulation_count}"
        )

        if to_delete_count == 0:
            continue

        selected_for_deletion = random.sample(simulation_dirs, to_delete_count)
        for sim_dir in sorted(selected_for_deletion, key=lambda path: path.name):
            if args.dry_run:
                print(f"  [DRY-RUN] Would delete {sim_dir}")
            else:
                try:
                    shutil.rmtree(sim_dir)
                    deleted_for_config += 1
                    total_deleted += 1
                    print(f"  [DELETED] {sim_dir}")
                except Exception as exc:
                    deletion_errors_for_config += 1
                    total_deletion_errors += 1
                    print(f"  [ERROR] Could not delete {sim_dir}: {exc}")

        per_configuration_summary.append(
            {
                "training": training_dir.name,
                "checkpoint": checkpoint_dir.name,
                "current": simulation_count,
                "to_delete": to_delete_count,
                "target": target_simulation_count,
                "deleted": deleted_for_config,
                "deletion_errors": deletion_errors_for_config,
            }
        )

    # Include configurations with no deletions in the summary table as well.
    summarized_keys = {
        (row["training"], row["checkpoint"]) for row in per_configuration_summary
    }
    for cfg in configurations:
        key = (cfg["training_dir"].name, cfg["checkpoint_dir"].name)
        if key in summarized_keys:
            continue
        per_configuration_summary.append(
            {
                "training": cfg["training_dir"].name,
                "checkpoint": cfg["checkpoint_dir"].name,
                "current": cfg["simulation_count"],
                "to_delete": 0,
                "target": target_simulation_count,
                "deleted": 0,
                "deletion_errors": 0,
            }
        )

    per_configuration_summary.sort(
        key=lambda row: (row["training"], row["checkpoint"])
    )

    print("\nSummary")
    print("-" * 60)
    print(f"Training directories processed: {training_count}")
    print(f"Checkpoint configurations processed: {checkpoint_count}")
    print(f"Natural minimum simulation count found: {natural_min_simulation_count}")
    if args.max_simulations is not None:
        print(f"Max simulations cap requested: {args.max_simulations}")
        if cap_applied:
            print(f"Target simulation count per configuration (cap applied): {target_simulation_count}")
        else:
            print(f"Target simulation count per configuration (cap not applied): {target_simulation_count}")
    else:
        print(f"Target simulation count per configuration: {target_simulation_count}")
    print(f"Total simulations before: {total_simulations_before}")

    if args.dry_run:
        print(f"Would delete (dry-run): {total_to_delete}")
        print(f"Total simulations after dry-run (projected): {total_simulations_before - total_to_delete}")
    else:
        print(f"Deleted: {total_deleted}")
        print(f"Deletion errors: {total_deletion_errors}")
        print(f"Total simulations after: {total_simulations_before - total_deleted}")

    print("\nPer-configuration deletion counts")
    print("-" * 60)
    for row in per_configuration_summary:
        if args.dry_run:
            print(
                f"{row['training']}/{row['checkpoint']} | "
                f"current={row['current']} | "
                f"would_delete={row['to_delete']} | "
                f"target={row['target']}"
            )
        else:
            print(
                f"{row['training']}/{row['checkpoint']} | "
                f"current={row['current']} | "
                f"to_delete={row['to_delete']} | "
                f"deleted={row['deleted']} | "
                f"deletion_errors={row['deletion_errors']} | "
                f"target={row['target']}"
            )

    if total_deletion_errors > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
