#!/usr/bin/env python3
"""
Delete simulation folders based on final-delivery threshold rules.

The script expects each simulation folder to contain exactly two CSV files named
"positions_<agent>.csv". For each CSV, it reads the last non-empty data row and
takes the last column value (expected to be "score"). The final total deliveries
for a simulation are computed as:

    total_deliveries = last_score_agent_1 + last_score_agent_2

By default, simulations with total_deliveries < --min_deliveries are deleted.
You can also delete simulations with total_deliveries > --min_deliveries
using --delete_mode above.

When --checkpoint_dir is not provided:
- If --training_id and --checkpoint_number are provided, process one checkpoint.
- If --training_id is provided but --checkpoint_number is omitted, process all
    checkpoint_* folders under that training.
- If both --training_id and --checkpoint_number are omitted, process all
    trainings and all checkpoints.

Delete modes:
- below: delete simulations where total_deliveries < D
- above: delete simulations where total_deliveries > D
- random: ignore deliveries and randomly delete simulations so each checkpoint
    has at most N simulations (--max_simulations)

Example:
    nohup python3 delete_wrong_delivery_simulations.py --dry_run --min_deliveries 10 --delete_mode above --checkpoint_dir /data/samuel_lozano/cooked/map_baseline_division_of_labor/simulations/Training_12345/checkpoint_50/ > log_cleanup.out 2>&1 &

    nohup python3 delete_wrong_delivery_simulations.py --delete_mode random --max_simulations 240 --map_nr baseline_division_of_labor_large --training_id 12345 > log_cleanup.out 2>&1 &

    nohup python3 delete_wrong_delivery_simulations.py --dry_run --min_deliveries 10 --delete_mode above --game_version classic_collision --map_nr baseline_division_of_labor_large --synergy 1.35 --specialization 0.25 > log_cleanup.out 2>&1 &
"""


import argparse
import csv
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def ensure_prefix(value: str, prefix: str) -> str:
    """Ensure the given value starts with the expected prefix."""
    return value if value.startswith(prefix) else f"{prefix}{value}"


def build_simulations_root(
    base_dir: str,
    map_nr: str,
    synergy: Optional[str] = None,
    specialization: Optional[str] = None,
    study_name: Optional[str] = None,
    game_version: Optional[str] = None,
) -> Path:
    """
    Build simulations root path from configuration parameters.

    Example layout:
    /data/samuel_lozano/cooked/map_<map_nr>/synergy_<x>/specialized_<y>/simulations/
    """
    path = Path(base_dir)

    if game_version:
        path = path / game_version

    path = path / ensure_prefix(map_nr, "map_")

    if synergy:
        path = path / ensure_prefix(synergy, "synergy_")

    if specialization:
        path = path / ensure_prefix(specialization, "specialized_")

    path = path / "simulations"

    if study_name:
        path = path / study_name

    return path


def build_checkpoint_dir(
    base_dir: str,
    map_nr: str,
    training_id: str,
    checkpoint_number: str,
    synergy: Optional[str] = None,
    specialization: Optional[str] = None,
    study_name: Optional[str] = None,
    game_version: Optional[str] = None,
) -> Path:
    """Build a single checkpoint directory path from configuration parameters."""
    path = build_simulations_root(
        base_dir=base_dir,
        map_nr=map_nr,
        synergy=synergy,
        specialization=specialization,
        study_name=study_name,
        game_version=game_version,
    )

    path = path / ensure_prefix(training_id, "Training_")
    path = path / ensure_prefix(str(checkpoint_number), "checkpoint_")

    return path


def read_final_score_from_positions_csv(csv_path: Path) -> float:
    """
    Read the final score from a positions CSV.

    Rules enforced:
    - Header must exist and its last column must be named "score".
    - At least one data row must exist.
    - Last non-empty data row must have the same number of columns as header.
    - Last column of that row must be numeric.
    """
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)

        if not header:
            raise ValueError("missing header")

        if header[-1].strip().lower() != "score":
            raise ValueError(
                f"invalid last header column '{header[-1]}' (expected 'score')"
            )

        last_data_row: Optional[List[str]] = None
        for row in reader:
            if row and any(cell.strip() for cell in row):
                last_data_row = row

        if last_data_row is None:
            raise ValueError("no data rows")

        if len(last_data_row) != len(header):
            raise ValueError(
                f"row length mismatch (header={len(header)}, row={len(last_data_row)})"
            )

        raw_score = last_data_row[-1].strip()
        if raw_score == "":
            raise ValueError("empty score value in last row")

        try:
            return float(raw_score)
        except ValueError as exc:
            raise ValueError(f"non-numeric score value '{raw_score}'") from exc


def validate_and_get_total_deliveries(sim_dir: Path) -> Tuple[float, List[Path]]:
    """
    Validate simulation structure and compute total final deliveries.

    Returns:
        (total_deliveries, positions_files)
    """
    position_files = sorted(sim_dir.glob("positions_*.csv"))

    if len(position_files) != 2:
        names = ", ".join(path.name for path in position_files) if position_files else "none"
        raise ValueError(
            f"expected exactly 2 files matching positions_*.csv, found {len(position_files)} ({names})"
        )

    total_deliveries = 0.0
    for csv_path in position_files:
        total_deliveries += read_final_score_from_positions_csv(csv_path)

    return total_deliveries, position_files


def iter_simulation_dirs(checkpoint_dir: Path) -> List[Path]:
    """Return sorted simulation_* directories directly inside the checkpoint directory."""
    return sorted(
        [
            path
            for path in checkpoint_dir.iterdir()
            if path.is_dir() and path.name.startswith("simulation_")
        ],
        key=lambda path: path.name,
    )


def iter_training_dirs(simulations_root: Path) -> List[Path]:
    """Return sorted Training_* directories directly inside simulations root."""
    return sorted(
        [
            path
            for path in simulations_root.iterdir()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete simulation_* folders based on final deliveries and a threshold rule. "
            "Final deliveries are computed by summing the last-column score from the "
            "last row of the two positions_<agent>.csv files."
        )
    )

    parser.add_argument(
        "--min_deliveries",
        type=float,
        default=None,
        help="Delivery threshold D used by delete_mode=below|above.",
    )
    parser.add_argument(
        "--max_simulations",
        type=int,
        default=None,
        help="Maximum simulations to keep per checkpoint when delete_mode=random.",
    )
    parser.add_argument(
        "--delete_mode",
        type=str,
        default="below",
        choices=["below", "above", "random"],
        help=(
            "Deletion rule: 'below' deletes when total_deliveries < D; "
            "'above' deletes when total_deliveries > D; "
            "'random' randomly deletes to enforce --max_simulations."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print what would be deleted, without deleting folders.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Direct path to target folder. Supported: checkpoint_*, Training_*, or "
            "a simulations root containing Training_* directories. If provided, map-based "
            "path construction is ignored."
        ),
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="/data/samuel_lozano/cooked",
        help="Base cooked directory used to build checkpoint path when --checkpoint-dir is not provided.",
    )
    parser.add_argument("--map_nr", type=str, default=None, help="Map name (with or without 'map_' prefix).")
    parser.add_argument(
        "--training_id",
        type=str,
        default=None,
        help="Training identifier (with or without 'Training_' prefix).",
    )
    parser.add_argument(
        "--checkpoint_number",
        type=str,
        default=None,
        help="Checkpoint number/name (with or without 'checkpoint_' prefix).",
    )
    parser.add_argument(
        "--synergy",
        type=str,
        default=None,
        help="Optional synergy segment (e.g., 1.70 or synergy_1.70).",
    )
    parser.add_argument(
        "--specialization",
        type=str,
        default=None,
        help="Optional specialization segment (e.g., 0.05 or specialized_0.05).",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Optional segment under simulations/ before Training_...",
    )
    parser.add_argument(
        "--game_version",
        type=str,
        default=None,
        help="Optional segment under base-dir before map_... (e.g., classic, competition).",
    )

    return parser.parse_args()


def resolve_checkpoint_dir(args: argparse.Namespace) -> Path:
    """Backward-compatible wrapper that resolves exactly one checkpoint directory."""
    checkpoint_dirs = resolve_target_checkpoint_dirs(args)
    if len(checkpoint_dirs) != 1:
        raise ValueError(
            "Arguments resolve to multiple checkpoint directories; use resolve_target_checkpoint_dirs instead"
        )
    return checkpoint_dirs[0]


def validate_mode_arguments(args: argparse.Namespace) -> None:
    """Validate mode-specific required arguments."""
    if args.delete_mode in ("below", "above"):
        if args.min_deliveries is None:
            raise ValueError("--min_deliveries is required when --delete_mode is below or above")
    elif args.delete_mode == "random":
        if args.max_simulations is None:
            raise ValueError("--max_simulations is required when --delete_mode is random")
        if args.max_simulations < 0:
            raise ValueError("--max_simulations must be >= 0")


def resolve_target_checkpoint_dirs(args: argparse.Namespace) -> List[Path]:
    """
    Resolve one or many checkpoint directories to process.

    Behavior when --checkpoint_dir is not provided:
    - training_id + checkpoint_number -> one checkpoint
    - training_id only -> all checkpoints for that training
    - neither training_id nor checkpoint_number -> all trainings and all checkpoints
    """
    checkpoint_dirs: List[Path] = []

    if args.checkpoint_dir:
        target = Path(args.checkpoint_dir)
        if not target.exists() or not target.is_dir():
            raise ValueError(
                f"--checkpoint_dir does not exist or is not a directory: {target}"
            )

        if target.name.startswith("checkpoint_"):
            return [target]

        if target.name.startswith("Training_"):
            if args.checkpoint_number:
                checkpoint_dir = target / ensure_prefix(str(args.checkpoint_number), "checkpoint_")
                if checkpoint_dir.is_dir():
                    return [checkpoint_dir]
                raise ValueError(f"Checkpoint folder not found: {checkpoint_dir}")

            checkpoint_dirs = iter_checkpoint_dirs(target)
            if not checkpoint_dirs:
                raise ValueError(f"No checkpoint_* directories found under: {target}")
            return checkpoint_dirs

        # Treat as simulations root containing Training_* directories.
        training_dirs = iter_training_dirs(target)
        if not training_dirs:
            raise ValueError(
                f"{target} is not a checkpoint_*, not a Training_*, and has no Training_* children"
            )

        for training_dir in training_dirs:
            if args.checkpoint_number:
                checkpoint_dir = training_dir / ensure_prefix(str(args.checkpoint_number), "checkpoint_")
                if checkpoint_dir.is_dir():
                    checkpoint_dirs.append(checkpoint_dir)
            else:
                checkpoint_dirs.extend(iter_checkpoint_dirs(training_dir))

        if not checkpoint_dirs:
            raise ValueError("No matching checkpoint directories found inside provided --checkpoint_dir")
        return checkpoint_dirs

    if not args.map_nr:
        raise ValueError("--map_nr is required when --checkpoint_dir is not provided")

    simulations_root = build_simulations_root(
        base_dir=args.base_dir,
        map_nr=args.map_nr,
        synergy=args.synergy,
        specialization=args.specialization,
        study_name=args.study_name,
        game_version=args.game_version,
    )

    if not simulations_root.exists() or not simulations_root.is_dir():
        raise ValueError(f"Simulations root does not exist or is not a directory: {simulations_root}")

    if args.training_id:
        training_dirs = [simulations_root / ensure_prefix(args.training_id, "Training_")]
        if not training_dirs[0].is_dir():
            raise ValueError(f"Training directory not found: {training_dirs[0]}")
    else:
        training_dirs = iter_training_dirs(simulations_root)
        if not training_dirs:
            raise ValueError(f"No Training_* directories found in: {simulations_root}")

    for training_dir in training_dirs:
        if args.checkpoint_number:
            checkpoint_dir = training_dir / ensure_prefix(str(args.checkpoint_number), "checkpoint_")
            if checkpoint_dir.is_dir():
                checkpoint_dirs.append(checkpoint_dir)
            else:
                print(f"[WARN] Missing checkpoint in {training_dir.name}: {checkpoint_dir.name}")
        else:
            training_checkpoints = iter_checkpoint_dirs(training_dir)
            if not training_checkpoints:
                print(f"[WARN] No checkpoint_* directories in {training_dir}")
            checkpoint_dirs.extend(training_checkpoints)

    if not checkpoint_dirs:
        raise ValueError("No checkpoint directories matched the provided arguments")

    # De-duplicate while preserving order.
    unique_dirs: List[Path] = []
    seen = set()
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_key = str(checkpoint_dir.resolve())
        if checkpoint_key not in seen:
            seen.add(checkpoint_key)
            unique_dirs.append(checkpoint_dir)

    return unique_dirs


def process_checkpoint_threshold_mode(checkpoint_dir: Path, args: argparse.Namespace) -> dict:
    """Process one checkpoint in below/above modes (delivery-threshold based)."""
    simulation_dirs = iter_simulation_dirs(checkpoint_dir)

    inspected = 0
    valid = 0
    kept = 0
    matching_rule = 0
    deleted = 0
    skipped_invalid = 0
    deletion_errors = 0

    for sim_dir in simulation_dirs:
        inspected += 1
        try:
            total_deliveries, position_files = validate_and_get_total_deliveries(sim_dir)
            valid += 1

            file_list = ", ".join(path.name for path in position_files)
            print(
                f"[VALID] {sim_dir.name} | files: {file_list} | total_deliveries={total_deliveries:g}"
            )

            should_delete = (
                total_deliveries < args.min_deliveries
                if args.delete_mode == "below"
                else total_deliveries > args.min_deliveries
            )

            if should_delete:
                matching_rule += 1
                if args.dry_run:
                    print(f"  [DRY-RUN] Would delete {sim_dir}")
                else:
                    try:
                        shutil.rmtree(sim_dir)
                        deleted += 1
                        print(f"  [DELETED] {sim_dir}")
                    except Exception as exc:
                        deletion_errors += 1
                        print(f"  [ERROR] Could not delete {sim_dir}: {exc}")
            else:
                kept += 1

        except Exception as exc:
            skipped_invalid += 1
            print(f"[SKIP-INVALID] {sim_dir.name} | reason: {exc}")

    return {
        "simulations_found": len(simulation_dirs),
        "inspected": inspected,
        "valid": valid,
        "kept": kept,
        "matching_rule": matching_rule,
        "deleted": deleted,
        "skipped_invalid": skipped_invalid,
        "deletion_errors": deletion_errors,
    }


def process_checkpoint_random_mode(checkpoint_dir: Path, args: argparse.Namespace) -> dict:
    """Process one checkpoint in random mode (cap number of simulation folders)."""
    simulation_dirs = iter_simulation_dirs(checkpoint_dir)
    simulations_found = len(simulation_dirs)

    to_delete_count = max(0, simulations_found - args.max_simulations)
    to_delete = random.sample(simulation_dirs, to_delete_count) if to_delete_count > 0 else []

    deleted = 0
    deletion_errors = 0

    for sim_dir in sorted(to_delete, key=lambda path: path.name):
        if args.dry_run:
            print(f"  [DRY-RUN] Would delete {sim_dir}")
        else:
            try:
                shutil.rmtree(sim_dir)
                deleted += 1
                print(f"  [DELETED] {sim_dir}")
            except Exception as exc:
                deletion_errors += 1
                print(f"  [ERROR] Could not delete {sim_dir}: {exc}")

    return {
        "simulations_found": simulations_found,
        "inspected": simulations_found,
        "valid": simulations_found,
        "kept": simulations_found - to_delete_count,
        "matching_rule": to_delete_count,
        "deleted": deleted,
        "skipped_invalid": 0,
        "deletion_errors": deletion_errors,
    }


def main() -> int:
    args = parse_args()

    try:
        validate_mode_arguments(args)
        checkpoint_dirs = resolve_target_checkpoint_dirs(args)
    except ValueError as exc:
        print(f"Argument error: {exc}")
        return 2

    print("Delivery-threshold simulation cleanup")
    print("=" * 60)
    print(f"Checkpoint directories to process: {len(checkpoint_dirs)}")
    if args.delete_mode == "random":
        print(f"Delete mode: random (keep max {args.max_simulations} simulations per checkpoint)")
    else:
        print(f"Threshold D: {args.min_deliveries}")
        if args.delete_mode == "below":
            print("Deletion rule: total_deliveries < D")
        else:
            print("Deletion rule: total_deliveries > D")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'LIVE DELETE'}")
    print("=" * 60)

    total_simulations_found = 0
    total_inspected = 0
    total_valid = 0
    total_kept = 0
    total_matching_rule = 0
    total_deleted = 0
    total_skipped_invalid = 0
    total_deletion_errors = 0

    for checkpoint_dir in checkpoint_dirs:
        print(f"\nCheckpoint: {checkpoint_dir}")
        print("-" * 60)

        if args.delete_mode == "random":
            stats = process_checkpoint_random_mode(checkpoint_dir, args)
        else:
            stats = process_checkpoint_threshold_mode(checkpoint_dir, args)

        print(
            f"Checkpoint summary | simulations_found={stats['simulations_found']} "
            f"matching_rule={stats['matching_rule']} deleted={stats['deleted']} "
            f"kept={stats['kept']}"
        )

        total_simulations_found += stats["simulations_found"]
        total_inspected += stats["inspected"]
        total_valid += stats["valid"]
        total_kept += stats["kept"]
        total_matching_rule += stats["matching_rule"]
        total_deleted += stats["deleted"]
        total_skipped_invalid += stats["skipped_invalid"]
        total_deletion_errors += stats["deletion_errors"]

    print("\nSummary")
    print("-" * 60)
    print(f"Checkpoint directories processed: {len(checkpoint_dirs)}")
    print(f"Simulation folders found: {total_simulations_found}")
    print(f"Inspected simulation folders: {total_inspected}")
    print(f"Valid simulations: {total_valid}")
    print(f"Invalid/skipped simulations: {total_skipped_invalid}")
    print(f"Simulations matching deletion rule: {total_matching_rule}")
    print(f"Simulations kept (not matching deletion rule): {total_kept}")

    if args.dry_run:
        print(f"Would be deleted (dry-run): {total_matching_rule}")
    else:
        print(f"Deleted: {total_deleted}")
        print(f"Deletion errors: {total_deletion_errors}")

    if total_deletion_errors > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())