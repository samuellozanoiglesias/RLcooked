#!/usr/bin/env python3
"""
Scan all simulation folders under ROOT and detect broken CSV files.

A file is considered BROKEN when, going down the rows, the column
`init_second` or `second` goes DOWN (time travels backwards), or contains
non-numeric values. Use --strict to also flag repeated values (value must
strictly go up on every row).

Files checked in every simulation folder:
    human_like_actions_ai_rl_1.csv
    human_like_actions_ai_rl_2.csv
    human_like_positions_ai_rl_1.csv
    human_like_positions_ai_rl_2.csv

Expected folder structures:
    ROOT/map_X/synergy_Y/specialized_Z/abilities_A_B/Training_T/checkpoint_C/simulation_S/
    ROOT/with_pretraining/map_X/.../simulation_S/        (flagged as pretraining=True)

Output:
    problematic_simulations.csv  -> one row per problematic simulation (with metadata)
    all_simulations_report.csv   -> (optional, --all-report) every simulation scanned

Usage:
    python check_broken_simulations.py --out problematic_simulations.csv --all-report all_simulations.csv
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

DEFAULT_ROOT = "/data/samuel_lozano/cooked/classic_collision/simulations/empty_init/"

TARGET_FILES = [
    "human_like_actions_ai_rl_1.csv",
    "human_like_actions_ai_rl_2.csv",
    "human_like_positions_ai_rl_1.csv",
    "human_like_positions_ai_rl_2.csv",
]
TIME_COLUMNS = ["init_second", "second"]


# --------------------------------------------------------------------------- #
# Path parsing
# --------------------------------------------------------------------------- #
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


def parse_sim_path(sim_dir, root):
    """Extract metadata (map, synergy, abilities, training id, ...) from the path."""
    rel = os.path.relpath(sim_dir, root)
    parts = rel.split(os.sep)

    info = {
        "pretraining": "with_pretraining" in parts,
        "map": None,
        "synergy": None,
        "specialized": None,
        "abilities": None,
        "training_id": None,
        "checkpoint": None,
        "simulation_id": None,
    }
    for p in parts:
        if p.startswith("map_"):
            info["map"] = p[len("map_"):]
        elif p.startswith("synergy_"):
            info["synergy"] = _num(p[len("synergy_"):])
        elif p.startswith("specialized_"):
            info["specialized"] = _num(p[len("specialized_"):])
        elif p.startswith("abilities_"):
            raw = p[len("abilities_"):]
            info["abilities"] = raw
            for i, v in enumerate(raw.split("_"), start=1):
                info[f"ability_agent_{i}"] = _num(v)
        elif p.startswith("Training_"):
            info["training_id"] = p[len("Training_"):]
        elif p.startswith("checkpoint_"):
            info["checkpoint"] = _num(p[len("checkpoint_"):])
        elif p.startswith("simulation_"):
            info["simulation_id"] = p[len("simulation_"):]

    info["relative_path"] = rel
    info["sim_path"] = os.path.abspath(sim_dir)
    return info


# --------------------------------------------------------------------------- #
# File checking
# --------------------------------------------------------------------------- #
def check_file(path, strict=False):
    """Return a dict describing whether the time columns of a CSV go up."""
    res = {"status": "ok", "n_rows": None, "details": ""}

    if not os.path.isfile(path):
        res.update(status="missing", details="file not found")
        return res

    try:
        header = pd.read_csv(path, nrows=0).columns
        cols = [c for c in TIME_COLUMNS if c in header]
        if not cols:
            res.update(status="no_time_columns",
                       details=f"none of {TIME_COLUMNS} in header")
            return res
        df = pd.read_csv(path, usecols=cols)
    except pd.errors.EmptyDataError:
        res.update(status="empty", details="empty file")
        return res
    except Exception as e:  # corrupted / truncated file, bad quoting, etc.
        res.update(status="read_error", details=f"{type(e).__name__}: {e}")
        return res

    res["n_rows"] = len(df)
    if len(df) == 0:
        res.update(status="empty", details="header only, no rows")
        return res

    details = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        n_nan = int(s.isna().sum())
        diff = s.diff().iloc[1:]
        bad_mask = (diff <= 0) if strict else (diff < 0)
        bad_idx = diff.index[bad_mask.to_numpy()]
        n_bad = len(bad_idx)

        res[f"{c}_violations"] = n_bad
        res[f"{c}_non_numeric"] = n_nan

        if n_bad:
            i = int(bad_idx[0])
            # line number in the file (1 = header)
            res[f"{c}_first_bad_line"] = i + 2
            details.append(
                f"{c}: {n_bad} drops, first at line {i + 2} "
                f"({s.iloc[i - 1]} -> {s.iloc[i]})"
            )
        if n_nan:
            details.append(f"{c}: {n_nan} non-numeric/NaN values")

        if n_bad or n_nan:
            res["status"] = "broken"

    res["details"] = "; ".join(details)
    return res


def process_simulation(sim_dir, root, strict=False):
    info = parse_sim_path(sim_dir, root)
    problem_files = []

    for fname in TARGET_FILES:
        key = fname.replace("human_like_", "").replace(".csv", "")
        r = check_file(os.path.join(sim_dir, fname), strict=strict)
        for k, v in r.items():
            info[f"{key}__{k}"] = v
        if r["status"] != "ok":
            problem_files.append(f"{fname} [{r['status']}]")

    info["is_problematic"] = bool(problem_files)
    info["n_problem_files"] = len(problem_files)
    info["problem_files"] = " | ".join(problem_files)
    return info


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def find_simulation_dirs(root):
    sims = []
    for dirpath, dirnames, filenames in os.walk(root):
        is_sim = os.path.basename(dirpath).startswith("simulation_") or any(
            f in filenames for f in TARGET_FILES
        )
        if is_sim:
            sims.append(dirpath)
            dirnames[:] = []  # don't descend further inside a simulation folder
        else:
            dirnames.sort()
    return sims


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default="problematic_simulations.csv",
                    help="output CSV with problematic simulations")
    ap.add_argument("--all-report", default=None,
                    help="optional CSV with ALL simulations (ok + problematic)")
    ap.add_argument("--strict", action="store_true",
                    help="also flag repeated values (require strictly increasing)")
    ap.add_argument("--ignore-missing", action="store_true",
                    help="do not count missing files as a problem")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    sims = find_simulation_dirs(root)
    print(f"Found {len(sims)} simulation folders under {root}")
    if not sims:
        sys.exit(0)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_simulation, s, root, args.strict): s for s in sims}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append({"sim_path": futs[fut], "is_problematic": True,
                             "problem_files": f"processing error: {e}"})
            if n % 200 == 0 or n == len(sims):
                print(f"  processed {n}/{len(sims)}")

    df = pd.DataFrame(rows)

    if args.ignore_missing:
        status_cols = [c for c in df.columns if c.endswith("__status")]
        not_ok = df[status_cols].apply(lambda col: ~col.isin(["ok", "missing"]))
        df["is_problematic"] = not_ok.any(axis=1)
        df["problem_files"] = df["problem_files"].fillna("").apply(
            lambda s: " | ".join(p for p in s.split(" | ") if "[missing]" not in p and p))
        df["n_problem_files"] = df["problem_files"].apply(
            lambda s: len([p for p in s.split(" | ") if p]))

    # nice column order: metadata first
    meta = ["pretraining", "map", "synergy", "specialized", "abilities"]
    meta += sorted(c for c in df.columns if c.startswith("ability_agent_"))
    meta += ["training_id", "checkpoint", "simulation_id",
             "is_problematic", "n_problem_files", "problem_files",
             "relative_path", "sim_path"]
    meta = [c for c in meta if c in df.columns]
    df = df[meta + [c for c in df.columns if c not in meta]]
    df = df.sort_values(["pretraining", "map", "synergy", "specialized",
                         "abilities", "training_id", "checkpoint", "simulation_id"],
                        na_position="last").reset_index(drop=True)

    bad = df[df["is_problematic"]]
    bad.to_csv(args.out, index=False)
    if args.all_report:
        df.to_csv(args.all_report, index=False)

    # summary
    print("\n===== SUMMARY =====")
    print(f"Total simulations : {len(df)}")
    print(f"Problematic       : {len(bad)}")
    print(f"  with pretraining: {int(bad['pretraining'].sum())}")
    print(f"  no pretraining  : {int((~bad['pretraining'].astype(bool)).sum())}")
    if len(bad):
        print("\nProblematic per map / pretraining:")
        print(bad.groupby(["pretraining", "map"]).size().to_string())
    print(f"\nSaved problematic simulations to: {os.path.abspath(args.out)}")
    if args.all_report:
        print(f"Saved full report to: {os.path.abspath(args.all_report)}")


if __name__ == "__main__":
    main()
