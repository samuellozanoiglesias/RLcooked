#!/usr/bin/env python3
"""
Inventory of the MARL trainings: which Training_* folders exist per configuration
and which seed each one used.

Scans every map_* folder in
    <base>/                       (trainings from scratch,  pretraining = no)
    <base>/with_pretraining/      (pretrained trainings,    pretraining = yes)
and every Training_* folder below each map (map_*/synergy_*/specialized_*/Training_*,
at any depth). From each training's config.txt it reads
    WALKING_SPEEDS, CUTTING_SPEEDS   -> abilities (walking/cutting speed of both agents)
    INITIAL_SEED                     -> seed

Configuration = pretraining x map x synergy x specialization x abilities.

Outputs (in --output_dir):
  trainings_per_configuration.csv  one row per configuration: number of trainings, how many
                                   have checkpoint_<N> (--checkpoint), number of distinct seeds,
                                   seeds and training ids (';'-separated), duplicate seeds
  trainings_list.csv               one row per training: configuration, training_id, seed,
                                   checkpoint present, problems, path

Usage:
    python3 trainings_inventory.py
    python3 trainings_inventory.py --base /data/samuel_lozano/cooked/classic_collision/empty_init \
        --output_dir ./figures --checkpoint 750
"""

import argparse
import ast
import os
import re
from pathlib import Path

import pandas as pd

DEFAULT_BASE = "/data/samuel_lozano/cooked/classic_collision/empty_init"
SKIP_DIRS = {"simulations"}          # never descend into these


def read_config(path: Path) -> dict:
    """WALKING_SPEEDS, CUTTING_SPEEDS and INITIAL_SEED of one config.txt ('KEY: value' lines)."""
    out = {"walking_speeds": None, "cutting_speeds": None, "seed": None, "problems": []}
    if not path.is_file():
        out["problems"].append("no config.txt")
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        for key, field in (("WALKING_SPEEDS", "walking_speeds"), ("CUTTING_SPEEDS", "cutting_speeds"),
                           ("INITIAL_SEED", "seed")):
            if line.startswith(key + ":") and out[field] is None:
                raw = line.split(":", 1)[1].strip()
                try:
                    out[field] = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    out[field] = raw          # keep the raw text (e.g. a seed written as a string)
    for key, field in (("WALKING_SPEEDS", "walking_speeds"), ("CUTTING_SPEEDS", "cutting_speeds"),
                       ("INITIAL_SEED", "seed")):
        if out[field] is None:
            out["problems"].append(f"no {key}")
    return out


def _speed(d, agent):
    try:
        return float(d[agent])
    except (TypeError, KeyError, ValueError):
        return float("nan")


def level_value(rel_parts, prefix):
    """Value of the last '<prefix>_<value>' folder in the path (e.g. synergy_1.35 -> '1.35')."""
    val = ""
    for p in rel_parts:
        if p.startswith(prefix + "_"):
            val = p[len(prefix) + 1:]
    return val


def find_trainings(map_dir: Path):
    """Every Training_* folder below map_dir (not descending into Training_* or simulations)."""
    for root, dirs, _ in os.walk(map_dir):
        keep = []
        for d in sorted(dirs):
            if d.startswith("Training_"):
                yield Path(root) / d
            elif d not in SKIP_DIRS:
                keep.append(d)
        dirs[:] = keep


def scan(base: Path, pretraining_subfolder: str, checkpoint: str) -> pd.DataFrame:
    sources = [("no", base), ("yes", base / pretraining_subfolder)]
    rows = []
    for pre, folder in sources:
        if not folder.is_dir():
            print(f"[WARN] folder not found: {folder}")
            continue
        maps = sorted(p for p in folder.iterdir() if p.is_dir() and p.name.startswith("map_"))
        print(f"{folder}: {len(maps)} map folders")
        for map_dir in maps:
            n = 0
            for tdir in find_trainings(map_dir):
                rel = tdir.relative_to(map_dir).parts
                cfg = read_config(tdir / "config.txt")
                ws, cs = cfg["walking_speeds"], cfg["cutting_speeds"]
                ckpt_dir = tdir / f"checkpoint_{checkpoint}"
                has_ckpt = ckpt_dir.is_dir() and any(ckpt_dir.iterdir())
                problems = list(cfg["problems"])
                if not has_ckpt:
                    problems.append(f"checkpoint_{checkpoint} missing or empty")
                row = {
                    "pretraining": pre,
                    "map": map_dir.name[len("map_"):],
                    "synergy": level_value(rel, "synergy"),
                    "specialization": level_value(rel, "specialized"),
                    "walking_speed_1": _speed(ws, "ai_rl_1"),
                    "cutting_speed_1": _speed(cs, "ai_rl_1"),
                    "walking_speed_2": _speed(ws, "ai_rl_2"),
                    "cutting_speed_2": _speed(cs, "ai_rl_2"),
                    "training_id": tdir.name,
                    "seed": "" if cfg["seed"] is None else str(cfg["seed"]),   # text: no 1 -> 1.0
                    f"has_checkpoint_{checkpoint}": has_ckpt,
                    "problems": "; ".join(problems),
                    "path": str(tdir),
                }
                # abilities label as in the simulation tree: abilities_<walk P1>_<cut P2>
                w1, c2 = row["walking_speed_1"], row["cutting_speed_2"]
                row["abilities"] = "unknown" if (w1 != w1 or c2 != c2) else f"{w1!r}_{c2!r}"
                rows.append(row)
                n += 1
            print(f"  {'pretrained' if pre == 'yes' else 'scratch':<10} {map_dir.name}: {n} trainings")
    return pd.DataFrame(rows)


CONFIG_KEYS = ["pretraining", "map", "synergy", "specialization", "abilities",
               "walking_speed_1", "cutting_speed_1", "walking_speed_2", "cutting_speed_2"]


def _num_key(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("inf")


def summarize(df: pd.DataFrame, checkpoint: str) -> pd.DataFrame:
    ck = f"has_checkpoint_{checkpoint}"
    rows = []
    for key, g in df.groupby(CONFIG_KEYS, dropna=False, sort=False):
        seeds = g["seed"].fillna("").astype(str)
        dup = sorted(seeds[seeds.duplicated(keep=False) & (seeds != "")].unique(), key=_num_key)
        rows.append({**dict(zip(CONFIG_KEYS, key)),
                     "n_trainings": len(g),
                     f"n_with_checkpoint_{checkpoint}": int(g[ck].sum()),
                     "n_distinct_seeds": seeds[seeds != ""].nunique(),
                     "duplicate_seeds": ";".join(dup),
                     "n_with_problems": int((g["problems"] != "").sum()),
                     "seeds": ";".join(v if v else "?" for v in seeds),
                     "training_ids": ";".join(g["training_id"].astype(str))})
    return pd.DataFrame(rows)


def sort_table(df: pd.DataFrame, extra=()) -> pd.DataFrame:
    d = df.copy()
    d["_syn"] = d["synergy"].map(_num_key)
    d["_spec"] = d["specialization"].map(_num_key)
    order = ["pretraining", "map", "_syn", "_spec", "walking_speed_1", "cutting_speed_2",
             "walking_speed_2", "cutting_speed_1", *extra]
    return d.sort_values(order, kind="stable").drop(columns=["_syn", "_spec"]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description="Trainings per configuration (abilities from config.txt) "
                                            "and their seeds (INITIAL_SEED).")
    p.add_argument("--base", default=DEFAULT_BASE, help=f"Trainings root (default {DEFAULT_BASE})")
    p.add_argument("--pretraining_subfolder", default="with_pretraining",
                   help="Subfolder of --base with the pretrained trainings")
    p.add_argument("--checkpoint", default="750", help="Checkpoint whose presence is checked (default 750)")
    p.add_argument("--output_dir", default=".", help="Where to write the two CSVs")
    p.add_argument("--prefix", default="", help="Optional prefix of the CSV names")
    args = p.parse_args()

    base = Path(args.base.rstrip("/"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = scan(base, args.pretraining_subfolder, args.checkpoint)
    if df.empty:
        raise SystemExit("No Training_* folders found.")

    trainings = sort_table(df, extra=("training_id",))
    cols = CONFIG_KEYS + ["training_id", "seed", f"has_checkpoint_{args.checkpoint}", "problems", "path"]
    list_path = out_dir / f"{args.prefix}trainings_list.csv"
    trainings[cols].to_csv(list_path, index=False)

    summary = sort_table(summarize(trainings, args.checkpoint))
    sum_path = out_dir / f"{args.prefix}trainings_per_configuration.csv"
    summary.to_csv(sum_path, index=False)

    print(f"\n{len(trainings)} trainings in {len(summary)} configurations "
          f"({(trainings['pretraining'] == 'no').sum()} from scratch, "
          f"{(trainings['pretraining'] == 'yes').sum()} with pretraining)")
    n_prob = int((trainings["problems"] != "").sum())
    if n_prob:
        print(f"{n_prob} trainings with problems (see the 'problems' column of {list_path.name})")
    n_dup = int((summary["duplicate_seeds"] != "").sum())
    if n_dup:
        print(f"{n_dup} configurations with repeated seeds (column 'duplicate_seeds')")
    print(f"-> {sum_path}\n-> {list_path}")


if __name__ == "__main__":
    main()
