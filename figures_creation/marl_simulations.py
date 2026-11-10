#!/usr/bin/env python3
"""
marl_simulations.py -- shared locator / loader for the MARL SIMULATION tree.

Every figure script imports this module (it must sit in the same folder, like
paper_stats.py). The simulated games live in

  <base>/data/samuel_lozano/cooked/[<study_name>/]<game_version>/simulations/<init_type>/
      map_<map_name>/synergy_<synergy>/specialized_<specialization>/
      abilities_<walk1>_<cut2>/Training_<training_id>/checkpoint_<checkpoint>/simulation_<simulation_id>/

e.g.
  /data/samuel_lozano/cooked/classic_collision/simulations/empty_init/
      map_baseline_division_of_labor_large/synergy_1.35/specialized_0.25/abilities_0.1_0.8/
      Training_2026-05-17_01-29-22/checkpoint_750/simulation_2026_10_29-04_25_41/

abilities_<walk1>_<cut2>:
    walk1 = walking speed of ai_rl_1, cut2 = cutting speed of ai_rl_2;
    the other two speeds (cutting speed of ai_rl_1, walking speed of ai_rl_2) are always 1.0.
The ability pair of a simulation is taken from this folder name (not from config.txt).
Every Training_* and every simulation_* folder is used; the checkpoint folder is
checkpoint_750 unless another one is requested.

Per-agent counts: deliveries come from the positions files, cuts from the actions file.
The two agents are matched across both files by the number in their id ('ai_rl_1' = '1' =
1.0), and an agent missing from a file counts as 0, so specialization_index is always defined
(it is computed from the four columns deliver_ai_rl_k / cut_ai_rl_k written in each record).

Folder values are matched NUMERICALLY (synergy_1.35 == synergy_1.350, abilities_1_1 ==
abilities_1.0_1.0), so the scripts do not depend on how the numbers were formatted.

Author: Samuel Lozano
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

DEFAULT_CHECKPOINT = "750"
DEFAULT_INIT_TYPE = "empty_init"
FIXED_CUT1 = 1.0          # cutting speed of ai_rl_1 (always 1 in this tree)
FIXED_WALK2 = 1.0         # walking speed of ai_rl_2 (always 1 in this tree)
SPEED_TOL = 1e-6

CLUSTER_DIRS = {
    "cuenca": "",
    "brigit": "/mnt/lustre/home/samuloza/",
    "local": "C:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado",
}


# ===========================================================================
# 1. PATHS
# ===========================================================================
def cluster_base_dir(cluster: str | None) -> str:
    """Base folder of a cluster. Unknown names fall back to the project's AnalysisConfig
    (if importable), else the name is taken as a literal base path."""
    cluster = cluster or "cuenca"
    if cluster in CLUSTER_DIRS:
        return CLUSTER_DIRS[cluster]
    try:
        from spoiled_broth.analysis.utils import AnalysisConfig   # optional project module
        paths = AnalysisConfig().cluster_paths
        if cluster in paths:
            return paths[cluster] or ""
    except Exception:
        pass
    return cluster


def is_none(v) -> bool:
    return v is None or str(v).strip().lower() in ("", "none", "null", "-")


def _to_float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _num_dir(parent: Path, prefix: str, value) -> Path:
    """Child '<prefix><value>' of parent, matched numerically when possible.
    Returns the literal path (which may not exist) when nothing matches."""
    literal = parent / f"{prefix}{value}"
    if literal.exists():
        return literal
    target = _to_float(value)
    if target is not None and parent.is_dir():
        for child in sorted(parent.glob(f"{prefix}*")):
            v = _to_float(child.name[len(prefix):])
            if child.is_dir() and v is not None and abs(v - target) < SPEED_TOL:
                return child
    return literal


def simulations_base(base_dir: str, game_version: str, init_type: str = DEFAULT_INIT_TYPE,
                     study_name: str | None = None) -> Path:
    """<base>/data/samuel_lozano/cooked/[<study>/]<game_version>/simulations/<init_type>"""
    root = Path(base_dir or "/") / "data" / "samuel_lozano" / "cooked"
    if study_name:
        root = root / study_name
    return root / game_version / "simulations" / init_type


def map_root(base_dir: str, game_version: str, map_name: str, synergy, specialization,
             init_type: str = DEFAULT_INIT_TYPE, study_name: str | None = None,
             path_template: str | None = None) -> Path:
    """Folder that contains the abilities_* folders of one map / reward configuration.

    synergy / specialization = 'none' drops that level. path_template (optional) is an
    explicit folder with '{map}' substituted; it replaces the whole resolution."""
    if path_template:
        return Path(path_template.replace("{map}", map_name))
    root = simulations_base(base_dir, game_version, init_type, study_name) / f"map_{map_name}"
    if not is_none(synergy):
        root = _num_dir(root, "synergy_", synergy)
    if not is_none(specialization):
        root = _num_dir(root, "specialized_", specialization)
    return root


def list_values(parent: Path, prefix: str) -> list[float]:
    """Numeric values of the '<prefix><value>' sub-folders of parent (e.g. synergy_)."""
    out = []
    if parent.is_dir():
        for child in parent.glob(f"{prefix}*"):
            v = _to_float(child.name[len(prefix):])
            if child.is_dir() and v is not None:
                out.append(v)
    return sorted(set(out))


# ===========================================================================
# 2. ABILITIES / TRAININGS / SIMULATIONS
# ===========================================================================
_ABIL_RE = re.compile(r"^abilities_([0-9.]+)_([0-9.]+)$")


def parse_abilities(name: str):
    """'abilities_0.1_0.8' -> (0.1, 0.8) = (walk speed of ai_rl_1, cut speed of ai_rl_2)."""
    m = _ABIL_RE.match(name)
    if not m:
        return None
    w, c = _to_float(m.group(1)), _to_float(m.group(2))
    return None if w is None or c is None else (w, c)


def speeds_of(walk1: float, cut2: float) -> dict:
    """The four speeds of an abilities_<walk1>_<cut2> folder."""
    return {"walking_speed_1": float(walk1), "cutting_speed_1": FIXED_CUT1,
            "walking_speed_2": FIXED_WALK2, "cutting_speed_2": float(cut2)}


def list_ability_dirs(root: Path) -> list[tuple[float, float, Path]]:
    out = []
    if root.is_dir():
        for child in sorted(root.glob("abilities_*")):
            p = parse_abilities(child.name)
            if child.is_dir() and p is not None:
                out.append((p[0], p[1], child))
    return out


def find_ability_dir(root: Path, walk1: float, cut2: float, tol: float = SPEED_TOL) -> Path | None:
    for w, c, path in list_ability_dirs(root):
        if abs(w - float(walk1)) < tol and abs(c - float(cut2)) < tol:
            return path
    return None


def _checkpoint_dirs(training_dir: Path, checkpoint) -> list[Path]:
    """checkpoint: '750' (literal folder checkpoint_750), 'all', or 'latest'
    (highest-numbered checkpoint_<int>)."""
    ck = DEFAULT_CHECKPOINT if checkpoint is None else str(checkpoint).strip()
    if ck.lower() == "all":
        return sorted(p for p in training_dir.glob("checkpoint_*") if p.is_dir())
    if ck.lower() == "latest":
        numbered = [p for p in training_dir.glob("checkpoint_*")
                    if p.is_dir() and p.name[len("checkpoint_"):].isdigit()]
        return [max(numbered, key=lambda p: int(p.name[len("checkpoint_"):]))] if numbered else []
    d = training_dir / f"checkpoint_{ck}"
    return [d] if d.is_dir() else []


@dataclass(frozen=True)
class SimRef:
    walk1: float            # walking speed of ai_rl_1
    cut2: float             # cutting speed of ai_rl_2
    training_id: str        # 'Training_<id>' folder name
    checkpoint: str         # '<n>' of checkpoint_<n>
    simulation: str         # 'simulation_<id>' folder name
    path: Path

    @property
    def pair(self):
        return (self.walk1, self.cut2)


def iter_simulations(root: Path, pairs: Iterable[tuple[float, float]] | None = None,
                     checkpoint=DEFAULT_CHECKPOINT) -> Iterator[SimRef]:
    """Every simulation_* folder under root/abilities_*/Training_*/checkpoint_<ck>/.
    pairs = None -> every abilities_* folder; else only the requested (walk1, cut2) pairs."""
    wanted = None if pairs is None else [(float(w), float(c)) for w, c in pairs]
    for w, c, adir in list_ability_dirs(root):
        if wanted is not None and not any(abs(w - a) < SPEED_TOL and abs(c - b) < SPEED_TOL
                                          for a, b in wanted):
            continue
        for tdir in sorted(p for p in adir.glob("Training_*") if p.is_dir()):
            for cdir in _checkpoint_dirs(tdir, checkpoint):
                for sdir in sorted(p for p in cdir.glob("simulation_*") if p.is_dir()):
                    yield SimRef(w, c, tdir.name, cdir.name[len("checkpoint_"):], sdir.name, sdir)


def describe_root(root: Path, checkpoint=DEFAULT_CHECKPOINT) -> str:
    """One-line inventory of a map root (for logs)."""
    if not root.exists():
        return f"[WARN] simulations root not found: {root}"
    dirs = list_ability_dirs(root)
    return (f"{root}: {len(dirs)} abilities_* folder(s), checkpoint_{checkpoint}" if dirs
            else f"[WARN] no abilities_* folders in {root}")


# ===========================================================================
# 3. METRICS OF ONE SIMULATED GAME  (same loaders as Fig. 2)
# ===========================================================================
def _extract_agent_id_from_filename(file_path: Path, prefixes: list[str]) -> str:
    stem = file_path.stem
    for prefix in prefixes:
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def load_positions(simulation_path: Path) -> pd.DataFrame | None:
    sim_csv = simulation_path / "simulation.csv"
    if sim_csv.exists():
        df = pd.read_csv(sim_csv)
        for old, new in [("tile_x", "x"), ("tile_y", "y"), ("tick", "frame")]:
            if new not in df.columns and old in df.columns:
                df[new] = df[old]
        return df
    pos_files = sorted(simulation_path.glob("positions_*.csv")) or \
        sorted(simulation_path.glob("human_like_positions_*.csv"))
    if not pos_files:
        return None
    frames = []
    for pf in pos_files:
        try:
            d = pd.read_csv(pf)
        except Exception:
            continue
        if "agent_id" not in d.columns:
            d["agent_id"] = _extract_agent_id_from_filename(pf, ["positions_", "human_like_positions_"])
        for old, new in [("tile_x", "x"), ("tile_y", "y"), ("tick", "frame")]:
            if new not in d.columns and old in d.columns:
                d[new] = d[old]
        frames.append(d)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else None


def _normalize_actions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "agent_id" not in df.columns:
        df["agent_id"] = df["player_id"] if "player_id" in df.columns else "unknown"
    if "action_category_name" not in df.columns:
        df["action_category_name"] = np.nan
    for col in ["action_long", "action_name", "action_type", "action"]:
        if col in df.columns:
            df["action_category_name"] = df["action_category_name"].fillna(df[col])
    df["action_category_name"] = df["action_category_name"].fillna("unknown")
    return df


def load_actions(simulation_path: Path) -> pd.DataFrame:
    meaningful = simulation_path / "meaningful_actions.csv"
    if meaningful.exists():
        try:
            return _normalize_actions(pd.read_csv(meaningful))
        except Exception:
            pass
    hl_files = sorted(simulation_path.glob("human_like_actions_*.csv"))
    if hl_files:
        frames = []
        for af in hl_files:
            try:
                d = pd.read_csv(af)
            except Exception:
                continue
            if "agent_id" not in d.columns:
                d["agent_id"] = d["player_id"] if "player_id" in d.columns else \
                    _extract_agent_id_from_filename(af, ["human_like_actions_"])
            frames.append(d)
        if frames:
            return _normalize_actions(pd.concat(frames, ignore_index=True, sort=False))
    actions_csv = simulation_path / "actions.csv"
    if actions_csv.exists():
        try:
            return _normalize_actions(pd.read_csv(actions_csv))
        except Exception:
            pass
    return pd.DataFrame(columns=["agent_id", "action_category_name"])


def read_init_period(config_path: Path) -> float:
    if not config_path.exists():
        return 0.0
    m = re.search(r"AGENT_INITIALIZATION_PERIOD\s*:\s*([\d.]+)", config_path.read_text(errors="replace"))
    return float(m.group(1)) if m else 0.0


def extract_metrics(simulation_path: Path, initialization_period: float | None = None) -> dict | None:
    """{'total_deliveries', 'deliveries_per_agent', 'cuts_per_agent'} of one simulated game.
    initialization_period None -> read from the simulation's config.txt."""
    if initialization_period is None:
        initialization_period = read_init_period(simulation_path / "config.txt")
    pos = load_positions(simulation_path)
    if pos is None or pos.empty or "score" not in pos.columns or "agent_id" not in pos.columns:
        return None
    pos = pos.copy()
    pos["agent_id"] = pos["agent_id"].astype(str)
    for _col in ("second", "frame", "score"):
        if _col in pos.columns:
            pos[_col] = pd.to_numeric(pos[_col], errors="coerce")
    if "second" in pos.columns:
        pos = pos.dropna(subset=["second"])
    elif "frame" in pos.columns:
        pos = pos.dropna(subset=["frame"])
    pos = pos.dropna(subset=["score"])
    if pos.empty:
        return None
    if "second" in pos.columns:
        pos["adj_sec"] = (pos["second"] - initialization_period).clip(lower=0)
    elif "frame" in pos.columns:
        pos["adj_sec"] = (pos["frame"] / 30.0 - initialization_period).clip(lower=0)
    else:
        pos["adj_sec"] = 0.0

    # Deliveries of each agent = FINAL value of that agent's score column (last row in time
    # order of its positions_<agent>.csv). The old count of score increases is kept only as a
    # diagnostic (score_increments) to compare the two definitions.
    deliveries_per_agent: dict[str, float] = {}
    increments_per_agent: dict[str, int] = {}
    for agent_id, grp in pos.groupby("agent_id"):
        s = grp.sort_values("adj_sec", kind="stable")["score"]
        last = float(s.iloc[-1])
        deliveries_per_agent[agent_id] = int(last) if last.is_integer() else last
        increments_per_agent[agent_id] = int((s.diff().fillna(0) > 0).sum())
    score_increments = int(sum(increments_per_agent.values()))

    actions = load_actions(simulation_path)
    cuts_per_agent: dict[str, int] = {}
    if not actions.empty and "action_category_name" in actions.columns:
        actions["agent_id"] = actions["agent_id"].astype(str)
        cut_mask = actions["action_category_name"].astype(str).str.contains("cut", case=False, na=False)
        for agent_id, grp in actions.groupby("agent_id"):
            cuts_per_agent[agent_id] = int(cut_mask[grp.index].sum())

    # Game score = final score of agent 1 + final score of agent 2
    total = sum(deliveries_per_agent.values())
    return {"total_deliveries": int(total) if float(total).is_integer() else total,
            "score_increments": score_increments,
            "deliveries_per_agent": deliveries_per_agent,
            "cuts_per_agent": cuts_per_agent}


_IGNORED_IDS = {"", "nan", "none", "unknown", "null"}


def _id_number(agent_id: str):
    """Number that identifies an agent id: 'ai_rl_1' -> 1, '1' -> 1, '1.0' -> 1, 'agent-2' -> 2;
    None when the id has no number."""
    s = str(agent_id).strip()
    try:
        f = float(s)
        if f.is_integer():
            return int(f)
    except ValueError:
        pass
    m = re.search(r"(\d+)\s*$", s)
    return int(m.group(1)) if m else None


def _valid_ids(d: dict) -> list[str]:
    return [a for a in d if str(a).strip().lower() not in _IGNORED_IDS]


def agent_slots(metrics: dict) -> dict:
    """Per-agent counts in slots 1 / 2 (ai_rl_1, ai_rl_2), matched across the positions file
    (deliveries) and the actions file (cuts) even when the two files write the agent ids
    differently ('ai_rl_1' vs '1' vs 1.0). Returns {'d1', 'd2', 'c1', 'c2', 'matched'}.

    Ids are matched by their number; the two numbers present (e.g. 1, 2 or 0, 1) become slots 1
    and 2 in increasing order. If the two files use different numberings (e.g. 0/1 in one and
    1/2 in the other), each file is ranked on its own. matched = False when the alignment had
    to fall back to that per-file ranking or an id had no number."""
    dels, cuts = metrics.get("deliveries_per_agent", {}), metrics.get("cuts_per_agent", {})
    d_ids, c_ids = _valid_ids(dels), _valid_ids(cuts)
    out = {"d1": 0, "d2": 0, "c1": 0, "c2": 0, "matched": True}

    def by_number(ids):
        nums = {a: _id_number(a) for a in ids}
        return nums if all(n is not None for n in nums.values()) else None

    dn, cn = by_number(d_ids), by_number(c_ids)
    if dn is not None and cn is not None:
        union = sorted(set(dn.values()) | set(cn.values()))
        if len(union) <= 2:
            slot = {n: k + 1 for k, n in enumerate(union)}
            for a, n in dn.items():
                out[f"d{slot[n]}"] += dels[a]
            for a, n in cn.items():
                out[f"c{slot[n]}"] += cuts[a]
            return out
    # fallback: rank the ids of each file separately (numbers if possible, else names)
    out["matched"] = False
    for ids, nums, src, key in ((d_ids, dn, dels, "d"), (c_ids, cn, cuts, "c")):
        order = sorted(ids, key=lambda a: (nums[a], a) if nums else (0, str(a)))[:2]
        for k, a in enumerate(order):
            out[f"{key}{k + 1}"] = src[a]
    return out


def specialization_from_counts(d1, d2, c1, c2) -> float:
    """|(c1 - c2)/(c1 + c2) + (d2 - d1)/(d1 + d2)| * 100 / 2 (as in Fig. 2);
    no deliveries or no cuts -> 0."""
    total_cuts, total_del = c1 + c2, d1 + d2
    if total_cuts == 0 or total_del == 0:
        return 0.0
    return abs((c1 - c2) / total_cuts + (d2 - d1) / total_del) * 100 / 2


def specialization_index(metrics: dict) -> float:
    """Specialization index of one simulated game (agents aligned with agent_slots)."""
    s = agent_slots(metrics)
    return specialization_from_counts(s["d1"], s["d2"], s["c1"], s["c2"])


def recompute_specialization(df: pd.DataFrame) -> pd.DataFrame:
    """Fill specialization_index from the per-agent columns deliver_ai_rl_k / cut_ai_rl_k
    (e.g. for record CSVs written before this fix, where the column is empty)."""
    cols = ["deliver_ai_rl_1", "deliver_ai_rl_2", "cut_ai_rl_1", "cut_ai_rl_2"]
    if not set(cols) <= set(df.columns):
        return df
    df = df.copy()
    d1, d2, c1, c2 = (pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(float) for c in cols)
    df["specialization_index"] = [specialization_from_counts(*v) for v in zip(d1, d2, c1, c2)]
    return df


def simulation_record(ref: SimRef) -> dict | None:
    """One flat row per simulated game: speeds, per-agent deliveries / cuts (ai_rl_1, ai_rl_2),
    total_deliveries and specialization_index (computed from those same four columns)."""
    met = extract_metrics(ref.path)
    if met is None:
        return None
    s = agent_slots(met)
    row = {"walk": ref.walk1, "cut": ref.cut2, **speeds_of(ref.walk1, ref.cut2),
           "training_id": ref.training_id, "checkpoint": ref.checkpoint, "simulation": ref.simulation,
           # game score = deliveries of agent 1 + deliveries of agent 2, by construction
           "total_deliveries": s["d1"] + s["d2"],
           "specialization_index": specialization_from_counts(s["d1"], s["d2"], s["c1"], s["c2"]),
           "deliver_ai_rl_1": s["d1"], "cut_ai_rl_1": s["c1"],
           "deliver_ai_rl_2": s["d2"], "cut_ai_rl_2": s["c2"],
           "agent_ids_matched": s["matched"],
           # agents found in each file (2 = both); an agent missing from a file counts as 0
           "n_agents_positions": len(_valid_ids(met["deliveries_per_agent"])),
           "n_agents_actions": len(_valid_ids(met["cuts_per_agent"])),
           "score_increments": met["score_increments"]}            # diagnostic only
    raw_total = sum(met["deliveries_per_agent"][a] for a in _valid_ids(met["deliveries_per_agent"]))
    if raw_total != row["total_deliveries"]:
        row["_lost"] = (raw_total, row["total_deliveries"])
    if not s["matched"]:
        row["_ids"] = (sorted(map(str, met["deliveries_per_agent"])), sorted(map(str, met["cuts_per_agent"])))
    return row


def collect_records(root: Path, pairs=None, checkpoint=DEFAULT_CHECKPOINT, verbose: bool = False,
                    extra: dict | None = None) -> pd.DataFrame:
    """DataFrame of simulation_record() rows for every simulation under root (see iter_simulations)."""
    rows, unmatched = [], []
    for ref in iter_simulations(root, pairs, checkpoint):
        rec = simulation_record(ref)
        if rec is None:
            if verbose:
                print(f"  [SKIP] no usable data in {ref.path}")
            continue
        ids = rec.pop("_ids", None)
        if ids is not None:
            unmatched.append((ref.path, ids))
        lost = rec.pop("_lost", None)
        if lost is not None:
            print(f"  [WARN] {ref.path}: sum of the agents' final scores = {lost[0]}, but only "
                  f"{lost[1]} assigned to ai_rl_1 / ai_rl_2 (more than two agent ids?)")
        if extra:
            rec.update(extra)
        rows.append(rec)
        if verbose:
            print(f"  abilities_{ref.walk1:g}_{ref.cut2:g}/{ref.training_id}/{ref.simulation}  "
                  f"deliveries={rec['total_deliveries']}  spec_idx={rec['specialization_index']:.3f}")
    if unmatched:
        path, (d_ids, c_ids) = unmatched[0]
        print(f"  [WARN] {len(unmatched)} simulated games: agent ids of the positions files {d_ids} and "
              f"of the actions file {c_ids} could not be matched by number; each file was ranked "
              f"separately (first case: {path})")
    df = pd.DataFrame(rows)
    if len(df):
        assert (df["total_deliveries"] == df["deliver_ai_rl_1"] + df["deliver_ai_rl_2"]).all()
        assert df["specialization_index"].notna().all()
        n_diff = int((df["total_deliveries"] != df["score_increments"]).sum())
        if n_diff:
            print(f"  [INFO] {n_diff} of {len(df)} simulated games: total_deliveries "
                  f"(sum of the final scores) differs from the old count of score increases")
    return df