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
      Training_2026-05-17_01-29-22/checkpoint_753/simulation_2026_10_29-04_25_41/

abilities_<walk1>_<cut2>:
    walk1 = walking speed of ai_rl_1, cut2 = cutting speed of ai_rl_2;
    the other two speeds (cutting speed of ai_rl_1, walking speed of ai_rl_2) are always 1.0.
The ability pair of a simulation is taken from this folder name (not from config.txt).
Every Training_* and every simulation_* folder is used; the checkpoint folder is
checkpoint_753 unless another one is requested.

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

DEFAULT_CHECKPOINT = "753"
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
    """checkpoint: '753' (literal folder checkpoint_753), 'all', or 'latest'
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

    score_ts = pos.groupby("adj_sec")["score"].max().sort_index()
    team_deliveries = int((score_ts.diff().fillna(0) > 0).sum())

    deliveries_per_agent: dict[str, int] = {}
    for agent_id, grp in pos.groupby("agent_id"):
        diffs = grp.sort_values("adj_sec")["score"].diff().fillna(0)
        deliveries_per_agent[agent_id] = int((diffs > 0).sum())
    if sum(deliveries_per_agent.values()) == 0 and team_deliveries > 0:
        agents = sorted(pos["agent_id"].unique())
        deliveries_per_agent = {a: team_deliveries // len(agents) for a in agents}

    actions = load_actions(simulation_path)
    cuts_per_agent: dict[str, int] = {}
    if not actions.empty and "action_category_name" in actions.columns:
        actions["agent_id"] = actions["agent_id"].astype(str)
        cut_mask = actions["action_category_name"].astype(str).str.contains("cut", case=False, na=False)
        for agent_id, grp in actions.groupby("agent_id"):
            cuts_per_agent[agent_id] = int(cut_mask[grp.index].sum())

    return {"total_deliveries": team_deliveries,
            "deliveries_per_agent": deliveries_per_agent,
            "cuts_per_agent": cuts_per_agent}


def specialization_index(metrics: dict) -> float | None:
    """|(cuts_a1 - cuts_a2)/total_cuts + (del_a2 - del_a1)/total_del| * 100 / 2 (as in Fig. 2);
    agents aligned by id; no deliveries or no cuts -> 0."""
    per_agent_cuts = metrics.get("cuts_per_agent", {})
    per_agent_del = metrics.get("deliveries_per_agent", {})
    agents = sorted(set(per_agent_cuts) & set(per_agent_del))[:2]
    if len(agents) < 2:
        return None
    vals_cuts = [per_agent_cuts[a] for a in agents]
    vals_del = [per_agent_del[a] for a in agents]
    total_cuts, total_del = sum(vals_cuts), sum(vals_del)
    if total_cuts == 0 or total_del == 0:
        return 0.0
    cuts_term = (vals_cuts[0] - vals_cuts[1]) / total_cuts
    del_term = (vals_del[1] - vals_del[0]) / total_del
    return abs(del_term + cuts_term) * 100 / 2


def _agent_slot(agent_id: str, agents: list[str]) -> int:
    """1 or 2: 'ai_rl_1' -> 1, 'ai_rl_2' -> 2; other ids by sorted order."""
    m = re.search(r"(\d+)$", agent_id)
    if m and all(re.search(r"(\d+)$", a) for a in agents):
        nums = sorted(int(re.search(r"(\d+)$", a).group(1)) for a in agents)
        return nums.index(int(m.group(1))) + 1
    return sorted(agents).index(agent_id) + 1


def simulation_record(ref: SimRef) -> dict | None:
    """One flat row per simulated game: speeds, per-agent deliveries / cuts (ai_rl_1, ai_rl_2),
    total_deliveries and specialization_index."""
    met = extract_metrics(ref.path)
    if met is None:
        return None
    row = {"walk": ref.walk1, "cut": ref.cut2, **speeds_of(ref.walk1, ref.cut2),
           "training_id": ref.training_id, "checkpoint": ref.checkpoint, "simulation": ref.simulation,
           "total_deliveries": met["total_deliveries"],
           "specialization_index": specialization_index(met)}
    agents = sorted(set(met["deliveries_per_agent"]) | set(met["cuts_per_agent"]))[:2]
    for k in (1, 2):
        row[f"deliver_ai_rl_{k}"] = 0
        row[f"cut_ai_rl_{k}"] = 0
    for a in agents:
        k = _agent_slot(a, agents)
        row[f"deliver_ai_rl_{k}"] = met["deliveries_per_agent"].get(a, 0)
        row[f"cut_ai_rl_{k}"] = met["cuts_per_agent"].get(a, 0)
    return row


def collect_records(root: Path, pairs=None, checkpoint=DEFAULT_CHECKPOINT, verbose: bool = False,
                    extra: dict | None = None) -> pd.DataFrame:
    """DataFrame of simulation_record() rows for every simulation under root (see iter_simulations)."""
    rows = []
    for ref in iter_simulations(root, pairs, checkpoint):
        rec = simulation_record(ref)
        if rec is None:
            if verbose:
                print(f"  [SKIP] no usable data in {ref.path}")
            continue
        if extra:
            rec.update(extra)
        rows.append(rec)
        if verbose:
            s = rec["specialization_index"]
            print(f"  abilities_{ref.walk1:g}_{ref.cut2:g}/{ref.training_id}/{ref.simulation}  "
                  f"deliveries={rec['total_deliveries']}  spec_idx={'N/A' if s is None else f'{s:.3f}'}")
    return pd.DataFrame(rows)