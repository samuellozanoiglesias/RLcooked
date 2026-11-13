#!/usr/bin/env python3
"""
Compute the map cooperation factor C_map for every map and save it to a lookup file.

C_map is built from two geometric ingredients, each a score in [0, 1]
(0 = no coordination pressure from that ingredient, 1 = maximal pressure).

(i)  Path diversity, D  (dominant)
     A walkable cell w is a *cut vertex* if removing it splits its connected
     walkable region into more pieces. A pair of distinct, mutually reachable
     walkable cells (u, v) is *fragile* if some cell w not in {u, v} is a cut
     vertex separating them, i.e. every route between u and v passes through
     one single cell. F is the fraction of reachable pairs that are fragile
     (every walkable cell is tested). It is mapped to a score with
         D = clip((F - D_OFFSET) / D_SCALE, 0, 1)

(ii) Task-sequence exposure, T  (secondary)
     For each agent start, the *optimal* route through the forced pipeline
         tomato -> cutting board -> counter k -> plate -> same counter k -> delivery
     is found by dynamic programming over the walkable access cells of every
     tile. The chopped tomato and the plate meet on the same counter, so both
     counter stages must use the same counter tile.
       time term     = clip((pipeline_time - TIME_OFFSET) / TIME_SCALE, 0, 1)
       exposure term = clip(E / E_SCALE, 0, 1), where E is the fraction of
                       pipeline steps spent on *unavoidable single-lane* cells.
                       A single-lane cell is walkable and blocked on both sides
                       along at least one axis (two agents cannot pass there);
                       it is unavoidable for a stage if blocking it lengthens or
                       breaks that stage's shortest path.
       T = W_TIME * time term + W_EXPOSURE * exposure term, averaged over agent starts.
     If any agent cannot complete the pipeline at all, T = 1.
     If a required tile type is missing, T is undefined and C_map = 2.0.

Combination:
     C_raw = W_DIVERSITY * D + W_TASK * T
     C_map = 0                     if F == 0 (no single point of failure anywhere)
           = clip(C_raw, 0.2, 2.0) otherwise
     W_DIVERSITY + W_TASK = 2.0, so only the lower clip can be active.

All distances are exact 4-connected shortest paths computed directly on the
grid with BFS (identical to A* distances on a unit-cost grid). If a
precomputed distance cache exists it is used only as a consistency check.

Outputs:
  cooperation_factors.json            map -> C_map (same format as before)
  cooperation_factor_components.csv   every ingredient and diagnostic per map
and a correlation matrix of the ingredients across the map bank is printed.

Usage:
  nohup python compute_cooperation_factor.py > compute_cooperation_factor.log 2>&1 &
  python compute_cooperation_factor.py --maps-dir path/to/maps   # single directory
"""

import argparse
import csv
import glob
import json
import os
from collections import deque
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Parameters (fixed a priori)
# ----------------------------------------------------------------------------
W_DIVERSITY = 1.10          # weight of D
W_TASK = 0.90               # weight of T
W_TIME = 0.65               # weight of the time term inside T
W_EXPOSURE = 0.35           # weight of the exposure term inside T
D_OFFSET = 0.27             # fragile-pair fraction with no diversity pressure
D_SCALE = 0.55              # extra fragile-pair fraction that saturates D
TIME_OFFSET = 35.0          # pipeline time (steps) with no time pressure
TIME_SCALE = 45.0           # extra pipeline time that saturates the time term
E_SCALE = 0.20              # single-lane exposure fraction that saturates the exposure term
C_MIN, C_MAX = 0.2, 2.0
MISSING_TILE_FALLBACK = 2.0

WALKABLE_CHARS = {' ', '1', '2'}
AGENT_CHARS = ('1', '2')
TILE_CHARS = {'tomato': 'T', 'cutting': 'B', 'counter': 'M', 'plate': 'X', 'delivery': 'D'}
NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))

CSV_FIELDS = [
    'map', 'status', 'C_map', 'C_raw', 'D', 'F', 'T', 'time_term', 'exposure_term', 'E',
    'pipeline_time', 'n_walkable', 'n_reachable_pairs', 'n_fragile_pairs',
    'n_cut_vertices', 'max_cut_criticality', 'n_single_lane',
    'n_unavoidable_single_lane_steps',
]


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


# ----------------------------------------------------------------------------
# Grid representation
# ----------------------------------------------------------------------------
class Grid:
    """Walkable cells as a graph. Positions are (row, col)."""

    def __init__(self, lines):
        self.rows = len(lines)
        self.cols = max((len(l) for l in lines), default=0)
        # Ragged lines are padded with walls.
        self.chars = [l.ljust(self.cols, '#') for l in lines]

        self.walkable = np.zeros((self.rows, self.cols), dtype=bool)
        self.tiles = {name: [] for name in TILE_CHARS}
        agents_by_char = {ch: [] for ch in AGENT_CHARS}
        char_to_tile = {v: k for k, v in TILE_CHARS.items()}

        for r, line in enumerate(self.chars):
            for c, ch in enumerate(line):
                if ch in WALKABLE_CHARS:
                    self.walkable[r, c] = True
                if ch in agents_by_char:
                    agents_by_char[ch].append((r, c))
                elif ch in char_to_tile:
                    self.tiles[char_to_tile[ch]].append((r, c))

        self.agents = [p for ch in AGENT_CHARS for p in agents_by_char[ch]]
        self.cells = [(r, c) for r in range(self.rows) for c in range(self.cols)
                      if self.walkable[r, c]]
        self.index = {p: i for i, p in enumerate(self.cells)}
        self.n = len(self.cells)
        self.adj = [[self.index[q] for q in self._walkable_neighbours(p)]
                    for p in self.cells]

    def is_walkable(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and self.walkable[r, c]

    def _walkable_neighbours(self, pos):
        r, c = pos
        return [(r + dr, c + dc) for dr, dc in NEIGHBOURS if self.is_walkable(r + dr, c + dc)]

    def access_cells(self, tile_pos):
        """Indices of walkable cells from which a (non-walkable) tile is used."""
        return [self.index[q] for q in self._walkable_neighbours(tile_pos)]

    def bfs(self, src, blocked=-1):
        """Shortest-path lengths from src (index); -1 = unreachable."""
        dist = np.full(self.n, -1, dtype=np.int64)
        if src == blocked:
            return dist
        dist[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                if v != blocked and dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    def all_pairs_distances(self):
        """n x n float matrix of exact distances, inf = unreachable."""
        if self.n == 0:
            return np.zeros((0, 0))
        d = np.stack([self.bfs(i) for i in range(self.n)]).astype(float)
        d[d < 0] = np.inf
        return d

    def component_labels(self, blocked=-1):
        """Connected-component label per cell (blocked cell gets -1)."""
        labels = np.full(self.n, -1, dtype=np.int64)
        k = 0
        for s in range(self.n):
            if s == blocked or labels[s] >= 0:
                continue
            labels[s] = k
            queue = deque([s])
            while queue:
                u = queue.popleft()
                for v in self.adj[u]:
                    if v != blocked and labels[v] < 0:
                        labels[v] = k
                        queue.append(v)
            k += 1
        return labels, k

    def single_lane_cells(self):
        """Walkable cells blocked on both sides along at least one axis."""
        out = set()
        for i, (r, c) in enumerate(self.cells):
            ns_blocked = not self.is_walkable(r - 1, c) and not self.is_walkable(r + 1, c)
            ew_blocked = not self.is_walkable(r, c - 1) and not self.is_walkable(r, c + 1)
            if ns_blocked or ew_blocked:
                out.add(i)
        return out


# ----------------------------------------------------------------------------
# Ingredient (i): path diversity D
# ----------------------------------------------------------------------------
def path_diversity(grid):
    """Return F (fragile-pair fraction), n_reachable_pairs, n_fragile_pairs, n_cut_vertices, max_criticality."""
    n = grid.n
    if n < 2:
        return 0.0, 0, 0, 0, 0.0

    base_labels, base_k = grid.component_labels()
    same_component = base_labels[:, None] == base_labels[None, :]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    reachable = same_component & upper
    n_reachable = int(reachable.sum())
    if n_reachable == 0:
        return 0.0, 0, 0, 0, 0.0

    fragile = np.zeros((n, n), dtype=bool)
    n_cut = 0
    max_criticality = 0.0
    for w in range(n):
        labels, k = grid.component_labels(blocked=w)
        if k <= base_k:          # removing w splits nothing → not a cut vertex
            continue
        n_cut += 1
        valid = labels >= 0
        separated = (labels[:, None] != labels[None, :]) & valid[:, None] & valid[None, :]
        separated &= reachable
        fragile |= separated
        max_criticality = max(max_criticality, separated.sum() / n_reachable)

    n_fragile = int(fragile.sum())
    return n_fragile / n_reachable, n_reachable, n_fragile, n_cut, float(max_criticality)


# ----------------------------------------------------------------------------
# Ingredient (ii): task-sequence exposure T
# ----------------------------------------------------------------------------
def optimal_pipeline(grid, dist, start):
    """
    Optimal route start → tomato → cutting → counter k → plate → counter k → delivery.
    Returns (total_time, [start, a1, ..., a6]) or (inf, None) if impossible.
    """
    def union_access(tiles):
        return sorted({a for t in tiles for a in grid.access_cells(t)})

    tomato = union_access(grid.tiles['tomato'])
    cutting = union_access(grid.tiles['cutting'])
    plate = union_access(grid.tiles['plate'])
    delivery = union_access(grid.tiles['delivery'])

    best_time, best_route = np.inf, None
    for counter_tile in grid.tiles['counter']:
        counter = grid.access_cells(counter_tile)
        layers = [[start], tomato, cutting, counter, plate, counter, delivery]
        if any(len(layer) == 0 for layer in layers):
            continue

        cost = np.zeros(1)
        back = []
        for prev_layer, layer in zip(layers[:-1], layers[1:]):
            total = cost[:, None] + dist[np.ix_(prev_layer, layer)]
            arg = np.argmin(total, axis=0)
            cost = total[arg, np.arange(len(layer))]
            back.append(arg)

        j = int(np.argmin(cost))
        if cost[j] < best_time:
            best_time = float(cost[j])
            route = [layers[-1][j]]
            for depth in range(len(back) - 1, -1, -1):
                j = int(back[depth][j])
                route.append(layers[depth][j])
            best_route = route[::-1]
    return best_time, best_route


def unavoidable_single_lane_steps(grid, dist, route, single_lane):
    """Number of steps of the route spent on single-lane cells that cannot be bypassed."""
    steps = 0
    for s, e in zip(route[:-1], route[1:]):
        d = dist[s, e]
        if s == e or d <= 1:
            continue
        on_shortest = [c for c in single_lane
                       if c not in (s, e) and dist[s, c] + dist[c, e] == d]
        for c in on_shortest:
            detour = grid.bfs(s, blocked=c)[e]
            if detour < 0 or detour > d:
                steps += 1
    return steps


def task_sequence(grid, dist, single_lane):
    """
    Return dict with T, time_term, E, pipeline_time, unavoidable_steps,
    or None if a required tile type is missing.
    """
    if any(len(grid.tiles[name]) == 0 for name in TILE_CHARS):
        return None

    starts = list(grid.agents)
    if not starts:
        # Same fallback as before: bottom-left and bottom-right walkable cells.
        if grid.n == 0:
            return None
        starts = [max(grid.cells, key=lambda p: (p[0], -p[1])),
                  max(grid.cells, key=lambda p: (p[0], p[1]))]
        print("  Warning: no agent start positions; using bottom-left/right walkable cells")

    per_agent = []
    for start in starts:
        total, route = optimal_pipeline(grid, dist, grid.index[start])
        if route is None or not np.isfinite(total):
            return {'T': 1.0, 'time_term': 1.0, 'exposure_term': 1.0, 'E': 1.0,
                    'pipeline_time': float('inf'), 'unavoidable_steps': -1}
        steps = unavoidable_single_lane_steps(grid, dist, route, single_lane)
        time_term = _clip((total - TIME_OFFSET) / TIME_SCALE)
        exposure = steps / total if total > 0 else 0.0
        exposure_term = _clip(exposure / E_SCALE)
        per_agent.append((W_TIME * time_term + W_EXPOSURE * exposure_term,
                          time_term, exposure_term, exposure, total, steps))

    T, time_term, exposure_term, E, total, steps = (float(np.mean(col)) for col in zip(*per_agent))
    return {'T': T, 'time_term': time_term, 'exposure_term': exposure_term, 'E': E,
            'pipeline_time': total, 'unavoidable_steps': steps}


# ----------------------------------------------------------------------------
# Optional consistency check against the precomputed A* distance cache
# ----------------------------------------------------------------------------
def check_distance_cache(grid, dist, cache_path):
    if not cache_path.exists():
        return
    try:
        data = np.load(str(cache_path))
        D_cache = data['D']
        pos_from = [tuple(p) for p in data['pos_from']]   # (x, y) = (col, row)
        pos_to = [tuple(p) for p in data['pos_to']]
    except Exception as e:
        print(f"  Warning: could not read distance cache {cache_path}: {e}")
        return

    mismatches = checked = 0
    for i, (x1, y1) in enumerate(pos_from):
        a = grid.index.get((y1, x1))
        if a is None:
            continue
        for j, (x2, y2) in enumerate(pos_to):
            b = grid.index.get((y2, x2))
            if b is None:
                continue
            cached = D_cache[i, j]
            ours = dist[a, b]
            checked += 1
            if (np.isnan(cached) and np.isfinite(ours)) or \
               (not np.isnan(cached) and cached != ours):
                mismatches += 1
    if mismatches:
        print(f"  Warning: {mismatches}/{checked} cached distances differ from exact BFS distances")
    else:
        print(f"  Distance cache consistent ({checked} entries checked)")


# ----------------------------------------------------------------------------
# Per-map computation
# ----------------------------------------------------------------------------
def compute_components(map_nr, maps_directory):
    """Compute every ingredient and C_map for one map. Returns a dict with CSV_FIELDS keys."""
    row = {k: '' for k in CSV_FIELDS}
    row['map'] = map_nr

    map_txt_path = os.path.join(maps_directory, f'{map_nr}.txt')
    if not os.path.exists(map_txt_path):
        print(f"Warning: map file {map_txt_path} not found")
        row.update(status='missing_file', C_map=1.0)
        return row

    with open(map_txt_path) as f:
        lines = [line.rstrip('\n') for line in f]
    grid = Grid(lines)
    dist = grid.all_pairs_distances()
    check_distance_cache(grid, dist,
                         Path(maps_directory).parent / "distance_cache" / f"distance_map_{map_nr}.npz")

    single_lane = grid.single_lane_cells()
    F, n_pairs, n_fragile, n_cut, max_crit = path_diversity(grid)
    D = _clip((F - D_OFFSET) / D_SCALE)
    row.update(D=D, F=F, n_walkable=grid.n, n_reachable_pairs=n_pairs, n_fragile_pairs=n_fragile,
               n_cut_vertices=n_cut, max_cut_criticality=max_crit,
               n_single_lane=len(single_lane))

    task = task_sequence(grid, dist, single_lane)
    if task is None:
        row.update(status='missing_tile_type', C_map=MISSING_TILE_FALLBACK)
        return row

    T = task['T']
    C_raw = W_DIVERSITY * D + W_TASK * T
    if F == 0:
        C_map, status = 0.0, 'ok_zero_floor'
    else:
        C_map, status = _clip(C_raw, C_MIN, C_MAX), 'ok'

    row.update(status=status, C_map=C_map, C_raw=C_raw, T=T, time_term=task['time_term'],
               exposure_term=task['exposure_term'], E=task['E'],
               pipeline_time=task['pipeline_time'],
               n_unavoidable_single_lane_steps=task['unavoidable_steps'])
    return row


def calculate_cooperation_factor_expensive(map_nr, maps_directory):
    """Kept for backward compatibility: returns only C_map."""
    row = compute_components(map_nr, maps_directory)
    print_components(row)
    return float(row['C_map'])


def print_components(row):
    print(f"Cooperation factor calculation for {row['map']} [{row['status']}]:")
    if row['D'] != '':
        print(f"  (i)  Path diversity D:  {row['D']:.3f}  "
              f"(F = {row['F']:.3f}: {row['n_fragile_pairs']}/{row['n_reachable_pairs']} pairs fragile, "
              f"{row['n_cut_vertices']} cut vertices)")
    if row['T'] != '':
        print(f"  (ii) Task exposure T:   {row['T']:.3f}  "
              f"(time term {row['time_term']:.3f} from {row['pipeline_time']:.1f} steps, "
              f"exposure term {row['exposure_term']:.3f} from E = {row['E']:.3f})")
        print(f"  C_raw = {W_DIVERSITY}·D + {W_TASK}·T = {row['C_raw']:.4f}")
    print(f"  → C_map = {float(row['C_map']):.4f}")


# ----------------------------------------------------------------------------
# Whole map bank
# ----------------------------------------------------------------------------
def print_correlations(rows):
    ok = [r for r in rows if str(r['status']).startswith('ok')]
    names = ['D', 'time_term', 'exposure_term', 'T']
    if len(ok) < 3:
        print("\nToo few maps for a correlation matrix.")
        return
    X = np.array([[float(r[k]) for k in names] for r in ok])
    keep = [i for i in range(len(names)) if np.std(X[:, i]) > 0]
    if len(keep) < 2:
        print("\nIngredients are constant across maps; no correlation matrix.")
        return
    C = np.corrcoef(X[:, keep], rowvar=False)
    labels = [names[i] for i in keep]
    print(f"\nPearson correlation of ingredients across {len(ok)} maps:")
    print("               " + "".join(f"{l:>14}" for l in labels))
    for l, line in zip(labels, C):
        print(f"{l:>15}" + "".join(f"{v:>14.3f}" for v in line))


def find_map_files(maps_dir=None):
    if maps_dir:
        dirs = [Path(maps_dir)]
    else:
        from spoiled_broth.maps.map_paths import iter_maps_txt_dirs
        dirs = list(iter_maps_txt_dirs())
    files = []
    for d in dirs:
        files.extend(glob.glob(str(Path(d) / "*.txt")))
    return sorted(f for f in files if not f.endswith('_info.txt'))


def compute_all_cooperation_factors(maps_dir=None, output_dir=None):
    output_dir = Path(output_dir) if output_dir else Path(__file__).parent
    output_json = output_dir / 'cooperation_factors.json'
    output_csv = output_dir / 'cooperation_factor_components.csv'

    map_files = find_map_files(maps_dir)
    print(f"Found {len(map_files)} map files. Computing cooperation factors...")
    print("=" * 60)

    rows, factors = [], {}
    for map_file in map_files:
        name = Path(map_file).stem
        print(f"\nProcessing: {name}")
        try:
            row = compute_components(name, str(Path(map_file).parent))
            print_components(row)
        except Exception as e:
            print(f"✗ Error processing {name}: {e}")
            row = {k: '' for k in CSV_FIELDS}
            row.update(map=name, status=f'error: {e}', C_map=1.0)
        rows.append(row)
        factors[name] = float(row['C_map'])

    with open(output_json, 'w') as f:
        json.dump(factors, f, indent=2, sort_keys=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ Saved {len(factors)} cooperation factors to {output_json}")
    print(f"✓ Saved per-map components to {output_csv}")

    if factors:
        values = list(factors.values())
        print(f"\nC_map: min {min(values):.4f}, max {max(values):.4f}, "
              f"mean {np.mean(values):.4f}")
        bad = [r['map'] for r in rows if not str(r['status']).startswith('ok')]
        if bad:
            print(f"  {len(bad)} map(s) used a fallback value: {', '.join(bad)}")
    print_correlations(rows)
    return factors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--maps-dir', help='single directory of map .txt files '
                        '(default: all directories from iter_maps_txt_dirs)')
    parser.add_argument('--output-dir', help='where to write the JSON and CSV '
                        '(default: next to this script)')
    args = parser.parse_args()
    compute_all_cooperation_factors(args.maps_dir, args.output_dir)