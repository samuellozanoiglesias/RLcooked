#!/usr/bin/env python3
"""
Compute cooperation factors for all maps and save to lookup file.
Contains the expensive max-flow calculation logic.

Usage:
nohup python compute_cooperation_factor.py > compute_cooperation_factor.log 2>&1 &
"""

import os
import sys
import json
import glob
import numpy as np
from pathlib import Path
from spoiled_broth.maps.map_paths import iter_maps_txt_dirs
from collections import deque

# A* from the engine (same algorithm used to build the distance cache)
try:
    from engine.extensions.topDownGridWorld import a_star as _a_star
    _ASTAR_AVAILABLE = True
except ImportError:
    _ASTAR_AVAILABLE = False


def calculate_cooperation_factor_expensive(map_nr, maps_directory):
    """
    Calculate cooperation factor using 4 key metrics (in order of importance):
    1. Path diversity - how many alternative paths exist to each tile
    2. Task sequence time - time for tomato->cut->counter->plate->counter->delivery + bottleneck exposure
    3. Bottleneck count - number of constrained passages
    4. Tile distribution - variety and spacing of resources (encourages sharing)
    
    Args:
        map_nr: Map identifier
        maps_directory: Path to maps directory
        
    Returns:
        float: Cooperation factor (0.2-2.0)
    """
    # Load map structure
    map_txt_path = os.path.join(maps_directory, f'{map_nr}.txt')
    if not os.path.exists(map_txt_path):
        print(f"Warning: Map file {map_txt_path} not found. Using default cooperation factor 1.0")
        return 1.0
    
    with open(map_txt_path, 'r') as f:
        map_lines = [line.rstrip('\n') for line in f.readlines()]
    
    rows, cols = len(map_lines), len(map_lines[0]) if map_lines else 0
    
    # Load distance cache for faster computation
    cache_dir = Path(maps_directory).parent / "distance_cache"
    cache_path = cache_dir / f"distance_map_{map_nr}.npz"
    distance_cache = None
    pos_from_idx = None
    pos_to_idx = None
    
    if cache_path.exists():
        try:
            cache_data = np.load(str(cache_path))
            D = cache_data['D']
            pos_from = [tuple(p) for p in cache_data['pos_from']]
            pos_to = [tuple(p) for p in cache_data['pos_to']]
            
            # Create lookup dictionaries for fast access
            pos_from_idx = {p: i for i, p in enumerate(pos_from)}
            pos_to_idx = {p: i for i, p in enumerate(pos_to)}
            distance_cache = D
            
            print(f"  Loaded distance cache with {len(pos_from)} x {len(pos_to)} distances")
        except Exception as e:
            print(f"  Warning: Failed to load distance cache: {e}. Using BFS fallback.")
            distance_cache = None
    
    # Create binary accessibility matrix (1=walkable, 0=blocked)
    accessibility = np.zeros((rows, cols), dtype=int)
    
    # Identify key locations by type
    agents = []
    tomato_dispensers = []  # T
    plate_dispensers = []   # X (not P!)
    cutting_boards = []     # B
    counters = []          # M
    deliveries = []        # D
    
    for i, line in enumerate(map_lines):
        for j, char in enumerate(line):
            # Only floor tiles are walkable
            if char in [' ', '1', '2']:
                accessibility[i, j] = 1

            # Record positions of all tile types (walkable or not)
            if char in ['1', '2']:
                agents.append((i, j))
            elif char == 'T':
                tomato_dispensers.append((i, j))
            elif char == 'X':  # Plate dispensers use X, not P
                plate_dispensers.append((i, j))
            elif char == 'B':
                cutting_boards.append((i, j))
            elif char == 'M':
                counters.append((i, j))
            elif char == 'D':
                deliveries.append((i, j))
    
    # BFS helper - returns path length (fallback when cache not available)
    def bfs_distance_fallback(start, end):
        queue = deque([(start[0], start[1], 0)])
        visited = set([start])
        
        while queue:
            x, y, dist = queue.popleft()
            if (x, y) == end:
                return dist
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < rows and 0 <= ny < cols and 
                    accessibility[nx, ny] == 1 and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))
        
        return float('inf')
    
    # Distance lookup - uses cache if available, otherwise BFS
    # All tile positions in this file are (row, col); cache uses (col, row) = (x, y).
    def bfs_distance(start, end):
        if distance_cache is not None and pos_from_idx is not None and pos_to_idx is not None:
            # Convert (row, col) → (col, row) for cache lookup
            start_xy = (start[1], start[0])
            end_xy   = (end[1],   end[0])
            if start_xy in pos_from_idx and end_xy in pos_to_idx:
                dist = distance_cache[pos_from_idx[start_xy], pos_to_idx[end_xy]]
                if not np.isnan(dist):
                    return float(dist)
            # Try reverse direction
            if end_xy in pos_from_idx and start_xy in pos_to_idx:
                dist = distance_cache[pos_from_idx[end_xy], pos_to_idx[start_xy]]
                if not np.isnan(dist):
                    return float(dist)

        # Fallback to BFS
        return bfs_distance_fallback(start, end)

    # Helper: walkable cells directly adjacent (4-connected) to a tile position.
    # Non-walkable objects (T, B, X, D, M ...) must be interacted with from an
    # adjacent walkable cell, so all distance calculations should target/source
    # those neighbors rather than the tile itself.
    def get_walkable_neighbors(pos):
        """Return all walkable 4-connected neighbors of pos."""
        r, c = pos
        result = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and accessibility[nr, nc] == 1:
                result.append((nr, nc))
        return result

    # Identify bottleneck cells (≤2 walkable neighbors)
    bottleneck_cells = set()
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if accessibility[i, j] == 1:
                neighbors = [
                    accessibility[i-1, j], accessibility[i+1, j],
                    accessibility[i, j-1], accessibility[i, j+1]
                ]
                if sum(neighbors) <= 2:
                    bottleneck_cells.add((i, j))
    
    # ------------------------------------------------------------------------
    # METRIC 1 (MOST IMPORTANT): PATH DIVERSITY
    # Test path robustness by blocking bottlenecks one at a time
    # If blocking a bottleneck breaks many paths, there's low diversity = high cooperation needed
    # If paths remain available even with bottlenecks blocked, high diversity = low cooperation needed
    # ------------------------------------------------------------------------

    # Lightweight grid wrapper so we can reuse the engine's A* unchanged.
    # Internally positions are (row, col); the wrapper exposes width=rows and
    # height=cols so that Node(row, col) maps correctly to tiles[row][col].
    class _Tile:
        __slots__ = ('is_walkable',)
        def __init__(self, walkable): self.is_walkable = walkable

    class _GridWrapper:
        """Minimal grid adapter for engine A* that treats one cell as a wall.
        Uses (row, col) = (x, y) convention internally (x=row, y=col).
        width = number of rows (x-extent), height = number of cols (y-extent).
        tiles[row][col] = tile at that position.
        All positions passed in must be (row, col) tuples.
        """
        def __init__(self, blocked_cell=None):
            self.width  = rows   # x = row ranges over 0..rows-1
            self.height = cols   # y = col ranges over 0..cols-1
            self.tiles  = [
                [
                    _Tile(accessibility[r, c] == 1 and (r, c) != blocked_cell)
                    for c in range(cols)
                ]
                for r in range(rows)
            ]

    def astar_path_exists(start_rc, end_rc, blocked_rc):
        """Return True when a path exists between (row,col) positions with blocked_rc walled off.
        Uses 4-directional A* (no diagonals) to match actual agent movement.
        """
        if start_rc == end_rc:
            return True
        if start_rc == blocked_rc or end_rc == blocked_rc:
            return False
        if _ASTAR_AVAILABLE:
            grid = _GridWrapper(blocked_rc)
            # Node(x, y) with our convention x=row, y=col
            path = _a_star.a_star(grid,
                                   _a_star.Node(start_rc[0], start_rc[1]),
                                   _a_star.Node(end_rc[0],   end_rc[1]))
            return path is not None and len(path) > 0
        # Fallback: plain BFS
        queue   = deque([start_rc])
        visited = {start_rc}
        while queue:
            r, c = queue.popleft()
            if (r, c) == end_rc:
                return True
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                nrc = (nr, nc)
                if (0 <= nr < rows and 0 <= nc < cols and
                        nrc != blocked_rc and
                        accessibility[nr, nc] == 1 and
                        nrc not in visited):
                    visited.add(nrc)
                    queue.append(nrc)
        return False
    
    def calculate_path_diversity_metric():
        """
        Measure global path diversity using the distance cache as the baseline set of
        reachable pairs.  For each bottleneck, block it (treat as a wall) and use BFS
        to count how many of those previously-reachable pairs become disconnected.

        High count = low diversity (critical bottlenecks) = high cooperation needed
        Low count = high diversity (many alternatives) = low cooperation needed
        """
        # Build all reachable walkable-to-walkable pairs in (row, col) space.
        # The cache uses engine (col, row) = (x, y), so we convert: rc = (y, x).
        walkable_tiles_rc = [
            (r, c) for r in range(rows) for c in range(cols) if accessibility[r, c] == 1
        ]
        if len(walkable_tiles_rc) < 2:
            return 0.5

        tile_pairs = []
        if distance_cache is not None and pos_from_idx is not None and pos_to_idx is not None:
            n = len(walkable_tiles_rc)
            for i in range(n):
                for j in range(i + 1, n):
                    src_rc = walkable_tiles_rc[i]  # (row, col)
                    dst_rc = walkable_tiles_rc[j]
                    # Convert to cache (col, row) = (x, y) for lookup
                    src_xy = (src_rc[1], src_rc[0])
                    dst_xy = (dst_rc[1], dst_rc[0])
                    d = np.nan
                    if src_xy in pos_from_idx and dst_xy in pos_to_idx:
                        d = distance_cache[pos_from_idx[src_xy], pos_to_idx[dst_xy]]
                    elif dst_xy in pos_from_idx and src_xy in pos_to_idx:
                        d = distance_cache[pos_from_idx[dst_xy], pos_to_idx[src_xy]]
                    if not np.isnan(d) and d > 0:
                        tile_pairs.append((src_rc, dst_rc))
        else:
            # No cache — use BFS to check reachability
            for i in range(len(walkable_tiles_rc)):
                for j in range(i + 1, len(walkable_tiles_rc)):
                    src_rc, dst_rc = walkable_tiles_rc[i], walkable_tiles_rc[j]
                    if bfs_distance_fallback(src_rc, dst_rc) != float('inf'):
                        tile_pairs.append((src_rc, dst_rc))

        if not tile_pairs:
            return 0.5

        # For each bottleneck, block it and count how many cached-reachable pairs break.
        bottleneck_criticality = []
        for bottleneck in bottleneck_cells:
            broken_paths = sum(
                1 for start, end in tile_pairs
                if bottleneck not in (start, end)
                and not astar_path_exists(start, end, bottleneck)
            )
            bottleneck_criticality.append(broken_paths / len(tile_pairs))

        if not bottleneck_criticality:
            return 0.0  # No bottlenecks = high diversity

        avg_criticality = np.mean(bottleneck_criticality)

        # Normalize: 0 % broken → 0.0 constraint (high diversity)
        #            16 %+ broken → 1.0 constraint (low diversity)
        diversity_constraint = min(1.0, avg_criticality * 6.0)

        print(f"  Path diversity: avg {avg_criticality:.1%} of paths broken per bottleneck "
              f"({len(bottleneck_cells)} bottlenecks, {len(tile_pairs)} pairs from cache)")

        return diversity_constraint
    
    path_diversity_score = calculate_path_diversity_metric()
    
    # ------------------------------------------------------------------------
    # METRIC 2: TASK SEQUENCE TIME + BOTTLENECK EXPOSURE
    # Simulate: tomato_dispenser -> cutting_board -> counter -> 
    #           plate_dispenser -> same_counter -> delivery
    # ------------------------------------------------------------------------
    def calculate_task_sequence_metric():
        """Higher time + bottleneck exposure = higher cooperation constraint."""
        if not (agents and tomato_dispensers and cutting_boards and
                counters and plate_dispensers and deliveries):
            missing = []
            if not agents: missing.append('agents')
            if not tomato_dispensers: missing.append('tomato(T)')
            if not cutting_boards: missing.append('cutting(B)')
            if not counters: missing.append('counters(M)')
            if not plate_dispensers: missing.append('plates(X)')
            if not deliveries: missing.append('delivery(D)')
            return None

        if distance_cache is None or pos_from_idx is None or pos_to_idx is None:
            return None

        def cache_dist(src, dst):
            """Look up distance from the pre-computed cache only (no BFS fallback).
            src, dst are (row, col) tuples; cache uses (col, row) = (x, y).
            """
            src_xy = (src[1], src[0])  # (row, col) → (col, row)
            dst_xy = (dst[1], dst[0])
            if src_xy in pos_from_idx and dst_xy in pos_to_idx:
                d = distance_cache[pos_from_idx[src_xy], pos_to_idx[dst_xy]]
                if not np.isnan(d):
                    return float(d)
            # Try reverse direction
            if dst_xy in pos_from_idx and src_xy in pos_to_idx:
                d = distance_cache[pos_from_idx[dst_xy], pos_to_idx[src_xy]]
                if not np.isnan(d):
                    return float(d)
            return float('inf')

        # Recipe sequence: tomato → cutting_board → counter → plate → counter → delivery
        agent_start = agents[0]

        # Non-walkable tiles are interacted with from an adjacent walkable cell.
        # For each tile list, pick the tile whose nearest walkable neighbor has
        # the shortest path from the current position.
        def closest_tile_access(tile_list, from_pos):
            """Return (tile_pos, access_pos) minimising cache_dist(from_pos, access_pos).
            Returns (None, None) when no tile has a reachable walkable neighbor.
            """
            best_tile, best_access, best_dist = None, None, float('inf')
            for tile in tile_list:
                for nb in get_walkable_neighbors(tile):
                    d = cache_dist(from_pos, nb)
                    if d < best_dist:
                        best_dist = d
                        best_tile = tile
                        best_access = nb
            return best_tile, best_access

        closest_tomato_tile,   tomato_access   = closest_tile_access(tomato_dispensers,  agent_start)
        closest_cutting_tile,  cutting_access  = closest_tile_access(cutting_boards,     tomato_access   or agent_start)
        closest_counter1_tile, counter1_access = closest_tile_access(counters,           cutting_access  or agent_start)
        closest_plate_tile,    plate_access    = closest_tile_access(plate_dispensers,   counter1_access or agent_start)
        closest_counter2_tile, counter2_access = closest_tile_access(counters,           plate_access    or agent_start)
        closest_delivery_tile, delivery_access = closest_tile_access(deliveries,         counter2_access or agent_start)

        # Abort early if any stage has no reachable access position
        if any(a is None for a in [tomato_access, cutting_access, counter1_access,
                                    plate_access, counter2_access, delivery_access]):
            missing_accesses = [
                name for name, a in [
                    ("tomato", tomato_access), ("cutting", cutting_access),
                    ("counter1", counter1_access), ("plate", plate_access),
                    ("counter2", counter2_access), ("delivery", delivery_access),
                ] if a is None
            ]
            return 1.0

        sequence = [
            (agent_start,    tomato_access,   "start→tomato_access"),
            (tomato_access,  cutting_access,  "tomato_access→cutting_access"),
            (cutting_access, counter1_access, "cutting_access→counter1_access"),
            (counter1_access, plate_access,   "counter1_access→plate_access"),
            (plate_access,   counter2_access, "plate_access→counter2_access"),
            (counter2_access, delivery_access, "counter2_access→delivery_access"),
        ]

        total_time      = 0
        bottleneck_time = 0

        for start, end, label in sequence:
            dist = cache_dist(start, end)
            if dist == float('inf'):
                return 1.0  # Unreachable = maximum constraint

            total_time += dist

            # Count bottleneck cells that lie on (or very close to) this path segment
            for cell in bottleneck_cells:
                d_s = cache_dist(start, cell)
                d_e = cache_dist(cell, end)
                if d_s != float('inf') and d_e != float('inf') and d_s + d_e <= dist + 2:
                    bottleneck_time += 1

        time_constraint       = min(1.0, max(0.0, (total_time - 15.0) / 50.0))
        bottleneck_constraint = min(1.0, bottleneck_time / 15.0)
        combined_constraint   = 0.7 * time_constraint + 0.3 * bottleneck_constraint

        return combined_constraint
    
    task_sequence_score = calculate_task_sequence_metric()
    
    # If task sequence cannot be computed, return maximum cooperation factor
    if task_sequence_score is None:
        return 2.0
    
    # ------------------------------------------------------------------------
    # METRIC 3: BOTTLENECK COUNT
    # Count cells with ≤2 walkable neighbors (tight corridors)
    # ------------------------------------------------------------------------
    def calculate_bottleneck_metric():
        """More bottleneck cells = more cooperation needed (agents must coordinate)."""
        walkable_cells = np.sum(accessibility)
        if walkable_cells == 0:
            return 0.5
        
        num_bottlenecks = len(bottleneck_cells)
        bottleneck_density = num_bottlenecks / walkable_cells

        return bottleneck_density
    
    bottleneck_score = calculate_bottleneck_metric()
    
    # ------------------------------------------------------------------------
    # METRIC 4: TILE DISTRIBUTION (variety and spacing)
    # More types + farther apart = less sharing needed
    # Only consider task-critical tiles: dispensers, cutting boards, delivery
    # (Counters are everywhere and not a bottleneck resource)
    # ------------------------------------------------------------------------
    def calculate_tile_distribution_metric():
        """Analyze tile variety and distribution of critical resources."""
        tile_types = {
            'tomato': tomato_dispensers,
            'plate': plate_dispensers,
            'cutting': cutting_boards,
            'delivery': deliveries
        }

        # Count distinct tile types present
        types_present = sum(1 for tiles in tile_types.values() if len(tiles) > 0)

        # Count total instances
        total_instances = sum(len(tiles) for tiles in tile_types.values())

        if total_instances == 0:
            return 0.5

        # Calculate average distance between instances of same type
        avg_distances = []
        for type_name, tile_list in tile_types.items():
            if len(tile_list) >= 2:
                distances = []
                for i in range(len(tile_list)):
                    for j in range(i+1, len(tile_list)):
                        nbs_i = get_walkable_neighbors(tile_list[i])
                        nbs_j = get_walkable_neighbors(tile_list[j])
                        if not nbs_i or not nbs_j:
                            continue
                        # Pick the pair of neighbors with the shortest inter-distance
                        best_dist = float('inf')
                        best_nb_i, best_nb_j = nbs_i[0], nbs_j[0]
                        for ni in nbs_i:
                            for nj in nbs_j:
                                d = bfs_distance(ni, nj)
                                if d < best_dist:
                                    best_dist = d
                                    best_nb_i, best_nb_j = ni, nj
                        if best_dist != float('inf'):
                            distances.append(best_dist)
                if distances:
                    type_avg = np.mean(distances)
                    avg_distances.append(type_avg)

        # Well-distributed = multiple instances far apart = low constraint
        variety_score = types_present / 4.0  # 4 critical tile types (no counters)

        if avg_distances:
            global_avg_spacing = np.mean(avg_distances)
            spacing_score = min(1.0, global_avg_spacing / 10.0)
        else:
            # If no duplicates, assume moderate distribution
            spacing_score = 0.5

        # Low variety + close spacing = high constraint (need to share)
        distribution_constraint = 1.0 - (0.5 * variety_score + 0.5 * spacing_score)

        return distribution_constraint
    
    tile_distribution_score = calculate_tile_distribution_metric()
    
    # ------------------------------------------------------------------------
    # WEIGHTED COMBINATION (in order of importance)
    # ------------------------------------------------------------------------
    weights = {
        'path_diversity': 1.40,       # MOST IMPORTANT: alternative routes
        'task_sequence': 0.50,        # Task time + bottleneck exposure
        'bottleneck_count': 0.05,     # Raw bottleneck count
        'tile_distribution': 0.05     # Resource sharing requirements
    }
    
    cooperation_score = (
        weights['path_diversity'] * path_diversity_score +
        weights['task_sequence'] * task_sequence_score +
        weights['bottleneck_count'] * bottleneck_score +
        weights['tile_distribution'] * tile_distribution_score
    )
    
    # Clamp to valid range (0.2-2.0) to avoid extreme values
    cooperation_factor = max(0.2, min(2.0, cooperation_score)) if min(path_diversity_score, task_sequence_score, bottleneck_score, tile_distribution_score) > 0 else 0.0
    
    # Debug information
    walkable = np.sum(accessibility)
    print(f"Cooperation factor calculation for {map_nr}:")
    print(f"  1. Path diversity:     {path_diversity_score:.3f} (weight: {weights['path_diversity']})")
    print(f"  2. Task sequence:      {task_sequence_score:.3f} (weight: {weights['task_sequence']})")
    print(f"  3. Bottleneck count:   {bottleneck_score:.3f} ({len(bottleneck_cells)} bottlenecks / {walkable} walkable = {len(bottleneck_cells)/walkable:.3f}, weight: {weights['bottleneck_count']})")
    print(f"  4. Tile distribution:  {tile_distribution_score:.3f} (weight: {weights['tile_distribution']})")
    print(f"  → Final cooperation factor: {cooperation_factor:.3f}")
    
    return cooperation_factor


def compute_all_cooperation_factors():
    """Compute cooperation factors for all map files in the maps directory."""
    
    output_file = Path(__file__).parent / 'cooperation_factors.json'
    
    print("Scanning for map files...")
    map_files = []
    for maps_txt_dir in iter_maps_txt_dirs():
        map_files.extend(glob.glob(str(maps_txt_dir / "*.txt")))
    map_files = [f for f in map_files if not f.endswith('_info.txt')]  # Exclude info files
    
    cooperation_factors = {}
    
    print(f"\nFound {len(map_files)} map files. Computing cooperation factors...")
    print("=" * 60)
    
    for map_file in sorted(map_files):
        map_name = Path(map_file).stem  # Get filename without .txt extension
        
        try:
            print(f"\nProcessing: {map_name}")
            
            # Calculate cooperation factor using the expensive method
            cooperation_factor = calculate_cooperation_factor_expensive(map_name, str(Path(map_file).parent))
            
            cooperation_factors[map_name] = cooperation_factor
            print(f"✓ {map_name}: {cooperation_factor:.3f}")
            
        except Exception as e:
            print(f"✗ Error processing {map_name}: {e}")
            cooperation_factors[map_name] = 1.0  # Default fallback
    
    # Save to JSON file
    print(f"\nSaving cooperation factors to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(cooperation_factors, f, indent=2, sort_keys=True)
    
    print(f"✓ Successfully saved {len(cooperation_factors)} cooperation factors")
    print(f"✓ File saved to: {output_file}")
    
    # Print summary statistics
    values = list(cooperation_factors.values())
    print(f"\nSummary Statistics:")
    print(f"  Min cooperation factor: {min(values):.3f}")
    print(f"  Max cooperation factor: {max(values):.3f}")
    print(f"  Mean cooperation factor: {sum(values)/len(values):.3f}")
    
    return cooperation_factors


if __name__ == "__main__":
    compute_all_cooperation_factors()