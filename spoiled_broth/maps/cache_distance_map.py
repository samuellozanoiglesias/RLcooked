"""
Cache and compute all-pairs shortest path distances from each floor tile to each clickable tile for a given map.

Uses A* from engine/extensions/topDownGridWorld/a_star.py

Usage:
nohup python cache_distance_map.py > cache_distance.log 2>&1 &
"""

import os
import shutil
import pickle
import numpy as np
from engine.extensions.topDownGridWorld import a_star
import glob
from spoiled_broth.game import SpoiledBroth

def get_floor_tiles(grid):
	floor_tiles = []
	for x in range(grid.width):
		for y in range(grid.height):
			tile = grid.tiles[x][y]
			if hasattr(tile, 'is_walkable') and tile.is_walkable:
				floor_tiles.append((x, y))
	return floor_tiles

def get_clickable_tiles(game):
	clickable = []
	grid = game.grid
	for idx in game.clickable_indices:
		x = idx % grid.width
		y = idx // grid.width
		tile = grid.tiles[x][y]
		clickable.append((idx, x, y))
	return clickable

def compute_distance_map(game, grid, cache_path=None):
	"""
	Returns: dict[(from_x, from_y)][(to_x, to_y)] = path_length (float or None)
	Computes distances from all floor tiles to all reachable tiles (both floor and clickable).
	Optionally caches to a pickle file.
	"""
	floor_tiles = get_floor_tiles(grid)
	clickable_tiles = get_clickable_tiles(game)
	
	# Create a combined list of all target tiles (floor + clickable positions)
	all_target_positions = set()
	
	# Add all floor tile positions
	for floor_pos in floor_tiles:
		all_target_positions.add(floor_pos)
	
	# Add all clickable tile positions  
	for idx, x, y in clickable_tiles:
		all_target_positions.add((x, y))
	
	# Helper to get adjacent walkable tiles
	def get_adjacent_walkable(x, y):
		adj = []
		for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
			nx, ny = x+dx, y+dy
			if 0 <= nx < grid.width and 0 <= ny < grid.height:
				tile = grid.tiles[nx][ny]
				if hasattr(tile, 'is_walkable') and tile.is_walkable:
					adj.append((nx, ny))
		return adj

	distance_map = {}
	max_distance = 0
	
	for from_xy in floor_tiles:
		distance_map[from_xy] = {}
		from_node = a_star.Node(*from_xy)
		
		for to_xy in all_target_positions:
			to_x, to_y = to_xy
			
			# If it's the same position, distance is 0
			if from_xy == to_xy:
				distance_map[from_xy][to_xy] = 0.0
				continue
			
			# Check if target position is walkable (for floor tiles)
			target_tile = grid.tiles[to_x][to_y]
			if hasattr(target_tile, 'is_walkable') and target_tile.is_walkable:
				# Direct path to walkable tile
				to_node = a_star.Node(to_x, to_y)
				path = a_star.find_path(grid, from_node, to_node)
				dist = a_star.path_length(path) if path else None
				distance_map[from_xy][to_xy] = dist
				if dist is not None and dist > max_distance:
					max_distance = dist
			else:
				# For non-walkable tiles (like clickable tiles), find path to adjacent walkable tiles
				adj_walkable = get_adjacent_walkable(to_x, to_y)
				min_dist = None
				for adj_xy in adj_walkable:
					to_node = a_star.Node(*adj_xy)
					path = a_star.find_path(grid, from_node, to_node)
					dist = a_star.path_length(path) if path else None
					# Add 1 for the step from adjacent tile to the target tile
					if dist is not None:
						dist += 1.0
					if dist is not None and (min_dist is None or dist < min_dist):
						min_dist = dist
				
				distance_map[from_xy][to_xy] = min_dist
				if min_dist is not None and min_dist > max_distance:
					max_distance = min_dist
	if cache_path:
		# Save a fast numpy representation (.npz) with separate from/to position lists
		try:
			# Collect ordered lists
			pos_from = sorted(distance_map.keys())
			pos_to_set = set()
			for frm, inner in distance_map.items():
				for to in inner.keys():
					pos_to_set.add(to)
			pos_to = sorted(pos_to_set)
			# Build matrix of shape (N_from, N_to)
			N_from = len(pos_from)
			N_to = len(pos_to)
			D = np.full((N_from, N_to), np.nan, dtype=np.float32)
			pos_to_idx = {p: i for i, p in enumerate(pos_to)}
			pos_from_idx = {p: i for i, p in enumerate(pos_from)}
			for frm, inner in distance_map.items():
				i = pos_from_idx[frm]
				for to, dist in inner.items():
					j = pos_to_idx.get(to)
					if j is not None and dist is not None:
						D[i, j] = float(dist)
			# Save compressed npz
			npz_path = os.path.splitext(cache_path)[0] + '.npz'
			np.savez_compressed(npz_path, D=D, pos_from=np.array(pos_from, dtype=np.int16), pos_to=np.array(pos_to, dtype=np.int16))
			# Save max_distance as a compact numpy file
			max_dist_path = os.path.splitext(cache_path)[0] + '_max_distance.npy'
			np.save(max_dist_path, np.array(max_distance))
		except Exception as e:
			print(f"[WARNING] Failed to save .npz distance map for {cache_path}: {e}")
		# Also write a readable .txt version
		cache_parent = os.path.dirname(cache_path)
		readable_dir = os.path.join(cache_parent, "readable_distances")
		os.makedirs(readable_dir, exist_ok=True)
		base_noext = os.path.splitext(os.path.basename(cache_path))[0]
		txt_path = os.path.join(readable_dir, base_noext + '.txt')
		with open(txt_path, 'w') as f:
			for from_xy, to_dict in sorted(distance_map.items()):
				f.write(f"FROM {from_xy}\n")
				# Sort by distance (None values go to the end)
				sorted_items = sorted(to_dict.items(), key=lambda x: (x[1] is None, x[1] if x[1] is not None else float('inf')))
				for to_xy, dist in sorted_items:
					f.write(f"  TO {to_xy}: {dist}\n")
		# Write max_distance to readable txt
		txt_max_path = os.path.join(readable_dir, base_noext + '_max_distance.txt')
		with open(txt_max_path, 'w') as f:
			f.write(f"max_distance: {max_distance}\n")
	return distance_map

def load_or_compute_distance_map(game, grid, map_id, cache_dir="./distance_cache"):
	os.makedirs(cache_dir, exist_ok=True)
	cache_base = os.path.join(cache_dir, f"distance_map_{map_id}")
	npz_path = cache_base + '.npz'
	# If npz exists, return its path
	if os.path.exists(npz_path):
		return npz_path
	# Otherwise compute and write the npz (compute_distance_map will create npz)
	# compute_distance_map expects a cache_path argument; pass cache_base so it will create cache_base.npz
	compute_distance_map(game, grid, cache_base)
	if os.path.exists(npz_path):
		return npz_path
	# Fallback: raise
	raise FileNotFoundError(f"Failed to create distance cache for map {map_id}")

# --- Main script for CLI execution ---
if __name__ == "__main__":
	print("[Distance Cache] Cleaning and recreating cache folders...")
	cache_dir = os.path.join(os.path.dirname(__file__), "./distance_cache")
	if os.path.exists(cache_dir):
		shutil.rmtree(cache_dir)
		print(f"  Deleted {cache_dir}")
	os.makedirs(cache_dir, exist_ok=True)
	print("[Distance Cache] Generating distance maps for all maps in spoiled_broth/maps/maps_txt/*.txt ...")
	map_dir = os.path.join(os.path.dirname(__file__), "maps_txt")
	map_files = glob.glob(os.path.join(map_dir, "*.txt"))
	for map_file in map_files:
		# Extract map_id from filename (e.g., map_1.txt -> 1)
		base = os.path.basename(map_file)
		map_id = os.path.splitext(base)[0]
		if map_id == "text_maps_info":
			print(f"  Skipping map {map_id} (info file)")
			continue
		print(f"  Processing map {map_id}...")
		try:
			# First, create a dummy max distance file to avoid circular dependency
			cache_dir = os.path.join(os.path.dirname(__file__), "./distance_cache")
			dummy_max_dist_path = os.path.join(cache_dir, f"distance_map_{map_id}_max_distance.npy")
			if not os.path.exists(dummy_max_dist_path):
				# Create a temporary dummy file with a reasonable default value
				np.save(dummy_max_dist_path, np.array(100.0))  # Temporary placeholder
			
			grid_size = (len(open(map_file).readlines()[0].rstrip("\n")), len(open(map_file).readlines()))
			game = SpoiledBroth(map_nr=map_id, grid_size=grid_size)
			grid = game.grid
			_ = load_or_compute_distance_map(game, grid, map_id)
			print(f"    Done: distance_map_{map_id}.npz")
		except Exception as e:
			print(f"    Failed for map {map_id}: {e}")
	print("[Distance Cache] All done.")