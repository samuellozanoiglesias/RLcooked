"""
Module for pre-computing and storing accessibility maps for kitchen layouts.

This module provides functionality to create and manage accessibility maps,
which are used to determine the navigable areas within a kitchen layout.

Usage:
nohup python accessibility_maps.py > accessibility.log 2>&1 &
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, Set, Tuple
from spoiled_broth.game import SpoiledBroth
from spoiled_broth.maps.map_paths import get_maps_txt_dir

def create_accessibility_map(grid) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
    """Creates a dictionary mapping each tile to its accessible neighbors"""
    accessibility = {}
    width = grid.width
    height = grid.height

    for x in range(width):
        for y in range(height):
            pos = (x, y)
            accessibility[pos] = set()

    # First, collect all walkable and accessible (not walkable but clickable and adjacent to walkable) tiles
    clickable_accessible = set()
    for x in range(width):
        for y in range(height):
            tile = grid.tiles[x][y]
            # Use 'clickable is not None' to determine if tile is clickable
            if not getattr(tile, "is_walkable", False) and getattr(tile, "clickable", 1) is not None:
                # Check if adjacent to a walkable tile
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        neighbor = grid.tiles[nx][ny]
                        if getattr(neighbor, "is_walkable", False):
                            clickable_accessible.add((x, y))
                            break

    for start_x in range(width):
        for start_y in range(height):
            start_pos = (start_x, start_y)
            start_tile = grid.tiles[start_x][start_y]

            if not getattr(start_tile, "is_walkable", False):
                continue

            to_visit = {start_pos}
            visited = set()

            while to_visit:
                curr = to_visit.pop()
                visited.add(curr)
                x, y = curr

                accessibility[start_pos].add(curr)

                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    next_x, next_y = x + dx, y + dy
                    next_pos = (next_x, next_y)

                    if (
                        0 <= next_x < width
                        and 0 <= next_y < height
                        and next_pos not in visited
                    ):
                        next_tile = grid.tiles[next_x][next_y]
                        if getattr(next_tile, "is_walkable", False):
                            to_visit.add(next_pos)
                        # If not walkable but accessible (clickable and adjacent to walkable), add as accessible but do not continue search from there
                        elif next_pos in clickable_accessible:
                            accessibility[start_pos].add(next_pos)

    # Optionally, store clickable_accessible for visualization
    accessibility['__clickable_accessible__'] = clickable_accessible
    return accessibility


def generate_map_from_file(filename: str, game_version: str = None, maps_folder: str = None) -> Tuple[Dict[Tuple[int, int], Set[Tuple[int, int]]], any]:
    """Generate accessibility map from a map file"""
    maps_dir = get_maps_txt_dir(game_version=game_version, maps_folder=maps_folder)
    full_path = os.path.join(maps_dir, f"{filename}.txt")

    with open(full_path, "r") as f:
        lines = [line.rstrip("\n") for line in f.readlines()]
        height = len(lines)
        width = max(len(line) for line in lines) if lines else 0

    grid_size = (width, height)
    game = SpoiledBroth(filename, grid_size=grid_size, game_version=game_version, maps_folder=maps_folder)
    return create_accessibility_map(game.grid), game.grid


def serialize_map(accessibility_map: Dict[Tuple[int, int], Set[Tuple[int, int]]]) -> Dict[str, list]:
    # Only serialize (x, y) tuple keys, skip special keys like '__clickable_accessible__'
    return {
        f"{x},{y}": [[p[0], p[1]] for p in accessible]
        for k, accessible in accessibility_map.items()
        if isinstance(k, tuple) and len(k) == 2
        for x, y in [k]
    }


def deserialize_map(serialized_map: Dict[str, list]) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
    return {
        tuple(map(int, pos.split(','))): {tuple(p) for p in accessible}
        for pos, accessible in serialized_map.items()
    }


def plot_accessibility_grid_and_save(grid, accessibility, filename):
    """Plot the accessibility from each tile and save as image"""
    width = grid.width
    height = grid.height

    fig, axes = plt.subplots(height, width, figsize=(width * 1.5, height * 1.5), constrained_layout=True)

    cmap = plt.get_cmap("Blues")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    clickable_accessible = accessibility.get('__clickable_accessible__', set())
    # Remove special keys from accessibility dict for iteration
    filtered_accessibility = {k: v for k, v in accessibility.items() if isinstance(k, tuple) and len(k) == 2}
    for y in range(height):
        for x in range(width):
            ax = axes[y][x] if height > 1 else axes[x]
            ax.set_xticks([])
            ax.set_yticks([])

            tile = grid.tiles[x][y]
            base = np.zeros((height, width))


            # Mark reachable walkable tiles as 1.0, clickable_accessible as 0.8
            reachable = filtered_accessibility.get((x, y), set())
            for (rx, ry) in reachable:
                if (rx, ry) in clickable_accessible:
                    base[ry][rx] = 0.8  # clickable but not walkable
                else:
                    base[ry][rx] = 1.0  # walkable
            # Mark origin
            base[y][x] = 0.5
            # Mark non-walkable tiles that are not reachable from this origin
            for rx in range(width):
                for ry in range(height):
                    if base[ry][rx] == 0.0:
                        base[ry][rx] = 0.1

            im = ax.imshow(base, cmap=cmap, norm=norm)

    # Colorbar (create inset for colorbar using `fig.colorbar`)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.set_label("Accessibility level", rotation=270, labelpad=15)

    # Add legend for accessible but not walkable
    legend_handles = [
        mpatches.Patch(color=cmap(norm(1.0)), label="Reachable (walkable)"),
        mpatches.Patch(color=cmap(norm(0.8)), label="Reachable (clickable, not walkable)"),
        mpatches.Patch(color=cmap(norm(0.5)), label="Origin"),
        mpatches.Patch(color=cmap(norm(0.1)), label="Non-reachable")
    ]
    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.15, 1))

    fig.suptitle(f"Accessibility map from each tile - {filename}", fontsize=16)

    images_dir = os.path.join(os.path.dirname(__file__), "images_accessibility_maps")
    os.makedirs(images_dir, exist_ok=True)
    plt.savefig(os.path.join(images_dir, f"{filename}_accessibility_grid.png"))
    plt.close()


def save_accessibility_maps(game_version: str = None, maps_folder: str = None):
    """Generate and save accessibility maps for all text map files"""
    maps_txt_dir = get_maps_txt_dir(game_version=game_version, maps_folder=maps_folder)
    output_file = os.path.join(os.path.dirname(__file__), 'precomputed_accessibility.json')

    all_maps = {}

    for filename in os.listdir(maps_txt_dir):
        if filename.endswith('.txt') and filename != 'text_maps_info.txt':
            try:
                name_only = os.path.splitext(filename)[0]
                print(f"Processing {name_only}...")
                accessibility_map, grid = generate_map_from_file(name_only, game_version=game_version, maps_folder=maps_folder)
                all_maps[name_only] = serialize_map(accessibility_map)

                # NEW: Save image of accessibility map
                plot_accessibility_grid_and_save(grid, accessibility_map, name_only)

                print(f"Successfully processed {name_only}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    with open(output_file, 'w') as f:
        json.dump(all_maps, f)

    print(f"Saved accessibility maps to {output_file}")


MAP_ACCESSIBILITY = {}

def get_accessibility_map(map_name: str, game_version: str = None, maps_folder: str = None) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
    if map_name not in MAP_ACCESSIBILITY:
        maps_dir = os.path.dirname(__file__)
        json_path = os.path.join(maps_dir, 'precomputed_accessibility.json')

        if not os.path.exists(json_path):
            save_accessibility_maps()

        with open(json_path) as f:
            all_maps = json.load(f)

        if map_name not in all_maps:
            # Fallback: lazily generate and cache missing map accessibility.
            try:
                generated_map, _ = generate_map_from_file(map_name, game_version=game_version, maps_folder=maps_folder)
                all_maps[map_name] = serialize_map(generated_map)
                with open(json_path, 'w') as f:
                    json.dump(all_maps, f)
            except Exception as e:
                raise KeyError(f"No accessibility map found for {map_name}") from e

        MAP_ACCESSIBILITY[map_name] = deserialize_map(all_maps[map_name])

    return MAP_ACCESSIBILITY[map_name]


if __name__ == '__main__':
    save_accessibility_maps()
