"""
CSV Writers - Helper functions for managing CSV files during simulation.

Provides functions to initialize and manage different types of CSV loggers:
- Actions CSV (basic action log)
- Positions CSV (basic position log per agent)
- Counters CSV (counter state log)
- Human-readable actions CSV (enriched action log with collaboration tracking)
- Human-readable positions CSV (enriched position log with distances)

Author: Samuel Lozano
"""

import csv
from pathlib import Path
from typing import Dict, Any, List


def create_actions_writer(simulation_dir: Path):
    """Create and initialize actions CSV writer."""
    actions_csv_path = simulation_dir / "actions.csv"
    actions_file = open(actions_csv_path, "w", newline="")
    actions_writer = csv.writer(actions_file)
    actions_writer.writerow([
        "tick", "second",
        "agent_id", "action_idx", "action_name", "action_type",
        "agent_tile_x", "agent_tile_y",
        "tile_x", "tile_y", "cancelled_by_collision",
        "collision_detected", "collision_rerouted",
    ])
    actions_file.flush()
    return actions_file, actions_writer, actions_csv_path


def create_position_writer(simulation_dir: Path, agent_id: str):
    """Create and initialize position CSV writer for a specific agent."""
    path = simulation_dir / f"positions_{agent_id}.csv"
    fh = open(path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow([
        "frame", "second",
        "tile_x", "tile_y",
        "pixel_x", "pixel_y",
        "item_id", "item", "score",
    ])
    return fh, writer


def create_counter_writer(simulation_dir: Path, counter_positions: List):
    """Create and initialize counter CSV writer."""
    counter_csv_path = simulation_dir / "counters.csv"
    counter_file = open(counter_csv_path, "w", newline="")
    counter_writer = csv.writer(counter_file)
    
    # Header: frame, second, adjusted_second, then per-counter item_id
    header = ["frame", "second", "adjusted_second"]
    for x, y in counter_positions:
        header.append(f"counter_{x}_{y}_id")
    counter_writer.writerow(header)
    counter_file.flush()
    
    return counter_file, counter_writer, counter_csv_path


def create_human_action_writer(simulation_dir: Path, agent_id: str):
    """Create and initialize human-readable action CSV writer for a specific agent."""
    path = simulation_dir / f"{agent_id}_actions.csv"
    fh = open(path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow([
        'second', 'item', 'item_id', 'action', 'target_type', 'target_position',
        'action_long', 'player_id', 'map_name', 'game_id',
        'distance_walked', 'distance_walked_since_last_action',
        'overall_score', 'player_score_change', 'player_score',
        'walking_speed', 'cutting_speed', 'start_pos',
        'last_touched', 'touched_list',
        'tomato_id', 'plate_id', 'tomato_cut_id', 'tomato_salad_id',
        'is_item_collaboration', 'is_exchange_collaboration',
        'who_picked_tomato', 'who_picked_plate', 'who_cutted', 'who_assembled', 'who_delivered',
        'number_of_counters_used', 'proportion_of_collaboration', 'cancelled_by_collision'
    ])
    return fh, writer


def create_human_position_writer(simulation_dir: Path, agent_id: str):
    """Create and initialize human-readable position CSV writer for a specific agent."""
    path = simulation_dir / f"{agent_id}_positions.csv"
    fh = open(path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow([
        'second', 'x', 'y', 'tile_x', 'tile_y',
        'distance_walked', 'walking_speed', 'cutting_speed',
        'start_pos', 'item', 'score', 'frame'
    ])
    return fh, writer
