"""
Enhanced data logger with built-in item tracking for human-readable outputs.

Records CSV files during simulation (no post-processing needed):
  - actions.csv          : basic action log with collision tracking
  - positions_{id}.csv   : basic position log  
  - counters.csv         : counter state log with item_id tracking
  - {agent_id}_actions.csv  : human-readable actions with collaboration tracking
  - {agent_id}_positions.csv : human-readable positions with distances

GAME STATE SYNCHRONIZATION:
============================
The tracker syncs with actual game state before processing each batch of actions.
This prevents desynchronization by reading the real state from:
- game.gameObjects[agent_id].item - what each agent is holding
- game.grid.tiles[x][y].item - what's on each counter

This eliminates the "shadow tracker" problem where parallel state tracking gets
out of sync with the actual game.

COLLISION TRACKING:
==================
The logger tracks actions cancelled by collisions to prevent phantom item updates.
When collisions can't be rerouted, the action is cancelled and marked with 
`cancelled_by_collision=True`. The item tracker skips these actions to maintain 
accurate state synchronization with the game.

Guards against phantom updates:
- inaccessible_tile: action never assigned (no path exists)
- do_nothing: no state change
- cancelled_by_collision: action cancelled mid-execution
- invalid location: safety check for (-1,-1) coordinates

ITEM TRACKING & LINEAGE:
=========================
Items are tracked with unique IDs and parent-child relationships:
- tomato → tomato_cut (via cutting)
- plate + tomato_cut → tomato_salad (via assembly)

Because the tracker syncs with game state, lineage tracking is now based on
the actual transformations happening in the game, not on guesses from action names.

PICK UP ACTIONS:
================
When action is "pick_up_X", X is the item being picked up from counter/dispenser.
With state synchronization, the tracker knows exactly what exists and what's new.

Author: Samuel Lozano
"""

import csv
import os
import time
import platform
import sys
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .item_tracker import ItemTracker
from . import csv_writers


# ============================================================================
# DATA LOGGER - Main logging class
# ============================================================================

class DataLogger:
    """Handles per-tick logging with built-in item tracking for human-readable outputs."""

    # ------------------------------------------------------------------ #
    #  Construction / setup                                                 #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        base_saving_path: Path,
        checkpoint_number: str,
        timestamp: str,
        simulation_config: Dict[str, Any],
    ):
        self.base_saving_path = base_saving_path
        self.checkpoint_number = checkpoint_number
        self.timestamp = timestamp
        self.simulation_config = simulation_config

        # Create simulation-specific directory
        self.simulation_dir = base_saving_path / f"simulation_{timestamp}"
        os.makedirs(self.simulation_dir, exist_ok=True)

        # File paths
        self.config_path = self.simulation_dir / "config.txt"

        # Per-agent position writers (initialised lazily on first log call)
        self._position_writers: Dict[str, csv.writer] = {}
        self._position_files: Dict[str, Any] = {}  # open file handles

        # Actions CSV (opened once, header written immediately)
        self._actions_file, self._actions_writer, self.actions_csv_path = (
            csv_writers.create_actions_writer(self.simulation_dir)
        )

        # Counter CSV (lazy-initialised on first call once game is available)
        self._counter_logger_initialized = False
        self._counter_positions: List[Tuple[int, int]] = []
        self._counter_file: Optional[Any] = None
        self._counter_writer: Optional[csv.writer] = None
        self.counter_csv_path = None

        # Item tracker for collaboration tracking
        self.item_tracker = ItemTracker()
        
        # Human-readable CSV writers
        self._human_action_writers: Dict[str, csv.writer] = {}
        self._human_action_files: Dict[str, Any] = {}
        self._human_position_writers: Dict[str, csv.writer] = {}
        self._human_position_files: Dict[str, Any] = {}
        
        # Per-agent tracking for human-readable outputs
        self._agent_data: Dict[str, Dict] = {}  # agent_id -> {prev_distance, prev_score, prev_pos, start_pos}
        
        # Get agent speeds from config
        self._walking_speeds = simulation_config.get('WALKING_SPEEDS', {})
        self._cutting_speeds = simulation_config.get('CUTTING_SPEEDS', {})
        self._map_name = simulation_config.get('MAP_NR', 'unknown')
        self._game_id = f"simulation_{timestamp}"
        self._init_period = simulation_config.get('AGENT_INITIALIZATION_PERIOD', 0.0)
        self._tick_rate = simulation_config.get('TICK_RATE', 2.0)
        self._last_tracker_sync_tick: Optional[int] = None
        
        self._create_config_file()

    def _sync_tracker_with_game_state(self, tick: int, game: Any):
        """Sync item tracker from actual game state at most once per tick."""
        if game is None:
            return
        if self._last_tracker_sync_tick == tick:
            return
        if not self._counter_logger_initialized:
            self._initialize_counter_logger(game)

        agent_ids = [aid for aid in getattr(game, 'gameObjects', {}).keys() if aid.startswith('ai_rl_')]
        self.item_tracker.sync_with_game_state(game, agent_ids, self._counter_positions)
        self._last_tracker_sync_tick = tick

    # ------------------------------------------------------------------ #
    #  Actions                                                              #
    # ------------------------------------------------------------------ #

    def _ensure_human_action_writer(self, agent_id: str):
        """Ensure human-readable action CSV writer exists for agent."""
        if agent_id not in self._human_action_writers:
            fh, writer = csv_writers.create_human_action_writer(self.simulation_dir, agent_id)
            self._human_action_files[agent_id] = fh
            self._human_action_writers[agent_id] = writer

    def log_actions(self, tick: int, tick_rate: int, logging_actions: Dict[str, Any], game: Any = None):
        """
        Write one row per agent that was assigned a *new* action this tick.
        Also tracks items and writes to human-readable action CSVs.
        """
        second = tick / tick_rate
        if second >= self._init_period and game is not None:
            self._sync_tracker_with_game_state(tick, game)
        
        for agent_id, data in logging_actions.items():
            # Internal coordinates in logging_actions are 0-indexed (grid space).
            # actions.csv is written as 1-indexed to match positions_{agent}.csv.
            tile_x = data.get("x", -1)
            tile_y = data.get("y", -1)
            csv_tile_x = tile_x + 1 if tile_x >= 0 else tile_x
            csv_tile_y = tile_y + 1 if tile_y >= 0 else tile_y
            agent_tile_x = data.get("agent_tile_x", -1)
            agent_tile_y = data.get("agent_tile_y", -1)
            cancelled = data.get("cancelled_by_collision", False)
            collision_detected = data.get("collision_detected", False)
            collision_rerouted = data.get("collision_rerouted", False)
            
            # Write to basic actions.csv
            self._actions_writer.writerow([
                tick,
                f"{second:.3f}",
                agent_id,
                data.get("action_idx", -1),
                data.get("action_name", ""),
                data.get("action_type", ""),
                agent_tile_x,
                agent_tile_y,
                csv_tile_x,
                csv_tile_y,
                cancelled,
                collision_detected,
                collision_rerouted,
            ])
            
            # Track items and write to human-readable CSV (only during gameplay)
            # Skip item tracking for cancelled actions
            if (
                second >= self._init_period
                and game is not None
                and data.get("action_type", "") != "collision_event"
            ):
                self._track_and_log_human_action(agent_id, data, tick, second, game)
        
        self._actions_file.flush()

    def _track_and_log_human_action(self, agent_id: str, action_data: Dict, tick: int, second: float, game: Any):
        """Track item changes and log to human-readable action CSV."""
        action_type = action_data.get("action_type", "").lower()
        action_name = action_data.get("action_name", "")
        tile_x = action_data.get("x", -1)
        tile_y = action_data.get("y", -1)
        location = (tile_x, tile_y)
        cancelled_by_collision = action_data.get("cancelled_by_collision", False)
        
        # Get current item from game object
        current_item = ""
        if hasattr(game, 'gameObjects') and agent_id in game.gameObjects:
            obj = game.gameObjects[agent_id]
            current_item = getattr(obj, 'item', '') or ''
        
        # Use tracker state synced from game (do not mutate on action assignment).
        item_id = self.item_tracker.agent_holding.get(agent_id)
        action_name_lower = action_name.lower()

        # ------------------------------------------------------------------ #
        # GUARD: Skip tracking for actions that didn't execute successfully  #
        # - inaccessible_tile: action was NOT executed (no path exists)      #
        # - do_nothing: no state change                                      #
        # - cancelled_by_collision: action was cancelled due to collision    #
        # - invalid location: safety check for (-1,-1) coordinates           #
        # ------------------------------------------------------------------ #
        tracking_should_update = (
            action_type != 'inaccessible_tile'
            and action_name_lower != 'do_nothing'
            and not cancelled_by_collision  # NEW: Don't track cancelled actions
            and location != (-1, -1)  # extra safety: inaccessible always has -1,-1
        )

        # IMPORTANT: actions.csv logs intent assignment, not interaction completion.
        # Mutating tracker here causes phantom item/counter transitions.
        if False and tracking_should_update:
            # ============================================================ #
            # SALAD ASSEMBLY: agent's item + counter item → tomato_salad
            # ============================================================ #
            if action_type == 'salad_assembly' or 'salad_assembly' in action_type:
                # Assembly was already tracked before sync in log_actions()
                # to preserve original item IDs. Just get the item_id.
                item_id = self.item_tracker.agent_holding.get(agent_id)
                # If agent isn't holding anything, check counter
                if not item_id:
                    item_id = self.item_tracker.counter_items.get(location)

            # ============================================================ #
            # PICK UP: agent picks up an item from dispenser or counter
            # ============================================================ #
            elif 'pick_up' in action_name_lower or 'pick up' in action_name_lower:
                # Determine item type from action name
                # Order matters: most-specific substrings first
                item_type = None
                if 'tomato_salad' in action_name_lower or 'salad' in action_name_lower:
                    item_type = 'tomato_salad'
                elif 'tomato_cut' in action_name_lower or 'cut' in action_name_lower:
                    item_type = 'tomato_cut'
                elif 'tomato' in action_name_lower:
                    item_type = 'tomato'
                elif 'plate' in action_name_lower or 'dish' in action_name_lower:
                    item_type = 'plate'
                
                if item_type:
                    item_id = self.item_tracker.track_pickup(agent_id, item_type, location)

            # ============================================================ #
            # CUTTING BOARD: agent transforms tomato → tomato_cut
            # ============================================================ #
            elif 'use_cutting_board' in action_name_lower:
                # Only track cutting if action was useful (agent had something to cut)
                if 'useful_cutting' in action_type:
                    # Transform tomato → tomato_cut
                    item_id = self.item_tracker.track_pickup(agent_id, 'tomato_cut', location,
                                                             from_cutting=True)
                    # The tomato_cut remains on the cutting board after this action
                    self.item_tracker.track_drop(agent_id, location)
                # else: useless_cutting_board - agent had nothing to cut, no tracking needed

            # ============================================================ #
            # PUT DOWN / DROP: agent places held item on counter
            # ============================================================ #
            elif 'put_down' in action_name_lower or 'drop' in action_name_lower or 'place' in action_name_lower:
                item_id = self.item_tracker.track_drop(agent_id, location)

            # ============================================================ #
            # DELIVERY: agent delivers held item (usually tomato_salad)
            # ============================================================ #
            elif 'deliver' in action_name_lower or 'delivery' in action_type:
                item_id = self.item_tracker.track_delivery(agent_id)
        
        # Get position/distance/score data
        if agent_id not in self._agent_data:
            self._agent_data[agent_id] = {
                'prev_distance': 0.0,
                'last_action_distance': 0.0,
                'prev_score': 0,
                'start_pos': '(0, 0)'
            }

        tracking = self._agent_data[agent_id]

        # Current cumulative distance (kept up-to-date by _log_human_position each tick)
        distance = tracking.get('prev_distance', 0.0)
        distance_since_last = distance - tracking.get('last_action_distance', 0.0)
        tracking['last_action_distance'] = distance

        # Per-agent score and global (overall) score
        score = tracking.get('prev_score', 0)
        if hasattr(game, 'gameObjects') and agent_id in game.gameObjects:
            obj = game.gameObjects[agent_id]
            score = getattr(obj, 'score', 0)
        overall_score = sum(
            getattr(obj, 'score', 0)
            for aid, obj in (game.gameObjects.items() if hasattr(game, 'gameObjects') else [])
            if aid.startswith('ai_rl_')
        )

        # Calculate per-agent score change
        score_change = score - tracking.get('prev_score', 0)
        tracking['prev_score'] = score
        
        # Get item data
        item_data = self.item_tracker.get_item_data(item_id)
        
        # Determine target type
        target_type = ''
        if 'counter' in action_type:
            target_type = 'counter'
        elif 'cutting' in action_type:
            target_type = 'cuttingboard'
        elif 'delivery' in action_type:
            target_type = 'delivery'
        
        # Write to human-readable CSV
        self._ensure_human_action_writer(agent_id)
        writer = self._human_action_writers[agent_id]
        
        # Adjust second to account for init period
        adjusted_second = second - self._init_period
        
        # Compute proportion_of_collaboration from item role breakdown
        proportion_str = self._compute_proportion_of_collaboration(item_data)

        writer.writerow([
            f"{adjusted_second:.3f}",
            current_item,
            item_id or '',
            action_name,
            target_type,
            f"({tile_x}, {tile_y})",
            action_name,
            agent_id,
            self._map_name,
            self._game_id,
            distance,
            round(distance_since_last, 4),
            overall_score,
            score_change,
            score,
            self._walking_speeds.get(agent_id, 1.0),
            self._cutting_speeds.get(agent_id, 1.0),
            tracking['start_pos'],
            item_data['last_touched'],
            item_data['touched_list'],
            item_data['tomato_id'],
            item_data['plate_id'],
            item_data['tomato_cut_id'],
            item_data['tomato_salad_id'],
            item_data['is_item_collaboration'],
            item_data['is_exchange_collaboration'],
            item_data['who_picked_tomato'],
            item_data['who_picked_plate'],
            item_data['who_cutted'],
            item_data['who_assembled'],
            item_data['who_delivered'],
            item_data['number_of_counters_used'],
            proportion_str,
            cancelled_by_collision
        ])
        self._human_action_files[agent_id].flush()

    def _compute_proportion_of_collaboration(self, item_data: Dict) -> str:
        """Return a per-agent proportion of how many roles each agent performed.

        Roles considered: who_picked_tomato, who_picked_plate, who_cutted,
        who_assembled, who_delivered.  Each filled role counts as one 'step'.
        The result is a list of floats in the same order as sorted agent IDs.
        """
        roles = [
            item_data.get('who_picked_tomato', ''),
            item_data.get('who_picked_plate', ''),
            item_data.get('who_cutted', ''),
            item_data.get('who_assembled', ''),
            item_data.get('who_delivered', ''),
        ]
        filled = [r for r in roles if r]
        agents = sorted(self._walking_speeds.keys())
        if not filled or not agents:
            return str([0.0] * len(agents))
        total = len(filled)
        proportions = [round(filled.count(a) / total, 4) for a in agents]
        return str(proportions)

    # ------------------------------------------------------------------ #
    #  Positions                                                            #
    # ------------------------------------------------------------------ #

    def _ensure_position_writer(self, agent_id: str):
        if agent_id not in self._position_writers:
            fh, writer = csv_writers.create_position_writer(self.simulation_dir, agent_id)
            self._position_files[agent_id] = fh
            self._position_writers[agent_id] = writer

    def _ensure_human_position_writer(self, agent_id: str):
        """Ensure human-readable position CSV writer exists for agent."""
        if agent_id not in self._human_position_writers:
            fh, writer = csv_writers.create_human_position_writer(self.simulation_dir, agent_id)
            self._human_position_files[agent_id] = fh
            self._human_position_writers[agent_id] = writer

    def log_positions(self, tick: int, tick_rate: int, game: Any):
        """Write one position row per RL agent every tick."""
        second = tick / tick_rate
        self._sync_tracker_with_game_state(tick, game)
        
        for agent_id, obj in game.gameObjects.items():
            if not agent_id.startswith("ai_rl_"):
                continue
            
            # Write to basic positions CSV
            self._ensure_position_writer(agent_id)
            writer = self._position_writers[agent_id]
            
            tile_x = getattr(obj, "slot_x", -1)
            tile_y = getattr(obj, "slot_y", -1)
            pixel_x = getattr(obj, "x", -1)
            pixel_y = getattr(obj, "y", -1)
            item = getattr(obj, "item", None) or ""
            item_id = self.item_tracker.agent_holding.get(agent_id, "") or ""
            score = getattr(obj, "score", 0)
            
            writer.writerow([
                tick,
                f"{second:.3f}",
                tile_x,
                tile_y,
                pixel_x,
                pixel_y,
                item_id,
                item,
                score,
            ])
            self._position_files[agent_id].flush()
            
            # Write to human-readable positions CSV (only during gameplay)
            if second >= self._init_period:
                self._log_human_position(agent_id, tick, second, tile_x, tile_y, pixel_x, pixel_y, item, score)

    def _log_human_position(self, agent_id: str, tick: int, second: float, 
                            tile_x: int, tile_y: int, pixel_x: float, pixel_y: float, 
                            item: str, score: int):
        """Log to human-readable position CSV with distance tracking."""
        # Initialize agent tracking if needed
        if agent_id not in self._agent_data:
            self._agent_data[agent_id] = {
                'prev_distance': 0.0,
                'prev_score': score,
                'prev_x': pixel_x,
                'prev_y': pixel_y,
                'start_pos': f"({tile_x}, {tile_y})"
            }
        
        tracking = self._agent_data[agent_id]
        
        # Calculate distance
        dx = pixel_x - tracking.get('prev_x', pixel_x)
        dy = pixel_y - tracking.get('prev_y', pixel_y)
        distance_step = math.sqrt(dx**2 + dy**2)
        current_distance = tracking['prev_distance'] + distance_step
        
        # Update tracking
        tracking['prev_distance'] = current_distance
        tracking['prev_x'] = pixel_x
        tracking['prev_y'] = pixel_y
        
        # Write to human-readable CSV
        self._ensure_human_position_writer(agent_id)
        writer = self._human_position_writers[agent_id]
        
        # Adjust second to account for init period
        adjusted_second = second - self._init_period
        
        writer.writerow([
            f"{adjusted_second:.3f}",
            pixel_x,
            pixel_y,
            tile_x,
            tile_y,
            current_distance,
            self._walking_speeds.get(agent_id, 1.0),
            self._cutting_speeds.get(agent_id, 1.0),
            tracking['start_pos'],
            item,
            score,
            tick
        ])
        self._human_position_files[agent_id].flush()

    # ------------------------------------------------------------------ #
    #  Counters                                                             #
    # ------------------------------------------------------------------ #

    def _initialize_counter_logger(self, game: Any):
        from spoiled_broth.world.tiles import Counter
        grid = getattr(game, "grid", None)
        if grid is not None:
            for x in range(grid.width):
                for y in range(grid.height):
                    tile = grid.tiles[x][y]
                    if isinstance(tile, Counter):
                        self._counter_positions.append((x + 1, y + 1))
        self._counter_positions.sort()

        self._counter_file, self._counter_writer, self.counter_csv_path = (
            csv_writers.create_counter_writer(self.simulation_dir, self._counter_positions)
        )
        self._counter_logger_initialized = True
        print(
            f"Counter logger initialised with {len(self._counter_positions)} counters: "
            f"{self._counter_positions}"
        )

    def log_counters(self, tick: int, tick_rate: int, game: Any):
        """Write one counter-state row per tick with item_id tracking."""
        if not self._counter_logger_initialized:
            self._initialize_counter_logger(game)
        self._sync_tracker_with_game_state(tick, game)

        second = tick / tick_rate
        # adjusted_second matches the timeline used by the human-readable CSVs
        adjusted_second = second - self._init_period
        row = [tick, f"{second:.3f}", f"{adjusted_second:.3f}"]
        
        for cx, cy in self._counter_positions:
            # Item ID comes from the shadow item tracker
            # The tracker uses 0-indexed action coordinates, so convert (1-indexed → 0-indexed)
            tracker_key = (cx - 1, cy - 1)
            item_id = self.item_tracker.counter_items.get(tracker_key, "")
            row.append(item_id or "")
        
        self._counter_writer.writerow(row)
        self._counter_file.flush()

    # ------------------------------------------------------------------ #
    #  Config file                                                          #
    # ------------------------------------------------------------------ #

    def _create_config_file(self):
        duration = self.simulation_config.get("DURATION", 0)
        init_period = self.simulation_config.get("AGENT_INITIALIZATION_PERIOD", 0.0)
        tick_rate = self.simulation_config.get("TICK_RATE", 24)
        total_time = duration + init_period
        total_frames = int(total_time * tick_rate)
        gameplay_frames = int(duration * tick_rate)
        init_frames = int(init_period * tick_rate)

        lines = [
            "# Simulation Configuration File",
            f"# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Platform: {platform.system()} {platform.release()}",
            f"# Python: {sys.version.split()[0]}",
            f"# Working Directory: {os.getcwd()}",
            f"# Command Line: {' '.join(sys.argv) if hasattr(sys, 'argv') else 'N/A'}",
            "",
            "[SIMULATION_INFO]",
            f"SIMULATION_ID: {self.timestamp}",
            f"CHECKPOINT_NUMBER: {self.checkpoint_number}",
            f"MAP_NR: {self.simulation_config.get('MAP_NR', 'unknown')}",
            f"NUM_AGENTS: {self.simulation_config.get('NUM_AGENTS', 'unknown')}",
            f"GAME_VERSION: {self.simulation_config.get('GAME_VERSION', 'unknown')}",
            f"TRAINING_ID: {self.simulation_config.get('TRAINING_ID', 'unknown')}",
            f"GAME_TYPE: {self.simulation_config.get('GAME_TYPE', 'unknown')}",
            f"COLLISION_ENABLED: {self.simulation_config.get('COLLISION_ENABLED', False)}",
            "",
            "[TIMING_CONFIGURATION]",
            f"AGENT_INITIALIZATION_PERIOD: {init_period}",
            f"DURATION_SECONDS: {duration}",
            f"TOTAL_SIMULATION_TIME: {total_time}",
            f"ENGINE_TICK_RATE: {tick_rate}",
            f"INITIALIZATION_FRAMES: {init_frames}",
            f"GAMEPLAY_FRAMES: {gameplay_frames}",
            f"TOTAL_FRAMES: {total_frames}",
            "",
            "[AGENT_SPEEDS]",
            f"WALKING_SPEEDS: {self.simulation_config.get('WALKING_SPEEDS', {})}",
            f"CUTTING_SPEEDS: {self.simulation_config.get('CUTTING_SPEEDS', {})}",
            "",
            "[VIDEO_SETTINGS]",
            f"ENABLE_VIDEO: {self.simulation_config.get('ENABLE_VIDEO', False)}",
            f"VIDEO_FPS: {self.simulation_config.get('VIDEO_FPS', 24)}",
            "",
            "[OUTPUT_FILES]",
            "ACTIONS_CSV: actions.csv",
            "POSITIONS_CSV: positions_{agent_id}.csv (one per agent)",
            "COUNTERS_CSV: counters.csv",
            "",
            "[PATHS]",
            f"SIMULATION_DIR: {self.simulation_dir}",
            f"CHECKPOINT_DIR: {self.simulation_config.get('CHECKPOINT_DIR', 'unknown')}",
            f"MAP_FILE: {self.simulation_config.get('MAP_FILE', 'unknown')}",
            "",
            "[COLUMN_DESCRIPTIONS]",
            "# actions.csv: tick, second, agent_id, action_idx, action_name, action_type, agent_tile_x, agent_tile_y, tile_x, tile_y, cancelled_by_collision, collision_detected, collision_rerouted",
            "#   Written on action assignment and on synthetic collision_event rows for ongoing collisions",
            "# positions_{agent_id}.csv: frame, second, tile_x, tile_y, pixel_x, pixel_y, item_id, item, score",
            "#   Written every tick",
            "# counters.csv: frame, second, counter_X_Y, ...",
            "#   Written every tick; X_Y are 1-indexed tile coordinates",
        ]
        try:
            with open(self.config_path, "w") as f:
                f.write("\n".join(lines))
            print(f"Configuration saved to: {self.config_path}")
        except Exception as e:
            print(f"Warning: could not create config file: {e}")

    # ------------------------------------------------------------------ #
    #  Cleanup                                                              #
    # ------------------------------------------------------------------ #

    def close(self):
        """Flush and close all open file handles."""
        self._actions_file.close()
        if self._counter_file is not None:
            self._counter_file.close()
        for fh in self._position_files.values():
            fh.close()
        for fh in self._human_action_files.values():
            fh.close()
        for fh in self._human_position_files.values():
            fh.close()

    def get_output_paths(self) -> Dict[str, Path]:
        """Return a dict of output file paths."""
        paths = {
            "simulation_dir": self.simulation_dir,
            "config_file": self.config_path,
            "actions_csv": self.actions_csv_path,
            "counter_csv": self.counter_csv_path,
        }
        for agent_id in self._position_writers:
            paths[f"positions_{agent_id}"] = (
                self.simulation_dir / f"positions_{agent_id}.csv"
            )
        for agent_id in self._human_action_writers:
            paths[f"{agent_id}_actions_human"] = (
                self.simulation_dir / f"{agent_id}_actions.csv"
            )
        for agent_id in self._human_position_writers:
            paths[f"{agent_id}_positions_human"] = (
                self.simulation_dir / f"{agent_id}_positions.csv"
            )
        return paths