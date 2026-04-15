"""
Enhanced data logger with built-in item tracking for human-readable outputs.

Records CSV files during simulation (no post-processing needed):
  - actions.csv          : basic action log with collision tracking
  - positions_{id}.csv   : basic position log  
  - counters.csv         : counter state log with item_id tracking
    - human_like_actions_{agent_id}.csv  : human-readable actions with collaboration tracking
    - human_like_positions_{agent_id}.csv : human-readable positions with distances

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

        # Collision CSV (one row per frame)
        self._collisions_file, self._collisions_writer, self.collisions_csv_path = (
            csv_writers.create_collisions_writer(self.simulation_dir)
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
        self._reconstructed_action_agents: List[str] = []
        self.items_csv_path = self.simulation_dir / "items.csv"
        
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
        self._previous_tracker_tick: Optional[int] = None
        self._previous_tracker_agent_holding: Dict[str, str] = {}
        self._previous_tracker_counter_items: Dict[Tuple[int, int], str] = {}
        
        self._create_config_file()

    def _sync_tracker_with_game_state(self, tick: int, game: Any):
        """Sync item tracker from actual game state at most once per tick."""
        if game is None:
            return
        if self._last_tracker_sync_tick == tick:
            return
        if not self._counter_logger_initialized:
            self._initialize_counter_logger(game)

        # Persist pre-sync state so action logging can reason about transitions
        # even if positions/counters already triggered sync on this tick.
        self._previous_tracker_tick = tick
        self._previous_tracker_agent_holding = dict(self.item_tracker.agent_holding)
        self._previous_tracker_counter_items = dict(self.item_tracker.counter_items)

        agent_ids = [aid for aid in getattr(game, 'gameObjects', {}).keys() if aid.startswith('ai_rl_')]
        second = tick / self._tick_rate if self._tick_rate else 0.0
        self.item_tracker.sync_with_game_state(game, agent_ids, self._counter_positions, tick=tick, second=second)
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

    def _get_action_execution_status(self, action_type: str, action_name: str, cancelled_by_collision: bool) -> Tuple[bool, str]:
        """Map the logger's action metadata to an execution outcome."""
        action_type = (action_type or '').lower()
        action_name = (action_name or '').lower()

        if cancelled_by_collision:
            return False, 'cancelled_collision'
        if action_type == 'inaccessible_tile':
            return False, 'inaccessible'
        if action_type == 'blocked':
            return False, 'blocked'
        if action_type == 'collision_event':
            return False, 'collision_event'
        if action_name == 'do_nothing':
            return True, 'no_op'
        if action_type.startswith('useless_'):
            return True, 'performed_useless'
        if action_type.startswith('destructive_'):
            return True, 'performed_destructive'
        return True, 'performed'

    @staticmethod
    def _is_salad_assembly_action(action_type: str, action_name: str) -> bool:
        """Return True when an action likely assembled salad on a counter."""
        action_type = (action_type or '').lower()
        action_name = (action_name or '').lower()

        if 'salad_assembly' in action_type:
            return True

        return (
            'pick_up_plate_from_counter' in action_name
            or 'pick_up_tomato_cut_from_counter' in action_name
        )

    def _infer_salad_origins_from_transition(
        self,
        agent_id: str,
        location: Tuple[int, int],
        previous_agent_holding: Dict[str, str],
        previous_counter_items: Dict[Tuple[int, int], str],
    ) -> Tuple[Optional[str], Dict[str, str]]:
        """Infer salad lineage using only local transition evidence.

        Uses the acting agent's previous hand and the previous item on the
        interacted counter tile. Does not use global fallbacks to avoid
        assigning unrelated plate/tomato_cut IDs across different salads.
        """
        salad_item_id = self.item_tracker.counter_items.get(location)
        if not salad_item_id or salad_item_id not in self.item_tracker.items:
            return None, {}
        if self.item_tracker.items[salad_item_id].get('type') != 'tomato_salad':
            return None, {}

        prev_counter_item_id = previous_counter_items.get(location)
        prev_agent_item_id = previous_agent_holding.get(agent_id)
        if not prev_counter_item_id or not prev_agent_item_id:
            return None, {}
        if prev_counter_item_id not in self.item_tracker.items or prev_agent_item_id not in self.item_tracker.items:
            return None, {}

        prev_counter_type = self.item_tracker.items[prev_counter_item_id]['type']
        prev_agent_type = self.item_tracker.items[prev_agent_item_id]['type']

        if prev_counter_type == 'plate' and prev_agent_type == 'tomato_cut':
            return salad_item_id, {'plate_id': prev_counter_item_id, 'tomato_cut_id': prev_agent_item_id}
        if prev_counter_type == 'tomato_cut' and prev_agent_type == 'plate':
            return salad_item_id, {'plate_id': prev_agent_item_id, 'tomato_cut_id': prev_counter_item_id}
        return None, {}

    def _infer_salad_origins_from_counter_appearance(
        self,
        agent_id: str,
        location: Tuple[int, int],
        previous_agent_holding: Dict[str, str],
        previous_counter_items: Dict[Tuple[int, int], str],
    ) -> Tuple[Optional[str], Dict[str, str]]:
        """Infer salad lineage from a newly appeared salad on the interacted counter.

        This is robust when action naming/classification is noisy. We use only
        state transition at the interacted counter and the acting agent's prior hand.
        """
        current_salad_id = self.item_tracker.counter_items.get(location)
        if not current_salad_id or current_salad_id not in self.item_tracker.items:
            return None, {}
        if self.item_tracker.items[current_salad_id].get('type') != 'tomato_salad':
            return None, {}

        prev_counter_id = previous_counter_items.get(location)
        # Must be a newly formed salad at this counter interaction, not an
        # existing salad already on the counter from earlier ticks.
        if prev_counter_id == current_salad_id:
            return None, {}
        prev_agent_id = previous_agent_holding.get(agent_id)
        if not prev_counter_id or not prev_agent_id:
            return None, {}
        if prev_counter_id not in self.item_tracker.items or prev_agent_id not in self.item_tracker.items:
            return None, {}

        prev_counter_type = self.item_tracker.items[prev_counter_id]['type']
        prev_agent_type = self.item_tracker.items[prev_agent_id]['type']

        # Salad can only be formed from complementary pair at this counter interaction.
        if prev_counter_type == 'plate' and prev_agent_type == 'tomato_cut':
            return current_salad_id, {'plate_id': prev_counter_id, 'tomato_cut_id': prev_agent_id}
        if prev_counter_type == 'tomato_cut' and prev_agent_type == 'plate':
            return current_salad_id, {'plate_id': prev_agent_id, 'tomato_cut_id': prev_counter_id}
        return None, {}

    def log_actions(self, tick: int, tick_rate: int, logging_actions: Dict[str, Any], game: Any = None):
        """
        Write one row per agent that was assigned a *new* action this tick.
        Also tracks items and writes to human-readable action CSVs.
        """
        second = tick / tick_rate
        if second >= self._init_period and game is not None:
            self._sync_tracker_with_game_state(tick, game)

        if self._previous_tracker_tick == tick:
            previous_agent_holding = dict(self._previous_tracker_agent_holding)
            previous_counter_items = dict(self._previous_tracker_counter_items)
        else:
            # Defensive fallback when a sync snapshot is unavailable.
            previous_agent_holding = dict(self.item_tracker.agent_holding)
            previous_counter_items = dict(self.item_tracker.counter_items)
        
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
            action_performed, action_execution_status = self._get_action_execution_status(
                data.get("action_type", ""),
                data.get("action_name", ""),
                cancelled,
            )
            
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
                action_performed,
                action_execution_status,
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
                self._track_and_log_human_action(
                    agent_id,
                    data,
                    tick,
                    second,
                    game,
                    previous_agent_holding,
                    previous_counter_items,
                )
        
        self._actions_file.flush()

    def log_collisions(self, tick: int, tick_rate: int, env: Any):
        """Write one per-frame collision summary row."""
        second = tick / tick_rate
        collision_stats = getattr(env, "_last_collision_stats", None) or {}
        detected_agents = sorted(collision_stats.get("agents_with_detected_collisions", []))
        rerouted_agents = sorted(collision_stats.get("agents_with_rerouted_collisions", []))
        failed_agents = sorted(collision_stats.get("agents_with_failed_collisions", []))
        collisions_detected = int(collision_stats.get("collisions_detected", 0) or 0)
        collisions_rerouted = int(collision_stats.get("collisions_rerouted", 0) or 0)
        collisions_failed = int(collision_stats.get("collisions_failed", 0) or 0)

        self._collisions_writer.writerow([
            tick,
            f"{second:.3f}",
            bool(detected_agents),
            collisions_detected,
            collisions_rerouted,
            collisions_failed,
            ";".join(detected_agents),
            ";".join(rerouted_agents),
            ";".join(failed_agents),
        ])
        self._collisions_file.flush()

    def _track_and_log_human_action(
        self,
        agent_id: str,
        action_data: Dict,
        tick: int,
        second: float,
        game: Any,
        previous_agent_holding: Dict[str, str],
        previous_counter_items: Dict[Tuple[int, int], str],
    ):
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

        # Backfill lineage for actions that create transformed items in the live game state.
        if tracking_should_update:
            # Prefer state-based detection: newly appeared salad on the interacted counter.
            salad_item_id, origins = self._infer_salad_origins_from_counter_appearance(
                agent_id,
                location,
                previous_agent_holding,
                previous_counter_items,
            )

            if not salad_item_id and self._is_salad_assembly_action(action_type, action_name):
                salad_item_id, origins = self._infer_salad_origins_from_transition(
                    agent_id,
                    location,
                    previous_agent_holding,
                    previous_counter_items,
                )

            if salad_item_id:
                created_source = 'counter_appearance_assembly'
                if self._is_salad_assembly_action(action_type, action_name):
                    created_source = 'salad_assembly' if 'salad_assembly' in action_type else 'counter_pickup_assembly'
                self.item_tracker.annotate_item(
                    salad_item_id,
                    created_by=agent_id,
                    created_tick=tick,
                    created_second=second,
                    created_source=created_source,
                    origins=origins,
                    touched_by=agent_id,
                )

            elif 'use_cutting_board' in action_name_lower and 'useful_cutting' in action_type:
                cut_item_id = self.item_tracker.counter_items.get(location) or self.item_tracker.agent_holding.get(agent_id)
                if cut_item_id and cut_item_id in self.item_tracker.items:
                    tomato_origin_id = previous_agent_holding.get(agent_id)
                    if tomato_origin_id and tomato_origin_id in self.item_tracker.items:
                        if self.item_tracker.items[tomato_origin_id]['type'] == 'tomato':
                            self.item_tracker.annotate_item(
                                cut_item_id,
                                created_by=agent_id,
                                created_tick=tick,
                                created_second=second,
                                created_source='cutting_board',
                                origins={'tomato_id': tomato_origin_id},
                                touched_by=agent_id,
                            )
            elif 'deliver' in action_name_lower or 'delivery' in action_type:
                delivered_source_id = previous_agent_holding.get(agent_id)
                if delivered_source_id and delivered_source_id in self.item_tracker.items:
                    delivered_source = self.item_tracker.items[delivered_source_id]
                    if delivered_source.get('type') == 'tomato_salad':
                        self.item_tracker.record_delivered_item(
                            agent_id,
                            delivered_source_id,
                            tick=tick,
                            second=second,
                        )
        
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
            "COLLISIONS_CSV: collisions.csv",
            "POSITIONS_CSV: positions_{agent_id}.csv (one per agent)",
            "COUNTERS_CSV: counters.csv",
            "ITEMS_CSV: items.csv",
            "AGENT_ACTIONS_CSV: human_like_actions_{agent_id}.csv (one per agent)",
            "AGENT_POSITIONS_CSV: human_like_positions_{agent_id}.csv (one per agent)",
            "",
            "[PATHS]",
            f"SIMULATION_DIR: {self.simulation_dir}",
            f"CHECKPOINT_DIR: {self.simulation_config.get('CHECKPOINT_DIR', 'unknown')}",
            f"MAP_FILE: {self.simulation_config.get('MAP_FILE', 'unknown')}",
            "",
            "[COLUMN_DESCRIPTIONS]",
            "# actions.csv: tick, second, agent_id, action_idx, action_name, action_type, agent_tile_x, agent_tile_y, tile_x, tile_y, action_performed, action_execution_status, cancelled_by_collision, collision_detected, collision_rerouted",
            "#   Written on action assignment and on synthetic collision_event rows for ongoing collisions",
            "# collisions.csv: frame, second, collision_occurred, collisions_detected, collisions_rerouted, collisions_failed, detected_agents, rerouted_agents, failed_agents",
            "#   Written once per tick with a predictive collision summary",
            "# positions_{agent_id}.csv: frame, second, tile_x, tile_y, pixel_x, pixel_y, item_id, item, score",
            "#   Written every tick",
            "# counters.csv: frame, second, counter_X_Y, ...",
            "#   Written every tick; X_Y are 1-indexed tile coordinates",
            "# items.csv: item_id, item_type, touched_by, last_touched, and lineage metadata",
            "# human_like_actions_{agent_id}.csv: derived per-agent action table enriched with item lineage",
            "# human_like_positions_{agent_id}.csv: per-agent human-readable position table",
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
        self._collisions_file.close()
        if self._counter_file is not None:
            self._counter_file.close()
        for fh in self._position_files.values():
            fh.close()
        for fh in self._human_action_files.values():
            fh.close()
        for fh in self._human_position_files.values():
            fh.close()
        try:
            self.item_tracker.export_items(self.items_csv_path)
        except Exception as exc:
            print(f"Warning: could not export items.csv: {exc}")
        try:
            self._rebuild_agent_action_tables()
        except Exception as exc:
            print(f"Warning: could not rebuild human_like_actions_{{agent}}.csv: {exc}")

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes", "y"}

    def _rebuild_agent_action_tables(self):
        """Reconstruct human_like_actions_{agent}.csv using actions/positions/counters/items CSVs."""
        actions_path = self.actions_csv_path
        if not actions_path or not Path(actions_path).exists():
            return

        with open(actions_path, "r", newline="") as fh:
            action_rows = list(csv.DictReader(fh))

        if not action_rows:
            return

        # Load positions per agent and compute distance timeline.
        positions_by_agent: Dict[str, List[Dict[str, Any]]] = {}
        distance_by_agent_tick: Dict[str, Dict[int, float]] = {}
        score_by_agent_tick: Dict[str, Dict[int, int]] = {}
        start_pos_by_agent: Dict[str, str] = {}

        for pos_path in self.simulation_dir.glob("positions_ai_rl_*.csv"):
            agent_id = pos_path.stem.replace("positions_", "", 1)
            with open(pos_path, "r", newline="") as fh:
                rows = list(csv.DictReader(fh))
            rows.sort(key=lambda row: self._safe_int(row.get("frame", 0)))
            positions_by_agent[agent_id] = rows

            cumulative = 0.0
            prev_x = None
            prev_y = None
            distance_timeline: Dict[int, float] = {}
            score_timeline: Dict[int, int] = {}
            for row in rows:
                frame = self._safe_int(row.get("frame", 0))
                px = self._safe_float(row.get("pixel_x", 0.0))
                py = self._safe_float(row.get("pixel_y", 0.0))
                if prev_x is not None and prev_y is not None:
                    cumulative += math.sqrt((px - prev_x) ** 2 + (py - prev_y) ** 2)
                prev_x, prev_y = px, py
                distance_timeline[frame] = cumulative
                score_timeline[frame] = self._safe_int(row.get("score", 0))

            distance_by_agent_tick[agent_id] = distance_timeline
            score_by_agent_tick[agent_id] = score_timeline

            if rows:
                first = rows[0]
                start_pos_by_agent[agent_id] = f"({self._safe_int(first.get('tile_x', 0))}, {self._safe_int(first.get('tile_y', 0))})"
            else:
                start_pos_by_agent[agent_id] = "(0, 0)"

        # Load counter timeline.
        counters_by_tick: Dict[int, Dict[str, str]] = {}
        if self.counter_csv_path and Path(self.counter_csv_path).exists():
            with open(self.counter_csv_path, "r", newline="") as fh:
                for row in csv.DictReader(fh):
                    counters_by_tick[self._safe_int(row.get("frame", 0))] = row

        # Load item metadata and index by creation (tick, agent).
        items_by_id: Dict[str, Dict[str, str]] = {}
        created_items: Dict[Tuple[int, str], List[Dict[str, str]]] = {}
        if Path(self.items_csv_path).exists():
            with open(self.items_csv_path, "r", newline="") as fh:
                for row in csv.DictReader(fh):
                    item_id = row.get("item_id", "")
                    if item_id:
                        items_by_id[item_id] = row
                    tick = row.get("created_tick", "")
                    creator = row.get("created_by", "")
                    if tick != "" and creator:
                        key = (self._safe_int(tick, -1), creator)
                        created_items.setdefault(key, []).append(row)

        def latest_row_for_tick(rows: List[Dict[str, Any]], tick: int) -> Optional[Dict[str, Any]]:
            latest = None
            for row in rows:
                frame = self._safe_int(row.get("frame", 0))
                if frame <= tick:
                    latest = row
                else:
                    break
            return latest

        def value_at_tick(timeline: Dict[int, Any], tick: int, default: Any):
            latest_key = None
            for key in sorted(timeline.keys()):
                if key <= tick:
                    latest_key = key
                else:
                    break
            return timeline[latest_key] if latest_key is not None else default

        def value_before_tick(timeline: Dict[int, Any], tick: int, default: Any):
            latest_key = None
            for key in sorted(timeline.keys()):
                if key < tick:
                    latest_key = key
                else:
                    break
            return timeline[latest_key] if latest_key is not None else default

        def _split_agents(value: Any) -> List[str]:
            if value is None:
                return []
            if isinstance(value, list):
                return [str(v).strip() for v in value if str(v).strip()]
            text = str(value).strip()
            if not text:
                return []
            return [part.strip() for part in text.split(';') if part.strip()]

        def _is_exchange_collaboration_from_item_data(item_data: Dict[str, Any]) -> bool:
            raw = item_data.get('is_exchange_collaboration', None)
            if raw not in (None, ''):
                return self._safe_bool(raw)
            touched_agents = set(_split_agents(item_data.get('touched_list', '')))
            return len(touched_agents) > 1

        def counter_item_before_tick(counter_x: int, counter_y: int, tick: int) -> str:
            if counter_x <= 0 or counter_y <= 0:
                return ""
            counter_key = f"counter_{counter_x}_{counter_y}_id"
            prev_row = counters_by_tick.get(tick - 1)
            if prev_row:
                return prev_row.get(counter_key, "") or ""
            # Fallback if exact tick-1 row is missing.
            candidate_tick = None
            for key in sorted(counters_by_tick.keys()):
                if key < tick:
                    candidate_tick = key
                else:
                    break
            if candidate_tick is None:
                return ""
            return counters_by_tick[candidate_tick].get(counter_key, "") or ""

        # Rebuild per-agent files.
        action_rows.sort(key=lambda row: (self._safe_int(row.get("tick", 0)), row.get("agent_id", "")))
        action_agents = sorted({row.get("agent_id", "") for row in action_rows if row.get("agent_id", "").startswith("ai_rl_")})
        self._reconstructed_action_agents = action_agents

        last_action_distance = {agent: 0.0 for agent in action_agents}
        last_action_score = {agent: 0 for agent in action_agents}
        pending_cut_origin_by_agent: Dict[str, str] = {}
        pending_cut_queue_by_agent: Dict[str, List[str]] = {agent: [] for agent in action_agents}
        last_seen_tomato_by_agent: Dict[str, str] = {}
        last_seen_cut_by_agent: Dict[str, str] = {}
        last_seen_plate_by_agent: Dict[str, str] = {}
        inferred_item_overrides: Dict[str, Dict[str, str]] = {}
        reconstructed_lineage_by_item: Dict[str, Dict[str, str]] = {}

        writers: Dict[str, csv.writer] = {}
        files: Dict[str, Any] = {}
        for agent in action_agents:
            fh, writer = csv_writers.create_agent_action_writer(self.simulation_dir, agent)
            files[agent] = fh
            writers[agent] = writer

        try:
            for row in action_rows:
                agent_id = row.get("agent_id", "")
                if agent_id not in writers:
                    continue

                tick = self._safe_int(row.get("tick", 0))
                second = self._safe_float(row.get("second", 0.0))
                action_type = row.get("action_type", "")
                action_name = row.get("action_name", "")
                tile_x = self._safe_int(row.get("tile_x", -1))
                tile_y = self._safe_int(row.get("tile_y", -1))
                cancelled_by_collision = self._safe_bool(row.get("cancelled_by_collision", False))
                action_performed = self._safe_bool(row.get("action_performed", False))
                action_type_lower = action_type.lower()

                pos_row = latest_row_for_tick(positions_by_agent.get(agent_id, []), tick)
                item_id = (pos_row or {}).get("item_id", "") if pos_row else ""

                created_key = (tick, agent_id)
                created_candidates = created_items.get(created_key, [])
                if created_candidates and action_performed:
                    preferred_type = None
                    if "cutting" in action_type:
                        preferred_type = "tomato_cut"
                    elif (
                        "salad_assembly" in action_type
                        or "pick_up_plate_from_counter" in action_name.lower()
                        or "pick_up_tomato_cut_from_counter" in action_name.lower()
                    ):
                        preferred_type = "tomato_salad"
                    elif "delivery" in action_type or "deliver" in action_name.lower():
                        preferred_type = "tomato_delivered"
                    if preferred_type:
                        typed = [candidate for candidate in created_candidates if candidate.get("item_type") == preferred_type]
                        if typed:
                            item_id = typed[-1].get("item_id", item_id)
                    if not item_id and created_candidates:
                        item_id = created_candidates[-1].get("item_id", "")

                if not item_id and tile_x > 0 and tile_y > 0:
                    counter_row = counters_by_tick.get(tick, {})
                    item_id = counter_row.get(f"counter_{tile_x}_{tile_y}_id", "")

                def _infer_item_type(item_identifier: str) -> str:
                    if item_identifier.startswith("tomato_cut_"):
                        return "tomato_cut"
                    if item_identifier.startswith("tomato_salad_"):
                        return "tomato_salad"
                    if item_identifier.startswith("tomato_delivered_"):
                        return "tomato_delivered"
                    if item_identifier.startswith("plate_"):
                        return "plate"
                    if item_identifier.startswith("tomato_"):
                        return "tomato"
                    return ""

                item_data = items_by_id.get(item_id, {
                    'item_type': _infer_item_type(item_id),
                    'last_touched': '', 'touched_list': '',
                    'tomato_id': '', 'plate_id': '', 'tomato_cut_id': '', 'tomato_salad_id': '',
                    'is_item_collaboration': 'False', 'is_exchange_collaboration': 'False',
                    'who_picked_tomato': '', 'who_picked_plate': '', 'who_cutted': '',
                    'who_assembled': '', 'who_delivered': '', 'number_of_counters_used': 0,
                })

                if item_id and item_id in reconstructed_lineage_by_item:
                    merged = dict(item_data)
                    merged.update(reconstructed_lineage_by_item[item_id])
                    item_data = merged

                if item_id in inferred_item_overrides:
                    merged = dict(item_data)
                    merged.update(inferred_item_overrides[item_id])
                    item_data = merged

                item_type = item_data.get('item_type', _infer_item_type(item_id))

                # Capture chronological hints for lineage inference.
                if item_type == 'tomato' and item_id:
                    last_seen_tomato_by_agent[agent_id] = item_id
                elif item_type == 'tomato_cut' and item_id:
                    last_seen_cut_by_agent[agent_id] = item_id
                elif item_type == 'plate' and item_id:
                    last_seen_plate_by_agent[agent_id] = item_id

                action_name_lower = action_name.lower()

                # Rebuild touched_list / last_touched from executed actions so the
                # fallback can repair incomplete items.csv interaction histories.
                touchworthy_action = (
                    item_id
                    and action_performed
                    and not cancelled_by_collision
                    and action_name_lower != 'do_nothing'
                    and action_type_lower not in {'inaccessible_tile', 'blocked', 'collision_event'}
                )
                if touchworthy_action:
                    override = inferred_item_overrides.setdefault(item_id, {})
                    current_touched = override.get('touched_list', item_data.get('touched_list', ''))
                    touched_agents = _split_agents(current_touched)
                    if agent_id not in touched_agents:
                        touched_agents.append(agent_id)
                    override['touched_list'] = ';'.join(touched_agents)
                    override['last_touched'] = agent_id
                    if len(set(touched_agents)) > 1:
                        override['is_exchange_collaboration'] = True
                    merged = dict(item_data)
                    merged.update(override)
                    item_data = merged

                if action_performed and 'use_cutting_board' in action_name_lower and item_type == 'tomato' and item_id:
                    pending_cut_origin_by_agent[agent_id] = item_id
                    pending_cut_queue_by_agent.setdefault(agent_id, []).append(item_id)

                if action_performed and item_type == 'tomato' and 'pick_up_tomato' in action_name_lower:
                    override = inferred_item_overrides.setdefault(item_id, {})
                    override['who_picked_tomato'] = override.get('who_picked_tomato', '') or agent_id
                    merged = dict(item_data)
                    merged.update(override)
                    item_data = merged

                if action_performed and item_type == 'plate' and 'pick_up_plate' in action_name_lower:
                    override = inferred_item_overrides.setdefault(item_id, {})
                    override['who_picked_plate'] = override.get('who_picked_plate', '') or agent_id
                    merged = dict(item_data)
                    merged.update(override)
                    item_data = merged

                # If a tomato_cut has missing tomato origin, infer it from the executed cut history.
                if item_type == 'tomato_cut' and item_id and not item_data.get('tomato_id'):
                    cut_queue = pending_cut_queue_by_agent.get(agent_id, [])
                    inferred_tomato = cut_queue.pop(0) if cut_queue else None
                    if not inferred_tomato:
                        inferred_tomato = pending_cut_origin_by_agent.get(agent_id) or last_seen_tomato_by_agent.get(agent_id)
                    if inferred_tomato:
                        override = inferred_item_overrides.setdefault(item_id, {})
                        override['tomato_id'] = inferred_tomato
                        source = items_by_id.get(inferred_tomato, {})
                        source_lineage = reconstructed_lineage_by_item.get(inferred_tomato, {})
                        picked_by = source_lineage.get('who_picked_tomato', '') or source.get('who_picked_tomato', '')
                        if picked_by:
                            override['who_picked_tomato'] = picked_by
                        override['who_cutted'] = override.get('who_cutted', '') or agent_id
                        pending_cut_origin_by_agent.pop(agent_id, None)
                        merged = dict(item_data)
                        merged.update(override)
                        item_data = merged

                # Lightweight fallback for salads with missing links.
                if item_type == 'tomato_salad' and item_id:
                    override = inferred_item_overrides.setdefault(item_id, {})

                    # Primary reconstruction: salad appears on counter at tick T.
                    # Ingredient on counter is what was there at tick T-1 on the target tile,
                    # and the complementary ingredient should be in the assembler's hand just before T.
                    counter_prev_item_id = counter_item_before_tick(tile_x, tile_y, tick)
                    agent_prev_row = latest_row_for_tick(positions_by_agent.get(agent_id, []), max(tick - 1, 0))
                    agent_prev_item_id = (agent_prev_row or {}).get('item_id', '') if agent_prev_row else ''

                    counter_prev_type = _infer_item_type(counter_prev_item_id)
                    agent_prev_type = _infer_item_type(agent_prev_item_id)

                    if counter_prev_type == 'plate' and not (override.get('plate_id') or item_data.get('plate_id')):
                        override['plate_id'] = counter_prev_item_id
                    if counter_prev_type == 'tomato_cut' and not (override.get('tomato_cut_id') or item_data.get('tomato_cut_id')):
                        override['tomato_cut_id'] = counter_prev_item_id

                    if agent_prev_type == 'plate' and not (override.get('plate_id') or item_data.get('plate_id')):
                        override['plate_id'] = agent_prev_item_id
                    if agent_prev_type == 'tomato_cut' and not (override.get('tomato_cut_id') or item_data.get('tomato_cut_id')):
                        override['tomato_cut_id'] = agent_prev_item_id

                    # Only stamp assembler when we have concrete local ingredient evidence.
                    has_local_pair = bool(
                        (override.get('plate_id') or item_data.get('plate_id'))
                        and (override.get('tomato_cut_id') or item_data.get('tomato_cut_id'))
                    )
                    if has_local_pair and not item_data.get('who_assembled'):
                        override['who_assembled'] = agent_id

                    cut_ref = override.get('tomato_cut_id') or item_data.get('tomato_cut_id', '')
                    if cut_ref:
                        cut_lineage = reconstructed_lineage_by_item.get(cut_ref, items_by_id.get(cut_ref, {}))
                        if cut_lineage.get('tomato_id') and not (override.get('tomato_id') or item_data.get('tomato_id')):
                            override['tomato_id'] = cut_lineage.get('tomato_id', '')
                        if cut_lineage.get('who_picked_tomato') and not (override.get('who_picked_tomato') or item_data.get('who_picked_tomato')):
                            override['who_picked_tomato'] = cut_lineage.get('who_picked_tomato', '')
                        if cut_lineage.get('who_cutted') and not (override.get('who_cutted') or item_data.get('who_cutted')):
                            override['who_cutted'] = cut_lineage.get('who_cutted', '')

                    plate_ref = override.get('plate_id') or item_data.get('plate_id', '')
                    if plate_ref:
                        plate_lineage = reconstructed_lineage_by_item.get(plate_ref, items_by_id.get(plate_ref, {}))
                        if plate_lineage.get('who_picked_plate') and not (override.get('who_picked_plate') or item_data.get('who_picked_plate')):
                            override['who_picked_plate'] = plate_lineage.get('who_picked_plate', '')
                    if override:
                        merged = dict(item_data)
                        merged.update(override)
                        item_data = merged

                if item_type == 'tomato_delivered' and item_id and action_performed:
                    override = inferred_item_overrides.setdefault(item_id, {})
                    override['who_delivered'] = override.get('who_delivered', '') or agent_id
                    salad_ref = item_data.get('tomato_salad_id', '')
                    if salad_ref:
                        salad_lineage = reconstructed_lineage_by_item.get(salad_ref, items_by_id.get(salad_ref, {}))
                        for key in ['tomato_id', 'plate_id', 'tomato_cut_id', 'who_picked_tomato', 'who_picked_plate', 'who_cutted', 'who_assembled']:
                            if salad_lineage.get(key) and not (override.get(key) or item_data.get(key)):
                                override[key] = salad_lineage.get(key, '')
                    merged = dict(item_data)
                    merged.update(override)
                    item_data = merged

                if item_id:
                    reconstructed_lineage_by_item[item_id] = {
                        'item_type': item_type,
                        'last_touched': item_data.get('last_touched', ''),
                        'touched_list': item_data.get('touched_list', ''),
                        'tomato_id': item_data.get('tomato_id', ''),
                        'plate_id': item_data.get('plate_id', ''),
                        'tomato_cut_id': item_data.get('tomato_cut_id', ''),
                        'tomato_salad_id': item_data.get('tomato_salad_id', ''),
                        'who_picked_tomato': item_data.get('who_picked_tomato', ''),
                        'who_picked_plate': item_data.get('who_picked_plate', ''),
                        'who_cutted': item_data.get('who_cutted', ''),
                        'who_assembled': item_data.get('who_assembled', ''),
                        'who_delivered': item_data.get('who_delivered', ''),
                    }

                distance = float(value_at_tick(distance_by_agent_tick.get(agent_id, {}), tick, 0.0))
                distance_since_last = distance - last_action_distance.get(agent_id, 0.0)
                last_action_distance[agent_id] = distance

                player_score = int(value_at_tick(score_by_agent_tick.get(agent_id, {}), tick, 0))
                overall_score = sum(int(value_at_tick(score_by_agent_tick.get(agent, {}), tick, 0)) for agent in action_agents)
                score_change = player_score - last_action_score.get(agent_id, 0)
                last_action_score[agent_id] = player_score

                proportion_str = self._compute_proportion_of_collaboration(item_data)

                target_type = ''
                if 'counter' in action_type:
                    target_type = 'counter'
                elif 'cutting' in action_type:
                    target_type = 'cuttingboard'
                elif 'delivery' in action_type:
                    target_type = 'delivery'

                writers[agent_id].writerow([
                    f"{second - self._init_period:.3f}",
                    (pos_row or {}).get("item", "") if pos_row else "",
                    item_id,
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
                    player_score,
                    self._walking_speeds.get(agent_id, 1.0),
                    self._cutting_speeds.get(agent_id, 1.0),
                    start_pos_by_agent.get(agent_id, '(0, 0)'),
                    item_data.get('last_touched', ''),
                    item_data.get('touched_list', ''),
                    item_data.get('tomato_id', ''),
                    item_data.get('plate_id', ''),
                    item_data.get('tomato_cut_id', ''),
                    item_data.get('tomato_salad_id', ''),
                    self._safe_bool(item_data.get('is_item_collaboration', False)),
                    _is_exchange_collaboration_from_item_data(item_data),
                    item_data.get('who_picked_tomato', ''),
                    item_data.get('who_picked_plate', ''),
                    item_data.get('who_cutted', ''),
                    item_data.get('who_assembled', ''),
                    item_data.get('who_delivered', ''),
                    self._safe_int(item_data.get('number_of_counters_used', 0), 0),
                    proportion_str,
                    cancelled_by_collision,
                ])
        finally:
            for fh in files.values():
                fh.flush()
                fh.close()

    def get_output_paths(self) -> Dict[str, Path]:
        """Return a dict of output file paths."""
        paths = {
            "simulation_dir": self.simulation_dir,
            "config_file": self.config_path,
            "actions_csv": self.actions_csv_path,
            "collisions_csv": self.collisions_csv_path,
            "items_csv": self.items_csv_path,
            "counter_csv": self.counter_csv_path,
        }
        for agent_id in self._position_writers:
            paths[f"positions_{agent_id}"] = (
                self.simulation_dir / f"positions_{agent_id}.csv"
            )
        action_agents = set(self._human_action_writers.keys()) | set(self._reconstructed_action_agents)
        for agent_id in action_agents:
            paths[f"human_like_actions_{agent_id}"] = (
                self.simulation_dir / f"human_like_actions_{agent_id}.csv"
            )
        for agent_id in self._human_position_writers:
            paths[f"human_like_positions_{agent_id}"] = (
                self.simulation_dir / f"human_like_positions_{agent_id}.csv"
            )
        return paths