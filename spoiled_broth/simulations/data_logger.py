"""
Enhanced data logger with built-in item tracking for human-readable outputs.

Records CSV files during simulation (no post-processing needed):
    - actions.csv          : basic action log with collision tracking
    - positions_{id}.csv   : basic position log  
    - counters.csv         : counter state log with item_id tracking
    - items.csv            : item ledger with lineage snapshots
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
        self._actions_header = [
            "tick", "second",
            "agent_id", "action_idx", "action_name", "action_type",
            "agent_tile_x", "agent_tile_y",
            "tile_x", "tile_y", "action_performed", "action_execution_status",
            "cancelled_by_collision", "collision_detected", "collision_rerouted",
        ]
        self._actions_rows: List[List[Any]] = []
        self._last_pending_action_row_by_agent: Dict[str, int] = {}

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
        self._human_action_header = [
            'init_second', 'finish_second', 'item', 'item_id', 'action', 'target_type', 'target_position',
            'action_long', 'player_id', 'map_name', 'game_id',
            'distance_walked', 'distance_walked_since_last_action',
            'overall_score', 'player_score_change', 'player_score',
            'walking_speed', 'cutting_speed', 'start_pos',
            'last_touched', 'touched_list', 'touched_list_history',
            'tomato_id', 'plate_id', 'tomato_cut_id', 'tomato_salad_id',
            'is_item_collaboration', 'is_history_collaboration',
            'who_picked_tomato', 'who_picked_plate', 'who_cutted', 'who_assembled', 'who_delivered',
            'number_of_counters_used', 'proportion_of_collaboration', 'cancelled_by_collision'
        ]
        self._human_action_rows: Dict[str, List[List[Any]]] = {}
        self._pending_human_actions: Dict[str, Dict[str, Any]] = {}
        self._human_position_writers: Dict[str, csv.writer] = {}
        self._human_position_files: Dict[str, Any] = {}
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
        self._pending_cut_pickup_rows: Dict[str, List[Any]] = {}
        self._pending_salad_assemblies: List[Dict[str, Any]] = []
        
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
        if agent_id not in self._human_action_rows:
            self._human_action_rows[agent_id] = []

    @staticmethod
    def _should_track_row_for_late_collision(
        action_type: str,
        action_name: str,
        cancelled_by_collision: bool,
    ) -> bool:
        """Return True when a row represents an ongoing action that may fail later."""
        action_type = (action_type or '').lower()
        action_name = (action_name or '').lower()

        if cancelled_by_collision:
            return False
        if action_type in {'inaccessible_tile', 'blocked', 'collision_event'}:
            return False
        if action_name == 'do_nothing':
            return False
        return True

    def _append_human_action_row(self, agent_id: str, row_values: List[Any]) -> int:
        """Append one row to per-agent human action CSV and in-memory cache."""
        self._ensure_human_action_writer(agent_id)
        self._human_action_writers[agent_id].writerow(row_values)
        self._human_action_rows[agent_id].append(list(row_values))
        return len(self._human_action_rows[agent_id]) - 1

    @staticmethod
    def _is_agent_idle_state(state: Dict[str, Any]) -> bool:
        """Return True when the env state indicates the action is fully finished."""
        return (
            state.get('current_action') is None
            and float(state.get('interaction_timer', 0.0) or 0.0) <= 0.0
            and len(state.get('current_path') or []) == 0
        )

    def _register_pending_human_action(
        self,
        agent_id: str,
        action_data: Dict[str, Any],
        init_second: float,
        previous_agent_holding: Dict[str, str],
        previous_counter_items: Dict[Tuple[int, int], str],
    ):
        """Store action context so the human-like row can be emitted on finish."""
        self._pending_human_actions[agent_id] = {
            'action_data': dict(action_data),
            'init_second': init_second,
            'previous_agent_holding': dict(previous_agent_holding),
            'previous_counter_items': dict(previous_counter_items),
        }

    def _backfill_latest_cancelled_action(self, agent_id: str):
        """Backfill cancellation flags onto the previously logged ongoing action row."""
        action_row_index = self._last_pending_action_row_by_agent.pop(agent_id, None)
        if action_row_index is not None and 0 <= action_row_index < len(self._actions_rows):
            row = self._actions_rows[action_row_index]
            row[10] = False  # action_performed
            row[11] = 'cancelled_collision'  # action_execution_status
            row[12] = True  # cancelled_by_collision
            row[13] = True  # collision_detected
            row[14] = False  # collision_rerouted

    def _rewrite_actions_csv(self):
        """Rewrite actions.csv from in-memory rows (used for cancellation backfills)."""
        with open(self.actions_csv_path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(self._actions_header)
            writer.writerows(self._actions_rows)

    def _rewrite_human_action_csv(self, agent_id: str):
        """Rewrite one human_like_actions_{agent}.csv from in-memory rows."""
        path = self.simulation_dir / f"human_like_actions_{agent_id}.csv"
        with open(path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(self._human_action_header)
            writer.writerows(self._human_action_rows.get(agent_id, []))

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

    @staticmethod
    def _is_pickup_counter_assembly_action(action_name: str, previous_item_type: str) -> bool:
        """Return True when a counter pickup action implies salad assembly intent."""
        action_name = (action_name or '').lower()
        previous_item_type = (previous_item_type or '').lower()

        if 'pick_up_tomato_cut_from_counter' in action_name and previous_item_type == 'plate':
            return True
        if 'pick_up_plate_from_counter' in action_name and previous_item_type == 'tomato_cut':
            return True
        return False

    def _extract_counter_assembly_origins(
        self,
        agent_id: str,
        location: Tuple[int, int],
        previous_agent_holding: Dict[str, str],
        previous_counter_items: Dict[Tuple[int, int], str],
    ) -> Dict[str, str]:
        """Extract local plate/tomato_cut origin IDs for a counter assembly interaction."""
        prev_counter_id = previous_counter_items.get(location)
        prev_agent_id = previous_agent_holding.get(agent_id)
        if not prev_counter_id or not prev_agent_id:
            return {}
        if prev_counter_id not in self.item_tracker.items or prev_agent_id not in self.item_tracker.items:
            return {}

        prev_counter_type = self.item_tracker.items[prev_counter_id]['type']
        prev_agent_type = self.item_tracker.items[prev_agent_id]['type']

        if prev_counter_type == 'plate' and prev_agent_type == 'tomato_cut':
            return {'plate_id': prev_counter_id, 'tomato_cut_id': prev_agent_id}
        if prev_counter_type == 'tomato_cut' and prev_agent_type == 'plate':
            return {'plate_id': prev_agent_id, 'tomato_cut_id': prev_counter_id}
        return {}

    def _queue_pending_salad_assembly(
        self,
        agent_id: str,
        tick: int,
        location: Tuple[int, int],
        previous_agent_holding: Dict[str, str],
        previous_counter_items: Dict[Tuple[int, int], str],
    ):
        """Queue a local assembly intent so who_assembled can be stamped on the created salad."""
        origins = self._extract_counter_assembly_origins(
            agent_id,
            location,
            previous_agent_holding,
            previous_counter_items,
        )
        if not origins:
            return

        plate_id = origins.get('plate_id', '')
        tomato_cut_id = origins.get('tomato_cut_id', '')
        if not plate_id or not tomato_cut_id:
            return

        for pending in self._pending_salad_assemblies:
            if (
                pending.get('agent_id') == agent_id
                and pending.get('plate_id') == plate_id
                and pending.get('tomato_cut_id') == tomato_cut_id
            ):
                pending['tick'] = tick
                return

        self._pending_salad_assemblies.append({
            'agent_id': agent_id,
            'tick': tick,
            'plate_id': plate_id,
            'tomato_cut_id': tomato_cut_id,
        })

    def _resolve_pending_salad_assemblies(self):
        """Apply queued who_assembled to matching tomato_salad items once they exist."""
        if not self._pending_salad_assemblies:
            return

        unresolved: List[Dict[str, Any]] = []

        for pending in self._pending_salad_assemblies:
            agent_id = pending.get('agent_id', '')
            plate_id = pending.get('plate_id', '')
            tomato_cut_id = pending.get('tomato_cut_id', '')
            matched_item_id = None

            for item_id, item in self.item_tracker.items.items():
                if item.get('type') != 'tomato_salad':
                    continue
                item_origins = item.get('origins', {})
                if item_origins.get('plate_id') != plate_id:
                    continue
                if item_origins.get('tomato_cut_id') != tomato_cut_id:
                    continue
                matched_item_id = item_id
                break

            if not matched_item_id:
                unresolved.append(pending)
                continue

            salad_item = self.item_tracker.items[matched_item_id]
            if agent_id and not salad_item.get('who_assembled'):
                salad_item['who_assembled'] = agent_id
            if agent_id and not salad_item.get('created_by'):
                salad_item['created_by'] = agent_id

        self._pending_salad_assemblies = unresolved

    def log_actions(
        self,
        tick: int,
        tick_rate: int,
        logging_actions: Dict[str, Any],
        game: Any = None,
        env: Any = None,
    ):
        """
        Write one row per agent that was assigned a *new* action this tick.
        Also tracks items and writes to human-readable action CSVs.
        """
        logging_actions = logging_actions or {}
        if game is None and env is not None:
            game = getattr(env, 'game', None)

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
        
        # Track agents whose pending action was cancelled this tick.
        cancelled_pending_agents = set()

        for agent_id, data in logging_actions.items():
            # Internal coordinates in logging_actions are 0-indexed (grid space).
            # actions.csv is written as 1-indexed to match positions_{agent}.csv.
            tile_x = data.get("x", -1)
            tile_y = data.get("y", -1)
            csv_tile_x = tile_x + 1 if tile_x >= 0 else tile_x
            csv_tile_y = tile_y + 1 if tile_y >= 0 else tile_y
            agent_tile_x = data.get("agent_tile_x", -1)
            agent_tile_y = data.get("agent_tile_y", -1)
            action_type_raw = data.get("action_type", "")
            cancelled = data.get("cancelled_by_collision", False) or action_type_raw == "blocked"
            collision_detected = data.get("collision_detected", False)
            collision_rerouted = data.get("collision_rerouted", False)

            # A collision cancellation can arrive N ticks after assignment via
            # synthetic collision_event rows. Backfill the original action row.
            if action_type_raw == 'collision_event' and cancelled:
                self._backfill_latest_cancelled_action(agent_id)

            action_performed, action_execution_status = self._get_action_execution_status(
                action_type_raw,
                data.get("action_name", ""),
                cancelled,
            )
            
            # Write to basic actions.csv
            action_row = [
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
            ]
            self._actions_writer.writerow(action_row)
            self._actions_rows.append(list(action_row))

            if action_type_raw != 'collision_event':
                if self._should_track_row_for_late_collision(
                    action_type_raw,
                    data.get("action_name", ""),
                    cancelled,
                ):
                    self._last_pending_action_row_by_agent[agent_id] = len(self._actions_rows) - 1
                else:
                    self._last_pending_action_row_by_agent.pop(agent_id, None)

            # Human-like rows are emitted on finish/cancellation, not assignment.
            if second >= self._init_period and game is not None:
                if action_type_raw != 'collision_event':
                    self._register_pending_human_action(
                        agent_id,
                        data,
                        init_second=second - self._init_period,
                        previous_agent_holding=previous_agent_holding,
                        previous_counter_items=previous_counter_items,
                    )
                    if cancelled:
                        pending = self._pending_human_actions.pop(agent_id, None)
                        if pending is not None:
                            cancelled_pending_agents.add(agent_id)
                            cancelled_action_data = dict(pending.get('action_data', {}))
                            cancelled_action_data['cancelled_by_collision'] = True
                            cancelled_action_data['collision_detected'] = bool(data.get('collision_detected', False))
                            cancelled_action_data['collision_rerouted'] = bool(data.get('collision_rerouted', False))
                            self._track_and_log_human_action(
                                agent_id,
                                cancelled_action_data,
                                tick,
                                second,
                                game,
                                pending.get('previous_agent_holding', previous_agent_holding),
                                pending.get('previous_counter_items', previous_counter_items),
                                init_second=pending.get('init_second', second - self._init_period),
                            )
                elif cancelled:
                    pending = self._pending_human_actions.pop(agent_id, None)
                    if pending is not None:
                        cancelled_pending_agents.add(agent_id)
                        cancelled_action_data = dict(pending.get('action_data', {}))
                        cancelled_action_data['cancelled_by_collision'] = True
                        cancelled_action_data['collision_detected'] = True
                        cancelled_action_data['collision_rerouted'] = bool(data.get('collision_rerouted', False))
                        self._track_and_log_human_action(
                            agent_id,
                            cancelled_action_data,
                            tick,
                            second,
                            game,
                            pending.get('previous_agent_holding', previous_agent_holding),
                            pending.get('previous_counter_items', previous_counter_items),
                            init_second=pending.get('init_second', second - self._init_period),
                        )

        # Flush completed pending actions using current env state.
        if second >= self._init_period and game is not None and env is not None:
            env_agent_state = getattr(env, 'agent_state', {}) or {}
            for agent_id in list(self._pending_human_actions.keys()):
                if agent_id in cancelled_pending_agents:
                    continue
                pending = self._pending_human_actions.get(agent_id)
                if pending is None:
                    continue
                state = env_agent_state.get(agent_id, {})
                if not self._is_agent_idle_state(state):
                    continue

                self._track_and_log_human_action(
                    agent_id,
                    pending.get('action_data', {}),
                    tick,
                    second,
                    game,
                    pending.get('previous_agent_holding', previous_agent_holding),
                    pending.get('previous_counter_items', previous_counter_items),
                    init_second=pending.get('init_second', second - self._init_period),
                )
                self._pending_human_actions.pop(agent_id, None)
        
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
        init_second: Optional[float] = None,
    ):
        """Track item changes and log to human-readable action CSV."""
        action_type = action_data.get("action_type", "").lower()
        action_name = action_data.get("action_name", "")
        tile_x = action_data.get("x", -1)
        tile_y = action_data.get("y", -1)
        location = (tile_x, tile_y)
        cancelled_by_collision = action_data.get("cancelled_by_collision", False) or action_type == 'blocked'
        
        # Get current item from game object
        current_item = ""
        if hasattr(game, 'gameObjects') and agent_id in game.gameObjects:
            obj = game.gameObjects[agent_id]
            current_item = getattr(obj, 'item', '') or ''
        
        # Use tracker state synced from game (do not mutate on action assignment).
        item_id = self.item_tracker.agent_holding.get(agent_id)
        action_name_lower = action_name.lower()
        delivered_source_id: Optional[str] = None
        inferred_salad_item_id: Optional[str] = None
        inferred_salad_assembly = False
        previous_item_id = previous_agent_holding.get(agent_id)
        previous_item_type = ''
        if previous_item_id and previous_item_id in self.item_tracker.items:
            previous_item_type = self.item_tracker.items[previous_item_id].get('type', '')
        assembly_intent = self._is_pickup_counter_assembly_action(action_name, previous_item_type)

        # ------------------------------------------------------------------ #
        # GUARD: Skip tracking for actions that didn't execute successfully  #
        # - inaccessible_tile: action was NOT executed (no path exists)      #
        # - do_nothing: no state change                                      #
        # - cancelled_by_collision: action was cancelled due to collision    #
        # - invalid location: safety check for (-1,-1) coordinates           #
        # ------------------------------------------------------------------ #
        tracking_should_update = (
            action_type != 'inaccessible_tile'
            and action_type != 'blocked'
            and action_type != 'collision_event'
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

            if salad_item_id and origins:
                self.item_tracker.backfill_item_origins(
                    salad_item_id,
                    origins,
                    agent_id=agent_id,
                )
                inferred_salad_item_id = salad_item_id
                inferred_salad_assembly = assembly_intent

            if assembly_intent and not inferred_salad_item_id:
                self._queue_pending_salad_assembly(
                    agent_id,
                    tick,
                    location,
                    previous_agent_holding,
                    previous_counter_items,
                )

            if 'deliver' in action_name_lower or 'delivery' in action_type:
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

        # Resolve pending assembly intents as soon as matching salads appear
        # in the live tracker so both items.csv and tomato_salad rows carry
        # who_assembled consistently.
        self._resolve_pending_salad_assemblies()
        
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
        row_item_id = item_id or ''
        action_long = action_name

        if inferred_salad_assembly and inferred_salad_item_id:
            # Assembly is inferred from state transition (counter item + held item -> salad).
            item_data = dict(self.item_tracker.get_item_data(inferred_salad_item_id))
            if not item_data.get('who_assembled'):
                item_data['who_assembled'] = agent_id
            row_item_id = inferred_salad_item_id
            action_long = 'assemble salad'
        elif assembly_intent:
            # Intent-based fallback when salad appears after this assignment row.
            action_long = 'assemble salad'

        # Delivery actions consume the held salad, so post-action item_id is often empty.
        # Use the pre-action held salad to keep lineage/roles (including who_delivered)
        # visible in human_like_actions_{agent}.csv.
        if delivered_source_id and delivered_source_id in self.item_tracker.items:
            delivered_source = self.item_tracker.items[delivered_source_id]
            if delivered_source.get('type') == 'tomato_salad':
                item_data = dict(self.item_tracker.get_item_data(delivered_source_id))
                item_data['who_delivered'] = agent_id
                row_item_id = delivered_source_id

        if not inferred_salad_assembly and not assembly_intent:
            if 'pick_up_tomato_from_dispenser' in action_name_lower:
                action_long = 'pick up tomato from dispenser'
            elif 'pick_up_plate_from_dispenser' in action_name_lower:
                action_long = 'pick up plate from dispenser'
            elif 'use_cutting_board' in action_name_lower and previous_item_type == 'tomato':
                action_long = 'start cutting tomato'
            elif 'pick_up_tomato_salad_from_counter' in action_name_lower:
                action_long = 'pick up tomato_salad from counter'
            elif 'pick_up_tomato_cut_from_counter' in action_name_lower:
                action_long = 'pick up tomato_cut from counter'
            elif 'pick_up_tomato_from_counter' in action_name_lower:
                action_long = 'pick up tomato from counter'
            elif 'pick_up_plate_from_counter' in action_name_lower:
                action_long = 'pick up plate from counter'
            elif 'use_delivery' in action_name_lower or 'delivery' in action_type:
                action_long = 'deliver tomato_salad'
            elif 'put_down_item_on_free_counter' in action_name_lower:
                counter_drop_map = {
                    'plate': 'put down plate on counter',
                    'tomato': 'put down tomato on counter',
                    'tomato_cut': 'put down tomato_cut on counter',
                    'tomato_salad': 'put down tomato_salad on counter',
                }
                action_long = counter_drop_map.get(previous_item_type, action_long)

        current_item_type = ''
        if row_item_id and row_item_id in self.item_tracker.items:
            current_item_type = self.item_tracker.items[row_item_id].get('type', '')
        elif current_item:
            current_item_type = current_item
        
        # Determine target type
        target_type = ''
        if 'delivery' in action_type or 'deliver' in action_name_lower:
            target_type = 'delivery'
        elif 'dispenser' in action_type or 'dispenser' in action_name_lower:
            target_type = 'dispenser'
        elif 'counter' in action_type or 'counter' in action_name_lower:
            target_type = 'counter'
        elif 'cutting' in action_type or 'cutting' in action_name_lower:
            target_type = 'cuttingboard'
        
        # Write to human-readable CSV
        self._ensure_human_action_writer(agent_id)
        
        # Human-like rows use finish time and explicit init time.
        finish_second = second - self._init_period
        row_init_second = finish_second if init_second is None else init_second
        
        # Compute proportion_of_collaboration from item role breakdown
        proportion_str = self._compute_proportion_of_collaboration(item_data)

        row_values = [
            f"{row_init_second:.3f}",
            f"{finish_second:.3f}",
            current_item,
            row_item_id,
            action_name,
            target_type,
            f"({tile_x}, {tile_y})",
            action_long,
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
            item_data['touched_list_history'],
            item_data['tomato_id'],
            item_data['plate_id'],
            item_data['tomato_cut_id'],
            item_data['tomato_salad_id'],
            item_data['is_item_collaboration'],
            item_data['is_history_collaboration'],
            item_data['who_picked_tomato'],
            item_data['who_picked_plate'],
            item_data['who_cutted'],
            item_data['who_assembled'],
            item_data['who_delivered'],
            item_data['number_of_counters_used'],
            proportion_str,
            cancelled_by_collision
        ]

        # After a previous "start cutting tomato", emit a synthetic
        # pickup-from-cuttingboard row when the corresponding tomato_cut
        # is observed in-hand. Build the row from current state so item
        # metadata stays consistent.
        pending_cut_row = self._pending_cut_pickup_rows.get(agent_id)
        pending_tomato_id = ''
        if pending_cut_row is not None and len(pending_cut_row) > 22:
            pending_tomato_id = pending_cut_row[22] or ''
        current_tomato_id = item_data.get('tomato_id', '')
        lineage_matches_pending_cut = (
            not pending_tomato_id
            or not current_tomato_id
            or pending_tomato_id == current_tomato_id
        )
        should_emit_cut_pickup = (
            tracking_should_update
            and pending_cut_row is not None
            and current_item_type == 'tomato_cut'
            and lineage_matches_pending_cut
        )
        if should_emit_cut_pickup:
            cutting_pickup_row = list(row_values)
            cutting_pickup_row[4] = 'pick_up_tomato_cut_from_cuttingboard'  # action
            cutting_pickup_row[5] = 'cuttingboard'  # target_type
            if len(pending_cut_row) > 6:
                cutting_pickup_row[6] = pending_cut_row[6]  # target_position
            cutting_pickup_row[7] = 'pick up tomato_cut from cuttingboard'  # action_long
            self._append_human_action_row(agent_id, cutting_pickup_row)
            self._pending_cut_pickup_rows.pop(agent_id, None)

        # If picking from dispenser while already holding an item, prepend a
        # synthetic "put down ... on dispenser" row using the same row context.
        dispenser_drop_action_long = ''
        if tracking_should_update and (
            'pick_up_tomato_from_dispenser' in action_name_lower
            or 'pick_up_plate_from_dispenser' in action_name_lower
        ):
            dispenser_drop_map = {
                'tomato': 'put down tomato on dispenser',
                'plate': 'put down plate on dispenser',
                'tomato_cut': 'put down tomato_cut on dispenser',
                'tomato_salad': 'put down tomato_salad on dispenser',
            }
            dispenser_drop_action_long = dispenser_drop_map.get(previous_item_type, '')

        if dispenser_drop_action_long:
            dispenser_row = list(row_values)
            dispenser_row[7] = dispenser_drop_action_long  # action_long
            self._append_human_action_row(agent_id, dispenser_row)
            self._append_human_action_row(agent_id, row_values)
        else:
            self._append_human_action_row(agent_id, row_values)

        if tracking_should_update and action_long == 'start cutting tomato':
            self._pending_cut_pickup_rows[agent_id] = list(row_values)

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
            "# items.csv: item_id, item_type, touched_list, touched_list_history, last_touched, and lineage metadata",
            "# human_like_actions_{agent_id}.csv: derived per-agent action table (init_second, finish_second) enriched with item lineage",
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
        try:
            self._rewrite_actions_csv()
        except Exception as exc:
            print(f"Warning: could not rewrite actions.csv: {exc}")
        self._collisions_file.close()
        if self._counter_file is not None:
            self._counter_file.close()
        for fh in self._position_files.values():
            fh.close()
        for fh in self._human_action_files.values():
            fh.close()
        for agent_id in self._human_action_rows:
            try:
                self._rewrite_human_action_csv(agent_id)
            except Exception as exc:
                print(f"Warning: could not rewrite human_like_actions_{agent_id}.csv: {exc}")
        for fh in self._human_position_files.values():
            fh.close()
        try:
            self.item_tracker.export_items(self.items_csv_path)
        except Exception as exc:
            print(f"Warning: could not export items.csv: {exc}")

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
        for agent_id in self._human_action_writers:
            paths[f"human_like_actions_{agent_id}"] = (
                self.simulation_dir / f"human_like_actions_{agent_id}.csv"
            )
        for agent_id in self._human_position_writers:
            paths[f"human_like_positions_{agent_id}"] = (
                self.simulation_dir / f"human_like_positions_{agent_id}.csv"
            )
        return paths