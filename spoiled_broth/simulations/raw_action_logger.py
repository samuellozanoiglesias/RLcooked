"""
Raw action logging utilities for capturing integer actions directly from controllers.

Author: Samuel Lozano
"""

import time
import csv
import threading
from pathlib import Path
from typing import Any, Optional


class RawActionLogger:
    """Logs raw integer actions directly from controllers with the same format as ActionTracker."""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.lock = threading.Lock()
        
        # Action ID counters per agent (for action_id - 0-indexed per agent)
        self.action_id_counters = {}
        
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize the raw action logging CSV file."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'action_id', 'agent_id', 'action_number', 'action_type', 
                'target_tile_type', 'target_tile_x', 'target_tile_y'
            ])
    
    def _get_tile_info(self, game: Any, action: int, clickable_indices: list) -> tuple[str, Optional[int], Optional[int]]:
        """Get tile information for a given action index."""
        tile_type, x_tile, y_tile = '', None, None
        
        try:
            if action == len(clickable_indices):
                # This is a "do_nothing" action
                return '', None, None
            
            if 0 <= action < len(clickable_indices):
                tile_index = clickable_indices[action]
                grid = getattr(game, 'grid', None)
                if grid is not None:
                    x = tile_index % grid.width
                    y = tile_index // grid.width
                    if 0 <= x < grid.width and 0 <= y < grid.height:
                        tile = grid.tiles[x][y]
                        if tile:
                            tile_type = tile.__class__.__name__.lower()
                            x_tile, y_tile = x + 1, y + 1  # Convert to 1-indexed
                            return tile_type, x_tile, y_tile
        except Exception as e:
            print(f"[RAW_ACTION_LOGGER] Error getting tile info: {e}")
        
        return tile_type, x_tile, y_tile
    
    def _calculate_rl_action_coordinates(self, agent_id: str, action_index: int, game: Any, game_mode: str = "classic") -> tuple[Optional[int], Optional[int]]:
        """Calculate target tile coordinates for an RL action using observation space (same as rllib_controller)."""
        try:            
            # Import here to avoid circular imports
            from spoiled_broth.rl.observation_space import game_to_obs_vector
            
            # Generate observation to get considered tiles (same as rllib_controller)
            path_processor = getattr(game, 'path_processor', None)
            obs_vector, considered_paths, considered_tiles = game_to_obs_vector(
                game, agent_id, game_mode=game_mode, path_processor=path_processor
            )
            
            # Get the tile index from the observation (same as rllib_controller)
            tile_index = considered_tiles[action_index] if action_index < len(considered_tiles) else None
            
            if tile_index is not None and tile_index != -1:
                # Convert tile index to coordinates (1-indexed)
                grid = getattr(game, 'grid', None)
                if grid is not None:
                    x = tile_index % grid.width
                    y = tile_index // grid.width
                    coords = (x + 1, y + 1)  # Convert to 1-indexed
                    return coords
                    
        except Exception as e:
            print(f"[RAW_ACTION_LOGGER] Error calculating coordinates for RL action: {e}")
            import traceback
            traceback.print_exc()
        
        return None, None
    
    def log_action(self, agent_id: str, raw_action: int, game: Any, clickable_indices: list):
        """Log a raw integer action to CSV."""
        with self.lock:
            try:
                # Get action_id for this agent (0-indexed per agent)
                if agent_id not in self.action_id_counters:
                    self.action_id_counters[agent_id] = 0
                else:
                    self.action_id_counters[agent_id] += 1
                
                action_id = self.action_id_counters[agent_id]
                
                # Determine action type and target info
                if raw_action == len(clickable_indices):
                    # This is a "do_nothing" action
                    action_type = 'do_nothing'
                    tile_type, x_tile, y_tile = '', '', ''
                    action_number = -1
                else:
                    # This is a click action
                    action_type = 'click'
                    tile_type, x_tile, y_tile = self._get_tile_info(game, raw_action, clickable_indices)
                    action_number = raw_action
                
                # Write action immediately to CSV in same format as ActionTracker
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [
                        action_id,
                        agent_id,
                        action_number,
                        action_type,
                        tile_type or '',
                        x_tile or '',
                        y_tile or ''
                    ]
                    writer.writerow(row)
                                        
            except Exception as e:
                print(f"[RAW_ACTION_LOGGER] Error logging action for {agent_id}: {e}")

    def log_rl_action(self, agent_id: str, rl_action_index: int, action_name: str, game: Any, game_mode: str = "classic"):
        """Log a high-level RL action to CSV."""
        with self.lock:
            try:
                # Get action_id for this agent (0-indexed per agent)
                if agent_id not in self.action_id_counters:
                    self.action_id_counters[agent_id] = 0
                else:
                    self.action_id_counters[agent_id] += 1
                
                action_id = self.action_id_counters[agent_id]

                # Function to extract the tile type automatically
                def get_tile_type(action: str) -> str:
                    # List of possible tile keywords
                    tile_keywords = [
                        "dispenser", "cutting_board", "delivery", "counter"
                    ]
                    for keyword in tile_keywords:
                        if keyword in action:
                            return keyword
                    return "unknown"  # default if nothing matches
                
                # For RL actions, we log the high-level action name
                action_type = action_name
                tile_type = get_tile_type(action_name)
                
                # Calculate target coordinates using observation space (same as rllib_controller)
                x_tile, y_tile = self._calculate_rl_action_coordinates(agent_id, rl_action_index, game, game_mode)
                action_number = rl_action_index
                                
                # Write action immediately to CSV in same format as ActionTracker
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [
                        action_id,
                        agent_id,
                        action_number,
                        action_type,
                        tile_type,
                        x_tile or '',  # Convert None to empty string
                        y_tile or ''   # Convert None to empty string
                    ]
                    writer.writerow(row)
                                        
            except Exception as e:
                print(f"[RAW_ACTION_LOGGER] Error logging RL action for {agent_id}: {e}")