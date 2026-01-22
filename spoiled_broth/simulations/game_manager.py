"""
Game management utilities for simulation runs.

Author: Samuel Lozano
"""

from typing import Tuple, Callable, Any
from spoiled_broth.game import SpoiledBroth as Game
from spoiled_broth.rl.path_processing import PathProcessor
from .simulation_config import SimulationConfig


class GameManager:
    """Manages game instance creation and configuration."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config

    def create_game_factory(self, map_nr: str, grid_size: Tuple[int, int], 
                           walking_speeds: dict = None, cutting_speeds: dict = None,
                           game_type: str = '') -> Callable:
        """
        Create a game factory function for the simulation.
        
        Args:
            map_nr: Name of the map to load
            grid_size: Grid size tuple (width, height)
            walking_speeds: Dictionary of walking speeds for each agent
            cutting_speeds: Dictionary of cutting speeds for each agent
            game_type: Game type (e.g., 'classic', 'classic_collision')
            
        Returns:
            Game factory function
        """
        def game_factory():
            
            game = Game(
                map_nr=map_nr,
                grid_size=grid_size,
                walking_speeds=walking_speeds,
                cutting_speeds=cutting_speeds
            )
            
            # Check if collision detection should be enabled
            collision_enabled = 'collision' in game_type.lower()
            
            # Create and attach PathProcessor if collisions are enabled
            if collision_enabled:
                game.path_processor = PathProcessor(map_nr, collision_enabled=True)
                print(f"PathProcessor created with collision detection ENABLED for game_type: {game_type}")
            else:
                game.path_processor = PathProcessor(map_nr, collision_enabled=False)
                print(f"PathProcessor created with collision detection DISABLED for game_type: {game_type}")
            
            # Reset game state
            self._reset_game_state(game)
            return game
        
        return game_factory
    
    def _reset_game_state(self, game: Any):
        """Reset game state to clean initial conditions."""
        try:
            grid = getattr(game, 'grid', None)
            if grid is not None:
                from spoiled_broth.world.tiles import Dispenser
                
                # Clear items on non-dispenser tiles
                for x in range(grid.width):
                    for y in range(grid.height):
                        tile = grid.tiles[x][y]
                        if tile is None:
                            continue
                        
                        if hasattr(tile, 'item') and not isinstance(tile, Dispenser):
                            tile.item = None
                        # Reset cutting board state by resetting cut_time_accumulated
                        # (cut_stage is a read-only property calculated from cut_time_accumulated)
                        if hasattr(tile, 'cut_time_accumulated'):
                            tile.cut_time_accumulated = 0
                        if hasattr(tile, 'cut_by'):
                            tile.cut_by = None
                        if hasattr(tile, 'cut_item'):
                            tile.cut_item = None
            
            # Reset agent states
            for agent_id, obj in list(game.gameObjects.items()):
                if agent_id.startswith('ai_rl_'):
                    if hasattr(obj, 'item'):
                        obj.item = None
                    if hasattr(obj, 'action_complete'):
                        obj.action_complete = True
                    if hasattr(obj, 'current_action'):
                        obj.current_action = None
                    if hasattr(obj, 'is_busy'):
                        obj.is_busy = False
                    if hasattr(obj, 'speed'):
                        obj.speed = self.config.agent_speed_px_per_sec
                        
        except Exception as e:
            print(f"Warning: Error resetting game state: {e}")