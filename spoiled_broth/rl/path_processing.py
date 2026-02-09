"""
Simplified path processing system for tick-based multi-agent RL training.

This module provides spatial pathfinding utilities with reactive collision handling:
1. Calculate paths considering only CURRENT agent positions as obstacles
2. No temporal/future collision prediction
3. Collision detection happens reactively during tick execution
"""

from typing import Dict, List, Tuple, Optional, NamedTuple
from engine.extensions.topDownGridWorld.a_star import Node, euclidean_distance, find_path


class AgentPathInfo(NamedTuple):
    """Information about an agent's active path and state (for tracking only)."""
    path: List[Node]
    current_position: Tuple[float, float]  # (x, y) in grid coordinates
    path_index: int  # Current position in path
    speed: float  # Agent's walking speed in tiles/second


class PathProcessor:
    """Spatial pathfinding system for tick-based RL training with reactive collision handling."""
    
    def __init__(self, map_nr: int, collision_enabled: bool = True):
        """Initialize path processor.
        
        Args:
            map_nr: Map number (unused, kept for compatibility)
            collision_enabled: Whether to treat other agents as obstacles
        """
        self.collision_enabled = collision_enabled
        self.active_paths: Dict[str, AgentPathInfo] = {}  # Track agent paths (for reference only)
        
    def is_enabled(self) -> bool:
        """Check if collision detection is enabled."""
        return self.collision_enabled
    
    def get_shortest_path_distance(self, grid, from_xy: Tuple[int, int], to_xy: Tuple[int, int], 
                                 agent_id: str = None, current_time: float = 0.0, 
                                 agent_speed: float = 1.875, game=None) -> Tuple[Optional[float], Optional[List[Node]]]:
        """Calculate shortest path distance considering only current agent positions.
        
        NO TEMPORAL COLLISION PREDICTION - only current positions treated as obstacles.
        Collisions are detected reactively during tick execution in GameEnv.
        
        Args:
            grid: Game grid for pathfinding
            from_xy: Start position (x, y)
            to_xy: Target position (x, y) - can be non-walkable; will find path to nearest walkable neighbor
            agent_id: ID of the requesting agent (to exclude from obstacle detection)
            current_time: Unused (kept for API compatibility)
            agent_speed: Unused (kept for API compatibility)
            game: Game instance to get current agent positions as obstacles
            
        Returns:
            tuple: (distance, path) where distance is path length or None if no path exists,
                  and path is list of Node objects or None
        """
        start_node = Node(from_xy[0], from_xy[1])
        
        # Get current agent positions as static obstacles (if collision detection enabled)
        static_obstacles = set()
        if self.collision_enabled and game is not None:
            for other_agent_id, other_agent in game.gameObjects.items():
                if (other_agent_id != agent_id and 
                    other_agent_id.startswith('ai_rl_') and
                    hasattr(other_agent, 'slot_x') and hasattr(other_agent, 'slot_y')):
                    static_obstacles.add((other_agent.slot_x, other_agent.slot_y))
        
        # Check if target is walkable; if not, find walkable neighbors
        target_x, target_y = to_xy
        target_is_walkable = grid.tiles[target_x][target_y].is_walkable
                
        if not target_is_walkable:
            # Target is not walkable (e.g., dispenser, cutting board)
            # Find walkable neighbors and pathfind to the closest one
            neighbors = self._get_walkable_neighbors_of_target(grid, to_xy)
            
            if not neighbors:
                return None, None
            
            # Check if agent is already at a walkable neighbor of the target
            if from_xy in neighbors:
                # Agent is already adjacent to the target - no movement needed
                return 0.0, [start_node]
            
            # Find shortest path to any of the walkable neighbors
            best_distance = None
            best_path = None
            
            for neighbor_xy in neighbors:
                neighbor_node = Node(neighbor_xy[0], neighbor_xy[1])
                path = find_path(grid, start_node, neighbor_node)
                
                # Check if path exists and doesn't go through obstacles
                if path and len(path) > 1:
                    # Validate path doesn't go through static obstacles
                    path_blocked = any((node.x, node.y) in static_obstacles for node in path[1:])  # Skip start node
                    
                    if not path_blocked:
                        distance = sum(euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1))
                        if best_distance is None or distance < best_distance:
                            best_distance = distance
                            best_path = path
            
            if best_path:
                return best_distance, best_path
            else:
                return None, None
        
        # Target is walkable - direct pathfinding
        target_node = Node(target_x, target_y)
        path = find_path(grid, start_node, target_node)
        
        # Validate path
        if path and len(path) > 1:
            # Check if path goes through static obstacles
            path_blocked = any((node.x, node.y) in static_obstacles for node in path[1:])  # Skip start node
            
            if not path_blocked:
                distance = sum(euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1))
                return distance, path
            else:
                return None, None
        else:
            return None, None
    
    def _get_walkable_neighbors_of_target(self, grid, target_xy: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all walkable neighbor tiles of a non-walkable target tile.
        
        Args:
            grid: Game grid
            target_xy: Target tile coordinates
            
        Returns:
            List of walkable neighbor coordinates
        """
        target_x, target_y = target_xy
        neighbors = []
        
        # Check all 4 cardinal directions
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = target_x + dx, target_y + dy
            
            # Check bounds
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                neighbor_tile = grid.tiles[nx][ny]
                if hasattr(neighbor_tile, 'is_walkable') and neighbor_tile.is_walkable:
                    neighbors.append((nx, ny))
        
        return neighbors
    
    # --- Legacy tracking methods (kept for API compatibility) ---
    # These methods don't affect pathfinding in the tick-based system
    
    def update_agent_path(self, agent_id: str, path: List[Node], current_position: Tuple[float, float],
                         path_index: int = 0, speed: float = 1.875):
        """Store agent's active path for tracking (no collision prediction).
        
        This is kept for compatibility but doesn't affect pathfinding in tick-based system.
        """
        if path:
            self.active_paths[agent_id] = AgentPathInfo(
                path=path,
                current_position=current_position,
                path_index=path_index,
                speed=speed
            )
    
    def update_agent_position(self, agent_id: str, current_position: Tuple[float, float], 
                            path_index: int):
        """Update agent's position in tracked path (for reference only)."""
        if agent_id in self.active_paths:
            info = self.active_paths[agent_id]
            self.active_paths[agent_id] = AgentPathInfo(
                path=info.path,
                current_position=current_position,
                path_index=path_index,
                speed=info.speed
            )
    
    def clear_agent_path(self, agent_id: str):
        """Clear agent's tracked path."""
        if agent_id in self.active_paths:
            del self.active_paths[agent_id]
    
    def get_performance_stats(self) -> Dict[str, int]:
        """Get performance statistics."""
        return {
            "active_paths": len(self.active_paths),
            "collision_enabled": self.collision_enabled
        }
