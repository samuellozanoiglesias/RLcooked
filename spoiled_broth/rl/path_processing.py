"""
Simple path processing system for multi-agent RL training.

This module implements efficient path caching:
1. Calculate paths once during observation space generation
2. Store and reuse paths during action execution
3. Simple collision detection for active paths
"""

from typing import Dict, List, Tuple, Optional, NamedTuple
import heapq
from engine.extensions.topDownGridWorld.a_star import Node, euclidean_distance, find_path


class AgentPathInfo(NamedTuple):
    """Information about an agent's active path and state."""
    path: List[Node]
    current_position: Tuple[float, float]  # (x, y) in grid coordinates
    path_index: int  # Current position in path
    speed: float  # Agent's walking speed in tiles/second


class PathProcessor:
    """Simple path calculation system for RL training."""
    
    def __init__(self, map_nr: int, collision_enabled: bool = True):
        self.collision_enabled = collision_enabled
        self.active_paths: Dict[str, AgentPathInfo] = {}  # Agent path information storage
        
    def is_enabled(self) -> bool:
        """Check if collision detection is enabled."""
        return self.collision_enabled
    
    def get_shortest_path_distance(self, grid, from_xy: Tuple[int, int], to_xy: Tuple[int, int], 
                                 agent_id: str = None, current_time: float = 0.0, 
                                 agent_speed: float = 1.875) -> Tuple[Optional[float], Optional[List[Node]]]:
        """Calculate shortest path distance and return both distance and path.
        
        Args:
            grid: Game grid for pathfinding
            from_xy: Start position (x, y)
            to_xy: Target position (x, y) - can be non-walkable; will find path to nearest walkable neighbor
            agent_id: ID of the requesting agent
            current_time: Current game time for collision detection
            agent_speed: Walking speed of the requesting agent in tiles/second (default 1.875 = 30/16)
            
        Returns:
            tuple: (distance, path) where distance is path length or None if no path exists,
                  and path is list of Node objects or None
        """
        start_node = Node(from_xy[0], from_xy[1])
        
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
                # Return distance 0 and a static path (just the current position)
                return 0.0, [start_node]
            
            # Find shortest path to any of the walkable neighbors
            best_distance = None
            best_path = None
            
            for neighbor_xy in neighbors:
                neighbor_node = Node(neighbor_xy[0], neighbor_xy[1])
                
                if not self.collision_enabled:
                    path = find_path(grid, start_node, neighbor_node)
                else:
                    path = self._find_collision_free_path(grid, start_node, neighbor_node, agent_id, current_time)
                
                if path and len(path) > 1:
                    distance = sum(euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1))
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_path = path
            
            if best_path:
                return best_distance, best_path
            else:
                # No path to any neighbor found
                if self.collision_enabled:
                    # Check if fallback path exists (collision blocking)
                    for neighbor_xy in neighbors:
                        neighbor_node = Node(neighbor_xy[0], neighbor_xy[1])
                        fallback = find_path(grid, start_node, neighbor_node)
                        if fallback and len(fallback) > 1:
                            return -1, -1
                return None, None
        
        # Target is walkable - use original logic
        target_node = Node(target_x, target_y)
                
        # Store agent speed for collision detection
        self._requesting_agent_speed = agent_speed
        
        if not self.collision_enabled:
            # Simple A* pathfinding without collision detection
            path = find_path(grid, start_node, target_node)
            if path and len(path) > 1:
                # Calculate actual path distance
                distance = sum(euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1))
                return distance, path
            else:
                return None, None
        else:
            # Collision-aware pathfinding
            path = self._find_collision_free_path(grid, start_node, target_node, agent_id, current_time)
            if path and len(path) > 1:
                distance = sum(euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1))
                return distance, path
            else:
                path = find_path(grid, start_node, target_node)
                if path and len(path) > 1:
                    return -1, -1  # Indicate path blocked by collisions
                else:
                    return None, None
    
    def _find_collision_free_path(self, grid, start_node: Node, target_node: Node, 
                                agent_id: str, current_time: float) -> Optional[List[Node]]:
        """Find path that avoids collisions with other agent paths.
        
        This implements collision-aware A* by checking if each potential path
        conflicts with stored active agent paths.
        Args:
            grid: Game grid for pathfinding
            start_node: Starting node
            target_node: Target node
            agent_id: ID of the requesting agent
            current_time: Current game time for collision detection
        """
        # Priority queue for A* with collision awareness
        # Use counter to break ties when f_score and g_score are equal
        counter = 0
        open_set = [(0, 0, counter, start_node, [start_node])]  # (f_score, g_score, counter, node, path)
        visited = set()
        
        iterations = 0
        while open_set:
            iterations += 1
            if iterations > 1000:  # Safety limit
                print(f"[_find_collision_free_path] Hit iteration limit!")
                break
            f_score, g_score, _, current_node, current_path = heapq.heappop(open_set)
            
            # Use node position as key for visited tracking
            node_key = (current_node.x, current_node.y)
            if node_key in visited:
                continue
                
            visited.add(node_key)
            
            # Check if we reached the goal
            if current_node.x == target_node.x and current_node.y == target_node.y:
                return current_path
            
            # Explore neighbors
            for neighbor in self._get_valid_neighbors(grid, current_node):
                neighbor_key = (neighbor.x, neighbor.y)
                if neighbor_key in visited:
                    continue
                
                new_path = current_path + [neighbor]
                
                # Check for collisions with other agent paths
                if self._path_has_collision(new_path, agent_id, current_time):
                    continue
                
                tentative_g = g_score + euclidean_distance(current_node, neighbor)
                h_score = euclidean_distance(neighbor, target_node)
                f = tentative_g + h_score
                
                counter += 1
                heapq.heappush(open_set, (f, tentative_g, counter, neighbor, new_path))
        
        return None  # No collision-free path found
    
    def _get_walkable_neighbors_of_target(self, grid, target_xy: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get walkable neighboring tiles around a target position.
        
        Args:
            grid: Game grid
            target_xy: Target position (x, y)
            
        Returns:
            List of (x, y) tuples for walkable neighbors
        """
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, Down, Right, Left
        
        for dx, dy in directions:
            new_x = target_xy[0] + dx
            new_y = target_xy[1] + dy
            
            # Check bounds
            if 0 <= new_x < grid.width and 0 <= new_y < grid.height:
                # Check if tile is walkable
                if grid.tiles[new_x][new_y].is_walkable:
                    neighbors.append((new_x, new_y))
        
        return neighbors
    
    def _get_valid_neighbors(self, grid, node: Node) -> List[Node]:
        """Get valid neighboring nodes for pathfinding."""
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, Down, Right, Left
        
        for dx, dy in directions:
            new_x, new_y = node.x + dx, node.y + dy
            
            # Check bounds
            if 0 <= new_x < grid.width and 0 <= new_y < grid.height:
                # Check if tile is walkable
                if grid.tiles[new_x][new_y].is_walkable:
                    neighbors.append(Node(new_x, new_y))
        
        return neighbors
    
    def _path_has_collision(self, path: List[Node], agent_id: str, current_time: float) -> bool:
        """Check if a path collides with any active agent paths.
        
        Uses temporal-spatial collision detection considering:
        - Current positions of all agents
        - Agent movement speeds
        - Future positions as agents move along their paths
        - Time-based prediction of when agents will occupy positions
        
        Args:
            path: The path to check for collisions
            agent_id: ID of the agent requesting the path
            current_time: Current game time
            
        Returns:
            bool: True if collision detected, False otherwise
        """
        if not self.collision_enabled:
            return False
            
        # Check against all other active agent paths
        for other_agent_id, other_path_info in self.active_paths.items():
            if other_agent_id == agent_id:
                continue
            
            # Perform temporal-spatial collision check
            if self._check_temporal_collision(path, other_path_info, current_time):
                return True
        
        return False
    
    def _check_temporal_collision(self, path: List[Node], other_path_info: AgentPathInfo, 
                                 current_time: float) -> bool:
        """Check if two paths will collide considering movement over time.
        
        This checks for exact tile-based collisions: agents collide when they occupy
        the same tile at the same time. Checks are performed every 0.5 seconds.
        
        Args:
            path: The candidate path to check
            other_path_info: Information about the other agent's path and state
            current_time: Current game time
            
        Returns:
            bool: True if paths will collide in space-time (same tile, same time)
        """
        other_path = other_path_info.path
        other_pos = other_path_info.current_position
        other_speed = other_path_info.speed
        other_path_idx = other_path_info.path_index
        
        # Get requesting agent speed (stored during pathfinding call) in tiles/second
        requesting_agent_speed = getattr(self, '_requesting_agent_speed', 1.875)
        
        # Time simulation parameters
        time_step = 0.5  # Check collisions every 0.5 seconds as requested
        max_simulation_time = 10.0  # Don't simulate beyond 10 time units
        
        # Simulate both agents moving along their paths
        requesting_time = 0.0
        
        requesting_idx = 0
        other_idx = other_path_idx
        
        # Calculate initial tile positions
        requesting_tile = self._get_current_tile_at_time(path, 0, requesting_agent_speed, requesting_time)
        other_tile = self._get_current_tile_at_time_for_agent(
            other_path, other_path_idx, other_pos, other_speed, requesting_time
        )
        
        while requesting_time < max_simulation_time:
            # Get tile positions at current time
            requesting_tile = self._get_current_tile_at_time(path, 0, requesting_agent_speed, requesting_time)
            other_tile = self._get_current_tile_at_time_for_agent(
                other_path, other_path_idx, other_pos, other_speed, requesting_time
            )
            
            # Check if both agents are still on valid paths
            if requesting_tile is None:
                break  # Requesting agent finished their path
            if other_tile is None:
                # Other agent finished, check if they stay at final position
                if len(other_path) > 0:
                    final_tile = (other_path[-1].x, other_path[-1].y)
                    if requesting_tile == final_tile:
                        return True
                break
            
            # Check for exact tile collision
            if requesting_tile == other_tile:
                return True
            
            # Advance time
            requesting_time += time_step
        
        return False
    
    def _distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return (dx * dx + dy * dy) ** 0.5
    
    def _move_toward(self, current: Tuple[float, float], target: Tuple[float, float], 
                    distance: float) -> Tuple[float, float]:
        """Move from current position toward target by specified distance."""
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        magnitude = self._distance(current, target)
        
        if magnitude == 0:
            return current
        
        # Normalize and scale by distance
        ratio = distance / magnitude
        new_x = current[0] + dx * ratio
        new_y = current[1] + dy * ratio
        
        return (new_x, new_y)
    
    def _get_current_tile_at_time(self, path: List[Node], start_idx: int, 
                                 speed: float, elapsed_time: float) -> Optional[Tuple[int, int]]:
        """Calculate which tile an agent occupies at a specific time along their path.
        
        Args:
            path: The agent's path
            start_idx: Starting path index (usually 0 for new paths)
            speed: Agent's movement speed in tiles/second
            elapsed_time: Time elapsed since starting the path
            
        Returns:
            (x, y) tile coordinates, or None if agent has finished the path
        """
        if not path or start_idx >= len(path):
            return None
        
        # Calculate total distance that can be traveled in elapsed_time
        distance_traveled = speed * elapsed_time
        current_distance = 0.0
        current_idx = start_idx
        
        # Walk through path segments until we've traveled the required distance
        while current_idx < len(path) - 1:
            next_idx = current_idx + 1
            segment_distance = self._distance(
                (path[current_idx].x, path[current_idx].y),
                (path[next_idx].x, path[next_idx].y)
            )
            
            if current_distance + segment_distance >= distance_traveled:
                # Agent is somewhere in this segment
                remaining_distance = distance_traveled - current_distance
                progress = remaining_distance / segment_distance if segment_distance > 0 else 0.0
                
                # Interpolate position within the segment
                start_pos = (path[current_idx].x, path[current_idx].y)
                end_pos = (path[next_idx].x, path[next_idx].y)
                current_pos = self._interpolate_position(start_pos, end_pos, progress)
                
                # Return the tile containing this position (round to nearest integer)
                return (round(current_pos[0]), round(current_pos[1]))
            
            current_distance += segment_distance
            current_idx += 1
        
        # Agent has reached the end of the path
        if current_idx < len(path):
            return (path[current_idx].x, path[current_idx].y)
        
        return None
    
    def _get_current_tile_at_time_for_agent(self, path: List[Node], start_path_idx: int,
                                           current_position: Tuple[float, float], speed: float,
                                           elapsed_time: float) -> Optional[Tuple[int, int]]:
        """Calculate which tile an agent occupies, accounting for their current position in their path.
        
        Args:
            path: The agent's full path
            start_path_idx: Current index in the path where the agent is
            current_position: Agent's current position (x, y)
            speed: Agent's movement speed in tiles/second
            elapsed_time: Time elapsed since the collision check started
            
        Returns:
            (x, y) tile coordinates, or None if agent has finished the path
        """
        if not path or start_path_idx >= len(path):
            return None
        
        # If no time has elapsed, return current tile
        if elapsed_time <= 0:
            return (round(current_position[0]), round(current_position[1]))
        
        # Calculate distance agent can travel in elapsed_time
        distance_traveled = speed * elapsed_time
        remaining_distance = distance_traveled
        current_pos = current_position
        current_idx = start_path_idx
        
        # Move through path segments
        while current_idx < len(path) and remaining_distance > 0:
            target_pos = (path[current_idx].x, path[current_idx].y)
            distance_to_target = self._distance(current_pos, target_pos)
            
            if distance_to_target <= remaining_distance:
                # Can reach this waypoint, move to it
                current_pos = target_pos
                remaining_distance -= distance_to_target
                current_idx += 1
            else:
                # Can't reach waypoint, stop partway
                progress = remaining_distance / distance_to_target if distance_to_target > 0 else 0.0
                current_pos = self._interpolate_position(current_pos, target_pos, progress)
                remaining_distance = 0
        
        # Return the tile containing the final position
        return (round(current_pos[0]), round(current_pos[1]))
    
    def _interpolate_position(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float],
                            progress: float) -> Tuple[float, float]:
        """Interpolate position between two points based on progress (0.0 to 1.0)."""
        x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
        y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
        return (x, y)
    
    def update_agent_path(self, agent_id: str, path: List[Node], current_position: Tuple[float, float],
                         path_index: int = 0, speed: float = 1.875):
        """Update or register an agent's active path information.
        
        Args:
            agent_id: ID of the agent
            path: The full path the agent is following
            current_position: Agent's current position (x, y) in grid coordinates
            path_index: Current index in the path
            speed: Agent's walking speed in tiles/second (default 1.875 = 30/16)
        """
        if path and len(path) > 0:
            self.active_paths[agent_id] = AgentPathInfo(
                path=path,
                current_position=current_position,
                path_index=path_index,
                speed=speed
            )
    
    def update_agent_position(self, agent_id: str, current_position: Tuple[float, float], 
                            path_index: int):
        """Update an agent's current position and path progress.
        
        Args:
            agent_id: ID of the agent
            current_position: Updated position (x, y) in grid coordinates
            path_index: Updated index in the path
        """
        if agent_id in self.active_paths:
            old_info = self.active_paths[agent_id]
            self.active_paths[agent_id] = AgentPathInfo(
                path=old_info.path,
                current_position=current_position,
                path_index=path_index,
                speed=old_info.speed
            )
    
    def clear_agent_path(self, agent_id: str):
        """Clear an agent's stored path when they finish or change course."""
        if agent_id in self.active_paths:
            del self.active_paths[agent_id]
    
    def get_performance_stats(self) -> Dict[str, int]:
        """Get simple performance statistics."""
        return {
            "active_paths": len(self.active_paths),
            "collision_checks": 0,
            "paths_found": 0,
            "paths_blocked": 0
        }