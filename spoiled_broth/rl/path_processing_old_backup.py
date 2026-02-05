"""
Simple path processing system for tick-based multi-agent RL training.

This module provides spatial pathfinding utilities with reactive collision handling:
1. Calculate paths considering only CURRENT agent positions as obstacles
2. No temporal/future collision prediction
3. Paths are validated at runtime during tick execution
"""

from typing import Dict, List, Tuple, Optional, NamedTuple
import heapq
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
        self.collision_enabled = collision_enabled
        self.active_paths: Dict[str, AgentPathInfo] = {}  # Track agent paths (for reference only)
        
    def is_enabled(self) -> bool:
        """Check if collision detection is enabled."""
        return self.collision_enabled
    
    def get_shortest_path_distance(self, grid, from_xy: Tuple[int, int], to_xy: Tuple[int, int], 
                                 agent_id: str = None, current_time: float = 0.0, 
                                 agent_speed: float = 1.875, game=None) -> Tuple[Optional[float], Optional[List[Node]]]:
        """Calculate shortest path distance considering only current agent positions.
        
        No temporal collision prediction - only current positions treated as obstacles.
        
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
            
            # Get current agent positions as static obstacles
            static_obstacles = set()
            if self.collision_enabled and game is not None:
                for aid, agent_obj in game.gameObjects.items():
                    if aid != agent_id and hasattr(agent_obj, 'slot_x'):
                        static_obstacles.add((agent_obj.slot_x, agent_obj.slot_y))
            
            # Find shortest path to any of the walkable neighbors
            best_distance = None
            best_path = None
            
            for neighbor_xy in neighbors:
                neighbor_node = Node(neighbor_xy[0], neighbor_xy[1])
                path = find_path(grid, start_node, neighbor_node, obstacles=static_obstacles)
                
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
                    # Get STATIONARY obstacles for fallback check - moving agents will move away
                    static_obstacles = set()
                    if game is not None:
                        for other_agent_id, other_agent in game.gameObjects.items():
                            if (other_agent_id != agent_id and 
                                other_agent_id.startswith('ai_rl_') and
                                hasattr(other_agent, 'slot_x') and hasattr(other_agent, 'slot_y')):
                                # Only treat as static obstacle if agent is NOT actively moving
                                if other_agent_id not in self.active_paths:
                                    static_obstacles.add((other_agent.slot_x, other_agent.slot_y))
                    
                    for neighbor_xy in neighbors:
                        # Check if path would be blocked by static obstacles
                        if neighbor_xy not in static_obstacles:
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
            # Collision-aware pathfinding with multiple path attempts
            path = self._find_collision_free_path_with_alternatives(grid, start_node, target_node, agent_id, current_time, game)
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
                                agent_id: str, current_time: float, game=None) -> Optional[List[Node]]:
        """Find path that avoids collisions with other agent paths - optimized version.
        
        Args:
            grid: Game grid for pathfinding
            start_node: Starting node
            target_node: Target node
            agent_id: ID of the requesting agent
            current_time: Current game time for collision detection
            game: Game instance to get all agent positions for obstacle detection
        """
        # Get all agent positions as static obstacles
        static_obstacles = set()
        if game is not None:
            for other_agent_id, other_agent in game.gameObjects.items():
                if (other_agent_id != agent_id and 
                    other_agent_id.startswith('ai_rl_') and
                    hasattr(other_agent, 'slot_x') and hasattr(other_agent, 'slot_y')):
                    static_obstacles.add((other_agent.slot_x, other_agent.slot_y))
        
        # Remove the problematic early return - always do collision detection when enabled
        # Priority queue for A* with collision awareness
        counter = 0
        open_set = [(0, 0, counter, start_node, [start_node])]
        visited = set()
        
        # Increase limits for circular maps that may need longer alternative routes
        max_iterations = 1500  # Increased for better alternative path exploration
        
        for iteration in range(max_iterations):
            if not open_set:
                break
                
            f_score, g_score, _, current_node, current_path = heapq.heappop(open_set)
            
            node_key = (current_node.x, current_node.y)
            if node_key in visited:
                continue
                
            visited.add(node_key)
            
            # Check if we reached the goal
            if current_node.x == target_node.x and current_node.y == target_node.y:
                return current_path
            
            # Increase path length limit for circular maps
            if len(current_path) > 40:  # Increased limit to allow longer alternative routes
                continue
            
            # Explore neighbors
            for neighbor in self._get_valid_neighbors(grid, current_node, static_obstacles):
                neighbor_key = (neighbor.x, neighbor.y)
                if neighbor_key in visited:
                    continue
                
                new_path = current_path + [neighbor]
                
                # Enhanced collision check: check both static obstacles and dynamic paths
                if self._node_has_collision(neighbor, static_obstacles):
                    continue
                
                # More lenient collision checking - only check path collisions for longer paths
                # This allows more exploration of alternative routes
                if len(new_path) > 5 and self._path_has_collision(new_path, agent_id, current_time):
                    continue
                
                tentative_g = g_score + 1  # Use Manhattan distance (faster than euclidean)
                h_score = abs(neighbor.x - target_node.x) + abs(neighbor.y - target_node.y)
                f = tentative_g + h_score
                
                counter += 1
                heapq.heappush(open_set, (f, tentative_g, counter, neighbor, new_path))
        
        return None  # No collision-free path found
    
    def _find_collision_free_path_with_alternatives(self, grid, start_node: Node, target_node: Node, 
                                                  agent_id: str, current_time: float, game=None) -> Optional[List[Node]]:
        """Find collision-free path by trying multiple alternative routes.
        
        First tries the standard collision-aware A*, then if that fails, tries:
        1. Different heuristic weights to favor exploration
        2. Randomized tie-breaking to find alternative routes
        3. Multiple starting directions to explore different paths
        
        Args:
            grid: Game grid for pathfinding
            start_node: Starting node
            target_node: Target node
            agent_id: ID of the requesting agent
            current_time: Current game time for collision detection
            game: Game instance to get all agent positions for obstacle detection
            
        Returns:
            Optional[List[Node]]: First collision-free path found, or None if all attempts fail
        """
        # First attempt: Standard collision-aware pathfinding
        path = self._find_collision_free_path(grid, start_node, target_node, agent_id, current_time, game)
        if path:
            return path
                
        # Strategy 1: Different heuristic weights (favor exploration over direct routes)
        heuristic_weights = [0.5, 1.5, 2.0, 0.2, 3.0]
        for i, heuristic_weight in enumerate(heuristic_weights):
            if i >= self.ALTERNATIVE_PATH_ATTEMPTS // 2:
                break
            path = self._find_collision_free_path_weighted(grid, start_node, target_node, agent_id, 
                                                         current_time, game, heuristic_weight)
            if path:
                return path
        
        # Strategy 2: Try different initial directions to force exploration of different routes
        start_neighbors = self._get_valid_neighbors(grid, start_node, set())
        for i, start_direction in enumerate(start_neighbors):
            if i >= self.ALTERNATIVE_PATH_ATTEMPTS // 2:
                break
            # Force path to go through this neighbor first
            path = self._find_collision_free_path_via_waypoint(grid, start_node, start_direction, 
                                                             target_node, agent_id, current_time, game)
            if path:
                return path
        
        # Strategy 3: Try waiting a bit (delayed start) to let other agent pass
        for wait_time in [0.5, 1.0, 1.5]:
            path = self._find_collision_free_path(grid, start_node, target_node, agent_id, current_time + wait_time, game)
            if path:
                # Path found with delayed start - the agent would wait then follow this path
                return path
        
        return None
    
    def _find_collision_free_path_weighted(self, grid, start_node: Node, target_node: Node, 
                                         agent_id: str, current_time: float, game=None,
                                         heuristic_weight: float = 1.0) -> Optional[List[Node]]:
        """Find collision-free path with weighted heuristic for alternative route exploration."""
        # Get all agent positions as static obstacles
        static_obstacles = set()
        if game is not None:
            for other_agent_id, other_agent in game.gameObjects.items():
                if (other_agent_id != agent_id and 
                    other_agent_id.startswith('ai_rl_') and
                    hasattr(other_agent, 'slot_x') and hasattr(other_agent, 'slot_y')):
                    static_obstacles.add((other_agent.slot_x, other_agent.slot_y))
        
        # Priority queue for A* with weighted heuristic
        counter = 0
        open_set = [(0, 0, counter, start_node, [start_node])]
        visited = set()
        
        # Increased limits for alternative path finding - allow more exploration
        max_iterations = 3000  # Increased from 2000
        
        for iteration in range(max_iterations):
            if not open_set:
                break
                
            f_score, g_score, _, current_node, current_path = heapq.heappop(open_set)
            
            node_key = (current_node.x, current_node.y)
            if node_key in visited:
                continue
                
            visited.add(node_key)
            
            # Check if we reached the goal
            if current_node.x == target_node.x and current_node.y == target_node.y:
                return current_path
            
            # Allow longer paths for alternative routes
            if len(current_path) > 50:
                continue
            
            # Explore neighbors
            for neighbor in self._get_valid_neighbors(grid, current_node, static_obstacles):
                neighbor_key = (neighbor.x, neighbor.y)
                if neighbor_key in visited:
                    continue
                
                new_path = current_path + [neighbor]
                
                # Check collisions with less aggressive early termination
                if self._node_has_collision(neighbor, static_obstacles):
                    continue
                
                # Only check path collisions for paths longer than 5 nodes to allow more exploration
                if len(new_path) > 5 and self._path_has_collision(new_path, agent_id, current_time):
                    continue
                
                tentative_g = g_score + 1
                h_score = (abs(neighbor.x - target_node.x) + abs(neighbor.y - target_node.y)) * heuristic_weight
                f = tentative_g + h_score
                
                counter += 1
                heapq.heappush(open_set, (f, tentative_g, counter, neighbor, new_path))
        
        return None
    
    def _find_collision_free_path_via_waypoint(self, grid, start_node: Node, waypoint_node: Node,
                                             target_node: Node, agent_id: str, current_time: float, 
                                             game=None) -> Optional[List[Node]]:
        """Find collision-free path by forcing it through a specific waypoint."""
        # First segment: start to waypoint
        first_path = self._find_collision_free_path(grid, start_node, waypoint_node, agent_id, current_time, game)
        if not first_path:
            return None
        
        # Calculate time at waypoint
        first_segment_time = len(first_path) * 0.53  # Approximate time per step
        waypoint_time = current_time + first_segment_time
        
        # Second segment: waypoint to target
        second_path = self._find_collision_free_path(grid, waypoint_node, target_node, agent_id, waypoint_time, game)
        if not second_path:
            return None
        
        # Combine paths (remove duplicate waypoint node)
        if len(second_path) > 1:
            combined_path = first_path + second_path[1:]
        else:
            combined_path = first_path
        
        return combined_path
    
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
    
    def _get_valid_neighbors(self, grid, node: Node, static_obstacles: set = None) -> List[Node]:
        """Get valid neighboring nodes for pathfinding, avoiding static obstacles."""
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, Down, Right, Left
        
        if static_obstacles is None:
            static_obstacles = set()
        
        for dx, dy in directions:
            new_x, new_y = node.x + dx, node.y + dy
            
            # Check bounds
            if 0 <= new_x < grid.width and 0 <= new_y < grid.height:
                # Check if tile is walkable and not blocked by static obstacles
                if (grid.tiles[new_x][new_y].is_walkable and 
                    (new_x, new_y) not in static_obstacles):
                    neighbors.append(Node(new_x, new_y))
        
        return neighbors
    
    def _node_has_collision(self, node: Node, static_obstacles: set) -> bool:
        """Check if a single node position collides with static obstacles."""
        return (node.x, node.y) in static_obstacles
    
    def _path_has_collision(self, path: List[Node], agent_id: str, current_time: float) -> bool:
        """Check if a path collides with any active agent paths with caching.
        
        Args:
            path: The path to check for collisions
            agent_id: ID of the agent requesting the path
            current_time: Current game time
            
        Returns:
            bool: True if collision detected, False otherwise
        """
        if not self.collision_enabled or not self.active_paths:
            return False
        
        # Create cache key for this collision check
        path_key = self._create_path_cache_key(path, agent_id, current_time)
        if path_key in self._collision_cache:
            self._cache_hits += 1
            return self._collision_cache[path_key]
        
        self._cache_misses += 1
        
        # Early termination: if path is very short, quick check
        if len(path) <= 2:
            for other_agent_id, other_path_info in self.active_paths.items():
                if other_agent_id != agent_id:
                    # Quick position-based check for short paths
                    other_pos = (round(other_path_info.current_position[0]), 
                                round(other_path_info.current_position[1]))
                    for node in path:
                        if (node.x, node.y) == other_pos:
                            self._collision_cache[path_key] = True
                            return True
        
        # Full collision check for longer paths
        for other_agent_id, other_path_info in self.active_paths.items():
            if other_agent_id == agent_id:
                continue
            
            collision_detected = self._check_temporal_collision(path, other_path_info, current_time, agent_id)
            
            if collision_detected:
                self._collision_cache[path_key] = True
                return True
        
        self._collision_cache[path_key] = False
        
        # Limit cache size to prevent memory bloat
        if len(self._collision_cache) > 1000:
            # Remove oldest half of cache entries
            keys_to_remove = list(self._collision_cache.keys())[:500]
            for key in keys_to_remove:
                del self._collision_cache[key]
        
        return False
    
    def _create_path_cache_key(self, path: List[Node], agent_id: str, current_time: float) -> str:
        """Create a cache key for path collision detection."""
        # Use path start/end and agent positions for cache key
        if not path:
            return f"{agent_id}_empty_{current_time:.1f}"
        
        start = (path[0].x, path[0].y)
        end = (path[-1].x, path[-1].y) if len(path) > 1 else start
        
        # Include other agent positions in cache key
        other_positions = []
        for other_id, info in self.active_paths.items():
            if other_id != agent_id:
                pos = (round(info.current_position[0]), round(info.current_position[1]))
                other_positions.append(f"{other_id}_{pos[0]}_{pos[1]}")
        
        other_pos_str = "|".join(sorted(other_positions))
        return f"{agent_id}_{start[0]}_{start[1]}_{end[0]}_{end[1]}_{len(path)}_{other_pos_str}"
    
    def _check_temporal_collision(self, path: List[Node], other_path_info: AgentPathInfo, 
                                 current_time: float, requesting_agent_id: str = None) -> bool:
        """Fast collision detection using precomputed path snapshots.
        
        Args:
            path: The candidate path to check
            other_path_info: Information about the other agent's path and state
            current_time: Current game time
            
        Returns:
            bool: True if paths will collide in space-time (same tile, same time)
        """
        # Quick path validation
        if not path or not other_path_info.path:
            return False
        
        # Get requesting agent speed - try to get it from stored paths first, then fallback
        requesting_agent_speed = getattr(self, '_requesting_agent_speed', 1.875)
        
        # Try to get the actual requesting agent's speed from stored paths if available
        if requesting_agent_id and requesting_agent_id in self.active_paths:
            requesting_agent_speed = self.active_paths[requesting_agent_id].speed
        
        # Calculate maximum time needed to complete both paths
        requesting_path_time = self._calculate_path_completion_time(path, 0, requesting_agent_speed)
        other_path_time = self._calculate_path_completion_time_for_agent(
            other_path_info.path, other_path_info.path_index, 
            other_path_info.current_position, other_path_info.speed
        )
        
        # Use the maximum time between both paths, but cap it to avoid overly long collision windows
        max_simulation_time = min(
            max(requesting_path_time, other_path_time) + self.FINAL_POSITION_OCCUPY_TIME,
            self.MAX_COLLISION_CHECK_TIME
        )
        
        # Precompute path snapshots for both agents until paths complete
        requesting_snapshots = self._get_path_snapshots(path, 0, requesting_agent_speed, max_simulation_time)
        other_snapshots = self._get_path_snapshots_for_agent(
            other_path_info.path, other_path_info.path_index, 
            other_path_info.current_position, other_path_info.speed, max_simulation_time
        )
        
        # Quick check: if paths don't overlap in time, no collision
        if not requesting_snapshots or not other_snapshots:
            return False
        
        # Convert to sets for fast intersection check
        requesting_tiles = set(requesting_snapshots.values())
        other_tiles = set(other_snapshots.values())
        
        # Quick overlap check - if no common tiles, no collision possible
        if not requesting_tiles.intersection(other_tiles):
            return False
        
        # Check for time-tile collisions with temporal tolerance
        collision_found = False
        tolerance_steps = int(self.TEMPORAL_TOLERANCE * 10)  # Convert seconds to 0.1s steps
        
        for time_key in requesting_snapshots:
            requesting_tile = requesting_snapshots[time_key]
            
            # Check if other agent occupies same tile within temporal tolerance window
            has_collision_in_window = False
            for time_offset in range(-tolerance_steps, tolerance_steps + 1):
                other_time_key = time_key + time_offset
                if other_time_key in other_snapshots:
                    if requesting_tile == other_snapshots[other_time_key]:
                        has_collision_in_window = True
                        break
            
            if has_collision_in_window:
                collision_found = True
                break
        
        return collision_found
    
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
    
    def _get_path_snapshots(self, path: List[Node], start_idx: int, speed: float, 
                           max_time: float) -> Dict[int, Tuple[int, int]]:
        """Precompute path positions at 0.1-second intervals until max_time is reached.
        Agent stays at final position after completing the path.
        
        Returns:
            Dict mapping time_step*10 -> (x, y) tile coordinates
        """
        if not path or start_idx >= len(path):
            return {}
        
        snapshots = {}
        time_step = 0.1
        current_time = 0.0
        
        # Calculate total path distance
        total_distance = 0.0
        for i in range(start_idx, len(path) - 1):
            dist = abs(path[i+1].x - path[i].x) + abs(path[i+1].y - path[i].y)
            total_distance += dist
        
        path_completion_time = total_distance / speed if speed > 0 else 0.0
        
        # Generate snapshots until max_time is reached
        while current_time <= max_time:
            time_key = int(round(current_time * 10))  # Convert to int key (0.1s = 1, 0.2s = 2, etc.)
            
            if current_time >= path_completion_time and path_completion_time > 0:
                # Agent has completed path - only occupy final position for limited time
                time_since_completion = current_time - path_completion_time
                if time_since_completion <= self.FINAL_POSITION_OCCUPY_TIME:
                    final_tile = (path[-1].x, path[-1].y)
                    snapshots[time_key] = final_tile
                else:
                    # After occupation time expires, agent no longer blocks this path
                    break
                
            else:
                # Agent is still moving along the path
                distance_traveled = speed * current_time
                
                # Find current position along path
                remaining_distance = distance_traveled
                current_path_idx = start_idx
                current_pos = (float(path[start_idx].x), float(path[start_idx].y))
                
                # Move along path segments
                while remaining_distance > 0 and current_path_idx < len(path) - 1:
                    next_node = path[current_path_idx + 1]
                    segment_distance = abs(next_node.x - current_pos[0]) + abs(next_node.y - current_pos[1])
                    
                    if remaining_distance >= segment_distance:
                        # Complete this segment
                        current_pos = (float(next_node.x), float(next_node.y))
                        remaining_distance -= segment_distance
                        current_path_idx += 1
                    else:
                        # Partial segment
                        if segment_distance > 0:
                            ratio = remaining_distance / segment_distance
                            current_pos = (
                                current_pos[0] + ratio * (next_node.x - current_pos[0]),
                                current_pos[1] + ratio * (next_node.y - current_pos[1])
                            )
                        remaining_distance = 0
                
                tile = (round(current_pos[0]), round(current_pos[1]))
                snapshots[time_key] = tile
            
            current_time += time_step
        
        return snapshots
    
    def _get_path_snapshots_for_agent(self, path: List[Node], start_path_idx: int,
                                     current_position: Tuple[float, float], speed: float,
                                     max_time: float) -> Dict[int, Tuple[int, int]]:
        """Generate snapshots for an agent already in motion until max_time is reached.
        Agent stays at final position after completing the path."""
        if not path or start_path_idx >= len(path):
            return {}
        
        snapshots = {}
        time_step = 0.1
        current_time = 0.0
        
        # Calculate remaining path distance and completion time
        remaining_distance = 0.0
        
        # Distance from current position to next waypoint
        if start_path_idx < len(path):
            target_pos = (path[start_path_idx].x, path[start_path_idx].y)
            remaining_distance += abs(target_pos[0] - current_position[0]) + abs(target_pos[1] - current_position[1])
        
        # Distance for remaining path segments
        for i in range(start_path_idx, len(path) - 1):
            dist = abs(path[i+1].x - path[i].x) + abs(path[i+1].y - path[i].y)
            remaining_distance += dist
        
        path_completion_time = remaining_distance / speed if speed > 0 else 0.0
        
        # Start from current position and simulate movement
        current_pos = current_position
        path_idx = start_path_idx
        
        while current_time <= max_time:
            time_key = int(round(current_time * 10))  # Convert to int key (0.1s = 1, 0.2s = 2, etc.)
            tile = (round(current_pos[0]), round(current_pos[1]))
            snapshots[time_key] = tile
            
            if current_time >= path_completion_time and path_completion_time > 0:
                # Agent has completed path - only occupy final position for limited time
                time_since_completion = current_time - path_completion_time
                if time_since_completion <= self.FINAL_POSITION_OCCUPY_TIME:
                    if path_idx < len(path):
                        # Move to final position if not already there
                        final_pos = (path[-1].x, path[-1].y)
                        current_pos = final_pos
                        path_idx = len(path)
                else:
                    # After occupation time expires, agent no longer blocks this path
                    break
            
            else:
                # Agent is still moving along the path
                distance_to_move = speed * time_step
                
                while distance_to_move > 0 and path_idx < len(path):
                    target = (path[path_idx].x, path[path_idx].y)
                    dist_to_target = abs(target[0] - current_pos[0]) + abs(target[1] - current_pos[1])
                    
                    if dist_to_target <= distance_to_move:
                        # Reach this waypoint
                        current_pos = target
                        distance_to_move -= dist_to_target
                        path_idx += 1
                        
                        # Check if we've completed the path
                        if path_idx >= len(path):
                            break
                    else:
                        # Partial movement toward target
                        if dist_to_target > 0:
                            ratio = distance_to_move / dist_to_target
                            old_pos = current_pos
                            current_pos = (
                                current_pos[0] + ratio * (target[0] - current_pos[0]),
                                current_pos[1] + ratio * (target[1] - current_pos[1])
                            )
                        
                        distance_to_move = 0
            
            current_time += time_step
        
        return snapshots
    
    def _binary_search_segment(self, cumulative_distances: List[float], target_distance: float) -> int:
        """Binary search to find which path segment contains the target distance."""
        left, right = 0, len(cumulative_distances) - 1
        
        while left < right:
            mid = (left + right) // 2
            if cumulative_distances[mid] <= target_distance:
                left = mid + 1
            else:
                right = mid
        
        return left - 1
    
    def _calculate_path_completion_time(self, path: List[Node], start_idx: int, speed: float) -> float:
        """Calculate how long it takes to complete a path from a given starting index.
        
        Args:
            path: The path to calculate completion time for
            start_idx: Starting index in the path
            speed: Movement speed in tiles/second
            
        Returns:
            Time in seconds to complete the path
        """
        if not path or start_idx >= len(path) or speed <= 0:
            return 0.0
        
        total_distance = 0.0
        for i in range(start_idx, len(path) - 1):
            dist = abs(path[i+1].x - path[i].x) + abs(path[i+1].y - path[i].y)
            total_distance += dist
        
        return total_distance / speed
    
    def _calculate_path_completion_time_for_agent(self, path: List[Node], start_path_idx: int,
                                                 current_position: Tuple[float, float], 
                                                 speed: float) -> float:
        """Calculate completion time for an agent already in motion on their path.
        
        Args:
            path: The agent's path
            start_path_idx: Current index in the path
            current_position: Agent's current position
            speed: Movement speed in tiles/second
            
        Returns:
            Time in seconds to complete remaining path
        """
        if not path or start_path_idx >= len(path) or speed <= 0:
            return 0.0
        
        total_distance = 0.0
        
        # Distance from current position to next waypoint
        if start_path_idx < len(path):
            target_pos = (path[start_path_idx].x, path[start_path_idx].y)
            total_distance += abs(target_pos[0] - current_position[0]) + abs(target_pos[1] - current_position[1])
        
        # Distance for remaining path segments
        for i in range(start_path_idx, len(path) - 1):
            dist = abs(path[i+1].x - path[i].x) + abs(path[i+1].y - path[i].y)
            total_distance += dist
        
        return total_distance / speed
    
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
        """Get performance statistics including cache efficiency."""
        total_checks = self._cache_hits + self._cache_misses
        cache_hit_rate = (self._cache_hits / total_checks * 100) if total_checks > 0 else 0
        
        return {
            "active_paths": len(self.active_paths),
            "cache_size": len(self._collision_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate_percent": round(cache_hit_rate, 1)
        }