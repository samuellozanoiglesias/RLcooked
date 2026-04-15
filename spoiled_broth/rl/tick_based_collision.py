"""
Collision detection and resolution for tick-based simulation.

This module handles:
- Predictive collision detection (before movement happens)
- Collision resolution through rerouting or cancellation
- Path-finding with dynamic obstacles
"""

import logging
from engine.extensions.topDownGridWorld.a_star import Node, find_path

logger = logging.getLogger(__name__)


def predict_next_tile_positions(env):
    """Predict where each agent will be in the next tick.
    
    Returns:
        tuple: (next_positions, agent_movement_status)
            - next_positions: dict agent_id -> (next_x, next_y) for ALL agents
            - agent_movement_status: dict agent_id -> bool (True if agent is moving, False if stationary)
    """
    next_positions = {}
    agent_movement_status = {}
    tick_duration = float(getattr(env, 'tick_duration', 0.5))
    
    for agent_id in env.agents:
        state = env.agent_state[agent_id]
        agent = env.agent_map[agent_id]
        
        # Check if agent is moving and will reach next tile
        if (len(state['current_path']) > 0 and 
            state['path_index'] < len(state['current_path'])):
            
            # Get agent's movement speed
            agent_speed_tiles = (agent.speed / 16.0)  # Convert pixels/sec to tiles/sec
            movement_distance = agent_speed_tiles * tick_duration  # Distance in next tick
            
            # Check if agent will reach next tile in this tick
            new_progress = state['movement_progress'] + movement_distance
            
            if new_progress >= 1.0:
                # Agent will move to next tile
                next_node = state['current_path'][state['path_index']]
                next_positions[agent_id] = (next_node.x, next_node.y)
                agent_movement_status[agent_id] = True  # Agent is moving
            else:
                # Agent won't reach next tile, stays at current position
                next_positions[agent_id] = (agent.slot_x, agent.slot_y)
                agent_movement_status[agent_id] = False  # Agent is stationary this tick
        else:
            # Agent is not moving, stays at current position
            next_positions[agent_id] = (agent.slot_x, agent.slot_y)
            agent_movement_status[agent_id] = False  # Agent is stationary
    
    return next_positions, agent_movement_status


def detect_predictive_collisions(env):
    """Detect collisions BEFORE they happen by predicting next tick positions.
    
    Smart collision detection rules:
    1. If Agent A is stationary and Agent B moves toward A's tile: only stop Agent B
    2. If both agents are moving to same tile: stop both (or try rerouting)
    3. If agents swap positions (both moving): stop both (or try rerouting)
    
    Returns:
        list: [(agent_to_stop, other_agent_id, collision_tile, collision_type), ...]
              collision_type: 'moving_into_stationary', 'mutual_collision', 'tile_swap'
    """
    if not env.path_processor.is_enabled():
        return []
    
    next_positions, agent_movement_status = predict_next_tile_positions(env)
    
    collision_events = []
    
    # Check for position collisions (agents going to same tile)
    tile_occupancy = {}  # (x, y) -> [agent_ids]
    for agent_id, (x, y) in next_positions.items():
        tile = (x, y)
        if tile not in tile_occupancy:
            tile_occupancy[tile] = []
        tile_occupancy[tile].append(agent_id)
    
    # Find collision pairs from position collisions
    for collision_tile, agents_on_tile in tile_occupancy.items():
        if len(agents_on_tile) > 1:
            # Determine which agents are moving and which are stationary
            moving_agents = [aid for aid in agents_on_tile if agent_movement_status[aid]]
            stationary_agents = [aid for aid in agents_on_tile if not agent_movement_status[aid]]
            
            if len(stationary_agents) > 0 and len(moving_agents) > 0:
                # Case 1: Moving agents trying to enter a tile occupied by stationary agent(s)
                # Only stop the moving agents, not the stationary ones
                for moving_agent in moving_agents:
                    # Pick first stationary agent as reference (could be any)
                    stationary_agent = stationary_agents[0]
                    collision_events.append((moving_agent, stationary_agent, collision_tile, 'moving_into_stationary'))
            
            elif len(moving_agents) > 1:
                # Case 2: Multiple moving agents trying to reach same tile
                # Stop all moving agents (mutual collision)
                for i in range(len(moving_agents)):
                    for j in range(i + 1, len(moving_agents)):
                        collision_events.append((moving_agents[i], moving_agents[j], collision_tile, 'mutual_collision'))
            
            # Note: If len(stationary_agents) > 1 and len(moving_agents) == 0, no collision needed
            # Multiple stationary agents can occupy same tile without issue
    
    # Check for tile swapping collisions (agents exchanging positions)
    current_positions = {agent_id: (env.agent_map[agent_id].slot_x, env.agent_map[agent_id].slot_y) 
                        for agent_id in next_positions.keys()}
    
    agents_list = list(next_positions.keys())
    for i in range(len(agents_list)):
        for j in range(i + 1, len(agents_list)):
            agent1_id = agents_list[i]
            agent2_id = agents_list[j]
            
            agent1_current = current_positions[agent1_id]
            agent1_next = next_positions[agent1_id]
            agent2_current = current_positions[agent2_id]
            agent2_next = next_positions[agent2_id]
            
            # Check if agents are swapping positions: A->B and B->A
            # Both agents must be moving for this to be a true swap collision
            if (agent1_current == agent2_next and agent2_current == agent1_next and 
                agent1_current != agent1_next and  # Ensure they're actually moving
                agent_movement_status[agent1_id] and agent_movement_status[agent2_id]):  # Both must be moving
                # Use the first agent's next position as the "collision tile" for reporting
                collision_events.append((agent1_id, agent2_id, agent1_next, 'tile_swap'))
    
    return collision_events


def find_path_avoiding_obstacles(grid, start, goal, blocked_tiles=None):
    """Enhanced pathfinding that treats blocked tiles as temporary obstacles.
    
    Args:
        grid: Game grid
        start: Start Node
        goal: Goal Node  
        blocked_tiles: Set of (x, y) tuples that should be treated as obstacles
        
    Returns:
        List[Node] or None: Path if found, None otherwise
    """
    if blocked_tiles is None:
        blocked_tiles = set()
    
    # Create a custom grid wrapper that treats blocked tiles as obstacles
    class GridWithObstacles:
        def __init__(self, original_grid, blocked_positions):
            self.original_grid = original_grid
            self.blocked_positions = blocked_positions
            # Copy attributes A* needs
            self.width = original_grid.width
            self.height = original_grid.height
            self.tile_size = getattr(original_grid, 'tile_size', 16)
            # Create a wrapper for tiles that intercepts walkability checks
            self._create_tile_wrapper()
            
        def _create_tile_wrapper(self):
            """Create a 2D array of tile wrappers that override walkability."""
            class TileWrapper:
                def __init__(self, original_tile, is_blocked):
                    self._original = original_tile
                    self._blocked = is_blocked
                    
                @property
                def is_walkable(self):
                    if self._blocked:
                        return False
                    return self._original.is_walkable
                
                def __getattr__(self, name):
                    # Delegate all other attributes to original tile
                    return getattr(self._original, name)
            
            # Create wrapped tiles
            self.tiles = []
            for x in range(self.width):
                column = []
                for y in range(self.height):
                    original_tile = self.original_grid.tiles[x][y]
                    is_blocked = (x, y) in self.blocked_positions
                    wrapped = TileWrapper(original_tile, is_blocked)
                    column.append(wrapped)
                self.tiles.append(column)
    
    grid_with_obstacles = GridWithObstacles(grid, blocked_tiles)
    path = find_path(grid_with_obstacles, start, goal)
    return path


def find_predictive_alternative_path(env, agent_id, collision_tile, other_agent_current_pos, other_agent_next_pos):
    """Try to find an alternative path that avoids both current and predicted positions of other agent.
    
    Uses A* pathfinding with blocked tiles to route around obstacles.
    If no alternative path exists, returns None (action will be cancelled).
    
    Args:
        env: GameEnv instance
        agent_id: ID of agent to reroute
        collision_tile: Tile coordinates where collision would occur
        other_agent_current_pos: Current position of other agent
        other_agent_next_pos: Where other agent will be next tick
        
    Returns:
        List[Node] or None: Alternative path if found, None otherwise
    """
    
    state = env.agent_state[agent_id]
    agent = env.agent_map[agent_id]
    
    # Get current position and destination
    current_pos = (agent.slot_x, agent.slot_y)
    target_tile_index = state['target_tile_index']
    
    if target_tile_index is None:
        return None
    
    # Calculate target position
    grid_w = env.game.grid.width
    target_x = target_tile_index % grid_w
    target_y = target_tile_index // grid_w
    target_tile = env.game.grid.tiles[target_x][target_y]
    
    # Determine which positions to block
    # Only block the other agent's NEXT position (where they're moving to)
    # Their CURRENT position will become free when they move away
    # Exception: If this is a tile-swapping collision, block both positions
    current_pos_tuple = (current_pos[0], current_pos[1])
    if (current_pos_tuple == other_agent_next_pos and 
        other_agent_current_pos == (target_x, target_y)):
        # Tile swapping: block both positions since it's an inherent conflict
        obstacles = {other_agent_current_pos, other_agent_next_pos}
    else:
        # Normal collision: only block where the other agent is going
        obstacles = {other_agent_next_pos}
    
    start_node = Node(current_pos[0], current_pos[1])
    
    # Try standard rerouting to the same destination using A* with blocked obstacles
    if hasattr(target_tile, 'is_walkable') and target_tile.is_walkable:
        # Target is walkable - use A* with blocked tiles
        goal_node = Node(target_x, target_y)
        alt_path = find_path_avoiding_obstacles(env.game.grid, start_node, goal_node, obstacles)
        
        if alt_path and len(alt_path) > 1:
            return alt_path
    else:
        # Target is not walkable - find walkable neighbors and use A* for each
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = target_x + dx, target_y + dy
            if 0 <= nx < env.game.grid.width and 0 <= ny < env.game.grid.height:
                neighbor_tile = env.game.grid.tiles[nx][ny]
                if hasattr(neighbor_tile, 'is_walkable') and neighbor_tile.is_walkable:
                    neighbors.append((nx, ny))
        
        # Try each neighbor using A* with blocked tiles
        for neighbor_pos in neighbors:
            goal_node = Node(neighbor_pos[0], neighbor_pos[1])
            alt_path = find_path_avoiding_obstacles(env.game.grid, start_node, goal_node, obstacles)
            
            if alt_path and len(alt_path) > 1:
                return alt_path
    
    # No alternative path found
    return None


def resolve_predictive_collisions(env):
    """Detect and resolve collisions BEFORE they happen.

    This implements the collision detection strategy:
    1. Predict where agents will move next tick
    2. Detect collisions with smart rules:
       - Only stop moving agent if entering stationary agent's tile
       - Stop both agents for mutual collisions or tile swaps
    3. Try to reroute moving agents
    4. If rerouting fails, cancel actions

    Returns:
        dict: Collision statistics including per-agent collision metadata
    """
    if not env.path_processor.is_enabled():
        return {
            'collisions_detected': 0,
            'collisions_rerouted': 0,
            'collisions_failed': 0,
            'agents_with_detected_collisions': [],
            'agents_with_rerouted_collisions': [],
            'agents_with_failed_collisions': []
        }

    # Import cancel function from tick_based_structure to avoid circular import
    from spoiled_broth.rl.tick_based_structure import cancel_agent_action

    # Detect collision events before they happen
    collision_events = detect_predictive_collisions(env)

    collisions_detected = len(collision_events)
    collisions_rerouted = 0
    collisions_failed = 0
    agents_with_failed_collisions = set()
    agents_with_detected_collisions = set()
    agents_with_rerouted_collisions = set()

    def _apply_reroute(agent_id, alt_path):
        """Apply alternative path and keep current progress toward movement in this tick."""
        state = env.agent_state[agent_id]
        state['current_path'] = alt_path[1:]  # Skip current position
        state['path_index'] = 0

    def _pair_has_next_tick_conflict(agent_a_id, agent_b_id):
        """Check if two agents would still overlap or swap this tick."""
        next_positions, movement_status = predict_next_tile_positions(env)
        if agent_a_id not in next_positions or agent_b_id not in next_positions:
            return False

        a_next = next_positions[agent_a_id]
        b_next = next_positions[agent_b_id]
        if a_next == b_next:
            return True

        a_current = (env.agent_map[agent_a_id].slot_x, env.agent_map[agent_a_id].slot_y)
        b_current = (env.agent_map[agent_b_id].slot_x, env.agent_map[agent_b_id].slot_y)
        return (
            movement_status.get(agent_a_id, False)
            and movement_status.get(agent_b_id, False)
            and a_current == b_next
            and b_current == a_next
        )

    # Process each collision event
    for agent_to_stop, other_agent_id, collision_tile, collision_type in collision_events:
        # Track all agents involved in detected collisions, regardless of outcome.
        agents_with_detected_collisions.add(agent_to_stop)
        agents_with_detected_collisions.add(other_agent_id)

        # Get current positions
        agent_to_stop_obj = env.agent_map[agent_to_stop]
        other_agent_obj = env.agent_map[other_agent_id]

        agent_to_stop_current = (agent_to_stop_obj.slot_x, agent_to_stop_obj.slot_y)
        other_agent_current = (other_agent_obj.slot_x, other_agent_obj.slot_y)

        rerouted = False

        if collision_type == 'moving_into_stationary':
            # Only try to reroute the moving agent
            alt_path = find_predictive_alternative_path(
                env, agent_to_stop, collision_tile, other_agent_current, collision_tile
            )

            if alt_path and len(alt_path) > 1:
                _apply_reroute(agent_to_stop, alt_path)
                if not _pair_has_next_tick_conflict(agent_to_stop, other_agent_id):
                    rerouted = True
                    collisions_rerouted += 1
                    agents_with_rerouted_collisions.add(agent_to_stop)
                else:
                    # Tentative reroute still collides this tick
                    cancel_agent_action(env, agent_to_stop)
                    agents_with_failed_collisions.add(agent_to_stop)
                    collisions_failed += 1
            else:
                # Could not reroute moving agent - cancel only the moving agent
                cancel_agent_action(env, agent_to_stop)
                agents_with_failed_collisions.add(agent_to_stop)
                collisions_failed += 1

        elif collision_type in ['mutual_collision', 'tile_swap']:
            # Try rerouting first agent
            alt_path1 = find_predictive_alternative_path(
                env, agent_to_stop, collision_tile, other_agent_current, collision_tile
            )
            if alt_path1 and len(alt_path1) > 1:
                _apply_reroute(agent_to_stop, alt_path1)
                if not _pair_has_next_tick_conflict(agent_to_stop, other_agent_id):
                    rerouted = True
                    collisions_rerouted += 1
                    agents_with_rerouted_collisions.add(agent_to_stop)

            # If first reroute is not valid, try rerouting second agent
            if not rerouted:
                alt_path2 = find_predictive_alternative_path(
                    env, other_agent_id, collision_tile, agent_to_stop_current, collision_tile
                )
                if alt_path2 and len(alt_path2) > 1:
                    _apply_reroute(other_agent_id, alt_path2)
                    if not _pair_has_next_tick_conflict(agent_to_stop, other_agent_id):
                        rerouted = True
                        collisions_rerouted += 1
                        agents_with_rerouted_collisions.add(other_agent_id)

            if not rerouted:
                # Could not find a valid reroute for this tick - cancel both
                cancel_agent_action(env, agent_to_stop)
                cancel_agent_action(env, other_agent_id)
                agents_with_failed_collisions.update([agent_to_stop, other_agent_id])
                collisions_failed += 1

    return {
        'collisions_detected': collisions_detected,
        'collisions_rerouted': collisions_rerouted,
        'collisions_failed': collisions_failed,
        'agents_with_detected_collisions': list(agents_with_detected_collisions),
        'agents_with_rerouted_collisions': list(agents_with_rerouted_collisions),
        'agents_with_failed_collisions': list(agents_with_failed_collisions)
    }
