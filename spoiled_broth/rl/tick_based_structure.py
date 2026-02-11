"""
Tick-based simulation helper functions for GameEnv.

This module contains all the helper methods for tick-based simulation including:
- Agent state management
- Movement and pathfinding
- Intelligent collision detection and resolution
- Interaction timing
"""

from engine.extensions.topDownGridWorld.a_star import Node, euclidean_distance, find_path


def agent_is_idle(env, agent_id):
    """
    Check if agent is ready to accept a new action.
    
    An agent is idle ONLY when:
    1. No current action assigned
    2. No interaction timer running (finished picking up, cutting, etc.)
    3. No movement path (not traveling to a tile)
    
    This ensures agents complete their full action (movement + intent time)
    before being allowed to take a new action.
    """
    state = env.agent_state[agent_id]
    
    is_idle = (state['current_action'] is None and 
               state['interaction_timer'] <= 0.0 and 
               len(state['current_path']) == 0)
    return is_idle


def assign_action(env, agent_id, action_idx, action_name, tile_index, cached_path, action_type, interaction_target_tile=None):
    """Assign a new action to an agent for execution.
    
    Args:
        env: GameEnv instance
        agent_id: ID of agent to assign action to
        action_idx: Index of action
        action_name: Name of action
        tile_index: Index of tile where agent will stand (destination)
        cached_path: Precomputed path to destination
        action_type: Type of action
        interaction_target_tile: (x, y) of tile being interacted with, or None for movement-only actions
    """
    state = env.agent_state[agent_id]
    agent = env.agent_map[agent_id]
    
    # Handle special case where tile_index == -1 (NOT_AVAILABLE)
    # In this case, we use the path that ignores agents and derive the target from the final path node
    if tile_index == -1 and cached_path and len(cached_path) > 0:
        # Get actual target tile from the end of the cached path
        final_node = cached_path[-1]
        grid_w = env.game.grid.width
        actual_tile_index = final_node.y * grid_w + final_node.x
        
        # Assign action with the actual target tile index
        state['current_action'] = action_name
        state['target_tile_index'] = actual_tile_index
        state['action_type'] = action_type
        state['interaction_target_tile'] = interaction_target_tile
        state['current_path'] = cached_path[1:]  # Skip current tile
        state['path_index'] = 0
        state['movement_progress'] = 0.0
        return
    
    # VALIDATE PATH FIRST - don't assign action until we know it's valid
    if cached_path and len(cached_path) > 1:
        # Valid path from observation computation - assign action and set up path
        state['current_action'] = action_name
        state['target_tile_index'] = tile_index
        state['action_type'] = action_type
        state['interaction_target_tile'] = interaction_target_tile
        state['current_path'] = cached_path[1:]  # Skip current tile
        state['path_index'] = 0
        state['movement_progress'] = 0.0
    else:
        # No path provided - check if agent is already at target
        current_pos = (agent.slot_x, agent.slot_y)
        
        grid_w = env.game.grid.width
        target_x = tile_index % grid_w
        target_y = tile_index // grid_w
        
        if current_pos == (target_x, target_y):
            # Agent is already at target - assign action and start interaction immediately
            from spoiled_broth.rl.game_env import INTENT_TIME, get_cutting_time
            
            if action_type == 'use_cutting_board':
                interaction_time = get_cutting_time(agent, env.game)
            else:
                interaction_time = INTENT_TIME
            
            # Now it's safe to assign the action since we know we can execute it
            state['current_action'] = action_name
            state['target_tile_index'] = tile_index
            state['action_type'] = action_type
            state['interaction_target_tile'] = interaction_target_tile
            state['interaction_timer'] = interaction_time
            state['current_path'] = []
            state['path_index'] = 0
            state['movement_progress'] = 0.0
        else:
            # Agent not at target and no path available - reject action
            # Don't assign anything - leave agent in idle state
            pass


def cancel_agent_action(env, agent_id):
    """Cancel agent's current action due to collision or invalidation."""
    state = env.agent_state[agent_id]
    state['current_action'] = None
    state['current_path'] = []
    state['path_index'] = 0
    state['movement_progress'] = 0.0
    state['interaction_timer'] = 0.0
    state['target_tile_index'] = None
    state['action_type'] = None
    state['interaction_target_tile'] = None
    
    # Clear path from collision processor
    if env.path_processor.is_enabled():
        env.path_processor.clear_agent_path(agent_id)


def advance_agent_movement(env, agent_id, tick_duration):
    """Move agent incrementally along their path for one tick.
    
    TIME MANAGEMENT:
    - Movement uses agent.walk_speed (multiplier on base 30 px/s)
    - Base speed: 30 px/s = 1.875 tiles/s (16 pixels per tile)
    - walk_speed=1.0 → 1.875 tiles/s → 0.533s per tile
    - walk_speed=0.5 → 0.9375 tiles/s → 1.067s per tile
    - Each tick advances movement_progress by (speed * tick_duration)
    - When progress ≥ 1.0, agent moves to next tile
    
    Args:
        env: GameEnv instance
        agent_id: ID of agent to move
        tick_duration: Time step duration in seconds (0.2s)
        
    Returns:
        tuple: (reached_next_tile, reached_final_tile, blocked)
    """
    state = env.agent_state[agent_id]
    agent = env.agent_map[agent_id]
    
    # Check if agent has a path to follow
    if not state['current_path'] or state['path_index'] >= len(state['current_path']):
        return False, False, False
    
    # Get agent's walking speed
    # agent.speed is already set to (30.0 * walk_speed) in Agent.__init__
    # walk_speed multiplier: 1.0 = normal, 0.5 = half speed, 2.0 = double speed
    agent_speed_pixels = agent.speed  # pixels/second (already includes walk_speed multiplier)
    agent_speed_tiles = agent_speed_pixels / 16.0  # Convert to tiles/second
    
    # Calculate movement distance this tick
    # Example: speed=30px/s * 0.2s = 6px = 0.375 tiles per tick
    movement_distance = agent_speed_tiles * tick_duration
    
    old_progress = state['movement_progress']
    # Add to progress toward next tile
    state['movement_progress'] += movement_distance
    

    
    # Check if we've reached the next tile
    if state['movement_progress'] >= 1.0:
        # Move to next tile
        state['movement_progress'] = 0.0
        old_index = state['path_index']
        state['path_index'] += 1
        

        
        # Check if we've reached the final tile
        if state['path_index'] >= len(state['current_path']):
            # Reached destination - set interaction timer for action completion
            from spoiled_broth.rl.game_env import INTENT_TIME, get_cutting_time
            
            # Determine interaction time based on action type
            action_type = state.get('action_type')
            if action_type == 'use_cutting_board':
                # Cutting actions use agent's cutting speed
                agent = env.agent_map[agent_id]
                interaction_time = get_cutting_time(agent, env.game)
            else:
                # All other actions use standard intent time (0.2s = 1 tick)
                interaction_time = INTENT_TIME
            
            state['interaction_timer'] = interaction_time
            
            target_node = state['current_path'][-1]
            agent.x = target_node.x * 16 + 8  # Center of tile
            agent.y = target_node.y * 16 + 8

            return True, True, False
        else:
            # Moved to intermediate tile - get the previous node we just reached
            previous_node = state['current_path'][state['path_index'] - 1]
            agent.x = previous_node.x * 16 + 8
            agent.y = previous_node.y * 16 + 8

            return True, False, False
    
    # Still moving within current tile segment
    return False, False, False



def predict_next_tile_positions(env):
    """Predict where each agent will be in the next tick.
    
    Returns:
        tuple: (next_positions, agent_movement_status)
            - next_positions: dict agent_id -> (next_x, next_y) for ALL agents
            - agent_movement_status: dict agent_id -> bool (True if agent is moving, False if stationary)
    """
    next_positions = {}
    agent_movement_status = {}
    
    for agent_id in env.agents:
        state = env.agent_state[agent_id]
        agent = env.agent_map[agent_id]
        
        # Check if agent is moving and will reach next tile
        if (len(state['current_path']) > 0 and 
            state['path_index'] < len(state['current_path'])):
            
            # Get agent's movement speed
            agent_speed_tiles = (agent.speed / 16.0)  # Convert pixels/sec to tiles/sec
            movement_distance = agent_speed_tiles * 0.2  # Distance in next tick (TICK_DURATION = 0.2s)
            
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
            # Copy other attributes A* needs
            self.width = original_grid.width
            self.height = original_grid.height
            self.tile_size = getattr(original_grid, 'tile_size', 16)
            self.tiles = original_grid.tiles
            
        def is_tile_walkable(self, x, y):
            # Check if position is blocked by our temporary obstacles
            if (x, y) in self.blocked_positions:
                return False
            # Check original walkability
            return self.original_grid.tiles[x][y].is_walkable
    
    # Monkey patch the is_obstacle function temporarily
    from engine.extensions.topDownGridWorld.a_star import is_obstacle
    original_is_obstacle = is_obstacle
    
    def custom_is_obstacle(grid_obj, x, y):
        if hasattr(grid_obj, 'is_tile_walkable'):
            return not grid_obj.is_tile_walkable(x, y)
        else:
            return original_is_obstacle(grid_obj, x, y)
    
    # Temporarily replace the function
    import engine.extensions.topDownGridWorld.a_star as astar_module
    astar_module.is_obstacle = custom_is_obstacle
    
    try:
        grid_with_obstacles = GridWithObstacles(grid, blocked_tiles)
        path = find_path(grid_with_obstacles, start, goal)
        return path
    finally:
        # Restore original function
        astar_module.is_obstacle = original_is_obstacle


def find_predictive_alternative_path(env, agent_id, collision_tile, other_agent_current_pos, other_agent_next_pos):
    """Try to find an alternative path that avoids both current and predicted positions of other agent.
    
    For interaction actions, this implements SMART REROUTING:
    1. First, try standard rerouting to the same destination using A* with blocked tiles
    2. If that fails and this is an interaction action, find alternative adjacent positions 
       that can interact with the same target tile
    3. Route to the closest alternative interaction position
    
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
    action_name = state.get('current_action', 'unknown')
    
    if target_tile_index is None:
        return None
    
    # Calculate target position
    grid_w = env.game.grid.width
    target_x = target_tile_index % grid_w
    target_y = target_tile_index // grid_w
    target_tile = env.game.grid.tiles[target_x][target_y]
    
    # SMARTER OBSTACLE LOGIC:
    # Only block the other agent's NEXT position (where they're moving to)
    # Their CURRENT position will become free when they move away
    # Exception: If this is a tile-swapping collision, we need different logic
    obstacles = {other_agent_next_pos}
    
    # Check if this is a tile swapping scenario
    current_pos_tuple = (current_pos[0], current_pos[1])
    if (current_pos_tuple == other_agent_next_pos and 
        other_agent_current_pos == (target_x, target_y)):
        # This is tile swapping: we're going where the other agent is, and they're coming here
        
        # For tile swapping, block both positions since it's an inherent conflict
        obstacles = {other_agent_current_pos, other_agent_next_pos}
    else:
        # Normal collision: only block where the other agent is going
        obstacles = {other_agent_next_pos}
    
    start_node = Node(current_pos[0], current_pos[1])
    
    # PHASE 1: Try standard rerouting to the same destination using A* with blocked obstacles
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
        for i, neighbor_pos in enumerate(neighbors):
            goal_node = Node(neighbor_pos[0], neighbor_pos[1])
            alt_path = find_path_avoiding_obstacles(env.game.grid, start_node, goal_node, obstacles)
            
            if alt_path and len(alt_path) > 1:
                return alt_path
    
    # PHASE 2: SMART INTERACTION REROUTING
    # If standard rerouting failed and this is an interaction action,
    # find alternative positions that can interact with the same target tile
    
    # Get the stored interaction target tile directly from agent state
    interaction_target_tile = state.get('interaction_target_tile')
    
    if interaction_target_tile is not None:
        # Find all walkable positions adjacent to the interaction target
        itx, ity = interaction_target_tile
        alternative_positions = []
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = itx + dx, ity + dy
            
            # Bounds check
            if not (0 <= nx < env.game.grid.width and 0 <= ny < env.game.grid.height):
                continue
            
            # Skip original destination
            if (nx, ny) == (target_x, target_y):
                continue
                
            neighbor_tile = env.game.grid.tiles[nx][ny]
            if hasattr(neighbor_tile, 'is_walkable') and neighbor_tile.is_walkable:
                # Check if this position is not blocked by obstacles
                if (nx, ny) not in obstacles:
                    alternative_positions.append((nx, ny))
        
        if not alternative_positions:
            return None
        
        # Try each alternative position, starting with the closest to agent's current position
        alternative_positions.sort(key=lambda pos: (pos[0] - current_pos[0])**2 + (pos[1] - current_pos[1])**2)
        
        for i, (alt_x, alt_y) in enumerate(alternative_positions):
            goal_node = Node(alt_x, alt_y)
            alt_path = find_path_avoiding_obstacles(env.game.grid, start_node, goal_node, obstacles)
            
            if alt_path and len(alt_path) > 1:
                
                # Update the agent's target_tile_index to the new destination
                new_target_index = alt_y * grid_w + alt_x
                old_target_index = state['target_tile_index']
                state['target_tile_index'] = new_target_index
                
                return alt_path
        
    else:
        pass
    
    return None

def is_tile_matching_action(tile, action_name):
    """Check if a tile matches the type that the action is trying to interact with.
    
    Args:
        tile: The tile object to check
        action_name: The action being performed
        
    Returns:
        bool: True if this tile is what the action wants to interact with
    """
    if not hasattr(tile, '_type'):
        return False
    
    tile_type = tile._type
    tile_item = getattr(tile, 'item', None)
    
    # Dispenser actions (type 3)
    if "from_dispenser" in action_name:
        if tile_type != 3:
            return False
        
        # Check specific dispenser type
        if "pick_up_tomato_from_dispenser" in action_name:
            result = tile_item == 'tomato'
            return result
        elif "pick_up_pumpkin_from_dispenser" in action_name:
            result = tile_item == 'pumpkin'
            return result
        elif "pick_up_plate_from_dispenser" in action_name:
            result = tile_item == 'plate'
            return result
        return True
    
    # Cutting board actions (type 4)
    elif "use_cutting_board" in action_name:
        result = tile_type == 4
        return result
    
    # Delivery actions (type 5)
    elif "use_delivery" in action_name:
        result = tile_type == 5
        return result
    
    # Counter actions (type 2)
    elif ("from_counter" in action_name or "on_free_counter" in action_name):
        result = tile_type == 2
        return result
    
    return False


def resolve_predictive_collisions(env):
    """Detect and resolve collisions BEFORE they happen.
    
    This implements the new collision detection strategy:
    1. Predict where agents will move next tick
    2. Detect collisions with smart rules:
       - Only stop moving agent if entering stationary agent's tile
       - Stop both agents for mutual collisions or tile swaps
    3. Try to reroute moving agents
    4. If rerouting fails, cancel actions
    
    Returns:
        dict: Collision statistics including agents_with_failed_collisions
    """
    if not env.path_processor.is_enabled():
        return {
            'collisions_detected': 0, 
            'collisions_rerouted': 0, 
            'collisions_failed': 0,
            'agents_with_failed_collisions': []
        }
    
    # Detect collision events before they happen
    collision_events = detect_predictive_collisions(env)
    
    collisions_detected = len(collision_events)
    collisions_rerouted = 0
    collisions_failed = 0
    agents_with_failed_collisions = []
    
    # Process each collision event
    for event_idx, (agent_to_stop, other_agent_id, collision_tile, collision_type) in enumerate(collision_events):
        

        # Get current and predicted positions
        agent_to_stop_obj = env.agent_map[agent_to_stop]
        other_agent_obj = env.agent_map[other_agent_id]
        
        agent_to_stop_current = (agent_to_stop_obj.slot_x, agent_to_stop_obj.slot_y)
        other_agent_current = (other_agent_obj.slot_x, other_agent_obj.slot_y)
        
        # Get agent states for additional debug info
        state_to_stop = env.agent_state[agent_to_stop]
        state_other = env.agent_state[other_agent_id]
        
        rerouted = False
        
        if collision_type == 'moving_into_stationary':
            # Only try to reroute the moving agent
            alt_path = find_predictive_alternative_path(
                env, agent_to_stop, collision_tile, other_agent_current, collision_tile  # Other agent stays put
            )
            
            if alt_path and len(alt_path) > 1:
                # Successfully rerouted the moving agent
                state_to_stop['current_path'] = alt_path[1:]  # Skip current position
                state_to_stop['path_index'] = 0
                state_to_stop['movement_progress'] = 0.0
                rerouted = True
                collisions_rerouted += 1
            else:
                # Could not reroute moving agent - cancel only the moving agent
                cancel_agent_action(env, agent_to_stop)
                agents_with_failed_collisions.append(agent_to_stop)
                collisions_failed += 1
                
        elif collision_type in ['mutual_collision', 'tile_swap']:
            # Try to reroute either agent (mutual collision or tile swap)
            alt_path1 = find_predictive_alternative_path(
                env, agent_to_stop, collision_tile, other_agent_current, collision_tile
            )
            
            if alt_path1 and len(alt_path1) > 1:
                # Successfully rerouted first agent
                state_to_stop['current_path'] = alt_path1[1:]  # Skip current position
                state_to_stop['path_index'] = 0
                state_to_stop['movement_progress'] = 0.0
                rerouted = True
                collisions_rerouted += 1
            else:
                # Try rerouting the other agent
                alt_path2 = find_predictive_alternative_path(
                    env, other_agent_id, collision_tile, agent_to_stop_current, collision_tile
                )
                
                if alt_path2 and len(alt_path2) > 1:
                    # Successfully rerouted second agent
                    state_other['current_path'] = alt_path2[1:]  # Skip current position
                    state_other['path_index'] = 0
                    state_other['movement_progress'] = 0.0
                    rerouted = True
                    collisions_rerouted += 1
                else:
                    # Could not find alternative path for either agent - cancel both
                    cancel_agent_action(env, agent_to_stop)
                    cancel_agent_action(env, other_agent_id)
                    agents_with_failed_collisions.extend([agent_to_stop, other_agent_id])
                    collisions_failed += 1
        
    result = {
        'collisions_detected': collisions_detected,
        'collisions_rerouted': collisions_rerouted,
        'collisions_failed': collisions_failed,
        'agents_with_failed_collisions': agents_with_failed_collisions
    }
    
    return result


def update_agent_interactions(env, agent_events, agent_penalties, tick_duration):
    """Update interaction timers and complete interactions when ready.
    
    TIME MANAGEMENT - INTERACTION TIMERS:
    After agent reaches destination tile, they must wait for intent time:
    
    1. CUTTING: Uses agent.cut_speed (multiplier on base 3.0s cutting time)
       - Base time: 3.0 seconds (15 ticks)
       - cut_speed=1.0 → 3.0s cutting time (15 ticks)
       - cut_speed=0.5 → 6.0s cutting time (30 ticks, slower)
       - cut_speed=2.0 → 1.5s cutting time (7.5 ticks, faster)
       - Formula: base_cutting_time / cut_speed
    
    2. OTHER ACTIONS: Use fixed INTENT_TIME (0.2s = 1 tick)
       - Picking up items: 0.2s (1 tick)
       - Putting down items: 0.2s (1 tick)
       - Delivering: 0.2s (1 tick)
       - Minimum time = TICK_DURATION (0.2s)
    
    Timer decrements each tick by tick_duration (0.2s).
    When timer reaches 0, action completes and agent becomes idle.
    
    Action completion updates:
    - Counters: Items added/removed from counter tiles
    - Hands: Agent's held item updated
    - Events: Delivery, salad creation, etc. recorded
    
    Args:
        env: GameEnv instance
        agent_events: Dictionary to accumulate agent events
        agent_penalties: Dictionary to accumulate penalties
        tick_duration: Time step duration in seconds (0.2s)
        
    Returns:
        agent_events: Updated events dictionary
    """
    for agent_id in env.agents:
        state = env.agent_state[agent_id]
        
        # Decrease interaction timer (counts down from intent time to 0)
        if state['interaction_timer'] > 0:
            old_timer = state['interaction_timer']
            state['interaction_timer'] -= tick_duration
            
            # Check if interaction just completed
            if state['interaction_timer'] <= 0:
                state['interaction_timer'] = 0.0
                
                # Complete the interaction
                agent = env.agent_map[agent_id]
                tile_index = state['target_tile_index']
                
                if tile_index is not None:
                    grid_w = env.game.grid.width
                    x = tile_index % grid_w
                    y = tile_index // grid_w
                    tile = env.game.grid.tiles[x][y]
                    
                    # Import and use the completion logic from game_step
                    from spoiled_broth.rl.game_step import complete_agent_action
                    
                    action_data = {'tile_index': tile_index}
                    agent_food_type = env.agent_food_type.get(agent_id) if hasattr(env, 'agent_food_type') and env.agent_food_type is not None else None
                    agent_events = complete_agent_action(env, agent_id, agent, action_data, agent_events, agent_food_type)
                    
                # Clear agent's action state
                cancel_agent_action(env, agent_id)
    
    return agent_events
