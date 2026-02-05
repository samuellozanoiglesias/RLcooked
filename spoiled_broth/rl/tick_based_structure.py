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
    return (state['current_action'] is None and 
            state['interaction_timer'] <= 0.0 and 
            len(state['current_path']) == 0)


def assign_action(env, agent_id, action_idx, action_name, tile_index, cached_path, action_type):
    """Assign a new action to an agent for execution."""
    state = env.agent_state[agent_id]
    agent = env.agent_map[agent_id]
    
    # Store action details
    state['current_action'] = action_name
    state['target_tile_index'] = tile_index
    state['action_type'] = action_type
    
    # Setup path for movement
    if cached_path and len(cached_path) > 1:
        # Use cached path from observation computation
        state['current_path'] = cached_path[1:]  # Skip current tile
        state['path_index'] = 0
        state['movement_progress'] = 0.0
    else:
        # Fallback - should rarely happen with proper caching
        state['current_path'] = []
        state['path_index'] = 0
        state['movement_progress'] = 0.0


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
    
    # Add to progress toward next tile
    state['movement_progress'] += movement_distance
    
    # Check if we've reached the next tile
    if state['movement_progress'] >= 1.0:
        # Move to next tile
        state['movement_progress'] = 0.0
        state['path_index'] += 1
        
        # Check if we've reached the final tile
        if state['path_index'] >= len(state['current_path']):
            # Reached destination
            target_node = state['current_path'][-1]
            agent.x = target_node.x * 16 + 8  # Center of tile
            agent.y = target_node.y * 16 + 8
            return True, True, False
        else:
            # Moved to intermediate tile
            current_node = state['current_path'][state['path_index']]
            agent.x = current_node.x * 16 + 8
            agent.y = current_node.y * 16 + 8
            return True, False, False
    
    # Still moving within current tile segment
    return False, False, False


def find_alternative_path_avoiding_collision(env, agent_id, collision_tile, other_agent_pos):
    """Try to find an alternative path that avoids the collision point.
    
    Args:
        env: GameEnv instance
        agent_id: ID of agent to reroute
        collision_tile: Tile coordinates where collision would occur
        other_agent_pos: Current position of other agent
        
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
    
    # Create obstacle set with both collision point and other agent's position
    obstacles = {collision_tile, other_agent_pos}
    
    # Try to find path to target (or walkable neighbor if target is non-walkable)
    start_node = Node(current_pos[0], current_pos[1])
    
    if hasattr(target_tile, 'is_walkable') and target_tile.is_walkable:
        # Target is walkable
        goal_node = Node(target_x, target_y)
        alt_path = find_path(env.game.grid, start_node, goal_node)
        
        # Check if path avoids obstacles
        if alt_path and len(alt_path) > 1:
            path_blocked = any((node.x, node.y) in obstacles for node in alt_path[1:])
            if not path_blocked:
                return alt_path
    else:
        # Target is not walkable - find walkable neighbors
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = target_x + dx, target_y + dy
            if 0 <= nx < env.game.grid.width and 0 <= ny < env.game.grid.height:
                neighbor_tile = env.game.grid.tiles[nx][ny]
                if hasattr(neighbor_tile, 'is_walkable') and neighbor_tile.is_walkable:
                    neighbors.append((nx, ny))
        
        # Try each neighbor
        for neighbor_pos in neighbors:
            goal_node = Node(neighbor_pos[0], neighbor_pos[1])
            alt_path = find_path(env.game.grid, start_node, goal_node)
            
            if alt_path and len(alt_path) > 1:
                path_blocked = any((node.x, node.y) in obstacles for node in alt_path[1:])
                if not path_blocked:
                    return alt_path
    
    return None


def detect_and_resolve_collisions(env):
    """Detect collisions and intelligently resolve them by trying alternative paths.
    
    Strategy:
    1. Detect which agents are on the same tile
    2. For each collision, try to find alternative path for one agent
    3. If rerouting succeeds, update that agent's path
    4. If rerouting fails, cancel both agents' actions
    
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
    
    # Track which tiles agents are attempting to occupy
    tile_occupancy = {}  # (x, y) -> [agent_ids]
    
    for agent_id in env.agents:
        agent = env.agent_map[agent_id]
        current_tile = (agent.slot_x, agent.slot_y)
        
        if current_tile not in tile_occupancy:
            tile_occupancy[current_tile] = []
        tile_occupancy[current_tile].append(agent_id)
    
    # Detect and resolve collisions
    collisions_detected = 0
    collisions_rerouted = 0
    collisions_failed = 0
    agents_with_failed_collisions = []  # Track agents whose collisions couldn't be rerouted
    
    for collision_tile, agents_on_tile in tile_occupancy.items():
        if len(agents_on_tile) > 1:
            collisions_detected += 1
            
            # Try to reroute one of the agents
            rerouted = False
            
            # Try rerouting each agent until one succeeds
            for i, agent_id in enumerate(agents_on_tile):
                # Get the other agent's position (use first other agent if multiple)
                other_agent_id = agents_on_tile[1] if i == 0 else agents_on_tile[0]
                other_agent = env.agent_map[other_agent_id]
                other_pos = (other_agent.slot_x, other_agent.slot_y)
                
                # Try to find alternative path for this agent
                alt_path = find_alternative_path_avoiding_collision(
                    env, agent_id, collision_tile, other_pos
                )
                
                if alt_path and len(alt_path) > 1:
                    # Successfully found alternative path - update agent's path
                    state = env.agent_state[agent_id]
                    state['current_path'] = alt_path[1:]  # Skip current position
                    state['path_index'] = 0
                    state['movement_progress'] = 0.0
                    
                    rerouted = True
                    collisions_rerouted += 1
                    break  # Only need to reroute one agent
            
            if not rerouted:
                # Could not find alternative path - cancel all agents in collision
                for agent_id in agents_on_tile:
                    cancel_agent_action(env, agent_id)
                    agents_with_failed_collisions.append(agent_id)
                
                collisions_failed += 1
    
    return {
        'collisions_detected': collisions_detected,
        'collisions_rerouted': collisions_rerouted,
        'collisions_failed': collisions_failed,
        'agents_with_failed_collisions': agents_with_failed_collisions
    }


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
                    agent_food_type = env.agent_food_type.get(agent_id) if hasattr(env, 'agent_food_type') else None
                    agent_events = complete_agent_action(env, agent_id, agent, action_data, agent_events, agent_food_type)
                
                # Clear agent's action state
                cancel_agent_action(env, agent_id)
    
    return agent_events
