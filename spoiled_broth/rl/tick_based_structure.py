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


def assign_action(env, agent_id, action_idx, action_name, tile_index, cached_path, action_type):
    """Assign a new action to an agent for execution."""
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
            # Moved to intermediate tile
            current_node = state['current_path'][state['path_index']]
            agent.x = current_node.x * 16 + 8
            agent.y = current_node.y * 16 + 8
            return True, False, False
    
    # Still moving within current tile segment
    return False, False, False



def predict_next_tile_positions(env):
    """Predict where each moving agent will be in the next tick.
    
    Returns:
        dict: agent_id -> (next_x, next_y) for agents that will move next tick
    """
    next_positions = {}
    
    for agent_id in env.agents:
        state = env.agent_state[agent_id]
        agent = env.agent_map[agent_id]
        
        # Skip agents that aren't moving
        if len(state['current_path']) == 0 or state['path_index'] >= len(state['current_path']):
            continue
        
        # Get agent's movement speed
        agent_speed_tiles = (agent.speed / 16.0)  # Convert pixels/sec to tiles/sec
        movement_distance = agent_speed_tiles * 0.2  # Distance in next tick (TICK_DURATION = 0.2s)
        
        # Check if agent will reach next tile in this tick
        new_progress = state['movement_progress'] + movement_distance
        
        if new_progress >= 1.0:
            # Agent will move to next tile
            next_node = state['current_path'][state['path_index']]
            next_positions[agent_id] = (next_node.x, next_node.y)
        # If agent won't reach next tile, they stay at current position (no entry in dict)
    
    return next_positions


def detect_predictive_collisions(env):
    """Detect collisions BEFORE they happen by predicting next tick positions.
    
    Returns:
        list: [(agent1_id, agent2_id, collision_tile), ...] - detected collision pairs
    """
    if not env.path_processor.is_enabled():
        return []
    
    next_positions = predict_next_tile_positions(env)
    
    # Group agents by their next tile positions
    tile_occupancy = {}  # (x, y) -> [agent_ids]
    for agent_id, (x, y) in next_positions.items():
        tile = (x, y)
        if tile not in tile_occupancy:
            tile_occupancy[tile] = []
        tile_occupancy[tile].append(agent_id)
    
    # Find collision pairs
    collision_pairs = []
    for collision_tile, agents_on_tile in tile_occupancy.items():
        if len(agents_on_tile) > 1:
            # Create all pairs of colliding agents
            for i in range(len(agents_on_tile)):
                for j in range(i + 1, len(agents_on_tile)):
                    collision_pairs.append((agents_on_tile[i], agents_on_tile[j], collision_tile))
    
    return collision_pairs


def find_predictive_alternative_path(env, agent_id, collision_tile, other_agent_current_pos, other_agent_next_pos):
    """Try to find an alternative path that avoids both current and predicted positions of other agent.
    
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
    
    # Create obstacle set with:
    # 1. The collision point (where both agents would go)
    # 2. Other agent's current position
    # 3. Other agent's predicted next position
    obstacles = {collision_tile, other_agent_current_pos, other_agent_next_pos}
    
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


def resolve_predictive_collisions(env):
    """Detect and resolve collisions BEFORE they happen.
    
    This implements the new collision detection strategy:
    1. Predict where agents will move next tick
    2. Detect if any will occupy the same tile
    3. Try to reroute one agent avoiding both current and predicted positions
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
    
    # Detect collision pairs before they happen
    collision_pairs = detect_predictive_collisions(env)
    
    collisions_detected = len(collision_pairs)
    collisions_rerouted = 0
    collisions_failed = 0
    agents_with_failed_collisions = []
    
    # Process each collision pair
    for agent1_id, agent2_id, collision_tile in collision_pairs:
        
        # Get current and predicted positions
        agent1 = env.agent_map[agent1_id]
        agent2 = env.agent_map[agent2_id]
        
        agent1_current = (agent1.slot_x, agent1.slot_y)
        agent2_current = (agent2.slot_x, agent2.slot_y)
        agent1_next = collision_tile
        agent2_next = collision_tile
        
        rerouted = False
        
        # Try rerouting agent1 first
        alt_path1 = find_predictive_alternative_path(
            env, agent1_id, collision_tile, agent2_current, agent2_next
        )
        
        if alt_path1 and len(alt_path1) > 1:
            # Successfully rerouted agent1
            state1 = env.agent_state[agent1_id]
            state1['current_path'] = alt_path1[1:]  # Skip current position
            state1['path_index'] = 0
            state1['movement_progress'] = 0.0
            rerouted = True
            collisions_rerouted += 1
        else:
            # Try rerouting agent2
            alt_path2 = find_predictive_alternative_path(
                env, agent2_id, collision_tile, agent1_current, agent1_next
            )
            
            if alt_path2 and len(alt_path2) > 1:
                # Successfully rerouted agent2
                state2 = env.agent_state[agent2_id]
                state2['current_path'] = alt_path2[1:]  # Skip current position
                state2['path_index'] = 0
                state2['movement_progress'] = 0.0
                rerouted = True
                collisions_rerouted += 1
        
        if not rerouted:
            # Could not find alternative path for either agent - cancel both
            cancel_agent_action(env, agent1_id)
            cancel_agent_action(env, agent2_id)
            agents_with_failed_collisions.extend([agent1_id, agent2_id])
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
