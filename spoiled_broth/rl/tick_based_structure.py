"""
Tick-based simulation helper functions for GameEnv.

This module contains all the helper methods for tick-based simulation including:
- Agent state management
- Movement and pathfinding
- Intelligent collision detection and resolution
- Interaction timing
"""

import logging
from engine.extensions.topDownGridWorld.a_star import Node, euclidean_distance, find_path

logger = logging.getLogger(__name__)

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
    
    Returns:
        int: Path length (number of tiles to traverse), or 0 if no movement needed
    """
    state = env.agent_state[agent_id]
    agent = env.agent_map[agent_id]
    
    # Calculate path length for return (used for adaptive cooperation penalty)
    path_length = len(cached_path) - 1 if cached_path and len(cached_path) > 1 else 0
    
    # Handle special case where tile_index == -1 (BLOCKED)
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
        
        if len(cached_path) > 1:
            state['current_path'] = cached_path[1:]  # Skip current tile
        else:
            # Agent is already at destination, set up for immediate interaction
            state['current_path'] = []
        state['path_index'] = 0
        state['movement_progress'] = 0.0
        # Calculate path length: blocked actions have cached_path with length
        path_length = len(cached_path) - 1 if cached_path and len(cached_path) > 1 else 0
        return path_length
    
    # VALIDATE PATH FIRST - don't assign action until we know it's valid
    if cached_path and len(cached_path) > 1:
        # Valid path with movement required - assign action and set up path
        state['current_action'] = action_name
        state['target_tile_index'] = tile_index
        state['action_type'] = action_type
        state['interaction_target_tile'] = interaction_target_tile
        state['current_path'] = cached_path[1:]  # Skip current tile
        state['path_index'] = 0
        state['movement_progress'] = 0.0
        # Return path length (number of steps to take)
        return len(cached_path) - 1
    elif cached_path and len(cached_path) == 1:
        # Path length is 1 - agent is already adjacent to the non-walkable target
        # This ONLY happens for non-walkable targets (dispenser, cutting board, counter, delivery)
        # where the agent is already at an adjacent tile (the destination for interaction)
        # Note: path_processing returns [start_node] when from_xy is in walkable neighbors
        current_pos = (agent.slot_x, agent.slot_y)
        path_pos = (cached_path[0].x, cached_path[0].y)
        
        if current_pos == path_pos:
            # Agent is at the correct position - set up for immediate interaction
            # IMPORTANT: Must set interaction_timer here since agent won't go through
            # the movement phase in game_env.py (which normally sets the timer)
            from spoiled_broth.rl.reward_analysis import get_cutting_time
            from spoiled_broth.rl.game_env import INTENT_TIME
            
            if action_name == "use_cutting_board":
                interaction_time = get_cutting_time(agent, env.game)
            else:
                interaction_time = INTENT_TIME
            
            state['current_action'] = action_name
            state['target_tile_index'] = tile_index
            state['action_type'] = action_type
            state['interaction_target_tile'] = interaction_target_tile
            state['current_path'] = []  # No movement needed
            state['path_index'] = 0
            state['movement_progress'] = 0.0
            state['interaction_timer'] = interaction_time  # Set timer for immediate interaction
            return 0  # No movement, so path length is 0
        else:
            # Path doesn't start at agent's position - invalid
            logger.debug(f"Path mismatch for {agent_id} action {action_name}: agent at {current_pos}, path starts at {path_pos}")
            return 0
    else:
        # No valid path - check if agent is already at exact target position
        current_pos = (agent.slot_x, agent.slot_y)
        
        grid_w = env.game.grid.width
        target_x = tile_index % grid_w
        target_y = tile_index // grid_w
        
        if current_pos == (target_x, target_y):
            # Agent is already at target - assign action and start interaction immediately
            from spoiled_broth.rl.reward_analysis import get_cutting_time
            from spoiled_broth.rl.game_env import INTENT_TIME
            
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
    
    # Default return: no movement (action rejected or agent already at position)
    return 0


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
            from spoiled_broth.rl.reward_analysis import get_cutting_time
            from spoiled_broth.rl.game_env import INTENT_TIME
            
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
