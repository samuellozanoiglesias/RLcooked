import numpy as np
import math
import os
import pickle

def get_tile_indices_by_type(game, tile_type):
    """
    Returns a list of (idx, x, y) for tiles of the given type.
    tile_type: str, one of ['tomato_dispenser', 'plate_dispenser', 'cutting_board', 'delivery', 'counter']
    """
    grid = game.grid
    indices = []
    for idx in game.clickable_indices:
        x = idx % grid.width
        y = idx // grid.width
        tile = grid.tiles[x][y]
        t = getattr(tile, '_type', None)
        item = getattr(tile, 'item', None)
        if tile_type == 'tomato_dispenser' and t == 3 and item == 'tomato':
            indices.append((idx, x, y))
        elif tile_type == 'plate_dispenser' and t == 3 and item == 'plate':
            indices.append((idx, x, y))
        elif tile_type == 'cutting_board' and t == 4:
            indices.append((idx, x, y))
        elif tile_type == 'delivery' and t == 5:
            indices.append((idx, x, y))
        elif tile_type == 'counter' and t == 2:
            indices.append((idx, x, y))
    return indices

def get_item_indices_on_counters(game, item_name):
    """
    Returns a list of (idx, x, y) for counters with the given item_name.
    """
    grid = game.grid
    indices = []
    for idx in game.clickable_indices:
        x = idx % grid.width
        y = idx // grid.width
        tile = grid.tiles[x][y]
        t = getattr(tile, '_type', None)
        item = getattr(tile, 'item', None)
        if t == 2 and item == item_name:
            indices.append((idx, x, y))
    return indices

def get_distance_and_path(path_processor, from_xy, to_xy, agent_id=None, game=None, current_time=0.0, agent_speed=1.875, force_ignore_agents=False):
    """
    Get shortest path distance between two positions using pathfinding.
    Returns path length if path exists, None if no path exists.
    
    Args:
        path_processor: PathProcessor instance for pathfinding calculations
        from_xy: Starting position (x, y)
        to_xy: Target position (x, y)
        agent_id: ID of the requesting agent (for collision detection)
        game: Game instance (needed for grid access)
        current_time: Current game time (for collision detection)
        agent_speed: Agent's walking speed in tiles/second (default 1.875 = 30/16)
        force_ignore_agents: If True, ignores other agents even if collision detection is enabled
                            Used for calculating ACCESSIBILITY (path exists ignoring agents)
    
    Returns:
        tuple: (distance, path) where distance is float or None, path is list of nodes or None
    """    
    if path_processor is None:
        # Fallback to Euclidean distance if no path processor
        dist = ((from_xy[0] - to_xy[0]) ** 2 + (from_xy[1] - to_xy[1]) ** 2) ** 0.5
        return dist, None
    
    if game is None:
        # Fallback to Euclidean distance if no game grid available
        dist = ((from_xy[0] - to_xy[0]) ** 2 + (from_xy[1] - to_xy[1]) ** 2) ** 0.5
        return dist, None
    
    # Temporarily override collision detection if force_ignore_agents=True
    # This allows us to calculate ACCESSIBILITY (ignoring agents) separately from AVAILABILITY (considering agents)
    original_collision_enabled = path_processor.collision_enabled
    if force_ignore_agents:
        path_processor.collision_enabled = False
    
    try:
        # Use path processor to calculate shortest path distance
        distance, path = path_processor.get_shortest_path_distance(
            game.grid, from_xy, to_xy, agent_id, current_time, agent_speed, game
        )
        return distance, path
    finally:
        # Restore original collision detection setting
        if force_ignore_agents:
            path_processor.collision_enabled = original_collision_enabled

# ---- Classic mode without ownership awareness ---- #
def game_to_obs_vector_classic(game, agent_id, path_processor=None):
    """
    Returns a vector observation for agent_id:
    For each tile type, includes:
      - distance to closest
      - distance to midpoint (between agents)
    Also includes one-hot agent inventory and other agent inventory.
    """
    normalization_factor = game.normalization_factor

    tile_types = ['tomato_dispenser', 'plate_dispenser', 'cutting_board', 'delivery']
    item_names = [None, 'tomato', 'plate', 'tomato_cut', 'tomato_salad']

    # Get agent walking and cutting speeds
    agent_walking_speed = game.walking_speeds.get(agent_id, 1) * game.walked_tiles_per_second
    # Cutting speed is inversely proportional: lower speed = more time
    # If cutting_speed = 0.3, then cutting takes 3/0.3 = 10 seconds instead of 3 seconds
    cutting_speed_multiplier = game.cutting_speeds.get(agent_id, 1)
    agent_cutting_speed = game.cutting_time / cutting_speed_multiplier if cutting_speed_multiplier > 0 else game.cutting_time

    # Get agent positions
    all_agent_ids = [aid for aid in game.gameObjects if aid.startswith('ai_rl_')]
    if agent_id not in all_agent_ids:
        raise ValueError(f"agent_id {agent_id} not found in gameObjects")
    
    # Handle single agent case
    other_agent_ids = [aid for aid in all_agent_ids if aid != agent_id]
    has_other_agent = len(other_agent_ids) > 0
    
    agent = game.gameObjects[agent_id]
    agent_pos = (agent.slot_x, agent.slot_y)
    
    if has_other_agent:
        other_agent_id = other_agent_ids[0]
        other_agent = game.gameObjects[other_agent_id]
        other_pos = (other_agent.slot_x, other_agent.slot_y)
        # Midpoint position (rounded to nearest int)
        midpoint = (int(round((agent.slot_x + other_agent.slot_x) / 2)), int(round((agent.slot_y + other_agent.slot_y) / 2)))
    else:
        other_agent = None
        other_pos = None  # No other agent
        # For single agent, midpoint is just the agent position
        midpoint = (agent.slot_x, agent.slot_y)
    obs_vector = []
    considered_paths = []
    considered_tiles = []
    
    # Add placeholder for do_nothing action (action index 0)
    # do_nothing doesn't require a tile or path
    considered_paths.append(None)
    considered_tiles.append(-2)  # Use -2 to indicate do_nothing
    

    # --- Add times to tile types ---
    # For each tile type, we calculate THREE indicators:
    # 1. ACCESSIBILITY: Can we reach this tile ignoring other agents? (0 or 1)
    # 2. AVAILABILITY: Can we reach this tile considering other agents' current positions? (0 or 1)
    # 3. TIME: Normalized time to reach the tile
    for tile_type in tile_types:
        if tile_type == 'cutting_board':
            action_time = agent_cutting_speed 
        else:
            action_time = 0

        indices = get_tile_indices_by_type(game, tile_type)

        # STEP 1: Calculate ACCESSIBILITY (ignoring other agents)
        # Find the closest tile of this type that has a valid path (ignoring agents)
        accessible_agent = [(_idx, x, y) for (_idx, x, y) in indices 
                           if get_distance_and_path(path_processor, agent_pos, (x, y), agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True)[0] is not None]
        
        min_dist_no_agents = None
        best_tile_idx = None
        best_tile_xy = None
        
        for _idx, x, y in accessible_agent:
            d, _ = get_distance_and_path(path_processor, agent_pos, (x, y), agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True)
            if d is not None and d >= 0:
                if min_dist_no_agents is None or d < min_dist_no_agents:
                    min_dist_no_agents = d
                    best_tile_idx = _idx
                    best_tile_xy = (x, y)
        
        # STEP 2: If accessible, calculate AVAILABILITY (considering other agents)
        # Only relevant if collision detection is enabled
        if best_tile_xy is not None:
            # Tile is accessible (path exists ignoring agents)
            accessibility = 1.0
            
            # Now check availability (considering agents)
            if path_processor.collision_enabled:
                # Calculate path considering other agents' positions
                dist_with_agents, path_with_agents = get_distance_and_path(
                    path_processor, agent_pos, best_tile_xy, agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=False
                )
                
                if dist_with_agents is not None and dist_with_agents >= 0:
                    # Path is available (not blocked by other agents)
                    availability = 1.0
                    final_dist = dist_with_agents
                    final_path = path_with_agents
                else:
                    # Path is blocked by other agents (tile is accessible but not currently available)
                    availability = 0.0
                    # Mark tile_index as -1 to indicate "not available" in game_env
                    best_tile_idx = -1
                    final_dist = min_dist_no_agents  # Use distance ignoring agents for observation
                    # But we DO provide the path ignoring agents for action execution
                    _, final_path = get_distance_and_path(
                        path_processor, agent_pos, best_tile_xy, agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True
                    )
            else:
                # Collision detection disabled: availability is always 1
                availability = 1.0
                final_dist = min_dist_no_agents
                _, final_path = get_distance_and_path(
                    path_processor, agent_pos, best_tile_xy, agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True
                )
            
            # Calculate time and add to observation
            time_to_tile = (final_dist / agent_walking_speed + action_time) / normalization_factor
            considered_paths.append(final_path)
            considered_tiles.append(best_tile_idx)
            obs_vector.append(accessibility)  # 1.0 = tile is accessible
            obs_vector.append(availability)   # 1.0 = tile is available, 0.0 = blocked by agents
            obs_vector.append(time_to_tile)
        else:
            # No accessible tile found (no path exists even ignoring agents)
            considered_paths.append(None)
            considered_tiles.append(None)
            obs_vector.append(0.0)  # Accessibility: 0 = inaccessible
            obs_vector.append(0.0)  # Availability: 0 = not available (because inaccessible)
            obs_vector.append(1.0)  # Max time (normalized)


    # --- Add times to items on counters ---
    # For each item type on counters, we calculate for the CLOSEST counter:
    # 1. PRESENCE: Does this item exist on any counter? (0 or 1)
    # 2. ACCESSIBILITY: Can we reach the closest counter with this item ignoring other agents? (0 or 1)
    # 3. AVAILABILITY: Can we reach it considering other agents? (0 or 1)
    # 4. TIME: Normalized time to reach the closest counter
    # Then for the MIDPOINT counter (closest to midpoint between agents):
    # 5. ACCESSIBILITY: Can we reach this specific counter ignoring other agents? (0 or 1)
    # 6. AVAILABILITY: Can we reach it considering other agents? (0 or 1)
    # 7. TIME: Normalized time to reach the midpoint counter
    for item_name in item_names:
        action_time = 0
        indices = get_item_indices_on_counters(game, item_name)

        if len(indices) > 0:
            obs_vector.append(1) # Presence: There is at least one counter with this item
            
            # STEP 1: Find CLOSEST counter with this item (ignoring agents)
            closest_accessible = [(_idx, x, y) for (_idx, x, y) in indices 
                                 if get_distance_and_path(path_processor, agent_pos, (x, y), agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True)[0] is not None]
            
            min_dist_no_agents = None
            best_closest_idx = None
            best_closest_xy = None
            
            for _idx, x, y in closest_accessible:
                d, _ = get_distance_and_path(path_processor, agent_pos, (x, y), agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True)
                if d is not None and d >= 0:
                    if min_dist_no_agents is None or d < min_dist_no_agents:
                        min_dist_no_agents = d
                        best_closest_idx = _idx
                        best_closest_xy = (x, y)
            
            # STEP 2: Check availability of closest counter
            if best_closest_xy is not None:
                # Closest counter is accessible
                accessibility_closest = 1.0
                
                if path_processor.collision_enabled:
                    # Check if path is available (not blocked by agents)
                    dist_with_agents, path_with_agents = get_distance_and_path(
                        path_processor, agent_pos, best_closest_xy, agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=False
                    )
                    
                    if dist_with_agents is not None and dist_with_agents >= 0:
                        availability_closest = 1.0
                        final_dist_closest = dist_with_agents
                        final_path_closest = path_with_agents
                    else:
                        # Path blocked by agents
                        availability_closest = 0.0
                        best_closest_idx = -1  # Mark as not available
                        final_dist_closest = min_dist_no_agents
                        # But provide the path ignoring agents for action execution
                        _, final_path_closest = get_distance_and_path(
                            path_processor, agent_pos, best_closest_xy, agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True
                        )
                else:
                    # Collision detection disabled
                    availability_closest = 1.0
                    final_dist_closest = min_dist_no_agents
                    _, final_path_closest = get_distance_and_path(
                        path_processor, agent_pos, best_closest_xy, agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True
                    )
                
                time_to_closest = (final_dist_closest / agent_walking_speed + action_time) / normalization_factor
                considered_paths.append(final_path_closest)
                considered_tiles.append(best_closest_idx)
                obs_vector.append(accessibility_closest)  # 1.0 = accessible
                obs_vector.append(availability_closest)   # 1.0 = available, 0.0 = blocked by agents
                obs_vector.append(time_to_closest)
            else:
                # No accessible counter found
                considered_paths.append(None)
                considered_tiles.append(None)
                obs_vector.append(0.0)  # Accessibility: not accessible
                obs_vector.append(0.0)  # Availability: not available
                obs_vector.append(1.0)  # Max time

            # STEP 3: Find counter closest to MIDPOINT (for coordination)
            # Choose the counter tile that is closest to the midpoint by Euclidean distance
            best = min(indices, key=lambda it: math.hypot(it[1] - midpoint[0], it[2] - midpoint[1]))
            _idx, bx, by = best
            
            # Check accessibility of midpoint counter (ignoring agents)
            dist_midpoint_no_agents, _ = get_distance_and_path(
                path_processor, agent_pos, (bx, by), agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True
            )
            
            if dist_midpoint_no_agents is not None and dist_midpoint_no_agents >= 0:
                # Midpoint counter is accessible
                accessibility_midpoint = 1.0
                
                if path_processor.collision_enabled:
                    # Check if available (not blocked by agents)
                    dist_with_agents, path_with_agents = get_distance_and_path(
                        path_processor, agent_pos, (bx, by), agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=False
                    )
                    
                    if dist_with_agents is not None and dist_with_agents >= 0:
                        availability_midpoint = 1.0
                        final_dist_mid = dist_with_agents
                        final_path_mid = path_with_agents
                        final_idx_mid = _idx
                    else:
                        # Blocked by agents
                        availability_midpoint = 0.0
                        final_dist_mid = dist_midpoint_no_agents
                        final_path_mid = None
                        final_idx_mid = -1
                else:
                    # Collision detection disabled
                    availability_midpoint = 1.0
                    final_dist_mid = dist_midpoint_no_agents
                    _, final_path_mid = get_distance_and_path(
                        path_processor, agent_pos, (bx, by), agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True
                    )
                    final_idx_mid = _idx
                
                time_to_midpoint = (final_dist_mid / agent_walking_speed + action_time) / normalization_factor
                considered_tiles.append(final_idx_mid)
                considered_paths.append(final_path_mid)
                obs_vector.append(accessibility_midpoint)  # 1.0 = accessible
                obs_vector.append(availability_midpoint)   # 1.0 = available, 0.0 = blocked
                obs_vector.append(time_to_midpoint)
            else:
                # Midpoint counter not accessible
                considered_tiles.append(None)
                considered_paths.append(None)
                obs_vector.append(0.0)  # Accessibility: not accessible
                obs_vector.append(0.0)  # Availability: not available
                obs_vector.append(1.0)  # Max time
        else:
            # No counters with this item exist
            obs_vector.append(0) # Presence: no tiles with this item
            # Closest counter values
            obs_vector.append(0.0) # Accessibility: no tiles exist
            obs_vector.append(0.0) # Availability: no tiles exist
            obs_vector.append(1.0) # Time: max
            # Midpoint counter values
            obs_vector.append(0.0) # Accessibility: no tiles exist
            obs_vector.append(0.0) # Availability: no tiles exist
            obs_vector.append(1.0) # Time: max
            considered_paths.append(None)
            considered_paths.append(None)
            considered_tiles.append(None)
            considered_tiles.append(None)

    # Distance to other agent - both Euclidean and pathfinding distances
    # For single agent case, use default values
    if has_other_agent:
        # 1. Euclidean distance (straight-line distance)
        euclidean_dist = math.sqrt((agent_pos[0] - other_pos[0])**2 + (agent_pos[1] - other_pos[1])**2)
        euclidean_time = (euclidean_dist / agent_walking_speed) / normalization_factor
        obs_vector.append(euclidean_time)
        
        # 2. Pathfinding distance (using path processor for accessibility)
        pathfinding_dist, path = get_distance_and_path(path_processor, agent_pos, other_pos, agent_id, game, 0.0, agent_walking_speed)
        pathfinding_time = (pathfinding_dist / agent_walking_speed) / normalization_factor if pathfinding_dist is not None else 1
        obs_vector.append(pathfinding_time)
    else:
        # Single agent case: append default values for other agent distances
        obs_vector.append(1.0)  # No other agent, use max distance
        obs_vector.append(1.0)  # No other agent, use max distance

    # One-hot agent inventory
    agent_inventory = np.zeros(len(item_names), dtype=np.float32)
    item = getattr(agent, 'item', None)
    if item in item_names:
        agent_inventory[item_names.index(item)] = 1.0
    obs_vector.extend(agent_inventory.tolist())

    # One-hot other agent inventory
    if has_other_agent:
        other_inventory = np.zeros(len(item_names), dtype=np.float32)
        item_other = getattr(other_agent, 'item', None)
        if item_other in item_names:
            other_inventory[item_names.index(item_other)] = 1.0
        obs_vector.extend(other_inventory.tolist())
    else:
        # Single agent case: append zeros for other agent inventory
        other_inventory = np.zeros(len(item_names), dtype=np.float32)
        obs_vector.extend(other_inventory.tolist())

    return np.array(obs_vector, dtype=np.float32), considered_paths, considered_tiles

# ---- Competition mode with ownership awareness ---- #
def game_to_obs_vector_competition(game, agent_id, path_processor=None):
    """
    Returns a vector observation for agent_id:
    For each tile type, includes:
      - distance to closest
      - distance to midpoint (between agents)
    Also includes one-hot agent inventory and other agent inventory.
    """
    normalization_factor = game.normalization_factor

    tile_types = ['tomato_dispenser', 'pumpkin_dispenser', 'plate_dispenser', 'cutting_board', 'delivery']
    item_names = [None, 'tomato', 'pumpkin', 'plate', 'tomato_cut', 'pumpkin_cut', 'tomato_salad', 'pumpkin_salad']

    # Get agent walking and cutting speeds
    agent_walking_speed = game.walking_speeds.get(agent_id, 1) * game.walked_tiles_per_second
    # Cutting speed is inversely proportional: lower speed = more time
    # If cutting_speed = 0.3, then cutting takes 3/0.3 = 10 seconds instead of 3 seconds
    cutting_speed_multiplier = game.cutting_speeds.get(agent_id, 1)
    agent_cutting_speed = game.cutting_time / cutting_speed_multiplier if cutting_speed_multiplier > 0 else game.cutting_time

    # Get agent positions
    all_agent_ids = [aid for aid in game.gameObjects if aid.startswith('ai_rl_')]
    if agent_id not in all_agent_ids:
        raise ValueError(f"agent_id {agent_id} not found in gameObjects")
    
    # Handle single agent case
    other_agent_ids = [aid for aid in all_agent_ids if aid != agent_id]
    has_other_agent = len(other_agent_ids) > 0
    
    agent = game.gameObjects[agent_id]
    agent_pos = (agent.slot_x, agent.slot_y)
    
    if has_other_agent:
        other_agent_id = other_agent_ids[0]
        other_agent = game.gameObjects[other_agent_id]
        other_pos = (other_agent.slot_x, other_agent.slot_y)
        midpoint = (int(round((agent.slot_x + other_agent.slot_x) / 2)), int(round((agent.slot_y + other_agent.slot_y) / 2)))
    else:
        other_agent = None
        other_pos = None  # No other agent
        # For single agent, midpoint is just the agent position
        midpoint = (agent.slot_x, agent.slot_y)
    obs_vector = []
    considered_paths = []
    considered_tiles = []
    
    # Add placeholder for do_nothing action (action index 0)
    # do_nothing doesn't require a tile or path  
    considered_paths.append(None)
    considered_tiles.append(-2)  # Use -2 to indicate do_nothing

    # --- Add times to tile types ---
    # Same three-indicator system as classic mode: [accessibility, availability, time]
    for tile_type in tile_types:
        if tile_type == 'cutting_board':
            action_time = agent_cutting_speed
        else:
            action_time = 0

        indices = get_tile_indices_by_type(game, tile_type)

        # STEP 1: Calculate ACCESSIBILITY (ignoring other agents)
        accessible_agent = [(_idx, x, y) for (_idx, x, y) in indices 
                           if get_distance_and_path(path_processor, agent_pos, (x, y), agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True)[0] is not None]
        
        min_dist_no_agents = None
        best_tile_idx = None
        best_tile_xy = None
        
        for _idx, x, y in accessible_agent:
            d, _ = get_distance_and_path(path_processor, agent_pos, (x, y), agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True)
            if d is not None and d >= 0:
                if min_dist_no_agents is None or d < min_dist_no_agents:
                    min_dist_no_agents = d
                    best_tile_idx = _idx
                    best_tile_xy = (x, y)
        
        # STEP 2: If accessible, calculate AVAILABILITY (considering other agents)
        if best_tile_xy is not None:
            accessibility = 1.0
            
            if path_processor.collision_enabled:
                dist_with_agents, path_with_agents = get_distance_and_path(
                    path_processor, agent_pos, best_tile_xy, agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=False
                )
                
                if dist_with_agents is not None and dist_with_agents >= 0:
                    availability = 1.0
                    final_dist = dist_with_agents
                    final_path = path_with_agents
                else:
                    availability = 0.0
                    best_tile_idx = -1
                    final_dist = min_dist_no_agents
                    final_path = None
            else:
                availability = 1.0
                final_dist = min_dist_no_agents
                _, final_path = get_distance_and_path(
                    path_processor, agent_pos, best_tile_xy, agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True
                )
            
            time_to_tile = (final_dist / agent_walking_speed + action_time) / normalization_factor
            considered_paths.append(final_path)
            considered_tiles.append(best_tile_idx)
            obs_vector.append(accessibility)  # 1.0 = accessible
            obs_vector.append(availability)   # 1.0 = available, 0.0 = blocked by agents
            obs_vector.append(time_to_tile)
        else:
            # No accessible tile found
            considered_paths.append(None)
            considered_tiles.append(None)
            obs_vector.append(0.0)  # Accessibility: inaccessible
            obs_vector.append(0.0)  # Availability: not available
            obs_vector.append(1.0)  # Max time

    # --- Add distances to items on counters ---
    # Same structure as classic mode: [presence, accessibility_closest, availability_closest, time_closest,
    #                                   accessibility_midpoint, availability_midpoint, time_midpoint]
    for item_name in item_names:
        action_time = 0
        indices = get_item_indices_on_counters(game, item_name)

        if len(indices) > 0:
            obs_vector.append(1) # Presence: item exists on counters
            
            # STEP 1: Find CLOSEST counter (ignoring agents)
            closest_accessible = [(_idx, x, y) for (_idx, x, y) in indices 
                                 if get_distance_and_path(path_processor, agent_pos, (x, y), agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True)[0] is not None]
            
            min_dist_no_agents = None
            best_closest_idx = None
            best_closest_xy = None
            
            for _idx, x, y in closest_accessible:
                d, _ = get_distance_and_path(path_processor, agent_pos, (x, y), agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True)
                if d is not None and d >= 0:
                    if min_dist_no_agents is None or d < min_dist_no_agents:
                        min_dist_no_agents = d
                        best_closest_idx = _idx
                        best_closest_xy = (x, y)
            
            # STEP 2: Check availability of closest counter
            if best_closest_xy is not None:
                accessibility_closest = 1.0
                
                if path_processor.collision_enabled:
                    dist_with_agents, path_with_agents = get_distance_and_path(
                        path_processor, agent_pos, best_closest_xy, agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=False
                    )
                    
                    if dist_with_agents is not None and dist_with_agents >= 0:
                        availability_closest = 1.0
                        final_dist_closest = dist_with_agents
                        final_path_closest = path_with_agents
                    else:
                        availability_closest = 0.0
                        best_closest_idx = -1
                        final_dist_closest = min_dist_no_agents
                        final_path_closest = None
                else:
                    availability_closest = 1.0
                    final_dist_closest = min_dist_no_agents
                    _, final_path_closest = get_distance_and_path(
                        path_processor, agent_pos, best_closest_xy, agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True
                    )
                
                time_to_closest = (final_dist_closest / agent_walking_speed + action_time) / normalization_factor
                considered_paths.append(final_path_closest)
                considered_tiles.append(best_closest_idx)
                obs_vector.append(accessibility_closest)
                obs_vector.append(availability_closest)
                obs_vector.append(time_to_closest)
            else:
                # No accessible counter
                considered_paths.append(None)
                considered_tiles.append(None)
                obs_vector.append(0.0)  # Accessibility: not accessible
                obs_vector.append(0.0)  # Availability: not available
                obs_vector.append(1.0)  # Max time

            # STEP 3: Find counter closest to MIDPOINT
            best = min(indices, key=lambda it: math.hypot(it[1] - midpoint[0], it[2] - midpoint[1]))
            _idx, bx, by = best
            
            dist_midpoint_no_agents, _ = get_distance_and_path(
                path_processor, agent_pos, (bx, by), agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True
            )
            
            if dist_midpoint_no_agents is not None and dist_midpoint_no_agents >= 0:
                accessibility_midpoint = 1.0
                
                if path_processor.collision_enabled:
                    dist_with_agents, path_with_agents = get_distance_and_path(
                        path_processor, agent_pos, (bx, by), agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=False
                    )
                    
                    if dist_with_agents is not None and dist_with_agents >= 0:
                        availability_midpoint = 1.0
                        final_dist_mid = dist_with_agents
                        final_path_mid = path_with_agents
                        final_idx_mid = _idx
                    else:
                        availability_midpoint = 0.0
                        final_dist_mid = dist_midpoint_no_agents
                        final_path_mid = None
                        final_idx_mid = -1
                else:
                    availability_midpoint = 1.0
                    final_dist_mid = dist_midpoint_no_agents
                    _, final_path_mid = get_distance_and_path(
                        path_processor, agent_pos, (bx, by), agent_id, game, 0.0, agent_walking_speed, force_ignore_agents=True
                    )
                    final_idx_mid = _idx
                
                time_to_midpoint = (final_dist_mid / agent_walking_speed + action_time) / normalization_factor
                considered_tiles.append(final_idx_mid)
                considered_paths.append(final_path_mid)
                obs_vector.append(accessibility_midpoint)
                obs_vector.append(availability_midpoint)
                obs_vector.append(time_to_midpoint)
            else:
                # Midpoint counter not accessible
                considered_tiles.append(None)
                considered_paths.append(None)
                obs_vector.append(0.0)  # Accessibility: not accessible
                obs_vector.append(0.0)  # Availability: not available
                obs_vector.append(1.0)  # Max time
        else:
            # No counters with this item
            obs_vector.append(0) # Presence: no tiles
            # Closest counter
            obs_vector.append(0.0) # Accessibility
            obs_vector.append(0.0) # Availability
            obs_vector.append(1.0) # Time
            # Midpoint counter
            obs_vector.append(0.0) # Accessibility
            obs_vector.append(0.0) # Availability
            obs_vector.append(1.0) # Time
            considered_paths.append(None)
            considered_paths.append(None)
            considered_tiles.append(None)
            considered_tiles.append(None)

    # Distance to other agent - both Euclidean and pathfinding distances
    if has_other_agent:
        # 1. Euclidean distance (straight-line distance)
        euclidean_dist = math.sqrt((agent_pos[0] - other_pos[0])**2 + (agent_pos[1] - other_pos[1])**2)
        euclidean_time = (euclidean_dist / agent_walking_speed) / normalization_factor
        obs_vector.append(euclidean_time)
        
        # 2. Pathfinding distance (using path processor for accessibility)
        pathfinding_dist, path = get_distance_and_path(path_processor, agent_pos, other_pos, agent_id, game, 0.0, agent_walking_speed)
        pathfinding_time = (pathfinding_dist / agent_walking_speed) / normalization_factor if pathfinding_dist is not None else 1
        obs_vector.append(pathfinding_time)
    else:
        # Single agent case: always use 1.0 (normalized max distance) for other agent distances
        obs_vector.append(1.0)
        obs_vector.append(1.0)

    # One-hot agent inventory
    agent_inventory = np.zeros(len(item_names), dtype=np.float32)
    item = getattr(agent, 'item', None)
    if item in item_names:
        agent_inventory[item_names.index(item)] = 1.0
    obs_vector.extend(agent_inventory.tolist())

    # One-hot other agent inventory
    if has_other_agent:
        other_inventory = np.zeros(len(item_names), dtype=np.float32)
        item_other = getattr(other_agent, 'item', None)
        if item_other in item_names:
            other_inventory[item_names.index(item_other)] = 1.0
        obs_vector.extend(other_inventory.tolist())
    else:
        # Single agent case: append zeros for other agent inventory
        other_inventory = np.zeros(len(item_names), dtype=np.float32)
        obs_vector.extend(other_inventory.tolist())
        
    return np.array(obs_vector, dtype=np.float32), considered_paths, considered_tiles

# Wrapper to select observation vector function based on game_mode.
def game_to_obs_vector(game, agent_id, game_mode="classic", path_processor=None):
    """
    Wrapper to select observation vector function based on game_mode.
    
    Args:
        path_processor: Path processor with pathfinder for distance calculations
        
    Returns:
        tuple: (obs_vector, action_paths, action_tiles)
            - obs_vector: numpy array with observation values  
            - action_paths: dict mapping action_idx to path (list of nodes)
            - action_tiles: dict mapping action_idx to target tile_index
    """
    if game_mode == "classic":
        result = game_to_obs_vector_classic(game, agent_id, path_processor)
    elif game_mode == "competition":
        result = game_to_obs_vector_competition(game, agent_id, path_processor)
    else:
        raise ValueError(f"Unknown game mode: {game_mode}")
    
    return result
    