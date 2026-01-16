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

def get_distance_and_path(path_processor, from_xy, to_xy, agent_id=None, game=None, current_time=0.0, agent_speed=1.875):
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
    
    # Use path processor to calculate shortest path distance
    distance, path = path_processor.get_shortest_path_distance(
        game.grid, from_xy, to_xy, agent_id, current_time, agent_speed, game
    )
    if distance == -1 or path == -1:
        # Path blocked by collisions
        print("[get_distance_and_path] Path blocked by collisions.")
    return distance, path

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
    

    # --- Add times to tile types ---
    for tile_type in tile_types:
        if tile_type == 'cutting_board':
            action_time = agent_cutting_speed 
        else:
            action_time = 0

        indices = get_tile_indices_by_type(game, tile_type)

        # Only consider accessible tiles for agent
        accessible_agent = [(_idx, x, y) for (_idx, x, y) in indices if get_distance_and_path(path_processor, agent_pos, (x, y), agent_id, game, 0.0, agent_walking_speed)[0] is not None]
        min_dist = None
        considered_path = None
        tile_index = None
        for _idx, x, y in accessible_agent:
            d, path = get_distance_and_path(path_processor, agent_pos, (x, y), agent_id, game, 0.0, agent_walking_speed)
            if d == -1:
                print(f"[game_to_obs_vector_classic] Path blocked by collisions for tile type {tile_type}.")
                print(f"  from {agent_pos} to {(x, y)}")
                print(f"  actual min_dist: {min_dist}")
            if min_dist is None or (d < min_dist and d >= 0):
                min_dist = d
                considered_path = path
                tile_index = _idx
        print(f"[DEBUG] Agent {agent_id} position: {agent_pos}")
        print(f"[DEBUG] Other agent position: {other_pos}")
        print(f"[DEBUG] Tile type: {tile_type}, min_dist: {min_dist}, tile_index: {tile_index}")
        if min_dist is not None and min_dist >= 0:
            time_to_tile = (min_dist / agent_walking_speed + action_time) / normalization_factor
            considered_paths.append(considered_path)
            considered_tiles.append(tile_index)
            obs_vector.append(time_to_tile)
        elif min_dist is not None and min_dist == -1:
            # Path blocked by collisions
            time_to_tile = 1
            considered_paths.append(considered_path)
            considered_tiles.append(-1)
            print(f"[DEBUG] Considered_tiles: {considered_tiles}, path blocked by collisions.")
            obs_vector.append(time_to_tile)
        else:
            considered_paths.append(None)
            considered_tiles.append(None)
            obs_vector.append(1)


    # --- Add times to items on counters ---
    for item_name in item_names:
        action_time = 0
        indices = get_item_indices_on_counters(game, item_name)

        if len(indices) > 0:
            obs_vector.append(1) # There is a tile with that item
            # Only consider accessible counters for agent
            accessible_agent = [(_idx, x, y) for (_idx, x, y) in indices if get_distance_and_path(path_processor, agent_pos, (x, y), agent_id, game, 0.0, agent_walking_speed)[0] is not None]
            min_dist = None
            considered_path = None
            tile_index = None
            for _idx, x, y in accessible_agent:
                d, path = get_distance_and_path(path_processor, agent_pos, (x, y), agent_id, game, 0.0, agent_walking_speed)
                if min_dist is None or (d < min_dist and d >= 0):
                    min_dist = d
                    considered_path = path
                    tile_index = _idx
            if min_dist is not None and min_dist >= 0:
                time_to_tile = (min_dist / agent_walking_speed + action_time) / normalization_factor
                considered_paths.append(considered_path)
                considered_tiles.append(tile_index)
                obs_vector.append(time_to_tile)
            elif min_dist == -1:
                # Path blocked by collisions
                considered_paths.append(considered_path)
                considered_tiles.append(-1)
                obs_vector.append(time_to_tile)
            else:
                considered_paths.append(None)
                considered_tiles.append(None)
                obs_vector.append(1)

            # choose tile index with minimum Euclidean distance to midpoint
            best = min(indices, key=lambda it: math.hypot(it[1] - midpoint[0], it[2] - midpoint[1]))
            _idx, bx, by = best
            path_dist, path = get_distance_and_path(path_processor, agent_pos, (bx, by), agent_id, game, 0.0, agent_walking_speed)
            if path_dist is not None and path_dist >= 0:
                time_to_midtile = (path_dist / agent_walking_speed + action_time) / normalization_factor
                considered_tiles.append(_idx)
                considered_paths.append(path)
            else:
                time_to_midtile = 1
                considered_tiles.append(None)
                considered_paths.append(None)
            obs_vector.append(time_to_midtile)
        else:
            # no counters with that item: append fallbacks for agent time and midpoint time
            obs_vector.append(0) # There are no tiles with that item
            obs_vector.append(1) # Fallback for agent time
            obs_vector.append(1) # Fallback for midpoint time
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

    print(f"[DEBUG] Considered_tiles: {considered_tiles}")

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

    # --- Add times to tile types ---
    for tile_type in tile_types:
        if tile_type == 'cutting_board':
            action_time = agent_cutting_speed
        else:
            action_time = 0

        indices = get_tile_indices_by_type(game, tile_type)
        # Only consider accessible tiles for agent
        accessible_agent = [(_idx, x, y) for (_idx, x, y) in indices if get_distance_and_path(path_processor, agent_pos, (x, y), agent_id, game, 0.0, agent_walking_speed)[0] is not None]
        min_dist = None
        considered_path = None
        tile_index = None
        for _idx, x, y in accessible_agent:
            d, path = get_distance_and_path(path_processor, agent_pos, (x, y), agent_id, game, 0.0, agent_walking_speed)
            if min_dist is None or (d < min_dist and d >= 0):
                min_dist = d
                considered_path = path
                tile_index = _idx
        if min_dist is not None and min_dist >= 0:
            time_to_tile = (min_dist / agent_walking_speed + action_time) / normalization_factor
            considered_paths.append(considered_path)
            considered_tiles.append(tile_index)
            obs_vector.append(time_to_tile)
        elif min_dist is not None and min_dist == -1:
            # Path blocked by collisions
            time_to_tile = 1
            considered_paths.append(considered_path)
            considered_tiles.append(-1)
            obs_vector.append(time_to_tile)
        else:
            considered_paths.append(None)
            considered_tiles.append(None)
            obs_vector.append(1)

    # --- Add distances to items on counters ---
    for item_name in item_names:
        action_time = 0
        indices = get_item_indices_on_counters(game, item_name)

        if len(indices) > 0:
            obs_vector.append(1) # There is a tile with that item
            # Only consider accessible counters for agent
            accessible_agent = [(_idx, x, y) for (_idx, x, y) in indices if get_distance_and_path(path_processor, agent_pos, (x, y), agent_id, game, 0.0, agent_walking_speed)[0] is not None]
            min_dist = None
            considered_path = None
            tile_index = None
            for _idx, x, y in accessible_agent:
                d, path = get_distance_and_path(path_processor, agent_pos, (x, y), agent_id, game, 0.0, agent_walking_speed)
                if min_dist is None or (d < min_dist and d >= 0):
                    min_dist = d
                    considered_path = path
                    tile_index = _idx
            if min_dist is not None and min_dist >= 0:
                time_to_tile = (min_dist / agent_walking_speed + action_time) / normalization_factor
                considered_paths.append(considered_path)
                considered_tiles.append(tile_index)
                obs_vector.append(time_to_tile)
            elif min_dist is not None and min_dist == -1:
                # Path blocked by collisions
                time_to_tile = 1
                considered_paths.append(considered_path)
                considered_tiles.append(-1)
                obs_vector.append(time_to_tile)
            else:
                considered_paths.append(None)
                considered_tiles.append(None)
                obs_vector.append(1)

            # Choose the counter tile that is closest to the midpoint by Euclidean distance
            best = min(indices, key=lambda it: math.hypot(it[1] - midpoint[0], it[2] - midpoint[1]))
            _idx, bx, by = best
            path_dist, path = get_distance_and_path(path_processor, agent_pos, (bx, by), agent_id, game, 0.0, agent_walking_speed)
            if path_dist is not None and path_dist >= 0:
                time_to_midtile = (path_dist / agent_walking_speed + action_time) / normalization_factor
                considered_tiles.append(_idx)
                considered_paths.append(path)
                obs_vector.append(time_to_midtile)
            else:
                time_to_midtile = 1
                considered_tiles.append(None)
                considered_paths.append(None)
                obs_vector.append(time_to_midtile)
        else:
            # no counters with that item: append fallbacks for agent distance and midpoint distance
            obs_vector.append(0) # There are no tiles with that item
            obs_vector.append(1) # Fallback for agent time
            obs_vector.append(1) # Fallback for midpoint time
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
        return game_to_obs_vector_classic(game, agent_id, path_processor)
    elif game_mode == "competition":
        return game_to_obs_vector_competition(game, agent_id, path_processor)
    else:
        raise ValueError(f"Unknown game mode: {game_mode}")
    