# --- New RL Action Space ---
# These are high-level, human-like actions that do not depend on the map layout

# For each action, add both 'closest' and 'midpoint' variants where relevant
RL_ACTIONS_CLASSIC = [
    "do_nothing",
    "pick_up_tomato_from_dispenser",
    "pick_up_plate_from_dispenser",
    "use_cutting_board",
    "use_delivery",
    "put_down_item_on_free_counter_closest",
    "put_down_item_on_free_counter_midpoint",
    "pick_up_tomato_from_counter_closest",
    "pick_up_tomato_from_counter_midpoint",
    "pick_up_plate_from_counter_closest",
    "pick_up_plate_from_counter_midpoint",
    "pick_up_tomato_cut_from_counter_closest",
    "pick_up_tomato_cut_from_counter_midpoint",
    "pick_up_tomato_salad_from_counter_closest",
    "pick_up_tomato_salad_from_counter_midpoint"
]

RL_ACTIONS_COMPETITION = [
    "do_nothing",
    "pick_up_tomato_from_dispenser",
    "pick_up_pumpkin_from_dispenser",
    "pick_up_plate_from_dispenser",
    "use_cutting_board",
    "use_delivery",
    "put_down_item_on_free_counter_closest",
    "put_down_item_on_free_counter_midpoint",
    "pick_up_tomato_from_counter_closest",
    "pick_up_tomato_from_counter_midpoint",
    "pick_up_pumpkin_from_counter_closest",
    "pick_up_pumpkin_from_counter_midpoint",
    "pick_up_plate_from_counter_closest",
    "pick_up_plate_from_counter_midpoint",
    "pick_up_tomato_cut_from_counter_closest",
    "pick_up_tomato_cut_from_counter_midpoint",
    "pick_up_pumpkin_cut_from_counter_closest",
    "pick_up_pumpkin_cut_from_counter_midpoint",
    "pick_up_tomato_salad_from_counter_closest",
    "pick_up_tomato_salad_from_counter_midpoint",
    "pick_up_pumpkin_salad_from_counter_closest",
    "pick_up_pumpkin_salad_from_counter_midpoint"
]

def get_rl_action_space(game_mode="classic"):
    if game_mode == "competition":
        return RL_ACTIONS_COMPETITION
    elif game_mode == "classic":
        return RL_ACTIONS_CLASSIC
    else:
        raise ValueError(f"Unknown game mode: {game_mode}")

# --- Tile selection helpers using distance map ---
def find_closest_tile(agent, tile_candidates, collision_processor):
    """
    agent: agent object with slot_x, slot_y
    tile_candidates: list of (idx, x, y)
    collision_processor: collision processor with pathfinder
    Returns the idx of the closest tile (or None if not reachable)
    """
    if not collision_processor or not collision_processor.pathfinder:
        # Fallback to Euclidean distance if no pathfinder
        agent_pos = (agent.slot_x, agent.slot_y)
        best = None
        best_dist = float('inf')
        for idx, x, y in tile_candidates:
            dist = ((agent_pos[0] - x) ** 2 + (agent_pos[1] - y) ** 2) ** 0.5
            if dist < best_dist:
                best = idx
                best_dist = dist
        return best
    
    from engine.extensions.topDownGridWorld.a_star import Node
    
    agent_pos = Node(agent.slot_x, agent.slot_y)
    best = None
    best_dist = float('inf')
    
    for idx, x, y in tile_candidates:
        target_pos = Node(x, y)
        path = collision_processor.pathfinder.find_path(agent_pos, target_pos)
        if path:
            path_length = len(path)
            if path_length < best_dist:
                best = idx
                best_dist = path_length
    return best

def find_midpoint_tile(agent, other_agent, tile_candidates, collision_processor):
    """
    agent, other_agent: agent objects with slot_x, slot_y
    tile_candidates: list of (idx, x, y)
    collision_processor: collision processor with pathfinder
    Returns the idx of the tile closest to the midpoint between the two agents (by path distance sum)
    """
    if not collision_processor or not collision_processor.pathfinder:
        # Fallback to Euclidean distance if no pathfinder
        pos1 = (agent.slot_x, agent.slot_y)
        pos2 = (other_agent.slot_x, other_agent.slot_y)
        best = None
        best_dist = float('inf')
        for idx, x, y in tile_candidates:
            d1 = ((pos1[0] - x) ** 2 + (pos1[1] - y) ** 2) ** 0.5
            d2 = ((pos2[0] - x) ** 2 + (pos2[1] - y) ** 2) ** 0.5
            total = d1 + d2
            if total < best_dist:
                best = idx
                best_dist = total
        return best
    
    from engine.extensions.topDownGridWorld.a_star import Node
    
    pos1 = Node(agent.slot_x, agent.slot_y)
    pos2 = Node(other_agent.slot_x, other_agent.slot_y)
    best = None
    best_dist = float('inf')
    
    for idx, x, y in tile_candidates:
        target_pos = Node(x, y)
        path1 = collision_processor.pathfinder.find_path(pos1, target_pos)
        path2 = collision_processor.pathfinder.find_path(pos2, target_pos)
        if path1 and path2:
            total = len(path1) + len(path2)
            if total < best_dist:
                best = idx
                best_dist = total
    return best

# Convert RL action to tile click --- LEGACY FUNCTION ---
def convert_action_to_tile(agent, game, action_name, collision_processor=None):
    """
    Given an agent, game state, and high-level action name (with _closest or _midpoint), return the tile index to click (or None for do_nothing).
    Uses the collision processor's pathfinder for efficient shortest path calculation.
    """
    # Handle do_nothing action
    if action_name == "do_nothing":
        return None
    
    # Parse action_name for target_mode
    if action_name.endswith("_midpoint"):
        base_action = action_name[:-9]
        target_mode = "midpoint"
    elif action_name.endswith("_closest"):
        base_action = action_name[:-8]
        target_mode = "closest"
    else:
        base_action = action_name
        target_mode = "closest"

    # Find the other agent (assume 2 agents)
    other_agent = None
    for a_id, a_obj in game.gameObjects.items():
        if hasattr(a_obj, 'slot_x') and a_obj is not agent:
            other_agent = a_obj
            break

    # Helper to get clickable tiles of a certain type
    def get_tiles_by_type(type_check):
        tiles = []
        grid = game.grid
        for idx in game.clickable_indices:
            x = idx % grid.width
            y = idx // grid.width
            tile = grid.tiles[x][y]
            if type_check(tile):
                tiles.append((idx, x, y))
        return tiles

    # Action logic for each base_action
    if base_action == "pick_up_tomato_from_dispenser":
        def is_tomato_dispenser(tile):
            return getattr(tile, "_type", None) == 3 and hasattr(tile, "item") and tile.item == "tomato"
        candidates = get_tiles_by_type(is_tomato_dispenser)
    elif base_action == "pick_up_pumpkin_from_dispenser":
        def is_pumpkin_dispenser(tile):
            return getattr(tile, "_type", None) == 3 and hasattr(tile, "item") and tile.item == "pumpkin"
        candidates = get_tiles_by_type(is_pumpkin_dispenser)
    elif base_action == "pick_up_plate_from_dispenser":
        def is_plate_dispenser(tile):
            return getattr(tile, "_type", None) == 3 and hasattr(tile, "item") and tile.item == "plate"
        candidates = get_tiles_by_type(is_plate_dispenser)
    elif base_action == "use_cutting_board":
        def is_cutting_board(tile):
            return getattr(tile, "_type", None) == 4 and not getattr(tile, "item", None)
        candidates = get_tiles_by_type(is_cutting_board)
    elif base_action == "use_delivery":
        def is_delivery(tile):
            return getattr(tile, "_type", None) == 5
        candidates = get_tiles_by_type(is_delivery)
    elif base_action == "put_down_item_on_free_counter":
        def is_free_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) is None
        candidates = get_tiles_by_type(is_free_counter)
    elif base_action == "pick_up_tomato_from_counter":
        def is_tomato_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "tomato"
        candidates = get_tiles_by_type(is_tomato_on_counter)
    elif base_action == "pick_up_pumpkin_from_counter":
        def is_pumpkin_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "pumpkin"
        candidates = get_tiles_by_type(is_pumpkin_on_counter)
    elif base_action == "pick_up_plate_from_counter":
        def is_plate_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "plate"
        candidates = get_tiles_by_type(is_plate_on_counter)
    elif base_action == "pick_up_tomato_cut_from_counter":
        def is_tomato_cut_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "tomato_cut"
        candidates = get_tiles_by_type(is_tomato_cut_on_counter)
    elif base_action == "pick_up_pumpkin_cut_from_counter":
        def is_pumpkin_cut_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "pumpkin_cut"
        candidates = get_tiles_by_type(is_pumpkin_cut_on_counter)
    elif base_action == "pick_up_tomato_salad_from_counter":
        def is_tomato_salad_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "tomato_salad"
        candidates = get_tiles_by_type(is_tomato_salad_on_counter)
    elif base_action == "pick_up_pumpkin_salad_from_counter":
        def is_pumpkin_salad_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "pumpkin_salad"
        candidates = get_tiles_by_type(is_pumpkin_salad_on_counter)
    else:
        return None

    # Use the appropriate selection helper
    if not candidates:
        return None
        
    if target_mode == "closest":
        result = find_closest_tile(agent, candidates, collision_processor)
    elif target_mode == "midpoint" and other_agent is not None:
        result = find_midpoint_tile(agent, other_agent, candidates, collision_processor)
    else:
        result = find_closest_tile(agent, candidates, collision_processor)
    
    return result


def get_tiles_for_action(game, action_name):
    """
    Get all candidate tile indices for a given action name.
    
    Args:
        game: SpoiledBroth game instance
        action_name: String name of the action
        
    Returns:
        List of tile indices that match the action criteria
    """
    def get_tiles_by_type(predicate_func):
        """Get all tiles that match the predicate function."""
        candidates = []
        for x in range(game.grid.width):
            for y in range(game.grid.height):
                tile = game.grid.tiles[x][y]
                if predicate_func(tile):
                    idx = y * game.grid.width + x
                    candidates.append(idx)
        return candidates
    
    # Parse action name to understand the requirements
    if action_name == "pick_up_tomato_from_dispenser":
        def is_tomato_dispenser(tile):
            return getattr(tile, "_type", None) == 3 and getattr(tile, "item", None) == "tomato"
        return get_tiles_by_type(is_tomato_dispenser)
    elif action_name == "pick_up_pumpkin_from_dispenser":
        def is_pumpkin_dispenser(tile):
            return getattr(tile, "_type", None) == 3 and getattr(tile, "item", None) == "pumpkin"
        return get_tiles_by_type(is_pumpkin_dispenser)
    elif action_name == "pick_up_plate_from_dispenser":
        def is_plate_dispenser(tile):
            return getattr(tile, "_type", None) == 3 and getattr(tile, "item", None) == "plate"
        return get_tiles_by_type(is_plate_dispenser)
    elif action_name == "use_cutting_board":
        def is_cutting_board(tile):
            return getattr(tile, "_type", None) == 4
        return get_tiles_by_type(is_cutting_board)
    elif action_name == "use_delivery":
        def is_delivery(tile):
            return getattr(tile, "_type", None) == 5
        return get_tiles_by_type(is_delivery)
    elif "put_down_item_on_free_counter" in action_name:
        def is_free_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) is None
        return get_tiles_by_type(is_free_counter)
    elif "pick_up_tomato_from_counter" in action_name:
        def is_tomato_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "tomato"
        return get_tiles_by_type(is_tomato_on_counter)
    elif "pick_up_pumpkin_from_counter" in action_name:
        def is_pumpkin_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "pumpkin"
        return get_tiles_by_type(is_pumpkin_on_counter)
    elif "pick_up_plate_from_counter" in action_name:
        def is_plate_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "plate"
        return get_tiles_by_type(is_plate_on_counter)
    elif "pick_up_tomato_cut_from_counter" in action_name:
        def is_tomato_cut_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "tomato_cut"
        return get_tiles_by_type(is_tomato_cut_on_counter)
    elif "pick_up_pumpkin_cut_from_counter" in action_name:
        def is_pumpkin_cut_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "pumpkin_cut"
        return get_tiles_by_type(is_pumpkin_cut_on_counter)
    elif "pick_up_tomato_salad_from_counter" in action_name:
        def is_tomato_salad_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "tomato_salad"
        return get_tiles_by_type(is_tomato_salad_on_counter)
    elif "pick_up_pumpkin_salad_from_counter" in action_name:
        def is_pumpkin_salad_on_counter(tile):
            return getattr(tile, "_type", None) == 2 and getattr(tile, "item", None) == "pumpkin_salad"
        return get_tiles_by_type(is_pumpkin_salad_on_counter)
    else:
        # Unknown action type
        return []