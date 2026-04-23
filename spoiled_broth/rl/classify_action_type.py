ACTION_TYPE_DO_NOTHING = "do_nothing"
ACTION_TYPE_FLOOR = "useless_floor"
ACTION_TYPE_WALL = "useless_wall"
ACTION_TYPE_USELESS_COUNTER = "useless_counter"
ACTION_TYPE_USEFUL_COUNTER = "useful_counter"
ACTION_TYPE_SALAD_ASSEMBLY = "salad_assembly"
ACTION_TYPE_USEFUL_FOOD_DISPENSER = "useful_food_dispenser"
ACTION_TYPE_USELESS_FOOD_DISPENSER = "destructive_food_dispenser"
ACTION_TYPE_USELESS_CUTTING_BOARD = "useless_cutting_board"
ACTION_TYPE_USEFUL_CUTTING_BOARD = "useful_cutting_board"
ACTION_TYPE_USEFUL_PLATE_DISPENSER = "useful_plate_dispenser"
ACTION_TYPE_USELESS_PLATE_DISPENSER = "destructive_plate_dispenser"
ACTION_TYPE_USELESS_DELIVERY = "useless_delivery"
ACTION_TYPE_USEFUL_DELIVERY = "useful_delivery"
ACTION_TYPE_INACCESSIBLE = "inaccessible_tile"
ACTION_TYPE_BLOCKED = "blocked"

# Detailed action types considering ownership and usefulness (competition mode)
ACTION_TYPE_OWN_SALAD_ASSEMBLY = "salad_assembly_own"
ACTION_TYPE_OTHER_SALAD_ASSEMBLY = "salad_assembly_other"
ACTION_TYPE_OWN_USEFUL_FOOD_DISPENSER = "useful_food_dispenser_own"
ACTION_TYPE_OWN_USELESS_FOOD_DISPENSER = "destructive_food_dispenser_own"
ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER = "useful_food_dispenser_other"
ACTION_TYPE_OTHER_USELESS_FOOD_DISPENSER = "destructive_food_dispenser_other"
ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD = "useful_cutting_board_own"
ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD = "useful_cutting_board_other"
ACTION_TYPE_OWN_USEFUL_DELIVERY = "useful_delivery_own"
ACTION_TYPE_OTHER_USEFUL_DELIVERY = "useful_delivery_other"

# Centralized list of all action types for import
ACTION_TYPE_LIST_CLASSIC = [
    ACTION_TYPE_DO_NOTHING,
    ACTION_TYPE_FLOOR,
    ACTION_TYPE_WALL,
    ACTION_TYPE_USELESS_COUNTER,
    ACTION_TYPE_USEFUL_COUNTER,
    ACTION_TYPE_SALAD_ASSEMBLY,
    ACTION_TYPE_USELESS_FOOD_DISPENSER,
    ACTION_TYPE_USEFUL_FOOD_DISPENSER,
    ACTION_TYPE_USELESS_CUTTING_BOARD,
    ACTION_TYPE_USEFUL_CUTTING_BOARD,
    ACTION_TYPE_USELESS_PLATE_DISPENSER,
    ACTION_TYPE_USEFUL_PLATE_DISPENSER,
    ACTION_TYPE_USELESS_DELIVERY,
    ACTION_TYPE_USEFUL_DELIVERY,
    ACTION_TYPE_INACCESSIBLE,
    ACTION_TYPE_BLOCKED
]

ACTION_TYPE_LIST_COMPETITION = [
    ACTION_TYPE_DO_NOTHING,
    ACTION_TYPE_FLOOR,
    ACTION_TYPE_WALL,
    ACTION_TYPE_USELESS_COUNTER,
    ACTION_TYPE_USEFUL_COUNTER,
    ACTION_TYPE_OWN_SALAD_ASSEMBLY,
    ACTION_TYPE_OTHER_SALAD_ASSEMBLY,
    ACTION_TYPE_OWN_USELESS_FOOD_DISPENSER,
    ACTION_TYPE_OWN_USEFUL_FOOD_DISPENSER,
    ACTION_TYPE_OTHER_USELESS_FOOD_DISPENSER,
    ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER,
    ACTION_TYPE_USELESS_CUTTING_BOARD,
    ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD,
    ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD,
    ACTION_TYPE_USELESS_PLATE_DISPENSER,
    ACTION_TYPE_USEFUL_PLATE_DISPENSER,
    ACTION_TYPE_USELESS_DELIVERY,
    ACTION_TYPE_OWN_USEFUL_DELIVERY,
    ACTION_TYPE_OTHER_USEFUL_DELIVERY,
    ACTION_TYPE_INACCESSIBLE,
    ACTION_TYPE_BLOCKED
]

# Helper: wrapper functions to get action type based on game mode
def get_action_type_list(game_mode):
    """
    Returns the list of action types based on the game mode.
    """
    if game_mode == "classic":
        return ACTION_TYPE_LIST_CLASSIC
    elif game_mode == "competition":
        return ACTION_TYPE_LIST_COMPETITION
    else:
        raise ValueError(f"Unknown game mode: {game_mode}")

# ---- Classic mode without ownership awareness ---- #
def get_action_type_classic(tile, agent, x, y, accessibility_map):
    """
    Determine the type of action based on the tile clicked and agent state.
    """

    if tile is None or not hasattr(tile, '_type'):
        return ACTION_TYPE_BLOCKED  # Default to blocked for None/invalid tiles

    # Check accessibility using pre-computed map
    agent_pos = (agent.slot_x, agent.slot_y)
    tile_pos = (x, y)

    if accessibility_map is not None and tile_pos not in accessibility_map.get(agent_pos, set()):
        return ACTION_TYPE_INACCESSIBLE

    # Agent holds something? (e.g., tomato, salad, etc.)
    holding_item = getattr(agent, "item", None)

    # Tile type 0: floor
    if tile._type == 0:
        return ACTION_TYPE_FLOOR

    # Tile type 1: wall
    if tile._type == 1:
        return ACTION_TYPE_WALL
    # Tile type 2: counter
    if tile._type == 2:
        # Useful if agent is holding something (can drop it), or counter has something and agent can pick it
        item_on_counter = getattr(tile, "item", None)
        valid_salad_items = ["tomato_cut", "pumpkin_cut", "cabbage_cut"]
        if (holding_item == "plate" and item_on_counter in valid_salad_items) or (holding_item in valid_salad_items and item_on_counter == "plate"):
            return ACTION_TYPE_SALAD_ASSEMBLY
        elif holding_item is not None or item_on_counter is not None:
            return ACTION_TYPE_USEFUL_COUNTER
        else:
            return ACTION_TYPE_USELESS_COUNTER

    # Tile type 3: dispenser
    if tile._type == 3:
        if getattr(tile, "item", None) == "plate":
            if holding_item is None:
                return ACTION_TYPE_USEFUL_PLATE_DISPENSER
            else:
                return ACTION_TYPE_USELESS_PLATE_DISPENSER
        else:
            if holding_item is None:
                return ACTION_TYPE_USEFUL_FOOD_DISPENSER
            else:
                return ACTION_TYPE_USELESS_FOOD_DISPENSER
            
    # Tile type 4: cutting board
    if tile._type == 4:
        # Agent holds an item that can be cut
        valid_cutting_items = ["tomato", "pumpkin", "cabbage"]
        if holding_item in valid_cutting_items:
            return ACTION_TYPE_USEFUL_CUTTING_BOARD
        else:
            return ACTION_TYPE_USELESS_CUTTING_BOARD

    # Tile type 5: delivery
    if tile._type == 5:
        # Useful if agent holds a deliverable item (e.g., salad)
        valid_delivery_items = ['tomato_salad', 'pumpkin_salad', 'cabbage_salad']
        if holding_item in valid_delivery_items:
            return ACTION_TYPE_USEFUL_DELIVERY
        else:
            return ACTION_TYPE_USELESS_DELIVERY

    # Default fallback
    return ACTION_TYPE_FLOOR

# ---- Competition mode with ownership awareness ---- #
def get_action_type_competition(tile, agent, own_food, x, y, accessibility_map):
    """
    Determine the type of action based on the tile clicked and agent state.
    Uses agent_food_type to distinguish own food from other food.
    """

    if tile is None or not hasattr(tile, '_type'):
        return ACTION_TYPE_BLOCKED  # Default to blocked for None/invalid tiles
    
    # Check accessibility using pre-computed map
    agent_pos = (agent.slot_x, agent.slot_y)
    tile_pos = (x, y)
    if accessibility_map is not None and tile_pos not in accessibility_map.get(agent_pos, set()):
        return ACTION_TYPE_INACCESSIBLE

    holding_item = getattr(agent, "item", None)

    # Tile type 0: floor
    if tile._type == 0:
        return ACTION_TYPE_FLOOR

    # Tile type 1: wall
    if tile._type == 1:
        return ACTION_TYPE_WALL

    # Tile type 2: counter
    if tile._type == 2:
        # Useful if agent is holding something (can drop it), or counter has something and agent can pick it
        item_on_counter = getattr(tile, "item", None)
        valid_items = ["tomato_cut", "pumpkin_cut", "cabbage_cut"]
        if (holding_item in valid_items and item_on_counter == "plate") or (holding_item == "plate" and item_on_counter in valid_items):
            if holding_item in valid_items:
                if holding_item.startswith(own_food):
                    return ACTION_TYPE_OWN_SALAD_ASSEMBLY
                else:
                    return ACTION_TYPE_OTHER_SALAD_ASSEMBLY
            else:
                if item_on_counter.startswith(own_food):
                    return ACTION_TYPE_OWN_SALAD_ASSEMBLY
                else:
                    return ACTION_TYPE_OTHER_SALAD_ASSEMBLY 
        elif holding_item is not None or item_on_counter is not None:
            return ACTION_TYPE_USEFUL_COUNTER
        else:
            return ACTION_TYPE_USELESS_COUNTER

    # Tile type 3: dispenser
    if tile._type == 3:
        disp_item = getattr(tile, "item", None)
        if disp_item == "plate":
            if holding_item is None:
                return ACTION_TYPE_USEFUL_PLATE_DISPENSER
            else:
                return ACTION_TYPE_USELESS_PLATE_DISPENSER
        elif disp_item in ["tomato", "pumpkin", "cabbage"]:
            if disp_item == own_food:
                if holding_item is None:
                    return ACTION_TYPE_OWN_USEFUL_FOOD_DISPENSER
                else:
                    return ACTION_TYPE_OWN_USELESS_FOOD_DISPENSER
            else:
                if holding_item is None:
                    return ACTION_TYPE_OTHER_USEFUL_FOOD_DISPENSER
                else:
                    return ACTION_TYPE_OTHER_USELESS_FOOD_DISPENSER
        else:
            return ACTION_TYPE_OTHER_USELESS_FOOD_DISPENSER
            
    # Tile type 4: cutting board
    if tile._type == 4:
        # Item in hand can be cut (not already cut/salad)
        valid_items = ["tomato", "pumpkin", "cabbage"]
        if holding_item in valid_items:
            if holding_item == own_food:
                return ACTION_TYPE_OWN_USEFUL_CUTTING_BOARD
            else:
                return ACTION_TYPE_OTHER_USEFUL_CUTTING_BOARD
        return ACTION_TYPE_USELESS_CUTTING_BOARD

    # Tile type 5: delivery
    if tile._type == 5:
        # Useful if agent holds a deliverable item (e.g., salad)
        valid_items = ['tomato_salad', 'pumpkin_salad', 'cabbage_salad']
        if holding_item in valid_items:
            if holding_item.startswith(own_food):
                return ACTION_TYPE_OWN_USEFUL_DELIVERY
            else:
                return ACTION_TYPE_OTHER_USEFUL_DELIVERY
        else:
            return ACTION_TYPE_USELESS_DELIVERY

    # Default fallback
    return ACTION_TYPE_FLOOR

def get_action_type(tile, agent, agent_id=None, agent_food_type=None, game_mode="classic", x=None, y=None, accessibility_map=None):
    """
    Wrapper to select appropriate action type function based on game mode.
    """
    if game_mode == "classic":
        return get_action_type_classic(tile, agent, x=x, y=y, accessibility_map=accessibility_map)
    elif game_mode == "competition":
        own_food = agent_food_type.get(agent_id, None)
        return get_action_type_competition(tile, agent, own_food, x=x, y=y, accessibility_map=accessibility_map)
    else:
        raise ValueError(f"Unknown game mode: {game_mode}")