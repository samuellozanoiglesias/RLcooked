"""
Optimized game step functions for CPU execution.
These functions handle game state updates efficiently on CPU while
neural network training happens on GPU in parallel.
"""

def complete_agent_action(self, agent_id, agent, action_data, agent_events, own_food):
    """Complete an agent's action by updating game state directly."""
    tile_index = action_data['tile_index']
    
    # Get the tile
    grid_w = self.game.grid.width
    x = tile_index % grid_w
    y = tile_index // grid_w
    tile = self.game.grid.tiles[x][y]
    
    # Ensure agent has finished moving to the target (if it's not walkable, position next to it)
    if hasattr(tile, 'is_walkable') and tile.is_walkable:
        agent.x = tile.slot_x * self.game.grid.tile_size + self.game.grid.tile_size // 2
        agent.y = tile.slot_y * self.game.grid.tile_size + self.game.grid.tile_size // 2
        # Note: slot_x and slot_y are calculated automatically from x and y pixel coordinates
    else:
        # Position agent at nearest walkable tile (should already be there from pathfinding)
        # Clear the path since we've reached the target
        agent.path = []
        agent.path_index = 0
    
    # Process the action based on tile type and agent state
    if hasattr(tile, '_type'):
        if tile._type == 3:  # Dispenser
            holding_item = getattr(agent, "item", None)
            handle_dispenser_action(agent, tile)
            new_holding_item = getattr(agent, "item", None)
            agent_events = log_dispenser_action(agent_id, holding_item, new_holding_item, agent_events, own_food)
        elif tile._type == 2:  # Counter
            holding_item = getattr(agent, "item", None)
            tile_item_before = getattr(tile, "item", None)
            handle_counter_action(agent, tile)
            new_tile_item = getattr(tile, "item", None)
            new_holding_item = getattr(agent, "item", None)
            agent_events = log_counter_action(agent_id, holding_item, new_holding_item, new_tile_item, agent_events, own_food)
        elif tile._type == 4:  # Cutting board
            holding_item = getattr(agent, "item", None)
            handle_cutting_board_action(agent, tile)
            new_holding_item = getattr(agent, "item", None)
            agent_events = log_cutting_board_action(agent_id, holding_item, new_holding_item, agent_events, own_food)
        elif tile._type == 5:  # Delivery
            holding_item = getattr(agent, "item", None)
            handle_delivery_action(self, agent, tile)
            new_holding_item = getattr(agent, "item", None)
            agent_events = log_delivery_action(agent_id, holding_item, new_holding_item, agent_events, own_food)
    
    # Clear agent's path from collision tracking if enabled
    if hasattr(self, 'path_processor') and self.path_processor.is_enabled():
        self.path_processor.clear_agent_path(agent_id)

    return agent_events

def log_dispenser_action(agent_id, holding_item, new_holding_item, agent_events, own_food):
    """Log dispenser action for analysis."""
    if new_holding_item == "plate":
        if holding_item is None or holding_item != "plate":
            agent_events[agent_id]["plate"] += 1
    else: 
        if own_food is not None:
            if new_holding_item == own_food:
                agent_events[agent_id]["raw_food_own"] += 1
            else:
                agent_events[agent_id]["raw_food_other"] += 1
        else:
            if holding_item is None or holding_item != new_holding_item:
                agent_events[agent_id]["raw_food"] += 1
    return agent_events

def log_counter_action(agent_id, holding_item, new_holding_item, new_tile_item, agent_events, own_food):
    """Log counter action for analysis."""
    valid_salad_items = ["tomato_salad", "pumpkin_salad", "cabbage_salad"]
    if own_food is not None:
        if new_holding_item is None and new_tile_item == f"{own_food}_salad":
            agent_events[agent_id]["salad_own"] += 1
        elif holding_item != new_tile_item and new_tile_item in valid_salad_items and new_holding_item is None:
            agent_events[agent_id]["salad_other"] += 1
        elif holding_item is not None or new_holding_item is not None:
            agent_events[agent_id]["counter"] += 1
    else:
        if holding_item != new_tile_item and new_tile_item in valid_salad_items and new_holding_item is None:
            agent_events[agent_id]["salad"] += 1
        elif holding_item is not None or new_holding_item is not None:
            agent_events[agent_id]["counter"] += 1
    return agent_events

def log_cutting_board_action(agent_id, holding_item, new_holding_item, agent_events, own_food):
    """Log cutting board action for analysis."""
    valid_items = ["tomato", "pumpkin", "cabbage"]
    if own_food is not None:
        if holding_item == own_food and new_holding_item == f"{own_food}_cut":
            agent_events[agent_id]["cut_own"] += 1
        elif holding_item in valid_items and new_holding_item == f"{holding_item}_cut":
            agent_events[agent_id]["cut_other"] += 1
    else:
        if holding_item in valid_items and new_holding_item == f"{holding_item}_cut":
            agent_events[agent_id]["cut"] += 1
    return agent_events

def log_delivery_action(agent_id, holding_item, new_holding_item, agent_events, own_food):
    """Log delivery action for analysis."""
    valid_items = ["tomato_salad", "pumpkin_salad", "cabbage_salad"]
    if own_food is not None:
        if holding_item == f"{own_food}_salad" and new_holding_item is None:
            agent_events[agent_id]["deliver_own"] += 1
        elif holding_item in valid_items and new_holding_item is None:
            agent_events[agent_id]["deliver_other"] += 1
    else:
        if holding_item in valid_items and new_holding_item is None:
            agent_events[agent_id]["deliver"] += 1
    return agent_events

def handle_dispenser_action(agent, tile):
    """Handle picking up items from dispensers."""
    if hasattr(tile, 'item'):
        # Always pick up from dispenser, replacing any item the agent currently has
        agent.item = tile.item
        
    # Position agent next to dispenser (dispensers are not walkable)
    # Agent should already be positioned by pathfinding, but ensure they're stopped
    agent.path = []
    agent.path_index = 0

def handle_counter_action(agent, tile):
    """Handle interactions with counters (placing/picking up items, making salads)."""
    # Check if this is a pick_up action
    valid_salad_items = ['tomato_cut', 'pumpkin_cut', 'cabbage_cut']

    if hasattr(tile, 'item') or agent.item is not None:
        # Special case: Agent has plate and counter has cut ingredient -> make salad
        if (agent.item == 'plate' and tile.item in valid_salad_items):
            tile.item = tile.item.split('_')[0] + '_salad'
            if hasattr(tile, 'salad_item'):
                tile.salad_item = tile.item
            agent.item = None  # Agent does not pick up the salad
                    
        # Special case: Agent has cut ingredient and counter has plate -> make salad  
        elif (agent.item in valid_salad_items and tile.item == 'plate'):
            tile.item = agent.item.split('_')[0] + '_salad'
            if hasattr(tile, 'salad_item'):
                tile.salad_item = tile.item
            agent.item = None  # Agent does not pick up the salad
                    
        # Normal item exchange
        else:
            temp_item = tile.item
            tile.item = agent.item
            agent.item = temp_item

    # Position agent next to counter (counters are not walkable)
    agent.path = []
    agent.path_index = 0

def handle_cutting_board_action(agent, tile):
    """Handle cutting actions on cutting board."""
    valid_cutting_items = ['tomato', 'pumpkin', 'cabbage']

    if agent.item in valid_cutting_items:
        # Instantly complete cutting action (as per original intent behavior)
        original_item = agent.item
        cut_item = f"{original_item}_cut"
        agent.item = cut_item
        
        # Update tile metadata
        if hasattr(tile, 'cut_item'):
            tile.cut_item = cut_item
        if hasattr(tile, 'item'):
            tile.item = None  # Clear the cutting board
    
    # Position agent next to cutting board (cutting boards are not walkable)
    # Agent should already be positioned by pathfinding, but ensure they're stopped
    agent.path = []
    agent.path_index = 0

def handle_delivery_action(self, agent, tile):
    """Handle delivery actions."""
    valid_delivery_items = ['tomato_salad', 'pumpkin_salad', 'cabbage_salad']
    
    if agent.item in valid_delivery_items:
        # Set delivery metadata
        if hasattr(tile, 'delivered_by'):
            tile.delivered_by = agent.id
        if hasattr(tile, 'delivered_item'):
            tile.delivered_item = agent.item
            
        # Clear agent item and increment score
        agent.item = None
        agent.score += 1
        
        # Update game score if it exists
        if ('score' in self.game.gameObjects and 
            self.game.gameObjects['score'] is not None):
            self.game.gameObjects['score'].score += 1
    
    # Position agent next to delivery (delivery tiles are not walkable)
    agent.path = []
    agent.path_index = 0