import numpy as np
import time

from engine.extensions.renderer2d.basic_2d import Basic2D
from engine.extensions.topDownGridWorld import agent
import random
from spoiled_broth.world.tiles import ITEM_LIST
from spoiled_broth.world.tiles import Dispenser
from spoiled_broth.agent.intents import PickUpIntent

class Agent(agent.Agent):
    def __init__(self, agent_id, grid, game, walk_speed=1, cut_speed=1):
        super().__init__(agent_id, grid)
        self.game = game
        self.path = []
        self.path_index = 0
        self.move_target = None

        self.last_click_target = None  # Store last RL click target for intent processing

        self.item = None
        self.provisional_item = None  # Item that is being processed (e.g., being cut)
        self.walk_speed = walk_speed
        self.cut_speed = cut_speed
        # Set movement speed: base 30 pixels/second * walk_speed multiplier
        # walk_speed=1.0 → 30 px/s (1.875 tiles/s)
        # walk_speed=0.5 → 15 px/s (0.9375 tiles/s)
        self.speed = 30.0 * walk_speed  # pixels/second
        self.action = None
        self.score = 0
        self.current_action = None
        self.is_busy = False  # Initially no action is in progress
        self.is_cutting = False  # Flag to indicate if the agent is currently cutting
        self.is_simulation = False  # Flag to indicate if this agent is in a simulation environment
        hair_n = random.randint(0, 8)
        mustache_n = random.randint(0, 8)
        skin_n = random.randint(0, 8)

        husk_2d = Basic2D(src='agent/cook-husk.png', z_index=1, src_y=0, src_w=16, src_h=16, width=16, height=16,
                          normalize=False)
        hair_2d = Basic2D(src=f'agent/hair/{hair_n}.png', z_index=2, src_w=16, src_h=16, width=16, height=16,
                          normalize=False)
        mustache_2d = Basic2D(src=f'agent/mustache/{mustache_n}.png', z_index=2, src_w=16, src_h=16, width=16,
                              height=16, normalize=False)
        head_2d = Basic2D(src=f'agent/skin/{skin_n}.png', z_index=2, src_y=0, src_w=16, src_h=16, width=16, height=16,
                          normalize=False)
        hands_no_item_2d = Basic2D(src=f'agent/skin/{skin_n}.png', z_index=2, src_y=16, src_w=16, src_h=16, width=16,
                                   height=16, normalize=False)
        hands_with_item_2d = Basic2D(src=f'agent/skin/{skin_n}.png', z_index=2, src_y=32, src_w=16, src_h=16, width=0,
                                     height=0, normalize=False)
        item_held_2d = Basic2D(src='agent/items-held.png', z_index=3, src_y=0, src_w=16, src_h=16, width=0, height=0,
                               normalize=False)

        self.src_x = 0
        self.src_x_n = 0
        self.animation_time = 0

        self.add_drawable(husk_2d)
        self.add_drawable(hair_2d)
        self.add_drawable(mustache_2d)
        self.add_drawable(head_2d)
        self.add_drawable(hands_no_item_2d)
        self.add_drawable(hands_with_item_2d)
        self.add_drawable(item_held_2d)

    @property
    def is_moving(self):
        return self.path_index < len(self.path)

    def update(self, actions: dict, delta_time: float):
        # Process RL 'click' actions to set movement target FIRST
        if self.id in actions:
            action = actions[self.id]
            if isinstance(action, dict) and action.get("type") == "click":
                tile_index = action.get("target")
                if tile_index is not None:
                    # Ensure tile_index is an integer
                    try:
                        tile_index = int(tile_index)
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid tile_index '{tile_index}' for agent {self.id}")
                        return
                    
                    grid_w = self.grid.width
                    x = tile_index % grid_w
                    y = tile_index // grid_w
                    tile = self.grid.tiles[x][y]
                    self.set_move_target(tile)
                    self.last_click_target = tile_index  # Track last RL click target

        # Check if there are actions for this agent
        if self.id in actions:
            # If we're getting a new action but the agent is busy, ignore it
            if getattr(self, 'is_busy', False):
                actions.pop(self.id)
            else:
                # Only set is_busy for non-movement actions (not RL click/move)
                action = actions[self.id]
                if action is not None and (not isinstance(action, dict) or action.get("type") != "click"):
                    self.is_busy = True

        # Always move agent if a path is set (for RL movement)
        self.move(delta_time)
        # Now call base update logic
        super().update(actions, delta_time)

        # After movement, check if agent is at the target and trigger intent(s) for the clickable tile
        if not self.is_moving and self.last_click_target is not None:
            grid_w = self.grid.width
            x = self.last_click_target % grid_w
            y = self.last_click_target // grid_w
            tile = self.grid.tiles[x][y]
            if hasattr(tile, "get_intent"):
                intents = tile.get_intent(self)
                if intents:
                    for intent in intents:
                        if type(intent).__name__ != "MoveToIntent":
                            try:
                                intent.update(self, delta_time)
                            except TypeError:
                                intent.update(self)
            self.last_click_target = None  # Reset after processing

        # Handle animation updates
        if not self.is_moving:
            self.animation_time = 0
            self.x = self.slot_x * self.grid.tile_size + self.grid.tile_size // 2
            self.y = self.slot_y * self.grid.tile_size + self.grid.tile_size // 2
            self.src_x_n = 0
        else:
            self.animation_time += delta_time
            if self.animation_time > 0.1:
                self.src_x_n += 1
                self.src_x_n %= 5
                self.animation_time = 0
        self.src_x = self.src_x_n * 16

        for drawable in self.drawables:
            drawable.src_x = self.src_x

        if self.item is not None:
            self.drawables[4].width = 0
            self.drawables[4].height = 0
            self.drawables[5].width = 16
            self.drawables[5].height = 16
            self.drawables[6].width = 16
            self.drawables[6].height = 16
            if self.item == 'tomato':
                self.drawables[6].src_y = 0
            if self.item == 'pumpkin':
                self.drawables[6].src_y = 16
            if self.item == 'cabbage':
                self.drawables[6].src_y = 32
            if self.item == 'plate':
                self.drawables[6].src_y = 48
            if self.item == 'tomato_cut':
                self.drawables[6].src_y = 9 * 16
            if self.item == 'pumpkin_cut':
                self.drawables[6].src_y = 10 * 16
            if self.item == 'cabbage_cut':
                self.drawables[6].src_y = 11 * 16
            if self.item == 'tomato_salad':
                self.drawables[6].src_y = 15 * 16
            if self.item == 'pumpkin_salad':
                self.drawables[6].src_y = 16 * 16
            if self.item == 'cabbage_salad':
                self.drawables[6].src_y = 17 * 16
        else:
            self.drawables[4].width = 16
            self.drawables[4].height = 16
            self.drawables[5].width = 0
            self.drawables[5].height = 0
            self.drawables[6].width = 0
            self.drawables[6].height = 0
