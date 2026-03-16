from engine.base_game import BaseGame

from engine.extensions.topDownGridWorld.grid import Grid
from spoiled_broth.agent.base import Agent
import spoiled_broth.agent.base as base

from spoiled_broth.world.tiles import COLOR_MAP, CHAR_MAP
from spoiled_broth.ui.score import Score

from pathlib import Path

import os
import numpy as np
import random

BASE_WALKING_SPEED = 30  # Base walking speed in pixels/second

class SpoiledBroth(BaseGame):
    def __init__(self, map_nr=None, grid_size=(8, 8), num_agents=2, seed=None, walking_speeds=None, cutting_speeds=None, cutting_time=3):
        super().__init__()
        self.rng = random.Random(seed)
        if map_nr is None:
            map_nr = str(self.rng.randint(1, 4))  # default maps
        width, height = grid_size
        self.num_agents = num_agents
        self.agent_start_tiles = {}
        self.walking_speeds = walking_speeds
        self.cutting_speeds = cutting_speeds
        self.walked_tiles_per_second = BASE_WALKING_SPEED / 16  # Base walking speed in tiles/second
        self.cutting_time = cutting_time
        max_distance = load_max_distance(map_nr)
        self.normalization_factor = max_distance / self.walked_tiles_per_second + self.cutting_time # Walked_tiles_per_second in tiles/second + cutting_time in seconds
        self.clickable_indices = []  # Initialize clickable indices storage
        # Track action completion status for each agent
        self.agent_action_status = {}
        
        # Add frame counter for timing
        self.frame_count = 0
        
        self.grid = Grid("grid", width, height, 16)
        maps_root = Path(__file__).parent / "maps"
        map_path_img = maps_root / f"{map_nr}.png"
        map_path_txt = maps_root / f"{map_nr}.txt"
        map_path_img_nested = maps_root / "maps_png" / f"{map_nr}.png"
        map_path_txt_nested = maps_root / "maps_txt" / f"{map_nr}.txt"

        if map_path_txt.exists():
            # Legacy text location under spoiled_broth/maps
            self.grid.init_from_text(map_path_txt, CHAR_MAP, self)
        elif map_path_txt_nested.exists():
            # Preferred text location under spoiled_broth/maps/maps_txt
            self.grid.init_from_text(map_path_txt_nested, CHAR_MAP, self)
        elif map_path_img.exists():
            # Legacy image location under spoiled_broth/maps
            self.grid.init_from_img(map_path_img, COLOR_MAP, self)
        elif map_path_img_nested.exists():
            # Image fallback under spoiled_broth/maps/maps_png
            self.grid.init_from_img(map_path_img_nested, COLOR_MAP, self)
        else:
            raise FileNotFoundError(
                f"Map '{map_nr}' not found. Checked: {map_path_img}, {map_path_img_nested}, {map_path_txt}, {map_path_txt_nested}"
            )
        self.score = Score()
        self.gameObjects['grid'] = self.grid
        self.gameObjects['score'] = self.score
        a1_tile = None
        a2_tile = None
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                tile = self.grid.tiles[x][y]
                if tile and hasattr(tile, 'char'):
                    if tile.char == '1':
                        a1_tile = tile
                    elif tile.char == '2':
                        a2_tile = tile
        self.agent_start_tiles = {'1': a1_tile, '2': a2_tile}

        # Calculate clickable indices first
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                tile = self.grid.tiles[x][y]
                # Only include tiles that are actually interactable
                if (tile and tile.clickable is not None):  # Exclude floor (0) and walls (1)
                    index = y * self.grid.width + x
                    self.clickable_indices.append(index)

    def add_agent(self, agent_id, walk_speed=1, cut_speed=1):
        # Get individual speeds from dictionaries if available
        if self.walking_speeds and agent_id in self.walking_speeds:
            walk_speed = self.walking_speeds[agent_id]
        if self.cutting_speeds and agent_id in self.cutting_speeds:
            cut_speed = self.cutting_speeds[agent_id]
            
        agent = Agent(agent_id, self.grid, self, walk_speed=walk_speed, cut_speed=cut_speed)

        # Get fixed A1/A2 tiles if present
        a1_tile = self.agent_start_tiles.get('1', None)
        a2_tile = self.agent_start_tiles.get('2', None)

        # Extract agent number from the ID (e.g., 'ai_rl_1' -> 1)
        agent_number = int(agent_id.split('_')[-1])

        if a1_tile and a2_tile:
            if self.num_agents == 1:
                start_tile = self.rng.choice([a1_tile, a2_tile])
            else:  # assume num_agents == 2
                start_tile = a1_tile if agent_number == 1 else a2_tile
        else:
            # Fallback to random walkable tile if no fixed positions found
            choices = []
            for x in range(self.grid.width):
                for y in range(self.grid.height):
                    tile = self.grid.tiles[x][y]
                    if tile and tile.is_walkable:
                        choices.append(tile)
            start_tile = self.rng.choice(choices)

        # Assign pixel position
        agent.x = start_tile.slot_x * self.grid.tile_size + self.grid.tile_size // 2
        agent.y = start_tile.slot_y * self.grid.tile_size + self.grid.tile_size // 2
        
        # Set agent speed in pixels/second for movement
        agent.speed = agent.walk_speed * BASE_WALKING_SPEED
    
        self.gameObjects[agent_id] = agent

    def step(self, actions: dict, delta_time: float):
        # Increment frame counter
        self.frame_count += 1

        # Filter out None actions and ensure proper structure
        filtered_actions = {}
        for agent_id, action in actions.items():
            if action is not None and isinstance(action, dict):
                filtered_actions[agent_id] = action
        
        super().step(filtered_actions, delta_time)

    @property
    def agent_scores(self):
        return {
            aid: agent.score
            for aid, agent in self.gameObjects.items()
            if aid.startswith('ai_rl_')
        }

MAX_PLAYERS = 4 # or import if needed

def random_game_state(game, game_mode="classic"):
    """
    Randomize the game state by placing random items on counters and in agent hands.
    Agent positions remain fixed (already set by add_agent).
    
    Args:
        game: SpoiledBroth game instance
        game_mode: "classic" (tomato only) or "competition" (tomato + pumpkin)
    """
    # Define possible items based on game mode
    if game_mode == "competition":
        # Competition mode: both tomato and pumpkin
        raw_items = ['tomato', 'pumpkin']
        cut_items = ['tomato_cut', 'pumpkin_cut']
        salad_items = ['tomato_salad', 'pumpkin_salad']
    else:  # classic mode
        # Classic mode: only tomato
        raw_items = ['tomato']
        cut_items = ['tomato_cut']
        salad_items = ['tomato_salad']
    
    # Build weighted item lists for counters (higher probability for simpler items)
    # Weights: raw/plate=4, cut=2, salad=1, None=9 (45% nothing, 20% raw, 20% plate, 10% cut, 5% salad)
    counter_items = (
        raw_items * 4 + 
        ['plate'] * 4 + 
        cut_items * 2 + 
        salad_items * 1 + 
        [None] * 9
    )
    
    # Build weighted item lists for agent hands (even more skewed towards simple items)
    # Weights: raw/plate=3, cut=1, salad=1, None=12 (60% nothing, 15% raw, 15% plate, 5% cut, 5% salad)
    agent_items = (
        raw_items * 3 + 
        ['plate'] * 3 + 
        cut_items * 1 + 
        salad_items * 1 + 
        [None] * 12
    )
    
    # Randomize items on counters
    for x in range(game.grid.width):
        for y in range(game.grid.height):
            tile = game.grid.tiles[x][y]
            # Only randomize Counter tiles (type 2)
            if hasattr(tile, '_type') and tile._type == 2:
                tile.item = game.rng.choice(counter_items)
    
    # Randomize items in agent hands
    for agent in game.gameObjects.values():
        if hasattr(agent, "item"):
            agent.item = game.rng.choice(agent_items)

def load_max_distance(map_id, cache_dir=None):
    """
    Loads the max distance for the given map_id from cache.
    """
    if cache_dir is None:
        # Default to spoiled_broth/maps/distance_cache
        cache_dir = os.path.join(os.path.dirname(__file__), './maps/distance_cache')
    max_dist_path = os.path.join(cache_dir, f"distance_map_{map_id}_max_distance.npy")
    if not os.path.exists(max_dist_path):
        raise FileNotFoundError(f"Max distance cache not found for map_id {map_id} at {max_dist_path}")
    return float(np.load(max_dist_path))