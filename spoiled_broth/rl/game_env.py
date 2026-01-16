import csv
import os
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
from spoiled_broth.config import *
from spoiled_broth.maps.accessibility_maps import get_accessibility_map
import pickle as _pickle
from spoiled_broth.rl.game_step import update_agents_directly, setup_agent_path
from spoiled_broth.rl.action_space import get_rl_action_space
from spoiled_broth.rl.observation_space import game_to_obs_vector
from spoiled_broth.rl.classify_action_type import get_action_type, get_action_type_list
from spoiled_broth.rl.reward_analysis import get_rewards
from spoiled_broth.rl.dynamic_rewards import calculate_dynamic_rewards
from spoiled_broth.game import SpoiledBroth, random_game_state
from spoiled_broth.rl.path_processing import PathProcessor

# Base intent time for non-cutting actions
INTENT_TIME = 0.5
MOVE_TIME = 0.2

def get_cutting_time(agent, game):
    """Calculate actual cutting time based on agent's cutting speed."""
    cutting_speed = getattr(agent, 'cut_speed', 1.0)
    base_cutting_time = getattr(game, 'cutting_time', 3.0)
    # Lower speed = more time (inverse relationship)
    return base_cutting_time / cutting_speed if cutting_speed > 0 else base_cutting_time

ACTIONS_OBSERVATION_MAPPING_CLASSIC = {
    0: 0, 1: 1, 2: 2, 3: 3, # Two dispensers, cutting board, delivery
    4: 5, 5: 6, # Free counter (closest and midpoint)
    6: 8, 7: 9, # Tomato on counter (closest and midpoint)
    8: 11, 9: 12, # Plate on counter (closest and midpoint)
    10: 14, 11: 15, # Tomato_cut on counter (closest and midpoint)
    12: 17, 13: 18, # Tomato_salad on counter (closest and midpoint)
}

ACTIONS_OBSERVATION_MAPPING_COMPETITION = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, # Three dispensers, cutting board, delivery
    5: 6, 6: 7, # Free counter (closest and midpoint)
    7: 9, 8: 10, # Tomato on counter (closest and midpoint)
    9: 12, 10: 13, # Pumpkin on counter (closest and midpoint)
    11: 15, 12: 16, # Plate on counter (closest and midpoint)
    13: 18, 14: 19, # Tomato_cut on counter (closest and midpoint)
    15: 21, 16: 22, # Pumpkin_cut on counter (closest and midpoint)
    17: 24, 18: 25, # Tomato_salad on counter (closest and midpoint)
    19: 26, 20: 27, # Pumpkin_salad on counter (closest and midpoint)
}

class DistanceMatrixWrapper:
    """
    Lightweight wrapper that provides the same .get(from).get(to) lookup
    semantics as the original nested dict, but backed by a numpy matrix
    and index maps for speed.
    """
    def __init__(self, D, pos_from, pos_to):
        # D: numpy array shape (N_from, N_to) with np.nan for missing
        self.D = D
        # pos_from / pos_to are arrays of shape (N,2)
        self.pos_from = [tuple(p) for p in pos_from.tolist()]
        self.pos_to = [tuple(p) for p in pos_to.tolist()]
        self.pos_from_idx = {p: i for i, p in enumerate(self.pos_from)}
        self.pos_to_idx = {p: i for i, p in enumerate(self.pos_to)}

    def get(self, from_xy, default=None):
        i = self.pos_from_idx.get(from_xy)
        if i is None:
            return {}
        # Return a dict-like object for compatibility: mapping to_xy -> distance
        row = self.D[i]
        result = {}
        for j, val in enumerate(row):
            if not np.isnan(val):
                result[self.pos_to[j]] = float(val)
        return result


def init_game(agents, map_nr=1, grid_size=(8, 8), seed=None, game_mode="classic", walking_speeds=None, cutting_speeds=None):
    num_agents = len(agents)
    game = SpoiledBroth(map_nr=map_nr, grid_size=grid_size, num_agents=num_agents, seed=seed, walking_speeds=walking_speeds, cutting_speeds=cutting_speeds)
    clickable_indices = game.clickable_indices
    # New action space: fixed action space for RL agents
    action_spaces = {
        agent: spaces.Discrete(len(get_rl_action_space(game_mode)))
        for agent in agents
    }
    _clickable_mask = np.zeros(game.grid.width * game.grid.height, dtype=np.int8)
    for idx in clickable_indices:
        _clickable_mask[idx] = 1
    for agent_id in agents:
        walk_speed = walking_speeds.get(agent_id, 1) if walking_speeds else 1
        cut_speed = cutting_speeds.get(agent_id, 1) if cutting_speeds else 1
        game.add_agent(agent_id, walk_speed, cut_speed)
        agent = game.gameObjects[agent_id]
        if hasattr(agent, 'game'):
            agent.game.clickable_indices = clickable_indices
    return game, action_spaces, _clickable_mask, clickable_indices

class GameEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "game_v0"}

    def __init__(
        self, 
        reward_weights=None, 
        map_nr=1, 
        game_mode="classic",
        inner_seconds=180,
        path="training_stats.csv",
        grid_size=(8, 8),
        payoff_matrix=[1,1,-2],
        initial_seed=0,
        wait_for_completion=True,  # New parameter to control action completion waiting
        start_episode=0,
        walking_speeds=None,
        cutting_speeds=None,
        distance_map=None,
        penalties_cfg=None,
        rewards_cfg=None,
        dynamic_rewards_cfg=None,
        collision_enabled=False  # New parameter for collision detection
    ):
        super().__init__()
        self.map_nr = map_nr
        self.episode_count = start_episode
        self.game_mode = game_mode
        self._max_seconds_per_episode = inner_seconds
        self._elapsed_time = 0.0
        self.render_mode = None
        self.write_header = True
        self.write_csv = False
        self.csv_path = os.path.join(path, "training_stats.csv")
        self.grid_size = grid_size
        self.seed = initial_seed
        self.payoff_matrix = payoff_matrix
        self.walking_speeds = walking_speeds
        self.cutting_speeds = cutting_speeds
        	
        # Initialize penalties and rewards
        default_penalties_cfg = {
            "busy": 0.01,
            "useless_action": 0.2,
            "destructive_action": 1.0,
            "not_available": 0.5,
            "inaccessible_tile": 1.0,
        }
        default_rewards_cfg = {
            "raw_food": 0.2,
            "plate": 0.2,
            "counter": 0.5,
            "cut": 2.0,
            "salad": 5.0,
            "deliver": 10.0,
        }
        self.penalties_cfg = penalties_cfg if penalties_cfg is not None else default_penalties_cfg
        self.rewards_cfg = rewards_cfg if rewards_cfg is not None else default_rewards_cfg
        self.initial_rewards_cfg = self.rewards_cfg.copy()  # Store initial rewards for dynamic updates
        self.dynamic_rewards_cfg = dynamic_rewards_cfg
        self.wait_for_action_completion = wait_for_completion

        self.clickable_indices = None  # Initialize clickable indices storage
        
        self.distance_map = None
        if isinstance(distance_map, str):
            # Only support .npz path strings now
            try:
                # direct path
                if distance_map.endswith('.npz') and os.path.exists(distance_map):
                    data = np.load(distance_map)
                    D = data['D']
                    pos_from = data['pos_from']
                    pos_to = data['pos_to']
                    self.distance_map = DistanceMatrixWrapper(D, pos_from, pos_to)
                else:
                    # Try the repo cache folder
                    possible = os.path.join(os.path.dirname(__file__), '..', 'maps', 'distance_cache', distance_map)
                    if possible.endswith('.npz') and os.path.exists(possible):
                        data = np.load(possible)
                        D = data['D']
                        pos_from = data['pos_from']
                        pos_to = data['pos_to']
                        self.distance_map = DistanceMatrixWrapper(D, pos_from, pos_to)
                    else:
                        # if not found or not .npz, set None
                        self.distance_map = None
            except Exception:
                self.distance_map = None
        else:
            # allow direct dict or wrapper being passed (for tests); otherwise None
            self.distance_map = distance_map
                    
        # Load the accessibility map for this map
        self.accessibility_map = get_accessibility_map(map_nr)
                    
        # Initialize path processing system
        self.path_processor = PathProcessor(map_nr, collision_enabled)

        # Determine agent IDs from reward_weights or default to two agents
        if reward_weights is not None:
            self.possible_agents = list(reward_weights.keys())
        else:
            self.possible_agents = ["ai_rl_1", "ai_rl_2"]
        self.agents = self.possible_agents[:]

        default_weights = {agent: (1.0, 0.0) for agent in self.agents}
        self.reward_weights = reward_weights if reward_weights is not None else default_weights
        self.wait_for_completion = wait_for_completion
        self.cumulated_pure_rewards = {agent: 0.0 for agent in self.agents}
        self.cumulated_modified_rewards = {agent: 0.0 for agent in self.agents}

        if self.game_mode == "competition":
            self.total_agent_events = {agent_id: {"deliver_own": 0, "deliver_other": 0, "salad_own": 0, "salad_other": 0, "cut_own": 0, "cut_other": 0, "plate": 0, "raw_food_own": 0, "raw_food_other": 0, "counter": 0} for agent_id in self.agents}
            self.action_obs_mapping = ACTIONS_OBSERVATION_MAPPING_COMPETITION
            # Assign food types dynamically based on actual agents
            food_types = ["tomato", "pumpkin"]
            self.agent_food_type = {
                agent_id: food_types[i % len(food_types)] 
                for i, agent_id in enumerate(sorted(self.agents))
            }
        elif self.game_mode == "classic":
            self.total_agent_events = {agent_id: {"deliver": 0, "salad": 0, "cut": 0, "plate": 0, "raw_food": 0, "counter": 0} for agent_id in self.agents}
            self.action_obs_mapping = ACTIONS_OBSERVATION_MAPPING_CLASSIC
        else:
            raise ValueError(f"Unknown game mode: {self.game_mode}")
        
        self.total_action_types = {
            agent_id: {action_type: 0 for action_type in get_action_type_list(self.game_mode)}
            for agent_id in self.agents
        }
        self.total_actions_asked = {agent_id: 0 for agent_id in self.agents}
        self.total_action_blocked = {agent_id: 0 for agent_id in self.agents}
        self.total_actions_not_available = {agent_id: 0 for agent_id in self.agents}
        self.total_actions_inaccessible = {agent_id: 0 for agent_id in self.agents}

        self.game, self.action_spaces, self._clickable_mask, self.clickable_indices = init_game(self.agents, map_nr=self.map_nr, grid_size=self.grid_size, seed=self.seed, game_mode=self.game_mode, walking_speeds=self.walking_speeds, cutting_speeds=self.cutting_speeds)

        self.agent_map = {agent_id: self.game.gameObjects[agent_id] for agent_id in self.agents}
        self.busy_until = {agent_id: None for agent_id in self.agents}
        self.action_info = {agent_id: None for agent_id in self.agents}

        # --- New observation space---
        obs_vector, _, _ = game_to_obs_vector(self.game, self.agents[0], game_mode=self.game_mode, path_processor=self.path_processor)
        obs_size = obs_vector.size
        self.observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
            for agent in self.agents
        }

        self.observations = {agent: np.zeros((obs_size,), dtype=np.float32) for agent in self.agents}
        self.modified_rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._last_score = 0

        self.obs_actions_mapping = {}
        
        # Initialize pre-calculated paths storage
        self.agent_action_paths = {agent_id: [] for agent_id in self.agents}
        self.agent_action_tiles = {agent_id: [] for agent_id in self.agents}

    def reset(self, seed=None, options=None):
        # Initialize or increment reset counter
        if not hasattr(self, '_reset_count'):
            self._reset_count = 0
        self._reset_count += 1

        # Create a unique seed by combining fixed seed and reset counter
        episode_seed = (self.seed + self._reset_count) if self.seed is not None else None

        self.agents = self.possible_agents[:]

        self.game, self.action_spaces, self._clickable_mask, self.clickable_indices = init_game(self.agents, map_nr=self.map_nr, grid_size=self.grid_size, seed=episode_seed, game_mode=self.game_mode, walking_speeds=self.walking_speeds, cutting_speeds=self.cutting_speeds)
        self.game.clickable_indices = self.clickable_indices
        random_game_state(self.game)

        self.agent_map = {agent_id: self.game.gameObjects[agent_id] for agent_id in self.agents}
        self.busy_until = {agent_id: None for agent_id in self.agents}
        self.action_info = {agent_id: None for agent_id in self.agents}

        # Initialize agent busy states
        for agent_id, agent in self.agent_map.items():
            if hasattr(agent, 'path'):
                agent.path = []
            if hasattr(agent, 'path_index'):
                agent.path_index = 0

        self.cumulated_pure_rewards = {agent: 0.0 for agent in self.agents}
        self.cumulated_modified_rewards = {agent: 0.0 for agent in self.agents}

        self._elapsed_time = 0.0

        # Update rewards dynamically if configured
        self._update_dynamic_rewards()

        # Clear path processor state for new episode
        if hasattr(self.path_processor, 'active_paths'):
            self.path_processor.active_paths.clear()
            
        # Clear pre-calculated paths
        self.agent_action_paths = {agent_id: [] for agent_id in self.agents}
        self.agent_action_tiles = {agent_id: [] for agent_id in self.agents}

        if self.game_mode == "competition":
            self.total_agent_events = {agent_id: {"deliver_own": 0, "deliver_other": 0, "salad_own": 0, "salad_other": 0, "cut_own": 0, "cut_other": 0, "plate": 0, "raw_food_own": 0, "raw_food_other": 0, "counter": 0} for agent_id in self.agents}
            # Agent food type already set in __init__, no need to reassign
        elif self.game_mode == "classic":
            self.total_agent_events = {agent_id: {"deliver": 0, "salad": 0, "cut": 0, "plate": 0, "raw_food": 0, "counter": 0} for agent_id in self.agents}
            self.agent_food_type = None
        else:
            raise ValueError(f"Unknown game mode: {self.game_mode}")

        self.total_action_types = {
            agent_id: {action_type: 0 for action_type in get_action_type_list(self.game_mode)}
            for agent_id in self.agents
        }
        self.total_actions_asked = {agent_id: 0 for agent_id in self.agents}
        self.total_actions_not_available = {agent_id: 0 for agent_id in self.agents}
        self.total_actions_inaccessible = {agent_id: 0 for agent_id in self.agents}

        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.modified_rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._last_score = 0

        return self.observations, self.infos

    def _update_dynamic_rewards(self):
        """Update rewards configuration based on dynamic rewards settings and current episode."""
        if self.dynamic_rewards_cfg is not None and self.dynamic_rewards_cfg.get("enabled", False):
            self.rewards_cfg = calculate_dynamic_rewards(
                self.episode_count,
                self.initial_rewards_cfg,
                self.dynamic_rewards_cfg
            )
            
            # Log reward changes periodically
            if self.episode_count % 1000 == 0:  # Log every 1000 episodes
                affected_rewards = self.dynamic_rewards_cfg.get("affected_rewards", [])
                if affected_rewards:
                    print(f"[Episode {self.episode_count}] Dynamic rewards updated:")
                    for reward_type in affected_rewards:
                        if reward_type in self.rewards_cfg:
                            initial_val = self.initial_rewards_cfg[reward_type]
                            current_val = self.rewards_cfg[reward_type]
                            print(f"  {reward_type}: {initial_val:.3f} -> {current_val:.3f} (ratio: {current_val/initial_val:.3f})")

    def observe(self, agent):
        obs_vector, considered_paths, considered_tiles = game_to_obs_vector(self.game, agent, game_mode=self.game_mode, path_processor=self.path_processor)
        
        # Store pre-calculated paths and tiles for this agent
        self.agent_action_paths[agent] = considered_paths
        self.agent_action_tiles[agent] = considered_tiles
        obs = obs_vector.flatten().astype(np.float32)
        return obs

    def step(self, actions):
        # --- Simultaneous agent update for minimal delta_time ---
        self.agent_map = {agent_id: self.game.gameObjects[agent_id] for agent_id in self.agents}
        agent_penalties = {agent_id: 0.0 for agent_id in self.agents}
        # Prepare agent_events dict
        if self.game_mode == "competition":
            agent_events = {agent_id: {"deliver_own": 0, "deliver_other": 0, "salad_own": 0, "salad_other": 0, "cut_own": 0, "cut_other": 0, "plate": 0, "raw_food_own": 0, "raw_food_other": 0, "counter": 0} for agent_id in self.agents}
        elif self.game_mode == "classic":
            agent_events = {agent_id: {"deliver": 0, "salad": 0, "cut": 0, "plate": 0, "raw_food": 0, "counter": 0} for agent_id in self.agents}
        else:
            raise ValueError(f"Unknown game mode: {self.game_mode}")

        # Track current actions and busy times
        busy_times = {}
        validated_actions = {}
        
        # Store validated actions for debug access (used by GameEnvDebug)
        self._logging_actions = {}
        
        # First, calculate remaining busy times for all agents
        for agent_id in self.agents:
            agent = self.agent_map[agent_id]
            if self.busy_until.get(agent_id) is not None and self.busy_until[agent_id] > self._elapsed_time:
                busy_times[agent_id] = self.busy_until[agent_id] - self._elapsed_time
            else:
                busy_times[agent_id] = 0  # Agent is not busy
        
        # Only process actions for agents that are not busy
        for agent_id, action_idx in actions.items():
            agent = self.agent_map[agent_id]

            # Check if agent is ready for a new action
            if busy_times[agent_id] <= 0:
                self.total_actions_asked[agent_id] += 1
                action_name = get_rl_action_space(self.game_mode)[action_idx]
                
                if agent_id in self.agent_action_tiles:        
                    cached_path = self.agent_action_paths[agent_id][action_idx]
                    tile_index = self.agent_action_tiles[agent_id][action_idx]
                else:
                    cached_path = None
                    tile_index = None
                
                if tile_index is None:
                    # No valid tile found during observation (inaccessible)
                    action_type = "inaccessible_tile"
                    logging_index = -1
                    logging_x, logging_y = -1, -1
                elif tile_index == -1:
                    # Path blocked by collisions
                    action_type = "not_available"
                    logging_index = -1
                    logging_x, logging_y = -1, -1
                else:
                    # Action is valid - set up the path for the agent
                    grid_w = self.game.grid.width
                    x = tile_index % grid_w
                    y = tile_index // grid_w
                    tile = self.game.grid.tiles[x][y]
                    action_type = get_action_type(tile, agent, agent_id, agent_food_type=self.agent_food_type, game_mode=self.game_mode, x=x, y=y, accessibility_map=self.accessibility_map)
                    logging_index = tile_index
                    logging_x = x
                    logging_y = y
                    
                    # Set up the cached path for the agent
                    if cached_path and len(cached_path) > 1:
                        agent.path = cached_path[1:]  # Skip current tile
                        agent.path_index = 0
                    else:
                        # Fallback to setup_agent_path if no cached path
                        setup_agent_path(self, agent, tile_index)
                    
                    # Update path processor with agent's active path
                    if self.path_processor.is_enabled() and hasattr(agent, 'path') and agent.path:
                        # Convert agent speed from pixels/second to tiles/second for collision detection
                        agent_speed_pixels = getattr(agent, 'speed', 30.0)
                        agent_speed_tiles = agent_speed_pixels / 16.0  # tile_size is 16 pixels
                        self.path_processor.update_agent_path(
                            agent_id=agent_id,
                            path=agent.path,
                            current_position=(agent.slot_x, agent.slot_y),
                            path_index=0,
                            speed=agent_speed_tiles
                        )
                    
                    # Set up action info for completion tracking
                    self.action_info[agent_id] = {
                        'tile_index': tile_index,
                        'action_name': action_name,
                        'action_type': action_type
                    }
                obs = self.observations.get(agent_id)
                
                # Store validated action for debug access
                self._logging_actions[agent_id] = {
                    'elapsed_time': self._elapsed_time,
                    'action_idx': action_idx,
                    'action_name': action_name,
                    'tile_index': logging_index,
                    'action_type': action_type,
                    'x': logging_x,
                    'y': logging_y
                }

                if action_type == "inaccessible_tile":
                    # No valid target found - action cannot be performed
                    self.total_actions_inaccessible[agent_id] += 1
                    self.total_action_types[agent_id][action_type] += 1
                    agent_penalties[agent_id] += self.penalties_cfg["inaccessible_tile"]
                    busy_times[agent_id] = MOVE_TIME
                    self.busy_until[agent_id] = self._elapsed_time + busy_times[agent_id]
                    continue
                elif action_type == "not_available":
                    self.total_actions_not_available[agent_id] += 1
                    self.total_action_types[agent_id][action_type] += 1
                    agent_penalties[agent_id] += self.penalties_cfg["not_available"]
                    busy_times[agent_id] = MOVE_TIME
                    self.busy_until[agent_id] = self._elapsed_time + busy_times[agent_id]
                    continue
                
                # Calculate movement time based on distance, but ensure minimum time
                if obs is not None and action_idx in self.action_obs_mapping:
                    distance_normalized = obs[self.action_obs_mapping[action_idx]]
                    move_time = distance_normalized * getattr(self.game, 'normalization_factor', 1.0)
                else:
                    move_time = MOVE_TIME
                
                # Add intent time - for cutting actions, use agent-specific cutting time
                intent_time = INTENT_TIME
                if action_name == "use_cutting_board" and agent_id in self.agent_map:
                    agent = self.agent_map[agent_id]
                    intent_time = get_cutting_time(agent, self.game)
                
                busy_time = move_time + intent_time
                self.busy_until[agent_id] = self._elapsed_time + busy_time
                busy_times[agent_id] = busy_time

                # Penalties
                if action_type.startswith("useless_"):
                    agent_penalties[agent_id] += self.penalties_cfg["useless_action"]
                elif action_type.startswith("destructive_"):
                    # Get the penalty for the destroyed item based on what the agent is carrying
                    destroyed_item_penalty = 0.0
                    if hasattr(agent, 'item') and agent.item:
                        # Map agent's item to corresponding reward value
                        if agent.item in ["tomato", "pumpkin"]:
                            destroyed_item_penalty = self.rewards_cfg["raw_food"]
                        elif agent.item == "plate":
                            destroyed_item_penalty = self.rewards_cfg["plate"]
                        elif agent.item in ["tomato_cut", "pumpkin_cut"]:
                            destroyed_item_penalty = self.rewards_cfg["cut"]
                        elif agent.item in ["tomato_salad", "pumpkin_salad"]:
                            destroyed_item_penalty = self.rewards_cfg["salad"]

                    # Apply both the base destructive penalty and the destroyed item penalty
                    agent_penalties[agent_id] += self.penalties_cfg["destructive_action"] + destroyed_item_penalty
                
                agent_penalties[agent_id] += self.penalties_cfg["busy"] * busy_time
                validated_actions[agent_id] = {"type": "click", "target": tile_index}
                self.action_info[agent_id] = {
                    "action_type": action_type,
                    "tile_index": tile_index,
                    "action_idx": action_idx
                }
                
                # Track action types immediately when selected (for all actions, including useless ones)
                self.total_action_types[agent_id][action_type] += 1
            else:
                # Agent is busy, ignore the new action and continue with current action
                pass

        # We need to advance time until at least one agent becomes available for a new action
        all_busy_times = [t for t in busy_times.values() if t > 0]

        if all_busy_times:
            # Advance to when the first agent finishes
            min_delta = min(all_busy_times)
        else:
            # All agents are ready, minimal advancement
            min_delta = 0.01
        
        next_time = min(self._elapsed_time + min_delta, self._max_seconds_per_episode)
        advanced_time = next_time - self._elapsed_time
        self._elapsed_time = next_time

        # Direct game state update instead of calling self.game.step()
        agent_events = update_agents_directly(self, advanced_time, agent_events, agent_food_type=self.agent_food_type, game_mode=self.game_mode)
        self.agent_map = {agent_id: self.game.gameObjects[agent_id] for agent_id in self.agents}

        # Update totals for actions that just completed (for logging purposes only)
        # We should use the events that were already recorded during action selection, not re-evaluate
        for agent_id in self.agents:
            for event_type in agent_events[agent_id]:
                if agent_events[agent_id][event_type] > 0:
                    self.total_agent_events[agent_id][event_type] += agent_events[agent_id][event_type]

        # Mark agents as available if their busy_until has passed
        for agent_id, agent in self.agent_map.items():
            if self.busy_until.get(agent_id) is not None and self._elapsed_time >= self.busy_until[agent_id]:
                self.busy_until[agent_id] = None
                self.action_info[agent_id] = None
                
                # Clear agent path when action completes
                if hasattr(agent, 'path'):
                    agent.path = []
                if hasattr(agent, 'path_index'):
                    agent.path_index = 0

        # Compute rewards
        self.cumulated_pure_rewards, self.cumulated_modified_rewards = get_rewards(self, agent_events, agent_penalties, self.rewards_cfg)

        # Check for episode termination
        should_truncate = self._elapsed_time >= self._max_seconds_per_episode
        if should_truncate:
            self.dones = {agent: True for agent in self.agents}
            self._elapsed_time = 0
            self.write_csv = True

        self.infos = {
            agent: {
                "agent_events": agent_events[agent],
                "action_types": self.total_action_types[agent],
                "score": self.agent_map[agent].score
            }
            for agent in self.agents
        }

        self.observations = {agent: self.observe(agent) for agent in self.agents}
        terminations = self.dones
        truncations = {agent: False for agent in self.agents}

        # If episode is done, aggregate and log
        if self.write_csv:
            if self.episode_count % 100 == 0:
                print(f"[Episode {self.episode_count}] Logging episode data to csv")
            row = {"episode": self.episode_count}
            for agent_id in self.agents:
                row[f"pure_reward_{agent_id}"] = float(self.cumulated_pure_rewards[agent_id])
                row[f"modified_reward_{agent_id}"] = float(self.cumulated_modified_rewards[agent_id])

                for result_event in self.total_agent_events[agent_id]:
                    row[f"{result_event}_{agent_id}"] = self.total_agent_events[agent_id][result_event]
                row[f"actions_asked_{agent_id}"] = self.total_actions_asked[agent_id]
                row[f"actions_not_available_{agent_id}"] = self.total_actions_not_available[agent_id]
                row[f"inaccessible_actions_{agent_id}"] = self.total_actions_inaccessible[agent_id]

                for action_type in get_action_type_list(self.game_mode):
                    row[f"{action_type}_{agent_id}"] = self.total_action_types[agent_id][action_type]

            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if self.write_header:
                    writer.writeheader()
                    self.write_header = False
                writer.writerow(row)

            self.episode_infos_log = {agent: [] for agent in self.agents}
            self.episode_count += 1
            self.write_csv = False

        return self.observations, self.modified_rewards, terminations, truncations, self.infos

    def render(self):
        print(f"[Game Render] Agents: {self.agents}")

    def close(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]
