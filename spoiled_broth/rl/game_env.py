import csv
import os
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
import pandas as pd  # For loading training_stats.csv
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
from spoiled_broth.rl.tick_based_structure import (
    agent_is_idle, assign_action, cancel_agent_action,
    advance_agent_movement, resolve_predictive_collisions, update_agent_interactions
)

# Tick-based simulation constants
TICK_DURATION = 0.2  # Fixed time step in seconds (200ms) - minimum wait time for ANY action

# Base intent time for non-cutting actions (1 tick minimum)
INTENT_TIME = TICK_DURATION  # 0.2s = 1 tick for pickup/delivery/put_down

# TIME MANAGEMENT SUMMARY:
# =======================
# 
# TICK DURATION (0.2s):
#   - Minimum time for ANY action or wait state
#   - Idle agents wait at least 1 tick before next action
#   - Rejected actions (inaccessible/not_available) wait 1 tick
#   - All times are multiples of TICK_DURATION
#
# MOVEMENT TIME (continuous, based on agent.walk_speed):
#   - Base speed: 30 pixels/second = 1.875 tiles/second
#   - Scaled by walk_speed: agent.speed = 30 * walk_speed
#   - walk_speed=1.0 → 0.533 seconds per tile (~2.7 ticks)
#   - walk_speed=0.5 → 1.067 seconds per tile (~5.3 ticks, slower)
#   - walk_speed=2.0 → 0.267 seconds per tile (~1.3 ticks, faster)
#   - Progress tracked per tick: movement_progress += (speed * TICK_DURATION)
#
# INTERACTION TIME (fixed duration after reaching destination):
#   - CUTTING: base_time / cut_speed (default 3.0s / cut_speed)
#     * cut_speed=1.0 → 3.0 seconds (15 ticks)
#     * cut_speed=0.5 → 6.0 seconds (30 ticks, slower)
#     * cut_speed=2.0 → 1.5 seconds (7.5 ticks, faster)
#   - OTHER ACTIONS: INTENT_TIME = 0.2 seconds (1 tick)
#     * Pickup, delivery, put_down all use 1 tick
#
# AGENT IDLE STATE:
#   Agent is idle ONLY when ALL of these are true:
#   1. current_action is None (no action assigned)
#   2. interaction_timer <= 0 (finished waiting for intent)
#   3. current_path is empty (not moving)
#   
#   This ensures agents complete BOTH movement AND interaction time
#   before being allowed to take a new action.
#   
#   When actions are rejected or do_nothing is selected, agents
#   remain idle and must wait until next tick to request new action.

def get_cutting_time(agent, game):
    """Calculate actual cutting time based on agent's cutting speed."""
    cutting_speed = getattr(agent, 'cut_speed', 1.0)
    base_cutting_time = getattr(game, 'cutting_time', 3.0)
    # Lower speed = more time (inverse relationship)
    return base_cutting_time / cutting_speed if cutting_speed > 0 else base_cutting_time

ACTIONS_OBSERVATION_MAPPING_CLASSIC = {
    # 0: do_nothing - no observation mapping needed
    # Tile types: [accessibility, time] for each type (indices 0-7)
    1: 1,  # pick_up_tomato_from_dispenser → tile_types[0] time_value
    2: 3,  # pick_up_plate_from_dispenser → tile_types[1] time_value
    3: 5,  # use_cutting_board → tile_types[2] time_value
    4: 7,  # use_delivery → tile_types[3] time_value
    # Items on counters: [presence, accessibility_closest, time_closest, accessibility_midpoint, time_midpoint]
    5: 10, 6: 12,   # put_down_item_on_free_counter → items[0] (None) closest/midpoint time
    7: 15, 8: 17,   # pick_up_tomato_from_counter → items[1] (tomato) closest/midpoint time
    9: 20, 10: 22,  # pick_up_plate_from_counter → items[2] (plate) closest/midpoint time
    11: 25, 12: 27,  # pick_up_tomato_cut_from_counter → items[3] (tomato_cut) closest/midpoint time
    13: 30, 14: 32,  # pick_up_tomato_salad_from_counter → items[4] (tomato_salad) closest/midpoint time
}

ACTIONS_OBSERVATION_MAPPING_COMPETITION = {
    # 0: do_nothing - no observation mapping needed
    # Tile types: [accessibility, time] for each type (5 types, indices 0-9)
    1: 1,  # pick_up_tomato_from_dispenser → tile_types[0] time_value
    2: 3,  # pick_up_pumpkin_from_dispenser → tile_types[1] time_value
    3: 5,  # pick_up_plate_from_dispenser → tile_types[2] time_value
    4: 7,  # use_cutting_board → tile_types[3] time_value
    5: 9,  # use_delivery → tile_types[4] time_value
    # Items on counters: [presence, accessibility_closest, time_closest, accessibility_midpoint, time_midpoint]
    6: 12, 7: 14,   # put_down_item_on_free_counter → items[0] (None) closest/midpoint time
    8: 17, 9: 19,   # pick_up_tomato_from_counter → items[1] (tomato) closest/midpoint time
    10: 22, 11: 24,  # pick_up_pumpkin_from_counter → items[2] (pumpkin) closest/midpoint time
    12: 27, 13: 29,  # pick_up_plate_from_counter → items[3] (plate) closest/midpoint time
    14: 32, 15: 34,  # pick_up_tomato_cut_from_counter → items[4] (tomato_cut) closest/midpoint time
    16: 37, 17: 39,  # pick_up_pumpkin_cut_from_counter → items[5] (pumpkin_cut) closest/midpoint time
    18: 42, 19: 44,  # pick_up_tomato_salad_from_counter → items[6] (tomato_salad) closest/midpoint time
    20: 47, 21: 49,  # pick_up_pumpkin_salad_from_counter → items[7] (pumpkin_salad) closest/midpoint time
}

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
        collision_enabled=False,  # New parameter for collision detection
        random_initial_state=False,  # New parameter to randomize initial game state
        reference_reward_cfg=None,  # Reference-based opportunity cost shaping
        solo_baselines=None  # Individual solo baselines for reference reward (dict: agent_id -> baseline)
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
            "collision": 0.5,  # Penalty when collision cannot be rerouted
            "specialization_penalty_scale": 0.0,  # Specialization penalty scale (lambda): 0=no penalty, >0=penalty scale
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
        self.random_initial_state = random_initial_state  # Store flag for random initial states
        
        self.clickable_indices = None  # Initialize clickable indices storage
        
        # Note: distance_map parameter is no longer used. Time normalization uses
        # max_distance loaded in game.py (from distance_map_{map_id}_max_distance.npy)
        # and observation space calculates paths online using PathProcessor + A*.
                    
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
        
        # Collision tracking statistics
        self.total_collisions_detected = 0
        self.total_collisions_rerouted = 0
        self.total_collisions_failed = 0

        self.game, self.action_spaces, self._clickable_mask, self.clickable_indices = init_game(self.agents, map_nr=self.map_nr, grid_size=self.grid_size, seed=self.seed, game_mode=self.game_mode, walking_speeds=self.walking_speeds, cutting_speeds=self.cutting_speeds)

        self.agent_map = {agent_id: self.game.gameObjects[agent_id] for agent_id in self.agents}
        
        # Tick-based agent state tracking (replaces busy_until)
        self.agent_state = {
            agent_id: {
                'current_action': None,  # Action name being executed
                'current_path': [],  # Path nodes for current action
                'path_index': 0,  # Current position in path
                'movement_progress': 0.0,  # Fractional progress to next tile [0, 1)
                'interaction_timer': 0.0,  # Time remaining for non-movement actions (cutting, pickup, etc.)
                'target_tile_index': None,  # Final tile for current action
                'action_type': None  # Type classification of current action
            }
            for agent_id in self.agents
        }
        
        # Legacy compatibility (will be removed after refactor complete)
        self.busy_until = {agent_id: None for agent_id in self.agents}
        self.action_info = {agent_id: None for agent_id in self.agents}

        # Team synergy-based shaping mechanism
        self.reference_reward_cfg = reference_reward_cfg if reference_reward_cfg is not None else {"enabled": False}
        self.solo_baselines = solo_baselines if solo_baselines is not None else {}
        self.solo_baseline_team = sum(self.solo_baselines.values()) if self.solo_baselines else None
        self.reference_reward_enabled = self.reference_reward_cfg.get("enabled", False) and self.solo_baselines
        if self.reference_reward_enabled:
            self.synergy_scaling_factor = self.reference_reward_cfg.get("synergy_scaling_factor", 0.5)
            print(f"[GameEnv] Team synergy enabled: synergy_scaling_factor={self.synergy_scaling_factor}, baselines={self.solo_baselines}, team={self.solo_baseline_team}")
            # Calculate competence using agent abilities (kappa values)
            self.agent_abilities = {}
            for agent_id in self.agents:
                # Get cutting and walking abilities (kappa values)
                cut_speed = self.cutting_speeds.get(agent_id, 1.0) if self.cutting_speeds else 1.0
                walk_speed = self.walking_speeds.get(agent_id, 1.0) if self.walking_speeds else 1.0
                self.agent_abilities[agent_id] = {
                    'cutting': cut_speed,
                    'walking': walk_speed,
                    'average': (cut_speed + walk_speed) / 2.0
                }
            print(f"[GameEnv] Agent abilities: {self.agent_abilities}")

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
        
        # Track which agents need observation refresh (only when they become idle)
        # This avoids recalculating expensive pathfinding when agents are busy
        self.agents_need_observation = {agent_id: True for agent_id in self.agents}

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
        
        # Only randomize initial state if flag is enabled
        if self.random_initial_state:
            random_game_state(self.game, game_mode=self.game_mode)

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
        
        # Reset collision statistics
        self.total_collisions_detected = 0
        self.total_collisions_rerouted = 0
        self.total_collisions_failed = 0

        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.modified_rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._last_score = 0
        
        # All agents start idle, so they all need observations
        self.agents_need_observation = {agent_id: True for agent_id in self.agents}

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
        """
        Tick-based simulation step with PREDICTIVE collision handling.
        
        NEW PROCESSING ORDER:
        1. Process new actions ONLY for idle agents
        2. PREDICTIVE collision detection: predict next tick positions and reroute BEFORE movement
        3. Execute one tick (all agents move/interact)
        4. Update observations ONLY for agents that became idle this tick
        
        OPTIMIZATION: Observations are only calculated for agents that CAN take actions.
        - Busy agents: Skip action processing, skip observation calculation
        - Idle agents: Process actions, calculate fresh observations for next step
        
        This avoids expensive pathfinding calculations (~80% of observation cost)
        when agents are executing actions and cannot make decisions.
        """
        # Initialize agent map and event tracking
        self.agent_map = {agent_id: self.game.gameObjects[agent_id] for agent_id in self.agents}
        agent_penalties = {agent_id: 0.0 for agent_id in self.agents}
        
        # Track which agents will need fresh observations after this step
        # (agents that are currently idle or will become idle during this tick)
        agents_becoming_idle = set()
        
        # Prepare agent_events dict
        if self.game_mode == "competition":
            agent_events = {agent_id: {"deliver_own": 0, "deliver_other": 0, "salad_own": 0, "salad_other": 0, "cut_own": 0, "cut_other": 0, "plate": 0, "raw_food_own": 0, "raw_food_other": 0, "counter": 0} for agent_id in self.agents}
        elif self.game_mode == "classic":
            agent_events = {agent_id: {"deliver": 0, "salad": 0, "cut": 0, "plate": 0, "raw_food": 0, "counter": 0} for agent_id in self.agents}
        else:
            raise ValueError(f"Unknown game mode: {self.game_mode}")
        
        # Store validated actions for debug access (used by GameEnvDebug)
        self._logging_actions = {}
        
        # --- Phase 1: Process new actions ONLY for idle agents ---
        # Busy agents are skipped - they already have actions executing
        for agent_id, action_idx in actions.items():
            # CRITICAL: Only process actions for idle agents
            # Busy agents cannot make decisions, so we skip them entirely
            is_idle = agent_is_idle(self, agent_id)
            if not is_idle:
                continue
            
            self.total_actions_asked[agent_id] += 1
            agent = self.agent_map[agent_id]
            action_name = get_rl_action_space(self.game_mode)[action_idx]
            
            # Handle do_nothing action
            if action_name == "do_nothing":
                # Store for logging
                self._logging_actions[agent_id] = {
                    'elapsed_time': self._elapsed_time,
                    'action_idx': action_idx,
                    'action_name': action_name,
                    'tile_index': -2,
                    'action_type': 'do_nothing',
                    'x': -2,
                    'y': -2
                }
                continue
            
            # Get cached path and tile from observation
            if agent_id in self.agent_action_tiles:
                cached_path = self.agent_action_paths[agent_id][action_idx]
                tile_index = self.agent_action_tiles[agent_id][action_idx]
            else:
                cached_path = None
                tile_index = None
            
            # --- THREE-TIER PENALTY SYSTEM ---
            # Validate action based on observation space indicators:
            # 
            # 1. INACCESSIBLE (tile_index=None): No path exists ignoring agents
            #    - Observation: accessibility=0, availability=0
            #    - Action: REJECTED immediately, agent stays idle
            #    - Penalty: penalties_cfg["inaccessible_tile"] (default: 5.0)
            #
            # 2. NOT_AVAILABLE (tile_index=-1): Path exists but blocked by agent's current position
            #    - Observation: accessibility=1, availability=0
            #    - Action: ACCEPTED and attempted (action is assigned to agent)
            #    - Penalty: penalties_cfg["not_available"] (default: 2.0) applied immediately
            #    - Note: Only occurs when collision_enabled=True
            #
            # 3. COLLISION (handled in Phase 2): Predictive collision detected and rerouting failed
            #    - Observation: accessibility=1, availability=1 (was available at observation time)
            #    - Prediction: Agents would collide in next movement tick
            #    - Action: Attempted, collision predicted before execution, rerouting fails
            #    - Penalty: penalties_cfg["collision"] (default: 0.5) applied when rerouting fails
            #    - Note: Only occurs when collision_enabled=True
            
            if tile_index is None:
                # CASE 1: INACCESSIBLE - No path exists at all (ignoring agents)
                # This means the tile is unreachable due to walls/obstacles, not other agents
                action_type = "inaccessible_tile"
                logging_index = -1
                logging_x, logging_y = -1, -1
                self.total_actions_inaccessible[agent_id] += 1
                self.total_action_types[agent_id][action_type] += 1
                agent_penalties[agent_id] += self.penalties_cfg["inaccessible_tile"]
                # Action is REJECTED - agent remains idle, will ask for new action next step
                # Mark that this agent needs a fresh observation for next step
                agents_becoming_idle.add(agent_id)
                
            elif tile_index == -1:
                # CASE 2: NOT_AVAILABLE - Path exists but blocked by other agent's current position
                # Tile is accessible (path exists ignoring agents) but not currently available
                # We ATTEMPT the action (assign it to agent) and let collision detection handle it
                # This only happens when collision_enabled=True
                action_type = "not_available"
                logging_index = -1
                logging_x, logging_y = -1, -1
                self.total_actions_not_available[agent_id] += 1
                self.total_action_types[agent_id][action_type] += 1
                agent_penalties[agent_id] += self.penalties_cfg["not_available"]
                
                # ASSIGN the action with path that ignores agents - let collision detection handle conflicts
                # The cached_path should contain the path ignoring agents (fixed in observation_space.py)
                assign_action(self, agent_id, action_idx, action_name, tile_index, cached_path, action_type)
                
            else:
                # CASE 3: VALID ACTION - Tile is both accessible and available
                # Action is assigned and will be executed
                # COLLISION penalty may be applied later if runtime collision occurs during movement
                grid_w = self.game.grid.width
                x = tile_index % grid_w
                y = tile_index // grid_w
                tile = self.game.grid.tiles[x][y]
                action_type = get_action_type(tile, agent, agent_id, agent_food_type=self.agent_food_type, game_mode=self.game_mode, x=x, y=y, accessibility_map=self.accessibility_map)
                logging_index = tile_index
                logging_x = x
                logging_y = y
                
                # Assign action to agent
                assign_action(self, agent_id, action_idx, action_name, tile_index, cached_path, action_type)
                
                # Track action type
                self.total_action_types[agent_id][action_type] += 1
                
                # Apply immediate penalties for useless/destructive actions
                if action_type.startswith("useless_"):
                    agent_penalties[agent_id] += self.penalties_cfg["useless_action"]
                elif action_type.startswith("destructive_"):
                    # Get penalty for destroyed item
                    destroyed_item_penalty = 0.0
                    if hasattr(agent, 'item') and agent.item:
                        if agent.item in ["tomato", "pumpkin"]:
                            destroyed_item_penalty = self.rewards_cfg["raw_food"]
                        elif agent.item == "plate":
                            destroyed_item_penalty = self.rewards_cfg["plate"]
                        elif agent.item in ["tomato_cut", "pumpkin_cut"]:
                            destroyed_item_penalty = self.rewards_cfg["cut"]
                        elif agent.item in ["tomato_salad", "pumpkin_salad"]:
                            destroyed_item_penalty = self.rewards_cfg["salad"]
                    agent_penalties[agent_id] += self.penalties_cfg["destructive_action"] + destroyed_item_penalty
            
            # Store for logging
            self._logging_actions[agent_id] = {
                'elapsed_time': self._elapsed_time,
                'action_idx': action_idx,
                'action_name': action_name,
                'tile_index': logging_index,
                'action_type': action_type,
                'x': logging_x,
                'y': logging_y
            }
        
        # --- Phase 2: Predictive Collision Detection and Rerouting ---
        # CRITICAL: This must happen BEFORE movement to prevent collisions
        # Instead of detecting collisions after they happen, predict and prevent them
        if self.path_processor.collision_enabled:
            collision_stats = resolve_predictive_collisions(self)
            
            # Update collision statistics
            self.total_collisions_detected += collision_stats['collisions_detected']
            self.total_collisions_rerouted += collision_stats['collisions_rerouted']
            self.total_collisions_failed += collision_stats['collisions_failed']
            
            # Apply penalties to agents whose collisions couldn't be rerouted
            # These agents had their actions cancelled in resolve_predictive_collisions
            for agent_id in collision_stats['agents_with_failed_collisions']:
                agent_penalties[agent_id] += self.penalties_cfg["collision"]
                # Mark these agents for observation refresh since they're now idle
                agents_becoming_idle.add(agent_id)
        # If collision detection is disabled, this entire block is skipped
        # No collision penalties, no rerouting, agents can overlap freely
        
        # --- Phase 3: Execute one tick of simulation ---
        # Advance time by one tick
        self._elapsed_time += TICK_DURATION
        
        # Move all agents
        agents_reached_destination = []
        for agent_id in self.agents:
            state = self.agent_state[agent_id]
            
            # Skip if agent is not moving
            if len(state['current_path']) == 0:
                continue
            
            # Advance movement
            reached_next, reached_final, blocked = advance_agent_movement(self, agent_id, TICK_DURATION)
            
            if reached_final:
                # Agent reached their destination tile
                agents_reached_destination.append(agent_id)
                
                # Start interaction timer for non-movement actions
                action_name = state['current_action']
                agent = self.agent_map[agent_id]
                
                if action_name == "use_cutting_board":
                    # Use agent-specific cutting time
                    state['interaction_timer'] = get_cutting_time(agent, self.game)
                elif action_name:
                    # All other interactions use INTENT_TIME
                    state['interaction_timer'] = INTENT_TIME
                
                # Clear movement path (keep action state for interaction completion)
                state['current_path'] = []
                state['path_index'] = 0
                state['movement_progress'] = 0.0
        
        # Update interaction timers and complete interactions
        # This also returns which agents completed their interactions (became idle)
        agent_events = update_agent_interactions(self, agent_events, agent_penalties, TICK_DURATION)
        
        # Mark agents that just finished interactions as needing new observations
        # Check which agents are now idle (completed their actions this tick)
        for agent_id in self.agents:
            if agent_is_idle(self, agent_id):
                # Agent is idle - either was already idle, or just finished
                # They'll need a fresh observation for the next decision
                agents_becoming_idle.add(agent_id)
        
        # Apply busy penalty for agents currently executing actions
        for agent_id in self.agents:
            state = self.agent_state[agent_id]
            if state['current_action'] is not None:
                agent_penalties[agent_id] += self.penalties_cfg["busy"] * TICK_DURATION
        
        # Apply specialization penalty
        penalty_scale = self.penalties_cfg.get("specialization_penalty_scale", 0.0)
        if penalty_scale > 0:
            for agent_id in self.agents:
                state = self.agent_state[agent_id]
                agent = self.agent_map[agent_id]
                action_type = state.get('action_type')
                
                if action_type and state['current_action']:
                    walk_speed = getattr(agent, 'walk_speed', 1.0)
                    cut_speed = getattr(agent, 'cut_speed', 1.0)
                    
                    # Apply penalty for cutting actions when cut_ability < 1
                    is_cutting_action = (
                        action_type in ["useful_cutting_board", "useful_cutting_board_own", "useful_cutting_board_other"]
                    )
                    if is_cutting_action and cut_speed < 1.0:
                        cutting_penalty = (1.0 - cut_speed) * penalty_scale * TICK_DURATION
                        agent_penalties[agent_id] += cutting_penalty
                    
                    # Apply penalty for delivery/walking actions when walk_speed < 1
                    is_delivery_action = (
                        action_type in ["useful_delivery", "useful_delivery_own", "useful_delivery_other"]
                    )
                    if is_delivery_action and walk_speed < 1.0:
                        delivery_penalty = (1.0 - walk_speed) * penalty_scale * TICK_DURATION
                        agent_penalties[agent_id] += delivery_penalty
        
        # Update totals for logged events
        for agent_id in self.agents:
            for event_type in agent_events[agent_id]:
                if agent_events[agent_id][event_type] > 0:
                    self.total_agent_events[agent_id][event_type] += agent_events[agent_id][event_type]
        
        # Compute rewards (includes reference-based opportunity cost if enabled)
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
        
        # Update observations ONLY for agents that need them
        # OPTIMIZATION: Only recalculate expensive pathfinding observations for agents that:
        # 1. Just finished their action (became idle)
        # 2. Had their action rejected (inaccessible or not available)
        # 3. Had their action cancelled due to collision
        # Busy agents keep their previous observations (they can't act anyway)
        for agent_id in agents_becoming_idle:
            self.observations[agent_id] = self.observe(agent_id)
        
        # Note: observations for busy agents are unchanged from previous step
        # They don't need fresh pathfinding data until they can make decisions
        
        terminations = self.dones
        truncations = {agent: False for agent in self.agents}
        
        # If episode is done, aggregate and log
        if self.write_csv:
            if self.episode_count % 100 == 0:
                print(f"[Episode {self.episode_count}] Logging episode data to csv")
            row = {"episode": self.episode_count}
            
            # Add collision statistics immediately after episode (episode-level, not per-agent)
            row["collisions_detected"] = self.total_collisions_detected
            row["collisions_rerouted"] = self.total_collisions_rerouted
            row["collisions_failed"] = self.total_collisions_failed
            
            for agent_id in self.agents:
                row[f"pure_reward_{agent_id}"] = float(self.cumulated_pure_rewards[agent_id])
                row[f"modified_reward_{agent_id}"] = float(self.cumulated_modified_rewards[agent_id])

                for result_event in self.total_agent_events[agent_id]:
                    row[f"{result_event}_{agent_id}"] = self.total_agent_events[agent_id][result_event]
                row[f"actions_asked_{agent_id}"] = self.total_actions_asked[agent_id]
                row[f"actions_not_available_{agent_id}"] = self.total_actions_not_available[agent_id]
                row[f"inaccessible_actions_{agent_id}"] = self.total_actions_inaccessible[agent_id]

                # Add action type columns for this specific agent
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