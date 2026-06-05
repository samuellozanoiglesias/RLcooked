import csv
import os
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np
import pandas as pd  # For loading training_stats.csv
from spoiled_broth.config import *
from spoiled_broth.maps.accessibility_maps import get_accessibility_map
import pickle as _pickle
# Note: game_step.py contains helper functions used by tick_based_structure.py
from spoiled_broth.rl.action_space import get_rl_action_space
from spoiled_broth.rl.observation_space import game_to_obs_vector
from spoiled_broth.rl.classify_action_type import get_action_type, get_action_type_list
from spoiled_broth.rl.reward_analysis import get_rewards, get_cutting_time, apply_adaptive_cooperation_penalty
from spoiled_broth.game import SpoiledBroth, random_game_state
from spoiled_broth.rl.path_processing import PathProcessor
from spoiled_broth.rl.tick_based_structure import (
    agent_is_idle, assign_action, cancel_agent_action,
    advance_agent_movement, update_agent_interactions
)
from spoiled_broth.rl.tick_based_collision import resolve_predictive_collisions

# Tick-based simulation constants
TICK_DURATION = 0.5  # Fixed time step in seconds (200ms) - minimum wait time for ANY action

# Base intent time for non-cutting actions (1 tick minimum)
INTENT_TIME = TICK_DURATION  # 0.2s = 1 tick for pickup/delivery/put_down

# TIME MANAGEMENT SUMMARY:
# =======================
# 
# TICK DURATION:
#   - Minimum time for ANY action or wait state
#   - Idle agents wait at least 1 tick before next action
#   - Rejected actions (inaccessible/blocked) wait 1 tick
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

def init_game(agents, map_nr=1, grid_size=(8, 8), seed=None, game_mode="classic", walking_speeds=None, cutting_speeds=None):
    num_agents = len(agents)
    game = SpoiledBroth(
        map_nr=map_nr,
        grid_size=grid_size,
        num_agents=num_agents,
        seed=seed,
        walking_speeds=walking_speeds,
        cutting_speeds=cutting_speeds,
        game_version=game_mode,
    )
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
        num_agents=None,
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
        intermediate_reward_decay_cfg=None,  # Configuration for intermediate reward decay
        collision_enabled=False,  # New parameter for collision detection
        random_initial_state=False,  # New parameter to randomize initial game state
        reference_reward_cfg=None,  # Reference-based opportunity cost shaping
        solo_baselines=None,  # Individual solo baselines for reference reward (dict: agent_id -> baseline)
        allow_blocked=False,  # Whether to allow blocked actions to be attempted
        enable_csv_logging=True  # Whether to write training_stats.csv (disable for simulations)
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
        self.enable_csv_logging = enable_csv_logging
        self.csv_path = os.path.join(path, "training_stats.csv") if enable_csv_logging else None
        self.grid_size = grid_size
        self.seed = initial_seed
        self.payoff_matrix = payoff_matrix
        self.walking_speeds = walking_speeds
        self.cutting_speeds = cutting_speeds
        	
        # Initialize penalties and rewards
        default_penalties_cfg = {
            "do_nothing": 1.0,  # Penalty for do_nothing action
            "useless_action": 0.2,
            "destructive_action": 1.0,
            "blocked": 0.5,  # Penalty when path is blocked (by agents if collision_enabled=True)
            "collision": 0.0,  # Penalty when a collision occurs (if collision_enabled=True)
            "inaccessible_tile": 1.0,  # Penalty when no path exists (walls/obstacles/no objects)
            "specialization_penalty_scale": 0.0,  # Specialization penalty scale (lambda): 0=no penalty, >0=penalty scale
            "specialization_theta": 1.0,  # Exponential sensitivity for specialization penalties
            "adaptive_cooperation_scale": 0.0,  # Path-length penalty for slow walkers: 0=disabled, >0=enabled (multiplied by collision_harshness when collisions enabled)
            "collision_harshness": 2.0,  # Multiplier for specialization and adaptive cooperation when collisions enabled (1.0=same, 2.0=double)
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
        
        # Initialize intermediate reward decay configuration
        from training_configuration.reward_penalties import get_intermediate_reward_decay_config
        default_intermediate_reward_decay_cfg = get_intermediate_reward_decay_config()
        self.intermediate_reward_decay_cfg = intermediate_reward_decay_cfg if intermediate_reward_decay_cfg is not None else default_intermediate_reward_decay_cfg
        
        self.wait_for_action_completion = wait_for_completion
        self.random_initial_state = random_initial_state  # Store flag for random initial states
        self.allow_blocked = allow_blocked  # Whether to allow blocked actions to be attempted
        
        # Extract collision harshness multiplier for specialization and adaptive cooperation
        self.collision_harshness = self.penalties_cfg.get("collision_harshness", 2.0)
        
        self.clickable_indices = None  # Initialize clickable indices storage
        
        # Note: distance_map parameter is no longer used. Time normalization uses
        # max_distance loaded in game.py (from distance_map_{map_id}_max_distance.npy)
        # and observation space calculates paths online using PathProcessor + A*.
                    
        # Load the accessibility map for this map
        self.accessibility_map = get_accessibility_map(map_nr, game_version=game_mode)
                    
        # Initialize path processing system
        self.path_processor = PathProcessor(map_nr, collision_enabled)
        
        # Store collision flag for reward calculations
        self.collision_enabled = collision_enabled

        # Determine agent IDs from reward_weights or an explicit agent-count override.
        if reward_weights is not None:
            self.possible_agents = list(reward_weights.keys())
        elif num_agents is not None:
            self.possible_agents = [f"ai_rl_{i}" for i in range(1, num_agents + 1)]
        else:
            self.possible_agents = ["ai_rl_1", "ai_rl_2"]
        self.agents = self.possible_agents[:]

        default_weights = {agent: (1.0, 0.0) for agent in self.agents}
        self.reward_weights = reward_weights if reward_weights is not None else default_weights
        self.wait_for_completion = wait_for_completion
        self.tick_duration = TICK_DURATION
        self.cumulated_pure_rewards = {agent: 0.0 for agent in self.agents}
        self.cumulated_modified_rewards = {agent: 0.0 for agent in self.agents}

        if self.game_mode == "competition":
            self.total_agent_events = {agent_id: {"deliver_own": 0, "deliver_other": 0, "salad_own": 0, "salad_other": 0, "cut_own": 0, "cut_other": 0, "plate": 0, "raw_food_own": 0, "raw_food_other": 0, "counter": 0} for agent_id in self.agents}
            # Assign food types dynamically based on actual agents
            food_types = ["tomato", "pumpkin"]
            self.agent_food_type = {
                agent_id: food_types[i % len(food_types)] 
                for i, agent_id in enumerate(sorted(self.agents))
            }
        elif self.game_mode == "classic":
            self.total_agent_events = {agent_id: {"deliver": 0, "salad": 0, "cut": 0, "plate": 0, "raw_food": 0, "counter": 0} for agent_id in self.agents}
        else:
            raise ValueError(f"Unknown game mode: {self.game_mode}")
        
        self.total_action_types = {
            agent_id: {action_type: 0 for action_type in get_action_type_list(self.game_mode)}
            for agent_id in self.agents
        }
        self.total_actions_asked = {agent_id: 0 for agent_id in self.agents}
        self.total_actions_blocked = {agent_id: 0 for agent_id in self.agents}
        self.total_actions_inaccessible = {agent_id: 0 for agent_id in self.agents}
        
        # Collision tracking statistics
        self.total_collisions_detected = 0
        self.total_collisions_rerouted = 0
        self.total_collisions_failed = 0
        self._last_collision_stats = {
            'collisions_detected': 0,
            'collisions_rerouted': 0,
            'collisions_failed': 0,
            'agents_with_detected_collisions': [],
            'agents_with_rerouted_collisions': [],
            'agents_with_failed_collisions': [],
        }

        self.game, self.action_spaces, self._clickable_mask, self.clickable_indices = init_game(self.agents, map_nr=self.map_nr, grid_size=self.grid_size, seed=self.seed, game_mode=self.game_mode, walking_speeds=self.walking_speeds, cutting_speeds=self.cutting_speeds)

        self.agent_map = {agent_id: self.game.gameObjects[agent_id] for agent_id in self.agents}
        
        # Tick-based agent state tracking
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

        # Team synergy-based shaping mechanism
        self.reference_reward_cfg = reference_reward_cfg if reference_reward_cfg is not None else {"enabled": False}
        self.solo_baselines = solo_baselines if solo_baselines is not None else {}
        self.solo_baseline_team = sum(self.solo_baselines.values()) if self.solo_baselines else None
        self.reference_reward_enabled = self.reference_reward_cfg.get("enabled", False) and self.solo_baselines
        
        # Always initialize kappa (competence transformation parameter)
        self.kappa = self.reference_reward_cfg.get("kappa", 10.0)
        
        # Initialize activate_synergy_positive flag (controls whether positive synergy signals are applied)
        self.activate_synergy_positive = self.reference_reward_cfg.get("activate_synergy_positive", False)
        
        # Initialize agent abilities (needed for both specialization and team synergy)
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
        
        if self.reference_reward_enabled:
            self.synergy_scaling_factor = self.reference_reward_cfg.get("synergy_scaling_factor", 0.5)
            print(f"[GameEnv] Team synergy enabled: synergy_scaling_factor={self.synergy_scaling_factor}, kappa={self.kappa}, activate_synergy_positive={self.activate_synergy_positive}, baselines={self.solo_baselines}, team={self.solo_baseline_team}")
            print(f"[GameEnv] Agent abilities: {self.agent_abilities}")

        # --- Initialize action history tracking (last 5 actions) ---
        self.action_history_length = 5
        # Initialize action history for each agent with -1 (no action taken yet)
        self.action_history = {
            agent: [-1] * self.action_history_length for agent in self.agents
        }
        
        # --- New observation space (includes action history) ---
        obs_vector, _, _, _ = game_to_obs_vector(self.game, self.agents[0], game_mode=self.game_mode, path_processor=self.path_processor)
        # Add space for action history (5 additional values)
        obs_size = obs_vector.size + self.action_history_length
        self.observation_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)
            for agent in self.agents
        }

        self.observations = {agent: np.zeros((obs_size,), dtype=np.float32) for agent in self.agents}
        self.modified_rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._last_score = 0
        
        # Initialize pre-calculated paths storage
        self.agent_action_paths = {agent_id: [] for agent_id in self.agents}
        self.agent_action_tiles = {agent_id: [] for agent_id in self.agents}
        self.agent_action_interaction_targets = {agent_id: [] for agent_id in self.agents}
        
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

        # Initialize agent busy states
        for agent_id, agent in self.agent_map.items():
            if hasattr(agent, 'path'):
                agent.path = []
            if hasattr(agent, 'path_index'):
                agent.path_index = 0

        self.cumulated_pure_rewards = {agent: 0.0 for agent in self.agents}
        self.cumulated_modified_rewards = {agent: 0.0 for agent in self.agents}

        self._elapsed_time = 0.0

        # Clear path processor state for new episode
        if hasattr(self.path_processor, 'active_paths'):
            self.path_processor.active_paths.clear()
            
        # Clear pre-calculated paths
        self.agent_action_paths = {agent_id: [] for agent_id in self.agents}
        self.agent_action_tiles = {agent_id: [] for agent_id in self.agents}
        self.agent_action_interaction_targets = {agent_id: [] for agent_id in self.agents}

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
        self.total_actions_blocked = {agent_id: 0 for agent_id in self.agents}
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
        
        # Reset action history for all agents
        self.action_history = {
            agent: [-1] * self.action_history_length for agent in self.agents
        }
        
        # All agents start idle, so they all need observations
        self.agents_need_observation = {agent_id: True for agent_id in self.agents}

        return self.observations, self.infos

    def observe(self, agent):
        obs_vector, considered_paths, considered_tiles, considered_interaction_targets = game_to_obs_vector(self.game, agent, game_mode=self.game_mode, path_processor=self.path_processor)
        
        # Store pre-calculated paths, tiles, and interaction targets for this agent
        self.agent_action_paths[agent] = considered_paths
        self.agent_action_tiles[agent] = considered_tiles
        self.agent_action_interaction_targets[agent] = considered_interaction_targets
        
        # Add action history to observation (last 5 actions as normalized values)
        # Normalize action indices to [0, 1] range. -1 (no action) becomes -1.0
        action_space_size = len(get_rl_action_space(self.game_mode))
        normalized_history = []
        for action_idx in self.action_history[agent]:
            if action_idx == -1:
                normalized_history.append(-1.0)  # No action taken yet
            else:
                normalized_history.append(float(action_idx) / max(1, action_space_size - 1))  # Normalize to [0, 1]
        
        # Combine original observation with action history
        obs_with_history = np.concatenate([obs_vector.flatten(), normalized_history])
        obs = obs_with_history.astype(np.float32)
        return obs

    def step(self, actions):
        """
        Tick-based simulation step with PREDICTIVE collision handling.
        
        PROCESSING ORDER:
        1. Process new actions ONLY for idle agents (ignore actions from busy agents)
        2. PREDICTIVE collision detection: predict next tick positions and reroute BEFORE movement
        3. Execute one tick (all agents move/interact)
        4. Update observations ONLY for agents that became idle (for next action decisions)
        """
        
        # Identify idle agents (only these can process new actions)
        idle_agents = {agent_id for agent_id in self.agents if agent_is_idle(self, agent_id)}
        
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
        # Process actions only from idle agents (ignore actions from busy agents)
        for agent_id, action_idx in actions.items():
            # Skip action processing if agent is not idle
            if agent_id not in idle_agents:
                continue
            self.total_actions_asked[agent_id] += 1
            agent = self.agent_map[agent_id]
            action_name = get_rl_action_space(self.game_mode)[action_idx]
            
            # Handle do_nothing action
            if action_name == "do_nothing":
                # Apply do_nothing penalty
                agent_penalties[agent_id] += self.penalties_cfg.get("do_nothing", 1.0)
                
                # Update action history for do_nothing action
                self.action_history[agent_id] = self.action_history[agent_id][1:] + [action_idx]
                
                # Store for logging
                self._logging_actions[agent_id] = {
                    'elapsed_time': self._elapsed_time,
                    'action_idx': action_idx,
                    'action_name': action_name,
                    'tile_index': -2,
                    'action_type': 'do_nothing',
                    'agent_tile_x': getattr(agent, 'slot_x', -1),
                    'agent_tile_y': getattr(agent, 'slot_y', -1),
                    'x': -2,
                    'y': -2,
                    'cancelled_by_collision': False,
                    'collision_detected': False,
                    'collision_rerouted': False
                }
                self.total_action_types[agent_id]['do_nothing'] += 1
                continue
            
            # Get cached path and tile from observation
            if agent_id in self.agent_action_tiles:
                cached_path = self.agent_action_paths[agent_id][action_idx]
                tile_index = self.agent_action_tiles[agent_id][action_idx]
                cached_interaction_target = self.agent_action_interaction_targets[agent_id][action_idx]
            else:
                cached_path = None
                tile_index = None
                cached_interaction_target = None
            
            # --- TWO-TIER PENALTY SYSTEM ---
            # Validate action based on observation space indicators:
            # 
            # 1. INACCESSIBLE (tile_index=None): No path exists due to walls/obstacles
            #    - Observation: accessibility=0, availability=0
            #    - Action: REJECTED immediately, agent stays idle
            #    - Penalty: penalties_cfg["inaccessible_tile"] (default: 1.0)
            #
            # 2. BLOCKED (tile_index=-1 OR collision during movement): Path blocked by agents
            #    - Observation: accessibility=1, availability=0 (blocked at observation time)
            #                   OR accessibility=1, availability=1 (collision during movement)
            #    - Action: ATTEMPTED, then cancelled if collision cannot be rerouted
            #    - Penalty: penalties_cfg["blocked"] (default: 0.5) applied when action fails
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
                # CASE 2: BLOCKED - Path exists but blocked by other agent's current position
                # Tile is accessible (path exists ignoring agents) but not currently available
                # Behavior controlled by ALLOW_BLOCKED parameter:
                #   - If True: ATTEMPT the action (assign it) and let collision detection handle it
                #   - If False: REJECT the action immediately (agent remains idle)
                # This only happens when collision_enabled=True
                action_type = "blocked"
                logging_index = -1
                logging_x, logging_y = -1, -1
                self.total_actions_blocked[agent_id] += 1
                self.total_action_types[agent_id][action_type] += 1
                
                # Base blocked penalty
                agent_penalties[agent_id] += self.penalties_cfg["blocked"]
                
                # ASSIGN the action only if ALLOW_BLOCKED is enabled
                if self.allow_blocked:
                    # ASSIGN the action with path that ignores agents - let collision detection handle conflicts
                    # The cached_path should contain the path ignoring agents (fixed in observation_space.py)
                    assign_action(self, agent_id, action_idx, action_name, tile_index, cached_path, action_type, cached_interaction_target)
                else:
                    # REJECT the action - agent remains idle and will ask for new action next step
                    agents_becoming_idle.add(agent_id)
                
            else:
                # CASE 2: VALID ACTION - Tile is both accessible and available
                # Action is assigned and will be executed
                # BLOCKED penalty may be applied later if runtime collision occurs during movement
                grid_w = self.game.grid.width
                x = tile_index % grid_w
                y = tile_index // grid_w
                tile = self.game.grid.tiles[x][y]
                action_type = get_action_type(tile, agent, agent_id, agent_food_type=self.agent_food_type, game_mode=self.game_mode, x=x, y=y, accessibility_map=self.accessibility_map)
                logging_index = tile_index
                logging_x = x
                logging_y = y
                
                # Assign action to agent
                path_length = assign_action(self, agent_id, action_idx, action_name, tile_index, cached_path, action_type, cached_interaction_target)
                
                # Apply adaptive cooperation penalty if enabled
                apply_adaptive_cooperation_penalty(self, agent_id, path_length, agent_penalties)
                
                # Track action type
                self.total_action_types[agent_id][action_type] += 1
                
                # Update action history (shift left and add new action)
                self.action_history[agent_id] = self.action_history[agent_id][1:] + [action_idx]
                
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
            is_blocked_action = action_type == "blocked"
            self._logging_actions[agent_id] = {
                'elapsed_time': self._elapsed_time,
                'action_idx': action_idx,
                'action_name': action_name,
                'tile_index': logging_index,
                'action_type': action_type,
                'agent_tile_x': getattr(agent, 'slot_x', -1),
                'agent_tile_y': getattr(agent, 'slot_y', -1),
                'x': logging_x,
                'y': logging_y,
                # blocked actions are failed due to occupancy conflict and should
                # be represented as collision-cancelled in downstream logs.
                'cancelled_by_collision': is_blocked_action,
                'collision_detected': False,
                'collision_rerouted': False
            }
        
        # --- Phase 2: Predictive Collision Detection and Rerouting ---
        # CRITICAL: This must happen BEFORE movement to prevent collisions
        # Instead of detecting collisions after they happen, predict and prevent them
        collision_stats = {
            'collisions_detected': 0,
            'collisions_rerouted': 0,
            'collisions_failed': 0,
            'agents_with_detected_collisions': [],
            'agents_with_rerouted_collisions': [],
            'agents_with_failed_collisions': [],
        }
        if self.path_processor.collision_enabled:
            collision_stats = resolve_predictive_collisions(self)
            
            # Update collision statistics
            self.total_collisions_detected += collision_stats['collisions_detected']
            self.total_collisions_rerouted += collision_stats['collisions_rerouted']
            self.total_collisions_failed += collision_stats['collisions_failed']

            # Mark collisions in action logs (including rerouted collisions).
            detected_agents = set(collision_stats.get('agents_with_detected_collisions', []))
            rerouted_agents = set(collision_stats.get('agents_with_rerouted_collisions', []))
            failed_agents = set(collision_stats.get('agents_with_failed_collisions', []))

            for agent_id in detected_agents:
                if agent_id in self._logging_actions:
                    self._logging_actions[agent_id]['collision_detected'] = True
                    self._logging_actions[agent_id]['collision_rerouted'] = agent_id in rerouted_agents
                else:
                    # Agent collided while executing an ongoing action from a previous tick.
                    # Emit a synthetic log row so actions.csv tracks all detected collisions.
                    state = self.agent_state[agent_id]
                    agent_obj = self.agent_map[agent_id]
                    self._logging_actions[agent_id] = {
                        'elapsed_time': self._elapsed_time,
                        'action_idx': -1,
                        'action_name': state.get('current_action') or 'ongoing_action',
                        'tile_index': -1,
                        'action_type': 'collision_event',
                        'agent_tile_x': getattr(agent_obj, 'slot_x', -1),
                        'agent_tile_y': getattr(agent_obj, 'slot_y', -1),
                        # Keep internal coordinates 0-indexed for logger/tracker consistency.
                        'x': (getattr(agent_obj, 'slot_x', 0) - 1) if getattr(agent_obj, 'slot_x', None) is not None else -1,
                        'y': (getattr(agent_obj, 'slot_y', 0) - 1) if getattr(agent_obj, 'slot_y', None) is not None else -1,
                        'cancelled_by_collision': agent_id in failed_agents,
                        'collision_detected': True,
                        'collision_rerouted': agent_id in rerouted_agents,
                    }
            
            # Apply penalties to agents whose collisions couldn't be rerouted
            # These agents had their actions cancelled in resolve_predictive_collisions
            for agent_id in collision_stats['agents_with_failed_collisions']:
                agent = self.agent_map[agent_id]
                
                # Base blocked penalty for all failed collisions
                agent_penalties[agent_id] += self.penalties_cfg["collision"]
                
                # Mark action as cancelled in logging
                if agent_id in self._logging_actions:
                    self._logging_actions[agent_id]['cancelled_by_collision'] = True
                
                # Mark these agents for observation refresh since they're now idle
                agents_becoming_idle.add(agent_id)
        # If collision detection is disabled, this entire block is skipped
        # No collision penalties, no rerouting, agents can overlap freely

            self._last_collision_stats = collision_stats
        
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
            agent = self.agent_map[agent_id]
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
        
        # Update totals for logged events
        for agent_id in self.agents:
            for event_type in agent_events[agent_id]:
                if agent_events[agent_id][event_type] > 0:
                    self.total_agent_events[agent_id][event_type] += agent_events[agent_id][event_type]
        
        # Compute rewards (includes reference-based opportunity cost if enabled)
        self.cumulated_pure_rewards, self.cumulated_modified_rewards = get_rewards(self, agent_events, agent_penalties, self.rewards_cfg, self.intermediate_reward_decay_cfg, self.episode_count)
        
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
        
        # Update observations ONLY for agents that became idle this tick
        # These are the agents that will be able to process new actions in the next step
        for agent_id in agents_becoming_idle:
            self.observations[agent_id] = self.observe(agent_id)
        
        # Note: Busy agents keep their previous observations since they can't act until they finish their current tasks
        
        terminations = self.dones
        truncations = {agent: False for agent in self.agents}
        
        # If episode is done, aggregate and log
        if self.write_csv and self.enable_csv_logging:
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
                row[f"actions_blocked_{agent_id}"] = self.total_actions_blocked[agent_id]
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