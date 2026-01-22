import os
import time
import torch
import numpy as np
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from spoiled_broth.rl.observation_space import game_to_obs_vector
from spoiled_broth.rl.action_space import get_rl_action_space, convert_action_to_tile
from engine.extensions.topDownGridWorld.ai_controller._base_controller import Controller

class RLlibController(Controller):
    def __init__(self, agent_id, checkpoint_path, policy_id, competition=False, agent_initialization_frames=0):
        super().__init__(agent_id)
        self.agent_id = agent_id
        self.policy_id = policy_id
        self.competition = competition
        self.agent_initialization_frames = agent_initialization_frames
        
        # RLModule inside the checkpoint
        rl_module_path = os.path.join(
            checkpoint_path,
            "learner_group",
            "learner",
            "rl_module"
        )

        # Load MultiRLModule from checkpoint
        self.multi_rl_module = MultiRLModule.from_checkpoint(rl_module_path)
        if self.policy_id not in self.multi_rl_module.keys():
            raise ValueError(f'Policy {self.policy_id} not found in the loaded module.')

        # Obtain the specific policy module
        self.policy_module = self.multi_rl_module[self.policy_id]

    def choose_action(self, observation):      
        # Set simulation flag on the agent if not already set
        if hasattr(self, 'agent') and self.agent is not None:
            if not hasattr(self.agent, 'is_simulation') or not self.agent.is_simulation:
                self.agent.is_simulation = True
                
            # Register agent with state manager if available
            if hasattr(self, 'agent_state_manager') and self.agent_id not in self.agent_state_manager.agent_state_history:
                self.agent_state_manager.register_agent(self.agent_id, self.agent)
        
        # Check and reset stuck agents if state manager is available
        if hasattr(self, 'agent_state_manager'):
            self.agent_state_manager.check_and_reset_stuck_agents()
        
        # Check if we're still in initialization period using frame count
        if self.agent_initialization_frames > 0:
            if hasattr(self, 'agent') and hasattr(self.agent, 'game'):
                game = self.agent.game
                
                # Get frame count from game for timing
                if hasattr(game, 'frame_count'):
                    frame_count = getattr(game, 'frame_count')
                    
                    # Use <= instead of < to be more conservative and ensure agents don't act
                    # until the initialization period is completely finished
                    if frame_count <= self.agent_initialization_frames:
                        # During initialization, return None (no action)
                        return None
                    
                    # Track when the first agent starts acting (globally)
                    if not hasattr(game, 'agents_start_acting_frame'):
                        game.agents_start_acting_frame = frame_count
                        print(f"{self.agent_id}: First agent starting to act at frame {frame_count}")

                else:
                    # No frame_count available - assume still in initialization
                    return None
        
        # If a previous action is still in progress, check if it should be completed
        if getattr(self.agent, 'is_busy', False):
            current_action = getattr(self.agent, 'current_action', None)
            is_moving = getattr(self.agent, 'is_moving', True)
                        
            # Check if the agent has finished moving
            if not is_moving:
                # Agent has finished moving, mark as not busy
                self.agent.is_busy = False
                if hasattr(self.agent, 'current_action'):
                    self.agent.current_action = None
            elif current_action is not None:
                # Check if action has been running too long (timeout mechanism)
                action_start_time = current_action.get('start_time', time.time())
                if time.time() - action_start_time > 10.0:  # 10 second timeout
                    self.agent.is_busy = False
                    self.agent.current_action = None
                else:
                    # Still moving, can't choose new action
                    return None
            else:
                # Busy but no current action - reset immediately
                self.agent.is_busy = False
        
        # Check if agent is still moving from a previous action
        elif getattr(self.agent, 'current_action', None) is not None:
            is_moving = getattr(self.agent, 'is_moving', False)
            current_action = self.agent.current_action
                        
            if is_moving:
                # Still moving to target, don't issue new action
                return None
            else:
                # Reached target, clear action tracking
                self.agent.current_action = None
        
        # Check if agent is in cutting state and still needs to wait
        if getattr(self.agent, 'is_cutting', False):
            cutting_start = getattr(self.agent, 'cutting_start_time', None)
            cutting_duration = getattr(self.agent, 'cutting_duration', 3.0)
            
            if cutting_start is not None:
                elapsed = time.time() - cutting_start
                if elapsed < cutting_duration:
                    # Still waiting for cutting to complete
                    return None
                else:
                    # Cutting time is done, release the agent
                    self.agent.is_cutting = False
                    self.agent.cutting_start_time = None
                    # Transfer the provisional item to actual item
                    if hasattr(self.agent, 'provisional_item') and self.agent.provisional_item is not None:
                        self.agent.item = self.agent.provisional_item
                        self.agent.provisional_item = None
                    # Continue to normal action selection below
        
        # Select correct obs function
        game_mode = "competition" if self.competition else "classic"
        path_processor = getattr(self.agent.game, 'path_processor', None)
        obs_vector, considered_paths, considered_tiles = game_to_obs_vector(
            self.agent.game, self.agent_id, game_mode=game_mode, path_processor=path_processor
        )
        input_dict = {"obs": torch.tensor(obs_vector, dtype=torch.float32)}

        # Obtain the action logits from the policy module
        action_output = self.policy_module.forward_inference(input_dict)
        action_logits = action_output["action_dist_inputs"]

        # Obtain the action distribution class from the policy module
        action_dist_class = self.policy_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(action_logits)
        action = action_dist.sample().item()
        
        # Convert RL action index to action name
        game_mode = "competition" if self.competition else "classic"
        action_space = get_rl_action_space(game_mode)
        
        if 0 <= action < len(action_space):
            action_name = action_space[action]
            
            # Get the tile index and path from observation generation (same as game_env)
            tile_index = considered_tiles[action] if action < len(considered_tiles) else None
            cached_path = considered_paths[action] if action < len(considered_paths) else None
            
            # Log the observation vector associated with this action if observation_logger exists
            if hasattr(self.agent.game, 'observation_logger'):
                try:
                    frame = getattr(self.agent.game, 'frame_count', 0)
                    self.agent.game.observation_logger.log_observation(
                        self.agent_id, frame, 'rl_action', action_name, obs_vector
                    )
                except Exception as e:
                    print(f"[RLLIB_CONTROLLER] Error logging observation for {self.agent_id}: {e}")
            
            # Log the raw action to CSV if raw_action_logger exists
            if hasattr(self.agent.game, 'raw_action_logger'):
                # For RL actions, we log the action name and eventual tile target
                try:
                    self.agent.game.raw_action_logger.log_rl_action(
                        self.agent_id, action, action_name, self.agent.game, game_mode
                    )
                except AttributeError:
                    # Fallback: log using original method but handle the mismatch gracefully
                    try:
                        self.agent.game.raw_action_logger.log_action(
                            self.agent_id, -1, self.agent.game, []  # Use empty list to avoid index errors
                        )
                    except Exception:
                        pass  # Skip logging if it fails
            
            # Execute action using the same logic as game_env.step()
            if tile_index is None:
                # No valid tile found during observation (inaccessible)
                return None
            elif tile_index == -1:
                # Path blocked by collisions
                return None
            else:
                # Action is valid - set up the path for the agent (same as game_env)
                grid = self.agent.grid
                x = tile_index % grid.width
                y = tile_index // grid.width
                tile = grid.tiles[x][y]
                
                if tile and hasattr(tile, "click"):
                    # Set up the cached path for the agent
                    if cached_path and len(cached_path) > 1:
                        self.agent.path = cached_path[1:]  # Skip current tile
                        self.agent.path_index = 0
                    else:
                        # No cached path - clear path
                        self.agent.path = []
                        self.agent.path_index = 0
                    
                    # Update path processor with agent's active path (same as game_env)
                    path_processor = getattr(self.agent.game, 'path_processor', None)
                    if path_processor and path_processor.is_enabled() and hasattr(self.agent, 'path') and self.agent.path:
                        # Convert agent speed from pixels/second to tiles/second for collision detection
                        agent_speed_pixels = getattr(self.agent, 'speed', 30.0)
                        agent_speed_tiles = agent_speed_pixels / 16.0  # tile_size is 16 pixels
                        path_processor.update_agent_path(
                            agent_id=self.agent_id,
                            path=self.agent.path,
                            current_position=(self.agent.slot_x, self.agent.slot_y),
                            path_index=0,
                            speed=agent_speed_tiles
                        )
                    
                    # Don't mark agent as busy immediately - let movement happen first
                    # The agent will be marked as busy by the intent system when it reaches the tile
                    action_dict = {
                        "type": "click", 
                        "target": tile_index,  # Use numeric tile_index instead of tile.id
                        "start_time": time.time()
                    }
                    # Store the current action for tracking, but don't set is_busy yet
                    self.agent.current_action = action_dict
                    
                    # Notify state manager of action
                    if hasattr(self, 'agent_state_manager'):
                        self.agent_state_manager.update_agent_action(self.agent_id)
                    
                    return action_dict
                else:
                    return None
        else:
            return None
        
        # If we reach here, no valid action was taken
        return None        
