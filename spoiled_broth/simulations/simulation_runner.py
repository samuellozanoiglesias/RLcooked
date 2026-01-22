"""
Main simulation runner class for managing complete simulation execution.

Author: Samuel Lozano
"""

import os
import time
import numpy as np
import sys
import threading
from pathlib import Path
from typing import Dict, Any, Callable, Optional

from engine.app.ai_only_app import AIOnlySessionApp
from engine.extensions.renderer2d.renderer_2d_offline import Renderer2DOffline
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule

from .simulation_config import SimulationConfig
from .path_manager import PathManager
from .controller_manager import ControllerManager
from .ray_manager import RayManager
from .game_manager import GameManager
from .data_logger import DataLogger
from .video_recorder import VideoRecorder
from .raw_action_logger import RawActionLogger
from .observation_logger import ObservationLogger
from spoiled_broth.maps.cache_distance_map import load_or_compute_distance_map


class SimulationRunner:
    """Main class for running simulations."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.path_manager = PathManager(config)
        self.controller_manager = ControllerManager(config)
        self.ray_manager = RayManager()
        self.game_manager = GameManager(config)
        
    def run_simulation(self, map_nr: str, num_agents: int,
                      game_version: str, training_id: str,
                      checkpoint_number: str, timestamp: str,
                      study_name: str = 'default',
                      game_type: str = '') -> Dict[str, Path]:
        """
        Run a complete simulation.
        
        Args:
            map_nr: Name of the map
            num_agents: Number of agents
            game_version: Game version identifier
            training_id: Training identifier
            checkpoint_number: Checkpoint number (integer or "final")
            timestamp: Timestamp for file naming
            study_name: Study name for organizing simulations
            
        Returns:
            Dictionary containing output file paths
        """        
        # Initialize Ray
        self.ray_manager.initialize_ray()
        
        try:
            # Setup paths
            paths = self.path_manager.setup_paths(
                map_nr, num_agents, game_version, training_id, checkpoint_number, study_name, game_type
            )
            
            # Get grid size
            grid_size = self.path_manager.get_grid_size_from_map(paths['map_txt_path'])
            
            # Initialize controllers
            controllers = self.controller_manager.initialize_controllers(
                num_agents, paths['checkpoint_dir'], game_version, 
                tick_rate=self.config.engine_tick_rate, custom_checkpoints=self.config.custom_checkpoints
            )
            
            if not controllers:
                raise ValueError("No controllers were successfully initialized")
            
            # Setup logging with simulation configuration
            simulation_config = {
                'MAP_NR': map_nr,
                'NUM_AGENTS': num_agents,
                'GAME_VERSION': game_version,
                'TRAINING_ID': training_id,
                'CLUSTER': self.config.cluster,
                'DURATION': self.config.duration_seconds,
                'TICK_RATE': self.config.engine_tick_rate,
                'AI_TICK_RATE': self.config.ai_tick_rate,
                'AGENT_SPEED': self.config.agent_speed_px_per_sec,
                'GRID_SIZE': grid_size,
                'TILE_SIZE': self.config.tile_size,
                'ENABLE_VIDEO': self.config.enable_video,
                'VIDEO_FPS': self.config.video_fps,
                'CHECKPOINT_DIR': str(paths['checkpoint_dir']),
                'MAP_FILE': str(paths['map_txt_path']),
                'AGENT_INITIALIZATION_PERIOD': self.config.agent_initialization_period,
                'TOTAL_SIMULATION_TIME': self.config.total_simulation_time,
                'TOTAL_FRAMES': self.config.total_frames,
                'WALKING_SPEEDS': self.config.walking_speeds,
                'CUTTING_SPEEDS': self.config.cutting_speeds
            }
            
            data_logger = DataLogger(paths['saving_path'], checkpoint_number, timestamp, simulation_config)
            
            # Create raw action logger for capturing integer actions directly from controllers
            raw_action_csv_path = data_logger.simulation_dir / "actions.csv"
            raw_action_logger = RawActionLogger(raw_action_csv_path)
            
            # Create observation logger for capturing observation vectors associated with actions
            observation_csv_path = data_logger.simulation_dir / "observations.csv"
            observation_logger = ObservationLogger(observation_csv_path)
            
            # Setup game
            game_factory = self.game_manager.create_game_factory(
                map_nr, grid_size, self.config.walking_speeds, self.config.cutting_speeds, game_type
            )
            
            # Run the simulation
            output_paths = self._execute_simulation(
                game_factory, controllers, paths, data_logger, 
                raw_action_logger, observation_logger, timestamp, map_nr
            )
            
            return output_paths
            
        finally:
            self.ray_manager.shutdown_ray()
    
    def _execute_simulation(self, game_factory: Callable, controllers: Dict,
                           paths: Dict, data_logger: DataLogger,
                           raw_action_logger: RawActionLogger,
                           observation_logger: ObservationLogger, 
                           timestamp: str, map_nr: str) -> Dict[str, Path]:
        """Execute the main simulation loop."""
        
        # Create engine app
        engine_app = AIOnlySessionApp(
            game_factory=game_factory,
            ui_modules=[Renderer2DModule()],
            agent_map=controllers,
            path_root=paths['path_root'],
            tick_rate=self.config.engine_tick_rate,
            ai_tick_rate=self.config.ai_tick_rate,
            is_max_speed=False,
            agent_initialization_period=self.config.agent_initialization_period
        )
        
        session = engine_app.get_session()
        game = session.engine.game
        
        # Load and attach distance map to the game
        try:
            distance_cache_dir = os.path.join(os.path.dirname(__file__), "../maps/distance_cache")
            distance_map_path = load_or_compute_distance_map(game, game.grid, map_nr, distance_cache_dir)
            
            # Load the distance map from the npz file
            data = np.load(distance_map_path)
            D = data['D']
            pos_from = data['pos_from']
            pos_to = data['pos_to']
            
            # Convert back to dictionary format
            distance_map = {}
            for i, from_pos in enumerate(pos_from):
                from_tuple = tuple(from_pos)
                distance_map[from_tuple] = {}
                for j, to_pos in enumerate(pos_to):
                    to_tuple = tuple(to_pos)
                    dist = D[i, j]
                    if not np.isnan(dist):
                        distance_map[from_tuple][to_tuple] = float(dist)
            
            # Attach distance map to game
            game.distance_map = distance_map
            print(f"Successfully loaded distance map for {map_nr} with {len(distance_map)} positions")
            
        except Exception as e:
            print(f"Warning: Could not load distance map for {map_nr}: {e}")
            print("Setting game.distance_map to empty dict to prevent crashes")
            game.distance_map = {}
        
        # Attach raw action logger
        game.raw_action_logger = raw_action_logger
        
        # Attach observation logger
        game.observation_logger = observation_logger
        
        # Attach controllers to agents
        for agent_id, controller in controllers.items():
            agent_obj = game.gameObjects.get(agent_id)
            if agent_obj is not None:
                controller.agent = agent_obj
        
        # Ensure agents have the configured individual speeds
        for agent_id, obj in list(game.gameObjects.items()):
            if agent_id.startswith('ai_rl_') and obj is not None:
                try:
                    # Use individual walking speed if available, otherwise use global default
                    if self.config.walking_speeds and agent_id in self.config.walking_speeds:
                        walking_speed = self.config.walking_speeds[agent_id]
                        obj.speed = walking_speed * self.config.agent_speed_px_per_sec
                    else:
                        obj.speed = self.config.agent_speed_px_per_sec
                except Exception:
                    pass
        
        # EXPLICIT AGENT STATE RESET: Clear any items and reset states from checkpoint
        # This ensures agents start fresh regardless of their training checkpoint state
        for agent_id, obj in list(game.gameObjects.items()):
            if agent_id.startswith('ai_rl_') and obj is not None:
                try:
                    # Clear items in hand
                    if hasattr(obj, 'item'):
                        obj.item = None
                    if hasattr(obj, 'provisional_item'):
                        obj.provisional_item = None
                    # Reset action states
                    if hasattr(obj, 'current_action'):
                        obj.current_action = None
                    if hasattr(obj, 'is_busy'):
                        obj.is_busy = False
                    if hasattr(obj, 'is_cutting'):
                        obj.is_cutting = False
                    if hasattr(obj, 'cutting_start_time'):
                        obj.cutting_start_time = None
                    # Reset movement states
                    if hasattr(obj, 'move_target'):
                        obj.move_target = None
                    if hasattr(obj, 'path'):
                        obj.path = []
                    if hasattr(obj, 'path_index'):
                        obj.path_index = 0
                    
                    # REPOSITION AGENTS TO CORRECT STARTING POSITIONS
                    # Extract agent number from ID (e.g., 'ai_rl_1' -> 1)
                    agent_number = int(agent_id.split('_')[-1])
                    
                    # Get the correct starting tile based on agent number
                    if hasattr(game, 'agent_start_tiles'):
                        start_tile_key = str(agent_number)
                        if start_tile_key in game.agent_start_tiles:
                            start_tile = game.agent_start_tiles[start_tile_key]
                            # Set pixel position to the correct starting tile
                            obj.x = start_tile.slot_x * game.grid.tile_size + game.grid.tile_size // 2
                            obj.y = start_tile.slot_y * game.grid.tile_size + game.grid.tile_size // 2
                            print(f"Repositioned agent {agent_id} to starting position: tile=({start_tile.slot_x}, {start_tile.slot_y}), pixel=({obj.x}, {obj.y})")
                        else:
                            print(f"Warning: No starting tile found for agent {agent_id} (number {agent_number})")
                    
                    print(f"Reset agent {agent_id} state: item={getattr(obj, 'item', None)}, position=({obj.x}, {obj.y})")
                except Exception as e:
                    print(f"Warning: Error resetting agent {agent_id} state: {e}")
        
        # Setup rendering
        renderer = Renderer2DOffline(
            path_root=paths['path_root'], 
            tile_size=self.config.tile_size
        )
        
        video_recorder = None
        if self.config.enable_video:
            video_path = data_logger.simulation_dir / f"offline_recording_{timestamp}.mp4"
            video_recorder = VideoRecorder(video_path, self.config.video_fps)
        
        try:
            # Start session
            try:
                session.start()
                print("Session started successfully")
            except RuntimeError as e:
                if "threads can only be started once" in str(e):
                    print("Session already started, continuing...")
                else:
                    raise e
            
            time.sleep(0.5)  # Allow startup
            
            # Run simulation loop
            self._run_simulation_loop(
                session, game, renderer, data_logger, 
                video_recorder, timestamp
            )
            
        finally:
            # Cleanup
            self._cleanup_simulation(
                video_recorder, engine_app, 
                game, session.engine, data_logger
            )
        
        return {
            'simulation_dir': data_logger.simulation_dir,
            'config_file': data_logger.config_path,
            'state_csv': data_logger.state_csv_path,
            'action_csv': data_logger.simulation_dir / "actions.csv",
            'observation_csv': data_logger.simulation_dir / "observations.csv",
            'counter_csv': data_logger.counter_csv_path,
            'video_file': data_logger.simulation_dir / f"offline_recording_{timestamp}.mp4" if self.config.enable_video else None
        }
    
    def _run_simulation_loop(self, session: Any, game: Any, renderer: Any,
                           data_logger: DataLogger, video_recorder: Optional[VideoRecorder],
                           timestamp: str):
        """Run the main simulation loop."""
        
        total_frames = self.config.total_frames
        frame_interval = 1.0 / self.config.engine_tick_rate
        start_time = time.time()
        
        progress_step = max(1, int(total_frames * 0.05))
        next_progress = progress_step
        
        # Wait for agents to be properly initialized and positioned
        self._wait_for_agent_initialization(game)
        print("System initialization complete, starting simulation...")
        
        prev_state = None
        
        for frame_idx in range(total_frames):
            # Wait for the right time for this frame (exactly like old code)
            target_time = start_time + frame_idx * frame_interval
            current_time = time.time()
            if current_time < target_time:
                time.sleep(target_time - current_time)
            
            # Set frame count on game object for controller synchronization
            game.frame_count = frame_idx
            
            # Get current state and render
            curr_state = self._serialize_ui_state(session, game)
            prev_state = prev_state or curr_state
            frame = renderer.render_to_array(prev_state, curr_state, 0.0)
            
            # Log state
            data_logger.log_state(frame_idx, game, self.config.engine_tick_rate)
            
            # Log counter state
            data_logger.log_counters(frame_idx, game, self.config.engine_tick_rate)
            
            # Record video
            if video_recorder:
                video_recorder.write_frame_with_hud(frame, game)
            
            prev_state = curr_state
            
            # Progress reporting (adjust for initialization period)
            simulation_time = frame_idx / self.config.engine_tick_rate
            if frame_idx + 1 >= next_progress:
                percent = int(100 * (frame_idx + 1) / total_frames)
                if simulation_time < self.config.agent_initialization_period:
                    status = f"(initialization: {simulation_time:.1f}s/{self.config.agent_initialization_period}s)"
                else:
                    active_time = simulation_time - self.config.agent_initialization_period
                    status = f"(active gameplay: {active_time:.1f}s/{self.config.duration_seconds}s)"
                print(f"Simulation progress: {percent}% ({frame_idx + 1}/{total_frames} frames) {status}")
                sys.stdout.flush()
                next_progress += progress_step
    
    def _serialize_ui_state(self, session: Any, game: Any) -> Dict:
        """Serialize UI state for rendering."""
        payload = {}
        for module in session.ui_modules:
            payload.update(module.serialize_for_agent(game, session.engine, None))
        payload['tick'] = session.engine.tick_count
        return payload
    
    def _cleanup_simulation(self, video_recorder: Optional[VideoRecorder],
                          engine_app: Any, game: Any, engine: Any,
                          data_logger: DataLogger):
        """Cleanup simulation resources."""
        print("Cleaning up simulation...")
        
        # Update config file with runtime information before cleanup
        print("Updating config file with runtime detection data...")
        data_logger.update_config_with_runtime_info(game)
        
        # Stop video recording
        if video_recorder:
            print("Stopping video recorder...")
            video_recorder.stop()
        
        # Stop engine
        print("Stopping engine...")
        self._stop_engine_with_timeout(engine_app)
        
        print("Simulation cleanup completed")
    
    def _stop_engine_with_timeout(self, engine_app: Any, timeout: float = 5.0):
        """Stop engine with timeout to prevent hanging."""
        try:
            # First try direct stop without threading
            engine_app.stop()
            print("Engine stopped successfully")
        except Exception as direct_error:
            print(f"Direct engine stop failed: {direct_error}, trying with timeout thread...")
            
            # Only use threading as fallback
            engine_stop_completed = threading.Event()
            engine_stop_result = {"error": None}
            
            def stop_engine():
                try:
                    engine_app.stop()
                    engine_stop_completed.set()
                except Exception as e:
                    engine_stop_result["error"] = e
                    engine_stop_completed.set()
            
            try:
                # Create and start thread - only when needed
                stop_thread = threading.Thread(target=stop_engine)
                stop_thread.daemon = True
                stop_thread.start()
                
                # Wait with timeout
                if engine_stop_completed.wait(timeout=timeout):
                    if engine_stop_result["error"]:
                        print(f"Error stopping engine: {engine_stop_result['error']}")
                    else:
                        print("Engine stopped successfully via thread")
                else:
                    print(f"Engine stop timed out after {timeout} seconds")
                    
            except RuntimeError as e:
                if "threads can only be started once" in str(e):
                    print("Threading error encountered, engine may already be stopped")
                else:
                    print(f"Threading error during engine stop: {e}")
            except Exception as e:
                print(f"Unexpected error during threaded engine stop: {e}")
    
    def _wait_for_agent_initialization(self, game: Any, max_wait_time: float = 60.0):
        """Wait for all agents to complete their initialization period and be ready to act."""
        
        start_wait = time.time()
        print("Waiting for agent initialization...")
        
        # First, wait for basic agent setup
        while time.time() - start_wait < max_wait_time:
            all_agents_ready = True
            agent_count = 0
            
            for agent_id, obj in game.gameObjects.items():
                if agent_id.startswith('ai_rl_'):
                    agent_count += 1
                    
                    # Check if agent has valid position
                    if not hasattr(obj, 'x') or not hasattr(obj, 'y') or obj.x is None or obj.y is None:
                        all_agents_ready = False
                        break
                    
                    # Check if agent has grid reference
                    if not hasattr(obj, 'grid') or obj.grid is None:
                        all_agents_ready = False
                        break
                        
                    # Check if agent has slot coordinates
                    try:
                        slot_x, slot_y = obj.slot_x, obj.slot_y
                        if slot_x < 0 or slot_y < 0:
                            all_agents_ready = False
                            break
                    except Exception:
                        all_agents_ready = False
                        break
            
            # Also check that we have the expected number of agents
            if agent_count == 0:
                all_agents_ready = False
            
            if all_agents_ready:
                print(f"Basic agent setup complete for {agent_count} agents")
                break
            
            time.sleep(0.1)  # Wait 100ms before checking again
        
        if not all_agents_ready:
            print(f"Warning: Basic agent setup timed out after {max_wait_time} seconds")
            # Log current agent states for debugging
            for agent_id, obj in game.gameObjects.items():
                if agent_id.startswith('ai_rl_'):
                    x = getattr(obj, 'x', 'MISSING')
                    y = getattr(obj, 'y', 'MISSING')
                    grid = getattr(obj, 'grid', 'MISSING')
                    print(f"  {agent_id}: position ({x}, {y}), grid={grid is not None}")
            return
        
        # Now wait for the frame-based initialization period to complete
        # This ensures agents don't start acting until the initialization period is over
        print("Basic agent setup complete, waiting for initialization period...")
        
        start_init_wait = time.time()
        initialization_complete = False
        
        while time.time() - start_init_wait < max_wait_time:
            # Check if initialization period is complete
            if hasattr(game, 'frame_count'):
                frame_count = game.frame_count
                
                # Get initialization frames from config
                init_frames = int(self.config.agent_initialization_period * self.config.engine_tick_rate)
                
                if frame_count >= init_frames:
                    initialization_complete = True
                    print(f"Initialization period complete at frame {frame_count} (required: {init_frames})")
                    break
            
            time.sleep(0.1)
            
        if not initialization_complete:
            print(f"Warning: Initialization period did not complete within {max_wait_time} seconds")
            if hasattr(game, 'frame_count'):
                print(f"  Current frame: {game.frame_count}, Required: {int(self.config.agent_initialization_period * self.config.engine_tick_rate)}")
            else:
                print("  Game has no frame_count attribute")
        
        print("All agents initialized successfully")
        # Log final agent positions
        for agent_id, obj in game.gameObjects.items():
            if agent_id.startswith('ai_rl_'):
                print(f"  {agent_id}: position ({obj.x}, {obj.y}), tile ({obj.slot_x}, {obj.slot_y})")