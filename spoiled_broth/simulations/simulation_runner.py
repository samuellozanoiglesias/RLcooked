"""
Simulation runner using GameEnv directly (tick-based, same dynamics as RL training).

This runner bypasses the Flask/engine layer entirely and instead:
  1. Creates a GameEnv instance (identical setup to RL training)
  2. Loads RLlib policies from checkpoints for inference
  3. Runs the tick-based step loop (env.step), logging every tick
  4. Optionally records a video by rendering game state directly

Collision support:  pass game_type containing 'collision' to enable.
Author: Samuel Lozano
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

from spoiled_broth.rl.game_env import GameEnv
from spoiled_broth.rl.tick_based_structure import agent_is_idle
from spoiled_broth.rl.action_space import get_rl_action_space

from .simulation_config import SimulationConfig
from .path_manager import PathManager
from .ray_manager import RayManager
from .data_logger import DataLogger
from .policy_loader import load_policies

# Must match RL's TICK_DURATION constant (game_env.py)
TICK_DURATION = 0.5  # seconds per env.step() call


class SimulationRunner:
    """Runs one simulation episode using GameEnv + RLlib policies."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.path_manager = PathManager(config)
        self.ray_manager = RayManager()

    # ------------------------------------------------------------------ #
    #  Public entry point                                                   #
    # ------------------------------------------------------------------ #

    def run_simulation(
        self,
        map_nr: str,
        num_agents: int,
        game_version: str,
        training_id: str,
        checkpoint_number: str,
        timestamp: str,
        study_name: str = "default",
        game_type: str = "",
        synergy_folder: str = None,
        specialization_folder: str = None,
    ) -> Dict[str, Path]:
        """Run a complete simulation and return output file paths."""

        self.ray_manager.initialize_ray()
        try:
            return self._run(
                map_nr, num_agents, game_version, training_id,
                checkpoint_number, timestamp, study_name, game_type,
                synergy_folder, specialization_folder,
            )
        finally:
            self.ray_manager.shutdown_ray()

    # ------------------------------------------------------------------ #
    #  Core implementation                                                  #
    # ------------------------------------------------------------------ #

    def _run(
        self,
        map_nr: str,
        num_agents: int,
        game_version: str,
        training_id: str,
        checkpoint_number: str,
        timestamp: str,
        study_name: str,
        game_type: str,
        synergy_folder: str,
        specialization_folder: str,
    ) -> Dict[str, Path]:

        # --- 1. Resolve paths ------------------------------------------------
        paths = self.path_manager.setup_paths(
            map_nr, num_agents, game_version, training_id,
            checkpoint_number, study_name, game_type,
            synergy_folder, specialization_folder,
        )
        grid_size = self.path_manager.get_grid_size_from_map(paths["map_txt_path"])

        # --- 2. Determine game mode and collision flag -----------------------
        game_mode = "competition" if game_version.upper() == "COMPETITION" else "classic"
        # Check both game_type and game_version for collision flag
        collision_enabled = "collision" in game_type.lower() or "collision" in game_version.lower()
        print(f"Game mode: {game_mode} | Collision: {collision_enabled}")

        # --- 3. Create DataLogger -------------------------------------------
        simulation_config_meta = {
            "MAP_NR": map_nr,
            "NUM_AGENTS": num_agents,
            "GAME_VERSION": game_version,
            "TRAINING_ID": training_id,
            "GAME_TYPE": game_type,
            "CLUSTER": self.config.cluster,
            "DURATION": self.config.duration_seconds,
            "TICK_RATE": 1.0 / TICK_DURATION,  # steps per second (for config logging)
            "COLLISION_ENABLED": collision_enabled,
            "ENABLE_VIDEO": self.config.enable_video,
            "VIDEO_FPS": self.config.video_fps,
            "CHECKPOINT_DIR": str(paths["checkpoint_dir"]),
            "MAP_FILE": str(paths["map_txt_path"]),
            "AGENT_INITIALIZATION_PERIOD": self.config.agent_initialization_period,
            "WALKING_SPEEDS": self.config.walking_speeds,
            "CUTTING_SPEEDS": self.config.cutting_speeds,
        }
        data_logger = DataLogger(
            paths["saving_path"],
            checkpoint_number,
            timestamp,
            simulation_config_meta,
        )

        # --- 4. Build GameEnv -----------------------------------------------
        print("Creating GameEnv …")
        env = GameEnv(
            map_nr=map_nr,
            game_mode=game_mode,
            inner_seconds=self.config.duration_seconds,
            grid_size=grid_size,
            walking_speeds=self.config.walking_speeds,
            cutting_speeds=self.config.cutting_speeds,
            collision_enabled=collision_enabled,
            path=str(data_logger.simulation_dir),
            enable_csv_logging=False,  # Disable training_stats.csv for simulations
        )
        obs, _infos = env.reset()
        # Don't capture game reference - always use env.game to get fresh state
        print(f"GameEnv ready. Agents: {env.agents}")

        # --- 5. Load policies -----------------------------------------------
        action_space = get_rl_action_space(game_mode)
        action_space_size = len(action_space)
        do_nothing_idx = (
            action_space.index("do_nothing") if "do_nothing" in action_space else 0
        )

        print("Loading RL policies …")
        policies = load_policies(
            num_agents=num_agents,
            checkpoint_dir=paths["checkpoint_dir"],
            game_version=game_version,
            action_space_size=action_space_size,
            custom_checkpoints=self.config.custom_checkpoints or {},
        )

        # --- 6. Simulation timing ------------------------------------------
        total_time = self.config.total_simulation_time  # init + gameplay
        total_ticks = int(round(total_time / TICK_DURATION))
        init_ticks = int(
            round(self.config.agent_initialization_period / TICK_DURATION)
        )

        print(f"Total ticks: {total_ticks} ({total_time}s @ {TICK_DURATION}s/tick)")
        print(
            f"Init ticks: {init_ticks} "
            f"({self.config.agent_initialization_period}s)"
        )

        # --- 7. Main simulation loop ----------------------------------------
        for tick in range(total_ticks):
            in_init = tick < init_ticks

            # ---- 8a. Decide actions ----------------------------------------
            if in_init:
                # Initialization period: keep agents idle with do_nothing
                actions = {agent_id: do_nothing_idx for agent_id in env.agents}
            else:
                # Active gameplay: idle agents pick actions from policy
                actions = {}
                for agent_id in env.agents:
                    if agent_is_idle(env, agent_id):
                        try:
                            action_idx = policies[agent_id].get_action(obs[agent_id])
                        except Exception as exc:
                            print(
                                f"Warning: policy error for {agent_id}: {exc}"
                            )
                            action_idx = do_nothing_idx
                        actions[agent_id] = action_idx
                    else:
                        # Busy agents: pass do_nothing (ignored by env anyway)
                        actions[agent_id] = do_nothing_idx

            # ---- 8b. Step one tick -----------------------------------------
            obs, _rewards, _terminations, _truncations, _infos = env.step(actions)

            # ---- 8c. Log (only outside init period) ------------------------
            data_logger.log_collisions(tick, 1.0 / TICK_DURATION, env)
            if not in_init:
                data_logger.log_positions(tick, 1.0 / TICK_DURATION, env.game)
                data_logger.log_counters(tick, 1.0 / TICK_DURATION, env.game)

                # _logging_actions populated by env.step for idle agents that
                # received a valid (non–do_nothing) action this tick
                if hasattr(env, "_logging_actions") and env._logging_actions:
                    data_logger.log_actions(
                        tick, 1.0 / TICK_DURATION, env._logging_actions, env.game
                    )

            # ---- 8d. Progress report ----------------------------------------
            if tick % max(1, total_ticks // 20) == 0:
                pct = int(100 * tick / total_ticks)
                sim_time = tick * TICK_DURATION
                if in_init:
                    status = (
                        f"(init {sim_time:.1f}s / "
                        f"{self.config.agent_initialization_period}s)"
                    )
                else:
                    active = sim_time - self.config.agent_initialization_period
                    status = (
                        f"(active {active:.1f}s / "
                        f"{self.config.duration_seconds}s)"
                    )
                print(f"  {pct}% — tick {tick}/{total_ticks} {status}")
                sys.stdout.flush()

        # --- 8. Cleanup and return results ----------------------------------
        data_logger.close()
        print("Simulation complete.")

        output_paths = data_logger.get_output_paths()
        return output_paths

