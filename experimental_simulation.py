#!/usr/bin/env python3
"""
Experimental simulation runner for reinforcement learning experiments.

This script runs simulations of trained RL agents with comprehensive logging,
video recording, and data collection capabilities. After the simulation completes,
it automatically performs meaningful actions analysis to detect and categorize
agent behaviors based on item state changes.

The script automatically saves all output (including meaningful actions analysis)
to a log file in the simulation directory.

IMPORTANT: The simulation includes a 15-second initialization period where agents
are positioned but do not perform any actions. This ensures complete system
stabilization and prevents teleportation artifacts.

Output location:
After the run, the simulation folder is moved into a tree that mirrors the training tree:
  training:   <base>/empty_init/map_X/synergy_S/specialized_P/Training_T/
  simulation: <base>/simulations/empty_init/map_X/synergy_S/specialized_P/abilities_<W>_<C>/Training_T/simulation_<timestamp>/
where W = WALKING_SPEEDS['ai_rl_1'] and C = CUTTING_SPEEDS['ai_rl_2'] from Training_T/config.txt.
For checkpoints other than "final", a checkpoint_<n>/ folder is added below Training_T/.
Options: --simulations_root DIR (override <base>/simulations), --training_path DIR (when not
using --custom_checkpoints), --no_relocate (keep the old behaviour).

Custom Checkpoints:
Use --custom_checkpoints to specify different policies/checkpoints for each agent.

When using custom checkpoints:
- training_id and checkpoint_number become OPTIONAL (default: "custom")
- If OMITTED: training_id, checkpoint_number, synergy, and specialization folders are automatically extracted from the FIRST agent's checkpoint path
  - Agent speeds are loaded from each agent's respective training config.txt
  - Output is organized using the first agent's training info with synergy/specialization structure
  - Example: .../map_X/synergy_0.40/specialized_0.05/simulations/Training_ID/checkpoint_final/
- If PROVIDED: Used to override the detection and organize output directories
- study_name is optional (default: empty, which saves under /simulations/ without study subfolder)

Checkpoint configuration file format (3 lines per agent):
Line 1: policy_id_to_load (e.g., policy_ai_rl_1, policy_ai_rl_2, etc.)
Line 2: checkpoint_number (e.g., "final", "1000", etc.)
Line 3: path_to_training_directory (not checkpoint directory)

Example checkpoints_config.txt for 2 agents:
policy_ai_rl_1
final
/path/to/Training_2025-11-15_13-23-45
policy_ai_rl_2  
1000
/path/to/Training_2025-11-16_14-30-12

Usage:
python experimental_simulation.py <map_nr> <game_version> [training_id] [checkpoint_number] [options]

Arguments:
  map_nr: Map name identifier
    game_version: Game version (classic/classic_collision/competition/competition_collision) - controls both game logic and collision detection
  training_id: Optional training identifier (auto-extracted from custom checkpoints if omitted)
  checkpoint_number: Optional checkpoint number (auto-extracted from custom checkpoints if omitted)

Note: training_id and checkpoint_number are optional when using --custom_checkpoints 
      (defaults to "custom" and auto-extracts from first agent's checkpoint path)
      study_name is optional (if omitted, simulations saved directly under /simulations/ without study subfolder)
    game_version="classic_collision" or "competition_collision" enables collision detection

For background execution:
nohup python experimental_simulation.py <map_nr> <game_version> [training_id] [checkpoint_number] [options] > experimental_simulation.log 2>&1 &

Examples:

# Using custom checkpoints with collision detection
nohup python experimental_simulation.py encouraged_division_of_labor_large classic_collision --custom_checkpoints ./cuenca/experimental_checkpoints/checkpoints_encouraged_collision_1.0.txt > experimental_simulation.log 2>&1 &

# Using custom checkpoints without collision (classic mode)
nohup python experimental_simulation.py encouraged_division_of_labor_large classic --custom_checkpoints ./cuenca/experimental_checkpoints/checkpoints_encouraged_collision_1.0.txt > experimental_simulation.log 2>&1 &

# Using custom checkpoints with specific training for speed loading
nohup python experimental_simulation.py encouraged_division_of_labor_large classic_collision Training_20250115 final --custom_checkpoints ./cuenca/experimental_checkpoints/checkpoints_encouraged_collision_1.0.txt > experimental_simulation.log 2>&1 &

# Using standard checkpoint loading (training_id and checkpoint_number required)
nohup python experimental_simulation.py encouraged_division_of_labor_large classic_collision Training_20250115 final > experimental_simulation.log 2>&1 &

# Additional options examples:
# --num_agents 2
# --enable_video true 
# --duration 180
# --agent_initialization_period 15
# --study_name collision_test
"""

import ast
import sys
import os
import shutil
import logging
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spoiled_broth.simulations import (
    setup_simulation_argument_parser,
    main_simulation_pipeline
)


class TeeLogger:
    """Class to duplicate output to both console and file"""
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_file = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def __enter__(self):
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
        sys.stdout = self
        sys.stderr = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log_file:
            self.log_file.close()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
    def write(self, text):
        self.original_stdout.write(text)
        self.original_stdout.flush()
        if self.log_file:
            self.log_file.write(text)
            self.log_file.flush()
            
    def flush(self):
        self.original_stdout.flush()
        if self.log_file:
            self.log_file.flush()


# =============================================================================
# Output relocation helpers
# =============================================================================
# The simulation pipeline writes its output next to the training folder. After the
# run finishes, the simulation folder is moved to a mirrored tree under
# <base>/simulations/, e.g.
#
#   training:   /data/.../cooked/classic_collision/empty_init/map_X/synergy_1.35/specialized_0.25/Training_T/
#   simulation: /data/.../cooked/classic_collision/simulations/empty_init/map_X/synergy_1.35/specialized_0.25/
#                   abilities_<walk ai_rl_1>_<cut ai_rl_2>/Training_T/simulation_<timestamp>/
#
# Abilities are read from the training's config.txt:
#   WALKING_SPEEDS: {'ai_rl_1': 0.4, 'ai_rl_2': 1.0}   -> 0.4 (ai_rl_1)
#   CUTTING_SPEEDS: {'ai_rl_1': 1.0, 'ai_rl_2': 0.2}   -> 0.2 (ai_rl_2)
# For checkpoints other than "final", a checkpoint_<n>/ level is added below Training_T/
# so that different checkpoints never mix in the same folder.


def read_custom_checkpoint_file(checkpoint_file):
    """Return (checkpoint_number, training_path) of the FIRST agent in a checkpoint file."""
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"Checkpoint file has fewer than 3 non-empty lines: {checkpoint_file}")
    return lines[1], Path(lines[2])


def _read_speed_dict(config_path, key):
    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(f"{key}:"):
                value = ast.literal_eval(stripped.split(':', 1)[1].strip())
                if not isinstance(value, dict):
                    raise ValueError(f"{key} in {config_path} is not a dict: {value!r}")
                return value
    raise ValueError(f"{key} not found in {config_path}")


def read_abilities(training_path):
    """Return (walking speed of ai_rl_1, cutting speed of ai_rl_2) from config.txt."""
    config_path = Path(training_path) / "config.txt"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.txt not found in {training_path}")
    walking = _read_speed_dict(config_path, "WALKING_SPEEDS")
    cutting = _read_speed_dict(config_path, "CUTTING_SPEEDS")
    return float(walking['ai_rl_1']), float(cutting['ai_rl_2'])


def compute_relocation_parent(training_path, checkpoint_number, simulations_root=None, output_subfolder=None):
    """Build the folder (without simulation_<timestamp>) where the simulation should live."""
    parts = Path(training_path).parts
    if not parts or not parts[-1].startswith("Training_"):
        raise ValueError(f"Training path does not end in a Training_* folder: {training_path}")

    map_idx = next((i for i in range(len(parts) - 2, -1, -1) if parts[i].startswith("map_")), None)
    if map_idx is None or map_idx < 2:
        raise ValueError(f"Could not find a map_* folder (with an init folder above it) in: {training_path}")

    init_folder = parts[map_idx - 1]                 # e.g. empty_init
    base_dir = Path(*parts[:map_idx - 1])            # e.g. /data/.../cooked/classic_collision
    middle = parts[map_idx:-1]                       # map_*/synergy_*/specialized_*
    training_name = parts[-1]

    walk_1, cut_2 = read_abilities(training_path)
    abilities_folder = f"abilities_{walk_1!r}_{cut_2!r}"

    root = Path(simulations_root) if simulations_root else base_dir / "simulations"
    target = root / init_folder
    if output_subfolder:
        target = target / output_subfolder          # e.g. empty_init/with_pretraining/
    target = target / Path(*middle) / abilities_folder / training_name
    if str(checkpoint_number) != "final":
        target = target / f"checkpoint_{checkpoint_number}"
    return target


def relocate_simulation_output(output_paths, target_parent):
    """Move the finished simulation folder into target_parent and rewrite output_paths."""
    old_dir = Path(output_paths['simulation_dir'])
    if not old_dir.name.startswith("simulation"):
        # Safety guard: never move a folder that could be shared by several runs.
        raise ValueError(f"Unexpected simulation folder name '{old_dir.name}', refusing to move it")

    target_parent.mkdir(parents=True, exist_ok=True)
    new_dir = target_parent / old_dir.name
    suffix = 1
    while new_dir.exists():
        new_dir = target_parent / f"{old_dir.name}_{suffix}"
        suffix += 1

    shutil.move(str(old_dir), str(new_dir))

    for key, value in list(output_paths.items()):
        if isinstance(value, (str, Path)):
            try:
                rel = Path(value).relative_to(old_dir)
            except ValueError:
                continue
            new_value = new_dir / rel
            output_paths[key] = new_value if isinstance(value, Path) else str(new_value)
    output_paths['simulation_dir'] = new_dir
    return old_dir, new_dir


def main():
    """Main execution function."""
    parser = setup_simulation_argument_parser()
    parser.add_argument(
        '--simulations_root', type=str, default=None,
        help="Root folder for relocated simulations (default: <base>/simulations, where <base> "
             "is the folder above the init folder, e.g. /data/.../cooked/classic_collision)")
    parser.add_argument(
        '--training_path', type=str, default=None,
        help="Training folder used to build the output structure when not using --custom_checkpoints")
    parser.add_argument(
        '--output_subfolder', type=str, default=None,
        help="Extra folder inserted after the init folder in the output path, "
             "e.g. with_pretraining -> simulations/empty_init/with_pretraining/map_*/...")
    parser.add_argument(
        '--original_training_path', type=str, default=None,
        help="Real training folder when the agents are simulated with overridden speeds "
             "(set by the launcher's --override_abilities); recorded in speed_override.txt")
    parser.add_argument(
        '--no_relocate', action='store_true',
        help="Keep the output where the simulation pipeline writes it")
    args = parser.parse_args()

    # Work out the final output location BEFORE running, so config problems show up early
    relocation_parent = None
    if not args.no_relocate:
        try:
            if args.custom_checkpoints:
                ckpt_number, training_path = read_custom_checkpoint_file(args.custom_checkpoints)
            elif args.training_path:
                ckpt_number, training_path = args.checkpoint_number, Path(args.training_path)
            else:
                raise ValueError("needs --custom_checkpoints or --training_path")
            relocation_parent = compute_relocation_parent(
                training_path, ckpt_number, args.simulations_root, args.output_subfolder)
        except Exception as e:
            print(f"WARNING: output relocation disabled ({e}). Output stays in the default location.")
    
    # Convert enable_video string to boolean
    enable_video = args.enable_video.lower() == 'true'
    
    # Print initial startup information
    startup_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Starting experimental simulation at: {startup_time}")
    print(f"Map: {args.map_nr}")
    print(f"Agents: {args.num_agents}")
    print(f"Game version: {args.game_version}")
    print(f"Training ID: {args.training_id}")
    print(f"Checkpoint: {args.checkpoint_number}")
    print(f"Video recording: {'Enabled' if enable_video else 'Disabled'}")
    print(f"Cluster: {args.cluster}")
    print(f"Duration: {args.duration} seconds gameplay (+ {args.agent_initialization_period}s initialization = {args.duration + args.agent_initialization_period}s total)")
    print(f"Tick rate: {args.tick_rate} FPS")
    print(f"Checkpoint config: {args.custom_checkpoints}")
    print(f"Study name: {args.study_name}")
    print(f"Output parent: {relocation_parent if relocation_parent else 'default pipeline location'}")
    print(f"Note: First {args.agent_initialization_period} seconds are agent initialization period (no actions)")
    print(f"Note: Requested duration refers to active gameplay time, not total simulation time")
    print("=" * 50)
    
    try:
        # Run main simulation pipeline (ALWAYS without video rendering)
        # Video will be generated afterwards from CSV files if requested
        output_paths = main_simulation_pipeline(
            map_nr=args.map_nr,
            num_agents=args.num_agents,
            game_version=args.game_version,
            training_id=args.training_id,
            checkpoint_number=args.checkpoint_number,
            enable_video=False,  # Disable video during simulation
            cluster=args.cluster,
            duration=args.duration,
            tick_rate=args.tick_rate,
            video_fps=args.video_fps,
            agent_initialization_period=args.agent_initialization_period,
            custom_checkpoints=args.custom_checkpoints,
            study_name=args.study_name,
        )
        
        # Move the finished simulation into the mirrored simulations/ tree
        relocation_info = None
        if relocation_parent is not None:
            try:
                relocation_info = relocate_simulation_output(output_paths, relocation_parent)
            except Exception as e:
                print(f"WARNING: could not relocate simulation output ({e}). "
                      f"Output stays in: {output_paths['simulation_dir']}")

        # Record where the agents came from when speeds were overridden
        if args.original_training_path:
            try:
                trained = read_abilities(args.original_training_path)
                simulated_from = read_custom_checkpoint_file(args.custom_checkpoints)[1] \
                    if args.custom_checkpoints else None
                simulated = read_abilities(simulated_from) if simulated_from else None
                with open(Path(output_paths['simulation_dir']) / "speed_override.txt", 'w', encoding='utf-8') as f:
                    f.write(f"agents_loaded_from: {args.original_training_path}\n")
                    f.write(f"trained_abilities (walk ai_rl_1, cut ai_rl_2): {trained}\n")
                    f.write(f"simulated_abilities (walk ai_rl_1, cut ai_rl_2): {simulated}\n")
            except Exception as e:
                print(f"WARNING: could not write speed_override.txt ({e})")

        # Setup logging to save in simulation directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = output_paths['simulation_dir'] / f"experimental_simulation_{timestamp}.log"
        
        # Continue with the rest of the execution while logging to file
        with TeeLogger(log_file_path):
            print("\nSimulation completed successfully!")
            print("=" * 50)
            if relocation_info:
                print(f"Simulation moved from: {relocation_info[0]}")
                print(f"                   to: {relocation_info[1]}")
            print("Output files:")
            print(f"  Simulation directory: {output_paths['simulation_dir']}")
            print(f"  Configuration file: {output_paths['config_file']}")
            print(f"  Actions CSV (basic): {output_paths.get('actions_csv', 'N/A')}")
            print(f"  Collisions CSV: {output_paths.get('collisions_csv', 'N/A')}")
            print(f"  Items CSV: {output_paths.get('items_csv', 'N/A')}")
            print(f"  Items full CSV: {output_paths.get('items_full_csv', 'N/A')}")
            print(f"  Counters CSV: {output_paths.get('counter_csv', 'N/A')}")
            print(f"  Counters full CSV: {output_paths.get('counter_full_csv', 'N/A')}")
            
            # Basic position files (one per agent)
            print("\n  Basic position files:")
            for key, path in output_paths.items():
                if key.startswith('positions_ai_rl_'):
                    print(f"    {key}: {path}")
            
            # Human-readable files (generated during simulation)
            print("\n  Derived per-agent action tables:")
            has_human_files = False
            for key, path in output_paths.items():
                if key.startswith('actions_ai_rl_'):
                    print(f"    {key}.csv: {path}")
                    has_human_files = True
            
            if not has_human_files:
                print("    (No per-agent action tables were generated)")
            
            print(f"\n  Log file: {log_file_path}")
            
            # Generate video from CSV if requested
            if enable_video:
                print("\n" + "=" * 50)
                print("Reconstructing video from CSV files...")
                print("=" * 50)
                
                from spoiled_broth.simulations.replay_simulation_video import replay_from_directory
                
                video_path = replay_from_directory(
                    str(output_paths['simulation_dir']),
                    fps=args.video_fps,
                    tile_size=96  # Larger tile size for better visibility
                )
                
                if video_path:
                    print(f"  Video file: {video_path}")
                    output_paths['video_file'] = video_path
                else:
                    print("  Video generation failed (see errors above)")
            else:
                print(f"\nVideo recording: Disabled (use --enable_video true to generate video)")
            
            print("\n" + "=" * 50)
            print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Full log saved to: {log_file_path}")
            print("\nNote: Derived per-agent action tables are written as actions_{agent_id}.csv.")
            print("      They are generated automatically during simulation with item tracking.")
        
    except Exception as e:
        # Try to save error log to simulation directory if possible
        error_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        error_msg = f"Error during simulation at {error_time}: {e}"
        print(error_msg)
        
        # If we have output_paths, try to save error log there
        try:
            if 'output_paths' in locals() and output_paths:
                error_log_path = output_paths['simulation_dir'] / f"experimental_simulation_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                with open(error_log_path, 'w') as f:
                    f.write(f"Experimental Simulation Error Log\n")
                    f.write(f"Time: {error_time}\n")
                    f.write(f"Error: {e}\n")
                    f.write(f"Arguments: {vars(args)}\n")
                print(f"Error log saved to: {error_log_path}")
        except Exception as log_error:
            print(f"Could not save error log: {log_error}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()