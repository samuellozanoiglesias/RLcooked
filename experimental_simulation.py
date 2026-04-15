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
  game_version: Game version (classic/classic_collision/competition) - controls both game logic and collision detection
  training_id: Optional training identifier (auto-extracted from custom checkpoints if omitted)
  checkpoint_number: Optional checkpoint number (auto-extracted from custom checkpoints if omitted)

Note: training_id and checkpoint_number are optional when using --custom_checkpoints 
      (defaults to "custom" and auto-extracts from first agent's checkpoint path)
      study_name is optional (if omitted, simulations saved directly under /simulations/ without study subfolder)
      game_version="classic_collision" enables collision detection

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
# --game_type classic_collision (for folder organization)
"""

import sys
import os
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


def main():
    """Main execution function."""
    parser = setup_simulation_argument_parser()
    args = parser.parse_args()
    
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
    print(f"Game type (folder): {args.game_type}")
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
            game_type=args.game_type
        )
        
        # Setup logging to save in simulation directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = output_paths['simulation_dir'] / f"experimental_simulation_{timestamp}.log"
        
        # Continue with the rest of the execution while logging to file
        with TeeLogger(log_file_path):
            print("\nSimulation completed successfully!")
            print("=" * 50)
            print("Output files:")
            print(f"  Simulation directory: {output_paths['simulation_dir']}")
            print(f"  Configuration file: {output_paths['config_file']}")
            print(f"  Actions CSV (basic): {output_paths.get('actions_csv', 'N/A')}")
            print(f"  Collisions CSV: {output_paths.get('collisions_csv', 'N/A')}")
            print(f"  Items CSV: {output_paths.get('items_csv', 'N/A')}")
            print(f"  Counters CSV: {output_paths.get('counter_csv', 'N/A')}")
            
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