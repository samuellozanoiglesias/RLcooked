#!/usr/bin/env python3
"""
Generate Grid Analysis Sequence GIF

This script runs grid_full_cooperative_analysis.py for multiple target episodes
and creates an animated GIF showing the progression of 2D grid plots over episodes.

Output format: Animated GIF (no ffmpeg or codec required — uses Pillow only)

Usage:
    # Generate PNGs and create GIF (default):
    python generate_grid_analysis_video.py --start_episode 100 --end_episode 10000 --step 100 \
        --cluster brigit --synergy 0.4 --specialization 0.05 --study_name MODIFIED_SPECIALIZATION \
        --num_episodes 10 --fps 2

    # Only create GIF from existing PNGs (skip generation):
    python generate_grid_analysis_video.py --video-only --start_episode 100 --end_episode 10000 \
        --step 100 --cluster brigit --synergy 0.4 --specialization 0.05 \
        --study_name MODIFIED_SPECIALIZATION --num_episodes 10 --fps 2

    # Default: start=100, end=10000, step=100, fps=2 (0.5 seconds per frame)
    python generate_grid_analysis_video.py --cluster brigit --synergy 0.4 --specialization 0.05 \
        --study_name MODIFIED_SPECIALIZATION

Example:
    nohup python generate_grid_analysis_video.py --start_episode 100 --end_episode 10000 --step 100 \
        --cluster brigit --synergy 0.4 --specialization 0.05 --study_name MODIFIED_SPECIALIZATION \
        --num_episodes 10 --video-only > generate_grid_video.log 2>&1 &

Installation:
    pip install pillow
"""

import argparse
import subprocess
import os
import sys
import glob
import time
from pathlib import Path
import re


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate 2D grid plots for multiple target episodes and create a video',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Episode range parameters
    parser.add_argument('--start_episode', type=int, default=100,
                        help='Starting episode number (default: 100)')
    parser.add_argument('--end_episode', type=int, default=10000,
                        help='Ending episode number (default: 10000)')
    parser.add_argument('--step', type=int, default=100,
                        help='Step size between episodes (default: 100)')
    
    # Analysis script parameters
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to include in each analysis (default: 10)')
    parser.add_argument('--cluster', type=str, default='brigit',
                        help='Cluster name (default: brigit)')
    parser.add_argument('--synergy', type=float, default=0.4,
                        help='Synergy value (default: 0.4)')
    parser.add_argument('--specialization', type=float, default=0.05,
                        help='Specialization value (default: 0.05)')
    parser.add_argument('--study_name', type=str, required=False,
                        help='Study name')
    parser.add_argument('--init_type', type=str, default='empty_init',
                        help='Initialization type (default: empty_init)')
    parser.add_argument('--game_type', type=str, default='classic_collision',
                        help='Game type (default: classic_collision)')
    
    # Grid analysis specific parameters
    parser.add_argument('--comparison_type', type=str, 
                        choices=['global', 'row', 'column', 'all'], default='all',
                        help='Comparison type for grid analysis: global, row, column, or all (default: all)')
    
    # Video parameters
    parser.add_argument('--fps', type=float, default=2,
                        help='Frames per second for video (default: 2, i.e., 0.5 seconds per frame)')
    parser.add_argument('--output_video', type=str, default=None,
                        help='Output video filename (default: auto-generated)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots and video (default: auto-detected from cluster)')
    
    # Execution parameters
    parser.add_argument('--video-only', action='store_true',
                        help='Skip PNG generation and only create video from existing plots')
    parser.add_argument('--generate-plots', action='store_true',
                        help='Generate PNG plots (default: True unless --video-only is specified)')
    parser.add_argument('--parallel', action='store_true',
                        help='Run plot generation in parallel (experimental)')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Maximum number of parallel workers (default: 4)')
    
    return parser.parse_args()


def run_grid_analysis_for_episode(target_episode, args, script_dir):
    """Run the grid analysis script for a specific target episode."""
    
    # Build command
    cmd = [
        'python', 
        os.path.join(script_dir, 'grid_full_cooperative_analysis.py'),
        '--episode_range', 'specific',
        '--target_episode', str(target_episode),
        '--num_episodes', str(args.num_episodes),
        '--cluster', args.cluster,
        '--synergy', str(args.synergy),
        '--specialization', str(args.specialization),
        '--init_type', args.init_type,
        '--game_type', args.game_type
    ]
    
    if args.study_name:
        cmd.extend(['--study_name', args.study_name])

    if args.output_dir:
        cmd.extend(['--output_dir', args.output_dir])

    if args.comparison_type != 'all':
        cmd.extend(['--comparison_type', args.comparison_type])
    
    # Create log file name
    log_file = os.path.join(script_dir, f'grid_full_analysis_ep{target_episode}.log')
    
    print(f"Running grid analysis for episode {target_episode}...")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log file: {log_file}")
    
    # Run command and capture output
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=script_dir,
                timeout=600  # 10 minute timeout per plot
            )
        
        if result.returncode == 0:
            print(f"  ✓ Successfully completed episode {target_episode}")
            return True
        else:
            print(f"  ✗ Failed for episode {target_episode} (return code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout for episode {target_episode}")
        return False
    except Exception as e:
        print(f"  ✗ Error for episode {target_episode}: {e}")
        return False


def find_generated_grid_plots(output_dir, comparison_type, args):
    """Find all generated grid plot files matching the pattern."""
    
    # Build search pattern based on arguments - must match grid_full_cooperative_analysis.py naming
    pattern_parts = ["full_grid"]
    
    # Add 'specific' since we use episode_range=specific
    pattern_parts.append("specific")
    
    # Add num_episodes if not 100
    if args.num_episodes != 100:
        pattern_parts.append(f"{args.num_episodes}ep")
    
    # The target episode will be added as target{episode}, so use wildcard
    # pattern_parts.append("target*")
    
    if args.init_type != 'empty_init':
        pattern_parts.append(args.init_type)
    
    if args.synergy != 0.0:
        pattern_parts.append(f"synergy{args.synergy}")
    
    if args.specialization is not None:
        pattern_parts.append(f"lambda{args.specialization}")
    
    if args.game_type != 'classic':
        pattern_parts.append(args.game_type)
    
    # Pattern for finding files: full_grid_specific_{num_episodes}ep_target{episode}_synergy{val}_lambda{val}_{game_type}_{comparison_type}.png
    base_pattern = "_".join(pattern_parts)
    
    # Look in the appropriate subfolder for comparison type
    if comparison_type == 'row':
        subfolder = os.path.join(output_dir, 'grid_per_row')
    elif comparison_type == 'column':
        subfolder = os.path.join(output_dir, 'grid_per_column')
    else:  # global
        subfolder = os.path.join(output_dir)
    
    # Simplify: just look for files with target* pattern and comparison type
    search_pattern = os.path.join(subfolder, f"full_grid_*_target*_{comparison_type}.png")
    
    print(f"\nSearching for plots with pattern: {search_pattern}")
    
    plot_files = glob.glob(search_pattern)
    
    if not plot_files:
        print(f"Warning: No plots found matching pattern")
        # Try a more general pattern
        general_pattern = os.path.join(subfolder, f"full_grid_*_{comparison_type}.png")
        print(f"Trying general pattern: {general_pattern}")
        plot_files = glob.glob(general_pattern)
    
    # Sort by episode number - look for target{episode} pattern
    def extract_episode_number(filename):
        match = re.search(r'_target(\d+)_', filename)
        return int(match.group(1)) if match else 0
    
    plot_files.sort(key=extract_episode_number)
    
    print(f"Found {len(plot_files)} plot files")
    if plot_files:
        print(f"  First: {os.path.basename(plot_files[0])}")
        print(f"  Last: {os.path.basename(plot_files[-1])}")
    
    return plot_files


def create_gif_pillow(plot_files, output_gif, fps):
    """Create an animated GIF using Pillow — no ffmpeg or codec required."""

    try:
        from PIL import Image
    except ImportError as e:
        print(f"  ✗ Pillow not available: {e}")
        print("  Install with: pip install pillow")
        return False

    if not plot_files:
        print("Error: No plot files to create GIF")
        return False

    duration_ms = int(1000 / fps)  # milliseconds per frame

    print(f"\nCreating animated GIF with Pillow...")
    print(f"  Output: {output_gif}")
    print(f"  FPS: {fps} ({duration_ms} ms per frame)")
    print(f"  Number of frames: {len(plot_files)}")

    try:
        frames = []
        for i, plot_file in enumerate(plot_files):
            img = Image.open(plot_file).convert("RGB")
            frames.append(img)
            if (i + 1) % 10 == 0:
                print(f"  Loaded frame {i + 1}/{len(plot_files)}")

        frames[0].save(
            output_gif,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )

        if os.path.exists(output_gif):
            print(f"  ✓ GIF created successfully: {output_gif}")
            size_mb = os.path.getsize(output_gif) / (1024 * 1024)
            print(f"  GIF size: {size_mb:.2f} MB")
            return True
        else:
            print(f"  ✗ GIF file was not created")
            return False

    except Exception as e:
        print(f"  ✗ Error creating GIF with Pillow: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    args = parse_arguments()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine output directory
    if args.output_dir is None:
        # Auto-detect based on cluster
        from spoiled_broth.analysis.utils import AnalysisConfig
        config = AnalysisConfig()
        local_path = config.cluster_paths.get(args.cluster, '')
        args.output_dir = f"{local_path}/data/samuel_lozano/cooked/grid_full_analysis_figures"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING GRID ANALYSIS SEQUENCE VIDEO")
    print("=" * 70)
    print(f"Episode range: {args.start_episode} to {args.end_episode} (step: {args.step})")
    print(f"Study: {args.study_name}")
    print(f"Cluster: {args.cluster}")
    print(f"Synergy: {args.synergy}")
    print(f"Specialization: {args.specialization}")
    print(f"Game type: {args.game_type}")
    print(f"Init type: {args.init_type}")
    print(f"Comparison types: {', '.join(['global', 'row', 'column'] if args.comparison_type == 'all' else [args.comparison_type])}")
    print(f"Number of episodes per analysis: {args.num_episodes}")
    print(f"Output directory: {args.output_dir}")  
    print(f"Video FPS: {args.fps} ({1/args.fps:.2f} seconds per frame)")
    print(f"Mode: {'Video only (using existing PNGs)' if args.video_only else 'Generate PNGs and create video'}")
    print("=" * 70)
    
    # Generate plots if not in video-only mode
    if not args.video_only:
        print("\nSTEP 1: Generating grid plots")
        print("=" * 70)
        
        episodes = range(args.start_episode, args.end_episode + 1, args.step)
        total_episodes = len(list(episodes))
        
        print(f"Will generate {total_episodes} grid plots")
        
        if args.parallel:
            print(f"Running with {args.max_workers} parallel workers")
            from concurrent.futures import ProcessPoolExecutor, as_completed
            
            with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {
                    executor.submit(run_grid_analysis_for_episode, ep, args, script_dir): ep 
                    for ep in episodes
                }
                
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    episode = futures[future]
                    try:
                        success = future.result()
                        status = "✓" if success else "✗"
                        print(f"[{completed}/{total_episodes}] {status} Episode {episode}")
                    except Exception as e:
                        print(f"[{completed}/{total_episodes}] ✗ Episode {episode} - Error: {e}")
        else:
            # Sequential execution
            completed = 0
            for target_episode in episodes:
                completed += 1
                print(f"\n[{completed}/{total_episodes}] Processing episode {target_episode}")
                run_grid_analysis_for_episode(target_episode, args, script_dir)
                time.sleep(0.5)  # Small delay between runs
        
        print("\n" + "=" * 70)
        print("Grid plot generation completed")
        print("=" * 70)
    
    # Create a GIF for each requested comparison type
    all_comparison_types = ['global', 'row', 'column'] if args.comparison_type == 'all' else [args.comparison_type]
    results = {}

    for comparison_type in all_comparison_types:
        print(f"\nSTEP 2/{all_comparison_types.index(comparison_type)+1}: Finding plots for '{comparison_type}'")
        print("=" * 70)

        plot_files = find_generated_grid_plots(str(output_dir), comparison_type, args)

        if not plot_files:
            print(f"Warning: No plots found for '{comparison_type}'. Skipping.")
            results[comparison_type] = None
            continue

        # Build GIF filename for this comparison type
        gif_name_parts = [
            "grid_sequence",
            comparison_type,
            args.init_type if args.init_type != 'empty_init' else None,
            f"synergy_{args.synergy:.1f}" if args.synergy != 0.0 else None,
            f"spec_{args.specialization:.2f}" if args.specialization is not None else None,
            args.game_type if args.game_type != 'classic' else None,
            f"ep{args.start_episode}-{args.end_episode}_step{args.step}"
        ]
        gif_name_parts = [p for p in gif_name_parts if p is not None]
        if args.study_name:
            gif_name_parts.insert(1, args.study_name)

        # Global GIF lives in root output_dir; row/column in their subfolders
        if comparison_type == 'row':
            gif_dir = str(output_dir / 'grid_per_row')
        elif comparison_type == 'column':
            gif_dir = str(output_dir / 'grid_per_column')
        else:
            gif_dir = str(output_dir)

        output_gif = os.path.join(gif_dir, "_".join(gif_name_parts) + ".gif")

        print(f"\nSTEP 3/{all_comparison_types.index(comparison_type)+1}: Creating GIF for '{comparison_type}'")
        print("=" * 70)
        success = create_gif_pillow(plot_files, output_gif, args.fps)
        results[comparison_type] = output_gif if success else None

    # Final summary
    print("\n" + "=" * 70)
    any_success = any(v is not None for v in results.values())
    if any_success:
        print("GRID ANALYSIS GIF GENERATION COMPLETED!")
        print("=" * 70)
        for ct, path in results.items():
            if path:
                n = len(find_generated_grid_plots(str(output_dir), ct, args))
                print(f"  [{ct}] {path}")
                print(f"        {n} frames, {n/args.fps:.1f} seconds")
            else:
                print(f"  [{ct}] SKIPPED (no plots found)")
    else:
        print("GIF GENERATION FAILED FOR ALL COMPARISON TYPES")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()