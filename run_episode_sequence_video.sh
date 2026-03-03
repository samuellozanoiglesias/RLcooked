#!/bin/bash
# Example script to run the episode sequence video generation
# 
# This script runs the video generation in the background with nohup
#
# Usage: 
#   ./run_episode_sequence_video.sh
#
# Or customize the parameters as needed

# Default parameters (modify as needed)
START_EPISODE=100
END_EPISODE=10000
STEP=100
CLUSTER="brigit"
SYNERGY=0.4
SPECIALIZATION=0.05
STUDY_NAME="MODIFIED_SPECIALIZATION"
NUM_EPISODES=10
FPS=2  # 2 fps = 0.5 seconds per frame

# Map names (optional, defaults will be used if not specified)
MAP_NAME_1="baseline_division_of_labor_large"
MAP_NAME_2="encouraged_division_of_labor_large"
INIT_TYPE="empty_init"

# Log file
LOG_FILE="generate_episode_sequence_video.log"

echo "Starting episode sequence video generation..."
echo "Parameters:"
echo "  Episode range: $START_EPISODE to $END_EPISODE (step $STEP)"
echo "  Study: $STUDY_NAME"
echo "  Cluster: $CLUSTER"
echo "  Synergy: $SYNERGY"
echo "  Specialization: $SPECIALIZATION"
echo "  Num episodes per plot: $NUM_EPISODES"
echo "  FPS: $FPS (${1/FPS} seconds per frame)"
echo "  Log file: $LOG_FILE"
echo ""
echo "Running in background with nohup..."

nohup python generate_episode_sequence_video.py \
    --start_episode $START_EPISODE \
    --end_episode $END_EPISODE \
    --step $STEP \
    --cluster $CLUSTER \
    --synergy $SYNERGY \
    --specialization $SPECIALIZATION \
    --study_name $STUDY_NAME \
    --num_episodes $NUM_EPISODES \
    --map_name_1 $MAP_NAME_1 \
    --map_name_2 $MAP_NAME_2 \
    --init_type $INIT_TYPE \
    --fps $FPS \
    > $LOG_FILE 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo "Monitor progress with: tail -f $LOG_FILE"
echo "To check if still running: ps -p $PID"
