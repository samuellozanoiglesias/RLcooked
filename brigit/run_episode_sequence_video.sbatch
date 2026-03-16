#!/bin/bash
#SBATCH --job-name=episode_sequence_video
#SBATCH -t 30-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=qos_gpu_long
#SBATCH --account=teruel
#SBATCH --ntasks=24
#SBATCH --gres=gpu:1
#SBATCH --output=results-episode_video-%j.out
#SBATCH --error=error-episode_video-%j.err

# Activar entorno conda
source ~/.bashrc
conda activate RLcooked

# Crear carpetas si no existen
mkdir -p logs

# Script path
SCRIPT_DIR='/mnt/lustre/home/samuloza/cooked'
VIDEO_SCRIPT="${SCRIPT_DIR}/generate_episode_sequence_video.py"

# Default parameters (modify as needed)
START_EPISODE=50
END_EPISODE=5950
STEP=50
CLUSTER="brigit"
SYNERGY=0.8
SPECIALIZATION=0.05
STUDY_NAME="HEATMAP"
NUM_EPISODES=5
FPS=2  # 2 fps = 0.5 seconds per frame

# Map names (optional, defaults will be used if not specified)
MAP_NAME_1="baseline_division_of_labor_large"
MAP_NAME_2="encouraged_division_of_labor_large"
INIT_TYPE="empty_init"

# Extended mode (set to --extended to enable, leave empty to disable)
EXTENDED_FLAG="--extended"

# Execution mode:
#   Leave empty to generate PNGs and create video
#   Set to "--video-only" to skip PNG generation and only create video from existing plots
EXECUTION_MODE=""

echo "=========================================="
echo "EPISODE SEQUENCE VIDEO GENERATION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Starting episode sequence video generation..."
echo "Parameters:"
echo "  Episode range: $START_EPISODE to $END_EPISODE (step $STEP)"
echo "  Study: $STUDY_NAME"
echo "  Cluster: $CLUSTER"
echo "  Synergy: $SYNERGY"
echo "  Specialization: $SPECIALIZATION"
echo "  Num episodes per plot: $NUM_EPISODES"
echo "  FPS: $FPS ($(python3 -c "print(1/$FPS)") seconds per frame)"
echo "  Map 1: $MAP_NAME_1"
echo "  Map 2: $MAP_NAME_2"
echo "  Init type: $INIT_TYPE"
echo "  Extended mode: ${EXTENDED_FLAG:-disabled}"
echo "  Execution mode: ${EXECUTION_MODE:-generate PNGs and video}"
echo "=========================================="
echo ""

# Build command
CMD="python3 $VIDEO_SCRIPT \
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
    --fps $FPS"

# Add optional flags
if [ -n "$EXTENDED_FLAG" ]; then
    CMD="$CMD $EXTENDED_FLAG"
fi

if [ -n "$EXECUTION_MODE" ]; then
    CMD="$CMD $EXECUTION_MODE"
fi

echo "Executing command:"
echo "$CMD"
echo ""

# Execute the command
eval $CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "VIDEO GENERATION COMPLETED SUCCESSFULLY"
else
    echo "VIDEO GENERATION FAILED (exit code: $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE
