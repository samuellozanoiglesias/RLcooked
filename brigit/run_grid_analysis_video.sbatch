#!/bin/bash
#SBATCH --job-name=grid_analysis_video
#SBATCH -t 12:00:00
#SBATCH --partition=short
#SBATCH --account=teruel
#SBATCH --ntasks=1
#SBATCH --output=results-grid_video-%j.out
#SBATCH --error=error-grid_video-%j.err

# Activar entorno conda
source ~/.bashrc
conda activate cooked

# Crear carpetas si no existen
mkdir -p logs

# Script path
SCRIPT_DIR='/mnt/lustre/home/samuloza/cooked'
VIDEO_SCRIPT="${SCRIPT_DIR}/generate_grid_analysis_video.py"

# Default parameters (modify as needed)
START_EPISODE=50
END_EPISODE=5950
STEP=50
CLUSTER="brigit"
SYNERGY=0.80
SPECIALIZATION=0.05
STUDY_NAME="HEATMAP"
NUM_EPISODES=10
FPS=2  # 2 fps = 0.5 seconds per frame

# Map and init parameters
INIT_TYPE="empty_init"
GAME_TYPE="classic_collision"

# Comparison type: global, row, column, or all (generates all three)
COMPARISON_TYPE="all"

# Execution mode:
#   Leave empty to generate PNGs and create GIF
#   Set to "--video-only" to skip PNG generation and only create GIF from existing plots
EXECUTION_MODE=""

echo "=========================================="
echo "GRID ANALYSIS SEQUENCE VIDEO GENERATION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Starting grid analysis video generation..."
echo "Parameters:"
echo "  Episode range: $START_EPISODE to $END_EPISODE (step $STEP)"
echo "  Study: $STUDY_NAME"
echo "  Cluster: $CLUSTER"
echo "  Synergy: $SYNERGY"
echo "  Specialization: $SPECIALIZATION"
echo "  Num episodes per plot: $NUM_EPISODES"
echo "  FPS: $FPS ($(python3 -c "print(1/$FPS)") seconds per frame)"
echo "  Init type: $INIT_TYPE"
echo "  Game type: $GAME_TYPE"
echo "  Comparison type: $COMPARISON_TYPE"
echo "  Execution mode: ${EXECUTION_MODE:-generate PNGs and GIF}"
echo "=========================================="
echo ""

# Build command
CMD="python $VIDEO_SCRIPT \
    --start_episode $START_EPISODE \
    --end_episode $END_EPISODE \
    --step $STEP \
    --cluster $CLUSTER \
    --synergy $SYNERGY \
    --specialization $SPECIALIZATION \
    --study_name $STUDY_NAME \
    --num_episodes $NUM_EPISODES \
    --fps $FPS \
    --init_type $INIT_TYPE \
    --game_type $GAME_TYPE \
    --comparison_type $COMPARISON_TYPE"

# Add optional flags
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
