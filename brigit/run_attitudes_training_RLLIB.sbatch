#!/bin/bash
#SBATCH --job-name=RLLIB_cooked_attitudes
#SBATCH -t 30-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=qos_gpu_long
#SBATCH --account=teruel
#SBATCH --ntasks=24
#SBATCH --gres=gpu:1
#SBATCH --output=results-GPU-%x-%j.out
#SBATCH --error=error-GPU-%x-%j.err

# Activar entorno conda
source ~/.bashrc
conda activate RLcooked

# Set environment variables for JAX GPU usage
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# Crear carpetas si no existen
mkdir -p inputs
mkdir -p logs

TRAIN_SCRIPT='/mnt/lustre/home/samuloza/cooked/training-DTDE-spoiled_broth.py'

CLUSTER=brigit
MAP_NR=baseline_competition
LR=3e-4
GAME_VERSION=competition
NUM_AGENTS=2
NUM_EPOCHS=2000  # Number of training epochs
SEEDS=(10 20 30 40 50 60 70 80 90)  # Different random seeds for runs
CHECKPOINT_PATHS=None  # Set to a specific checkpoint name or None
REWARDS_ON_DELIVERY_ONLY=True  # Whether to use rewards on delivery only

# Define arrays for angles and seeds
ANGLE_PAIRS=("0 0" "0 45" "0 315" "45 0" "315 0")
WALKING_SPEED_1=1.0
WALKING_SPEED_2=1.0
CUTTING_SPEED_1=1.0
CUTTING_SPEED_2=1.0

# Maximum number of parallel jobs
MAX_PARALLEL=2  # Set this to control how many jobs run in parallel

# Count number of jobs = seeds × angle pairs
NUM_SEEDS=${#SEEDS[@]}
NUM_ANGLES=${#ANGLE_PAIRS[@]}
NUM_JOBS=$(( NUM_SEEDS * NUM_ANGLES ))

TOTAL_CPUS=$SLURM_NTASKS
TOTAL_GPUS=1

# Resource allocation per job
export RLLIB_GPU_FRACTION=$(python3 - <<EOF
print(1)  # Use full GPU per job (Ray requires whole numbers)
EOF
)

# CPUs per job (minimum 1)
export RLLIB_CPUS_PER_WORKER=$(python3 - <<EOF
import math
print(max(1, math.floor($TOTAL_CPUS / $MAX_PARALLEL)))
EOF
)

echo "=== Auto resource allocation ==="
echo "NUM_SEEDS = $NUM_SEEDS"
echo "NUM_ANGLES = $NUM_ANGLES"
echo "NUM_JOBS = $NUM_JOBS"
echo "MAX_PARALLEL = $MAX_PARALLEL"
echo "GPU per job = 1 (full GPU)"
echo "CPUs per job = $RLLIB_CPUS_PER_WORKER"
echo "================================"

# Function to wait for jobs to complete
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 30
    done
}

# Create all job combinations first
declare -a job_list=()
for ANGLES in "${ANGLE_PAIRS[@]}"; do
    read ANGLE1 ANGLE2 <<< "$ANGLES"
    for SEED in "${SEEDS[@]}"; do
        job_list+=("$ANGLE1 $ANGLE2 $SEED")
    done
done

echo "Total jobs to run: ${#job_list[@]}"

# Execute jobs with parallel limit
job_count=0
for job in "${job_list[@]}"; do
    read ANGLE1 ANGLE2 SEED <<< "$job"
    
    # Wait for a slot to be available
    wait_for_slot
    
    job_count=$((job_count + 1))
    
    RAD1=$(python3 -c "import numpy as np; print(np.radians($ANGLE1))")
    RAD2=$(python3 -c "import numpy as np; print(np.radians($ANGLE2))")
    ALPHA1=$(python3 -c "import numpy as np; print(np.cos($RAD1))")
    BETA1=$(python3 -c "import numpy as np; print(np.sin($RAD1))")
    ALPHA2=$(python3 -c "import numpy as np; print(np.cos($RAD2))")
    BETA2=$(python3 -c "import numpy as np; print(np.sin($RAD2))")

    INPUT_FILE="inputs/input_${ANGLE1}_${ANGLE2}.txt"
    LOG_FILE="logs/out_${ANGLE1}_${ANGLE2}_lr${LR}_${GAME_VERSION}_${MAP_NR}_seed_${SEED}_${REWARDS_ON_DELIVERY_ONLY}.log"

    echo -e "${ALPHA1} ${BETA1}\n${WALKING_SPEED_1} ${CUTTING_SPEED_1}\n${ALPHA2} ${BETA2}\n${WALKING_SPEED_2} ${CUTTING_SPEED_2}" > "$INPUT_FILE"

    echo "Starting job $job_count/${#job_list[@]}: MAP_NR=$MAP_NR ANGLE1=$ANGLE1 ANGLE2=$ANGLE2 LR=$LR SEED=$SEED"
    python3 "$TRAIN_SCRIPT" $CLUSTER $INPUT_FILE $MAP_NR $LR $GAME_VERSION $NUM_AGENTS $NUM_EPOCHS $SEED $CHECKPOINT_PATHS $REWARDS_ON_DELIVERY_ONLY > "$LOG_FILE" 2>&1 &
    
    sleep 30  # Delay between job starts
done

# Wait for all remaining jobs to complete
echo "Waiting for all jobs to complete..."
wait
