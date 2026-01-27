#!/bin/bash
#SBATCH --job-name=baseline
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
conda activate cooked

# Set environment variables for JAX GPU usage
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# Crear carpetas si no existen
mkdir -p inputs
mkdir -p logs

TRAIN_SCRIPT='/mnt/lustre/home/samuloza/cooked/training-DTDE-spoiled_broth.py'

CLUSTER=brigit
MAP_NR=encouraged_division_of_labor_large
LR=3e-4
GAME_VERSION=classic
NUM_AGENTS=1
NUM_EPOCHS=50  # Number of training epochs
SEEDS=(0 42 96)  # Different random seeds for runs
REWARDS_ON_DELIVERY_ONLY=False  # Whether to use rewards on delivery only
RANDOM_INITIAL_STATE=True  # Whether to randomize initial state
ABILITY_RISK_ENABLED=False
AGENTS_TO_TRAIN=("1" "2")

# Define arrays for angles and seeds
ANGLE_PAIRS=("0 0")

# Define specific speed pairs
# Format: "cut_speed1,walk_speed1 cut_speed2,walk_speed2"
SPEED_PAIRS=(
    "1.0,1.0 1.0,1.0"
    "1.0,0.4 0.2,1.0"
)

# Define checkpoint paths for each speed pair
# Key format: "cut1,walk1 cut2,walk2" -> Value: checkpoint path or "None"
declare -A CHECKPOINT_MAP
CHECKPOINT_MAP["1.0,1.0 1.0,1.0"]="none"
CHECKPOINT_MAP["1.0,0.4 0.2,1.0"]="none"

# Define ETA values (opportunity cost sensitivity)
ETA_VALUES=(0.0)

# Define specialization penalty options
SPECIALIZATION_OPTIONS=(False)

# Maximum number of parallel jobs
MAX_PARALLEL=10 # Set this to control how many jobs run in parallel

# Count number of jobs = seeds × angle pairs × speed pairs × eta values × specialization options × agents_to_train (if NUM_AGENTS=1)
NUM_SEEDS=${#SEEDS[@]}
NUM_ANGLES=${#ANGLE_PAIRS[@]}
NUM_SPEEDS=${#SPEED_PAIRS[@]}
NUM_ETAS=${#ETA_VALUES[@]}
NUM_SPEC_OPTIONS=${#SPECIALIZATION_OPTIONS[@]}
NUM_AGENTS_TO_TRAIN=${#AGENTS_TO_TRAIN[@]}
if [ $NUM_AGENTS -eq 1 ]; then
    NUM_JOBS=$(( NUM_SEEDS * NUM_ANGLES * NUM_SPEEDS * NUM_ETAS * NUM_SPEC_OPTIONS * NUM_AGENTS_TO_TRAIN ))
else
    NUM_JOBS=$(( NUM_SEEDS * NUM_ANGLES * NUM_SPEEDS * NUM_ETAS * NUM_SPEC_OPTIONS ))
fi

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
echo "NUM_SPEEDS = $NUM_SPEEDS"
echo "NUM_ETAS = $NUM_ETAS"
echo "NUM_SPEC_OPTIONS = $NUM_SPEC_OPTIONS"
if [ $NUM_AGENTS -eq 1 ]; then
    echo "NUM_AGENTS_TO_TRAIN = $NUM_AGENTS_TO_TRAIN"
fi
echo "NUM_JOBS = $NUM_JOBS"
echo "MAX_PARALLEL = $MAX_PARALLEL"
echo "GPU per job = 1 (full GPU)"
echo "CPUs per job = $RLLIB_CPUS_PER_WORKER"
echo "================================="

# Function to wait for jobs to complete
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 10
    done
}

# Create all job combinations first
declare -a job_list=()
if [ $NUM_AGENTS -eq 1 ]; then
    # Include AGENTS_TO_TRAIN as outer loop when NUM_AGENTS is 1
    for AGENTS in "${AGENTS_TO_TRAIN[@]}"; do
        for ANGLES in "${ANGLE_PAIRS[@]}"; do
            read ANGLE1 ANGLE2 <<< "$ANGLES"
            for SPEEDS in "${SPEED_PAIRS[@]}"; do
                read AGENT1_SPEEDS AGENT2_SPEEDS <<< "$SPEEDS"
                IFS=',' read CUTTING_SPEED_1 WALKING_SPEED_1 <<< "$AGENT1_SPEEDS"
                IFS=',' read CUTTING_SPEED_2 WALKING_SPEED_2 <<< "$AGENT2_SPEEDS"
                for ETA in "${ETA_VALUES[@]}"; do
                    for SPEC_PENALTY in "${SPECIALIZATION_OPTIONS[@]}"; do
                        for SEED in "${SEEDS[@]}"; do
                            job_list+=("$ANGLE1 $ANGLE2 $CUTTING_SPEED_1 $WALKING_SPEED_1 $CUTTING_SPEED_2 $WALKING_SPEED_2 $ETA $SPEC_PENALTY $SEED $AGENTS")
                        done
                    done
                done
            done
        done
    done
else
    # Original loop without AGENTS_TO_TRAIN
    for ANGLES in "${ANGLE_PAIRS[@]}"; do
        read ANGLE1 ANGLE2 <<< "$ANGLES"
        for SPEEDS in "${SPEED_PAIRS[@]}"; do
            read AGENT1_SPEEDS AGENT2_SPEEDS <<< "$SPEEDS"
            IFS=',' read CUTTING_SPEED_1 WALKING_SPEED_1 <<< "$AGENT1_SPEEDS"
            IFS=',' read CUTTING_SPEED_2 WALKING_SPEED_2 <<< "$AGENT2_SPEEDS"
            for ETA in "${ETA_VALUES[@]}"; do
                for SPEC_PENALTY in "${SPECIALIZATION_OPTIONS[@]}"; do
                    for SEED in "${SEEDS[@]}"; do
                        job_list+=("$ANGLE1 $ANGLE2 $CUTTING_SPEED_1 $WALKING_SPEED_1 $CUTTING_SPEED_2 $WALKING_SPEED_2 $ETA $SPEC_PENALTY $SEED")
                    done
                done
            done
        done
    done
fi

echo "Total jobs to run: ${#job_list[@]}"

# Execute jobs with parallel limit
job_count=0
for job in "${job_list[@]}"; do
    if [ $NUM_AGENTS -eq 1 ]; then
        read ANGLE1 ANGLE2 CUTTING_SPEED_1 WALKING_SPEED_1 CUTTING_SPEED_2 WALKING_SPEED_2 ETA SPEC_PENALTY SEED AGENTS <<< "$job"
    else
        read ANGLE1 ANGLE2 CUTTING_SPEED_1 WALKING_SPEED_1 CUTTING_SPEED_2 WALKING_SPEED_2 ETA SPEC_PENALTY SEED <<< "$job"
        AGENTS=""  # Empty when not used
    fi
    
    # Wait for a slot to be available
    wait_for_slot
    
    job_count=$((job_count + 1))
    
    RAD1=$(python -c "import numpy as np; print(np.radians($ANGLE1))")
    RAD2=$(python -c "import numpy as np; print(np.radians($ANGLE2))")
    ALPHA1=$(python -c "import numpy as np; print(np.cos($RAD1))")
    BETA1=$(python -c "import numpy as np; print(np.sin($RAD1))")
    ALPHA2=$(python -c "import numpy as np; print(np.cos($RAD2))")
    BETA2=$(python -c "import numpy as np; print(np.sin($RAD2))")

    # Get checkpoint path for this speed pair
    SPEED_KEY="${CUTTING_SPEED_1},${WALKING_SPEED_1} ${CUTTING_SPEED_2},${WALKING_SPEED_2}"
    CHECKPOINT_PATHS="${CHECKPOINT_MAP[$SPEED_KEY]}"

    if [ $NUM_AGENTS -eq 1 ]; then
        INPUT_FILE="inputs/input_${ANGLE1}_${ANGLE2}_c${CUTTING_SPEED_1}${CUTTING_SPEED_2}_w${WALKING_SPEED_1}${WALKING_SPEED_2}_eta${ETA}_spec${SPEC_PENALTY}_agent${AGENTS}.txt"
        LOG_FILE="logs/out_${ANGLE1}_${ANGLE2}_c${CUTTING_SPEED_1}${CUTTING_SPEED_2}_w${WALKING_SPEED_1}${WALKING_SPEED_2}_lr${LR}_${GAME_VERSION}_${MAP_NR}_seed_${SEED}_${RANDOM_INITIAL_STATE}_eta${ETA}_spec${SPEC_PENALTY}_agent${AGENTS}.log"
    else
        INPUT_FILE="inputs/input_${ANGLE1}_${ANGLE2}_c${CUTTING_SPEED_1}${CUTTING_SPEED_2}_w${WALKING_SPEED_1}${WALKING_SPEED_2}_eta${ETA}_spec${SPEC_PENALTY}.txt"
        LOG_FILE="logs/out_${ANGLE1}_${ANGLE2}_c${CUTTING_SPEED_1}${CUTTING_SPEED_2}_w${WALKING_SPEED_1}${WALKING_SPEED_2}_lr${LR}_${GAME_VERSION}_${MAP_NR}_seed_${SEED}_${RANDOM_INITIAL_STATE}_eta${ETA}_spec${SPEC_PENALTY}.log"
    fi

    echo -e "${ALPHA1} ${BETA1}\n${WALKING_SPEED_1} ${CUTTING_SPEED_1}\n${ALPHA2} ${BETA2}\n${WALKING_SPEED_2} ${CUTTING_SPEED_2}" > "$INPUT_FILE"

    if [ $NUM_AGENTS -eq 1 ]; then
        echo "Starting job $job_count/${#job_list[@]}: MAP_NR=$MAP_NR ANGLE1=$ANGLE1 ANGLE2=$ANGLE2 C_SPEEDS=[$CUTTING_SPEED_1,$CUTTING_SPEED_2] W_SPEEDS=[$WALKING_SPEED_1,$WALKING_SPEED_2] LR=$LR SEED=$SEED ETA=$ETA SPEC_PENALTY=$SPEC_PENALTY CHECKPOINT=$CHECKPOINT_PATHS AGENTS_TO_TRAIN=$AGENTS"
        python "$TRAIN_SCRIPT" $CLUSTER $INPUT_FILE $MAP_NR $LR $GAME_VERSION $NUM_AGENTS $NUM_EPOCHS $SEED $CHECKPOINT_PATHS $REWARDS_ON_DELIVERY_ONLY $RANDOM_INITIAL_STATE $ABILITY_RISK_ENABLED $ETA $SPEC_PENALTY $AGENTS > "$LOG_FILE" 2>&1 &
    else
        echo "Starting job $job_count/${#job_list[@]}: MAP_NR=$MAP_NR ANGLE1=$ANGLE1 ANGLE2=$ANGLE2 C_SPEEDS=[$CUTTING_SPEED_1,$CUTTING_SPEED_2] W_SPEEDS=[$WALKING_SPEED_1,$WALKING_SPEED_2] LR=$LR SEED=$SEED ETA=$ETA SPEC_PENALTY=$SPEC_PENALTY CHECKPOINT=$CHECKPOINT_PATHS"
        python "$TRAIN_SCRIPT" $CLUSTER $INPUT_FILE $MAP_NR $LR $GAME_VERSION $NUM_AGENTS $NUM_EPOCHS $SEED $CHECKPOINT_PATHS $REWARDS_ON_DELIVERY_ONLY $RANDOM_INITIAL_STATE $ABILITY_RISK_ENABLED $ETA $SPEC_PENALTY > "$LOG_FILE" 2>&1 &
    fi
    
    sleep 45  # Delay between job starts
done

# Wait for all remaining jobs to complete
echo "Waiting for all jobs to complete..."
wait

