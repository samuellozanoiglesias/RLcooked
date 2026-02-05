#!/bin/bash
#SBATCH --job-name=encouraged_collision
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
MAP_NRS=(baseline_division_of_labor_large encouraged_division_of_labor_large)  # List of maps to run
LR=3e-4
GAME_VERSIONS=(classic classic_collision)  # List of game versions to run
NUM_AGENTS=2
NUM_EPOCHS=10  # Number of training epochs
SEEDS=(0)  # Different random seeds for runs
REWARDS_ON_DELIVERY_ONLY=False  # Whether to use rewards on delivery only
RANDOM_INITIAL_STATE=True  # Whether to randomize initial state
ABILITY_RISK_ENABLED=False
AGENTS_TO_TRAIN=("")

# Define arrays for angles and seeds
ANGLE_PAIRS=("0 0")

# Define specific speed pairs
# Format: "cut_speed1,walk_speed1 cut_speed2,walk_speed2"
SPEED_PAIRS=(
    "1.0,1.0 1.0,1.0"
    "1.0,0.4 0.2,1.0"
)

# Define SYNERGY_SCALING_FACTOR values (team synergy sensitivity)
SYNERGY_SCALING_FACTOR_VALUES=(2.0)

# Define specialization penalty scale values
SPECIALIZATION_PENALTY_SCALE_VALUES=(1.0)

# Maximum number of parallel jobs
MAX_PARALLEL=10 # Set this to control how many jobs run in parallel

# Function to get checkpoint name based on map and game version
get_checkpoint_name() {
    local map_name=$1
    local game_version=$2
    local speeds=$3
    
    # Extract first word from map name (e.g., "baseline" from "baseline_division_of_labor_large")
    local map_prefix=$(echo "$map_name" | cut -d'_' -f1)
    
    # Determine suffix based on speeds
    if [ "$speeds" = "1.0,1.0 1.0,1.0" ]; then
        local speed_suffix="stars"
    elif [ "$speeds" = "1.0,0.4 0.2,1.0" ]; then
        local speed_suffix="mixed"
    else
        local speed_suffix="custom"
    fi
    
    # Generate checkpoint path based on game version
    if [ "$game_version" = "classic" ]; then
        echo "checkpoints/checkpoints_${map_prefix}_${speed_suffix}.txt"
    elif [ "$game_version" = "classic_collision" ]; then
        echo "checkpoints/checkpoints_${map_prefix}_collision_${speed_suffix}.txt"
    else
        echo "checkpoints/checkpoints_${map_prefix}_${game_version}_${speed_suffix}.txt"
    fi
}

# Count number of jobs = seeds × angle pairs × speed pairs × eta values × specialization options × game versions × maps × agents_to_train (if NUM_AGENTS=1)
NUM_SEEDS=${#SEEDS[@]}
NUM_ANGLES=${#ANGLE_PAIRS[@]}
NUM_SPEEDS=${#SPEED_PAIRS[@]}
NUM_SYNERGY_VALUES=${#SYNERGY_SCALING_FACTOR_VALUES[@]}
NUM_SPEC_SCALE_VALUES=${#SPECIALIZATION_PENALTY_SCALE_VALUES[@]}
NUM_GAME_VERSIONS=${#GAME_VERSIONS[@]}
NUM_MAPS=${#MAP_NRS[@]}
NUM_AGENTS_TO_TRAIN=${#AGENTS_TO_TRAIN[@]}
if [ $NUM_AGENTS -eq 1 ]; then
    NUM_JOBS=$(( NUM_SEEDS * NUM_ANGLES * NUM_SPEEDS * NUM_SYNERGY_VALUES * NUM_SPEC_SCALE_VALUES * NUM_GAME_VERSIONS * NUM_MAPS * NUM_AGENTS_TO_TRAIN ))
else
    NUM_JOBS=$(( NUM_SEEDS * NUM_ANGLES * NUM_SPEEDS * NUM_SYNERGY_VALUES * NUM_SPEC_SCALE_VALUES * NUM_GAME_VERSIONS * NUM_MAPS ))
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
echo "NUM_SYNERGY_VALUES = $NUM_SYNERGY_VALUES"
echo "NUM_SPEC_SCALE_VALUES = $NUM_SPEC_SCALE_VALUES"
echo "NUM_GAME_VERSIONS = $NUM_GAME_VERSIONS"
echo "NUM_MAPS = $NUM_MAPS"
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
                for SYNERGY_SCALING_FACTOR in "${SYNERGY_SCALING_FACTOR_VALUES[@]}"; do
                    for SPEC_PENALTY_SCALE in "${SPECIALIZATION_PENALTY_SCALE_VALUES[@]}"; do
                        for SEED in "${SEEDS[@]}"; do
                            for GAME_VERSION in "${GAME_VERSIONS[@]}"; do
                                for MAP_NR in "${MAP_NRS[@]}"; do
                                    job_list+=("$ANGLE1 $ANGLE2 $CUTTING_SPEED_1 $WALKING_SPEED_1 $CUTTING_SPEED_2 $WALKING_SPEED_2 $SYNERGY_SCALING_FACTOR $SPEC_PENALTY_SCALE $SEED $AGENTS $GAME_VERSION $MAP_NR")
                                done
                            done
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
            for SYNERGY_SCALING_FACTOR in "${SYNERGY_SCALING_FACTOR_VALUES[@]}"; do
                for SPEC_PENALTY_SCALE in "${SPECIALIZATION_PENALTY_SCALE_VALUES[@]}"; do
                    for SEED in "${SEEDS[@]}"; do
                        for GAME_VERSION in "${GAME_VERSIONS[@]}"; do
                            for MAP_NR in "${MAP_NRS[@]}"; do
                                job_list+=("$ANGLE1 $ANGLE2 $CUTTING_SPEED_1 $WALKING_SPEED_1 $CUTTING_SPEED_2 $WALKING_SPEED_2 $SYNERGY_SCALING_FACTOR $SPEC_PENALTY_SCALE $SEED $GAME_VERSION $MAP_NR")
                            done
                        done
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
        read ANGLE1 ANGLE2 CUTTING_SPEED_1 WALKING_SPEED_1 CUTTING_SPEED_2 WALKING_SPEED_2 SYNERGY_SCALING_FACTOR SPEC_PENALTY_SCALE SEED AGENTS GAME_VERSION MAP_NR <<< "$job"
    else
        read ANGLE1 ANGLE2 CUTTING_SPEED_1 WALKING_SPEED_1 CUTTING_SPEED_2 WALKING_SPEED_2 SYNERGY_SCALING_FACTOR SPEC_PENALTY_SCALE SEED GAME_VERSION MAP_NR <<< "$job"
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

    # Get checkpoint path based on map and game version
    SPEED_KEY="${CUTTING_SPEED_1},${WALKING_SPEED_1} ${CUTTING_SPEED_2},${WALKING_SPEED_2}"
    CHECKPOINT_PATHS=$(get_checkpoint_name "$MAP_NR" "$GAME_VERSION" "$SPEED_KEY")

    if [ $NUM_AGENTS -eq 1 ]; then
        INPUT_FILE="inputs/input_${ANGLE1}_${ANGLE2}_c${CUTTING_SPEED_1}${CUTTING_SPEED_2}_w${WALKING_SPEED_1}${WALKING_SPEED_2}_synergy${SYNERGY_SCALING_FACTOR}_spec${SPEC_PENALTY_SCALE}_agent${AGENTS}.txt"
        LOG_FILE="logs/out_${ANGLE1}_${ANGLE2}_c${CUTTING_SPEED_1}${CUTTING_SPEED_2}_w${WALKING_SPEED_1}${WALKING_SPEED_2}_lr${LR}_${GAME_VERSION}_${MAP_NR}_seed_${SEED}_${RANDOM_INITIAL_STATE}_synergy${SYNERGY_SCALING_FACTOR}_spec${SPEC_PENALTY_SCALE}_agent${AGENTS}.log"
    else
        INPUT_FILE="inputs/input_${ANGLE1}_${ANGLE2}_c${CUTTING_SPEED_1}${CUTTING_SPEED_2}_w${WALKING_SPEED_1}${WALKING_SPEED_2}_synergy${SYNERGY_SCALING_FACTOR}_spec${SPEC_PENALTY_SCALE}.txt"
        LOG_FILE="logs/out_${ANGLE1}_${ANGLE2}_c${CUTTING_SPEED_1}${CUTTING_SPEED_2}_w${WALKING_SPEED_1}${WALKING_SPEED_2}_lr${LR}_${GAME_VERSION}_${MAP_NR}_seed_${SEED}_${RANDOM_INITIAL_STATE}_synergy${SYNERGY_SCALING_FACTOR}_spec${SPEC_PENALTY_SCALE}.log"
    fi

    echo -e "${ALPHA1} ${BETA1}\n${WALKING_SPEED_1} ${CUTTING_SPEED_1}\n${ALPHA2} ${BETA2}\n${WALKING_SPEED_2} ${CUTTING_SPEED_2}" > "$INPUT_FILE"

    if [ $NUM_AGENTS -eq 1 ]; then
        echo "Starting job $job_count/${#job_list[@]}: GAME_VERSION=$GAME_VERSION MAP_NR=$MAP_NR ANGLE1=$ANGLE1 ANGLE2=$ANGLE2 C_SPEEDS=[$CUTTING_SPEED_1,$CUTTING_SPEED_2] W_SPEEDS=[$WALKING_SPEED_1,$WALKING_SPEED_2] LR=$LR SEED=$SEED SYNERGY_SCALING_FACTOR=$SYNERGY_SCALING_FACTOR SPEC_PENALTY_SCALE=$SPEC_PENALTY_SCALE CHECKPOINT=$CHECKPOINT_PATHS AGENTS_TO_TRAIN=$AGENTS"
        python "$TRAIN_SCRIPT" $CLUSTER $INPUT_FILE $MAP_NR $LR $GAME_VERSION $NUM_AGENTS $NUM_EPOCHS $SEED $CHECKPOINT_PATHS $REWARDS_ON_DELIVERY_ONLY $RANDOM_INITIAL_STATE $ABILITY_RISK_ENABLED $SYNERGY_SCALING_FACTOR $SPEC_PENALTY_SCALE $AGENTS > "$LOG_FILE" 2>&1 &
    else
        echo "Starting job $job_count/${#job_list[@]}: GAME_VERSION=$GAME_VERSION MAP_NR=$MAP_NR ANGLE1=$ANGLE1 ANGLE2=$ANGLE2 C_SPEEDS=[$CUTTING_SPEED_1,$CUTTING_SPEED_2] W_SPEEDS=[$WALKING_SPEED_1,$WALKING_SPEED_2] LR=$LR SEED=$SEED SYNERGY_SCALING_FACTOR=$SYNERGY_SCALING_FACTOR SPEC_PENALTY_SCALE=$SPEC_PENALTY_SCALE CHECKPOINT=$CHECKPOINT_PATHS"
        python "$TRAIN_SCRIPT" $CLUSTER $INPUT_FILE $MAP_NR $LR $GAME_VERSION $NUM_AGENTS $NUM_EPOCHS $SEED $CHECKPOINT_PATHS $REWARDS_ON_DELIVERY_ONLY $RANDOM_INITIAL_STATE $ABILITY_RISK_ENABLED $SYNERGY_SCALING_FACTOR $SPEC_PENALTY_SCALE > "$LOG_FILE" 2>&1 &
    fi
    
    sleep 45  # Delay between job starts
done

# Wait for all remaining jobs to complete
echo "Waiting for all jobs to complete..."
wait

