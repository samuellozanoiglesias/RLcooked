#!/bin/bash
#SBATCH --job-name=pretrained
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

TRAIN_SCRIPT='/mnt/lustre/home/samuloza/RLcooked/training-DTDE-spoiled_broth.py'

CLUSTER=brigit
MAP_NRS=(baseline_division_of_labor_large
        encouraged_division_of_labor_large)  # List of maps to run
LR=3e-4
GAME_VERSIONS=(classic_collision)  # List of game versions to run
NUM_AGENTS=2
NUM_EPOCHS=2000  # Number of training epochs
SEEDS=(0)  # Different random seeds for runs
REWARDS_ON_DELIVERY_ONLY=True  # Whether to use rewards on delivery only
RANDOM_INITIAL_STATE=False  # Whether to randomize initial state
AGENTS_TO_TRAIN=("")  # Specify which agent to train when NUM_AGENTS=1 (e.g., "1" for agent 1, "2" for agent 2)
COUNTER_REWARD=("0.0")  # Penalty for collisions (set to 0 to disable)
COLLISION_PENALTY=("0.0")  # Penalty for collisions (set to 0 to disable)
ALLOW_BLOCKED=False  # Whether to allow agents to attempt blocked actions

# Custom checkpoint directory (optional)
# If set, this will be used for all simulations instead of auto-generated paths
# Leave empty to use automatic checkpoint name generation
CUSTOM_CHECKPOINT_DIR="checkpoints/checkpoints_pretrained.txt"  # e.g., "checkpoints/my_custom_checkpoint.txt"
ACTIVATE_CHECKPOINTS=True  # Set to False to disable checkpoint loading

# Entropy coefficient for training
ENTROPY_COEF=0.001  # Default value, can be changed

# Define arrays for agent angle vectors and speeds
ANGLE_PAIRS=("0 0")  # Format: "angle_agent_1 angle_agent_2 ... angle_agent_N"

# Define agent speed vectors
# Format: "cut_speed_1,walk_speed_1 cut_speed_2,walk_speed_2 ..."
SPEED_PAIRS=(
    "1.0,1.0 1.0,1.0"
    "1.0,0.2 1.0,1.0"
    "1.0,0.7 0.4,1.0"
    "1.0,0.4 0.2,1.0"
    "1.0,0.1 0.1,1.0"
)

#SPEED_PAIRS=(
#    "1.0,1.0 1.0,1.0"
#    #"1.0,0.9 0.9,1.0"
#    #"1.0,0.8 0.8,1.0"
#    #"1.0,0.7 0.7,1.0"
#    #"1.0,0.6 0.6,1.0"
#    #"1.0,0.5 0.5,1.0"
#    #"1.0,0.4 0.4,1.0"
#    #"1.0,0.3 0.3,1.0"
#    #"1.0,0.2 0.2,1.0"
#    #"1.0,0.1 0.1,1.0"
#)

# Define SYNERGY_SCALING_FACTOR values (team synergy sensitivity)
SYNERGY_SCALING_FACTOR_VALUES=(0)

# Define specialization penalty scale values
SPECIALIZATION_PENALTY_SCALE_VALUES=(0)

# Define theta values for exponential specialization penalty
# penalty = base_reward * (exp(theta * (1.0 - speed)) - 1) * specialization_scale * event_count
SPECIALIZATION_THETA_VALUES=(1.0)

# Define kappa values (competence transformation parameter for team synergy distribution)
KAPPA_VALUES=(20.0)

# Define activate_synergy_positive values (whether to activate positive synergy signals)
ACTIVATE_SYNERGY_POSITIVE_VALUES=(False)

# Define enable_reward_decay values (exponentially decay intermediate rewards after episode 200)
# True = enabled, False = disabled
ENABLE_REWARD_DECAY_VALUES=(False)

# Define collision harshness values (multiplier when collisions enabled)
# Controls how much stronger specialization penalty becomes in collision scenarios
# 1.0 = same strength, 2.0 = double strength (default), >2.0 = even stronger
COLLISION_HARSHNESS_VALUES=(10.0)

# Maximum number of parallel jobs
MAX_PARALLEL=12 # Set this to control how many jobs run in parallel

# Function to get checkpoint name based on map and game version
get_checkpoint_name() {
    local map_name=$1
    local game_version=$2
    local speeds=$3
    
    # If custom checkpoint directory is set, use it for all simulations
    if [ -n "$CUSTOM_CHECKPOINT_DIR" ]; then
        echo "$CUSTOM_CHECKPOINT_DIR"
        return
    fi
    
    # Otherwise, auto-generate checkpoint path based on parameters
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

sanitize_for_path() {
    local value=$1
    value=${value// /_}
    value=${value//,/x}
    echo "$value"
}

# Count number of jobs = seeds × angle pairs × speed pairs × eta values × specialization options × theta values × kappa values × collision penalties × game versions × maps × agents_to_train (if NUM_AGENTS=1) × activate_synergy_positive × enable_reward_decay × collision_harshness
NUM_SEEDS=${#SEEDS[@]}
NUM_ANGLES=${#ANGLE_PAIRS[@]}
NUM_SPEEDS=${#SPEED_PAIRS[@]}
NUM_SYNERGY_VALUES=${#SYNERGY_SCALING_FACTOR_VALUES[@]}
NUM_SPEC_SCALE_VALUES=${#SPECIALIZATION_PENALTY_SCALE_VALUES[@]}
NUM_THETA_VALUES=${#SPECIALIZATION_THETA_VALUES[@]}
NUM_KAPPA_VALUES=${#KAPPA_VALUES[@]}
NUM_COUNTER_REWARDS=${#COUNTER_REWARD[@]}
NUM_COLLISION_PENALTY_VALUES=${#COLLISION_PENALTY[@]}
NUM_GAME_VERSIONS=${#GAME_VERSIONS[@]}
NUM_MAPS=${#MAP_NRS[@]}
NUM_AGENTS_TO_TRAIN=${#AGENTS_TO_TRAIN[@]}
NUM_ACTIVATE_SYNERGY_VALUES=${#ACTIVATE_SYNERGY_POSITIVE_VALUES[@]}
NUM_ENABLE_REWARD_DECAY_VALUES=${#ENABLE_REWARD_DECAY_VALUES[@]}
NUM_COLLISION_HARSHNESS_VALUES=${#COLLISION_HARSHNESS_VALUES[@]}
if [ $NUM_AGENTS -eq 1 ]; then
    NUM_JOBS=$(( NUM_SEEDS * NUM_ANGLES * NUM_SPEEDS * NUM_SYNERGY_VALUES * NUM_SPEC_SCALE_VALUES * NUM_THETA_VALUES * NUM_KAPPA_VALUES * NUM_COUNTER_REWARDS * NUM_COLLISION_PENALTY_VALUES * NUM_GAME_VERSIONS * NUM_MAPS * NUM_AGENTS_TO_TRAIN * NUM_ACTIVATE_SYNERGY_VALUES * NUM_ENABLE_REWARD_DECAY_VALUES * NUM_COLLISION_HARSHNESS_VALUES ))
else
    NUM_JOBS=$(( NUM_SEEDS * NUM_ANGLES * NUM_SPEEDS * NUM_SYNERGY_VALUES * NUM_SPEC_SCALE_VALUES * NUM_THETA_VALUES * NUM_KAPPA_VALUES * NUM_COUNTER_REWARDS * NUM_COLLISION_PENALTY_VALUES * NUM_GAME_VERSIONS * NUM_MAPS * NUM_ACTIVATE_SYNERGY_VALUES * NUM_ENABLE_REWARD_DECAY_VALUES * NUM_COLLISION_HARSHNESS_VALUES ))
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
echo "NUM_THETA_VALUES = $NUM_THETA_VALUES"
echo "NUM_KAPPA_VALUES = $NUM_KAPPA_VALUES"
echo "NUM_COUNTER_REWARDS = $NUM_COUNTER_REWARDS"
echo "NUM_COLLISION_PENALTY_VALUES = $NUM_COLLISION_PENALTY_VALUES"
echo "NUM_ACTIVATE_SYNERGY_VALUES = $NUM_ACTIVATE_SYNERGY_VALUES"
echo "NUM_ENABLE_REWARD_DECAY_VALUES = $NUM_ENABLE_REWARD_DECAY_VALUES"
echo "NUM_COLLISION_HARSHNESS_VALUES = $NUM_COLLISION_HARSHNESS_VALUES"
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

# Wait until the launched job has written something to its log file.
# If the process exits before writing, continue after emitting a warning.
wait_for_log_output() {
    local log_file=$1
    local job_pid=$2

    while [ ! -s "$log_file" ]; do
        sleep 1
    done

    return 0
}

# Create all job combinations first
declare -a job_list=()
if [ $NUM_AGENTS -eq 1 ]; then
    # Include AGENTS_TO_TRAIN as outer loop when NUM_AGENTS is 1
    for AGENTS in "${AGENTS_TO_TRAIN[@]}"; do
        for ANGLES in "${ANGLE_PAIRS[@]}"; do
            read ANGLE1 ANGLE2 <<< "$ANGLES"
            for ENABLE_REWARD_DECAY in "${ENABLE_REWARD_DECAY_VALUES[@]}"; do
                for SEED in "${SEEDS[@]}"; do
                    for SPEEDS in "${SPEED_PAIRS[@]}"; do
                        read AGENT1_SPEEDS AGENT2_SPEEDS <<< "$SPEEDS"
                        IFS=',' read CUTTING_SPEED_1 WALKING_SPEED_1 <<< "$AGENT1_SPEEDS"
                        IFS=',' read CUTTING_SPEED_2 WALKING_SPEED_2 <<< "$AGENT2_SPEEDS"
                        for COLLISION_PENALTY_VALUE in "${COLLISION_PENALTY[@]}"; do
                            for COUNTER_REWARD_VALUE in "${COUNTER_REWARD[@]}"; do
                                for KAPPA in "${KAPPA_VALUES[@]}"; do
                                    for ACTIVATE_SYNERGY_POSITIVE in "${ACTIVATE_SYNERGY_POSITIVE_VALUES[@]}"; do
                                        for COLLISION_HARSHNESS in "${COLLISION_HARSHNESS_VALUES[@]}"; do
                                            for SYNERGY_SCALING_FACTOR in "${SYNERGY_SCALING_FACTOR_VALUES[@]}"; do
                                                for SPEC_PENALTY_SCALE in "${SPECIALIZATION_PENALTY_SCALE_VALUES[@]}"; do
                                                    for SPECIALIZATION_THETA in "${SPECIALIZATION_THETA_VALUES[@]}"; do
                                                        for GAME_VERSION in "${GAME_VERSIONS[@]}"; do
                                                            for MAP_NR in "${MAP_NRS[@]}"; do
                                                                # Use placeholder for empty AGENTS to prevent field misalignment
                                                                AGENTS_PLACEHOLDER="${AGENTS:-none}"
                                                                job_list+=("$ANGLE1 $ANGLE2 $CUTTING_SPEED_1 $WALKING_SPEED_1 $CUTTING_SPEED_2 $WALKING_SPEED_2 $SYNERGY_SCALING_FACTOR $SPEC_PENALTY_SCALE $SPECIALIZATION_THETA $KAPPA $COUNTER_REWARD_VALUE $SEED $AGENTS_PLACEHOLDER $GAME_VERSION $MAP_NR $COLLISION_PENALTY_VALUE $ACTIVATE_SYNERGY_POSITIVE $ENABLE_REWARD_DECAY $COLLISION_HARSHNESS")
                                                            done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
else
    # Multi-agent loop without AGENTS_TO_TRAIN
    for ANGLES in "${ANGLE_PAIRS[@]}"; do
        for ENABLE_REWARD_DECAY in "${ENABLE_REWARD_DECAY_VALUES[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                for SPEEDS in "${SPEED_PAIRS[@]}"; do
                    for COLLISION_PENALTY_VALUE in "${COLLISION_PENALTY[@]}"; do
                        for COUNTER_REWARD_VALUE in "${COUNTER_REWARD[@]}"; do
                            for KAPPA in "${KAPPA_VALUES[@]}"; do
                                for ACTIVATE_SYNERGY_POSITIVE in "${ACTIVATE_SYNERGY_POSITIVE_VALUES[@]}"; do
                                    for COLLISION_HARSHNESS in "${COLLISION_HARSHNESS_VALUES[@]}"; do
                                        for SYNERGY_SCALING_FACTOR in "${SYNERGY_SCALING_FACTOR_VALUES[@]}"; do
                                            for SPEC_PENALTY_SCALE in "${SPECIALIZATION_PENALTY_SCALE_VALUES[@]}"; do
                                                for SPECIALIZATION_THETA in "${SPECIALIZATION_THETA_VALUES[@]}"; do
                                                    for GAME_VERSION in "${GAME_VERSIONS[@]}"; do
                                                        for MAP_NR in "${MAP_NRS[@]}"; do
                                                            job_list+=("$ANGLES|$SPEEDS|$SYNERGY_SCALING_FACTOR|$SPEC_PENALTY_SCALE|$SPECIALIZATION_THETA|$KAPPA|$COUNTER_REWARD_VALUE|$SEED|$GAME_VERSION|$MAP_NR|$COLLISION_PENALTY_VALUE|$ACTIVATE_SYNERGY_POSITIVE|$ENABLE_REWARD_DECAY|$COLLISION_HARSHNESS")
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
fi

echo "Total jobs to run: ${#job_list[@]}"

# Generate input files once (overwrite) based on angles/speeds.
declare -A input_generated=()
if [ $NUM_AGENTS -eq 1 ]; then
    for job in "${job_list[@]}"; do
        read ANGLE1 ANGLE2 CUTTING_SPEED_1 WALKING_SPEED_1 CUTTING_SPEED_2 WALKING_SPEED_2 SYNERGY_SCALING_FACTOR SPEC_PENALTY_SCALE SPECIALIZATION_THETA KAPPA COUNTER_REWARD_VALUE SEED AGENTS GAME_VERSION MAP_NR COLLISION_PENALTY_VALUE ACTIVATE_SYNERGY_POSITIVE ENABLE_REWARD_DECAY COLLISION_HARSHNESS <<< "$job"
        if [ "$AGENTS" = "none" ]; then
            AGENTS=""
        fi

        INPUT_FILE="inputs/input_${ANGLE1}_${ANGLE2}_c${CUTTING_SPEED_1}${CUTTING_SPEED_2}_w${WALKING_SPEED_1}${WALKING_SPEED_2}_agent${AGENTS}.txt"
        if [ -z "${input_generated[$INPUT_FILE]+x}" ]; then
            input_generated[$INPUT_FILE]=1

            RAD1=$(python3 -c "import math; print(math.radians($ANGLE1))")
            RAD2=$(python3 -c "import math; print(math.radians($ANGLE2))")
            ALPHA1=$(python3 -c "import math; print(math.cos($RAD1))")
            BETA1=$(python3 -c "import math; print(math.sin($RAD1))")
            ALPHA2=$(python3 -c "import math; print(math.cos($RAD2))")
            BETA2=$(python3 -c "import math; print(math.sin($RAD2))")

            tmp_file="${INPUT_FILE}.tmp.$$"
            printf '%s %s\n%s %s\n%s %s\n%s %s\n' "$ALPHA1" "$BETA1" "$WALKING_SPEED_1" "$CUTTING_SPEED_1" "$ALPHA2" "$BETA2" "$WALKING_SPEED_2" "$CUTTING_SPEED_2" > "$tmp_file"
            mv "$tmp_file" "$INPUT_FILE"
        fi
    done
else
    for job in "${job_list[@]}"; do
        IFS='|' read -r ANGLES SPEEDS SYNERGY_SCALING_FACTOR SPEC_PENALTY_SCALE SPECIALIZATION_THETA KAPPA COUNTER_REWARD_VALUE SEED GAME_VERSION MAP_NR COLLISION_PENALTY_VALUE ACTIVATE_SYNERGY_POSITIVE ENABLE_REWARD_DECAY COLLISION_HARSHNESS <<< "$job"

        ANGLE_TAG=$(sanitize_for_path "$ANGLES")
        SPEED_TAG=$(sanitize_for_path "$SPEEDS")
        INPUT_FILE="inputs/input_${ANGLE_TAG}_c${SPEED_TAG}.txt"
        if [ -z "${input_generated[$INPUT_FILE]+x}" ]; then
            input_generated[$INPUT_FILE]=1

            read -r -a ANGLE_VALUES <<< "$ANGLES"
            read -r -a SPEED_VALUES <<< "$SPEEDS"

            tmp_file="${INPUT_FILE}.tmp.$$"
            : > "$tmp_file"
            for ((idx = 0; idx < ${#ANGLE_VALUES[@]}; idx++)); do
                ANGLE_VALUE=${ANGLE_VALUES[$idx]}
                SPEED_VALUE=${SPEED_VALUES[$idx]}
                IFS=',' read -r CUTTING_SPEED WALKING_SPEED <<< "$SPEED_VALUE"

                RADIAN_VALUE=$(python3 -c "import math; print(math.radians($ANGLE_VALUE))")
                ALPHA_VALUE=$(python3 -c "import math; print(math.cos($RADIAN_VALUE))")
                BETA_VALUE=$(python3 -c "import math; print(math.sin($RADIAN_VALUE))")

                printf '%s %s\n%s %s\n' "$ALPHA_VALUE" "$BETA_VALUE" "$WALKING_SPEED" "$CUTTING_SPEED" >> "$tmp_file"
            done
            mv "$tmp_file" "$INPUT_FILE"
        fi
    done
fi

# Execute jobs with parallel limit
job_count=0
for job in "${job_list[@]}"; do
    if [ $NUM_AGENTS -eq 1 ]; then
        read ANGLE1 ANGLE2 CUTTING_SPEED_1 WALKING_SPEED_1 CUTTING_SPEED_2 WALKING_SPEED_2 SYNERGY_SCALING_FACTOR SPEC_PENALTY_SCALE SPECIALIZATION_THETA KAPPA COUNTER_REWARD_VALUE SEED AGENTS GAME_VERSION MAP_NR COLLISION_PENALTY_VALUE ACTIVATE_SYNERGY_POSITIVE ENABLE_REWARD_DECAY COLLISION_HARSHNESS <<< "$job"
        # Convert placeholder back to empty string
        if [ "$AGENTS" = "none" ]; then
            AGENTS=""
        fi
    else
        IFS='|' read -r ANGLES SPEEDS SYNERGY_SCALING_FACTOR SPEC_PENALTY_SCALE SPECIALIZATION_THETA KAPPA COUNTER_REWARD_VALUE SEED GAME_VERSION MAP_NR COLLISION_PENALTY_VALUE ACTIVATE_SYNERGY_POSITIVE ENABLE_REWARD_DECAY COLLISION_HARSHNESS <<< "$job"
        AGENTS=""  # Empty when not used

        read -r -a ANGLE_VALUES <<< "$ANGLES"
        read -r -a SPEED_VALUES <<< "$SPEEDS"

        if [ ${#ANGLE_VALUES[@]} -ne $NUM_AGENTS ]; then
            echo "Error: ANGLE_PAIRS entry '$ANGLES' has ${#ANGLE_VALUES[@]} values but NUM_AGENTS=$NUM_AGENTS"
            exit 1
        fi

        if [ ${#SPEED_VALUES[@]} -ne $NUM_AGENTS ]; then
            echo "Error: SPEED_PAIRS entry '$SPEEDS' has ${#SPEED_VALUES[@]} values but NUM_AGENTS=$NUM_AGENTS"
            exit 1
        fi
    fi
    
    # Wait for a slot to be available
    wait_for_slot
    
    job_count=$((job_count + 1))
    
    if [ $NUM_AGENTS -eq 1 ]; then
        SPEED_KEY="${CUTTING_SPEED_1},${WALKING_SPEED_1} ${CUTTING_SPEED_2},${WALKING_SPEED_2}"
        INPUT_FILE="inputs/input_${ANGLE1}_${ANGLE2}_c${CUTTING_SPEED_1}${CUTTING_SPEED_2}_w${WALKING_SPEED_1}${WALKING_SPEED_2}_agent${AGENTS}.txt"
        LOG_FILE="logs/Out_${ANGLE1}_${ANGLE2}_c${CUTTING_SPEED_1}${CUTTING_SPEED_2}_w${WALKING_SPEED_1}${WALKING_SPEED_2}_lr${LR}_${GAME_VERSION}_${MAP_NR}_seed_${SEED}_${RANDOM_INITIAL_STATE}_synergy${SYNERGY_SCALING_FACTOR}_spec${SPEC_PENALTY_SCALE}_theta${SPECIALIZATION_THETA}_kappa${KAPPA}_counter${COUNTER_REWARD_VALUE}_collision${COLLISION_PENALTY_VALUE}_activatesynergy${ACTIVATE_SYNERGY_POSITIVE}_rewarddecay${ENABLE_REWARD_DECAY}_harsh${COLLISION_HARSHNESS}_agent${AGENTS}.log"
    else
        SPEED_KEY="$SPEEDS"
        ANGLE_TAG=$(sanitize_for_path "$ANGLES")
        SPEED_TAG=$(sanitize_for_path "$SPEEDS")
        INPUT_FILE="inputs/input_${ANGLE_TAG}_c${SPEED_TAG}.txt"
        LOG_FILE="logs/Out_${ANGLE_TAG}_c${SPEED_TAG}_lr${LR}_${GAME_VERSION}_${MAP_NR}_seed_${SEED}_synergy${SYNERGY_SCALING_FACTOR}_spec${SPEC_PENALTY_SCALE}_theta${SPECIALIZATION_THETA}_kappa${KAPPA}_agent${AGENTS}.log"
    fi

    # Get checkpoint path based on map and game version
    if [ "$ACTIVATE_CHECKPOINTS" = "True" ]; then
        CHECKPOINT_PATHS=$(get_checkpoint_name "$MAP_NR" "$GAME_VERSION" "$SPEED_KEY")
    else
        CHECKPOINT_PATHS=none
    fi

    if [ $NUM_AGENTS -eq 1 ]; then
        echo "Starting job $job_count/${#job_list[@]}: GAME_VERSION=$GAME_VERSION MAP_NR=$MAP_NR ANGLE1=$ANGLE1 ANGLE2=$ANGLE2 C_SPEEDS=[$CUTTING_SPEED_1,$CUTTING_SPEED_2] W_SPEEDS=[$WALKING_SPEED_1,$WALKING_SPEED_2] LR=$LR SEED=$SEED SYNERGY_SCALING_FACTOR=$SYNERGY_SCALING_FACTOR SPEC_PENALTY_SCALE=$SPEC_PENALTY_SCALE SPECIALIZATION_THETA=$SPECIALIZATION_THETA KAPPA=$KAPPA CHECKPOINT=$CHECKPOINT_PATHS COUNTER_REWARD=$COUNTER_REWARD_VALUE COLLISION_PENALTY=$COLLISION_PENALTY_VALUE ALLOW_BLOCKED=$ALLOW_BLOCKED ACTIVATE_SYNERGY_POSITIVE=$ACTIVATE_SYNERGY_POSITIVE ENABLE_REWARD_DECAY=$ENABLE_REWARD_DECAY COLLISION_HARSHNESS=$COLLISION_HARSHNESS AGENTS_TO_TRAIN=$AGENTS"
            python3 "$TRAIN_SCRIPT" "$CLUSTER" "$INPUT_FILE" "$MAP_NR" "$LR" "$GAME_VERSION" "$NUM_AGENTS" "$NUM_EPOCHS" "$SEED" "$CHECKPOINT_PATHS" "$REWARDS_ON_DELIVERY_ONLY" "$RANDOM_INITIAL_STATE" "$SYNERGY_SCALING_FACTOR" "$SPEC_PENALTY_SCALE" "$COUNTER_REWARD_VALUE" "$COLLISION_PENALTY_VALUE" "$ALLOW_BLOCKED" "$KAPPA" "$ACTIVATE_SYNERGY_POSITIVE" "$ENABLE_REWARD_DECAY" "$COLLISION_HARSHNESS" "$ENTROPY_COEF" "$AGENTS" "$SPECIALIZATION_THETA" > "$LOG_FILE" 2>&1 &
    else
        echo "Starting job $job_count/${#job_list[@]}: GAME_VERSION=$GAME_VERSION MAP_NR=$MAP_NR ANGLES=[$ANGLES] SPEEDS=[$SPEEDS] LR=$LR SEED=$SEED SYNERGY_SCALING_FACTOR=$SYNERGY_SCALING_FACTOR SPEC_PENALTY_SCALE=$SPEC_PENALTY_SCALE SPECIALIZATION_THETA=$SPECIALIZATION_THETA KAPPA=$KAPPA CHECKPOINT=$CHECKPOINT_PATHS COUNTER_REWARD=$COUNTER_REWARD_VALUE COLLISION_PENALTY=$COLLISION_PENALTY_VALUE ALLOW_BLOCKED=$ALLOW_BLOCKED ACTIVATE_SYNERGY_POSITIVE=$ACTIVATE_SYNERGY_POSITIVE ENABLE_REWARD_DECAY=$ENABLE_REWARD_DECAY COLLISION_HARSHNESS=$COLLISION_HARSHNESS"
            python3 "$TRAIN_SCRIPT" "$CLUSTER" "$INPUT_FILE" "$MAP_NR" "$LR" "$GAME_VERSION" "$NUM_AGENTS" "$NUM_EPOCHS" "$SEED" "$CHECKPOINT_PATHS" "$REWARDS_ON_DELIVERY_ONLY" "$RANDOM_INITIAL_STATE" "$SYNERGY_SCALING_FACTOR" "$SPEC_PENALTY_SCALE" "$COUNTER_REWARD_VALUE" "$COLLISION_PENALTY_VALUE" "$ALLOW_BLOCKED" "$KAPPA" "$ACTIVATE_SYNERGY_POSITIVE" "$ENABLE_REWARD_DECAY" "$COLLISION_HARSHNESS" "$ENTROPY_COEF" "$SPECIALIZATION_THETA" > "$LOG_FILE" 2>&1 &
    fi
    
    wait_for_log_output "$LOG_FILE" "$job_pid"
    sleep 10  # Delay between job starts after the log becomes active
done

# Wait for all remaining jobs to complete
echo "Waiting for all jobs to complete..."
wait