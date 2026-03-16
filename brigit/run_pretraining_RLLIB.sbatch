#!/bin/bash
#SBATCH --job-name=RLLIB_cooked
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

LRS=(3e-3 3e-2)
MAP_NR=baseline_division_of_labor
COOPERATIVE=1
GAME_VERSION=classic
INTENT_VERSION=v3.1
SEED=0
NUM_AGENTS=1
TRAIN_SCRIPT='/mnt/lustre/home/samuloza/cooked/training-DTDE-spoiled_broth.py'

for LR in "${LRS[@]}"; do
    ANGLE1=0
    RAD1=$(python3 -c "import numpy as np; print(np.radians($ANGLE1))")
    ALPHA1=$(python3 -c "import numpy as np; print(np.cos($RAD1))")
    BETA1=$(python3 -c "import numpy as np; print(np.sin($RAD1))")

    INPUT_FILE="inputs/input_${ANGLE1}.txt"
    LOG_FILE="logs/out_${ANGLE1}_lr${LR}_cooperative${COOPERATIVE}_map_${MAP_NR}.log"

    echo -e "${ALPHA1} ${BETA1}" > "$INPUT_FILE"

    echo "Entrenando con MAP_NR=$MAP_NR ANGLE1=$ANGLE1 LR=$LR"
    python3 "$TRAIN_SCRIPT" "$INPUT_FILE" $MAP_NR $LR $COOPERATIVE $GAME_VERSION $INTENT_VERSION $SEED $NUM_AGENTS > "$LOG_FILE" 2>&1

done
