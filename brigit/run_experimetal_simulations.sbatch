#!/bin/bash
#SBATCH --job-name=exp_simulation      # Job name
#SBATCH --output=exp_simulation_%j.out # Stdout file (%j expands to jobId)
#SBATCH --error=exp_simulation_%j.err  # Stderr file
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --ntasks=1

# Load conda
eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate cooked

# Features
MAP_NR=forced_division_of_labor
NUM_AGENTS=2
INTENT_VERSION=v3.1
COOPERATIVE=1
GAME_VERSION=classic
TRAINING_ID=2025-09-14_19-15-40
VIDEO=True
NUM_EXPERIMENTS=3

TRAIN_SCRIPT='/mnt/lustre/home/samuloza/cooked/experimental_simulation.py'

# List of checkpoints
checkpoints=("0" "1000" "6000" "11000" "final")

# Run each checkpoint in background
for ckpt in "${checkpoints[@]}"; do
    for ((i=1; i<=NUM_EXPERIMENTS; i++)); do
        LOG_FILE="logs_experimental_simulations/out_${MAP_NR}_training_${TRAINING_ID}_cooperative${COOPERATIVE}_checkpoint_${ckpt}_${i}.log"
        python3 "$TRAIN_SCRIPT" \
            $MAP_NR \
            $NUM_AGENTS \
            $INTENT_VERSION \
            $COOPERATIVE \
            $GAME_VERSION \
            $TRAINING_ID \
            "$ckpt" \
            $VIDEO \
            > "$LOG_FILE" 2>&1
    done
done

# Wait for all runs to finish
wait