#!/bin/bash
#SBATCH --job-name=launch_py
#SBATCH -t 12:00:00
#SBATCH --partition=short
#SBATCH --account=teruel
#SBATCH --ntasks=1
#SBATCH --output=tar-%x-%j.out
#SBATCH --error=tar-%x-%j.err

# Activar tu entorno
source ~/.bashrc
conda activate cooked 

# Ejecutar el script (secuencial, uno a uno)
for target_ep in $(seq 50 50 5950); do
    python ../grid_full_cooperative_analysis.py --episode_range specific --init_type empty_init --num_episodes 10 --cluster brigit --specialization 0.05 --synergy 0.80 --target_episode $target_ep --study_name HEATMAP > full_grid_empty_init_ep${target_ep}.log 2>&1
done