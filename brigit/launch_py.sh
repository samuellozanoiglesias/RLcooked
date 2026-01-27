#!/bin/bash
#SBATCH --job-name=analysis_classic
#SBATCH -t 30-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=qos_gpu_long
#SBATCH --account=teruel
#SBATCH --ntasks=24
#SBATCH --gres=gpu:1
#SBATCH --output=results-GPU-%x-%j.out
#SBATCH --error=error-GPU-%x-%j.err

# Activar tu entorno
source ~/.bashrc
conda activate cooked 

# Ejecutar el script
python ../figure_gridsearch_analysis.py --cluster brigit --episode_range final --num_final_episodes 20 