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
conda activate RLcooked 

# Ejecutar el script (secuencial, uno a uno)
nohup python ../analysis_classic.py encouraged_division_of_labor_large --game_type classic_collision --individual_training yes --init_type empty_init --cluster brigit > analysis_classic.log 2>&1