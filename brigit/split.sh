#!/bin/bash
#SBATCH --job-name=tar_UCL_TRAININGS
#SBATCH -t 01:00:00
#SBATCH --partition=gpu
#SBATCH --qos=qos_gpu_long
#SBATCH --account=teruel
#SBATCH --ntasks=1
#SBATCH --output=tar-%x-%j.out
#SBATCH --error=tar-%x-%j.err

# Create the tar archive
split -b 10G /mnt/lustre/home/samuloza/data/samuel_lozano/cooked/UCL_TRAININGS.tar /mnt/lustre/home/samuloza/data/samuel_lozano/cooked/UCL_TRAININGS.tar.part-
