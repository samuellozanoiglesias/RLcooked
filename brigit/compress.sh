#!/bin/bash
#SBATCH --job-name=tar_old_v2_collisions
#SBATCH -t 01:00:00
#SBATCH --partition=long
#SBATCH --account=teruel
#SBATCH --ntasks=1
#SBATCH --output=tar-%x-%j.out
#SBATCH --error=tar-%x-%j.err

# Create the tar archive
tar -cvf /mnt/lustre/home/samuloza/data/samuel_lozano/cooked/1-GRID_RESULTS.tar \
/mnt/lustre/home/samuloza/data/samuel_lozano/cooked/1-GRID_RESULTS
