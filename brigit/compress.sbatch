#!/bin/bash
#SBATCH --job-name=compress
#SBATCH -t 12:00:00
#SBATCH --partition=short
#SBATCH --account=teruel
#SBATCH --ntasks=1
#SBATCH --output=tar-%x-%j.out
#SBATCH --error=tar-%x-%j.err

# Create the tar archive
tar -cvf /mnt/lustre/home/samuloza/data/samuel_lozano/cooked/classic_collision/empty_init/PNAS_RESULTS.tar \
/mnt/lustre/home/samuloza/data/samuel_lozano/cooked/classic_collision/empty_init/PNAS_RESULTS
