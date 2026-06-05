#!/bin/bash
#SBATCH --job-name=compress
#SBATCH -t 01:00:00
#SBATCH --partition=short
#SBATCH --account=teruel
#SBATCH --ntasks=1
#SBATCH --output=tar-%x-%j.out
#SBATCH --error=tar-%x-%j.err

# Create the tar archive
tar -cvf /mnt/lustre/home/samuloza/data/samuel_lozano/cooked/pretraining_competition.tar \
/mnt/lustre/home/samuloza/data/samuel_lozano/cooked/pretraining
