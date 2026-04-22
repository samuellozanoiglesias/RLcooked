#!/bin/bash
#SBATCH --job-name=compress
#SBATCH -t 01:00:00
#SBATCH --partition=short
#SBATCH --account=teruel
#SBATCH --ntasks=1
#SBATCH --output=tar-%x-%j.out
#SBATCH --error=tar-%x-%j.err

# Create the tar archive
tar -cvf /mnt/lustre/home/samuloza/data/samuel_lozano/cooked/classic_collision/empty_init/map_encouraged_division_of_labor_large/synergy_1.70/specialized_0.05/2d_grid_abilities_encouraged_map.tar \
/mnt/lustre/home/samuloza/data/samuel_lozano/cooked/classic_collision/empty_init/map_encouraged_division_of_labor_large/synergy_1.70/specialized_0.05/Training_*
