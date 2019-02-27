#!/bin/bash
#
#SBATCH --job-name=number_counts

#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --ntasks-per-node=24

#SBATCH --mem=60GB

module load python/2.7.13
source /home/rtobar/venvs/shark/bin/activate

MPLBACKEND=agg /home/rtobar/venvs/shark/bin/python selection-galaxies-test.py
#number-counts.py
