#!/bin/bash
#
#SBATCH --job-name=number_counts_ud

#SBATCH -A pawsey0119
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=20GB 
#SBATCH --cpus-per-task=10 
#SBATCH -n 1

module load python/3.9.15 py-h5py/3.4.0 py-matplotlib/3.4.3 py-scipy/1.7.1 py-astropy/4.2.1

export MPLBACKEND=agg
python selection-galaxies-atlast-udeep.py > lala
#extinction_calculation.py
#selection-galaxies-submm.py
#fir-flux-redshift-analysis.py
#temperature_calculation.py
#number-counts.py
#selection-galaxies-test.py
#number-counts.py
