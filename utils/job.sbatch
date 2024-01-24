#!/bin/bash

#SBATCH --job-name=syntheticData ##job name
#SBATCH -o ../output/syntheticData-%j.output  ## output file name
#SBATCH -e ../output/syntheticData-%j.error ## error log
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=10:00:00
#SBATCH --tmp=50g
#SBATCH --mem=40g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yang7492@umn.edu
#SBATCH -p agsmall                                             

source activate syntheticData
python mapMatching.py
