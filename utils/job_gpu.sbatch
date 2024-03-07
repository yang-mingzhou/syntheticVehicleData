#!/bin/bash

#SBATCH --job-name=node2vec ##job name
#SBATCH -o ../output/node2vec.output  ## output file name
#SBATCH -e ../output/node2vec.error ## error log
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --time=10:00:00
#SBATCH --tmp=2g
#SBATCH --mem=120g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yang7492@umn.edu

#SBATCH -p a100-4                                             
#SBATCH --gres=gpu:a100:1

source activate a40
module load cuda cuda-sdk
deviceQuery | grep NumDevs
nvcc --version
python node2vec.py