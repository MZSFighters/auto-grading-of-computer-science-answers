#!/bin/bash
#SBATCH --job-name=Unsupervised
#SBATCH --output=/home-mscluster/mmoonsamy/zPhase1/unsupervised/sampling/output/%N_%j.out
#SBATCH --error=/home-mscluster/mmoonsamy/zPhase1/unsupervised/sampling/output/%N_%j.err
#SBATCH --nodes=1
#SBATCH -w mscluster88
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=bigbatch

##SBATCH --gpus=1
##SBATCH --mem=10
##SBATCH --time=10:00:00


export PATH=/usr/local/cuda-11.5/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.5/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Activate the environment
source /home-mscluster/mmoonsamy/miniconda3/etc/profile.d/conda.sh
#conda init
conda activate researchenv

# Run your script
srun python3 random_num_clusters.py
