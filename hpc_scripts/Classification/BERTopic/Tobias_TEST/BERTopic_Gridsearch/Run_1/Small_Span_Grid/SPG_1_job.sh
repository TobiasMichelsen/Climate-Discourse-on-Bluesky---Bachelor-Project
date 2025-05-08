#!/bin/bash

#SBATCH --job-name=SPG_1_BERT            # Job name
#SBATCH --output=SPG_1_BERT.%j.out       # Output file (%j = job ID)
#SBATCH --error=SPG_1_BERT.%j.err        # Error file
#SBATCH --cpus-per-task=4                # Number of CPU cores
#SBATCH --mem=32G                        # Memory
#SBATCH --gres=gpu:v100:1                # GPU request
#SBATCH --time=36:00:00                  # Max run time
#SBATCH --partition=acltr                # Partition
#SBATCH --mail-type=FAIL                 # Email on failure

# Load CUDA
module load CUDA/12.1.1

# Load Python
module load Python/3.12.3-GCCcore-13.3.0
python --version
hostname

# Activate your environment
source ../../../envs/BERTopic_env/bin/activate

# Run the SPG_1 script
python SPG_1.py
