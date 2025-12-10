#!/bin/bash
# SBATCH --job-name=disaggregated            # Job name

# SBATCH --account=jiexiao                   # Replace with your netid
# SBATCH --partition=gpu                   # Partition name (default is gpu)
# SBATCH --nodes=1                         # One node (tomago/tempura)
# SBATCH --ntasks-per-node=1               # Number of CPU cores (vary it as needed)
# SBATCH --mem=16G                         # Memory per node
# SBATCH --gpus=2                          # Number of GPUs (for HW1, let it be 1)
# SBATCH --time=02:00:00                   # Max runtime = 2 hours (cluster limit)

# SBATCH --chdir=/homes/iws/jiexiao/assignment3-rl  # Working directory
# SBATCH --export=ALL                      # Export environment variables
# SBATCH --output=jiexiao-slurm-%j.out             # STDOUT log file (%j = JobID)
# SBATCH --error=jiexiao-slurm-%j.err              # STDERR log file (%j = JobID)

# --- Example Command to run ---
srun uv run python ./cse599o_alignment/train_grpo_ray_disaggregated.py --steps 3