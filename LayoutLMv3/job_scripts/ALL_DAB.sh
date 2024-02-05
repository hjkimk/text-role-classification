#!/bin/sh
#SBATCH --output=/LayoutLMv3/slurm_logs/slurm-%A_%a.out
#SBATCH --error=/LayoutLMv3/slurm_logs/slurm-%A_%a.out
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
source venv/bin/activate
python3 main.py /LayoutLMv3/training_config/ALL_DAB.json

# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
