#!/bin/bash
#SBATCH -J garch
#SBATCH -o garch-out
#SBATCH -e garch-err
#SBATCH -n 1
#SBATCH -t 1000
#SBATCH --array=200-2518
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=ddyi@fas.harvard.edu # send-to addressfor i in {1..100000}; do
#SBATCH -p stats

python3 cpt_garch.py ${SLURM_ARRAY_TASK_ID}
