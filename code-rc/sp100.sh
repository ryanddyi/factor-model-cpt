#!/bin/bash
#SBATCH -J sp100
#SBATCH -o sp-out
#SBATCH -e sp-err
#SBATCH -n 1
#SBATCH -t 1000
#SBATCH --array=200-2518
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=ddyi@fas.harvard.edu # send-to addressfor i in {1..100000}; do
#SBATCH -p stats

python3 cpt_sp100.py ${SLURM_ARRAY_TASK_ID}
