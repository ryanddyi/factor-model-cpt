#!/bin/bash
#SBATCH -J sim
#SBATCH -o sim-out
#SBATCH -e sim-err
#SBATCH -n 1
#SBATCH -t 100
#SBATCH --array=1-2
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=ddyi@fas.harvard.edu # send-to addressfor i in {1..100000}; do
#SBATCH -p stats

python cpt_batch.py ${SLURM_ARRAY_TASK_ID}
