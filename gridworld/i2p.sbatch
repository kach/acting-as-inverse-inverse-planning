#!/bin/bash
#SBATCH --job-name=i2p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1G
#SBATCH --time=0:20:00
#SBATCH --output=slurm-out.out
#SBATCH --error=slurm-error.out
#SBATCH --array=0-4

#                                      runtime/beam_size/beam_cuts
# help, hinder, indifferent          - 15/500/1
# help-then-hinder, hinder-then-help - 15/500/1
# cinderella, irony                  - 25/500/1
# flashback-help, flashback-hinder   -  3/500/100
# experiment    					 -  7/100/100

python i2p.py\
	--seed $SLURM_ARRAY_TASK_ID\
	--which "help"\
	--runtime 15\
	--beam_size 500\
	--beam_cuts 1\
	--video_mode pov\
	--no-deus
