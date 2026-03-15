#!/bin/bash
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --job-name=merge_latents_pose
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=merge_latents_pose_%j.out

set -e
source $(conda info --base)/etc/profile.d/conda.sh
conda activate worldplay2

cd /home/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego/worldcompass/prepare_dataset

python merge_latents_pose.py \
  --latents_json /scratch/peilab/ysunem/ys_26.2/3.13_real_mani/latents_output_3.15/latents.json \
  --poses_dir /scratch/peilab/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego/gf_prepare_output/poses \
  --output_json /scratch/peilab/ysunem/ys_26.2/3.13_real_mani/latents_output_3.15/latents_with_pose.json
