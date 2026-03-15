#!/bin/bash
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --job-name=prepare_latent_4gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=30:00:00
#SBATCH --output=log_latent_4gpu_%j.out

set -e
source $(conda info --base)/etc/profile.d/conda.sh
conda activate worldplay

cd /home/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego/worldcompass

python -m torch.distributed.run  --nproc_per_node=4 prepare_dataset/prepare_image_text_latent_simple.py \
  --input_json /scratch/peilab/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego/gf_prepare_output/image_caption_list.json \
  --output_dir /scratch/peilab/ysunem/ys_26.2/3.13_real_mani/latents_output_3.15 \
  --hunyuan_checkpoint_path /scratch/peilab/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego \
  --skip_existing
