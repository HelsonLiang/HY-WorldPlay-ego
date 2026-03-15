#!/bin/bash
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --job-name=prepare_dataset_1811
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --output=log_%j.out


# 激活环境（若尚未激活）
source $(conda info --base)/etc/profile.d/conda.sh
conda activate worldplay

# 进入脚本目录
cd /home/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego/worldcompass/prepare_dataset


python prepare_image_text_latent_simple.py \
  --input_json /scratch/peilab/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego/gf_prepare_output/image_caption_list.json \
  --output_dir /scratch/peilab/ysunem/ys_26.2/3.13_real_mani/latents_output_3.15 \
  --hunyuan_checkpoint_path /scratch/peilab/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego
