#!/bin/bash
#SBATCH --account=peilab
#SBATCH --job-name=hyvideo
#SBATCH --partition=preempt
#SBATCH --time=24:00:00
#SBATCH --output=assets/logs/%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=30:00:00


source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate worldplay2

cd /home/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego
export PYTHONPATH=$PYTHONPATH:.

python -c "
import torch
print('torch:', torch.__version__, torch.__file__)
print('cuda:', torch.version.cuda)
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
import flash_attn
print('flash_attn: ok')
"



export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export I2V_REWRITE_MODEL_NAME="<your_model_name>"

## changable:
PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water.  Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  The overall composition emphasizes the peaceful and harmonious nature of the landscape. I looked to the right and saw a red monster with nine eyes, in a Cthulhu-like style, with bold colors and rugged, uneven skin. When I looked upward, the sky became gloomy.'
## changable:
IMAGE_PATH=./assets/img/test.png # Now we only provide the i2v model, so the path cannot be None
SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p # Now we only provide the 480p model
OUTPUT_PATH=./outputs/
MODEL_PATH=/scratch/peilab/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego
AR_RL_ACTION_MODEL_PATH=/scratch/peilab/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego/ar_rl_model/diffusion_pytorch_model.safetensors
AR_DISTILL_ACTION_MODEL_PATH=/scratch/peilab/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego/ar_distilled_action_model/diffusion_pytorch_model.safetensors
## changable:
POSE='right-12, a-3, w-31, up-33'                   # Camera trajectory: pose string (e.g., 'w-31' means generating [1 + 31] latents) or JSON file path
## changable:
NUM_FRAMES=317
WIDTH=832
HEIGHT=480

# Configuration for faster inference
# The maximum number recommended is 8.
## changable:
N_INFERENCE_GPU=1 # Parallel inference GPU count.

# Configuration for better quality
REWRITE=false   # Enable prompt rewriting. Please ensure rewrite vLLM server is deployed and configured.
ENABLE_SR=false # Enable super resolution. When the NUM_FRAMES == 125, you can set it to true

# inference with bidirectional model
# python -m torch.distributed.run --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
#   --prompt "$PROMPT" \
#   --image_path $IMAGE_PATH \
#   --resolution $RESOLUTION \
#   --aspect_ratio $ASPECT_RATIO \
#   --video_length $NUM_FRAMES \
#   --seed $SEED \
#   --rewrite $REWRITE \
#   --sr $ENABLE_SR --save_pre_sr_video \
#   --pose "$POSE" \
#   --output_path $OUTPUT_PATH \
#   --model_path $MODEL_PATH \
#   --action_ckpt $BI_ACTION_MODEL_PATH \
#   --few_step false \
#   --model_type 'bi'

# inference with autoregressive model
# python -m torch.distributed.run --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
#   --prompt "$PROMPT" \
#   --image_path $IMAGE_PATH \
#   --resolution $RESOLUTION \
#   --aspect_ratio $ASPECT_RATIO \
#   --video_length $NUM_FRAMES \
#   --seed $SEED \
#   --rewrite $REWRITE \
#   --sr $ENABLE_SR --save_pre_sr_video \
#   --pose "$POSE" \
#   --output_path $OUTPUT_PATH \
#   --model_path $MODEL_PATH \
#   --action_ckpt $AR_ACTION_MODEL_PATH \
#   --few_step false \
#   --width $WIDTH \
#   --height $HEIGHT \
#   --model_type 'ar'

# inference with autoregressive distilled model
python -m torch.distributed.run --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py \
  --prompt "$PROMPT" \
  --image_path $IMAGE_PATH \
  --resolution $RESOLUTION \
  --aspect_ratio $ASPECT_RATIO \
  --video_length $NUM_FRAMES \
  --seed $SEED \
  --rewrite $REWRITE \
  --sr $ENABLE_SR --save_pre_sr_video \
  --pose "$POSE" \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH \
  --action_ckpt $AR_DISTILL_ACTION_MODEL_PATH \
  --few_step true \
  --num_inference_steps 4 \
  --model_type 'ar' \
  --use_vae_parallel false \
  --use_sageattn false \
  --use_fp8_gemm false \
  2>&1 | tee -a assets/job_logs/train_$(date +%Y%m%d_%H%M%S).out