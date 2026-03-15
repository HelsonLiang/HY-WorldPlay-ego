#!/bin/bash
#SBATCH --account=peilab
#SBATCH --job-name=flash-attn
#SBATCH --partition=preempt
#SBATCH --time=01:00:00
#SBATCH --output=../assets/logs/flash_attn_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

# 与 run.sh 一致的环境，便于在 sbatch 里装 flash-attn（编译时的 GPU/CUDA 与推理时一致）
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate worldplay_fa

cd /home/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego
export PYTHONPATH=$PYTHONPATH:.

echo "===== env check ====="
python -c "
import torch
print('torch:', torch.__version__, torch.__file__)
print('cuda:', torch.version.cuda)
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
"
echo ""

echo "===== uninstall old flash-attn (if any) ====="
pip uninstall -y flash-attn 2>/dev/null || true
pip cache remove flash-attn 2>/dev/null || true
echo ""

echo "===== install deps ====="
pip install -q packaging psutil ninja
echo ""

echo "===== install flash-attn (from source, may take several min) ====="
MAX_JOBS=4 python -m pip install flash-attn --no-build-isolation --no-cache-dir --no-binary flash-attn
echo ""

echo "===== verify ====="
python -c "
from flash_attn import flash_attn_func
print('flash_attn: OK')
"
echo "===== done ====="
