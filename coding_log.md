# GameFactory → Middle Training / RL 流程日志（粗线条）

## 计划

- 用 **GameFactory (GF-Minecraft)** 数据做 middle training 和 RL。
- 数据源：`/scratch/peilab/ysunem/ys_26.2/3.13_real_mani/game_factory`（含 HuggingFace 下载的 data_2003：video + metadata + annotation.csv）。
- 目标：生成 hy_worldplay_ego 需要的 latents.json + .pt、pose JSON，再跑 WorldCompass RL / trainer 监督。

## 做了啥

### 1. 数据解压

- 在 `game_factory/GF-Minecraft/data_2003/` 下合并 part_* → `data_2003.zip`，解压得到 `data_2003/video/`、`metadata/`、`annotation.csv`。
- 实际数据目录：`/scratch/peilab/ysunem/ys_26.2/3.13_real_mani/game_factory/GF-Minecraft/data_2003/data_2003/`。

### 2. 2.1：image + caption 列表（供潜空间提取）

- **脚本**：`worldcompass/prepare_dataset/prepare_gamefactory_image_caption.py`
- 读 annotation.csv + 从每个 clip 的 mp4 抽一帧 → 写出图片 + 一份 JSON：`[{"image_path", "caption"}, ...]`。
- CSV 里视频名为 `seed_N_part.mp4`，实际文件为 `seed_N_part_N.mp4`，脚本里做了该映射。
- **输出**：图片目录 + `image_caption_list.json`，我们放在  
  `/scratch/peilab/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego/gf_prepare_output/`（frames/ + image_caption_list.json，约 15 万条）。

### 3. 2.2：metadata → pose JSON（WorldCompass / AR 用）

- **脚本**：`worldcompass/prepare_dataset/prepare_gamefactory_pose.py`
- 读每个 metadata JSON 的 actions（pos, pitch, yaw），转成 c2w + K，写出每 clip 一个 pose 文件（含 extrinsic/K 与 w2c/intrinsic 双格式）。
- **输出**：`gf_prepare_output/poses/*.json`（与 2.1 同根目录下）。

### 4. 潜空间提取（2.3）

- **脚本**：`worldcompass/prepare_dataset/prepare_image_text_latent_simple.py`  
  需在 **worldcompass** 目录下跑（否则 `import fastvideo` 失败）。
- **入口脚本**：`worldcompass/prepare_dataset/run_latent_4gpu.bash`（4 卡，`python -m torch.distributed.run`）。
- 输入：上面 2.1 的 `image_caption_list.json`；Hunyuan 权重：`/scratch/peilab/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego`（vae、vision_encoder/siglip、text_encoder/llm、Glyph-SDXL-v2 等）。
- 输出：`/scratch/peilab/ysunem/ys_26.2/3.13_real_mani/latents_output_3.15/`（latents/*.pt + latents.json）。

### 5. 合并 pose_path（为 middle / RL 准备）

- **脚本**：`worldcompass/prepare_dataset/merge_latents_pose.py`
- 读潜空间输出的 `latents.json`（含 latent_path、image_path、caption），按 image 文件名推出对应 pose 文件（seed_N_part_f*.jpg → poses/seed_N_part_N.json），给每条加上 `pose_path`，写出 `latents_with_pose.json`。
- 用法（潜空间跑完后执行）：
  ```bash
  cd worldcompass/prepare_dataset
  python merge_latents_pose.py \
    --latents_json /scratch/.../latents_output_3.15/latents.json \
    --poses_dir /scratch/.../gf_prepare_output/poses \
    --output_json /scratch/.../latents_output_3.15/latents_with_pose.json
  ```

### 6. Middle training（方案 A：AR 支持单帧，已做）

- **入口**：`scripts/training/hyvideo15/run_ar_hunyuan_action_mem.sh`，数据用上面带 `pose_path` 的 `latents_with_pose.json`（`--json_path`）。
- **方案 A 改动**（`trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py`）：
  - 单帧不再 skip：当 `latent_length < window_frames` 时用 `max_length = latent_length`、`max_frames = max_length`，保留该样本。
  - Pose 键：`pose_keys` 按数字排序；按帧取 pose 时用 `min(4*(i-1)+4, len(pose_keys)-1)` 防越界，单帧只用 `pose_keys[0]`。
  - 单帧时不做「选窗外帧」：仅当 `latent.shape[1] >= window_frames` 且 `select_prob < 0.8` 才做 memory 窗口选择；否则用 `pred_latent_size = latent.shape[1]`（单帧为 1）。
  - 单帧时 on-the-fly action：`c2ws.shape[0] > 1` 才做 `C_inv @ c2ws[1:]`，避免 `c2ws[:-1]` 为空报错。
- **脚本**：`run_ar_hunyuan_action_mem.sh` 顶部已加 GameFactory 单帧说明，`--json_path` 可指向 `latents_output_3.15/latents_with_pose.json`；单帧建议 `--window_frames 1`，且 `sp_size` 需整除 window_frames。

### 7. WorldCompass RL 代码改动（已做）

- **camera_dataset.py**（`worldcompass/fastvideo/dataset/camera_dataset.py`）  
  - 在 `HunyuanImageJsonDataset.__getitem__` 中：若当前条目的 `json_data` 含有 **`pose_path`** 且该文件存在，则从该路径读取 pose JSON（单条 pose 序列，格式同原 random_pose 中一项），用其建 w2c/intrinsic/action；否则仍用 **`random_pose[idx % len(self.random_pose)]`**。  
  - 对从文件读入的 pose 用 `sorted(..., key=int)` 取帧序，并检查帧数 ≥ `window_frames`。
- **train_worldcompass.sh**（`worldcompass/scripts/train_worldcompass.sh`，未改 backup）  
  - **TRAIN_LATENTS_DIR**：`/scratch/peilab/ysunem/ys_26.2/3.13_real_mani/latents_output_3.15`  
  - **EVAL_LATENTS_DIR**：同上（可后续改为单独 eval 目录或子集）。  
  - **--json_path / --eval_json_path**：改为使用 **`latents_with_pose.json`**（不再用 `latents.json`）。  
  - **POSE_PATH**：默认 `.../worldcompass/prepare_dataset/dataset/harder_random_poses.json`，供「无 pose_path 的样本」兜底；需保证该文件存在（或改为其他合法 pose 列表 JSON）。

### 8. 依赖与路径小记

- Glyph byt5：需 `text_encoder/Glyph-SDXL-v2/checkpoints/byt5_model.pt`，可用 ModelScope 下载 AI-ModelScope/Glyph-SDXL-v2 后把 checkpoints 拷进去。
- 多卡时用 `python -m torch.distributed.run`，避免用到 `~/.local/bin/torchrun`（未在 conda 环境里）。

## 关键路径汇总

| 用途           | 路径 |
|----------------|------|
| GameFactory 数据根 | `/scratch/peilab/ysunem/ys_26.2/3.13_real_mani/game_factory` |
| data_2003 解压后  | `.../game_factory/GF-Minecraft/data_2003/data_2003/`（video, metadata, annotation.csv） |
| 2.1/2.2 输出根   | `/scratch/peilab/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego/gf_prepare_output/` |
| image_caption 列表 | `gf_prepare_output/image_caption_list.json` |
| pose 文件       | `gf_prepare_output/poses/*.json` |
| Hunyuan 权重    | `/scratch/peilab/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego` |
| 潜空间输出      | `/scratch/peilab/ysunem/ys_26.2/3.13_real_mani/latents_output_3.15/` |
| 4 卡潜空间脚本  | `worldcompass/prepare_dataset/run_latent_4gpu.bash` |
| 合并 pose 脚本  | `worldcompass/prepare_dataset/merge_latents_pose.py` |
| 带 pose 的 latents（middle/RL 用） | `latents_output_3.15/latents_with_pose.json`（合并脚本输出） |
| Middle 训练入口 | `scripts/training/hyvideo15/run_ar_hunyuan_action_mem.sh` |
| RL 训练脚本（已改） | `worldcompass/scripts/train_worldcompass.sh`（未改 train_worldcompass_backup.sh） |
| RL 用 dataset（per-sample pose） | `worldcompass/fastvideo/dataset/camera_dataset.py` HunyuanImageJsonDataset |
| POSE_PATH 兜底 | `worldcompass/prepare_dataset/dataset/harder_random_poses.json` |



ps. 

python and torch for flash attention: 
(worldplay_fa) ysunem@slogin-02:~/ys_26.2/3.13_real_mani/hy_worldplay_ego/worldcompass/prepare_dataset$ which python
python -c "import torch; print(torch.__version__, torch.__file__)"
/home/ysunem/miniconda3/envs/worldplay_fa/bin/python
2.6.0+cu124 /home/ysunem/miniconda3/envs/worldplay_fa/lib/python3.10/site-packages/torch/__init__.py