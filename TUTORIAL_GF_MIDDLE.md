# GameFactory → Middle Training 极简教程

## 0. 路径约定

- **项目根**：`/home/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego`
- **Scratch**：`/scratch/peilab/ysunem/ys_26.2/3.13_real_mani`
- **数据根**：`.../game_factory/GF-Minecraft/data_2003/data_2003/`（含 `video/`、`metadata/`、`annotation.csv`）
- **2.1/2.2 输出**：`.../hy_worldplay_ego/gf_prepare_output/`
- **潜空间输出**：`.../latents_output_3.15/`

---

## 1. 下载数据集

- 从 HuggingFace 下载 GameFactory GF-Minecraft **data_2003**（part_* 分片）。
- 在 `game_factory/GF-Minecraft/data_2003/` 下合并为 `data_2003.zip` 并解压，得到 `data_2003/video/`、`metadata/`、`annotation.csv`。
- 解压后目录示例：`/scratch/peilab/ysunem/ys_26.2/3.13_real_mani/game_factory/GF-Minecraft/data_2003/data_2003/`。

---

## 2. 数据处理（按顺序）

### 2.1 抽帧 + image_caption 列表

- **脚本**：`worldcompass/prepare_dataset/prepare_gamefactory_image_caption.py`
- **无需 GPU**，耗时可较长，建议 `nohup`/`screen` 或直接跑。

```bash
cd /home/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego

python worldcompass/prepare_dataset/prepare_gamefactory_image_caption.py \
  --data_dir /scratch/peilab/ysunem/ys_26.2/3.13_real_mani/game_factory/GF-Minecraft/data_2003/data_2003 \
  --output_image_dir /scratch/peilab/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego/gf_prepare_output/frames \
  --output_json /scratch/peilab/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego/gf_prepare_output/image_caption_list.json \
  --frame_mode start \
  --skip_existing
```

**输出**：`gf_prepare_output/frames/`、`gf_prepare_output/image_caption_list.json`。

---

### 2.2 metadata → pose JSON

- **脚本**：`worldcompass/prepare_dataset/prepare_gamefactory_pose.py`
- **无需 GPU**，直接跑即可。

```bash
cd /home/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego

python worldcompass/prepare_dataset/prepare_gamefactory_pose.py \
  --data_dir /scratch/peilab/ysunem/ys_26.2/3.13_real_mani/game_factory/GF-Minecraft/data_2003/data_2003 \
  --output_pose_dir /scratch/peilab/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego/gf_prepare_output/poses \
  --max_frames 2000 \
  --skip_existing
```

**输出**：`gf_prepare_output/poses/*.json`。

---

### 2.3 潜空间提取（4 卡，用 sbatch）

- **脚本**：`worldcompass/prepare_dataset/run_latent_4gpu.bash`（内已配 SBATCH，需在 worldcompass 下跑、conda `worldplay`）。

```bash
cd /home/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego
sbatch worldcompass/prepare_dataset/run_latent_4gpu.bash
```

**输出**：`/scratch/peilab/ysunem/ys_26.2/3.13_real_mani/latents_output_3.15/`（`latents/*.pt` + `latents.json`）。

---

### 2.4 合并 pose_path

- **脚本**：`misc_code/merge_latents_pose.sh`（内已配 SBATCH，调 `merge_latents_pose.py`）。

```bash
cd /home/ysunem/ys_26.2/3.13_real_mani/hy_worldplay_ego
sbatch misc_code/merge_latents_pose.sh
```

**输出**：`latents_output_3.15/latents_with_pose.json`。

---

## 3. 做 Middle Training 前要改什么

### 3.1 训练脚本 `run_ar_hunyuan_action_mem.sh`

**文件**：`scripts/training/hyvideo15/run_ar_hunyuan_action_mem.sh`

必改项：

| 项 | 说明 |
|----|------|
| `--json_path` | 改为 `latents_with_pose.json` 的完整路径，例如：`/scratch/peilab/ysunem/ys_26.2/3.13_real_mani/latents_output_3.15/latents_with_pose.json` |
| `--window_frames` | 单帧数据改为 **1**；且 `sp_size` 需整除 window_frames（用 1） |
| `MODEL_PATH` | 填 HunyuanVideo-1.5 权重目录（如 `.../hy_worldplay_ego`） |
| `--load_from_dir` | 填 pretrained transformer 目录 |
| `--ar_action_load_from_dir` | 填 pretrained AR action 模型目录 |
| `--output_dir` | 填本次 middle 输出目录 |
| `--wandb_key` / `--wandb_entity` / `--tracker_project_name` | 按需填写或关闭 wandb |

可选：`CUDA_VISIBLE_DEVICES`、`NUM_GPUS` 按机器改。

### 3.2 AR Dataset 里 negative prompt 路径

**文件**：`trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py`（约 405–412 行）

把两处占位路径改成你机器上的实际路径：

- `"/your_path/to/hunyuan_neg_prompt.pt"` → 实际 `hunyuan_neg_prompt.pt` 路径
- `"/your_path/to/hunyuan_neg_byt5_prompt.pt"` → 实际 `hunyuan_neg_byt5_prompt.pt` 路径

若还没有这两个 .pt，需用同一套 text encoder + byt5 对空字符串 encode 一次并保存（格式见 `coding_log.md` 或此前对话）。

### 3.3 代码层面：按方案 A 改 AR 支持单帧

做 middle 前需要先**按方案 A 改 AR dataset**（`trainer/dataset/ar_camera_hunyuan_w_mem_dataset.py`），否则单帧数据会被全部 skip，middle 跑不起来。改动要点：单帧不 skip（`latent_length < window_frames` 时用 `max_length = latent_length` 保留样本）、单帧 pose 索引（`pose_keys` 排序 + 取帧防越界）、单帧不做「选窗外」（仅当 `latent.shape[1] >= window_frames` 才做 memory 窗口选择）、单帧时 action 的 c2ws 保护（`c2ws.shape[0] > 1` 才做 `C_inv @ c2ws[1:]`）。改好 3.1、3.2、3.3 后再开跑。

---

## 4. 顺序小结

| 步骤 | 做什么 | 怎么跑 |
|------|--------|--------|
| 1 | 下载并解压 data_2003 | 手动 |
| 2.1 | 抽帧 + image_caption_list.json | 直接跑 `prepare_gamefactory_image_caption.py`（见上） |
| 2.2 | metadata → poses/*.json | 直接跑 `prepare_gamefactory_pose.py`（见上） |
| 2.3 | 潜空间 latents + latents.json | **sbatch** `worldcompass/prepare_dataset/run_latent_4gpu.bash` |
| 2.4 | 合并 pose_path → latents_with_pose.json | **sbatch** `misc_code/merge_latents_pose.sh` |
| 3 | Middle 开跑前 | 改 AR 支持单帧（方案 A，见 3.3）+ 改 `run_ar_hunyuan_action_mem.sh`（json_path、window_frames=1、模型/输出/wandb）+ 改 dataset 里 neg_prompt/neg_byt5 的 .pt 路径 |
