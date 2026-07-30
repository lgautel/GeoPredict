# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and Cursor when working with code in this repository.

## Project Overview

GeoPredict is a geometry-aware Vision-Language-Action (VLA) framework for robotic manipulation (CVPR 2026 Highlight). It augments a Pi0-style VLA backbone with predictive kinematic and 3D Gaussian geometry modules that are used only during training, keeping inference lightweight.

## Commands

### Environment Setup
```bash
conda create -n geopredict python=3.10 -y
conda activate geopredict
pip install -r requirements.txt
```
RoboCasa simulator must be installed separately from https://github.com/robocasa/robocasa.

### Training (RoboCasa)
```bash
bash train_robocasa.sh
```
Uses DeepSpeed ZeRO Stage 3 (bf16). Pretrained weights go in `./ckpts/pi0_base.pth`. Training data expected at `./data/robocasa/`.

### Evaluation (RoboCasa)
```bash
bash test_robocasa.sh
```
Requires MuJoCo with OSMesa rendering (`MUJOCO_GL=osmesa`). Distributes 24 tasks across 8 GPUs. Checkpoints and tokenizer model must be in `./ckpts/`.

### Run a single evaluation task
```bash
MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES=0 python tools/test_robocasa.py \
    --suite_idx 0 --seed 7 --num_trials_per_task 50 --replan_steps 5 \
    --weight_path ./ckpts/GeoPredict_robocasa.pth \
    --task_data_path ./ckpts/robocasa
```
`--suite_idx` selects from the 24 RoboCasa tasks (0-23, defined in `ROBOCASA_TASKS` list in `tools/test_robocasa.py`).

## Architecture

### Model (GeoPredict in `models/geopredict.py`)

The model processes a heterogeneous token sequence through a shared Gemma LLM backbone. The token sequence is built in two stages:

**Prefix tokens** (`embed_prefix`): image tokens (SigLIP) + language tokens (Gemma embedder) + history keypoint tokens (TrackEncoder) + future keypoint query tokens + spatial query tokens (for 3D Gaussian prediction). Bidirectional attention within each group, autoregressive between groups.

**Suffix tokens** (`embed_suffix`): state token + noisy action tokens (flow matching). Suffix attends to prefix but not vice versa.

**Training** (`compute_loss`): four loss terms jointly optimized:
1. **Action loss** - flow matching velocity prediction (noise - actions)
2. **Current keypoint loss** - predict 8 robot joint 3D positions at current timestep
3. **Future keypoint loss** - predict keypoint trajectories across action horizon (50 steps) using sinusoidal time embeddings
4. **Depth rendering loss** - decode spatial tokens into 3D Gaussian voxels (40x40x25 grid), render depth maps via differentiable Gaussian splatting, supervise with ground truth depths from left/right cameras. Track-guided refinement densifies Gaussians near predicted keypoint locations.

**Inference** (`sample_actions`): only action denoising runs. Prefix is computed once and cached (KV cache). Iterative denoising (default 10 steps) refines actions from noise using cached prefix context. Keypoint/Gaussian modules are unused.

### Backbone Components

- **Gemma** (`models/gemma.py`): multi-expert Gemma LLM with two width configs (2B: width=2048 for prefix, 300M: width=1024 for action expert). Shared attention (GQA with RoPE), separate FFN per expert. 18 layers.
- **SigLIP** (`models/siglip.py`): ViT image encoder (patch_size=14, width=1152, 27 layers) projecting to 2048-dim tokens.
- **TrackEncoder** (`models/keypoints.py`): encodes variable-length keypoint history via patch embedding (stride=4 over time) + cross-attention with learnable queries, producing per-joint tokens.
- **VoxelDecoder** (`models/head.py`): upsamples spatial tokens (8x8x5) to voxel grid (40x40x25) via transposed 3D convolutions, outputting Gaussian parameters (14 params x 4 Gaussians per voxel).
- **GaussianRenderer** (`models/gaussian.py`): differentiable Gaussian splatting using `diff_gaussian_rasterization`. Renders depth (and RGB/alpha) from voxelized 3D Gaussians given camera intrinsics/extrinsics.

### Data Pipeline

- **RobocasaDataset** (`data_processing/robocasa_dataset.py`): loads episodes from disk. Each sample includes left/right/wrist images (224x224), robot state (8-dim padded to 32), action chunks (horizon=50), keypoint history/future, depth maps, and camera parameters. Uses PaliGemma tokenizer for language prompts.
- **Normalization**: state and actions are z-score normalized using precomputed stats from `./ckpts/robocasa_norm_stats.json`.
- **Image augmentation**: ColorJitter (brightness, contrast, saturation) during training; images rescaled to [-1, 1].

### Key Constants
- `action_dim=32` (12 actual dims padded to 32), `action_horizon=50`, `max_token_len=48`
- Voxel grid: coarse 8x8x5, decoded to 40x40x25, point cloud range [0,0,0] to [1.6,1.6,1.0]
- 8 robot keypoints (7 arm links + gripper EEF)
- Gaussian params per primitive: 14 (3 offset + 1 opacity + 3 scale + 4 rotation + 3 RGB)

### Training Infrastructure
- DeepSpeed ZeRO Stage 3 with bf16, gradient clipping 1.0
- EMA model maintained alongside training model; EMA weights are saved as checkpoints
- AdamW optimizer with cosine decay LR schedule (warmup 1k steps, decay over 30k steps)
- Missing keys for keypoint/spatial/gs_decoder/renderer/refine modules are expected (initialized from scratch when fine-tuning from Pi0 base)

# 介绍

- 这是论文[GeoPredict: Leveraging Predictive Kinematics and 3D Gaussian Geometry for Precise VLA Manipulation](https://arxiv.org/abs/2512.16811)的代码库, 论文的html版在 https://arxiv.org/html/2512.16811v2 , 论文的本地 tex 版在`b/d/paper/TeX_Source`, 
- 论文的项目主页在 https://jingjingqian75.github.io/GeoPredict-Page/  , 
- 论文的GitHUb在 https://github.com/jingjingqian75/GeoPredict  ,


# 设计/方案/分析/解释和写文档的注意点

* 图表用mermaid, 数学相关的用LaTex, 必要时可以用py脚本画一些更能帮助读者理解的图片(图片中的文字用英文). 这些脚本和图一般放在与生成的文档同目录的`asset`子文件夹中.
* 如果在公式和内容中用到了数学符号或代号, 请在该公式或内容的附近对该符号给予解释.
* 分析,解析和撰写文档时, 可以参考论文或代码库的官网, 官方文档, GItHUb, 参考github中的issues, 代码和pull requests, 也可参考网上其它可信来源的相关文章, 但参考内容要列出, 所生产的文档中若有与被参考对象相关的内容也要指出内容的出处. 
* 分析要深入仔细, 既要包括纵向分析(算法或方法的由来与演进历史, 以及在该算法或方法的基础上又演进和优化出了些什么解决类似问题的方法, 新老方法各有什么优缺点, 各适合应用到什么场景), 纵向分析(同时期同类算法的对比分析, 不同算法或方法各有什么优缺点, 各适合应用到什么场景), 和 消融分析(算法或方法中哪些点是在benchmark实验或实践中被证明有效的, 哪些点相对来说更有效, 哪些没那么有效).
* 记得深入分析模型或方法的输入,输出,在输入输出间做了些什么处理. 当然, 各组成模块的输入输出以及中间的处理也要分析. 为了训这个模型用了什么数据集和任务, 训出来后能做什么任务, 训练和推理时的输入输出数据格式大概长什么样.
* 系统或程序的设计要包括静态架构(组件图,类图,组件和类的职责与关系等等)和动态架构(数据流图,序列图,工作流图,不同场景下的各组件或类的调用与协调图.如果是算法还会涉及forward阶段的数据流,模型组件间的调用,以及backwawrd阶段的数据流,gradient流,哪些权重冻结哪些会被更新,和模型组件间的调用等等).
* 代码还是以该代码库的本地代码为准, 但可用参考网上GitHub的issues, commits, pull requests等.
* 解释要深入浅出, 图文并茂, 可以举一些易于理解的例子帮助说明, 对关键的逻辑也要进行深入的代码解读, 要用严谨的科普论文的风格.

