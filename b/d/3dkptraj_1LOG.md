# 3D Keypoint Trajectory 实施日志

> 对应方案文档: [`b/d/3dkptraj_1.md`](3dkptraj_1.md)  
> 实施日期: 2026-08-10

---

## Phase 1 — 环境准备

### 操作记录

```bash
# RoboTwin 环境 (SAPIEN 提取)
/home/luogang/miniforge3/envs/RoboTwin/bin/pip install pyarrow pytest matplotlib

# GeoPredict 训练环境
conda create -n geopredict python=3.10 -y
/home/luogang/miniforge3/envs/geopredict/bin/pip install -r requirements.txt
/home/luogang/miniforge3/envs/geopredict/bin/pip install opencv-python pyarrow av pandas matplotlib pytest sentencepiece einops

# ckpts 准备
mkdir -p ckpts
ln -sf /home/luogang/.cache/openpi/big_vision/paligemma_tokenizer.model ckpts/paligemma_tokenizer.model
python tools/compute_robotwin_norm_stats.py
```

### 关键路径

| 用途 | 路径 |
|:---|:---|
| RoboTwin conda | `/home/luogang/miniforge3/envs/RoboTwin` |
| GeoPredict conda | `/home/luogang/miniforge3/envs/geopredict` |
| 源数据集 | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three/` |
| URDF | `/home/luogang/share/zwy/Projects/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf` |
| 输出数据集 | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three_kptsim/` |
| GeoPredict 根目录 | `/home/luogang/SRC/Robot/GeoPredict` |

### 文件变更

| 文件 | 操作 | 原因 |
|:---|:---|:---|
| `ckpts/paligemma_tokenizer.model` | 添加 symlink | RoboTwinDataset 需要 PaliGemma tokenizer |
| `ckpts/robotwin_norm_stats.json` | 新增 | 14 维 state/action 归一化统计 |

---

## Phase 2 — SAPIEN Keypoint 提取代码

### 操作记录

```bash
cd /home/luogang/SRC/Robot/GeoPredict
conda activate RoboTwin

# 单元测试
python -m pytest b/script/kpt_tst/ -v

# 全量提取
python b/script/kpt/run_extract.py

# 验收
python b/script/kpt_tst/validate_all.py
```

### 文件变更

| 文件 | 操作 | 原因 |
|:---|:---|:---|
| `b/script/kpt/__init__.py` | 新增 | Python 包 |
| `b/script/kpt/config.py` | 新增 | 路径、K=14、EEF/体素常量 |
| `b/script/kpt/sapien_env.py` | 新增 | AlohaFKScene: URDF 加载 + FK |
| `b/script/kpt/joint_mapper.py` | 新增 | 14 维 state → 38 维 qpos |
| `b/script/kpt/eef_calculator.py` | 新增 | TCP 位置 (_trans_endpose 复刻) |
| `b/script/kpt/coord_transform.py` | 新增 | 自动 offset + 范围验证 |
| `b/script/kpt/keypoint_extractor.py` | 新增 | 两遍扫描提取核心逻辑 |
| `b/script/kpt/run_extract.py` | 新增 | CLI 入口 |
| `b/script/kpt_tst/conftest.py` | 新增 | pytest 路径配置 |
| `b/script/kpt_tst/test_sapien_load.py` | 新增 | URDF 加载验证 |
| `b/script/kpt_tst/test_joint_mapper.py` | 新增 | 关节映射验证 |
| `b/script/kpt_tst/test_fk_home.py` | 新增 | Home FK 验证 |
| `b/script/kpt_tst/test_eef_tcp.py` | 新增 | TCP 计算验证 |
| `b/script/kpt_tst/test_extract_single.py` | 新增 | 单 episode 集成测试 |
| `b/script/kpt_tst/validate_all.py` | 新增 | 全量验收 + 可视化 |

### 错误与 Fix

#### Error 1: `test_arms_y_symmetric` 失败

- **现象**: `abs(left_y_mean + right_y_mean) = 0.847 > 0.3`
- **根因**: ALOHA root pose 绕 Z 轴旋转 90° 后，左右臂 Y 坐标均为负值，并非关于 Y=0 对称
- **Fix**: 将测试改为 `test_arms_y_similar`，验证左右臂 Y 均值差 < 0.02

---

## Phase 3 — 数据集生成与验收

### 操作记录

```bash
python b/script/kpt/run_extract.py \
  --dataset_dir /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three \
  --urdf_path /home/luogang/share/zwy/Projects/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf \
  --output_dir /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three_kptsim
```

### 结果

- 50 episodes, 23,550 frames, 耗时 ~7.6s
- 自动 offset: `[-0.8117267, -1.0236192, 0.50463176]`
- 变换后范围: X=[0.405, 1.195], Y=[0.365, 1.235], Z=[0.253, 0.747]
- `validate_all.py`: **PASS** (50/50 episodes)
- 可视化: `stack_bowls_three_kptsim/vis/keypoints_3d_samples.png`, `eef_trajectories_ep0.png`

---

## Phase 4 — GeoPredict 适配与 Smoke Training

### 操作记录

```bash
conda activate geopredict
cd /home/luogang/SRC/Robot/GeoPredict

CUDA_VISIBLE_DEVICES=0 python tools/train_robotwin_smoke.py \
  --dataset_dir /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three \
  --keypoints_dir /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three_kptsim \
  --num_train_steps 500 --batch_size 2 --log_interval 50 --num_workers 2
```

### 文件变更

| 文件 | 操作 | 原因 |
|:---|:---|:---|
| `models/geopredict.py` | 修改 | 添加 `joint_num`/`use_depth_loss` 参数；跳过 depth loss |
| `models/gaussian.py` | 修改 | lazy import diff_gaussian_rasterization |
| `data_processing/robotwin_dataset.py` | 新增 | LeRobot + kptsim 双路径数据加载 |
| `tools/compute_robotwin_norm_stats.py` | 新增 | 计算归一化统计 |
| `tools/train_robotwin_smoke.py` | 新增 | 无 DeepSpeed 的 smoke training |

### 错误与 Fix

#### Error 2: state 归一化维度不匹配

- **现象**: `RuntimeError: size of tensor a (32) must match size of tensor b (14)`
- **根因**: 先 pad 到 32 再归一化，但 norm_stats 是 14 维
- **Fix**: 先对 14 维归一化，再 `pad_to_dim` 到 32

#### Error 3: collate_fn 嵌套 dict 失败

- **现象**: `TypeError: expected Tensor ... but got dict` (images / cam_infos)
- **根因**: `images` 和 `cam_infos` 是二层嵌套 dict，通用 collate 无法处理
- **Fix**: 为 `images`、`image_masks`、`cam_infos`、`step` 添加专用 collate 分支

#### Error 4: step 字段无法 unsqueeze

- **现象**: `AttributeError: 'list' object has no attribute 'unsqueeze'`
- **根因**: collate 将 int 类型的 step 保留为 list
- **Fix**: `step` 专用分支 `torch.tensor(values, dtype=torch.long)`

### Smoke Training 结果

| Step | total_loss | action_loss | current_kpt | future_kpt |
|:---:|:---:|:---:|:---:|:---:|
| 50 | 3.3840 | 1.4933 | 0.9167 | 0.9740 |
| 100 | 1.3468 | 1.2358 | 0.0400 | 0.0710 |
| 500 | 0.4749 | 0.4551 | 0.0087 | 0.0110 |

- 500 步耗时 ~354s (~6 min)，无 NaN/崩溃，loss 稳定下降
- 未加载预训练权重 (用户选择 skip pretrain)

---

## 数据集 Schema 与使用说明

### 目录结构

```
stack_bowls_three_kptsim/
├── episode_000000/
│   └── keypoints.npy          # float32 [T, 42]
├── episode_000001/
│   └── keypoints.npy
├── ...
├── episode_000049/
│   └── keypoints.npy
├── keypoints_meta.json        # 元信息
└── vis/                       # 验收可视化
    ├── keypoints_3d_samples.png
    └── eef_trajectories_ep0.png
```

### keypoints.npy

| 属性 | 值 |
|:---|:---|
| dtype | `float32` |
| shape | `[T, 42]`，T = 该 episode 帧数 |
| reshape | `[T, 14, 3]` 得 14 个关键点 XYZ |
| 坐标系 | GeoPredict 体素空间 (已减 offset) |
| 范围 | 均在 `[0, 1.6] × [0, 1.6] × [0, 1.0]` 内 |

### 关键点索引 (K=14)

| Index | Name | 含义 |
|:---:|:---|:---|
| 0–5 | fl_link1 ~ fl_link6 | 左臂 6 个 link |
| 6 | fl_eef_tcp | 左臂 TCP (gripper_bias=0.12m) |
| 7–12 | fr_link1 ~ fr_link6 | 右臂 6 个 link |
| 13 | fr_eef_tcp | 右臂 TCP |

### keypoints_meta.json 字段

```json
{
  "K": 14,
  "keypoint_names": ["fl_link1", "...", "fr_eef_tcp"],
  "coord_offset": [-0.812, -1.024, 0.505],
  "world_range_min": [...],
  "world_range_max": [...],
  "transformed_range_min": [...],
  "transformed_range_max": [...],
  "urdf_path": "...",
  "dataset_dir": "...",
  "total_episodes": 50
}
```

### 如何使用

**1. 重新生成 keypoints**

```bash
conda activate RoboTwin
cd /home/luogang/SRC/Robot/GeoPredict
python b/script/kpt/run_extract.py
```

**2. GeoPredict 训练 (smoke test)**

```bash
conda activate geopredict
cd /home/luogang/SRC/Robot/GeoPredict

# 计算归一化统计 (首次)
python tools/compute_robotwin_norm_stats.py

# 训练
CUDA_VISIBLE_DEVICES=0 python tools/train_robotwin_smoke.py \
  --dataset_dir /path/to/stack_bowls_three \
  --keypoints_dir /path/to/stack_bowls_three_kptsim \
  --num_train_steps 500 --batch_size 2
```

**3. 数据对齐关系**

- LeRobot 主数据 (`stack_bowls_three`) 提供: Parquet state/action、MP4 三视角、prompt
- kptsim 数据提供: `episode_{idx:06d}/keypoints.npy`
- 通过 `episode_index` 对齐，step 索引与 Parquet 行号一致

**4. 图像映射 (RoboTwin → GeoPredict)**

| GeoPredict 输入 | RoboTwin 视频 |
|:---|:---|
| `left_rgb` | `observation.images.cam_high` |
| `right_rgb` | `observation.images.cam_right_wrist` |
| `wrist_rgb` | `observation.images.cam_left_wrist` |

**5. 模型配置**

- `joint_num=14` (双臂 K=14)
- `use_depth_loss=False` (RoboTwin 无深度 GT)
- state/action: 14 维关节空间，pad 到 32

---

## 总结

| 阶段 | 状态 |
|:---|:---|
| SAPIEN 提取代码 | ✅ 完成 |
| 单元测试 (31/31) | ✅ 通过 |
| 数据集生成 (50 ep) | ✅ 完成 |
| 验收脚本 | ✅ PASS |
| GeoPredict smoke training (500 step) | ✅ 稳定运行 |
