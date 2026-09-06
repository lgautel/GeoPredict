# RoboTwin 2.0 全部 50 子任务 3D 关键点位置 + 姿态四元数抽取处理日志

> 日期: 2026-09-05  
> 对应方案文档: [`3dkptraj_1.md`](3dkptraj_1.md)（附录二: SAPIEN 方案(含关键点及其位姿)详细实施设计）  
> 环境: conda `RoboTwin`，批处理脚本 `b/script/kpt/batch_extract_lrb3_kptsim7.py`  
> 输出格式: LeRobot v3.0  
> 输出后缀: `_lrb3_kptsim7`

---

## 1. 任务概述

对 RoboTwin 2.0 的全部 50 个子任务原始训练数据（每个子任务 50 个 episode），通过 SAPIEN FK（正运动学）抽取 14 个关键点的 **3D 位置** 和 **姿态四元数**，并将结果以 LeRobot v3.0 格式保存。

- **输入**: `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/<task>/`（LeRobot v2.1 格式，只读）
- **输出**: `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/<task>_lrb3_kptsim7/`（LeRobot v3.0 格式）
- **约束**: 原始数据只读，不做任何修改

### 输出数据内容

| 特征列 | 维度 | 说明 |
|--------|------|------|
| `observation.state` | [14] | 原始关节状态（直接复制） |
| `action` | [14] | 原始动作（直接复制） |
| `observation.keypoint_3d` | [42] | 14 关键点 × 3D 位置，经 auto-offset 映射到体素空间 |
| `observation.keypoint_quat` | [56] | 14 关键点 × 4D 四元数 (wxyz)，世界坐标系下的姿态 |

### 14 个关键点

| 索引 | 名称 | 来源 |
|------|------|------|
| 0-5 | fl_link1 ~ fl_link6 | 左臂 6 个 link 的 body-frame pose |
| 6 | fl_eef_tcp | 左臂 TCP（经 gripper_bias + GLOBAL_TRANS_MATRIX 变换） |
| 7-12 | fr_link1 ~ fr_link6 | 右臂 6 个 link 的 body-frame pose |
| 13 | fr_eef_tcp | 右臂 TCP |

---

## 2. 环境与配置

| 项目 | 值 |
|------|-----|
| Conda 环境 | `RoboTwin` (`/home/luogang/miniforge3/envs/RoboTwin`) |
| Python | 3.10 |
| SAPIEN | 3.0.0b1 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 (升级修复，见下文) |
| URDF | `/home/luogang/share/zwy/Projects/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf` |
| 数据根目录 | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/` |
| 四元数约定 | Hamilton [w,x,y,z]（SAPIEN / transforms3d 标准） |
| 体素空间范围 | [0, 1.6] × [0, 1.6] × [0, 1.0] |

---

## 3. 代码修改记录

### 3.1 附录二实施代码（在本次批处理之前已完成）

以下文件在 `GeoPredict/b/script/kpt/` 下：

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `config.py` | 追加 | 新增 `QUAT_DIM=4`, `QUAT_CONVENTION="wxyz"`, `QUAT_FILENAME="keypoint_quats.npy"` |
| `sapien_env.py` | 追加方法 | 新增 `get_link_poses()` 方法，返回位置+四元数 |
| `eef_calculator.py` | 追加函数 | 新增 `compute_tcp_pose()` 函数，计算 TCP 位置+姿态 |
| `keypoint_pose_extractor.py` | 新建 | `KeypointPoseExtractor` 继承 `KeypointExtractor`，重写 4 个方法 |

### 3.2 批处理脚本（本次新建）

| 文件 | 说明 |
|------|------|
| `batch_extract_lrb3_kptsim7.py` | 批量处理脚本：SAPIEN FK 抽取 + v2.1 → v3 转换 + 验收 |

---

## 4. 处理过程

### 4.1 首次测试: scan_object（成功）

```bash
cd /home/luogang/SRC/Robot/GeoPredict
conda run -n RoboTwin python b/script/kpt/batch_extract_lrb3_kptsim7.py --tasks scan_object
```

结果: **PASS**，8463 帧，3.7 秒完成。

验证:
- `observation.keypoint_3d`: shape [42], 范围 [0.217, 0.982] — 在体素空间内
- `observation.keypoint_quat`: shape [56], 四元数范数 ≈ 1.0
- v3 格式结构完整: info.json, keypoints_meta.json, stats.json, episodes parquet, norm_stat.json

### 4.2 Error 1: SciPy / NumPy 版本不兼容

**现象**: 运行时报错
```
AttributeError: _ARRAY_API not found
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```

**根因**: RoboTwin 环境中 SciPy 1.10.1 与 NumPy 2.2.6 不兼容。SciPy 1.10.1 编译时基于 NumPy 1.x，不支持 NumPy 2.x 的 ABI。

**修复**:
```bash
conda run -n RoboTwin pip install "scipy>=1.14"
# scipy 1.10.1 → 1.15.3
```

### 4.3 Error 2: SAPIEN GPU 资源耗尽（Segfault）

**现象**: 首次全量运行 50 个子任务时，在第 18 个子任务（open_microwave）处崩溃:
```
RuntimeError: Failed to find a supported physical device "cuda:0"
Segmentation fault (core dumped)  [exit code 139]
```

实际已完成 26 个子任务的输出目录，因 `conda run` 缓冲了 stdout，日志只显示到第 17 个。

**根因**: 初版脚本为每个子任务创建新的 SAPIEN Engine + Scene + RenderSystem。SAPIEN 3.0 的 `Scene()` 构造函数会分配 GPU 渲染资源（`sapien.render.RenderSystem()`）。在同一进程中反复创建/销毁（即使调用了 `close()` 将引用置 None），底层原生 GPU 资源并未被完全释放，导致资源泄漏。经过 ~26 次创建后 GPU 资源耗尽，触发 segfault。

**修复**: 重构 `batch_extract_lrb3_kptsim7.py`，改为在全局创建一个共享的 `KeypointPoseExtractor`（仅一个 SAPIEN 场景），在不同子任务间复用。每次切换任务只需更改 `dataset_dir` 并清空缓存:

```python
def create_shared_extractor(urdf_path: Path) -> KeypointPoseExtractor:
    """Create a single extractor whose SAPIEN scene is reused across tasks."""
    return KeypointPoseExtractor(urdf_path=urdf_path, ...)

def extract_keypoints_for_task(task_name, task_dir, extractor):
    extractor.dataset_dir = task_dir
    extractor._world_cache.clear()
    extractor._quat_cache.clear()
    ...
```

### 4.4 第二次运行: 剩余 24 个子任务（成功）

```bash
conda run -n RoboTwin python b/script/kpt/batch_extract_lrb3_kptsim7.py --skip_existing
```

结果: **全部 24/24 PASS**，91 秒完成。

### 4.5 全量验证: 50 个子任务（成功）

```python
# 验证脚本检查:
# 1. keypoint_3d 位置在体素空间 [0, 1.6]×[0, 1.6]×[0, 1.0] 内
# 2. keypoint_quat 四元数范数 |q| ≈ 1.0（误差 < 1e-3）
# 3. 所有 meta 文件存在 (info.json, keypoints_meta.json, stats.json, tasks.parquet, episodes parquet, norm_stat.json)
```

结果: **ALL PASS (50 tasks), 549,787 total frames**

---

## 5. 输出目录结构（LeRobot v3 格式）

每个子任务的输出目录 `<task>_lrb3_kptsim7/` 结构如下:

```
<task>_lrb3_kptsim7/
├── data/
│   └── chunk-000/
│       └── file-000.parquet           # 全部 episode 合并的数据 parquet
├── meta/
│   ├── info.json                      # v3.0 元信息（含 features 定义）
│   ├── keypoints_meta.json            # 关键点元信息（K=14, offset, ranges, quat config）
│   ├── stats.json                     # 全局特征统计（min/max/mean/std/count）
│   ├── tasks.parquet                  # 任务文本 → task_index 映射
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet       # 每 episode 的元信息 + 统计
├── norm_stat.json                     # 归一化统计（mean/std per feature）
└── videos/
    ├── observation.images.cam_high/
    │   └── chunk-000/
    │       ├── file-000.mp4 → (symlink to original episode_000000.mp4)
    │       ├── file-001.mp4 → ...
    │       └── ...
    ├── observation.images.cam_left_wrist/
    │   └── chunk-000/ ...
    └── observation.images.cam_right_wrist/
        └── chunk-000/ ...
```

**视频处理说明**: 视频文件以 symlink 方式链接到原始 v2.1 episode 视频，每个 episode 对应独立的 `file-{ep_idx:03d}.mp4`。Episodes parquet 中记录了每个 episode 对应的 `file_index`。

---

## 6. 各子任务处理结果

| 子任务 | Episodes | Frames | Auto Offset [x, y, z] | 状态 |
|--------|----------|--------|------------------------|------|
| adjust_bottle | 50 | 7,188 | [-0.7899, -1.0709, 0.4863] | PASS |
| beat_block_hammer | 50 | 5,682 | [-0.8022, -1.0436, 0.4937] | PASS |
| blocks_ranking_rgb | 50 | 23,041 | [-0.7940, -1.0424, 0.4982] | PASS |
| blocks_ranking_size | 50 | 23,170 | [-0.7775, -1.0393, 0.5187] | PASS |
| click_alarmclock | 50 | 4,252 | [-0.8046, -1.0709, 0.5645] | PASS |
| click_bell | 50 | 3,855 | [-0.8042, -1.0732, 0.5082] | PASS |
| dump_bin_bigbin | 50 | 12,122 | [-0.8539, -1.0233, 0.5169] | PASS |
| grab_roller | 50 | 4,728 | [-0.9804, -1.1273, 0.4770] | PASS |
| handover_block | 50 | 14,084 | [-0.8569, -0.9790, 0.5648] | PASS |
| handover_mic | 50 | 11,092 | [-0.8041, -1.0775, 0.4743] | PASS |
| hanging_mug | 50 | 16,889 | [-0.7718, -1.0504, 0.4779] | PASS |
| lift_pot | 50 | 5,554 | [-0.8077, -1.0281, 0.4945] | PASS |
| move_can_pot | 50 | 7,568 | [-0.8672, -1.0234, 0.5540] | PASS |
| move_pillbottle_pad | 50 | 7,345 | [-0.8020, -1.0381, 0.6028] | PASS |
| move_playingcard_away | 50 | 5,884 | [-0.8136, -1.0351, 0.4990] | PASS |
| move_stapler_pad | 50 | 7,749 | [-0.8024, -1.0637, 0.5248] | PASS |
| open_laptop | 50 | 10,412 | [-0.7834, -1.0504, 0.4611] | PASS |
| open_microwave | 50 | 24,333 | [-0.8234, -1.0518, 0.6074] | PASS |
| pick_diverse_bottles | 50 | 6,060 | [-0.7991, -0.9885, 0.5159] | PASS |
| pick_dual_bottles | 50 | 6,129 | [-0.8133, -0.9867, 0.5391] | PASS |
| place_a2b_left | 50 | 7,451 | [-0.8004, -1.0629, 0.5238] | PASS |
| place_a2b_right | 50 | 7,349 | [-0.8056, -1.0672, 0.4780] | PASS |
| place_bread_basket | 50 | 11,956 | [-0.8047, -1.0390, 0.4826] | PASS |
| place_bread_skillet | 50 | 8,277 | [-0.7939, -1.0416, 0.4794] | PASS |
| place_burger_fries | 50 | 12,046 | [-0.8034, -1.0839, 0.4629] | PASS |
| place_can_basket | 50 | 12,618 | [-0.6296, -1.0068, 0.6445] | PASS |
| place_cans_plasticbox | 50 | 14,375 | [-0.7848, -1.0908, 0.5373] | PASS |
| place_container_plate | 50 | 7,934 | [-0.7971, -1.0437, 0.5092] | PASS |
| place_dual_shoes | 50 | 11,510 | [-0.9987, -1.1721, 0.4348] | PASS |
| place_empty_cup | 50 | 8,617 | [-0.8085, -1.0458, 0.5154] | PASS |
| place_fan | 50 | 7,358 | [-0.8052, -1.0974, 0.6169] | PASS |
| place_mouse_pad | 50 | 7,531 | [-0.8037, -1.0663, 0.4769] | PASS |
| place_object_basket | 50 | 12,298 | [-0.6635, -1.0147, 0.6258] | PASS |
| place_object_scale | 50 | 7,266 | [-0.8024, -1.0420, 0.5507] | PASS |
| place_object_stand | 50 | 6,952 | [-0.7924, -1.0412, 0.5261] | PASS |
| place_phone_stand | 50 | 6,357 | [-0.8055, -1.0211, 0.4763] | PASS |
| place_shoe | 50 | 8,982 | [-0.5260, -1.1831, 0.5174] | PASS |
| press_stapler | 50 | 5,953 | [-0.7986, -1.0455, 0.5171] | PASS |
| put_bottles_dustbin | 50 | 31,231 | [-0.7804, -0.9858, 0.3335] | PASS |
| put_object_cabinet | 50 | 13,460 | [-0.7832, -1.0884, 0.4781] | PASS |
| rotate_qrcode | 50 | 7,724 | [-0.8001, -1.0678, 0.5232] | PASS |
| scan_object | 50 | 8,463 | [-0.6748, -1.0345, 0.6219] | PASS |
| shake_bottle | 50 | 12,436 | [-0.8092, -1.0058, 0.5162] | PASS |
| shake_bottle_horizontally | 50 | 13,911 | [-0.8111, -1.0062, 0.5087] | PASS |
| stack_blocks_three | 50 | 23,619 | [-0.7940, -1.0418, 0.4895] | PASS |
| stack_blocks_two | 50 | 15,647 | [-0.7941, -1.0424, 0.4837] | PASS |
| stack_bowls_three | 50 | 23,550 | [-0.8117, -1.0236, 0.5046] | PASS |
| stack_bowls_two | 50 | 15,637 | [-0.8079, -1.0243, 0.5372] | PASS |
| stamp_seal | 50 | 7,279 | [-0.8064, -1.0379, 0.5433] | PASS |
| turn_switch | 50 | 4,863 | [-0.8554, -1.1267, 0.4790] | PASS |

**汇总**: 50 子任务 × 50 episodes = 2,500 episodes, 共 **549,787 帧**, 全部 **PASS**

---

## 7. 关键文件路径

| 文件 | 路径 |
|------|------|
| 批处理脚本 | `GeoPredict/b/script/kpt/batch_extract_lrb3_kptsim7.py` |
| 关键点+姿态抽取器 | `GeoPredict/b/script/kpt/keypoint_pose_extractor.py` |
| FK 场景 | `GeoPredict/b/script/kpt/sapien_env.py` |
| TCP 位姿计算 | `GeoPredict/b/script/kpt/eef_calculator.py` |
| 配置常量 | `GeoPredict/b/script/kpt/config.py` |
| 坐标变换 | `GeoPredict/b/script/kpt/coord_transform.py` |
| 原始数据 | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/<task>/` |
| 输出数据 | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/<task>_lrb3_kptsim7/` |
| 批处理摘要 | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/_batch_lrb3_kptsim7_summary.json` |

---

## 8. 复现命令

```bash
# 1. 激活环境
conda activate RoboTwin

# 2. 进入 GeoPredict 目录
cd /home/luogang/SRC/Robot/GeoPredict

# 3. 全量处理（跳过已完成的）
python b/script/kpt/batch_extract_lrb3_kptsim7.py --skip_existing

# 4. 处理指定子任务
python b/script/kpt/batch_extract_lrb3_kptsim7.py --tasks scan_object hanging_mug

# 5. 从头处理全部（会覆盖已有输出）
python b/script/kpt/batch_extract_lrb3_kptsim7.py
```
