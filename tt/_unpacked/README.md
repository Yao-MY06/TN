# 乒乓球视觉智能裁判 · 裁判核心与计分状态机交付包

本项目为 602Lab 乒乓球视觉智能裁判系统的**共用裁判核心层（Core Engine）**。依据组长下发的权威方案《TTNet、TrackNet、YOLO 共用计分状态机方案》实现。

本交付件负责接收各模型（TrackNet / YOLO / TTNet）适配器导出的**统一逐帧观测（JSONL）**，自动执行轨迹平滑滤波、物理事件判定、落点单应性解算与 6 状态机自动计分。

---

## 目录结构

```text
乒乓球裁判状态机_交付件/
├── core/                         # 裁判核心引擎
│   ├── schemas.py                # 统一数据契约（FrameObservation / RefereeEvent）
│   ├── state_machine.py          # 乒乓球 6 状态计分状态机（含 5 条判罚规则）
│   ├── trajectory_processor.py   # 轨迹处理与物理事件提取（卡尔曼平滑/击球反折/落台检测）
│   ├── geometry.py               # 球台几何与机位标定预设（test2 / game2 等）
│   ├── homography.py             # 球台单应性变换与 2D 物理落点解算
│   └── base_adapter.py           # 适配器抽象基类
├── visualizer/                   # 选配：转播级比分渲染看板
│   └── broadcast_hud.py          # 左上角计分牌 + 轨迹尾迹 + 2D 俯视落点雷达
├── examples/                     # 样例数据
│   └── sample_test2_video_600f.jsonl # TrackNet 产出的 600 帧真实观测样例
├── run_state_machine.py          # 交付运行入口脚本
├── requirements.txt              # 极简运行依赖
└── README.md                     # 本说明文档
```

---

## 环境要求

本核心模块针对工程化交付做了严格解耦，**完全不依赖 PyTorch / CUDA 等重型模型框架**，在任何常规 Python 环境下均可飞速运行（上万帧每秒）。

### 安装依赖
```bash
pip install -r requirements.txt
```
*(仅依赖 `numpy>=1.20.0` 和 `opencv-python>=4.5.0`)*

---

## 快速使用

### 1. 纯命令行极速评测（出分与判罚日志）
将你用 TrackNet 导出的 `.jsonl` 观测文件传入：

```bash
python run_state_machine.py --jsonl examples/sample_test2_video_600f.jsonl --preset test2
```

**预期终端输出：**
```text
========================================================================
【乒乓球智能裁判】开始评测
输入观测文件: sample_test2_video_600f.jsonl
球台几何预设: test2 (球网X像素位置: 929)
========================================================================
  [帧 0070 |  0.583s] 事件: RACKET_HIT  半台: A       ID: racket_hit_00001
  [帧 0111 |  0.925s] 事件: BOUNCE      半台: B       ID: bounce_00001
  [帧 0111 |  0.925s] >> 判罚: P2(B) 得分 [SERVE_FAULT] 球未落发球(A)半台落在 B 0:1
  [帧 0147 |  1.225s] 事件: RACKET_HIT  半台: B       ID: racket_hit_00002
  [帧 0179 |  1.492s] 事件: BOUNCE      半台: A       ID: bounce_00002
  ...
========================================================================
评测完成！
处理观测总帧数 : 600 帧
状态机处理用时 : 0.041 秒 (吞吐速度: 14634.1 FPS)
提取物理事件统计: {'RACKET_HIT': 9, 'BOUNCE': 6}
最终判定比分   : 选手P1 (半台A) 0 : 1 选手P2 (半台B)
========================================================================
```

---

### 2. 同步渲染带转播级 HUD 比分板的视频（可选）
如果需要向组长或展示汇报，传入原视频路径 `--video` 和输出路径 `--output`，系统会自动将实时比分板、轨迹尾迹和落点雷达叠加到视频中：

```bash
python run_state_machine.py \
    --jsonl examples/sample_test2_video_600f.jsonl \
    --preset test2 \
    --video path/to/test_2.mp4 \
    --output output_referee_test2.mp4
```

---

## 输入数据契约说明（组长方案 §7.1）

输入的 `.jsonl` 文件每行必须为一个合法的 JSON 对象，格式如下：

```json
{
  "frame_id": 1234,
  "timestamp_s": 10.283,
  "source": "tracknet",
  "image_size": [1920, 1080],
  "ball": {
    "visible": true,
    "center_px": [850.0, 420.0],
    "bbox_xyxy": null,
    "confidence": 0.749
  },
  "event_scores": {
    "bounce": null,
    "net_hit": null
  }
}
```

### 硬性规则说明：
1. **漏检不得补位**：当 `visible=false` 时，`center_px`、`bbox_xyxy`、`confidence` 必须全部为 `null`（严禁沿用上一帧，严禁使用 `(0,0)` 占位，否则会引起轨迹断裂判定失效）。
2. **坐标原图还原**：`center_px` 必须还原为原视频分辨率（如 1920×1080）下的像素坐标。
3. **时间戳定义**：`timestamp_s = frame_id / fps`，以被预测帧为基准。

---

## 常用参数表

| 参数名 | 说明 | 缺省值 / 示例 |
| :--- | :--- | :--- |
| `--jsonl` | 必填，输入的统一观测 JSONL 文件 | `--jsonl your_obs.jsonl` |
| `--preset` | 球台几何与机位标定预设 | 默认 `test2`。支持 `test2`、`game2`、`rally`、`plos` |
| `--video` | 可选，原始视频文件路径 | `--video data/test_2.mp4` |
| `--output` | 可选，渲染后保存的视频路径 | `--output out.mp4` |
