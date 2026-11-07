# 实时层简化版：用视频文件模拟摄像头

> 给执行的 AI：本文件是完整任务说明。工作目录 `E:\20-项目\ppq\TN`（三个仓库并排在 `E:\20-项目\ppq` 下）。
> 先读 `cascade/README.md`、`cascade/run_cascade.py`、`cascade/fuse_obs.py`、`cascade/ttnet_reviewer.py`，再动手。

## 0. 目标与范围

把现在的"预先算好结果 → 逐帧回放"变成**真正并行的流式运行**：视频文件按原始帧率吐帧，当作摄像头；YOLO、TrackNet、TTNet 在各自的线程里跑，结果在线汇合，判分边跑边出。

架构按级联方案的图：

```text
VideoFileCamera（按 fps 吐帧，模拟摄像头）
   │ (session_id, frame_id, timestamp_s, BGR)
   ├──→ FrameRing（最近 N 秒、TTNet 尺寸的帧）◄───────────────┐ 读取候选区间
   ├──→ YOLO worker ──────┐                                    │
   └──→ TrackNet worker ──┴→ 按帧号汇合 → fuse → 提议 → 区间调度 → TTNet worker（有界队列）
                                                                  ↓ ReviewResult
                                              事件验证、去重、按时间提交 → 共用状态机
```

**这次做**：

- 帧源抽象：先实现视频文件版，接口留给以后换真摄像头。
- 原始帧环形缓存。
- 三个模型各一个 worker 线程。
- TTNet 用有界任务队列。
- 两种节奏：不丢帧、按真实时间丢帧。
- 运行统计：每个阶段的帧率、丢帧数、队列深度、端到端延迟。

**这次不做**：

- 真实摄像头驱动；
- TensorRT 或其他推理加速；
- 多卡或嵌入式板部署；
- 进程间通信。

这些等接入硬件时再做。

## 1. 硬约束（违反任何一条都算没完成）

1. **只改 `cascade/`**。不改 `tt/`、`adapter/`、`TrackNet/`、`third_party/`，也不改另外三个仓库（`TTNet`、`TTNet1.1`、`pingpong-smart-referee`）。需要的东西用 import 引用，不复制代码。
2. **不重装、不升级 torch**；`pip install ultralytics` 必须加 `--no-deps`。
3. **回放结果不能变**：重构完成后，`python -m unittest discover -s cascade/tests -t .` 全部通过；`python -m cascade.run_batch --manifest cascade/batch_manifest.json --out-dir outputs/cascade_batch_check` 生成的 `summary.md` 必须与 `outputs/cascade_batch/summary.md` 逐字相同。
4. **超时和持续条件一律按秒**（共用方案 §4.1），不按帧数。
5. **git**：
   - 作者和提交者都是 `Yao-MY06 <Yao-MY06@users.noreply.github.com>`，不加任何 Co-Authored-By 或 Claude 字样。
   - 在 `fusion/three-way` 上提交，再 cherry-pick 到 `feat/cascade`。
   - 不 push。
6. **这台电脑没有 GPU、没有 torch**（Python 3.10 + numpy/cv2）。所以每个 worker 都要有"回放模式"：读已有的 JSONL 或缓存，并按设定耗时 sleep 来模拟推理。这样所有测试在没有 GPU 的机器上也能跑。真模型只在原机上验收。
7. 不要和另一个 AI 同时改这个目录。

## 2. 任务

### R1. 把主循环抽成可复用的核心（不改行为）

`run_cascade.run()` 里的主循环目前是"同步复核 + 按时间模拟复核延迟"。要把它拆成一个类，让回放和实时共用同一套逻辑：

```python
class CascadeCore:
    def __init__(self, poly, calib, reviewer_name, params, commit, merge, max_clip_s,
                 pre_ctx, post_ctx, fps, session_id): ...
    def on_frame(self, obs: dict, yolo_obs: Optional[dict]) -> List[Clip]
        # 已融合的一帧 → 提议 → 调度；返回本帧关闭、需要复核的区间
    def on_result(self, clip_id: str, result: ReviewResult, now_t: float) -> None
        # 复核结果到达（可能乱序）
    def on_timeout(self, clip_id: str, now_t: float) -> None
    def advance(self, now_t: float) -> None      # 推进水位线
    def finish(self) -> Dict[str, Any]           # 返回和现在 run() 一样的 res
```

- `run()` 改为调用 `CascadeCore`，复核仍然同步执行，用 `review_latency_s` / `review_deadline_s` 模拟耗时和截止，**行为保持不变**（硬约束 3）。
- 水位线 `bounce_unknown_before()` 的语义不能变：还在 TTNet 队列里、或正在推理的区间，都要压住水位线；超时的区间不再压。

### R2. 帧源与环形缓存（`cascade/realtime/frames.py`）

- `Camera` 抽象：`start()`、`stop()`、`frames()`。`frames()` 逐帧产出 `(frame_id, timestamp_s, bgr)`。时间戳由采集端生成，所有阶段共用同一套 frame_id 和时间戳。
- `VideoFileCamera(path, pace="realtime" | "all", speed=1.0)`：
  - `realtime`：按 `frame_id / fps / speed` 的墙钟时间吐帧，下游跟不上不等待。
  - `all`：等下游消费完再吐下一帧（背压），不丢帧。
- `NullCamera(n_frames, fps)`：只产出帧号和时间戳，`bgr=None`，给回放模式和测试用。
- `FrameRing(capacity_s=3.0, fps)`：线程安全；按 frame_id 存 **TTNet 输入尺寸**的 RGB 帧（320×128，与 `ttnet_engine` 的预处理一致）。
  - 不存原图：原图 3 秒约 2 GB，太大。
  - `get(frame_id)` 在帧已被淘汰时返回 `None`。
  - 统计 `evicted_before_read` 次数。

### R3. YOLO 与 TrackNet worker（`cascade/realtime/workers.py`）

每个 worker 是一个线程：从自己的输入队列取帧，把 `(frame_id, Observation dict)` 放进输出队列。

- **队列**：输入队列有上限。`realtime` 节奏下队列满了就丢最旧的帧，被丢的帧输出"未处理"的不可见观测，并计数 `dropped`。`all` 节奏下阻塞等待。
- **YoloWorker**：
  - 真模型：用 `pingpong-smart-referee/adapters/yolo_adapter.py` 的 `predict_frame`。构造参数（preset、ROI、阈值）照 `pingpong-smart-referee/run_referee.py --preset test2` 的写法。
  - 回放：读 `_data/obs/yolo/test_2.jsonl`，每帧 sleep `--yolo-ms`。
- **TrackNetWorker**：
  - 真模型：流式版 TrackNet，**从 `adapter/tracknet_adapter.py` 导入** `BallTrackerNet`、`postprocess`、`_heatmap_peak`、`MODEL_W/H`、`SPACE_SCALE`、`make_observation`，不复制。
  - 三帧输入顺序必须和 `run_video` 一致：由新到旧 `(t, t-1, t-2)`；前两帧输出不可见；`min_conf=0.5`；坐标换算一致。
  - 允许小批量：最多 4 帧，最多等 10 ms。
  - 回放：读 TrackNet 观测 JSONL，每帧 sleep `--tracknet-ms`。
- **汇合**：主线程按 frame_id 汇合两路观测，然后调用 `fuse_obs.fuse`。
  - 需要把 `fuse` 改成**逐帧在线版**，或包一层在线适配。它的连续性门限要看之后 `gate_s` 秒，所以帧要晚 `gate_s` 才能送出。
  - 在线结果必须和离线 `fuse` 逐帧一致，并写单测。
  - 某一路超过 `--join-timeout-s`（默认 0.5 s）还没给出该帧，就按"该模型此帧不可见"处理，并计数。

### R4. TTNet worker 与有界任务队列

- 区间关闭后进入 TTNet 队列，队列容量 `--ttnet-queue` 默认 8 个区间。
  - **队列满时**：新区间直接生成 `status="timeout"`、`reason="queue_full"` 的结果，进人工确认清单（复用 `review_queue.py`）。不能阻塞主线程，也不能悄悄丢掉。
  - **截止时间**：`--review-deadline-s` 按墙钟计时；超时规则与现在一致，迟到的结果只写 `late_results.jsonl`。
- 推理方式：
  - 真模型：`LiveScores(engine, frame_source=ring.get)`。如果区间要用的帧已被淘汰，返回 `error` 和 `missing_context`。
  - 回放：`ReplayScores` 读 `_data/ttnet/test_2.jsonl`，每个窗口 sleep `--ttnet-ms-per-window`（默认 14）。
- 复核结果通过线程安全队列交回主线程，调用 `CascadeCore.on_result`。

### R5. 命令行与统计（`cascade/run_realtime.py`）

```bash
# 这台电脑（无 GPU）：全回放模式
python -m cascade.run_realtime --null-camera 3600 --fps 120 --pace all \
    --yolo-replay _data/obs/yolo/test_2.jsonl --tracknet-replay adapter/examples/test_2_observations.jsonl \
    --ttnet-cache _data/ttnet/test_2.jsonl --reviewer ttnet --preset test2 --out-dir outputs/realtime/replay_all

# 原机：真模型 + 视频文件
python -m cascade.run_realtime --video <test_2.mp4> --pace realtime \
    --yolo-weights <...> --tracknet-weights <...> --ttnet-weights <...> --ttnet-src ../TTNet/src \
    --reviewer ttnet --preset test2 --out-dir outputs/realtime/a_realtime
```

- 输出文件和 `run_cascade --out-dir` 完全一样，额外加一个 `realtime_stats.json`，内容包括：
  - 每个阶段：处理帧数、丢帧数、实际帧率、单帧耗时 p50 / p95；
  - 队列深度最大值；
  - TTNet：`queue_full` 次数、`missing_context` 次数、窗口数；
  - 环形缓存的 `evicted_before_read` 次数；
  - **端到端延迟**：帧采集 → 判罚输出，p50 / p95 / max；
  - 总墙钟时间。
- 运行时每秒打印一行进度：各阶段帧率和队列深度。

### R6. 测试（`cascade/tests/test_realtime.py`，没有 GPU 也要能跑）

1. `FrameRing`：容量按秒计算、淘汰后 `get` 返回 `None`、多线程读写不出错。
2. 在线版 `fuse` 和离线版逐帧一致（test_2 真数据）。
3. **一致性**：`--pace all`、全回放、sleep 设为 0 时，`decisions`、`events.jsonl`、比分与 `run_cascade` 相同参数的回放**完全一致**。
4. TTNet 队列满：容量设 1、每窗耗时设得很大，应出现 `queue_full`，相应区间进人工确认清单，程序正常结束。
5. 截止时间：设一个小于复核耗时的截止，应生成 `deadline_exceeded`，且迟到结果不进状态机。
6. `realtime` 节奏下 TrackNet 跟不上（例如 `--tracknet-ms 42`）：丢帧数 > 0、程序不卡死、能出比分。

## 3. 验收

在这台电脑上：

- 硬约束 3 通过：原有测试全过，批量汇总逐字相同。
- R6 的测试全部通过。
- 回放一致性（R6-3）通过：test_2 比分 0:2，落台 F1 与回放相同（规则复核 0.903，TTNet 复核 0.871）。
- 用 `--pace realtime --yolo-ms 5 --tracknet-ms 42 --ttnet-ms-per-window 14` 模拟原机速度跑一次 test_2，把 `realtime_stats.json` 的要点写进回执：丢帧比例、端到端延迟、落台 F1、比分。

在原机上（给出命令和预期，由用户转交）：

- `--pace all` 用真模型跑 test_2：落台 F1 和比分应与回放接近。流式 TrackNet 与批量版因 AMP 可能有极小差异，需要列出差异帧数。
- `--pace realtime` 用真模型跑 test_2：如实报告每个阶段的帧率、丢帧比例、端到端延迟，以及此时的落台 F1 和比分。**不要为了"跑得上"改任何阈值。**

## 4. 交付

1. 代码：
   - `cascade/realtime/__init__.py`、`frames.py`、`workers.py`
   - `cascade/run_realtime.py`
   - `cascade/tests/test_realtime.py`
   - `run_cascade.py` 的重构
   - `cascade/README.md` 新增"实时层（简化版）"一节
2. 提交：按任务拆成几个 commit，作者见硬约束 5，两个分支都要有。
3. 回执 `outputs/realtime/_回执.md`，内容包括：
   - 每项任务完成情况；
   - 测试输出；
   - 回放一致性结果；
   - 模拟原机速度那次运行的统计；
   - 给原机的运行命令；
   - 遇到的问题，以及没做或做不到的地方，照实写。
