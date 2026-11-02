# TrackNet 适配器（统一观测格式）

把 TrackNet 的输出转换成统一 **Observation 逐帧观测**，供下游“轨迹/事件处理 → 共用计分状态机”使用。
本目录只负责 TrackNet 这一路。

```
TrackNet ──本适配器──┐
YOLO ────适配器──────┼→ 轨迹/事件处理 → 共用计分状态机 → 比分
TTNet ───适配器──────┘
```

## 1. 统一格式

字段定义来自《TTNet、TrackNet、YOLO 共用状态机方案》第 7.1 节，三路共用的契约文件统一放在**仓库根目录的 `contract/`**：

- `../contract/observation.schema.json` —— Observation 的 JSON Schema
- `../contract/event.schema.json` —— Event 的 JSON Schema
- `../contract/README.md` —— 契约说明与变更记录
- `observation.py` —— 构造 / 校验 / 读写（`make_observation`、`validate_observation`、`write_observations`）

一条观测（JSONL 每行一条）：

```json
{
  "frame_id": 1234,
  "timestamp_s": 10.283,
  "source": "tracknet",
  "image_size": [1920, 1080],
  "ball": {"visible": true, "center_px": [850.0, 420.0], "bbox_xyxy": null,
           "confidence": 0.749, "predicted": false},
  "event_scores": {"bounce": null, "net_hit": null}
}
```

约定：

- 坐标一律**还原到原始图像像素**（左上角原点，x 向右、y 向下）。适配器内部已完成缩放还原，下游不需要知道模型输入尺寸。
- 漏检帧输出 `visible=false` 且 `center_px/bbox_xyxy/confidence` 全为 `null` —— **不填上一帧位置，也不用 (0,0) 表示丢失**。轨迹预测属于下游模块，且必须标记为“预测”。
- `predicted=true` 表示该帧位置是**轨迹推断**出来的、不是模型看到的（方案文档 7.3 要求明确标记）。
  此时 `confidence` 必须为 `null`（没有检测就没有分数）。该字段**可选**，缺省视为 `false`。
- `timestamp_s` 是**被预测帧**的视频时间（三帧输入的最后一帧），不是推理完成时间。
- TrackNet 没有落台/擦网能力，`event_scores` 两个字段恒为 `null`；事件由下游轨迹层推断。

## 2. 字段映射（TrackNet）

| 统一字段 | TrackNet 来源 |
|---|---|
| `center_px` | 热图 → 阈值 → HoughCircles 圆心，再乘回原图比例 |
| `bbox_xyxy` | 无检测框，恒为 `null` |
| `confidence` | 检出中心 ±2 像素邻域的热图峰值 / 255（见下） |
| `predicted` | 普通推理恒为 `false`；只有从补帧 CSV（带 `interp` 列）转换时，补出来的帧标 `true` |
| `event_scores.bounce` / `net_hit` | 模型无此能力，恒为 `null` |

### confidence 的定义

TrackNet 是逐像素 256 类分类，argmax 后的类值就是该像素的高斯热图灰度（0~255）。
这里取“检出中心 ±2 像素窗口内热图最大值 / 255”，落在 [0, 1]。

它是**热图峰强度**，不是概率，**不能和 YOLO 的检测置信度直接比较**。

实测（test_2 前 600 帧、124 帧有真值）：`confidence ≥ 0.5` 的检出中位定位误差 **3.5 px**；
`confidence < 0.5` 的 12 帧中位误差 **20.0 px**（基本是误检）。下游可据此过滤，但适配器本身不过滤。

## 3. 用法

两种输入模式。**推荐直接跑视频**（能给出 confidence）：

```bash
# 视频模式：自己跑推理
python tracknet_adapter.py \
    --video /path/to/test_2.mp4 \
    --model_path ../TrackNet/TrackNet/exps/tt_g45_aug/model_best.pt \
    --out test_2.observations.jsonl \
    --min_radius 2 --max_radius 12        # 乒乓球；网球用 --max_radius 7

# CSV 模式：转换 track_video.py 的逐帧坐标（快，但没有热图，confidence=null）
python tracknet_adapter.py --csv track.csv --fps 120 --image_size 1920 1080 \
    --out test_2.observations.jsonl
```

CSV 若带 `interp` 列（`interpolate_track.py` 的补帧产物），补出来的帧会自动标成
`predicted=true` 且 `confidence=null`，下游不会把它当真实检出：

```bash
python tracknet_adapter.py --csv ../TrackNet/TrackNet/outputs/2026-10-21/csv/tt_g345_aug_test_2_filled.csv \
    --fps 120 --image_size 1920 1080 --out test_2_filled.observations.jsonl
# -> NOTE: 142 frames are trajectory-predicted (predicted=true, confidence=null)
# -> visible: 1245/3600 (34.6%)  [其中 142 帧为轨迹补帧]
```

两种模式产出的两份样例都在 `examples/`：`test_2_observations.jsonl`（纯模型）、
`test_2_observations_filled.jsonl`（含补帧标记）。

常用参数：`--max_frames`（截断）、`--batch_size`（默认 8）、`--no_amp`（关闭 fp16）。
速度：RTX 5060 上 1080p@120fps 视频约 **28 fps** 处理，3,600 帧约 2 分钟。

## 3.1 导出 MOT 标准格式（推荐，用于跨组对比）

JSONL 是我们自己的格式，信息全但不通用。用 `write_mot()` 可以再导一份
**MOT Challenge 格式**（10 字段，事实标准），直接对接 `py-motmetrics` / `supervision` / `trackers`：

```python
import observation as O
obs = O.load_observations("test_2.observations.jsonl")
O.write_mot("test_2.txt", obs)          # 漏检帧不写行；帧号自动转 1-based
```

然后用标准工具评估（不用自己实现匹配算法）：

```bash
python eval_mot.py --pred test_2.txt \
    --gt-json <OpenTTGames>/test_annotations/test_2_extracted/ball_markup.json \
    --max-frames 3600 --max-iou 0.3
```

TrackNet 只出球心没有框，导出时用「中心点 ± 半径」造框，默认半径 = 图像宽 × 1.2%
（1920 宽 → 23 px，直径 46 px，与实测球径 47 px 一致），可用 `ball_radius` 覆盖。

**补帧不会污染 MOT 指标**：`write_mot()` 默认跳过 `predicted=true` 的帧（MOT 格式评估的是
“模型看到了什么”，把推断位置写进去会虚高）。需要一并导出时显式传 `include_predicted=True`。

**注意口径**：`--max-iou 0.3` 在 46 px 的框上约等于中心误差 ≤ 14 px，比我们报告里用的
20 px 更严格，所以 Recall 会低一些（新权重 test_2：MOT 口径 0.6016 vs 自算 0.6637）。两套数字不要混用。

## 3.2 从 Observation 生成事件（Event）

适配器只输出**观测**，不输出事件；事件由下游从轨迹推出来。本分支已经实现了这一层，入口是
`TrackNet/TrackNet/events_from_observations.py`，它**只读统一 Observation**，不依赖任何模型私有文件：

```bash
cd TrackNet/TrackNet
python events_from_observations.py --obs ../../adapter/examples/test_2_observations.jsonl \
    --game test_2 --event_jsonl events.jsonl --fps 120
```

输出是方案文档 7.4 的 `Event`：

```json
{"event_id": "bounce_00000", "timestamp_s": 0.9, "type": "BOUNCE", "side": "B",
 "position_px": [1183.5, 553.5], "confidence": null, "sources": ["tracknet"]}
```

`side` 由台面四角单应性变换到俯视坐标后与球网比较得到，`A` 是 `far_left/near_left` 一侧。
规则法没有概率输出，所以 `confidence` 恒为 `null`，不填凑数。

**YOLO / TTNet 两路可以直接复用这一层**：只要你们的适配器输出合规的 Observation，事件逻辑不用重写，
唯一需要准备的是各自机位的球台四角（写进 `calib/table_quads.json`）。
`predicted=true` 的帧默认不参与事件判定。

## 4. 自查

```bash
python check_observations.py test_2.observations.jsonl --expect-frames 3600 --fps 120
python check_observations.py test_2.events.jsonl --fps 120     # 事件文件，自动识别
```

输出会给出：观测条数、帧是否连续、可见率、补帧数、置信度范围、时间戳一致性、以及 schema 错误列表。
事件文件则给出类型分布、半台分布、间隔统计。当前 TrackNet 结果（`tt_g345_aug`，test_2 前 3,600 帧）：
`PASS: schema OK`，1103/3600 可见（30.6%），与 `track_video.py` 的 CSV **坐标逐帧完全一致（0.00 px）**，
时间戳与 `frame_id/fps` 零偏差。

两份契约的机器可读定义在 `../contract/`（`observation.schema.json` 与 `event.schema.json`），
已用 `jsonschema` 校验过本身合法性，样例数据也全部通过。

## 5. 适用范围

本目录只负责 **TrackNet** 这一路，YOLO / TTNet 两路由其他成员负责，不在此实现。

`observation.py` 里的构造与校验函数与具体模型无关（`source` 只是枚举字段），
格式本身由 `../contract/observation.schema.json` 固定，不需要在本目录改动。

## 6. 业界同类方案（写代码前先看这里）

`RELATED_WORK.md` 记录了调研结论，简单说：

- **格式**：MOT Challenge 已经是事实标准，我们只做「导出」，不另立标准；
- **评估**：用 `py-motmetrics`（本目录 `eval_mot.py`），不自己写 P/R 匹配；
- **事件层**：抄 PlaynTrack 的三段式（Rally → Shot → Bounce）和默认参数；
- **状态机**：抄 jainr3/TableTennisTracker 的 FSM（MIT 许可，状态定义直接可用）；
- **可视化/多目标**：需要时用 `supervision` + `roboflow/trackers`，不要自己写追踪。

另外提醒一个指标陷阱：TTNet 论文的「97% 事件准确率」是**给定窗口内的分类准确率**，
不是端到端事件检测 F1，和我们的数字不能直接对比。

