# 统一观测契约（Observation / Event）

> 版本：**v0.1**　｜　生效日期：2026-11-02　｜　状态：已冻结，变更需走第 7 节流程
>
> 机器可读定义：`observation.schema.json`、`event.schema.json`（JSON Schema draft-07）
>
> 上游依据：《TTNet、TrackNet、YOLO 共用状态机方案》第 7 节（`docs/otherdata/`）

这个目录是**三路模型唯一的对接面**。三个模型、三个仓库、三个人，只要都按这两份结构产出数据，下游的轨迹层、事件层、判罚状态机就可以互换使用，不需要知道对方模型长什么样。

---

## 1. 数据流

```text
TTNet   ──┐
TrackNet ─┼→ Observation（逐帧，§2）→ 轨迹/事件层 → Event（事件，§3）→ 计分状态机 → 判罚
YOLO    ──┘
```

两层的边界要划清楚：

| 层 | 产出 | 谁产出 | 回答什么问题 |
|---|---|---|---|
| 观测层 | `Observation` | **各模型适配器** | 这一帧球在哪、看没看见 |
| 事件层 | `Event` | 轨迹/事件处理 | 这一拍发生了什么（落台/击球/擦网/出界） |

**适配器不产出 Event。** 模型自带的落台/擦网能力只能作为 `event_scores` 里的候选分数，进不了 `Event`——候选分数要经时间峰值提取、几何验证、去重之后，才算一个事件。

---

## 2. Observation：逐帧观测

一条观测对应**一帧**，JSONL 每行一条。漏检的帧也必须输出一条。

| 字段 | 类型 | 必填 | 约束 |
|---|---|---|---|
| `frame_id` | int | ✅ | ≥ 0，**从 0 开始**，对应被预测帧 |
| `timestamp_s` | number | ✅ | ≥ 0，**被预测帧**的视频时间，不是推理完成时间 |
| `source` | string | ✅ | `yolo` / `tracknet` / `ttnet` |
| `image_size` | [w, h] | ✅ | **原始**图像尺寸 |
| `ball.visible` | bool | ✅ | 该帧是否检测到球 |
| `ball.center_px` | [x, y] \| null | ✅ | 球心，**原始图像像素**；`visible=false` 时必须 null |
| `ball.bbox_xyxy` | [x1,y1,x2,y2] \| null | ✅ | 检测框；无框的模型（TrackNet）填 null |
| `ball.confidence` | number \| null | ✅ | 0~1，模型自身分数；无可靠分数填 null |
| `ball.predicted` | bool | ⬜ 可选 | true = 该点由轨迹推断，非模型检出；缺省 false |
| `event_scores.bounce` | number \| null | ✅ | 模型自带落台候选分数；无此能力填 null |
| `event_scores.net_hit` | number \| null | ✅ | 模型自带擦网候选分数；无此能力填 null |

### 三条硬规则

**① 坐标一律还原到原始图像。** 左上角为原点，x 向右、y 向下。适配器负责撤销缩放、裁剪、letterbox 补边，下游不需要知道模型输入尺寸。

**② 漏检就写漏检。** `visible=false` 时 `center_px / bbox_xyxy / confidence` 三个字段**全部为 null**。
不填上一帧位置，不用 `(0, 0)` 表示丢失。轨迹补帧是下游的事，而且必须用 `predicted=true` 标出来。

**③ 时间戳属于被预测帧。** 三帧输入预测中间帧，时间戳就是中间帧的时间。一次输出多帧位置，就拆成多条观测。

### 分路映射

| 统一字段 | YOLO | TrackNet | TTNet |
|---|---|---|---|
| `center_px` | 检测框中心 | 热图峰值 / 连通域中心 | 球位置输出 |
| `bbox_xyxy` | 检测框 | null | 无框则 null |
| `confidence` | 检测置信度 | 热图峰值强度（见下） | 对应球检测分数 |
| `event_scores` | null | 全 null | 落台/擦网分数 |

> **`confidence` 不可跨模型比较。** TrackNet 填的是"中心 ±2px 邻域的热图峰值 / 255"，不是概率；YOLO 填的是分类置信度。两者量纲不同，任何跨模型阈值都必须各自标定。

---

## 3. Event：事件

| 字段 | 类型 | 必填 | 约束 |
|---|---|---|---|
| `event_id` | string | ✅ | 唯一，建议 `<type小写>_<5位序号>`，如 `bounce_00042` |
| `timestamp_s` | number | ✅ | 事件发生的视频时间 |
| `type` | string | ✅ | `BOUNCE` / `RACKET_HIT` / `NET_TOUCH` / `OUT` |
| `side` | string | ✅ | `A` / `B` / `UNKNOWN` |
| `position_px` | [x, y] \| null | ✅ | 原始图像像素；无位置信息填 null |
| `confidence` | number \| null | ✅ | 规则法没有概率输出时填 **null**，不要填凑数的值 |
| `sources` | string[] | ✅ | 产生该事件的观测来源，如 `["tracknet"]` |

### 事件枚举的四条约定

**① `side` 的 A/B 是球台的物理左/右半台，不是发球方/接发方。**
A = `far_left` / `near_left` 所在侧（俯视时图像左侧）。判定靠四角单应性把落点投到俯视坐标，再用 `net_x` 分界。

> ⚠️ **换机位必须重新标定四角**，否则 A/B 会整体翻转而且不报错。四角写在 `calib/table_quads.json`。

**② 事件和模型分数是两套名字，不要混用。**
`event_scores.bounce` ↔ `BOUNCE`，`event_scores.net_hit` ↔ `NET_TOUCH`。观测层没有 `OUT` / `RACKET_HIT` 的分数槽——模型不直接给这两种分，只能由事件层从轨迹推断。

**③ `LET`（擦网重发）与"谁得分"不是物理事件，不进本枚举。** 它们是状态机的输出。

**④ `confidence=null` 是合法值，不要为了"看起来完整"而填 0 或 0.5。** 下游有义务区分"没有分数"和"分数很低"——这正是当前最大的一个对接缺陷，见第 6 节。

---

## 4. 校验

```bash
# 单份观测：帧连续性、可见率、补帧数、时间戳一致性、schema 合法性
python adapter/check_observations.py adapter/examples/test_2_observations.jsonl \
       --expect-frames 3600 --fps 120

# 事件文件：类型分布、半台分布、事件间隔
python adapter/check_observations.py adapter/examples/test_2_events.jsonl --fps 120
```

两份 schema 自身也用 `jsonschema` 校验过，样例数据全部通过。

---

## 5. 变更记录

| 日期 | 变更 | 影响面 |
|---|---|---|
| 2026-10-22 | `frame_id` 统一从 **0** 开始；导出 MOT 时 +1 转 1-based，对外仍 0-based | 三路 |
| 2026-10-22 | `side` 的 A/B 明确为**物理左/右半台**，A = 左侧 | 三路 |
| 2026-10-22 | 字段命名统一按方案文档 §7.1；§2 的 `ball_xy` 扁平写法**作废** | 三路 |
| 2026-10-22 | 事件枚举定为 `BOUNCE / RACKET_HIT / NET_TOUCH / OUT`（裸 `HIT` / `NET` 有歧义，且 `net_hit` 已被观测层占用） | 三路 |
| 2026-11-02 | `ball.predicted` 作为**可选**字段写入 schema。依据是方案文档 7.3 要求"明确标记预测位置"但没给字段名 | 三路 |

---

## 6. 已知契约级问题（待上游确认）

这三条不是实现 bug，是**契约本身没说清或下游理解不一致**，需要在组内定调。

| # | 问题 | 后果 | 位置 |
|---|---|---|---|
| 1 | `confidence` 允许 null，但下游把它当成 `0.35` 的硬门限 | CSV 路线（无热图 → confidence 全 null）跑全片 **零事件**；填 0.9 立刻出 232 条落台 | `core/trajectory_processor.py:459` |
| 2 | `ball.predicted` 被 `from_dict` **静默丢弃**，补帧点被当成真实检测用 | 轨迹外推点参与事件判定，无法区分"看到的"和"猜的" | 状态机 `FrameObservation.from_dict` |
| 3 | 观测层**没有击球分数槽**，`RACKET_HIT` 只能靠轨迹层推 | 击球瞬间球被球员遮挡，自测召回不达标，只能依赖状态机内置轨迹层 | 契约设计 |

第 1 条的临时处置：跑真实推理（有热图）而不是读 CSV。根本修法要下游改判空语义。

---

## 7. 改契约的流程

1. 先在群里说明改哪个字段、为什么、影响哪几路；
2. 三方**口头同意后**才动 schema；
3. 动 schema 必须同时更新：`observation.schema.json` / `event.schema.json`、本文件第 2~5 节、以及 `adapter/observation.py` 的构造函数；
4. 提交信息带 `contract:` 前缀，例如 `contract: 增补 ball.predicted 可选字段`；
5. 破坏性变更（删字段、改语义、改枚举）必须升 **minor 版本**，并在第 5 节写清迁移方式。

**不要各改各的。** 三个仓库的代码可以不同步，契约必须同步——这是建这个仓库的唯一理由。
