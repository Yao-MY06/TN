# 同类方案调研：哪些轮子已经有了

> 目的：在做「统一观测格式 + 事件层 + 共用状态机」之前，先确认业界是否已有现成实现。
> 结论：**统一格式这件事已经有事实标准（MOT Challenge），直接对齐即可；事件/计分这一层
> 有几个开源项目可以抄结构，但精度都不够直接用。**
> 调研时间：2026-10-20

---

## 一、可以直接用的（不用自己写）

### 1. MOT Challenge 格式 —— 轨迹交换的事实标准

每行 10 个逗号分隔字段，**帧号 1-based**：

```
frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
1, 1, 794.2, 47.5, 71.2, 174.8, 1.0, -1, -1, -1
```

- 2D 场景 x/y/z 一律填 `-1`（这是官方写法，不是缺失值）
- 出处：MOTChallenge 2015, arXiv:1504.01942, Table II
- 生态：`py-motmetrics`（官方评估，出 MOTA / IDF1 / MOTP）、`supervision`、`trackers`、
  `motrackers` 都直接吃这个格式

**对我们的意义**：不用自己定义 Precision/Recall 的口径，也不用自己实现匹配算法。
队友的 YOLO、学长的 TTNet 只要各导出同样一份 `<seq>.txt`，指标天然可比。

### 2. Supervision（roboflow，38k stars）—— 若要做可视化

`sv.Detections` 是检测结果的统一容器（`xyxy` / `confidence` / `class_id` / `tracker_id`），
提供 20+ annotator（含 `TraceAnnotator` 拖尾）和 ByteTrack。

**对我们的意义**：`track_video.py` 里的红尾绘制如果要做成可交互/多目标版本，
可以换成 supervision 的 annotator，省掉自己写绘制逻辑。当前单球场景不是必须。

### 3. trackers（roboflow，Apache 2.0）—— 若要做多目标追踪

SORT / ByteTrack / OC-SORT / BoT-SORT / C-BIoU / McByte 的 clean-room 实现，
统一 `update(detections, frame)` 接口，原生吃 `supervision.Detections`，
在 MOT17 / SportsMOT / SoccerNet 上有基准数据。

**对我们的意义**：现在球是单目标，用不上；如果以后要同时追踪球和两名球员，
直接用这个而不是自己写卡尔曼关联。注意 BoxMOT 是 AGPL-3.0，商用要避开。

---

## 二、结构可以抄的（同题项目，但精度不够）

### 4. PlaynTrack（Rogit-28，2025）—— **和学长方案最接近的一个**

用 OpenTTGames 做乒乓球分析：YOLOv8 检测 + Kalman 平滑 + 事件分段 + 统计 + Gradio 面板 + CSV 导出。

```
VideoLoader → BallDetector + PlayerDetector
            → KalmanTracker
            → EventSegmenter → RallyDetector / ShotDetector / BounceDetector
            → compute_match_stats
            → AR Visualizer / Dashboard / CSV Export
```

**可直接抄的**（`src/segmentation/` 的模块划分）：
- `event_types.py` / `event_segmenter.py` / `rally_detector.py` / `shot_detector.py` / `bounce_detector.py`
- 默认参数（`config.yaml`，可当我们的起点）：
  - `rally_gap_seconds: 2.0` —— 超过 2 秒没球就切分回合
  - `min_shots_per_rally: 2` —— 少于 2 拍不算一个回合
  - `shot_angle_threshold_deg: 90.0` —— 击球判定的方向反转角度
  - `max_missed_frames: 10` —— 允许连续丢帧数

**不能抄的**：它的检测和事件都是规则/阈值实现的，没有定量指标，追踪画质一般。
我们的 TrackNet 在定位精度上是明显更强的。

### 5. TableTennisTracker（jainr3，MIT 许可）—— 状态机可以直接照搬

有完整的乒乓球**计分状态机**（README 里那张 fsm.png）：
发球（球被抛到一定高度 + 提示音）→ 发球方击球 → 检测 bounce / hit / timeout / 换边 → 判分。

**对我们的意义**：学长方案里"共用状态机"这一层，**状态名和转移条件照这个抄就行**，
不用重新设计。MIT 许可，可以直接移植代码。
它的短板是检测用颜色阈值（橙色球 + HSV 掩码），换场馆就崩，所以只能抄逻辑不能抄视觉。

### 6. Table-Tennis-Umpire（MatthewDulcich）—— 事件二分类的现成做法

同样用 OpenTTGames，YOLOv5nu 检球 + 两个 Keras 模型（`ball_tracker_model.keras` 出位置、
`ball_event_model.keras` 出事件），支持摄像头和视频，标注 bounce / net hit / empty。

**对我们的意义**：验证了「位置和事件用两个模型分开做」是可行的路线，
也是我们 `event_scores` 字段填 `null` 时的过渡方案（TrackNet 不出事件，另起一个轻量事件模型）。

### 7. Ping-Pong-Judge / PingPongCourt —— 多类别检测的思路

YOLOv8 同时检测 ball / hand / table / net，据此判 bounce / double bounce / floor hit 并计分。
**对我们的意义**：如果只靠球轨迹判 bounce 误检多，可以引入 table/net 的检测结果做约束
（球必须落在台面多边形内才算 bounce）——这条路比纯轨迹阈值稳。

---

## 三、一个必须澄清的指标问题

TTNet 论文里写的「事件识别 97.0% 准确率」，**不是端到端事件检测的 F1**，
而是在**已经给定事件窗口**的前提下，判断这个窗口属于 bounce / net hit / empty 的分类准确率。
球定位的 2 px RMSE 同理，是在已检测到的帧上算的。

所以我们和 TTNet 比时，如果要用「97%」这个数，必须说明口径；
真正的端到端比较应该用同一套（事件级 P/R/F1 + 时序容差），或者直接对齐它的官方评估方式。

---

## 四、已经落地的改动（本次）

### 统一格式 → MOT 导出

`observation.py` 新增：

| 函数 | 作用 |
|---|---|
| `obs_to_mot(obs, track_id, ball_radius)` | 一条 Observation → MOT 十元组；漏检返回 None |
| `write_mot(path, observations, ...)` | 批量导出 `<seq>.txt` |
| `mot_file_for(seq, dir)` | 按 MOT 约定拼文件名 |

球没有 bbox（TrackNet 只出中心点）时，用**中心点 ± 固定半径**造框，
默认半径 = 图像宽 × 1.2%（1920 宽 → 23 px，直径 46 px，与实测球径 47 px 一致）。

### 用标准工具评估

`eval_mot.py`：把 OpenTTGames 的 `ball_markup.json` 转成 GT，用 `py-motmetrics` 算
MOTA / IDF1 / MOTP / Precision / Recall，**不再自己实现匹配**。

实测（test_2 前 3600 帧，只在 779 个标注帧上评估，max_iou=0.3）：

| 指标 | 值 | 说明 |
|---|---|---|
| 标注帧 | 779（其中 773 有球） | 事件中心标注，全片覆盖约 10% |
| Precision | 0.8358 | 比自算的 0.9638 低——IoU 0.3 约等于中心误差 14 px，比 20 px 严 |
| Recall | 0.5071 | 自算 20 px 口径是 0.5802 |
| MOTA | 0.4075 | 标准定义，误检直接扣分 |
| IDF1 | 0.6312 | 单目标场景下≈轨迹连续性 |
| MOTP | 0.1567 | 平均 1−IoU，越小越好 |

**两个口径不要混用**：报告里给学长看的是 20 px 自算口径；对外和队友横向对比时用 MOT 口径
（因为它有公开定义、别人能复现）。

> 小坑：`motmetrics 1.4.0` 还在用 `np.asfarray`，numpy 2.x 已移除。
> 不要为此降级 numpy（会和 torch 冲突），`eval_mot.py` 里补了一个等价 shim。

---

## 五、对学长方案的修正建议

| 学长的设计 | 建议 | 理由 |
|---|---|---|
| 自定义 Observation JSON | **保留**，但**同时**能导 MOT txt | JSONL 信息更全（可见性/事件分数/时间戳），MOT 是交换格式，两者互补 |
| 自己实现事件层 | 抄 PlaynTrack 的三段式（Rally→Shot→Bounce）+ 它的默认参数 | 结构已经验证过，省一轮设计 |
| 自己设计状态机 | 抄 jainr3 的 FSM 状态与转移（MIT） | 状态定义是领域知识，重写容易漏边界情况 |
| 自己算 Precision/Recall | 用 py-motmetrics | 定义公开、可复现、能和别人比 |

---

## 六、许可证提醒

| 项目 | 许可 | 能不能直接用 |
|---|---|---|
| MOT Challenge 格式 | 格式规范，无版权限制 | ✅ |
| py-motmetrics | MIT | ✅ |
| supervision | MIT | ✅ |
| trackers (roboflow) | Apache 2.0 | ✅ |
| jainr3/TableTennisTracker | MIT | ✅ 可移植状态机 |
| PlaynTrack | "educational and research purposes" | ⚠️ 只抄结构，别直接搬代码 |
| BoxMOT | AGPL-3.0 | ❌ 商用要避开 |
| OpenTTGames 数据集 | CC BY-NC-SA 4.0 | ⚠️ 非商用，对外发布要注明 |

---

## 六之二、其他乒乓球公开数据集（学长提供的备选清单）

学长给的资料里列了四个替代数据集。这批数据集里**没有一个同时覆盖"球检测 + 落台/擦网 + 完整判分"**，所以事件层不能指望直接拿一个现成数据集训到端到端。

**但对 OpenTTGames 要更正一条**：它自带 `events_markup.json`，逐帧标注了 `bounce`（落台）/ `net`（擦网）/ `empty_event`——test_2 有 28 个 bounce + 21 个 net，game_3/4/5 各 240/231/248 个 bounce。**事件层是有监督信号、也有评估集可用的**，缺的只是判分（比分）。之前"事件层只能自己搭"的说法过于悲观。

| 数据集 | 规模与标注 | 能不能喂 TrackNet | 许可 |
|---|---|---|---|
| **TT ball** | 3,739 张图，7 类目标检测（球/球台/球员/裁判/球网/球拍/记分牌） | ❌ 静态图、无连续帧，热图法用不上；**适合 YOLO 分支**训球检测 | CC BY 4.0，需署名，仍要核实图片来源 |
| **Actual Table Tennis** | 1,050 张检测图，可导出 YOLO/COCO | ❌ 同上；且**公开版全部是训练集，无验证/测试划分** | CC BY 4.0，同样需核实来源 |
| **TT3D** | 130 条球轨迹，侧视/斜视视频 + 200 Hz 三维真值 | ⚠️ 可作**跨机位轨迹与落点验证集**，规模太小不能当训练主力 | 商用许可未确认，需联系作者 |
| **TTStroke-21** | 20 类击球动作 + 1 类非击球，标注动作起止时间 | ❌ 无球位置、无判分；单球员拍摄 | 需遵守组织方协议 |

几条能直接用的教训：
- **Actual Table Tennis 全部塞进训练集、没有划分**——我们按 `run_id` 划分训练/验证（`--split_runs`）的做法是对的，同一段视频的相邻帧绝不能跨集，否则评估虚高。这条可以直接写进报告当方法论。
- **TT3D 的 200 Hz 是三维真值跟踪的采样率，不代表视频是 200 fps**，引用时别搞混。
- 若要补检测类数据（YOLO 分支），TT ball 优先；若要补**跨机位/跨赛事**的轨迹泛化验证，TT3D 比 TT ball 更对口。
- 我们当前的 test_2 与训练用的 game_3/4/5 属**同一赛事 TT-LIGA.PRO**（抽帧核对过：同样的蓝幕布、蓝台面、黄地板、同款围挡），所以"加数据带来的提升"只能算同赛事的机位/光照覆盖，**不能宣称跨场馆泛化**。真要测跨场馆，得靠 TT3D 或自录素材。

---

## 参考链接

- MOTChallenge 2015 格式：https://arxiv.org/abs/1504.01942
- py-motmetrics：https://github.com/cheind/py-motmetrics
- Supervision：https://github.com/roboflow/supervision
- trackers：https://github.com/roboflow/trackers
- SportsLabKit：https://sportslabkit.readthedocs.io/
- PlaynTrack：https://github.com/Rogit-28/PlaynTrack
- TableTennisTracker：https://github.com/jainr3/TableTennisTracker
- Table-Tennis-Umpire：https://github.com/MatthewDulcich/Table-Tennis-Umpire
- TTNet 论文：https://arxiv.org/abs/2004.09927
- OpenTTGames：https://lab.osai.ai/
