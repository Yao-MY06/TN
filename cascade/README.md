# cascade/ — YOLO → TrackNet 提议 → TTNet 区间复核 级联层

> 状态：第 1–2 步完成（2026-11-03）。TTNet 复核器尚未接入，当前用**规则复核器**（C3a）占位。
> 计划文档：《YOLO→TrackNet→TTNet 级联融合实施计划》

只在外层包装：`tt/`、`adapter/`、`TrackNet/`、状态机交付件、另外两个仓库**一行未改**。

## 跑一下

在 TN 仓库根目录，只用仓库内样例，不需要视频、权重或 GPU：

```bash
# 级联回放（规则复核 = C3a）
python -m cascade.run_cascade --jsonl adapter/examples/test_2_observations.jsonl \
    --preset test2 --out-dir outputs/cascade/test_2_rule

# 模拟复核耗时 30 帧：判罚不变，只是输出变晚
python -m cascade.run_cascade --jsonl adapter/examples/test_2_observations.jsonl \
    --preset test2 --review-latency-frames 30

# 回归测试（11 项，约 4 s）
python -m unittest discover -s cascade/tests -t .
```

`--out-dir` 下会写出 `proposals.jsonl`、`clips.jsonl`、`reviews.jsonl`、`events.jsonl`（格式同
`run_mixed.py --events-out`）、`decisions.jsonl`、`log.txt`、`summary.json`。

## 模块

| 文件 | 作用 |
|---|---|
| `frame_index.py` | 帧号 ↔ 时间戳；区间按时间表达，查表得帧号 |
| `online_bounce.py` | `detect_bounce.detect()` 的流式等价版（前瞻 12 帧），只找 y 局部极大 |
| `proposals.py` | ProposalGenerator：局部极大（宽松阈值）、方向突变、台面附近丢球、YOLO 分歧四个触发器 |
| `scheduler.py` | ClipScheduler：合并重叠提议、判断收口与上下文帧、统计 TTNet 窗口数 |
| `reviewers.py` | 复核器接口；`RuleReviewer`（C3a，离线严格阈值）、`OracleReviewer`（已知事件，上限分析） |
| `online_out.py` | `run_mixed.derive_out_events` 的流式等价版 |
| `committer.py` | 按事件时间提交：在滞后的虚拟时钟上原样执行 `run_mixed` 的主循环 |
| `run_cascade.py` | 命令行入口与汇总 |
| `eval_proposals.py` | 用 `events_markup.json` 算提议召回、漏提议明细、确认落台 P/R/F1（±5 / ±12 帧） |

### 提交层为什么能保证"延迟不改判罚"

外部事件（落台、出界）在虚拟时刻 `ts + 0.07 s` 才会注入状态机（沿用 `run_mixed`
的注入延迟）。提交层维护 W = 尚未确定的外部事件时间戳下界，只把时间早于 `W + 0.07 s`
的帧交给状态机。于是无论提议、复核多慢，状态机看到的 `(tick, 事件)` 序列都与离线
`run_mixed` 完全相同，延迟只推迟判罚的输出时刻。W 被违反时直接抛错，不会悄悄判错。

W 的三个来源：尚未发出的提议（锚点 ≥ 当前帧 − 12 帧）、已发出未复核的片段、尚未定论的出界候选（丢球后要等 0.4 s）。

## 当前结果（test_2 前 30 s，3600 帧）

| 项 | 合并（默认） | 不合并 |
|---|---|---|
| 判罚 / 事件流 | 与 `run_mixed` 逐条相同（0:2，3 条） | 相同 |
| 提议 | 78 条（局部极大 44、丢球 23、方向突变 11） | 同左 |
| 复核覆盖时长 | 61.8% | 61.8% |
| TTNet 窗口 / 总帧数 | 69.8% | 131.0%（重叠区间重复计算） |
| 事件提交延迟 P50 / P95 | 0.61 s / 1.08 s | 0.30 s / 0.31 s |

延迟按视频时间计、复核耗时记 0；接 TTNet 后要加上实测推理时间。

**待第 4 步处理**：覆盖率远高于 ≤ 40% 的目标——`[t0 − 0.2 s, t0 + 0.3 s]` 的区间偏宽，丢球触发器在
TrackNet 可见率只有 30.6% 的视频上触发很多。要等拿到 `events_markup.json` 后按提议召回率一起调，
不能只为压覆盖率而调。另外重叠片段可以复用逐帧分数，TTNet 复核器接入时按"去重后的帧数"计窗口。

## 用独立真值核对（Extended OpenTTGames）

[moamal01/table_tennis_data](https://github.com/moamal01/table_tennis_data)（CC BY-NC-SA 4.0，arXiv:2512.19327）
给 OpenTTGames 补了**击球**和**每分的结束方式**标注（不含落台）。文件放在 `_data/gt/extended_openttgames/`
（`_data/` 不进 git），核对脚本：

```bash
python -m cascade.eval_extended --run-dir outputs/cascade/test_2_rule \
    --game-json _data/gt/extended_openttgames/game_data/test/test_2.json
```

test_2 前 30 s 的结果：

| 项 | 结果 |
|---|---|
| 每一分得分方 | **2/2 正确**：9.45 s `left_net` → 我方 9.675 s 判右方得分（`OWN_COURT`）；23.833 s `left_out` → 我方同一帧判右方得分（`OUT`） |
| 击球召回 / 侧别 | **29/29**，侧别 29/29 |
| 击球时间 | 系统性**偏晚 7–23 帧，中位 15 帧（125 ms）**；按 ±5 帧对称容差算 F1 为 0 |
| 多出的击球 | 11 条全在第一分：10 条与标注的右方击球严格 A/B 交替，疑似标注漏了左方回球；9.817 s 一条在该分结束后，是真误报 |

**击球滞后的影响**：轨迹层要等球重新出现、看到水平速度反向才报击球，而击球瞬间 TrackNet
在 ±12 帧内多数只看到 0–3 帧球。`run_mixed` 的 0.07 s 注入延迟是按"滞后 0–17 ms"定的，
实测约 125 ms，快速对拉里"落台排到击球前面"的风险比原先估计的大。这一条值得同步给状态机那边。

## 下一步

1. 原机对 test_2 跑 YOLO 适配器（`pingpong-smart-referee/run_referee.py --video … --preset test2
   --export-jsonl …`，视频必须与 TrackNet 观测同一个文件，帧号才对得上）→ `--yolo` 打开分歧触发器。
2. 拷 test_2 的 `events_markup.json` →
   `python -m cascade.eval_proposals --run-dir <run_cascade 输出目录> --events-markup <json> --jsonl <TrackNet 观测>`。
3. TTNet 到位 → `TTNetReviewer`（replay / live），`--pre-ctx 8-k --post-ctx k`。
