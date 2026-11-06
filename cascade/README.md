# cascade/ — YOLO → TrackNet 提议 → TTNet 区间复核 级联层

> 状态（2026-11-04）：第 1–2 步完成；第 4 步在 test_2 上完成（推断落台真值）。
> 第 5 步代码完成：`--reviewer ttnet`（replay / live）已接入，备用权重 `ttnet.pth` 已验证可加载；
> **还没有真实 TTNet 分数**，要等原机跑 `cascade/tools/run_machine_a.ps1`。YOLO 观测尚未生成。
> 11-05：原机首跑完成，TTNet 复核把 test_2 落台召回从 0.857 提到 0.964，判罚不变（见「原机实测」）。
> 11-04：核对了 chagoup 自训权重（与备用权重不是同一份）；新增 `--state-machine` 可选 TTNet1.1 状态机（默认不变）。
> 11-06（B 机）：T1–T5 代码完成——四角构造标定、复核超时、人工确认清单、批量汇总、片段外漏检统计；真实 TTNet 缓存验收待原机数据。
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

# 复核超过 60 帧仍未完成：记 timeout，迟到的结果只写 late_results.jsonl
python -m cascade.run_cascade --jsonl adapter/examples/test_2_observations.jsonl \
    --preset test2 --review-latency-frames 90 --review-deadline-frames 60

# 回归测试（57 项；没有 TTNet 权重时，需要权重的测试自动跳过）
python -m unittest discover -s cascade/tests -t .
```

`--out-dir` 下会写出 `proposals.jsonl`、`clips.jsonl`、`reviews.jsonl`、`late_results.jsonl`、
`review_queue.jsonl`、`events.jsonl`（格式同 `run_mixed.py --events-out`）、`decisions.jsonl`、
`log.txt`、`summary.json`；TTNet 组另有 `ttnet_reviews.jsonl`。

## 模块

| 文件 | 作用 |
|---|---|
| `frame_index.py` | 帧号 ↔ 时间戳；区间按时间表达，查表得帧号 |
| `online_bounce.py` | `detect_bounce.detect()` 的流式等价版（前瞻 12 帧），只找 y 局部极大 |
| `proposals.py` | ProposalGenerator：局部极大（宽松阈值）、方向突变、台面附近丢球、YOLO 分歧四个触发器 |
| `scheduler.py` | ClipScheduler：合并重叠提议、判断收口与上下文帧、统计 TTNet 窗口数 |
| `calib.py` | 从 `calib/table_quads.json` 的四角构造新机位 `TableCalibration` |
| `reviewers.py` | 复核器接口；`RuleReviewer`（C3a，离线严格阈值）、`OracleReviewer`（已知事件，上限分析） |
| `online_out.py` | `run_mixed.derive_out_events` 的流式等价版 |
| `committer.py` | 按事件时间提交：在滞后的虚拟时钟上原样执行 `run_mixed` 的主循环 |
| `run_cascade.py` | 命令行入口与汇总 |
| `eval_proposals.py` | 用 `events_markup.json` 算提议召回、漏提议明细、确认落台 P/R/F1（±5 / ±12 帧） |
| `eval_ttnet_outside.py` | §7：全片 TTNet 驼峰中，峰值不落在任何复核片段内、且与真值落台 ±5 帧匹配的漏检 |
| `review_queue.py` | 汇总 `uncertain` / `error` / `timeout` 片段，生成待人工确认清单 |
| `run_batch.py` | 按清单批量跑 rule / ttnet，生成 `summary.md` / `summary.csv` 与 holdout 池化合计 |
| `state_machines.py` | 计分状态机选择：`tn`（默认）或 TTNet1.1 的 `TTNetStateMachine` 两个档位 |
| `ttnet_engine.py` | TTNet 单窗推理：`TorchEngine`（用 TTNet 仓库的 create_model，在外层兼容 CPU）/ `NumpyEngine`（不依赖 torch 的复刻，自带 .pth 读取） |
| `ttnet_cache.py` | 视频 → 逐窗口 TTNet 分数 JSONL（原机用） |
| `ttnet_reviewer.py` | `TTNetReviewer`：replay / live 两个后端，驼峰检测 + 印证判定 |
| `ttnet_k_check.py` | 用真值核实事件对齐偏移 k |
| `tools/run_machine_a.ps1` | 原机一键脚本 |

`run_cascade.py` 的 `--preset` 若是 core 内置预设（`game2` / `rally` / `plos` / `test4` / `test2`）或
`game4`，仍走 `run_mixed.build_calib`；其他新机位按 `--quad`（缺省由 preset 推出）从四角构造标定。

### 提交层为什么能保证"延迟不改判罚"

外部事件（落台、出界）在虚拟时刻 `ts + 0.07 s` 才会注入状态机（沿用 `run_mixed`
的注入延迟）。提交层维护 W = 尚未确定的外部事件时间戳下界，只把时间早于 `W + 0.07 s`
的帧交给状态机。于是无论提议、复核多慢，状态机看到的 `(tick, 事件)` 序列都与离线
`run_mixed` 完全相同，延迟只推迟判罚的输出时刻。W 被违反时直接抛错，不会悄悄判错。

W 的三个来源：尚未发出的提议（锚点 ≥ 当前帧 − 12 帧）、已发出未复核的片段、尚未定论的出界候选（丢球后要等 0.4 s）。

## 当前结果（test_2 前 30 s，3600 帧）

| 项 | 方案初值（±0.2/0.3 s，三个触发器） | 现默认（±0.1 s，关方向突变） |
|---|---|---|
| 判罚 / 事件流 | 与 `run_mixed` 逐条相同（0:2，3 条） | 相同 |
| 提议 | 78 条 | 67 条（局部极大 44、丢球 23） |
| 落台提议召回（28 个真值，距片段边缘 ≥ 6 帧） | 28/28 | 28/28 |
| 复核覆盖时长 | 61.8% | **29.9%** |
| TTNet 窗口 / 总帧数 | 69.8% | **30.8%** |
| 事件提交延迟 P50 / P95 | 0.61 s / 1.08 s | **0.30 s / 0.34 s** |

延迟按视频时间计、复核耗时记 0；接 TTNet 后要加上实测推理时间。关闭合并会把窗口数翻倍
（重叠区间重复计算），TTNet 复核器接入时应复用重叠帧的分数，按去重后的帧数计窗口。

## 第 4 步：落台真值与提议召回

**落台真值从球位置标注推断**（`infer_markup.py`）。Extended 仓库里 test_2 的事件文件恰好没有落台，
但 OpenTTGames 的球位置是围绕事件采的（每个事件一段 17 帧，事件在段内第 5 帧），可以反推：

* 在另外 6 段有完整标注的测试视频上验证：370 个落台/过网**逐帧全部找回**，类型判对 98.6%
* test_2 推出 28 个落台，与 TTNet 报告的"真值 28 次"一致；用它评估 TN 的 34 条检测得
  F1 **0.7742**、31 条得 P/R/F1 **0.6774 / 0.75 / 0.7119**，与 TN 报告逐位相同

所以在 test_2 上可以当原版用；拿到原版 `events_markup.json` 后以原版为准。

**提议层结论**（扫描 pre/post × 触发器开关，共 80 组）：

* 28 个落台里有 **4 个 TrackNet 根本没看到**（2.95 / 5.77 / 7.62 / 8.57 s，附近没有局部极大），
  正是规则检测漏掉的那 4 个。它们只被**丢球触发器**的片段覆盖（距边缘 10–12 帧）——
  这是级联的价值所在：只有 TTNet 复核才可能把它们找回来。关掉丢球触发器，召回降到 24/28。
* 方向突变触发器没带来任何召回，只增加覆盖，默认关。
* 用"完美复核器"（`--reviewer oracle`，真值落台）跑，test_2 的判罚仍是 2/2：这 4 个落台在第一分
  对拉中间，不影响得分方。**test_2 只有 2 分，不足以区分方案好坏**，需要更多视频。

参数只在一段视频上定的，有过拟合风险。test_1、3–7 都有落台与回合结束标注，只差 TrackNet 观测。

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

## 计分状态机的选择（`--state-machine`）

chagoup/TTNet1.1（放在 `ppq/TTNet1.1`）移植了同一套共用状态机，并以继承方式加了 TTNet 专属档位。
它的 `yolo_sm/state_machine.py`、`schemas.py` 与 TN `core/` 逐字节相同（只差导入路径），所以 `state_machines.py`
把它的导入映射到 TN 的 `core` 直接加载，不复制、不修改 TTNet1.1 的文件。

| `--state-machine` | test_2 规则复核 | test_2 真值落台复核 |
|---|---|---|
| `tn`（默认） | 0:2，2/2 分正确 | 0:2，2/2 |
| `ttnet11-yolo` | 与 `tn` 逐条相同（有单测） | 同左 |
| `ttnet11-ttnet` | **0:6**，多判 4 个 `DOUBLE_BOUNCE` | **3:3**，多判 5 分、漏判 1 分 |

`ttnet11-ttnet` 的"击球不清空落台侧"假设落台不漏：3.383 s 那次误判前，B 侧有一个 TrackNet 没看到的落台（真值 2.95 s），
于是 2.508 s、3.383 s 两次 A 侧落台被当成连跳两次。这个档位是按 TTNet 自己的事件流调的，放进级联会多判分，
所以默认仍用 `tn`，`ttnet11-*` 只作对照。

## 下一步

1. **更多视频**（最要紧）：原机用 TrackNet 适配器跑 test_1、3–7 出观测 → 这几段都有落台与回合结束标注，
   可以直接复核提议参数、核对更多分的判罚。新机位要先在 `calib/table_quads.json` 标四角并加预设。
2. 原机对 test_2 跑 YOLO 适配器（`pingpong-smart-referee/run_referee.py --video … --preset test2
   --export-jsonl …`，视频必须与 TrackNet 观测同一个文件）→ `--yolo` 打开分歧触发器。
3. 拿到原版 `events_markup.json` 时，用它替换推断真值复跑一遍
   `python -m cascade.eval_proposals --run-dir <输出目录> --events-markup <json> --jsonl <TrackNet 观测>`。
4. TTNet 复核器已接入（见下节）。剩下的是在原机跑一键脚本，拿到真实分数并核实 k。

## 第 5 步：TTNet 复核器

### 备用权重（2026-11-03 在云端验证，没有 GPU、也装不上 torch）

| 项 | 结果 |
|---|---|
| 来源 | SLAM-TT v1.0.0 Release 的 `ttnet.pth`，189,408,362 B，sha256 `370f1b667ee7717fcdb851df88801533665275abacd5fe9df2dba31a0d4ef6d9` |
| 内容 | `ttnet_3rd_phase_epoch_30`，epoch 30 |
| 键比对 | 用 TTNet 仓库自己的 `parse_configs` + `create_model`（借 torch 替身只枚举键名与形状）对照：226 / 226，missing 0、unexpected 0、形状不符 0（`BallDetection_right` 结构，键带 `model.` 前缀） |
| 前向 (1,27,128,320) | numpy 引擎正常；2 核 CPU 约 0.5–0.6 s/窗（含分割约 0.75 s）。**torch 的 CPU/GPU 耗时未测**，看原机的 `_data/ttnet/test_2.jsonl.meta.json` |
| 语义抽查 | 合成视频（暗背景上的白点）上，解码出的球位置与真实位置差 2–6 px：global→裁剪→local 链路复刻正确，也印证“球位置对应窗口末帧” |

权重没有放进任何仓库或这台电脑；原机自备。

### 规则

- **k（事件对齐偏移）**：结束于第 e 帧的窗口，事件分数对应第 e−k 帧，默认 k=4。依据是训练代码
  `get_events_infor`：9 帧窗口以 `smooth_idx` 为中心，事件标签峰在中间帧；球位置标签才取 `last_f_idx = smooth_idx + 4`。
  chagoup 的适配器把事件绑在末帧，相当于 k=0。**必须用真值核实**：`ttnet_k_check.py` 在 k=0 下统计峰值相对最近真值的中位偏差，
  它就是 k 的估计。`--reviewer ttnet` 默认 `--pre-ctx 8-k --post-ctx k`。
- **驼峰**：bounce 上阈 0.65 / 下阈 0.55（chagoup `EventExtractor` 同规则），只在片段 [start_frame_id, end_frame_id] 内找；片段末尾未结束的驼峰也算。
- **判定**：调度器标了 `missing_context` 或缓存缺窗口 → `error`；无驼峰 → `rejected`；
  有驼峰，且峰值 ±5 帧内有 TrackNet/YOLO 真实检出（visible、非 predicted，取离峰值最近的一次，同距取较早）、
  该点在台面多边形内、`detect_bounce.side_of` 为 A/B → `confirmed`，输出 BOUNCE（时间 = 峰值帧 / fps，sources `["ttnet", <印证来源>]`）；
  否则 `uncertain`（原因 `no_detection_near_peak` / `off_table` / `side_unknown`）。
- **net 分数**只记进 `ttnet_reviews.jsonl` 的 `net_cross`（上阈 0.5 / 下阈 0.4），**不生成 NET_TOUCH**。
- **窗口去重**：同一次运行内按窗口末帧缓存，重叠片段复用分数。`summary.json` 的 `ttnet.windows_dedup` 是去重后的窗口数，
  `clips.ttnet_windows` 仍是逐片段累计（未去重）。`ReviewResult.n_windows` 是该片段新算的窗口数。

### 怎么跑

```bash
# replay（读原机产出的缓存）
python -m cascade.run_cascade --jsonl adapter/examples/test_2_observations.jsonl --preset test2 \
    --reviewer ttnet --ttnet-cache _data/ttnet/test_2.jsonl --ttnet-k 4 \
    [--yolo _data/obs/yolo/test_2.jsonl] --out-dir outputs/cascade/test_2_ttnet

# live（从视频现算；有 torch 用 torch，没有就用 numpy，约 0.5 s/窗）
python -m cascade.run_cascade --jsonl ... --reviewer ttnet --video <test_2.mp4> \
    --ttnet-weights <ttnet.pth> [--ttnet-src ../TTNet/src] [--ttnet-engine numpy] --ttnet-device cuda:0

# 只生成缓存（原机）
python -m cascade.ttnet_cache --video <test_2.mp4> --weights <ttnet.pth> --ttnet-src ../TTNet/src \
    --engine torch --device cuda:0 --out _data/ttnet/test_2.jsonl

# 核实 k
python -m cascade.ttnet_k_check --ttnet-cache _data/ttnet/test_2.jsonl --gt \
    _data/gt/extended_openttgames/inferred/test_2_events_markup_inferred.json --gt-end-frame 3226
```

`--out-dir` 下多一个 `ttnet_reviews.jsonl`（每个片段的驼峰、过网证据、原因码）。

### 复核超时与人工确认（T2 / T3）

- `--review-deadline-frames N`：片段提交后等待 N 帧；默认 0 表示不启用。正常完成优先于截止，
  因此 `latency <= N` 与非截止运行逐项一致。超过截止时生成 `status="timeout"`、`reason="deadline_exceeded"` 的
  空事件结果，水位线不再等待；原复核结果后来到达时只写入 `late_results.jsonl`，不交给状态机。
- `summary.json` 新增 `timeout_count`、`timeout_ratio`、`late_result_count`、`manual_confirm_count`、
  `manual_confirm_ratio`、`unreviewable_count`、`missing_context_ratio`。
- 视频结束时尚未完成的复核同样按截止时间判：预计完成晚于截止的记 `timeout`，结果只进 `late_results.jsonl`。
- `review_queue.jsonl` 每条带 `category`：`manual`（需要人看）或 `unreviewable`（复核器无法判断，如规则复核遇到
  只有丢球触发的片段，原因码 `no_reviewable_candidate`）。人工确认比例只算 `manual`，否则规则组会被这类片段撑大，
  和 TTNet 组没有可比性（test_2：规则组 9 条全是 `unreviewable`）。
- `review_queue.jsonl` 只收 `uncertain` / `error` / `timeout`（`rejected` 不算），每条含 clip 时间/帧范围、
  原因码、TTNet 最高驼峰、片段前后 1 秒内的判罚。

### 批量运行与 §7 漏检（T4 / T5）

清单是一个 JSON；相对路径相对清单文件解析：

```json
{"videos": [{
  "name": "test_2", "split": "tuning", "preset": "test2", "quad": "test_2",
  "tracknet": "_data/obs/tracknet/test_2.jsonl",
  "yolo": "_data/obs/yolo/test_2.jsonl",
  "ttnet_cache": "_data/ttnet/test_2.jsonl",
  "events_markup": "_data/openttgames/test_annotations/test_2_extracted/events_markup.json",
  "extended": "_data/gt/extended_openttgames/game_data/test/test_2.json"
}]}
```

```bash
# 先复制样例并按实际文件改路径：cascade/batch_manifest.example.json → cascade/batch_manifest.json
python -m cascade.run_batch --manifest cascade/batch_manifest.json --out-dir outputs/cascade_batch
```

每个视频各跑 `_rule` / `_ttnet` 两组并接 `eval_proposals`、`eval_extended` 与
`eval_ttnet_outside`。`summary.md` / `summary.csv` 每段一行；`split="holdout"` 的合计行按
tp/fp/fn 汇总后重算 P/R/F1，不是平均各视频 F1。清单缺 TTNet 缓存时自动只跑 rule，并在备注列说明。

单段 T5 也可直接跑：

```bash
python -m cascade.eval_ttnet_outside --ttnet-cache _data/ttnet/test_2.jsonl \
    --clips outputs/cascade_batch/test_2_ttnet/clips.jsonl \
    --events-markup <events_markup.json> --k 4 --out <输出.json>
```

**原机一键脚本**（PowerShell，路径都是参数）：

```powershell
powershell -ExecutionPolicy Bypass -File cascade\tools\run_machine_a.ps1 `
  -TNRoot E:\20-项目\ppq\TN -VideoDir <OpenTTGames 测试视频目录> `
  -TTNetWeights <ttnet.pth> -TTNetSrc E:\20-项目\ppq\TTNet\src -Python <TN venv 的 python.exe>
```

依次做：自检（含权重 sha256）→ test_1、3–7 的 TrackNet 观测（在 `TrackNet/TrackNet` 下跑 `adapter/tracknet_adapter.py`，
权重 `exps/tt_g345_aug/model_best.pt`）→ test_2 的 YOLO 导出 → test_2 的 TTNet 缓存，以及 torch 对 numpy 的一致性测试 →
k 核实 → rule 和 ttnet 两组 run_cascade，每组接 eval_proposals 和 eval_extended，结果写到 `outputs/cascade_step5/`。
TTNet 依赖与 TN 的 venv 冲突时加 `-TTNetPython <另一个 python.exe>`。各步可以用 `-Skip*` 跳过；已有输出默认不重跑，加 `-Force` 重跑。
**这个脚本还没在 Windows 上实际跑过。**

**新机位标定**：test_1、test_3–7 在 `calib/table_quads.json` 里还没有四角。脚本会对这些视频跑
`calibrate_table.py export --frame 1200`，把带原图坐标网格的帧写到 `_data/calib/<game>_grid.jpg`。
按 far_left、far_right、near_right、near_left 的顺序读出坐标，照 test_2 的格式写进 `calib/table_quads.json`，
再用 `calibrate_table.py draw` 画回去核对。TrackNet 观测本身不需要四角，但提议、落台、侧别和复核都需要。

### 已验证（没有真实 TTNet 分数时能验证的部分）

- 规则复核路径不变：`--reviewer rule` 在 test_2 上仍是 0:2、3 条判罚。
- 端到端冒烟：构造一份“理想 TTNet”缓存（在 28 个推断真值落台处放驼峰，k=4）走 `--reviewer ttnet`：
  确认 26 / 28（另 2 个：一个附近没有 TrackNet 检出，判 uncertain；一个缺上下文，判 error），P 1.000 / R 0.929 / F1 0.963；
  判罚仍是 0:2、3 条；去重窗口 1084（30.1%），逐片段累计 1109；`ttnet_k_check` 估出 k=4。
- 合成视频上 replay 与 live 逐窗口分数差 ≤ 1e-3（numpy 引擎）。

### 原机实测（2026-11-05，RTX 5060，test_2）

第一次用真实 TTNet 分数（chagoup 自训权重 `ttnet_3rd_phase_epoch_30.pth`）跑完整级联。

| 项 | 规则复核（C3a） | TTNet 复核（C3，k=4） |
|---|---|---|
| 比分 / 判罚 | 0:2，3 条 | 0:2，3 条（逐条相同） |
| 每一分得分方（对照 Extended 标注） | 2/2 | 2/2 |
| 落台 P / R / F1（±5 帧，推断真值 28 个） | 0.828 / 0.857 / 0.842 | 0.794 / **0.964** / **0.871** |
| tp / fp / fn | 24 / 5 / 4 | 27 / 7 / 1 |
| 复核覆盖 / TTNet 窗口（去重） | 29.9% | 29.9% / 1084 窗（30.1%） |
| TTNet 单窗耗时（torch，cuda） | — | 中位 13.6 ms，p95 16.1 ms |

- **k = 4 得到确认**：29 个偏移样本中位 4，k=4 时平均绝对误差 0.54 帧；k=3 为 0.61，k=5 为 1.46。chagoup 适配器绑末帧（k=0）的做法会让事件晚约 4 帧。
- **级联的核心作用兑现了**：TrackNet 完全没看到的 4 个落台（2.95 / 5.77 / 7.62 / 8.57 s）只靠丢球触发器送进复核，TTNet **4 个全部找回**。
- **TTNet 没能否决规则的误检**：规则的 5 个误检 TTNet 都给了驼峰，一个没拦下；另多出 2 个只有 TTNet 认的误检（2.42 s 一个与规则重合，3.88 s、9.72 s 两个只在 TTNet）。这几个点要看视频判断是真误检还是标注遗漏。
- 唯一漏掉的真值（10.00 s）在这一分结束之后的死球段，TTNet 驼峰落在台外，判 uncertain，不影响比分。
- 只算 30% 的帧：全片逐窗推理要 3592 窗（74 s），级联只需 1084 窗（约 15 s）。
- test_2 只有 2 分，判罚上分不出两种复核的差别；要更多视频（test_1、3–7）才能看出对判罚的影响。

### 待核实

1. ~~k~~ **已确认 k = 4**（11-05 原机实测）。
2. **torch 单窗耗时**，以及 **torch 对 numpy 的一致性**（`tests/test_ttnet_weights.py`，需要设置 `TTNET_WEIGHTS` / `TTNET_SRC`）。
3. ~~备用权重和 chagoup 资产包是否同一份~~ **已核对（11-04）：不是同一份。** chagoup 的自训权重
   `ttnet_3rd_phase_epoch_30.pth`（放在 `ppq/` 根目录）sha256 `5e09137ece48f538a014c699744c8d45a225e39000c06c3efea3abe1951b6785`，
   与备用权重 `370f1b…6d9` 大小相同、内容不同；TTNet1.1 仓库里 LFS 存的 `ttnet.pth` 就是备用权重那份。
   自训权重在本机用 numpy 引擎加载、前向正常，`test_ttnet_weights` 的 replay/live 一致性通过（约 0.46 s/窗）；
   torch 对 numpy 的一致性仍待原机验证（本机没有 torch）。chagoup 的 WEIGHT_COMPARE_REPORT 称两份效果几乎一致。
4. **真实分数下的 C2 / C3 对照**：落台 F1 要不低于 C1 和 C2，重复计分为 0，TTNet 窗口调用数要低于全量。
