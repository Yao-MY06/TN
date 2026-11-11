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
    --preset test2 --review-latency-s 0.25

# 复核超过 60 帧仍未完成：记 timeout，迟到的结果只写 late_results.jsonl
python -m cascade.run_cascade --jsonl adapter/examples/test_2_observations.jsonl \
    --preset test2 --review-latency-s 0.75 --review-deadline-s 0.5

# 回归测试（116 项；没有 TTNet 权重时，需要权重的测试自动跳过；语音文本依赖缺失时对应测试跳过）
python -m unittest discover -s cascade/tests -t .
```

`--out-dir` 下会写出 `proposals.jsonl`、`clips.jsonl`、`reviews.jsonl`、`late_results.jsonl`、
`review_queue.jsonl`、`events.jsonl`（格式同 `run_mixed.py --events-out`）、`decisions.jsonl`、
`log.txt`、`summary.json`、`fused_obs.jsonl`（逐帧融合后球位，只给画视频用）；TTNet 组另有 `ttnet_reviews.jsonl`。

## 实时层（简化版）

实时层用视频文件模拟摄像头，并按每秒墙钟时间处理超时与持续时间：

```text
VideoFileCamera → FrameRing
   ├─→ YOLO worker ────┐
   └─→ TrackNet worker ┴→ 按 frame_id 汇合 + OnlineFusion(gate_s)
                         → CascadeCore → TTNet 有界队列 → 状态机
```

无 GPU 的全回放：

```bash
python -m cascade.run_realtime --null-camera 3600 --fps 120 --pace all \
    --yolo-replay _data/obs/yolo/test_2.jsonl \
    --tracknet-replay adapter/examples/test_2_observations.jsonl \
    --ttnet-cache _data/ttnet/test_2.jsonl --reviewer ttnet --preset test2 \
    --ttnet-ms-per-window 0 --out-dir outputs/realtime/replay_all
```

原机真模型（视频输入）：

```powershell
python -m cascade.run_realtime --video <test_2.mp4> --pace realtime \
    --yolo-weights E:\20-项目\ppq\pingpong-smart-referee\weights\yolo11n_clean_best.pt \
    --tracknet-weights TrackNet\TrackNet\exps\tt_g345_aug\model_best.pt \
    --ttnet-weights E:\20-项目\ppq\ttnet_3rd_phase_epoch_30.pth \
    --ttnet-src ..\TTNet\src --reviewer ttnet --preset test2 \
    --out-dir outputs/realtime/a_realtime
```

- `--pace all` 由有界队列形成背压，**不丢帧、不丢复核**：TTNet 队列满时主线程等待（同时回收已完成结果），
  结果与 `run_cascade` 同步回放逐条一致；汇合也不设超时，模型再慢也等两路结果到齐（`--join-timeout-s` 只在 realtime 下生效）。`--pace realtime` 按视频 fps / speed 吐帧，worker 队列满时
  丢最旧帧并输出不可见观测，迟到结果若晚于 `--join-timeout-s` 会被忽略并计数。
- 输入队列 `--yolo-queue` / `--tracknet-queue` 默认 all=64、realtime=2：实时节奏下只处理最新帧，
  否则 TrackNet（~42 ms/帧）排 64 帧要 2.7 s，结果全部晚于汇合超时而作废。
- TrackNet 实时跳帧推理时，输入仍是相机上的**连续**三帧 (t, t-1, t-2)：每帧（含不推理的帧）入队前
  都缩放后存入 `ConsecutiveFrames`，推理时按帧号取；缺帧则该帧输出不可见。
- TTNet 队列容量 `--ttnet-queue` 默认 8。`--pace realtime` 下队列满立即生成 `timeout / queue_full`，
  进人工确认清单；`--review-deadline-s` 按墙钟秒判截止，迟到结果只进 `late_results.jsonl`。
- 输出文件与 `run_cascade --out-dir` 相同，另写 `realtime_stats.json`：各阶段处理数/丢帧/帧率/耗时
  p50/p95、队列深度、TTNet queue_full/missing_context/窗口数、Ring `evicted_before_read`、
  判决端到端延迟 p50/p95/max 和总墙钟时间。
- 帧环只保存 TTNet 的 320×128 RGB 小图，不保存原图；容量由 `--ring-s`（默认 3 s）按秒计算。

test_2 全回放（sleep=0）与 `run_cascade` 的 proposals / clips / `events.jsonl`、判罚文本和比分逐项一致；
规则复核落台 F1 0.903，TTNet 复核 0.871。

## 球拍证据：RACKET_HIT 的 shadow / drop（默认关）

球拍缓存来自 `_data/racket/test_N.jsonl`，按 `racket.v1` 读取，不依赖球拍仓库运行时。
默认 `--racket-hit-hold off` 时完全不加载缓存，`events.jsonl` / `decisions.jsonl` 与旧行为逐字节一致。

### 固定参数

参数只在 test_2 上扫，选定后固定：

- Δt = 0.15 s
- 窗口中心相对 RACKET_HIT 前移 0.05 s
- 球拍置信度阈值 c = 0.50
- 球拍到球的距离按球拍框对角线归一化；NEAR 阈值 3.0
- 真击球匹配沿用 `eval_extended` 的 [-5, +25] 帧，并要求侧别相同

分类规则：

- `NEAR`：窗口内有同侧球拍，且框到球的距离 ≤ 阈值
- `FAR`：窗口内有同侧球拍，但距离 > 阈值
- `NONE`：窗口内没有同侧球拍
- `OTHER_SIDE`：只看到另一侧球拍

`shadow` 只给 RACKET_HIT 加 `racket_evidence` 标签，并把诊断项追加到 `review_queue.jsonl`；`drop` 只撤销 `FAR`。
没有同侧候选、缓存缺失或该帧未处理时立即按 `NONE` 放行；有同侧候选时最多扣留到窗口结束（0.10 s）。
`summary.json` 的 `racket_evidence` 记录四类计数、held/dropped/released 和额外延迟。

### 跑证据分析和模式对比

```bash
python -m cascade.racket_evidence \
    --manifest cascade/batch_manifest.json \
    --runs-root outputs/cascade_batch_racket/off \
    --racket-dir _data/racket \
    --out-dir outputs/racket_eval

python -m cascade.racket_compare \
    --base outputs/cascade_batch_racket/off \
    --compare outputs/cascade_batch_racket/shadow outputs/cascade_batch_racket/drop \
    --manifest cascade/batch_manifest.json \
    --out-dir outputs/cascade_batch_racket
```

清单项新增：

```json
"racket": "../_data/racket/test_2.jsonl",
"racket_hit_hold": "shadow",
"racket_static_filter": false
```

`racket_hit_hold` 可取 `off` / `shadow` / `drop`。静止过滤是可选项，开启后会过滤位置和框大小长期不变的目标；
test_7 的 26,010 条 visible UNKNOWN 检出中过滤掉 14,879 条，A/B 侧证据不变。

### 7 段结果

| 模式 | 复核 | test_2 判对 | 留出含 test_4 判对 | 留出不含 test_4 判对 | 含 test_4 多判/死球 | 不含 test_4 多判/死球 | 击球 P/R（含 test_4） |
|---|---|---:|---:|---:|---:|---:|---:|
| off | rule | 2/2 | 34/56 | 21/35 | 28/35 | 15/22 | 0.606/0.881 |
| shadow | rule | 2/2 | 34/56 | 21/35 | 28/35 | 15/22 | 0.606/0.881 |
| drop | rule | 2/2 | 34/56 | 22/35 | 41/29 | 35/19 | 0.651/0.722 |
| off | ttnet | 2/2 | 34/56 | 23/35 | 19/32 | 11/23 | 0.606/0.881 |
| shadow | ttnet | 2/2 | 34/56 | 23/35 | 19/32 | 11/23 | 0.606/0.881 |
| drop | ttnet | 2/2 | 32/56 | 22/35 | 39/24 | 33/18 | 0.651/0.722 |

test_4 是球拍模型训练集，必须单独看；留出集合计同时给出含/不含 test_4。

结论：

- 球拍证据在 test_2 能分开大部分假击球：FAR 为 0 真 / 9 假，`drop` 把击球精确率从 0.725 提到 0.935，召回保持 1.000。
- 但留出集不稳定：含 test_4 时 FAR 为 47 真 / 55 假，不含 test_4 时仍有 36 真 / 36 假。`drop` 把召回压到 0.722（含 test_4）和 0.744（不含），说明 FAR 不能可靠代表假击球。
- 判分上没有收益：rule 含 test_4 判对数不变，不含 test_4 仅 +1；TTNet 含 test_4 -2，不含 test_4 -1。多判显著增加，死球减少。
- `shadow` 与 off 的判分表完全一致，只增加标签和诊断清单。
- 因此 `drop` 不值得默认开启，保持 `off`；`shadow` 可用于继续收集证据。
## 计分 UI 与语音

三模型级联只负责产生 `decisions.jsonl`。级联状态机照常独立运行，不回写；语音仓库的 `ScoreEngine`
是视觉、语音和 UI 唯一共同读写的记分牌。

```text
run_cascade / run_realtime decisions
              │
              ▼
       ScoreboardBridge
       ├─ 物理 A/B → 选手 A/B（按 engine.side_of_a）
       ├─ POINT / DEAD_BALL / LET
       ├─ 待确认超时
       └─ scoreboard_log.jsonl
              │
              ▼
       SharedScoreBoard.engine (ScoreEngine)
       ├─ ScoreServer GET /api/state → 浏览器计分 UI
       └─ tt_voice Pipeline.push_text / feed → 同一 engine
```

### 判罚映射

| 级联 kind | 动作 | 播报 |
|---|---|---|
| `POINT` | 物理侧先按 `side_of_a` 换成选手；有视觉待确认项先 `reject_pending()`，再 `finish_rally(..., reason="vision:<rule>", confidence=1.0, source="vision")` | “左方得分，左方3比2右方” |
| `DEAD_BALL` | 默认 `propose_pending("UNKNOWN", reason="vision:<rule>", source="vision")`；`--dead-ball-mode ignore` 时只记日志 | “这一分请确认” |
| `LET` | `let(reason="vision:<rule>", source="vision")` | 不额外播报 |
| 其他 | 只写桥接日志 | 无 |

- `engine.side_of_a == "left"` 时，物理 A→选手 A、B→B；每局结束换边后，物理 A→选手 B、B→A。
- 视觉和语音写入在同一个主循环里串行执行；启用 UI 时，写操作和 UI 读取共用 `ScoreServer._server.lock`。
- 待确认超时按视频秒数计算：`--pending-timeout-s` 默认 5 s。
- `--say` 默认开启，`--no-say` 关闭。播报文本只含比分/提示，不含唤醒词。

### 回放入口

`run_scoreboard.py` 的主要开关：`--ui` / `--ui-port` / `--ui-window` / `--open-browser`，
`--voice none|text|mic`，`--say` / `--no-say`，`--dead-ball-mode pending|ignore`，
`--pending-timeout-s`（秒），`--speed`，`--max-frames`，`--linger-s`（回放结束后 UI/语音继续运行的秒数，现场说口令、录屏收尾用）。`run_realtime.py` 需先加 `--scoreboard`，
再使用同一组 UI/语音开关。

```powershell
cd E:\20-项目\ppq\TN
$env:PYTHONUTF8='1'
python -m cascade.run_scoreboard `
  --run-dir E:\20-项目\ppq\TN\outputs\cascade_batch_sm\tn\test_2_rule `
  --video D:\OpenTTGames\test\videos\test_2.mp4 `
  --speed 1 --max-frames 3600 `
  --ui --open-browser --ui-port 8765 --ui-window 1024x600 `
  --voice none --say `
  --out-dir E:\20-项目\ppq\TN\outputs\scoreboard_replay\test_2_ui
```

预期：播放 test_2 视频时打开 `http://127.0.0.1:8765/`，先显示 0:0，两个 POINT 依次到 0:2；
每次视觉加分听到“右方得分，左方0比1/2右方”。结束时终端打印物理累计 `A:0 B:2`，
并写 `scoreboard_log.jsonl`、`scoreboard_final.json`。

文本模式（stdin 与视觉在同一个主循环处理）：

```powershell
python -m cascade.run_scoreboard `
  --run-dir E:\20-项目\ppq\TN\outputs\cascade_batch_sm\tn\test_2_rule `
  --video D:\OpenTTGames\test\videos\test_2.mp4 `
  --speed 1 --max-frames 3600 `
  --ui --open-browser --ui-port 8765 --ui-window 1024x600 `
  --voice text --say `
  --out-dir E:\20-项目\ppq\TN\outputs\scoreboard_replay\test_2_text
```

预期：视觉到 0:2 后，在终端输入 `小裁判 撤销上一分`，UI 回到 0:1 并播报“已撤销”；
再输入 `小裁判 左方加一分`，UI 变为 1:1。
文本模式会把行首的唤醒词拆出来，先送唤醒词、再送口令（语音仓库在聆听态下整句能匹配唤醒词时只回“在”，不解析口令）；
只写口令也可以，聆听窗口已结束时会自动补一次唤醒词。

麦克风模式（先确认设备编号；没有默认输入设备时用 `python -m tt_voice.cli devices`）：

```powershell
python -m cascade.run_scoreboard `
  --run-dir E:\20-项目\ppq\TN\outputs\cascade_batch_sm\tn\test_2_rule `
  --video D:\OpenTTGames\test\videos\test_2.mp4 `
  --speed 1 --max-frames 3600 `
  --ui --open-browser --ui-port 8765 --ui-window 1024x600 `
  --voice mic --input-device 0 --say `
  --out-dir E:\20-项目\ppq\TN\outputs\scoreboard_replay\test_2_mic
```

预期：先说“小裁判”，听到“在”；再说“撤销上一分”或“左方加一分”，UI 与播报同步更新。
如果 `0` 不是你的麦克风编号，替换成 `devices` 输出中的编号。
麦克风模式按语音仓库 `config.json` 的 `engine`（默认 vosk）构造识别引擎，需要先下载模型（`python tools\download_model.py`）。
test_2 只有 30 s，0:2 出现在约 24 s；现场说口令建议加 `--speed 0.5` 或 `--linger-s 30`。
建议先单独说“小裁判”，听到“在”再说口令。

`run_realtime.py` 同样支持这些开关，只需加 `--scoreboard`：

```powershell
python -m cascade.run_realtime --video D:\OpenTTGames\test\videos\test_2.mp4 `
  --pace realtime --reviewer rule --tracknet-replay E:\20-项目\ppq\TN\adapter\examples\test_2_observations.jsonl `
  --preset test2 --scoreboard --ui --open-browser --say --voice none `
  --out-dir E:\20-项目\ppq\TN\outputs\scoreboard_replay\realtime_test_2
```

### 已知限制

- 语音/UI 的人工撤销、补分、重置只作用于 `ScoreEngine`，不会回写级联状态机。桥接层会把这类动作记为
  `manual_override`，但之后的视觉判罚仍按级联自己的状态继续，因此人工改分后两边可能不一致。
- `ScoreEngine` 是 11 分、五局三胜、两次发球换发；级联状态机只负责产生事件和判分建议。回放验收比较的是
  `physical_points`（视觉 POINT 按当时的物理侧累计），UI 当前局分是 `snapshot.points` / `snapshot.sets`，长局结束后当前局分会归零。
- UI 是只读计分牌，只有语音/视觉主循环能写；语音仓库与 UI 仓库包名相同，cascade 只按路径挂载
  `pingpong-voice-referee`，不会同时加载另一份 `tt_referee`。
- `DEAD_BALL` 的待确认侧别是 `UNKNOWN`。只说“确认这一分”不会猜赢家，而是保留待确认并提示补说“左方加一分/右方加一分”；直接说方向会补上该分。
- Windows 播报走 SAPI；Linux 只输出控制台文本。麦克风需要 `sounddevice`，语音文本需要 `pypinyin` 等语音仓库依赖。
- 本机无 `pypinyin`，所以语音文本 / 麦克风相关测试自动跳过；回放、UI、桥接和计分测试不依赖该包。
- 同组仓库按 `cascade/_paths.py` 的顺序查找：环境变量 `PPQ_ROOT`（或 TN 的上一级）→ TN 根目录 → `TN/third_party`。
  原机的布局是 TN 仓库直接克隆在 `ppq/`，语音仓库放在 `ppq/pingpong-voice-referee` 即可，不需要额外建目录联接。
- `gd*` 状态机需要国栋仓库 `main` 73ea183 或更新（门控版）；`fusion/three-way` 自带的 `third_party` 快照是旧版，加载时会报“版本过旧”，测试自动跳过。

## 画视频（事后渲染）

`render_video.py` 把任一 `run_cascade` / `run_realtime` 输出目录画成带 HUD 的 mp4：球位与尾迹（TrackNet 橙、YOLO 青）、
当前提议区间及其 TTNet 复核状态、确认的事件和最近落点、比分与判罚、底部全片时间轴。渲染在运行结束后单独做，不影响实时计时。

```bash
python -m cascade.render_video --run-dir outputs/realtime/a_all --video <test_2.mp4> \
    --preset test2 --max-frames 3600 --out outputs/realtime/a_all/cascade_hud.mp4
```

球位优先用运行目录里的 `fused_obs.jsonl`。旧版运行没有这个文件时，可加 `--tracknet-obs`（可再加 `--yolo-obs`）离线融合出球位，
画面会标 `ball: offline obs (not this run)`，实时节奏下的丢帧不会反映在球位上。`--scale 0.5` 可输出 960×540。
区间颜色是该区间最终的复核结果（事后才知道），不代表复核在区间开始时就已完成。

## 网附近异常：NET_TOUCH 触发 + 网带击球扣留（默认关闭）

`--net-hit-hold` 同时开启两个默认关闭的能力：

1. **6.2 提议**：核心轨迹层实际发出 `NET_TOUCH` 时生成 `net_anomaly` 提议，区间固定为
   `[t_net - 0.1 s, t_net + 0.3 s]`，不再另做一套几何检测。只开触发的消融键是清单字段
   `"net_trigger": true`。
2. **6.3 扣住与裁决**：committer 拦截同时满足以下条件的 `RACKET_HIT`：
   `|x - calib.net_x| <= 30 px`，且距离最近 `NET_TOUCH` 不超过 `0.15 s`。扣住期间水位线不前进。
   随后在固定 `0.3 s` 窗口内看融合轨迹：球向击球方向离网超过 `60 px`（2 倍带宽）且 y 变小，
   才按原时间戳放行；否则撤销该击球。帧不足或看不到球时放行并进人工确认。

这些数值是 6.3 预注册值，不在留出集上调整。撤销事件在 `summary.json.net_hit_hold.events`
和 `review_queue.jsonl` 中可查；人工清单类别为 `net_reversal`。net_anomaly 专用片段不与普通提议
交叉合并，也不会被通用 bounce 复核误当成落台。

```bash
python -m cascade.run_cascade --jsonl _data/obs/tracknet/test_4.jsonl \
    --preset test4 --quad test_4 --reviewer rule --net-hit-hold \
    --out-dir outputs/net_diag/test_4_hold
```

`run_batch` 清单项支持 `"net_hit_hold": true`，以及仅用于消融的 `"net_trigger": true`。
3.3 网附近丢球专项没有实现（诊断 B 类为 0）。对打中的 `NET_TOUCH` 仍由共用状态机忽略；
发球阶段 `NET_TOUCH -> LET` 的风险属于状态机缺口，cascade 层不绕过它自行判分。

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
| `run_batch.py` | 按清单批量跑 rule / ttnet，生成 `summary.md` / `summary.csv` 与 holdout 池化合计；支持 `--state-machine` |
| `compare_sm.py` | 只读批量输出，比较多个状态机的判分、门因归因、门控计数与出界诊断 |
| `racket_evidence.py` | 球拍证据 NEAR/FAR/NONE/OTHER_SIDE 分类、test_2 参数扫描与静止目标过滤 |
| `racket_compare.py` | 对比 off/shadow/drop 的判分、击球精确率/召回和逐分改判 |
| `render_video.py` | 事后把一次运行画成带 HUD 的视频（球位来源、复核区间状态、事件、比分、时间轴） |
| `racket_cache.py` | 视频 → 逐帧球拍观测缓存（racket.v1，朗然的 `pingpong-racket-detector`，侧别按本机位网线 x 判定），原机用 |
| `run_realtime.py` | 实时层入口：视频/空帧源、并行 YOLO/TrackNet/TTNet worker、有界队列与运行统计；可接计分牌 |
| `scoreboard_bridge.py` | 级联判罚 → 语音仓库 `ScoreEngine` 的映射、待确认超时与日志 |
| `run_scoreboard.py` | 回放判罚驱动计分 UI/语音，并写 `scoreboard_log.jsonl` / `scoreboard_final.json` |
| `realtime/frames.py` | `Camera` / `VideoFileCamera` / `NullCamera` 与线程安全 320×128 帧环形缓存 |
| `realtime/workers.py` | YOLO/TrackNet 逐帧 worker、TTNet 区间 worker；均支持无 GPU 的回放 sleep 模式 |
| `state_machines.py` | 计分状态机选择：`tn`（默认）、`gd` / `gd-traj` / `gd-nr` 或 TTNet1.1 的两个档位 |
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

国栋仓库（`ppq/pingpong-smart-referee`，main `73ea183`）的门控版也按路径直接加载：

- `gd`：只换 `core/state_machine.py`。发球准入、发球前 1.5 s 内须有击球、回合内须有击球才判分（OUT 同样受限）、判分后冷却 2 s、`NO_RETURN` 默认关、按 `event_id` 去重。
- `gd-traj`：在 `gd` 基础上再换 `core/trajectory_processor.py`（关联速度上限 9000 px/s、确认帧防劫持）。
- `gd-nr`：国栋状态机，仅把 `MatchRules.enable_no_return` 打开到 TN 默认值；只用于归因分析，不是采用候选。
- 加载前会核对国栋与 TN 的 `core/schemas.py`（忽略 CRLF 差异）；不一致直接报错，不会混用事件类型。
- 两个仓库都只读引用，不复制文件。`summary.json` 的 `state_machine` 记录实际档位，`sm_gates` 记录门控计数（`tn` 为 `null`），`state_machine_event_id_renames` 记录送入状态机前的 ID 改名次数。

cascade 层会把送给状态机的事件 ID 在单次运行内改成唯一值：首项保持原名，重复项稳定加 `~2`、`~3` 后缀。`events.jsonl`、`decisions.jsonl` 和 `event_sources` 仍使用原 ID，因此 tn 的对外输出不变。这个处理修掉了核心轨迹层与 `OnlineOutDetector` 都生成 `out_00001` 起的 ID 时的撞号；国栋状态机不会再误把真实 OUT 当成重复事件。

| `--state-machine` | test_2 规则复核 | test_2 TTNet 复核 |
|---|---|---|
| `tn`（默认） | 0:2，2/2 分正确 | 0:2，2/2 |
| `gd` | 0:1，1/2；`left_net` 被发球准入挡下 | 0:1，1/2 |
| `gd-traj` | 与 `gd` 逐条相同 | 与 `gd` 逐条相同 |
| `gd-nr` | 0:1，1/2；同一分仍被发球准入挡下 | 0:1，1/2 |
| `ttnet11-yolo` | 与 `tn` 逐条相同（有单测） | 同左 |
| `ttnet11-ttnet` | **0:6**，多判 4 个 `DOUBLE_BOUNCE` | **3:3**，多判 5 分、漏判 1 分 |

`ttnet11-ttnet` 的"击球不清空落台侧"假设落台不漏：3.383 s 那次误判前，B 侧有一个 TrackNet 没看到的落台（真值 2.95 s），
于是 2.508 s、3.383 s 两次 A 侧落台被当成连跳两次。这个档位是按 TTNet 自己的事件流调的，放进级联会多判分，
所以默认仍用 `tn`，`ttnet11-*` 只作对照。

### 7 段视频上的 tn / gd / gd-traj / gd-nr 对比

同一份 `cascade/batch_manifest.json`，每个状态机各跑 rule 与 TTNet 两组：

```bash
for sm in tn gd gd-traj gd-nr; do
  python -m cascade.run_batch --manifest cascade/batch_manifest.json \
      --out-dir outputs/cascade_batch_sm/$sm --state-machine $sm
done
python -m cascade.compare_sm \
    --base outputs/cascade_batch_sm/tn \
    --compare outputs/cascade_batch_sm/gd outputs/cascade_batch_sm/gd-traj outputs/cascade_batch_sm/gd-nr \
    --out outputs/cascade_batch_sm
```

`compare.md` 分五段：

1. **合计表**：每个状态机 × 每种复核，test_2 单列，留出集合计；列出判对、出界/下网/其他未判、判错方、多判、死球。
2. **逐分改判表**：相对 tn，只列“原来对→现在错”和“原来错→现在对”，并给出双方的判罚。
3. **门因归因表**：对“原来对→现在错”的每一分，用已保存的 `events.jsonl`、`fused_obs.jsonl`、`decisions.jsonl` 回放状态机定位门因。
4. **门控计数**：每个状态机每道门挡掉的事件数，按视频和复核器列出。
5. **出界诊断**：每个未判 `*_out` 的分，在“上一个真值分结束 → 本题 `gt_t + 2 s`”窗口内有没有 OUT 事件、判罚是什么。

留出集结果（从 `outputs/cascade_batch_sm/compare.md` 抄录）：

| 状态机 | 复核 | 得分方判对 | 出界漏判 | 下网漏判 | 判错 | 多判 | 死球 |
|---|---|---:|---:|---:|---:|---:|---:|
| `tn` | rule | 34/56 | 8 | 3 | 8 | 28 | 35 |
| `tn` | ttnet | 34/56 | 11 | 2 | 7 | 19 | 32 |
| `gd` | rule | 14/56 | 21 | 13 | 3 | 2 | 65 |
| `gd` | ttnet | 13/56 | 21 | 14 | 3 | 1 | 67 |
| `gd-traj` | rule | 14/56 | 21 | 13 | 3 | 2 | 65 |
| `gd-traj` | ttnet | 13/56 | 21 | 14 | 3 | 1 | 67 |
| `gd-nr` | rule | 21/56 | 19 | 9 | 4 | 5 | 54 |
| `gd-nr` | ttnet | 19/56 | 18 | 11 | 6 | 4 | 54 |

结论：

- 修掉 OUT ID 撞号后，`gd` 比上一轮多判对 3 分（rule 11→14）和 4 分（TTNet 9→13）；`gd-traj` 仍与 `gd` 逐项相同。
- `gd-nr` 相比 `gd` 再多判对 7 分（rule 14→21）和 6 分（TTNet 13→19），说明 `NO_RETURN` 默认关闭是上一轮最主要的人为差距之一。
- 但 `gd-nr` 仍比 tn 少判对 13 分（rule）和 15 分（TTNet）。留出集剩余“原本判对→现在未判”为 rule 16 条、TTNet 17 条；按门因分别为：
  - rule：发球准入未完成 5、发球阶段判罚被忽略 5、回合内须有击球 3、同侧两弹夹击球 1、判分后冷却 1、未进入可判分回合 1。
  - ttnet：发球阶段判罚被忽略 7、发球准入未完成 5、未进入可判分回合 2、回合内须有击球 2、同侧两弹夹击球 1。
- 修复后门因表不再出现事件 ID 撞号；所有新跑法中 `duplicate_event_ids_ignored` 都是 0。`gd-traj` 与 `gd` 的门因也完全一致。
- 出界漏判仍高于 tn：`gd` 为 rule 21 / TTNet 21，`gd-nr` 为 rule 19 / TTNet 18；多判虽从 tn 的 28/19 降到 5/4（gd-nr），死球仍明显增加到 54/54。
- 送入状态机的 ID 改名次数（rule/TTNet）：`gd`、`gd-traj`、`tn` 相同，依次为 test_1 0/0、test_2 0/0、test_3 1/1、test_4 4/5、test_5 1/1、test_6 2/2、test_7 1/1；`gd-nr` 除 test_4 为 5/6 外其余相同。

`tn` 回归：`outputs/cascade_batch_sm/tn_verify/summary.md` 与旧 `outputs/cascade_batch_sm/tn/summary.md` 逐字节相同；14 组 `events.jsonl` 和 14 组 `decisions.jsonl` 也全部逐字节相同。

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

- `--review-deadline-s S`：片段提交后最多等待 S 秒（按时间戳，不按帧数）；默认 0 表示不启用。正常完成优先于截止，
  因此 `latency <= S` 与非截止运行逐项一致。超过截止时生成 `status="timeout"`、`reason="deadline_exceeded"` 的
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
python -m cascade.run_batch --manifest cascade/batch_manifest.json \
    --out-dir outputs/cascade_batch --state-machine tn
```

清单项也可以写 `"state_machine": "gd"`，清单项优先于命令行默认值。

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

## YOLO 与 TrackNet 的观测融合（`fuse_obs.py`）

按级联方案 §1–§2：YOLO 实时定位，TrackNet 读连续图像给出连续轨迹，YOLO 在模型外辅助
（候选关联、质量检查、重新捕获），不替代 TrackNet 的图像输入。给了 `--yolo` 时默认：

- **重新捕获**：某帧 TrackNet 没看到球、而 YOLO 有可信检出时，用 YOLO 的位置补上。
  可信 = 置信度 ≥ 0.5，且前后 0.017 s 内有位置连得上的 YOLO 检出（孤立跳点视为误检）。
- **质量检查**：两者都看到但位置分歧 > 40 px 并持续 ≥ 0.015 s，提议层 `yolo_disagree` 触发复核。
- 融合后仍是统一 Observation，每帧 `source` 标明球位置来自哪个模型；事件的 `sources`
  也据此标注（`events.jsonl`），TTNet 确认的落台标为 `["ttnet", <证据来源>]`。
- `--no-yolo-recapture` 关掉重新捕获（只保留分歧触发），用于复现旧结果。
- `--primary yolo` 是试验选项：YOLO 主跟踪、回合中 YOLO 丢球（最近 `--fill-window-s` 秒内见过球）
  才调 TrackNet，用来估算 TrackNet 能省多少算力。与方案中 TrackNet 的角色不同，不作默认。

所有持续/超时条件按时间戳（共用状态机方案 §4.1）：`--dropout-min-s`、`--review-latency-s`、
`--review-deadline-s`、`--fill-window-s`；默认值在 120 fps 下与原来的帧数等价，7 段批量结果不变。

test_2（调参视频，2026-11-07）：

| 轨迹 | 复核 | 落台 P / R / F1 | 得分方判对 | TrackNet 调用帧 |
|---|---|---|---|---|
| 仅 TrackNet | 规则 | 0.828 / 0.857 / 0.842 | 2/2 | 100% |
| 仅 TrackNet | TTNet | 0.788 / 0.929 / 0.853 | 2/2 | 100% |
| **TrackNet + YOLO 重新捕获（默认）** | **规则** | **0.824 / 1.000 / 0.903** | **2/2** | 100% |
| TrackNet + YOLO 重新捕获（默认） | TTNet | 0.794 / 0.964 / 0.871 | 2/2 | 100% |
| YOLO 主跟踪（试验） | 规则 | 0.824 / 1.000 / 0.903 | 1/2 | 27% |

YOLO 主跟踪丢的一分（9.45 s，A 下网）：球撞网弹回，水平速度反向，轨迹层把它当成 B 击球。
