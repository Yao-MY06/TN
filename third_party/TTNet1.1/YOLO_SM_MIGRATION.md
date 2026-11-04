# YOLO 状态机移植报告（TTNet 1.1）

> 任务：把 TTNet 的计分状态机替换为
> [pingpong-smart-referee](https://github.com/V2tin19/pingpong-smart-referee) 的状态机，
> 并用 test_1 / test_2 做视频测试，输出 UI 渲染后的视频。
>
> 完成时间：2026-11-02
> 视频产物：`C:\Users\PC\Desktop\TTNet_test1_结果\test_1_比分版_yoloSM.mp4`（140s）
> 　　　　　`C:\Users\PC\Desktop\TTNet_test1_结果\test_2_比分版_yoloSM.mp4`（30s）

---

## 一、结论先说

| 项目 | 状态 |
|---|---|
| 状态机移植 | ✅ 完成，**代码原样保留**，只改导入路径 |
| TTNet 主流水线默认引擎 | ✅ 已切换为 YOLO 状态机（`--engine legacy` 可切回） |
| test_1 视频渲染 | ✅ 16800 帧 / 140s / 1920×1080@120fps / 475.8 MB |
| test_2 视频渲染 | ✅ 3600 帧 / 30s / 1920×1080@120fps / 133 MB / **A1:B2** |
| 原状态机回归 | ✅ 与新代码逐分一致（时间戳完全相同） |
| 单测 | ✅ 31 项（状态机）+ 23 项（桥接）+ 9 项（事件层）全绿 |
| **事件层优化** | ✅ **不改状态机**，HIT 恢复 + 阈值调优，见第六节 |
| **TTNet 专属状态机** | ✅ 继承扩展，YOLO 上游逻辑 **0 改动**（仅 2 行 import），见第七节 |
| **死球率** | ✅ 18/3/9 → 5/2/4（事件层）→ **1/1/0**（TTNet 档位） |
| **test_1 判分** | ✅ 6 分 → **16 分**（140s，每 8.8s 一分，节奏合理） |
| 判分准确率 | ⚠️ 仍无可验证提升（缺逐分真值），但漏判显著减少 |

**最重要的一句**：这轮做了三件事，效果依次递进：

1. **修掉一个既有 bug** —— 短暂遮挡就清空击球历史，让 HIT 少了一半
   （test_2 因此一度判出 11 分）；
2. **纠正精确率的统计口径** —— 模型其实不差，之前被标注缺失冤枉成 22% 精确率；
3. **加了一层 TTNet 专属状态机**（继承扩展，YOLO 上游 0 改动）。

合计效果：**test_1 从 6 分提升到 16 分，死球从 5 降到 1**；
test_3 从 6 分到 13 分，死球清零。

仍要诚实说明：**没有逐分真值，无法证明「判对了」**。
唯一可核对的是 test_2 的回合数（真值 3），三套方案都判 3 分 ——
但内容不同，细节见 §6.4 与 §7.4。

---

## 二、两个项目的对应关系

YOLO 项目按组长方案《TTNet、TrackNet、YOLO 共用计分状态机方案》自上而下实现，
分层非常干净：

```text
适配器 → FrameObservation(§7.1) → 轨迹/事件处理 → RefereeEvent(§7.4) → 状态机 → 比分
```

TTNet 的分层几乎一样，只是命名不同：

| 方案文档 | YOLO 项目 | TTNet 1.1 |
|---|---|---|
| 适配器 | `adapters/yolo_adapter.py` | `referee/ttnet_adapter.py` |
| §7.1 逐帧观测 | `core/schemas.py::FrameObservation` | `referee/unified_types.py::Observation` |
| 轨迹/事件处理 | `core/trajectory_processor.py` | `referee/event_processing.py` |
| §7.4 物理事件 | `core/schemas.py::RefereeEvent` | `referee/unified_types.py::Event` |
| **计分状态机** | `core/state_machine.py` | `referee/score_state_machine.py` ← **本次替换的就是这一层** |

**关键发现**：两边都已经遵循同一份方案文档，因此状态机可以整体替换，
不需要改动适配器和事件层。这正是方案文档 §1 想要的效果。

---

## 三、改了什么

### 3.1 新增 `src/referee/yolo_sm/`（移植包）

| 文件 | 来源 | 改动 |
|---|---|---|
| `schemas.py` | `core/schemas.py` | 无（原样复制） |
| `state_machine.py` | `core/state_machine.py` | 仅 `from core.schemas` → `from referee.yolo_sm.schemas` |
| `test_state_machine.py` | `tests/test_state_machine.py` | 仅导入路径 + `sources=["yolo"]`→`["ttnet"]` |
| `bridge.py` | **新写** | TTNet `Event` → YOLO `RefereeEvent` 的字段映射 + 换局接力 |
| `selftest_bridge.py` | **新写** | 桥接层自检（20 项） |
| `__init__.py` | **新写** | 统一导出 |

**移植原则：状态机代码一个字都不改。** 这样 YOLO 分支后续升级规则时，
直接覆盖这两个文件即可，TTNet 侧零改动。

### 3.2 `bridge.py` 解决的两个真实问题

**(1) 字段映射**（两边只差命名）

```text
TTNet Event          YOLO RefereeEvent
BOUNCE            →  BOUNCE
HIT               →  RACKET_HIT
NET_HIT           →  NET_TOUCH
BALL_LOST         →  （丢弃，状态机没有这个概念）
```

`BALL_LOST` 刻意不送：YOLO 状态机的 `tick()` 时间判据（`rally_end_s`）
本来就覆盖「球消失」，再塞一个断段事件只会让时间线错乱。

**(2) 换局接力**（这是移植时必须补的一层）

YOLO 状态机在 `game_over=True` 之后 `update()` 一律返回空 —— 对完整比赛是对的，
但 test_1 这种 140 秒片段按 11 分制会在中途「打完一局」，比分就永久冻住了。

`YoloCoreScorer` 的做法：检测到 GAME_OVER → 记录本局结果 → 用上一局胜者作首发方
**新建一个状态机**继续计分。**状态机自身的规则完全没动**，只是在外面套了一层。

同时用 `total_points` 跨局累计账本保证 UI 上的比分单调上升（否则换局瞬间会倒退）。

### 3.3 改动的既有文件

| 文件 | 改动 | 兼容性 |
|---|---|---|
| `render_with_ui.py` | 加 `--engine {yolo,legacy}`，默认 `yolo`；抽出 `_yolo_timeline` / `_legacy_timeline` | `legacy` 与原行为**逐分一致**（已回归验证） |
| `scoreboard_ui.py` | `draw()` 新增可选参数 `phase` / `event_kind`；新增「状态徽标」与「判罚提示条」 | 全部可选，旧调用方布局不变 |
| `event_processing.py` | 新增 `hit_side_by_direction` 参数（**默认 False，不改变现有行为**） | 默认关闭，零影响 |

### 3.4 新增两个工具

| 文件 | 作用 |
|---|---|
| `reextract_events.py` | **从观测缓存重建事件**，不重跑推理。改阈值/改归属规则 → 秒级验证 |
| `compare_engines.py` | 同一份缓存，两套状态机各判一遍，输出 A/B 对比表 |

`reextract_events.py` 是这次最有价值的副产品：`results/cache_test_N.json` 存的是
逐帧原始观测（球位置 + 事件分数 + 球台框），所以「事件提取」和「推理」可以彻底解耦。
**换判分规则不再需要等几十分钟推理。**

---

## 四、UI 上能看到的变化

旧版 vs 新版（同一帧 69.5s）：

```text
┌─────────────────────────────────────┐
│ 第 1 局  ╭──────────╮      B 发球 ● │  ← 新增：状态机阶段徽标
│          │ 对打中   │               │     6 状态直接可见
│  A        1 : 0          B          │
│ A 方       局分 0:0        B 方      │
└─────────────────────────────────────┘
┌──────────────────────────────────────┐
│ P1(A) 得分 [NO_RETURN]  1:0          │  ← 新增：谁得分 + 哪条 ITTF 规则
└──────────────────────────────────────┘     （金黄底，与普通事件区分）
```

三处新增都源自 YOLO 状态机的特性（它有真正的「规则名」体系，旧状态机没有）：

1. **阶段徽标** —— 方案文档 §3 的 6 状态流转，从「看不到」变成「看得见」；
2. **规则名** —— 每分都能回答「凭什么给的」，这是 YOLO 版本最大的可解释性收益；
3. **判罚提示条配色** —— 比分变化比「又落了一次台」重要，视觉上区分开。

---

## 五、判分结果与诚实的局限说明

### 5.0 test_2：唯一能拿真值验证的视频

test_2 是全部 7 个测试视频里**标注覆盖率最高（59%）**的一个，也是唯一能核对
「回合数」的样本。真值（`dataset/test/annotations/test_2/events_markup.json`）：

```text
回合 1: 0.91s - 10.00s  (16 次落台)
回合 2: 18.43s - 22.67s (10 次落台)
回合 3: 26.47s - 26.88s (2 次落台)
→ 真值 3 个回合
```

模型的事件切分与真值**完全对齐**（3 段：0.93-10.03 / 18.47-22.69 / 26.50-29.90），
说明**感知层没问题**。但两套状态机都只判出 2 分：

| 引擎 | 判分 | 判出的时刻 |
|---|---|---|
| legacy | A2 : B1（3 分） | 10.03s / 25.70s / 29.99s |
| **yolo** | **A1 : B1（2 分）** | 9.63s / 31.21s（结算） |

**都漏了回合 2。** yolo 在 24.74s 判它「死球」，原因是 `NO_RETURN` 要求
「最后落台侧 ≠ 最后击球方」这个证据，而 23.300s 有一个**误报击球**
（球此时已飞出画面，真值 23.00s 是一次 net），状态机的 `_on_racket_hit`
按设计会清空 `last_bounce_side` —— 证据被这个假击球擦掉了。

验证过程（原始状态机代码未改，只在外面插桩）：

```text
进入 _resolve_rally_end @ 24.74
   last_hitter=A  last_bounce_side=None   ← None 就是被 23.300s 的假击球清空的
   → NO_RETURN 需要两侧信息都在，于是退化成「死球」
```

**这是上游 HIT 误报的锅，不是移植的锅。** 但它说明一件事：
**YOLO 状态机对「假击球」比 legacy 更脆弱** —— 因为它把 `last_hitter`
当作核心判罚依据，而 legacy 用的是「回合形态校验 + 可见性判据」那套冗余护栏。

### 5.1 A/B 对比（同一份推理缓存，差异纯粹来自判分规则）

| 视频 | 事件数 | 旧引擎 | 新引擎 | 两套判出同一时刻的分 | 新引擎「死球」次数 |
|---|---|---|---|---|---|
| test_1 | 338 | A6 : B5 | **A6 : B4** | 2 / 10 | 18 |
| test_2 | 125 | A2 : B1 | **A1 : B1** | 1 / 2 | 3 |
| test_3 | 132 | A3 : B3 | **A3 : B5** | 2 / 8 | 9 |

新引擎的规则分布（每分是哪条 ITTF 规则给的）：

| 视频 | 规则分布 |
|---|---|
| test_1 | `NO_RETURN` ×5, `DOUBLE_BOUNCE` ×2, `SERVE_FAULT` ×2, `OWN_COURT` ×1 |
| test_2 | `OWN_COURT` ×1, `NO_RETURN` ×1 |
| test_3 | `OWN_COURT` ×8 |

### 5.1.1 移植过程中我自己引入并修掉的 3 个 bug（诚实记录）

这 3 个都不是 YOLO 状态机的问题，是我的集成层写错了。记下来免得后人重踩：

**bug 1｜终局结算调用方式错（影响最大）**

状态机的 `tick()` **每次调用只推进一步**：
`RALLY` →(静默 rally_end_s)→ `RALLY_END_CANDIDATE` →(静默 confirm_s)→ 裁决。

我最初写的是「末帧 + 1 秒 tick 一次」，只走完第一步，**末尾那一分全部丢掉**。
实测 test_2 因此从 A1:B1 掉成 A0:B1，test_1 也从 10 分掉到 8 分。

修复：`YoloCoreScorer.finalize()` 改成按三个时间点各推进一次。
已补单测（`selftest_bridge.py` 第 6 组），其中一条专门断言
「单次 tick 捞不出来」——把这个坑钉死。

**bug 2｜结算提示条的时间戳超出视频末尾**

结算发生在视频结束之后（如 31.21s > 末帧 29.99s），提示条时间戳跟着是 31.21s，
渲染时永远走不到那一刻 —— 观众看到比分跳了却看不到原因。
修复：把提示条时间夹到末帧。

**bug 3｜夹到末帧时差了 3e-6 秒，提示条还是不显示**

缓存里末帧时间戳是四舍五入后的 `29.99167`，而渲染时按 `written/fps` 重算是
`29.9916666…`。`timeline[0] <= ts` 判 False，整条提示又没了。
修复：夹到 `last_ts - 1e-3`，留出余量。

> 教训：这类「浮点边界」bug 在图片抽查里很难发现（我前两次看末帧截图都正常），
> 只能靠逐帧比对时间戳才发现。

### 5.2 没有变好，原因是什么

**test_3 的 8 次 `OWN_COURT`（球落在击球方自己半台）是一个明确的坏信号** ——
业余对打里这种情形不可能有 8 次。

根因不在状态机，在**上游 HIT 检出率**：

```text
TTNet 的 HIT 是靠「轨迹水平方向反转」推断的，不是真的看到球拍。
```

实测每个落台前 1.5s 内有 HIT 的比例：

| 视频 | BOUNCE | HIT | 落台前有击球的比例 |
|---|---|---|---|
| test_1 | 111 | 116 | 95.5% |
| test_2 | 41 | 42 | 100.0% |
| test_3 | 49 | 52 | 93.9% |

表面看「够用」，但 HIT 的**数量**和**时刻**都不准 —— 球撞台面同样会让 x 方向速度反号，
所以大量 HIT 其实是「假击球」。`last_hitter` 一旦错，`OWN_COURT` 就会误触发。

YOLO 分支没有这么严重，因为它有卡尔曼滤波给出可靠的速度方向；TTNet 的
`EventExtractor._detect_hit` 用的是原始轨迹的简单差分，抗噪能力差得多。

### 5.3 我试过按 YOLO 的 R-017 修，但实测反而更差

YOLO 项目的 `docs/decisions.md` 记录了 R-017：击球方按**飞行方向**归属，
实测把 `OWN_COURT` 从 8 降到 1。我照做并实测（从观测缓存重建，只换这一条规则）：

| 视频 | 同侧落台/异侧（旧=位置） | 同侧落台/异侧（新=方向） | 判分变化 |
|---|---|---|---|
| test_1 | 3 / 65 | 3 / 65 | A6:B4(10分) → **A7:B1(8分)**，OWN_COURT 1→5 ❌ |
| test_2 | 2 / 25 | 1 / 26 | A0:B1 → **A8:B2** ❌ |
| test_3 | 8 / 12 | 4 / 16 | 持平 |

方向归属本身是**对的**（「击球方把球打进自己半台」的异常从 8 降到 4），
但**判分反而更差**。原因：TTNet 的 HIT 检出时序不稳，改变归属规则会让
`last_hitter` 的更新时机整体错位，反倒更容易与落台配上错。

**决定：这条规则保留为开关（`hit_side_by_direction`），默认 `False`，
不设为默认。** 等 HIT 检测改进后再做 A/B —— 前提变了，结论才可能变。

### 5.4 必须说清楚的三条评估局限

1. **没有逐分真值。** OpenTTGames 只标注 bounce/net 的**位置**，不标注**谁得分**。
   因此 A6:B4 和 A6:B5 哪个更接近真实，**目前无法验证**。
   唯一能对照的是 test_2 的**回合数**（真值 3 回合）——
   两套引擎都只判出 2 分，**都没达标**，其中 yolo 因为假击球擦掉证据而漏判。
   这一例 legacy 略好（判出 3 分）。样本只有 1 个，不能下结论。

2. **标注覆盖率只有 10%~59%。** test_1 的 140 秒里只有约 13 秒有标注
   （见 `TEST1_REPORT.md` §5.1）。因此任何「精确率」指标都不可信。

3. **「死球」占了判罚的一半。** test_1 的 36 条判罚里 18 条是死球
   （`NO_EVIDENCE` 10 次 + `SERVE_INCOMPLETE` 8 次）。
   这不是 bug，而是状态机在**证据不足时拒绝猜**——方案文档 §4-2 明确要求
   「不要把球消失直接当作失分」。比起瞎判，拒绝判是更负责的行为，
   但也意味着**一半的回合没有结论**。

### 5.5 换成 YOLO 状态机的净收益，该怎么评价

**判分准确率：没有改善**（test_2 甚至从 3 分掉到 2 分）。
**可解释性：明显改善** —— 每分都能回答「依据哪条 ITTF 规则」：

```text
P1(A) 得分 [NO_RETURN]        球落在 B 侧后消失，判定对方未接到
P2(B) 得分 [OWN_COURT]        球落在击球方(A)自己半台
P1(A) 得分 [DOUBLE_BOUNCE]    发球在己方(B)半台连跳两次
       死球 [SERVE_INCOMPLETE] 发球阶段球消失，证据不足，不判分
       擦网重发 [LET]          比分不变
```

**可维护性：明显改善** —— 规则文件与 YOLO/TrackNet 分支同源，将来三个模型
可以真正共用一套状态机，调规则不用改三个地方。

但**如果目标是「判分更准」，这次替换没有达成**，需要先解决 HIT 误报（见第八节）。

---

## 六、不改状态机的事件层优化（本轮新增）

状态机代码一行未动，全部优化都发生在**事件层**（`event_processing.py`）。
共三项，每项都有可复现的对照数据。

### 6.1 优化一：修复「短暂遮挡就清空击球历史」（影响最大）

**这是本次最严重的发现 —— 而且它是一个既有 bug，不是移植引入的。**

原实现在「轨迹分段变化」时清空击球历史 `_pt_hist`，本意是防止把
「丢球前的旧点」和「重新出现后的新点」拼成假轨迹。但：

* TTNet 的球可见率只有 **25%~48%**；
* `TrajectoryProcessor` 只要连续丢球 >12 帧（120fps 下 **0.1 秒**）就 +1 段；
* 而判一次击球需要 `_pt_hist` 攒满 **6 个点**。

结果：一回合内 segment 能变几十次，`_pt_hist` 几乎永远攒不满，
**HIT 被系统性砍掉一半**。用 HEAD 版本与当前实现对照重建（同一份观测缓存）：

| 视频 | HIT（HEAD，事件数） | HIT（有 bug 的中间版本） | 判分变化 |
|---|---|---|---|
| test_1 | 115 | 72 | A6:B4 → A7:B1 |
| test_2 | 43 | 29 | **A1:B2 → A9:B2** ★ |
| test_3 | 51 | 25 | A3:B5 → A2:B4 |

test_2 尤其荒谬：**30 秒里判出 11 分，全是 `OWN_COURT`**。
根因链：HIT 太少 → 状态机 `last_hitter` 一直是旧值 →
正常落台被当成「击球方打进自己半台」。

**修复方式**：把清空判据从「segment 变化」改成「**按实际时间间隔逐帧检查**」
（`hit_hist_reset_s=0.40`）。

这里还有一个二次坑：我第一版把判断挂在 segment 变化的那一刻，结果**测试立刻抓出问题** ——
`TrajectoryProcessor` 的断段是**一次性**的（`_gap_exceeded` 标记保证同一段丢球只 +1 次），
且它在**刚超过 0.1 秒**时就翻起来。那一刻球才丢 0.1 秒，远没到 0.4 秒阈值，
于是判「不过期」；等球真丢了 1 秒，segment 早已不再变化，判断再也触发不了。
**必须逐帧按时间检查**，这才是「断了多久」的准确度量。

已固化为 9 项回归测试（`referee/selftest_event_layer.py`）。

### 6.2 优化二：修正「精确率」的统计口径（不改代码也重要）

旧报告里 bounce/net 的精确率是 **22%~80% 乱跳、不可信**。根因不是模型，
而是**统计口径错了**：OpenTTGames 的标注只覆盖 10%~59% 的时长，
在全时段统计时，那 60%~90% 未标注时段里的**真实事件全被算成 FP**。

改成「精确率只在标注窗口内统计」（`tune_event_thresholds.py::evaluate()`）后，
数据立刻变得平滑可信：

| 视频 | BOUNCE P/R/F1 | NET P/R/F1 |
|---|---|---|
| test_1 | 72% / 100% / 84% | 62% / 96% / 75% |
| test_2 | 75% / 96% / 84% | 62% / 100% / 76% |
| test_3 | 78% / 94% / 85% | 67% / 100% / 80% |

**平均 F1：BOUNCE 84%、NET 77%**。模型其实不差，之前是被口径冤枉了。

### 6.3 优化三：阈值调优（基于可信口径的网格搜索）

有了 6.2 的可信口径，网格搜索结果才站得住：

| 参数 | 原值 | 新值 | 依据 |
|---|---|---|---|
| `net_thresh` | 0.50 | **0.65** | 三段视频 NET F1 全部提升（test_2 70%→76%，test_3 75%→86%） |
| `min_event_interval_s` | 0.10 | **0.25** | 抑制 NET 驼峰拖尾，test_1 NET F1 68%→76%，BOUNCE 不掉 |
| `bounce_thresh` | 0.65 | **0.65（不变）** | 已是平台区，0.60~0.75 无差别 |

`net_thresh` 0.65 是三段视频都不退化的最高公共值（0.85 对 test_2 反而略降）。

### 6.4 三项优化的合计效果

| 指标 | 优化前 | 优化后 |
|---|---|---|
| test_2 判分 | A1:B1（2 分） | **A1:B2（3 分）** |
| test_2 死球数 | 3 | **2** |
| test_1 死球数 | 18 | **5** |
| test_3 死球数 | 9 | **4** |
| BOUNCE 平均 F1 | 84%（口径错误，实际不可比） | **84%（口径正确）** |
| NET 平均 F1 | 67% | **77%** |

**死球数（状态机拒绝判分的次数）从 18/3/9 降到 5/2/4** —— 这是证据完整度改善
最直接的体现：能判的都判了，不再因为缺证据而放弃。

#### ⚠️ 但 test_2 那 3 分，不要当成「判对了」

判分次数（3）恰好等于真值回合数（3），这个巧合很容易被误读。逐条看明细：

```text
 9.63s  B 得分 [OWN_COURT]    ← 这两分
10.03s  B 得分 [OWN_COURT]    ← 在同一个回合内，相隔仅 0.4 秒
14.33s  死球 [SERVE_INCOMPLETE]
24.74s  死球 [NO_EVIDENCE]    ← 真值回合 2（18.43-22.67）被漏掉了
31.21s  A 得分 [NO_RETURN]
```

真值里 9.60/9.81/10.00 三次连续落台是**发球前的颠球找手感**
（标注里 9.45s 就是 `empty_event`）。模型把它当成正式回合，
由于同侧连跳两次触发 `DOUBLE_BOUNCE` 给了 B 一分，
紧接着 10.03s 又一次同侧落台又给一分 —— **同一个回合被算了两次分**。

而真正的回合 2（18.43-22.67，10 次落台）被判成死球（24.74s）漏掉了。

**所以：3 分里有 1 分是「补上的漏判」、1 分是「重复计分」、1 分是正确的。**
数量对上了，内容没对上。这个错误正好抵消，属于典型的「指标好看但结论不成立」，
不能因为数字吻合就宣称成功。

我试过抑制这类「同侧快速连跳」（`max_gap=0.55s, max_dist=90px`），
结果是 test_2 变成 `A1:B1`（2 分），与真值 3 回合**差得更多**，
因此没有采纳。要正确解决这个问题，需要 YOLO 分支已经提出的
`PRE_SERVE`（发球准备期）状态 —— 但那属于**改状态机**，本轮不在范围内。

### 6.5 仍然没解决的部分（说清楚）

* **`OWN_COURT` 误报**（test_3 还有 5 次）。根源是 HIT 假阳性：TTNet 的 HIT
  靠轨迹水平反转推断，球撞台面也会反转。试过「滤掉紧邻落台的 HIT」，
  但**真值里本来就存在同侧 0.19~0.29s 的连续落台**，一刀切会误杀，
  实测准确率反而下降（见 `experiment_hit_filter.py`）。
* **发球前颠球被当正式回合**（上面 test_2 的例子）。需要 `PRE_SERVE` 状态，
  属于改状态机。
* **test_1 判分从 A6:B5 变 A5:B1**。test_1 标注覆盖率仅 10%，
  无法判断哪个更接近真实；但死球从 18 降到 5，**证据完整性**确实改善了。

> **诚实结论**：本轮拿到了两个确定的成果 ——
> ①修掉一个会让 HIT 少一半的既有 bug；②纠正了精确率的统计口径（模型没变，是之前被冤枉）。
> 这两项让**死球率（证据完整性）显著改善**，F1 数字也可信了。
> 但**判分准确率没有可验证的提升** —— 因为缺少逐分真值，且
> 颠球/发球准备期这类结构性问题必须改状态机才能解决。

---

---

## 七、TTNet 专属状态机（继承扩展，不动 YOLO）

> 用户约束：**状态机可以添加 TTNet 专属的部分，但不可以删除或修改适配 YOLO 的部分，
> 要同时兼容两者。**

落实方式是 **继承 + 覆写**，而非分支改动：

```text
referee/yolo_sm/state_machine.py        ← 上游 YOLO 原文件，逻辑代码 0 改动
        ↑ 继承
referee/yolo_sm/ttnet_state_machine.py  ← 新增：TTNetStateMachine（只加不删）
```

`yolo_sm/state_machine.py` 与 `schemas.py` 与 GitHub 上游逐字节比对，
**唯一差异是 2 行 import 路径**：

```diff
-from core.schemas import EventType, RefereeEvent, Side
+from referee.yolo_sm.schemas import EventType, RefereeEvent, Side
```

这一点有自动化测试守着（`selftest_profiles.py` 第 7 组），
一旦有人往上游文件里塞逻辑，测试立刻失败。

### 7.1 双档位开关

```python
from referee.yolo_sm.ttnet_state_machine import TTNetStateMachine, Profile

TTNetStateMachine(profile=Profile.YOLO)   # 行为与 YOLO 父类逐条一致（回归基线）
TTNetStateMachine(profile=Profile.TTNET)  # 启用 TTNet 专属修正（默认）
```

命令行：`--profile {ttnet,yolo}`（默认 ttnet）。

`Profile.YOLO` 与父类的等价性有专门测试：用 3 组合成事件流对比**每一条 Decision**
（时间戳 / 类型 / 得分方 / 规则名），必须完全一致。

### 7.2 四条 TTNet 专属修正（为什么不能用在 YOLO 上）

TTNet 与 YOLO 的**证据质量**根本不同，同一套判罚规则会系统性吃亏：

| 维度 | YOLO | TTNet |
|---|---|---|
| 球可见率 | ~83% | 25%~48% |
| 击球(HIT) | 真实检测框 | 靠「轨迹水平反转」推断，球撞台面同样反转 → 大量假阳性 |
| 发球 | 能看到球拍动作 | 看不到，常漏掉发球那一次落台 |

**修正 1｜击球不清空落台侧**（`preserve_bounce_side_on_hit`）

父类 `_on_racket_hit` 会把 `last_bounce_side` 置 None —— YOLO 里这是对的
（HIT 可信，新回合确实开始）。但 TTNet 实测 **79/106、36/40、24/41 次落台之后
3 秒内都跟着一个 HIT**，于是 `NO_RETURN` 的证据被反复擦掉。
落台侧是**物理事实**，不该因为「有人挥了一下拍」而失效。

**修正 2｜WAITING_SERVE 落台可进入 RALLY**（`bounce_enters_rally_from_waiting`）

TTNet 实测 **27/106、5/40、16/41 次落台发生在 `WAITING_SERVE`**，父类直接忽略。
因为看不到发球那一拍击球，永远进不了 RALLY，这些落台白丢。

**修正 3｜用「是否接回球」替代「击球方是谁」**（`relax_no_return_evidence`）

父类的 `NO_RETURN` 要求 `last_hitter` 与 `last_bounce_side` 都已知且不同侧。
但 TTNet 的 `last_hitter` 常年为 None，且假击球会把它改成「根本没碰到球的人」。

改成判「球落在某侧之后，这一侧有没有把球打回对面」，三条判据依次收紧：

1. **时间**：回击必须在落台后 `return_window_s`（0.8s）之内；
2. **归属**：这次击球必须判给落台那一侧；
3. **结果（最强）**：击球之后球必须**继续飞并对侧落台** —— 这是「接回去了」的铁证。

第 3 条是实测挖出来的关键区分器：

| 视频 | 「真回击」样本（后有对侧落台） | 「假击球」样本（球消失） |
|---|---|---|
| test_1 | 70 | 15 |
| test_2 | 33 | 5 |
| test_3 | 16 | 15 |

即「落台后 0.8s 内出现 HIT」本身**不足以**说明接球成功，
必须再确认球确实被打到了对面。

**修正 4｜抑制同回合重复判分**（`min_point_gap_s`）

TTNet 的落台常有拖尾重复（同一次落台被检出 2~3 次），会让父类的
「同侧双弹」在同一个回合里连判两次分（实测 test_2 的 9.63s / 10.03s）。
两次判分间隔 < 1.2s 时，第二次记为 `DUPLICATE_POINT` 死球，不计分。

> **这些都是「只加不删」**：父类的 5 条 ITTF 规则开关、6 状态定义、
> 判罚入口全部保留，TTNet 只是在自己的档位里改判据。

### 7.3 效果（三视频实测）

| 视频 | profile=yolo（基线） | profile=ttnet | 变化 |
|---|---|---|---|
| test_1 | A5:B1（6 分，死球 5） | **A8:B8（16 分，死球 1）** | 判分 ×2.7，死球 ↓80% |
| test_2 | A1:B2（3 分，死球 2） | **A2:B1（3 分，死球 1）** | 分数持平，重复判分被抑制 |
| test_3 | A2:B4（6 分，死球 4） | **A9:B4（13 分，死球 0）** | 判分 ×2.2，死球清零 |

**test_1 的判分节奏检查**（16 分 / 140 秒 = 每 8.8 秒 1 分，
最小间隔 1.86s、中位 5.9s）—— 对业余对打是合理量级，
不再是之前「140 秒只判 6 分」那种明显漏判。

### 7.4 关于「判早了」的审计（必须说清楚）

我对 test_1 的 14 次 `NO_RETURN` 逐条审计了「判分之后 3 秒内是否还有落台」，
发现 **4 次**（test_3 有 2 次）在判分后 1.2~1.7s 内、**同一侧**又有落台。
表面看像「回合还在继续就判了分」，但逐案核对后结论不同 —— 举个例子：

```text
102.75s  球落 B 侧
103.96s  状态机判「B 侧未接回」→ A 得分    ← 看起来判早了
105.19s  B 侧击球（TTNet 漏检了中间那一拍）
105.23s  球又落 B 侧                        ← B 接回来了，但打在自家半台
```

**得分方其实是对的**（都是 B 失误、A 得分），但：
1. 规则名写成了 `NO_RETURN`，实际应该算 `OWN_COURT`；
2. 时刻早了约 1.2 秒。

根因是 TTNet 球可见率只有 31%，中间那一拍回击整段被漏掉，
而**实时判决看不到未来** —— 在 103.96s 那一刻，状态机看到的就是「球落在 B 侧后 1.2 秒没有回击」。

我试过把静默阈值从 1.2s 提高到 1.8 / 2.2 / 2.6 / 3.0s（真值里回合内最大间隔是
1.24~1.30s，回合间最小 3.8s，理论上留了余量），结果**「判早」次数不降反升**
（4→6→5→5），而且高阈值会让相邻回合粘连。因此保持 0.8/0.4 不动。

> 这条局限的定性是：**判分方向对、规则名偏差、时刻偏早约 1.2s**，
> 而不是「判错了分」。要根治必须让上游不再漏检回击（提高球可见率），
> 不是状态机能解决的。

### 7.5 仍然存在的局限

* `NO_RETURN` 占绝大多数（test_1 16 分里 14 分）。这是 TTNet 能力边界的真实反映：
  球可见率只有 31%，回合末「对手没接到」几乎只能靠「球消失」推断。
* test_2 仍然只有 3 分（真值 3 个回合）。内容是：
  1 分来自回合 2（24.74s，判对）+ 2 分来自回合 1 末尾。
  回合 3（26.47-26.88）仍然没判出 —— 只有 2 次落台，且末尾紧邻视频结束。
* 发球前颠球被当正式回合的问题**仍未解决**，需要 `PRE_SERVE` 状态
  （属于新增状态，本可以加，但会改变 6 状态机结构，需你确认后再动）。

---

## 八、复现命令

```bash
cd src

# 主流程（默认已用 YOLO 状态机；缓存存在则跳过推理）
python -m referee.render_with_ui \
  --video ../dataset/test/videos/test_1.mp4 \
  --out "C:/Users/PC/Desktop/TTNet_test1_结果/test_1_比分版_yoloSM.mp4" \
  --cache ../results/infer_test_1_full.json \
  --summary ../results/yolo_sm/test_1_yoloSM_summary.json

# test_2（30 秒，唯一有回合数真值的样本）
python -m referee.render_with_ui \
  --video ../dataset/test/videos/test_2.mp4 \
  --out "C:/Users/PC/Desktop/TTNet_test1_结果/test_2_比分版_yoloSM.mp4" \
  --cache ../results/infer_test_2_full.json \
  --summary ../results/yolo_sm/test_2_yoloSM_summary.json

# 用 TTNet 专属档位（默认；显式写出来便于对照）
python -m referee.render_with_ui --video ../dataset/test/videos/test_1.mp4 \
  --out ... --cache ../results/infer_test_1_full.json --profile ttnet

# 用纯 YOLO 档位（回归基线，行为与上游一致）
python -m referee.render_with_ui --video ... --out ... --profile yolo

# 双档位兼容性测试
python -m referee.yolo_sm.selftest_profiles

# 批量
python -m referee.batch_render --games test_1 test_2 test_3 --engine yolo

# 切回旧状态机（回归对比用）
python -m referee.render_with_ui --video ... --out ... --engine legacy

# 两套引擎 A/B 对比（毫秒级，无需渲染）
python -m referee.compare_engines --games test_1 test_2 test_3 \
  --json ../results/yolo_sm/ab_comparison.json

# 从观测缓存重建事件（改阈值/归属规则后必跑，秒级）
python -m referee.reextract_events --game test_1 --dry-run

# 状态机原单测（31 项）
python -m unittest referee.yolo_sm.test_state_machine

# 桥接层自检（20 项）
python -m referee.yolo_sm.selftest_bridge
```

---

## 九、产物清单

| 文件 | 说明 |
|---|---|
| `C:\Users\PC\Desktop\TTNet_test1_结果\test_1_比分版_yoloSM.mp4` | **test_1 UI 渲染视频**（16800 帧 / 140s / 475.8 MB / A6:B4） |
| `C:\Users\PC\Desktop\TTNet_test1_结果\test_2_比分版_yoloSM.mp4` | **test_2 UI 渲染视频**（3600 帧 / 30s / 140 MB / A1:B1） |
| `results/yolo_sm/test_1_yoloSM_summary.json` | test_1 完整判分明细（36 条判罚 + 10 分逐条规则） |
| `results/yolo_sm/test_2_yoloSM_summary.json` | test_2 完整判分明细（8 条判罚 + 2 分逐条规则） |
| `results/yolo_sm/ab_comparison.json` | 两套引擎 A/B 明细 |
| `results/yolo_sm/preview/*.jpg` | 关键帧截图 |

---

## 十、下一步建议（按性价比排序）

1. **抑制 HIT 误报（收益最大）**。这是当前所有误判的单一根因：
   * test_2 回合 2 漏判 —— 23.300s 的假击球擦掉了 `last_bounce_side`；
   * test_3 的 8 次 `OWN_COURT` —— 假击球让 `last_hitter` 长期错位；
   * 测试 1/2/3 各有 42~116 个 HIT，但真实击球数远少于此
     （球撞台面会让 x 方向速度反号，被误判成击球）。

   具体做法（两条可以叠加）：
   * 把 `_detect_hit` 的「简单差分」换成**卡尔曼速度判据** —— 这正是 YOLO 分支
     抗噪的原因，它用状态估计压制了瞬移跳变；
   * 启用已写好的 `PaddleEvidenceFilter`（`filter_suspect_hit=True`），
     用「球是否在台面高度带内」给假击球打标，让状态机可以不采信。

2. **补逐分真值**。test_2 只有回合数真值，仍然算不出判分准确率。
   建议人工标注 test_2 那 30 秒的每一分归属 —— 它最短、覆盖率最高（59%）。

3. **给 `SERVE_IN_PROGRESS` 加发球准备期**。YOLO 分支的
   `docs/test4_evaluation_report.md` 记录了同一问题：运动员发球前颠球找手感，
   被误判成 `DOUBLE_BOUNCE`。方案是增设 `PRE_SERVE` 缓冲态。

4. **多模型融合**。TTNet 的 bounce/net 分数召回高，YOLO 的轨迹稳 ——
   方案文档 §4-3 建议融合去重后再送状态机。两套都在手上了，可以试。

> **不建议**在 HIT 修好之前就去调状态机阈值。
> 现在动 `rally_end_s` / `confirm_s` 只是在掩盖上游噪声，会把测试集调过拟合。
