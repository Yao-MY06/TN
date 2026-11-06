# 本周计划（2026-11-06 起）：多视频验证 + 两个小工程件

> 给接手实现的 AI：先读 `cascade/README.md`（现状、全部数字和结论），再按本文件做。
> 分支：`feat/cascade`（级联层）与 `fusion/three-way`（= feat/cascade + 三路代码快照），两边保持同步。

## 目标

1. **在更多视频上验证级联**：test_2 只有 2 分，判罚层面看不出 TTNet 复核的作用。把 test_1、test_3～7 跑起来，按视频给出规则复核（C3a）与 TTNet 复核（C3）的对照。
2. **补两个小工程件**：复核超时、待人工确认清单（学长方案 §5-6、§6、§12.7 要求报告超时比例与人工确认比例）。
3. **方案 §7 的离线漏检统计**：用全片 TTNet 分数统计“TTNet 能认出、但提议没覆盖”的事件。

**本周不做**：原始帧环形缓存与钉帧、异步实时队列、人工改分与回合版本、回合结束补充复核、球拍接口。等学长确定要现场演示、或多视频结果显示提议有漏检时再做。

## 约束（必须遵守）

- 不修改 `tt/`、`adapter/`、`TrackNet/`、`third_party/`、`tt/_unpacked/core` 的任何文件；只在 `cascade/` 里加或改。
- 现有测试必须一直通过：TN 根目录 `python -m unittest discover -s cascade/tests -t .`（目前 47 项，2 项需 torch/权重时跳过）。新功能要带测试。
- `--reviewer rule` 在 test_2 上的事件流必须与 `tt/run_mixed.py` 逐字节一致（已有回归测试，不能破）。
- 默认参数不变：提议区间 ±0.1 s、方向突变触发器关、状态机 `tn`、TTNet k=4。**不要为新视频重新调参**（见任务 7 的划分）。
- git：作者与提交者都是 `Yao-MY06 <Yao-MY06@users.noreply.github.com>`，提交说明里**不加** Claude 的署名行；不要 push、不要改写已有历史。
- `pip install ultralytics` 会把 torch 换成 CPU 版，必须加 `--no-deps`（原机已踩过）。

## 任务

机器说明：**A = 原机**（RTX 5060、OpenTTGames 视频与原版标注、TN 目录 `E:\bc\xm\ppq`）；**B = 另一台**（`E:\20-项目\ppq\TN`，有 Extended 标注，无 GPU）。代码任务两台都能做。

### 在 B 机（`E:\20-项目\ppq\TN`）实现时

- **只做 T1–T5**；T6–T8 要视频和 GPU，留给原机。
- 工作目录就是 `E:\20-项目\ppq\TN`，不用从 GitHub 拉。分支纪律：在 `feat/cascade` 上开发、提交，再把每个提交 `cherry-pick` 到 `fusion/three-way`，两边保持同步。仓库本地 git 配置已设成 Yao-MY06，直接 commit 即可。
- 需要 Python ≥ 3.10，带 numpy、opencv-python、pandas；没有 torch 也行（TTNet 测试会自动跳过）。
- 原机首跑的完整产出在 `outputs/cascade_step5/`（不进 git）：两组运行目录（含 `reviews.jsonl`、`ttnet_reviews.jsonl`、`clips.jsonl`、`summary.json`、`eval*.json`）、`k_check_test_2.json` 和原机回执。写 T3/T4/T5 时用它对照字段和预期数字。
- **原机的全片 TTNet 缓存 `_data/ttnet/test_2.jsonl` 还没拷到 B 机。** 没有它时：T3、T5 先用测试里构造的分数验证逻辑；T4 的 rule 组可以在 B 机复现 0.842，ttnet 组的 0.871 和 T3、T5 的真实验收等缓存拷来后再补跑（放到 `TN/_data/ttnet/test_2.jsonl`；YOLO 观测放 `TN/_data/obs/yolo/test_2.jsonl`）。
- 同一时间只让一个 AI 改这个文件夹。

### 代码（任一台，先做）

**T1 新机位：从四角标定构造机位参数**
- 现状：`run_mixed.build_calib(name)` 只认 core 里的 5 个预设（game2 / rally / plos / test4 / test2），外加从 `calib/table_quads.json` 算的 game4。test_1、3、5、6、7 没有预设，级联跑不起来；test_4 有预设但 `table_quads.json` 里没有四角（提议与落台规则要用四角）。
- 做法：新增 `cascade/calib.py`，`build_calib_from_quad(key)` 按 `run_mixed.build_calib` 里 game4 分支的同一取法（net_x、net_zone、roi）构造 `TableCalibration`；`run_cascade.py` 的 `--preset` 不是 core 预设时，回退到用 `--quad`（或由 preset 推出的键）构造。
- 验收：对 game_4，`build_calib_from_quad("game_4")` 与 `run_mixed.build_calib("game4")` 所有字段相等（单测）；预设名照旧走原路径，test_2 回归测试不变。

**T2 复核超时（简版）**
- 每个片段设截止时间 `--review-deadline-frames`（按帧计，默认不启用 = 现在的行为）。片段提交后超过截止仍未完成 → 生成 `status="timeout"` 的 ReviewResult（无事件，原因码 `deadline_exceeded`），提交层的水位线**不再等它**。超时后才到的结果只写进 `late_results.jsonl`，**不提交**给状态机。
- 用现有的 `--review-latency-frames` 模拟慢复核即可测试。
- 验收：延迟 < 截止时，输出与不设截止完全相同（单测）；延迟 > 截止时，相关片段为 timeout、判罚不卡住、迟到结果不入状态机（单测）；`summary.json` 增加 `timeout_count / timeout_ratio`。

**T3 待人工确认清单**
- 每次运行在输出目录写 `review_queue.jsonl`：所有 `uncertain` / `error` / `timeout` 片段，带片段时间范围、原因码、驼峰峰值（若有）、片段前后 1 s 内的判罚。`rejected` 不算。
- `summary.json` 增加 `manual_confirm_count`、`manual_confirm_ratio`（= 清单条数 / 片段总数）、`missing_context_ratio`。
- 验收：用 replay 跑原机已有的 test_2 TTNet 缓存，清单应为 2 条（`s01:c0022` uncertain，原因 `off_table@1216`；`s01:c0042` error）。

**T4 批量运行与汇总表**
- 新增 `cascade/run_batch.py`：读一个清单（每段视频的 TrackNet 观测、YOLO 观测可选、TTNet 缓存、原版 `events_markup.json`、Extended `game_data/test/test_N.json` 可选、quad 键），对每段视频跑 rule 与 ttnet 两组 `run_cascade`，接 `eval_proposals` 与 `eval_extended`，输出 `outputs/cascade_batch/summary.md` 与 `summary.csv`。
- 每段一行：提议召回、覆盖率、TTNet 窗口占比、落台 P/R/F1（两组）、每分得分方正确数 / 总分数（两组）、多判的分、人工确认比例、超时比例。最后一行是 test_1、3～7 的合计（落台按 tp/fp/fn 汇总再算，不是平均 F1）。
- 验收：只放 test_2 时，两组落台 F1 复现 README 的 0.842 / 0.871。

**T5 §7 离线漏检统计**
- 新增 `cascade/eval_ttnet_outside.py`：输入全片 TTNet 缓存（`ttnet_cache` 的输出）、一次运行的 `clips.jsonl`、真值标注。在全片分数上按复核器同一规则（k、驼峰阈值）找驼峰，列出**落在所有片段之外**的驼峰，标出其中与真值落台 ±5 帧匹配的条数——这就是“提议层漏掉、而 TTNet 本可找回”的事件。
- 验收：在 test_2 上能跑出结果（test_2 提议召回 28/28，预期 0 条真值漏检）；接进 T4 的汇总表一列。

### 数据与运行（原机 A）

**T6 标定 test_1、test_3～7 的球台四角**
- `TrackNet/TrackNet/calibrate_table.py export --video <视频> --frame 1200 --out _data/calib/<game>_grid.jpg` 导出带坐标网格的帧；按 far_left、far_right、near_right、near_left 顺序读出原图坐标，照 test_2 的格式写进 `calib/table_quads.json`；再用 `calibrate_table.py draw` 画回去核对。
- 能看图的 AI 可以自己读网格；不能看图就交给人。**四角错了会导致 A/B 整体翻转且不报错**，务必画回去核对。test_4 已有 core 预设，但也要补四角。

**T7 跑全部 7 段视频**
- 把 `cascade/tools/run_machine_a.ps1` 的 YOLO 与 TTNet 缓存两步从只跑 test_2 扩展到对清单里的每段视频都跑（TrackNet 一步已经支持多段）。
- 真值：落台用原机的**原版** `events_markup.json`（在 `_data/openttgames/test_annotations/test_N_extracted/`）；每分与击球用 Extended 标注（B 机 `TN/_data/gt/extended_openttgames/` 拷过来，CC BY-NC-SA 4.0，见其中 `SOURCE.txt`）。
- 用 T4 出汇总表。**划分**：test_2 是调参视频，单列；test_1、3～7 是留出验证，参数一律不改，合计行只算这 6 段。

**T8 交回**
- `outputs/cascade_batch/`（汇总表与各运行目录），`_data/ttnet/test_2.jsonl`（全片 TTNet 缓存），`_data/obs/yolo/test_2.jsonl`（查 YOLO 分歧触发器为何 0 条），以及 test_2 上 7 个误检时刻（2.42 / 3.88 / 5.28 / 6.15 / 8.04 / 9.72 / 20.12 s）前后各 10 帧的截图，供人工判断是真误检还是标注遗漏。

## 顺序与依赖

```text
T1 ─┐
T2  ├─(任一台，可并行)─→ T4 ─┐
T3  │                        ├─→ T7（原机）─→ T8
T5 ─┘        T6（原机）──────┘
```

T6 需要人工核对，可以和 T1–T5 同时开始。

## 完成标准

- T1–T5 代码合入 `feat/cascade` 并同步到 `fusion/three-way`，测试全过，`cascade/README.md` 补上新参数与新输出说明。
- 交回 7 段视频的汇总表（test_2 单列 + 6 段留出合计），重点回答：留出视频上，TTNet 复核相比规则复核，落台 F1 和每分判罚正确数各变了多少；提议层有没有漏掉 TTNet 本可找回的事件。
