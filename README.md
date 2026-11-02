# 乒乓球自动裁判 · 统一观测契约与联调层

三人组课题：**TTNet / TrackNet / YOLO** 三路球追踪，各自出自不同模型，但要共用同一套事件层与判罚状态机。

这个仓库装的是**三路共用的那一层**——接口契约、各路的适配器、以及把观测接到判罚状态机上的联调代码。
模型训练代码也在本仓库（`TrackNet/`），与上游的关系见 [与上游的关系](#与上游的关系)。

- 契约定义 → [`contract/`](contract/README.md)（**改接口先看这里**）
- 当前进度 → [当前状态](#当前状态)
- 测试结论 → [`视频判罚测试报告.md`](视频判罚测试报告.md)

---

## 目录

```text
.
├── contract/                    三路共用契约（唯一对接面）
│   ├── observation.schema.json     §7.1 Observation 的 JSON Schema
│   ├── event.schema.json           §7.4 Event 的 JSON Schema
│   └── README.md                   字段表、硬规则、变更记录、已知问题
├── adapter/                     TrackNet 一路的适配器
│   ├── tracknet_adapter.py         模型输出 → Observation（视频 / CSV 两种模式）
│   ├── observation.py              构造、校验、读写
│   ├── check_observations.py       观测/事件文件自查
│   ├── eval_mot.py                 MOT 指标评估（py-motmetrics）
│   └── examples/                   可跑通 demo 的样例数据
├── tt/                          状态机联调层
│   ├── run_mixed.py                混合事件源运行器（主力）
│   ├── build_timeline.py           事件/判罚时间线 HTML
│   ├── export_his_racket_hits.py   导出状态机轨迹层的击球事件
│   ├── probe_heatmap.py            热图峰值逐帧探测（诊断用）
│   ├── probe_resolution.py         多分辨率对比（诊断用）
│   ├── shift_events.py             事件流裁时间窗并平移
│   ├── 状态机联调评估.md            交付件评估 + 5 条改进建议
│   └── 事件顺序修复方案.md          事件错位的成因、修复与验证记录
├── TrackNet/                    TrackNet 上游实现 + 我方的乒乓球改造
│   ├── TrackNet/                  实际工作目录（脚本都在这里跑）
│   │   ├── prepare_openttgames.py   数据转换
│   │   ├── detect_bounce.py         落台检测
│   │   ├── events_from_observations.py  Observation → 事件
│   │   └── exps/ datasets/          训练输出与数据（不进仓库）
│   └── calib/                     标定过程截图
├── calib/table_quads.json       球台四角标定（换机位必须重标）
├── docs/otherdata/              方案文档、对接清单等外部资料
└── 四份阶段报告                 见文末「文档索引」
```

---

## 三路分工

| 分支 | 模型 | 产出 | 负责人 |
|---|---|---|---|
| TTNet | 端到端，直接出球位置与事件分数 | Observation（含 `event_scores`） | 组内 |
| **TrackNet** | 三帧输入 → 高斯热图，峰值即球心 | Observation（`event_scores` 全 null） | **本仓库作者** |
| YOLO | 逐帧检测 | Observation（带检测框） | 组内 |

三路都只做到 Observation。事件层与判罚规则共用状态机那一份，谁都不用改判罚逻辑来适配自己的模型。

---

## 快速开始

### 1. 环境

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows bash；PowerShell 用 .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

模型推理另需 PyTorch，版本跟随显卡，不写死在这里。

### 2. 取状态机交付件（**必需**）

`tt/` 依赖组长发的状态机包，本仓库不收录（第三方交付件，不分发）。

```bash
# 拿到 乒乓球裁判状态机_交付件.zip 之后：
unzip 乒乓球裁判状态机_交付件.zip -d tt/_unpacked/
```

`run_mixed.py` 通过 `sys.path.insert(0, tt/_unpacked)` 导入 `core.*` 与 `visualizer.*`，不需要安装。

### 3. 跑一段端到端（自包含，不用视频、不用权重）

```bash
python tt/run_mixed.py \
    --jsonl   adapter/examples/test_2_observations.jsonl \
    --bounces adapter/examples/test_2_events.jsonl \
    --preset  test2 --jitter-merge
```

预期输出：

```text
观测帧数        : 3600
事件统计        : {'RACKET_HIT': 40, 'BOUNCE': 31, 'OUT': 1}
最终比分        : P1(A) 0 : 2 P2(B)
判罚条数        : 3
```

这两份样例就是 test_2 前 30 秒的真实产出，直接复现报告里的比分。

### 4. 从视频重跑（完整链路）

```bash
# a) 推理 → Observation（要权重）
cd TrackNet/TrackNet
python ../../adapter/tracknet_adapter.py \
    --video ../../_data/test_2_1080p_crf22.mp4 \
    --model_path exps/tt_g45_aug/model_best.pt \
    --out ../../outputs/observations/test_2.jsonl --min_conf 0.5

# b) Observation → 落台事件
python events_from_observations.py \
    --obs ../../outputs/observations/test_2.jsonl \
    --game test_2 --event_jsonl ../../outputs/events/test_2_bounces.jsonl

# c) 落台事件 + 状态机轨迹层 → 判罚
cd ../..
python tt/run_mixed.py \
    --jsonl outputs/observations/test_2.jsonl \
    --bounces outputs/events/test_2_bounces.jsonl \
    --preset test2 --jitter-merge
```

`--preset` 只有 `test2` 和 `game4` 两套，对应两个机位的球台四角。**换机位必须重新标定**，否则半台归属会整体翻转且不报错。

---

## 当前状态

两段真实比赛视频，完整链路（观测 → 事件 → 判罚 → HUD）已验证：

| 视频 | 时长 | 机位 | 比分 | 落台 P / R / F1 |
|---|---|---|---|---|
| `game_4_seg470` | 56 s | 训练集同机位 | **1:1** | 1.000 / 0.907 / **0.951** |
| `test_2` | 30 s | 未参与训练的换机位 | **0:2** | 0.677 / 0.750 / **0.712** |

- 共 7 条判罚，逐条对照标注核对过，**无位置或半台判错**；其中 5 条有标注事件支撑，2 条是"证据不足不判分"的死球收尾。
- 标注帧上的模型召回 **77.9%**（±5 帧容差）。全片检出率 30.6% 不代表模型差——标注只覆盖 21.6% 的帧。
- 落台事件不依赖模型私有格式：`events_from_observations.py`（TrackNet 侧）只读 Observation，另两路可直接复用。

判罚事件源的分工：**击球**用状态机内置轨迹层，**落台**用我方检测，**出界**由外层按"最后可见位置"推断补入。
为什么这么分，见 [`tt/状态机联调评估.md`](tt/状态机联调评估.md) 第 6 节。

---

## 已知问题

| 问题 | 性质 | 状态 |
|---|---|---|
| `confidence=null` 被下游当成 `0.35` 硬门限，CSV 路线跑全片零事件 | 契约级，需下游修 | 已绕开（改走视频推理），待上游确认 |
| 擦网 `NET_TOUCH` 单目侧视不可分 | 物理限制 | 四个假设全部实测否定，暂不支持 |
| 球飞出台外后模型彻底丢失（热图峰值 231→**0**，持续 2.84 s） | 模型能力边界 | 已定位，非阈值/分辨率问题；标注同样空白 |
| 远端高球飞出（图像 y 递减）不出 `OUT` | 现有判据的理论缺口 | 两段视频暂无实锤案例 |
| 状态机 `_unpacked` 不在本仓库 | 授权 / 分发限制 | 需向组长索取 |

`confidence=null` 那条值得单独说：方案 §7.1 允许填 null，下游却拿它当门槛用。**这种问题只有把三方的约定放在一起才看得见，也是建这个仓库的主要理由。**

---

## 与上游的关系

`TrackNet/` 是 [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) 的克隆，**现已并入本仓库统一管理**。
并入前确认过：两个克隆都停在上游同一个提交（`730ea17`），**0 个本地提交、无 stash**，所以合并不丢任何历史。
原 `.git` 已移出到仓库外（见文末「仓库合并记录」）。

上游代码上我方的改动与新增，全部在 `TrackNet/TrackNet/` 下：

| 文件 | 作用 | 状态 |
|---|---|---|
| `prepare_openttgames.py` | OpenTTGames 乒乓球数据 → TrackNet 训练格式（含 `run_id` 防跨片段取帧） | 新增 |
| `detect_bounce.py` | 落台检测（规则法） | 新增 |
| `events_from_observations.py` | Observation → 落台事件，**只读统一格式**，另两路可直接复用 | 新增 |
| `calibrate_table.py` | 球台四角标定 | 新增 |
| `gt_gen.py` / `main.py` / `datasets.py` / `general.py` / `test.py` | 乒乓球适配改造 | 修改 |

**不进仓库的**：`datasets/`（5.2 G）、`exps/`（410 M 训练输出）、`weights/`（82 M）、`.venv/`（7.6 G）、`outputs/`（2.6 G）、`*.pt`。

> 上游仓库未附 LICENSE 文件，这里的代码仅用于课程项目。对外分发前要确认授权。

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [`contract/README.md`](contract/README.md) | 契约字段表、硬规则、变更记录 |
| [`adapter/README.md`](adapter/README.md) | TrackNet 适配器用法、confidence 定义、MOT 导出 |
| [`统一观测适配器报告.md`](统一观测适配器报告.md) | 适配器设计与自查结果 |
| [`落台事件检测.md`](落台事件检测.md) | 落台判据、参数、F1 对比 |
| [`tt/状态机联调评估.md`](tt/状态机联调评估.md) | 交付件评估与 5 条改进建议 |
| [`tt/事件顺序修复方案.md`](tt/事件顺序修复方案.md) | 事件错位的成因、修复、验证（含一条被 GT 否决的方案） |
| [`视频判罚测试报告.md`](视频判罚测试报告.md) | 两段视频的逐条判罚核对与指标 |
| [`TrackNet复现报告.md`](TrackNet复现报告.md) | 模型复现过程与训练结果 |
| [`复现仓库评估.md`](复现仓库评估.md) | 三个候选方案的技术选型 |

---

## 仓库合并记录

**2026-11-02：把 `TrackNet/` 并入本仓库。**

`TrackNet/` 与 `TrackNet/TrackNet/` 原本各带一个 `.git`（都是 yastrebksv/TrackNet 的克隆），
根仓库无法跟踪另一个仓库里的文件，于是把两个 `.git` 移到了仓库外：

```text
E:/bc/xm/ppq_git_backup_2026-11-02/TrackNet_outer.git   6.9 M
E:/bc/xm/ppq_git_backup_2026-11-02/TrackNet_inner.git   6.9 M
```

移出前确认过：两者都停在上游同一提交、**0 个本地提交、无 stash**，所以只断开 `git log` / `git diff` 的上游参照，不丢工作。
需要恢复对比时，把备份目录改名回 `TrackNet/.git` 即可；确认不再需要可直接删除。
