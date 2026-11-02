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
│   ├── _unpacked/                 状态机交付件（原样入库，见「状态机交付件」）
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
│   │   ├── exps/tt_*/model_best.pt  四份乒乓球权重（入库，各 41 MB）
│   │   └── datasets/                训练数据 5.2 G（不进仓库）
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

### 2. 状态机交付件（已随仓库提供，无需额外操作）

`tt/_unpacked/` 就是组长下发的 `乒乓球裁判状态机_交付件.zip` 解压后的**原样内容**，
入库时与本机解压结果做过 `diff -r` 逐文件比对，**零差异**，一行未改。clone 下来即可直接跑。

`run_mixed.py` 通过 `sys.path.insert(0, tt/_unpacked)` 导入 `core.*` 与 `visualizer.*`，不需要安装。

> 原始 zip 没有一起入库——内容与 `tt/_unpacked/` 完全相同，避免同一份代码存两份拷贝后逐渐漂移。
> 需要以 zip 为准覆盖时：`unzip -o 乒乓球裁判状态机_交付件.zip -d tt/_unpacked/`

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

> **前置**：模型权重**已随仓库提供**（`TrackNet/TrackNet/exps/tt_g345_aug/model_best.pt`，41 MB，clone 下来就有）。
> 只需另外准备待测视频（`_data/`），获取方式见文末「不进仓库的」。
> **只想验证链路是否通，用第 3 步的自包含 demo 就够**，不需要这些。

```bash
# a) 推理 → Observation（要权重）
cd TrackNet/TrackNet
python ../../adapter/tracknet_adapter.py \
    --video ../../_data/test_2_1080p_crf22.mp4 \
    --model_path exps/tt_g345_aug/model_best.pt \
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

`confidence=null` 那条值得单独说：方案 §7.1 允许填 null，下游却拿它当门槛用。**这种问题只有把三方的约定放在一起才看得见，也是建这个仓库的主要理由。**

---

## 状态机交付件（`tt/_unpacked/`）

判罚规则来自组长下发的 `乒乓球裁判状态机_交付件.zip`（2026-10-30 收）。已解压入库，**原样保留、一行未改**。

| 项 | 值 |
|---|---|
| 位置 | `tt/_unpacked/`，13 个文件 / 318 KB |
| 内容 | `core/`（几何、单应性、schema、状态机、轨迹层）、`visualizer/broadcast_hud.py`、`run_state_machine.py`，以及包内自带的 `README.md`、`requirements.txt`、`examples/` 样例数据 |
| 完整性 | 入库时与本机 zip 解压结果 `diff -r` **零差异**；zip 的 SHA-256 为 `ba9f3b21…f9efcae` |
| 版本 | 2026-10-30 版。内置预设为 game2 / rally / plos / test4 / test2，**不含 game_4**，片段的球台四角由我方在 `run_mixed.py` 里另加 |

**为什么不改它。** 对外的四个干预——落台换源、注入延迟、出界推断、发球守卫——全在 `tt/run_mixed.py`，状态机文件未动。理由有三：对方能独立升级，我这层不用跟着改；判错时能立刻分清是规则问题还是检测问题；对方拿自己那份代码就能复现我的结果，不必先同步我的补丁。唯一伸进内部的操作是调用它自己的 `_reset_rally(keep_server=True)`（`core/state_machine.py:517`），属正常调用私有 API，不是打补丁。

**`OUT` 不是新造的类型。** 状态机原生支持——`core/state_machine.py:421` 的 `_on_out()`，发球阶段判"发球出界"、对拉阶段判 `opponent(last_hitter)` 得分，判给谁由它算。外层只回答"什么时候算球出界"。

**包内已知问题**（`confidence` 那条是契约级；完整 5 条见 [`tt/状态机联调评估.md`](tt/状态机联调评估.md) 第五节「建议」，均未改代码，而是绕开或提建议）：

- `confidence=null` 被当成硬门限——`core/trajectory_processor.py:460`（默认值在 `:154`，`conf_new=0.35`）
- `ball.predicted` 被 `from_dict` 静默丢弃，补帧点当真实检测用
- HUD 发球指示与 `P1` 文字重叠成 `P1erve`（`visualizer/broadcast_hud.py:254`），外层用 `patch_scoreboard()` 绕开

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

**已入库的乒乓球权重**（4 份，各 41 MB，合计 164 MB）。四个训练实验的 `model_best.pt` 全部随仓库提供，clone 下来即可推理：

| 文件 | md5 前 8 位 | 训练实验 | 说明 |
|---|---|---|---|
| `TrackNet/TrackNet/exps/tt_g345_aug/model_best.pt` | `a496d256` | game3/4/5，带增强，约 16,300 帧 | **当前使用的一份**，与 2026-10-26 交付包内文件 md5 一致 |
| `TrackNet/TrackNet/exps/tt_g45_aug/model_best.pt` | `dae24d10` | game4/5，带增强 | 中间版，报告里 `tt_g45_aug_*` 那批结果由它产出 |
| `TrackNet/TrackNet/exps/tt_game4_aug/model_best.pt` | `bc0d0625` | game_4 单场，带增强 | 中间版 |
| `TrackNet/TrackNet/exps/tt_game4_run/model_best.pt` | `fc8428b6` | game_4 单场 | 第一版 |

四份架构完全相同（`BallTrackerNet`，10.72 M 参数，126 层），**`.pt` 加载不会报错，权重错了只会静默跑出坏结果**，认文件请对 md5。同一实验的 `model_last.pt`、`model_best_ep2.pt` 是中间态，未入库。

**不在仓库里、但名字很像的两份**（用错不会报错）：

| 路径 | 是什么 | 用途 |
|---|---|---|
| `TrackNet/TrackNet/weights/model_best.pt` | **上游 2023 年官方网球权重**，与 `datasets/raw/model_best.pt` 是同一个文件（md5 `3675ee22…`，42,886,919 B） | 迁移学习的起点，**不能**直接跑乒乓球 |
| `TrackNet/TrackNet/weights/tt_game4_best.pt` | 是 `exps/tt_game4_run/model_best.pt` 的副本（md5 相同） | 早期单独拷出来的一份，已被 `exps/` 里的取代 |

> `weights/` 这个目录名最容易误导：里面 **一份是上游网球权重、一份是自训乒乓球权重的副本**，只凭文件名分不出来。

**仍然不进仓库的**（合计约 54 G，权重之外的其余大文件）：

| 目录 / 文件 | 体积 | 说明 |
|---|---|---|
| `_data/` | 38 G | 原始比赛视频与标注，OpenTTGames 官方源可重新下载 |
| `TrackNet/.venv/` | 7.6 G | `pip install -r requirements.txt` 重建 |
| `TrackNet/TrackNet/datasets/` | 5.2 G | `prepare_openttgames.py` 从 `_data/` 生成 |
| `outputs/` | 2.7 G | 追踪视频与运行产物；**单个 mp4 就 1.9 G，超 GitHub 100 MB 硬限，技术上进不去** |
| `TrackNet/TrackNet/exps/` 其余 | 246 M | `model_last.pt` ×4、`model_best_ep2.pt` ×2，同实验中间态 |
| `backup/` | 250 M | 改动前的本地快照（旧代码 + 权重），仅作回滚用 |

**为什么这些不放进 Git。** 两条，对所有大文件都成立：

1. **单文件硬上限 100 MB。** GitHub 超过直接拒收（50 MB 起警告），一次 `push` 就被整条驳回。
2. **Git 存的是快照，不是增量。** 对 `.pt` / `.mp4` / `.jpg` 这类二进制，git 既不能 delta 压缩也不能 diff，每提交一版就多一份完整副本；更糟的是**写进历史就删不掉**——`git rm` 只是让它在最新提交里消失，体积还在 `.git` 里，要真删得用 `filter-repo` 重写全部历史。

**权重是唯一的破例**，因为它不可重建（要 38 G 数据加完整训练），而 41 MB 未触到硬限，代价可控。这意味着**以后每换一版权重，仓库历史就多 41 MB 且无法回收**——所以只放 4 个实验的 best，不放 last 和中间 epoch。

**视频和图**没法这么处理（单文件上 G），需要走 Release 附件或网盘；仓库里只留 `outputs/` 的体积清单和代表性截图。

> 上游仓库未附 LICENSE 文件，这里的代码仅用于课程项目。权重由 OpenTTGames 数据训练得到，公开分发前需确认该数据集的条款。

---

## 文档索引

按**阶段**排列。**报告写于不同阶段，数字口径不同，不能横向比较**——差异说明见表格下方。

| 阶段 | 文档 | 内容 | 数字口径 |
|---|---|---|---|
| ① 选型 | [`复现仓库评估.md`](复现仓库评估.md) | 三个候选方案的技术选型 | 无指标，只有结论 |
| ② 复现 | [`TrackNet复现报告.md`](TrackNet复现报告.md) | 模型复现过程与训练结果 | 网球验证集 F1 0.9567 / 0.9625；test_2 **球位置** F1 0.7893 |
| ③ 适配 | [`contract/README.md`](contract/README.md) | 契约字段表、硬规则、变更记录 | — |
| ③ 适配 | [`adapter/README.md`](adapter/README.md) | 适配器用法、`confidence` 定义、MOT 导出 | — |
| ③ 适配 | [`统一观测适配器报告.md`](统一观测适配器报告.md) | 适配器设计与自查结果 | MOT IoU 0.3 口径：IDF1 0.7132 / MOTA 0.5162 |
| ④ 落台 | [`落台事件检测.md`](落台事件检测.md) | 落台判据、参数、F1 对比 | **开发期口径**：game_4 全片 231 真值 F1 0.942；test_2 34 条 F1 0.774 |
| ⑤ 联调 | [`tt/状态机联调评估.md`](tt/状态机联调评估.md) | 交付件评估与 5 条改进建议（第五节） | 草稿口径，见下方第 3 条 |
| ⑤ 联调 | [`tt/事件顺序修复方案.md`](tt/事件顺序修复方案.md) | 事件错位的成因、修复、验证（含一条被 GT 否决的方案） | 修复过程，非终值 |
| ⑥ 最终 | [`视频判罚测试报告.md`](视频判罚测试报告.md) | 两段视频的逐条判罚核对与指标 | **最终口径，对外引用以这份为准** |

**三处看似矛盾的数字，实际是口径不同：**

1. **test_2 落台 F1：0.774（④）对 0.712（⑥）。** 前者是检测器直接输出的 34 条；后者加了 `--jitter-merge`（同侧 0.6 s / 60 px 内合并），把标注把一次弹跳标成连续三次的抖动合并掉，剩 31 条。**F1 降 0.042 换来 1 分判罚的修正**（0:3 → 0:2）。
2. **game_4 F1：0.942（④）对 0.951（⑥）。** 前者是 8 分 46 秒**全片**、231 个真值；后者只算 56 秒**片段**、40 条检出。样本不同，不是模型变好了。
3. **联调评估第七节那段回复草稿写于修复前**，里面「34 条 / 0.774 / 0:1」现为「31 条 / 0.712 / 0:2」，其余技术判断仍然成立。文档顶部已加时效说明。

另外两种口径本身不可混用：落台 P/R/F1 用 **±5 帧容差**；球位置用 **20 px 命中**（验证集上是 10 px ≈ 15 px）；MOT 用 **IoU 0.3**（46 px 框上约等于 14 px）。同一份结果在三套口径下数字不同，切换口径时必须整段重算。

---

## 仓库合并记录

**2026-11-02：把 `TrackNet/` 并入本仓库。**

`TrackNet/` 与 `TrackNet/TrackNet/` 原本各带一个 `.git`（都是 yastrebksv/TrackNet 的克隆），
根仓库无法跟踪另一个仓库里的文件，于是把两个 `.git` 移到了仓库外：

```text
<本机仓库外的备份目录>/TrackNet_outer.git   6.9 M
<本机仓库外的备份目录>/TrackNet_inner.git   6.9 M
```

移出前确认过：两者都停在上游同一提交、**0 个本地提交、无 stash**，所以只断开 `git log` / `git diff` 的上游参照，不丢工作。
需要恢复对比时，把备份目录改名回 `TrackNet/.git` 即可；确认不再需要可直接删除。

**同日：把状态机交付件纳入本仓库。** `tt/_unpacked/` 原先在忽略列表里，靠组长单独发 zip。
现在整目录入库（13 个文件、318 KB），clone 下来即可直接跑，不必再额外索取交付件。
入库时做过 `diff -r` 校验，与 zip 解压结果零差异；原始 zip 不入库，避免同一份代码两份拷贝。
详见上文「状态机交付件」一节。
