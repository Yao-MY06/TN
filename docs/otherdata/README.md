# 参考资料（otherdata）

本目录存放**外部来源**的材料：组长下发的方案文档、组内讨论产出、项目背景资料。这些文件不是本项目产出的成果，只作查阅，原件仍以微信/群里发的版本为准。

## 文件清单

| 文件 | 来源 | 作用 |
|---|---|---|
| `TTNet_TrackNet_YOLO_共用状态机方案.md` | 组长下发 | 三路（TTNet / TrackNet / YOLO）共用状态机与统一 Observation 接口的方案，§7 是接口契约本体 |
| `pingpong.md` | 项目背景方案 | 低成本乒乓球智能裁判的产品/硬件/算法选型方案，含实施顺序与数据集调研 |
| `数据集.docx` | 组内资料 | OpenTTGames 数据集说明 |
| `对接问题清单.md` | 组内（YOLO 分支起草） | Q1–Q15 待确认的接口对接问题，含各家暂定答案 |

## 与本项目产出的对应关系

本项目自己的产出**不在本目录**，在仓库根目录：

| 产出 | 位置 |
|---|---|
| TrackNet 复现与改造（含三阶段同口径指标） | `TrackNet复现报告.md` |
| 统一 Observation / Event 适配器 | `统一观测适配器报告.md`、`adapter/` |
| 落台事件检测与半台归属 | `落台事件检测.md`、`TrackNet/TrackNet/detect_bounce.py` |
| 四个候选仓库的实现完整度评估 | `复现仓库评估.md` |

## 约定提醒

- `TTNet_TrackNet_YOLO_共用状态机方案.md` §7.1 是**冻结接口**，改动需在群里同步，不要各分支自行加字段。
- 事件枚举已统一为 `BOUNCE / RACKET_HIT / NET_TOUCH / OUT`（在 `adapter/observation.py` 与 `adapter/event.schema.json` 中实现），与本文档 §7.4 的示例不冲突但更完整。
- `side` 语义已确认：**A = 球台物理左半台**（由标定四角命名锚定，`far_left` 恒映射俯视坐标 `x = 0`）；`frame_id` 从 0 开始。
