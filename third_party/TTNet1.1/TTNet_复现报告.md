# TTNet 复现与乒乓球多任务改造报告

## 一、核心结论

- 完整复现 TTNet（CVPRW 2020 乒乓球实时分析系统）的**多任务架构 + 三阶段训练策略**，在 OpenTTGames 官方数据集（33GB、12 个视频）上从零训练至收敛；
- 自训练模型球检测 **overall RMSE = 2.1px**（1920×1080 尺度，对比官方预训练权重 1.9px），达到同等精度水平；
- 四个任务协同输出：**全局/局部球检测（global + local）+ 事件识别（bounce / net）+ 语义分割（运动员 / 球台 / 记分牌）**；
- 推理速度 1080p 约 **25 FPS**、短片段 demo 约 **40 FPS**，在 RTX 2080 Ti 上满足实时分析需求；
- 已用自训练模型完成 test_2~7 六个视频的批量推理（6 段结果视频合计 2.2GB），球轨迹连续、无"乱飞"现象。

## 二、方法要点

### 1. 多任务统一架构
TTNet 在一个骨干网络（ResNet-18 类）上同时学习四类任务，共享浅层特征、多路输出：

| 任务 | 输出 | 作用 |
| --- | --- | --- |
| 全局球检测 global | 球位置热图（粗定位） | 全图快速找球，作为后续任务地基 |
| 局部球检测 local | 球位置热图（精定位） | 基于 global 结果裁剪后高分辨率精定位 |
| 事件识别 event | bounce / net 二分类概率 | 判断弹跳、触网 |
| 语义分割 seg | 运动员 / 球台 / 记分牌掩码 | 场景解析 |

模型总参数量 **15.76M**，最终 checkpoint 181MB。

### 2. 三阶段递进训练策略
按任务依赖关系拆分训练，避免多任务从零联合训练互相拖累：

| 阶段 | 训练模块 | 关键配置 | 产出 |
| --- | --- | --- | --- |
| Phase 1 | global + seg | 冻结 local/event，global 权重 5× | 打地基 |
| Phase 2 | local + event | 冻结 global/seg，全局→局部权重热启动，event 权重 2× | 上层任务 |
| Phase 3 | 端到端微调 | 全部解冻，学习率降为 1/10 | 最终模型 |

三阶段各 30 epochs，学习率分别为 0.001 / 0.001 / 0.0001，最终测试 Loss：Phase 1 = 0.032、Phase 2 = 0.086、Phase 3 = 0.052。

### 3. 乒乓球数据适配
| 项 | 内容 |
| --- | --- |
| 数据来源 | OpenTTGames 官方数据集（lab.osai.ai） |
| 规模 | 训练 game_1~5（28GB）+ 测试 test_1~7（5.6GB）+ 标注 60MB |
| 抽帧 | `smooth-labelling` 平滑标注方案：empty_event ±4 帧、bounce/net ±8 帧 |
| 帧数 | 训练 53,265 + 测试 7,189 = **60,454 帧** |
| 样本扩展 | 事件中心标注仅 3,806 条，平滑标注扩展至 **25,120** 条训练样本（6.6×） |

## 三、关键踩坑：球检测"乱飞"的定位与修复

**问题现象**：首轮自训练模型在 demo 中球轨迹出现跳变/乱飞，肉眼可见追踪失效。

**量化定位**（自写评估脚本 `evaluate_ball.py`，1920×1080 尺度）：

| 指标 | 预训练权重 | 首轮自训练 ❌ | 重训后 ✅ |
| --- | --- | --- | --- |
| global RMSE | 0.6px | 8.5px | **0.7px** |
| local RMSE | 1.8px | **195.9px** | **2.0px** |
| overall RMSE | 1.9px | **195.9px** | **2.1px** |

**根因**：首轮为提速，在 Phase 1 误删了 `--smooth-labelling`。虽然 Phase 1 不训练事件，但平滑标注同时扩展了球检测样本（25,120 → 3,806，骤减 6.6 倍）。global 球检测是整条链路"地基"，训练样本不足 → global 定位变差 → local 依赖 global 输出做裁剪，级联失效（195.9px 比随机猜测还差）。

**修复**：恢复三阶段 `--smooth-labelling` 完整重训，local RMSE 从 195.9px 降至 2.0px，与预训练权重持平，球追踪恢复正常。

## 四、工程修复（现代环境适配）

原项目基于 Python 3.6 + PyTorch 1.5，需全面适配本机 Python 3.11 + PyTorch 2.6 + CUDA 12.4 + Windows：

| 问题 | 修复 |
| --- | --- |
| 权重托管 Google Drive（被墙） | 改用 SLAM-TT 项目 GitHub Release 的等价预训练权重 `ttnet.pth`（181MB），经 `ghfast.top` 镜像下载 |
| 权重结构不匹配（ball detection 输出 tuple） | 移植修正版 `BallDetection_right`（x/y 双分支）到 `TTNet.py`、`post_processing.py` |
| `torch.load` 默认 `weights_only=True` | 全部改为 `weights_only=False` |
| numpy 2.x 移除 `np.int`/`np.float` | 改为 `np.int64`/`np.float32` |
| `turbojpeg` 依赖（Windows 无 DLL） | 改用 `cv2.imread` |
| 训练链路适配 tuple 输出 | 新增 `Ball_Detection_Loss_right` + `create_target_ball_right`（losses.py、data_utils.py、unbalanced_loss_model.py） |
| 抽帧脚本 `cv2.waitKey(10)` 拖慢 10× | 重写 `extract_frames_fast.py`，去 waitKey + 断点续抽 |
| 1920×1080 大图加载瓶颈（1.1s/batch） | 预缩放 60,454 帧到 320×128，数据加载 0.012s/batch（提速 12×） |
| 大文件下载断流 | 重写下载脚本：`-C -` 断点续传 + `--retry 200 --retry-all-errors` |
| Windows 多进程 Pool 不稳 | 抽帧/缩放改用线程池 |

> 所有改动均已备份原文件为 `*.orig`。

## 五、最终指标汇总

| 维度 | 指标 | 结果 |
| --- | --- | --- |
| 球检测 | overall RMSE（1920×1080） | **2.1px**（预训练 1.9px） |
| 事件识别 | bounce / net | 训练收敛，demo 中正确给出 is bounce 概率 |
| 语义分割 | 运动员/球台/记分牌 | 掩码干净准确，背景噪声少于预训练权重 |
| 推理速度 | 1080p 流式 | 约 25 FPS（批量）/ 40 FPS（短片段） |
| 模型规模 | 参数量 / checkpoint | 15.76M / 181MB |
| 训练耗时 | 三阶段 30 epochs | 约 9.4 小时 |

## 六、交付物清单

| 文件 | 说明 |
| --- | --- |
| `checkpoints/ttnet_3rd_phase/ttnet_3rd_phase_epoch_30.pth` | 最终自训练权重（181MB） |
| `checkpoints/ttnet_3rd_phase/ttnet.pth` | 官方预训练权重（等价来源） |
| `results/demo/batch/test_2~7.mp4` | 6 段批量推理结果视频 |
| `run_train.sh` / `run_demo.sh` | 一键训练 / 一键推理脚本 |
| `evaluate_ball.py` | 球检测 RMSE 量化评估脚本 |
| `extract_frames_fast.py` / `scale_frames.py` | 抽帧与预缩放优化脚本 |
| `DEPLOY_NOTES.md` | 部署与复现说明 |

## 七、后续可优化方向

1. **事件识别量化评估**：当前 bounce/net 仅通过 demo 定性验证，可参照官方 `test_3rd_phase.sh` 补全 PCE/精确率指标；
2. **推理侧时序平滑**：若需进一步抑制帧间跳动，可对球位置序列加中值滤波/卡尔曼平滑；
3. **视频末尾越界 bug**：原项目 `ttnet_video_loader.py` 读帧越界导致 demo 末尾 `AssertionError`，不影响结果但可修复。
