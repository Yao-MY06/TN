# TTNet 基线备份与其他 PC 部署

本说明适用于本目录 `TTNet` 基线。预定代码远程为 https://github.com/chagoup/TTNet 。模型算法保持当前基线，不适用于相邻的 `TTnet1.0` 或 `TTnet_test`。

## 1. 备份边界和验收状态

代码 Git 仓库与本地独立资产包需要分别保存：

- Git：源码、脚本、说明和依赖清单。`.gitignore` 已排除 `checkpoints/`（**主力权重除外**，见下条）、`dataset/`、视频及生成结果。
- **模型权重（Git LFS 管理）**：`checkpoints/ttnet_3rd_phase/ttnet.pth`（当前基线主力权重，约 180 MB）经由 **Git LFS** 纳入仓库，clone 后需 `git lfs pull` 才能取到实体（否则只是指针文件）。其余 4 个 `.pth` 体积大且非必需，仍不纳入 Git，按 `baseline-assets.zip` 恢复。
- `baseline-assets.zip`：位于相邻目录 `TTNet-backup/`，包含 `checkpoints/` 下三个 `.pth` 与 `dataset/test/videos/test_2.mp4`，实测 503 MB。
- `assets-manifest.json`：与该包同目录，记录文件路径、字节数与 SHA-256；压缩包已通过 `zipfile.testzip()` 与逐项大小核对。
- 该包**不含完整训练数据**、其余 6 个测试视频、抽帧、球/事件标注和分割掩码，也不备份原有结果视频。
- 该资产包为本地文件，**未上传至 Git 仓库**（体积过大且属二进制大文件），需手动拷贝或另行托管。主力权重 `ttnet.pth` 例外：已通过 Git LFS 纳入仓库（见上）。

本次部署修改只做受管 Python 3.13 的语法/CLI 检查，不执行原 conda，不安装训练依赖。Python 3.11 + CUDA 的模型前向、视频解码及真实训练须在目标 PC 验收；不能将静态检查视为运行成功。

## 2. 目标环境与依赖

推荐 Windows 或 Linux x86-64、Python **3.11**、NVIDIA GPU 和适配 CUDA 12.4 PyTorch wheel 的驱动。安装 PyTorch wheel 后一般不需另装 CUDA Toolkit，但驱动必须兼容，实际以 CUDA 探测为准。单 GPU 为本说明默认路径；Windows 多卡 NCCL 不在本次验收范围。macOS/Apple MPS 未适配。

本机基线环境仅只读检查 `dist-info` 元数据，主要版本如下：

| 依赖 | 元数据版本 |
| --- | --- |
| torch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 |
| numpy / opencv-python | 2.4.6 / 5.0.0.93 |
| easydict / scikit-learn | 1.13 / 1.9.1 |
| tensorboard / tqdm | 2.21.0 / 4.70.1 |
| matplotlib / Pillow | 3.11.2 / 12.3.0 |

`requirements-pc.txt` 固定上述直接依赖，并包含数据下载脚本使用的 `wget==3.2`。它不是完整传递依赖锁文件，未在干净目标 PC 执行安装验证。OpenCV 5 的元数据要求 Python >=3.9 时 NumPy >=2，因此不应单独改成 NumPy 1.x。旧 `requirement.txt` 保留用于历史记录（torch 1.5 / Python 3.6 时代），其他 PC 部署不要使用它。

## 3. 获取代码与创建环境

代码已推送至远程私有仓库 `main` 分支（提交 `241a5b7`，74 个文件）。另一台电脑直接克隆：

```bash
git clone https://github.com/chagoup/TTNet.git
cd TTNet
```

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements-pc.txt
.\.venv\Scripts\python.exe -m pip check
```

Linux Bash：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r requirements-pc.txt
.venv/bin/python -m pip check
```

先安装 CUDA wheel，再安装通用清单；通用清单的 `torch==2.6.0` 可匹配已安装的 `2.6.0+cu124`。不要使用 `--upgrade` 重新覆盖 CUDA wheel。如固定版本在目标平台的源中不可用，应先核对源和 Python 版本，不能把依赖安装成功写成已完成验收。

后续示例假定已激活新建 `.venv`，`python` 指向它。也可始终使用上面的完整解释器路径，无需激活。

## 4. 从本地资产包恢复

将可信的 `baseline-assets.zip`、`assets-manifest.json` 复制到克隆目录，先查看包内路径：

```bash
python -m zipfile -l baseline-assets.zip
```

预期条目直接以 `checkpoints/` 和 `dataset/` 开头（不应再套一层 `TTNet/`），且仅包含清单所列资产。确认目录结构后，在项目根目录解压：

```bash
python -m zipfile -e baseline-assets.zip .
```

按交付的 `assets-manifest.json` 逐项核对文件大小、SHA-256 与文件数；当前自检入口只验证选定演示权重和视频非空，**不代替清单校验**。已存在同名资产时先备份，解压会覆盖同名文件。不要将资产包、原数据或 `.venv` 强行加入 Git。

校验示例：

```bash
python - <<'EOF'
import hashlib, json, os
m = json.load(open('assets-manifest.json'))
for f in m['files']:
    h = hashlib.sha256()
    with open(f['path'], 'rb') as fh:
        for c in iter(lambda: fh.read(1 << 22), b''):
            h.update(c)
    ok = h.hexdigest() == f['sha256'] and os.path.getsize(f['path']) == f['size']
    print(('OK  ' if ok else 'FAIL'), f['path'])
EOF
```

默认演示必须具备：

```text
TTNet/
  deploy_pc.py
  requirements-pc.txt
  checkpoints/ttnet_3rd_phase/ttnet.pth
  dataset/test/videos/test_2.mp4
  src/
```

其他 `.pth` 按原相对路径恢复，以清单为准。`ttnet.pth` 对应当前基线的 `BallDetection_right` 双分支结构；不要随意换用原版网络权重。已有加载逻辑含 `torch.load(..., weights_only=False)`，只加载自己确认可信的本地 checkpoint。

## 5. 跨平台自检和演示

`deploy_pc.py` 从自身位置解析项目根目录，并以 `src/` 为子进程工作目录，避免源码中的相对路径随调用位置变化。用户传入的相对视频/权重路径统一以**项目根目录**为基准；绝对路径同样支持。

```bash
# 无第三方依赖；Python 3.13 也可做此项静态检查，不生成 pyc
python deploy_pc.py check --static
python deploy_pc.py --help

# 在新建 Python 3.11 环境检查依赖元数据及默认资产是否存在
python deploy_pc.py check

# 实际导入依赖并做 CUDA 张量探测；不加载权重
python deploy_pc.py check --runtime

# 加载默认 ttnet.pth，执行原 verify.py 的随机输入前向
python deploy_pc.py verify

# 先打印命令，再限制为 60 个处理序列做短演示
python deploy_pc.py demo --max-frames 60 --dry-run
python deploy_pc.py demo --max-frames 60 --saved-fn smoke_test_2

# 完整演示；可显式指定其他本地权重和视频
python deploy_pc.py demo --video dataset/test/videos/test_2.mp4 --weights checkpoints/ttnet_3rd_phase/ttnet.pth --saved-fn test_2
```

演示调用已有流式写视频模式，输出为 `results/demo/<saved-fn>/result.mp4`，避免生成大量帧和再合成；未修改推理算法。已有写出逻辑使用 `min(原视频 FPS, 30)`，高帧率输入的输出播放时长可能变化，不能用结果视频时长直接衡量处理耗时。新机器须实际播放短演示产物，检查 MP4 编解码与输出内容。

`--saved-fn` 只允许英文字母、数字、下划线、短横线；运行前选用新的输出名称，避免覆盖已有结果。`--gpu N` 从当前可见设备中选择一张，子进程将其映射为 `cuda:0`，兼容原演示中直接 `model.cuda()` 的行为。已设置 `CUDA_VISIBLE_DEVICES` 时 `N` 是过滤后的索引。

Bash / Git Bash 兼容入口默认使用 `python`，可通过环境变量选解释器：

```bash
PYTHON="/path/to/python" bash run_demo.sh dataset/test/videos/test_2.mp4 --max-frames 60
PYTHON="/path/to/python" bash run_train.sh --dry-run
```

训练脚本也兼容 `PY`，其优先级为 `PY` > `PYTHON` > `python`。Windows 原生 PowerShell 请直接调用 `deploy_pc.py`；解释器路径含空格时，PowerShell 用 `& "路径/python.exe" deploy_pc.py ...`。旧示例中相对于 `src/` 的 `../dataset/...` 路径不适用于这两个根目录新入口。

## 6. 训练恢复条件

```bash
python deploy_pc.py train --dry-run
# 仅在完整数据准备妥当并检查输出目录后运行
python deploy_pc.py train --batch-size 8 --num-workers 4
```

训练参数与原串联脚本一致：三个阶段各 30 epochs；阶段 2 读取阶段 1 的 epoch_30，阶段 3 读取阶段 2 的 epoch_30；任一阶段失败立即停止。显存不足时可降低 `--batch-size`；Windows 数据加载问题可先用 `--num-workers 0` 排查。入口不保证任意 batch size 的训练效果，也未执行本次训练验收。

需补齐 `dataset/training` 中 `game_1` 到 `game_5`、`dataset/test` 中 `test_1` 到 `test_7`：`images/<game>/img_*.jpg`、`annotations/<game>/ball_markup.json`、`events_markup.json`、`segmentation_masks/*.png`。具体准备方法见 `prepare_dataset/README.md`。入口检查各组文件/目录是否存在，不逐帧验证全量数据一致性。

`src/main.py` 即使传 `--no-test` 仍会先构造测试 dataloader；因此只有训练数据也不够。资产包中的一段测试视频不能替代训练/测试标注与抽帧。已有恢复的 epoch checkpoint 与重新训练输出名称相同，重新训练前应另存旧 checkpoint 和日志，避免覆盖。

## 7. CPU 与其他现存路径限制

- 纯 CPU 可运行标准库静态自检、命令预览和资产清单核对。实际演示、原 verify 及训练包含 `.cuda()` / `torch.cuda.synchronize()` 等调用，本入口对无 CUDA 机器明确报错；不承诺 `--no_cuda` 可用，不提供 CPU 回退。
- 推荐入口仅覆盖基线 demo、verify 和三阶段训练。`src/batch_demo.py` 需要额外的多段测试视频；`src/evaluate_ball.py` 与测试入口需要完整测试抽帧和标注。
- `src/referee/batch_render.py` 尚有原作者本机桌面输出默认路径，使用该辅助命令须显式给 `--out-dir`；其他 referee 渲染辅助脚本及字体选择不在本次 GPU 基线验收范围。
- `make_video.py` 和旧 `src/*.sh` 仍保留历史使用方式；新根目录入口已固定工作目录和默认资产。此说明不把缺失数据、字体或旧辅助脚本默认路径问题视为已修复。
