# third_party：三路融合的另外两路代码

本分支 `fusion/three-way` 把三路代码放在同一个仓库里，单独克隆即可运行级联：

| 路 | 位置 | 来源 | 来源提交 |
|---|---|---|---|
| TrackNet | 仓库根目录（`TrackNet/`、`adapter/`、`tt/`） | Yao-MY06/TN | `feat/cascade` 之上 |
| YOLO | `third_party/pingpong-smart-referee/` | https://github.com/V2tin19/pingpong-smart-referee | `728ef76` |
| TTNet | `third_party/TTNet1.1/` | https://github.com/chagoup/TTNet1.1 | `8316a1a` |
| 级联层 | `cascade/` | 本仓库 | 见 `cascade/README.md` |

- 两份第三方代码是原样快照（不含各自的 `.git`），**不要在这里改**；要改请回各自仓库，再重新同步快照。
- 受本仓库 `.gitignore` 影响，以下文件没有进入快照：`*.gif`（TTNet1.1 的演示动图）、`*.pth`（TTNet1.1 仓库里只有一个 LFS 指针）。
  YOLO 权重 `weights/yolo11n_clean_best.pt`（5.4 MB）强制加入了，开箱即可推理。
- TTNet 权重不在仓库里：备用权重见 `cascade/README.md`「第 5 步」，chagoup 自训权重 `ttnet_3rd_phase_epoch_30.pth` 需自备。
- `cascade/_paths.py` 优先使用并排的独立仓库（`../TTNet`、`../TTNet1.1`、`../pingpong-smart-referee`），找不到时才用这里的副本。
