# table-calib —— 球台 / 球网自动标定（机位固定）

调研见项目文档 claude/net_calib_research.md。

## 目标
输入一段固定机位视频的若干帧，输出：
- 球台四角（远左、远右、近右、近左，原图像素）。角点定义为**白边外沿**两条直线的交点；
- 球网：网底线、网顶线、网柱、网面四边形；
- 国栋 `TableCalibration` 的字段：`table_polygon`、`net_x`、`net_zone`、`roi_box`（建议值）；
- 置信度；置信度低时走兜底（半自动：粗点四下再精修；或全手动）。

## 方法
1. 粗定位：缩小图上按色度 k-means，挑最像台面的一簇（实心、接近四边形、色度高），
   用轮廓上靠近各边的点拟合四条边求交（角被挡住时也能外推）。
2. 精修：原分辨率下沿每条边的外法线扫亮度剖面，找"台面→白边→外面"这一对边缘，取外沿亚像素位置。
3. RANSAC 拟合四条直线，交点为四角；以此为初值再窄带精修一轮。
4. 多帧：逐角中位数，剔除偏离大的帧，标准差即抖动。
5. 球网：网底线 = 两条长边中点连线（单应性）；PnPf（焦距未知，主点取图像中心）解相机，
   投影网顶（15.25 cm）和网柱（伸出边线 15.25 cm）。
6. 兼容字段：
   - `net_x`：网底线在四角重心 y 处的 x（与国栋 `calibration_net_x_cm()` 的自检口径一致，反推为 137 cm）；
   - `net_zone`：网顶再高 4 cm（一个球径）的投影 y 到网底（含网柱外延）的 y；
   - `roi_box`：台面外扩（两端 120 cm、前后 60 cm、高 120 cm）的长方体投影外接框，**仅建议值**。

## 直接从视频标定（推荐）

```bash
# 开场标定（默认）：开头 10 秒内均匀取 24 帧
python scripts/calibrate_video.py --video test_2.mp4 --name test_2 --out out/test_2
# 离线验收：整段视频 2%～98% 均匀取 24 帧；另存第 1200 帧原图供核对真值（不参与标定）
python scripts/calibrate_video.py --video test_2.mp4 --name test_2 --out out/test_2 --mode spread --save-frame 1200
# 批量 + 验收 + 打包（原机用）
python scripts/batch_videos.py --videos videos.txt --out-root out --mode spread --count 24 \
    --save-frame test_2:1200 --truth ../TN/calib/table_quads.json --zip results.zip
```

- 抽到的**所有帧**都计入置信度（有效帧占比、抖动）；汇总时剔除检测失败和偏离大的帧。不按清晰度挑帧。
- 定位后核对实际帧号，对不上的记在 `result.json` 的 `video.seek_mismatch`。
- 每段输出：`result.json`、`frames.csv`、`overlay.jpg`、`corners.png`、`frames/*.png`（`--save-frame` 指定的帧）。
- 自动失败：先加 `--init-quad "x,y;x,y;x,y;x,y"` 走半自动，再不行用 `scripts/click_corners.py` 人工点。
- 结果只写到输出目录。`table_quads.json` 是验收用的人工真值，**不要把自动结果写回去**。

## 用法
```
python scripts/calib_dir.py --frames frames/test_2 --name test_2 --out out/test_2
python scripts/evaluate.py                        # 与 ../TN/calib/table_quads.json 比较 → out/eval.md
python scripts/check_compat.py out/*/result.json  # 用国栋的 TableCalibration / TableHomography 读一遍
python scripts/corner_crops.py --image frames/test_2/f001200.png --out out/test_2/corners.png \
    --result out/test_2/result.json --cand "manual=540,516;1330,532;1484,656;352,631"
python scripts/click_corners.py --frames frames/test_2 --name test_2 --out out/test_2 --refine   # 兜底，需显示器
python -m unittest discover -s tests              # 合成图 + 视频入口回归测试
```
只依赖 numpy、opencv-python。

## 验收
- 角点：与人工四角比，每角 ≤5 px 合格，5～8 px 目检，>8 px 不合格；9 段中至少 8 段全部合格。
- 抖动：每角多帧标准差 ≤2 px（test_6 ≤5 px）。
- 网：画回原图目检（test_2 重点）；网顶投影与可见网顶白带 ≤4 px；PnPf 重投影 ≤2 px。
- 兼容：国栋的 `from_dict` 能读，`calibration_net_x_cm()` 在 137±1 cm。
- 耗时：CPU 上单段 ≤30 s。

注意：table_quads.json 里除 test_2（手标）外都来自自动拟合，只能说明与旧方法一致；
test_2 的手标与同机位的 test_3、国栋 game2 在远左角差 17～21 px，先用原分辨率帧重新核定。

## 目录
- tcalib/   算法（table / camera / detect / calibrate / video / report / viz / synth）
- scripts/  命令行
- tests/    合成图测试
- frames/   抽出的原图帧（不进 git）
- out/      结果和可视化（不进 git）
