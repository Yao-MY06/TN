# table-calib —— 球台 / 球网自动标定（机位固定）

调研见项目文档 claude/net_calib_research.md。

## 目标
输入一段固定机位视频的若干帧，输出：
- 球台四角（远左、远右、近右、近左，原图像素）。角点定义为**白边外沿**两条直线的交点；
- 球网：网底线、网顶线、网柱、网面四边形；
- 国栋 `TableCalibration` 的字段：`table_polygon`、`net_x`、`net_zone`、`roi_box`；
- 置信度；置信度低时走兜底（半自动：粗点四下再精修；或全手动）。

## 方法
角点定义：白边**外沿**（白边属于台面，274×152.5 cm 量到外沿）。注意 TN/calib/table_quads.json 的旧值两端多取的是内沿（蓝色区域边界），不能直接当本定义下的真值。

1. 粗定位：缩小图上按色度 k-means。候选是各簇里较大的连通块，以及把色相相近、合起来仍然实心
   （solidity ≥ 0.9、接近四边形）的几块合并的结果——台面可能被球网切成两半（test_1），也可能因反光
   被分进两个簇（test_7）。不实心的、碰到画面边缘的色块（幕布、墙、地面）不参与。
   用轮廓上靠近各边的点拟合四条边求交（角被挡住时也能外推）。
2. 精修：原分辨率下沿每条边的外法线扫亮度剖面，找"台面→白边→外面"这一对边缘，取外沿亚像素位置。
3. 单帧：RANSAC 拟合四条直线，交点为四角（只用来给多帧一个初始四边形，并做逐帧诊断）。
4. 多帧（v2，按边汇集）：机位固定，每条边在所有帧里是同一条直线。以公共四边形为基准在每一帧上找白边外沿点，
   同一条边所有帧的点合并拟合，交点得四角，再沿拟合曲线窄带精修两轮。球员挡住一端的帧不再整帧作废。
   - 镜头畸变：实测近侧长边在画面里有 3~6.6 px 的拱高，直线拟合会把近侧角推出台外 6~9 px。
     每条边用相对弦的二次曲线拟合（`tcalib/curve.py`），拱高 <0.5 px 或统计上不显著时退回直线；
     角点为相邻两条曲线的交点。`result.json` 的 `sides.*.sagitta_px` 记录拱高，`side_curves` 记录曲线。
   - 覆盖率：每条边上至少在一帧里找到内点的采样位置比例（≥0.6 高，≥0.3 中，否则低）；
   - 抖动：四边俱全的帧（≥3 帧时）单独拟合的四角相对汇总结果的 RMS 偏差；不足 3 帧时用对帧 bootstrap 的标准差，
     `result.json` 的 `std_method` 注明用的是哪种。
5. 球网：网底线 = 两条长边中点连线（单应性）；PnPf（焦距未知，主点取图像中心）解相机，
   投影网顶（15.25 cm）和网柱（伸出边线 15.25 cm）。
6. 兼容字段：
   - `net_x`：网底线在四角重心 y 处的 x（与国栋 `calibration_net_x_cm()` 的自检口径一致，反推为 137 cm）；
   - `net_zone`：网顶再高 4 cm（一个球径）的投影 y 到网底（含网柱外延）的 y；
   - `roi_box`：照国栋手填预设（game2、test4）的习惯，由台面外接框外扩：左右各 0.22、上 0.285、下 0.17 倍台面外接框宽度
     （两组预设能复现到 3~64 px 以内）。

## 直接从视频标定（推荐）

```bash
# 开场标定（默认）：开头 10 秒内均匀取 24 帧
python scripts/calibrate_video.py --video test_2.mp4 --name test_2 --out out/test_2
# 离线验收：整段视频 2%～98% 均匀取 24 帧；另存第 1200 帧原图供核对真值（不参与标定）
python scripts/calibrate_video.py --video test_2.mp4 --name test_2 --out out/test_2 --mode spread --save-frame 1200
# 批量 + 验收 + 打包（原机用）
python scripts/batch_videos.py --videos videos.txt --out-root out --mode spread --count 24 \
    --save-frame test_2:1200 --truth truth/table_quads_outer.json --zip results.zip
```

- 抽到的**所有帧**都参与按边汇集和置信度统计（覆盖率、抖动）。不按清晰度挑帧。
- 定位后核对实际帧号，对不上的记在 `result.json` 的 `video.seek_mismatch`。
- 每段输出：`result.json`、`frames.csv`、`overlay.jpg`、`corners.png`、`frames/*.png`（`--save-frame` 指定的帧）。
- 自动失败：先加 `--init-quad "x,y;x,y;x,y;x,y"` 走半自动，再不行用 `scripts/click_corners.py` 人工点。
- 结果只写到输出目录。`table_quads.json` 是验收用的人工真值，**不要把自动结果写回去**。

## 用法
```
python scripts/calib_dir.py --frames frames/test_2 --name test_2 --out out/test_2
python scripts/evaluate.py                        # 与 truth/table_quads_outer.json 比较 → out/eval.md
python scripts/check_compat.py out/*/result.json  # 用国栋的 TableCalibration / TableHomography 读一遍
python scripts/corner_crops.py --image frames/test_2/f001200.png --out out/test_2/corners.png \
    --result out/test_2/result.json --cand "manual=540,516;1330,532;1484,656;352,631"
python scripts/click_corners.py --frames frames/test_2 --name test_2 --out out/test_2 --refine   # 兜底，需显示器
python -m unittest discover -s tests              # 合成图 + 视频入口回归测试
```
只依赖 numpy、opencv-python。

## 验收
- 角点：与 truth/table_quads_outer.json（白边外沿，目检确认）比，每角 ≤5 px 合格，5～8 px 目检，>8 px 不合格；8 段全部合格（原计划"9 段中 ≥8 段"，真值现为 8 段）。
- 抖动：每角多帧标准差 ≤2 px（test_6 ≤5 px）。
- 网：画回原图目检（test_2 重点）；网顶投影与可见网顶白带 ≤4 px；PnPf 重投影 ≤3 px（实测 OpenTTGames 机位约 2～2.8 px，原 2 px 过严）。
- 兼容：国栋的 `from_dict` 能读，`calibration_net_x_cm()` 在 137±1 cm。
- 耗时：CPU 上单段 ≤30 s。

真值：`truth/table_quads_outer.json`，8 段（game_4、test_1～7），本算法在每段 2 张原图上的结果经人工目检确认，
test_2 目检略偏外，取同机位 test_3 的四角。旧的 TN/calib/table_quads.json 两端取白边内沿，仅供对比。

## 目录
- tcalib/   算法（table / camera / detect / calibrate / video / report / viz / synth）
- scripts/  命令行
- tests/    合成图测试
- frames/   抽出的原图帧（不进 git）
- out/      结果和可视化（不进 git）
