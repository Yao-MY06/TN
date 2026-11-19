"""对一个机位的一组帧做自动标定。

用法：
  python scripts/calib_dir.py --frames frames/test_2 --name test_2 --out out/test_2
  # 半自动兜底：给一个粗四边形（远左、远右、近右、近左；误差 20px 内即可）
  python scripts/calib_dir.py --frames frames/test_2 --name test_2 --out out/test_2 \
      --init-quad "540,516;1330,532;1484,656;352,631"

输出：
  <out>/result.json     主结果（四角、网、置信度、相机、TableCalibration 字段）
  <out>/overlay.jpg     画回原图（取参与汇总的中间一帧）
  <out>/frames.csv      逐帧检测结果
  <out>/corners.png     四个角的放大图
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tcalib.calibrate import calibrate_frames, result_to_dict  # noqa: E402
from tcalib.report import write_outputs  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png")


def load_frames(folder: str, prefer_png: bool = False):
    """读取 f<帧号>.jpg/png。同一帧号既有 jpg 又有 png 时，默认用 jpg（与其它帧一致）。"""
    by_id = {}
    for fn in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in IMG_EXT:
            continue
        if stem in by_id and (ext.lower() == ".png") != prefer_png:
            continue
        by_id[stem] = os.path.join(folder, fn)
    frames = {}
    for k, p in by_id.items():
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] 读不了 {p}", file=sys.stderr)
            continue
        frames[k] = img
    return frames


def parse_quad(s: str) -> np.ndarray:
    pts = [list(map(float, p.split(","))) for p in s.split(";")]
    if len(pts) != 4:
        raise SystemExit("--init-quad 需要 4 个点")
    return np.array(pts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-quad", default=None)
    args = ap.parse_args()

    frames = load_frames(args.frames)
    if not frames:
        raise SystemExit(f"{args.frames} 里没有图片")
    sizes = {f.shape[:2] for f in frames.values()}
    if len(sizes) != 1:
        raise SystemExit(f"帧分辨率不一致：{sizes}")
    init = parse_quad(args.init_quad) if args.init_quad else None
    method = "auto_edge_v1" if init is None else "semi_auto_edge_v1"

    t0 = time.time()
    res = calibrate_frames(frames, init_quad=init)
    dt = time.time() - t0
    d = result_to_dict(res, args.name, method)
    d["runtime_s"] = round(dt, 2)
    d["image_size"] = [int(list(sizes)[0][1]), int(list(sizes)[0][0])]

    write_outputs(args.out, d, res, frames)

    pts = d["points"]
    print(f"{args.name}: conf={d['confidence']} used {d['n_used']}/{d['n_frames']} {dt:.1f}s")
    print(f"  points={pts}")
    print(f"  std={d['per_corner_std_px']}  reasons={d['confidence_reasons']}")
    if d.get("table_calibration"):
        tc = d["table_calibration"]
        print(f"  net_x={tc['net_x']} net_zone={tc['net_zone']} roi_box={tc['roi_box']}")


if __name__ == "__main__":
    main()
