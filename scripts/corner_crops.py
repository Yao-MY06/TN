"""角点放大对比图：把几组候选四角叠在原分辨率帧的四个角上，供目检定真值。

用法：
  python scripts/corner_crops.py --image frames/test_2/f001200.png --out out/test_2/corners_f001200.png \
      --cand "test_2手标=540,516;1330,532;1484,656;352,631" \
      --cand "test_3自动=561,516;1334,532;1480,660;357,636" \
      --cand "国栋game2=557,519;1329,535;1473,654;367,631" \
      --result out/test_2/result.json

--result 会把自动标定结果也作为一组候选加进来。
OpenCV 不能画中文，图例里的中文名会被替换成 cand1、cand2…，对应关系打印在终端。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tcalib.viz import corner_crops  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cand", action="append", default=[], help="名字=x,y;x,y;x,y;x,y")
    ap.add_argument("--result", default=None)
    ap.add_argument("--half", type=int, default=24)
    ap.add_argument("--zoom", type=int, default=8)
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"读不了 {args.image}")
    cands = {}
    if args.result:
        with open(args.result, encoding="utf-8") as f:
            pts = json.load(f).get("points")
        if pts:
            cands["auto"] = pts
    for s in args.cand:
        name, q = s.split("=", 1)
        pts = [list(map(float, p.split(","))) for p in q.split(";")]
        if len(pts) != 4:
            raise SystemExit(f"{name} 不是 4 个点")
        cands[name] = pts
    if not cands:
        raise SystemExit("至少给一组候选")
    safe = {}
    for i, (k, v) in enumerate(cands.items()):
        ascii_name = k if k.isascii() else f"cand{i + 1}"
        safe[ascii_name] = v
        print(f"{ascii_name} = {k}: {v}")
    out = corner_crops(img, safe, half=args.half, zoom=args.zoom)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, out)
    print(f"写出 {args.out}")


if __name__ == "__main__":
    main()
