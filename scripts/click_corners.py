"""兜底：在原机上手动点四个角（需要显示器，OpenCV 窗口）。

两种用法：
  # 半自动：粗点四下（误差 20px 内即可），再按白边精修，对目录里所有帧汇总
  python scripts/click_corners.py --frames frames/test_2 --name test_2 --out out/test_2 --refine
  # 全手动：点到哪就是哪（在放大镜里点准），不做精修
  python scripts/click_corners.py --frames frames/test_2 --name test_2 --out out/test_2

操作：按 远左 → 远右 → 近右 → 近左 的顺序左键点；右上角的放大镜显示鼠标附近 8 倍像素。
  u 撤销上一点   r 全部重来   Enter 确认   Esc 放弃
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from calib_dir import load_frames  # noqa: E402
from tcalib.calibrate import calibrate_frames, result_to_dict  # noqa: E402
from tcalib.camera import net_geometry, solve_camera  # noqa: E402
from tcalib.calibrate import compat_fields  # noqa: E402
from tcalib.table import CORNER_NAMES  # noqa: E402
from tcalib.viz import draw_overlay  # noqa: E402

WIN = "click corners"
LOUPE_HALF, LOUPE_ZOOM = 20, 8


def click(img: np.ndarray, display_scale: float):
    pts, mouse = [], [0, 0]

    def on_mouse(ev, x, y, flags, param):
        mouse[0], mouse[1] = x / display_scale, y / display_scale
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / display_scale, y / display_scale))

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        view = cv2.resize(img, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
        for i, (x, y) in enumerate(pts):
            p = (int(x * display_scale), int(y * display_scale))
            cv2.circle(view, p, 4, (0, 255, 0), -1)
            cv2.putText(view, CORNER_NAMES[i], (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        nxt = CORNER_NAMES[len(pts)] if len(pts) < 4 else "Enter to accept"
        cv2.putText(view, f"next: {nxt}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # 放大镜
        cx, cy = int(round(mouse[0])), int(round(mouse[1]))
        H, W = img.shape[:2]
        x0, y0 = min(max(cx - LOUPE_HALF, 0), W - 2 * LOUPE_HALF - 1), min(max(cy - LOUPE_HALF, 0), H - 2 * LOUPE_HALF - 1)
        crop = img[y0 : y0 + 2 * LOUPE_HALF + 1, x0 : x0 + 2 * LOUPE_HALF + 1]
        big = cv2.resize(crop, None, fx=LOUPE_ZOOM, fy=LOUPE_ZOOM, interpolation=cv2.INTER_NEAREST)
        c = int((cx - x0 + 0.5) * LOUPE_ZOOM), int((cy - y0 + 0.5) * LOUPE_ZOOM)
        cv2.line(big, (c[0] - 12, c[1]), (c[0] + 12, c[1]), (0, 0, 255), 1)
        cv2.line(big, (c[0], c[1] - 12), (c[0], c[1] + 12), (0, 0, 255), 1)
        cv2.putText(big, f"({mouse[0]:.0f},{mouse[1]:.0f})", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        bh, bw = big.shape[:2]
        if bh < view.shape[0] and bw < view.shape[1]:
            view[0:bh, view.shape[1] - bw :] = big
        cv2.imshow(WIN, view)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            return None
        if k == ord("u") and pts:
            pts.pop()
        if k == ord("r"):
            pts.clear()
        if k in (13, 10) and len(pts) == 4:
            cv2.destroyAllWindows()
            return np.array(pts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refine", action="store_true", help="粗点后按白边精修（半自动）")
    ap.add_argument("--display-scale", type=float, default=0.75)
    args = ap.parse_args()

    frames = load_frames(args.frames)
    ids = sorted(frames)
    img = frames[ids[len(ids) // 2]]
    pts = click(img, args.display_scale)
    if pts is None:
        raise SystemExit("已放弃")
    os.makedirs(args.out, exist_ok=True)
    if args.refine:
        res = calibrate_frames(frames, init_quad=pts)
        d = result_to_dict(res, args.name, "semi_auto_edge_pooled_v3")
        d["clicked_points"] = pts.round(1).tolist()
    else:
        h, w = img.shape[:2]
        cam = solve_camera(pts, (w, h))
        net = net_geometry(pts, cam)
        d = {
            "points": pts.round(2).tolist(),
            "corner_order": list(CORNER_NAMES),
            "method": "manual_click",
            "confidence": "manual",
            "net": {k: [[round(float(x), 2), round(float(y), 2)] for x, y in v] for k, v in net.items()},
            "camera": cam.to_dict(),
            "table_calibration": dict(name=args.name, **compat_fields(pts, cam, net, (w, h))),
        }
    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    cv2.imwrite(os.path.join(args.out, "overlay.jpg"), draw_overlay(img, d))
    print(json.dumps({k: d[k] for k in ("points", "method", "confidence")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
