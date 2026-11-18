"""可视化：把四角、球网、兼容字段画回原图；角点放大对比图。"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .table import CORNER_NAMES

# BGR
COLORS = {
    "quad": (0, 220, 0),
    "net_base_line": (0, 220, 255),
    "net_top_line": (255, 0, 255),
    "net_post": (255, 200, 0),
    "net_x": (0, 0, 255),
    "net_zone": (0, 0, 255),
    "roi": (200, 200, 200),
}

CANDIDATE_COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

_SHIFT = 4


def _pt(p) -> Tuple[int, int]:
    return int(round(p[0] * (1 << _SHIFT))), int(round(p[1] * (1 << _SHIFT)))


def _line(img, a, b, color, th=2):
    cv2.line(img, _pt(a), _pt(b), color, th, cv2.LINE_AA, _SHIFT)


def draw_overlay(img: np.ndarray, d: Dict, compat: bool = True) -> np.ndarray:
    """d 是 calibrate.result_to_dict 的输出。"""
    out = img.copy()
    pts = d.get("points")
    if pts is None:
        cv2.putText(out, "NO RESULT", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        return out
    net = d.get("net", {})
    tc = d.get("table_calibration", {})
    if compat and tc:
        if tc.get("roi_box"):
            y0, y1, x0, x1 = tc["roi_box"]
            cv2.rectangle(out, (x0, y0), (x1, y1), COLORS["roi"], 1)
        if tc.get("net_zone"):
            z0, z1 = tc["net_zone"]
            x = tc["net_x"]
            overlay = out.copy()
            cv2.rectangle(overlay, (x - 30, z0), (x + 30, z1), COLORS["net_zone"], -1)
            cv2.addWeighted(overlay, 0.15, out, 0.85, 0, dst=out)
            cv2.line(out, (x, z0), (x, z1), COLORS["net_x"], 1)
    q = np.array(pts)
    for i in range(4):
        _line(out, q[i], q[(i + 1) % 4], COLORS["quad"], 1)
    for i, (x, y) in enumerate(q):
        cv2.circle(out, _pt((x, y)), 4 << _SHIFT, COLORS["quad"], 1, cv2.LINE_AA, _SHIFT)
        cv2.putText(out, CORNER_NAMES[i], (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["quad"], 1)
    if "net_base_line" in net:
        _line(out, *net["net_base_line"], COLORS["net_base_line"], 1)
    if "net_top_line" in net:
        _line(out, *net["net_top_line"], COLORS["net_top_line"], 1)
        _line(out, *net["net_bottom_full"], COLORS["net_base_line"], 1)
        _line(out, *net["net_post_far"], COLORS["net_post"], 1)
        _line(out, *net["net_post_near"], COLORS["net_post"], 1)
    lines = [f"{d.get('method', '')}  conf={d.get('confidence')}  used {d.get('n_used')}/{d.get('n_frames')}"]
    if d.get("per_corner_std_px"):
        lines.append("std px: " + " ".join(f"{v:.1f}" for v in d["per_corner_std_px"]))
    cam = d.get("camera")
    if cam:
        lines.append(f"f={cam['f_px']:.0f}px reproj={cam['reproj_rms_px']:.2f}px")
    for i, s in enumerate(lines):
        cv2.putText(out, s, (20, 40 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(out, s, (20, 40 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    return out


def corner_crops(
    img: np.ndarray,
    candidates: Dict[str, Sequence[Sequence[float]]],
    center: Optional[Sequence[Sequence[float]]] = None,
    half: int = 24,
    zoom: int = 8,
) -> np.ndarray:
    """四个角各裁一块放大，叠上各组候选角点（不同颜色），拼成一张图。

    candidates: 名字 → 4 个角（far_left, far_right, near_right, near_left）。
    center: 每块裁切的中心，默认取第一组候选。
    像素放大用最近邻，像素格子清晰可数；候选点按亚像素位置画。
    """
    names = list(candidates)
    if center is None:
        center = candidates[names[0]]
    tiles = []
    for i in range(4):
        cx, cy = int(round(center[i][0])), int(round(center[i][1]))
        x0, y0 = cx - half, cy - half
        crop = np.zeros((2 * half + 1, 2 * half + 1, 3), np.uint8)
        H, W = img.shape[:2]
        sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + 2 * half + 1, W), min(y0 + 2 * half + 1, H)
        crop[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
        big = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        for j, nm in enumerate(names):
            px, py = candidates[nm][i]
            # 原图像素 (x, y) 的中心在放大图里位于 ((x - x0) + 0.5) * zoom
            bx, by = ((px - x0) + 0.5) * zoom, ((py - y0) + 0.5) * zoom
            col = CANDIDATE_COLORS[j % len(CANDIDATE_COLORS)]
            r = 6 << _SHIFT
            p = _pt((bx, by))
            cv2.line(big, (p[0] - r, p[1]), (p[0] + r, p[1]), col, 2, cv2.LINE_AA, _SHIFT)
            cv2.line(big, (p[0], p[1] - r), (p[0], p[1] + r), col, 2, cv2.LINE_AA, _SHIFT)
        cv2.putText(big, f"{CORNER_NAMES[i]} ({cx},{cy})", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(big)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack([tiles[3], tiles[2]])])
    legend_h = 26 * len(names) + 10
    legend = np.zeros((legend_h, grid.shape[1], 3), np.uint8)
    for j, nm in enumerate(names):
        col = CANDIDATE_COLORS[j % len(CANDIDATE_COLORS)]
        pts = " ".join(f"({p[0]:.0f},{p[1]:.0f})" for p in candidates[nm])
        cv2.putText(legend, f"+ {nm}: {pts}", (8, 24 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return np.vstack([grid, legend])
