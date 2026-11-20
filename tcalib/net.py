"""从图像中检测球网白色顶带。"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

from .camera import CameraModel
from .detect import DetectParams, _whiteness, fit_line
from .table import NET_POST_OUT_CM, NET_X_CM, TABLE_WIDTH_CM

NET_TOP_HEIGHT_CM = 14.5
NET_TOP_MIN_COVERAGE = 0.35
NET_TOP_SAMPLE_SPACING_PX = 2.0
NET_TOP_SEARCH_PX = 15.0
NET_TOP_MIN_CONTRAST = 18.0


def projected_net_top(cam: CameraModel) -> np.ndarray:
    y0, y1 = -NET_POST_OUT_CM, TABLE_WIDTH_CM + NET_POST_OUT_CM
    return cam.project(np.array([[NET_X_CM, y0, NET_TOP_HEIGHT_CM], [NET_X_CM, y1, NET_TOP_HEIGHT_CM]]))


def _profile_points(white: np.ndarray, line: np.ndarray):
    d = line[1] - line[0]
    length = float(np.linalg.norm(d))
    tangent = d / length
    normal = np.array([-tangent[1], tangent[0]])
    n = max(int(0.9 * length / NET_TOP_SAMPLE_SPACING_PX), 2)
    centers = line[0] + np.linspace(0.05, 0.95, n)[:, None] * d
    offsets = np.arange(-NET_TOP_SEARCH_PX, NET_TOP_SEARCH_PX + 0.01, 0.5)
    coords = centers[:, None, :] + offsets[None, :, None] * normal
    prof = cv2.remap(white, coords[..., 0].astype(np.float32), coords[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    prof = cv2.GaussianBlur(prof, (0, 1), sigmaX=0.65)
    points, indices = [], []
    for i, (center, row) in enumerate(zip(centers, prof)):
        peak = int(np.argmax(row))
        if peak < 7 or peak >= len(offsets) - 8:
            continue
        left = float(row[peak - 7:peak - 3].mean())
        right = float(row[peak + 4:peak + 8].mean())
        if row[peak] - max(left, right) < NET_TOP_MIN_CONTRAST:
            continue
        y0, y1, y2 = row[peak - 1:peak + 2]
        den = float(y0 - 2 * y1 + y2)
        delta = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if abs(den) > 1e-9 else 0.0
        points.append(center + (offsets[peak] + delta * 0.5) * normal)
        indices.append(i)
    return np.asarray(points, dtype=np.float64).reshape(-1, 2), np.asarray(indices, dtype=int), n


def detect_net_top(
    frames: Dict[str, np.ndarray], cam: CameraModel, projected: Sequence[Sequence[float]] | None = None,
) -> Tuple[np.ndarray, Dict]:
    """多帧合并检测网顶线，返回端点和质量统计。"""
    prior = projected_net_top(cam) if projected is None else np.asarray(projected, dtype=np.float64).reshape(2, 2)
    samples = [_profile_points(_whiteness(frames[name]), prior) for name in sorted(frames)]
    points = np.vstack([s[0] for s in samples])
    indices = np.concatenate([s[1] for s in samples])
    n_samples = samples[0][2]
    params = DetectParams(ransac_thresh_px=1.5, ransac_iters=500)
    fit = fit_line(points, params, np.random.default_rng(0)) if len(points) >= 5 else None
    if fit is None:
        coverage = len(np.unique(indices)) / n_samples
        return prior.copy(), {"coverage": coverage, "n_points": len(points), "n_inliers": 0, "rms_px": None, "offset_px": None}
    origin, direction, inliers, rms = fit
    d = prior[1] - prior[0]
    length = float(np.linalg.norm(d))
    t = d / length
    # 把投影网柱位置垂直到检测直线，作为最终网顶端点。
    feet = origin + ((prior - origin) @ direction)[:, None] * direction
    coverage = len(np.unique(indices[inliers])) / n_samples
    signed = (points[inliers] - prior[0]) @ np.array([-t[1], t[0]])
    stats = {
        "coverage": coverage, "n_points": len(points), "n_inliers": int(inliers.sum()),
        "rms_px": float(rms), "offset_px": float(np.median(signed)),
    }
    return feet, stats
