"""由球台四角求单应性和相机（PnPf：焦距未知），并把球网投影回图像。

相机模型的假设：主点在图像中心、像素是方的、不考虑镜头畸变。
4 个共面点给出 8 个约束，而相机只有 7 个未知量（旋转 3、平移 3、焦距 1），
所以多出的 1 个约束可以拿来自检：重投影误差大，说明四角不自洽，
或者上面的假设不成立（比如画面被裁过、主点不在中心）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .table import (
    NET_HEIGHT_CM,
    NET_POST_OUT_CM,
    NET_X_CM,
    TABLE_CORNERS_CM,
    TABLE_LENGTH_CM,
    TABLE_WIDTH_CM,
)


def homography_from_corners(corners_px: Sequence[Sequence[float]]) -> np.ndarray:
    """台面厘米坐标 → 图像像素的单应矩阵（3x3）。"""
    src = TABLE_CORNERS_CM.astype(np.float32)
    dst = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return ph[:, :2] / ph[:, 2:3]


def focal_from_homography(H: np.ndarray, principal: Tuple[float, float]) -> Optional[float]:
    """闭式解焦距（Zhang 的两条正交约束，主点已知、像素方形）。

    返回 None 表示退化（台面几乎正对相机，焦距不可观测）或无正解。
    """
    cx, cy = principal
    T = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    Hp = T @ H
    h1, h2 = Hp[:, 0], Hp[:, 1]
    # 约束写成 a * w + b = 0，w = 1/f^2
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    denom = float(a @ a)
    if denom < 1e-18:
        return None
    w = -float(a @ b) / denom
    if w <= 0:
        return None
    return float(1.0 / np.sqrt(w))


@dataclass
class CameraModel:
    f: float
    cx: float
    cy: float
    rvec: np.ndarray
    tvec: np.ndarray
    up_sign: float  # 台面坐标里"向上"对应的 Z 符号（+1 或 -1）
    reproj_rms_px: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0.0, self.cx], [0.0, self.f, self.cy], [0.0, 0.0, 1.0]])

    def project(self, pts_xyh: np.ndarray) -> np.ndarray:
        """台面坐标 (x, y, 高度h) → 图像像素。高度向上为正。"""
        p = np.asarray(pts_xyh, dtype=np.float64).reshape(-1, 3).copy()
        p[:, 2] *= self.up_sign
        img, _ = cv2.projectPoints(p, self.rvec, self.tvec, self.K, None)
        return img.reshape(-1, 2)

    def camera_center_cm(self) -> np.ndarray:
        R, _ = cv2.Rodrigues(self.rvec)
        c = -R.T @ self.tvec.reshape(3)
        c[2] *= self.up_sign
        return c

    def to_dict(self) -> Dict:
        return {
            "f_px": round(self.f, 2),
            "principal_px": [round(self.cx, 2), round(self.cy, 2)],
            "rvec": [round(float(v), 6) for v in self.rvec.reshape(3)],
            "tvec_cm": [round(float(v), 3) for v in self.tvec.reshape(3)],
            "up_sign": int(self.up_sign),
            "camera_center_cm": [round(float(v), 1) for v in self.camera_center_cm()],
            "reproj_rms_px": round(self.reproj_rms_px, 3),
        }


def _pnp_at_focal(corners_px: np.ndarray, f: float, cx: float, cy: float):
    obj = np.hstack([TABLE_CORNERS_CM, np.zeros((4, 1))]).astype(np.float64)
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    ok, rvec, tvec = cv2.solvePnP(obj, corners_px.astype(np.float64), K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - corners_px) ** 2, axis=1))))
    return rms, rvec, tvec


def solve_camera(corners_px: Sequence[Sequence[float]], image_size: Tuple[int, int]) -> CameraModel:
    """PnPf：在焦距上做一维搜索，每个焦距下用 IPPE 解位姿，取重投影误差最小者。

    image_size = (width, height)。
    """
    w, h = image_size
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    c = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)

    def cost(logf: float) -> float:
        r = _pnp_at_focal(c, float(np.exp(logf)), cx, cy)
        # IPPE 在个别焦距上会返回 NaN，当成无效
        return 1e9 if r is None or not np.isfinite(r[0]) else r[0]

    # 粗扫：0.3w ~ 10w，对数均匀
    grid = np.linspace(np.log(0.3 * w), np.log(10.0 * w), 120)
    costs = np.array([cost(g) for g in grid])
    f0 = focal_from_homography(homography_from_corners(c), (cx, cy))
    if f0 is not None and 0.3 * w < f0 < 10.0 * w:
        grid = np.append(grid, np.log(f0))
        costs = np.append(costs, cost(np.log(f0)))
    order = np.argsort(grid)
    grid, costs = grid[order], costs[order]
    i = int(np.argmin(costs))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, len(grid) - 1)]
    # 黄金分割细化
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    x1, x2 = b - g * (b - a), a + g * (b - a)
    c1, c2 = cost(x1), cost(x2)
    for _ in range(60):
        if c1 < c2:
            b, x2, c2 = x2, x1, c1
            x1 = b - g * (b - a)
            c1 = cost(x1)
        else:
            a, x1, c1 = x1, x2, c2
            x2 = a + g * (b - a)
            c2 = cost(x2)
    best = x1 if c1 < c2 else x2
    if cost(best) > costs[i]:
        best = grid[i]
    f = float(np.exp(best))
    rms, rvec, tvec = _pnp_at_focal(c, f, cx, cy)

    R, _ = cv2.Rodrigues(rvec)
    center = -R.T @ tvec.reshape(3)
    up_sign = 1.0 if center[2] > 0 else -1.0
    return CameraModel(f=f, cx=cx, cy=cy, rvec=rvec, tvec=tvec, up_sign=up_sign, reproj_rms_px=rms)


def net_geometry(corners_px: Sequence[Sequence[float]], cam: Optional[CameraModel]) -> Dict:
    """球网在图像上的几何。

    网底线只用单应性（两条长边中点的连线），不依赖相机假设；
    网顶线和网柱需要相机，cam 为 None 时省略。
    所有点的顺序都是"远侧在前、近侧在后"。
    """
    H = homography_from_corners(corners_px)
    base = apply_h(H, np.array([[NET_X_CM, 0.0], [NET_X_CM, TABLE_WIDTH_CM]]))
    out: Dict = {"net_base_line": base.tolist()}
    if cam is None:
        return out
    y_far, y_near = -NET_POST_OUT_CM, TABLE_WIDTH_CM + NET_POST_OUT_CM
    pts = np.array(
        [
            [NET_X_CM, y_far, 0.0],
            [NET_X_CM, y_near, 0.0],
            [NET_X_CM, y_far, NET_HEIGHT_CM],
            [NET_X_CM, y_near, NET_HEIGHT_CM],
            [NET_X_CM, 0.0, NET_HEIGHT_CM],
            [NET_X_CM, TABLE_WIDTH_CM, NET_HEIGHT_CM],
        ]
    )
    p = cam.project(pts)
    out["net_bottom_full"] = [p[0].tolist(), p[1].tolist()]  # 网底延长到两侧网柱
    out["net_top_line"] = [p[2].tolist(), p[3].tolist()]  # 网顶，两端到网柱
    out["net_top_over_table"] = [p[4].tolist(), p[5].tolist()]
    out["net_post_far"] = [p[0].tolist(), p[2].tolist()]  # 底 → 顶
    out["net_post_near"] = [p[1].tolist(), p[3].tolist()]
    # 网面四边形：远底、近底、近顶、远顶
    out["net_polygon"] = [p[0].tolist(), p[1].tolist(), p[3].tolist(), p[2].tolist()]
    return out
