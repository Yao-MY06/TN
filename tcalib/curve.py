"""球台边在画面里的二次曲线模型（应对镜头畸变）。

镜头有桶形畸变时，台面长边在画面里是一条弧：game_4 的近边，两端相对中段偏了约 8 px。
用直线拟合，角点会被推到台外。这里把每条边写成相对弦的二次曲线：

    以弦 a→b 为基准，s = 沿弦距离 / 弦长（0~1），v = 沿外法线的偏移（像素）
    v(s) = c0 + c1*s + c2*s^2

c2 = 0 时就是直线。拱高（sagitta）= |c2| / 4，即曲线中点偏离端点连线的距离。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .table import SIDES


def side_frame(quad: np.ndarray, side_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """与 detect.refine_side 相同的取法：(起点 a, 弦方向, 外法线, 弦长)。"""
    _, ia, ib = SIDES[side_idx]
    a, b = quad[ia], quad[ib]
    d = b - a
    L = float(np.linalg.norm(d))
    t_hat = d / L
    n_hat = np.array([t_hat[1], -t_hat[0]])
    if np.dot((a + b) / 2 - quad.mean(axis=0), n_hat) < 0:
        n_hat = -n_hat
    return a.astype(np.float64), t_hat, n_hat, L


@dataclass
class SideCurve:
    a: np.ndarray
    t_hat: np.ndarray
    n_hat: np.ndarray
    L: float
    coef: np.ndarray  # (c0, c1, c2)

    def uv(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rel = np.asarray(pts, dtype=np.float64).reshape(-1, 2) - self.a
        return rel @ self.t_hat / self.L, rel @ self.n_hat

    def v_of(self, s: np.ndarray) -> np.ndarray:
        c0, c1, c2 = self.coef
        return c0 + c1 * s + c2 * s * s

    def residual(self, pts: np.ndarray) -> np.ndarray:
        """点到曲线的有符号距离（沿外法线；曲率小时近似成立）。"""
        s, v = self.uv(pts)
        return v - self.v_of(s)

    def point(self, s: np.ndarray) -> np.ndarray:
        s = np.atleast_1d(np.asarray(s, dtype=np.float64))
        return self.a[None, :] + (s * self.L)[:, None] * self.t_hat[None, :] + self.v_of(s)[:, None] * self.n_hat[None, :]

    @property
    def sagitta_px(self) -> float:
        return abs(float(self.coef[2])) / 4.0

    def polyline(self, s0: float = 0.0, s1: float = 1.0, n: int = 24) -> np.ndarray:
        return self.point(np.linspace(s0, s1, n))


def _lsq(s: np.ndarray, v: np.ndarray, quadratic: bool, fixed_c2: Optional[float] = None) -> np.ndarray:
    if fixed_c2 is not None:
        A = np.stack([np.ones_like(s), s], axis=1)
        c, *_ = np.linalg.lstsq(A, v - fixed_c2 * s * s, rcond=None)
        return np.array([c[0], c[1], fixed_c2])
    if quadratic:
        A = np.stack([np.ones_like(s), s, s * s], axis=1)
        c, *_ = np.linalg.lstsq(A, v, rcond=None)
        return c
    A = np.stack([np.ones_like(s), s], axis=1)
    c, *_ = np.linalg.lstsq(A, v, rcond=None)
    return np.array([c[0], c[1], 0.0])


def fit_curve(
    pts: np.ndarray,
    quad: np.ndarray,
    side_idx: int,
    init_mask: np.ndarray,
    thresh_px: float,
    max_sagitta_px: float,
    iters: int = 6,
    min_sagitta_px: float = 0.5,
) -> Tuple[SideCurve, np.ndarray]:
    """以 init_mask（直线 RANSAC 的内点）为起点，交替"二次拟合 → 按阈值重选内点"。

    弯曲的两端最初会被直线判成离群点，二次模型逐轮把它们收回来。
    拱高超过 max_sagitta_px 视为拟合到了别的东西，退回直线；
    拱高小于 min_sagitta_px 或统计上不显著时，用直线（少一个参数，抖动更小）。
    """
    a, t_hat, n_hat, L = side_frame(quad, side_idx)
    cur, mask = _iterate(pts, a, t_hat, n_hat, L, init_mask, thresh_px, max_sagitta_px, iters, quadratic=True)
    if not _curvature_significant(cur, pts[mask], min_sagitta_px):
        cur, mask = _iterate(pts, a, t_hat, n_hat, L, init_mask, thresh_px, max_sagitta_px, iters, quadratic=False)
    return cur, mask


def _curvature_significant(cur: "SideCurve", pts: np.ndarray, min_sagitta_px: float) -> bool:
    """拱高既要不小于 min_sagitta_px，又要在统计上显著（|c2| > 3 倍标准误）。"""
    if cur.coef[2] == 0.0 or len(pts) < 10 or cur.sagitta_px < min_sagitta_px:
        return False
    s, v = cur.uv(pts)
    A = np.stack([np.ones_like(s), s, s * s], axis=1)
    r = v - A @ cur.coef
    sigma2 = float(r @ r) / max(len(s) - 3, 1)
    try:
        cov = sigma2 * np.linalg.inv(A.T @ A)
    except np.linalg.LinAlgError:
        return False
    return abs(cur.coef[2]) > 3.0 * np.sqrt(max(cov[2, 2], 1e-18))


def _iterate(pts, a, t_hat, n_hat, L, init_mask, thresh_px, max_sagitta_px, iters, quadratic):
    cur = SideCurve(a, t_hat, n_hat, L, np.zeros(3))
    s, v = cur.uv(pts)
    mask = init_mask.copy()
    for _ in range(iters):
        if mask.sum() < 5:
            break
        coef = _lsq(s[mask], v[mask], quadratic)
        if quadratic and abs(coef[2]) / 4.0 > max_sagitta_px:
            quadratic = False
            coef = _lsq(s[mask], v[mask], False)
        cur = SideCurve(a, t_hat, n_hat, L, coef)
        new = np.abs(v - cur.v_of(s)) < thresh_px
        if np.array_equal(new, mask):
            break
        mask = new
    return cur, mask


def refit(curve: SideCurve, pts: np.ndarray, weights: Optional[np.ndarray] = None,
          fixed_c2: Optional[float] = None) -> SideCurve:
    """在同一坐标系下用给定点（可带整数权重）重新最小二乘。"""
    s, v = curve.uv(pts)
    if weights is not None:
        s, v = np.repeat(s, weights), np.repeat(v, weights)
    # 原曲线是直线（c2=0）时只拟合直线；fixed_c2 给定时只拟合 c0、c1
    coef = _lsq(s, v, quadratic=curve.coef[2] != 0.0, fixed_c2=fixed_c2)
    return SideCurve(curve.a, curve.t_hat, curve.n_hat, curve.L, coef)


def intersect_curves(A: SideCurve, B: SideCurve, s_guess: float) -> Optional[np.ndarray]:
    """曲线 A 上找一点使其落在曲线 B 上（割线法，从 A 的参数 s_guess 出发）。"""
    f = lambda s: float(B.residual(A.point(s))[0])  # noqa: E731
    s0, s1 = s_guess - 0.01, s_guess + 0.01
    f0, f1 = f(s0), f(s1)
    for _ in range(50):
        if abs(f1 - f0) < 1e-12:
            break
        s2 = s1 - f1 * (s1 - s0) / (f1 - f0)
        s0, f0, s1, f1 = s1, f1, s2, f(s2)
        if abs(f1) < 1e-6:
            break
    if not np.isfinite(s1) or abs(f1) > 0.05 or not (-0.5 < s1 < 1.5):
        return None
    return A.point(s1)[0]


def corners_from_curves(curves) -> Optional[np.ndarray]:
    """curves 按 far, right, near, left 排列；角 i = 边 (i-1) ∩ 边 i。

    在边 (i-1) 上，角 i 位于其终点附近（s≈1）。
    """
    out = []
    for i in range(4):
        x = intersect_curves(curves[(i - 1) % 4], curves[i], 1.0)
        if x is None:
            return None
        out.append(x)
    return np.array(out)


def offset_fn(curve: SideCurve, quad: np.ndarray, side_idx: int):
    """给 refine_side 用：以新四边形的弦为基准，曲线相对弦的外法线偏移 t∈[0,1] → px。"""
    a, t_hat, n_hat, L = side_frame(quad, side_idx)

    def fn(t: np.ndarray) -> np.ndarray:
        q = a[None, :] + (np.asarray(t) * L)[:, None] * t_hat[None, :]
        return -curve.residual(q) * float(np.dot(curve.n_hat, n_hat))

    return fn
