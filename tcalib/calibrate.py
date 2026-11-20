"""多帧标定：把同一机位的若干帧合成一份标定结果（四角、相机、球网、兼容字段）。

做法（v2，按边汇集）
  机位固定，所以四条边在所有帧里是同一条直线。打球时球员常站在球台两端，
  单帧里某条短边经常被挡掉大半；v1 要求"每帧四条边都找得到"，这样的帧整帧作废，
  导致有效帧很少。v2 改为：
  1. 每帧照常做单帧检测，只用来得到初始四边形和逐帧诊断；
  2. 以一个公共四边形为基准，在**每一帧**上沿四条边找白边外沿点，
     同一条边在所有帧上的点合在一起做 RANSAC 拟合（不同帧挡住的位置不同，合起来就补全了）；
  3. 每条边用二次曲线拟合（镜头畸变下边是弯的，见 curve.py），相邻两条曲线的交点为四角；
     以新四角为基准、沿曲线再窄带精修两轮。
  质量指标：
  * 覆盖率：每条边上，至少在一帧里找到内点的采样位置所占比例；
  * 逐帧抖动：四条边在同一帧里都有足够内点的帧，单独拟合出的四角相对汇总结果的偏差；
  * bootstrap：对帧有放回重采样后重新拟合，四角的标准差（输出本身稳不稳）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from .camera import CameraModel, net_geometry, solve_camera
from .curve import SideCurve, corners_from_curves, fit_curve, offset_fn, refit, side_frame
from .detect import (
    DetectParams,
    FrameResult,
    _is_convex,
    _whiteness,
    detect_frame,
    fit_line,
    order_corners,
    refine_side,
)
from .net import NET_TOP_HEIGHT_CM, NET_TOP_MIN_COVERAGE, detect_net_top, projected_net_top
from .table import CORNER_NAMES, NET_HEIGHT_CM, NET_POST_OUT_CM, NET_X_CM, SIDES, TABLE_WIDTH_CM

# 置信度阈值（验收标准里的数字，集中放在这里）
HIGH_MAX_STD_PX = 2.0
MEDIUM_MAX_STD_PX = 5.0
MAX_REPROJ_PX = 3.0
HIGH_MIN_COVERAGE = 0.6
MEDIUM_MIN_COVERAGE = 0.3
MIN_FULL_FRAMES_FOR_JITTER = 3  # 少于这么多"四边俱全"的帧，抖动改用 bootstrap
BOOTSTRAP_ROUNDS = 40
# 一帧在某条边上至少要有这么多内点（占采样数的比例），才算这帧"这条边看得到"
FRAME_SIDE_MIN_FRAC = 0.3
# 边的拱高上限（占弦长比例）：超过就当作拟合到了别的东西，退回直线
MAX_SAGITTA_FRAC = 0.02
# 检测裁剪框 roi_box 相对台面外接框的余量（占外接框宽度的比例），照国栋手填预设的习惯：
# pingpong-smart-referee/core/geometry.py 的 game2、test4（与本项目同一套 OpenTTGames 机位）
# 两组手填值折算下来左右各约 0.19~0.28、上方约 0.28~0.29、下方约 0.16~0.18。
ROI_MARGIN_SIDE = 0.22
ROI_MARGIN_TOP = 0.285
ROI_MARGIN_BOTTOM = 0.17
# 球擦网时球心最多高出网顶一个球径（4 cm），net_zone 的上沿按此外扩
NET_ZONE_EXTRA_CM = 4.0


@dataclass
class SideStat:
    n_samples: int
    n_points: int
    n_inliers: int
    coverage: float
    rms_px: float
    frame_inliers: List[int]
    sagitta_px: float = 0.0  # 边的弯曲程度（拱高，像素）

    def to_dict(self) -> Dict:
        return {
            "n_samples": self.n_samples,
            "n_points": self.n_points,
            "n_inliers": self.n_inliers,
            "coverage": round(self.coverage, 3),
            "rms_px": round(self.rms_px, 3),
            "sagitta_px": round(self.sagitta_px, 2),
        }


@dataclass
class CalibResult:
    corners: Optional[np.ndarray]
    confidence: str  # high / medium / low
    reasons: List[str]
    per_corner_std_px: Optional[np.ndarray]
    std_method: str  # frame_jitter / bootstrap / none
    n_frames: int
    n_used: int  # 至少给一条边贡献了内点的帧数
    frame_ids: List[str]
    used_ids: List[str]
    frame_results: Dict[str, FrameResult] = field(repr=False, default_factory=dict)
    side_stats: Dict[str, SideStat] = field(default_factory=dict)
    frame_jitter_px: Optional[np.ndarray] = None
    bootstrap_std_px: Optional[np.ndarray] = None
    full_frame_ids: List[str] = field(default_factory=list)
    camera: Optional[CameraModel] = None
    net: Dict = field(default_factory=dict)
    compat: Dict = field(default_factory=dict)
    side_curves: Dict[str, SideCurve] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 按边汇集拟合
# ---------------------------------------------------------------------------


@dataclass
class _SidePool:
    pts: np.ndarray
    idx: np.ndarray
    frame: np.ndarray
    inlier: np.ndarray
    n_samples: int
    curve: SideCurve


def _pooled_pass(whites: List[np.ndarray], quad: np.ndarray, band: float, params: DetectParams, rng,
                 prev: Optional[Dict[str, "_SidePool"]] = None):
    """以 quad 为基准在所有帧上采点，每条边合并拟合（二次曲线）。失败返回 (None, 原因)。

    prev 给定时：搜索带沿上一轮的曲线弯曲，初始内点按上一轮曲线选；否则先用直线 RANSAC 起步。
    """
    p_fit = replace(params, ransac_thresh_px=params.pooled_ransac_thresh_px, ransac_iters=max(params.ransac_iters, 400))
    curves, pools = [], {}
    for si, (name, _, _) in enumerate(SIDES):
        off = offset_fn(prev[name].curve, quad, si) if prev is not None else None
        P, I, F, n_s = [], [], [], 0
        for fi, w in enumerate(whites):
            pts, n_s, idx = refine_side(w, quad, si, band, params, return_index=True, center_offset=off)
            P.append(pts)
            I.append(idx)
            F.append(np.full(len(pts), fi, dtype=int))
        P, I, F = np.vstack(P), np.concatenate(I), np.concatenate(F)
        if len(P) < 5:
            return None, f"side_{name}_no_points"
        if prev is not None:
            init = np.abs(prev[name].curve.residual(P)) < params.pooled_ransac_thresh_px
        else:
            fit = fit_line(P, p_fit, rng)
            if fit is None:
                return None, f"side_{name}_no_points"
            init = fit[2]
        _, _, _, L = side_frame(quad, si)
        curve, mask = fit_curve(P, quad, si, init, params.pooled_ransac_thresh_px, MAX_SAGITTA_FRAC * L)
        if mask.sum() < 5:
            return None, f"side_{name}_no_points"
        curves.append(curve)
        pools[name] = _SidePool(P, I, F, mask, n_s, curve)
    corners = corners_from_curves(curves)
    if corners is None:
        return None, "curves_do_not_intersect"
    if not _is_convex(corners):
        return None, "bad_quad"
    return (corners, pools), ""


def _side_stats(pools: Dict[str, _SidePool], n_frames: int) -> Dict[str, SideStat]:
    out = {}
    for name, _, _ in SIDES:
        pl = pools[name]
        p = pl.pts[pl.inlier]
        rms = float(np.sqrt(np.mean(pl.curve.residual(p) ** 2))) if len(p) else float("nan")
        cov = len(np.unique(pl.idx[pl.inlier])) / max(pl.n_samples, 1)
        fin = np.bincount(pl.frame[pl.inlier], minlength=n_frames).tolist()
        out[name] = SideStat(pl.n_samples, len(pl.pts), int(pl.inlier.sum()), cov, rms, fin, pl.curve.sagitta_px)
    return out


def _frame_jitter(pools: Dict[str, _SidePool], corners: np.ndarray, n_frames: int, ids: List[str]):
    """四条边在同一帧里都有足够内点的帧，单独拟合四角（曲率沿用汇总结果，只拟合位置和方向）；
    返回 (每角相对汇总四角的 RMS 偏差, 这些帧的帧号)。"""
    per_frame = []
    full_ids = []
    for fi in range(n_frames):
        curves = []
        for name, _, _ in SIDES:
            pl = pools[name]
            sel = pl.inlier & (pl.frame == fi)
            if sel.sum() < max(5, FRAME_SIDE_MIN_FRAC * pl.n_samples):
                break
            curves.append(refit(pl.curve, pl.pts[sel], fixed_c2=float(pl.curve.coef[2])))
        else:
            c = corners_from_curves(curves)
            if c is not None:
                per_frame.append(c)
                full_ids.append(ids[fi])
    if len(per_frame) < MIN_FULL_FRAMES_FOR_JITTER:
        return None, full_ids
    arr = np.stack(per_frame)
    return np.sqrt(np.mean(np.sum((arr - corners[None]) ** 2, axis=2), axis=0)), full_ids


def _bootstrap(pools: Dict[str, _SidePool], n_frames: int, rng) -> Optional[np.ndarray]:
    """对帧有放回重采样，用各边已定的内点重新最小二乘，四角的标准差。"""
    if n_frames < 2:
        return None
    samples = []
    for _ in range(BOOTSTRAP_ROUNDS):
        pick = np.bincount(rng.integers(0, n_frames, n_frames), minlength=n_frames)
        curves = []
        for name, _, _ in SIDES:
            pl = pools[name]
            w = pick[pl.frame[pl.inlier]]
            if w.sum() < 5:
                break
            curves.append(refit(pl.curve, pl.pts[pl.inlier], weights=w))
        else:
            c = corners_from_curves(curves)
            if c is not None:
                samples.append(c)
    if len(samples) < 5:
        return None
    arr = np.stack(samples)
    m = arr.mean(axis=0)
    return np.sqrt(np.mean(np.sum((arr - m[None]) ** 2, axis=2), axis=0))


def _initial_quad(results: Dict[str, FrameResult]) -> Optional[np.ndarray]:
    ok = [r.corners for r in results.values() if r.ok and r.corners is not None]
    if ok:
        return np.median(np.stack(ok), axis=0)
    coarse = [r.coarse for r in results.values() if r.coarse is not None]
    if coarse:
        return np.median(np.stack(coarse), axis=0)
    return None


# ---------------------------------------------------------------------------
# 兼容字段
# ---------------------------------------------------------------------------


def compat_fields(corners: np.ndarray, cam: Optional[CameraModel], net: Dict, image_size: Tuple[int, int]) -> Dict:
    """国栋 TableCalibration 需要的字段（像素，整数）。"""
    w, h = image_size
    out: Dict = {
        "table_polygon": [[int(round(x)), int(round(y))] for x, y in corners],
        "base_width": int(w),
        "base_height": int(h),
    }
    # net_x：网底线在"四角重心的 y"处的 x。这样取与国栋 TableHomography.calibration_net_x_cm()
    # 的自检口径一致（默认也取四角重心的 y），反推回去正好是 137 cm。
    base = np.array(net["net_base_line"])
    yc = float(np.mean(corners[:, 1]))
    (bx0, by0), (bx1, by1) = base
    t = 0.0 if abs(by1 - by0) < 1e-9 else (yc - by0) / (by1 - by0)
    out["net_x"] = int(round(bx0 + t * (bx1 - bx0)))
    out["roi_box"] = roi_like_presets(corners, image_size)
    if cam is None:
        out["net_zone"] = None
        out["note"] = "相机解算失败：只给出网底线对应的 net_x，net_zone 需手填"
        return out
    top_projected = projected_net_top(cam)
    above = cam.project(
        np.array(
            [
                [NET_X_CM, -NET_POST_OUT_CM, NET_TOP_HEIGHT_CM + NET_ZONE_EXTRA_CM],
                [NET_X_CM, TABLE_WIDTH_CM + NET_POST_OUT_CM, NET_TOP_HEIGHT_CM + NET_ZONE_EXTRA_CM],
            ]
        )
    )
    top_line = np.asarray(net["net_top_line"], dtype=float)
    i = int(np.argmin(top_line[:, 1]))
    y0 = int(np.floor(top_line[i, 1] + above[i, 1] - top_projected[i, 1]))
    y1 = int(np.ceil(np.array(net["net_bottom_full"])[:, 1].max()))
    out["net_zone"] = [max(y0, 0), min(y1, h)]
    return out


def roi_like_presets(corners: np.ndarray, image_size: Tuple[int, int]) -> List[int]:
    """按国栋手填预设的习惯，由台面外接框外扩得到检测裁剪框 (y_min, y_max, x_min, x_max)。"""
    w, h = image_size
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    tw = x1 - x0
    return [
        max(int(round(y0 - ROI_MARGIN_TOP * tw)), 0),
        min(int(round(y1 + ROI_MARGIN_BOTTOM * tw)), h),
        max(int(round(x0 - ROI_MARGIN_SIDE * tw)), 0),
        min(int(round(x1 + ROI_MARGIN_SIDE * tw)), w),
    ]


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def calibrate_frames(
    frames: Dict[str, np.ndarray],
    params: DetectParams = DetectParams(),
    init_quad: Optional[np.ndarray] = None,
    seed: int = 0,
) -> CalibResult:
    """frames: 帧号 → BGR 图像（同一机位、同一分辨率）。init_quad 给定时用它当初始四边形（半自动）。"""
    ids = sorted(frames)
    if not ids:
        raise ValueError("no frames")
    h, w = frames[ids[0]].shape[:2]
    rng = np.random.default_rng(seed)
    results = {k: detect_frame(frames[k], init_quad=init_quad, params=params, seed=i) for i, k in enumerate(ids)}

    def fail(reason: str) -> CalibResult:
        return CalibResult(None, "low", [reason], None, "none", len(ids), 0, ids, [], results)

    quad = order_corners(np.asarray(init_quad, float)) if init_quad is not None else _initial_quad(results)
    if quad is None:
        return fail("no_initial_quad")
    whites = [_whiteness(frames[k]) for k in ids]
    pools = None
    for band in (params.band_px, params.band2_px, params.band2_px):
        out, why = _pooled_pass(whites, quad, band, params, rng, prev=pools)
        if out is None:
            return fail(why)
        quad, pools = out
    corners = quad

    stats = _side_stats(pools, len(ids))
    jitter, full_ids = _frame_jitter(pools, corners, len(ids), ids)
    boot = _bootstrap(pools, len(ids), rng)
    if jitter is not None:
        std, std_method = jitter, "frame_jitter"
    elif boot is not None:
        std, std_method = boot, "bootstrap"
    else:
        std, std_method = None, "none"
    used = [ids[fi] for fi in range(len(ids)) if any(stats[n].frame_inliers[fi] > 0 for n, _, _ in SIDES)]

    reasons: List[str] = []
    level = {"high": 0, "medium": 1, "low": 2}
    conf = "high"

    def downgrade(to: str, why: str) -> None:
        nonlocal conf
        reasons.append(why)
        if level[to] > level[conf]:
            conf = to

    for name, st in stats.items():
        if st.coverage < MEDIUM_MIN_COVERAGE:
            downgrade("low", f"coverage_{name}={st.coverage:.2f}<{MEDIUM_MIN_COVERAGE}")
        elif st.coverage < HIGH_MIN_COVERAGE:
            downgrade("medium", f"coverage_{name}={st.coverage:.2f}<{HIGH_MIN_COVERAGE}")
    if std is None:
        downgrade("medium", "std_unavailable")
    else:
        max_std = float(std.max())
        if max_std > MEDIUM_MAX_STD_PX:
            downgrade("low", f"max_std={max_std:.1f}px>{MEDIUM_MAX_STD_PX}")
        elif max_std > HIGH_MAX_STD_PX:
            downgrade("medium", f"max_std={max_std:.1f}px>{HIGH_MAX_STD_PX}")

    cam: Optional[CameraModel]
    try:
        cam = solve_camera(corners, (w, h))
    except Exception as e:  # noqa: BLE001  相机失败不影响四角本身
        cam = None
        downgrade("medium", f"camera_failed:{e}")
    if cam is not None:
        if cam.reproj_rms_px > MAX_REPROJ_PX:
            downgrade("medium", f"reproj={cam.reproj_rms_px:.2f}px>{MAX_REPROJ_PX}")
        if not (0.3 * w < cam.f < 10 * w):
            downgrade("low", f"focal={cam.f:.0f}px_implausible")
    net = net_geometry(corners, cam)
    if cam is not None:
        projected = projected_net_top(cam)
        detected, net_stats = detect_net_top(frames, cam, projected)
        net["net_top_line_projected"] = projected.tolist()
        net["net_top_stats"] = net_stats
        if net_stats["coverage"] >= NET_TOP_MIN_COVERAGE:
            net["net_top_line"] = detected.tolist()
            net["net_top_source"] = "detected"
        else:
            net["net_top_line"] = projected.tolist()
            net["net_top_source"] = "projected"
    else:
        net["net_top_line_projected"] = None
        net["net_top_source"] = "projected"
        net["net_top_stats"] = {"coverage": 0.0, "n_points": 0, "n_inliers": 0,
                                   "rms_px": None, "offset_px": None}

    res = CalibResult(
        corners, conf, reasons, std, std_method, len(ids), len(used), ids, used, results,
        stats, jitter, boot, full_ids, cam, net,
    )
    res.compat = compat_fields(corners, cam, net, (w, h))
    res.side_curves = {name: pools[name].curve for name, _, _ in SIDES}
    return res


def _r(a: Optional[np.ndarray]):
    return None if a is None else [round(float(v), 2) for v in a]


def result_to_dict(res: CalibResult, name: str, method: str) -> Dict:
    """主输出格式：沿用 table_quads.json 的写法，再加网、置信度、相机、每条边的统计。"""
    d: Dict = {
        "points": None if res.corners is None else [[round(float(x), 2), round(float(y), 2)] for x, y in res.corners],
        "corner_order": list(CORNER_NAMES),
        "corner_definition": "white_line_outer_edge",
        "method": method,
        "confidence": res.confidence,
        "confidence_reasons": res.reasons,
        "per_corner_std_px": _r(res.per_corner_std_px),
        "std_method": res.std_method,
        "frame_jitter_px": _r(res.frame_jitter_px),
        "bootstrap_std_px": _r(res.bootstrap_std_px),
        "n_frames": res.n_frames,
        "n_used": res.n_used,
        "n_full_frames": len(res.full_frame_ids),
        "sides": {k: v.to_dict() for k, v in res.side_stats.items()},
        "used_frames": res.used_ids,
        "per_frame_ok": [k for k, r in res.frame_results.items() if r.ok],
    }
    if res.corners is not None:
        d["net"] = {}
        for k, v in res.net.items():
            if k == "net_top_stats":
                d["net"][k] = {name: round(value, 3) if isinstance(value, float) else value
                               for name, value in v.items()}
            elif k == "net_top_source" or v is None:
                d["net"][k] = v
            else:
                d["net"][k] = [[round(float(x), 2), round(float(y), 2)] for x, y in v]
        d["camera"] = None if res.camera is None else res.camera.to_dict()
        d["table_calibration"] = dict(name=name, **res.compat)
        # 每条边的拟合曲线（画回原图目检用；镜头畸变下边是弯的）
        d["side_curves"] = {
            k: [[round(float(x), 2), round(float(y), 2)] for x, y in c.polyline(-0.02, 1.02, 26)]
            for k, c in res.side_curves.items()
        }
    return d
