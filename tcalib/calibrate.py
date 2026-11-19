"""多帧汇总 + 相机 + 球网 + 兼容字段：把一个机位的若干帧变成一份标定结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .camera import CameraModel, net_geometry, solve_camera, suggest_roi
from .detect import DetectParams, FrameResult, detect_frame
from .table import CORNER_NAMES, NET_HEIGHT_CM, NET_POST_OUT_CM, NET_X_CM, TABLE_WIDTH_CM

# 置信度阈值（验收标准里的数字，集中放在这里）
HIGH_MAX_STD_PX = 2.0
MEDIUM_MAX_STD_PX = 5.0
MAX_REPROJ_PX = 2.0
MIN_VALID_FRAC = 0.5
# 球擦网时球心最多高出网顶一个球径（4 cm），net_zone 的上沿按此外扩
NET_ZONE_EXTRA_CM = 4.0


@dataclass
class CalibResult:
    corners: Optional[np.ndarray]
    confidence: str  # high / medium / low
    reasons: List[str]
    per_corner_std_px: Optional[np.ndarray]
    n_frames: int
    n_used: int
    frame_ids: List[str]
    used_ids: List[str]
    frame_results: Dict[str, FrameResult] = field(repr=False, default_factory=dict)
    camera: Optional[CameraModel] = None
    net: Dict = field(default_factory=dict)
    compat: Dict = field(default_factory=dict)


def aggregate(results: Dict[str, FrameResult], outlier_px: float = 4.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """逐角中位数；与中位数偏离过大的帧整帧剔除后再算一次。

    返回 (四角, 每个角的标准差, 参与汇总的帧号)。
    """
    ok = {k: r.corners for k, r in results.items() if r.ok and r.corners is not None}
    if not ok:
        return None, None, []
    ids = sorted(ok)
    arr = np.stack([ok[k] for k in ids])  # (N, 4, 2)
    med = np.median(arr, axis=0)
    dev = np.linalg.norm(arr - med[None], axis=2)  # (N, 4)
    mad = np.median(dev, axis=0)
    thr = np.maximum(outlier_px, 3.0 * 1.4826 * mad)
    keep = np.all(dev <= thr[None], axis=1)
    if keep.sum() >= max(2, int(0.3 * len(ids))):
        arr = arr[keep]
        ids = [k for k, m in zip(ids, keep) if m]
        med = np.median(arr, axis=0)
    std = np.sqrt(np.mean(np.sum((arr - med[None]) ** 2, axis=2), axis=0)) if len(arr) > 1 else np.zeros(4)
    return med, std, ids


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
    if cam is None:
        out["net_zone"] = None
        out["roi_box"] = None
        out["note"] = "相机解算失败：只给出网底线对应的 net_x，net_zone / roi_box 需手填"
        return out
    top = cam.project(
        np.array(
            [
                [NET_X_CM, -NET_POST_OUT_CM, NET_HEIGHT_CM + NET_ZONE_EXTRA_CM],
                [NET_X_CM, TABLE_WIDTH_CM + NET_POST_OUT_CM, NET_HEIGHT_CM + NET_ZONE_EXTRA_CM],
            ]
        )
    )
    y0 = int(np.floor(top[:, 1].min()))
    y1 = int(np.ceil(np.array(net["net_bottom_full"])[:, 1].max()))
    out["net_zone"] = [max(y0, 0), min(y1, h)]
    out["roi_box"] = list(suggest_roi(cam, image_size))
    return out


def calibrate_results(
    results: Dict[str, FrameResult],
    image_size: Tuple[int, int],
) -> CalibResult:
    """对已经完成单帧检测的结果做多帧汇总；CLI 预检后复用，避免重复检测。"""
    ids = sorted(results)
    if not ids:
        raise ValueError("no frames")
    w, h = image_size
    corners, std, used = aggregate(results)
    reasons: List[str] = []
    if corners is None:
        return CalibResult(None, "low", ["no_valid_frame"], None, len(ids), 0, ids, [], results)

    cam: Optional[CameraModel]
    try:
        cam = solve_camera(corners, (w, h))
    except Exception as e:  # noqa: BLE001  相机失败不影响四角本身
        cam = None
        reasons.append(f"camera_failed:{e}")
    net = net_geometry(corners, cam)

    valid_frac = len(used) / len(ids)
    max_std = float(std.max())
    conf = "high"
    if valid_frac < MIN_VALID_FRAC:
        reasons.append(f"valid_frac={valid_frac:.2f}<{MIN_VALID_FRAC}")
        conf = "low"
    if max_std > MEDIUM_MAX_STD_PX:
        reasons.append(f"max_std={max_std:.1f}px>{MEDIUM_MAX_STD_PX}")
        conf = "low"
    elif max_std > HIGH_MAX_STD_PX and conf == "high":
        reasons.append(f"max_std={max_std:.1f}px>{HIGH_MAX_STD_PX}")
        conf = "medium"
    if cam is None:
        conf = "low" if conf == "low" else "medium"
    else:
        if cam.reproj_rms_px > MAX_REPROJ_PX:
            reasons.append(f"reproj={cam.reproj_rms_px:.2f}px>{MAX_REPROJ_PX}")
            conf = "low" if conf == "low" else "medium"
        if not (0.3 * w < cam.f < 10 * w):
            reasons.append(f"focal={cam.f:.0f}px_implausible")
            conf = "low"
    res = CalibResult(corners, conf, reasons, std, len(ids), len(used), ids, used, results, cam, net)
    res.compat = compat_fields(corners, cam, net, (w, h))
    return res


def calibrate_frames(
    frames: Dict[str, np.ndarray],
    params: DetectParams = DetectParams(),
    init_quad: Optional[np.ndarray] = None,
) -> CalibResult:
    """frames: 帧号 → BGR 图像（同一机位、同一分辨率）。init_quad 给定时走半自动。"""
    ids = sorted(frames)
    if not ids:
        raise ValueError("no frames")
    h, w = frames[ids[0]].shape[:2]
    results = {k: detect_frame(frames[k], init_quad=init_quad, params=params, seed=i)
               for i, k in enumerate(ids)}
    return calibrate_results(results, (w, h))


def result_to_dict(res: CalibResult, name: str, method: str) -> Dict:
    """主输出格式：沿用 table_quads.json 的写法，再加网、置信度、相机。"""
    d: Dict = {
        "points": None if res.corners is None else [[round(float(x), 2), round(float(y), 2)] for x, y in res.corners],
        "corner_order": list(CORNER_NAMES),
        "method": method,
        "confidence": res.confidence,
        "confidence_reasons": res.reasons,
        "per_corner_std_px": None if res.per_corner_std_px is None else [round(float(v), 2) for v in res.per_corner_std_px],
        "n_frames": res.n_frames,
        "n_used": res.n_used,
        "used_frames": res.used_ids,
        "rejected_frames": {
            k: r.reason or "outlier" for k, r in res.frame_results.items() if k not in res.used_ids
        },
    }
    if res.corners is not None:
        d["net"] = {k: [[round(float(x), 2), round(float(y), 2)] for x, y in v] for k, v in res.net.items()}
        d["camera"] = None if res.camera is None else res.camera.to_dict()
        d["table_calibration"] = dict(name=name, **res.compat)
    return d
