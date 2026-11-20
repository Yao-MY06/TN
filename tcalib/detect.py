"""单帧球台四角检测：台面粗定位 → 沿白边外沿精修 → 四条直线 → 交点。

角点定义：白边**外沿**两条直线的交点（白边属于台面，ITTF 2.1.5）。

流程
1. 粗定位（coarse_quad）：在缩小图上按色度做 k-means，挑出最像球台的那一簇
   （面积合适、实心、接近四边形），取凸包的四个角。
   也可以由调用方直接给一个粗四边形（半自动兜底：人粗点四下）。
2. 精修（refine_side）：在原分辨率下，沿每条粗边的外法线扫亮度剖面，
   找"台面 → 白边"（内沿，亮度上跳）和"白边 → 外面"（外沿，亮度下跳）这一对边缘，
   外沿位置取亚像素。
3. 拟合（fit_line）：RANSAC 剔除遮挡、球网、球员等造成的离群点，再对内点做最小二乘。
   内点比例就是这条边的置信度。
4. 四条边两两求交得四角；以新四边形为初值再精修一轮（窄带）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .table import EDGE_LINE_CM, SIDES, TABLE_LENGTH_CM, TABLE_WIDTH_CM

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------


@dataclass
class DetectParams:
    coarse_width: int = 640  # 粗定位时缩到的宽度
    kmeans_k: int = 6
    min_area_frac: float = 0.01  # 球台面积占画面比例的下限
    max_area_frac: float = 0.4
    min_chroma: float = 20.0
    min_solidity: float = 0.85  # 色块面积 / 凸包面积 的下限
    max_pieces: int = 8  # 每个颜色簇最多考虑的连通块数
    max_seeds: int = 10  # 最多以几块（按面积从大到小）为起点去合并
    merge_ab_dist: float = 30.0  # 两个簇的 Lab 色度 (a,b) 距离不超过此值才允许合并
    min_piece_frac: float = 0.003  # 参与合并的连通块面积下限（占画面比例）
    merge_min_solidity: float = 0.9  # 合并后 solidity 不得低于此值
    merge_min_quad_fit: float = 0.9  # 合并后凸包与其四边形近似的面积比不得低于此值  # Lab 色度低于此值的簇按比例降权
    band_px: float = 28.0  # 第一轮精修：粗边两侧的搜索半宽
    band2_px: float = 8.0  # 第二轮精修
    profile_step: float = 0.5  # 剖面采样步长（像素）
    sample_spacing_px: float = 3.0  # 沿边的采样间距
    end_margin_frac: float = 0.06  # 每条边两端各留出的比例（避开角点附近的混叠）
    min_edge_grad: float = 6.0  # 内外沿最小梯度（灰度/像素）
    min_contrast: float = 18.0  # 白边比两侧亮多少（灰度）
    ransac_thresh_px: float = 1.5
    ransac_iters: int = 300
    min_inlier_ratio: float = 0.35  # 低于此值，这条边判为不可靠
    pooled_ransac_thresh_px: float = 2.0  # 多帧合并拟合时的内点阈值（容许机位轻微晃动）


@dataclass
class SideFit:
    name: str
    point: np.ndarray  # 直线上一点
    direction: np.ndarray  # 单位方向
    n_samples: int
    n_valid: int  # 剖面上找到白边的采样点数
    n_inliers: int
    rms_px: float
    points: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 2)))
    inlier_mask: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, bool))

    @property
    def inlier_ratio(self) -> float:
        return self.n_inliers / max(self.n_samples, 1)

    def to_dict(self) -> Dict:
        return {
            "n_samples": self.n_samples,
            "n_valid": self.n_valid,
            "n_inliers": self.n_inliers,
            "inlier_ratio": round(self.inlier_ratio, 3),
            "rms_px": round(self.rms_px, 3),
        }


@dataclass
class FrameResult:
    ok: bool
    corners: Optional[np.ndarray]  # 4x2，far_left, far_right, near_right, near_left
    sides: Dict[str, SideFit]
    coarse: Optional[np.ndarray]
    reason: str = ""

    @property
    def min_inlier_ratio(self) -> float:
        if not self.sides:
            return 0.0
        return min(s.inlier_ratio for s in self.sides.values())


# ---------------------------------------------------------------------------
# 粗定位
# ---------------------------------------------------------------------------


def order_corners(pts: np.ndarray) -> np.ndarray:
    """把任意顺序的 4 点排成 far_left, far_right, near_right, near_left。

    远边是 y 较小的两个点（机位从近侧长边一侧拍）。
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    idx = np.argsort(pts[:, 1])
    far, near = pts[idx[:2]], pts[idx[2:]]
    far = far[np.argsort(far[:, 0])]
    near = near[np.argsort(near[:, 0])]
    return np.array([far[0], far[1], near[1], near[0]])


def _quad_from_hull(hull: np.ndarray) -> Optional[np.ndarray]:
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.005, 0.08, 40):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float64)
        if len(approx) < 4:
            break
    # 退化：用对角方向的极值点
    p = hull.reshape(-1, 2).astype(np.float64)
    s, d = p.sum(1), p[:, 0] - p[:, 1]
    return np.array([p[np.argmin(s)], p[np.argmax(d)], p[np.argmax(s)], p[np.argmin(d)]])


def _coarse_lines(contour: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """用轮廓上靠近各边的点重新拟合四条边，再求交。

    角被球员挡住时，近似多边形会把角"切掉"，但两条边各自剩下的部分仍在原直线上，
    外推求交就能把角找回来。
    """
    rng = np.random.default_rng(0)
    size = float(np.linalg.norm(quad[2] - quad[0]))
    tol = max(0.03 * size, 3.0)
    p = DetectParams(ransac_thresh_px=max(1.5, 0.004 * size), ransac_iters=200)
    lines = []
    for _, ia, ib in SIDES:
        a, b = quad[ia], quad[ib]
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-6:
            return quad
        t_hat = d / L
        n_hat = np.array([-t_hat[1], t_hat[0]])
        rel = contour - a
        along = rel @ t_hat
        dist = np.abs(rel @ n_hat)
        sel = contour[(dist < tol) & (along > 0.1 * L) & (along < 0.9 * L)]
        fit = fit_line(sel, p, rng) if len(sel) >= 5 else None
        if fit is None:
            lines.append((a, t_hat))
        else:
            lines.append((fit[0], fit[1]))
    out = []
    for i in range(4):
        x = intersect(lines[(i - 1) % 4][0], lines[(i - 1) % 4][1], lines[i][0], lines[i][1])
        if x is None:
            return quad
        out.append(x)
    out = np.array(out)
    # 外推出来的角跑得离谱时退回原四边形
    if np.max(np.linalg.norm(out - quad, axis=1)) > 0.5 * size:
        return quad
    return out


def _quad_fit(hull: np.ndarray) -> float:
    q = _quad_from_hull(hull)
    qa = abs(cv2.contourArea(q.astype(np.float32)))
    ha = cv2.contourArea(hull)
    return min(qa, ha) / max(qa, ha, 1.0)


def _external_points(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.int32)


def _score_candidate(pts: np.ndarray, area: float, sh: int, sw: int, chroma: float, params: "DetectParams") -> Dict:
    """给一个候选色块（若干连通块合并）打分；不合格的 score=0。"""
    frac = area / (sh * sw)
    out: Dict = {"area_frac": round(frac, 4), "score": 0.0}
    if not (params.min_area_frac <= frac <= params.max_area_frac):
        return out
    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)
    solidity = area / max(hull_area, 1.0)
    out["solidity"] = round(solidity, 3)
    # 固定机位下台面不会顶到画面边上；顶到边的多是墙面、幕布、地面
    bx, by, bw_, bh_ = cv2.boundingRect(pts)
    if bx <= 1 or by <= 1 or bx + bw_ >= sw - 1 or by + bh_ >= sh - 1:
        out["touches_border"] = True
        return out
    # 台面是实心的凸四边形：凹凸不平的大色块（幕布、地面）直接排除，不让面积项把它们抬上来
    if solidity < params.min_solidity:
        return out
    quad = _quad_from_hull(hull)
    quad_area = abs(cv2.contourArea(quad.astype(np.float32)))
    quad_fit = min(quad_area, hull_area) / max(quad_area, hull_area, 1.0)
    # 球台大致在画面水平方向中部
    x0, _, bw, _ = cv2.boundingRect(pts)
    cx = x0 + bw / 2.0
    center_w = 1.0 - min(abs(cx / sw - 0.5) * 2, 1.0) * 0.5
    # 台面是彩色（蓝/绿），地面、墙、白边大多是灰的：色度低的簇降权
    chroma_w = min(chroma / params.min_chroma, 1.0) ** 2
    score = solidity**2 * quad_fit**2 * center_w * chroma_w * np.sqrt(frac)
    out.update(quad_fit=round(quad_fit, 3), chroma=round(chroma, 1), score=round(float(score), 4))
    out["quad"] = _coarse_lines(pts.astype(np.float64), order_corners(quad))
    return out


def coarse_quad(img: np.ndarray, params: DetectParams = DetectParams()) -> Tuple[Optional[np.ndarray], Dict]:
    """返回 (原分辨率下的粗四边形, 调试信息)。找不到返回 (None, info)。"""
    h, w = img.shape[:2]
    scale = params.coarse_width / float(w)
    small = cv2.resize(img, (params.coarse_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    # 主要按色度聚类，亮度权重低：对左右光照不均（test_2）更稳
    feat = np.dstack([lab[..., 0] * 0.25, lab[..., 1], lab[..., 2]]).reshape(-1, 3)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(12345)
    _, labels, centers = cv2.kmeans(feat, params.kmeans_k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(small.shape[:2])

    sh, sw = small.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    best, best_score, info = None, 0.0, {"candidates": []}
    # 候选块：每个颜色簇里较大的几个连通块（灰色簇不要：台面是彩色的）
    ab = centers[:, 1:3]
    chroma = np.hypot(ab[:, 0] - 128.0, ab[:, 1] - 128.0)
    pieces = []  # (簇号, 面积, 外轮廓点)
    for k in range(params.kmeans_k):
        if chroma[k] < 0.5 * params.min_chroma:
            continue
        m = (labels == k).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        n, cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        order = 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA]) if n > 1 else []
        for j in list(order)[: params.max_pieces]:
            if stats[j, cv2.CC_STAT_AREA] >= params.min_piece_frac * sh * sw:
                pieces.append((k, float(stats[j, cv2.CC_STAT_AREA]), _external_points((cc == j).astype(np.uint8))))
    pieces.sort(key=lambda t: -t[1])
    # 台面可能被球网切成两半（test_1），也可能因反光被分进两个色相相近的簇（test_7）：
    # 以较大的块为起点，把色相相近、并进来后仍然实心的块合并，再给合并结果打分。
    groups = []
    for si, (ks, area, pts) in enumerate(pieces[: params.max_seeds]):
        groups.append([si])  # 单块本身也是候选
        members = [si]
        for j, (kj, aj, pj) in enumerate(pieces):
            if j in members or np.linalg.norm(ab[kj] - ab[ks]) > params.merge_ab_dist:
                continue
            p2 = np.vstack([pts, pj])
            s2 = (area + aj) / max(cv2.contourArea(cv2.convexHull(p2)), 1.0)
            if s2 >= params.merge_min_solidity and _quad_fit(cv2.convexHull(p2)) >= params.merge_min_quad_fit:
                members, area, pts = members + [j], area + aj, p2
        if len(members) > 1:
            groups.append(members)
    for members in groups:
        area = sum(pieces[j][1] for j in members)
        pts = np.vstack([pieces[j][2] for j in members])
        ch = sum(chroma[pieces[j][0]] * pieces[j][1] for j in members) / area
        cand = _score_candidate(pts, area, sh, sw, float(ch), params)
        cand["clusters"], cand["pieces"] = sorted({int(pieces[j][0]) for j in members}), len(members)
        info["candidates"].append({kk: vv for kk, vv in cand.items() if kk != "quad"})
        if cand["score"] > best_score:
            best_score = cand["score"]
            best = cand["quad"] / scale
    info["best_score"] = round(float(best_score), 4)
    if best is None:
        return None, info
    return order_corners(best), info


# ---------------------------------------------------------------------------
# 精修
# ---------------------------------------------------------------------------


def _whiteness(img: np.ndarray) -> np.ndarray:
    """白的程度：三个通道的最小值。白边高，蓝/绿台面和多数背景低。"""
    return img.min(axis=2).astype(np.float32)


def _expected_line_px(quad: np.ndarray, side_idx: int) -> float:
    name, a, b = SIDES[side_idx]
    length_px = float(np.linalg.norm(quad[b] - quad[a]))
    real = TABLE_LENGTH_CM if name in ("far", "near") else TABLE_WIDTH_CM
    # 白边沿法线方向的宽度：用近似的像素/厘米比例；短边受透视压缩，取较宽的上限
    return EDGE_LINE_CM * length_px / real


def refine_side(
    white: np.ndarray,
    quad: np.ndarray,
    side_idx: int,
    band: float,
    params: DetectParams,
    line_px_hint: Optional[float] = None,
    return_index: bool = False,
    center_offset=None,
):
    """沿一条边找白边外沿的亚像素点。

    返回 (Nx2 点, 采样数)；return_index=True 时再返回每个点对应的采样序号（用于算覆盖率）。
    center_offset：可选，函数 t∈[0,1] → 像素，把每个采样点的剖面中心沿外法线挪动
    （边在画面里是弯的时，让搜索带跟着弯曲走）。
    """
    _, ia, ib = SIDES[side_idx]
    a, b = quad[ia], quad[ib]
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 10:
        return (np.zeros((0, 2)), 0, np.zeros(0, int)) if return_index else (np.zeros((0, 2)), 0)
    t_hat = d / length
    n_hat = np.array([t_hat[1], -t_hat[0]])
    center = quad.mean(axis=0)
    if np.dot((a + b) / 2 - center, n_hat) < 0:
        n_hat = -n_hat  # 外法线

    m = params.end_margin_frac
    n_s = max(int(length * (1 - 2 * m) / params.sample_spacing_px), 5)
    ts = np.linspace(m, 1 - m, n_s) * length
    ds = np.arange(-band, band + 1e-9, params.profile_step)
    base = a[None, :] + ts[:, None] * t_hat[None, :]
    if center_offset is not None:
        base = base + np.asarray(center_offset(ts / length), dtype=np.float64)[:, None] * n_hat[None, :]
    pts = base[:, None, :] + ds[None, :, None] * n_hat[None, None, :]
    mapx = pts[..., 0].astype(np.float32)
    mapy = pts[..., 1].astype(np.float32)
    prof = cv2.remap(white, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    prof = cv2.GaussianBlur(prof, (0, 1), sigmaX=0.6) if prof.shape[1] > 3 else prof

    step = params.profile_step
    grad = np.diff(prof, axis=1) / step  # 沿外法线的导数
    line_px = line_px_hint if line_px_hint is not None else _expected_line_px(quad, side_idx)
    max_sep = max(4.0, 2.5 * line_px) + 2.0
    min_sep = 1.0
    max_sep_i, min_sep_i = int(round(max_sep / step)), int(round(min_sep / step))
    win = max(int(round(2.0 / step)), 1)

    out, out_idx = [], []
    for r in range(len(ts)):
        g = grad[r]
        # 候选外沿：最负的几个梯度
        cand = np.argsort(g)[:4]
        best, best_score = None, 0.0
        for jo in cand:
            go = -g[jo]
            if go < params.min_edge_grad:
                continue
            lo, hi = max(jo - max_sep_i, 0), jo - min_sep_i
            if hi <= lo:
                continue
            ji = lo + int(np.argmax(g[lo:hi]))
            gi = g[ji]
            if gi < params.min_edge_grad:
                continue
            p = prof[r]
            ridge = p[ji + 1 : jo + 1].mean()
            inside = p[max(ji - win, 0) : ji + 1].mean()
            outside = p[jo + 1 : jo + 1 + win].mean() if jo + 1 < len(p) else p[-1]
            contrast = ridge - max(inside, outside)
            if contrast < params.min_contrast:
                continue
            score = min(go, gi) * contrast
            if score > best_score:
                best_score, best = score, jo
        if best is None:
            continue
        jo = best
        # 亚像素：对 -g 在 jo 附近做抛物线拟合
        off = 0.0
        if 0 < jo < len(g) - 1:
            y0, y1, y2 = -g[jo - 1], -g[jo], -g[jo + 1]
            den = y0 - 2 * y1 + y2
            if abs(den) > 1e-9:
                off = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
        dpos = ds[0] + (jo + 0.5 + off) * step
        out.append(base[r] + dpos * n_hat)
        out_idx.append(r)
    pts_out = np.asarray(out, dtype=np.float64).reshape(-1, 2)
    if return_index:
        return pts_out, len(ts), np.asarray(out_idx, dtype=int)
    return pts_out, len(ts)


def fit_line(points: np.ndarray, params: DetectParams, rng: np.random.Generator):
    """RANSAC + 内点最小二乘。返回 (point, direction, inlier_mask, rms)。"""
    n = len(points)
    if n < 5:
        return None
    best_mask, best_cnt = None, -1
    for _ in range(params.ransac_iters):
        i, j = rng.choice(n, 2, replace=False)
        d = points[j] - points[i]
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        nrm = np.array([-d[1], d[0]]) / L
        dist = np.abs((points - points[i]) @ nrm)
        mask = dist < params.ransac_thresh_px
        c = int(mask.sum())
        if c > best_cnt:
            best_cnt, best_mask = c, mask
    for _ in range(2):  # 内点最小二乘后再重选一次内点
        p = points[best_mask]
        mean = p.mean(axis=0)
        _, _, vt = np.linalg.svd(p - mean)
        dirv = vt[0]
        nrm = np.array([-dirv[1], dirv[0]])
        dist = np.abs((points - mean) @ nrm)
        best_mask = dist < params.ransac_thresh_px
        if best_mask.sum() < 2:
            return None
    rms = float(np.sqrt(np.mean(dist[best_mask] ** 2)))
    return mean, dirv, best_mask, rms


def intersect(p1, d1, p2, d2) -> Optional[np.ndarray]:
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    s = np.linalg.solve(A, p2 - p1)
    return p1 + s[0] * d1


def _is_convex(q: np.ndarray) -> bool:
    signs = []
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        u, v = b - a, c - b
        signs.append(u[0] * v[1] - u[1] * v[0])
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def detect_frame(
    img: np.ndarray,
    init_quad: Optional[np.ndarray] = None,
    params: DetectParams = DetectParams(),
    seed: int = 0,
) -> FrameResult:
    """单帧检测。init_quad 给定时跳过粗定位（半自动兜底）。"""
    rng = np.random.default_rng(seed)
    if init_quad is None:
        coarse, _ = coarse_quad(img, params)
        if coarse is None:
            return FrameResult(False, None, {}, None, "coarse_failed")
    else:
        coarse = order_corners(np.asarray(init_quad, dtype=np.float64))
    white = _whiteness(img)
    h, w = white.shape

    quad = coarse.copy()
    sides: Dict[str, SideFit] = {}
    for band in (params.band_px, params.band2_px):
        sides = {}
        lines = []
        for si, (name, _, _) in enumerate(SIDES):
            pts, n_s = refine_side(white, quad, si, band, params)
            fit = fit_line(pts, params, rng) if len(pts) >= 5 else None
            if fit is None:
                return FrameResult(False, None, sides, coarse, f"side_{name}_no_points")
            p0, dv, mask, rms = fit
            sides[name] = SideFit(name, p0, dv, n_s, len(pts), int(mask.sum()), rms, pts, mask)
            lines.append((p0, dv))
        new = []
        for i in range(4):
            # 角 i 由边 (i-1) 与边 i 相交：far_left = left ∩ far
            pl, dl = lines[(i - 1) % 4]
            pc, dc = lines[i]
            x = intersect(pl, dl, pc, dc)
            if x is None:
                return FrameResult(False, None, sides, coarse, "parallel_sides")
            new.append(x)
        new = np.array(new)
        if not _is_convex(new) or np.any(new < -0.5 * w) or np.any(new > 1.5 * w):
            return FrameResult(False, None, sides, coarse, "bad_quad")
        quad = new
    ok = all(s.inlier_ratio >= params.min_inlier_ratio for s in sides.values())
    return FrameResult(ok, quad, sides, coarse, "" if ok else "low_inlier_ratio")
