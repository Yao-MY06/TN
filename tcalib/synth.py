"""合成测试图：按已知相机渲染一张带白边、球网、遮挡和光照不均的球台。

只用于单元测试和算法自检，真值是精确的（四角 = 白边外沿交点）。
不是为了逼真，而是为了覆盖真实画面里会让算法出错的几类情况。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .table import EDGE_LINE_CM, NET_HEIGHT_CM, NET_POST_OUT_CM, NET_X_CM, TABLE_LENGTH_CM, TABLE_WIDTH_CM


@dataclass
class SynthCamera:
    f: float
    R: np.ndarray
    t: np.ndarray
    size: Tuple[int, int]  # (w, h)

    def project(self, pts_xyh: np.ndarray) -> np.ndarray:
        """台面坐标 (x, y, 高度) → 像素。合成世界里"向上"是 -Z。"""
        p = np.asarray(pts_xyh, dtype=np.float64).reshape(-1, 3).copy()
        p[:, 2] = -p[:, 2]
        pc = p @ self.R.T + self.t
        w, h = self.size
        u = self.f * pc[:, 0] / pc[:, 2] + (w - 1) / 2.0
        v = self.f * pc[:, 1] / pc[:, 2] + (h - 1) / 2.0
        return np.stack([u, v], axis=1)


def make_camera(
    f: float = 1700.0,
    cam_xyh: Tuple[float, float, float] = (137.0, 152.5 + 420.0, 190.0),
    target_xyh: Tuple[float, float, float] = (137.0, 76.0, 0.0),
    roll_deg: float = 0.0,
    size: Tuple[int, int] = (1920, 1080),
) -> SynthCamera:
    C = np.array([cam_xyh[0], cam_xyh[1], -cam_xyh[2]], dtype=np.float64)
    T = np.array([target_xyh[0], target_xyh[1], -target_xyh[2]], dtype=np.float64)
    fwd = T - C
    fwd /= np.linalg.norm(fwd)
    x_axis = np.array([1.0, 0.0, 0.0])
    right = x_axis - (x_axis @ fwd) * fwd
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    if roll_deg:
        a = np.deg2rad(roll_deg)
        right, down = np.cos(a) * right + np.sin(a) * down, -np.sin(a) * right + np.cos(a) * down
    R = np.stack([right, down, fwd])
    return SynthCamera(f=f, R=R, t=-R @ C, size=size)


SS = 4  # 超采样倍数：OpenCV 的抗锯齿填充会把多边形外扩约 0.6 px，超采样后再缩小才能让真值精确


def _hi(pts: np.ndarray) -> np.ndarray:
    """低分辨率像素坐标 → 超采样画布坐标（低分辨率像素 i 的中心对应 SS*i + (SS-1)/2）。"""
    return pts * SS + (SS - 1) / 2.0


def _fill(img: np.ndarray, pts: np.ndarray, color, alpha: float = 1.0) -> None:
    shift = 4
    p = np.round(_hi(pts) * (1 << shift)).astype(np.int32).reshape(-1, 1, 2)
    if alpha >= 1.0:
        cv2.fillPoly(img, [p], color, lineType=cv2.LINE_8, shift=shift)
        return
    layer = img.copy()
    cv2.fillPoly(layer, [p], color, lineType=cv2.LINE_8, shift=shift)
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, dst=img)


def _quad(x0, x1, y0, y1, h=0.0):
    return np.array([[x0, y0, h], [x1, y0, h], [x1, y1, h], [x0, y1, h]], dtype=np.float64)


def render(
    cam: SynthCamera,
    seed: int = 0,
    table_bgr=(150, 90, 30),
    light_gradient: float = 0.55,
    n_occluders: int = 2,
    jpeg_quality: Optional[int] = 95,
    noise_sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (BGR 图像, 真值四角 4x2)。light_gradient 是左端亮度相对右端的比例。"""
    rng = np.random.default_rng(seed)
    w, h = cam.size
    # 背景：带纹理的地面
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = 60 + 40 * (yy / h) + rng.normal(0, 6, (h, w)).astype(np.float32)
    img = np.dstack([bg * 0.9, bg * 0.95, bg]).clip(0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.resize(img, (w * SS, h * SS), interpolation=cv2.INTER_NEAREST)

    L, W, e = TABLE_LENGTH_CM, TABLE_WIDTH_CM, EDGE_LINE_CM
    # 近侧台面侧板（看得到的立面，深色）
    apron = np.array([[0, W, 0], [L, W, 0], [L, W, -3.0], [0, W, -3.0]], dtype=np.float64)
    _fill(img, cam.project(apron), (40, 30, 20))
    # 台面：白边外框 → 内部蓝色 → 中线
    _fill(img, cam.project(_quad(0, L, 0, W)), (235, 235, 235))
    _fill(img, cam.project(_quad(e, L - e, e, W - e)), table_bgr)
    _fill(img, cam.project(_quad(e, L - e, W / 2 - 0.15, W / 2 + 0.15)), (230, 230, 230))

    # 球网：半透明深色网面 + 白色网顶带 + 网柱
    yf, yn = -NET_POST_OUT_CM, W + NET_POST_OUT_CM
    net = np.array([[NET_X_CM, yf, 0], [NET_X_CM, yn, 0], [NET_X_CM, yn, NET_HEIGHT_CM], [NET_X_CM, yf, NET_HEIGHT_CM]])
    _fill(img, cam.project(net), (20, 20, 20), alpha=0.45)
    tape = np.array([[NET_X_CM, yf, NET_HEIGHT_CM - 1.5], [NET_X_CM, yn, NET_HEIGHT_CM - 1.5],
                     [NET_X_CM, yn, NET_HEIGHT_CM], [NET_X_CM, yf, NET_HEIGHT_CM]])
    _fill(img, cam.project(tape), (240, 240, 240))
    for y in (yf, yn):
        post = np.array([[NET_X_CM - 1, y, 0], [NET_X_CM + 1, y, 0], [NET_X_CM + 1, y, NET_HEIGHT_CM + 1],
                         [NET_X_CM - 1, y, NET_HEIGHT_CM + 1]])
        _fill(img, cam.project(post), (30, 30, 30))

    # 遮挡物：模拟球员身体/手臂，压在台面边缘上（不压角点）
    corners = cam.project(np.hstack([np.array([[0, 0], [L, 0], [L, W], [0, W]]), np.zeros((4, 1))]))
    for k in range(n_occluders):
        side = rng.integers(0, 4)
        a, b = corners[side], corners[(side + 1) % 4]
        s = rng.uniform(0.3, 0.7)
        c = a + s * (b - a) + rng.normal(0, 10, 2)
        axes = (int(rng.uniform(15, 40)), int(rng.uniform(40, 110)))
        color = tuple(int(v) for v in rng.uniform(40, 200, 3))
        ch = _hi(c)
        cv2.ellipse(img, (int(ch[0]), int(ch[1])), (axes[0] * SS, axes[1] * SS), float(rng.uniform(0, 180)), 0, 360,
                    color, -1, cv2.LINE_8)

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    # 光照不均：左暗右亮
    gain = (light_gradient + (1 - light_gradient) * (xx / w))[..., None]
    img = (img.astype(np.float32) * gain).clip(0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    if noise_sigma > 0:
        img = (img.astype(np.float32) + rng.normal(0, noise_sigma, img.shape)).clip(0, 255).astype(np.uint8)
    if jpeg_quality is not None:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img, corners
