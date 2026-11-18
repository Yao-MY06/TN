"""ITTF 标准球台与球网的物理尺寸，以及台面坐标系约定。

台面坐标系（单位 cm）：
  * 原点在远左角 far_left；
  * X 轴沿长边，far_left → far_right，0 ~ 274；
  * Y 轴沿短边，远边 → 近边，0 ~ 152.5；
  * 台面在 Z = 0 平面上。"向上"是 +Z 还是 -Z 由相机所在一侧决定，
    见 camera.CameraModel.up_sign，调用方不必关心。

四角顺序与 TN/calib/table_quads.json、pingpong-smart-referee 的 table_polygon 一致：
  far_left, far_right, near_right, near_left（顺时针）。
"""

from __future__ import annotations

import numpy as np

TABLE_LENGTH_CM = 274.0
TABLE_WIDTH_CM = 152.5
NET_HEIGHT_CM = 15.25
# 网柱外侧到台面边线的距离（ITTF 2.2.2：网柱伸出边线 15.25 cm）
NET_POST_OUT_CM = 15.25
# 台面白边宽度（ITTF 2.1.5：边线宽 2 cm）
EDGE_LINE_CM = 2.0

NET_X_CM = TABLE_LENGTH_CM / 2.0

CORNER_NAMES = ("far_left", "far_right", "near_right", "near_left")

# 四角的台面坐标，顺序与 CORNER_NAMES 一致
TABLE_CORNERS_CM = np.array(
    [
        [0.0, 0.0],
        [TABLE_LENGTH_CM, 0.0],
        [TABLE_LENGTH_CM, TABLE_WIDTH_CM],
        [0.0, TABLE_WIDTH_CM],
    ],
    dtype=np.float64,
)

# 四条边：(起点角下标, 终点角下标)，名字与方向
SIDES = (
    ("far", 0, 1),
    ("right", 1, 2),
    ("near", 2, 3),
    ("left", 3, 0),
)
