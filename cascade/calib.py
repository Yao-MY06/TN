"""从 calib/table_quads.json 的四角构造状态机机位参数。"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from ._paths import CALIB_QUADS, PRESET_TO_QUAD

from core.geometry import ROIBox, TableCalibration


def _canonical_key(key: str) -> str:
    return PRESET_TO_QUAD.get(key, key)


def load_quad(key: str) -> List[List[float]]:
    """读取一个机位的四角；键同时接受预设别名（game4/test2）。"""
    key = _canonical_key(key)
    with open(CALIB_QUADS, encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    if key not in data:
        available = sorted(k for k in data if not k.startswith("_"))
        raise KeyError(f"未知机位四角 {key!r}；可用：{available}")
    points = data[key].get("points")
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError(f"{key} 的四角必须是 4 个点，实际为 {points!r}")
    return points


def build_calib_from_quad(key: str) -> TableCalibration:
    """按 run_mixed.game4 的同一取法构造 TableCalibration。

    四角顺序固定为 far_left、far_right、near_right、near_left。ROI 与 net_zone
    沿用 game4 分支的取法，保证 game_4 与 run_mixed.build_calib("game4") 完全一致。
    """
    canonical = _canonical_key(key)
    quad = load_quad(canonical)
    (flx, fly), (frx, _fry), (_nrx, nry), (nlx, nly) = quad
    net_x = int(round(((flx + frx) / 2.0 + (nlx + _nrx) / 2.0) / 2.0))
    name = "game4" if canonical == "game_4" else canonical
    return TableCalibration(
        name=name,
        base_width=1920,
        base_height=1080,
        table_polygon=tuple(tuple(p) for p in quad),
        roi_box=ROIBox(y_min=250, y_max=920, x_min=150, x_max=1750),
        net_x=net_x,
        net_zone=(int(fly) - 70, int(max(nry, nly)) - 12),
    )
