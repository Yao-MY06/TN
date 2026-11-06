"""YOLO 主跟踪 + TrackNet 补洞：融合成一条逐帧观测流。

学长定的顺序是 YOLO（实时）→ TrackNet（实时提议）→ TTNet（区间检测）。这里实现第一段：

  * 每帧先看 YOLO。置信度够、且前后 ``gate_frames`` 帧内有位置连得上的 YOLO 检出，
    就直接用 YOLO 的球位置。
  * YOLO 这一帧没给出可用的球，并且最近 ``fill_window_frames`` 帧内还看到过球
    （说明回合还在进行，只是 YOLO 跟丢了），才"调用" TrackNet 补这一帧。
  * 长时间谁都没看到球 = 不在回合中，TrackNet 不跑，这一帧记为无球。

融合后的观测流交给原来的提议层（局部 y 最大值、丢球触发），后面的区间调度、
TTNet 复核和状态机都不变。

TrackNet 的观测是预先逐帧算好的；这里只读"会被调用的那些帧"，并统计调用比例，
作为实时部署时 TrackNet 的算力占比。连续性门限要看后 ``gate_frames`` 帧，
因此融合流比原始帧晚 ``gate_frames`` 帧（默认 2 帧 = 17 ms）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from cascade import _paths  # noqa: F401


@dataclass
class FuseParams:
    yolo_min_conf: float = 0.5        # 与状态机 obs_min_conf 一致：低于它轨迹层本来就当无检测
    gate_frames: int = 2              # 连续性门限看前后几帧
    gate_px: float = 80.0             # 相邻帧位置跳变超过它就不算连得上
    fill_window_frames: Optional[int] = 60   # 最近多少帧见过球才调 TrackNet；None = YOLO 一丢就调


def _xy(obs: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    if not obs:
        return None
    b = obs["ball"]
    if not b["visible"] or b.get("predicted", False) or b.get("center_px") is None:
        return None
    return float(b["center_px"][0]), float(b["center_px"][1])


def _invisible(frame_id: int, ts: float, image_size, source: str) -> Dict[str, Any]:
    return {"frame_id": frame_id, "timestamp_s": ts, "source": source, "image_size": image_size,
            "ball": {"visible": False, "center_px": None, "bbox_xyxy": None,
                     "confidence": None, "predicted": False},
            "event_scores": {"bounce": None, "net_hit": None}}


def fuse(yolo_obs: List[Dict[str, Any]], tracknet_obs: Optional[List[Dict[str, Any]]],
         params: Optional[FuseParams] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """返回 (融合观测流, 统计)。帧序以 YOLO 观测为准（YOLO 是逐帧实时跑的那一层）。"""
    p = params or FuseParams()
    yolo = sorted(yolo_obs, key=lambda d: d["frame_id"])
    tn = {d["frame_id"]: d for d in (tracknet_obs or [])}

    # 置信度过门限的 YOLO 检出
    ydet: Dict[int, Tuple[float, float]] = {}
    for d in yolo:
        xy = _xy(d)
        c = d["ball"].get("confidence")
        if xy is not None and (c is None or c >= p.yolo_min_conf):
            ydet[d["frame_id"]] = xy

    def yolo_ok(f: int) -> bool:
        xy = ydet.get(f)
        if xy is None:
            return False
        for g in range(f - p.gate_frames, f + p.gate_frames + 1):
            if g == f or g not in ydet:
                continue
            q = ydet[g]
            if math.hypot(xy[0] - q[0], xy[1] - q[1]) <= p.gate_px * abs(g - f):
                return True
        return False

    out: List[Dict[str, Any]] = []
    stats = {"frames": 0, "yolo_used": 0, "yolo_rejected": 0, "tracknet_called": 0,
             "tracknet_used": 0, "no_ball": 0}
    last_seen: Optional[int] = None
    for d in yolo:
        f, ts = int(d["frame_id"]), float(d["timestamp_s"])
        stats["frames"] += 1
        if yolo_ok(f):
            out.append(d)
            stats["yolo_used"] += 1
            last_seen = f
            continue
        if f in ydet:
            stats["yolo_rejected"] += 1
        in_rally = (p.fill_window_frames is None
                    or (last_seen is not None and f - last_seen <= p.fill_window_frames))
        t = tn.get(f) if in_rally else None
        if in_rally and tracknet_obs is not None:
            stats["tracknet_called"] += 1
        if t is not None and _xy(t) is not None:
            out.append(t)
            stats["tracknet_used"] += 1
            last_seen = f
            continue
        out.append(_invisible(f, ts, d.get("image_size"), "yolo"))
        stats["no_ball"] += 1

    n = max(1, stats["frames"])
    stats["tracknet_call_ratio"] = round(stats["tracknet_called"] / n, 4)
    stats["ball_visible_ratio"] = round((stats["yolo_used"] + stats["tracknet_used"]) / n, 4)
    stats["params"] = asdict(p)
    return out, stats
