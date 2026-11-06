"""YOLO 与 TrackNet 的逐帧观测融合（适配器之后、事件提取之前）。

方案（级联方案 §1、§2；共用状态机方案 §4.3）：TrackNet 读连续图像给出连续轨迹，
YOLO 实时定位，**在模型外**辅助：候选关联、质量检查、TrackNet 丢球时重新捕获。
YOLO 坐标不替代 TrackNet 的图像输入。融合后仍是统一的逐帧 Observation，
再交给同一个提议层 / 事件层 / 计分状态机。

两种主轨迹：

* ``primary="tracknet"``（默认，按方案）：TrackNet 逐帧跑。某帧 TrackNet 没看到球、
  而 YOLO 在这一帧有可信检出时，用 YOLO 的位置补上（重新捕获）。两者都看到但位置
  分歧的帧仍保留 TrackNet，分歧由提议层的 ``yolo_disagree`` 触发器送去复核（质量检查）。
* ``primary="yolo"``（试验）：YOLO 主跟踪，只在回合进行中 YOLO 丢球时才调用 TrackNet，
  用来估算 TrackNet 的算力能省多少。与方案中 TrackNet 的角色不同，不作默认。

"可信的 YOLO 检出"：置信度 ≥ ``yolo_min_conf``，且前后 ``gate_s`` 秒内有位置连得上
（位移 ≤ ``gate_speed_px_s`` × 时间差）的另一条 YOLO 检出；孤立跳点视为误检。
门限要看之后 ``gate_s`` 秒，融合流因此晚 ``gate_s``（默认 0.017 s ≈ 2 帧 @120 fps）。

所有持续条件按时间戳（共用状态机方案 §4.1），不按帧数。
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from cascade import _paths  # noqa: F401

_EPS = 1e-6


@dataclass
class FuseParams:
    primary: str = "tracknet"            # tracknet（方案）| yolo（试验）
    yolo_min_conf: float = 0.5           # 与状态机 obs_min_conf 一致：低于它轨迹层本来就当无检测
    gate_s: float = 0.017                # 连续性门限的时间窗（秒）
    gate_speed_px_s: float = 9600.0      # 相邻检出允许的最大位移速度（80 px/帧 @120 fps）
    fill_window_s: Optional[float] = 0.5  # 仅 primary=yolo：最近多久见过球才调 TrackNet；None=一丢就调


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


class _YoloGate:
    """可信 YOLO 检出的判定（置信度 + 前后时间窗内的位置连续性）。"""

    def __init__(self, yolo_obs: List[Dict[str, Any]], p: FuseParams) -> None:
        self.p = p
        det = []
        for d in yolo_obs:
            xy = _xy(d)
            c = d["ball"].get("confidence")
            if xy is not None and (c is None or c >= p.yolo_min_conf):
                det.append((float(d["timestamp_s"]), int(d["frame_id"]), xy))
        det.sort()
        self.ts = [x[0] for x in det]
        self.det = det
        self.by_fid = {fid: i for i, (_, fid, _) in enumerate(det)}

    def ok(self, fid: int) -> bool:
        i = self.by_fid.get(fid)
        if i is None:
            return False
        t, _, (x, y) = self.det[i]
        lo = bisect_left(self.ts, t - self.p.gate_s - _EPS)
        for j in range(lo, len(self.det)):
            tj, _, (xj, yj) = self.det[j]
            if tj > t + self.p.gate_s + _EPS:
                break
            if j != i and math.hypot(x - xj, y - yj) <= self.p.gate_speed_px_s * abs(tj - t) + _EPS:
                return True
        return False

    def has(self, fid: int) -> bool:
        return fid in self.by_fid


def fuse(yolo_obs: List[Dict[str, Any]], tracknet_obs: Optional[List[Dict[str, Any]]],
         params: Optional[FuseParams] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """返回 (融合观测流, 统计)。每条观测的 ``source`` 标明该帧球位置来自哪个模型。"""
    p = params or FuseParams()
    if p.primary not in ("tracknet", "yolo"):
        raise ValueError(f"未知 primary: {p.primary}")
    gate = _YoloGate(yolo_obs, p)
    yolo_by = {int(d["frame_id"]): d for d in yolo_obs}
    tn_by = {int(d["frame_id"]): d for d in (tracknet_obs or [])}

    stats = {"primary": p.primary, "frames": 0, "tracknet_used": 0, "yolo_used": 0,
             "yolo_rejected": 0, "tracknet_called": 0, "no_ball": 0}
    out: List[Dict[str, Any]] = []

    if p.primary == "tracknet":
        if tracknet_obs is None:
            raise ValueError("primary=tracknet 需要 TrackNet 观测")
        for d in sorted(tracknet_obs, key=lambda o: o["frame_id"]):
            f = int(d["frame_id"])
            stats["frames"] += 1
            stats["tracknet_called"] += 1
            if _xy(d) is not None:
                out.append(d)
                stats["tracknet_used"] += 1
            elif gate.ok(f):
                out.append(yolo_by[f])                 # 重新捕获：TrackNet 丢球，YOLO 看到
                stats["yolo_used"] += 1
            else:
                if gate.has(f):
                    stats["yolo_rejected"] += 1
                out.append(d)                          # 保留 TrackNet 原样（不可见或推断点）
                stats["no_ball"] += 1
    else:
        last_seen_t: Optional[float] = None
        for d in sorted(yolo_obs, key=lambda o: o["frame_id"]):
            f, ts = int(d["frame_id"]), float(d["timestamp_s"])
            stats["frames"] += 1
            if gate.ok(f):
                out.append(d)
                stats["yolo_used"] += 1
                last_seen_t = ts
                continue
            if gate.has(f):
                stats["yolo_rejected"] += 1
            in_rally = (p.fill_window_s is None
                        or (last_seen_t is not None and ts - last_seen_t <= p.fill_window_s + _EPS))
            t = tn_by.get(f) if in_rally else None
            if in_rally and tracknet_obs is not None:
                stats["tracknet_called"] += 1
            if t is not None and _xy(t) is not None:
                out.append(t)
                stats["tracknet_used"] += 1
                last_seen_t = ts
                continue
            out.append(_invisible(f, ts, d.get("image_size"), "yolo"))
            stats["no_ball"] += 1

    n = max(1, stats["frames"])
    stats["tracknet_call_ratio"] = round(stats["tracknet_called"] / n, 4)
    stats["ball_visible_ratio"] = round((stats["yolo_used"] + stats["tracknet_used"]) / n, 4)
    stats["params"] = asdict(p)
    return out, stats
