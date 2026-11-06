"""ProposalGenerator：逐帧读 TrackNet（+ 可选 YOLO）观测，输出待复核区间。

提议只说"这里值得 TTNet 看一眼"，不代表事件已发生（方案 §3）。
触发器宁多勿漏；每个提议都在锚点之后至多 ``win`` 帧内发出，
因此"以后还会出现的提议，其锚点不早于 当前帧 - win + 1"——
提交层靠这个下界算水位线。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .frame_index import FrameIndex
from .online_bounce import LocalMax, StreamingLocalMax


@dataclass
class Proposal:
    proposal_id: str
    session_id: str
    anchor_frame_id: int
    anchor_timestamp_s: float
    start_timestamp_s: float
    end_timestamp_s: float
    candidate_types: List[str]
    reasons: List[str]
    created_frame_id: int            # 真实流上发出提议的那一帧（算延迟用）
    created_timestamp_s: float
    side_hint: str = "UNKNOWN"
    rally_id: Optional[str] = None
    track_id: Optional[str] = None
    status: str = "waiting_frames"
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("anchor_timestamp_s", "start_timestamp_s", "end_timestamp_s", "created_timestamp_s"):
            d[k] = round(d[k], 6)
        return d


@dataclass
class ProposalParams:
    win: int = 12                     # 局部极大判定窗（帧），与 detect_bounce 默认一致
    fps: float = 120.0                # 只用于把帧差换算成 px/s，与 detect_bounce 一致
    # 区间 [t0 - pre, t0 + post]。方案初值 0.2 / 0.3 覆盖了 62% 的视频；在 test_2（推断真值，
    # 28 个落台）上扫描后取 0.1 / 0.1：落台仍全部覆盖且距片段边缘 >= 6 帧，覆盖降到 30%。
    # 只在一段视频上定的，换视频要复核（见 README「第 4 步」）。
    pre_s: float = 0.1
    post_s: float = 0.1
    table_margin_px: float = 30.0     # 台面多边形外扩（宽松）
    bounce_min_drop: float = 8.0      # 落差（宽松；离线严格版是 15）
    # 方向突变：test_2 上没有带来任何落台召回，只增加覆盖，默认关
    hit_trigger: bool = False
    hit_min_vx: float = 300.0         # 水平速度反号且任一侧 |vx| >= 该值（px/s）
    dropout_trigger: bool = True
    # 方案 §4.1：持续条件按时间戳，不按帧数。默认值在 120 fps 下与原来的 6 帧 / 3 帧等价。
    dropout_min_s: float = 0.05       # 台面附近持续丢球多久（秒）
    disagree_trigger: bool = True     # 仅在给了 YOLO 观测时生效
    disagree_px: float = 40.0
    disagree_min_s: float = 0.015     # YOLO 与 TrackNet 持续分歧多久（首末帧时间差，秒）


class ProposalGenerator:
    def __init__(self, table_poly, frame_index: FrameIndex,
                 params: Optional[ProposalParams] = None, session_id: str = "s01") -> None:
        self.p = params or ProposalParams()
        if max(self.p.dropout_min_s, self.p.disagree_min_s) * self.p.fps > self.p.win:
            raise ValueError("dropout/disagree 触发时长不能超过 win 帧对应的时间，否则违反前瞻上界")
        self.poly = np.asarray(table_poly, dtype=np.float32)
        self.index = frame_index
        self.session_id = session_id
        self.localmax = StreamingLocalMax(self.p.win, self.p.fps)
        self._n = 0
        self._done = False
        self.stats: Dict[str, int] = {"local_y_max": 0, "direction_change": 0,
                                      "dropout": 0, "yolo_disagree": 0,
                                      "localmax_seen": 0}
        # 方向突变：最近两个真实检出
        self._prev_det: List[Tuple[int, float, float, float]] = []   # (f, t, x, y)
        # 丢球
        self._last_vis: Optional[Tuple[int, float, float, float]] = None
        self._gap_emitted = False
        # 两路分歧
        self._dis_run: List[Tuple[int, float]] = []
        self._dis_emitted = False
        self._now_frame: Optional[int] = None

    # ------------------------------------------------------------------
    def signed_dist(self, x: float, y: float) -> float:
        """到球台多边形的有符号距离（台内为正，像素）。"""
        return float(cv2.pointPolygonTest(self.poly, (float(x), float(y)), True))

    def _new(self, anchor_f: int, anchor_t: float, types: List[str], reasons: List[str],
             features: Dict[str, Any]) -> Proposal:
        self._n += 1
        now_f = self._now_frame if self._now_frame is not None else anchor_f
        now_t = self.index.time_of(now_f) if self.index.has(now_f) else anchor_t
        return Proposal(
            proposal_id=f"{self.session_id}:p{self._n:04d}",
            session_id=self.session_id,
            anchor_frame_id=anchor_f,
            anchor_timestamp_s=anchor_t,
            start_timestamp_s=anchor_t - self.p.pre_s,
            end_timestamp_s=anchor_t + self.p.post_s,
            candidate_types=types,
            reasons=reasons,
            created_frame_id=now_f,
            created_timestamp_s=now_t,
            features=features,
        )

    # ------------------------------------------------------------------
    def update(self, frame_id: int, timestamp_s: float,
               track_xy: Optional[Tuple[float, float]],
               yolo_xy: Optional[Tuple[float, float]] = None,
               yolo_given: bool = False) -> List[Proposal]:
        """喂一帧。``track_xy`` 只填 TrackNet 的**真实检出**（visible 且非补帧）。"""
        self._now_frame = frame_id
        out: List[Proposal] = []
        for lm in self.localmax.push(frame_id, timestamp_s, track_xy):
            p = self._from_localmax(lm)
            if p is not None:
                out.append(p)
        if self.p.hit_trigger and track_xy is not None:
            p = self._direction_change(frame_id, timestamp_s, track_xy)
            if p is not None:
                out.append(p)
        if self.p.dropout_trigger:
            p = self._dropout(frame_id, timestamp_s, track_xy)
            if p is not None:
                out.append(p)
        if self.p.disagree_trigger and yolo_given:
            p = self._disagree(frame_id, timestamp_s, track_xy, yolo_xy)
            if p is not None:
                out.append(p)
        return out

    def flush(self) -> List[Proposal]:
        out = [p for p in (self._from_localmax(lm) for lm in self.localmax.flush()) if p is not None]
        self._done = True
        return out

    def min_future_anchor_t(self) -> float:
        """以后才会发出的提议，其锚点时间的下界。流结束后为 +inf。"""
        if self._done:
            return float("inf")
        if self._now_frame is None:
            return float("-inf")
        # 本帧（位置 len-1）及以前已处理完。之后在位置 P >= len 发出的提议，
        # 锚点位置 >= P - win >= len - win，即倒数第 win 个位置。
        if self.p.win <= 0:
            return self.index.last_timestamp
        t = self.index.time_at_position_from_end(self.p.win)
        return t if t is not None else float("-inf")

    # ------------------------------------------------------------------
    def _from_localmax(self, lm: LocalMax) -> Optional[Proposal]:
        self.stats["localmax_seen"] += 1
        sd = self.signed_dist(lm.x, lm.y)
        if sd < -self.p.table_margin_px:
            return None
        if lm.drop_before < self.p.bounce_min_drop and lm.drop_after < self.p.bounce_min_drop:
            return None
        types = ["BOUNCE"]
        if lm.vx_in * lm.vx_out < 0 and max(abs(lm.vx_in), abs(lm.vx_out)) >= self.p.hit_min_vx:
            types.append("HIT")
        types.append("UNKNOWN")
        self.stats["local_y_max"] += 1
        return self._new(lm.frame_id, lm.timestamp_s, types, ["local_y_max"], {
            "local_max": True, "x": lm.x, "y": lm.y,
            "drop_before": lm.drop_before, "drop_after": lm.drop_after,
            "vx_in": lm.vx_in, "vx_out": lm.vx_out, "signed_dist_px": sd,
        })

    def _direction_change(self, f: int, t: float, xy) -> Optional[Proposal]:
        det = (f, t, float(xy[0]), float(xy[1]))
        self._prev_det.append(det)
        if len(self._prev_det) > 3:
            self._prev_det.pop(0)
        if len(self._prev_det) < 3:
            return None
        (f0, _t0, x0, _), (f1, t1, x1, y1), (f2, _t2, x2, _) = self._prev_det
        # 下一检出离得太远就不判：否则提议发出时间会超过前瞻上界（交给丢球触发器）
        if f2 - f1 > self.p.win or f1 - f0 > self.p.win:
            return None
        vin = (x1 - x0) / ((f1 - f0) / self.p.fps)
        vout = (x2 - x1) / ((f2 - f1) / self.p.fps)
        if vin * vout >= 0 or max(abs(vin), abs(vout)) < self.p.hit_min_vx:
            return None
        self.stats["direction_change"] += 1
        return self._new(f1, t1, ["HIT", "BOUNCE", "UNKNOWN"], ["direction_change"],
                         {"x": x1, "y": y1, "vx_in": vin, "vx_out": vout})

    def _dropout(self, f: int, t: float, xy) -> Optional[Proposal]:
        if xy is not None:
            self._last_vis = (f, t, float(xy[0]), float(xy[1]))
            self._gap_emitted = False
            return None
        if self._last_vis is None or self._gap_emitted:
            return None
        lf, lt, lx, ly = self._last_vis
        if t - lt < self.p.dropout_min_s - 1e-6:
            return None
        self._gap_emitted = True
        sd = self.signed_dist(lx, ly)
        if sd < -self.p.table_margin_px:
            return None
        self.stats["dropout"] += 1
        return self._new(lf, lt, ["UNKNOWN"], ["dropout"], {"x": lx, "y": ly, "signed_dist_px": sd})

    def _disagree(self, f: int, t: float, txy, yxy) -> Optional[Proposal]:
        far = (txy is not None and yxy is not None
               and float(np.hypot(txy[0] - yxy[0], txy[1] - yxy[1])) > self.p.disagree_px)
        if not far:
            self._dis_run = []
            self._dis_emitted = False
            return None
        self._dis_run.append((f, t))
        if self._dis_emitted or t - self._dis_run[0][1] < self.p.disagree_min_s - 1e-6:
            return None
        self._dis_emitted = True
        f0, t0 = self._dis_run[0]
        self.stats["yolo_disagree"] += 1
        return self._new(f0, t0, ["UNKNOWN"], ["yolo_disagree"],
                         {"tracknet_xy": list(txy), "yolo_xy": list(yxy)})
