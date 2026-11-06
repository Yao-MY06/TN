"""``run_mixed.derive_out_events`` 的流式等价实现。

离线版看"球最后一次可见的位置 + 之后多久没再出现"，是双向依赖；
流式版在丢球时长达到 ``min_gap_s`` 的那一帧就能确定"这段空洞够长"，
再等到该时刻之前的落台全部已知（落台守卫要用），才给出结论。
判据、编号、统计口径与离线逐条相同（有单测）。
"""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ._paths import SM_ROOT  # noqa: F401  确保 core 可导入
from core.schemas import EventType, FrameObservation, RefereeEvent  # noqa: E402


class OnlineOutDetector:
    def __init__(self, calib, min_out_px: float = 35.0, min_gap_s: float = 0.4,
                 need_descending: bool = True, bounce_guard_s: float = 0.6,
                 cooldown_s: float = 1.5, source: str = "tracknet") -> None:
        self.poly = np.array(list(calib.table_polygon), dtype=np.float32)
        self.net_x = calib.net_x
        self.min_out_px = min_out_px
        self.min_gap_s = min_gap_s
        self.need_descending = need_descending
        self.bounce_guard_s = bounce_guard_s
        self.cooldown_s = cooldown_s
        self.source = source
        self.stats: Dict[str, int] = {}
        self._recent: deque = deque(maxlen=5)       # 最近 5 帧原始观测（离线 [idx-4, idx]）
        self._prev: Optional[Tuple[FrameObservation, List[float]]] = None
        self._prev_queued = False
        self._queue: List[Tuple[FrameObservation, List[float]]] = []
        self._last_ts: Optional[float] = None
        self._last_out_t = -1e9
        self._count = 0
        self._done = False

    # ------------------------------------------------------------------
    def push(self, obs: FrameObservation) -> None:
        """喂一帧**原始**观测（不做低置信过滤：出界后的低置信点恰是唯一证据）。"""
        self._recent.append(obs)
        self._last_ts = obs.timestamp_s
        if obs.ball.visible:
            if self._prev is not None and not self._prev_queued \
                    and obs.timestamp_s - self._prev[0].timestamp_s >= self.min_gap_s:
                self._queue.append(self._prev)
            earlier = [p.ball.center_px[1] for p in self._recent
                       if p.ball.visible and p.ball.center_px is not None]
            self._prev = (obs, earlier)
            self._prev_queued = False
        elif self._prev is not None and not self._prev_queued \
                and obs.timestamp_s - self._prev[0].timestamp_s >= self.min_gap_s:
            self._queue.append(self._prev)
            self._prev_queued = True

    def finish(self) -> None:
        """流结束：离线版对末尾空洞按"最后一帧 - 最后可见"判定，此处已在 push 中等价处理。"""
        self._done = True

    def min_unknown_t(self) -> float:
        """尚未确定的 OUT 事件，其时间戳的下界。"""
        if self._queue:
            return self._queue[0][0].timestamp_s
        if self._done:
            return float("inf")
        if self._prev is not None and not self._prev_queued:
            return self._prev[0].timestamp_s
        return self._last_ts if self._last_ts is not None else float("-inf")

    def finalize(self, bounces_known_before_t: float, bounce_times: List[float]) -> List[RefereeEvent]:
        """对落台信息已完整的候选给出结论。``bounce_times`` 为已定稿（抖动合并后）的落台时间，升序。"""
        out: List[RefereeEvent] = []
        while self._queue and self._queue[0][0].timestamp_s + 1e-9 < bounces_known_before_t:
            obs, earlier = self._queue.pop(0)
            ev = self._try(obs, earlier, bounce_times)
            if ev is not None:
                out.append(ev)
        return out

    # ------------------------------------------------------------------
    def _try(self, o: FrameObservation, earlier: List[float], bt: List[float]) -> Optional[RefereeEvent]:
        if o.ball.center_px is None:
            return None
        x, y = o.ball.center_px
        signed = cv2.pointPolygonTest(self.poly, (float(x), float(y)), True)
        if signed > -self.min_out_px:
            return None
        if self.need_descending:
            if len(earlier) < 2 or earlier[-1] <= earlier[0]:
                return None
        t = o.timestamp_s
        i = bisect_left(bt, t - self.bounce_guard_s)
        if i < len(bt) and bt[i] <= t + 1e-9:
            self.stats["out_suppressed_by_bounce"] = self.stats.get("out_suppressed_by_bounce", 0) + 1
            return None
        if t - self._last_out_t < self.cooldown_s:
            self.stats["out_suppressed_by_cooldown"] = self.stats.get("out_suppressed_by_cooldown", 0) + 1
            return None
        self._last_out_t = t
        self._count += 1
        return RefereeEvent(
            event_id=f"out_{self._count:05d}",
            timestamp_s=t,
            type=EventType.OUT.value,
            side="A" if x < self.net_x else "B",
            position_px=(float(x), float(y)),
            confidence=None,
            sources=[o.source or self.source],
        )
