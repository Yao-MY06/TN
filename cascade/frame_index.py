"""帧号 <-> 被预测帧时间戳的索引。

方案 §3：区间按时间表达，再查帧索引得到帧号；不能用 ``t * fps`` 去算，
变帧率或丢帧时会错位。帧必须按时间顺序到达。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Dict, List, Optional


class FrameIndex:
    def __init__(self) -> None:
        self._fids: List[int] = []
        self._ts: List[float] = []
        self._pos: Dict[int, int] = {}

    # ------------------------------------------------------------------
    def add(self, frame_id: int, timestamp_s: float) -> None:
        if self._fids:
            if frame_id <= self._fids[-1]:
                raise ValueError(f"frame_id must increase: {frame_id} after {self._fids[-1]}")
            if timestamp_s < self._ts[-1]:
                raise ValueError(f"timestamp went backwards: {timestamp_s} < {self._ts[-1]}")
        self._pos[frame_id] = len(self._fids)
        self._fids.append(int(frame_id))
        self._ts.append(float(timestamp_s))

    def __len__(self) -> int:
        return len(self._fids)

    @property
    def last_frame_id(self) -> Optional[int]:
        return self._fids[-1] if self._fids else None

    @property
    def last_timestamp(self) -> Optional[float]:
        return self._ts[-1] if self._ts else None

    @property
    def first_timestamp(self) -> Optional[float]:
        return self._ts[0] if self._ts else None

    # ------------------------------------------------------------------
    def has(self, frame_id: int) -> bool:
        return frame_id in self._pos

    def time_of(self, frame_id: int) -> float:
        return self._ts[self._pos[frame_id]]

    def seen_through(self, t: float) -> bool:
        """是否已经收齐了所有时间戳 <= t 的帧。"""
        return bool(self._ts) and self._ts[-1] >= t

    def first_at_or_after(self, t: float) -> Optional[int]:
        i = bisect_left(self._ts, t)
        return self._fids[i] if i < len(self._fids) else None

    def last_at_or_before(self, t: float) -> Optional[int]:
        i = bisect_right(self._ts, t) - 1
        return self._fids[i] if i >= 0 else None

    def shift(self, frame_id: int, k: int) -> Optional[int]:
        """往后（k>0）或往前（k<0）数 k 个**实际存在**的帧。越界返回 None。"""
        i = self._pos[frame_id] + k
        if 0 <= i < len(self._fids):
            return self._fids[i]
        return None

    def count_between(self, f0: int, f1: int) -> int:
        """[f0, f1] 闭区间内实际存在的帧数。"""
        if f1 < f0:
            return 0
        return bisect_right(self._fids, f1) - bisect_left(self._fids, f0)

    def time_at_position_from_end(self, back: int) -> Optional[float]:
        """倒数第 ``back`` 个位置之后那一帧的时间（back=0 → 下一帧尚未到达，返回 None）。"""
        i = len(self._ts) - back
        if 0 <= i < len(self._ts):
            return self._ts[i]
        return None
