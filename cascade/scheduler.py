"""ClipScheduler：把提议合并成待复核片段（clip），等收口且上下文帧到齐后交给复核器。

* 合并：同一时段重叠的提议并成一个 clip，保留全部 proposal_id 与各自锚点；
  一个 clip 允许复核出多个事件，不压成一个（方案 §5-4）。
* 收口：以后的提议起点都不早于 ``min_future_start_t``，当 clip 结束时间早于它，
  就不会再有提议并进来。
* 上下文：TTNet 给区间 [a, b] 内每帧打分，要读帧 [a - pre_ctx, b + post_ctx]。
  帧不足（视频开头/结尾）记 ``missing_context``，不拿复制帧凑。
* 算力：clip 内每帧对应一次窗口推理，``n_windows`` 就是 TTNet 调用量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .frame_index import FrameIndex
from .proposals import Proposal


@dataclass
class Clip:
    clip_id: str
    proposals: List[Proposal]
    start_timestamp_s: float
    end_timestamp_s: float
    start_frame_id: Optional[int] = None
    end_frame_id: Optional[int] = None
    ctx_first_frame_id: Optional[int] = None
    ctx_last_frame_id: Optional[int] = None
    missing_context: bool = False
    n_windows: int = 0
    submitted_frame_id: Optional[int] = None
    submitted_timestamp_s: Optional[float] = None

    @property
    def proposal_ids(self) -> List[str]:
        return [p.proposal_id for p in self.proposals]

    def to_dict(self) -> Dict:
        return {
            "clip_id": self.clip_id,
            "proposal_ids": self.proposal_ids,
            "anchors": [p.anchor_frame_id for p in self.proposals],
            "start_timestamp_s": round(self.start_timestamp_s, 6),
            "end_timestamp_s": round(self.end_timestamp_s, 6),
            "start_frame_id": self.start_frame_id,
            "end_frame_id": self.end_frame_id,
            "ctx_first_frame_id": self.ctx_first_frame_id,
            "ctx_last_frame_id": self.ctx_last_frame_id,
            "missing_context": self.missing_context,
            "n_windows": self.n_windows,
            "submitted_frame_id": self.submitted_frame_id,
        }


class ClipScheduler:
    def __init__(self, frame_index: FrameIndex, pre_ctx_frames: int = 0, post_ctx_frames: int = 0,
                 merge: bool = True, max_clip_s: float = 1.0, max_pending: int = 64,
                 session_id: str = "s01") -> None:
        self.index = frame_index
        self.pre_ctx = int(pre_ctx_frames)
        self.post_ctx = int(post_ctx_frames)
        self.merge = merge
        self.max_clip_s = float(max_clip_s)
        self.max_pending = int(max_pending)
        self.session_id = session_id
        self._open: List[Clip] = []
        self._n = 0
        self.submitted: List[Clip] = []
        self.stats: Dict[str, int] = {"proposals": 0, "clips": 0, "merged_into_existing": 0,
                                      "missing_context": 0, "queue_overflow": 0}

    # ------------------------------------------------------------------
    def add(self, proposals: List[Proposal]) -> None:
        for p in proposals:
            self.stats["proposals"] += 1
            target = None
            if self.merge:
                p_net_only = bool(p.reasons and all(r == "net_anomaly" for r in p.reasons))
                for c in self._open:
                    c_net_only = bool(c.proposals and all(
                        q.reasons and all(r == "net_anomaly" for r in q.reasons)
                        for q in c.proposals))
                    if p_net_only != c_net_only:
                        continue
                    overlap = p.start_timestamp_s <= c.end_timestamp_s and p.end_timestamp_s >= c.start_timestamp_s
                    span = max(c.end_timestamp_s, p.end_timestamp_s) - min(c.start_timestamp_s, p.start_timestamp_s)
                    if overlap and span <= self.max_clip_s + 1e-9:
                        target = c
                        break
            if target is None:
                self._n += 1
                self._open.append(Clip(f"{self.session_id}:c{self._n:04d}", [p],
                                       p.start_timestamp_s, p.end_timestamp_s))
                self.stats["clips"] += 1
            else:
                target.proposals.append(p)
                target.start_timestamp_s = min(target.start_timestamp_s, p.start_timestamp_s)
                target.end_timestamp_s = max(target.end_timestamp_s, p.end_timestamp_s)
                self.stats["merged_into_existing"] += 1
        self._open.sort(key=lambda c: (c.end_timestamp_s, c.start_timestamp_s))

    def earliest_open_start_t(self) -> float:
        return min((c.start_timestamp_s for c in self._open), default=float("inf"))

    def open_clips(self) -> List[Clip]:
        return list(self._open)

    # ------------------------------------------------------------------
    def poll(self, min_future_start_t: float, final: bool = False) -> List[Clip]:
        """返回已收口且帧已到齐的 clip（按结束时间排序）。``final=True`` 表示流已结束。"""
        ready: List[Clip] = []
        still: List[Clip] = []
        last_t = self.index.last_timestamp
        for c in self._open:
            # 不合并时无需等待后续提议；合并时要等到不可能再有提议与之重叠
            closed = final or not self.merge or c.end_timestamp_s < min_future_start_t
            if not closed or not self._resolve(c, final):
                still.append(c)
                continue
            c.submitted_frame_id = self.index.last_frame_id
            c.submitted_timestamp_s = last_t
            ready.append(c)
        self._open = still
        if len(self._open) > self.max_pending:
            self.stats["queue_overflow"] += 1
        for c in ready:
            c.proposals.sort(key=lambda p: (p.anchor_frame_id, p.proposal_id))
            if c.missing_context:
                self.stats["missing_context"] += 1
        self.submitted.extend(ready)
        return ready

    def _resolve(self, c: Clip, final: bool) -> bool:
        """确定 clip 的帧范围与上下文；帧还没到齐返回 False。"""
        idx = self.index
        if not final and not idx.seen_through(c.end_timestamp_s):
            return False
        f0 = idx.first_at_or_after(c.start_timestamp_s)
        f1 = idx.last_at_or_before(c.end_timestamp_s)
        if f0 is None or f1 is None or f1 < f0:
            f0 = f1 = min(p.anchor_frame_id for p in c.proposals)
        last_ctx = idx.shift(f1, self.post_ctx)
        if last_ctx is None and not final:
            return False                      # 后续上下文帧还没到
        first_ctx = idx.shift(f0, -self.pre_ctx)
        c.start_frame_id, c.end_frame_id = f0, f1
        c.ctx_first_frame_id = first_ctx if first_ctx is not None else idx.shift(f0, 0)
        c.ctx_last_frame_id = last_ctx if last_ctx is not None else idx.last_frame_id
        c.missing_context = first_ctx is None or last_ctx is None
        c.n_windows = idx.count_between(f0, f1)
        return True

    # ------------------------------------------------------------------
    def coverage(self, t_begin: float, t_end: float) -> float:
        """已提交 clip 的 [start, end] 并集时长 / 视频时长。"""
        spans = sorted((max(t_begin, c.start_timestamp_s), min(t_end, c.end_timestamp_s))
                       for c in self.submitted)
        total, cur_a, cur_b = 0.0, None, None
        for a, b in spans:
            if b <= a:
                continue
            if cur_b is None or a > cur_b:
                if cur_b is not None:
                    total += cur_b - cur_a
                cur_a, cur_b = a, b
            else:
                cur_b = max(cur_b, b)
        if cur_b is not None:
            total += cur_b - cur_a
        dur = max(t_end - t_begin, 1e-9)
        return total / dur

    def total_windows(self) -> int:
        """各 clip 窗口数之和（clip 之间若有重叠会重复计算，这正是实际调用量）。"""
        return sum(c.n_windows for c in self.submitted)
