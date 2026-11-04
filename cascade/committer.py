"""EventCommitter：按**事件发生时间**把事件送进状态机。

做法：在一个滞后的"虚拟时钟"上，**原样执行** ``tt/run_mixed.py`` 的逐帧主循环
（``sm.tick`` → ``MixedEventSource.update`` → ``sm.update``），对方状态机和
``MixedEventSource`` 一行不改。

* 真实侧（每来一帧）：收观测、算低置信掩码与盲区、推断出界；提议/复核在外面跑，
  确认的落台通过 ``add_bounces`` 交进来。
* 虚拟侧：设 W = 尚未确定的外部事件（落台/出界）时间戳的下界。外部事件在
  虚拟时刻 ``ts + delay`` 才会被注入，所以虚拟时钟可以安全推进到 ``W + delay`` 之前。
  这样无论复核多慢，状态机看到的 (tick, 事件) 序列都和离线 run_mixed 完全一样——
  延迟只会推迟判罚的**真实输出时刻**，不会改变判罚本身。
"""

from __future__ import annotations

import math
from bisect import insort
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from ._paths import SM_ROOT  # noqa: F401
from .online_out import OnlineOutDetector

import run_mixed as rm  # noqa: E402  (tt/run_mixed.py)
from core.schemas import EventType, FrameObservation, RefereeEvent  # noqa: E402
from core.state_machine import MatchRules, TableTennisStateMachine  # noqa: E402
from core.trajectory_processor import TrajectoryEventProcessor  # noqa: E402

_EPS = 1e-6


@dataclass
class CommitConfig:
    """默认值与 run_mixed 的命令行默认值、以及 README 里 test_2 的跑法（--jitter-merge）一致。"""
    bounce_delay: float = 0.07
    obs_min_conf: float = 0.5
    jitter_merge: bool = True
    jitter_max_dt: float = 0.6
    jitter_max_dist: float = 60.0
    use_out: bool = True
    out_min_px: float = 35.0
    out_gap_s: float = 0.4
    out_need_descending: bool = True
    out_bounce_guard: float = 0.6
    out_cooldown: float = 1.5
    out_after_point_guard: float = 2.0
    post_point_lockout: float = 0.0
    serve_guard_lookback: float = 0.45
    serve_guard_max_hole: float = 0.08
    keep_inner_bounce: bool = False


class StreamingReferee:
    def __init__(self, calib, cfg: Optional[CommitConfig] = None) -> None:
        self.calib = calib
        self.cfg = cfg or CommitConfig()
        self.sm = TableTennisStateMachine(MatchRules())
        self.source: Optional[rm.MixedEventSource] = None
        self.out_det = OnlineOutDetector(
            calib, self.cfg.out_min_px, self.cfg.out_gap_s, self.cfg.out_need_descending,
            self.cfg.out_bounce_guard, self.cfg.out_cooldown) if self.cfg.use_out else None

        # 真实侧状态
        self._vqueue: Deque[Tuple[FrameObservation, float]] = deque()   # (轨迹层观测, 到达时的真实时间)
        self._now_t: Optional[float] = None
        self._dropouts: List[Tuple[float, float]] = []
        self._last_visible_t: Optional[float] = None
        self._gap_open = False
        self.n_lowconf = 0

        # 落台：确认 → 抖动合并定稿 → 注入
        self._confirmed: List[Tuple[float, int, RefereeEvent]] = []   # 按时间排序，等待定稿
        self._seq = 0
        self._jitter_last: Optional[RefereeEvent] = None
        self.jitter_dropped: List[RefereeEvent] = []
        self.bounce_times: List[float] = []          # 定稿后的落台时间（出界守卫用）
        self.kept_bounces: List[RefereeEvent] = []
        self.out_events: List[RefereeEvent] = []
        self._staged: List[Tuple[float, int, int, RefereeEvent]] = []  # source 建立前的暂存

        # 输出（格式同 run_mixed）
        self.counts: Counter = Counter()
        self.decisions: List[str] = []
        self.decision_records: List[Dict[str, Any]] = []
        self.log_lines: List[str] = []
        self.dumped_events: List[Dict[str, Any]] = []
        self.release_lag: List[Tuple[str, float, float]] = []   # (event_id, 事件时间, 真实释放时刻)
        self.n_virtual = 0
        self._last_virtual_t = float("-inf")
        self._finished = False

    # ------------------------------------------------------------------ 真实侧
    def push_frame(self, raw: FrameObservation) -> None:
        self._now_t = raw.timestamp_s
        masked, n = rm.mask_low_confidence([raw], self.cfg.obs_min_conf)
        self.n_lowconf += n
        tob = masked[0]
        self._vqueue.append((tob, raw.timestamp_s))
        self._update_dropouts(tob)
        if self.out_det is not None:
            self.out_det.push(raw)

    def _update_dropouts(self, o: FrameObservation) -> None:
        """与 run_mixed.find_dropouts 等价的增量版；进行中的盲区终点记为当前帧。"""
        t = o.timestamp_s
        if o.ball.visible:
            if self._gap_open:
                self._dropouts[-1] = (self._dropouts[-1][0], t)
                self._gap_open = False
            self._last_visible_t = t
        elif self._last_visible_t is not None:
            if not self._gap_open:
                self._dropouts.append((self._last_visible_t, t))
                self._gap_open = True
            else:
                self._dropouts[-1] = (self._dropouts[-1][0], t)

    def add_bounces(self, events: List[Dict[str, Any]]) -> None:
        """复核确认的落台（§7.4 dict）。与 run_mixed.load_bounces 一样只收 A/B 侧。"""
        for d in events:
            if d.get("type") != EventType.BOUNCE.value or d.get("side") not in ("A", "B"):
                continue
            ev = RefereeEvent.from_dict(d)
            self._seq += 1
            insort(self._confirmed, (ev.timestamp_s, self._seq, ev))

    # ------------------------------------------------------------------ 推进
    def advance(self, bounce_unknown_before_t: float) -> None:
        """``bounce_unknown_before_t``：尚未确认的落台，其时间戳不会早于这个值。"""
        # 1) 落台定稿：早于下界的已全部确认，可以按时间顺序做抖动合并
        while self._confirmed and self._confirmed[0][0] + 1e-9 < bounce_unknown_before_t:
            _, _, ev = self._confirmed.pop(0)
            self._finalize_bounce(ev)
        # 2) 出界定稿：需要它之前的落台全部定稿
        if self.out_det is not None:
            for ev in self.out_det.finalize(bounce_unknown_before_t, self.bounce_times):
                self.out_events.append(ev)
                self._inject(ev, kind=1)
        # 3) 虚拟时钟推进
        w = bounce_unknown_before_t
        if self.out_det is not None:
            w = min(w, self.out_det.min_unknown_t())
        limit = w + self.cfg.bounce_delay - _EPS
        while self._vqueue and self._vqueue[0][0].timestamp_s < limit:
            obs, _ = self._vqueue.popleft()
            self._virtual_frame(obs)

    def finish(self) -> None:
        if self.out_det is not None:
            self.out_det.finish()
        self.advance(float("inf"))
        self._finished = True

    def _finalize_bounce(self, ev: RefereeEvent) -> None:
        if self.cfg.jitter_merge and self._jitter_last is not None:
            prev = self._jitter_last
            dt = ev.timestamp_s - prev.timestamp_s
            near = (ev.position_px is not None and prev.position_px is not None
                    and math.dist(ev.position_px, prev.position_px) < self.cfg.jitter_max_dist)
            if ev.side == prev.side and dt < self.cfg.jitter_max_dt and near:
                self.jitter_dropped.append(ev)
                return
        self._jitter_last = ev
        self.kept_bounces.append(ev)
        self.bounce_times.append(ev.timestamp_s)
        self._inject(ev, kind=0)

    def _inject(self, ev: RefereeEvent, kind: int) -> None:
        """排进 MixedEventSource.pending。键 (时间, 落台在前出界在后) 与 run_mixed 的稳定排序一致。"""
        if ev.timestamp_s + self.cfg.bounce_delay <= self._last_virtual_t + 1e-9:
            # 虚拟时钟已经走过了这条事件该注入的时刻：水位线算错了。宁可报错，不悄悄判错。
            raise RuntimeError(
                f"watermark violated: {ev.event_id} @ {ev.timestamp_s:.6f}s arrived after "
                f"virtual clock reached {self._last_virtual_t:.6f}s")
        self._seq += 1
        item = (ev.timestamp_s, kind, self._seq, ev)
        if self.source is None:
            insort(self._staged, item)
            return
        self._insert_pending(item)

    def _insert_pending(self, item) -> None:
        pend = self.source.pending
        key = (item[0], item[1])
        lo, hi = 0, len(pend)
        while lo < hi:
            mid = (lo + hi) // 2
            m = pend[mid]
            mk = (m.timestamp_s, 0 if m.type == EventType.BOUNCE.value else 1)
            if mk <= key:
                lo = mid + 1
            else:
                hi = mid
        pend.insert(lo, item[3])

    # ------------------------------------------------------------------ 虚拟侧：run_mixed 主循环体
    def _virtual_frame(self, obs: FrameObservation) -> None:
        cfg, sm = self.cfg, self.sm
        if self.source is None:
            src = obs.source.value if hasattr(obs.source, "value") else str(obs.source)
            inner = TrajectoryEventProcessor(self.calib, source=src)
            self.source = rm.MixedEventSource(
                inner, [], drop_inner=not cfg.keep_inner_bounce, delay_s=cfg.bounce_delay, sm=sm,
                dropouts=None, serve_guard_lookback=cfg.serve_guard_lookback,
                serve_guard_max_hole=cfg.serve_guard_max_hole)
            self.source.dropouts = self._dropouts          # 共享同一个列表，增量更新可见
            for item in self._staged:
                self._insert_pending(item)
            self._staged = []
        source = self.source
        self.n_virtual += 1
        t = obs.timestamp_s
        self._last_virtual_t = t

        for d in sm.tick(t):
            self._record_decision(obs, t, d)
            source.out_lockout_until = t + cfg.out_after_point_guard
            if cfg.post_point_lockout > 0:
                source.lockout_until = t + cfg.post_point_lockout

        n_inj_before = len(source.injected)
        for ev in source.update(obs):
            self.counts[ev.type] += 1
            pos = None
            if ev.position_px is not None:
                pos = (round(float(ev.position_px[0]), 1), round(float(ev.position_px[1]), 1))
            self.dumped_events.append({
                "event_id": ev.event_id,
                "timestamp_s": round(float(ev.timestamp_s), 6),
                "type": ev.type,
                "side": ev.side,
                "position_px": list(pos) if pos else None,
                "confidence": ev.confidence,
            })
            self.log_lines.append(
                f"  [帧 {obs.frame_id:04d} | {t:6.3f}s] 事件: {ev.type:<11} 半台: {ev.side:<7} ID: {ev.event_id}")
            for d in sm.update(ev):
                self._record_decision(obs, t, d)
                source.out_lockout_until = t + cfg.out_after_point_guard
                if cfg.post_point_lockout > 0:
                    source.lockout_until = t + cfg.post_point_lockout
        for ev in source.injected[n_inj_before:]:
            self.release_lag.append((ev.event_id, ev.timestamp_s, self._now_t))

    def _record_decision(self, obs, t: float, d) -> None:
        text = d.describe()
        self.decisions.append(f"[{t:6.3f}s] {text}")
        self.decision_records.append({"virtual_t": round(t, 6), "real_t": round(self._now_t, 6),
                                      "frame_id": obs.frame_id, "text": text})
        self.log_lines.append(f"  [帧 {obs.frame_id:04d} | {t:6.3f}s] >> 判罚: {text}")

    # ------------------------------------------------------------------ 汇总
    @property
    def score(self) -> Tuple[int, int]:
        return self.sm.points["A"], self.sm.points["B"]

    @property
    def backlog_frames(self) -> int:
        return len(self._vqueue)
