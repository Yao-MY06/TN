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
from dataclasses import dataclass, replace
from typing import Any, Deque, Dict, List, Optional, Tuple

from ._paths import SM_ROOT  # noqa: F401
from .online_out import OnlineOutDetector
from .racket_evidence import RacketEvidence

import run_mixed as rm  # noqa: E402  (tt/run_mixed.py)
from core.schemas import EventType, FrameObservation, RefereeEvent  # noqa: E402
from core.state_machine import MatchRules, TableTennisStateMachine  # noqa: E402
from .state_machines import trajectory_processor_class  # noqa: E402

_EPS = 1e-6
NET_HALF_WIDTH_PX = 30.0
NET_HOLD_AFTER_S = 0.15
NET_RETURN_S = 0.30
NET_ESCAPE_FACTOR = 2.0


def unique_state_machine_event_id(event_id: str, counts: Dict[str, int],
                                  used: set[str]) -> Tuple[str, bool]:
    """返回一次运行内唯一的 event_id；重复项稳定地加 ``~2``、``~3`` 后缀。"""
    n = counts.get(event_id, 0) + 1
    candidate = event_id if n == 1 else f"{event_id}~{n}"
    while candidate in used:
        n += 1
        candidate = f"{event_id}~{n}"
    counts[event_id] = n
    used.add(candidate)
    return candidate, candidate != event_id


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
    # 网附近反向击球：扣住网带内、NET_TOUCH 后 0.15 s 内的 RACKET_HIT，
    # 用之后 0.3 s 的融合轨迹裁决；默认关闭。
    net_hit_hold: bool = False
    # 球拍证据：off=默认不变；shadow=只打标签；drop=撤销 FAR 击球。
    racket_cache: Optional[str] = None
    racket_hit_hold: str = "off"
    racket_static_filter: bool = False
    # 计分状态机：tn（默认，与 run_mixed 一致）/ gd* / ttnet11-*，见 state_machines.py
    state_machine: str = "tn"


class StreamingReferee:
    def __init__(self, calib, cfg: Optional[CommitConfig] = None) -> None:
        self.calib = calib
        self.cfg = cfg or CommitConfig()
        from .state_machines import make_state_machine
        self.sm = make_state_machine(self.cfg.state_machine)
        self.source: Optional[rm.MixedEventSource] = None
        # 状态机只接受一次运行内唯一的 event_id；对外输出仍保留原 ID。
        self._sm_event_id_counts: Dict[str, int] = {}
        self._sm_used_event_ids: set[str] = set()
        self.sm_event_id_renames = 0
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
        self.event_sources: Dict[str, List[str]] = {}

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

        # 网附近反向击球：NET_TOUCH 回调、被扣住的同帧事件序列与裁决记录。
        self.event_callback = None
        self._last_net_touch_t: Optional[float] = None
        self._held_batch: Optional[Dict[str, Any]] = None
        self.net_adjudications: List[Dict[str, Any]] = []

        # 球拍证据：窗口结束前扣住 RACKET_HIT，shadow 只贴标签，drop 撤销 FAR。
        if self.cfg.racket_hit_hold not in ("off", "shadow", "drop"):
            raise ValueError("racket_hit_hold 只能是 off/shadow/drop")
        self.racket_provider = (RacketEvidence.from_cache(
            self.cfg.racket_cache, static_filter=self.cfg.racket_static_filter)
            if self.cfg.racket_hit_hold != "off" else None)
        self._racket_held: Optional[Dict[str, Any]] = None
        self.racket_events: List[Dict[str, Any]] = []
        self.racket_class_counts = {k: 0 for k in ("NEAR", "FAR", "NONE", "OTHER_SIDE")}
        self.racket_held_count = 0
        self.racket_dropped = 0
        self.racket_released = 0
        self._racket_delays: List[float] = []
        self._ball_by_frame: Dict[int, Tuple[float, float]] = {}
        self._ball_history: Deque[Tuple[float, int]] = deque()

    def held_hit_watermark(self) -> float:
        """未裁决的网带/球拍击球必须压住虚拟水位线。"""
        marks = []
        if self._held_batch is not None:
            marks.append(float(self._held_batch["events"][0].timestamp_s))
        if self._racket_held is not None:
            marks.append(float(self._racket_held["events"][0].timestamp_s))
        return min(marks) if marks else float("inf")

    # ------------------------------------------------------------------ 真实侧
    def push_frame(self, raw: FrameObservation) -> None:
        self._now_t = raw.timestamp_s
        masked, n = rm.mask_low_confidence([raw], self.cfg.obs_min_conf)
        self.n_lowconf += n
        tob = masked[0]
        self._vqueue.append((tob, raw.timestamp_s))
        if raw.ball.visible and raw.ball.center_px is not None:
            fid = int(raw.frame_id)
            self._ball_by_frame[fid] = tuple(float(v) for v in raw.ball.center_px[:2])
            self._ball_history.append((float(raw.timestamp_s), fid))
            while self._ball_history and self._ball_history[0][0] < float(raw.timestamp_s) - 1.0:
                _, old = self._ball_history.popleft()
                self._ball_by_frame.pop(old, None)
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
        if self._racket_held is not None:
            if not self._try_resolve_racket_held():
                return
        if self._held_batch is not None:
            if not self._try_resolve_held():
                return
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
            if self._held_batch is not None and not self._try_resolve_held():
                break

    def finish(self) -> None:
        if self._racket_held is not None:
            self._try_resolve_racket_held(force=True)
        if self._held_batch is not None:
            self._try_resolve_held(force=True)
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
            inner = trajectory_processor_class(cfg.state_machine)(self.calib, source=src)
            self.source = rm.MixedEventSource(
                inner, [], drop_inner=not cfg.keep_inner_bounce, delay_s=cfg.bounce_delay, sm=sm,
                dropouts=None, serve_guard_lookback=cfg.serve_guard_lookback,
                serve_guard_max_hole=cfg.serve_guard_max_hole)
            self.source.dropouts = self._dropouts
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
        evs = source.update(obs)
        injected_ids = {id(e) for e in source.injected[n_inj_before:]}
        for ev in evs:
            if ev.type == EventType.NET_TOUCH.value:
                self._last_net_touch_t = float(ev.timestamp_s)
                if self.event_callback is not None:
                    self.event_callback(ev)
        if evs:
            self._process_event_batch(evs, obs, t, injected_ids)

    def _is_hold_candidate(self, ev: RefereeEvent) -> bool:
        if (not self.cfg.net_hit_hold or self.event_callback is None
                or ev.type != EventType.RACKET_HIT.value):
            return False
        if ev.side not in ("A", "B") or ev.position_px is None:
            return False
        if self._last_net_touch_t is None:
            return False
        dt = float(ev.timestamp_s) - self._last_net_touch_t
        if dt < -_EPS or dt > NET_HOLD_AFTER_S + _EPS:
            return False
        return abs(float(ev.position_px[0]) - float(self.calib.net_x)) <= NET_HALF_WIDTH_PX + _EPS

    def _process_event_batch(self, events: List[RefereeEvent], obs: FrameObservation,
                             t: float, injected_ids: set[int]) -> None:
        for i, ev in enumerate(events):
            evidence = None
            if self.racket_provider is not None and ev.type == EventType.RACKET_HIT.value:
                same_side = self.racket_provider.has_same_side_candidates(ev.side, ev.timestamp_s)
                decide_at = self.racket_provider.decision_time(ev.timestamp_s)
                if same_side and self._now_t is not None and self._now_t + _EPS < decide_at:
                    self._racket_held = {
                        "events": list(events[i:]), "obs": obs, "t": t,
                        "injected_ids": set(injected_ids),
                    }
                    self.racket_held_count += 1
                    self._try_resolve_racket_held()
                    return
                evidence = self._classify_racket(ev)
                if self.cfg.racket_hit_hold == "drop" and evidence["label"] == "FAR":
                    self._record_racket(ev, evidence, "dropped")
                    self.racket_dropped += 1
                    continue
                self._record_racket(ev, evidence, "emitted")
            if self._is_hold_candidate(ev):
                self._held_batch = {
                    "events": list(events[i:]), "obs": obs, "t": t,
                    "injected_ids": set(injected_ids),
                }
                self._try_resolve_held()
                return
            self._emit_event(ev, obs, t, injected_ids, racket_evidence=evidence)

    def _emit_event(self, ev: RefereeEvent, obs: FrameObservation, t: float,
                    injected_ids: set[int], racket_evidence: Optional[Dict[str, Any]] = None) -> None:
        cfg, sm = self.cfg, self.sm
        if id(ev) not in injected_ids:
            ev.sources = [obs.source]
        self.counts[ev.type] += 1
        pos = None
        if ev.position_px is not None:
            pos = (round(float(ev.position_px[0]), 1), round(float(ev.position_px[1]), 1))
        dumped = {
            "event_id": ev.event_id,
            "timestamp_s": round(float(ev.timestamp_s), 6),
            "type": ev.type,
            "side": ev.side,
            "position_px": list(pos) if pos else None,
            "confidence": ev.confidence,
        }
        if racket_evidence is not None:
            dumped["racket_evidence"] = dict(racket_evidence)
        self.dumped_events.append(dumped)
        self.event_sources[ev.event_id] = list(ev.sources)
        self.log_lines.append(
            f"  [帧 {obs.frame_id:04d} | {t:6.3f}s] 事件: {ev.type:<11} 半台: {ev.side:<7} ID: {ev.event_id}")
        sm_ev = self._state_machine_event(ev)
        for d in sm.update(sm_ev):
            self._record_decision(obs, t, d)
            self.source.out_lockout_until = t + cfg.out_after_point_guard
            if cfg.post_point_lockout > 0:
                self.source.lockout_until = t + cfg.post_point_lockout
        if id(ev) in injected_ids:
            self.release_lag.append((ev.event_id, ev.timestamp_s, self._now_t))

    def _state_machine_event(self, ev: RefereeEvent) -> RefereeEvent:
        """给状态机一份唯一 event_id 的副本；dumped_events 仍保留原始 ID。"""
        candidate, renamed = unique_state_machine_event_id(
            ev.event_id, self._sm_event_id_counts, self._sm_used_event_ids)
        if not renamed:
            return ev
        self.sm_event_id_renames += 1
        return replace(ev, event_id=candidate)


    def _ball_at_frame(self, frame_id: int) -> Optional[Tuple[float, float]]:
        for delta in range(0, 4):
            for fid in ((frame_id,) if delta == 0 else (frame_id - delta, frame_id + delta)):
                if fid in self._ball_by_frame:
                    return self._ball_by_frame[fid]
        return None

    def _classify_racket(self, ev: RefereeEvent) -> Dict[str, Any]:
        return self.racket_provider.classify(
            ev.side, ev.timestamp_s, ev.position_px, self._ball_at_frame)

    def _record_racket(self, ev: RefereeEvent, evidence: Dict[str, Any], action: str,
                       release_delay_s: float = 0.0) -> None:
        self.racket_class_counts[evidence["label"]] += 1
        self.racket_events.append({
            "event_id": ev.event_id, "timestamp_s": round(float(ev.timestamp_s), 6),
            "side": ev.side, "action": action, "label": evidence["label"],
            "distance_ratio": evidence.get("distance_ratio"),
            "racket_frame_id": evidence.get("racket_frame_id"),
            "racket_confidence": evidence.get("racket_confidence"),
            "window_s": evidence.get("window_s"),
            "decision_time_s": evidence.get("decision_time_s"),
            "release_delay_s": round(max(0.0, float(release_delay_s)), 6),
            "missing_cache": bool(evidence.get("missing_cache")),
        })

    def _try_resolve_racket_held(self, force: bool = False) -> bool:
        held = self._racket_held
        if held is None:
            return True
        hit = held["events"][0]
        decide_at = self.racket_provider.decision_time(hit.timestamp_s)
        if not force and (self._now_t is None or self._now_t + _EPS < decide_at):
            return False
        evidence = self._classify_racket(hit)
        delay = max(0.0, float(self._now_t or hit.timestamp_s) - float(hit.timestamp_s))
        self._racket_delays.append(delay)
        self._racket_held = None
        if self.cfg.racket_hit_hold == "drop" and evidence["label"] == "FAR":
            self._record_racket(hit, evidence, "dropped", delay)
            self.racket_dropped += 1
        else:
            self._record_racket(hit, evidence, "emitted", delay)
            self.racket_released += 1
            self._emit_event(hit, held["obs"], held["t"], held["injected_ids"],
                             racket_evidence=evidence)
        rest = list(held["events"][1:])
        if rest:
            self._process_event_batch(rest, held["obs"], held["t"], held["injected_ids"])
        return True

    def racket_summary(self) -> Optional[Dict[str, Any]]:
        if self.cfg.racket_hit_hold == "off":
            return None
        delays = sorted(self._racket_delays)
        return {
            "enabled": True, "mode": self.cfg.racket_hit_hold,
            "cache": self.cfg.racket_cache,
            "provider": self.racket_provider.describe() if self.racket_provider else None,
            "classes": dict(self.racket_class_counts),
            "held": self.racket_held_count, "dropped": self.racket_dropped,
            "released": self.racket_released,
            "extra_delay_s": {
                "p50": delays[len(delays) // 2] if delays else 0.0,
                "max": max(delays) if delays else 0.0,
            },
            "events": list(self.racket_events),
        }

    def racket_review_queue(self) -> List[Dict[str, Any]]:
        if self.cfg.racket_hit_hold == "off":
            return []
        category = "racket_shadow" if self.cfg.racket_hit_hold == "shadow" else "racket_drop"
        out = []
        for item in self.racket_events:
            t = float(item["timestamp_s"])
            out.append({
                "clip_id": f"racket:{item['event_id']}", "category": category,
                "proposal_ids": [], "status": "diagnostic",
                "reason": f"racket_{item['label'].lower()}",
                "reason_codes": [f"racket_{item['label'].lower()}"],
                "start_timestamp_s": round(t - 0.2, 6), "end_timestamp_s": round(t + 0.2, 6),
                "start_frame_id": None, "end_frame_id": None, "missing_context": False,
                "hump_peak": None, "humps": [], "nearby_decisions": [],
                "reviewer": "racket", "racket_evidence": dict(item),
            })
        return out
    def _try_resolve_held(self, force: bool = False) -> bool:
        """返回是否已裁决。帧不够且非收尾时继续扣住。"""
        held = self._held_batch
        if held is None:
            return True
        decision, reason = self._adjudicate_held(held, force=force)
        if decision is None:
            return False
        hit = held["events"][0]
        events = list(held["events"])
        if decision == "reverse":
            events = events[1:]
        sources = list(hit.sources)
        if decision in ("reverse", "uncertain") and "suspect_net" not in sources:
            sources.append("suspect_net")
        self.net_adjudications.append({
            "event_id": hit.event_id,
            "timestamp_s": round(float(hit.timestamp_s), 6),
            "side": hit.side,
            "position_px": list(hit.position_px) if hit.position_px else None,
            "decision": decision,
            "reason": reason,
            "sources": sources,
        })
        for ev in events:
            self._emit_event(ev, held["obs"], held["t"], held["injected_ids"])
        self._held_batch = None
        return True

    def _adjudicate_held(self, held: Dict[str, Any], force: bool = False) -> Tuple[Optional[str], str]:
        hit = held["events"][0]
        hit_t = float(hit.timestamp_s)
        end_t = hit_t + NET_RETURN_S
        if not force and (self._now_t is None or self._now_t + _EPS < end_t):
            return None, "window_not_arrived"
        # 用尚未交给虚拟时钟的融合观测做裁决；扣住期间它们不会被消费。
        interval = [(o, t) for o, t in self._vqueue
                    if hit_t - _EPS <= float(o.timestamp_s) <= end_t + _EPS]
        if not force and not any(float(o.timestamp_s) >= end_t - _EPS for o, _ in interval):
            return None, "frame_window_incomplete"
        if len(interval) < 2:
            return "uncertain", "frame_window_incomplete"
        direction = 1.0 if hit.side == "A" else -1.0
        hit_y = float(hit.position_px[1]) if hit.position_px else None
        visible = []
        for o, _ in interval:
            if o.ball.visible and o.ball.center_px is not None:
                visible.append((float(o.ball.center_px[0]), float(o.ball.center_px[1])))
        if not visible:
            return "uncertain", "no_visible_trajectory_after_net_touch"
        escaped = [(x, y) for x, y in visible
                   if direction * (x - float(self.calib.net_x)) > NET_ESCAPE_FACTOR * NET_HALF_WIDTH_PX]
        if hit_y is not None and any(y < hit_y - _EPS for _, y in escaped):
            return "released", "true_hit_moves_away_and_rises"
        return "reverse", "no_escape_and_rise_after_net_touch"

    def _record_decision(self, obs, t: float, d) -> None:
        text = d.describe()
        self.decisions.append(f"[{t:6.3f}s] {text}")
        kind = getattr(d, "kind", None)
        self.decision_records.append({"virtual_t": round(t, 6), "real_t": round(self._now_t, 6),
                                      "frame_id": obs.frame_id,
                                      "kind": kind.value if hasattr(kind, "value") else kind,
                                      "winner": getattr(d, "winner", None),
                                      "rule": getattr(d, "rule", None), "text": text})
        self.log_lines.append(f"  [帧 {obs.frame_id:04d} | {t:6.3f}s] >> 判罚: {text}")

    # ------------------------------------------------------------------ 汇总
    @property
    def score(self) -> Tuple[int, int]:
        return self.sm.points["A"], self.sm.points["B"]

    @property
    def backlog_frames(self) -> int:
        return len(self._vqueue)
