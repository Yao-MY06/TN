"""复核器：对一个 clip 给出 ReviewResult。

所有后端同一个接口，提交层不关心是谁复核的：

* ``RuleReviewer``   —— C3a。用 detect_bounce 的**严格**阈值确认提议里的局部极大点。
                       与离线 ``events_from_observations.py`` 逐条一致（有单测）。
* ``OracleReviewer`` —— 给一份已知事件（标注或离线结果），落在 clip 内的才算确认。
                       用于回答"复核器完美时，提议漏检会损失多少"。
* ``TTNetReviewer``  —— 第 5 步再接（replay / live 两个后端）。

``max_lead_s``：复核出的事件时间最多比 clip 内最早锚点早多少秒。规则复核
就在锚点上，取 0；TTNet 可能在区间内任意位置，取 pre_s。提交层用它算水位线。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ._paths import load_detect_bounce
from .scheduler import Clip

from observation import make_event  # noqa: E402  (adapter/observation.py)

STATUSES = ("confirmed", "rejected", "uncertain", "error", "timeout")


@dataclass
class ReviewResult:
    clip_id: str
    proposal_ids: List[str]
    status: str
    events: List[Dict[str, Any]]                  # §7.4 Event dict（make_event 构造）
    reason: str
    per_proposal: Dict[str, str] = field(default_factory=dict)
    reviewer: str = ""
    submitted_frame_id: Optional[int] = None
    completed_frame_id: Optional[int] = None
    n_windows: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id, "proposal_ids": self.proposal_ids, "status": self.status,
            "events": self.events, "reason": self.reason, "per_proposal": self.per_proposal,
            "reviewer": self.reviewer, "submitted_frame_id": self.submitted_frame_id,
            "completed_frame_id": self.completed_frame_id, "n_windows": self.n_windows,
        }


class RuleReviewer:
    """严格规则：落差 >= 15、台面多边形内（不外扩）、与上一个确认点间隔 >= 15 帧。

    最小间隔是**顺序相关**的（离线按帧序依次比较），所以必须按帧序复核；
    调度器按结束时间交出 clip、clip 内按锚点排序，满足这一点。乱序时直接报错，
    不悄悄给出与离线不同的结果。
    """

    name = "rule"
    max_lead_s = 0.0

    def __init__(self, table_poly, fps: float = 120.0, min_drop: float = 15.0,
                 margin: float = 0.0, min_gap: int = 15, dead: float = 0.05,
                 source: str = "tracknet", source_of=None) -> None:
        db = load_detect_bounce()
        self._db = db
        self.poly = np.asarray(table_poly, dtype=float)
        self.poly_strict = db.shrink_poly(self.poly, margin)
        self.H = db.table_homography(self.poly)
        self.fps = float(fps)
        self.min_drop = float(min_drop)
        self.min_gap = int(min_gap)
        self.dead = float(dead)
        self.source = source
        self.source_of = source_of      # frame_id -> 该帧球位置来自哪个模型（融合后逐帧不同）
        self._last_kept = -10 ** 9
        self._last_seen = -10 ** 9
        self._n_events = 0

    def review(self, clip: Clip) -> ReviewResult:
        events: List[Dict[str, Any]] = []
        per: Dict[str, str] = {}
        n_bounce_candidates = 0
        for p in clip.proposals:
            f = p.features
            if not f.get("local_max"):
                per[p.proposal_id] = "no_rule_candidate"
                continue
            n_bounce_candidates += 1
            fr = p.anchor_frame_id
            if fr < self._last_seen:
                raise RuntimeError(f"RuleReviewer 收到乱序候选：{fr} < {self._last_seen}")
            self._last_seen = fr
            if f["drop_before"] < self.min_drop and f["drop_after"] < self.min_drop:
                per[p.proposal_id] = "rejected:drop"
                continue
            inside = bool(self._db._pip(np.array([[f["x"], f["y"]]]), self.poly_strict)[0])
            if not inside:
                per[p.proposal_id] = "rejected:outside_table"
                continue
            if fr - self._last_kept < self.min_gap:
                per[p.proposal_id] = "rejected:min_gap"
                continue
            self._last_kept = fr
            side, _ = self._db.side_of((f["x"], f["y"]), self.H, self.dead)
            # 与离线 build_events 相同：编号按全部确认点（含 UNKNOWN 侧）递增，时间取 frame / fps
            events.append(make_event(
                event_id=f"bounce_{self._n_events:05d}",
                timestamp_s=fr / self.fps,
                event_type="BOUNCE",
                side=side,
                position_px=[f["x"], f["y"]],
                confidence=None,
                sources=[(self.source_of(fr) if self.source_of else None) or self.source],
            ))
            self._n_events += 1
            per[p.proposal_id] = f"confirmed:{side}"
        if events:
            status, reason = "confirmed", "rule_criteria_met"
        elif n_bounce_candidates:
            status, reason = "rejected", "rule_criteria_not_met"
        else:
            status, reason = "uncertain", "no_reviewable_candidate"
        return ReviewResult(clip.clip_id, clip.proposal_ids, status, events, reason, per,
                            reviewer=self.name, submitted_frame_id=clip.submitted_frame_id,
                            n_windows=clip.n_windows)


class OracleReviewer:
    """已知事件落在 clip 区间内即确认；每条已知事件至多确认一次。"""

    name = "oracle"

    def __init__(self, known_events: List[Dict[str, Any]], max_lead_s: float = 0.2) -> None:
        self.known = sorted(known_events, key=lambda e: e["timestamp_s"])
        self.used = [False] * len(self.known)
        self.max_lead_s = float(max_lead_s)

    def review(self, clip: Clip) -> ReviewResult:
        events = []
        for i, e in enumerate(self.known):
            if self.used[i]:
                continue
            if clip.start_timestamp_s <= e["timestamp_s"] <= clip.end_timestamp_s:
                self.used[i] = True
                events.append(dict(e))
        status = "confirmed" if events else "rejected"
        return ReviewResult(clip.clip_id, clip.proposal_ids, status, events,
                            "known_event_in_clip" if events else "no_known_event_in_clip",
                            reviewer=self.name, submitted_frame_id=clip.submitted_frame_id,
                            n_windows=clip.n_windows)

    def missed(self) -> List[Dict[str, Any]]:
        """没落进任何 clip 的已知事件 = 被提议层漏掉的事件。"""
        return [e for e, u in zip(self.known, self.used) if not u]
