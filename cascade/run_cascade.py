"""级联回放运行器：观测 JSONL → 提议 → 调度 → 复核 → 按事件时间提交 → 状态机。

逐帧流式处理（因果），"真实时刻"按视频帧时间计；``--review-latency-s``
模拟复核耗时。判罚应与延迟无关（见 committer.py 的说明），延迟只影响输出时刻。

用法（在 TN 仓库根目录）::

    python -m cascade.run_cascade --jsonl adapter/examples/test_2_observations.jsonl \\
        --preset test2 --out-dir outputs/cascade/test_2_rule

    # 已知事件当复核器（上限分析：提议漏检会损失多少）
    python -m cascade.run_cascade --jsonl ... --reviewer oracle \\
        --oracle-events adapter/examples/test_2_events.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, replace
from typing import Any, Dict, List, Optional

_EPS_T = 1e-6   # 时间比较容差（秒）

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "cascade"  # noqa: A001

from cascade._paths import PRESET_TO_QUAD, load_detect_bounce  # noqa: E402
from cascade.calib import build_calib_from_quad  # noqa: E402
from cascade.committer import CommitConfig, StreamingReferee  # noqa: E402
from cascade.frame_index import FrameIndex  # noqa: E402
from cascade.fuse_obs import FuseParams, fuse  # noqa: E402
from cascade.proposals import ProposalGenerator, ProposalParams  # noqa: E402
from cascade.review_queue import build_review_queue  # noqa: E402
from cascade.reviewers import OracleReviewer, ReviewResult, RuleReviewer  # noqa: E402
from cascade.scheduler import ClipScheduler  # noqa: E402
from cascade.ttnet_reviewer import add_ttnet_args, build_ttnet_reviewer  # noqa: E402

import run_mixed as rm  # noqa: E402
from core.geometry import CALIBRATION_PRESETS  # noqa: E402
from core.schemas import FrameObservation  # noqa: E402
from observation import load_observations  # noqa: E402


def _pct(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class CascadeCore:
    """回放与实时共用的因果核心。

    调用方只负责提供已融合的逐帧观测和复核结果；本类负责提议、片段调度、
    水位线、提交层以及统一输出。``on_frame`` 返回的 clip 在结果返回前一直压住
    水位线，直到 ``on_result`` 或 ``on_timeout``。
    """

    def __init__(self, poly, calib, reviewer_name: str, params: ProposalParams,
                 commit: CommitConfig, merge: bool, max_clip_s: float,
                 pre_ctx: int, post_ctx: int, fps: float, session_id: str = "s01",
                 *, review_latency_s: float = 0.0, input_name: str = "",
                 primary: str = "tracknet", fuse_stats: Optional[Dict[str, Any]] = None,
                 reviewer_obj: Any = None) -> None:
        self.poly = poly
        self.calib = calib
        self.reviewer_name = reviewer_name
        self.params = params
        self.commit_cfg = commit
        self.merge = bool(merge)
        self.max_clip_s = float(max_clip_s)
        self.pre_ctx = int(pre_ctx)
        self.post_ctx = int(post_ctx)
        self.fps = float(fps)
        self.session_id = session_id
        self.review_latency_s = float(review_latency_s)
        self.input_name = input_name
        self.primary = primary
        self.fuse_stats = fuse_stats
        self.reviewer = reviewer_obj
        # Oracle / TTNet 的峰值可能出现在锚点之前；规则复核固定在锚点。
        self.max_lead_s = float(params.pre_s) if reviewer_name in ("oracle", "ttnet") else 0.0

        self.index = FrameIndex()
        self.generator = ProposalGenerator(poly, self.index, params, session_id)
        self.scheduler = ClipScheduler(self.index, pre_ctx, post_ctx, merge=merge,
                                       max_clip_s=max_clip_s, session_id=session_id)
        self.referee = StreamingReferee(calib, commit)
        self.referee.event_callback = self._on_referee_event if params.net_trigger else None
        self.proposals: List[Any] = []
        self.results: List[ReviewResult] = []
        self.late_results: List[ReviewResult] = []
        self._awaiting: Dict[str, Any] = {}
        self._timed_out: set[str] = set()
        self._frames = 0
        self._finished = False
        self._last_now_t = 0.0

    def _on_referee_event(self, ev) -> None:
        """核心 NET_TOUCH → net_anomaly 提议（只在开关开启时）。"""
        if not self.params.net_trigger or ev.type != "NET_TOUCH":
            return
        prop = self.generator.add_net_anomaly(float(ev.timestamp_s))
        if prop is not None:
            self.proposals.append(prop)
            self.scheduler.add([prop])

    def _poll_ready(self, final: bool = False) -> List[Any]:
        if final:
            ready = self.scheduler.poll(float("inf"), final=True)
        else:
            ready = self.scheduler.poll(self.generator.min_future_anchor_t() - self.params.pre_s)
        for clip in ready:
            self._awaiting[clip.clip_id] = clip
        return ready

    def earliest_event_t(self, clip) -> float:
        anchor = min(p.anchor_timestamp_s for p in clip.proposals) - self.max_lead_s
        return max(clip.start_timestamp_s, anchor)

    def on_frame(self, obs: Dict[str, Any], yolo_obs: Optional[Dict[str, Any]]) -> List[Any]:
        """喂入已融合的一帧，返回本帧已关闭、需要复核的 clip。"""
        fid, ts = int(obs["frame_id"]), float(obs["timestamp_s"])
        self.index.add(fid, ts)
        self.referee.push_frame(FrameObservation.from_dict(obs))
        b = obs["ball"]
        txy = tuple(b["center_px"]) if b["visible"] and not b.get("predicted", False) else None
        yxy, ygiven = None, False
        if yolo_obs is not None:
            ygiven = True
            yb = yolo_obs["ball"]
            if yb["visible"] and not yb.get("predicted", False):
                yxy = tuple(yb["center_px"])
        new = self.generator.update(fid, ts, txy, yxy, ygiven)
        self.proposals.extend(new)
        self.scheduler.add(new)
        ready = self._poll_ready()
        self._frames += 1
        self._last_now_t = ts
        return ready

    def flush(self) -> List[Any]:
        """流结束后冲刷前瞻窗口，返回最后一批 clip。"""
        tail = self.generator.flush()
        self.proposals.extend(tail)
        self.scheduler.add(tail)
        return self._poll_ready(final=True)

    def on_result(self, clip_id: str, result: ReviewResult, now_t: float) -> None:
        """正常或迟到结果到达。迟到结果不提交，只进诊断输出。"""
        self._last_now_t = float(now_t)
        if clip_id in self._timed_out:
            result.completed_frame_id = self.index.last_frame_id
            self.late_results.append(result)
            return
        if clip_id not in self._awaiting:
            raise KeyError(f"未知或重复的复核结果: {clip_id}")
        self._awaiting.pop(clip_id)
        result.completed_frame_id = self.index.last_frame_id
        self.results.append(result)
        self.referee.add_bounces(result.events)

    def on_timeout(self, clip_id: str, now_t: float, reason: str = "deadline_exceeded") -> None:
        """复核截止/队列满：合成 timeout，并释放该片段对水位线的阻塞。"""
        self._last_now_t = float(now_t)
        if clip_id in self._timed_out:
            return
        clip = self._awaiting.pop(clip_id, None)
        if clip is None:
            raise KeyError(f"未知或已完成的复核片段: {clip_id}")
        self._timed_out.add(clip_id)
        self.results.append(ReviewResult(
            clip.clip_id, clip.proposal_ids, "timeout", [], reason,
            per_proposal={pid: f"timeout:{reason}" for pid in clip.proposal_ids},
            reviewer=self.reviewer_name, submitted_frame_id=clip.submitted_frame_id,
            completed_frame_id=self.index.last_frame_id, n_windows=clip.n_windows))

    def bounce_unknown_before(self) -> float:
        w = self.generator.min_future_anchor_t() - self.max_lead_s
        for clip in self.scheduler.open_clips():
            w = min(w, self.earliest_event_t(clip))
        for clip in self._awaiting.values():
            w = min(w, self.earliest_event_t(clip))
        w = min(w, self.referee.held_hit_watermark())
        return w

    def advance(self, now_t: float) -> List[Any]:
        self._last_now_t = float(now_t)
        self.referee.advance(self.bounce_unknown_before())
        return self._poll_ready()

    def finish(self) -> Dict[str, Any]:
        if self._finished:
            raise RuntimeError("CascadeCore.finish() called twice")
        self.referee.finish()
        self._finished = True
        t0 = self.index.first_timestamp or 0.0
        t1 = self.index.last_timestamp or 0.0
        rel_lag = [real - evt for _, evt, real in self.referee.release_lag]
        dec_lag = [r["real_t"] - r["virtual_t"] for r in self.referee.decision_records]
        reasons = Counter(r for p in self.proposals for r in p.reasons)
        review_queue = build_review_queue({
            "results": self.results, "clips": self.scheduler.submitted,
            "reviewer": self.reviewer, "referee": self.referee,
        })
        for adj in self.referee.net_adjudications:
            if adj["decision"] == "reverse":
                category, reason = "net_reversal", "net_reversal"
            elif adj["decision"] == "uncertain":
                category, reason = "manual", "net_hit_uncertain"
            else:
                continue
            t = float(adj["timestamp_s"])
            near = [dict(d) for d in self.referee.decision_records
                    if t - 1.0 <= float(d.get("virtual_t", 0.0)) <= t + 1.0]
            review_queue.append({
                "clip_id": f"net_hold:{adj['event_id']}", "category": category,
                "proposal_ids": [], "status": "uncertain", "reason": reason,
                "reason_codes": [reason], "start_timestamp_s": round(t - 0.1, 6),
                "end_timestamp_s": round(t + 0.3, 6),
                "start_frame_id": None, "end_frame_id": None, "missing_context": False,
                "hump_peak": None, "humps": [], "nearby_decisions": near,
                "reviewer": "net_hold", "net_decision": adj,
            })
        timeout_count = sum(r.status == "timeout" for r in self.results)
        manual_count = sum(q["category"] in ("manual", "net_reversal") for q in review_queue)
        summary = {
            "input": self.input_name, "preset": getattr(self, "preset", ""),
            "reviewer": self.reviewer_name,
            "frames": self._frames, "duration_s": round(t1 - t0, 3),
            "score": {"A": self.referee.score[0], "B": self.referee.score[1]},
            "decisions": self.referee.decisions,
            "event_counts": dict(self.referee.counts),
            "source_stats": dict(self.referee.source.stats) if self.referee.source else {},
            "proposals": {"total": len(self.proposals), "by_reason": dict(reasons),
                          "generator_stats": self.generator.stats},
            "clips": {"total": len(self.scheduler.submitted), **self.scheduler.stats,
                      "coverage": round(self.scheduler.coverage(t0, t1), 4),
                      "ttnet_windows": self.scheduler.total_windows(),
                      "ttnet_window_ratio": round(self.scheduler.total_windows()
                                                  / max(1, self._frames), 4)},
            "reviews": dict(Counter(r.status for r in self.results)),
            "timeout_count": timeout_count,
            "timeout_ratio": round(timeout_count / max(1, len(self.scheduler.submitted)), 4),
            "late_result_count": len(self.late_results),
            "manual_confirm_count": manual_count,
            "manual_confirm_ratio": round(manual_count / max(1, len(self.scheduler.submitted)), 4),
            "unreviewable_count": sum(q["category"] == "unreviewable" for q in review_queue),
            "missing_context_ratio": round(self.scheduler.stats["missing_context"]
                                           / max(1, len(self.scheduler.submitted)), 4),
            "bounces": {"confirmed": sum(len(r.events) for r in self.results),
                        "kept_after_jitter": len(self.referee.kept_bounces),
                        "jitter_dropped": [round(e.timestamp_s, 3)
                                           for e in self.referee.jitter_dropped]},
            "out_events": [{"t": round(e.timestamp_s, 3), "side": e.side}
                           for e in self.referee.out_events],
            "latency_s": {
                "event_release_p50": _pct(rel_lag, 0.5), "event_release_p95": _pct(rel_lag, 0.95),
                "event_release_max": max(rel_lag) if rel_lag else None,
                "decision_p50": _pct(dec_lag, 0.5), "decision_p95": _pct(dec_lag, 0.95),
                "review_latency_s": self.review_latency_s,
            },
            "n_lowconf_masked": self.referee.n_lowconf,
            "serve_guard_suppressed": [round(ev.timestamp_s, 3)
                                       for ev, _ in (self.referee.source.suppressed_serve
                                                     if self.referee.source else [])],
            "params": asdict(self.params), "commit": asdict(self.commit_cfg),
            "merge": self.merge, "max_clip_s": self.max_clip_s,
            "pre_ctx": self.pre_ctx, "post_ctx": self.post_ctx,
            "primary": self.primary,
            "net_hit_hold": {
                "enabled": bool(self.commit_cfg.net_hit_hold),
                "held": len(self.referee.net_adjudications),
                "released": sum(a["decision"] == "released"
                                for a in self.referee.net_adjudications),
                "reversed": sum(a["decision"] == "reverse"
                                for a in self.referee.net_adjudications),
                "uncertain": sum(a["decision"] == "uncertain"
                                 for a in self.referee.net_adjudications),
                "events": list(self.referee.net_adjudications),
            },
        }
        if self.fuse_stats is not None:
            summary["fusion"] = self.fuse_stats
        if self.reviewer_name == "ttnet" and self.reviewer is not None:
            ms = sorted(self.reviewer.scores.infer_ms)
            summary["ttnet"] = {
                "backend": self.reviewer.scores.kind, "k": self.reviewer.k,
                "windows_dedup": self.reviewer.n_windows,
                "windows_dedup_ratio": round(self.reviewer.n_windows / max(1, self._frames), 4),
                "infer_ms_p50": _pct(ms, 0.5), "infer_ms_p95": _pct(ms, 0.95),
            }
        if self.reviewer_name == "oracle" and self.reviewer is not None:
            summary["oracle_missed"] = [round(e["timestamp_s"], 3)
                                         for e in self.reviewer.missed()]
        return {"summary": summary, "proposals": self.proposals,
                "clips": self.scheduler.submitted, "results": self.results,
                "late_results": self.late_results, "review_queue": review_queue,
                "referee": self.referee, "reviewer": self.reviewer}


def run(jsonl: str, preset: str = "test2", quad: Optional[str] = None,
        reviewer: str = "rule", oracle_events: Optional[str] = None,
        yolo_jsonl: Optional[str] = None,
        params: Optional[ProposalParams] = None, commit: Optional[CommitConfig] = None,
        review_latency_s: float = 0.0, review_deadline_s: float = 0.0,
        merge: bool = True, max_clip_s: float = 1.0,
        pre_ctx: int = 0, post_ctx: int = 0, session_id: str = "s01",
        fps: float = 120.0, ttnet_args: Any = None,
        primary: str = "tracknet", fuse_params: Optional[FuseParams] = None,
        yolo_recapture: bool = True, net_hit_hold: bool = False,
        net_trigger: bool = False) -> Dict[str, Any]:
    # 方案 §4.1：超时一律按时间戳，不按固定帧数（各模型帧率、延迟不同）。
    if review_latency_s < 0 or review_deadline_s < 0:
        raise ValueError("review latency/deadline must be >= 0 s")
    params = params or ProposalParams()
    commit = commit or CommitConfig()
    if net_hit_hold or net_trigger:
        params = replace(params, net_trigger=True)
    if net_hit_hold:
        commit = replace(commit, net_hit_hold=True)
    db = load_detect_bounce()
    quad_key = quad or PRESET_TO_QUAD.get(preset, preset)
    poly = db.load_quad(quad_key)
    if preset in CALIBRATION_PRESETS or preset == "game4":
        calib = rm.build_calib(preset)
    else:
        # test_1、test_3、test_5~7 等新机位没有 core 预设，四角是唯一机位参数来源。
        calib = build_calib_from_quad(quad_key)

    raw = load_observations(jsonl)
    yolo = {d["frame_id"]: d for d in load_observations(yolo_jsonl)} if yolo_jsonl else None
    # 观测融合（fuse_obs.py）。默认按方案：TrackNet 连续轨迹，YOLO 在 TrackNet 丢球的帧
    # 重新捕获；两者分歧仍交给提议层的 yolo_disagree 触发器（质量检查）。
    fuse_stats = None
    if primary not in ("tracknet", "yolo"):
        raise SystemExit(f"未知 primary: {primary}")
    if primary == "yolo" and yolo is None:
        raise SystemExit("--primary yolo 需要 --yolo")
    if yolo is not None and (primary == "yolo" or yolo_recapture):
        fp = replace(fuse_params or FuseParams(), primary=primary)
        raw, fuse_stats = fuse(list(yolo.values()), raw, fp)
        if primary == "yolo":
            yolo = None  # YOLO 已是主轨迹，分歧触发器不再适用
    src_of = {int(d["frame_id"]): d["source"] for d in raw}

    index = FrameIndex()
    gen = ProposalGenerator(poly, index, params, session_id)
    sched = ClipScheduler(index, pre_ctx, post_ctx, merge=merge, max_clip_s=max_clip_s,
                          session_id=session_id)
    if reviewer == "rule":
        rev = RuleReviewer(poly, fps=fps, source_of=src_of.get)
    elif reviewer == "oracle":
        if not oracle_events:
            raise SystemExit("--reviewer oracle 需要 --oracle-events")
        rev = OracleReviewer(_read_jsonl(oracle_events), max_lead_s=params.pre_s)
    elif reviewer == "ttnet":
        if ttnet_args is None:
            raise SystemExit("--reviewer ttnet 需要 TTNet 参数（--ttnet-cache 或 --video + --ttnet-weights）")
        evidence = list(raw) + (list(yolo.values()) if yolo else [])
        rev = build_ttnet_reviewer(ttnet_args, poly, evidence, fps=fps, pre_s=params.pre_s)
    else:
        raise SystemExit(f"未知复核器: {reviewer}（TTNet 复核器在第 5 步接入）")
    core = CascadeCore(
        poly, calib, reviewer, params, commit, merge, max_clip_s, pre_ctx, post_ctx,
        fps, session_id, review_latency_s=review_latency_s,
        input_name=os.path.basename(jsonl), primary=primary,
        fuse_stats=fuse_stats, reviewer_obj=rev)
    core.preset = preset

    # 每项：模拟完成时刻、截止时刻（视频时间）、clip 和已经算出的结果。
    inflight: List[Dict[str, Any]] = []

    def schedule_reviews(ready, now_t: float) -> None:
        for c in ready:
            inflight.append({
                "clip": c,
                "result": rev.review(c),
                "due_t": now_t + review_latency_s,
                "deadline_t": now_t + review_deadline_s if review_deadline_s else None,
                "timed_out": False,
            })

    def deliver(now_t: float, force: bool = False) -> None:
        """交付到期结果；超时项由 core 合成 timeout，原结果稍后记为迟到。"""
        still: List[Dict[str, Any]] = []
        for item in inflight:
            c = item["clip"]
            if item["timed_out"]:
                if force or item["due_t"] <= now_t + _EPS_T:
                    core.on_result(c.clip_id, item["result"], now_t)
                else:
                    still.append(item)
                continue

            deadline_t = item["deadline_t"]
            due = item["due_t"] <= now_t + _EPS_T
            late_at_end = (force and not due and deadline_t is not None
                           and item["due_t"] > deadline_t + _EPS_T)
            if (force or due) and not late_at_end:
                core.on_result(c.clip_id, item["result"], now_t)
                continue

            if deadline_t is not None and (deadline_t <= now_t + _EPS_T or late_at_end):
                core.on_timeout(c.clip_id, now_t)
                item["timed_out"] = True
                if force:
                    core.on_result(c.clip_id, item["result"], now_t)
                else:
                    still.append(item)
            else:
                still.append(item)
        inflight[:] = still

    for d in raw:
        ts = float(d["timestamp_s"])
        fid = int(d["frame_id"])
        ready = core.on_frame(d, yolo.get(fid) if yolo is not None else None)
        schedule_reviews(ready, ts)
        deliver(ts)
        schedule_reviews(core.advance(ts), ts)

    end_t = (core.index.last_timestamp + 1.0 / fps) if raw else 0.0
    schedule_reviews(core.flush(), end_t)
    deliver(end_t, force=True)
    return core.finish()


def write_outputs(out_dir: str, res: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def dump(name, rows):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("proposals.jsonl", [p.to_dict() for p in res["proposals"]])
    dump("clips.jsonl", [c.to_dict() for c in res["clips"]])
    dump("reviews.jsonl", [r.to_dict() for r in res["results"]])
    dump("late_results.jsonl", [r.to_dict() for r in res.get("late_results", [])])
    dump("review_queue.jsonl", res.get("review_queue") or build_review_queue(res))
    ref = res["referee"]
    dump("events.jsonl", [{**e, "sources": ref.event_sources.get(e["event_id"], [])}
                          for e in ref.dumped_events])
    dump("decisions.jsonl", res["referee"].decision_records)
    if getattr(res.get("reviewer"), "name", "") == "ttnet":
        dump("ttnet_reviews.jsonl", res["reviewer"].log)
    with open(os.path.join(out_dir, "log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(res["referee"].log_lines))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res["summary"], f, ensure_ascii=False, indent=2)


def print_summary(s: Dict[str, Any]) -> None:
    line = "=" * 74
    print(line)
    print(f"【级联回放】{s['input']}  预设 {s['preset']}  复核器 {s['reviewer']}")
    print(line)
    p, c = s["proposals"], s["clips"]
    print(f"提议            : {p['total']} 条  {p['by_reason']}")
    print(f"复核片段        : {c['total']} 个  覆盖时长 {c['coverage'] * 100:.1f}%  "
          f"TTNet 窗口 {c['ttnet_windows']} / {s['frames']} 帧 ({c['ttnet_window_ratio'] * 100:.1f}%)")
    print(f"复核结果        : {s['reviews']}")
    b = s["bounces"]
    print(f"落台            : 确认 {b['confirmed']}，抖动合并后 {b['kept_after_jitter']}，"
          f"合并丢弃 {b['jitter_dropped']}")
    if s["out_events"]:
        print(f"出界            : {s['out_events']}")
    if "ttnet" in s:
        t = s["ttnet"]
        print(f"TTNet           : {t['backend']}  k={t['k']}  去重窗口 {t['windows_dedup']} "
              f"({t['windows_dedup_ratio'] * 100:.1f}%)  单窗 P50 {t['infer_ms_p50']} ms / P95 {t['infer_ms_p95']} ms")
    if "oracle_missed" in s:
        print(f"未被提议覆盖    : {len(s['oracle_missed'])} 条 {s['oracle_missed']}")
    lat = s["latency_s"]
    fmt = lambda v: "-" if v is None else f"{v:.3f}s"  # noqa: E731
    print(f"提交延迟        : 事件 P50 {fmt(lat['event_release_p50'])} / P95 {fmt(lat['event_release_p95'])}；"
          f"判罚 P50 {fmt(lat['decision_p50'])} / P95 {fmt(lat['decision_p95'])}"
          f"（复核模拟耗时 {lat['review_latency_s']} s）")
    print(f"事件统计        : {s['event_counts']}")
    print(f"事件源统计      : {s['source_stats']}")
    if s["serve_guard_suppressed"]:
        print(f"发球守卫拦截    : {s['serve_guard_suppressed']}")
    print(f"最终比分        : P1(A) {s['score']['A']} : {s['score']['B']} P2(B)")
    print(f"判罚条数        : {len(s['decisions'])}")
    for d in s["decisions"]:
        print(f"   - {d}")
    print(line)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="YOLO → TrackNet 提议 → 复核 → 状态机 级联回放")
    ap.add_argument("--jsonl", required=True, help="TrackNet 逐帧观测（§7.1）")
    ap.add_argument("--yolo", default=None, help="同一视频的 YOLO 逐帧观测；--primary yolo 时必需")
    ap.add_argument("--primary", default="tracknet", choices=["tracknet", "yolo"],
                    help="主轨迹：tracknet=按方案（TrackNet 连续轨迹，YOLO 辅助）；"
                         "yolo=试验（YOLO 主跟踪，TrackNet 只补 YOLO 丢球的帧）")
    ap.add_argument("--no-yolo-recapture", action="store_true",
                    help="primary=tracknet 时不用 YOLO 补 TrackNet 丢球的帧（只做分歧触发）")
    ap.add_argument("--net-hit-hold", action="store_true",
                    help="用核心 NET_TOUCH 触发网附近提议，并扣住网带内击球做 0.3s 裁决")
    ap.add_argument("--yolo-min-conf", type=float, default=0.5)
    ap.add_argument("--fill-window-s", type=float, default=0.5,
                    help="primary=yolo：最近多久见过球才调用 TrackNet 补位（秒）；-1=YOLO 一丢就调")
    ap.add_argument("--preset", default="test2")
    ap.add_argument("--quad", default=None, help="calib/table_quads.json 的键，默认由 preset 推出")
    ap.add_argument("--reviewer", default="rule", choices=["rule", "oracle", "ttnet"])
    ap.add_argument("--oracle-events", default=None)
    ap.add_argument("--review-latency-s", type=float, default=0.0, help="模拟复核耗时（秒）")
    ap.add_argument("--review-deadline-s", type=float, default=0.0,
                    help="片段提交后最多等待复核多少秒；0=不启用（默认）")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--max-clip-s", type=float, default=1.0)
    ap.add_argument("--pre-ctx", type=int, default=None,
                    help="复核器在区间前需要的上下文帧；默认 TTNet 取 8-k，其他取 0")
    ap.add_argument("--post-ctx", type=int, default=None,
                    help="复核器在区间后需要的上下文帧；默认 TTNet 取 k，其他取 0")
    ap.add_argument("--pre-s", type=float, default=0.1)
    ap.add_argument("--post-s", type=float, default=0.1)
    ap.add_argument("--bounce-min-drop", type=float, default=8.0)
    ap.add_argument("--table-margin-px", type=float, default=30.0)
    ap.add_argument("--hit-trigger", action="store_true", help="打开方向突变触发器（默认关）")
    ap.add_argument("--no-dropout-trigger", action="store_true")
    ap.add_argument("--dropout-min-s", type=float, default=0.05, help="台面附近持续丢球多久才提议（秒）")
    ap.add_argument("--no-jitter-merge", action="store_true")
    ap.add_argument("--no-out", action="store_true")
    ap.add_argument("--state-machine", default="tn", choices=["tn", "ttnet11-yolo", "ttnet11-ttnet"],
                    help="计分状态机：tn=TN 共用版（默认）；ttnet11-*=chagoup/TTNet1.1 的 TTNetStateMachine 两个档位")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    add_ttnet_args(ap)
    a = ap.parse_args(argv)
    if a.pre_ctx is None:
        a.pre_ctx = 8 - a.ttnet_k if a.reviewer == "ttnet" else 0
    if a.post_ctx is None:
        a.post_ctx = a.ttnet_k if a.reviewer == "ttnet" else 0

    params = ProposalParams(pre_s=a.pre_s, post_s=a.post_s, bounce_min_drop=a.bounce_min_drop,
                            table_margin_px=a.table_margin_px, hit_trigger=a.hit_trigger,
                            dropout_trigger=not a.no_dropout_trigger,
                            dropout_min_s=a.dropout_min_s)
    commit = CommitConfig(jitter_merge=not a.no_jitter_merge, use_out=not a.no_out,
                          state_machine=a.state_machine)
    res = run(a.jsonl, a.preset, a.quad, a.reviewer, a.oracle_events, a.yolo, params, commit,
              a.review_latency_s, a.review_deadline_s,
              merge=not a.no_merge, max_clip_s=a.max_clip_s,
              pre_ctx=a.pre_ctx, post_ctx=a.post_ctx,
              ttnet_args=a if a.reviewer == "ttnet" else None,
              primary=a.primary, yolo_recapture=not a.no_yolo_recapture,
              net_hit_hold=a.net_hit_hold,
              fuse_params=FuseParams(yolo_min_conf=a.yolo_min_conf,
                                     fill_window_s=None if a.fill_window_s < 0 else a.fill_window_s))
    if not a.quiet:
        print_summary(res["summary"])
    if a.out_dir:
        write_outputs(a.out_dir, res)
        print(f"输出已写入: {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
