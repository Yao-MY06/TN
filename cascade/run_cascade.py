"""级联回放运行器：观测 JSONL → 提议 → 调度 → 复核 → 按事件时间提交 → 状态机。

逐帧流式处理（因果），"真实时刻"按视频帧时间计；``--review-latency-frames``
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
from dataclasses import asdict
from typing import Any, Dict, List, Optional

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


def run(jsonl: str, preset: str = "test2", quad: Optional[str] = None,
        reviewer: str = "rule", oracle_events: Optional[str] = None,
        yolo_jsonl: Optional[str] = None,
        params: Optional[ProposalParams] = None, commit: Optional[CommitConfig] = None,
        review_latency_frames: int = 0, review_deadline_frames: int = 0,
        merge: bool = True, max_clip_s: float = 1.0,
        pre_ctx: int = 0, post_ctx: int = 0, session_id: str = "s01",
        fps: float = 120.0, ttnet_args: Any = None,
        primary: str = "tracknet", fuse_params: Optional[FuseParams] = None) -> Dict[str, Any]:
    if review_latency_frames < 0 or review_deadline_frames < 0:
        raise ValueError("review latency/deadline frames must be >= 0")
    params = params or ProposalParams()
    commit = commit or CommitConfig()
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
    fuse_stats = None
    if primary == "yolo":
        # YOLO 主跟踪，TrackNet 只在回合中 YOLO 丢球的帧补位（fuse_obs.py）。
        if yolo is None:
            raise SystemExit("--primary yolo 需要 --yolo")
        raw, fuse_stats = fuse(list(yolo.values()), raw, fuse_params)
        yolo = None  # YOLO 已经是主轨迹，“YOLO 与 TrackNet 分歧”触发器不再适用
    elif primary != "tracknet":
        raise SystemExit(f"未知 primary: {primary}")

    index = FrameIndex()
    gen = ProposalGenerator(poly, index, params, session_id)
    sched = ClipScheduler(index, pre_ctx, post_ctx, merge=merge, max_clip_s=max_clip_s,
                          session_id=session_id)
    if reviewer == "rule":
        rev = RuleReviewer(poly, fps=fps)
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
    ref = StreamingReferee(calib, commit)

    proposals, results, late_results = [], [], []
    # 每项：提交位置、模拟完成位置、截止位置、clip 和已经算出的结果。
    # review() 在当前实现中是同步的；due_pos 模拟结果到达提交层的时间。
    inflight: List[Dict[str, Any]] = []

    def earliest_event_t(c) -> float:
        a = min(p.anchor_timestamp_s for p in c.proposals) - rev.max_lead_s
        return max(c.start_timestamp_s, a)

    def deliver(pos: int, force: bool = False) -> None:
        """交付到期结果；超时项合成 timeout，原结果稍后只写 late_results。"""
        still: List[Dict[str, Any]] = []
        for item in inflight:
            c = item["clip"]
            if item["timed_out"]:
                # 迟到的真实结果绝不送状态机，只留作诊断。
                if force or item["due_pos"] <= pos:
                    r = item["result"]
                    r.completed_frame_id = index.last_frame_id
                    late_results.append(r)
                else:
                    still.append(item)
                continue

            deadline_pos = item["deadline_pos"]
            due = item["due_pos"] <= pos
            # 流结束（force）时也按截止时间判：预计完成晚于截止的，照样记 timeout，
            # 不能因为视频到头了就把本该超时的结果当正常结果提交。
            late_at_end = (force and not due and deadline_pos is not None
                           and item["due_pos"] > deadline_pos)
            if (force or due) and not late_at_end:
                r = item["result"]
                r.completed_frame_id = index.last_frame_id
                results.append(r)
                ref.add_bounces(r.events)
                continue

            if deadline_pos is not None and (deadline_pos <= pos or late_at_end):
                timeout = ReviewResult(
                    c.clip_id, c.proposal_ids, "timeout", [], "deadline_exceeded",
                    per_proposal={pid: "timeout:deadline_exceeded" for pid in c.proposal_ids},
                    reviewer=rev.name, submitted_frame_id=c.submitted_frame_id,
                    completed_frame_id=index.last_frame_id, n_windows=c.n_windows)
                results.append(timeout)
                item["timed_out"] = True
                if force:
                    item["result"].completed_frame_id = index.last_frame_id
                    late_results.append(item["result"])
                else:
                    still.append(item)
            else:
                still.append(item)
        inflight[:] = still

    def bounce_unknown_before() -> float:
        w = gen.min_future_anchor_t() - rev.max_lead_s
        for c in sched.open_clips():
            w = min(w, earliest_event_t(c))
        for item in inflight:
            if not item["timed_out"]:
                w = min(w, earliest_event_t(item["clip"]))
        return w

    for pos, d in enumerate(raw):
        fid, ts = int(d["frame_id"]), float(d["timestamp_s"])
        index.add(fid, ts)
        ref.push_frame(FrameObservation.from_dict(d))
        b = d["ball"]
        txy = tuple(b["center_px"]) if b["visible"] and not b.get("predicted", False) else None
        yxy, ygiven = None, False
        if yolo is not None:
            yd = yolo.get(fid)
            ygiven = yd is not None
            if yd is not None and yd["ball"]["visible"]:
                yxy = tuple(yd["ball"]["center_px"])
        new = gen.update(fid, ts, txy, yxy, ygiven)
        proposals.extend(new)
        sched.add(new)
        for c in sched.poll(gen.min_future_anchor_t() - params.pre_s):
            inflight.append({
                "clip": c,
                "result": rev.review(c),
                "due_pos": pos + review_latency_frames,
                "deadline_pos": pos + review_deadline_frames if review_deadline_frames else None,
                "timed_out": False,
            })
        deliver(pos)
        ref.advance(bounce_unknown_before())

    tail = gen.flush()
    proposals.extend(tail)
    sched.add(tail)
    for c in sched.poll(float("inf"), final=True):
        inflight.append({
            "clip": c,
            "result": rev.review(c),
            "due_pos": len(raw) + review_latency_frames,
            "deadline_pos": len(raw) + review_deadline_frames if review_deadline_frames else None,
            "timed_out": False,
        })
    deliver(len(raw), force=True)
    ref.finish()

    # ---------------------------------------------------------------- 汇总
    t0, t1 = index.first_timestamp or 0.0, index.last_timestamp or 0.0
    rel_lag = [real - evt for _, evt, real in ref.release_lag]
    dec_lag = [r["real_t"] - r["virtual_t"] for r in ref.decision_records]
    reasons = Counter(r for p in proposals for r in p.reasons)
    review_queue = build_review_queue({
        "results": results, "clips": sched.submitted, "reviewer": rev, "referee": ref,
    })
    summary = {
        "input": os.path.basename(jsonl), "preset": preset, "reviewer": rev.name,
        "frames": len(raw), "duration_s": round(t1 - t0, 3),
        "score": {"A": ref.score[0], "B": ref.score[1]},
        "decisions": ref.decisions,
        "event_counts": dict(ref.counts),
        "source_stats": dict(ref.source.stats) if ref.source else {},
        "proposals": {"total": len(proposals), "by_reason": dict(reasons),
                      "generator_stats": gen.stats},
        "clips": {"total": len(sched.submitted), **sched.stats,
                  "coverage": round(sched.coverage(t0, t1), 4),
                  "ttnet_windows": sched.total_windows(),
                  "ttnet_window_ratio": round(sched.total_windows() / max(1, len(raw)), 4)},
        "reviews": dict(Counter(r.status for r in results)),
        "timeout_count": sum(r.status == "timeout" for r in results),
        "timeout_ratio": round(sum(r.status == "timeout" for r in results)
                               / max(1, len(sched.submitted)), 4),
        "late_result_count": len(late_results),
        # 规则复核判断不了的片段（no_reviewable_candidate）不算“需要人工确认”，单独计数
        "manual_confirm_count": sum(q["category"] == "manual" for q in review_queue),
        "manual_confirm_ratio": round(sum(q["category"] == "manual" for q in review_queue)
                                      / max(1, len(sched.submitted)), 4),
        "unreviewable_count": sum(q["category"] == "unreviewable" for q in review_queue),
        "missing_context_ratio": round(sched.stats["missing_context"]
                                       / max(1, len(sched.submitted)), 4),
        "bounces": {"confirmed": sum(len(r.events) for r in results),
                    "kept_after_jitter": len(ref.kept_bounces),
                    "jitter_dropped": [round(e.timestamp_s, 3) for e in ref.jitter_dropped]},
        "out_events": [{"t": round(e.timestamp_s, 3), "side": e.side} for e in ref.out_events],
        "latency_s": {
            "event_release_p50": _pct(rel_lag, 0.5), "event_release_p95": _pct(rel_lag, 0.95),
            "event_release_max": max(rel_lag) if rel_lag else None,
            "decision_p50": _pct(dec_lag, 0.5), "decision_p95": _pct(dec_lag, 0.95),
            "review_latency_frames": review_latency_frames,
        },
        "n_lowconf_masked": ref.n_lowconf,
        "serve_guard_suppressed": [round(ev.timestamp_s, 3) for ev, _ in (ref.source.suppressed_serve if ref.source else [])],
        "params": asdict(params), "commit": asdict(commit),
        "merge": merge, "max_clip_s": max_clip_s, "pre_ctx": pre_ctx, "post_ctx": post_ctx,
    }
    summary["primary"] = primary
    if fuse_stats is not None:
        summary["fusion"] = fuse_stats
    if rev.name == "ttnet":
        ms = sorted(rev.scores.infer_ms)
        summary["ttnet"] = {
            "backend": rev.scores.kind, "k": rev.k,
            "windows_dedup": rev.n_windows,
            "windows_dedup_ratio": round(rev.n_windows / max(1, len(raw)), 4),
            "infer_ms_p50": _pct(ms, 0.5), "infer_ms_p95": _pct(ms, 0.95),
        }
    if isinstance(rev, OracleReviewer):
        summary["oracle_missed"] = [round(e["timestamp_s"], 3) for e in rev.missed()]
    return {"summary": summary, "proposals": proposals, "clips": sched.submitted,
            "results": results, "late_results": late_results, "review_queue": review_queue,
            "referee": ref, "reviewer": rev}


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
    dump("events.jsonl", res["referee"].dumped_events)
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
          f"（复核模拟耗时 {lat['review_latency_frames']} 帧）")
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
                    help="主跟踪层：tracknet=原方案；yolo=YOLO 逐帧主跟踪、TrackNet 只补 YOLO 丢球的帧")
    ap.add_argument("--yolo-min-conf", type=float, default=0.5)
    ap.add_argument("--fill-window-frames", type=int, default=60,
                    help="最近多少帧内见过球才调用 TrackNet 补位；-1=YOLO 一丢就调")
    ap.add_argument("--preset", default="test2")
    ap.add_argument("--quad", default=None, help="calib/table_quads.json 的键，默认由 preset 推出")
    ap.add_argument("--reviewer", default="rule", choices=["rule", "oracle", "ttnet"])
    ap.add_argument("--oracle-events", default=None)
    ap.add_argument("--review-latency-frames", type=int, default=0)
    ap.add_argument("--review-deadline-frames", type=int, default=0,
                    help="片段提交后等待复核的帧数；0=不启用（默认）")
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
    ap.add_argument("--dropout-min-frames", type=int, default=6)
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
                            dropout_min_frames=a.dropout_min_frames)
    commit = CommitConfig(jitter_merge=not a.no_jitter_merge, use_out=not a.no_out,
                          state_machine=a.state_machine)
    res = run(a.jsonl, a.preset, a.quad, a.reviewer, a.oracle_events, a.yolo, params, commit,
              a.review_latency_frames, a.review_deadline_frames,
              merge=not a.no_merge, max_clip_s=a.max_clip_s,
              pre_ctx=a.pre_ctx, post_ctx=a.post_ctx,
              ttnet_args=a if a.reviewer == "ttnet" else None,
              primary=a.primary,
              fuse_params=FuseParams(yolo_min_conf=a.yolo_min_conf,
                                     fill_window_frames=None if a.fill_window_frames < 0
                                     else a.fill_window_frames))
    if not a.quiet:
        print_summary(res["summary"])
    if a.out_dir:
        write_outputs(a.out_dir, res)
        print(f"输出已写入: {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
