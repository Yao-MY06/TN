"""实时层简化版入口：视频文件模拟摄像头，多 worker 并行，TTNet 有界队列。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cascade._paths import PRESET_TO_QUAD, load_detect_bounce
from cascade.calib import build_calib_from_quad
from cascade.committer import CommitConfig
from cascade.fuse_obs import FuseParams, OnlineFusion, _invisible
from cascade.proposals import ProposalParams
from cascade.realtime.frames import FrameRing, NullCamera, VideoFileCamera
from cascade.realtime.workers import (TTNetWorker, TrackNetWorker, YoloWorker,
                                      make_live_ttnet_reviewer, make_replay_ttnet_reviewer,
                                      make_tracknet_batch_predictor, make_yolo_predictor)
from cascade.reviewers import RuleReviewer
from cascade.run_cascade import CascadeCore, _pct, write_outputs
from cascade.ttnet_reviewer import DEFAULT_K, detections_from_observations
from core.geometry import CALIBRATION_PRESETS


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _clip_geometry(preset: str, quad: Optional[str]):
    db = load_detect_bounce()
    quad_key = quad or PRESET_TO_QUAD.get(preset, preset)
    poly = db.load_quad(quad_key)
    if preset in CALIBRATION_PRESETS or preset == "game4":
        import run_mixed as rm
        calib = rm.build_calib(preset)
    else:
        calib = build_calib_from_quad(quad_key)
    return poly, calib


def run_realtime(args) -> Dict[str, Any]:
    started = time.monotonic()
    poly, calib = _clip_geometry(args.preset, args.quad)
    params = ProposalParams(pre_s=args.pre_s, post_s=args.post_s,
                            bounce_min_drop=args.bounce_min_drop,
                            table_margin_px=args.table_margin_px,
                            hit_trigger=args.hit_trigger,
                            dropout_trigger=not args.no_dropout_trigger,
                            dropout_min_s=args.dropout_min_s)
    commit = CommitConfig(jitter_merge=not args.no_jitter_merge, use_out=not args.no_out,
                          state_machine=args.state_machine)
    fp = FuseParams(primary=args.primary, yolo_min_conf=args.yolo_min_conf,
                    gate_s=args.fuse_gate_s,
                    fill_window_s=None if args.fill_window_s < 0 else args.fill_window_s)

    ring = FrameRing(args.ring_s, args.fps)
    if args.video:
        camera = VideoFileCamera(args.video, pace=args.pace, speed=args.speed)
        input_name = os.path.basename(args.video)
    else:
        camera = NullCamera(args.null_camera, args.fps, pace=args.pace, speed=args.speed)
        input_name = "<null-camera>"

    yolo_replay = _read_jsonl(args.yolo_replay) if args.yolo_replay else None
    track_replay = _read_jsonl(args.tracknet_replay) if args.tracknet_replay else None
    if track_replay is None and not (args.video and args.tracknet_weights):
        raise SystemExit("需要 --tracknet-replay，或 --video + --tracknet-weights")

    yolo_pred = None
    if yolo_replay is None and args.video and args.yolo_weights:
        yolo_pred = make_yolo_predictor(args.yolo_weights, calib, args.yolo_imgsz,
                                        args.yolo_conf, args.yolo_device)
    yolo_worker = None
    if yolo_replay is not None or yolo_pred is not None:
        yolo_worker = YoloWorker(replay=yolo_replay, sleep_ms=args.yolo_ms,
                                 pace=args.pace, queue_size=args.yolo_queue,
                                 predictor=yolo_pred)

    tn_pred = None
    if track_replay is None and args.video and args.tracknet_weights:
        tn_pred = make_tracknet_batch_predictor(args.tracknet_weights, args.tracknet_device,
                                                args.tracknet_min_conf)
    track_worker = TrackNetWorker(replay=track_replay, sleep_ms=args.tracknet_ms,
                                  pace=args.pace, queue_size=args.tracknet_queue,
                                  batch_predictor=tn_pred)

    reviewer = None
    ttnet_worker = None
    source_of: Dict[int, str] = {}
    if args.reviewer == "rule":
        reviewer = RuleReviewer(poly, fps=args.fps, source_of=source_of.get)
    else:
        if args.ttnet_cache:
            reviewer = make_replay_ttnet_reviewer(args.ttnet_cache, poly, args.fps,
                                                  args.ttnet_k, args.ttnet_ms_per_window / 1000.0)
        elif args.video and args.ttnet_weights:
            reviewer = make_live_ttnet_reviewer(args.ttnet_weights, ring, poly, args.fps,
                                                args.ttnet_k, args.ttnet_src,
                                                args.ttnet_engine, args.ttnet_device)
        else:
            raise SystemExit("--reviewer ttnet 需要 --ttnet-cache 或 --video + --ttnet-weights")
        ttnet_worker = TTNetWorker(reviewer, args.ttnet_queue)

    core = CascadeCore(poly, calib, args.reviewer, params, commit,
                       merge=not args.no_merge, max_clip_s=args.max_clip_s,
                       pre_ctx=(8 - args.ttnet_k) if args.reviewer == "ttnet" else 0,
                       post_ctx=args.ttnet_k if args.reviewer == "ttnet" else 0,
                       fps=args.fps, session_id="s01", input_name=input_name,
                       primary=args.primary, reviewer_obj=reviewer)
    core.preset = args.preset

    if yolo_worker is not None:
        yolo_worker.start()
    track_worker.start()
    if ttnet_worker is not None:
        ttnet_worker.start()

    online = OnlineFusion(fp)
    join: Dict[int, Dict[str, Any]] = {}
    order_rows: List[Dict[str, Any]] = []
    next_order = 0
    detections: Dict[int, List[Tuple[float, float, str]]] = {}
    yolo_raw_by_fid: Dict[int, Optional[Dict[str, Any]]] = {}
    submitted_tasks: set[str] = set()
    active_deadlines: Dict[str, float] = {}
    queue_full = 0
    join_timeouts = 0
    late_worker_results = 0
    decision_lags: List[float] = []
    seen_decisions = 0
    last_video_t = 0.0
    last_progress = started

    def record_decisions() -> None:
        nonlocal seen_decisions
        recs = core.referee.decision_records
        for row in recs[seen_decisions:]:
            cap = next((r["capture_wall"] for r in order_rows if r["frame_id"] == row["frame_id"]), None)
            if cap is not None:
                decision_lags.append(time.monotonic() - cap)
        seen_decisions = len(recs)

    def drain_workers() -> None:
        nonlocal late_worker_results
        now = time.monotonic()
        if yolo_worker is not None:
            for obs in yolo_worker.drain():
                fid = int(obs["frame_id"])
                if fid not in join:
                    late_worker_results += 1
                    continue
                join[fid]["yolo"] = obs
                join[fid]["yolo_at"] = now
                yolo_raw_by_fid[fid] = obs
        for obs in track_worker.drain():
            fid = int(obs["frame_id"])
            if fid not in join:
                late_worker_results += 1
                continue
            join[fid]["tracknet"] = obs
            join[fid]["tracknet_at"] = now

    def process_fused(fused: Dict[str, Any], yolo_raw) -> None:
        nonlocal last_video_t, queue_full
        fid = int(fused["frame_id"])
        last_video_t = float(fused["timestamp_s"])
        if args.reviewer == "rule":
            source_of[fid] = fused["source"]
        evidence = [fused] + ([yolo_raw] if yolo_raw is not None else [])
        for fid_, dets in detections_from_observations(evidence).items():
            detections.setdefault(fid_, []).extend(dets)
        ready = core.on_frame(fused, yolo_raw)
        now_wall = time.monotonic()
        for clip in ready:
            if args.reviewer == "rule":
                core.on_result(clip.clip_id, reviewer.review(clip), last_video_t)
            else:
                if ttnet_worker.submit(clip, dict(detections)):
                    submitted_tasks.add(clip.clip_id)
                    if args.review_deadline_s:
                        active_deadlines[clip.clip_id] = now_wall + args.review_deadline_s
                else:
                    queue_full += 1
                    core.on_timeout(clip.clip_id, last_video_t, reason="queue_full")
        core.advance(last_video_t)
        record_decisions()

    def emit_join(force: bool = False) -> None:
        nonlocal next_order, join_timeouts
        now = time.monotonic()
        while next_order < len(order_rows):
            row = order_rows[next_order]
            fid = row["frame_id"]
            ent = join[fid]
            timed_out = False
            if not force and yolo_worker is not None and ent["yolo"] is None:
                if now - row["capture_wall"] < args.join_timeout_s:
                    break
                ent["yolo"] = _invisible(fid, row["timestamp_s"], (1920, 1080), "yolo")
                timed_out = True
            if not force and ent["tracknet"] is None:
                if now - row["capture_wall"] < args.join_timeout_s:
                    break
                ent["tracknet"] = _invisible(fid, row["timestamp_s"], (1920, 1080), "tracknet")
                timed_out = True
            if yolo_worker is not None and ent["yolo"] is None:
                break
            if ent["tracknet"] is None:
                break
            if timed_out:
                join_timeouts += 1
            fused = online.push(fid, row["timestamp_s"], ent["yolo"], ent["tracknet"],
                                now_t=row["timestamp_s"])
            for obs in fused:
                process_fused(obs, yolo_raw_by_fid.get(int(obs["frame_id"])))
            join.pop(fid, None)
            next_order += 1

    def drain_ttnet() -> None:
        if ttnet_worker is None:
            return
        for cid, result, wall in ttnet_worker.drain():
            core.on_result(cid, result, last_video_t)
            submitted_tasks.discard(cid)
            active_deadlines.pop(cid, None)
        now = time.monotonic()
        for cid, deadline in list(active_deadlines.items()):
            if now >= deadline:
                core.on_timeout(cid, last_video_t, reason="deadline_exceeded")
                active_deadlines.pop(cid, None)
        core.advance(last_video_t)
        record_decisions()

    camera.start()
    try:
        for fid, ts, bgr in camera.frames():
            if args.max_frames is not None and fid >= args.max_frames:
                break
            wall = time.monotonic()
            order_rows.append({"frame_id": int(fid), "timestamp_s": float(ts),
                               "capture_wall": wall})
            join[int(fid)] = {"yolo": None, "tracknet": None,
                              "yolo_at": None, "tracknet_at": None}
            if bgr is not None:
                ring.push(fid, bgr)
            if yolo_worker is not None:
                yolo_worker.submit(fid, ts, bgr)
            track_worker.submit(fid, ts, bgr)
            drain_workers()
            emit_join(force=False)
            drain_ttnet()
            if not args.quiet and wall - last_progress >= args.progress_s:
                ys = yolo_worker.stats() if yolo_worker else {"processed": 0, "dropped": 0}
                ts_ = track_worker.stats()
                print(f"[realtime] frame={fid} yolo={ys['processed']}/{ys['dropped']} "
                      f"tracknet={ts_['processed']}/{ts_['dropped']} "
                      f"queues={yolo_worker.input.qsize() if yolo_worker else 0}/"
                      f"{track_worker.input.qsize()}/{ttnet_worker.pending if ttnet_worker else 0}",
                      flush=True)
                last_progress = wall
    finally:
        camera.stop()

    if yolo_worker is not None:
        yolo_worker.close()
    track_worker.close()
    drain_workers()
    emit_join(force=True)
    for obs in online.flush():
        process_fused(obs, None)
    for clip in core.flush():
        if args.reviewer == "rule":
            core.on_result(clip.clip_id, reviewer.review(clip), last_video_t)
        else:
            if ttnet_worker.submit(clip, dict(detections)):
                submitted_tasks.add(clip.clip_id)
                if args.review_deadline_s:
                    active_deadlines[clip.clip_id] = time.monotonic() + args.review_deadline_s
            else:
                queue_full += 1
                core.on_timeout(clip.clip_id, last_video_t, reason="queue_full")
    core.advance(last_video_t)
    record_decisions()

    if ttnet_worker is not None:
        while True:
            drain_ttnet()
            if not submitted_tasks and not active_deadlines:
                break
            time.sleep(0.002)
        ttnet_worker.close()
        for cid, result, wall in ttnet_worker.drain():
            core.on_result(cid, result, last_video_t)
            submitted_tasks.discard(cid)
        core.advance(last_video_t)
        record_decisions()

    res = core.finish()
    wall_s = time.monotonic() - started
    realtime_stats = {
        "camera": {"kind": "video" if args.video else "null", "pace": args.pace,
                   "frames": len(order_rows),
                   "fps": round(len(order_rows) / wall_s, 3) if wall_s > 0 else None},
        "stages": {
            "yolo": yolo_worker.stats() if yolo_worker else None,
            "tracknet": track_worker.stats(),
            "ttnet": ttnet_worker.stats() if ttnet_worker else None,
        },
        "queues": {
            "yolo_depth_max": yolo_worker.queue_depth_max if yolo_worker else 0,
            "tracknet_depth_max": track_worker.queue_depth_max,
            "ttnet_depth_max": ttnet_worker.queue_depth_max if ttnet_worker else 0,
        },
        "join_timeouts": join_timeouts,
        "late_worker_results": late_worker_results,
        "ttnet": {
            "queue_full": queue_full,
            "missing_context": sum(1 for r in res["results"] if r.reason == "missing_context"),
            "windows": reviewer.n_windows if reviewer is not None and hasattr(reviewer, "n_windows") else 0,
        },
        "frame_ring": ring.stats(),
        "end_to_end_latency_s": {
            "decision_p50": _pct(decision_lags, 0.5),
            "decision_p95": _pct(decision_lags, 0.95),
            "decision_max": max(decision_lags) if decision_lags else None,
        },
        "wall_s": round(wall_s, 3),
    }
    res["realtime_stats"] = realtime_stats
    return res


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="实时层简化版：视频模拟摄像头 + 并行 worker")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--null-camera", type=int)
    ap.add_argument("--pace", choices=["all", "realtime"], default="all")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=120.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--yolo-replay")
    ap.add_argument("--tracknet-replay")
    ap.add_argument("--yolo-weights")
    ap.add_argument("--tracknet-weights")
    ap.add_argument("--yolo-ms", type=float, default=0.0)
    ap.add_argument("--tracknet-ms", type=float, default=0.0)
    ap.add_argument("--ttnet-ms-per-window", type=float, default=14.0)
    ap.add_argument("--yolo-queue", type=int, default=64)
    ap.add_argument("--tracknet-queue", type=int, default=64)
    ap.add_argument("--ttnet-queue", type=int, default=8)
    ap.add_argument("--join-timeout-s", type=float, default=0.5)
    ap.add_argument("--ring-s", type=float, default=3.0)
    ap.add_argument("--reviewer", choices=["rule", "ttnet"], default="ttnet")
    ap.add_argument("--review-deadline-s", type=float, default=0.0)
    ap.add_argument("--ttnet-cache")
    ap.add_argument("--ttnet-weights")
    ap.add_argument("--ttnet-src")
    ap.add_argument("--ttnet-engine", choices=["auto", "torch", "numpy"], default="auto")
    ap.add_argument("--ttnet-device", default="cuda:0")
    ap.add_argument("--ttnet-k", type=int, default=DEFAULT_K)
    ap.add_argument("--preset", default="test2")
    ap.add_argument("--quad")
    ap.add_argument("--primary", choices=["tracknet", "yolo"], default="tracknet")
    ap.add_argument("--no-yolo-recapture", action="store_true")
    ap.add_argument("--yolo-min-conf", type=float, default=0.5)
    ap.add_argument("--fill-window-s", type=float, default=0.5)
    ap.add_argument("--fuse-gate-s", type=float, default=0.017)
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--max-clip-s", type=float, default=1.0)
    ap.add_argument("--pre-s", type=float, default=0.1)
    ap.add_argument("--post-s", type=float, default=0.1)
    ap.add_argument("--bounce-min-drop", type=float, default=8.0)
    ap.add_argument("--table-margin-px", type=float, default=30.0)
    ap.add_argument("--hit-trigger", action="store_true")
    ap.add_argument("--no-dropout-trigger", action="store_true")
    ap.add_argument("--dropout-min-s", type=float, default=0.05)
    ap.add_argument("--no-jitter-merge", action="store_true")
    ap.add_argument("--no-out", action="store_true")
    ap.add_argument("--state-machine", choices=["tn", "ttnet11-yolo", "ttnet11-ttnet"], default="tn")
    ap.add_argument("--yolo-imgsz", type=int, default=960)
    ap.add_argument("--yolo-conf", type=float, default=0.18)
    ap.add_argument("--yolo-device", default=None)
    ap.add_argument("--tracknet-device", default=None)
    ap.add_argument("--tracknet-min-conf", type=float, default=0.5)
    ap.add_argument("--progress-s", type=float, default=1.0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv=None) -> int:
    a = _build_parser().parse_args(argv)
    if a.join_timeout_s < 0 or a.review_deadline_s < 0:
        raise SystemExit("join/deadline 秒数必须 >= 0")
    res = run_realtime(a)
    if not a.quiet:
        print(json.dumps(res["realtime_stats"], ensure_ascii=False, indent=2))
        print("score", res["summary"]["score"], "decisions", res["summary"]["decisions"])
    if a.out_dir:
        write_outputs(a.out_dir, res)
        with open(os.path.join(a.out_dir, "realtime_stats.json"), "w", encoding="utf-8") as f:
            json.dump(res["realtime_stats"], f, ensure_ascii=False, indent=2)
        print(f"输出已写入: {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
