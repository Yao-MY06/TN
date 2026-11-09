from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest

from cascade.eval_proposals import evaluate as evaluate_proposals
from cascade.fuse_obs import FuseParams, OnlineFusion, fuse
from cascade.realtime.frames import FrameRing, VideoFileCamera
from cascade.run_cascade import run as run_cascade, write_outputs
from cascade.run_realtime import _build_parser, run_realtime

TN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OBS = os.path.join(TN_ROOT, "adapter", "examples", "test_2_observations.jsonl")
YOLO = os.path.join(TN_ROOT, "_data", "obs", "yolo", "test_2.jsonl")
TTNET = os.path.join(TN_ROOT, "_data", "ttnet", "test_2.jsonl")
MARKUP = os.path.join(TN_ROOT, "_data", "gt", "extended_openttgames", "inferred",
                      "test_2_events_markup_inferred.json")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def _event_rows(res):
    ref = res["referee"]
    return [{**e, "sources": ref.event_sources.get(e["event_id"], [])}
            for e in ref.dumped_events]


class TestFrameRing(unittest.TestCase):
    def test_capacity_and_eviction(self):
        ring = FrameRing(capacity_s=2.0, fps=10.0)
        import numpy as np
        frame = np.zeros((128, 320, 3), dtype=np.uint8)
        for fid in range(30):
            ring.push(fid, frame)
        self.assertEqual(ring.capacity_frames, 20)
        self.assertEqual(len(ring), 20)
        self.assertIsNone(ring.get(9))
        self.assertGreaterEqual(ring.evicted_before_read, 1)
        self.assertIsNotNone(ring.get(20))

    def test_multithreaded_read_write(self):
        import numpy as np
        ring = FrameRing(capacity_s=5.0, fps=100.0)
        frame = np.zeros((128, 320, 3), dtype=np.uint8)
        errors = []

        def writer():
            try:
                for fid in range(500):
                    ring.push(fid, frame)
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for fid in range(500):
                    ring.get(fid)
            except Exception as exc:
                errors.append(exc)

        a, b = threading.Thread(target=writer), threading.Thread(target=reader)
        a.start(); b.start(); a.join(); b.join()
        self.assertEqual(errors, [])


class TestVideoFileCamera(unittest.TestCase):
    def test_video_file_camera_all_pace(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (64, 48))
            if not writer.isOpened():
                self.skipTest("cv2 VideoWriter(MJPG) 不可用")
            for _ in range(5):
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            cam = VideoFileCamera(path, pace="all")
            cam.start()
            frames = list(cam.frames())
            cam.stop()
            self.assertEqual([f[0] for f in frames], list(range(5)))
            self.assertAlmostEqual(frames[-1][1], 4 / 20.0, places=3)
            self.assertEqual(frames[0][2].shape, (48, 64, 3))


@unittest.skipUnless(os.path.isfile(YOLO), "缺 test_2 YOLO 观测")
class TestOnlineFusion(unittest.TestCase):
    def test_matches_offline_per_frame(self):
        tn = _load(OBS); yo = _load(YOLO)
        tb = {int(x["frame_id"]): x for x in tn}
        yb = {int(x["frame_id"]): x for x in yo}
        for primary in ("tracknet", "yolo"):
            p = FuseParams(primary=primary)
            offline, _ = fuse(yo, tn, p)
            online = OnlineFusion(p)
            out = []
            for fid in sorted(set(tb) | set(yb)):
                ts = (tb.get(fid) or yb.get(fid))["timestamp_s"]
                out.extend(online.push(fid, ts, yb.get(fid), tb.get(fid), now_t=ts))
            out.extend(online.flush())
            self.assertEqual(out, offline, primary)


@unittest.skipUnless(os.path.isfile(YOLO) and os.path.isfile(TTNET) and os.path.isfile(MARKUP),
                     "缺实时回放验收数据")
class TestReplayConsistency(unittest.TestCase):
    def _args(self, reviewer, ttnet_ms="0", ttnet_queue="8"):
        argv = ["--null-camera", "3600", "--fps", "120", "--pace", "all",
                "--yolo-replay", YOLO, "--tracknet-replay", OBS,
                "--reviewer", reviewer, "--preset", "test2", "--quiet"]
        if reviewer == "ttnet":
            argv += ["--ttnet-cache", TTNET, "--ttnet-k", "4",
                     "--ttnet-ms-per-window", ttnet_ms, "--ttnet-queue", ttnet_queue]
        return _build_parser().parse_args(argv)

    def test_pace_all_backpressure_when_ttnet_is_slow(self):
        # --pace all 不丢复核：TTNet 慢、队列只有 1 时也必须等，结果与同步回放逐条一致
        rt = run_realtime(self._args("ttnet", ttnet_ms="2", ttnet_queue="1"))
        self.assertEqual(rt["realtime_stats"]["ttnet"]["queue_full"], 0)
        self.assertEqual(rt["summary"]["timeout_count"], 0)
        tt_args = type("A", (), {"ttnet_cache": TTNET, "video": None, "ttnet_weights": None,
                                 "ttnet_src": None, "ttnet_engine": "auto",
                                 "ttnet_device": "cpu", "ttnet_k": 4})()
        base = run_cascade(OBS, preset="test2", reviewer="ttnet", yolo_jsonl=YOLO,
                           pre_ctx=4, post_ctx=4, ttnet_args=tt_args)
        self.assertEqual(_event_rows(rt), _event_rows(base))
        self.assertEqual(rt["summary"]["decisions"], base["summary"]["decisions"])
        self.assertEqual(rt["summary"]["score"], {"A": 0, "B": 2})

    def test_rule_and_ttnet_match_run_cascade(self):
        expected_f1 = {"rule": 0.903, "ttnet": 0.871}
        for reviewer in ("rule", "ttnet"):
            with self.subTest(reviewer=reviewer):
                rt = run_realtime(self._args(reviewer))
                tt_args = None
                if reviewer == "ttnet":
                    tt_args = type("A", (), {"ttnet_cache": TTNET, "video": None,
                                             "ttnet_weights": None, "ttnet_src": None,
                                             "ttnet_engine": "auto", "ttnet_device": "cpu",
                                             "ttnet_k": 4})()
                base = run_cascade(
                    OBS, preset="test2", reviewer=reviewer, yolo_jsonl=YOLO,
                    pre_ctx=(8 - 4) if reviewer == "ttnet" else 0,
                    post_ctx=4 if reviewer == "ttnet" else 0, ttnet_args=tt_args)
                self.assertEqual(_event_rows(rt), _event_rows(base))
                self.assertEqual(rt["summary"]["decisions"], base["summary"]["decisions"])
                self.assertEqual(rt["summary"]["score"], base["summary"]["score"])
                rt_dec = [{k: v for k, v in row.items() if k != "real_t"}
                          for row in rt["referee"].decision_records]
                base_dec = [{k: v for k, v in row.items() if k != "real_t"}
                            for row in base["referee"].decision_records]
                self.assertEqual(rt_dec, base_dec)
                with tempfile.TemporaryDirectory() as tmp:
                    write_outputs(tmp, rt)
                    ev = evaluate_proposals(tmp, MARKUP, OBS, 120.0)
                    self.assertAlmostEqual(ev["confirmed_bounce_tol5"]["f1"],
                                           expected_f1[reviewer], places=3)
                self.assertEqual(rt["summary"]["score"], {"A": 0, "B": 2})


@unittest.skipUnless(os.path.isfile(YOLO) and os.path.isfile(TTNET), "缺 test_2 回放数据")
class TestRealtimeFailures(unittest.TestCase):
    def test_ttnet_queue_full_becomes_manual_timeout(self):
        a = _build_parser().parse_args([
            "--null-camera", "500", "--fps", "120", "--pace", "realtime",
            "--yolo-replay", YOLO, "--tracknet-replay", OBS,
            "--reviewer", "ttnet", "--ttnet-cache", TTNET,
            "--ttnet-ms-per-window", "80", "--ttnet-queue", "1",
            "--preset", "test2", "--quiet"])
        res = run_realtime(a)
        self.assertGreater(res["realtime_stats"]["ttnet"]["queue_full"], 0)
        self.assertTrue(any(q["reason"] == "queue_full" for q in res["review_queue"]))
        self.assertEqual(res["summary"]["score"], {"A": 0, "B": 0})

    def test_deadline_timeout_keeps_late_results_out_of_state_machine(self):
        a = _build_parser().parse_args([
            "--null-camera", "500", "--fps", "120", "--pace", "all",
            "--yolo-replay", YOLO, "--tracknet-replay", OBS,
            "--reviewer", "ttnet", "--ttnet-cache", TTNET,
            "--ttnet-ms-per-window", "20", "--review-deadline-s", "0.01",
            "--preset", "test2", "--quiet"])
        res = run_realtime(a)
        self.assertGreater(res["summary"]["timeout_count"], 0)
        timeout_rows = [r for r in res["results"] if r.status == "timeout"]
        self.assertTrue(timeout_rows)
        self.assertTrue(all(r.reason == "deadline_exceeded" for r in timeout_rows))
        late_ids = {e["event_id"] for r in res["late_results"] for e in r.events}
        committed = {e["event_id"] for e in res["referee"].dumped_events}
        self.assertFalse(late_ids & committed)


@unittest.skipUnless(os.path.isfile(YOLO), "缺 test_2 YOLO 观测")
class TestRealtimeDrop(unittest.TestCase):
    def test_tracknet_backlog_drops_without_hanging(self):
        a = _build_parser().parse_args([
            "--null-camera", "800", "--fps", "120", "--pace", "realtime", "--speed", "20",
            "--yolo-replay", YOLO, "--tracknet-replay", OBS,
            "--yolo-ms", "5", "--tracknet-ms", "42", "--tracknet-queue", "8",
            "--join-timeout-s", "0.01", "--reviewer", "rule",
            "--preset", "test2", "--quiet"])
        res = run_realtime(a)
        self.assertGreater(res["realtime_stats"]["stages"]["tracknet"]["dropped"], 0)
        self.assertGreater(res["realtime_stats"]["join_timeouts"], 0)
        self.assertGreater(res["realtime_stats"]["late_worker_results"], 0)
        self.assertIn("A", res["summary"]["score"])
        self.assertIn("B", res["summary"]["score"])


class TestConsecutiveFrames(unittest.TestCase):
    def test_window_is_camera_consecutive_even_when_inference_skips_frames(self):
        from cascade.realtime.workers import ConsecutiveFrames
        h = ConsecutiveFrames(lambda x: x, keep=8)
        for f in range(10):
            h.observe(f, f"img{f}")          # 每帧都缓存，哪怕之后不推理
        self.assertEqual(h.window(9), ("img9", "img8", "img7"))   # (t, t-1, t-2)
        self.assertEqual(h.window(5), ("img5", "img4", "img3"))
        self.assertIsNone(h.window(2))       # 0、1 已被淘汰（keep=8）
        g = ConsecutiveFrames(lambda x: x)
        g.observe(0, "a"); g.observe(2, "c")
        self.assertIsNone(g.window(2))       # 缺第 1 帧 → 不推理，输出不可见

    def test_frame_hook_sees_dropped_frames(self):
        from cascade.realtime.workers import TrackNetWorker
        seen = []
        def pred(batch):
            return []
        pred.observe = lambda fid, bgr: seen.append(fid)
        w = TrackNetWorker(batch_predictor=pred, pace="realtime", queue_size=1)
        for f in range(5):                   # 未启动线程：队列满后丢最旧帧
            w.submit(f, f / 120, object())
        self.assertEqual(seen, [0, 1, 2, 3, 4])
        self.assertGreater(w.dropped, 0)


class TestQueueDefaults(unittest.TestCase):
    def test_realtime_uses_small_input_queues(self):
        from cascade.run_realtime import _resolve_queue_sizes
        a = _build_parser().parse_args(["--null-camera", "10", "--pace", "realtime"])
        _resolve_queue_sizes(a)
        self.assertEqual((a.yolo_queue, a.tracknet_queue), (2, 2))
        b = _build_parser().parse_args(["--null-camera", "10", "--pace", "all"])
        _resolve_queue_sizes(b)
        self.assertEqual((b.yolo_queue, b.tracknet_queue), (64, 64))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(os.path.isfile(YOLO), "缺 test_2 YOLO 观测")
class TestPaceAllNoJoinTimeout(unittest.TestCase):
    """--pace all 不丢帧：模型比汇合超时还慢时也必须等结果，不能把排队中的观测当超时作废。"""

    def _args(self, tracknet_ms):
        return _build_parser().parse_args([
            "--null-camera", "400", "--fps", "120", "--pace", "all",
            "--yolo-replay", YOLO, "--tracknet-replay", OBS,
            "--yolo-ms", "0", "--tracknet-ms", tracknet_ms,
            "--join-timeout-s", "0.01", "--reviewer", "rule",
            "--preset", "test2", "--quiet"])

    def test_slow_models_still_join_every_frame(self):
        fast = run_realtime(self._args("0"))
        slow = run_realtime(self._args("3"))
        self.assertEqual(slow["realtime_stats"]["join_timeouts"], 0)
        self.assertEqual(slow["realtime_stats"]["late_worker_results"], 0)
        self.assertEqual(slow["fused_trace"], fast["fused_trace"])
        self.assertEqual(_event_rows(slow), _event_rows(fast))
