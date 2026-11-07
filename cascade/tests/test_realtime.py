from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest

from cascade.eval_proposals import evaluate as evaluate_proposals
from cascade.fuse_obs import FuseParams, OnlineFusion, fuse
from cascade.realtime.frames import FrameRing
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
    def _args(self, reviewer, **kw):
        argv = ["--null-camera", "3600", "--fps", "120", "--pace", "all",
                "--yolo-replay", YOLO, "--tracknet-replay", OBS,
                "--reviewer", reviewer, "--preset", "test2", "--quiet"]
        if reviewer == "ttnet":
            argv += ["--ttnet-cache", TTNET, "--ttnet-k", "4",
                     "--ttnet-ms-per-window", "0"]
        return _build_parser().parse_args(argv)

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
            "--null-camera", "500", "--fps", "120", "--pace", "all",
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
            "--ttnet-ms-per-window", "80", "--review-deadline-s", "0.01",
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
            "--reviewer", "rule", "--preset", "test2", "--quiet"])
        res = run_realtime(a)
        self.assertGreater(res["realtime_stats"]["stages"]["tracknet"]["dropped"], 0)
        self.assertIn("A", res["summary"]["score"])
        self.assertIn("B", res["summary"]["score"])


if __name__ == "__main__":
    unittest.main()
