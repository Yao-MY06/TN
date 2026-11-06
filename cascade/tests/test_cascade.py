"""级联层回归测试。只用仓库内样例（test_2 前 30 s），不需要视频、权重或 GPU。

    python -m unittest discover -s cascade/tests -t .        # 在 TN 根目录
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TN_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if TN_ROOT not in sys.path:
    sys.path.insert(0, TN_ROOT)

from cascade._paths import EXAMPLES_DIR, load_detect_bounce  # noqa: E402
from cascade.committer import CommitConfig  # noqa: E402
from cascade.frame_index import FrameIndex  # noqa: E402
from cascade.online_out import OnlineOutDetector  # noqa: E402
from cascade.proposals import ProposalParams  # noqa: E402
from cascade.run_cascade import run  # noqa: E402

import run_mixed as rm  # noqa: E402
from core.schemas import FrameObservation  # noqa: E402

OBS = os.path.join(EXAMPLES_DIR, "test_2_observations.jsonl")
EVENTS = os.path.join(EXAMPLES_DIR, "test_2_events.jsonl")


def _offline_bounces():
    """离线基线：events_from_observations.py 的核心流程（detect + build_events）。"""
    db = load_detect_bounce()
    sys.path.insert(0, os.path.join(TN_ROOT, "TrackNet", "TrackNet"))
    from events_from_observations import obs_to_arrays  # noqa: E402
    from observation import load_observations  # noqa: E402
    obs = load_observations(OBS)
    x, y, det = obs_to_arrays(obs)
    poly = db.load_quad("test_2")
    dets = db.detect(x, y, det, poly, 12, 15.0, margin=0.0, min_gap=15,
                     H=db.table_homography(poly), dead=0.05)
    return db.build_events(dets, 120.0, source="tracknet")


class TestFrameIndex(unittest.TestCase):
    def test_lookup(self):
        ix = FrameIndex()
        for f in range(10):
            ix.add(f, f / 120.0)
        self.assertEqual(ix.first_at_or_after(0.05), 6)
        self.assertEqual(ix.last_at_or_before(0.05), 6)
        self.assertEqual(ix.last_at_or_before(0.049), 5)
        self.assertEqual(ix.shift(3, 4), 7)
        self.assertIsNone(ix.shift(8, 4))
        self.assertTrue(ix.seen_through(9 / 120.0))
        with self.assertRaises(ValueError):
            ix.add(9, 1.0)


class TestOfflineParity(unittest.TestCase):
    """在线实现必须与离线脚本逐条一致。"""

    @classmethod
    def setUpClass(cls):
        cls.res = run(OBS, preset="test2")

    def test_rule_reviewer_equals_detect_bounce(self):
        online = [e for r in self.res["results"] for e in r.events]
        offline = _offline_bounces()
        self.assertEqual(len(offline), 34)
        self.assertEqual(online, offline)

    def test_offline_bounces_equal_checked_in_sample(self):
        with open(EVENTS, encoding="utf-8") as f:
            sample = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(_offline_bounces(), sample)

    def test_every_offline_bounce_is_proposed(self):
        anchors = {p.anchor_frame_id for p in self.res["proposals"]}
        for e in _offline_bounces():
            self.assertIn(round(e["timestamp_s"] * 120), anchors)

    def test_dropouts_equal_find_dropouts(self):
        ref = self.res["referee"]
        with open(OBS, encoding="utf-8") as f:
            obs = [FrameObservation.from_dict(json.loads(line)) for line in f if line.strip()]
        track, _ = rm.mask_low_confidence(obs, 0.5)
        self.assertEqual(ref._dropouts, rm.find_dropouts(track))

    def test_online_out_equals_derive_out_events(self):
        with open(OBS, encoding="utf-8") as f:
            obs = [FrameObservation.from_dict(json.loads(line)) for line in f if line.strip()]
        calib = rm.build_calib("test2")
        bt = [e.timestamp_s for e in self.res["referee"].kept_bounces]
        for bounce_times, desc in ((bt, True), ([], True), ([], False)):
            off = rm.derive_out_events(obs, calib, bounce_times=bounce_times, need_descending=desc)
            det = OnlineOutDetector(calib, need_descending=desc)
            for o in obs:
                det.push(o)
            det.finish()
            on = det.finalize(float("inf"), sorted(bounce_times))
            self.assertEqual([e.to_dict() for e in on], [e.to_dict() for e in off],
                             f"bounce_times={len(bounce_times)} desc={desc}")


class TestMatchesRunMixed(unittest.TestCase):
    """C3a（规则复核）必须与 run_mixed（C1）给出完全相同的事件流与判罚。"""

    def test_event_stream_and_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev_path = os.path.join(tmp, "c1.jsonl")
            proc = subprocess.run(
                [sys.executable, os.path.join(TN_ROOT, "tt", "run_mixed.py"), "--jsonl", OBS,
                 "--bounces", EVENTS, "--preset", "test2", "--jitter-merge", "--events-out", ev_path],
                capture_output=True, text=True, encoding="utf-8", cwd=TN_ROOT)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(ev_path, encoding="utf-8") as f:
                c1 = [json.loads(line) for line in f if line.strip()]
        res = run(OBS, preset="test2")
        ref = res["referee"]
        self.assertEqual(ref.dumped_events, c1)
        self.assertEqual(ref.score, (0, 2))
        decided = [line.split("- ", 1)[1] for line in proc.stdout.splitlines()
                   if line.strip().startswith("- [")]
        self.assertEqual(ref.decisions, decided)


class TestLatencyInvariance(unittest.TestCase):
    """复核延迟、是否合并只改变输出时刻，不改变判罚和事件流。"""

    def test_invariance(self):
        base = run(OBS, preset="test2")["referee"]
        for lat in (1, 37, 240):
            for merge in (True, False):
                r = run(OBS, preset="test2", review_latency_frames=lat, merge=merge)["referee"]
                self.assertEqual(r.dumped_events, base.dumped_events, f"lat={lat} merge={merge}")
                self.assertEqual(r.decisions, base.decisions, f"lat={lat} merge={merge}")

    def test_oracle_reviewer_same_result(self):
        res = run(OBS, preset="test2", reviewer="oracle", oracle_events=EVENTS)
        base = run(OBS, preset="test2")["referee"]
        self.assertEqual(res["summary"]["oracle_missed"], [])
        self.assertEqual(res["referee"].dumped_events, base.dumped_events)


class TestReviewDeadline(unittest.TestCase):
    def test_completion_before_deadline_matches_no_deadline(self):
        base = run(OBS, preset="test2", review_latency_frames=10)
        limited = run(OBS, preset="test2", review_latency_frames=10,
                      review_deadline_frames=20)
        self.assertEqual(base["referee"].dumped_events, limited["referee"].dumped_events)
        self.assertEqual(base["referee"].decisions, limited["referee"].decisions)
        self.assertEqual([r.to_dict() for r in base["results"]],
                         [r.to_dict() for r in limited["results"]])
        self.assertEqual(limited["summary"]["timeout_count"], 0)
        self.assertEqual(limited["late_results"], [])

    def test_timeout_does_not_block_commit_and_late_events_are_ignored(self):
        res = run(OBS, preset="test2", review_latency_frames=240,
                  review_deadline_frames=30)
        s = res["summary"]
        self.assertGreater(s["timeout_count"], 0)
        self.assertEqual(s["timeout_count"], len(res["late_results"]))
        self.assertGreater(s["timeout_ratio"], 0.0)
        self.assertEqual(s["reviews"].get("timeout"), s["timeout_count"])
        timeout_ids = {r.clip_id for r in res["results"] if r.status == "timeout"}
        late_ids = {r.clip_id for r in res["late_results"]}
        self.assertEqual(timeout_ids, late_ids)
        late_event_ids = {e["event_id"] for r in res["late_results"] for e in r.events}
        committed_event_ids = {e["event_id"] for e in res["referee"].dumped_events}
        self.assertTrue(late_event_ids)
        self.assertFalse(late_event_ids & committed_event_ids)

        queue = res["review_queue"]
        timeout_rows = [row for row in queue if row["status"] == "timeout"]
        self.assertEqual(len(timeout_rows), s["timeout_count"])
        self.assertEqual(s["manual_confirm_count"], len(queue))
        self.assertGreaterEqual(len(queue), s["timeout_count"])
        for row in timeout_rows:
            self.assertIn("deadline_exceeded", row["reason_codes"])
            self.assertIsNotNone(row["start_frame_id"])
            self.assertIsNotNone(row["end_frame_id"])


class TestReviewQueue(unittest.TestCase):
    def test_manual_statuses_include_range_reason_hump_and_nearby_decisions(self):
        from types import SimpleNamespace
        from cascade.review_queue import build_review_queue
        from cascade.reviewers import ReviewResult
        from cascade.scheduler import Clip

        clips = [
            Clip("s01:c0001", [], 1.0, 1.2, start_frame_id=120, end_frame_id=144),
            Clip("s01:c0002", [], 5.0, 5.2, start_frame_id=600, end_frame_id=624),
            Clip("s01:c0003", [], 8.0, 8.2, start_frame_id=960, end_frame_id=984),
            Clip("s01:c0004", [], 9.0, 9.2, start_frame_id=1080, end_frame_id=1104),
        ]
        results = [
            ReviewResult("s01:c0001", ["p1"], "uncertain", [], "off_table@1216",
                         reviewer="ttnet"),
            ReviewResult("s01:c0002", ["p2"], "error", [], "missing_context",
                         reviewer="ttnet"),
            ReviewResult("s01:c0003", ["p3"], "rejected", [], "no_hump", reviewer="ttnet"),
            ReviewResult("s01:c0004", ["p4"], "timeout", [], "deadline_exceeded",
                         reviewer="ttnet"),
        ]
        reviewer = SimpleNamespace(log=[
            {"clip_id": "s01:c0001", "humps": [
                {"start": 120, "end": 122, "peak_frame": 121, "peak": 0.72, "truncated": False},
            ]},
            {"clip_id": "s01:c0002", "reasons": ["missing_context"]},
        ])
        referee = SimpleNamespace(decision_records=[
            {"virtual_t": 1.8, "real_t": 2.0, "winner": "A", "rule": "OUT", "text": "near"},
            {"virtual_t": 7.0, "real_t": 7.1, "winner": "B", "rule": "OUT", "text": "far"},
        ])
        res = {"clips": clips, "results": results, "reviewer": reviewer, "referee": referee}
        queue = build_review_queue(res)
        self.assertEqual([row["clip_id"] for row in queue],
                         ["s01:c0001", "s01:c0002", "s01:c0004"])
        first = queue[0]
        self.assertEqual(first["reason_codes"], ["off_table@1216"])
        self.assertEqual(first["hump_peak"], {"frame": 121, "score": 0.72})
        self.assertEqual([d["text"] for d in first["nearby_decisions"]], ["near"])
        self.assertEqual(queue[1]["reason_codes"], ["missing_context"])
        self.assertIsNone(queue[1]["hump_peak"])
        self.assertEqual(queue[2]["status"], "timeout")


class TestConfigVariants(unittest.TestCase):
    def test_no_out_no_jitter_still_consistent(self):
        """关掉出界推断与抖动合并时，与 run_mixed 对应开关的结果一致。"""
        with tempfile.TemporaryDirectory() as tmp:
            ev_path = os.path.join(tmp, "c1.jsonl")
            proc = subprocess.run(
                [sys.executable, os.path.join(TN_ROOT, "tt", "run_mixed.py"), "--jsonl", OBS,
                 "--bounces", EVENTS, "--preset", "test2", "--no-out", "--events-out", ev_path],
                capture_output=True, text=True, encoding="utf-8", cwd=TN_ROOT)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(ev_path, encoding="utf-8") as f:
                c1 = [json.loads(line) for line in f if line.strip()]
        r = run(OBS, preset="test2", commit=CommitConfig(jitter_merge=False, use_out=False),
                review_latency_frames=50)["referee"]
        self.assertEqual(r.dumped_events, c1)

    def test_strict_proposals_still_cover(self):
        """宽松阈值收紧到与离线相同，提议仍覆盖全部离线落台（严格 ⊆ 宽松 的边界情况）。"""
        p = ProposalParams(bounce_min_drop=15.0, table_margin_px=0.0, hit_trigger=False,
                           dropout_trigger=False)
        res = run(OBS, preset="test2", params=p)
        online = [e for r in res["results"] for e in r.events]
        self.assertEqual(online, _offline_bounces())


class TestQuadCalibration(unittest.TestCase):
    def test_game4_from_quad_matches_run_mixed(self):
        from cascade.calib import build_calib_from_quad
        self.assertEqual(build_calib_from_quad("game_4"), rm.build_calib("game4"))

    def test_preset_alias_is_accepted(self):
        from cascade.calib import build_calib_from_quad
        self.assertEqual(build_calib_from_quad("game4"), rm.build_calib("game4"))


EXT_GT = os.path.join(TN_ROOT, "_data", "gt", "extended_openttgames", "game_data", "test", "test_2.json")


@unittest.skipUnless(os.path.isfile(EXT_GT), "缺 Extended OpenTTGames 标注（_data/gt/extended_openttgames）")
class TestExtendedGroundTruth(unittest.TestCase):
    """独立真值：每一分的得分方必须正确，击球不得漏检。"""

    def test_points_and_hits(self):
        from cascade.eval_extended import evaluate, parse_game_json
        from cascade.run_cascade import write_outputs
        strokes, endings = parse_game_json(EXT_GT)
        self.assertEqual((len(strokes), len(endings)), (29, 2))
        with tempfile.TemporaryDirectory() as tmp:
            res = run(OBS, preset="test2")
            write_outputs(tmp, res)
            with open(os.path.join(tmp, "review_queue.jsonl"), encoding="utf-8") as f:
                queue = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(queue), res["summary"]["manual_confirm_count"])
            self.assertIn("manual_confirm_ratio", res["summary"])
            self.assertIn("missing_context_ratio", res["summary"])
            r = evaluate(tmp, EXT_GT)
        self.assertEqual(r["points"]["correct"], 2)
        self.assertEqual(r["points"]["spurious_points"], [])
        self.assertEqual(r["hits"]["lag_aware"]["r"], 1.0)
        self.assertEqual(r["hits"]["lag_aware"]["side_agree"], "29/29")


EXT_ROOT = os.path.join(TN_ROOT, "_data", "gt", "extended_openttgames")
T2_BALL = os.path.join(EXT_ROOT, "ball_data", "test", "test_2.json")


@unittest.skipUnless(os.path.isfile(T2_BALL), "缺 Extended OpenTTGames 球位置标注")
class TestInferredBounceGroundTruth(unittest.TestCase):
    """推断真值的可靠性，以及默认提议参数下的落台召回。"""

    def test_inference_rule_on_six_videos(self):
        from cascade.infer_markup import validate
        t = validate(EXT_ROOT)["total"]
        self.assertEqual(t["found"], t["gt"])                 # 370/370 逐帧找回
        self.assertGreaterEqual(t["typed_ok"] / t["typed"], 0.97)

    def test_inferred_test2_matches_tn_report(self):
        """推断真值复现 TN 报告的 0.774（34 条，全片口径）。"""
        from cascade.eval_proposals import match
        from cascade.infer_markup import infer, load_ball
        inf = infer(load_ball(T2_BALL), 929.0)
        gt = sorted(f for f, v in inf.items() if v == "bounce")
        self.assertEqual(len(gt), 28)
        with open(EVENTS, encoding="utf-8") as f:
            pred = [round(json.loads(line)["timestamp_s"] * 120) for line in f if line.strip()]
        self.assertAlmostEqual(match(pred, gt, 5)["f1"], 0.7742, places=4)

    def test_default_proposals_cover_all_bounces_with_margin(self):
        from cascade.infer_markup import infer, load_ball
        inf = infer(load_ball(T2_BALL), 929.0)
        gt = [f for f, v in inf.items() if v == "bounce"]
        res = run(OBS, preset="test2")
        clips = res["clips"]
        for g in gt:
            self.assertTrue(any(c.start_frame_id + 6 <= g <= c.end_frame_id - 6 for c in clips), g)
        self.assertLess(res["summary"]["clips"]["coverage"], 0.40)


if __name__ == "__main__":
    unittest.main()
