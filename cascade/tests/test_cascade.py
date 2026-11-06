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
            write_outputs(tmp, run(OBS, preset="test2"))
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
