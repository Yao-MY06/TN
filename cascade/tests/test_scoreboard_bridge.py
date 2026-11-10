from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TN_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if TN_ROOT not in sys.path:
    sys.path.insert(0, TN_ROOT)

from cascade.scoreboard_bridge import ScoreboardBridge, VOICE_ROOT  # noqa: E402
from cascade.run_scoreboard import _build_parser, run_scoreboard  # noqa: E402

HAS_VOICE = os.path.isdir(VOICE_ROOT)
RULE_RUNS = os.path.join(TN_ROOT, "outputs", "cascade_batch_sm", "tn")
HAS_RUNS = all(os.path.isfile(os.path.join(RULE_RUNS, f"test_{n}_rule", "summary.json"))
               for n in range(1, 8))


def _point(side, rule="OUT", t=1.0):
    return {"virtual_t": t, "kind": "POINT", "winner": side, "rule": rule, "text": ""}


def _dead(rule="NO_EVIDENCE", t=1.0):
    return {"virtual_t": t, "kind": "DEAD_BALL", "winner": None, "rule": rule, "text": ""}


@unittest.skipUnless(HAS_VOICE, "缺语音仓库：" + VOICE_ROOT)
class TestScoreboardBridge(unittest.TestCase):
    def _bridge(self, **kwargs):
        return ScoreboardBridge(say=False, **kwargs)

    def test_physical_to_player_mapping_after_set_switch(self):
        bridge = self._bridge()
        self.assertEqual(bridge.map_winner("A"), "A")
        self.assertEqual(bridge.map_winner("B"), "B")
        engine = bridge.engine
        engine.points = {"A": 10, "B": 0}
        engine.award("A", reason="test", source="test")
        self.assertEqual(engine.side_of_a, "right")
        self.assertEqual(bridge.map_winner("A"), "B")
        self.assertEqual(bridge.map_winner("B"), "A")

    def test_point_dead_ball_let_and_timeout(self):
        bridge = self._bridge(pending_timeout_s=5.0)
        out = bridge.apply_decision(_point("A", "OUT", 1.0), 1.0)
        self.assertEqual(bridge.engine.points, {"A": 1, "B": 0})
        self.assertEqual(out["mapped_side"], "A")
        self.assertEqual(bridge.physical_points, {"A": 1, "B": 0})

        bridge.apply_decision(_dead("NO_EVIDENCE", 2.0), 2.0)
        self.assertEqual(bridge.engine.pending["side"], "UNKNOWN")
        self.assertEqual(bridge.engine.state, "PENDING")
        bridge.tick(6.999)
        self.assertIsNotNone(bridge.engine.pending)
        bridge.tick(7.0)
        self.assertIsNone(bridge.engine.pending)
        self.assertIn("pending_timeout", [e.get("action") for e in bridge.logs])

        before = dict(bridge.engine.points)
        bridge.apply_decision({"kind": "LET", "rule": "LET", "winner": None}, 8.0)
        self.assertEqual(bridge.engine.points, before)

    def test_visual_point_rejects_existing_visual_pending(self):
        bridge = self._bridge(pending_timeout_s=30.0)
        bridge.apply_decision(_dead("NO_EVIDENCE", 1.0), 1.0)
        self.assertIsNotNone(bridge.engine.pending)
        bridge.apply_decision(_point("A", "OUT", 2.0), 2.0)
        self.assertIsNone(bridge.engine.pending)
        self.assertEqual(bridge.engine.points, {"A": 1, "B": 0})
        actions = [entry.get("action") for entry in bridge.logs]
        self.assertIn("reject_pending_before_point", actions)

    def test_voice_text_undo_and_confirm_dead_ball(self):
        try:
            from tt_referee.voice_compat import SharedScoreBoard
            from tt_voice.app import Pipeline
            from tt_voice.config import load_config
            from tt_voice.feedback import NullSpeaker
        except Exception as exc:
            self.skipTest(f"语音文本依赖不可用：{exc}")

        bridge = self._bridge(pending_timeout_s=30.0)
        pipe = Pipeline(load_config(), engine=None, board=bridge.board,
                        speaker=NullSpeaker(), start_awake=True)
        bridge.apply_decision(_point("A", "OUT", 1.0), 1.0)
        pipe.push_text("小裁判 撤销上一分", now=2.0)
        self.assertEqual(bridge.engine.points, {"A": 0, "B": 0})

        bridge.apply_decision(_dead("NO_EVIDENCE", 3.0), 3.0)
        self.assertEqual(bridge.engine.pending["side"], "UNKNOWN")
        pipe.push_text("小裁判 左方加一分", now=4.0)
        self.assertIsNone(bridge.engine.pending)
        self.assertEqual(bridge.engine.points, {"A": 1, "B": 0})

    def test_unknown_pending_confirm_keeps_pending_and_requests_side(self):
        bridge = self._bridge(pending_timeout_s=30.0)
        bridge.apply_decision(_dead("NO_EVIDENCE", 1.0), 1.0)
        result = bridge.board.apply("confirm", raw_text="确认这一分", score=0.99)
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "side_required")
        self.assertIsNotNone(bridge.engine.pending)

    def test_manual_override_is_logged(self):
        bridge = self._bridge()
        bridge.apply_decision(_point("A", "OUT", 1.0), 1.0)
        bridge.engine.undo()
        bridge.tick(2.0)
        self.assertEqual(bridge.manual_override_count, 1)
        self.assertIn("manual_override", [e.get("action") for e in bridge.logs])

    def test_score_server_state_matches_engine(self):
        try:
            from tt_referee.ui.server import ScoreServer
        except Exception as exc:
            self.skipTest(f"UI 依赖不可用：{exc}")
        bridge = self._bridge()
        bridge.apply_decision(_point("A", "OUT", 1.0), 1.0)
        server = ScoreServer(bridge.engine, port=0, allow_demo=False)
        server.start(open_browser=False)
        try:
            with urllib.request.urlopen(server.url + "api/state", timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload["points"], bridge.engine.points)
        finally:
            server.stop()


@unittest.skipUnless(HAS_VOICE and HAS_RUNS, "缺语音仓库或 7 段 rule 输出")
class TestScoreboardReplay(unittest.TestCase):
    def test_seven_rule_runs_match_summary_physical_points(self):
        for n in range(1, 8):
            with self.subTest(video=f"test_{n}"):
                run_dir = os.path.join(RULE_RUNS, f"test_{n}_rule")
                with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
                    expected = json.load(f)["score"]
                with tempfile.TemporaryDirectory() as tmp:
                    result = run_scoreboard(
                        run_dir, speed=1e8, ui=False, voice="none", say=False,
                        dead_ball_mode="ignore", out_dir=tmp, quiet=True)
                    self.assertEqual(result["physical_points"], expected)
                    self.assertTrue(os.path.isfile(os.path.join(tmp, "scoreboard_log.jsonl")))
                    self.assertTrue(os.path.isfile(os.path.join(tmp, "scoreboard_final.json")))


class TestRunScoreboardCLI(unittest.TestCase):
    def test_parser_flags(self):
        a = _build_parser().parse_args([
            "--run-dir", "run", "--video", "video.mp4", "--speed", "2",
            "--max-frames", "100", "--ui", "--ui-port", "0",
            "--ui-window", "800x480", "--open-browser", "--voice", "mic",
            "--no-say", "--dead-ball-mode", "ignore", "--pending-timeout-s", "2.5",
            "--input-device", "3"])
        self.assertEqual(a.run_dir, "run")
        self.assertEqual(a.speed, 2.0)
        self.assertTrue(a.ui)
        self.assertEqual(a.ui_port, 0)
        self.assertTrue(a.open_browser)
        self.assertEqual(a.voice, "mic")
        self.assertFalse(a.say)
        self.assertEqual(a.dead_ball_mode, "ignore")
        self.assertEqual(a.pending_timeout_s, 2.5)
        self.assertEqual(a.input_device, 3)
if __name__ == "__main__":
    unittest.main()