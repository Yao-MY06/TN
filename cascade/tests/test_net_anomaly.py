from __future__ import annotations

import os
import unittest
from collections import deque
from types import SimpleNamespace

from cascade._paths import load_detect_bounce
from cascade.committer import CommitConfig, StreamingReferee
from cascade.proposals import ProposalGenerator, ProposalParams
from cascade.run_cascade import CascadeCore, run
from cascade.frame_index import FrameIndex
from core.schemas import FrameObservation, RefereeEvent

import run_mixed as rm
from observation import make_observation

TN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T3 = os.path.join(TN_ROOT, "_data", "obs", "tracknet", "test_3.jsonl")
T4 = os.path.join(TN_ROOT, "_data", "obs", "tracknet", "test_4.jsonl")


def _calib():
    return rm.build_calib("test4")


def _poly():
    return load_detect_bounce().load_quad("test_4")


def _frame(fid, t, x, y):
    d = make_observation(fid, t, "tracknet", (1920, 1080), center_px=(x, y), confidence=0.9)
    return FrameObservation.from_dict(d)


def _hit(t, side="A", x=None, y=600.0):
    c = _calib()
    return RefereeEvent(
        event_id=f"hit_{t:.3f}", timestamp_s=t, type="RACKET_HIT", side=side,
        position_px=(c.net_x if x is None else x, y), confidence=0.9, sources=["tracknet"])


class TestNetTrigger(unittest.TestCase):
    def _core(self, enabled=True):
        params = ProposalParams(net_trigger=enabled)
        core = CascadeCore(_poly(), _calib(), "rule", params, CommitConfig(),
                           merge=True, max_clip_s=1.0, pre_ctx=0, post_ctx=0, fps=120.0)
        for fid in range(0, 150):
            core.index.add(fid, fid / 120.0)
        core.generator._now_frame = 120
        return core

    def test_net_touch_creates_fixed_interval(self):
        core = self._core(True)
        core._on_referee_event(SimpleNamespace(type="NET_TOUCH", timestamp_s=1.0))
        self.assertEqual(len(core.proposals), 1)
        p = core.proposals[0]
        self.assertEqual(p.reasons, ["net_anomaly"])
        self.assertEqual(p.candidate_types, ["NET", "HIT", "BOUNCE", "UNKNOWN"])
        self.assertAlmostEqual(p.start_timestamp_s, 0.9)
        self.assertAlmostEqual(p.end_timestamp_s, 1.3)
        self.assertEqual((p.anchor_frame_id, p.anchor_timestamp_s), (120, 1.0))

    def test_trigger_is_default_off_and_ignores_non_net_touch(self):
        core = self._core(False)
        core._on_referee_event(SimpleNamespace(type="NET_TOUCH", timestamp_s=1.0))
        core._on_referee_event(SimpleNamespace(type="RACKET_HIT", timestamp_s=1.0))
        self.assertEqual(core.proposals, [])
        self.assertEqual(ProposalParams().net_trigger, False)


class TestNetHitHold(unittest.TestCase):
    def setUp(self):
        self.calib = _calib()
        self.ref = StreamingReferee(self.calib, CommitConfig(net_hit_hold=True))
        self.ref.event_callback = lambda ev: None  # 模拟 trigger 已开启
        self.ref._last_net_touch_t = 1.0

    def test_hold_boundaries_are_fixed(self):
        self.assertTrue(self.ref._is_hold_candidate(_hit(1.0)))
        self.assertTrue(self.ref._is_hold_candidate(_hit(1.15)))
        self.assertFalse(self.ref._is_hold_candidate(_hit(1.151)))
        self.assertFalse(self.ref._is_hold_candidate(_hit(0.99)))
        self.assertFalse(self.ref._is_hold_candidate(_hit(1.0, x=self.calib.net_x + 30.1)))
        self.ref.cfg = CommitConfig(net_hit_hold=False)
        self.assertFalse(self.ref._is_hold_candidate(_hit(1.0)))

    def _held(self, hit):
        return {"events": [hit], "obs": _frame(120, hit.timestamp_s, *hit.position_px),
                "t": hit.timestamp_s, "injected_ids": set()}

    def _set_trace(self, hit, points):
        self.ref._now_t = 1.3
        self.ref._vqueue = deque((_frame(121 + i, t, x, y), t)
                                 for i, (t, x, y) in enumerate(points))

    def test_true_hit_is_released(self):
        hit = _hit(1.0, side="A")
        self._set_trace(hit, [(1.0, self.calib.net_x, 600), (1.15, self.calib.net_x + 30, 590),
                              (1.3, self.calib.net_x + 80, 540)])
        decision, reason = self.ref._adjudicate_held(self._held(hit))
        self.assertEqual(decision, "released")
        self.assertIn("rises", reason)

    def test_net_reversal_is_rejected(self):
        hit = _hit(1.0, side="A")
        self._set_trace(hit, [(1.0, self.calib.net_x + 5, 600), (1.15, self.calib.net_x + 10, 620),
                              (1.3, self.calib.net_x + 25, 650)])
        decision, reason = self.ref._adjudicate_held(self._held(hit))
        self.assertEqual(decision, "reverse")
        self.assertIn("no_escape", reason)

    def test_incomplete_window_becomes_uncertain_on_finish(self):
        hit = _hit(1.0, side="A")
        self.ref._now_t = 1.2
        self.ref._vqueue = deque([(_frame(121, 1.1, self.calib.net_x, 600), 1.1)])
        decision, reason = self.ref._adjudicate_held(self._held(hit), force=True)
        self.assertEqual(decision, "uncertain")
        self.assertIn("incomplete", reason)


@unittest.skipUnless(os.path.isfile(T3) and os.path.isfile(T4), "缺 test_3/test_4 观测")
class TestNetHoldIntegration(unittest.TestCase):
    def test_test3_trigger_and_true_hits_release(self):
        res = run(T3, preset="test_3", quad="test_3", reviewer="rule", net_hit_hold=True)
        s = res["summary"]
        self.assertTrue(s["commit"]["net_hit_hold"])
        self.assertTrue(s["params"]["net_trigger"])
        self.assertGreater(s["proposals"]["by_reason"].get("net_anomaly", 0), 0)
        self.assertGreater(s["net_hit_hold"]["released"], 0)
        self.assertEqual(s["score"], {"A": 3, "B": 4})

    def test_test3_trigger_only_does_not_hold_hits(self):
        res = run(T3, preset="test_3", quad="test_3", reviewer="rule", net_trigger=True)
        s = res["summary"]
        self.assertTrue(s["params"]["net_trigger"])
        self.assertFalse(s["commit"]["net_hit_hold"])
        self.assertGreater(s["proposals"]["by_reason"].get("net_anomaly", 0), 0)
        self.assertEqual(s["net_hit_hold"]["held"], 0)

    def test_test4_reversed_hits_enter_net_reversal_queue(self):
        res = run(T4, preset="test4", quad="test_4", reviewer="rule", net_hit_hold=True)
        s = res["summary"]
        self.assertGreater(s["net_hit_hold"]["reversed"], 0)
        rows = [r for r in res["review_queue"] if r["category"] == "net_reversal"]
        self.assertEqual(len(rows), s["net_hit_hold"]["reversed"])
        self.assertTrue(all(r["net_decision"]["sources"][-1] == "suspect_net" for r in rows))
        self.assertEqual(s["score"], {"A": 13, "B": 15})


if __name__ == "__main__":
    unittest.main()
