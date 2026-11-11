from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
TN_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if TN_ROOT not in sys.path:
    sys.path.insert(0, TN_ROOT)

from cascade.committer import CommitConfig, StreamingReferee  # noqa: E402
from cascade.run_cascade import run  # noqa: E402
from core.schemas import RefereeEvent  # noqa: E402

OBS = os.path.join(TN_ROOT, "adapter", "examples", "test_2_observations.jsonl")
BASE = os.path.join(TN_ROOT, "outputs", "cascade_batch_sm", "tn", "test_2_rule")


class _FakeProvider:
    missing = False
    static_filter = False

    def __init__(self, labels=None, decision_delay=0.0):
        self.labels = labels or {}
        self.decision_delay = decision_delay
        self.calls = 0

    def describe(self):
        return {"fake": True}

    def decision_time(self, hit_t):
        return float(hit_t) + self.decision_delay

    def has_same_side_candidates(self, side, hit_t):
        return True

    def classify(self, side, hit_t, position, ball_at_frame=None):
        self.calls += 1
        label = self.labels.get(side, "NEAR")
        return {"label": label, "distance_ratio": 0.1, "window_s": [hit_t - 0.2, hit_t + 0.1],
                "decision_time_s": self.decision_time(hit_t), "racket_frame_id": 1,
                "racket_side": side, "racket_confidence": 0.9, "missing_cache": False}


def _hit(event_id, side, t=1.0):
    return RefereeEvent(event_id=event_id, timestamp_s=t, type="RACKET_HIT", side=side,
                        position_px=(100, 100), confidence=0.8)


def _obs(t):
    return SimpleNamespace(source="tracknet", frame_id=1, timestamp_s=t)


class TestRacketHold(unittest.TestCase):
    def _ref(self, mode, provider):
        ref = StreamingReferee(object(), CommitConfig(
            use_out=False, state_machine="tn", racket_hit_hold=mode))
        ref.racket_provider = provider
        ref.source = SimpleNamespace(out_lockout_until=0.0, lockout_until=0.0)
        ref._now_t = 1.0
        return ref

    def test_shadow_preserves_event_sequence_and_adds_label(self):
        events = [_hit("hit_1", "A"), RefereeEvent(
            event_id="b_1", timestamp_s=1.0, type="BOUNCE", side="B", position_px=(200, 200))]
        base = self._ref("off", None)
        base._process_event_batch(events, _obs(1.0), 1.0, set())
        shadow = self._ref("shadow", _FakeProvider({"A": "FAR"}))
        shadow._process_event_batch(events, _obs(1.0), 1.0, set())
        self.assertEqual([e["type"] for e in shadow.dumped_events],
                         [e["type"] for e in base.dumped_events])
        self.assertEqual([e["event_id"] for e in shadow.dumped_events],
                         [e["event_id"] for e in base.dumped_events])
        self.assertEqual(shadow.dumped_events[0]["racket_evidence"]["label"], "FAR")

    def test_drop_removes_only_far_hit(self):
        events = [_hit("far", "A", 1.0), _hit("near", "B", 1.0)]
        ref = self._ref("drop", _FakeProvider({"A": "FAR", "B": "NEAR"}))
        ref._process_event_batch(events, _obs(1.0), 1.0, set())
        ids = [e["event_id"] for e in ref.dumped_events if e["type"] == "RACKET_HIT"]
        self.assertEqual(ids, ["near"])
        summary = ref.racket_summary()
        self.assertEqual(summary["classes"]["FAR"], 1)
        self.assertEqual(summary["classes"]["NEAR"], 1)
        self.assertEqual(summary["dropped"], 1)

    def test_hold_blocks_watermark_until_window_end(self):
        provider = _FakeProvider({"A": "NEAR"}, decision_delay=0.1)
        ref = self._ref("drop", provider)
        hit = _hit("h", "A", 1.0)
        ref._process_event_batch([hit], _obs(1.0), 1.0, set())
        self.assertEqual(ref.held_hit_watermark(), 1.0)
        self.assertFalse(ref.dumped_events)
        ref._now_t = 1.1
        self.assertTrue(ref._try_resolve_racket_held())
        self.assertEqual([e["event_id"] for e in ref.dumped_events], ["h"])

    def test_missing_cache_is_none_and_does_not_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing.jsonl")
            ref = StreamingReferee(object(), CommitConfig(
                use_out=False, racket_cache=missing, racket_hit_hold="drop"))
            ref._now_t = 1.0
            ref._process_event_batch([_hit("h", "A", 1.0)], _obs(1.0), 1.0, set())
            self.assertEqual([e["event_id"] for e in ref.dumped_events], ["h"])
            self.assertEqual(ref.racket_summary()["classes"]["NONE"], 1)


class TestRacketDefaultRegression(unittest.TestCase):
    def test_default_off_adds_no_racket_fields(self):
        res = run(OBS, preset="test2", commit=CommitConfig())
        self.assertIsNone(res["summary"]["racket_evidence"])
        self.assertTrue(all("racket_evidence" not in e for e in res["referee"].dumped_events))
        self.assertFalse(any(q.get("reviewer") == "racket"
                             for q in res["review_queue"]))


if __name__ == "__main__":
    unittest.main()