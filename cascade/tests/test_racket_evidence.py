from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TN_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if TN_ROOT not in sys.path:
    sys.path.insert(0, TN_ROOT)

from cascade.racket_evidence import RacketEvidence, filter_static_observations  # noqa: E402


def _obs(frame_id, t, side, center, confidence=0.8, bbox=None, visible=True):
    cx, cy = center
    if bbox is None:
        bbox = [cx - 10, cy - 10, cx + 10, cy + 10]
    return {
        "schema_version": "racket.v1", "frame_id": frame_id, "timestamp_s": t,
        "visible": visible, "side": side, "bbox_xyxy": bbox,
        "center_px": list(center), "confidence": confidence, "track_id": f"r_{frame_id}",
    }


def _write_cache(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestRacketEvidence(unittest.TestCase):
    def _provider(self, rows, tmp, **kwargs):
        path = os.path.join(tmp, "racket.jsonl")
        _write_cache(path, rows)
        return RacketEvidence.from_cache(path, **kwargs)

    def test_four_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _obs(1, 0.8, "A", (100, 100)),
                _obs(2, 1.8, "A", (520, 520)),
                _obs(3, 2.8, "B", (100, 100)),
            ]
            p = self._provider(rows, tmp, dt_s=0.15, offset_s=-0.15, conf=0.5,
                               distance_ratio=3.0)
            near = p.classify("A", 1.0, (100, 100), lambda fid: (100, 100))
            far = p.classify("A", 2.0, (100, 100), lambda fid: (100, 100))
            other = p.classify("A", 3.0, (100, 100), lambda fid: (100, 100))
            none = p.classify("A", 4.0, (100, 100), lambda fid: (100, 100))
            self.assertEqual(near["label"], "NEAR")
            self.assertEqual(far["label"], "FAR")
            self.assertEqual(none["label"], "NONE")
            self.assertEqual(other["label"], "OTHER_SIDE")
            self.assertLess(near["distance_ratio"], far["distance_ratio"])

    def test_missing_cache_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = RacketEvidence.from_cache(os.path.join(tmp, "missing.jsonl"))
            out = p.classify("A", 1.0, (100, 100), lambda fid: (100, 100))
            self.assertEqual(out["label"], "NONE")
            self.assertTrue(p.missing)

    def test_static_filter_marks_long_stationary_object(self):
        rows = []
        for i in range(30):
            rows.append(_obs(i, i / 10.0, "UNKNOWN", (906 + i % 2, 378)))
        for i in range(10):
            rows.append(_obs(100 + i, 4.0 + i / 10.0, "A", (200 + i * 30, 300)))
        by_frame = {}
        for o in rows:
            by_frame.setdefault(o["frame_id"], []).append(o)
        stats = filter_static_observations(
            by_frame, min_static_s=2.0, max_gap_s=0.2, center_bin_px=32, size_bin_px=8)
        self.assertGreaterEqual(stats["filtered"], 30)
        self.assertTrue(all(o.get("_static_filtered") for o in by_frame[0]))
        self.assertFalse(all(o.get("_static_filtered") for o in by_frame[109]))


if __name__ == "__main__":
    unittest.main()