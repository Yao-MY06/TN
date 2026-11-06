from __future__ import annotations

import json
import os
import tempfile
import unittest

from cascade.eval_ttnet_outside import evaluate


class TestTTNetOutside(unittest.TestCase):
    def test_outside_humps_and_truth_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "ttnet.jsonl")
            clips = os.path.join(tmp, "clips.jsonl")
            markup = os.path.join(tmp, "events_markup.json")
            rows = []
            scores = {100: 0.9, 200: 0.8, 300: 0.9}
            for event_frame in range(0, 351):
                rows.append({"window_end": event_frame + 4,
                             "frame_id": event_frame + 4,
                             "bounce": scores.get(event_frame, 0.0), "net": 0.0})
            with open(cache, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            with open(clips, "w", encoding="utf-8") as f:
                f.write(json.dumps({"clip_id": "c1", "start_frame_id": 195,
                                    "end_frame_id": 205}) + "\n")
            with open(markup, "w", encoding="utf-8") as f:
                json.dump({"100": "bounce", "250": "bounce", "300": "bounce"}, f)

            result = evaluate(cache, clips, markup, k=4, fps=120.0)
            self.assertEqual(result["humps_total"], 3)
            self.assertEqual(result["humps_outside"], 2)
            self.assertEqual(result["outside_truth_matches"], 2)
            self.assertEqual([row["peak_frame"] for row in result["outside_rows"]], [100, 300])
            self.assertEqual([row["truth_matched"] for row in result["outside_rows"]],
                             [True, True])

    def test_threshold_reuses_reviewer_hysteresis(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "ttnet.jsonl")
            clips = os.path.join(tmp, "clips.jsonl")
            markup = os.path.join(tmp, "events_markup.json")
            with open(cache, "w", encoding="utf-8") as f:
                for e in range(20):
                    f.write(json.dumps({"window_end": e + 4, "bounce": 0.64}) + "\n")
            with open(clips, "w", encoding="utf-8") as f:
                f.write("")
            with open(markup, "w", encoding="utf-8") as f:
                json.dump({}, f)
            result = evaluate(cache, clips, markup, k=4)
            self.assertEqual(result["humps_total"], 0)


if __name__ == "__main__":
    unittest.main()
