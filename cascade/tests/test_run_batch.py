from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest

from cascade.run_batch import _build_parser, aggregate_holdout, run_batch

TN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OBS = os.path.join(TN_ROOT, "adapter", "examples", "test_2_observations.jsonl")
MARKUP = os.path.join(TN_ROOT, "_data", "gt", "extended_openttgames", "inferred",
                      "test_2_events_markup_inferred.json")
EXTENDED = os.path.join(TN_ROOT, "_data", "gt", "extended_openttgames", "game_data",
                        "test", "test_2.json")


def _stats(tp, fp, fn, points_correct=1, points_total=1, clips=10,
           manual=1, timeout=0, gt_bounce=10):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "p": round(p, 4), "r": round(r, 4),
        "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        "points_correct": points_correct, "points_total": points_total,
        "spurious_points": 0, "manual_confirm_count": manual,
        "timeout_count": timeout, "clips": clips,
        "ttnet_outside_truth_misses": 0,
        "_gt_bounce": gt_bounce, "_proposal_hits": gt_bounce,
    }


class TestAggregate(unittest.TestCase):
    def test_holdout_f1_is_pooled_not_averaged(self):
        rows = [
            {"video": "a", "split": "holdout", "frames": 100, "coverage": 0.2,
             "ttnet_window_ratio": 0.3, "proposal_recall": 1.0,
             "rule": _stats(8, 2, 2), "ttnet": None, "notes": []},
            {"video": "b", "split": "holdout", "frames": 300, "coverage": 0.4,
             "ttnet_window_ratio": 0.5, "proposal_recall": 1.0,
             "rule": _stats(1, 9, 9), "ttnet": None, "notes": []},
            {"video": "test_2", "split": "tuning", "frames": 999, "coverage": 0.9,
             "ttnet_window_ratio": 0.9, "proposal_recall": 1.0,
             "rule": _stats(99, 0, 0), "ttnet": None, "notes": []},
        ]
        total = aggregate_holdout(rows)
        self.assertEqual(total["frames"], 400)
        self.assertEqual(total["rule"]["tp"], 9)
        self.assertEqual(total["rule"]["fp"], 11)
        self.assertEqual(total["rule"]["fn"], 11)
        self.assertEqual(total["rule"]["f1"], 0.45)   # 不是 (0.8 + 0.1) / 2
        self.assertAlmostEqual(total["coverage"], 0.35)
        self.assertAlmostEqual(total["ttnet_window_ratio"], 0.45)


@unittest.skipUnless(os.path.isfile(MARKUP) and os.path.isfile(EXTENDED), "缺推断真值或 Extended 标注")
class TestRunBatchTest2(unittest.TestCase):
    def test_rule_metrics_and_summary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = os.path.join(tmp, "manifest.json")
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump({"videos": [{
                    "name": "test_2", "split": "tuning", "preset": "test2",
                    "tracknet": OBS, "events_markup": MARKUP, "extended": EXTENDED,
                }]}, f)
            out = os.path.join(tmp, "out")
            result = run_batch(manifest, out)
            row = result["rows"][0]
            self.assertIsNone(row["ttnet"])
            self.assertIn("缺 TTNet 缓存", row["notes"][0])
            self.assertEqual(row["proposal_recall"], 1.0)
            self.assertEqual((row["rule"]["p"], row["rule"]["r"], row["rule"]["f1"]),
                             (0.8276, 0.8571, 0.8421))
            self.assertEqual((row["rule"]["points_correct"], row["rule"]["points_total"]), (2, 2))
            self.assertTrue(os.path.isfile(result["summary_md"]))
            self.assertTrue(os.path.isfile(result["summary_csv"]))
            with open(result["summary_csv"], encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertIn("ttnet_outside_truth_misses", rows[0])
            self.assertEqual(rows[0]["rule_f1"], "0.842")

    def test_manifest_state_machine_overrides_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = os.path.join(tmp, "manifest.json")
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump({"videos": [{
                    "name": "test_2", "split": "tuning", "preset": "test2",
                    "tracknet": OBS, "events_markup": MARKUP, "extended": EXTENDED,
                    "state_machine": "gd",
                }]}, f)
            out = os.path.join(tmp, "out")
            result = run_batch(manifest, out, state_machine="tn")
            with open(os.path.join(out, "test_2_rule", "summary.json"), encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary["commit"]["state_machine"], "gd")
            self.assertEqual(summary["state_machine"], "gd")
            self.assertIsInstance(summary["sm_gates"], dict)
            with open(result["summary_md"], encoding="utf-8") as f:
                md = f.read()
            self.assertIn("状态机: `gd`", md)

    def test_ttnet_path_and_outside_column(self):
        with open(MARKUP, encoding="utf-8") as f:
            markup = {int(k): v for k, v in json.load(f).items()}
        gt = sorted(f for f, v in markup.items() if v == "bounce")
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "ttnet.jsonl")
            scores = {}
            for g in gt:
                for d, value in ((-2, 0.7), (-1, 0.85), (0, 0.95), (1, 0.85), (2, 0.7)):
                    scores[g + d] = value
            with open(cache, "w", encoding="utf-8") as f:
                for e in range(8, 3700):
                    score = scores.get(e - 4, 0.0)
                    f.write(json.dumps({"window_end": e, "frame_id": e,
                                        "bounce": score, "net": 0.0, "infer_ms": 1.0}) + "\n")
            manifest = os.path.join(tmp, "manifest.json")
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump({"videos": [{
                    "name": "test_2", "split": "tuning", "preset": "test2",
                    "tracknet": OBS, "events_markup": MARKUP, "extended": EXTENDED,
                    "ttnet_cache": cache, "ttnet_k": 4,
                }]}, f)
            result = run_batch(manifest, os.path.join(tmp, "out"))
            row = result["rows"][0]
            self.assertIsNotNone(row["ttnet"])
            self.assertEqual(row["ttnet"]["ttnet_outside_truth_misses"], 0)
            self.assertTrue(os.path.isfile(os.path.join(
                tmp, "out", "test_2_ttnet", "eval_ttnet_outside.json")))


class TestRunBatchCLI(unittest.TestCase):
    def test_state_machine_defaults_to_tn_and_accepts_gd(self):
        parser = _build_parser()
        self.assertEqual(parser.parse_args(["--manifest", "x"]).state_machine, "tn")
        self.assertEqual(parser.parse_args(
            ["--manifest", "x", "--state-machine", "gd-nr"]).state_machine, "gd-nr")


class TestManifestExample(unittest.TestCase):
    def test_example_manifest_structure(self):
        import json as _json
        path = os.path.join(TN_ROOT, "cascade", "batch_manifest.example.json")
        with open(path, encoding="utf-8") as f:
            doc = _json.load(f)
        vids = {v["name"]: v for v in doc["videos"]}
        self.assertEqual(sorted(vids), [f"test_{n}" for n in range(1, 8)])
        self.assertEqual(vids["test_2"]["split"], "tuning")
        self.assertTrue(all(v["split"] == "holdout" for k, v in vids.items() if k != "test_2"))
        self.assertEqual(vids["test_4"]["quad"], "test_4")
        for v in vids.values():
            for key in ("tracknet", "events_markup", "quad"):
                self.assertIn(key, v)


if __name__ == "__main__":
    unittest.main()
