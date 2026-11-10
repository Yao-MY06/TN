from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest

from cascade.compare_sm import _classify_gate_cause, build_report, write_report


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row(gt_t, gt, winner, our_t=None, our_winner=None, rule=None, correct=False):
    row = {
        "gt_t": gt_t,
        "gt": gt,
        "gt_winner": winner,
        "our_t": our_t,
        "correct": correct,
    }
    if our_t is None:
        row["note"] = "未判分"
    else:
        row.update({"our_winner": our_winner, "our_rule": rule})
    return row


def _write_case(root, name, state_machine, reviewer, rows, gates=None,
                spurious=0, dead=0, events=None, decisions=None):
    run_dir = os.path.join(root, name)
    _write_json(os.path.join(run_dir, "eval_extended.json"), {
        "points": {
            "rows": rows,
            "spurious_points": [{"t": 70 + i, "winner": "A", "rule": "OUT"}
                                for i in range(spurious)],
            "dead_ball": [{"t": 80 + i, "rule": "NO_EVIDENCE"} for i in range(dead)],
        }
    })
    _write_json(os.path.join(run_dir, "summary.json"), {
        "state_machine": state_machine,
        "sm_gates": gates,
    })
    _write_jsonl(os.path.join(run_dir, "events.jsonl"), events or [])
    _write_jsonl(os.path.join(run_dir, "decisions.jsonl"), decisions or [])


class TestCompareSM(unittest.TestCase):
    def _fixture(self, tmp):
        base = os.path.join(tmp, "tn")
        gd = os.path.join(tmp, "gd")
        traj = os.path.join(tmp, "gd-traj")

        tn_rows = [
            _row(10.0, "left_out", "B", 10.1, "B", "OUT", True),
            _row(20.0, "right_out", "A"),
            _row(30.0, "left_net", "B", 30.2, "A", "OUT", False),
            _row(40.0, "left_out", "B"),
            _row(50.0, "right_out", "A"),
        ]
        gd_rows = [
            _row(10.0, "left_out", "B", 10.1, "A", "NO_RETURN", False),
            _row(20.0, "right_out", "A", 20.3, "A", "OUT", True),
            _row(30.0, "left_net", "B", 30.2, "A", "OUT", False),
            _row(40.0, "left_out", "B"),
            _row(50.0, "right_out", "A"),
        ]
        tuning_rows = [_row(5.0, "left_net", "B", 5.1, "B", "OWN_COURT", True)]

        base_events = [
            {"event_id": "out_1", "timestamp_s": 20.2, "type": "OUT", "side": "B"},
            {"event_id": "out_2", "timestamp_s": 60.0, "type": "OUT", "side": "A"},
        ]
        base_decisions = [
            {"virtual_t": 20.5, "kind": "DEAD_BALL", "winner": None,
             "rule": "NO_EVIDENCE", "text": "死球"},
        ]
        _write_case(base, "test_1_rule", "tn", "rule", tn_rows,
                    events=base_events, decisions=base_decisions, spurious=2, dead=1)
        _write_case(base, "test_1_ttnet", "tn", "ttnet",
                    [_row(10.0, "left_out", "B", 10.1, "B", "OUT", True)])
        _write_case(base, "test_2_rule", "tn", "rule", tuning_rows)
        _write_case(gd, "test_1_rule", "gd", "rule", gd_rows,
                    gates={"serve_no_hit_blocked": 2, "no_hit_skips": 1},
                    events=base_events, decisions=base_decisions, spurious=1, dead=2)
        gd_tuning_rows = [_row(5.0, "left_net", "B", 5.1, "A", "OUT", False)]
        _write_case(gd, "test_2_rule", "gd", "rule", gd_tuning_rows,
                    gates={"serve_no_hit_blocked": 1})
        _write_case(traj, "test_1_rule", "gd-traj", "rule", gd_rows,
                    gates={"splice_skips": 1}, events=base_events,
                    decisions=base_decisions, spurious=0, dead=3)
        _write_case(traj, "test_2_rule", "gd-traj", "rule", tuning_rows,
                    gates={"cooldown_ignored_events": 1})
        return base, [gd, traj]

    def test_report_aggregates_changes_gates_and_out_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, compare = self._fixture(tmp)
            report = build_report(base, compare)
            totals = {(r["state_machine"], r["reviewer"], r["scope"]): r
                      for r in report["totals"]}

            tn = totals[("tn", "rule", "holdout")]
            self.assertEqual(tn["correct"], 1)
            self.assertEqual(tn["missed_out"], 3)
            self.assertEqual(tn["missed_net"], 0)
            self.assertEqual(tn["wrong"], 1)
            self.assertEqual((tn["wrong_a"], tn["wrong_b"]), (1, 0))
            self.assertEqual(tn["spurious"], 2)
            self.assertEqual(tn["dead"], 1)
            self.assertEqual((tn["no_out_event"], tn["out_event_not_scored"]), (2, 1))
            self.assertEqual(totals[("tn", "ttnet", "holdout")]["correct"], 1)

            self.assertEqual(totals[("gd", "rule", "tuning")]["correct"], 0)
            changes = {(c["state_machine"], c["gt_t"]): c for c in report["changes"]}
            self.assertEqual(changes[("gd", 10.0)]["change"], "right_to_wrong")
            self.assertEqual(changes[("gd", 20.0)]["change"], "wrong_to_right")

            gates = {(g["state_machine"], g["video"], g["reviewer"]): g["gates"]
                     for g in report["gates"]}
            self.assertEqual(gates[("gd", "test_1", "rule")]["serve_no_hit_blocked"], 2)

            diag = {(d["state_machine"], d["video"], d["gt_t"]): d
                    for d in report["out_diagnostics"]}
            self.assertEqual(diag[("tn", "test_1", 20.0)]["category"], "OUT_EVENT_NOT_SCORED")
            self.assertEqual(diag[("tn", "test_1", 40.0)]["category"], "NO_OUT_EVENT")
            self.assertTrue(any(d["rule"] == "NO_EVIDENCE"
                                for d in diag[("tn", "test_1", 20.0)]["decisions"]))

    def test_write_report_outputs_markdown_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, compare = self._fixture(tmp)
            out = os.path.join(tmp, "out")
            paths = write_report(base, compare, out)
            self.assertTrue(os.path.isfile(paths["markdown"]))
            self.assertTrue(os.path.isfile(paths["csv"]))
            with open(paths["markdown"], encoding="utf-8") as f:
                md = f.read()
            for text in ("合计表", "test_2", "留出集", "逐分改判", "门因归因表", "门控计数", "出界诊断",
                         "原来对→现在错", "原来错→现在对", "没有 OUT 事件"):
                self.assertIn(text, md)
            holdout_line = next(line for line in md.splitlines()
                                if line.startswith("- `gd` / rule 留出集"))
            self.assertIn("原对→现错 1", holdout_line)
            with open(paths["csv"], encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(rows)
            self.assertIn("state_machine", rows[0])
            self.assertIn("missed_out", rows[0])


class TestGateCause(unittest.TestCase):
    def test_classifier_prioritizes_duplicate_and_defaults(self):
        point = {"kind": "POINT", "rule": "OUT", "winner": "A"}
        rec = {"before_state": "RALLY", "before_ctx": {}, "event_type": "OUT",
               "gate_delta": [("duplicate_event_ids_ignored", 1)]}
        self.assertEqual(_classify_gate_cause(point, rec)[0], "EVENT_ID_DUPLICATE")

        no_return = {"kind": "POINT", "rule": "NO_RETURN", "winner": "A"}
        self.assertEqual(_classify_gate_cause(no_return, None)[0], "NO_RETURN_DISABLED")

        serve = {"before_state": "SERVE_IN_PROGRESS", "before_ctx": {},
                 "event_type": "BOUNCE", "gate_delta": []}
        self.assertEqual(_classify_gate_cause(point, serve)[0], "SERVE_ADMISSION_INCOMPLETE")

        no_hit = {"before_state": "RALLY_END_CANDIDATE",
                  "before_ctx": {"rally_has_hit": False}, "gate_delta": []}
        self.assertEqual(_classify_gate_cause(no_return, no_hit, no_return_enabled=True)[0], "NO_HIT_REQUIRED")
if __name__ == "__main__":
    unittest.main()