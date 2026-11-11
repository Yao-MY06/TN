"""状态机选择：TN、TTNet1.1 与国栋门控版都必须可构造并跑通。缺相邻仓库时跳过。"""
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

from cascade._paths import EXAMPLES_DIR  # noqa: E402
from cascade.committer import CommitConfig, StreamingReferee  # noqa: E402
from cascade.run_cascade import run, write_outputs  # noqa: E402
from cascade.state_machines import (CHOICES, GD_ROOT, TTNET11_ROOT, _load_gd, gd_status,  # noqa: E402
                                      gate_stats, make_state_machine)
from core.schemas import RefereeEvent  # noqa: E402

OBS = os.path.join(EXAMPLES_DIR, "test_2_observations.jsonl")
HAS_11 = os.path.isfile(os.path.join(TTNET11_ROOT, "src", "referee", "yolo_sm", "ttnet_state_machine.py"))
GD_REASON = gd_status()
HAS_GD = GD_REASON is None


class TestDefault(unittest.TestCase):
    def test_default_is_tn(self):
        self.assertEqual(CommitConfig().state_machine, "tn")
        self.assertEqual(type(make_state_machine("tn")).__name__, "TableTennisStateMachine")
        with self.assertRaises(ValueError):
            make_state_machine("nope")


@unittest.skipUnless(HAS_11, "缺 TTNet1.1 仓库：" + TTNET11_ROOT)
class TestTTNet11(unittest.TestCase):
    def test_yolo_profile_equals_tn(self):
        base = run(OBS, preset="test2")["referee"]
        r = run(OBS, preset="test2", commit=CommitConfig(state_machine="ttnet11-yolo"))["referee"]
        self.assertEqual(r.dumped_events, base.dumped_events)
        self.assertEqual(r.decisions, base.decisions)

    def test_ttnet_profile_runs(self):
        r = run(OBS, preset="test2", commit=CommitConfig(state_machine="ttnet11-ttnet"))["referee"]
        self.assertGreater(len(r.decisions), 0)


class TestGdChoices(unittest.TestCase):
    def test_choices_include_gd_variants(self):
        self.assertIn("gd", CHOICES)
        self.assertIn("gd-traj", CHOICES)
        self.assertIn("gd-nr", CHOICES)


@unittest.skipUnless(HAS_GD, GD_REASON or "")
class TestGd(unittest.TestCase):
    def test_gd_constructs_and_exposes_gate_stats(self):
        sm = make_state_machine("gd")
        gates = gate_stats(sm)
        self.assertIsInstance(gates, dict)
        self.assertIn("serve_no_hit_blocked", gates)
        self.assertIsNone(gate_stats(make_state_machine("tn")))
        self.assertFalse(make_state_machine("gd").rules.enable_no_return)
        self.assertTrue(make_state_machine("gd-nr").rules.enable_no_return)

    def test_gd_rule_run_and_outputs(self):
        for kind in ("gd", "gd-traj", "gd-nr"):
            with self.subTest(kind=kind):
                res = run(OBS, preset="test2", commit=CommitConfig(state_machine=kind))
                summary = res["summary"]
                self.assertEqual(summary["state_machine"], kind)
                self.assertIsInstance(summary["sm_gates"], dict)
                self.assertIsInstance(summary["state_machine_event_id_renames"], int)
                self.assertEqual(set(summary["score"]), {"A", "B"})
                with tempfile.TemporaryDirectory() as tmp:
                    write_outputs(tmp, res)
                    with open(os.path.join(tmp, "summary.json"), encoding="utf-8") as f:
                        written = json.load(f)
                    self.assertEqual(written["state_machine"], kind)
                    self.assertIsInstance(written["sm_gates"], dict)


class TestStateMachineEventIds(unittest.TestCase):
    def _feed_duplicate_out(self, kind):
        ref = StreamingReferee(object(), CommitConfig(state_machine=kind, use_out=False))
        seen = []
        original_update = ref.sm.update

        def recording_update(event):
            seen.append(event.event_id)
            return original_update(event)

        ref.sm.update = recording_update
        obs = SimpleNamespace(source="tracknet", frame_id=1)
        for timestamp in (1.0, 2.0):
            ref._emit_event(
                RefereeEvent(event_id="out_00001", timestamp_s=timestamp,
                             type="OUT", side="A"),
                obs, timestamp, set())
        return ref, seen

    def test_duplicate_out_ids_are_unique_only_inside_state_machine(self):
        ref, seen = self._feed_duplicate_out("gd")
        self.assertEqual(seen, ["out_00001", "out_00001~2"])
        self.assertEqual([e["event_id"] for e in ref.dumped_events],
                         ["out_00001", "out_00001"])
        self.assertEqual(ref.sm_event_id_renames, 1)
        self.assertEqual(gate_stats(ref.sm)["duplicate_event_ids_ignored"], 0)

    def test_tn_dumped_event_ids_remain_unchanged(self):
        ref, seen = self._feed_duplicate_out("tn")
        self.assertEqual(seen, ["out_00001", "out_00001~2"])
        self.assertEqual([e["event_id"] for e in ref.dumped_events],
                         ["out_00001", "out_00001"])
        self.assertEqual(ref.sm_event_id_renames, 1)
class TestGdLoaderErrors(unittest.TestCase):
    def test_missing_repository_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "找不到国栋的 state_machine.py"):
                _load_gd("state_machine.py", root=tmp)

    def test_schema_mismatch_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            core = os.path.join(tmp, "core")
            os.makedirs(core)
            with open(os.path.join(core, "state_machine.py"), "w", encoding="utf-8") as f:
                f.write("")
            with open(os.path.join(core, "schemas.py"), "w", encoding="utf-8") as f:
                f.write("# deliberately different\n")
            with self.assertRaisesRegex(RuntimeError, "schemas.py.*不一致"):
                _load_gd("state_machine.py", root=tmp)

    def test_old_snapshot_without_gates_has_clear_error(self):
        import shutil
        import core.schemas as tn_schemas
        with tempfile.TemporaryDirectory() as tmp:
            core = os.path.join(tmp, "core")
            os.makedirs(core)
            shutil.copyfile(tn_schemas.__file__, os.path.join(core, "schemas.py"))
            with open(os.path.join(core, "state_machine.py"), "w", encoding="utf-8") as f:
                f.write("class TableTennisStateMachine:\n    pass\n")
            self.assertIn("版本过旧", gd_status(tmp))
            with self.assertRaisesRegex(RuntimeError, "版本过旧"):
                _load_gd("state_machine.py", root=tmp)
if __name__ == "__main__":
    unittest.main()
