"""状态机选择：TTNet1.1 的 Profile.YOLO 必须与 TN 共用状态机逐条一致。缺 TTNet1.1 仓库时跳过。"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TN_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if TN_ROOT not in sys.path:
    sys.path.insert(0, TN_ROOT)

from cascade._paths import EXAMPLES_DIR  # noqa: E402
from cascade.committer import CommitConfig  # noqa: E402
from cascade.run_cascade import run  # noqa: E402
from cascade.state_machines import TTNET11_ROOT, make_state_machine  # noqa: E402

OBS = os.path.join(EXAMPLES_DIR, "test_2_observations.jsonl")
HAS_11 = os.path.isfile(os.path.join(TTNET11_ROOT, "src", "referee", "yolo_sm", "ttnet_state_machine.py"))


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


if __name__ == "__main__":
    unittest.main()
