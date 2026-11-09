"""render_video 与 fused_obs.jsonl 的回归测试（不需要 GPU、权重或真视频）。"""
import json
import os
import shutil
import tempfile
import unittest

from cascade import run_cascade
from cascade.render_video import load_ball_track, render, score_at

TN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OBS = os.path.join(TN_ROOT, "adapter", "examples", "test_2_observations.jsonl")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


class RenderVideoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="render_video_")
        cls.run_dir = os.path.join(cls.tmp, "run")
        res = run_cascade.run(OBS, preset="test2", reviewer="rule")
        run_cascade.write_outputs(cls.run_dir, res)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_fused_obs_written_per_frame(self):
        rows = _read(os.path.join(self.run_dir, "fused_obs.jsonl"))
        src = _read(OBS)
        self.assertEqual(len(rows), len(src))
        self.assertEqual([r["frame_id"] for r in rows], [d["frame_id"] for d in src])
        for r, d in zip(rows, src):
            self.assertEqual(r["visible"], bool(d["ball"]["visible"]))
            self.assertEqual(r["center_px"], d["ball"]["center_px"])

    def test_offline_fallback_when_no_fused_obs(self):
        d = os.path.join(self.tmp, "nofused")
        os.makedirs(d, exist_ok=True)
        track, label = load_ball_track(d, OBS, None)
        self.assertIn("offline", label)
        self.assertEqual(len(track), len(_read(OBS)))
        track, label = load_ball_track(d, None, None)
        self.assertEqual(track, {})
        track, label = load_ball_track(self.run_dir, OBS, None)
        self.assertIn("this run", label)

    def test_score_from_decisions(self):
        decs = [{"t": 1.0, "kind": "POINT", "winner": "B"}, {"t": 2.0, "kind": "DEAD_BALL", "winner": None},
                {"t": 3.0, "kind": "POINT", "winner": "A"}]
        self.assertEqual(score_at(decs, 0.5), (0, 0))
        self.assertEqual(score_at(decs, 2.5), (0, 1))
        self.assertEqual(score_at(decs, 9.0), (1, 1))

    def test_render_small_video(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("需要 opencv-python")
        src = os.path.join(self.tmp, "in.avi")
        w = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"MJPG"), 120.0, (320, 180))
        if not w.isOpened():
            self.skipTest("cv2 VideoWriter(MJPG) 不可用")
        frame = np.full((180, 320, 3), 60, np.uint8)
        for _ in range(48):
            w.write(frame)
        w.release()
        out = os.path.join(self.tmp, "hud.mp4")
        try:
            info = render(self.run_dir, src, out, preset="test2", max_frames=40)
        except SystemExit as e:
            if "无法写视频" in str(e):
                self.skipTest("cv2 VideoWriter(mp4v) 不可用")
            raise
        self.assertEqual(info["frames"], 40)
        cap = cv2.VideoCapture(out)
        n = 0
        while cap.read()[0]:
            n += 1
        cap.release()
        self.assertEqual(n, 40)


if __name__ == "__main__":
    unittest.main()
