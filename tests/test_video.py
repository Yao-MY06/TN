"""视频入口的测试：抽帧、定位核对、置信度不被挑帧抬高、批量与打包。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from tcalib import synth  # noqa: E402
from tcalib.video import evenly_spaced_indices, open_video, sample_ids, sample_video_frames, sharpness  # noqa: E402


def _write_video(path, frames, fps=25.0):
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not wr.isOpened():
        return False
    for f in frames:
        wr.write(f)
    wr.release()
    return True


class TestIndices(unittest.TestCase):
    def test_evenly_spaced(self):
        self.assertEqual(evenly_spaced_indices(0, 100, 5), [0, 25, 50, 74, 99])
        self.assertEqual(evenly_spaced_indices(10, 12, 1), [11])
        self.assertEqual(evenly_spaced_indices(10, 10, 5), [])

    def test_sharpness(self):
        rng = np.random.default_rng(0)
        sharp = rng.integers(0, 255, (80, 120, 3), dtype=np.uint8)
        self.assertGreater(sharpness(sharp), sharpness(cv2.GaussianBlur(sharp, (15, 15), 0)))


class TestSampling(unittest.TestCase):
    def test_modes_and_frame_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.avi")
            frames = []
            for i in range(100):
                img = np.zeros((64, 96, 3), np.uint8)
                img[:, :, 2] = i  # 帧号写进像素，用来核对读到的是不是目标帧
                frames.append(img)
            if not _write_video(path, frames, fps=10.0):
                self.skipTest("cv2 VideoWriter(MJPG) 不可用")
            cap, info = open_video(path)
            cap.release()
            self.assertEqual(sample_ids(info, "spread", 5), [2, 26, 50, 73, 97])
            self.assertEqual(sample_ids(info, "opening", 5, start_s=1.0, window_s=2.0), evenly_spaced_indices(10, 30, 5))
            got, sampled, info, extras = sample_video_frames(path, mode="spread", count=5, extra_ids=[42])
            self.assertEqual(list(got), [f"f{i:06d}" for i in (2, 26, 50, 73, 97)])
            for x in sampled:
                self.assertLessEqual(abs(int(np.median(x.image[:, :, 2])) - x.frame_id), 2)
            self.assertIn(42, extras)
            self.assertEqual(info.unreadable, [])


class TestVideoCalibration(unittest.TestCase):
    """端到端：一半以上的帧被严重遮挡。所有抽到的帧都要计入统计，四角仍应准确。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cam = synth.make_camera(f=1800, cam_xyh=(120, 560, 160))
        frames = []
        for s in range(24):
            img, gt = synth.render(cam, seed=1000 + s, n_occluders=8 if s < 18 else 1)
            frames.append(img)
        cls.gt = gt
        cls.video = os.path.join(cls.tmp.name, "hard.avi")
        cls.ok = _write_video(cls.video, frames)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        if not self.ok:
            self.skipTest("cv2 VideoWriter(MJPG) 不可用")

    def test_video_end_to_end(self):
        from calibrate_video import run_video

        out = os.path.join(self.tmp.name, "out", "hard")
        d = run_video(self.video, "hard", out, mode="spread", count=24, save_frame_ids=[5])
        self.assertEqual(d["n_frames"], len(d["sampling"]["frames"]))  # 全部抽到的帧都计入
        self.assertGreaterEqual(d["n_frames"], 20)
        self.assertLess(len(d["per_frame_ok"]), d["n_frames"])  # 确有单帧失败
        self.assertLess(np.linalg.norm(np.array(d["points"]) - self.gt, axis=1).max(), 1.5)
        for fn in ("result.json", "frames.csv", "overlay.jpg", "corners.png", os.path.join("frames", "f000005.png")):
            self.assertTrue(os.path.exists(os.path.join(out, fn)), fn)

    def test_batch_and_zip(self):
        lst = os.path.join(self.tmp.name, "videos.txt")
        with open(lst, "w", encoding="utf-8") as f:
            f.write(f"# 注释\nhard  {self.video}\nmissing  {os.path.join(self.tmp.name, 'nope.mp4')}\n")
        truth = os.path.join(self.tmp.name, "truth.json")
        with open(truth, "w", encoding="utf-8") as f:
            json.dump({"_comment": "x", "hard": {"points": self.gt.tolist(), "method": "synthetic"}}, f)
        root = os.path.join(self.tmp.name, "bout")
        z = os.path.join(self.tmp.name, "results.zip")
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "batch_videos.py"), "--videos", lst,
                            "--out-root", root, "--count", "24", "--save-frame", "hard:3", "--truth", truth,
                            "--zip", z], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(root, "batch_summary.json"), encoding="utf-8") as f:
            s = json.load(f)
        self.assertTrue(s["videos"]["hard"]["ok"])
        self.assertFalse(s["videos"]["missing"]["ok"])
        names = zipfile.ZipFile(z).namelist()
        self.assertIn("bout/hard/result.json", names)
        self.assertIn("bout/eval.md", names)
        self.assertIn("bout/hard/frames/f000003.png", names)


if __name__ == "__main__":
    unittest.main()
