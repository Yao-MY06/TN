from __future__ import annotations

import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tcalib.video import evenly_spaced_indices, sample_video_frames, sharpness  # noqa: E402


class TestIndices(unittest.TestCase):
    def test_evenly_spaced(self):
        self.assertEqual(evenly_spaced_indices(0, 100, 5), [0, 25, 50, 74, 99])
        self.assertEqual(evenly_spaced_indices(10, 12, 1), [11])
        self.assertEqual(evenly_spaced_indices(10, 10, 5), [])


class TestSharpness(unittest.TestCase):
    def test_blur_is_lower(self):
        rng = np.random.default_rng(0)
        sharp = rng.integers(0, 255, (80, 120, 3), dtype=np.uint8)
        blur = cv2.GaussianBlur(sharp, (15, 15), 0)
        self.assertGreater(sharpness(sharp), sharpness(blur))


class TestSampleVideo(unittest.TestCase):
    def test_sample_from_synthetic_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64))
            if not writer.isOpened():
                self.skipTest("cv2 VideoWriter(MJPG) 不可用")
            for i in range(40):
                img = np.zeros((64, 96, 3), dtype=np.uint8)
                cv2.rectangle(img, (10 + i, 15), (45 + i, 50), (255, 255, 255), -1)
                writer.write(img)
            writer.release()
            frames, sampled = sample_video_frames(
                path, count=4, start_s=0.0, window_s=4.0, candidate_factor=3)
            self.assertEqual(len(frames), 4)
            self.assertEqual(len(sampled), 4)
            self.assertEqual(list(frames), [f"f{x.frame_id:06d}" for x in sampled])


if __name__ == "__main__":
    unittest.main()