"""racket_cache：用假检测器和合成小视频跑通（不需要 ultralytics、权重和真视频）。"""
import json
import os
import tempfile
import unittest

from cascade.racket_cache import RACKET_ROOT, load_racket_obs, run

HAS_RACKET = os.path.isdir(os.path.join(RACKET_ROOT, "racket"))


@unittest.skipUnless(HAS_RACKET, "缺球拍仓库：" + RACKET_ROOT)
class TestRacketCache(unittest.TestCase):
    def test_fake_detector_sides_by_net_x(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("需要 opencv-python")
        from cascade.racket_cache import ensure_racket_path
        ensure_racket_path()
        from racket.detectors import StaticDetector
        from racket.tracker import Detection

        with tempfile.TemporaryDirectory() as tmp:
            video = os.path.join(tmp, "v.avi")
            w = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 120.0, (640, 360))
            if not w.isOpened():
                self.skipTest("cv2 VideoWriter(MJPG) 不可用")
            for _ in range(6):
                w.write(np.zeros((360, 640, 3), np.uint8))
            w.release()

            # 网线在 x=400（不是画面正中 320）：x=350 的框应判 A，x=500 判 B
            def script(frame, size):
                return [Detection(bbox=(330, 100, 370, 160), confidence=0.9),
                        Detection(bbox=(480, 100, 520, 160), confidence=0.8)]
            out = os.path.join(tmp, "r.jsonl")
            meta = run(video, out, detector=StaticDetector(script), net_x=400.0,
                       stride=2, quiet=True)
            self.assertEqual(meta["frames_read"], 6)
            self.assertEqual(meta["frames_processed"], 3)
            obs = load_racket_obs(out)
            self.assertEqual(sorted(obs), [0, 2, 4])
            sides = {(o["frame_id"], o["side"]) for v in obs.values() for o in v}
            self.assertIn((0, "A"), sides)
            self.assertIn((0, "B"), sides)
            for v in obs.values():
                for o in v:
                    self.assertEqual(o["schema_version"], "racket.v1")
                    self.assertAlmostEqual(o["timestamp_s"], o["frame_id"] / 120.0, places=5)
            with open(out + ".meta.json", encoding="utf-8") as f:
                self.assertEqual(json.load(f)["net_x"], 400.0)


if __name__ == "__main__":
    unittest.main()
