# -*- coding: utf-8 -*-
"""需要 TTNet 权重的测试；缺权重（或缺 torch）自动跳过。

权重路径：环境变量 TTNET_WEIGHTS，或默认 <TN>/_data/weights/ttnet.pth。
torch 引擎还需要 TTNET_SRC 指向 TTNet/src。

- replay vs live：cv2 合成一小段视频（暗背景上移动的白点），先用 ttnet_cache
  生成缓存，再用 LiveScores 从同一视频现算，逐窗口原始分数差 <= 1e-3。
  有 torch 用 torch 引擎，没有就用 numpy 引擎。
- torch vs numpy：两引擎在同一输入上的输出差（只在有 torch 时跑）。
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS = os.environ.get('TTNET_WEIGHTS', os.path.join(_ROOT, '_data', 'weights', 'ttnet.pth'))
from cascade._paths import TTNET_ROOT  # noqa: E402
TTNET_SRC = os.environ.get('TTNET_SRC', os.path.join(TTNET_ROOT, 'src'))
N_FRAMES = int(os.environ.get('TTNET_TEST_FRAMES', '20'))

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

HAS_WEIGHTS = os.path.isfile(WEIGHTS)


def synth_video(path, n=N_FRAMES, w=1920, h=1080, fps=120.0):
    import cv2
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    if not vw.isOpened():
        raise unittest.SkipTest('cv2.VideoWriter(MJPG) 不可用')
    for i in range(n):
        img = np.full((h, w, 3), 20, np.uint8)
        x = int(300 + i * 40)
        y = int(500 + 120 * np.sin(i / 4.0))
        cv2.circle(img, (x, y), 10, (255, 255, 255), -1)
        vw.write(img)
    vw.release()


def _engine():
    from cascade.ttnet_engine import make_engine
    if HAS_TORCH:
        dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        return make_engine('torch', WEIGHTS, TTNET_SRC, dev)
    return make_engine('numpy', WEIGHTS)


@unittest.skipUnless(HAS_WEIGHTS, 'TTNet 权重不存在：{}'.format(WEIGHTS))
class TestReplayVsLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.video = os.path.join(cls.tmp, 'synth.avi')
        synth_video(cls.video)
        cls.engine = _engine()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_replay_equals_live(self):
        from cascade.ttnet_cache import build_cache, VideoFrameSource
        from cascade.ttnet_reviewer import ReplayScores, LiveScores, TTNetReviewer
        out = os.path.join(self.tmp, 'cache.jsonl')
        meta = build_cache(self.engine, self.video, out, progress_every=0)
        self.assertEqual(meta['n_windows'], N_FRAMES - 8)
        replay = ReplayScores(out)
        src = VideoFrameSource(self.video)
        live = LiveScores(self.engine, src, src.timestamp_of)
        ends = sorted(replay.rows)
        self.assertEqual(ends[0], 8)
        worst = 0.0
        for e in ends:
            a, b = replay.get(e), live.get(e)
            for key in ('bounce', 'net'):
                worst = max(worst, abs(a[key] - b[key]))
            for key in ('global_peak', 'local_peak'):
                worst = max(worst, float(np.max(np.abs(np.subtract(a[key], b[key])))))
            self.assertEqual(a['crop'], b['crop'])
        self.assertLessEqual(worst, 1e-3)
        # 通过复核器：重叠区间复用分数，按去重窗口计数
        live2 = LiveScores(self.engine, VideoFrameSource(self.video), None)
        rev = TTNetReviewer(live2, {}, None, k=4)
        from cascade.scheduler import Clip

        def clip(a, b):
            return Clip('c', [], a / 120.0, b / 120.0, start_frame_id=a, end_frame_id=b)
        rev.review(clip(4, 10))
        rev.review(clip(8, 12))
        self.assertEqual(rev.n_windows, len(set(range(8, 15)) | set(range(12, 17))))
        # 视频开头缺上下文 → error
        self.assertEqual(rev.review(clip(0, 5)).status, 'error')


@unittest.skipUnless(HAS_WEIGHTS and HAS_TORCH, '需要 torch 和权重')
class TestTorchVsNumpy(unittest.TestCase):
    def test_parity(self):
        from cascade.ttnet_engine import make_engine
        dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        t = make_engine('torch', WEIGHTS, TTNET_SRC, dev)
        n = make_engine('numpy', WEIGHTS)
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 255, (27, 128, 320)).astype(np.float32)
        a, b = t.infer(x), n.infer(x)
        for key in ('global_x', 'global_y', 'local_x', 'local_y', 'events'):
            np.testing.assert_allclose(a[key], b[key], atol=1e-3, err_msg=key)
        self.assertEqual(tuple(a['crop']), tuple(b['crop']))


if __name__ == '__main__':
    unittest.main()
