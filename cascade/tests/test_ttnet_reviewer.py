# -*- coding: utf-8 -*-
"""TTNet 复核器：不需要 torch / 权重 / 视频的测试（用户电脑上也必须能跑）。"""
import json
import os
import unittest

import numpy as np

from cascade import ttnet_reviewer as tr
from cascade import ttnet_engine as te
from cascade import ttnet_cache as tc

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(_ROOT, 'calib', 'table_quads.json'), encoding='utf-8') as _f:
    QUAD_T2 = json.load(_f)['test_2']['points']
# test_2 台面内、远离球网的两点（左半台 A、右半台 B），以及网线附近一点
PT_A = (600.0, 580.0)
PT_B = (1250.0, 590.0)
PT_OFF = (100.0, 900.0)


def _clip(a, b, cid='s01:c0001', missing=False):
    from cascade.scheduler import Clip
    return Clip(cid, [], a / 120.0, b / 120.0, start_frame_id=a, end_frame_id=b, missing_context=missing)


def _rows(scores_by_event_frame, k, n=200, net=None):
    """按“事件帧”给分数，换算成窗口末帧行（e = f + k）。"""
    rows = []
    for e in range(8, n):
        f = e - k
        rows.append({'window_end': e, 'frame_id': e, 'bounce': scores_by_event_frame.get(f, 0.0),
                     'net': (net or {}).get(f, 0.0), 'infer_ms': 1.0})
    return rows


class TestHumps(unittest.TestCase):
    def test_single_hump_hysteresis(self):
        fr = list(range(10))
        sc = [0.1, 0.6, 0.66, 0.7, 0.6, 0.56, 0.8, 0.5, 0.1, 0.1]
        h = tr.find_humps(fr, sc)
        self.assertEqual(len(h), 1)
        self.assertEqual((h[0].start, h[0].end, h[0].peak_frame), (2, 6, 6))
        self.assertAlmostEqual(h[0].peak, 0.8)
        self.assertFalse(h[0].truncated)

    def test_two_humps_and_truncated(self):
        fr = list(range(8))
        sc = [0.7, 0.5, 0.3, 0.9, 0.9, 0.2, 0.66, 0.6]
        h = tr.find_humps(fr, sc)
        self.assertEqual([x.peak_frame for x in h], [0, 3, 6])
        self.assertTrue(h[-1].truncated)

    def test_below_high_no_hump(self):
        self.assertEqual(tr.find_humps(range(5), [0.6, 0.64, 0.64, 0.5, 0.0]), [])


class TestKMapping(unittest.TestCase):
    def test_default_k4(self):
        self.assertEqual(tr.window_end_for(100, 4), 104)
        self.assertEqual(tr.event_frame_for(104, 4), 100)
        self.assertEqual(tr.context_range(100, 120, 4), (96, 124))

    def test_k0_is_adapter_assumption(self):
        self.assertEqual(tr.context_range(100, 120, 0), (92, 120))

    def test_bad_k(self):
        with self.assertRaises(ValueError):
            tr.check_k(9)

    def test_reviewer_ctx(self):
        r = tr.TTNetReviewer(tr.ReplayScores([]), k=3, pre_s=0.2)
        self.assertEqual((r.pre_ctx, r.post_ctx), (5, 3))
        self.assertEqual(r.max_lead_s, 0.2)


class TestJudge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = tr.TableGeometry(QUAD_T2)

    def test_geometry_sanity(self):
        self.assertTrue(self.geo.inside(*PT_A))
        self.assertEqual(self.geo.side(*PT_A), 'A')
        self.assertEqual(self.geo.side(*PT_B), 'B')
        self.assertFalse(self.geo.inside(*PT_OFF))

    def _h(self, f, peak=0.9):
        return tr.Hump(f - 2, f + 2, f, peak, False)

    def test_table(self):
        det = {103: [(PT_A[0], PT_A[1], 'tracknet')]}
        st, ev, _ = tr.judge([self._h(100)], det, self.geo)
        self.assertEqual(st, 'confirmed')
        self.assertEqual(ev[0]['side'], 'A')
        self.assertEqual(ev[0]['evidence_frame'], 103)
        # ±5 帧之外 → uncertain
        st, ev, why = tr.judge([self._h(100)], {106: [(PT_A[0], PT_A[1], 'tracknet')]}, self.geo)
        self.assertEqual((st, ev), ('uncertain', []))
        self.assertIn('no_detection', why[0])
        # 出台 → uncertain
        st, _, why = tr.judge([self._h(100)], {100: [(PT_OFF[0], PT_OFF[1], 'yolo')]}, self.geo)
        self.assertEqual(st, 'uncertain')
        self.assertIn('off_table', why[0])
        # 没有驼峰 → rejected
        self.assertEqual(tr.judge([], det, self.geo)[0], 'rejected')

    def test_side_unknown_near_net(self):
        # 网线：两条长边中点连线；取其中点
        q = np.asarray(QUAD_T2, float)
        mid = ((q[0] + q[1]) / 2 + (q[2] + q[3]) / 2) / 2
        self.assertEqual(self.geo.side(*mid), 'UNKNOWN')
        st, _, why = tr.judge([self._h(50)], {50: [(mid[0], mid[1], 'tracknet')]}, self.geo)
        self.assertEqual(st, 'uncertain')
        self.assertIn('side_unknown', why[0])

    def test_nearest_detection_prefers_closest_then_earlier(self):
        det = {98: [(1, 1, 'a')], 102: [(2, 2, 'b')], 104: [(3, 3, 'c')]}
        self.assertEqual(tr.nearest_detection(det, 100)[0], 98)
        self.assertEqual(tr.nearest_detection(det, 103)[0], 102)

    def test_detections_skip_predicted(self):
        obs = [
            {'frame_id': 1, 'source': 'tracknet', 'ball': {'visible': True, 'center_px': [1, 2]}},
            {'frame_id': 2, 'source': 'tracknet', 'ball': {'visible': True, 'center_px': [1, 2], 'predicted': True}},
            {'frame_id': 3, 'source': 'yolo', 'ball': {'visible': False, 'center_px': None}},
        ]
        self.assertEqual(list(tr.detections_from_observations(obs)), [1])


class TestReviewer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = tr.TableGeometry(QUAD_T2)

    def _rev(self, rows, det=None, k=4):
        return tr.TTNetReviewer(tr.ReplayScores(rows), det or {}, self.geo, k=k, fps=120.0)

    def test_confirmed_and_k_shift(self):
        k = 4
        rows = _rows({60: 0.7, 61: 0.9, 62: 0.7}, k)
        r = self._rev(rows, {61: [(PT_B[0], PT_B[1], 'tracknet')]}, k)
        res = r.review(_clip(50, 70))
        self.assertEqual(res.status, 'confirmed')
        ev = res.events
        self.assertEqual(len(ev), 1)
        self.assertEqual((ev[0]['type'], ev[0]['side'], ev[0]['sources']), ('BOUNCE', 'B', ['ttnet', 'tracknet']))
        self.assertEqual(res.reviewer, 'ttnet')
        self.assertAlmostEqual(ev[0]['timestamp_s'], 61 / 120.0, places=5)
        # 窗口末帧 = 61 + 4 = 65 的行承载这个峰
        self.assertEqual(max(rows, key=lambda x: x['bounce'])['window_end'], 65)

    def test_wrong_k_moves_peak(self):
        rows = _rows({61: 0.9}, 4)
        r = self._rev(rows, {}, k=0)  # 用 k=0 读 k=4 的数据：峰被认到 65
        res = r.review(_clip(50, 70))
        self.assertEqual(r.log[-1]['humps'][0]['peak_frame'], 65)

    def test_rejected_when_flat(self):
        r = self._rev(_rows({}, 4))
        self.assertEqual(r.review(_clip(50, 70)).status, 'rejected')

    def test_uncertain_without_evidence(self):
        r = self._rev(_rows({61: 0.95}, 4))
        res = r.review(_clip(50, 70))
        self.assertEqual(res.status, 'uncertain')
        self.assertEqual(res.events, [])

    def test_missing_context_is_error(self):
        rows = [x for x in _rows({}, 4) if x['window_end'] != 66]
        r = self._rev(rows)
        res = r.review(_clip(50, 70))
        self.assertEqual(res.status, 'error')
        self.assertEqual(r.log[-1]['missing_windows'], [66])
        # 视频开头：需要的窗口末帧 < 8 不存在
        res = r.review(_clip(0, 10))
        self.assertEqual(res.status, 'error')

    def test_net_never_net_touch(self):
        r = self._rev(_rows({}, 4, net={60: 0.9, 61: 0.95}), {61: [(PT_A[0], PT_A[1], 'tracknet')]})
        res = r.review(_clip(50, 70))
        self.assertEqual(res.status, 'rejected')
        self.assertEqual(res.events, [])
        self.assertEqual(len(r.log[-1]['net_cross']), 1)

    def test_overlap_dedup_window_count(self):
        r = self._rev(_rows({}, 4))
        r.review(_clip(50, 70))   # 窗口 54..74 → 21
        r.review(_clip(60, 80))   # 窗口 64..84，新增 75..84 → 10
        self.assertEqual(r.n_windows, 31)

    def test_scheduler_missing_context_is_error(self):
        r = self._rev(_rows({}, 4))
        res = r.review(_clip(50, 70, missing=True))
        self.assertEqual((res.status, res.reason), ('error', 'missing_context'))


class TestNumpyOps(unittest.TestCase):
    """numpy 引擎的算子对照朴素循环实现（不依赖 torch）。"""

    def test_conv3x3(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(3, 6, 7)).astype(np.float32)
        w = rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
        b = rng.normal(size=4).astype(np.float32)
        got = te._conv2d(x, w, b, 1)
        xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
        ref = np.zeros((4, 6, 7), np.float32)
        for o in range(4):
            for i in range(6):
                for j in range(7):
                    ref[o, i, j] = (xp[:, i:i + 3, j:j + 3] * w[o]).sum() + b[o]
        np.testing.assert_allclose(got, ref, atol=1e-4)

    def test_tconv_definition(self):
        # ConvTranspose2d 定义：out[o, i*s - p + ki, j*s - p + kj] += x[c,i,j] * w[c,o,ki,kj]
        rng = np.random.default_rng(2)
        for (k, s, p, op) in ((3, 2, 1, 1), (3, 2, 0, 0)):
            x = rng.normal(size=(2, 4, 5)).astype(np.float32)
            w = rng.normal(size=(2, 3, k, k)).astype(np.float32)
            b = rng.normal(size=3).astype(np.float32)
            H = (4 - 1) * s - 2 * p + k + op
            W = (5 - 1) * s - 2 * p + k + op
            full = np.zeros((3, H + 2 * p + k, W + 2 * p + k), np.float64)
            for c in range(2):
                for i in range(4):
                    for j in range(5):
                        full[:, i * s:i * s + k, j * s:j * s + k] += x[c, i, j] * w[c]
            ref = full[:, p:p + H, p:p + W] + b[:, None, None]
            got = te._tconv2d(x, w, b, s, p, op)
            self.assertEqual(got.shape, ref.shape)
            np.testing.assert_allclose(got, ref, atol=1e-4)

    def test_pool_and_nearest(self):
        x = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)
        self.assertEqual(te._pool2(x).shape, (2, 2, 3))
        self.assertEqual(te._pool2(x)[0, 0, 0], 7)
        np.testing.assert_array_equal(te._SRC_X[:13], [0] * 6 + [1] * 6 + [2])
        self.assertEqual(te._SRC_X[-1], 319)
        self.assertEqual(te._SRC_Y[-1], 127)
        self.assertTrue(np.all(np.diff(te._SRC_Y) >= 0))

    def test_crop_params(self):
        gx = np.zeros(320); gy = np.zeros(128)
        gx[0] = gy[0] = 1
        self.assertEqual(te.NumpyEngine.crop_params(gx, gy), (0, 320, 0, 128, 0, 0))
        gx[:] = 0; gy[:] = 0; gx[319] = gy[127] = 1
        # x_center=1914 → x_min=1754，x_max=1920（宽 166）→ x_pad=77
        self.assertEqual(te.NumpyEngine.crop_params(gx, gy), (1754, 1920, 1007, 1080, 77, 27))

    def test_decode_ball_and_guard(self):
        out = {'global_x': np.zeros(320), 'global_y': np.zeros(128),
               'local_x': np.zeros(320), 'local_y': np.zeros(128)}
        self.assertEqual(tc.decode_ball(out), (None, None))
        out['global_x'][100] = 0.9; out['global_y'][64] = 0.8
        out['local_x'][170] = 0.7; out['local_y'][60] = 0.6
        c, conf = tc.decode_ball(out)
        self.assertEqual(c, [100 * 6 + 170 - 160, 64 * 8.4375 + 60 - 64])
        self.assertAlmostEqual(conf, 0.6)
        g = tc.BallGuard()
        self.assertEqual(g(c, 0.2), (None, None))          # 低置信
        res = [g(c, 0.9)[0] is not None for _ in range(5)]  # 冻结 >= 3 次后丢弃
        self.assertEqual(res, [True, True, True, False, False])


if __name__ == '__main__':
    unittest.main()


class TestKCheck(unittest.TestCase):
    def test_recovers_k(self):
        from cascade.ttnet_k_check import k_report
        gt = [100, 200, 300]
        k_true = 4
        sc = {}
        for g in gt:
            for d, v in ((-2, 0.7), (-1, 0.8), (0, 0.95), (1, 0.8), (2, 0.7)):
                sc[g + k_true + d] = v
        rows = [{'window_end': e, 'bounce': sc.get(e, 0.0)} for e in range(8, 400)]
        rep = k_report(rows, gt)
        self.assertEqual(rep['k_estimate_median_offset'], 4.0)
        self.assertEqual(rep['best_k_by_abs_err'], 4)
        self.assertEqual(rep['per_k'][4]['f1'], 1.0)


class TestRunCascadeReplay(unittest.TestCase):
    """端到端：test_2 样例观测 + 构造的 TTNet 缓存（在推断真值落台处放驼峰），走 run_cascade。"""

    GT = os.path.join(_ROOT, '_data', 'gt', 'extended_openttgames', 'inferred',
                      'test_2_events_markup_inferred.json')
    OBS = os.path.join(_ROOT, 'adapter', 'examples', 'test_2_observations.jsonl')

    def test_replay_end_to_end(self):
        if not os.path.isfile(self.GT):
            self.skipTest('缺推断真值 ' + self.GT)
        import tempfile
        from cascade.run_cascade import run
        from types import SimpleNamespace
        with open(self.GT, encoding='utf-8') as f:
            gt = sorted(int(k) for k, v in json.load(f).items() if v == 'bounce')
        k = 4
        sc = {}
        for g in gt:
            for d, v in ((-2, 0.7), (-1, 0.85), (0, 0.95), (1, 0.85), (2, 0.7)):
                sc[g + d] = v
        rows = _rows(sc, k, n=3700)
        tmp = tempfile.mkdtemp()
        cache = os.path.join(tmp, 't2.jsonl')
        with open(cache, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        args = SimpleNamespace(ttnet_cache=cache, video=None, ttnet_weights=None, ttnet_src=None,
                               ttnet_engine='auto', ttnet_device='cpu', ttnet_k=k)
        res = run(self.OBS, 'test2', reviewer='ttnet', pre_ctx=8 - k, post_ctx=k, ttnet_args=args)
        s = res['summary']
        self.assertEqual(s['reviewer'], 'ttnet')
        self.assertGreater(s['bounces']['confirmed'], 0)
        # 去重窗口不多于逐片段累计
        self.assertLessEqual(s['ttnet']['windows_dedup'], s['clips']['ttnet_windows'])
        # 每个确认落台都对应一个真值（±2 帧内）
        for r in res['results']:
            for e in r.events:
                f = round(e['timestamp_s'] * 120)
                self.assertTrue(any(abs(f - g) <= 2 for g in gt), f)
        self.assertNotIn('NET_TOUCH', [e['type'] for r in res['results'] for e in r.events])
