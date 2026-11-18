"""合成图上的回归测试：真值精确已知。python -m unittest discover -s tests"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tcalib import camera, detect, synth  # noqa: E402
from tcalib.calibrate import calibrate_frames, result_to_dict  # noqa: E402
from tcalib.table import NET_HEIGHT_CM, NET_POST_OUT_CM, NET_X_CM, TABLE_WIDTH_CM  # noqa: E402
from tcalib.viz import corner_crops, draw_overlay  # noqa: E402

CAMS = [
    {},
    {"f": 1400, "cam_xyh": (60, 600, 140)},
    {"f": 2200, "cam_xyh": (200, 700, 260), "roll_deg": 2},
    {"f": 1900, "cam_xyh": (137, 532, 150)},
]


class TestCamera(unittest.TestCase):
    def test_pnpf_recovers_focal_and_net(self):
        for kw in CAMS:
            cam = synth.make_camera(**kw)
            _, gt = synth.render(cam, seed=0, n_occluders=0)
            m = camera.solve_camera(gt, cam.size)
            self.assertAlmostEqual(m.f / cam.f, 1.0, places=3)
            self.assertLess(m.reproj_rms_px, 1e-3)
            net = camera.net_geometry(gt, m)
            true_top = cam.project(
                np.array([[NET_X_CM, -NET_POST_OUT_CM, NET_HEIGHT_CM], [NET_X_CM, TABLE_WIDTH_CM + NET_POST_OUT_CM, NET_HEIGHT_CM]])
            )
            self.assertLess(np.abs(np.array(net["net_top_line"]) - true_top).max(), 0.01)
            # 网顶在图像上必须比网底高（y 更小）
            self.assertLess(np.mean(np.array(net["net_top_line"])[:, 1]), np.mean(np.array(net["net_bottom_full"])[:, 1]))

    def test_closed_form_focal(self):
        cam = synth.make_camera()
        _, gt = synth.render(cam, seed=0, n_occluders=0)
        f0 = camera.focal_from_homography(camera.homography_from_corners(gt), ((1920 - 1) / 2, (1080 - 1) / 2))
        self.assertAlmostEqual(f0 / cam.f, 1.0, places=3)


class TestDetect(unittest.TestCase):
    def test_single_frame_subpixel(self):
        for seed, kw in enumerate(CAMS):
            cam = synth.make_camera(**kw)
            img, gt = synth.render(cam, seed=seed)
            r = detect.detect_frame(img)
            self.assertTrue(r.ok, r.reason)
            err = np.linalg.norm(r.corners - gt, axis=1)
            self.assertLess(err.max(), 1.5, f"cam {kw}: {err}")

    def test_dim_left_half(self):
        """test_2 那种左半台明显偏暗。"""
        cam = synth.make_camera()
        img, gt = synth.render(cam, seed=5, light_gradient=0.35)
        r = detect.detect_frame(img)
        self.assertTrue(r.ok, r.reason)
        self.assertLess(np.linalg.norm(r.corners - gt, axis=1).max(), 1.5)

    def test_green_table(self):
        cam = synth.make_camera(f=1600)
        img, gt = synth.render(cam, seed=6, table_bgr=(60, 110, 40))
        r = detect.detect_frame(img)
        self.assertTrue(r.ok, r.reason)
        self.assertLess(np.linalg.norm(r.corners - gt, axis=1).max(), 1.5)

    def test_semi_auto_from_rough_clicks(self):
        cam = synth.make_camera()
        img, gt = synth.render(cam, seed=7)
        rough = gt + np.random.default_rng(0).uniform(-15, 15, gt.shape)
        r = detect.detect_frame(img, init_quad=rough)
        self.assertTrue(r.ok, r.reason)
        self.assertLess(np.linalg.norm(r.corners - gt, axis=1).max(), 1.5)

    def test_order_corners(self):
        q = np.array([[10, 50], [100, 0], [0, 0], [90, 60]], float)
        o = detect.order_corners(q)
        np.testing.assert_array_equal(o, [[0, 0], [100, 0], [90, 60], [10, 50]])


class TestCalibrate(unittest.TestCase):
    def test_multi_frame_with_occlusion(self):
        cam = synth.make_camera(f=1800, cam_xyh=(120, 560, 160))
        frames, gt = {}, None
        for s in range(8):
            img, gt = synth.render(cam, seed=100 + s, n_occluders=3)
            frames[f"f{s:06d}"] = img
        res = calibrate_frames(frames)
        self.assertIsNotNone(res.corners)
        self.assertLess(np.linalg.norm(res.corners - gt, axis=1).max(), 1.0)
        self.assertEqual(res.confidence, "high", res.reasons)
        d = result_to_dict(res, "synth", "auto_edge_v1")
        tc = d["table_calibration"]
        self.assertEqual(len(tc["table_polygon"]), 4)
        self.assertLess(tc["net_zone"][0], tc["net_zone"][1])
        y0, y1, x0, x1 = tc["roi_box"]
        self.assertTrue(0 <= y0 < y1 <= 1080 and 0 <= x0 < x1 <= 1920)
        # net_x 在四角重心 y 处反投影应为 137 cm（国栋 calibration_net_x_cm 的口径）
        H = camera.homography_from_corners(np.array(tc["table_polygon"], float))
        yc = np.mean(np.array(tc["table_polygon"], float)[:, 1])
        x_cm = camera.apply_h(np.linalg.inv(H), np.array([[tc["net_x"], yc]]))[0, 0]
        self.assertAlmostEqual(x_cm, 137.0, delta=1.0)
        # 可视化能跑
        img = frames["f000000"]
        self.assertEqual(draw_overlay(img, d).shape, img.shape)
        self.assertEqual(corner_crops(img, {"auto": d["points"], "gt": gt.tolist()}).ndim, 3)


if __name__ == "__main__":
    unittest.main()
