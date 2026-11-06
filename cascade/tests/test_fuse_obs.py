import unittest

from cascade.fuse_obs import FuseParams, fuse


def obs(f, xy=None, conf=0.8, source="yolo"):
    vis = xy is not None
    return {"frame_id": f, "timestamp_s": f / 120.0, "source": source, "image_size": [1920, 1080],
            "ball": {"visible": vis, "center_px": list(xy) if vis else None,
                     "bbox_xyxy": None, "confidence": conf if vis else None, "predicted": False},
            "event_scores": {"bounce": None, "net_hit": None}}


class FuseTest(unittest.TestCase):
    def test_yolo_primary_tracknet_fills_gap_in_rally(self):
        yolo = [obs(f, (100 + 10 * f, 500)) for f in range(10)] + [obs(f) for f in range(10, 20)]
        tn = [obs(f, (100 + 10 * f, 500), source="tracknet") for f in range(20)]
        out, st = fuse(yolo, tn, FuseParams(fill_window_frames=60))
        self.assertEqual([d["source"] for d in out[:10]], ["yolo"] * 10)
        self.assertEqual([d["source"] for d in out[10:]], ["tracknet"] * 10)
        self.assertEqual(st["tracknet_called"], 10)

    def test_tracknet_not_called_outside_rally(self):
        yolo = [obs(f) for f in range(100)]
        tn = [obs(f, (500, 500), source="tracknet") for f in range(100)]
        out, st = fuse(yolo, tn, FuseParams(fill_window_frames=60))
        self.assertEqual(st["tracknet_called"], 0)
        self.assertFalse(any(d["ball"]["visible"] for d in out))
        # None：YOLO 一丢就调
        _, st2 = fuse(yolo, tn, FuseParams(fill_window_frames=None))
        self.assertEqual(st2["tracknet_called"], 100)

    def test_isolated_and_low_conf_yolo_rejected(self):
        yolo = [obs(0), obs(1), obs(2, (800, 300)), obs(3), obs(4),
                obs(5, (100, 500), conf=0.3), obs(6, (105, 500), conf=0.3)]
        out, st = fuse(yolo, None)
        self.assertFalse(any(d["ball"]["visible"] for d in out))
        self.assertEqual(st["yolo_rejected"], 1)          # 孤立点；低置信不算检出
        self.assertEqual(st["tracknet_called"], 0)


if __name__ == "__main__":
    unittest.main()
