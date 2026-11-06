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
        out, st = fuse(yolo, tn, FuseParams(primary="yolo", fill_window_s=0.5))
        self.assertEqual([d["source"] for d in out[:10]], ["yolo"] * 10)
        self.assertEqual([d["source"] for d in out[10:]], ["tracknet"] * 10)
        self.assertEqual(st["tracknet_called"], 10)

    def test_tracknet_not_called_outside_rally(self):
        yolo = [obs(f) for f in range(100)]
        tn = [obs(f, (500, 500), source="tracknet") for f in range(100)]
        out, st = fuse(yolo, tn, FuseParams(primary="yolo", fill_window_s=0.5))
        self.assertEqual(st["tracknet_called"], 0)
        self.assertFalse(any(d["ball"]["visible"] for d in out))
        # None：YOLO 一丢就调
        _, st2 = fuse(yolo, tn, FuseParams(primary="yolo", fill_window_s=None))
        self.assertEqual(st2["tracknet_called"], 100)

    def test_isolated_and_low_conf_yolo_rejected(self):
        yolo = [obs(0), obs(1), obs(2, (800, 300)), obs(3), obs(4),
                obs(5, (100, 500), conf=0.3), obs(6, (105, 500), conf=0.3)]
        out, st = fuse(yolo, None, FuseParams(primary="yolo"))
        self.assertFalse(any(d["ball"]["visible"] for d in out))
        self.assertEqual(st["yolo_rejected"], 1)          # 孤立点；低置信不算检出
        self.assertEqual(st["tracknet_called"], 0)


    def test_tracknet_primary_yolo_recaptures_only_tracknet_gaps(self):
        # 按方案：TrackNet 连续轨迹；TrackNet 丢球而 YOLO 可信时用 YOLO 补；两者都有时保留 TrackNet
        tn = [obs(f, (100 + 10 * f, 500), source="tracknet") for f in range(5)] + \
             [obs(f, source="tracknet") for f in range(5, 10)]
        yolo = [obs(f, (103 + 10 * f, 502)) for f in range(10)]
        out, st = fuse(yolo, tn, FuseParams(primary="tracknet"))
        self.assertEqual([d["source"] for d in out], ["tracknet"] * 5 + ["yolo"] * 5)
        self.assertEqual(out[0]["ball"]["center_px"], [100, 500])
        self.assertEqual((st["tracknet_used"], st["yolo_used"], st["tracknet_call_ratio"]), (5, 5, 1.0))

    def test_gate_is_time_based(self):
        # 同样相隔 1 帧，60 fps 下时间差 0.0167 s 仍在门限内；时间戳稀疏（0.1 s）则不算连续
        mk = lambda f, t, x: {**obs(f, (x, 500)), "timestamp_s": t}
        ok, _ = fuse([mk(0, 0.0, 100), mk(1, 1 / 60, 110)], None, FuseParams(primary="yolo"))
        far, _ = fuse([mk(0, 0.0, 100), mk(1, 0.1, 110)], None, FuseParams(primary="yolo"))
        self.assertTrue(all(d["ball"]["visible"] for d in ok))
        self.assertFalse(any(d["ball"]["visible"] for d in far))


if __name__ == "__main__":
    unittest.main()
