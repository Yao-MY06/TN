"""直接从视频做自动标定：抽帧 → 逐帧检测 → 多帧汇总 → 四角 / 球网 / 兼容字段。

用法：
  # 开场标定（默认）：开头 10 秒内均匀取 24 帧
  python scripts/calibrate_video.py --video test_2.mp4 --name test_2 --out out/test_2
  # 离线验收：整段视频 2%～98% 均匀取 24 帧，并另存第 1200 帧原图（PNG，不参与标定）
  python scripts/calibrate_video.py --video test_2.mp4 --name test_2 --out out/test_2 \
      --mode spread --save-frame 1200
  # 半自动兜底：给粗四角（远左、远右、近右、近左；误差 20px 内即可）
  python scripts/calibrate_video.py --video test_2.mp4 --name test_2 --out out/test_2 \
      --init-quad "540,516;1330,532;1484,656;352,631"

所有抽到的帧都参与置信度统计（有效帧占比、抖动），汇总时剔除检测失败和偏离大的帧。
结果只写到 --out 目录，不改任何其它文件。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tcalib.calibrate import calibrate_frames, result_to_dict  # noqa: E402
from tcalib.report import write_outputs  # noqa: E402
from tcalib.video import sample_video_frames  # noqa: E402


def parse_quad(value: str) -> np.ndarray:
    pts = [list(map(float, p.split(","))) for p in value.split(";")]
    if len(pts) != 4:
        raise SystemExit("--init-quad 需要 4 个点")
    return np.array(pts, dtype=np.float64)


def run_video(
    video: str,
    name: str,
    out: str,
    *,
    mode: str = "opening",
    count: int = 24,
    start_s: float = 0.0,
    window_s: float = 10.0,
    init_quad: Optional[np.ndarray] = None,
    save_frame_ids: Sequence[int] = (),
) -> Dict:
    t0 = time.time()
    frames, sampled, info, extras = sample_video_frames(
        video, mode=mode, count=count, start_s=start_s, window_s=window_s, extra_ids=save_frame_ids
    )
    if len(frames) < 2:
        raise RuntimeError(f"只读到 {len(frames)} 帧，无法做多帧汇总")
    res = calibrate_frames(frames, init_quad=init_quad)
    method = "auto_edge_pooled_v3" if init_quad is None else "semi_auto_edge_pooled_v3"
    d = result_to_dict(res, name, method)
    d["runtime_s"] = round(time.time() - t0, 2)
    d["image_size"] = [info.width, info.height]
    d["video"] = info.to_dict()
    d["sampling"] = {
        "mode": mode,
        "count": count,
        **({"start_s": start_s, "window_s": window_s} if mode == "opening" else {"range": "2%-98%"}),
        "frames": [
            {"frame_id": x.frame_id, "timestamp_s": round(x.timestamp_s, 3), "sharpness": round(x.sharpness, 1)}
            for x in sampled
        ],
    }
    if init_quad is not None:
        d["init_quad"] = np.asarray(init_quad).round(1).tolist()
    show_id = write_outputs(out, d, res, frames)
    d["overlay_frame"] = show_id
    if extras:
        fdir = os.path.join(out, "frames")
        os.makedirs(fdir, exist_ok=True)
        for fid, img in extras.items():
            cv2.imwrite(os.path.join(fdir, f"f{fid:06d}.png"), img)
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description="固定机位视频自动标定球台四角与球网")
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("opening", "spread"), default="opening",
                    help="opening：开头一段时间内取帧（开场标定）；spread：整段视频均匀取帧（离线验收）")
    ap.add_argument("--count", type=int, default=24, help="抽帧数")
    ap.add_argument("--start-s", type=float, default=0.0, help="opening 模式的起点（秒）")
    ap.add_argument("--window-s", type=float, default=10.0, help="opening 模式的时长（秒）")
    ap.add_argument("--init-quad", default=None, help="粗四角 far_left;far_right;near_right;near_left，坐标 x,y")
    ap.add_argument("--save-frame", type=int, action="append", default=[],
                    help="另存这些帧号的原分辨率 PNG 到 <out>/frames/（不参与标定），可重复")
    args = ap.parse_args()

    init = parse_quad(args.init_quad) if args.init_quad else None
    d = run_video(args.video, args.name, args.out, mode=args.mode, count=args.count, start_s=args.start_s,
                  window_s=args.window_s, init_quad=init, save_frame_ids=args.save_frame)
    print(f"{args.name}: conf={d['confidence']} used {d['n_used']}/{d['n_frames']} {d['runtime_s']:.1f}s")
    print(f"  points={d.get('points')}")
    print(f"  std={d.get('per_corner_std_px')} reasons={d.get('confidence_reasons')}")
    tc = d.get("table_calibration") or {}
    if tc:
        print(f"  net_x={tc.get('net_x')} net_zone={tc.get('net_zone')} roi_box={tc.get('roi_box')}")
    if d["video"]["seek_mismatch"]:
        print(f"  [warn] 定位不准的帧：{d['video']['seek_mismatch']}", file=sys.stderr)
    if not d.get("points"):
        print("自动检测失败：用 --init-quad 给粗四角，或运行 scripts/click_corners.py 人工兜底", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
