"""固定机位视频自动标定入口：抽 5~10 个清晰帧 → 四角/球网 → 兼容结果。

用法：
  python scripts/calibrate_video.py --video test_2.mp4 --name test_2 --out out/test_2
  python scripts/calibrate_video.py --video test_2.mp4 --name test_2 --out out/test_2 \
      --init-quad "540,516;1330,532;1484,656;352,631"
  python scripts/calibrate_video.py --video test_2.mp4 --name test_2 --out out/test_2 \
      --tn-quads ../TN/calib/table_quads.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tcalib.calibrate import calibrate_frames, calibrate_results, result_to_dict  # noqa: E402
from tcalib.detect import DetectParams, detect_frame  # noqa: E402
from tcalib.table import CORNER_NAMES  # noqa: E402
from tcalib.video import sample_video_frames  # noqa: E402
from tcalib.viz import corner_crops, draw_overlay  # noqa: E402


def parse_quad(value: str) -> np.ndarray:
    pts = [list(map(float, p.split(","))) for p in value.split(";")]
    if len(pts) != 4:
        raise SystemExit("--init-quad 需要 4 个点")
    return np.array(pts, dtype=np.float64)


def write_result(out: str, name: str, method: str, res, sampled, video: str,
                 runtime_s: float) -> dict:
    d = result_to_dict(res, name, method)
    d["runtime_s"] = round(runtime_s, 2)
    d["video"] = os.path.abspath(video)
    d["sample"] = [
        {"frame_id": x.frame_id, "timestamp_s": round(x.timestamp_s, 3),
         "sharpness": round(x.sharpness, 2)} for x in sampled
    ]
    if res.corners is not None:
        d["image_size"] = [int(sampled[0].image.shape[1]), int(sampled[0].image.shape[0])]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "frames.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        head = ["frame", "ok", "reason", "used"]
        head += [f"{c}_{axis}" for c in CORNER_NAMES for axis in "xy"]
        head += [f"{side}_inlier" for side in ("far", "right", "near", "left")]
        wr.writerow(head)
        for key in res.frame_ids:
            item = res.frame_results[key]
            row = [key, int(item.ok), item.reason, int(key in res.used_ids)]
            row += ([round(float(v), 2) for v in item.corners.reshape(-1)]
                    if item.corners is not None else [""] * 8)
            row += [round(item.sides[s].inlier_ratio, 3) if s in item.sides else ""
                    for s in ("far", "right", "near", "left")]
            wr.writerow(row)
    if res.corners is not None and sampled:
        show = next((x for x in sampled if f"f{x.frame_id:06d}" in res.used_ids), sampled[0])
        overlay = draw_overlay(show.image, d)
        cv2.imwrite(os.path.join(out, "overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
        crops = corner_crops(show.image, {"auto": d["points"]})
        cv2.imwrite(os.path.join(out, "corners.png"), crops)
    return d


def merge_tn_quads(path: str, name: str, d: dict) -> None:
    if not d.get("points"):
        raise SystemExit("没有四角结果，不能写入 TN table_quads.json")
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data[name] = {
        "points": d["points"],
        "method": d.get("method", "auto_edge_v2"),
        "verified_frames": [x["frame_id"] for x in d.get("sample", [])],
        "confidence": d.get("confidence"),
        "net_x": (d.get("table_calibration") or {}).get("net_x"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="固定机位视频自动标定球台四角与球网")
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=8, help="最终使用的清晰帧数，建议 5~10")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--candidate-factor", type=int, default=3)
    ap.add_argument("--no-prefilter", action="store_true",
                    help="关闭单帧预检；默认先抽候选帧并筛掉检测失败的帧")
    ap.add_argument("--min-sharpness", type=float, default=0.0)
    ap.add_argument("--init-quad", default=None,
                    help="粗四角，顺序 far_left;far_right;near_right;near_left，坐标 x,y")
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--tn-quads", default=None, help="可选：把结果合并进 TN 的 table_quads.json")
    args = ap.parse_args()

    t0 = time.time()
    init = parse_quad(args.init_quad) if args.init_quad else None
    sample_count = args.count if args.no_prefilter else max(
        args.count, args.count * args.candidate_factor)
    frames, sampled = sample_video_frames(
        args.video, count=sample_count, start_s=args.start_s,
        window_s=args.window_s, candidate_factor=1,
        min_sharpness=args.min_sharpness)
    if len(frames) < 2:
        raise SystemExit(f"只抽到 {len(frames)} 帧，无法做多帧汇总")
    method = "auto_edge_v2" if init is None else "semi_auto_edge_v2"
    if args.no_prefilter:
        res = calibrate_frames(frames, init_quad=init)
    else:
        by_key = {f"f{x.frame_id:06d}": x for x in sampled}
        checked = {}
        for i, (key, image) in enumerate(frames.items()):
            result = detect_frame(image, init_quad=init, params=DetectParams(), seed=i)
            checked[key] = result
        valid = [(key, result) for key, result in checked.items() if result.ok]
        valid.sort(key=lambda kr: (
            sum(s.inlier_ratio for s in kr[1].sides.values()) / max(1, len(kr[1].sides)),
            by_key[kr[0]].sharpness), reverse=True)
        selected = valid[:args.count]
        if len(selected) < 2:
            print("单帧预检后有效帧不足 2 个，继续用全部候选结果做低置信度汇总",
                  file=sys.stderr)
            selected = list(checked.items())
        selected_keys = {key for key, _ in selected}
        sampled = sorted((by_key[key] for key in selected_keys), key=lambda x: x.frame_id)
        first = next(iter(frames.values()))
        res = calibrate_results({key: checked[key] for key in selected_keys},
                                (first.shape[1], first.shape[0]))
    d = write_result(args.out, args.name, method, res, sampled, args.video,
                     time.time() - t0)

    if args.save_frames:
        frame_dir = os.path.join(args.out, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        for item in sampled:
            cv2.imwrite(os.path.join(frame_dir, f"f{item.frame_id:06d}.png"), item.image)

    pts = d.get("points")
    print(f"{args.name}: conf={d['confidence']} used {d['n_used']}/{d['n_frames']} "
          f"{d['runtime_s']:.1f}s")
    print(f"  points={pts}")
    print(f"  std={d.get('per_corner_std_px')} reasons={d.get('confidence_reasons')}")
    tc = d.get("table_calibration") or {}
    if tc:
        print(f"  net_x={tc.get('net_x')} net_zone={tc.get('net_zone')} roi_box={tc.get('roi_box')}")
    if args.tn_quads:
        merge_tn_quads(args.tn_quads, args.name, d)
        print(f"  已合并到 {os.path.abspath(args.tn_quads)}")
    if not pts:
        print("自动检测失败：用 --init-quad 给粗四角，或运行 scripts/click_corners.py 人工兜底",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())