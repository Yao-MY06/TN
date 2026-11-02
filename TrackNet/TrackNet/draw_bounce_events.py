"""把落台事件画到视频帧上，用于核对落点位置与半台归属。

画出：台面四角多边形、球网（长边中点连线）、每个落点的圆与 A/B 标签，
并把若干帧拼成一张图，便于目视检查单应性是否可信、A/B 方向是否正确。

用法：
  python draw_bounce_events.py --video <test_2.mp4> --game test_2 \
      --events <test_2_bounce_events.jsonl> --out <bounce_side_check.jpg>
"""
import argparse
import json
import os

import cv2
import numpy as np

from detect_bounce import (NET_X_M, TABLE_WID_M, load_quad,
                           table_homography)


def net_in_image(H):
    """球网两端（俯视坐标）反投影回图像坐标。"""
    h_inv = np.linalg.inv(H)
    pts = np.array([[[NET_X_M, 0.0]],
                    [[NET_X_M, TABLE_WID_M]]], dtype=np.float32)
    return cv2.perspectiveTransform(pts, h_inv).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--fps", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--thumb_w", type=int, default=640)
    ap.add_argument("--limit", type=int, default=6)
    args = ap.parse_args()

    poly = load_quad(args.game)
    H = table_homography(poly)
    net = net_in_image(H).astype(int)

    events = [json.loads(l) for l in open(args.events, encoding="utf-8") if l.strip()]
    # 均匀取若干条，避免只看到开头
    if len(events) > args.limit:
        idx = np.linspace(0, len(events) - 1, args.limit).round().astype(int)
        events = [events[i] for i in idx]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频 {args.video}")

    tiles = []
    for ev in events:
        frame = int(round(ev["timestamp_s"] * args.fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        if not ok:
            print(f"跳过 frame {frame}（读不到）")
            continue

        cv2.polylines(img, [poly.astype(np.int32)], True, (0, 200, 0), 2)
        cv2.line(img, tuple(net[0]), tuple(net[1]), (255, 200, 0), 2)

        cx, cy = [int(round(v)) for v in ev["position_px"]]
        color = (0, 0, 255) if ev["side"] == "A" else (255, 0, 255)
        cv2.circle(img, (cx, cy), 16, color, 2)
        cv2.circle(img, (cx, cy), 3, color, -1)
        cv2.putText(img, f"{ev['side']} f{frame}", (cx + 20, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        h, w = img.shape[:2]
        scale = args.thumb_w / w
        tile = cv2.resize(img, (args.thumb_w, int(h * scale)))
        tiles.append(tile)

    cap.release()
    if not tiles:
        raise SystemExit("没有可用帧")

    th, tw = tiles[0].shape[:2]
    cols = min(args.cols, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    grid = np.full((rows * th, cols * tw, 3), 255, dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f"写入 {args.out}  ({len(tiles)} 帧, 网线端点(图像)={net.tolist()})")


if __name__ == "__main__":
    main()
