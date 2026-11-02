"""Streaming TrackNet inference for arbitrary-length / arbitrary-resolution videos.

Improvements over the original infer_on_video.py:
  * Streaming (two passes) -> constant memory, no more 15GB OOM on long 720p videos.
  * Coordinates rescaled from the 720x1280 model space to the *actual* frame size,
    so 1080p / 4K / portrait inputs are handled correctly.
  * Batched inference (much faster than the original per-frame loop).
  * Direct .mp4 output + optional per-frame CSV of ball coordinates.

Usage:
    python track_video.py --model_path weights/model_best.pt \
        --video_path in.mp4 --out_path out.mp4 [--extrapolation] [--csv out.csv]
"""
import os
import csv
import time
import argparse
from collections import deque
from itertools import groupby

import cv2
import numpy as np
import torch
from scipy.spatial import distance

from model import BallTrackerNet
from general import postprocess

MODEL_H, MODEL_W = 360, 640          # network input size
SPACE_SCALE = 2                      # postprocess() returns coords in (H*2, W*2)


def probe_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    meta = dict(
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
    )
    cap.release()
    return meta


def preprocess(frame):
    img = cv2.resize(frame, (MODEL_W, MODEL_H))
    return img


# --------------------------------------------------------------------------- #
# pass 1: run the network over the video and collect the ball track
# --------------------------------------------------------------------------- #
def predict_track(video_path, model, device, batch_size=4, max_frames=None, log_every=200, amp=True,
                  min_radius=2, max_radius=7):
    meta = probe_video(video_path)
    n = meta["n"] if max_frames is None else min(meta["n"], max_frames)
    sx, sy = meta["w"] / (MODEL_W * SPACE_SCALE), meta["h"] / (MODEL_H * SPACE_SCALE)

    cap = cv2.VideoCapture(video_path)
    buf = deque(maxlen=3)
    ball_track = [(None, None), (None, None)]     # first two frames have no prediction
    dists = [-1, -1]

    pending_frames = []                            # list of 9ch inputs waiting for a forward
    pending_idx = []

    def flush():
        if not pending_frames:
            return
        inp = torch.from_numpy(np.stack(pending_frames)).float().to(device)
        with torch.no_grad():
            if amp and device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(inp)
                out = out.float()
            else:
                out = model(inp)
        outs = out.argmax(dim=1).detach().cpu().numpy()
        for k, o in enumerate(outs):
            x, y = postprocess(o, height=MODEL_H, width=MODEL_W,
                               min_radius=min_radius, max_radius=max_radius)
            if x is not None:
                x *= sx
                y *= sy
            idx = pending_idx[k]
            ball_track.append((x, y))
            if x is not None and ball_track[-2][0] is not None:
                dist = distance.euclidean(ball_track[-1], ball_track[-2])
            else:
                dist = -1
            dists.append(dist)
        pending_frames.clear()
        pending_idx.clear()

    cur = 0
    t0 = time.time()
    while cur < n:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(preprocess(frame))
        if len(buf) == 3:
            imgs = np.concatenate((buf[2], buf[1], buf[0]), axis=2).astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)
            pending_frames.append(imgs)
            pending_idx.append(cur)
            if len(pending_frames) >= batch_size:
                flush()
        cur += 1
        if log_every and cur % log_every == 0:
            el = time.time() - t0
            print(f"  [infer] {cur}/{n} frames  {cur/max(el,1e-6):.1f} fps", flush=True)
    flush()
    cap.release()
    return ball_track, dists, meta


# --------------------------------------------------------------------------- #
# post-processing (same logic as the original script)
# --------------------------------------------------------------------------- #
def remove_outliers(ball_track, dists, max_dist=100):
    """Drop single-frame spikes.

    `max_dist` is in *frame pixels*. The classic value 100 assumes a 1280x720
    frame; for a 1920x1080 or 4K capture scale it by the frame width so the
    criterion stays resolution-independent.
    """
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if (dists[i + 1] > max_dist) | (dists[i + 1] == -1):
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i - 1] == -1:
            ball_track[i - 1] = (None, None)
    return ball_track


def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    list_det = [0 if x[0] else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]
    cursor = 0
    min_value = 0
    result = []
    for i, (k, l) in enumerate(groups):
        if (k == 1) & (i > 0) & (i < len(groups) - 1):
            dist = distance.euclidean(ball_track[cursor - 1], ball_track[cursor + l])
            if (l >= max_gap) | (dist / l > max_dist_gap):
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                    min_value = cursor + l - 1
        cursor += l
    if len(list_det) - min_value > min_track:
        result.append([min_value, len(list_det)])
    return result


def interpolation(coords):
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    x = np.array([p[0] if p[0] is not None else np.nan for p in coords])
    y = np.array([p[1] if p[1] is not None else np.nan for p in coords])
    if np.all(np.isnan(x)) or np.all(np.isnan(y)):
        return coords
    nons, yy = nan_helper(x)
    x[nons] = np.interp(yy(nons), yy(~nons), x[~nons])
    nans, xx = nan_helper(y)
    y[nans] = np.interp(xx(nans), xx(~nans), y[~nans])
    return [*zip(x, y)]


# --------------------------------------------------------------------------- #
# pass 2: draw the track and write the annotated video
# --------------------------------------------------------------------------- #
def write_track(video_path, ball_track, out_path, meta, trace=7, max_frames=None):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, meta["fps"], (meta["w"], meta["h"]))
    cap = cv2.VideoCapture(video_path)
    n = meta["n"] if max_frames is None else min(meta["n"], max_frames)
    for num in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        for i in range(trace):
            j = num - i
            if j > 0 and j < len(ball_track) and ball_track[j][0] is not None:
                x, y = int(ball_track[j][0]), int(ball_track[j][1])
                r = max(1, trace - i)
                cv2.circle(frame, (x, y), r, (0, 0, 255), -1)
            elif j > 0:
                break
        vw.write(frame)
        if num and num % 500 == 0:
            print(f"  [write] {num}/{n}", flush=True)
    cap.release()
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--video_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--extrapolation", action="store_true")
    ap.add_argument("--csv", type=str, default=None, help="optional path to dump per-frame coords")
    ap.add_argument("--max_frames", type=int, default=None, help="only process the first N frames")
    ap.add_argument("--trace", type=int, default=7)
    ap.add_argument("--no_amp", action="store_true", help="disable fp16 autocast (slower, more VRAM)")
    ap.add_argument("--min_radius", type=int, default=2,
                    help="HoughCircles min radius, model space (tennis=2)")
    ap.add_argument("--max_radius", type=int, default=7,
                    help="HoughCircles max radius, model space (tennis=7; table tennis ~12)")
    ap.add_argument("--outlier_dist", type=float, default=None,
                    help="spike threshold in frame pixels (default: 100 * frame_width/1280)")
    ap.add_argument("--gap_dist", type=float, default=None,
                    help="track-split distance in frame pixels (default: 80 * frame_width/1280)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    model = BallTrackerNet()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device).eval()
    amp = not args.no_amp

    meta = probe_video(args.video_path)
    n = meta["n"] if args.max_frames is None else min(meta["n"], args.max_frames)
    print(f"video: {args.video_path}")
    print(f"  {n} frames | {meta['w']}x{meta['h']} | {meta['fps']:.1f} fps | device={device}")

    t0 = time.time()
    ball_track, dists, meta = predict_track(
        args.video_path, model, device, args.batch_size, args.max_frames, amp=amp,
        min_radius=args.min_radius, max_radius=args.max_radius)
    dt = time.time() - t0
    print(f"  inference done: {len(ball_track)} frames in {dt:.1f}s ({len(ball_track)/max(dt,1e-6):.1f} fps)")

    # keep the classic thresholds meaningful at any frame width
    res_scale = meta["w"] / 1280.0
    max_dist = args.outlier_dist if args.outlier_dist is not None else 100 * res_scale
    max_dist_gap = args.gap_dist if args.gap_dist is not None else 80 * res_scale
    print(f"  scale-adjusted thresholds: outlier_dist={max_dist:.0f}px gap_dist={max_dist_gap:.0f}px")

    ball_track = remove_outliers(ball_track, dists, max_dist=max_dist)
    n_det = sum(1 for p in ball_track if p[0] is not None)
    print(f"  detections: {n_det}/{len(ball_track)} ({n_det/max(len(ball_track),1)*100:.1f}%)")

    if args.extrapolation:
        for r in split_track(ball_track, max_dist_gap=max_dist_gap):
            ball_track[r[0]:r[1]] = interpolation(ball_track[r[0]:r[1]])
        print("  extrapolation applied")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    write_track(args.video_path, ball_track, args.out_path, meta, args.trace, args.max_frames)
    print(f"  saved -> {args.out_path}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "x", "y", "detected"])
            for i, (x, y) in enumerate(ball_track):
                w.writerow([i, "" if x is None else round(float(x), 2),
                            "" if y is None else round(float(y), 2), 0 if x is None else 1])
        print(f"  coords -> {args.csv}")


if __name__ == "__main__":
    main()
