"""Evaluate a TrackNet checkpoint on OpenTTGames test videos.

Two modes:

  --mode range   evaluate only on annotated frames (fast, gives the "can it
                 find the ball when the ball is actually there" number)
  --mode video   run the full video and report the overall detection rate
                 (biased by long stretches where the ball is absent)

The metric definition mirrors evaluate.py / general.validate() so numbers are
comparable with the tennis reproduction:

    hit  = a ball is detected AND within `min_dist` px of the annotation
    miss = ball detected but too far, OR no detection while the ball is there
    false= ball detected while the annotation says the ball is absent

Usage
-----
python eval_tt_video.py --model_path weights/tt_game4_best.pt \
    --video E:/.../test_videos/test_2.mp4 \
    --ball_markup E:/.../test_annotations/test_2_extracted/ball_markup.json \
    --max_frames 3000 --mode range
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
from scipy.spatial import distance

from model import BallTrackerNet
from general import postprocess

MODEL_H, MODEL_W = 360, 640
SPACE_SCALE = 2


def load_markup(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        x, y = v.get("x"), v.get("y")
        out[int(k)] = (None, None) if x in (-1, None) or y in (-1, None) else (x, y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--ball_markup", required=True)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--min_dist", type=float, default=20.0,
                    help="hit radius in FRAME pixels (frames are 1920x1080 here)")
    ap.add_argument("--min_radius", type=int, default=2)
    ap.add_argument("--max_radius", type=int, default=12)
    ap.add_argument("--mode", choices=["range", "video"], default="range")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BallTrackerNet()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device).eval()

    marks = load_markup(args.ball_markup)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        n_total = min(n_total, args.max_frames)
    sx, sy = W / (MODEL_W * SPACE_SCALE), H / (MODEL_H * SPACE_SCALE)

    print(f"{os.path.basename(args.video)}  {W}x{H}  {n_total} frames  mode={args.mode}")
    print(f"model={os.path.basename(args.model_path)}  hough=[{args.min_radius},{args.max_radius}]")

    def grab(fn):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ok, img = cap.read()
        return img if ok else None

    hit = miss = false_pos = no_det = 0
    errs = []

    # need frames fn-2, fn-1, fn
    fn = 2
    while fn < n_total:
        f3 = grab(fn)
        f2 = grab(fn - 1)
        f1 = grab(fn - 2)
        if f3 is None or f2 is None or f1 is None:
            break
        stack = [cv2.resize(im, (MODEL_W, MODEL_H)) for im in (f3, f2, f1)]
        inp = np.concatenate(stack, axis=2).astype(np.float32) / 255.0
        inp = np.rollaxis(inp, 2, 0)[None]
        with torch.no_grad():
            t = torch.from_numpy(inp).float().to(device)
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(t)
                out = out.float()
            else:
                out = model(t)
        o = out.argmax(dim=1).detach().cpu().numpy()[0]
        px, py = postprocess(o, height=MODEL_H, width=MODEL_W,
                             min_radius=args.min_radius, max_radius=args.max_radius)

        gx, gy = marks.get(fn, (None, None))
        if args.mode == "range" and gx is None:
            fn += 1
            continue

        if px is not None:
            p = (px * sx, py * sy)
            if gx is not None:
                d = distance.euclidean(p, (gx, gy))
                errs.append(d)
                if d <= args.min_dist:
                    hit += 1
                else:
                    miss += 1
            else:
                false_pos += 1
        else:
            if gx is not None:
                no_det += 1

        if fn % 500 == 0:
            print(f"  {fn}/{n_total}", flush=True)
        fn += 1

    cap.release()

    eps = 1e-9
    if args.mode == "range":
        tp, fp, fn_ = hit, miss, no_det       # false_pos not counted: mode=range skips absent frames
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn_ + eps)
    else:
        tp, fp, fn_ = hit, miss + false_pos, no_det
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn_ + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    res = {
        "video": os.path.basename(args.video),
        "model": os.path.basename(args.model_path),
        "mode": args.mode,
        "frames_evaluated": fn,
        "hit": hit, "miss": miss, "false_pos": false_pos, "no_detection": no_det,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "median_err_px": round(float(np.median(errs)), 2) if errs else None,
        "mean_err_px": round(float(np.mean(errs)), 2) if errs else None,
        "min_dist": args.min_dist,
    }
    print("\n================ RESULT ================")
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
