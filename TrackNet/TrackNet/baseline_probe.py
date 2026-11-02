"""Probe how a (tennis) TrackNet checkpoint behaves on table-tennis frames.

This is a *diagnostic*, not part of the pipeline. It answers one question:

    if we simply point the tennis model at table-tennis frames, where do the
    predictions land relative to the annotated ball?

It reads frames straight out of the source video (random access by frame no.),
runs the same 3-frame stack the training pipeline uses, and reports the
distance between prediction and ground truth per frame.

Usage
-----
python baseline_probe.py \
    --model_path weights/model_best.pt \
    --video      /path/game_4.mp4 \
    --ball_markup /path/ball_markup.json \
    --start 2464 --end 2487 \
    --min_radius 2 --max_radius 7
"""
import argparse
import json

import cv2
import numpy as np
import torch
from scipy.spatial import distance

from model import BallTrackerNet
from general import postprocess

MODEL_H, MODEL_W = 360, 640
SPACE_SCALE = 2          # postprocess returns coords in (H*2, W*2)


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
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--min_radius", type=int, default=2)
    ap.add_argument("--max_radius", type=int, default=7)
    ap.add_argument("--save_frames", default=None,
                    help="optional dir to dump annotated frames for visual inspection")
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
    sx, sy = W / (MODEL_W * SPACE_SCALE), H / (MODEL_H * SPACE_SCALE)

    print(f"video {W}x{H} | frames {args.start}..{args.end} | device={device}")
    print(f"hough radius [{args.min_radius}, {args.max_radius}] | coord scale ({sx:.3f}, {sy:.3f})")

    def grab(fn):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {fn}")
        return img

    if args.save_frames:
        import os
        os.makedirs(args.save_frames, exist_ok=True)

    hits, misses, no_det = 0, 0, 0
    dists = []
    # need 3 consecutive frames: use fn-2, fn-1, fn as the model input
    for fn in range(args.start + 2, args.end + 1):
        f3 = grab(fn)
        f2 = grab(fn - 1)
        f1 = grab(fn - 2)
        stack = [cv2.resize(im, (MODEL_W, MODEL_H)) for im in (f3, f2, f1)]
        inp = np.concatenate(stack, axis=2).astype(np.float32) / 255.0
        inp = np.rollaxis(inp, 2, 0)[None]
        with torch.no_grad():
            out = model(torch.from_numpy(inp).float().to(device))
        o = out.argmax(dim=1).detach().cpu().numpy()[0]
        px, py = postprocess(o, height=MODEL_H, width=MODEL_W,
                             min_radius=args.min_radius, max_radius=args.max_radius)
        gx, gy = marks.get(fn, (None, None))

        line = f"frame {fn}: "
        if px is not None and gx is not None:
            p = (px * sx, py * sy)
            d = distance.euclidean(p, (gx, gy))
            dists.append(d)
            line += f"pred=({p[0]:.0f},{p[1]:.0f}) gt=({gx},{gy}) dist={d:.1f}px"
            if d < 20:
                hits += 1
                line += "  HIT"
            else:
                misses += 1
        elif px is None:
            no_det += 1
            line += f"pred=NONE gt=({gx},{gy})" if gx else "pred=NONE gt=NONE"
        else:
            line += f"pred=({px*sx:.0f},{py*sy:.0f}) gt=NONE"
        print(line)

        if args.save_frames:
            vis = f3.copy()
            if gx is not None:
                cv2.circle(vis, (int(gx), int(gy)), 30, (0, 255, 0), 3)      # GT green
            if px is not None:
                cv2.circle(vis, (int(px * sx), int(py * sy)), 30, (0, 0, 255), 3)  # pred red
            cv2.imwrite(f"{args.save_frames}/{fn}.jpg", vis)

    cap.release()
    n = hits + misses
    print()
    print("================ SUMMARY ================")
    print(f"frames evaluated      : {n + no_det}")
    print(f"no detection          : {no_det}")
    if n:
        print(f"prediction within 20px: {hits}/{n} ({hits/n*100:.1f}%)")
        print(f"median distance       : {np.median(dists):.1f}px")
        print(f"mean distance         : {np.mean(dists):.1f}px")
    else:
        print("no comparable frames (every frame had a detection miss)")


if __name__ == "__main__":
    main()
