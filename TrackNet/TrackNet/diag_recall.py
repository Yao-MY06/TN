"""Diagnose WHY annotated frames are missed (no detection).

For every annotated frame we record:
  * gt_peak   : max value of the argmax heatmap in a small window around the GT
                centre (model space, 0-255). High = the network DID localise the
                ball, so the miss comes from postprocess (HoughCircles).
  * glob_peak : max value of the whole argmax heatmap.
  * detected  : whether postprocess returned a circle.

Then we split the annotated frames into hit / miss / no_detection and compare
the gt_peak distributions. That tells us whether the recall bottleneck is the
network or the postprocess -- the answer decides if more training is worth it.

Usage:
    python diag_recall.py --model exps/tt_g45_aug/model_best.pt \
        --video E:/bc/xm/ppq/_data/openttgames/test_videos/test_2.mp4 \
        --markup E:/bc/xm/ppq/_data/openttgames/test_annotations/test_2_extracted/ball_markup.json \
        --max_frames 3600
"""
import argparse
import json
from collections import deque

import cv2
import numpy as np
import torch

from general import postprocess
from model import BallTrackerNet

MODEL_H, MODEL_W = 360, 640
SPACE_SCALE = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--markup', required=True)
    ap.add_argument('--max_frames', type=int, default=3600)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--win', type=int, default=4,
                    help='half window (model px) around GT when reading gt_peak')
    ap.add_argument('--min_radius', type=int, default=2)
    ap.add_argument('--max_radius', type=int, default=12)
    ap.add_argument('--min_dist', type=float, default=20.0,
                    help='frame-space px, same as analyze_track_csv')
    ap.add_argument('--out_csv', default='outputs/diag_recall.csv')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BallTrackerNet()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device).eval()

    with open(args.markup) as f:
        marks = json.load(f)
    gt = {int(k): (float(v['x']), float(v['y']))
          for k, v in marks.items() if v.get('x') is not None}

    cap = cv2.VideoCapture(args.video)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = vw / (MODEL_W * SPACE_SCALE), vh / (MODEL_H * SPACE_SCALE)

    buf = deque(maxlen=3)
    rows = []
    pending, pending_idx = [], []

    def flush():
        if not pending:
            return
        inp = torch.from_numpy(np.stack(pending)).float().to(device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16,
                                             enabled=device == 'cuda'):
            out = model(inp)
        outs = out.float().argmax(dim=1).detach().cpu().numpy()
        for k, o in enumerate(outs):
            fn = pending_idx[k]
            # the head flattens the spatial dims -> (230400,); restore 2D for peeking
            o2d = o.reshape(MODEL_H, MODEL_W)
            x, y = postprocess(o.copy(), height=MODEL_H, width=MODEL_W,
                               min_radius=args.min_radius, max_radius=args.max_radius)
            detected = x is not None
            if detected:
                x *= sx
                y *= sy
            glob_peak = int(o2d.max())
            gt_peak, gt_dist = -1, -1.0
            if fn in gt:
                gx, gy = gt[fn]
                # frame space -> model space (postprocess output is *2 of argmax space)
                mx, my = gx / sx / SPACE_SCALE, gy / sy / SPACE_SCALE
                ix, iy = int(round(mx)), int(round(my))
                w = args.win
                x0, x1 = max(0, ix - w), min(MODEL_W, ix + w + 1)
                y0, y1 = max(0, iy - w), min(MODEL_H, iy + w + 1)
                if x0 < x1 and y0 < y1:
                    gt_peak = int(o2d[y0:y1, x0:x1].max())
                if detected:
                    gt_dist = float(np.hypot(x - gx, y - gy))
            rows.append(dict(frame=fn, detected=int(detected),
                             glob_peak=glob_peak, gt_peak=gt_peak,
                             gt_dist=gt_dist, annotated=int(fn in gt)))
        pending.clear()
        pending_idx.clear()

    cur = 0
    while cur < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(cv2.resize(frame, (MODEL_W, MODEL_H)))
        if len(buf) == 3:
            img = np.concatenate((buf[2], buf[1], buf[0]), axis=2).astype(np.float32) / 255.0
            pending.append(np.rollaxis(img, 2, 0))
            pending_idx.append(cur)
            if len(pending) >= args.batch_size:
                flush()
        cur += 1
    flush()
    cap.release()

    import csv
    with open(args.out_csv, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # ---- classify annotated frames ------------------------------------- #
    ann = [r for r in rows if r['annotated']]
    hit = [r for r in ann if r['detected'] and 0 <= r['gt_dist'] < args.min_dist]
    far = [r for r in ann if r['detected'] and r['gt_dist'] >= args.min_dist]
    nodet = [r for r in ann if not r['detected']]

    def stat(name, rs):
        if not rs:
            print(f'{name:<14}: n=0')
            return
        gp = np.array([r['gt_peak'] for r in rs])
        gb = np.array([r['glob_peak'] for r in rs])
        print(f'{name:<14}: n={len(rs):<4} gt_peak median={np.median(gp):5.0f} '
              f'mean={gp.mean():6.1f} p25={np.percentile(gp, 25):5.0f} '
              f'p75={np.percentile(gp, 75):5.0f} | share>=128: {(gp >= 128).mean() * 100:5.1f}% '
              f'| glob_peak median={np.median(gb):5.0f}')

    print(f'\nframes written : {len(rows)}  ({args.out_csv})')
    print(f'annotated      : {len(ann)}')
    print(f'  hit(<{args.min_dist:g}px)  : {len(hit)}')
    print(f'  far(off)     : {len(far)}')
    print(f'  no_detection : {len(nodet)}')
    print('\ngt_peak distribution (argmax heatmap value near the GT, 0-255):')
    stat('hit', hit)
    stat('far', far)
    stat('no_det', nodet)
    stat('all annotated', ann)


if __name__ == '__main__':
    main()
