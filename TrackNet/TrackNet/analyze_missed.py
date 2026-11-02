"""Analyse WHAT the missed annotated frames look like (CPU only).

Joins outputs/diag_recall_*.csv with ball_markup.json and answers:
  1. where in an annotation run does the miss happen (head / middle / tail)?
  2. are misses isolated single frames (recoverable by interpolation) or whole
     stretches (the ball is genuinely gone)?
  3. how fast was the ball, and where on the image was it?
  4. does the network still fire a peak there (gt_peak), i.e. "sees but cannot
     postprocess" vs "blind"?

Usage:
    python analyze_missed.py --diag outputs/diag_recall_g345.csv \
        --markup E:/.../test_2_extracted/ball_markup.json
"""
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd


def runs_of(frames):
    out, cur = [], [frames[0]]
    for a, b in zip(frames, frames[1:]):
        if b - a == 1:
            cur.append(b)
        else:
            out.append(cur)
            cur = [b]
    out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag', required=True)
    ap.add_argument('--markup', required=True)
    ap.add_argument('--min_dist', type=float, default=20.0)
    args = ap.parse_args()

    df = pd.read_csv(args.diag)
    with open(args.markup) as f:
        m = json.load(f)
    gt = {int(k): (float(v['x']), float(v['y']))
          for k, v in m.items() if v.get('x') is not None}
    ann_frames = sorted(k for k in gt if k < len(df))

    d = df.set_index('frame')
    det = {int(f): bool(d.loc[f, 'detected']) for f in ann_frames}
    peak = {int(f): float(d.loc[f, 'gt_peak']) for f in ann_frames}
    gdist = {int(f): float(d.loc[f, 'gt_dist']) for f in ann_frames}

    hit = {f for f in ann_frames
           if det[f] and 0 <= gdist[f] < args.min_dist}
    nodet = [f for f in ann_frames if not det[f]]

    print(f'annotated frames : {len(ann_frames)}   hit {len(hit)}   no_det {len(nodet)}')

    # ---- 1. position inside the annotation run ------------------------- #
    runs = runs_of(ann_frames)
    print(f'\nruns (annotation segments) : {len(runs)}  '
          f'len {min(len(r) for r in runs)}~{max(len(r) for r in runs)}')

    print('\n--- 1. miss rate by frame index inside the run ---')
    print(f'{"idx":>4} {"n":>5} {"miss":>5} {"miss%":>7}')
    by_idx = {}
    for r in runs:
        for i, f in enumerate(r):
            by_idx.setdefault(i, [0, 0])
            by_idx[i][0] += 1
            by_idx[i][1] += int(f not in hit)
    for i in sorted(by_idx):
        n, ms = by_idx[i]
        print(f'{i:>4} {n:>5} {ms:>5} {ms / n * 100:>6.1f}%')

    def zone(i, L):
        t = i / max(L - 1, 1)
        return 'head' if t <= 0.25 else ('tail' if t >= 0.75 else 'middle')

    zs = Counter()
    zn = Counter()
    for r in runs:
        L = len(r)
        for i, f in enumerate(r):
            z = zone(i, L)
            zn[z] += 1
            zs[z] += int(f not in hit)
    print('\n--- 1b. by zone ---')
    for z in ('head', 'middle', 'tail'):
        if zn[z]:
            print(f'{z:<8} n={zn[z]:<4} miss={zs[z]:<4} miss rate={zs[z] / zn[z] * 100:5.1f}%')

    # ---- 2. isolated vs consecutive misses ----------------------------- #
    print('\n--- 2. consecutive-miss streak lengths ---')
    streak, cur = [], 0
    for f in ann_frames:
        if f not in hit:
            cur += 1
        elif cur:
            streak.append(cur)
            cur = 0
    if cur:
        streak.append(cur)
    cnt = Counter(streak)
    print(f'streaks={len(streak)}  total missed frames={sum(streak)}')
    for k in sorted(cnt):
        print(f'  length {k:>2} : {cnt[k]:>3} streak(s)  ({cnt[k] * k:>3} frames)')

    # how many misses sit between two hits -> recoverable by interpolation
    recov = 0
    for r in runs:
        for i, f in enumerate(r):
            if f in hit:
                continue
            left = any(x in hit for x in r[:i])
            right = any(x in hit for x in r[i + 1:])
            if left and right:
                recov += 1
    print(f'\nmisses flanked by hits on BOTH sides (interpolatable) : '
          f'{recov} / {len(nodet)} = {recov / max(len(nodet), 1) * 100:.1f}%')

    # ---- 3. ball speed and image position ----------------------------- #
    print('\n--- 3. GT ball speed (px/frame @1920) ---')
    spd = {}
    for r in runs:
        for i, f in enumerate(r):
            if i == 0 or r[i - 1] != f - 1:
                spd[f] = np.nan
            else:
                p, q = gt[r[i - 1]], gt[f]
                spd[f] = float(np.hypot(q[0] - p[0], q[1] - p[1]))
    for name, grp in (('hit', hit), ('no_det', set(nodet))):
        v = np.array([spd[f] for f in grp if not np.isnan(spd.get(f, np.nan))])
        if len(v):
            print(f'{name:<7} n={len(v):<4} median={np.median(v):6.1f} '
                  f'p25={np.percentile(v, 25):6.1f} p75={np.percentile(v, 75):6.1f} '
                  f'max={v.max():6.1f}')

    print('\n--- 4. GT position on the image (1920x1080) ---')
    for name, grp in (('hit', hit), ('no_det', set(nodet))):
        pts = np.array([gt[f] for f in grp])
        if len(pts):
            edge = ((pts[:, 0] < 100) | (pts[:, 0] > 1820) |
                    (pts[:, 1] < 60) | (pts[:, 1] > 1020)).sum()
            print(f'{name:<7} n={len(pts):<4} x median={np.median(pts[:, 0]):6.0f} '
                  f'y median={np.median(pts[:, 1]):6.0f}  near-edge(<100px)={edge}')

    print('\n--- 5. does the network still see the ball? (gt_peak) ---')
    for name, grp in (('hit', hit), ('no_det', set(nodet))):
        v = np.array([peak[f] for f in grp])
        if len(v):
            print(f'{name:<7} n={len(v):<4} median={np.median(v):5.0f} '
                  f'>=128: {(v >= 128).sum():<4} ({(v >= 128).mean() * 100:4.1f}%)  '
                  f'0 exactly: {(v == 0).sum()}')
    v = np.array([peak[f] for f in nodet])
    if len(v):
        print(f'\nno_det with a peak >=128 ({int((v >= 128).sum())} frames) '
              f'-> network localised it, postprocess dropped it')
        print(f'no_det with peak == 0 ({int((v == 0).sum())} frames) '
              f'-> network blind, only temporal extrapolation can help')


if __name__ == '__main__':
    main()
