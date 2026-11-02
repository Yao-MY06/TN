"""Constrained gap filling for a track_video.py CSV (CPU only).

Why not track_video.py --extrapolation: its `interpolation()` is a bare
np.interp over every NaN, so it (a) bridges arbitrarily long gaps with a
straight line and (b) paints the end-point value outside the observed range.
On a 30%-detection-rate track that injects far more wrong points than right
ones.

This version only fills gaps that are physically plausible:
  * both ends of the gap must have a real detection,
  * the gap must be <= --max_gap frames,
  * the interpolated speed must stay <= --max_speed px/frame,
  * the gap must not look like a track switch (turn angle check).

Filled points are flagged in the extra `interp` column so downstream metrics
can be reported with and without them.

Usage:
    python interpolate_track.py --csv outputs/xxx.csv --out outputs/xxx_filled.csv
"""
import argparse

import numpy as np
import pandas as pd


def fill(df, max_gap, max_speed, mode='linear'):
    x = df['x'].to_numpy(dtype=float).copy()
    y = df['y'].to_numpy(dtype=float).copy()
    det = df['detected'].to_numpy(dtype=int).copy()
    interp = np.zeros(len(df), dtype=int)

    idx = np.where(det == 1)[0]
    if len(idx) < 2:
        return df, interp, dict(gaps=0, filled=0, skipped_far=0, skipped_speed=0,
                                skipped_turn=0)

    stats = dict(gaps=0, filled=0, skipped_far=0, skipped_speed=0, skipped_turn=0)

    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if gap <= 0:
            continue
        stats['gaps'] += 1
        if gap > max_gap:
            stats['skipped_far'] += 1
            continue

        ax, ay = x[a], y[a]
        bx, by = x[b], y[b]
        span = b - a

        # straight-line speed across the gap
        if np.hypot(bx - ax, by - ay) / span > max_speed:
            stats['skipped_speed'] += 1
            continue

        # turn-angle sanity: direction before the gap vs across the gap
        if a - 1 >= 0 and det[a - 1] == 1:
            v0 = np.array([ax - x[a - 1], ay - y[a - 1]])
            v1 = np.array([bx - ax, by - ay])
            n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
            if n0 > 1e-6 and n1 > 1e-6:
                cos = np.dot(v0, v1) / (n0 * n1)
                if cos < 0.0:          # >90 deg flip => probably another ball/segment
                    stats['skipped_turn'] += 1
                    continue

        for f in range(a + 1, b):
            t = (f - a) / span
            if mode == 'quad' and a - 1 >= 0 and b + 1 < len(df) \
                    and det[a - 1] == 1 and det[b + 1] == 1:
                # quadratic through (a-1, a, b) evaluated at f
                ts = np.array([a - 1, a, b], dtype=float)
                fx = np.polyfit(ts, np.array([x[a - 1], ax, bx]), 2)
                fy = np.polyfit(ts, np.array([y[a - 1], ay, by]), 2)
                px, py = np.polyval(fx, f), np.polyval(fy, f)
            else:
                px = ax + (bx - ax) * t
                py = ay + (by - ay) * t
            x[f], y[f] = px, py
            det[f] = 1
            interp[f] = 1
            stats['filled'] += 1

    out = df.copy()
    out['x'], out['y'], out['detected'] = x, y, det
    return out, interp, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_gap', type=int, default=12,
                    help='only bridge gaps up to this many frames (120fps: 12 = 0.1s)')
    ap.add_argument('--max_speed', type=float, default=120.0,
                    help='px per frame at frame resolution (median ball speed ~24)')
    ap.add_argument('--mode', choices=['linear', 'quad'], default='linear')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for c in ('x', 'y'):
        df[c] = pd.to_numeric(df[c], errors='coerce')

    out, interp, stats = fill(df, args.max_gap, args.max_speed, args.mode)
    out['interp'] = interp
    out.to_csv(args.out, index=False)

    before = int(df['detected'].sum())
    after = int(out['detected'].sum())
    print(f'csv            : {args.csv}')
    print(f'filled csv     : {args.out}')
    print(f'mode           : {args.mode}  max_gap={args.max_gap}  max_speed={args.max_speed}')
    print(f'gaps seen      : {stats["gaps"]}')
    print(f'  skipped far  : {stats["skipped_far"]}')
    print(f'  skipped speed: {stats["skipped_speed"]}')
    print(f'  skipped turn : {stats["skipped_turn"]}')
    print(f'frames filled  : {stats["filled"]}')
    print(f'detected       : {before} -> {after}  ({after / len(out) * 100:.1f}%)')


if __name__ == '__main__':
    main()
