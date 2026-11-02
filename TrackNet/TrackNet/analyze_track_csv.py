"""Compute detection metrics from a track_video.py CSV + OpenTTGames ball_markup.json.

The CSV (frame, x, y, detected) comes from track_video.py at native frame
resolution (1920x1080 here), same space as ball_markup.json.
Metrics (min_dist=20 px in frame space, mirroring eval_tt_video.py):

  overall detection rate : detected frames / total frames (whole CSV)
  on annotated frames    : hit  = detected and dist(pred, gt) < min_dist
                           miss = detected but dist >= min_dist
                           no_detection = no detection on an annotated frame
  precision / recall / F1 computed on the annotated subset
  error stats            : median / mean of dist over hits

Usage:
    python analyze_track_csv.py --csv outputs/xx.csv --markup path/ball_markup.json [--min_dist 20]
"""
import argparse
import json

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--markup', required=True)
    ap.add_argument('--min_dist', type=float, default=20.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    with open(args.markup) as f:
        marks = json.load(f)
    gt = {int(k): (float(v['x']), float(v['y']))
          for k, v in marks.items() if v.get('x') is not None}

    n = len(df)
    det = int(df['detected'].sum())
    print(f'frames            : {n}')
    print(f'detected          : {det} ({det / n * 100:.1f}%)  <- overall detection rate')

    hit = miss = no_det = 0
    errs = []
    for fn, (gx, gy) in gt.items():
        if fn >= n:
            continue
        row = df[df['frame'] == fn]
        if row.empty or int(row['detected'].iloc[0]) == 0:
            no_det += 1
            continue
        px, py = float(row['x'].iloc[0]), float(row['y'].iloc[0])
        d = float(np.hypot(px - gx, py - gy))
        if d < args.min_dist:
            hit += 1
            errs.append(d)
        else:
            miss += 1

    total_ann = hit + miss + no_det
    prec = hit / (hit + miss) if hit + miss else 0.0
    rec = hit / total_ann if total_ann else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    errs = np.array(errs)
    print(f'annotated frames  : {total_ann}')
    print(f'hit               : {hit}')
    print(f'miss (far off)    : {miss}')
    print(f'no detection      : {no_det}')
    print(f'precision         : {prec:.4f}   (on annotated frames)')
    print(f'recall            : {rec:.4f}   (on annotated frames)')
    print(f'f1                : {f1:.4f}')
    if len(errs):
        print(f'error median      : {np.median(errs):.2f} px')
        print(f'error mean        : {errs.mean():.2f} px')


if __name__ == '__main__':
    main()
