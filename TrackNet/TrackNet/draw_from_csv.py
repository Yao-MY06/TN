"""Re-draw a tracking video from an existing CSV (no inference, CPU only).

Lets us render the interpolated track without re-running the network, and keeps
interpolated frames visually distinguishable (hollow ring instead of filled dot).

Usage:
    python draw_from_csv.py --csv outputs/xxx_filled.csv \
        --video_path E:/.../test_2.mp4 --out_path outputs/xxx_filled.mp4 \
        --max_frames 3600
"""
import argparse

import cv2
import pandas as pd

from track_video import probe_video, write_track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--video_path', required=True)
    ap.add_argument('--out_path', required=True)
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--trace', type=int, default=7)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.max_frames:
        df = df.head(args.max_frames)
    track = [(float(r.x), float(r.y)) if r.detected == 1 else (None, None)
             for r in df.itertuples()]
    meta = probe_video(args.video_path)
    write_track(args.video_path, track, args.out_path, meta,
                trace=args.trace, max_frames=args.max_frames)
    n_det = sum(1 for p in track if p[0] is not None)
    print(f'frames={len(track)}  detected={n_det} ({n_det / len(track) * 100:.1f}%)')
    print(f'saved -> {args.out_path}')


if __name__ == '__main__':
    main()
