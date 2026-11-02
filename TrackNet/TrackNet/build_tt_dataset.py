"""Build the full TrackNet table-tennis dataset from OpenTTGames.

Runs the two existing tools back to back for every game that is present:

    prepare_openttgames.py   labels + frames   ->  trackNet/images/<game>/Clip1
    gt_gen.py --split_runs   heat-maps + split ->  trackNet/gts/... + labels_*.csv

It is idempotent: games whose Label.csv already exists are skipped unless
--force is given, so an interrupted run can simply be restarted.

Usage
-----
python build_tt_dataset.py                     # all games found
python build_tt_dataset.py --games game4,game5
python build_tt_dataset.py --games game5 --force
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

DATA_ROOT = "E:/bc/xm/ppq/_data/openttgames"
VIDEO_DIR = os.path.join(DATA_ROOT, "train_videos")
ANNOT_DIR = os.path.join(DATA_ROOT, "train_annotations")
OUT_ROOT = os.path.join(DATA_ROOT, "trackNet")

# heat-map parameters for table tennis (smaller, faster ball than tennis)
HEATMAP_SIZE = 10
HEATMAP_VAR = 5
FRAME_W = 1280          # frames are downscaled 1920 -> 1280
FRAME_H = 720


def run(cmd, title):
    print(f"\n{'='*70}\n{title}\n{'='*70}", flush=True)
    print(" ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {title}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="",
                    help="comma separated, e.g. game4,game5 (default: all present)")
    ap.add_argument("--force", action="store_true", help="rebuild even if Label.csv exists")
    ap.add_argument("--min_run", type=int, default=6)
    ap.add_argument("--train_rate", type=float, default=0.85)
    args = ap.parse_args()

    if args.games:
        games = [g.strip() for g in args.games.split(",") if g.strip()]
    else:
        games = [f"game{i}" for i in range(1, 6)]

    # only keep games that have BOTH the video and the extracted annotations
    usable = []
    for g in games:
        vid = os.path.join(VIDEO_DIR, f"{g}.mp4")
        ann = os.path.join(ANNOT_DIR, f"{g}_extracted", "ball_markup.json")
        if os.path.exists(vid) and os.path.exists(ann):
            # the dataset folder has always been named without the underscore
            # (game4/, not game_4/) and labels_*.csv reference those paths
            usable.append(g)
        else:
            print(f"skip {g}: video={os.path.exists(vid)} annotation={os.path.exists(ann)}")

    if not usable:
        raise SystemExit("no usable games found")

    def gname(g):
        """game_4 -> game4 (matches the existing dataset layout)."""
        return g.replace("_", "")

    print(f"building dataset for: {', '.join(usable)}")

    # ---- step 1: frames + Label.csv -------------------------------------- #
    for g in usable:
        clip_dir = os.path.join(OUT_ROOT, "images", gname(g), "Clip1")
        label = os.path.join(clip_dir, "Label.csv")
        if os.path.exists(label) and not args.force:
            n = len(os.listdir(clip_dir)) - 1
            print(f"\n{g}: already has {n} frames + Label.csv, skip (use --force to redo)")
            continue
        run([PY, "prepare_openttgames.py",
             "--ball_markup", os.path.join(ANNOT_DIR, f"{g}_extracted", "ball_markup.json"),
             "--video", os.path.join(VIDEO_DIR, f"{g}.mp4"),
             "--out_root", OUT_ROOT,
             "--game", gname(g), "--clip", "Clip1",
             "--min_run", str(args.min_run),
             "--resize_w", str(FRAME_W)],
            f"prepare {g}")

    # ---- step 2: heat-maps + train/val split ----------------------------- #
    present = [gname(g) for g in usable
               if os.path.exists(os.path.join(OUT_ROOT, "images", gname(g), "Clip1", "Label.csv"))]
    run([PY, "gt_gen.py",
         "--path_input", os.path.join(OUT_ROOT, "images"),
         "--path_output", OUT_ROOT,
         "--size", str(HEATMAP_SIZE),
         "--variance", str(HEATMAP_VAR),
         "--width", str(FRAME_W),
         "--height", str(FRAME_H),
         "--games", ",".join(present),
         "--split_runs",
         "--train_rate", str(args.train_rate)],
        f"heat-maps + split for {','.join(present)}")

    print("\nDONE")
    for f in ("labels_train.csv", "labels_val.csv"):
        p = os.path.join(OUT_ROOT, f)
        if os.path.exists(p):
            with open(p) as fh:
                print(f"  {f}: {sum(1 for _ in fh) - 1} samples")


if __name__ == "__main__":
    main()
