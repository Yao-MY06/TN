"""OpenTTGames (OSAI) table-tennis annotations -> TrackNet training format.

The OpenTTGames ball markup is *event-centred*: ball coordinates only exist for
short windows around each annotated event (roughly 4 frames before / 12 after).
This script turns those sparse windows into a TrackNet-ready dataset:

    openttgames images (extracted frames) + ball_markup.json
        ->  datasets/trackNet/images/<game>/<clip>/<frame>.png
        ->  datasets/trackNet/images/<game>/<clip>/Label.csv
            (file name, visibility, x, y, status)

Only *contiguous* runs of ball annotation are exported, so every emitted clip is
a valid series with no temporal holes (TrackNet needs 3 consecutive frames).

Usage
-----
# 1) from a folder of already-extracted frames (one flat dir or per-frame names)
python prepare_openttgames.py \
    --ball_markup  path/to/ball_markup.json \
    --frames_dir   path/to/extracted_frames \
    --out_root     datasets/trackNet \
    --game game1 --clip Clip1

# 2) or read frames straight out of the source video (random access by frame no.)
python prepare_openttgames.py \
    --ball_markup  path/to/ball_markup.json \
    --video        path/to/test_1.mp4 \
    --out_root     datasets/trackNet \
    --game game1 --clip Clip1 \
    --min_run 6 --max_gap 1

Notes
-----
* `--min_run` drops annotation runs shorter than N frames (too short to be useful
  for 3-frame training and prone to label noise).
* `visibility` follows the TrackNet convention: 1 = ball visible.
  OpenTTGames uses (-1, -1) to mean "ball absent"; those frames are kept with
  visibility 0 so the model can learn to *not* fire.
* Frames are written as PNG because OpenTTGames is a 120 fps industrial capture
  where JPEG compression hurts the very small ball.
"""
import os
import csv
import json
import argparse

import cv2


# --------------------------------------------------------------------------- #
def load_ball_markup(path):
    """-> {frame_no(int): (x, y, visibility)}"""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        fn = int(k)
        x, y = v.get("x"), v.get("y")
        if x in (-1, None) or y in (-1, None):
            out[fn] = (None, None, 0)          # ball absent
        else:
            out[fn] = (x, y, 1)
    return out


def contiguous_runs(frames, max_gap=1):
    """Group sorted frame numbers into runs allowing small gaps.

    max_gap=1 -> strictly consecutive.
    """
    fs = sorted(frames)
    if not fs:
        return []
    runs, start, prev = [], fs[0], fs[0]
    for f in fs[1:]:
        if f - prev <= max_gap:
            prev = f
            continue
        runs.append((start, prev))
        start = prev = f
    runs.append((start, prev))
    return runs


def frame_reader(frames_dir, video_path):
    """Return a callable frame_no -> BGR image (or None)."""
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")

        # Frame numbers are requested in ascending order, so scan forward with
        # grab() instead of seeking.  CAP_PROP_POS_FRAMES on a 1080p@120fps
        # video has to decode from the previous keyframe every time: on
        # game_3.mp4 a random seek costs ~30 ms, which makes a 7k-frame
        # extraction take hours.  grab() only decodes the container, so a full
        # 74k-frame pass takes a couple of minutes.
        pos = {"next": 0}

        def read(fn):
            if fn < pos["next"]:                    # backwards: must seek
                cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                pos["next"] = fn
            while pos["next"] < fn:
                if not cap.grab():
                    return None
                pos["next"] += 1
            ok, img = cap.read()
            if ok:
                pos["next"] += 1
            return img if ok else None

        return read, cap

    # build an index of available frame files
    files = {}
    for name in os.listdir(frames_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            if stem.lstrip("-").isdigit():
                files[int(stem)] = os.path.join(frames_dir, name)
    if not files:
        # flat dir of sequential images (0.png, 1.png ...) already handled above
        raise RuntimeError(f"no numerically-named frames found in {frames_dir}")

    def read(fn):
        p = files.get(fn)
        if p is None:
            return None
        return cv2.imread(p)

    return read, None


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ball_markup", required=True, help="path to ball_markup.json")
    ap.add_argument("--frames_dir", default=None, help="folder with extracted frames")
    ap.add_argument("--video", default=None, help="source video (alternative to frames_dir)")
    ap.add_argument("--out_root", required=True, help="TrackNet datasets/trackNet root")
    ap.add_argument("--game", required=True, help="game folder name, e.g. game1")
    ap.add_argument("--clip", required=True, help="clip folder name, e.g. Clip1")
    ap.add_argument("--min_run", type=int, default=6, help="drop annotation runs shorter than N")
    ap.add_argument("--max_gap", type=int, default=1, help="allow gaps up to N frames within a run")
    ap.add_argument("--start_index", type=int, default=0,
                    help="first output frame number (TrackNet convention starts at 0)")
    ap.add_argument("--jpeg", action="store_true", help="write JPEG instead of PNG")
    ap.add_argument("--resize_w", type=int, default=0,
                    help="if >0, downscale frames to this width (keeps aspect); coordinates "
                         "are scaled accordingly. OpenTTGames is 1920x1080 and TrackNet only "
                         "needs 640x360, so 1280 saves ~55%% I/O with no accuracy loss.")
    args = ap.parse_args()

    if not args.frames_dir and not args.video:
        raise SystemExit("need --frames_dir or --video")

    marks = load_ball_markup(args.ball_markup)
    print(f"loaded {len(marks)} annotated frames "
          f"({sum(1 for v in marks.values() if v[2])} with ball)")

    runs = [r for r in contiguous_runs(marks.keys(), args.max_gap)
            if (r[1] - r[0] + 1) >= args.min_run]
    print(f"kept {len(runs)} runs (>= {args.min_run} frames)")

    clip_dir = os.path.join(args.out_root, "images", args.game, args.clip)
    os.makedirs(clip_dir, exist_ok=True)

    read_frame, cap = frame_reader(args.frames_dir, args.video)
    ext = ".jpg" if args.jpeg else ".png"

    # optional downscale (coordinates follow the same scale factor)
    scale = None
    if args.resize_w > 0 and cap is not None:
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if args.resize_w != src_w:
            scale = args.resize_w / float(src_w)
            out_w, out_h = args.resize_w, int(round(src_h * scale))
            print(f"downscaling {src_w}x{src_h} -> {out_w}x{out_h} (x{scale:.4f})")

    rows = []
    idx = args.start_index
    written = 0
    missing = 0
    for run_id, (a, b) in enumerate(runs):
        for fn in range(a, b + 1):
            img = read_frame(fn)
            if img is None:
                missing += 1
                continue
            if scale is not None:
                img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
            name = f"{idx}{ext}"
            cv2.imwrite(os.path.join(clip_dir, name), img)
            x, y, vis = marks.get(fn, (None, None, 0))
            if x is not None and scale is not None:
                x, y = round(x * scale, 2), round(y * scale, 2)
            x_s = "" if x is None else x
            y_s = "" if y is None else y
            rows.append([name, vis, x_s, y_s, 0, run_id])
            idx += 1
            written += 1

    if cap is not None:
        cap.release()

    label_path = os.path.join(clip_dir, "Label.csv")
    with open(label_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # run_id marks which independent frame-run a row belongs to; gt_gen.py
        # --split_runs uses it so no 3-frame sample straddles a temporal gap.
        w.writerow(["file name", "visibility", "x-coordinate", "y-coordinate", "status", "run_id"])
        w.writerows(rows)

    print(f"wrote {written} frames -> {clip_dir}")
    print(f"wrote {len(rows)} rows  -> {label_path}")
    if missing:
        print(f"WARNING: {missing} frames unavailable (video shorter than annotation?)")


if __name__ == "__main__":
    main()
