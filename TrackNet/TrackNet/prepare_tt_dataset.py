"""Convert table-tennis annotations (TTNet / OSAI format) into TrackNet dataset format.

TTNet `ball_markup.json`  : {"<frame_index>": {"x": int, "y": int}, ...}
TTNet `events_markup.json`: {"<frame_index>": "<bounce|net|...>", ...}

This tool builds the layout TrackNet expects:

    datasets/trackNet/
        images/<game>/<clip>/0000.jpg ...
        gts/<game>/<clip>/0000.jpg ...      (produced afterwards by gt_gen.py)
        Label.csv                           (one per clip, inside images/<game>/<clip>/)

NOTE: TTNet releases only ball coordinates + segmentation masks, NOT video frames.
      You must supply the source video (see --video) or your own table-tennis clip.

Usage
-----
# 1) with a video: extract the annotated frames and write Label.csv
python prepare_tt_dataset.py --markup_dir Train/game_1 --video tt.mp4 \
       --out_root datasets/trackNet --game game1 --clip Clip1

# 2) frames already extracted on disk (named <idx>.jpg) -> only write Label.csv
python prepare_tt_dataset.py --markup_dir Train/game_1 --frames_dir my_frames \
       --out_root datasets/trackNet --game game1 --clip Clip1
"""
import argparse
import csv
import json
import os

import cv2


def load_markup(markup_dir):
    """Return {frame_index(int): {'x':int,'y':int,'event':str|None}}."""
    ball_path = os.path.join(markup_dir, "ball_markup.json")
    with open(ball_path, encoding="utf-8") as f:
        ball = json.load(f)

    events = {}
    ev_path = os.path.join(markup_dir, "events_markup.json")
    if os.path.exists(ev_path):
        with open(ev_path, encoding="utf-8") as f:
            events = json.load(f)

    out = {}
    for k, v in ball.items():
        try:
            idx = int(k)
        except ValueError:
            continue
        if isinstance(v, dict) and v.get("x") is not None and v.get("y") is not None:
            out[idx] = {"x": float(v["x"]), "y": float(v["y"]), "event": events.get(k)}
    return out


def extract_from_video(video_path, markup, dst_dir):
    """Save only the annotated frames from a video into dst_dir as 0000.jpg style."""
    os.makedirs(dst_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(markup.keys())
    saved, idx = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in wanted:
            cv2.imwrite(os.path.join(dst_dir, f"{idx:04d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"video frames={total}, annotated={len(wanted)}, saved={saved} -> {dst_dir}")
    return saved


def write_label_csv(markup, dst_dir, img_dir_for_csv):
    """Write TrackNet-style Label.csv.

    Columns: file name, visibility, x, y, status  (status kept for compatibility)
    All annotated frames are marked visibility=1 (visible); frames listed but with
    no coordinates would be visibility=0, which TTNet does not provide.
    """
    os.makedirs(dst_dir, exist_ok=True)
    csv_path = os.path.join(dst_dir, "Label.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file name", "visibility", "x", "y", "status"])
        for idx in sorted(markup.keys()):
            m = markup[idx]
            w.writerow([f"{idx:04d}.jpg", 1, m["x"], m["y"], 0])
    print(f"wrote {csv_path}  ({len(markup)} rows)  frames expected under {img_dir_for_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--markup_dir", required=True, help="folder containing ball_markup.json")
    ap.add_argument("--video", default=None, help="source video to extract frames from")
    ap.add_argument("--frames_dir", default=None, help="pre-extracted frames dir (idx.jpg)")
    ap.add_argument("--out_root", default="datasets/trackNet", help="TrackNet dataset root")
    ap.add_argument("--game", default="game1")
    ap.add_argument("--clip", default="Clip1")
    args = ap.parse_args()

    markup = load_markup(args.markup_dir)
    print(f"annotated frames: {len(markup)}")

    img_dir = os.path.join(args.out_root, "images", args.game, args.clip)
    if args.video:
        extract_from_video(args.video, markup, img_dir)
    elif args.frames_dir:
        os.makedirs(img_dir, exist_ok=True)
        missing = 0
        for idx in markup:
            src = os.path.join(args.frames_dir, f"{idx:04d}.jpg")
            if os.path.exists(src):
                import shutil
                shutil.copy(src, os.path.join(img_dir, f"{idx:04d}.jpg"))
            else:
                missing += 1
        print(f"copied frames to {img_dir}; missing={missing}")
    write_label_csv(markup, img_dir, img_dir)

    print("\nnext steps:")
    print(f"  1) python gt_gen.py --path_input {os.path.join(args.out_root, 'images')} "
          f"--path_output {args.out_root}   # or per-clip run")
    print("  2) python main.py --exp_id tt")
