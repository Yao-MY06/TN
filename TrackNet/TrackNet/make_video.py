"""Build a test .mp4 from a frame folder (for infer_on_video.py reproduction).

Usage: python make_video.py <frames_dir> <out_mp4> [fps]
"""
import os, sys, cv2

def main(frames_dir, out_mp4, fps=30):
    files = sorted(f for f in os.listdir(frames_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        print("no frames found"); return
    first = cv2.imread(os.path.join(frames_dir, files[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))
    for f in files:
        img = cv2.imread(os.path.join(frames_dir, f))
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        vw.write(img)
    vw.release()
    print(f"wrote {out_mp4}  frames={len(files)}  {w}x{h}@{fps}fps")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 30)
