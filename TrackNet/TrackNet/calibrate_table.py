"""导出一帧带坐标网格的图，用于人工标定球台四角。

用法：python calibrate_table.py export --video <mp4> --frame 5000 --out <jpg>
坐标网格式：原图分辨率下每 100 px 一条细线、每 200 px 一条粗线，
图上标注的是**原图坐标**，直接读到即可用。
"""
import argparse
import cv2
import numpy as np


def grid_frame(video, frame_idx, out, scale=0.5):
    cap = cv2.VideoCapture(video)
    if frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"读取失败: {video} frame={frame_idx}")

    H, W = img.shape[:2]
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    step = 100                       # 原图坐标步长
    minor = int(step * scale)
    for x in range(0, W, step):
        xs = int(x * scale)
        if xs >= small.shape[1]:
            break
        bold = (x % 500 == 0)
        cv2.line(small, (xs, 0), (xs, small.shape[0]),
                 (0, 0, 255) if bold else (0, 255, 255), 1 if bold else 1)
        if bold:
            cv2.putText(small, str(x), (xs + 2, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255), 1, cv2.LINE_AA)
    for y in range(0, H, step):
        ys = int(y * scale)
        if ys >= small.shape[0]:
            break
        bold = (y % 500 == 0)
        cv2.line(small, (0, ys), (small.shape[1], ys),
                 (0, 0, 255) if bold else (255, 255, 0), 1)
        if bold:
            cv2.putText(small, str(y), (2, ys - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out, small, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"saved {out}  src={W}x{H}  shown={small.shape[1]}x{small.shape[0]}  frame={frame_idx}")


def draw_poly(video, frame_idx, pts, out, scale=0.5):
    """把标定好的四角画回原图，用于目视验证。"""
    cap = cv2.VideoCapture(video)
    if frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("读取失败")
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    poly = (np.array(pts, dtype=float) * scale).astype(np.int32)
    cv2.polylines(small, [poly], True, (0, 0, 255), 2)
    for i, p in enumerate(pts):
        c = tuple((np.array(p) * scale).astype(int))
        cv2.circle(small, c, 5, (0, 255, 0), -1)
        cv2.putText(small, "ABCD"[i], (c[0] + 6, c[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    # 画中线（网），取 A-C 与 B-D 的中点连线
    a, b, c, d = [np.array(p, dtype=float) * scale for p in pts]
    cv2.line(small, tuple(((a + b) / 2).astype(int)), tuple(((c + d) / 2).astype(int)),
             (255, 0, 255), 2)
    cv2.imwrite(out, small, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export")
    e.add_argument("--video", required=True)
    e.add_argument("--frame", type=int, default=0)
    e.add_argument("--out", required=True)

    d = sub.add_parser("draw")
    d.add_argument("--video", required=True)
    d.add_argument("--frame", type=int, default=0)
    d.add_argument("--out", required=True)
    d.add_argument("--pts", required=True,
                   help='四角，按顺时针，格式 "x1,y1 x2,y2 x3,y3 x4,y4"')

    a = ap.parse_args()
    if a.cmd == "export":
        grid_frame(a.video, a.frame, a.out)
    else:
        pts = [tuple(float(v) for v in p.split(",")) for p in a.pts.split()]
        assert len(pts) == 4, "需要 4 个点"
        draw_poly(a.video, a.frame, pts, a.out)
