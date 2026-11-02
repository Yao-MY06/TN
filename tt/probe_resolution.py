"""验证：推理时提高输入分辨率，能否让"消失的球"重新出现。

TrackNet 是全卷积，训练用 360x640，推理可换更大尺寸（等效于更细的采样）。
只推理、不重训 —— 若有效，是零成本改进。

用法:
  python probe_resolution.py --video ../_data/test_2_1080p_crf22.mp4 \
      --model ../TrackNet/TrackNet/exps/tt_g45_aug/model_best.pt --start 2830 --end 2880
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "TrackNet", "TrackNet"))


def run(model, device, cap, w, h, start, end, in_w, in_h, tag):
    import torch
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start - 2))
    buf = deque(maxlen=3)
    fid = max(0, start - 2) - 1
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fid += 1
        if fid > end:
            break
        buf.append(cv2.resize(frame, (in_w, in_h)))
        if len(buf) < 3:
            continue
        stack = np.concatenate((buf[2], buf[1], buf[0]), axis=2).astype(np.float32) / 255.0
        t = torch.from_numpy(np.rollaxis(stack, 2, 0)[None]).float().to(device)
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(t)
                logits = logits.float()
            else:
                logits = model(t)
        cm = logits.argmax(dim=1)[0].cpu().numpy().reshape(in_h, in_w).astype(np.uint8)
        oh, ow = in_h, in_w          # 输入需为 8 的倍数，输出与输入同尺寸
        peak = int(cm.max())
        py, px = np.unravel_index(int(cm.argmax()), cm.shape)
        out.append((fid, peak, px * w / ow, py * h / oh))
        print(f"  [{tag}] 帧{fid} {fid/120:6.3f}s 峰值{peak:3d} ({px*w/in_w:6.0f},{py*h/in_h:6.0f})", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--start", type=int, default=2830)
    ap.add_argument("--end", type=int, default=2880)
    args = ap.parse_args()

    import torch
    from model import BallTrackerNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = BallTrackerNet()
    m.load_state_dict(torch.load(args.model, map_location=device))
    m = m.to(device).eval()

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    res = {}
    for (iw, ih) in [(640, 360), (960, 544), (1280, 720)]:
        print(f"\n=== 输入 {iw}x{ih} ===")
        res[(iw, ih)] = run(m, device, cap, w, h, args.start, args.end, iw, ih, f"{iw}x{ih}")
    cap.release()

    print("\n=== 峰值对比（帧 | 640x360 | 960x544 | 1280x720）===")
    for i in range(len(res[(640, 360)])):
        fid = res[(640, 360)][i][0]
        a = res[(640, 360)][i][1]
        b = res[(960, 544)][i][1] if i < len(res[(960, 544)]) else -1
        c = res[(1280, 720)][i][1] if i < len(res[(1280, 720)]) else -1
        mark = "  <<< 360p 全 0" if (a == 0 and (b > 0 or c > 0)) else ""
        print(f"  帧{fid} {fid/120:6.3f}s | {a:3d} | {b:3d} | {c:3d}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
