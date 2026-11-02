"""诊断：最后一球消失后，模型热图到底有没有响应。

不跑后处理（不阈值、不霍夫），直接看 argmax 后的热图：
  * 全局峰值是多少、在哪
  * 在"球真实所在位置"邻域的峰值是多少

用法:
  python probe_heatmap.py --video ../_data/test_2_1080p_crf22.mp4 \
      --model ../TrackNet/TrackNet/exps/tt_g45_aug/model_best.pt \
      --start 2790 --end 3100
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

MODEL_H, MODEL_W = 360, 640


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--start", type=int, default=2790)
    ap.add_argument("--end", type=int, default=3100)
    ap.add_argument("--out_dir", default="../outputs/_heat_probe")
    ap.add_argument("--save_every", type=int, default=6, help="每 N 帧存一张热图 PNG")
    args = ap.parse_args()

    import torch
    from model import BallTrackerNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BallTrackerNet()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device).eval()

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video {w}x{h} {n} frames | device={device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # 需要从前 2 帧建立三帧栈，所以从 start-2 开始读
    s = max(0, args.start - 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    buf = deque(maxlen=3)
    fid = s - 1
    rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fid += 1
        if fid > args.end:
            break
        small = cv2.resize(frame, (MODEL_W, MODEL_H))
        buf.append(small)
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
        cm = logits.argmax(dim=1)[0].cpu().numpy().reshape(MODEL_H, MODEL_W).astype(np.uint8)

        peak = int(cm.max())
        py, px = np.unravel_index(int(cm.argmax()), cm.shape)
        # 原图坐标（模型输入 640x360 → 原图）
        gx, gy = px * (w / MODEL_W), py * (h / MODEL_H)
        rows.append((fid, fid / 120.0, peak, gx, gy))
        print(f"帧{fid} {fid/120:6.3f}s 峰值{peak:3d} 位置({gx:6.0f},{gy:6.0f})", flush=True)

        if (fid - s) % args.save_every == 0:
            cv2.imwrite(os.path.join(args.out_dir, f"hm_{fid:05d}.png"),
                        cv2.resize(cm, (MODEL_W * 2, MODEL_H * 2), interpolation=cv2.INTER_NEAREST))

    cap.release()
    arr = np.array([r[2] for r in rows])
    print(f"\n=== 汇总 {len(rows)} 帧 ===")
    print(f"峰值 max={arr.max()}  中位={int(np.median(arr))}  min={arr.min()}")
    print(f"峰值 >=127 的帧: {(arr >= 127).sum()}")
    print(f"峰值 >=96  的帧: {(arr >= 96).sum()}")
    print(f"峰值 >=64  的帧: {(arr >= 64).sum()}")
    print(f"峰值 <64 的帧: {(arr < 64).sum()}  ← 模型基本无响应")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
