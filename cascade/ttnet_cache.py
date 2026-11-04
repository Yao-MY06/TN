# -*- coding: utf-8 -*-
"""在一段视频上逐窗口跑 TTNet，输出 JSONL（每个窗口末帧一行）。给原机（GPU + 视频）用。

    python -m cascade.ttnet_cache --video test_2.mp4 --weights ttnet.pth \
        --ttnet-src ../TTNet/src --out _data/ttnet/test_2.jsonl

每行字段：
    frame_id        = 窗口末帧 e（球位置对应这一帧）
    timestamp_s     = e 的视频时间
    window_start / window_end = e-8 / e
    bounce / net    = 事件分数（对应第 e-k 帧，k 由复核器决定，这里不做偏移）
    ball            = {visible, center_px, confidence}，按 ttnet_adapter 语义解码：
                      峰值 < 0.05 不可见；global+local 合成坐标；conf = min(峰值)；
                      conf < 0.25 丢弃；同一坐标连续冻结 >= 3 帧丢弃
    ball_raw        = 不经过 0.25 / 冻结守卫的坐标与置信度（便于以后重调）
    global_peak / local_peak = [x 峰值, y 峰值]
    crop            = 局部阶段裁剪参数 [x_min, x_max, y_min, y_max, x_pad, y_pad]
    infer_ms        = 单窗推理耗时（毫秒，含 host<->device 拷贝，不含解码）
前 8 帧凑不满窗口，没有行。另写一个 <out>.meta.json。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

from cascade.ttnet_engine import (H_IN, H_ORG, NUM_FRAMES, W_IN, W_ORG, make_engine,  # noqa: E402
                                  small_rgb, stack_window)

BALL_THRESH = 0.05
MIN_CONF = 0.25
STUCK_PATIENCE = 3
W_RATIO, H_RATIO = W_ORG / W_IN, H_ORG / H_IN


def _decode_1d(prob, thresh=BALL_THRESH):
    prob = np.asarray(prob, dtype=np.float32).ravel()
    peak = float(prob.max())
    if peak < thresh:
        return None, peak
    return float(int(prob.argmax())), peak


def decode_ball(out, thresh=BALL_THRESH):
    """ttnet_adapter.build_observation 的坐标还原（argmax 解码），无状态部分。"""
    gx, pgx = _decode_1d(out['global_x'], thresh)
    gy, pgy = _decode_1d(out['global_y'], thresh)
    lx, plx = _decode_1d(out['local_x'], thresh)
    ly, ply = _decode_1d(out['local_y'], thresh)
    if gx is None or gy is None:
        return None, None
    ox, oy, conf = gx * W_RATIO, gy * H_RATIO, min(pgx, pgy)
    if lx is not None and ly is not None:
        ox = gx * W_RATIO + lx - W_IN / 2.0
        oy = gy * H_RATIO + ly - H_IN / 2.0
        conf = min(conf, plx, ply)
    return [float(np.clip(ox, 0, W_ORG - 1)), float(np.clip(oy, 0, H_ORG - 1))], float(conf)


class BallGuard:
    """ttnet_adapter._guard_invalid 的有状态部分（低置信 + 坐标冻结）。"""

    def __init__(self, min_conf=MIN_CONF, patience=STUCK_PATIENCE):
        self.min_conf, self.patience = min_conf, patience
        self.last, self.stuck = None, 0

    def __call__(self, center, conf):
        if center is None:
            return None, None
        if conf is None or conf < self.min_conf:
            self.last, self.stuck = None, 0
            return None, None
        if self.last is not None and abs(center[0] - self.last[0]) < 0.5 and abs(center[1] - self.last[1]) < 0.5:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last = center
        if self.stuck >= self.patience:
            return None, None
        return center, conf


def row_from_output(window_end, timestamp_s, out, infer_ms, guard=None):
    center, conf = decode_ball(out)
    g_center, g_conf = guard(center, conf) if guard is not None else (center, conf)
    ev = np.asarray(out['events'], dtype=np.float64).ravel()
    return {
        'frame_id': int(window_end),
        'timestamp_s': None if timestamp_s is None else round(float(timestamp_s), 6),
        'window_start': int(window_end) - (NUM_FRAMES - 1),
        'window_end': int(window_end),
        'bounce': round(float(ev[0]), 6),
        'net': round(float(ev[1]), 6),
        'ball': {'visible': g_center is not None,
                 'center_px': None if g_center is None else [round(v, 2) for v in g_center],
                 'confidence': None if g_conf is None else round(g_conf, 6)},
        'ball_raw': {'center_px': None if center is None else [round(v, 2) for v in center],
                     'confidence': None if conf is None else round(conf, 6)},
        'global_peak': [round(float(np.max(out['global_x'])), 6), round(float(np.max(out['global_y'])), 6)],
        'local_peak': [round(float(np.max(out['local_x'])), 6), round(float(np.max(out['local_y'])), 6)],
        'crop': [int(v) for v in out['crop']],
        'infer_ms': None if infer_ms is None else round(float(infer_ms), 3),
    }


class VideoFrameSource:
    """顺序解码视频，只保留 320x128 RGB 小图（每帧 0.12 MB）。

    __call__(frame_id) 返回小图；需要时往后解码。只保留最近 ``keep`` 帧，
    被淘汰的帧再请求返回 None（复核器会记 error，不会拿邻帧凑数）。
    """

    def __init__(self, video, keep=2400):
        import cv2
        self.cap = cv2.VideoCapture(video)
        if not self.cap.isOpened():
            raise RuntimeError('无法打开视频: {}'.format(video))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.keep = keep
        self.frames = {}
        self.next_id = 0
        self.eof = False

    def timestamp_of(self, frame_id):
        return frame_id / self.fps if self.fps else None

    def __call__(self, frame_id):
        while not self.eof and self.next_id <= frame_id:
            ok, frame = self.cap.read()
            if not ok:
                self.eof = True
                break
            self.frames[self.next_id] = small_rgb(frame)
            self.next_id += 1
            if len(self.frames) > self.keep:
                del self.frames[min(self.frames)]
        return self.frames.get(frame_id)

    def __iter__(self):
        f = 0
        while True:
            img = self(f)
            if img is None:
                return
            yield f, img
            f += 1


def build_cache(engine, video, out_path, max_frames=None, start=0, progress_every=500):
    src = VideoFrameSource(video, keep=NUM_FRAMES + 2)
    guard = BallGuard()
    window = []
    n, times = 0, []
    t_all = time.perf_counter()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for fid, img in src:
            if max_frames is not None and fid >= max_frames:
                break
            window.append(img)
            if len(window) > NUM_FRAMES:
                window.pop(0)
            if len(window) < NUM_FRAMES or fid < start:
                continue
            t0 = time.perf_counter()
            out = engine.infer(stack_window(window))
            ms = (time.perf_counter() - t0) * 1000.0
            times.append(ms)
            f.write(json.dumps(row_from_output(fid, src.timestamp_of(fid), out, ms, guard)) + '\n')
            n += 1
            if progress_every and n % progress_every == 0:
                print('  [ttnet_cache] {} 窗口 (frame {})  中位 {:.1f} ms'.format(n, fid, float(np.median(times))))
    meta = {
        'video': os.path.abspath(video), 'fps': src.fps, 'n_windows': n,
        'engine': getattr(engine, 'name', '?'), 'window': NUM_FRAMES,
        'input_size': [W_IN, H_IN], 'note': 'bounce/net 对应 e-k 帧，k 由复核器决定',
        'infer_ms_median': float(np.median(times)) if times else None,
        'infer_ms_p95': float(np.percentile(times, 95)) if times else None,
        'wall_s': round(time.perf_counter() - t_all, 2),
    }
    with open(out_path + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description='TTNet 逐窗口分数缓存')
    ap.add_argument('--video', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--engine', default='auto', choices=['auto', 'torch', 'numpy'])
    ap.add_argument('--ttnet-src', default=None, help='torch 引擎需要：TTNet/src 目录')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max-frames', type=int, default=None)
    a = ap.parse_args(argv)
    eng = make_engine(a.engine, a.weights, a.ttnet_src, a.device)
    meta = build_cache(eng, a.video, a.out, a.max_frames)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
