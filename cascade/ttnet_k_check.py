# -*- coding: utf-8 -*-
"""用真值核实事件对齐偏移 k。

原理：某个落台真值帧 f 的分数峰出现在窗口末帧 e = f + k_true。
在整段缓存上按 k=0 找 bounce 驼峰，峰值帧与最近真值之差（预测 − 真值）的中位数
就是 k_true 的估计；再对 k=0..8 各算一次 ±5 帧口径的 P/R/F1 与平均 |Δt|。

    python -m cascade.ttnet_k_check --ttnet-cache _data/ttnet/test_2.jsonl \
        --gt <events_markup.json 或含它的目录> --game test_2 --out outputs/k_check.json

真值格式（自动识别）：
  - OpenTTGames events_markup.json：{"<frame>": "bounce" | "net" | "empty_event", ...}
  - 列表：[{"frame": 123, "event": "bounce"}, ...]（键名 frame/frame_id，event/type/label）
目录时递归找文件名同时含 game 和 "event"/"markup" 的 .json。
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from cascade.ttnet_reviewer import BOUNCE_HIGH, BOUNCE_LOW, find_humps, load_jsonl


def load_gt_bounces(path, game=None, max_frame=None):
    if os.path.isdir(path):
        cands = [p for p in glob.glob(os.path.join(path, '**', '*.json'), recursive=True)
                 if (game is None or game in p) and ('event' in os.path.basename(p).lower()
                                                     or 'markup' in os.path.basename(p).lower())]
        if not cands:
            raise SystemExit('在 {} 下没找到 {} 的事件真值 JSON'.format(path, game))
        path = sorted(cands)[0]
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    frames = []
    if isinstance(data, dict):
        for k, v in data.items():
            lab = v if isinstance(v, str) else (v.get('event') or v.get('type') or v.get('label'))
            if str(lab).lower() == 'bounce':
                frames.append(int(k))
    else:
        for r in data:
            f = r.get('frame', r.get('frame_id'))
            lab = r.get('event') or r.get('type') or r.get('label')
            if f is not None and str(lab).lower() == 'bounce':
                frames.append(int(f))
    frames = sorted(frames)
    if max_frame is not None:
        frames = [f for f in frames if f <= max_frame]
    return frames, path


def match(pred, gt, tol):
    used, pairs = set(), []
    for p in pred:
        best, bd = None, None
        for i, g in enumerate(gt):
            if i in used:
                continue
            d = abs(p - g)
            if d <= tol and (bd is None or d < bd):
                best, bd = i, d
        if best is not None:
            used.add(best)
            pairs.append((p, gt[best]))
    tp = len(pairs)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gt) if gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return pairs, prec, rec, f1


def k_report(rows, gt, tol=5, wide=12, fps=120.0, high=BOUNCE_HIGH, low=BOUNCE_LOW, gt_end=None):
    rows = sorted(rows, key=lambda r: r['window_end'])
    ends = [int(r['window_end']) for r in rows]
    sc = [float(r['bounce']) for r in rows]
    peaks_e = [h.peak_frame for h in find_humps(ends, sc, high, low)]
    if gt_end is not None:
        gt = [g for g in gt if g <= gt_end]
    # k=0 下与最近真值的带符号偏差
    offs = []
    for p in peaks_e:
        if not gt:
            break
        g = min(gt, key=lambda x: abs(x - p))
        if abs(g - p) <= wide:
            offs.append(p - g)
    per_k = []
    for k in range(9):
        pred = [p - k for p in peaks_e]
        if gt_end is not None:
            pred = [p for p in pred if p <= gt_end]
        pairs, prec, rec, f1 = match(pred, gt, tol)
        err = [abs(p - g) for p, g in pairs]
        per_k.append({'k': k, 'n_pred': len(pred), 'n_gt': len(gt), 'tp': len(pairs),
                      'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4),
                      'mean_abs_err_frames': round(float(np.mean(err)), 3) if err else None,
                      'mean_abs_err_ms': round(float(np.mean(err)) / fps * 1000, 2) if err else None})
    best = min((r for r in per_k if r['mean_abs_err_frames'] is not None),
               key=lambda r: (r['mean_abs_err_frames'], -r['f1']), default=None)
    return {'k_estimate_median_offset': float(np.median(offs)) if offs else None,
            'n_offsets': len(offs), 'offset_hist': {str(int(v)): int(c) for v, c in
                                                     zip(*np.unique(offs, return_counts=True))} if offs else {},
            'best_k_by_abs_err': best['k'] if best else None, 'per_k': per_k}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ttnet-cache', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--game', default='test_2')
    ap.add_argument('--gt-end-frame', type=int, default=None, help='真值标注覆盖的最后一帧（test_2 约 26.88 s）')
    ap.add_argument('--fps', type=float, default=120.0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    gt, gt_path = load_gt_bounces(a.gt, a.game)
    rep = k_report(load_jsonl(a.ttnet_cache), gt, fps=a.fps, gt_end=a.gt_end_frame)
    rep['gt_path'] = gt_path
    txt = json.dumps(rep, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            f.write(txt)


if __name__ == '__main__':
    main()
