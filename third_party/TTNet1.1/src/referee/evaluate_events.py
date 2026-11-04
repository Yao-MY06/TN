# -*- coding: utf-8 -*-
"""量化 TTNet 事件检测（bounce / net）在测试集上的精确率/召回率。

这是判分可信度的前提：如果 bounce 的精确率只有 50%，状态机就只能当参考。

评估方式：时间窗口匹配（容差默认 ±0.1s）。TTNet 的单帧预测受 9 帧窗口影响，
存在少量时间偏移，用窗口匹配比逐帧严格相等更合理。
"""

import json
import os
import sys
import argparse

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from referee.ttnet_adapter import TTNetAdapter
from referee.event_processing import TrajectoryProcessor, EventExtractor
from referee.run_referee import build_configs, DEFAULT_WEIGHTS
from referee.unified_types import EventType


def load_gt(game):
    """读取真值事件，并返回标注有效区间。

    OpenTTGames 的标注不覆盖整段视频（test_2 长 30.0s，事件只标到 26.88s），
    因此必须只在标注区间内统计，否则末尾未标注段的正常预测会被误判为 FP。
    """
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    p = os.path.join(root, 'dataset', 'test', 'annotations', game, 'events_markup.json')
    with open(p) as f:
        d = json.load(f)
    fps = 120.0
    out = {'bounce': [], 'net': []}
    for k, v in d.items():
        if v in out:
            out[v].append(float(k) / fps)
    out = {k: sorted(v) for k, v in out.items()}
    all_ts = out['bounce'] + out['net']
    span = (min(all_ts), max(all_ts)) if all_ts else (0.0, float('inf'))
    return out, span


def match(pred, gt, tol):
    """一对一窗口匹配，返回 (tp, fp, fn, 匹配误差列表)。"""
    pred = sorted(pred)
    gt = sorted(gt)
    used = set()
    tp, errs = 0, []
    for p in pred:
        best, best_d = None, None
        for i, g in enumerate(gt):
            if i in used:
                continue
            d = abs(p - g)
            if d <= tol and (best_d is None or d < best_d):
                best, best_d = i, d
        if best is not None:
            used.add(best)
            tp += 1
            errs.append(best_d)
    return tp, len(pred) - tp, len(gt) - tp, errs


def evaluate(game='test_2', weights=DEFAULT_WEIGHTS, max_frames=None,
             tol=0.1, bounce_thresh=0.65, net_thresh=0.65, device='cuda:0'):
    gt, span = load_gt(game)
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    video = os.path.join(root, 'dataset', 'test', 'videos', '{}.mp4'.format(game))

    configs = build_configs()
    adapter = TTNetAdapter(configs, weights, device=device)
    tracker = TrajectoryProcessor()
    extractor = EventExtractor(bounce_thresh=bounce_thresh, net_thresh=net_thresh)

    pred = {'bounce': [], 'net': []}
    n_visible = n_total = 0

    def _collect(ev_list):
        for ev in ev_list:
            ts = ev.timestamp_s
            # 只在标注有效区间内计入，区间外的预测无法判定对错
            if not (span[0] - 0.5 <= ts <= span[1] + 0.5):
                continue
            if ev.type == EventType.BOUNCE:
                pred['bounce'].append(ts)
            elif ev.type == EventType.NET_HIT:
                pred['net'].append(ts)

    for obs in adapter.observations_from_video(video, max_frames=max_frames, progress_every=1000):
        n_total += 1
        if obs.ball.visible:
            n_visible += 1
        traj = tracker.update(obs)
        _collect(extractor.process(obs, traj))
    _collect(extractor.flush())

    print()
    print('=' * 72)
    print('TTNet 事件检测评估 — {} (容差 ±{:.2f}s, 标注区间 {:.2f}-{:.2f}s)'.format(
        game, tol, span[0], span[1]))
    print('=' * 72)
    for k in ('bounce', 'net'):
        tp, fp, fn, errs = match(pred[k], gt[k], tol)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        print('  {:<7} 预测 {} | 真值 {} | TP {} FP {} FN {}'.format(
            k, len(pred[k]), len(gt[k]), tp, fp, fn))
        print('          Precision {:.1%}  Recall {:.1%}  F1 {:.1%}  时间误差 {}'.format(
            p, r, f1, '{:.1f}ms'.format(np.mean(errs) * 1000) if errs else '-'))
    print('  球可见率 {:.1%}  |  低置信弃权 {} 帧  |  冻结丢弃 {} 帧'.format(
        n_visible / max(n_total, 1), adapter.low_conf_dropped,
        adapter.frozen_frames_dropped))
    print('  事件过滤 {}'.format(extractor.stats))
    print('=' * 72)
    return pred, gt


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--game', default='test_2')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--tol', type=float, default=0.1)
    ap.add_argument('--bounce-thresh', type=float, default=0.65)
    ap.add_argument('--net-thresh', type=float, default=0.5)
    a = ap.parse_args()
    evaluate(a.game, max_frames=a.max_frames, tol=a.tol,
             bounce_thresh=a.bounce_thresh, net_thresh=a.net_thresh)
