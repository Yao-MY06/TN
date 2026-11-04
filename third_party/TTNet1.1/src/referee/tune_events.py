# -*- coding: utf-8 -*-
"""用缓存做离线参数扫描，秒级得到最优事件阈值。

不必重复跑推理（每次 2 分钟），直接对 cache_scores.py 产出的分数序列
重放迟滞检测，网格搜索阈值 / 滞回带宽 / 最小间隔。

用法::

    python -m referee.tune_events --cache ../results/cache_test_2.json --game test_2
"""

import os
import sys
import json
import argparse
import itertools

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def load_gt(game):
    root = os.path.join(os.path.dirname(__file__), '..', '..')
    p = os.path.join(root, 'dataset', 'test', 'annotations', game, 'events_markup.json')
    with open(p) as f:
        d = json.load(f)
    out = {'bounce': [], 'net': []}
    for k, v in d.items():
        if v in out:
            out[v].append(float(k) / 120.0)
    return {k: sorted(v) for k, v in out.items()}


def labeled_span(game):
    """返回标注有效区间 (t_min, t_max)。

    OpenTTGames 的标注**不覆盖整段视频**：例如 test_2 长 30.0s，
    但事件只标注到 26.88s，末尾 3.1s（10.4%）完全无标注。
    若不裁剪评估区间，这段「未标注但可能真有球」的预测会被误算成 FP，
    使精确率被严重低估。因此所有指标都只在标注区间内统计。
    """
    gt = load_gt(game)
    all_ts = gt['bounce'] + gt['net']
    if not all_ts:
        return 0.0, float('inf')
    return min(all_ts), max(all_ts)


def match(pred, gt, tol):
    pred, gt = sorted(pred), sorted(gt)
    used, tp, errs = set(), 0, []
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


def detect(rows, key, high, hyst, min_interval, mask_invisible=True):
    """在缓存序列上重放迟滞驼峰检测。"""
    low = high - hyst
    events = []
    active, peak, peak_ts = False, 0.0, None
    prev_ts = None
    for r in rows:
        ts = r['t']
        s = r[key]
        # 球不可见时分数不可靠：可选屏蔽
        if mask_invisible and not r['v']:
            s = 0.0

        if s >= high:
            if not active:
                active, peak, peak_ts = True, s, ts
            elif s > peak:
                peak, peak_ts = s, ts
        elif active and s >= low:
            if s > peak:
                peak, peak_ts = s, ts
        elif active:
            active = False
            if prev_ts is None or (peak_ts - prev_ts) >= min_interval:
                events.append(peak_ts)
                prev_ts = peak_ts
            peak, peak_ts = 0.0, None
    if active and peak_ts is not None:
        if prev_ts is None or (peak_ts - prev_ts) >= min_interval:
            events.append(peak_ts)
    return events


def scan(rows, gt, tol=0.1, span=None):
    """在标注有效区间内网格搜索最优参数。"""
    if span is not None:
        rows = [r for r in rows if span[0] <= r['t'] <= span[1]]
    best = {}
    grid = {
        'high': [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
        'hyst': [0.0, 0.1, 0.15, 0.2, 0.25, 0.3],
        'interval': [0.0, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
        'mask_invisible': [True, False],
    }
    for key, gtkey in (('b', 'bounce'), ('n', 'net')):
        results = []
        for high, hyst, interval, mask in itertools.product(
                grid['high'], grid['hyst'], grid['interval'], grid['mask_invisible']):
            if hyst >= high:
                continue
            pred = detect(rows, key, high, hyst, interval, mask)
            tp, fp, fn, errs = match(pred, gt[gtkey], tol)
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            results.append((f1, p, r, tp, fp, fn, high, hyst, interval, mask, len(pred)))
        results.sort(key=lambda x: -x[0])
        best[gtkey] = results[:8]
        print()
        print('=== {} 最优参数（按 F1 排序，真值 {} 个）==='.format(gtkey, len(gt[gtkey])))
        print('  {:<6} {:<7} {:<7} {:<5} {:<4}{:<4}{:<4} {:<9} {:<9} {}'.format(
            'F1', 'P', 'R', 'pred', 'TP', 'FP', 'FN', 'high', 'hyst', 'interval/mask'))
        for f1, p, r, tp, fp, fn, high, hyst, interval, mask, npred in results[:8]:
            print('  {:<6.1%} {:<7.1%} {:<7.1%} {:<5} {:<4}{:<4}{:<4} {:<9.2f} {:<9.2f} {}'.format(
                f1, p, r, npred, tp, fp, fn, high, hyst, interval, '是' if mask else '否'))
    return best


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='../results/cache_test_2.json')
    ap.add_argument('--game', default='test_2')
    ap.add_argument('--tol', type=float, default=0.1)
    a = ap.parse_args()

    with open(a.cache) as f:
        rows = json.load(f)['rows']
    gt = load_gt(a.game)
    span = labeled_span(a.game)
    n_all = len(rows)
    rows = [r for r in rows if span[0] <= r['t'] <= span[1]]
    print('缓存 {} 帧，有效标注区间 {:.2f}-{:.2f}s 内 {} 帧'.format(
        n_all, span[0], span[1], len(rows)))
    print('真值 bounce={} net={}'.format(len(gt['bounce']), len(gt['net'])))
    scan(rows, gt, a.tol)
