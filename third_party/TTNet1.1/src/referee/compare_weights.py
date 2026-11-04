# -*- coding: utf-8 -*-
"""对比自训练权重与预训练权重在全部测试视频上的表现。

评估维度
--------
1. **球可见率** —— 模型能"看到"球的帧占比，直接决定事件覆盖度；
2. **事件检测** P/R/F1 —— 在标注区间内按 ±0.1s 窗口匹配；
3. **判分结果** —— 回合数与比分。

重要前提
--------
OpenTTGames 的 test 标注是**抽样式**的（覆盖率 10%~59%），
未标注段落里的正确检出会被算成 FP，因此 **精确率被系统性低估**。
跨权重对比时，两个模型面对同样的标注缺失，所以**相对比较仍然有效**，
但绝对值不可当真。
"""

import json
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from referee.tune_events import load_gt, labeled_span, detect, match
from referee.render_with_ui import build_score_timeline


ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
GAMES = ['test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6', 'test_7']


def load(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def eval_cache(cache, game, tol=0.1):
    """在标注区间内评估一个推理缓存。"""
    gt = load_gt(game)
    span = labeled_span(game)

    frames = [x for x in cache['frames'] if span[0] <= x[1] <= span[1]]
    vis_rate = float(np.mean([x[2] for x in frames])) if frames else 0.0

    evb = [e['timestamp_s'] for e in cache['events']
           if e['type'] == 'BOUNCE' and span[0] - 0.5 <= e['timestamp_s'] <= span[1] + 0.5]
    evn = [e['timestamp_s'] for e in cache['events']
           if e['type'] == 'NET_HIT' and span[0] - 0.5 <= e['timestamp_s'] <= span[1] + 0.5]

    out = {'vis_rate': vis_rate}
    for pred, name in ((evb, 'bounce'), (evn, 'net')):
        tp, fp, fn, errs = match(pred, gt[name], tol)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        out[name] = {'p': p, 'r': r, 'f1': f1,
                     'tp': tp, 'fp': fp, 'fn': fn,
                     'err_ms': float(np.mean(errs) * 1000) if errs else 0.0}
    return out


def main(games=None, tol=0.1):
    games = games or GAMES
    rows = []

    for g in games:
        self_cache = load(os.path.join(ROOT, 'results', 'infer_{}_full.json'.format(g)))
        pre_cache = load(os.path.join(ROOT, 'results',
                                      'infer_{}_full_pretrained.json'.format(g)))
        if self_cache is None:
            print('[跳过] {} 无自训练缓存'.format(g))
            continue

        rec = {'game': g}
        rec['self'] = eval_cache(self_cache, g, tol)
        if pre_cache is not None:
            rec['pretrained'] = eval_cache(pre_cache, g, tol)
            _, _, m_self = build_score_timeline(self_cache, verbose=False)
            _, _, m_pre = build_score_timeline(pre_cache, verbose=False)
            rec['points_self'] = len(m_self.points)
            rec['points_pre'] = len(m_pre.points)
        rows.append(rec)

    # ---- 输出表格
    print()
    print('=' * 100)
    print('自训练 vs 预训练 对比（标注区间内，时间容差 ±{:.2f}s）'.format(tol))
    print('=' * 100)
    hdr = '{:<8} | {:^22} | {:^22} | {:^22} | {:^12}'.format(
        '视频', '球可见率', 'bounce F1', 'net F1', '判分数')
    print(hdr)
    print('{:<8} | {:>10} {:>10} | {:>10} {:>10} | {:>10} {:>10} | {:>5} {:>5}'.format(
        '', '自训练', '预训练', '自训练', '预训练', '自训练', '预训练', '自', '预'))
    print('-' * 100)

    agg = {'self': [], 'pre': []}
    for r in rows:
        s, p = r['self'], r.get('pretrained')
        def fmt(v, key):
            return '{:.0%}'.format(v[key]) if v else '—'
        def f1f(v):
            return '{:.1%}'.format(v['f1']) if v else '—'
        print('{:<8} | {:>10} {:>10} | {:>10} {:>10} | {:>10} {:>10} | {:>5} {:>5}'.format(
            r['game'],
            '{:.1%}'.format(s['vis_rate']),
            '{:.1%}'.format(p['vis_rate']) if p else '—',
            f1f(s['bounce']), f1f(p['bounce']) if p else '—',
            f1f(s['net']), f1f(p['net']) if p else '—',
            r.get('points_self', '—'), r.get('points_pre', '—')))
        agg['self'].append(r['self'])
        if p:
            agg['pre'].append(p)

    print('-' * 100)
    def avg(lst, key, sub=None):
        vals = []
        for x in lst:
            v = x[key]
            vals.append(v[sub] if sub else v)
        return float(np.mean(vals)) if vals else None

    for label, lst in (('自训练', agg['self']), ('预训练', agg['pre'])):
        if not lst:
            continue
        print('{:<8} | {:>10} {:>10} | {:>10} {:>10} | {:>10} {:>10} |'.format(
            label + '均值',
            '{:.1%}'.format(avg(lst, 'vis_rate')), '',
            '{:.1%}'.format(avg(lst, 'bounce', 'f1')), '',
            '{:.1%}'.format(avg(lst, 'net', 'f1')), ''))
    print('=' * 100)

    # 存 JSON
    out_path = os.path.join(ROOT, 'results', '权重对比.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1, default=float)
    print('\n明细已写入: {}'.format(out_path))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', nargs='+', default=None)
    ap.add_argument('--tol', type=float, default=0.1)
    a = ap.parse_args()
    main(a.games, a.tol)
