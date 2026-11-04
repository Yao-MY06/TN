# -*- coding: utf-8 -*-
"""事件层阈值网格搜索（不改状态机，只调事件提取参数）。

为什么需要"只在标注窗口内算精确率"
--------------------------------
OpenTTGames 的标注是**抽样式**的：test_1 的 140 秒里只有约 13 秒有标注。
如果在全时段统计精确率，那 108 秒里检出的真实事件**全部被算成 FP** ——
这就是上一版报告里「精确率 22%~80% 波动极大、不可信」的真正原因。

正确的算法是只评判**模型确实被监督到的地方**：

* ``Recall``   = 匹配上的真值 / 全部真值            （全时段，标注是完整的"该报的"）
* ``Precision``= 匹配上的预测 / **落在标注窗口内的预测**
  （窗口外的预测既不能算对也不能算错，直接不计入）

这样得到的精确率才是可信的，也才能用来选阈值。

搜索的参数
----------
* ``bounce_thresh`` / ``net_thresh``：驼峰上阈值
* ``min_event_interval_s``：同类事件最小间隔（去重强度）

用法::

    cd src
    python -m referee.tune_event_thresholds --game test_2
    python -m referee.tune_event_thresholds --game test_2 --apply
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
TOL_S = 0.10          # 匹配容差（与既有报告一致）
WINDOW_PAD_S = 1.5    # 标注窗口两端各留的缓冲


# ---------------------------------------------------------------- 真值
def load_gt(game: str) -> Tuple[List[float], List[float], List[Tuple[float, float]]]:
    """返回 (bounce 时刻, net 时刻, 标注窗口列表)。

    窗口按 bounce 间隔 > 2s 切分，两端各加 ``WINDOW_PAD_S``。
    这样「窗口内」= 有真值监督的时段，「窗口外」= 未标注、不可评判。
    """
    path = os.path.join(ROOT, 'dataset', 'test', 'annotations', game,
                        'events_markup.json')
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    ev = sorted((int(k) / 120.0, v) for k, v in d.items() if v in ('bounce', 'net'))
    b = [t for t, v in ev if v == 'bounce']
    n = [t for t, v in ev if v == 'net']

    rallies: List[List[float]] = []
    cur = [b[0]]
    for a, c in zip(b, b[1:]):
        if c - a > 2.0:
            rallies.append(cur)
            cur = [c]
        else:
            cur.append(c)
    rallies.append(cur)
    windows = [(r[0] - WINDOW_PAD_S, r[-1] + WINDOW_PAD_S) for r in rallies]
    return b, n, windows


def _match(pred: Sequence[float], gt: Sequence[float], tol=TOL_S):
    """一对一贪心匹配，返回 (匹配数, 匹配上的 pred 索引集合)。"""
    used = set()
    hit_pred = set()
    for i, x in enumerate(pred):
        for j, y in enumerate(gt):
            if j in used:
                continue
            if abs(x - y) <= tol:
                used.add(j)
                hit_pred.add(i)
                break
    return len(used), hit_pred


def _in_windows(t: float, windows) -> bool:
    return any(lo <= t <= hi for lo, hi in windows)


def evaluate(pred: Sequence[float], gt: Sequence[float], windows) -> Dict[str, float]:
    """在「标注窗口内」评估 precision，全时段评估 recall。"""
    tp, hit_idx = _match(pred, gt)
    # 落在标注窗口内的预测才算"可评判"
    judged = [i for i, t in enumerate(pred) if _in_windows(t, windows)]
    tp_in = sum(1 for i in hit_idx if i in set(judged))
    P = tp_in / len(judged) if judged else 0.0
    R = tp / len(gt) if gt else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {'P': P, 'R': R, 'F1': F, 'tp': tp,
            'n_pred': len(pred), 'n_pred_judged': len(judged), 'n_gt': len(gt)}


# ---------------------------------------------------------------- 重建
def rebuild_cache(obs_cache, bounce_thresh, net_thresh, min_interval):
    from referee.reextract_events import rebuild
    return rebuild(obs_cache, bounce_thresh=bounce_thresh,
                   net_thresh=net_thresh, verbose=False,
                   min_event_interval_s=min_interval)


def _ts(cache, etype):
    """取某类事件的时间戳。

    注意：``rebuild()`` 返回的是 **TTNet 原生事件名**（``NET_HIT``），
    不是桥接后的 YOLO 名（``NET_TOUCH``）。这里统一做一次映射，
    否则 ``NET_TOUCH`` 会永远取到空列表、看起来像"一个都没检出"。
    """
    alias = {'NET_TOUCH': 'NET_HIT'}
    etype = alias.get(etype, etype)
    return [e['timestamp_s'] for e in cache['events'] if e['type'] == etype]


# ---------------------------------------------------------------- 主流程
def main(games, apply_best=False, quick=False):
    bounce_grid = [0.55, 0.65, 0.75, 0.85] if quick else [0.50, 0.55, 0.60, 0.65,
                                                          0.70, 0.75, 0.80, 0.85, 0.90]
    net_grid = [0.40, 0.55, 0.70] if quick else [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    interval_grid = [0.10, 0.30] if quick else [0.10, 0.20, 0.30, 0.40]

    for game in games:
        obs_path = os.path.join(ROOT, 'results', 'cache_{}.json'.format(game))
        if not os.path.isfile(obs_path):
            print('[跳过] {} 没有观测缓存'.format(game))
            continue
        with open(obs_path, encoding='utf-8') as f:
            obs_cache = json.load(f)
        gb, gn, windows = load_gt(game)

        print('\n' + '=' * 88)
        print('{}  真值: bounce {} / net {}，标注窗口 {} 段，覆盖 {:.0f}s'.format(
            game, len(gb), len(gn), len(windows),
            sum(hi - lo for lo, hi in windows)))
        print('=' * 88)

        # ---- bounce 阈值（net 用默认，interval 用默认）
        print('\nbounce_thresh（只影响 BOUNCE；P 只在标注窗口内统计）')
        print('  {:>7}{:>8}{:>8}{:>8}{:>9}{:>12}'.format(
            'thresh', 'P', 'R', 'F1', '预测数', '窗口内预测'))
        best_b = (None, -1)
        for bt in bounce_grid:
            c = rebuild_cache(obs_cache, bt, 0.5, 0.10)
            m = evaluate(_ts(c, 'BOUNCE'), gb, windows)
            flag = ''
            if m['F1'] > best_b[1]:
                best_b = (bt, m['F1'])
                flag = '  ←'
            print('  {:>7.2f}{:>7.0f}%{:>7.0f}%{:>8.0f}{:>9}{:>12}{}'.format(
                bt, m['P'] * 100, m['R'] * 100, m['F1'] * 100,
                m['n_pred'], m['n_pred_judged'], flag))

        # ---- net 阈值
        print('\nnet_thresh（只影响 NET_TOUCH）')
        print('  {:>7}{:>8}{:>8}{:>8}{:>9}{:>12}'.format(
            'thresh', 'P', 'R', 'F1', '预测数', '窗口内预测'))
        best_n = (None, -1)
        for nt in net_grid:
            c = rebuild_cache(obs_cache, 0.65, nt, 0.10)
            m = evaluate(_ts(c, 'NET_TOUCH'), gn, windows)
            flag = ''
            if m['F1'] > best_n[1]:
                best_n = (nt, m['F1'])
                flag = '  ←'
            print('  {:>7.2f}{:>7.0f}%{:>7.0f}%{:>8.0f}{:>9}{:>12}{}'.format(
                nt, m['P'] * 100, m['R'] * 100, m['F1'] * 100,
                m['n_pred'], m['n_pred_judged'], flag))

        # ---- 去重间隔（对两类事件都生效）
        print('\nmin_event_interval_s（同类事件去重强度）')
        print('  {:>7}{:>14}{:>14}'.format('间隔', 'BOUNCE F1', 'NET F1'))
        best_i = (None, -1)
        for mi in interval_grid:
            c = rebuild_cache(obs_cache, best_b[0], best_n[0], mi)
            mb = evaluate(_ts(c, 'BOUNCE'), gb, windows)
            mn = evaluate(_ts(c, 'NET_TOUCH'), gn, windows)
            score = mb['F1'] + mn['F1']
            flag = ''
            if score > best_i[1]:
                best_i = (mi, score)
                flag = '  ←'
            print('  {:>7.2f}{:>13.0f}%{:>13.0f}%{}'.format(
                mi, mb['F1'] * 100, mn['F1'] * 100, flag))

        print('\n最优: bounce_thresh={:.2f}  net_thresh={:.2f}  min_interval={:.2f}'
              .format(best_b[0], best_n[0], best_i[0]))

        if apply_best:
            out = rebuild_cache(obs_cache, best_b[0], best_n[0], best_i[0])
            path = os.path.join(ROOT, 'results', 'infer_{}_full.json'.format(game))
            if os.path.isfile(path) and not os.path.isfile(path + '.bak'):
                os.replace(path, path + '.bak')
                print('已备份 → {}'.format(os.path.basename(path) + '.bak'))
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(out, f, ensure_ascii=False)
            print('已写入 → {}'.format(path))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--game', nargs='+', default=['test_2'])
    ap.add_argument('--apply', action='store_true', help='把最优参数写回推理缓存')
    ap.add_argument('--quick', action='store_true', help='小网格，快速看趋势')
    a = ap.parse_args()
    main(a.game, a.apply, a.quick)
