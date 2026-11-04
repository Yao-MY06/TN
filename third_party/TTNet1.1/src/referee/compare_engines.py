# -*- coding: utf-8 -*-
"""A/B 对比：同一批 TTNet 事件，两套计分状态机各判一遍。

用途
----
验证「换成 YOLO 状态机」到底改变了什么。因为两套引擎吃的是**同一份推理缓存**，
差异完全来自判分规则，与模型精度无关 —— 这就是方案文档 §6 说的
「用同一段视频分别测试，可以直接比较最终判分准确率」。

用法::

    cd src
    # 全部有缓存的视频，给一张对比表
    python -m referee.compare_engines

    # 指定视频 + 落盘 json
    python -m referee.compare_engines --games test_1 test_2 --json ../results/yolo_sm/ab.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from referee.render_with_ui import build_score_timeline


ROOT = os.path.join(os.path.dirname(__file__), '..', '..')


def discover(prefix='infer_', suffix='_full.json'):
    """扫出所有推理缓存，返回 {game: path}。"""
    found = {}
    for p in sorted(glob.glob(os.path.join(ROOT, 'results', prefix + '*' + suffix))):
        name = os.path.basename(p)                      # infer_test_1_full.json
        m = re.match(r'^{}(\w+?){}$'.format(re.escape(prefix), re.escape(suffix)), name)
        if m:
            found[m.group(1)] = p
    return found


def summarize(engine, machine):
    """把两种引擎的机器状态压成同一形状。"""
    if engine == 'yolo':
        pts = machine.points
        return {
            'engine': 'yolo',
            'points_total': sum(pts),
            'final': {'A': pts[0], 'B': pts[1]},
            'games_won': dict(machine.games),
            'rule_counts': dict(machine.rule_counts),
            'decision_count': len(machine.all_decisions),
            'dead_ball': (machine.rule_counts.get('NO_EVIDENCE', 0)
                          + machine.rule_counts.get('SERVE_INCOMPLETE', 0)),
            'lets': machine.rule_counts.get('LET', 0),
            'point_events': [
                {'ts': round(d.timestamp_s, 3), 'winner': d.winner,
                 'rule': d.rule, 'score': list(d.score)}
                for d in machine.all_decisions
                if d.kind.value == 'POINT'],
        }
    return {
        'engine': 'legacy',
        'points_total': len(machine.points),
        'final': machine.scoreboard['score'],
        'games_won': machine.scoreboard['games'],
        'rule_counts': {},       # legacy 没有规则名体系
        'decision_count': len(machine.points),
        'dead_ball': None,
        'lets': None,
        'suppressed_by_guard': machine.suppressed_count,
        'point_events': [
            {'ts': round(p['ts'], 3), 'winner': p['winner'],
             'rule': None, 'score': [machine.scoreboard['score'][k] for k in ('A', 'B')]}
            for p in machine.points],
    }


def compare_one(game, cache_path, verbose=False):
    with open(cache_path, encoding='utf-8') as f:
        cache = json.load(f)

    out = {'game': game, 'cache': os.path.relpath(cache_path, ROOT),
           'frames': len(cache.get('frames', [])),
           'detected_events': cache.get('detected', {})}

    for eng in ('legacy', 'yolo'):
        if not verbose:
            # build_score_timeline 会打印进度，对比表模式下静音
            import contextlib
            import io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _, _, _, m = build_score_timeline(cache, engine=eng)
        else:
            _, _, _, m = build_score_timeline(cache, engine=eng)
        out[eng] = summarize(eng, m)

    # 时间线上有多少分是两套都判的（±0.5s 内视为同一分）
    l_ts = [p['ts'] for p in out['legacy']['point_events']]
    y_ts = [p['ts'] for p in out['yolo']['point_events']]
    matched = sum(1 for t in y_ts if any(abs(t - u) <= 0.5 for u in l_ts))
    out['agreement'] = {
        'legacy_points': len(l_ts),
        'yolo_points': len(y_ts),
        'matched_within_0.5s': matched,
    }
    return out


def main(games, json_path=None, verbose=False):
    caches = discover()
    if not caches:
        print('没找到任何推理缓存（results/infer_*_full.json）。'
              '请先跑一次推理，或用 batch_render 生成。')
        return 1
    targets = games or sorted(caches, key=lambda g: (len(g), g))

    rows = []
    for g in targets:
        if g not in caches:
            print('[跳过] {} 没有推理缓存'.format(g))
            continue
        print('\n>>> {}'.format(g))
        rows.append(compare_one(g, caches[g], verbose=verbose))

    # ---- 对比表
    print('\n' + '=' * 100)
    print('两套计分状态机 A/B 对比（同一份 TTNet 推理缓存，差异只来自判分规则）')
    print('=' * 100)
    hdr = ('{:<10}{:>10}{:>14}{:>14}{:>10}{:>10}'.format(
        '视频', '事件数', 'legacy 判分', 'yolo 判分', '两套一致', 'yolo死球'))
    print(hdr)
    print('-' * 100)
    for r in rows:
        ev = r['detected_events']
        n_ev = sum(v for k, v in ev.items() if k != 'BALL_LOST')
        lg, yl = r['legacy'], r['yolo']
        print('{:<10}{:>10}{:>14}{:>14}{:>10}{:>10}'.format(
            r['game'], n_ev,
            'A{}-B{}'.format(lg['final']['A'], lg['final']['B']),
            'A{}-B{}'.format(yl['final']['A'], yl['final']['B']),
            '{}/{}'.format(r['agreement']['matched_within_0.5s'],
                           r['agreement']['yolo_points']),
            yl['dead_ball']))

    print('\nyolo 引擎的规则分布（每分是哪条 ITTF 规则给的）：')
    for r in rows:
        rc = r['yolo']['rule_counts']
        decided = {k: v for k, v in rc.items() if k not in ('LET', 'NO_EVIDENCE',
                                                            'SERVE_INCOMPLETE')}
        print('  {:<10} {}'.format(r['game'], decided or '（无判分）'))

    if json_path:
        d = os.path.dirname(os.path.abspath(json_path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print('\n明细已写入: {}'.format(json_path))
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', nargs='+', default=None)
    ap.add_argument('--json', default=None)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    sys.exit(main(a.games, a.json, a.verbose))
