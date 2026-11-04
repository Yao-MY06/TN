# -*- coding: utf-8 -*-
"""事件层优化的对照实验（不改状态机）。

目的
----
YOLO 状态机对「假击球」很敏感：`_on_racket_hit` 会清空 `last_bounce_side`，
一个误报的 HIT 就能把整段的判罚证据擦掉。已确认 test_2 回合 2 漏判就是这个原因。

因此在不改状态机的前提下，唯一有效的杠杆是**把假击球在上游滤掉**。
本脚本对比几种过滤策略，用可验证的指标挑出最优者：

评估指标（都可测，不依赖缺失的逐分真值）
--------------------------------------
1. **test_2 判分数** —— 唯一有回合数真值的样本，真值 3 分；
2. **OWN_COURT 次数** —— 业余对打中「球落在击球方自己半台」极罕见，越少越好；
3. **死球占比** —— 状态机拒绝判分的比例，越低说明证据越完整；
4. **HIT 数量** —— 过滤后还剩多少，太少会让状态机缺击球证据。

用法::

    cd src
    python -m referee.experiment_hit_filter
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from referee.yolo_sm.bridge import to_referee_events
from referee.yolo_sm.state_machine import MatchRules


ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
GAMES = ('test_1', 'test_2', 'test_3')
GT_RALLIES = {'test_2': 3}    # 只有 test_2 有回合数真值


# ---------------------------------------------------------------- 过滤器
def filter_hits(events: List[Dict[str, Any]], *,
                drop_near_bounce_s: Optional[float] = None,
                drop_all_suspect: bool = False) -> List[Dict[str, Any]]:
    """按策略过滤 HIT 事件，其余事件原样保留。

    :param drop_near_bounce_s: 丢弃「距最近 BOUNCE ≤ 该秒数」的 HIT。
        依据：球撞台面会让 x 方向速度反号，被 `_detect_hit` 误判成击球；
        这类假击球的时刻必然紧贴落台。真实击球时球在台面上方，与落台的
        时间差通常在 0.3s 以上（球要飞过半个球台）。
    :param drop_all_suspect: 丢弃带 ``suspect_bounce`` 标记的 HIT
        （由 `PaddleEvidenceFilter` 的「台面高度带」判据标记）。
    """
    bounce_ts = [e['timestamp_s'] for e in events if e['type'] == 'BOUNCE']
    out = []
    for e in events:
        if e['type'] == 'HIT':
            if drop_all_suspect and 'suspect_bounce' in (e.get('sources') or []):
                continue
            if drop_near_bounce_s is not None and bounce_ts:
                d = min(abs(e['timestamp_s'] - b) for b in bounce_ts)
                if d <= drop_near_bounce_s:
                    continue
        out.append(e)
    return out


# ---------------------------------------------------------------- 跑分
def score(cache: Dict[str, Any], rules: MatchRules):
    """用 YOLO 状态机跑一遍，返回判分统计。"""
    from referee.render_with_ui import build_score_timeline

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _, _, _, m = build_score_timeline(cache, engine='yolo', rules=rules)
    total = sum(m.points)
    dead = (m.rule_counts.get('NO_EVIDENCE', 0)
            + m.rule_counts.get('SERVE_INCOMPLETE', 0))
    n_dec = len(m.all_decisions)
    return {
        'points': (m.points[0], m.points[1]),
        'total': total,
        'own_court': m.rule_counts.get('OWN_COURT', 0),
        'no_return': m.rule_counts.get('NO_RETURN', 0),
        'double_bounce': m.rule_counts.get('DOUBLE_BOUNCE', 0),
        'serve_fault': m.rule_counts.get('SERVE_FAULT', 0),
        'dead': dead,
        'dead_ratio': dead / n_dec if n_dec else 0.0,
        'decisions': n_dec,
        'hit_count': sum(1 for e in cache['events'] if e['type'] == 'HIT'),
        'bounce_count': sum(1 for e in cache['events'] if e['type'] == 'BOUNCE'),
    }


def main():
    from referee.reextract_events import rebuild

    rules = MatchRules()
    caches = {g: json.load(open(os.path.join(ROOT, 'results',
                                             'infer_{}_full.json'.format(g)),
                                encoding='utf-8'))
              for g in GAMES}

    variants = [
        ('基线（现状）', {}),
        ('滤掉紧邻落台的HIT(0.10s)', {'drop_near_bounce_s': 0.10}),
        ('滤掉紧邻落台的HIT(0.15s)', {'drop_near_bounce_s': 0.15}),
        ('滤掉紧邻落台的HIT(0.25s)', {'drop_near_bounce_s': 0.25}),
    ]

    print('=' * 96)
    print('事件层 HIT 过滤策略对照（状态机代码未改，只改送进去的事件）')
    print('=' * 96)
    header = '{:<26}{:>10}{:>10}{:>8}{:>8}{:>10}{:>12}'.format(
        '策略', 'test_1', 'test_2', 'test_3', 'HIT数', 'OWN_COURT', '死球占比')
    print(header)
    print('-' * 96)

    for label, kw in variants:
        row = []
        hits = []
        own = 0
        deadr = []
        for g in GAMES:
            c = dict(caches[g])
            c['events'] = filter_hits(c['events'], **kw)
            s = score(c, rules)
            row.append('A%d:B%d' % s['points'])
            hits.append(s['hit_count'])
            own += s['own_court']
            deadr.append(s['dead_ratio'])
        print('{:<26}{:>10}{:>10}{:>8}{:>8}{:>10}{:>12}'.format(
            label, row[0], row[1], row[2],
            '/'.join(str(h) for h in hits), own,
            '/'.join('%.0f%%' % (d * 100) for d in deadr)))

    print('\ntest_2 真值 = 3 分；OWN_COURT 越少越好；死球占比越低说明证据越完整')


if __name__ == '__main__':
    main()
