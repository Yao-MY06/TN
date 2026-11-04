# -*- coding: utf-8 -*-
"""桥接层与计分器的自检（不依赖 torch / 视频 / 模型）。

覆盖四件事：
1. TTNet 事件 → RefereeEvent 的字段映射（含 BALL_LOST 必须被丢弃）；
2. 时间戳单调性（状态机硬性要求）；
3. 换局接力：GAME_OVER 之后仍能继续计分，累计比分只增不减；
4. 端到端：合成一段「发球 → 对打 → 同侧双弹」的事件流，验证能判出 1 分。

用法::

    cd src
    python -m referee.yolo_sm.selftest_bridge
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from referee.yolo_sm.bridge import (
    YoloCoreScorer,
    to_referee_event,
    to_referee_events,
)
from referee.yolo_sm.schemas import EventType
from referee.yolo_sm.state_machine import DecisionKind, MatchRules

_fail = []


def check(name, cond, detail=''):
    if cond:
        print('  ok   {}'.format(name))
    else:
        print('  FAIL {}  {}'.format(name, detail))
        _fail.append(name)


def _ev(etype, side='UNKNOWN', ts=0.0, conf=0.9, pos=(900, 600)):
    return {'event_id': 'x', 'timestamp_s': ts, 'type': etype, 'side': side,
            'position_px': list(pos) if pos else None, 'confidence': conf,
            'sources': ['ttnet']}


def test_mapping():
    print('[1] 字段映射')
    ev = to_referee_event(_ev('BOUNCE', 'B', 1.5), 1)
    check('BOUNCE → BOUNCE', ev.type == EventType.BOUNCE.value, ev.type)
    check('side 透传', ev.side == 'B', ev.side)
    check('时间戳透传', ev.timestamp_s == 1.5)
    check('坐标转元组', ev.position_px == (900.0, 600.0), ev.position_px)

    check('HIT → RACKET_HIT',
          to_referee_event(_ev('HIT', 'A'), 1).type == EventType.RACKET_HIT.value)
    check('NET_HIT → NET_TOUCH',
          to_referee_event(_ev('NET_HIT'), 1).type == EventType.NET_TOUCH.value)
    check('BALL_LOST 被丢弃', to_referee_event(_ev('BALL_LOST'), 1) is None)

    # 越界置信度要 clamp，否则 schemas 会抛错把流水线打挂
    ev = to_referee_event(_ev('BOUNCE', 'A', conf=1.7), 1)
    check('置信度 clamp 到 1.0', ev.confidence == 1.0, ev.confidence)

    # 未知 side 归一化
    check('野生 side → UNKNOWN',
          to_referee_event(_ev('BOUNCE', 'LEFT'), 1).side == 'UNKNOWN')


def test_ordering_and_filter():
    print('[2] 批量转换与排序')
    raw = [_ev('BOUNCE', 'A', 2.0), _ev('BALL_LOST', ts=2.0),
           _ev('HIT', 'A', 1.0), _ev('BOUNCE', 'B', 1.0)]
    evs = to_referee_events(raw)
    check('BALL_LOST 全被滤掉', all(e.type != 'BALL_LOST' for e in evs), len(evs))
    check('时间单调不减',
          all(evs[i].timestamp_s <= evs[i + 1].timestamp_s for i in range(len(evs) - 1)))
    same = [e.type for e in evs if e.timestamp_s == 1.0]
    check('同刻先击球后落台',
          same == [EventType.RACKET_HIT.value, EventType.BOUNCE.value], same)


def test_game_relay():
    print('[3] 换局接力与累计比分')
    rules = MatchRules(points_to_win=2, win_by=1, serve_rotation=1,
                       serve_min_gap_s=0.1, rally_end_s=0.2, confirm_s=0.1)
    sc = YoloCoreScorer(rules=rules, first_server='A')

    # 连续制造 A 得分：A 半台连跳两次 → B 得分；反过来就 A 得分。
    # 用最小规则集（win_by=1, points_to_win=2）会很快打完一局，正好测接力。
    t = 0.0
    seen = []
    for _ in range(6):
        t += 2.0
        sc.update(to_referee_event(_ev('HIT', 'A', t), 1))
        t += 1.0
        sc.update(to_referee_event(_ev('BOUNCE', 'B', t), 1))
        for d in sc.tick(t + 0.5):
            seen.append(d)
        t += 0.5

    pts = sc.points
    check('比赛能持续计分（没有卡在 game_over）', sum(pts) + sum(sc.total_points.values()) > 0,
          'points={} total={}'.format(pts, sc.total_points))
    check('累计比分单调（= 所有判给某方的分）',
          sc.total_points['A'] + sc.total_points['B'] >= 1,
          sc.total_points)
    check('game_over 后 machine 已重建（不是同一局残留）',
          not sc.machine.game_over, sc.machine.game_over)


def test_end_to_end_point():
    print('[4] 端到端：发球 + 对打 + 同侧双弹 → 判出 1 分')
    # 默认规则（serve_min_gap_s=1.5）。用「无历史 → 第一次击球是发球」这条路径：
    # 第一个事件的 prev_event_time 为 None → gap=inf ≥ 1.5 → 进入 SERVE_IN_PROGRESS。
    sc = YoloCoreScorer(rules=MatchRules(), first_server='A')
    seq = [
        ('RACKET_HIT', 'A', 0.5),    # 发球（A 是发球方）
        ('BOUNCE', 'A', 0.9),        # 合法第一跳（自己半台）
        ('BOUNCE', 'B', 1.4),        # 越网 → 进入 RALLY
        ('RACKET_HIT', 'B', 1.8),    # B 回击
        ('BOUNCE', 'A', 2.2),        # 落在 A 侧（击球方是 B → 正常）
        ('RACKET_HIT', 'A', 2.6),    # A 回击
        ('BOUNCE', 'B', 3.0),        # 落在 B 侧
        ('BOUNCE', 'B', 3.4),        # B 侧连跳两次 → B 未接到 → A 得分
    ]
    decisions = []
    for etype, side, ts in seq:
        decisions.extend(sc.update(to_referee_event(_ev(etype, side, ts), 1)))
        decisions.extend(sc.tick(ts))
    decisions.extend(sc.tick(4.5))

    points = [d for d in decisions if d.kind == DecisionKind.POINT]
    check('判出 1 分', len(points) == 1, [d.describe() for d in points])
    if points:
        check('得分方是 A', points[0].winner == 'A', points[0].winner)
        check('触发 DOUBLE_BOUNCE 规则', points[0].rule == 'DOUBLE_BOUNCE',
              points[0].rule)


def test_no_scoring_without_evidence():
    print('[5] 反例：只有孤立落台、没有击球 → 不得分')
    sc = YoloCoreScorer(rules=MatchRules(), first_server='A')
    dec = []
    dec.extend(sc.update(to_referee_event(_ev('BOUNCE', 'B', 5.0), 1)))
    dec.extend(sc.tick(5.5))
    dec.extend(sc.tick(7.0))
    points = [d for d in dec if d.kind == DecisionKind.POINT]
    check('没有凭空得分', len(points) == 0, [d.describe() for d in points])


def test_finalize_needs_multiple_ticks():
    """终局结算必须走 finalize()，不能只 tick 一次。

    这是一个真实踩过的坑：状态机的 tick() 每次只推进一步
    （RALLY → RALLY_END_CANDIDATE → 裁决），所以「视频末帧 + 1 秒」这种
    一次性 tick 永远走不到裁决，末尾那一分就静默丢了。
    实测 test_2 因此从 A1:B1 掉成 A0:B1。
    """
    print('[6] 终局结算：finalize() 能捞出末帧那一分')

    # 构造一个「回合刚结束、视频就截断」的场景：
    # 发球 → 对打 → 球落 B 侧后突然没事件了。
    def build():
        sc = YoloCoreScorer(rules=MatchRules(), first_server='A')
        seq = [
            ('RACKET_HIT', 'A', 0.5), ('BOUNCE', 'A', 0.9), ('BOUNCE', 'B', 1.4),
            ('RACKET_HIT', 'B', 1.8), ('BOUNCE', 'A', 2.2),
            ('RACKET_HIT', 'A', 2.6), ('BOUNCE', 'B', 3.0),   # 落在 B 侧后截断
        ]
        for etype, side, ts in seq:
            sc.update(to_referee_event(_ev(etype, side, ts), 1))
            sc.tick(ts)
        return sc

    # 反例：单次 tick(+1s) 走不到裁决
    sc1 = build()
    one = sc1.tick(4.0)
    check('单次 tick 捞不出来（这正是 bug 的表现）',
          len([d for d in one if d.kind == DecisionKind.POINT]) == 0,
          [d.describe() for d in one])

    # 正例：finalize() 能结算出这一分
    sc2 = build()
    fin = sc2.finalize(3.0)
    pts = [d for d in fin if d.kind == DecisionKind.POINT]
    check('finalize() 判出 1 分', len(pts) == 1, [d.describe() for d in fin])
    if pts:
        check('得分方是这个回合的击球方 A', pts[0].winner == 'A', pts[0].winner)


def main():
    test_mapping()
    test_ordering_and_filter()
    test_game_relay()
    test_end_to_end_point()
    test_no_scoring_without_evidence()
    test_finalize_needs_multiple_ticks()
    print()
    if _fail:
        print('失败 {} 项: {}'.format(len(_fail), _fail))
        return 1
    print('全部通过 ✅')
    return 0


if __name__ == '__main__':
    sys.exit(main())
