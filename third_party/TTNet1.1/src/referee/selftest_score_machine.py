# -*- coding: utf-8 -*-
"""计分状态机回归自测（无需模型、无需视频，纯事件重放）。

覆盖 v3 的两类要求：

**必须挡掉的三类现场误得分**（用户实际遇到的问题）
    1. 手摸 / 擦桌面            —— 孤立落台 + 长时间静默，不得分；
    2. 桌面颠球                 —— 同侧反复落台，不得分；
    3. 换发球时传球过网         —— 无球拍击球的「本方→对方」落台，不得分。

**必须保留的正常判分**（v2 曾在此大量漏判）
    4. 真实回合·同侧双弹        —— 对手没接到，必须判分；
    5. 真实回合·球出界静默      —— 球飞出台面无人接回，必须判分；
    6. 真实回合·视频末尾截断    —— 末尾残留回合要结算；
    7. 发球直接得分             —— 发球后对手完全没碰到，必须判分。

用法::

    cd src
    python -m referee.selftest_score_machine
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from referee.score_state_machine import Phase, TableTennisStateMachine
from referee.unified_types import Event, EventType, Side


def ev(etype, ts, side=Side.UNKNOWN, conf=None, eid=None):
    return Event(event_id=eid or '{}_{:.3f}'.format(etype, ts), timestamp_s=ts,
                 type=etype, side=side, confidence=conf,
                 position_px=[900, 600], sources=['selftest'])


def replay(machine, events, end_ts, step=0.05):
    """按时间推进回放事件，模拟渲染链路的逐帧 tick。"""
    events = sorted(events, key=lambda e: (e.timestamp_s, e.type))
    i = 0
    ts = 0.0
    while ts <= end_ts + 1e-9:
        while i < len(events) and events[i].timestamp_s <= ts + 1e-9:
            machine.consume(events[i])
            i += 1
        machine.tick_deep(ts)
        ts += step
    machine.finalize(end_ts)
    return machine


def rally_prefix(t0=1.0, a_serve=True):
    """构造一段「像正式比赛」的前置对打：合法发球 + 两个来回。

    用来满足形态校验（跨网交替 + 球拍击球），模拟真实视频里得分之前
    一定已经打过几个来回。返回 (events, 下一拍的起始时间)。
    """
    A, B = Side.A, Side.B
    srv, rcv = (A, B) if a_serve else (B, A)
    evs = [
        ev(EventType.HIT, t0 - 0.2, srv),
        ev(EventType.BOUNCE, t0, srv, 0.9),          # 本方台面
        ev(EventType.BOUNCE, t0 + 0.5, rcv, 0.9),    # 跨网
    ]
    t = t0 + 0.5
    for hitter, land in ((rcv, srv), (srv, rcv)):
        evs.append(ev(EventType.HIT, t + 0.3, hitter))
        evs.append(ev(EventType.BOUNCE, t + 0.6, land, 0.9))
        t += 0.6
    return evs, t


# ---------------------------------------------------------------- 必须挡掉
def case_hand_touch_table():
    """手摸桌面：孤立落台误报 + 长时间静默，绝不能得分。"""
    events = [
        ev(EventType.HIT, 1.0, Side.A),
        ev(EventType.BOUNCE, 1.2, Side.A, 0.9),   # 手碰桌面被误报为落台
    ]
    m = replay(TableTennisStateMachine(verbose=False, cumulative_mode=True),
               events, end_ts=20.0)
    assert len(m.points) == 0, '手摸桌面被判分: {}'.format(m.points)
    return '孤立误报落台 + 静默 18s → 0 分'


def case_juggle_on_table():
    """桌面颠球：同侧反复落台，属于准备动作，绝不能得分。"""
    events = []
    t = 1.0
    for _ in range(12):                      # 连续 12 次同侧落台，间隔 0.6s
        events.append(ev(EventType.HIT, t - 0.25, Side.A))
        events.append(ev(EventType.BOUNCE, t, Side.A, 0.95))
        t += 0.6
    m = replay(TableTennisStateMachine(verbose=False, cumulative_mode=True),
               events, end_ts=30.0)
    assert len(m.points) == 0, '桌面颠球被判分: {}'.format(m.points)
    return '12 次同侧连弹（含击球）→ 0 分'


def case_throw_across_after_point():
    """换发球时把球从桌面扔到对面：无球拍击球，不得分。"""
    pre, t = rally_prefix(t0=1.0)
    events = list(pre)
    # 结束这一分（同侧双弹 → 对方得分）
    events.append(ev(EventType.HIT, t + 0.3, Side.B))
    events.append(ev(EventType.BOUNCE, t + 0.6, Side.A, 0.9))
    events.append(ev(EventType.BOUNCE, t + 1.0, Side.A, 0.9))
    first_pt = t + 1.0
    # 换发球：拿球后直接把球从桌面扔/拍到对面 —— 全程无球拍击球
    events.append(ev(EventType.BOUNCE, first_pt + 3.5, Side.B, 0.9))
    events.append(ev(EventType.BOUNCE, first_pt + 4.1, Side.A, 0.9))
    m = replay(TableTennisStateMachine(verbose=False, cumulative_mode=True),
               events, end_ts=first_pt + 15.0)
    assert len(m.points) == 1, '传球过网被判分: {} 分 {}'.format(
        len(m.points), [(p['ts'], p['reason']) for p in m.points])
    return '正常 1 分 + 传球过网 → 仍为 1 分'


# ---------------------------------------------------------------- 必须保留
def case_real_rally_double_bounce():
    """真实回合：对手没接到，同一侧连弹两次 → 必须判分。"""
    pre, t = rally_prefix(t0=1.0)          # 下一个落台在 A 侧结束
    events = list(pre)
    events.append(ev(EventType.HIT, t + 0.3, Side.B))
    events.append(ev(EventType.BOUNCE, t + 0.6, Side.A, 0.9))
    events.append(ev(EventType.BOUNCE, t + 1.0, Side.A, 0.9))   # A 侧两连弹
    m = replay(TableTennisStateMachine(verbose=False, cumulative_mode=True),
               events, end_ts=30.0)
    assert len(m.points) == 1, '正常回合未判分或重复判分: {}'.format(m.points)
    assert m.points[0]['winner'] == Side.B, m.points[0]
    return 'A 侧两连弹 → B 得 1 分（不多不少）'


def case_real_rally_ball_out_silence():
    """真实回合：球打飞出台，之后长时间无事件 → 必须判分（静默结算）。"""
    pre, t = rally_prefix(t0=1.0)
    events = list(pre)
    events.append(ev(EventType.HIT, t + 0.3, Side.A))
    events.append(ev(EventType.BOUNCE, t + 0.6, Side.B, 0.9))   # 落在 B 侧后球飞走
    m = replay(TableTennisStateMachine(verbose=False, cumulative_mode=True),
               events, end_ts=30.0)
    assert len(m.points) == 1, '球出界静默未判分: {}'.format(m.points)
    assert m.points[0]['winner'] == Side.A, m.points[0]
    return '球落在 B 侧后飞走 → A 得 1 分'


def case_real_rally_video_cut():
    """真实回合打到一半视频结束 → 末尾必须结算，不能白丢一分。"""
    pre, t = rally_prefix(t0=1.0)
    events = list(pre)
    events.append(ev(EventType.HIT, t + 0.3, Side.A))
    events.append(ev(EventType.BOUNCE, t + 0.6, Side.B, 0.9))
    m = replay(TableTennisStateMachine(verbose=False, cumulative_mode=True),
               events, end_ts=t + 1.2)     # 视频在回合中途截断
    assert len(m.points) == 1, '视频末尾未结算: {}'.format(m.points)
    return '回合中途视频结束 → 结算 1 分'


def case_serve_ace():
    """发球直接得分：发球后对手完全没碰到球 → 必须判分。

    这是 v2 漏判的典型场景：v2 要求「回合内两侧都落过台」才判分，
    而发球 ACE 只有「本方落台 → 对方落台 → 对方再弹」，
    若发球那次本方落台漏检（可见率仅 27%），就被整段丢掉。
    """
    events = [
        ev(EventType.HIT, 0.8, Side.A),          # 发球击球
        ev(EventType.BOUNCE, 1.0, Side.A, 0.9),  # 本方台面
        ev(EventType.BOUNCE, 1.5, Side.B, 0.9),  # 对方台面（对手没碰到）
        ev(EventType.BOUNCE, 2.0, Side.B, 0.9),  # 对方台面再弹一次
    ]
    m = replay(TableTennisStateMachine(verbose=False, cumulative_mode=True),
               events, end_ts=20.0)
    assert len(m.points) == 1, '发球直接得分未判分: {}'.format(m.points)
    assert m.points[0]['winner'] == Side.A, m.points[0]
    return '发球 ACE（B 侧两连弹）→ A 得 1 分'


# ------------------------------------------------ 真实样本回归（test_4）
def _replay_with_visibility(events, end_ts, visible_ranges, **machine_kw):
    """构造带球可见性数据的回放。

    :param visible_ranges: [(t0, t1)] 在这些区间内球可见，其余不可见
    """
    machine = TableTennisStateMachine(verbose=False, cumulative_mode=True,
                                      **machine_kw)
    samples = []
    t = 0.0
    while t <= end_ts:
        vis = 1 if any(a <= t <= b for a, b in visible_ranges) else 0
        samples.append((t, vis))
        t += 1.0 / 120.0
    machine.set_visibility_source(samples)
    return replay(machine, events, end_ts)


def case_real_sample_true_point():
    """真实样本：test_4 的 7.39s（用户确认应判 B 得分），必须判出。

    该点特征：A 侧同侧双弹（6.91s → 7.39s），球全程可见，之前有球拍击球。
    v3 曾要求「窗口内必须出现跨网交替」，把它误杀了 —— 而真实的一分
    完全可以是单侧两弹（对手没接到）。这正是用户本次反馈的漏判。
    """
    events = [
        ev(EventType.HIT, 4.31, Side.A),
        ev(EventType.BOUNCE, 6.91, Side.A, 0.89),
        ev(EventType.BOUNCE, 7.39, Side.A, 0.95),
    ]
    m = _replay_with_visibility(events, 20.0, visible_ranges=[(6.8, 7.6)])
    assert len(m.points) == 1, '真实得分被漏判: {}'.format(m.points)
    assert m.points[0]['winner'] == Side.B, m.points[0]
    return '7.39s A 侧双弹（球可见）→ B 得 1 分'


def case_real_sample_false_point():
    """真实样本：test_4 的 24.70s（用户确认为误判，只是摸了一下球），必须挡住。

    该点特征：A 侧"双弹"（24.44s → 24.70s），但**球可见率仅 20%** ——
    模型在没看到球的情况下凭空报了落台（幻觉），置信度也偏低（0.65）。
    这类误判只能靠「球是否真的被看到」拦，靠击球/跨网都拦不住。
    """
    events = [
        ev(EventType.HIT, 24.33, Side.A),
        ev(EventType.BOUNCE, 24.44, Side.A, 0.65),
        ev(EventType.BOUNCE, 24.70, Side.A, 0.78),
    ]
    # 只在极短瞬间可见，模拟模型幻觉
    m = _replay_with_visibility(events, 40.0, visible_ranges=[(24.40, 24.46)])
    assert len(m.points) == 0, '幻觉落台被判分: {}'.format(m.points)
    return '24.70s 双弹但球几乎不可见 → 0 分'


def case_paddle_marks_bounce_hit():
    """球拍过滤器 A：球贴台面高度带内的 HIT 应被标记为可疑。

    依据：TTNet 的 HIT 由轨迹方向反转推断，球撞台面同样会反转。
    真实击球时球在空中，撞台时球贴台 —— 这个几何差别可用于标记。
    """
    from referee.paddle_detector import PaddleEvidenceFilter

    f = PaddleEvidenceFilter()
    # 喂入足够多的落台点，估计台面高度带（y 越大越靠下）
    for i in range(40):
        f.observe_bounce(i * 0.5, [900, 660 + (i % 5)])
    assert f._table_y_lo is not None, '未能估计台面高度带'

    # 球在台面高度带内 → 可疑
    bad, reasons = f.is_hit_suspicious(20.0, [900, 700])
    assert bad, '贴台的 HIT 未被标记'
    assert 'ball_in_table_zone' in reasons, reasons

    # 球明显在台面上方（空中）→ 不应被标记
    ok, reasons2 = f.is_hit_suspicious(21.0, [900, 480])
    assert not ok, '空中的 HIT 被误标: {}'.format(reasons2)
    return '贴台 HIT 被标记 / 空中 HIT 不标'


def case_paddle_one_way_veto():
    """球拍过滤器 B：明确看到两侧球拍都远离球时，否决该击球。

    这是零样本球拍检测唯一能被安全利用的方式：
    **只在有把握时否决，没检出球拍时不表态**（否则会因漏检误杀）。
    """
    from referee.paddle_detector import PaddleEvidenceFilter

    f = PaddleEvidenceFilter()
    # 两块拍都离球很远（球在 900,300，拍在 100,700 和 1700,700）
    f.set_paddle_source({10.0: [([80, 650, 200, 780], 'A'),
                               ([1650, 650, 1780, 780], 'B')]})
    bad, reasons = f.is_hit_suspicious(10.0, [900, 300])
    assert bad and any('paddles_far' in r for r in reasons), reasons

    # 没有球拍数据时不能否决（不表态）
    f2 = PaddleEvidenceFilter()
    ok, reasons2 = f2.is_hit_suspicious(10.0, [900, 300])
    assert not ok, '无球拍数据时不应否决: {}'.format(reasons2)
    return '球拍远离→否决 / 无数据→不表态'


CASES = [
    ('手摸/擦桌面', case_hand_touch_table),
    ('桌面颠球', case_juggle_on_table),
    ('换发球传球过网', case_throw_across_after_point),
    ('真实回合·双弹', case_real_rally_double_bounce),
    ('真实回合·球出界', case_real_rally_ball_out_silence),
    ('真实回合·视频截断', case_real_rally_video_cut),
    ('发球直接得分', case_serve_ace),
    ('实测样本·真分', case_real_sample_true_point),
    ('实测样本·幻觉分', case_real_sample_false_point),
    ('球拍·台面弹跳标记', case_paddle_marks_bounce_hit),
    ('球拍·单向否决', case_paddle_one_way_veto),
]


def main():
    print('=' * 70)
    print('计分状态机 v4 回归自测')
    print('=' * 70)
    failed = 0
    for name, fn in CASES:
        try:
            detail = fn()
            print('  ✅ {:<18} {}'.format(name, detail))
        except AssertionError as e:
            failed += 1
            print('  ❌ {:<18} {}'.format(name, e))
    print('-' * 70)
    print('  {}/{} 通过'.format(len(CASES) - failed, len(CASES)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
