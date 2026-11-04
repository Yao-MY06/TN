# -*- coding: utf-8 -*-
"""事件层优化的回归测试（不需要 torch / 视频 / 状态机）。

守住三条实测踩过的坑，每条都对应一次真实回归：

1. **击球历史不能因「短暂遮挡」被清空**（最严重，曾让 test_2 判出 11 分）；
2. **HIT 侧别默认按位置推断**，方向推断默认关闭（开关不能被悄悄改成默认开）；
3. **事件类型名要区分**：`EventExtractor` 产出 `NET_HIT`，
   桥接后才变成 `NET_TOUCH`，混用会得到空列表。

用法::

    cd src
    python -m referee.selftest_event_layer
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from referee.event_processing import EventExtractor, TrajectoryProcessor
from referee.unified_types import BallObservation, EventType, Observation, Side

_fail = []


def check(name, cond, detail=''):
    if cond:
        print('  ok   {}'.format(name))
    else:
        print('  FAIL {}  {}'.format(name, detail))
        _fail.append(name)


def _obs(fid, vis, xy=None, bounce=None, net=None, seg_break=False):
    ball = (BallObservation(visible=True, center_px=list(xy), confidence=0.9)
            if vis else BallObservation(visible=False))
    return Observation(frame_id=fid, timestamp_s=fid / 120.0, source='ttnet',
                       image_size=[1920, 1080], ball=ball,
                       event_scores={'bounce': bounce, 'net_hit': net},
                       extras={'table_box': [360.0, 500.0, 1120.0, 150.0]})


def test_hit_survives_brief_occlusion():
    """球短暂丢失（< hit_hist_reset_s）不应清空击球历史。

    这是最严重的一次回归：原实现在「segment 变化」时就清空 `_pt_hist`，
    而 TTNet 球可见率仅 25%~48%，segment 一次回合内能变几十次，
    导致 HIT 被系统性地砍掉一半，状态机的 last_hitter 长期错位，
    test_2 因此判出 11 分（真值 3 个回合）。
    """
    print('[1] 短暂遮挡不清空击球历史')
    tracker = TrajectoryProcessor()
    ex = EventExtractor()

    # 造一段「水平向右飞」然后「向左飞」的轨迹，中间挖掉 0.15s 的球。
    # 120fps，lost frames = 18 > max_gap_frames(12) → 必然触发 segment +1。
    hits = 0
    fid = 0
    # 阶段 A：向右飞（x 递增），每帧 12px → 1440px/s
    for _ in range(8):
        obs = _obs(fid, True, (300 + 12 * 6, 500))
        tr = tracker.update(obs)
        hits += sum(1 for e in ex.process(obs, tr) if e.type == EventType.HIT)
        fid += 1
    # 遮挡 18 帧（0.15s），强行让 segment 变化
    for _ in range(18):
        obs = _obs(fid, False)
        tr = tracker.update(obs)
        ex.process(obs, tr)
        fid += 1
    # 阶段 B：向左飞（x 递减）→ 方向反转应被检出为一次击球
    base_x = 600
    for k in range(8):
        obs = _obs(fid, True, (base_x - 12 * k, 500))
        tr = tracker.update(obs)
        hits += sum(1 for e in ex.process(obs, tr) if e.type == EventType.HIT)
        fid += 1

    check('遮挡 0.15s 后仍检出击球', hits >= 1, 'hits={}'.format(hits))


def test_long_loss_does_clear():
    """反过来：球断了很久（> hit_hist_reset_s）时必须清空，避免拼假轨迹。"""
    print('[2] 长时间断球会清空击球历史（反例保护）')
    tracker = TrajectoryProcessor()
    ex = EventExtractor(hit_hist_reset_s=0.40)

    fid = 0
    for _ in range(8):
        obs = _obs(fid, True, (300 + 12 * 6, 500))
        ex.process(obs, tracker.update(obs))
        fid += 1
    # 断 1.0s（120 帧），远超 hit_hist_reset_s
    for _ in range(120):
        obs = _obs(fid, False)
        ex.process(obs, tracker.update(obs))
        fid += 1
    before = len(ex._pt_hist)
    hits = 0
    for k in range(8):
        obs = _obs(fid, True, (600 - 12 * k, 500))
        tr = tracker.update(obs)
        hits += sum(1 for e in ex.process(obs, tr) if e.type == EventType.HIT)
        fid += 1
    check('长断后历史已清空', before == 0, 'pt_hist={}'.format(before))
    # 清空后点数不够（需 6 个），不应立刻凭空报一次击球
    check('长断后不凭空产生击球', hits == 0, 'hits={}'.format(hits))


def test_attribution_default_off():
    """HIT 侧别默认按位置推断；方向推断必须显式开启。"""
    print('[3] HIT 侧别归属的默认值')
    ex = EventExtractor()
    check('hit_side_by_direction 默认 False', ex.hit_side_by_direction is False,
          ex.hit_side_by_direction)
    check('net_thresh 默认 0.65', abs(ex.net_thresh - 0.65) < 1e-9, ex.net_thresh)
    check('min_event_interval_s 默认 0.25',
          abs(ex.min_event_interval_s - 0.25) < 1e-9, ex.min_event_interval_s)
    check('hit_hist_reset_s 默认 0.40',
          abs(ex.hit_hist_reset_s - 0.40) < 1e-9, ex.hit_hist_reset_s)


def test_event_name_mapping():
    """EventExtractor 产出 NET_HIT；桥接后才叫 NET_TOUCH。混用会全是空。"""
    print('[4] 事件名语义（NET_HIT vs NET_TOUCH）')
    check('EventType.NET_HIT 存在', hasattr(EventType, 'NET_HIT'))

    from referee.yolo_sm.bridge import to_referee_event
    ev = to_referee_event({'event_id': 'n1', 'timestamp_s': 1.0,
                           'type': 'NET_HIT', 'side': 'UNKNOWN',
                           'position_px': None, 'confidence': 0.9,
                           'sources': ['ttnet']}, 1)
    check('NET_HIT 桥接为 NET_TOUCH', ev is not None and ev.type == 'NET_TOUCH',
          getattr(ev, 'type', None))


def test_dedup_interval_effect():
    """加大去重间隔后，同一段观测的事件数不应增加（单调性）。"""
    print('[5] 去重间隔的单调性')

    # 造一段持续高分的 bounce 驼峰，间隔应当合并成一个事件
    def run(interval):
        ex = EventExtractor(bounce_thresh=0.65, min_event_interval_s=interval)
        tracker = TrajectoryProcessor()
        n = 0
        for fid in range(120):
            # 每 10 帧一个驼峰，峰值 0.9
            ph = fid % 10
            score = 0.9 if ph in (4, 5) else 0.1
            obs = _obs(fid, True, (400 + 5 * (fid % 20), 560), bounce=score)
            tr = tracker.update(obs)
            n += sum(1 for e in ex.process(obs, tr) if e.type == EventType.BOUNCE)
        n += sum(1 for e in ex.flush() if e.type == EventType.BOUNCE)
        return n

    small = run(0.10)
    large = run(0.25)
    check('大间隔的事件数 <= 小间隔', large <= small,
          'interval0.10={} interval0.25={}'.format(small, large))


def main():
    test_hit_survives_brief_occlusion()
    test_long_loss_does_clear()
    test_attribution_default_off()
    test_event_name_mapping()
    test_dedup_interval_effect()
    print()
    if _fail:
        print('失败 {} 项: {}'.format(len(_fail), _fail))
        return 1
    print('全部通过 ✅')
    return 0


if __name__ == '__main__':
    sys.exit(main())
