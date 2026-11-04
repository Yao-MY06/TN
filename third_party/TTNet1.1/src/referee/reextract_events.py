# -*- coding: utf-8 -*-
"""从「逐帧原始观测」缓存重建事件，无需重跑 TTNet 推理。

为什么需要这个
--------------
``results/cache_test_N.json`` 存的是**逐帧原始观测**（球位置 / 有无 / bounce 与
net 的分数 / 球台框），而 ``results/infer_test_N_full.json`` 存的是**已提取的事件**。

事件提取逻辑（阈值、驼峰检测、击球侧别归属…）一旦改动，只改代码是没用的 ——
旧的事件缓存已经把结论冻住了。重跑推理要几十分钟到几小时，而观测缓存就在这里，
因此从观测缓存重放一遍事件提取，几秒钟就能得到新结论。

这就把「推理」和「事件提取 + 判分」彻底解耦了：
* 换阈值 / 换击球归属规则 / 换状态机 → 全部秒级验证；
* 只有换权重 / 换输入分辨率才需要真的重跑推理。

用法::

    cd src
    # 重建 test_1 的事件缓存（覆盖 results/infer_test_1_full.json 前先备份）
    python -m referee.reextract_events --game test_1

    # 只对比「旧事件缓存」与「新规则下的事件」差异，不写文件
    python -m referee.reextract_events --game test_1 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from referee.event_processing import TrajectoryProcessor, EventExtractor
from referee.unified_types import BallObservation, Observation


ROOT = os.path.join(os.path.dirname(__file__), '..', '..')


def _obs_from_row(row: Dict[str, Any]) -> Observation:
    """把 cache_test_N.json 的一行还原成 Observation（§7.1 契约）。

    注意：``v=False`` 时 BallObservation 的其它字段必须全为 None。
    观测缓存里 ``p`` 已经是 None，这里直接透传即可。
    """
    visible = bool(row.get('v'))
    ball = BallObservation(
        visible=visible,
        center_px=list(row['p']) if visible and row.get('p') else None,
        bbox_xyxy=None,
        confidence=(float(row['cp']) if visible and row.get('cp') is not None else None),
    )
    # 观测缓存可能没有逐帧置信度（老版本），那就把事件分数当不了置信度，留 None。
    if visible and ball.confidence is None:
        ball = BallObservation(visible=True, center_px=list(row['p']),
                               bbox_xyxy=None, confidence=None)

    extras = {}
    if row.get('tb'):
        extras['table_box'] = list(row['tb'])

    return Observation(
        frame_id=int(row['f']),
        timestamp_s=float(row['t']),
        source='ttnet',
        image_size=[1920, 1080],
        ball=ball,
        event_scores={'bounce': row.get('b'), 'net_hit': row.get('n')},
        extras=extras,
    )


def rebuild(obs_cache: Dict[str, Any], bounce_thresh: float = 0.65,
            net_thresh: float = 0.5, filter_suspect_hit: bool = False,
            hit_side_by_direction: bool = False, verbose: bool = True,
            min_event_interval_s: float = 0.25):
    """从逐帧观测重建「帧号 → 可见性」与事件列表。

    :param min_event_interval_s: 同类事件的最小间隔（去重强度），
        透传给 ``EventExtractor``。调大可抑制驼峰拖尾产生的重复事件。
    """
    tracker = TrajectoryProcessor()
    extractor = EventExtractor(bounce_thresh=bounce_thresh, net_thresh=net_thresh,
                               filter_suspect_hit=filter_suspect_hit,
                               hit_side_by_direction=hit_side_by_direction,
                               min_event_interval_s=min_event_interval_s)

    frame_obs: List[List[Any]] = []
    events: List[Dict[str, Any]] = []
    detected = {'BOUNCE': 0, 'NET_HIT': 0, 'HIT': 0, 'BALL_LOST': 0}

    def _handle(evs):
        for ev in evs:
            detected[ev.type] = detected.get(ev.type, 0) + 1
            events.append(ev.to_dict())

    for row in obs_cache['rows']:
        obs = _obs_from_row(row)
        traj = tracker.update(obs)
        _handle(extractor.process(obs, traj))
        frame_obs.append((obs.frame_id, round(obs.timestamp_s, 5),
                          1 if obs.ball.visible else 0))
    _handle(extractor.flush())

    if verbose:
        print('  重建完成：{} 帧，事件 {}'.format(len(frame_obs), detected))
    meta = obs_cache.get('meta', {})
    return {
        'frames': frame_obs,
        'events': events,
        'detected': detected,
        'low_conf_dropped': meta.get('low_conf_dropped', 0),
        'frozen_frames_dropped': meta.get('frozen_dropped', 0),
    }


def _stats(cache: Dict[str, Any]) -> Dict[str, Any]:
    evs = [e for e in cache['events'] if e['type'] != 'BALL_LOST']
    hits = [e for e in evs if e['type'] == 'HIT']
    bounces = [e for e in evs if e['type'] == 'BOUNCE']
    # 「击球侧」与「随后首个落台侧」同侧的比例。物理上几乎不可能同侧
    # （球员必须把球打向对方半台），因此这个比例越低越说明归属是对的。
    same = diff = 0
    for h in hits:
        nxt = [b for b in bounces if b['timestamp_s'] > h['timestamp_s']]
        if not nxt:
            continue
        nb = nxt[0]
        if nb['timestamp_s'] - h['timestamp_s'] > 1.5:
            continue
        if nb['side'] == h['side']:
            same += 1
        elif nb['side'] in ('A', 'B') and h['side'] in ('A', 'B'):
            diff += 1
    vis = (sum(f[2] for f in cache['frames']) / len(cache['frames'])
           if cache['frames'] else 0.0)
    return {'events': len(evs), 'hits': len(hits), 'bounces': len(bounces),
            'hit_followed_by_same_side_bounce': same,
            'hit_followed_by_other_side_bounce': diff,
            'ball_visibility': round(vis, 4)}


def main(games, cache_dir=None, bounce_thresh=0.65, net_thresh=0.65,
         dry_run=False, filter_suspect_hit=False, hit_side_by_direction=False):
    cache_dir = cache_dir or os.path.join(ROOT, 'results')
    rc = 0
    for game in games:
        obs_path = os.path.join(cache_dir, 'cache_{}.json'.format(game))
        if not os.path.isfile(obs_path):
            print('[跳过] {} 没有观测缓存 {}'.format(game, obs_path))
            rc = 1
            continue

        print('\n' + '=' * 74)
        print('{}  观测缓存: {}'.format(game, os.path.basename(obs_path)))
        print('=' * 74)
        with open(obs_path, encoding='utf-8') as f:
            obs_cache = json.load(f)

        new = rebuild(obs_cache, bounce_thresh, net_thresh,
                      filter_suspect_hit=filter_suspect_hit,
                      hit_side_by_direction=hit_side_by_direction)

        old_path = os.path.join(cache_dir, 'infer_{}_full.json'.format(game))
        if os.path.isfile(old_path):
            with open(old_path, encoding='utf-8') as f:
                old = json.load(f)
            so, sn = _stats(old), _stats(new)
            print('  旧缓存: 事件 {events} | HIT {hits} | 同侧落台 '
                  '{hit_followed_by_same_side_bounce} / 异侧 '
                  '{hit_followed_by_other_side_bounce}'.format(**so))
            print('  新重建: 事件 {events} | HIT {hits} | 同侧落台 '
                  '{hit_followed_by_same_side_bounce} / 异侧 '
                  '{hit_followed_by_other_side_bounce}'.format(**sn))
        else:
            print('  （无旧事件缓存可对比）')

        if dry_run:
            print('  [dry-run] 不写入')
            continue

        out_path = os.path.join(cache_dir, 'infer_{}_full.json'.format(game))
        if os.path.isfile(out_path):
            bak = out_path + '.bak'
            if not os.path.isfile(bak):
                os.replace(out_path, bak)
                print('  已备份旧缓存 → {}'.format(os.path.basename(bak)))
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(new, f, ensure_ascii=False)
        print('  已写入 → {}'.format(out_path))
    return rc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--game', nargs='+', default=['test_1'])
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--bounce-thresh', type=float, default=0.65)
    ap.add_argument('--net-thresh', type=float, default=0.5)
    ap.add_argument('--filter-suspect-hit', action='store_true')
    ap.add_argument('--hit-side-by-direction', action='store_true',
                    help='击球侧按飞行方向推断（R-017 做法）。默认关闭 —— 该项目实测无收益')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    sys.exit(main(a.game, a.cache_dir, a.bounce_thresh, a.net_thresh,
                  a.dry_run, a.filter_suspect_hit, a.hit_side_by_direction))
