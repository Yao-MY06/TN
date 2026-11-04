# -*- coding: utf-8 -*-
"""
TTNet 乒乓球裁判：适配器 → 轨迹/事件处理 → 共用计分状态机。

用法::

    cd src
    python -m referee.run_referee --video ../dataset/test/videos/test_2.mp4

    # 只跑前 N 帧（快速验证）
    python -m referee.run_referee --video ... --max-frames 1200

    # 导出逐帧观测 JSON，便于和 YOLO / TrackNet 横向对比
    python -m referee.run_referee --video ... --dump-observations out.json
"""

import os
import sys
import json
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from referee.ttnet_adapter import TTNetAdapter
from referee.event_processing import TrajectoryProcessor, EventExtractor
from referee.score_state_machine import TableTennisStateMachine
from referee.unified_types import EventType


DEFAULT_WEIGHTS = os.path.join('..', 'checkpoints', 'ttnet_3rd_phase',
                               'ttnet_3rd_phase_epoch_30.pth')


def build_configs(working_dir='../'):
    """构造 TTNet configs（避免依赖命令行参数）。"""
    from config.config import parse_configs

    argv = ['referee', '--working-dir', working_dir, '--saved_fn', 'referee',
            '--gpu_idx', '0', '--smooth-labelling']
    sys.argv = argv
    return parse_configs()


def run(video_path, weights, max_frames=None, dump_path=None,
        bounce_thresh=0.65, net_thresh=0.65, ball_thresh=0.05,
        decode='argmax', device='cuda:0', verbose=True):
    """跑通完整裁判流程，返回统计结果。"""
    configs = build_configs()

    if verbose:
        print('=' * 70)
        print('TTNet 乒乓球裁判流程')
        print('  视频  : {}'.format(video_path))
        print('  权重  : {}'.format(weights))
        print('  设备  : {}'.format(device))
        print('=' * 70)

    # ---- 三个层次，职责清晰
    adapter = TTNetAdapter(configs, weights, device=device,
                           ball_thresh=ball_thresh, decode=decode)
    tracker = TrajectoryProcessor()
    extractor = EventExtractor(bounce_thresh=bounce_thresh, net_thresh=net_thresh)
    machine = TableTennisStateMachine(verbose=verbose)

    all_events = []
    observations = []
    n_frames = 0
    n_visible = 0

    for obs in adapter.observations_from_video(video_path, max_frames=max_frames):
        n_frames += 1
        if obs.ball.visible:
            n_visible += 1

        traj = tracker.update(obs)
        events = extractor.process(obs, traj)

        for ev in events:
            all_events.append(ev)
            machine.consume(ev)
            if verbose and ev.type in (EventType.BOUNCE, EventType.NET_HIT):
                print('  [事件] {:<9} @ {:7.3f}s  side={:<7} conf={}'.format(
                    ev.type, ev.timestamp_s, ev.side,
                    '{:.2f}'.format(ev.confidence) if ev.confidence else '-'))
        machine.tick(obs.timestamp_s)

        if dump_path:
            observations.append(obs.to_dict())

    # 视频结束：补发仍在驼峰中的事件
    for ev in extractor.flush():
        all_events.append(ev)
        machine.consume(ev)
        if verbose and ev.type in (EventType.BOUNCE, EventType.NET_HIT):
            print('  [事件] {:<9} @ {:7.3f}s  side={:<7} conf={}  (尾部补发)'.format(
                ev.type, ev.timestamp_s, ev.side,
                '{:.2f}'.format(ev.confidence) if ev.confidence else '-'))

    # ---- 汇总
    summary = {
        'video': video_path,
        'frames_processed': n_frames,
        'ball_visible_frames': n_visible,
        'ball_visible_rate': round(n_visible / n_frames, 4) if n_frames else 0.0,
        'suspicious_jumps': tracker.suspicious_count,
        'segments': tracker.segment_id + 1,
        'event_counts': _count_events(all_events),
        'event_filter_stats': extractor.stats,
        'scoreboard': machine.scoreboard,
        'points': machine.points,
        'reviews': machine.reviews,
    }

    if dump_path:
        with open(dump_path, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'observations': observations,
                       'events': [e.to_dict() for e in all_events]},
                      f, ensure_ascii=False, indent=1)

    _print_summary(summary)
    return summary


def _count_events(events):
    counts = {}
    for e in events:
        counts[e.type] = counts.get(e.type, 0) + 1
    return counts


def _print_summary(s):
    print()
    print('=' * 70)
    print('处理结果')
    print('=' * 70)
    print('  处理帧数      : {}'.format(s['frames_processed']))
    print('  球可见帧数    : {} ({:.1%})'.format(
        s['ball_visible_frames'], s['ball_visible_rate']))
    print('  轨迹断段数    : {}'.format(s['segments']))
    print('  可疑跳变次数  : {}'.format(s['suspicious_jumps']))
    print('  事件统计      : {}'.format(
        ', '.join('{}={}'.format(k, v) for k, v in s['event_counts'].items()) or '无'))
    print('  事件过滤      : {}'.format(s['event_filter_stats']))
    sb = s['scoreboard']
    print('  当前比分      : A={} B={} (局分 {}:{})'.format(
        sb['score']['A'], sb['score']['B'], sb['games']['A'], sb['games']['B']))
    print('  发球方        : {}'.format(sb['server']))
    if s['points']:
        print('  判分明细:')
        for p in s['points'][-10:]:
            print('    {:>7.3f}s  {} 得分  [{}]  比分 {}'.format(
                p['ts'], p['winner'], p['reason'], p['score']))
    if s['reviews']:
        print('  待人工确认    : {} 条'.format(len(s['reviews'])))
    print('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--weights', default=DEFAULT_WEIGHTS)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--dump-observations', default=None)
    parser.add_argument('--bounce-thresh', type=float, default=0.65)
    parser.add_argument('--net-thresh', type=float, default=0.5)
    parser.add_argument('--ball-thresh', type=float, default=0.05)
    parser.add_argument('--decode', default='argmax', choices=['argmax', 'centroid'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    run(args.video, args.weights, max_frames=args.max_frames,
        dump_path=args.dump_observations,
        bounce_thresh=args.bounce_thresh, net_thresh=args.net_thresh,
        ball_thresh=args.ball_thresh, decode=args.decode,
        device=args.device, verbose=not args.quiet)
