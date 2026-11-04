# -*- coding: utf-8 -*-
"""对完整视频跑裁判流程，并把实时比分 UI 烧进结果视频。

计分引擎可切换（``--engine``）
------------------------------
============================  ==================================================
``yolo``（**默认**）            ``referee.yolo_sm`` —— 移植自
                              https://github.com/V2tin19/pingpong-smart-referee
                              的共用计分状态机。6 状态序列 + 5 条 ITTF 判罚规则，
                              每条规则独立开关，判罚带规则名。
``legacy``                    ``referee.score_state_machine`` —— TTNet 原状态机
                              （v3 出口侧形态校验版），保留用于回归对比。
============================  ==================================================

两者吃的是**同一批 TTNet 事件**，因此同一段视频、同一份推理缓存，
切换引擎就能直接看出「同一批证据、两套规则」的判分差异。
这正是方案文档 §6「用同一段视频分别测试，可以直接比较最终判分准确率」。

与 run_referee.py 的区别
------------------------
run_referee.py 只做「离线分析」，输出文字统计；
本脚本做「在线可视化」，边推理边绘制比分 UI，最终产出可播放的 mp4。

关键实现细节
------------
1. **时间对齐**：TTNet 输入 9 帧、预测末帧，因此前 8 帧没有观测。
   为保证输出视频与原视频帧数一致、不丢开头，开头 8 帧沿用初始比分（0:0）绘制。
2. **事件滞后**：比分在事件发生的那一帧才更新，因此 UI 用「事件时间戳」判断
   当前帧应显示哪个比分，而不是用推理完成时刻。
3. **两遍式设计**：先完整推理一遍收集「帧号 → 比分/事件」，再渲染视频。
   这样避免在单遍流式推理中处理复杂的时间回溯，逻辑更可靠。
"""

import os
import sys
import json
import argparse
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from referee.ttnet_adapter import TTNetAdapter
from referee.event_processing import TrajectoryProcessor, EventExtractor
from referee.score_state_machine import TableTennisStateMachine
from referee.unified_types import EventType, Side
from referee.scoreboard_ui import ScoreboardUI, load_font
from referee.run_referee import build_configs, DEFAULT_WEIGHTS


ENGINES = ('yolo', 'legacy')
DEFAULT_ENGINE = 'yolo'

EVENT_TEXT = {
    EventType.BOUNCE: '落台',
    EventType.NET_HIT: '擦网',
    EventType.HIT: '击球',
}

# 两套状态机的阶段名 → 中文。已统一的键直接对齐；legacy 的私有阶段也在此登记，
# 免得 UI 上出现陌生的英文状态名。
PHASE_TEXT = {
    # yolo（6 状态，方案文档 §3）
    'WAITING_SERVE': '等待发球',
    'SERVE_IN_PROGRESS': '发球进行中',
    'RALLY': '对打中',
    'RALLY_END_CANDIDATE': '回合结束候选',
    'CONFIRM_POINT': '确认得分',
    'MANUAL_REVIEW': '待人工确认',
    # legacy
    'ROUND_ENDING': '回合结束',
    'COOLDOWN': '判分冷却',
    'NEEDS_REVIEW': '待人工确认',
}


def phase_text(phase: Optional[str]) -> Optional[str]:
    if not phase:
        return None
    return PHASE_TEXT.get(phase, phase)


def analyze(adapter, video_path, bounce_thresh, net_thresh, max_frames=None, verbose=True):
    """第一遍：推理并记录「逐帧观测 + 事件」。

    为了支持「一次推理、反复调判分规则」，这里只保存原始观测与事件，
    不在这一步跑状态机。判分由 build_score_timeline() 单独完成（毫秒级）。
    """
    tracker = TrajectoryProcessor()
    extractor = EventExtractor(bounce_thresh=bounce_thresh, net_thresh=net_thresh)

    frame_obs = []      # [(frame_id, ts, visible)]
    events = []         # [dict]
    detected = {'BOUNCE': 0, 'NET_HIT': 0, 'HIT': 0, 'BALL_LOST': 0}

    def _handle(evs):
        for ev in evs:
            detected[ev.type] = detected.get(ev.type, 0) + 1
            events.append(ev.to_dict())

    for obs in adapter.observations_from_video(video_path, max_frames=max_frames,
                                               progress_every=2000):
        traj = tracker.update(obs)
        _handle(extractor.process(obs, traj))
        frame_obs.append((obs.frame_id, round(obs.timestamp_s, 5),
                          1 if obs.ball.visible else 0))
    _handle(extractor.flush())

    if verbose:
        print('  推理完成：{} 帧，事件 {}'.format(len(frame_obs), detected))
    return {'frames': frame_obs, 'events': events, 'detected': detected,
            'low_conf_dropped': adapter.low_conf_dropped,
            'frozen_frames_dropped': adapter.frozen_frames_dropped}


# ---------------------------------------------------------------- 第二遍：判分
def _legacy_timeline(cache, verbose_machine=False, cumulative_mode=True,
                     auto_award_on_silence=True):
    """旧引擎：跑 TTNet 原状态机，产出与 yolo 引擎同形的四元组。"""
    from referee.unified_types import Event

    machine = TableTennisStateMachine(verbose=verbose_machine,
                                      cumulative_mode=cumulative_mode,
                                      auto_award_on_silence=auto_award_on_silence,
                                      auto_award_on_end=auto_award_on_silence)
    machine.set_visibility_source([(f[1], f[2]) for f in cache.get('frames', [])])
    evs = sorted([e for e in cache['events'] if e['type'] != EventType.BALL_LOST],
                 key=lambda e: (e['timestamp_s'], e['type']))

    snapshots: Dict[int, Dict[str, Any]] = {}
    event_timeline: List[Tuple[float, str]] = []
    decision_timeline: List[Tuple[float, str]] = []
    decided = 0
    ev_idx = 0

    def _drain():
        nonlocal decided
        for p in machine.points[decided:]:
            decision_timeline.append((
                p['ts'],
                '{} 得分 [{}]  {}'.format(
                    'P1(A)' if p['winner'] == 'A' else 'P2(B)',
                    p['reason'][:22], p['score'])))
        decided = len(machine.points)

    for frame_id, ts, _vis in cache.get('frames', []):
        while ev_idx < len(evs) and evs[ev_idx]['timestamp_s'] <= ts:
            d = evs[ev_idx]
            ev = Event(event_id=d['event_id'], timestamp_s=d['timestamp_s'],
                       type=d['type'], side=d['side'],
                       position_px=d.get('position_px'),
                       confidence=d.get('confidence'),
                       sources=d.get('sources', []))
            machine.consume(ev)
            _drain()
            if ev.type in (EventType.BOUNCE, EventType.NET_HIT):
                suffix = ' · {} 侧'.format(ev.side) if ev.side in (Side.A, Side.B) else ''
                event_timeline.append(
                    (ev.timestamp_s, EVENT_TEXT.get(ev.type, ev.type) + suffix))
            ev_idx += 1
        machine.tick_deep(ts)
        _drain()
        sb = machine.scoreboard
        snapshots[frame_id] = {
            'score_a': sb['score']['A'], 'score_b': sb['score']['B'],
            'games_a': sb['games']['A'], 'games_b': sb['games']['B'],
            'server': sb['server'], 'phase': sb['phase'],
        }

    if cache.get('frames'):
        last_fid, last_ts = cache['frames'][-1][0], cache['frames'][-1][1]
        if machine.finalize(last_ts):
            _drain()
            sb = machine.scoreboard
            snapshots[last_fid] = {
                'score_a': sb['score']['A'], 'score_b': sb['score']['B'],
                'games_a': sb['games']['A'], 'games_b': sb['games']['B'],
                'server': sb['server'], 'phase': sb['phase'],
            }

    if machine.points:
        last = machine.points[-1]['score']
        print('  判分完成：{} 分（护栏拦下 {} 次）末局分 {}'.format(
            len(machine.points), machine.suppressed_count, last))
    return snapshots, event_timeline, decision_timeline, machine


def _yolo_timeline(cache, rules=None, verbose_machine=False, profile='ttnet'):
    """新引擎：跑共用状态机，产出与 legacy 引擎同形的四元组。

    :param profile: ``'ttnet'``（默认）启用 :mod:`referee.yolo_sm.ttnet_state_machine`
        的 TTNet 专属修正；``'yolo'`` 则与原 YOLO 分支逐条一致。
    """
    from referee.yolo_sm.bridge import YOLO_EVENT_TEXT, YoloCoreScorer, to_referee_events
    from referee.yolo_sm.state_machine import DecisionKind, MatchRules

    scorer = YoloCoreScorer(rules=rules, first_server='A',
                            verbose=verbose_machine, profile=profile)
    # BALL_LOST 不送：YOLO 状态机没有这个概念，回合结束由 tick() 的时间判据负责。
    events = to_referee_events(cache['events'])
    frames = cache.get('frames') or []

    snapshots: Dict[int, Dict[str, Any]] = {}
    event_timeline: List[Tuple[float, str]] = []
    decision_timeline: List[Tuple[float, str]] = []
    ev_idx = 0

    for frame_id, ts, _vis in frames:
        while ev_idx < len(events) and events[ev_idx].timestamp_s <= ts:
            ev = events[ev_idx]
            for d in scorer.update(ev):
                decision_timeline.append((d.timestamp_s, _decision_text(d, DecisionKind)))
            if ev.type in ('BOUNCE', 'NET_TOUCH', 'RACKET_HIT'):
                suffix = ' · {} 侧'.format(ev.side) if ev.side in ('A', 'B') else ''
                event_timeline.append(
                    (ev.timestamp_s, YOLO_EVENT_TEXT.get(ev.type, ev.type) + suffix))
            ev_idx += 1
        for d in scorer.tick(ts):
            decision_timeline.append((d.timestamp_s, _decision_text(d, DecisionKind)))
        snapshots[frame_id] = scorer.scoreboard()

    # 末尾：把残留回合结算掉；结果只写最后一帧，绝不向前倒灌。
    #
    # 必须走 scorer.finalize() 而不是「一次 tick(+1s)」：状态机的 tick() 每次只推进
    # 一步（RALLY → RALLY_END_CANDIDATE → 裁决），一次性跨到确认时刻只会走完第一步，
    # 末尾那一分就静默丢了。实测 test_2 因此从 A1:B1 掉成 A0:B1。
    if frames:
        last_fid, last_ts = frames[-1][0], frames[-1][1]
        for d in scorer.finalize(last_ts):
            # 结算发生在「视频结束之后」，其时间戳（如 31.21s）会超过末帧（29.99s）。
            # 比分本身写在末帧（见下面 snapshots[last_fid]），若提示条沿用
            # 31.21s，渲染时永远走不到那一刻 —— 观众看到比分跳了却看不到原因。
            # 因此把提示条时间**夹到末帧**：它和比分变化是同一时刻、同一件事。
            #
            # 必须留一点点余量（-1e-3）：末帧的时间在渲染时是按 written/fps 重算的
            # （3599/120 = 29.9916666…），而缓存里存的是四舍五入后的 29.99167。
            # 直接取等会差 3e-6，导致 `timeline[0] <= ts` 判 False，提示条整条不显示。
            shown_ts = min(d.timestamp_s, last_ts - 1e-3)
            decision_timeline.append((shown_ts, _decision_text(d, DecisionKind)))
        snapshots[last_fid] = scorer.scoreboard()

    pts = scorer.points
    dead = (scorer.rule_counts.get('NO_EVIDENCE', 0)
            + scorer.rule_counts.get('SERVE_INCOMPLETE', 0))
    print('  判分完成：{} 分（A{} : B{}，共 {} 条判罚，其中死球 {} 次）'.format(
        pts[0] + pts[1], pts[0], pts[1], len(scorer.all_decisions), dead))
    return snapshots, event_timeline, decision_timeline, scorer


def _decision_text(d, DecisionKind) -> str:
    """把 YOLO 状态机的 Decision 转成 UI 文案。"""
    if d.kind == DecisionKind.POINT:
        who = 'P1(A)' if d.winner == 'A' else 'P2(B)'
        return '{} 得分 [{}]  {}:{}'.format(who, d.rule, d.score[0], d.score[1])
    if d.kind == DecisionKind.LET:
        return '擦网重发 [{}]  比分不变'.format(d.rule)
    if d.kind == DecisionKind.GAME_OVER:
        return '本局结束 {}:{}'.format(d.score[0], d.score[1])
    if d.kind == DecisionKind.REVIEW:
        return '待人工确认 [{}]'.format(d.rule)
    return '死球 [{}]  不判分'.format(d.rule)


def build_score_timeline(cache, verbose=True, cumulative_mode=True,
                         auto_award_on_silence=True, verbose_machine=False,
                         engine=DEFAULT_ENGINE, rules=None, profile='ttnet'):
    """第二遍（毫秒级）：在缓存的事件上跑状态机，产出「帧号 → 比分快照」。

    与推理完全解耦，因此可以随意调状态机参数并立即看到效果。

    :param engine: ``'yolo'``（默认，YOLO 共用状态机）或 ``'legacy'``（TTNet 原状态机）。
    :param cumulative_mode: 仅 legacy 引擎使用。测试片只有几十秒到几分钟，
        不存在「打完一局」，按 11 分制会在 11 分处归零。yolo 引擎用跨局累计
        账本达到同样效果，无需该开关。
    :param auto_award_on_silence: 仅 legacy 引擎使用 —— 落台后长时间无后续是否判分。
        yolo 引擎由 ``MatchRules.rally_end_s`` + ``NO_RETURN`` 规则承担同一职责。
    :param verbose_machine: 打印状态机内部决策，用于排查某个落台为何未计分。
    :param rules: 仅 yolo 引擎使用，见 ``referee.yolo_sm.state_machine.MatchRules``。

    :return: (snapshots, event_timeline, decision_timeline, machine)
    """
    if engine not in ENGINES:
        raise ValueError('engine must be one of {}, got {!r}'.format(ENGINES, engine))

    if engine == 'yolo':
        return _yolo_timeline(cache, rules=rules, verbose_machine=verbose_machine,
                              profile=profile)
    return _legacy_timeline(cache, verbose_machine=verbose_machine,
                            cumulative_mode=cumulative_mode,
                            auto_award_on_silence=auto_award_on_silence)


def render(video_path, out_path, snapshots, event_timeline, decision_timeline, fps,
           ui=None, fade_s=1.6, decision_fade_s=2.6, progress_every=2000,
           max_frames=None, cumulative=True, show_phase=True):
    """第三遍：逐帧回放原视频，叠加对应时刻的比分 UI。

    为保证「比分与画面同步」，用帧号查快照；事件提示用时间差淡出。

    注意：快照的键与视频帧号并不完全对齐 —— 适配器前 8 帧没有观测（9 帧窗口
    需要预热），因此快照从 frame_id=8 开始。这里统一按「帧号」查询，
    并在找不到时向前回溯最近一个快照，避免开头几帧与末尾帧丢掉比分。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('无法打开视频: {}'.format(video_path))

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        n_total = min(n_total, max_frames)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ui = ui or ScoreboardUI()

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # 初始状态（前 8 帧还没有观测）
    cur = {'score_a': 0, 'score_b': 0, 'games_a': 0, 'games_b': 0,
           'server': 'A', 'phase': None}
    written = 0
    ev_idx = 0
    dec_idx = 0
    last_ev = None
    last_dec = None

    # 按帧号排序的快照键，用于「找不到精确帧号时取最近一帧」
    snap_keys = sorted(snapshots.keys())
    snap_pos = 0

    while True:
        if max_frames is not None and written >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        ts = written / float(fps)

        # 推进快照指针：取所有 frame_id <= written 的最后一个快照
        while snap_pos < len(snap_keys) and snap_keys[snap_pos] <= written:
            cur = snapshots[snap_keys[snap_pos]]
            snap_pos += 1

        # 推进事件指针（timeline 按时间递增）
        while ev_idx < len(event_timeline) and event_timeline[ev_idx][0] <= ts:
            last_ev = event_timeline[ev_idx]
            ev_idx += 1
        while dec_idx < len(decision_timeline) and decision_timeline[dec_idx][0] <= ts:
            last_dec = decision_timeline[dec_idx]
            dec_idx += 1

        ev_age = (ts - last_ev[0]) if last_ev else None
        dec_age = (ts - last_dec[0]) if last_dec else None

        # 判罚优先于普通事件：比分变化比「又落了一次台」重要。
        if dec_age is not None and dec_age < decision_fade_s:
            text, age, kind = last_dec[1], dec_age, 'point'
        elif ev_age is not None and ev_age < fade_s:
            text, age, kind = last_ev[1], ev_age, 'event'
        else:
            text, age, kind = None, None, 'event'

        ui.draw(frame, score_a=cur['score_a'], score_b=cur['score_b'],
                games_a=cur['games_a'], games_b=cur['games_b'],
                server=cur['server'], event_text=text, event_age=age,
                cumulative=cumulative,
                phase=phase_text(cur.get('phase')) if show_phase else None,
                event_kind=kind)
        writer.write(frame)
        written += 1
        if progress_every and written % progress_every == 0:
            print('  渲染 {}/{} 帧 ({:.0%})'.format(written, n_total, written / max(n_total, 1)))

    cap.release()
    writer.release()
    return written


def run(video_path, out_path, weights=DEFAULT_WEIGHTS,
        bounce_thresh=0.65, net_thresh=0.65, max_frames=None, device='cuda:0',
        cache_path=None, auto_award_on_silence=True, engine=DEFAULT_ENGINE,
        rules=None, verbose_machine=False, show_phase=True, decision_fade_s=2.6,
        profile='ttnet'):
    t0 = time.time()
    print('=' * 70)
    print('TTNet 裁判 · 带比分 UI 视频渲染')
    print('  计分引擎: {}'.format(
        'yolo（pingpong-smart-referee 共用状态机）' if engine == 'yolo'
        else 'legacy（TTNet 原 v3 状态机）'))
    print('  输入 : {}'.format(video_path))
    print('  输出 : {}'.format(out_path))
    print('=' * 70)

    # ---- 第一遍：推理（可复用缓存）
    if cache_path and os.path.isfile(cache_path):
        print('[1/4] 载入推理缓存…')
        with open(cache_path, encoding='utf-8') as f:
            cache = json.load(f)
    else:
        configs = build_configs()
        adapter = TTNetAdapter(configs, weights, device=device)
        print('[1/4] 推理分析…')
        cache = analyze(adapter, video_path, bounce_thresh, net_thresh, max_frames)
        if cache_path:
            d = os.path.dirname(os.path.abspath(cache_path))
            if d:
                os.makedirs(d, exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False)

    # ---- 第二遍：判分（毫秒级，可反复调参）
    print('[2/4] 状态机判分…')
    snapshots, ev_tl, dec_tl, machine = build_score_timeline(
        cache, auto_award_on_silence=auto_award_on_silence, engine=engine,
        rules=rules, verbose_machine=verbose_machine, profile=profile)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # ---- 第三遍：渲染
    print('[3/4] 渲染视频 + 比分 UI…')
    cumulative = engine == 'legacy'
    written = render(video_path, out_path, snapshots, ev_tl, dec_tl, fps,
                     max_frames=max_frames, cumulative=cumulative,
                     show_phase=show_phase, decision_fade_s=decision_fade_s)

    print('[4/4] 完成')
    size = os.path.getsize(out_path) / 1024 / 1024 if os.path.isfile(out_path) else 0
    print('  耗时 {:.1f}s | 输出 {} 帧 | 文件 {:.1f} MB'.format(
        time.time() - t0, written, size))

    summary = {
        'engine': engine,
        'profile': profile,
        'video': video_path,
        'output': out_path,
        'frames': written,
        'fps': fps,
        'detected_events': cache['detected'],
        'auto_award_on_silence': auto_award_on_silence,
        'low_conf_dropped': cache.get('low_conf_dropped'),
        'frozen_dropped': cache.get('frozen_frames_dropped'),
    }
    if engine == 'yolo':
        pts = machine.points
        summary.update({
            'final_score': {'A': pts[0], 'B': pts[1]},
            'games_won': dict(machine.games),
            'server': machine.server,
            'rule_counts': dict(machine.rule_counts),
            'points': [{'ts': round(d.timestamp_s, 3), 'winner': d.winner,
                        'rule': d.rule, 'score': list(d.score)}
                       for d in machine.all_decisions
                       if d.kind.value == 'POINT'],
            'decisions': [{'ts': round(d.timestamp_s, 3), 'kind': d.kind.value,
                           'winner': d.winner, 'rule': d.rule,
                           'reason': d.reason, 'score': list(d.score),
                           'state': d.state}
                          for d in machine.all_decisions],
        })
    else:
        summary.update({
            'final_score': machine.scoreboard['score'],
            'points': machine.points,
            'suppressed_by_guard': machine.suppressed_count,
            'legal_serves': machine.serve_count,
            'rejected_bounces': machine.rejected,
        })
    return summary


def _build_rules(a):
    """从命令行构造 yolo 引擎的 MatchRules（默认即 YOLO 项目的原始值）。"""
    from referee.yolo_sm.state_machine import MatchRules

    kw = {}
    if a.rally_end_s is not None:
        kw['rally_end_s'] = a.rally_end_s
    if a.confirm_s is not None:
        kw['confirm_s'] = a.confirm_s
    if a.serve_min_gap_s is not None:
        kw['serve_min_gap_s'] = a.serve_min_gap_s
    if a.points_to_win is not None:
        kw['points_to_win'] = a.points_to_win
    for name in ('enable_serve_fault', 'enable_double_bounce', 'enable_own_court',
                 'enable_out', 'enable_no_return'):
        v = getattr(a, name, None)
        if v is not None:
            kw[name] = v
    return MatchRules(**kw)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--bounce-thresh', type=float, default=0.65)
    ap.add_argument('--net-thresh', type=float, default=0.65)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cache', default=None,
                    help='推理缓存路径；存在则直接复用，避免重复推理')
    ap.add_argument('--profile', choices=('ttnet', 'yolo'), default='ttnet',
                    help='证据质量档位：ttnet（默认，启用 TTNet 专属修正）/ yolo（与 YOLO 分支一致）')
    ap.add_argument('--engine', choices=ENGINES, default=DEFAULT_ENGINE,
                    help='计分引擎：yolo（默认，YOLO 共用状态机）/ legacy（TTNet 原状态机）')
    ap.add_argument('--no-silence-award', action='store_true',
                    help='[legacy] 关闭「落台后长期无后续」的判分')
    ap.add_argument('--no-phase-badge', action='store_true',
                    help='不在 UI 上显示状态机阶段徽标')
    ap.add_argument('--decision-fade', type=float, default=2.6,
                    help='判罚提示条停留秒数')
    ap.add_argument('--verbose-machine', action='store_true',
                    help='打印状态机内部决策（排查某分为何未判）')
    ap.add_argument('--summary', default=None, help='把汇总 JSON 写到该路径')

    ap.add_argument('--rally-end-s', type=float, default=None,
                    help='[yolo] 无事件多久算回合可能结束（默认 0.8）')
    ap.add_argument('--confirm-s', type=float, default=None,
                    help='[yolo] 回合结束候选的消抖窗口（默认 0.4）')
    ap.add_argument('--serve-min-gap-s', type=float, default=None,
                    help='[yolo] 死球停顿多久后的第一次击球才算发球（默认 1.5）')
    ap.add_argument('--points-to-win', type=int, default=None,
                    help='[yolo] 每局目标分（默认 11）')
    for name in ('enable_serve_fault', 'enable_double_bounce', 'enable_own_court',
                 'enable_out', 'enable_no_return'):
        ap.add_argument('--{}'.format(name.replace('_', '-')), dest=name,
                        type=lambda s: s.lower() not in ('0', 'false', 'no'),
                        default=None,
                        help='[yolo] 规则开关：{}'.format(name))
    a = ap.parse_args()

    rules = _build_rules(a) if (a.engine == 'yolo' and a.profile == 'ttnet') else None
    s = run(a.video, a.out, bounce_thresh=a.bounce_thresh,
            net_thresh=a.net_thresh, max_frames=a.max_frames,
            device=a.device, cache_path=a.cache,
            auto_award_on_silence=not a.no_silence_award,
            engine=a.engine, rules=rules,
            verbose_machine=a.verbose_machine,
            show_phase=not a.no_phase_badge,
            decision_fade_s=a.decision_fade, profile=a.profile)
    print(json.dumps({k: v for k, v in s.items() if k != 'decisions'},
                     ensure_ascii=False, indent=1))
    if a.summary:
        d = os.path.dirname(os.path.abspath(a.summary))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.summary, 'w', encoding='utf-8') as f:
            json.dump(s, f, ensure_ascii=False, indent=1)
        print('汇总已写入: {}'.format(a.summary))
