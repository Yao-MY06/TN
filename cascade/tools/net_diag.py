"""生成网附近下网分的逐点诊断卡、轨迹/分数图与汇总。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
TN_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if TN_ROOT not in sys.path:
    sys.path.insert(0, TN_ROOT)

from cascade._paths import PRESET_TO_QUAD, load_detect_bounce  # noqa: E402
from cascade.calib import build_calib_from_quad  # noqa: E402
from cascade.fuse_obs import FuseParams, fuse  # noqa: E402
from cascade.ttnet_reviewer import DEFAULT_K  # noqa: E402
from core.geometry import CALIBRATION_PRESETS  # noqa: E402


def _read_json(path: str) -> Any:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve(base: str, value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value if os.path.isabs(value) else os.path.normpath(os.path.join(base, value))


def _calib_for(entry: Dict[str, Any]):
    preset = entry.get('preset') or entry['name']
    quad = entry.get('quad') or PRESET_TO_QUAD.get(preset, preset)
    if preset in CALIBRATION_PRESETS:
        import run_mixed as rm
        return rm.build_calib(preset)
    if preset == 'game4':
        import run_mixed as rm
        return rm.build_calib(preset)
    return build_calib_from_quad(quad)


def _net_endings(entry: Dict[str, Any], manifest_dir: str) -> List[Dict[str, Any]]:
    game_json = _resolve(manifest_dir, entry['extended'])
    raw = {int(k): v for k, v in _read_json(game_json).items()}
    out = []
    for frame, label in sorted(raw.items()):
        head = label.split()[0]
        if not (head.endswith('_net') or head in ('left_net', 'right_net')):
            continue
        loser_word = head.split('_', 1)[0]
        loser = 'A' if loser_word == 'left' else 'B'
        out.append({'frame': frame, 'timestamp_s': frame / float(entry.get('fps', 120.0)),
                    'label': label, 'loser': loser,
                    'winner': 'B' if loser == 'A' else 'A'})
    return out


def _eval_rows(path: str) -> Dict[Tuple[float, str], Dict[str, Any]]:
    if not os.path.isfile(path):
        return {}
    data = _read_json(path)
    return {(round(float(r['gt_t']), 3), r['gt']): r for r in data['points']['rows']}


def _visible(obs: Optional[Dict[str, Any]]) -> bool:
    return bool(obs and obs.get('ball', {}).get('visible'))


def _predicted(obs: Optional[Dict[str, Any]]) -> bool:
    return bool(obs and obs.get('ball', {}).get('predicted'))


def _xy(obs: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    if not _visible(obs):
        return None
    b = obs['ball']
    c = b.get('center_px')
    if c is None:
        return None
    return float(c[0]), float(c[1])


def _trace_rows(entry: Dict[str, Any], manifest_dir: str, calib, t0: float,
                t1: float, fps: float) -> List[Dict[str, Any]]:
    tn = _read_jsonl(_resolve(manifest_dir, entry['tracknet']))
    yolo_path = _resolve(manifest_dir, entry.get('yolo'))
    yolo = _read_jsonl(yolo_path) if yolo_path else []
    tn_by = {int(o['frame_id']): o for o in tn}
    yolo_by = {int(o['frame_id']): o for o in yolo}
    fused_by: Dict[int, Dict[str, Any]] = {}
    if yolo:
        fused, _ = fuse(yolo, tn, FuseParams())
        fused_by = {int(o['frame_id']): o for o in fused}
    else:
        fused_by = tn_by
    f0, f1 = int(np.floor(t0 * fps)), int(np.ceil(t1 * fps))
    rows = []
    for fid in range(max(0, f0), f1 + 1):
        ts = fid / fps
        tn_o, y_o = tn_by.get(fid), yolo_by.get(fid)
        fo = fused_by.get(fid)
        size = (fo or tn_o or y_o or {}).get('image_size') or [1920, 1080]
        c = calib.scaled_to(int(size[0]), int(size[1]))
        xy = _xy(fo)
        rows.append({
            'frame_id': fid, 'timestamp_s': round(ts, 6),
            'x_rel': None if xy is None else round(xy[0] - c.net_x, 2),
            'y': None if xy is None else round(xy[1], 2),
            'source': None if fo is None else fo.get('source'),
            'tracknet_visible': _visible(tn_o), 'tracknet_predicted': _predicted(tn_o),
            'yolo_visible': _visible(y_o), 'yolo_predicted': _predicted(y_o),
        })
    return rows


def _ttnet_curve(entry: Dict[str, Any], manifest_dir: str, t0: float, t1: float,
                 fps: float, k: int) -> List[Dict[str, Any]]:
    path = _resolve(manifest_dir, entry.get('ttnet_cache'))
    if not path or not os.path.isfile(path):
        return []
    rows = []
    for r in _read_jsonl(path):
        if 'window_end' not in r:
            continue
        f = int(r['window_end']) - k
        ts = f / fps
        if t0 <= ts <= t1:
            rows.append({'frame_id': f, 'timestamp_s': round(ts, 6),
                         'bounce': float(r.get('bounce') or 0.0),
                         'net': float(r.get('net') or 0.0)})
    return sorted(rows, key=lambda x: x['frame_id'])


def _score_peaks(curve: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not curve:
        return {'max_bounce': None, 'bounce_frame': None, 'max_net': None, 'net_frame': None}
    b = max(curve, key=lambda r: r['bounce'])
    n = max(curve, key=lambda r: r['net'])
    return {'max_bounce': round(b['bounce'], 6), 'bounce_frame': b['frame_id'],
            'max_net': round(n['net'], 6), 'net_frame': n['frame_id']}


def _overlap_clips(clips: List[Dict[str, Any]], frame: int,
                   pad_frames: int = 0) -> List[Dict[str, Any]]:
    return [c for c in clips if c.get('start_frame_id') is not None
            and c.get('end_frame_id') is not None
            and c['start_frame_id'] - pad_frames <= frame <= c['end_frame_id'] + pad_frames]


def _group_context(run_dir: str, frame: int) -> Dict[str, Any]:
    props = _read_jsonl(os.path.join(run_dir, 'proposals.jsonl'))
    clips = _read_jsonl(os.path.join(run_dir, 'clips.jsonl'))
    reviews = _read_jsonl(os.path.join(run_dir, 'reviews.jsonl'))
    events = _read_jsonl(os.path.join(run_dir, 'events.jsonl'))
    decisions = _read_jsonl(os.path.join(run_dir, 'decisions.jsonl'))
    pmap = {p['proposal_id']: p for p in props}
    rmap = {r['clip_id']: r for r in reviews}
    hits = _overlap_clips(clips, frame)
    if not hits:
        hits = _overlap_clips(clips, frame, pad_frames=12)
    used_props = []
    for c in hits:
        for pid in c.get('proposal_ids', []):
            if pid in pmap:
                used_props.append(pmap[pid])
    return {'clips': hits, 'proposals': used_props, 'reviews': [rmap[c['clip_id']] for c in hits
                                                                       if c['clip_id'] in rmap],
            'events': events, 'decisions': decisions}


def _event_near(events: List[Dict[str, Any]], etype: str, t0: float, t1: float) -> List[Dict[str, Any]]:
    return [e for e in events if e.get('type') == etype and t0 <= float(e['timestamp_s']) <= t1]


def _initial_category(card: Dict[str, Any], rule_ctx: Dict[str, Any], tt_ctx: Dict[str, Any]) -> Tuple[str, str]:
    gt = card['gt']
    t = gt['timestamp_s']
    if card['rule'].get('correct') or card['ttnet'].get('correct'):
        chain = []
        if card['rule'].get('correct'):
            chain.append('rule')
        if card['ttnet'].get('correct'):
            chain.append('ttnet')
        return 'E', '已判对；有效链路：' + '+'.join(chain)

    events = rule_ctx['events'] + tt_ctx['events']
    net_x = card['net_x']
    zone = card['net_zone']
    hits = [e for e in events if e.get('type') == 'RACKET_HIT'
            and t - 0.8 <= float(e['timestamp_s']) <= t + 0.8]
    near_hits = [e for e in hits if e.get('position_px')
                 and abs(float(e['position_px'][0]) - net_x) <= 30
                 and zone[0] <= float(e['position_px'][1]) <= zone[1]]
    if any(e.get('side') != gt['loser'] for e in near_hits):
        return 'A', '网带内出现与真实丢分方相反的 RACKET_HIT：' + ','.join(
            f"{e['timestamp_s']:.3f}s/{e['side']}" for e in near_hits)
    if _event_near(events, 'NET_TOUCH', t - 0.6, t + 1.0):
        return 'C', '窗口内有 NET_TOUCH，但当前链路仍未判对'
    if any(r.get('reason') == 'NO_EVIDENCE' for r in card['nearby_rules']):
        return 'B', '最终为 NO_EVIDENCE，网附近轨迹/证据不足'
    return 'E', '未命中 A–D 的明确自动规则，需要人工看图/轨迹确认'


def _nearest_net_time(net_points: List[Dict[str, Any]], t: float) -> Optional[float]:
    if not net_points:
        return None
    return min((abs(p['timestamp_s'] - t), p['timestamp_s']) for p in net_points)[1]


def _fmt(v, digits=3):
    if v is None:
        return '—'
    if isinstance(v, float):
        return f'{v:.{digits}f}'
    return str(v)


def _draw_card(path: str, card: Dict[str, Any]) -> None:
    W, H = 1400, 1020
    canvas = np.full((H, W, 3), 255, np.uint8)
    x0, x1 = 90, W - 30
    t0, t1 = card['window'][0], card['window'][1]
    panels = [(70, 320), (390, 650), (720, 940)]
    bounce_col = card['score_curve']
    trace = card['trace']
    colors = {'tracknet': (20, 120, 20), 'yolo': (200, 110, 20), 'fused': (30, 80, 220)}

    def tx(t):
        return int(x0 + (float(t) - t0) / max(t1 - t0, 1e-9) * (x1 - x0))

    def panel(p, y, ymin, ymax):
        a, b = p
        yy = int(b - (float(y) - ymin) / max(ymax - ymin, 1e-9) * (b - a))
        return max(a, min(b, yy))

    for idx, (a, b) in enumerate(panels):
        cv2.rectangle(canvas, (x0, a), (x1, b), (210, 210, 210), 1)
        for sec in range(int(np.ceil(t0)), int(np.floor(t1)) + 1):
            xx = tx(sec)
            cv2.line(canvas, (xx, a), (xx, b), (235, 235, 235), 1)
            cv2.putText(canvas, f'{sec}s', (xx - 13, b + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'ball x - net_x (px)', (12, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'ball y (px)', (12, 365), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'TTNet bounce / net score', (12, 695), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (20, 20, 20), 1, cv2.LINE_AA)

    # net bands / GT line
    cv2.rectangle(canvas, (x0, 70), (x1, 320), (245, 255, 245), -1)
    cv2.rectangle(canvas, (x0, 70), (x1, 320), (210, 210, 210), 1)
    cv2.line(canvas, (x0, panel(panels[0], 0, -600, 600)), (x1, panel(panels[0], 0, -600, 600)),
             (80, 80, 80), 1)
    cv2.rectangle(canvas, (x0, panel(panels[0], 30, -600, 600)),
                  (x1, panel(panels[0], -30, -600, 600)), (220, 240, 220), -1)
    c = card['calib_scaled']
    cv2.rectangle(canvas, (x0, panel(panels[1], int(c.net_zone[0]), 300, 900)),
                  (x1, panel(panels[1], int(c.net_zone[1]), 300, 900)), (220, 240, 220), -1)
    gx = tx(card['gt']['timestamp_s'])
    for a, b in panels:
        cv2.line(canvas, (gx, a), (gx, b), (0, 0, 220), 2)
    cv2.putText(canvas, 'GT', (gx + 3, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)

    # traces
    for src, color in (('tracknet', colors['tracknet']), ('yolo', colors['yolo'])):
        pts = []
        for r in trace:
            if r['source'] == src and r['x_rel'] is not None:
                pts.append((r['timestamp_s'], r['x_rel'], r['y'], r['frame_id']))
            else:
                if len(pts) > 1:
                    for j in range(1, len(pts)):
                        cv2.line(canvas, (tx(pts[j-1][0]), panel(panels[0], pts[j-1][1], -600, 600)),
                                 (tx(pts[j][0]), panel(panels[0], pts[j][1], -600, 600)), color, 2)
                        cv2.line(canvas, (tx(pts[j-1][0]), panel(panels[1], pts[j-1][2], 300, 900)),
                                 (tx(pts[j][0]), panel(panels[1], pts[j][2], 300, 900)), color, 2)
                pts = []
        if len(pts) > 1:
            for j in range(1, len(pts)):
                cv2.line(canvas, (tx(pts[j-1][0]), panel(panels[0], pts[j-1][1], -600, 600)),
                         (tx(pts[j][0]), panel(panels[0], pts[j][1], -600, 600)), color, 2)
                cv2.line(canvas, (tx(pts[j-1][0]), panel(panels[1], pts[j-1][2], 300, 900)),
                         (tx(pts[j][0]), panel(panels[1], pts[j][2], 300, 900)), color, 2)

    # events
    ev_colors = {'RACKET_HIT': (180, 80, 0), 'BOUNCE': (180, 0, 180),
                 'NET_TOUCH': (0, 140, 210), 'OUT': (0, 0, 180)}
    labels = []
    for e in card['events']:
        xx = tx(float(e['timestamp_s']))
        col = ev_colors.get(e.get('type'), (100, 100, 100))
        cv2.line(canvas, (xx, 70), (xx, 940), col, 1)
        labels.append((xx, e.get('type', '?'), col))
    for i, (xx, lab, col) in enumerate(labels[:12]):
        cv2.putText(canvas, lab[:3], (xx + 2, 105 + 13 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, col, 1, cv2.LINE_AA)

    # scores
    cv2.line(canvas, (x0, panel(panels[2], 0.65, 0, 1)), (x1, panel(panels[2], 0.65, 0, 1)),
             (200, 200, 200), 1)
    cv2.line(canvas, (x0, panel(panels[2], 0.50, 0, 1)), (x1, panel(panels[2], 0.50, 0, 1)),
             (220, 220, 220), 1)
    for key, color in (('bounce', colors['tracknet']), ('net', (0, 150, 230))):
        pts = [(r['timestamp_s'], r[key]) for r in bounce_col]
        for j in range(1, len(pts)):
            cv2.line(canvas, (tx(pts[j-1][0]), panel(panels[2], pts[j-1][1], 0, 1)),
                     (tx(pts[j][0]), panel(panels[2], pts[j][1], 0, 1)), color, 2)

    cv2.putText(canvas, 'green=tracknet orange=yolo  event colors: HIT/BOU/TOU/OUT', (90, 975),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{card['game']}  {card['gt']['timestamp_s']:.3f}s  {card['gt']['label']}  "
                       f"category {card['category']}", (90, 1000), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (20, 20, 20), 1, cv2.LINE_AA)
    ok, enc = cv2.imencode('.png', canvas)
    if not ok:
        raise RuntimeError(f'cannot encode image: {path}')
    enc.tofile(path)


def _trace_table(rows: List[Dict[str, Any]]) -> str:
    out = ['<details><summary>逐帧轨迹（每帧）</summary>', '',
           '| frame | t | x-net_x | y | source | TN vis/pred | YOLO vis/pred |',
           '|---:|---:|---:|---:|---|---|---|']
    for r in rows:
        out.append(f"| {r['frame_id']} | {r['timestamp_s']:.3f} | {_fmt(r['x_rel'], 1)} | "
                   f"{_fmt(r['y'], 1)} | {r['source'] or '—'} | "
                   f"{'Y' if r['tracknet_visible'] else 'N'}/{'Y' if r['tracknet_predicted'] else 'N'} | "
                   f"{'Y' if r['yolo_visible'] else 'N'}/{'Y' if r['yolo_predicted'] else 'N'} |")
    out.extend(['', '</details>', ''])
    return '\n'.join(out)


def _event_table(events: List[Dict[str, Any]]) -> str:
    if not events:
        return '_无事件_\n'
    out = ['| t | type | side | position_px | sources |', '|---:|---|---|---|---|']
    for e in events:
        out.append(f"| {float(e['timestamp_s']):.3f} | {e.get('type')} | {e.get('side')} | "
                   f"{e.get('position_px') or '—'} | {', '.join(e.get('sources') or []) or '—'} |")
    return '\n'.join(out) + '\n'


def _coverage_text(ctx: Dict[str, Any]) -> str:
    if not ctx['clips']:
        return '无覆盖片段'
    parts = []
    for c, rev in zip(ctx['clips'], ctx['reviews']):
        reasons = []
        for p in ctx['proposals']:
            if p['proposal_id'] in c.get('proposal_ids', []):
                reasons.extend(p.get('reasons') or [])
        parts.append(f"{c['clip_id']} [{c.get('start_frame_id')},{c.get('end_frame_id')}] "
                     f"原因={','.join(sorted(set(reasons))) or '—'} "
                     f"复核={rev.get('status')}/{rev.get('reason')}")
    return '；'.join(parts)


def build_cards(manifest_path: str, batch_dir: str, out_dir: str, k: int = DEFAULT_K) -> Dict[str, Any]:
    manifest_path = os.path.abspath(manifest_path)
    manifest_dir = os.path.dirname(manifest_path)
    manifest = _read_json(manifest_path)
    entries = manifest.get('videos', [])
    os.makedirs(os.path.join(out_dir, 'img'), exist_ok=True)
    cards = []
    let_rows = []
    for entry in entries:
        name = entry['name']
        calib = _calib_for(entry)
        rule_dir = os.path.join(batch_dir, f'{name}_rule')
        tt_dir = os.path.join(batch_dir, f'{name}_ttnet')
        rule_eval = _eval_rows(os.path.join(rule_dir, 'eval_extended.json'))
        tt_eval = _eval_rows(os.path.join(tt_dir, 'eval_extended.json'))
        net_points = _net_endings(entry, manifest_dir)
        for gt in net_points:
            key = (round(gt['timestamp_s'], 3), gt['label'])
            row_rule = rule_eval.get(key, {'gt_t': gt['timestamp_s'], 'gt': gt['label'],
                                           'gt_winner': gt['winner'], 'our_t': None,
                                           'correct': False, 'note': '未找到 rule 行'})
            row_tt = tt_eval.get(key, {'gt_t': gt['timestamp_s'], 'gt': gt['label'],
                                       'gt_winner': gt['winner'], 'our_t': None,
                                       'correct': False, 'note': '未找到 ttnet 行'})
            fps = float(entry.get('fps', 120.0))
            w0, w1 = gt['timestamp_s'] - 1.5, gt['timestamp_s'] + 1.0
            trace = _trace_rows(entry, manifest_dir, calib, w0, w1, fps)
            curve = _ttnet_curve(entry, manifest_dir, w0, w1, fps, k)
            rule_ctx = _group_context(rule_dir, gt['frame'])
            tt_ctx = _group_context(tt_dir, gt['frame'])
            events = sorted([e for e in (rule_ctx['events'] + tt_ctx['events'])
                             if w0 <= float(e['timestamp_s']) <= w1],
                            key=lambda e: (float(e['timestamp_s']), e.get('type', '')))
            # 同一事件可能在 rule/ttnet 各出现一次，按字段去重
            uniq = {}
            for e in events:
                uniq[(round(float(e['timestamp_s']), 6), e.get('type'), e.get('side'),
                      tuple(e.get('position_px') or []), tuple(e.get('sources') or []))] = e
            events = sorted(uniq.values(), key=lambda e: (float(e['timestamp_s']), e.get('type', '')))
            nearby_rules = [d for d in (rule_ctx['decisions'] + tt_ctx['decisions'])
                            if w0 <= float(d.get('virtual_t', -1e9)) <= w1]
            rr = {k2: d for k2, d in ((round(float(d.get('virtual_t', 0)), 6), d)
                                      for d in nearby_rules)}
            nearby_rules = sorted(rr.values(), key=lambda d: d.get('virtual_t', 0))
            tw = 1920; th = 1080
            # 根据观测尺寸取实际缩放标定
            first_obs_path = _resolve(manifest_dir, entry['tracknet'])
            obs = _read_jsonl(first_obs_path)
            if obs:
                tw, th = obs[0].get('image_size') or [tw, th]
            calib_scaled = calib.scaled_to(int(tw), int(th))
            card = {'game': name, 'gt': gt, 'rule': row_rule, 'ttnet': row_tt,
                    'trace': trace, 'score_curve': curve, 'events': events,
                    'nearby_rules': nearby_rules, 'rule_ctx': rule_ctx, 'tt_ctx': tt_ctx,
                    'net_x': calib_scaled.net_x, 'net_zone': calib_scaled.net_zone,
                    'calib_scaled': calib_scaled, 'window': (w0, w1),
                    'score_peaks': _score_peaks(curve)}
            category, basis = _initial_category(card, rule_ctx, tt_ctx)
            card['category'], card['basis'] = category, basis
            cards.append(card)

    # 手工复核后的分类覆盖：键为 game@timestamp。自动规则先把候选证据打出来，
    # 最终以这里的人工结论为准，避免脚本把“有 NET_TOUCH”直接当成同一类错误。
    overrides = {
        'test_1@138.225': ('E', '判对：A 侧落台触发 OWN_COURT，rule/ttnet 链路均有效'),
        'test_2@9.450': ('E', '判对：A 侧落台触发 OWN_COURT，默认融合下 rule/ttnet 均有效'),
        'test_3@9.833': ('E', 'rule 判对 OWN_COURT；TTNet 未确认。网带内有反向击球，但未改变已成立的落台链'),
        'test_3@19.425': ('A', '19.458s 网带内 RACKET_HIT A，与丢分方 B 相反；rule/ttnet 均漏判'),
        'test_3@46.683': ('E', '反向击球位置 x-net_x=-52.5 px，超出 30 px 网带；无 NET_TOUCH，归其他'),
        'test_4@7.067': ('A', '7.267s 网带内 RACKET_HIT B（丢分方 A 的反侧）并有 NET_TOUCH；未形成判罚'),
        'test_4@31.283': ('C', '31.317s 已发 NET_TOUCH B 并伴随 RACKET_HIT B，但复核/状态机未采用，最终漏判'),
        'test_4@37.750': ('E', '判对：OUT 链路，rule/ttnet 均正确'),
        'test_4@63.692': ('E', '判对：OWN_COURT，rule/ttnet 均正确；之后网带内反向击球未改判'),
        'test_4@138.317': ('A', '138.342s NET_TOUCH B 后 138.350s 网带内 RACKET_HIT A（丢分方 B 的反侧），最终误判 OUT B'),
        'test_4@257.058': ('E', 'rule 判对 OWN_COURT；TTNet 漏判，但已有落台链正确'),
        'test_4@266.533': ('A', '266.708s 网带内 RACKET_HIT B（丢分方 A 的反侧）并有 NET_TOUCH；未形成判罚'),
        'test_5@58.575': ('A', '58.775s 网带内 RACKET_HIT A（丢分方 B 的反侧）；网带多次 NET_TOUCH，rule/ttnet 均漏判'),
        'test_6@53.417': ('A', '53.458s 网带内 RACKET_HIT B（丢分方 A 的反侧），53.433s 有 NET_TOUCH B；两侧漏判'),
        'test_6@88.100': ('E', '判对：NO_RETURN 链路，rule/ttnet 均正确'),
        'test_7@23.925': ('E', 'TTNet 判对 OWN_COURT；rule 漏判。之后网带内反向击球未改变正确落台链'),
        'test_7@64.750': ('C', '64.925s NET_TOUCH A 后 65.033s RACKET_HIT A；证据未被采用，rule/ttnet 均漏判'),
        'test_7@128.758': ('C', '129.033s NET_TOUCH B 后 129.058s RACKET_HIT B；证据未被采用，rule/ttnet 均漏判'),
    }
    for card in cards:
        key = f"{card['game']}@{card['gt']['timestamp_s']:.3f}"
        if key in overrides:
            card['category'], card['basis'] = overrides[key]

    for card in cards:
        slug = f"{card['game']}_{card['gt']['timestamp_s']:.3f}".replace('.', '_')
        img_rel = os.path.join('img', slug + '.png')
        _draw_card(os.path.join(out_dir, img_rel), card)
        card['image'] = img_rel.replace('\\', '/')

    # LET / SERVE_INCOMPLETE: holdout only
    for entry in entries:
        if entry.get('name') == 'test_2':
            continue
        name = entry['name']
        rule_dir = os.path.join(batch_dir, f'{name}_rule')
        tt_dir = os.path.join(batch_dir, f'{name}_ttnet')
        net_points = _net_endings(entry, manifest_dir)
        seen = set()
        for group, d in (('rule', rule_dir), ('ttnet', tt_dir)):
            for dec in _read_jsonl(os.path.join(d, 'decisions.jsonl')):
                if dec.get('kind') not in ('LET',) and dec.get('rule') != 'SERVE_INCOMPLETE':
                    continue
                key = (group, dec.get('frame_id'), dec.get('rule'), dec.get('virtual_t'))
                if key in seen:
                    continue
                seen.add(key)
                t = float(dec.get('virtual_t', 0))
                nearest = _nearest_net_time(net_points, t)
                dt = None if nearest is None else round(t - nearest, 3)
                if dt is None:
                    relation = '该视频无下网真值'
                elif abs(dt) <= 1.0:
                    relation = '时间上邻近下网真值，需核对是否同一回合'
                elif abs(dt) <= 2.0:
                    relation = '接近但非紧邻，可能是回合后残留或漏检'
                else:
                    relation = '与下网真值无时间邻近关系'
                let_rows.append({'game': name, 'group': group, 't': t, 'kind': dec.get('kind'),
                                 'rule': dec.get('rule'), 'text': dec.get('text'),
                                 'nearest_net_t': nearest, 'dt_s': dt, 'relation': relation})

    counts = Counter(c['category'] for c in cards if c['game'] != 'test_2')
    total_counts = Counter(c['category'] for c in cards)
    holdout_correct_rule = sum(bool(c['rule'].get('correct')) for c in cards if c['game'] != 'test_2')
    holdout_correct_tt = sum(bool(c['ttnet'].get('correct')) for c in cards if c['game'] != 'test_2')
    lines = ['# 网附近下网分诊断', '',
             f'对象：留出集 {sum(1 for c in cards if c["game"] != "test_2")} 分 + test_2 '
             f'1 分；共 {len(cards)} 张卡。', '',
             f'- 留出集 rule 判对：{holdout_correct_rule}/17',
             f'- 留出集 TTNet 判对：{holdout_correct_tt}/17', '',
             '## 汇总表', '',
             '| 视频 | 真值 | 得分方 | rule | TTNet | 分类 | 依据摘要 |',
             '|---|---:|---|---|---|---|---|']
    for c in cards:
        r, t = c['rule'], c['ttnet']
        rs = f"{_fmt(r.get('our_t'))} {r.get('our_winner') or ''} {r.get('our_rule') or ''} " \
             f"{'✓' if r.get('correct') else '✗'}"
        ts = f"{_fmt(t.get('our_t'))} {t.get('our_winner') or ''} {t.get('our_rule') or ''} " \
             f"{'✓' if t.get('correct') else '✗'}"
        lines.append(f"| {c['game']} | {c['gt']['timestamp_s']:.3f}s {c['gt']['label']} | "
                     f"{c['gt']['winner']} | {rs} | {ts} | **{c['category']}** | {c['basis']} |")

    lines += ['', '## 类别计数（留出集）', '',
              '| A 撞网反向 | B 网附近丢球 | C 有证据未采用 | D 本方落台漏检 | E 其他/已判对 |',
              '|---:|---:|---:|---:|---:|',
              f"| {counts['A']} | {counts['B']} | {counts['C']} | {counts['D']} | {counts['E']} |", '',
              '按第 3 节门槛：A、B、C、D 各自需要留出集至少 3 分才建议实现对应部分；本轮只诊断。', '',
              '**建议**：', '',
              f"- A={counts['A']}（≥3）：建议第二步做 3.1 网附近异常触发器 + 3.2 网带内击球先扣住、复核后撤销/放行；"
              "必须带区间约束和消融。",
              f"- B={counts['B']}（<3）：不实现 3.3 的网附近丢球专项。",
              f"- C={counts['C']}（≥3）：三例都有 `NET_TOUCH` 却被对打状态忽略；这是共用状态机能力缺口，"
              "应在回执中提交给状态机维护方，不建议在 cascade 层绕过状态机自行判分。",
              f"- D={counts['D']}（<3）：不单独立项，落台召回问题已在相关卡片中记录。", '',
              '## LET / SERVE_INCOMPLETE 对照（留出集）', '']
    if let_rows:
        lines += ['| 视频 | 组 | t | kind | rule | 最近下网真值 | Δt | 判断 |', '|---|---|---:|---|---|---:|---:|---|']
        for x in let_rows:
            lines.append(f"| {x['game']} | {x['group']} | {x['t']:.3f} | {x['kind']} | {x['rule']} | "
                         f"{_fmt(x['nearest_net_t'])} | {_fmt(x['dt_s'])} | {x['relation']} |")
    else:
        lines.append('_未发现 LET / SERVE_INCOMPLETE 判罚。_')

    lines += ['', '## 诊断卡', '']
    for i, c in enumerate(cards, 1):
        r, t = c['rule'], c['ttnet']
        lines += [f"### {i}. {c['game']} / {c['gt']['timestamp_s']:.3f}s / {c['gt']['label']}", '',
                  f"**分类：{c['category']}** — {c['basis']}", '',
                  '| 项 | 结果 |', '|---|---|',
                  f"| 真值 | {c['gt']['timestamp_s']:.3f}s，{c['gt']['label']}，得分方 {c['gt']['winner']} |",
                  f"| rule | {_fmt(r.get('our_t'))}s，{r.get('our_winner') or '—'}，"
                  f"{r.get('our_rule') or r.get('note') or '—'}，{'✓' if r.get('correct') else '✗'} |",
                  f"| TTNet | {_fmt(t.get('our_t'))}s，{t.get('our_winner') or '—'}，"
                  f"{t.get('our_rule') or t.get('note') or '—'}，{'✓' if t.get('correct') else '✗'} |",
                  f"| 网线/网带 | net_x={c['net_x']} px，net_zone={c['net_zone']} |", '',
                  f"![轨迹与分数]({c['image']})", '',
                  '**覆盖与复核**', '',
                  f"- rule：{_coverage_text(c['rule_ctx'])}",
                  f"- TTNet：{_coverage_text(c['tt_ctx'])}",
                  f"- TTNet 分数峰值：bounce={_fmt(c['score_peaks']['max_bounce'])}"
                  f"@{c['score_peaks']['bounce_frame']}，net={_fmt(c['score_peaks']['max_net'])}"
                  f"@{c['score_peaks']['net_frame']}", '',
                  '**事件时间线（真值前 1.5 s 到后 1.0 s）**', '',
                  _event_table(c['events']),
                  '**逐帧轨迹**', '',
                  _trace_table(c['trace'])]
    md = os.path.join(out_dir, 'net_points.md')
    with open(md, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    recommendations = {
        'A': 'implement_3_1_and_3_2' if counts['A'] >= 3 else 'below_threshold',
        'B': 'implement_3_3' if counts['B'] >= 3 else 'below_threshold',
        'C': 'state_machine_net_touch_support' if counts['C'] >= 3 else 'below_threshold',
        'D': 'bounce_recall' if counts['D'] >= 3 else 'below_threshold',
    }
    raw = {'cards': [], 'let': let_rows, 'counts_holdout': dict(counts),
           'counts_all': dict(total_counts), 'recommendations': recommendations}
    for c in cards:
        raw['cards'].append({k: v for k, v in c.items()
                             if k not in ('calib_scaled', 'trace', 'score_curve', 'events',
                                          'rule_ctx', 'tt_ctx')})
    with open(os.path.join(out_dir, 'diagnostics.json'), 'w', encoding='utf-8') as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    return {'cards': cards, 'let': let_rows, 'counts_holdout': dict(counts),
            'counts_all': dict(total_counts), 'net_points_md': md}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='生成网附近下网分诊断卡')
    ap.add_argument('--manifest', default=os.path.join('cascade', 'batch_manifest.json'))
    ap.add_argument('--batch-dir', default=os.path.join('outputs', 'cascade_batch'))
    ap.add_argument('--out-dir', default=os.path.join('outputs', 'net_diag'))
    ap.add_argument('--ttnet-k', type=int, default=DEFAULT_K)
    a = ap.parse_args(argv)
    result = build_cards(a.manifest, a.batch_dir, a.out_dir, a.ttnet_k)
    print(f"诊断卡：{result['net_points_md']}")
    print(f"留出集类别：{result['counts_holdout']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
