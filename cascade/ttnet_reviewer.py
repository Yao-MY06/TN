# -*- coding: utf-8 -*-
"""TTNetReviewer：在提议区间上用 TTNet 的 bounce 分数复核。

接口与 reviewers.py 一致：``review(clip) -> ReviewResult``，属性 ``name``、``max_lead_s``。

帧与窗口的对应（事件对齐偏移 k）
--------------------------------
TTNet 一个窗口是 9 帧 [e-8, e]。球位置对应末帧 e；**事件分数对应 e-k**。
默认 k=4：训练代码 ``get_events_infor`` 里窗口以 ``smooth_idx`` 为中心
（[smooth_idx-4, smooth_idx+4]），事件标签 ``smooth_event_labelling(.., smooth_idx, ..)``
的峰值落在中间帧；而球位置标签取 ``last_f_idx = smooth_idx + 4``。
chagoup 的 ttnet_adapter 把事件也绑到末帧，等于假设 k=0。
**k 必须用真值核实**（见 README「待核实」）。

给区间 [a, b] 内每帧打分，需要窗口末帧 e ∈ [a+k, b+k]，即读帧
[a-(8-k), b+k]：pre_ctx = 8-k，post_ctx = k。

判定表
------
- 缺上下文帧（replay 缓存里缺窗口 / live 帧源取不到帧）→ error
- 上下文完整、区间内没有 bounce 驼峰 → rejected
- 有驼峰，峰值 ±5 帧内有真实检出（TrackNet/YOLO，visible 且非 predicted），
  该检出点在台面多边形内，side_of 为 A/B → confirmed，输出 BOUNCE
- 有驼峰但缺上述任一印证 → uncertain（附原因码）

TTNet 的 net 分数只作“过网”证据写进 details，**绝不**生成 NET_TOUCH。
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import namedtuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

WINDOW = 9
DEFAULT_K = 4
BOUNCE_HIGH, BOUNCE_LOW = 0.65, 0.55
NET_HIGH, NET_LOW = 0.50, 0.40
EVIDENCE_TOL = 5

Hump = namedtuple('Hump', 'start end peak_frame peak truncated')


# ---------------------------------------------------------------------------
# 纯函数（不依赖 torch / 视频，可单测）
# ---------------------------------------------------------------------------
def check_k(k):
    k = int(k)
    if not 0 <= k <= WINDOW - 1:
        raise ValueError('k 必须在 0..8，得到 {}'.format(k))
    return k


def window_end_for(frame, k):
    """为第 frame 帧打事件分数要用的窗口末帧。"""
    return frame + k


def event_frame_for(window_end, k):
    """窗口末帧 e 的事件分数对应的帧：e - k。"""
    return window_end - k


def context_range(a, b, k):
    """区间 [a, b] 需要读的帧范围（含两端）：[a-(8-k), b+k]。"""
    k = check_k(k)
    return a - (WINDOW - 1 - k), b + k


def find_humps(frames, scores, high=BOUNCE_HIGH, low=BOUNCE_LOW):
    """迟滞双阈值驼峰检测（与 chagoup event_processing._feed_hump 同规则）。

    上穿 high 进入驼峰；在驼峰内分数 >= low 都算驼峰；跌破 low 结束。
    区间末尾仍在驼峰中 → 照样输出，truncated=True。
    峰值取驼峰内最大分数，并列取最早一帧。
    """
    out = []
    active = False
    start = peak_f = None
    peak = -1.0
    last_f = None
    for f, s in zip(frames, scores):
        if s is None or (isinstance(s, float) and np.isnan(s)):
            s = 0.0
        if not active:
            if s >= high:
                active, start, peak_f, peak = True, f, f, s
        else:
            if s >= low:
                if s > peak:
                    peak_f, peak = f, s
            else:
                out.append(Hump(start, last_f, peak_f, float(peak), False))
                active = False
        last_f = f
    if active:
        out.append(Hump(start, last_f, peak_f, float(peak), True))
    return out


def nearest_detection(detections, frame, tol=EVIDENCE_TOL):
    """detections: {frame_id: [(x, y, source), ...]}。返回 ±tol 内离 frame 最近的一次
    真实检出 (frame_id, x, y, source)；距离相同取较早的一帧（更接近“最近一次已发生”）。"""
    best = None
    for d in sorted(range(-tol, tol + 1), key=lambda v: (abs(v), v)):
        dets = detections.get(frame + d)
        if dets:
            x, y, src = dets[0]
            best = (frame + d, float(x), float(y), src)
            break
    return best


class TableGeometry:
    """台面多边形 + 侧别。侧别用 detect_bounce.side_of（单应性投影到俯视平面）。"""

    def __init__(self, quad, dead=0.05):
        from ._paths import load_detect_bounce
        db = load_detect_bounce()
        self._db = db
        self.poly = np.asarray(quad, dtype=float)
        self.H = db.table_homography(self.poly)
        self.dead = dead

    def inside(self, x, y):
        return bool(self._db._pip(np.array([[x, y]], dtype=float), self.poly)[0])

    def side(self, x, y):
        return self._db.side_of((x, y), self.H, self.dead)[0]


def judge(humps, detections, geometry, tol=EVIDENCE_TOL):
    """判定表。返回 (status, events, reasons)。不处理 error（调用方先查上下文）。

    events: [{'frame_id', 'side', 'position_px', 'confidence', 'evidence_frame', 'evidence_source'}]
    """
    if not humps:
        return 'rejected', [], ['no_hump']
    events, reasons = [], []
    for h in humps:
        det = nearest_detection(detections, h.peak_frame, tol)
        if det is None:
            reasons.append('no_detection_near_peak@{}'.format(h.peak_frame))
            continue
        f, x, y, src = det
        if geometry is None:
            reasons.append('no_table_geometry')
            continue
        if not geometry.inside(x, y):
            reasons.append('off_table@{}'.format(h.peak_frame))
            continue
        side = geometry.side(x, y)
        if side not in ('A', 'B'):
            reasons.append('side_unknown@{}'.format(h.peak_frame))
            continue
        events.append({'frame_id': int(h.peak_frame), 'side': side, 'position_px': [x, y],
                       'confidence': float(h.peak), 'evidence_frame': int(f),
                       'evidence_source': src, 'truncated': bool(h.truncated)})
    if events:
        return 'confirmed', events, reasons
    return 'uncertain', [], reasons


def detections_from_observations(observations):
    """统一 Observation（dict）流 → {frame_id: [(x, y, source)]}，只收真实检出。"""
    det = {}
    for o in observations:
        b = o.get('ball') or {}
        if not b.get('visible') or b.get('predicted', False) or b.get('center_px') is None:
            continue
        x, y = b['center_px']
        det.setdefault(int(o['frame_id']), []).append((float(x), float(y), o.get('source', '?')))
    return det


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# 分数后端：按窗口末帧去重
# ---------------------------------------------------------------------------
class ReplayScores:
    """读 ttnet_cache.py 产出的 JSONL。"""
    kind = 'replay'

    def __init__(self, path_or_rows):
        rows = load_jsonl(path_or_rows) if isinstance(path_or_rows, str) else list(path_or_rows)
        self.rows = {int(r['window_end']): r for r in rows if 'window_end' in r}
        self.used = set()
        self.infer_ms = []

    def get(self, window_end):
        r = self.rows.get(int(window_end))
        if r is None:
            return None
        if window_end not in self.used:
            self.used.add(window_end)
            if r.get('infer_ms') is not None:
                self.infer_ms.append(float(r['infer_ms']))
        return r


class LiveScores:
    """从帧源现算。frame_source(frame_id) -> (128,320,3) RGB uint8 或 None。"""
    kind = 'live'

    def __init__(self, engine, frame_source, timestamp_of=None):
        from .ttnet_engine import stack_window
        self._stack = stack_window
        self.engine = engine
        self.frame_source = frame_source
        self.timestamp_of = timestamp_of
        self.rows = {}
        self.used = set()
        self.infer_ms = []

    def get(self, window_end):
        e = int(window_end)
        if e in self.rows:
            return self.rows[e]
        if e - (WINDOW - 1) < 0:
            return None
        frames = []
        for f in range(e - (WINDOW - 1), e + 1):
            img = self.frame_source(f)
            if img is None:
                return None
            frames.append(img)
        from .ttnet_cache import row_from_output
        t0 = time.perf_counter()
        out = self.engine.infer(self._stack(frames))
        ms = (time.perf_counter() - t0) * 1000.0
        ts = self.timestamp_of(e) if self.timestamp_of else None
        row = row_from_output(e, ts, out, ms)
        self.rows[e] = row
        self.used.add(e)
        self.infer_ms.append(ms)
        return row


# ---------------------------------------------------------------------------
# 复核器
# ---------------------------------------------------------------------------
class TTNetReviewer:
    """与 RuleReviewer / OracleReviewer 同接口。

    ``max_lead_s = pre_s``：TTNet 的峰可以落在片段内任意位置，最早到片段起点
    （锚点 − pre_s），提交层据此算水位线。
    """

    name = 'ttnet'

    def __init__(self, scores, detections=None, geometry=None, k=DEFAULT_K, pre_s=0.1,
                 fps=120.0, high=BOUNCE_HIGH, low=BOUNCE_LOW, tol=EVIDENCE_TOL):
        """
        :param scores: ReplayScores 或 LiveScores
        :param detections: {frame_id: [(x, y, source)]}，TrackNet/YOLO 真实检出
        :param geometry: TableGeometry（台面四角）
        :param k: 事件对齐偏移
        :param pre_s: 提议区间前沿（ProposalParams.pre_s）
        """
        self.scores = scores
        self.detections = detections or {}
        self.geometry = geometry
        self.k = check_k(k)
        self.pre_ctx = WINDOW - 1 - self.k
        self.post_ctx = self.k
        self.max_lead_s = float(pre_s)
        self.fps = float(fps)
        self.high, self.low, self.tol = high, low, tol
        self.n_reviews = 0
        self._n_events = 0
        self.log = []          # 每个片段的驼峰与过网证据，run_cascade 写进 ttnet_reviews.jsonl

    @property
    def n_windows(self):
        """实际用到的窗口数（重叠片段去重后）——TTNet 调用比的分子。"""
        return len(self.scores.used)

    def frame_scores(self, a, b):
        """区间 [a, b] 内逐帧 (bounce, net)；缺窗口返回 (None, 缺的窗口末帧列表)。"""
        frames, bounce, net, missing = [], [], [], []
        for f in range(a, b + 1):
            r = self.scores.get(window_end_for(f, self.k))
            if r is None:
                missing.append(window_end_for(f, self.k))
                continue
            frames.append(f)
            bounce.append(float(r['bounce']))
            net.append(float(r['net']))
        if missing:
            return None, missing
        return (frames, bounce, net), []

    def _result(self, clip, status, events, reason, per, n_new):
        from .reviewers import ReviewResult
        return ReviewResult(clip.clip_id, clip.proposal_ids, status, events, reason, per,
                            reviewer=self.name, submitted_frame_id=clip.submitted_frame_id,
                            n_windows=n_new)

    def review(self, clip):
        from observation import make_event
        self.n_reviews += 1
        a, b = int(clip.start_frame_id), int(clip.end_frame_id)
        used_before = len(self.scores.used)
        got, missing = (None, []) if clip.missing_context else self.frame_scores(a, b)
        n_new = len(self.scores.used) - used_before
        entry = {'clip_id': clip.clip_id, 'start_frame_id': a, 'end_frame_id': b, 'k': self.k}
        if got is None:
            reason = 'missing_context'
            entry.update(status='error', missing_windows=missing[:20], n_missing=len(missing))
            self.log.append(entry)
            per = {pid: 'error:missing_context' for pid in clip.proposal_ids}
            return self._result(clip, 'error', [], reason, per, n_new)

        frames, bounce, net = got
        humps = find_humps(frames, bounce, self.high, self.low)
        net_humps = find_humps(frames, net, NET_HIGH, NET_LOW)   # 只作过网证据，不生成 NET_TOUCH
        status, evs, reasons = judge(humps, self.detections, self.geometry, self.tol)
        events = []
        for e in evs:
            events.append(make_event(
                event_id='ttnet_bounce_{:05d}'.format(self._n_events),
                timestamp_s=e['frame_id'] / self.fps,
                event_type='BOUNCE', side=e['side'], position_px=e['position_px'],
                confidence=e['confidence'], sources=['ttnet', e['evidence_source']]))
            self._n_events += 1

        # 每条提议：落在它时间范围内的确认事件 → confirmed:<侧>；否则沿用片段结论
        per = {}
        for p in clip.proposals:
            lo = p.start_timestamp_s * self.fps - 0.5
            hi = p.end_timestamp_s * self.fps + 0.5
            hit = [e for e in evs if lo <= e['frame_id'] <= hi]
            per[p.proposal_id] = 'confirmed:{}'.format(hit[0]['side']) if hit else (
                'rejected:no_hump' if status == 'rejected' else
                'uncertain' if status == 'uncertain' else 'outside_confirmed_peak')
        reason = {'confirmed': 'ttnet_hump_with_evidence', 'rejected': 'no_hump'}.get(
            status, ';'.join(reasons) or status)
        entry.update(status=status, reasons=reasons, humps=[h._asdict() for h in humps],
                     net_cross=[h._asdict() for h in net_humps],
                     max_bounce=round(max(bounce), 4) if bounce else None, n_new_windows=n_new)
        self.log.append(entry)
        return self._result(clip, status, events, reason, per, n_new)


# ---------------------------------------------------------------------------
# 构造（run_cascade.py 用）
# ---------------------------------------------------------------------------
def add_ttnet_args(ap):
    g = ap.add_argument_group('TTNet 复核器（--reviewer ttnet）')
    g.add_argument('--ttnet-cache', default=None, help='replay：ttnet_cache.py 产出的 JSONL')
    g.add_argument('--video', default=None, help='live：视频路径（必须与观测同一个文件）')
    g.add_argument('--ttnet-weights', default=None, help='live：TTNet 权重 .pth')
    g.add_argument('--ttnet-src', default=None, help='live + torch 引擎：TTNet/src，默认 ../TTNet/src')
    g.add_argument('--ttnet-engine', default='auto', choices=['auto', 'torch', 'numpy'])
    g.add_argument('--ttnet-device', default='cuda:0')
    g.add_argument('--ttnet-k', type=int, default=DEFAULT_K, help='事件对齐偏移，默认 4（待真值核实）')
    return g


def build_ttnet_reviewer(args, poly, observations, fps=120.0, pre_s=0.1):
    """observations：TrackNet（及 YOLO）逐帧观测 dict，用作印证。"""
    if args.ttnet_cache:
        scores = ReplayScores(args.ttnet_cache)
    elif args.video and args.ttnet_weights:
        from ._paths import TTNET_ROOT
        from .ttnet_engine import make_engine
        from .ttnet_cache import VideoFrameSource
        src_dir = args.ttnet_src or os.path.join(TTNET_ROOT, 'src')
        eng = make_engine(args.ttnet_engine, args.ttnet_weights, src_dir, args.ttnet_device)
        src = VideoFrameSource(args.video)
        scores = LiveScores(eng, src, src.timestamp_of)
    else:
        raise SystemExit('--reviewer ttnet 需要 --ttnet-cache（replay）或 --video + --ttnet-weights（live）')
    return TTNetReviewer(scores, detections_from_observations(observations), TableGeometry(poly),
                         k=args.ttnet_k, pre_s=pre_s, fps=fps)
