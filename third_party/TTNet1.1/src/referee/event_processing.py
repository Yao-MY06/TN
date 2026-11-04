# -*- coding: utf-8 -*-
"""
轨迹处理 + 事件提取层。

职责划分（对应方案文档第 3 节「分开事件提取，共用计分规则」）：
- 轨迹处理：平滑、断段、区分「真实检测」与「预测位置」；
- 事件提取：把模型的事件候选分数 + 轨迹几何，转换成状态机可消费的 Event。

关键设计取舍
------------
1. **事件分数是候选，不是事件**
   TTNet 的 bounce 分数是逐帧回归值（训练目标是 cos 衰减的可乐形曲线，
   峰值在事件帧、两侧衰减），因此一次落台会在连续多帧产生高分。
   必须做「时间峰值提取」（局部极大 + 最小间隔）才能得到唯一事件，
   否则状态机会重复加分。

2. **bounce 必须过几何校验**
   实测 game_1 中 96% 的 bounce 落在球台掩码内，说明该标签指「落台」而非
   任意弹跳。因此要求落点位于球台 ROI 内（含少量容差）才采纳，
   过滤掉球在台外或画面外的误检。

3. **球消失不等于失分**
   漏检、遮挡、飞出画面都会让球不可见。这里只标记 BALL_LOST 用于断段，
   判分完全交给状态机结合前后事件决定。

4. **使用时间戳而非帧数**
   不同模型处理帧率、连续帧输入、输出延迟都不同，统一用 seconds 表示时间。
"""

from collections import deque

import numpy as np

from referee.unified_types import Event, EventType, Side


class TrajectoryProcessor:
    """把逐帧 Observation 整理成干净轨迹。

    - 中值/均值平滑，抑制帧间抖动；
    - 记录丢球断段，避免把跨段的两点连成一条「飞线」；
    - 区分 detected / predicted，预测点绝不冒充真实检测。
    """

    def __init__(self, smooth_window=5, max_gap_frames=12, jump_limit_px=400.0):
        """
        :param smooth_window: 中值平滑窗口（帧）
        :param max_gap_frames: 超过该帧数仍无球则判定断段
        :param jump_limit_px: 单帧位移超过该值视为异常跳变（用于标记可疑点）
        """
        self.smooth_window = smooth_window
        self.max_gap_frames = max_gap_frames
        self.jump_limit_px = jump_limit_px

        self._buffer = deque(maxlen=smooth_window)
        self._last_seen_frame = None
        self._last_seen_xy = None
        self.segment_id = 0
        self.suspicious_count = 0
        self._gap_exceeded = False   # 防止同一段丢球被重复断段

    def update(self, obs):
        """输入一帧观测，返回带平滑与段标记的轨迹点。

        :return: dict(point, visible, is_predicted, segment_id, suspicious)
        """
        if not obs.ball.visible:
            gap = (obs.frame_id - self._last_seen_frame) if self._last_seen_frame is not None else 0
            # 只在「首次」超过阈值时断段一次，否则连续丢球会反复 +1
            if (self._last_seen_frame is not None and gap > self.max_gap_frames
                    and not self._gap_exceeded):
                self.segment_id += 1
                self._buffer.clear()
                self._last_seen_xy = None
                self._gap_exceeded = True
            return {
                'frame_id': obs.frame_id,
                'timestamp_s': obs.timestamp_s,
                'visible': False,
                'point': None,
                'is_predicted': False,
                'segment_id': self.segment_id,
                'suspicious': False,
            }

        raw = obs.ball.center_px
        suspicious = False
        if self._last_seen_xy is not None:
            jump = float(np.hypot(raw[0] - self._last_seen_xy[0], raw[1] - self._last_seen_xy[1]))
            if jump > self.jump_limit_px:
                suspicious = True
                self.suspicious_count += 1

        self._buffer.append(raw)
        if len(self._buffer) >= 3:
            arr = np.array(self._buffer, dtype=np.float32)
            smoothed = [float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))]
        else:
            smoothed = [float(raw[0]), float(raw[1])]

        self._last_seen_frame = obs.frame_id
        self._last_seen_xy = raw
        self._gap_exceeded = False   # 重新看到球，重置断段标记

        return {
            'frame_id': obs.frame_id,
            'timestamp_s': obs.timestamp_s,
            'visible': True,
            'point': smoothed,
            'raw_point': raw,
            'is_predicted': False,
            'segment_id': self.segment_id,
            'suspicious': suspicious,
            'confidence': obs.ball.confidence,
        }


class EventExtractor:
    """把事件候选分数与轨迹几何转成统一 Event（含去重）。

    可疑击球的标记（可选，`filter_suspect_hit=True` 时启用）
    ------------------------------------------------------
    HIT 目前由「轨迹水平方向反转」推断，而**球撞台面同样会让 x 方向速度
    反号**，因此会混入大量假击球。实测 test_4：168 个 HIT 中有 60 个（36%）
    落在台面高度带内，其中 63% 与某个落台时刻相距 ≤0.3s —— 典型的
    「球落台 → 方向反转 → 被误判为击球」。

    真实击球时球在台面上方的空中，这个几何差别不需要新模型就能利用。
    开启后这类 HIT 的 confidence 会被降到 `SUSPECT_HIT_CONFIDENCE` 并在
    sources 里加 'suspect_bounce'，**但事件本身仍会输出** ——
    由状态机决定是否采信，事件层不擅自删数据。

    峰值提取：迟滞双阈值（Schmitt trigger）驼峰检测
    ------------------------------------------------
    实测发现（test_2，真值 bounce=28、net=21）：

    - TTNet 的事件输出**召回率很高（96%~100%）但精确率只有约 55%**，
      即「该报的都报了，但报多了」。预测 49 次 bounce，真值仅 28 次。
    - 分数曲线是单峰驼峰（训练目标是 cos 衰减），一次事件会连续多帧高分。
    - 原先的「最小间隔 + 逐帧上升沿」策略有两个缺陷：
        a) 驼峰内部的小幅抖动会被当成新的上升沿，重复产出；
        b) 固定间隔 0.25s 与 120fps 的节奏不匹配，长驼峰仍会重复触发。

    改用迟滞双阈值：
      分数上穿 high（默认 0.5）→ 进入驼峰，记录峰值；
      分数下穿 low（默认 0.35）→ 驼峰结束，若峰值足够高则**只输出一次**事件。
    这样一个驼峰严格对应一个事件，从根本上消除重复计数；
    同时 low < high 的滞回带能避免临界抖动反复穿越。

    几何校验（可选）：bounce 落点若不在球台 ROI 内则丢弃。
    """

    # 被标记为「疑似台面弹跳」的 HIT 所使用的置信度。
    # 状态机可用它区分「可信击球」与「疑似假击球」。
    SUSPECT_HIT_CONFIDENCE = 0.4

    def __init__(self, bounce_thresh=0.65, net_thresh=0.65,
                 hysteresis=0.1, min_event_interval_s=0.25,
                 table_margin_ratio=0.06, require_table_for_bounce=False,
                 filter_suspect_hit=False, hit_side_by_direction=False,
                 hit_hist_reset_s=0.40):
        """        :param bounce_thresh: bounce 驼峰上阈值。默认 0.65 —— 由离线网格搜索
            （tune_events.py）在 test_2 上确定，并在 test_3 上验证可泛化：
            test_2 bounce F1 88.5%（P81.8/R96.4），test_3 F1 75.0%（P61.2/R96.8）。
        :param net_thresh: net 驼峰上阈值。**默认 0.65**（原 0.5）。

            为什么从 0.5 提到 0.65（实测依据）
            ----------------------------------
            旧报告里 net 的 F1（test_2 73.7% / test_3 73.5%）是**在全时段**算的，
            而那些视频的标注只覆盖 10%~59% 的时长 —— 未标注时段的正确检出
            全被算成 FP，精确率被系统性压低，于是"看起来 0.5 就够了"。

            改成**只在标注窗口内统计精确率**后（`tune_event_thresholds.py`
            的 `evaluate()`），网格搜索的结果很一致：抬高阈值有明确收益。

            ========  ==========  ==========  ==========
            视频      th=0.5 F1   th=0.65 F1   th=0.85 F1
            ========  ==========  ==========  ==========
            test_1    68%         68%          68%
            test_2    70%         76%          74%
            test_3    75%         86%          86%
            ========  ==========  ==========  ==========

            0.65 是三段视频 F1 都不退化的最高公共值（0.85 对 test_2 反而略降）。
        :param min_event_interval_s: 同类事件最小间隔（秒），防止双峰误判。
            **默认 0.25**（原 0.10）。同样是网格搜索结论：0.25 能在保住
            BOUNCE F1 的同时把两段视频的 NET F1 平均抬高约 4 个百分点
            （0.10 → 0.25 时 test_1 NET 68%→76%），
            因为 NET 的驼峰常出现"双峰拖尾"，间隔太短会把同一次擦网报两次。
        :param table_margin_ratio: 球台 ROI 外扩比例
        :param require_table_for_bounce: 是否强制 bounce 落点必须在球台内。
            默认 False —— 「落在台外」本身也是有效判罚信息（出界），
            应由状态机结合上下文判断，而不是在事件层直接丢弃。
        :param filter_suspect_hit: 是否给「球在台面高度带内」的 HIT 打可疑标记。
            默认 False（保持既有行为）。开启后这类 HIT 的 confidence 会被降到
            `SUSPECT_HIT_CONFIDENCE`，并在 sources 里加 'suspect_bounce'。
            依据：TTNet 的 HIT 由「轨迹水平方向反转」推断，球撞台面时同样会
            反转，因此会混入大量假击球（实测 test_4 有 36% 的 HIT 落在台面
            高度带内，其中 63% 与某个落台时刻相距 ≤0.3s）。
            状态机可用 sources 判断是否采信，因此这里只打标记、不直接丢弃 ——
            「只标注、不删数据」是本项目一贯原则。
        :param hit_side_by_direction: **击球方按飞行方向归属，不按检出位置**。
            默认 **False（保持现有行为不变）**。这是 YOLO 分支 R-017 的做法，
            在本项目实测**未获收益**，因此只作为可选项保留，不设为默认。

            原理（为什么它本该更好）
            ------------------------
            假设球员一定把球打向对方半台，那么「球往哪飞」才是击球方的可靠判据；
            而「击球瞬间位置在左侧 → 击球方是 A」在高速球上并不可靠：水平速度反号
            这个判据是在击球之后几帧才被检出，120fps 下球每帧走约 10px，
            检出时球可能已越过球网，于是击球方被判成了对面。

            为什么在本项目不设为默认（实测结论）
            ----------------------------------
            用 ``referee/reextract_events.py`` 从观测缓存重建事件后对比
            （同一批观测、只换这一条归属规则，因此差异纯粹来自归属）：

            ========  ==========================  ======================
            视频      同侧落台 / 异侧落台（旧=位置）  同侧落台 / 异侧落台（新=方向）
            ========  ==========================  ======================
            test_1    3 / 65                          3 / 65
            test_2    2 / 25                          1 / 26
            test_3    8 / 12                          4 / 16
            ========  ==========================  ======================

            方向上确有改善（「击球方把球打进自己半台」变少了），但**判分反而更差**：
            test_1 的 yolo 引擎从 A6:B4（10 分）变成 A7:B1（8 分），
            且 OWN_COURT 从 1 次升到 5 次。原因是本项目的 HIT 检出率远低于 YOLO
            分支（TTNet 的 HIT 靠轨迹反转推断，每段视频只有几十个），
            一旦归属变化，``last_hitter`` 的更新时序随之改变，反而更容易与
            落台错位。

            结论：这条规则的效果**强依赖上游 HIT 召回率**，在 HIT 可靠之前不应
            默认开启。保留为开关，等 HIT 检测改进后再做 A/B。

            注：YOLO 分支能靠它把 OWN_COURT 从 8 降到 1，是因为它有可靠的
            卡尔曼轨迹速度；两者前提不同，结论不能直接照搬。
        """
        self.bounce_thresh = bounce_thresh
        self.net_thresh = net_thresh
        self.hysteresis = hysteresis
        self.min_event_interval_s = min_event_interval_s
        self.table_margin_ratio = table_margin_ratio
        self.require_table_for_bounce = require_table_for_bounce
        self.filter_suspect_hit = filter_suspect_hit
        self.hit_side_by_direction = hit_side_by_direction
        self.hit_hist_reset_s = float(hit_hist_reset_s)

        # 球拍证据过滤器（可选，用于标记不可信的击球）
        self._paddle_filter = None
        if filter_suspect_hit:
            from referee.paddle_detector import PaddleEvidenceFilter
            self._paddle_filter = PaddleEvidenceFilter()

        self._last_emit = {}
        self._counter = {}
        self._rejected = {'bounce_off_table': 0, 'interval': 0}

        # 驼峰状态机：{type: {'active': bool, 'peak': float, 'peak_ts': float, 'peak_pt': list}}
        self._hump = {
            'bounce': {'active': False, 'peak': 0.0, 'peak_ts': None, 'peak_pt': None},
            'net': {'active': False, 'peak': 0.0, 'peak_ts': None, 'peak_pt': None},
        }

        # 击球检测状态
        self._pt_hist = deque(maxlen=7)
        self._last_hit_ts = None
        self._lost_since = None
        self._segment_id = None

    # ---------------------------------------------------------------- 工具
    def _new_id(self, etype):
        self._counter[etype] = self._counter.get(etype, 0) + 1
        return '{}_{:05d}'.format(etype.lower(), self._counter[etype])

    @staticmethod
    def _side_of(point, table_box):
        """按球台外接框中线判断落点属于 A 侧（左）还是 B 侧（右）。"""
        if point is None or table_box is None:
            return Side.UNKNOWN
        mid_x = table_box[0] + table_box[2] / 2.0
        return Side.A if point[0] < mid_x else Side.B

    def _in_table(self, point, table_box):
        """落点是否在球台 ROI 内（含外扩容差）。"""
        if point is None or table_box is None:
            return True
        x, y, w, h = table_box
        mx = w * self.table_margin_ratio
        my = h * self.table_margin_ratio
        return (x - mx) <= point[0] <= (x + w + mx) and (y - my) <= point[1] <= (y + h + my)

    def _interval_ok(self, etype, ts):
        last = self._last_emit.get(etype)
        return last is None or (ts - last) >= self.min_event_interval_s

    # ---------------------------------------------------------------- 驼峰
    def _feed_hump(self, name, score, ts, point, high, table_box, obs_source):
        """把一个分数送入迟滞驼峰检测器，返回在「驼峰结束时」产生的事件（或 None）。

        迟滞逻辑：
          score >= high            → 进入/维持驼峰，更新峰值
          low <= score < high 且 active → 仍在驼峰内（滞回带），继续更新峰值
          score < low 且 active     → 驼峰结束，产出事件
        """
        if score is None:
            return None
        low = high - self.hysteresis
        h = self._hump[name]

        if score >= high:
            if not h['active']:
                h['active'] = True
                h['peak'] = score
                h['peak_ts'] = ts
                h['peak_pt'] = point
            elif score > h['peak']:
                h['peak'] = score
                h['peak_ts'] = ts
                h['peak_pt'] = point      # 峰值位置更可信
            return None

        if h['active'] and score >= low:
            # 滞回带内，维持驼峰
            if score > h['peak']:
                h['peak'] = score
                h['peak_ts'] = ts
                h['peak_pt'] = point
            return None

        if h['active']:
            # 驼峰结束 → 产出唯一事件
            h['active'] = False
            peak, peak_ts, peak_pt = h['peak'], h['peak_ts'], h['peak_pt']
            h['peak'] = 0.0
            h['peak_ts'] = None
            h['peak_pt'] = None

            if not self._interval_ok(name, peak_ts):
                self._rejected['interval'] += 1
                return None

            if name == 'bounce' and self.require_table_for_bounce \
                    and not self._in_table(peak_pt, table_box):
                self._rejected['bounce_off_table'] += 1
                return None

            self._last_emit[name] = peak_ts
            etype = EventType.BOUNCE if name == 'bounce' else EventType.NET_HIT
            return Event(
                event_id=self._new_id(etype),
                timestamp_s=peak_ts,
                type=etype,
                side=(self._side_of(peak_pt, table_box)
                      if name == 'bounce' else Side.UNKNOWN),
                position_px=[int(p) for p in peak_pt] if peak_pt else None,
                confidence=peak,
                sources=[obs_source],
            )

        return None

    # ---------------------------------------------------------------- 主逻辑
    def process(self, obs, traj_point):
        """处理一帧，返回本帧新产生的事件列表（通常为空）。"""
        events = []
        table_box = obs.extras.get('table_box')
        ts = obs.timestamp_s
        scores = obs.event_scores or {}
        point = traj_point.get('point')

        # 轨迹断段时处理击球历史。
        #
        # 这里有一个**实测过的坑**，改之前务必看懂：
        #
        # 原实现是「只要 segment 变化就清空 _pt_hist」。本意是防止把
        # 「丢球前的旧点位」和「重新出现后的新点位」拼成一段轨迹，
        # 再由水平方向反转「推断」出并不存在的击球。
        #
        # 但 TTNet 的球可见率只有 25%~48%，而 `TrajectoryProcessor` 只要
        # 连续丢球超过 `max_gap_frames=12`（120fps 下仅 0.1 秒）就 +1 段。
        # 于是 segment 在一次回合里能变几十次，`_pt_hist`（需要 6 个点才能
        # 判一次击球）几乎永远攒不满 —— HIT 被系统性砍掉一半：
        #
        #   ========  ==============  ==============  ==================
        #   视频      HIT(清空版)     HIT(修复后)     判分变化
        #   ========  ==============  ==============  ==================
        #   test_1    72              115             A7:B1  → A6:B4
        #   test_2    29              43              A9:B2  → A1:B2  ★
        #   test_3    25              51              A2:B4  → A3:B5
        #   ========  ==============  ==============  ==================
        #
        # test_2 尤其严重：30 秒里判出 11 分（真值是 3 个回合），全是
        # OWN_COURT。根因是 HIT 太少 → 状态机的 `last_hitter` 长期停留在
        # 旧值 → 正常落台被当成「击球方打进自己半台」。
        #
        # 驼峰（落台/擦网）**总是**在断段时清空：它属于已经断掉的轨迹段，
        # 若保留到新段再补发，会得到一个时间戳很久以前的「落台」，
        # 造成事件时间轴错位。这一点与击球历史不同 —— 击球是按点的
        # 相对位移算的，只要点本身间隔不远就不会算错。
        seg = traj_point.get('segment_id')
        if seg is not None and seg != self._segment_id:
            self._segment_id = seg
            for name in ('bounce', 'net'):
                self._hump[name].update({'active': False, 'peak': 0.0,
                                         'peak_ts': None, 'peak_pt': None})

        # ---- 击球历史的清空：按「实际时间间隔」逐帧检查，不挂在断段事件上
        #
        # 为什么不能只在 segment 变化的那一刻判断：
        # `TrajectoryProcessor` 的断段是**一次性**的 —— 它用一个
        # `_gap_exceeded` 标记保证「同一段丢球只 +1 次」，且这个标记在
        # **刚超过 12 帧（0.1s）时**就翻起来了。可那一刻球才丢了 0.1 秒，
        # 远没到 hit_hist_reset_s（0.40s），于是 stale 判 False，历史没清；
        # 等球真的丢了 1 秒，segment 早已不再变化，这个判断再也不会被触发。
        #
        # 结果就是：长时间断球后 _pt_hist 里仍留着 1 秒前的旧点，
        # 球一回来就与旧点拼成一段「跨越大半个球台」的轨迹，
        # 由水平方向反转凭空造出一次击球（实测复现：断 1 秒后仍会误报）。
        #
        # 因此改成逐帧按时间检查 —— 这才是「断了多久」的准确度量。
        if self._pt_hist and (ts - self._pt_hist[-1][0]) > self.hit_hist_reset_s:
            self._pt_hist.clear()
            self._last_hit_ts = None

        # 事件分数只在球可见时有意义：丢球期间模型输出不可靠
        if point is not None:
            # 注意：统一接口里 net 的字段名是 'net_hit'，而驼峰检测器内部用 'net'，
            # 两者必须显式映射，否则 scores.get('net') 永远是 None（曾导致 net 完全检测不到）。
            for name, key, thresh in (('bounce', 'bounce', self.bounce_thresh),
                                      ('net', 'net_hit', self.net_thresh)):
                ev = self._feed_hump(name, scores.get(key), ts, point,
                                     thresh, table_box, obs.source)
                if ev is not None:
                    events.append(ev)
                    # 把已确认的落台点喂给击球过滤器，让它在线估计
                    # 「台面高度带」—— 真实击球的球在台面之上，撞台时球贴台。
                    if name == 'bounce' and self._paddle_filter is not None:
                        self._paddle_filter.observe_bounce(ev.timestamp_s,
                                                           ev.position_px)

        # ---- 击球检测：球在某一侧发生水平方向反转
        if point is not None:
            self._pt_hist.append((ts, point))
            hit = self._detect_hit(obs, table_box)
            if hit is not None:
                events.append(hit)
            self._lost_since = None
        else:
            if self._lost_since is None:
                self._lost_since = ts
                events.append(Event(
                    event_id=self._new_id(EventType.BALL_LOST),
                    timestamp_s=ts,
                    type=EventType.BALL_LOST,
                    side=Side.UNKNOWN,
                    confidence=None,
                    sources=[obs.source],
                ))

        return events

    def _detect_hit(self, obs, table_box):
        """用轨迹水平方向反转检测击球。要求方向明确反转且速度足够。"""
        if len(self._pt_hist) < 6:
            return None
        ts_now = self._pt_hist[-1][0]
        if self._last_hit_ts is not None and (ts_now - self._last_hit_ts) < 0.20:
            return None

        pts = list(self._pt_hist)

        def vx(i, j):
            dt = pts[j][0] - pts[i][0]
            if dt <= 1e-6:
                return 0.0
            return (pts[j][1][0] - pts[i][1][0]) / dt

        v_before = vx(0, 2)
        v_after = vx(3, 5)
        min_speed = 60.0
        if abs(v_before) < min_speed or abs(v_after) < min_speed:
            return None
        if v_before * v_after >= 0:
            return None

        point = pts[-1][1]
        self._last_hit_ts = ts_now

        # ---- 击球方归属：按飞行方向，不按检出位置（R-017）
        #
        # v_after 是击出后的水平速度。球员一定把球打向对面，因此：
        #   球往右飞 (v_after > 0) → 击球方在左 (A)
        #   球往左飞 (v_after < 0) → 击球方在右 (B)
        # 退化情形（关掉开关 / v_after 恰为 0）仍按位置推断。
        if self.hit_side_by_direction and abs(v_after) > 1e-6:
            hit_side = Side.A if v_after > 0 else Side.B
        else:
            hit_side = self._side_of(point, table_box)

        hit = Event(
            event_id=self._new_id(EventType.HIT),
            timestamp_s=ts_now,
            type=EventType.HIT,
            side=hit_side,
            position_px=[int(point[0]), int(point[1])],
            confidence=None,
            sources=[obs.source, 'trajectory'],
        )

        # ---- 可疑击球标记（可选）
        #
        # 「水平方向反转」无法区分「球拍击球」与「球撞台面」：两者都会让
        # 球的 x 方向速度反号。而球撞台面时球贴在台面高度带内，真实击球时
        # 球在空中 —— 这个几何差别不需要新模型就能利用。
        # 这里只降置信度 + 加来源标记，绝不丢弃事件（保留给状态机裁决）。
        if self._paddle_filter is not None:
            suspicious, reasons = self._paddle_filter.is_hit_suspicious(ts_now, point)
            if suspicious:
                hit.confidence = self.SUSPECT_HIT_CONFIDENCE
                hit.sources = list(hit.sources) + ['suspect_bounce']
                self._rejected['suspect_hit'] = \
                    self._rejected.get('suspect_hit', 0) + 1

        return hit

    def flush(self):
        """视频结束：把仍处于驼峰中的事件补发出来。"""
        out = []
        for name, thresh in (('bounce', self.bounce_thresh), ('net', self.net_thresh)):
            h = self._hump[name]
            if h['active'] and h['peak'] >= thresh:
                if self._interval_ok(name, h['peak_ts']):
                    etype = EventType.BOUNCE if name == 'bounce' else EventType.NET_HIT
                    self._last_emit[name] = h['peak_ts']
                    out.append(Event(
                        event_id=self._new_id(etype),
                        timestamp_s=h['peak_ts'],
                        type=etype,
                        side=Side.UNKNOWN,
                        position_px=[int(p) for p in h['peak_pt']] if h['peak_pt'] else None,
                        confidence=h['peak'],
                        sources=['ttnet'],
                    ))
                h['active'] = False
        return out

    @property
    def stats(self):
        return dict(self._rejected)
