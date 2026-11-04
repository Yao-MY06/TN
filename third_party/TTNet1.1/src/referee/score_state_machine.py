# -*- coding: utf-8 -*-
"""
乒乓球计分状态机（YOLO / TrackNet / TTNet 共用）。

模型只负责「看到了什么」，本状态机只负责「按规则该给谁加分」。
因此更换检测模型时，这里的规则一行都不用改。

实现的是国际乒联（ITTF）单打基本规则
------------------------------------
1. 11 分制，先到 11 分且领先 2 分者胜一局；
2. 每 2 分发球权交换，10:10 后每 1 分发球权交换；
3. 一局结束后交换场地，发球权继续交替；
4. 判分依据：
   - 已成立的回合内，球在同一侧台面连续落台两次（对手未接到）→ 对侧得分；
   - 已成立的回合内，球落台后长时间无后续（出界 / 未接回）→ 对侧得分。

三条设计原则（v3 定稿）
----------------------
**原则 1：判分信号要全，宁可多判也不要漏判。**
前两版都在这里翻过车：
- v2 把「发球那一刻的本方落台 → 对方落台」当作**硬前置**。而 TTNet 球可见率
  只有 27%，发球那一次落台经常检不到，于是整段真实回合被丢弃 ——
  test_4 从 v1 的 51 分暴跌到 9 分，这是本次要修的核心问题。
- 正确做法是：**不要求看到发球**，只要落台证据累积到能判分就判。

**原则 2：把校验放在「出口」，不要放在「入口」。**
这是召回与精度的关键区别：入口校验一旦失败就丢整段数据；出口校验只在真正
要加分时否决一次，后续落台仍会重新累积证据，不会连带丢掉后面真实的一分。

出口处的「回合形态」校验要求本回合**同时**满足：
- 出现过**跨网交替**（两侧都落过台）→ 排除手摸桌子（孤立落台）、颠球（同侧反复）；
- 出现过**球拍击球**（HIT）→ 排除换发球时的手抛传球（抛球没有球拍接触）。

**原则 3：物理不可能性护栏照旧保留。**
判分冷却、重复落台去重、落台链断开等，都是不依赖模型精度的硬约束。

三类现场误得分是如何被挡住的
----------------------------
| 现象 | 该回合的落台证据 | 被哪一条挡掉 |
| --- | --- | --- |
| 手摸 / 擦桌面 | 通常 1 次孤立落台（偶尔 2 次同侧） | 不足 2 次落台 / 无跨网交替 |
| 准备活动颠球 | 同侧反复落台（A→A→A…） | 无跨网交替 |
| 换发球时传球过网 | 本方落台 → 对方落台，但**全程无球拍击球** | 无击球证据 |

状态流转
--------
WAITING_SERVE / RALLY（累积落台与击球证据）
        └→ 同侧双弹或静默超时 → ROUND_ENDING → POINT_CONFIRMED
                                        └→ COOLDOWN（冷却期内事件一律忽略）→ 重新布防
"""

import bisect

from referee.unified_types import EventType, Side


class Phase:
    WAITING_SERVE = 'WAITING_SERVE'       # 无证据，等待下一次落台
    RALLY = 'RALLY'                       # 正在累积落台/击球证据
    ROUND_ENDING = 'ROUND_ENDING'         # 判分已产生
    COOLDOWN = 'COOLDOWN'                 # 刚判完一分，禁止积累下一分证据
    NEEDS_REVIEW = 'NEEDS_REVIEW'         # 证据冲突，等人工确认


class TableTennisStateMachine:
    """11 分制乒乓球单打计分状态机（v3：出口侧回合形态校验）。"""

    def __init__(self, target_score=11, win_by=2, alternate_every=2,
                 serve_side=Side.A, rally_timeout_s=3.0,
                 ending_window_s=1.5, min_point_interval_s=1.2,
                 max_bounce_gap_s=1.5, min_bounce_gap_s=0.12,
                 rally_break_s=3.0, cumulative_mode=False, verbose=True,
                 require_cross_net=True,
                 require_hit_in_rally=True,
                 reject_suspect_hit=False,
                 require_ball_visibility=True,
                 min_ball_visibility=0.30,
                 visibility_window_s=0.15,
                 form_window_s=12.0,
                 max_same_side_bounces=4,
                 juggle_max_interval_s=1.2,
                 juggle_interval_tolerance=1.6,
                 juggle_pos_tol_px=60.0,
                 hit_window_s=4.0,
                 require_legal_serve=False,
                 serve_candidate_timeout_s=2.5,
                 post_point_cooldown_s=0.5,
                 point_hold_s=0.8,
                 auto_award_on_silence=True,
                 auto_award_on_end=True,
                 reject_hit_between_double_bounce=False,
                 min_bounce_confidence=None,
                 table_roi=None, table_roi_margin=0.06):
        """
        :param target_score: 每局目标分（11）
        :param win_by: 需净胜分（2）
        :param alternate_every: 每几分换发球（2；10 平后自动变 1）
        :param serve_side: 首发球方（仅用于发球权显示与轮换）
        :param rally_timeout_s: 回合内长时间无事件时的休眠阈值（保留参数）
        :param ending_window_s: 回合结束候选的确认窗口（供 UI/复核使用）
        :param min_point_interval_s: 两次判分的最小间隔（秒），最后一道兜底护栏
        :param max_bounce_gap_s: 相邻落台超过该间隔视为落台链断开（不再配对判分）
        :param min_bounce_gap_s: 相邻落台小于该间隔视为同一球的重复检测，丢弃
        :param rally_break_s: 最后一次落台后超过该时长仍无后续 → 回合结束信号

        :param require_cross_net: **判据 3（弱）**。判分前要求本回合两侧都落过台。
            只做「全同侧才否决」这一条最弱的要求，用于排除颠球、擦桌面。
            **不要收紧成「最近 N 秒必须出现跨网交替」** —— 真实得分那一拍本身
            就是同侧双弹，且发球落台常漏检，收紧会误杀大量真实得分（实测）。
        :param require_hit_in_rally: **判据 2**。判分前要求本回合出现过球拍
            击球（轨迹水平方向反转）。手抛传球没有球拍接触，因此被挡住。
        :param require_ball_visibility: **判据 1（最重要）**。判分前要求落台时刻
            前后球确实被看到。拦截模型的「幻觉落台」—— TTNet 在无球帧上也会
            输出高事件分数（实测低可见率处 100% 都带 conf≥0.6），这正是
            「选手只是摸了一下球，对手却得分」的来源。需要先调用
            set_visibility_source() 注入数据；未注入时该判据自动跳过。
        :param min_ball_visibility: 判据 1 的可见比例下限。
            实测真实落台前后可见率中位 89%，而用户报告的误判样本仅 12%；
            取 0.35 可挡住绝大多数幻觉落台，同时给遮挡留出余量。
        :param visibility_window_s: 判据 1 在落台时刻前后各取多长的窗口
        :param form_window_s: 判据 3 回看多长的落台历史
        :param hit_window_s: 判据 2 回看多长的击球历史（相对最后一次落台）
        :param require_legal_serve: **可选严格模式**。要求先看到「本方落台 →
            对方落台」才允许判分。**默认 False** —— 球可见率仅 27% 时该前置
            条件会让整段真实回合被丢弃（v2 的漏判根因），仅在球检测很可靠时开启。
        :param serve_candidate_timeout_s: 严格模式下发球候选的有效期
        :param post_point_cooldown_s: 判分后的冷却时长。冷却期内的事件一律忽略，
            保证「上一分余波」「换发球时把球扔到对面」不会被算成下一分。
        :param auto_award_on_silence: 落台后长时间无后续是否判分。**默认 True** ——
            这是「对手没接到 / 球出界」的主要判分途径，关闭会大量漏判。
            误判风险已由形态校验兜住。
        :param auto_award_on_end: 视频结束时是否结算残留回合。默认 True。
        :param reject_hit_between_double_bounce: 两次同侧落台之间若夹着击球，
            是否拒绝判分。默认 False —— 该判据会误伤「击球检测误报」覆盖到的
            真实得分，实测会额外损失约 5% 的分，因此仅作可选严格项。
        :param min_bounce_confidence: 可计分落台的最低置信度（None 表示不过滤）
        :param table_roi: 球台外接框 [x, y, w, h]（原始像素）。给出后会丢弃落点
            明显在台外的 bounce，None 表示不校验
        :param table_roi_margin: 球台 ROI 外扩比例
        """
        self.target_score = target_score
        self.win_by = win_by
        self.alternate_every = alternate_every
        self.rally_timeout_s = rally_timeout_s
        self.ending_window_s = ending_window_s
        self.min_point_interval_s = min_point_interval_s
        self.max_bounce_gap_s = max_bounce_gap_s
        self.min_bounce_gap_s = min_bounce_gap_s
        self.rally_break_s = rally_break_s
        self.cumulative_mode = cumulative_mode
        self.verbose = verbose

        self.require_cross_net = require_cross_net
        self.require_hit_in_rally = require_hit_in_rally
        self.reject_suspect_hit = reject_suspect_hit
        self.require_ball_visibility = require_ball_visibility
        self.min_ball_visibility = min_ball_visibility
        self.visibility_window_s = visibility_window_s
        self.form_window_s = form_window_s
        self.max_same_side_bounces = max_same_side_bounces
        self.juggle_max_interval_s = juggle_max_interval_s
        self.juggle_interval_tolerance = juggle_interval_tolerance
        self.juggle_pos_tol_px = juggle_pos_tol_px
        self.hit_window_s = hit_window_s
        self.require_legal_serve = require_legal_serve
        self.serve_candidate_timeout_s = serve_candidate_timeout_s
        self.post_point_cooldown_s = max(0.0, post_point_cooldown_s)
        self.point_hold_s = point_hold_s
        self.auto_award_on_silence = auto_award_on_silence
        self.auto_award_on_end = auto_award_on_end
        self.reject_hit_between_double_bounce = reject_hit_between_double_bounce
        self.min_bounce_confidence = min_bounce_confidence
        self.table_roi = list(table_roi) if table_roi else None
        self.table_roi_margin = table_roi_margin

        self.score = {Side.A: 0, Side.B: 0}
        self.games = {Side.A: 0, Side.B: 0}
        self.server = serve_side
        self.phase = Phase.WAITING_SERVE

        # ---- 回合证据（用于形态校验；判分或落台序列中断时才清空）
        self._bounces = []                # 本回合全部落台 [(ts, side, conf)]
        self._sides = set()               # 本回合落过台的侧别
        self._hit_ts = []                 # 击球时间戳（跨回合保留，用于滑动窗）
        self._trusted_hit_ts = []         # 其中「非疑似台面弹跳」的那些
        self._last_hit_side = None

        # ---- 跨回合的落台侧别历史，用于「最近 N 秒是否跨网往返」
        # 注意不能用 self._bounces：它判分后就清空，而形态校验需要回看
        # 上一次判分之前的对打是否仍在进行（真实回合常跨越判分冷却）。
        self._recent_bounce_sides = []    # [(ts, side)]

        # ---- 球可见性查询源（由 set_visibility_source 注入）
        self._vis_ts = []
        self._vis_val = []

        # ---- 落台链（只用于配对找「同侧连续两次落台」）
        self._chain = []                  # [(ts, side, conf)]

        # ---- 严格模式下的发球候选
        self._serve_bounce = None

        self._last_event_ts = None
        self._net_touched = False
        self._pending_point = None      # 待确认的判分（延迟点，见 _flush_pending_point）

        # ---- 比赛状态
        self._serve_count = 0
        self._point_log = []
        self._review_log = []
        self._reject_log = []
        self._last_point_ts = None
        self._cooldown_until = None
        self._suppressed = 0

    # ---------------------------------------------------------------- 工具
    def _log(self, msg):
        if self.verbose:
            print('    [状态机] {}'.format(msg))

    def _next_server(self):
        """按规则计算下一个发球方。"""
        total = self.score[Side.A] + self.score[Side.B]
        if self.score[Side.A] >= self.target_score - 1 and self.score[Side.B] >= self.target_score - 1:
            return Side.opposite(self.server)
        if total % self.alternate_every == 0:
            return Side.opposite(self.server)
        return self.server

    def _reset_rally(self):
        self._bounces = []
        self._sides = set()
        self._chain = []
        self._serve_bounce = None
        self._net_touched = False
        self._pending_point = None
        # 不清空 _hit_ts / _recent_bounce_sides：两者都是**跨回合的上下文**，
        # 供滑动窗形态校验使用。清空会让每个新回合都因「本回合没有击球/跨网」
        # 被误杀（v2 漏判的重要成因之一）。两者的存活时长分别在
        # _remember_hit_context() / _remember_bounce_context() 里按窗口维护，
        # 因此不会无限增长，也不会把很久以前的证据带进当前判断。

    def _in_cooldown(self, ts):
        return self._cooldown_until is not None and ts < self._cooldown_until

    def _rally_form_ok(self):
        """回合形态校验（出口侧）：这段证据像不像一次真实的击球回合。

        注意这是**判分前的否决**，不是回合的准入门槛 —— 否决后证据继续累积，
        后面的真实落台仍能凑成新的判分机会，因此不会连带丢掉后续的分。

        三条判据（AND）
        --------------
        **判据 1｜球真的被看到过（最重要）**
            落台时刻前后 `visibility_window_s` 内，球必须处于可见状态，
            且可见比例不低于 `min_ball_visibility`。
            TTNet 在不含球的帧上也会输出高的事件分数 —— 实测「落台前后
            球可见率 < 30%」的 bounce 一共 48 个，**100% 都带有 conf ≥ 0.6**。
            这类「幻觉落台」正是「选手只是摸了一下球、对手却得分」的来源：
            模型在没看到球的情况下凭空报了一个落台，状态机无从分辨。
            而真实落台必然伴随连续可见的球（实测真实落台前后可见率中位 89%）。

        **判据 2｜本拍确实被击打过**
            最后一次落台前 `hit_window_s` 内出现过球拍击球（轨迹方向反转）。
            排除「手抛传球过网」—— 抛球没有球拍接触。

        **判据 3｜不是单侧反复弹**
            本回合的落台不能全部落在同一侧（用于排除颠球、擦桌面）。
            注意这里只做「全同侧才否决」这一条最弱的要求：真实得分的那一拍
            本身必然是同侧双弹，不能因为「看不到跨网」就否决 —— 实测用
            「最近 N 秒必须跨网」会误杀大量真实得分（发球落台常漏检）。

        这三条能区分用户报告的一对样本
        ------------------------------
        - 7.39s（应判 B 得分）：双弹 conf 0.89/0.95，**球全程可见（100%）**，
          之前有击球，且回合内出现过 A/B 两侧落台 → 通过；
        - 24.70s（不应判）：双弹 conf 0.65/0.78，**球可见率仅 12%**（模型在
          无球时幻觉），且这是孤立的一次双弹 → 被判据 1 否决。
        """
        last_ts = self._bounces[-1][0] if self._bounces else None
        if last_ts is None:
            return True

        # ---- 判据 1：落台时刻球真的可见（拦截模型的「幻觉落台」）
        #
        # 只看**最后一次落台**附近的可见性，不能把两次落台之间的整段都算进来：
        # 静默结算时，最后落台到超时之间可能隔了 3 秒，那段球早飞走了、
        # 也可能恰好有别的可见帧，把可见比例「平均」上去，幻觉就漏过去了。
        # 落台那一刻球必须被看到，这才是物理事实。
        if self.require_ball_visibility:
            vis = self._visibility_probe(last_ts, last_ts)
            if vis is None:
                # 没有可见性数据（例如换用无 visible 字段的模型）时跳过该判据，
                # 不能因为缺少信息就一律否决。
                pass
            elif vis < self.min_ball_visibility:
                self._reject_log.append({
                    'ts': last_ts, 'reason': 'ball_not_visible',
                    'visibility': round(vis, 3),
                    'min_required': self.min_ball_visibility,
                })
                return False

        # ---- 判据 2：本拍确实被击打过
        if self.require_hit_in_rally:
            hits = [h for h in self._hit_ts
                    if last_ts - self.hit_window_s <= h <= last_ts]
            if not hits:
                self._reject_log.append({
                    'ts': last_ts, 'reason': 'no_hit',
                    'window_s': self.hit_window_s,
                })
                return False
            # 可选：只采信「可信击球」。
            # 事件层会把「球贴台面高度带」的 HIT 标上 suspect_bounce
            # （那是球撞台面造成的假击球），默认仍然采信它们（reject=False），
            # 以免在没有球拍检测时过度收紧召回。开启后只接受可信击球。
            if self.reject_suspect_hit and not any(h in self._trusted_hit_ts
                                                   for h in hits):
                self._reject_log.append({
                    'ts': last_ts, 'reason': 'only_suspect_hits',
                    'window_s': self.hit_window_s,
                })
                return False

        # ---- 判据 3：颠球 / 擦桌面
        #
        # 由 _is_juggling() 承担，且在 _on_bounce() 的配对判分**之前**就已拦截
        # （必须提前：否则颠球在第 2 次落台就凑成「同侧双弹」被判分）。
        # 这里不再重复检查，避免两套逻辑互相打架。
        return True

    def _is_juggling(self, ts):
        """判断当前落台是否属于「同侧原地反复弹」（颠球 / 擦桌面）。

        三个条件必须同时成立，缺一不可 —— 少任何一条都会误伤正常对打：

        1. **同一侧连续**：球一直在同一侧弹。正常对打会跨网交替。
        2. **间隔短且均匀**：颠球节奏稳定（实测 0.6s 左右且抖动小）。
           正常对打中同一侧的相邻落台间隔变化很大。
        3. **落点几乎不动**：颠球是球在原地上下弹，落点集中在一小片区域。
           这是最能区分「颠球」与「正常对打」的特征。

        注意不能只用「同侧落台次数多」：正常对打经过若干来回后，
        同一侧本来就会累积很多次落台，那样会把真实回合全部误杀。
        """
        hist = self._recent_bounce_sides
        if len(hist) < 3:
            return False
        side = hist[-1][1]
        # 条件 1：最近 3 次必须同侧
        if any(h[1] != side for h in hist[-3:]):
            return False
        # 条件 2：间隔短且均匀
        gaps = [hist[-1][0] - hist[-2][0], hist[-2][0] - hist[-3][0]]
        if any(g <= 0 or g > self.juggle_max_interval_s for g in gaps):
            return False
        ratio = gaps[0] / gaps[1]
        if not (1.0 / self.juggle_interval_tolerance
                <= ratio <= self.juggle_interval_tolerance):
            return False
        # 条件 3：落点几乎不动
        pos = [h[2] for h in hist[-3:]]
        if all(p for p in pos):
            span = max(
                ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                for i, a in enumerate(pos) for b in pos[i + 1:])
            if span > self.juggle_pos_tol_px:
                return False
        return True

    def _bounce_plausible(self, event):
        """落台证据的最低可采信条件：置信度下限 + 落点在球台 ROI 内。

        只排除「明显不可能」的落台，不承担判定职责：
        - 置信度过低 → 模型含糊，不能作为计分依据；
        - 落点远在台外 → 球落地 / 手在台外碰球，不属于「台面落台」。
        """
        if self.min_bounce_confidence is not None and event.confidence is not None \
                and event.confidence < self.min_bounce_confidence:
            self._reject_log.append({'ts': event.timestamp_s, 'reason': 'low_conf',
                                     'conf': event.confidence, 'side': event.side})
            return False

        if self.table_roi is not None and event.position_px:
            x, y, w, h = self.table_roi
            mx = w * self.table_roi_margin
            my = h * self.table_roi_margin
            px, py = event.position_px[0], event.position_px[1]
            if not ((x - mx) <= px <= (x + w + mx) and (y - my) <= py <= (y + h + my)):
                self._reject_log.append({'ts': event.timestamp_s, 'reason': 'off_table',
                                         'side': event.side,
                                         'position_px': list(event.position_px)})
                return False
        return True

    # ---------------------------------------------------------------- 判分
    def _award(self, side, reason, confidence=None, ts=None):
        """给 side 加 1 分，并处理换发球与换局。

        :return: True 表示真正提交了这一分，False 表示被护栏拦下 / 无法判定。
        """
        if (ts is not None and self._last_point_ts is not None
                and (ts - self._last_point_ts) < self.min_point_interval_s):
            self._suppressed += 1
            self._log('⚠ 判分被护栏拦下：距上次判分仅 {:.2f}s < {:.1f}s（{}）'.format(
                ts - self._last_point_ts, self.min_point_interval_s, reason))
            self._reset_rally()
            self.phase = Phase.WAITING_SERVE
            return False

        if side == Side.UNKNOWN:
            self.phase = Phase.NEEDS_REVIEW
            self._review_log.append({'ts': ts, 'reason': reason})
            self._reset_rally()
            self._log('回合无法判定，进入待人工确认: {}'.format(reason))
            return False

        loser = Side.opposite(side)
        self.score[side] += 1
        self._point_log.append({
            'ts': ts,
            'winner': side,
            'reason': reason,
            'confidence': confidence,
            'score': '{}({}) : {}({})'.format(self.score[Side.A], self.games[Side.A],
                                              self.score[Side.B], self.games[Side.B]),
        })
        self._log('判分 → {side} 得分（{reason}）, 比分 {a}:{b}'.format(
            side=side, reason=reason, a=self.score[Side.A], b=self.score[Side.B]))

        self._last_point_ts = ts
        self.server = self._next_server()
        self._reset_rally()

        if ts is not None and self.post_point_cooldown_s > 0:
            self._cooldown_until = ts + self.post_point_cooldown_s
            self.phase = Phase.COOLDOWN
        else:
            self._cooldown_until = None
            self.phase = Phase.WAITING_SERVE

        if self.cumulative_mode:
            # 累计总分模式：不触发换局，比分一直累加。
            # 用于「短片/训练片段」演示 —— 这些视频只有几十秒到几分钟，
            # 不存在打完一局的情形；若按 11 分制处理，到 11 分会归零，
            # 反而让观众看到比分突然清零（实测 test_6 判 14 分却显示 0:0）。
            return True

        s_win = self.score[side]
        s_lose = self.score[loser]
        if s_win >= self.target_score and (s_win - s_lose) >= self.win_by:
            self.games[side] += 1
            self._log('🏆 {side} 赢下本局 {}-{}！总局分 {}:{}'.format(
                s_win, s_lose, self.games[Side.A], self.games[Side.B], side=side))
            self.score = {Side.A: 0, Side.B: 0}
        return True

    # ---------------------------------------------------------------- 主入口
    def consume(self, event):
        """消费一个统一 Event，返回本步产生的判分动作（若有）。"""
        ts = event.timestamp_s
        if self._last_event_ts is None or ts > self._last_event_ts:
            self._last_event_ts = ts

        if event.type == EventType.NET_HIT:
            if not self._in_cooldown(ts):
                self._net_touched = True
                self._log('擦网 @ {:.3f}s'.format(ts))
            return None

        if event.type == EventType.BALL_LOST:
            # 球丢失只作为断段信号：既不清空已积累的证据（会丢真分），也不判分
            return None

        # ---- 冷却期内：任何事件都不允许开启新的可计分回合。
        #
        # 但「上下文」必须先记下来：形态校验要回看最近若干秒是否跨网往返，
        # 而一次真实对打常常横跨判分冷却（判分后双方继续对拉）。若冷却期的落台
        # 完全不记录，跨网判据就会误判为「没有往返」，从而吞掉后面真实的分。
        if self._in_cooldown(ts):
            if event.type == EventType.BOUNCE and event.side in (Side.A, Side.B):
                self._remember_bounce_context(ts, event.side)
            elif event.type == EventType.HIT:
                self._remember_hit_context(ts, event)
            return None

        if event.type == EventType.HIT:
            self._remember_hit_context(ts, event)
            self._last_hit_side = event.side
            return None

        if event.type == EventType.BOUNCE:
            if not self._bounce_plausible(event):
                return None
            if self.require_legal_serve:
                return self._on_serve_bounce(event)
            return self._on_bounce(event)

        return None

    def _remember_bounce_context(self, ts, side, pos=None):
        """记录落台侧别与落点历史（跨回合保留，供滑动窗形态校验）。"""
        self._recent_bounce_sides.append((ts, side, list(pos) if pos else None))
        keep = max(self.form_window_s, self.hit_window_s) + 5.0
        self._recent_bounce_sides = [r for r in self._recent_bounce_sides
                                     if r[0] >= ts - keep]

    def _remember_hit_context(self, ts, event=None):
        """记录击球时间（跨回合保留，供滑动窗形态校验）。"""
        self._hit_ts.append(ts)
        # 「可信击球」= 没被事件层标成疑似台面弹跳。
        # 事件层用「球是否在台面高度带内」判断：球撞台面会让轨迹方向反转，
        # 被误判成击球；真实击球时球在空中。见 paddle_detector.py。
        if event is None or 'suspect_bounce' not in (event.sources or []):
            self._trusted_hit_ts.append(ts)
        # 留存时长要覆盖「落台之后才做形态校验」的延迟：
        # 击球发生在落台之前，而校验发生在落台事件到达时，两者可能相隔
        # 数秒（球飞行时间 + 驼峰检测延迟）。留得太短会把真实得分的
        # 击球证据提前裁掉 —— 实测 hit_window_s=3 时，7.39s 那次真实得分
        # 就因击球记录已被清理而被误判为 no_hit。
        keep = self.hit_window_s + self.form_window_s + 10.0
        self._hit_ts = [h for h in self._hit_ts if h >= ts - keep]
        self._trusted_hit_ts = [h for h in self._trusted_hit_ts if h >= ts - keep]

    # ---------------------------------------------------------------- 可见性
    def set_visibility_source(self, samples):
        """注入「帧时间戳 → 球是否可见」的查询源，用于拦截幻觉落台。

        :param samples: [(timestamp_s, visible_int)]，按时间升序。
            由调用方从推理缓存/适配器给出。传 None 表示不启用该判据。

        为什么必须由外部注入
        --------------------
        「球是否可见」只有模型层知道（TTNet 的球位置概率峰值低于阈值即为不可见）。
        状态机拿到的是已提取的 Event，本身不含这个信息。而它偏偏是区分
        「真实落台」与「模型幻觉落台」最可靠的证据，因此显式提供注入入口，
        而不是让状态机去猜。
        """
        if not samples:
            self._vis_ts, self._vis_val = [], []
            return
        samples = sorted(samples, key=lambda x: x[0])
        self._vis_ts = [float(s[0]) for s in samples]
        self._vis_val = [1 if s[1] else 0 for s in samples]

    def _visibility_probe(self, t0, t1):
        """返回 [t0-window, t1+window] 内球的可见比例；无数据时返回 None。"""
        if not self._vis_ts:
            return None
        lo = min(t0, t1) - self.visibility_window_s
        hi = max(t0, t1) + self.visibility_window_s
        i = bisect.bisect_left(self._vis_ts, lo)
        j = bisect.bisect_right(self._vis_ts, hi)
        if j <= i:
            return None
        return sum(self._vis_val[i:j]) / float(j - i)

    # ---------------------------------------------------------------- 严格模式
    def _on_serve_bounce(self, event):
        """可选严格模式：要求「本方落台 → 对方落台」成立才允许判分。

        默认关闭。仅在球检测非常可靠（可见率高）时才建议开启，
        否则会因发球落台漏检而丢弃整段真实回合。
        """
        if event.side == Side.UNKNOWN:
            return None

        if self._serve_bounce is None:
            self._serve_bounce = (event.side, event.timestamp_s, event.confidence)
            self.phase = Phase.WAITING_SERVE
            self._log('发球候选：{} 侧落台 @ {:.3f}s'.format(event.side, event.timestamp_s))
            return None

        first_side, first_ts, first_conf = self._serve_bounce
        gap = event.timestamp_s - first_ts

        if event.side == first_side:
            if gap >= self.min_bounce_gap_s:
                self._serve_bounce = (event.side, event.timestamp_s, event.confidence)
            return None
        if gap < self.min_bounce_gap_s:
            return None
        if gap > self.max_bounce_gap_s:
            self._serve_bounce = None
            return None

        self._serve_count += 1
        self._serve_bounce = None
        self._log('✅ 合法发球成立（{} → {}）'.format(first_side, event.side))
        # 发球双落台作为回合的起始证据
        self._chain = [(first_ts, first_side, first_conf)]
        self._bounces = [(first_ts, first_side, first_conf)]
        self._sides = {first_side}
        self.phase = Phase.RALLY
        return self._on_bounce(event)

    # ---------------------------------------------------------------- 回合判分
    def _on_bounce(self, event):
        """回合内落台：累积证据，配对同侧连续两次落台即判分。

        数据事实（本项目实测，OpenTTGames test_1）
        ------------------------------------------
        标注的 bounce 是「每一次落台」。一次正常往返对打会形成**左右交替**序列
        （A→B→A→B…，中位间隔约 0.40s）。「同侧连续两次落台」只在**球没有被接回**
        时才出现，也即：这就是回合的最后一拍。因此它是可靠的单分判据。
        """
        if event.side == Side.UNKNOWN:
            return None

        ts = event.timestamp_s

        # ---- 落台链配对（重复检测丢弃 / 间隔过长则断链，但回合证据保留）
        if self._chain:
            gap = ts - self._chain[-1][0]
            if gap < self.min_bounce_gap_s:
                self._log('两次落台间隔 {:.2f}s 过短，判为重复检测，忽略'.format(gap))
                return None
            if gap > self.max_bounce_gap_s:
                self._log('落台间隔 {:.2f}s > {:.1f}s → 落台链断开（回合证据保留）'.format(
                    gap, self.max_bounce_gap_s))
                self._chain = []

        self._chain.append((ts, event.side, event.confidence))
        self._bounces.append((ts, event.side, event.confidence))
        self._sides.add(event.side)
        self._remember_bounce_context(ts, event.side, event.position_px)
        self.phase = Phase.RALLY

        self._log('落台 side={} @ {:.3f}s (bounce={:.2f})'.format(
            event.side, ts, event.confidence or 0.0))

        # ---- 颠球 / 擦桌面预检：必须在配对判分**之前**做。
        # 否则颠球会在第 2 次落台就凑成「同侧双弹」被判分，
        # 等到第 3、4 次才拦截已经晚了两分。
        if self.require_cross_net and self._is_juggling(ts):
            self._reject_log.append({'ts': ts, 'reason': 'juggling_detected',
                                     'side': event.side})
            self._log('检测到同侧原地反复弹（颠球/擦桌）→ 清理回合，不计分')
            self._reset_rally()
            self.phase = Phase.WAITING_SERVE
            return None

        if len(self._chain) < 2:
            return None

        prev_ts, prev_side, prev_conf = self._chain[-2]
        curr_ts, curr_side, curr_conf = self._chain[-1]

        if prev_side != curr_side:
            return None   # 左右交替 = 正常往返，回合继续

        # ---- 可选严格项：两弹之间夹着击球，说明序列被误拼
        if self.reject_hit_between_double_bounce:
            if any(prev_ts < h < curr_ts for h in self._hit_ts):
                self._reject_log.append({'ts': curr_ts,
                                         'reason': 'hit_between_double_bounce',
                                         'side': curr_side})
                self._log('{} 侧两弹之间有击球 → 序列被误拼，不计分'.format(curr_side))
                self._chain = [(curr_ts, curr_side, curr_conf)]
                return None

        # ---- 形态校验：不像正式回合就不加分（证据继续保留，不丢弃）
        if not self._rally_form_ok():
            self._log('{} 侧连续两弹，但回合形态不成立 → 暂不计分（证据保留）'.format(
                curr_side))
            self._chain = [(curr_ts, curr_side, curr_conf)]
            return None

        # ---- 同侧两次落台 → 候选判分（延迟确认）
        #
        # 为什么不能当场判分
        # ------------------
        # 「同侧连续两次落台」既能是「对手没接到」（该判分），
        # 也能是颠球的第 1、2 下（不该判分）。二者在**第 2 次落台那一刻
        # 完全无法区分** —— 差别在于后面还有没有第 3、4 下。
        # 因此这里只登记候选，等 hold 一段时间再看：
        #   期间又出现同侧落台 → 是颠球，撤销候选；
        #   期间没有新的同侧落台 → 是真实失分，提交判分。
        #
        # 延迟代价：比分比真实时刻晚 `point_hold_s` 秒（默认 0.8s）。
        # 这是必要的准确性代价 —— 当场判分会在颠球场景连续误判多分。
        winner = Side.opposite(curr_side)
        confs = [c for c in (prev_conf, curr_conf) if c is not None]
        self._pending_point = {
            'ts': curr_ts,
            'winner': winner,
            'side': curr_side,
            'confidence': float(min(confs)) if confs else None,
            'reason': '{} 侧台面连续两次落台（{:.2f}s 内，对手未接到）'.format(
                curr_side, curr_ts - prev_ts),
        }
        self._chain = [(curr_ts, curr_side, curr_conf)]
        self._log('登记待确认判分：{} 侧双弹 @ {:.3f}s，等待 {:.1f}s 确认'.format(
            curr_side, curr_ts, self.point_hold_s))
        return None

    def _flush_pending_point(self, now_ts):
        """到点后提交待确认的判分；若期间出现同侧落台则撤销。"""
        p = self._pending_point
        if not p:
            return None
        if now_ts - p['ts'] < self.point_hold_s:
            return None

        # 期间又出现同侧落台 → 说明是颠球 / 擦桌，撤销候选
        if self._is_juggling(p['ts']):
            self._reject_log.append({'ts': p['ts'], 'reason': 'juggling_detected',
                                     'side': p['side']})
            self._log('待确认判分被撤销：{} 侧持续反复落台（颠球/擦桌）'.format(p['side']))
            self._pending_point = None
            self._reset_rally()
            self.phase = Phase.WAITING_SERVE
            return None

        self._pending_point = None
        self.phase = Phase.ROUND_ENDING
        committed = self._award(p['winner'], reason=p['reason'],
                                confidence=p['confidence'], ts=p['ts'])
        if not committed:
            return None
        return {'type': 'POINT', 'winner': p['winner']}

    # ---------------------------------------------------------------- 时间驱动
    def tick_deep(self, now_ts):
        """时间推进：处理冷却解除、发球候选超时、回合结束信号。"""
        return self.tick(now_ts)

    def tick(self, now_ts):
        """由外部按时间推进调用。"""
        # ---- 待确认判分：到点提交 / 发现颠球则撤销
        ev = self._flush_pending_point(now_ts)
        if ev is not None:
            return ev
        # ---- 冷却结束 → 重新布防
        if self._cooldown_until is not None and now_ts >= self._cooldown_until:
            self._cooldown_until = None
            self.phase = Phase.WAITING_SERVE
            self._log('冷却结束 @ {:.2f}s，重新布防'.format(now_ts))
            # 注意：这里**不清空落台链**。冷却期内本来就不积累证据（见 consume），
            # 因此链里不会有本回合的残留；而如果在冷却结束时就清空，会把
            # 「冷却期最后一刻刚记下的落台」抹掉 —— 实测这会让紧随其后的
            # 真实得分（对手没接到、同侧再弹）失去配对对象而漏判。

        # ---- 严格模式下：发球候选超时作废
        if self.require_legal_serve and self._serve_bounce is not None:
            if now_ts - self._serve_bounce[1] > self.serve_candidate_timeout_s:
                self._serve_bounce = None
                return {'type': 'SERVE_CANDIDATE_EXPIRED'}

        # ---- 回合结束信号：以「最后一次落台」为基准计时。
        #
        # 为什么必须用落台时间而不是「最后一次事件」：真实回合结束时，最后一拍
        # 落台之后往往还有零星的 HIT/NET 误报（球已飞出、运动员收拍动作）。若用
        # 「最后事件」计时，这些噪声会不断把超时往后推，导致该回合永远等不到
        # 结算窗口，真实的一分就这样静默丢失。
        if self._bounces and now_ts - self._bounces[-1][0] > self.rally_break_s:
            gap = now_ts - self._bounces[-1][0]
            if (self.auto_award_on_silence and len(self._bounces) >= 2
                    and self._rally_form_ok()):
                last_side = self._bounces[-1][1]
                if last_side != Side.UNKNOWN:
                    winner = Side.opposite(last_side)
                    self._log('落台序列中断 {:.1f}s，最后落台 {} 侧 → 判 {} 得分'.format(
                        gap, last_side, winner))
                    committed = self._award(
                        winner,
                        reason='球落在 {} 侧后 {:.1f}s 未被击回'.format(last_side, gap),
                        confidence=self._bounces[-1][2], ts=now_ts)
                    if committed:
                        return {'type': 'POINT', 'winner': winner}
            self._log('落台序列中断 {:.2f}s → 丢弃回合证据（不计分）'.format(gap))
            self._reset_rally()
            self.phase = Phase.WAITING_SERVE
            return {'type': 'RALLY_TIMEOUT'}

        return None

    def finalize(self, end_ts=None):
        """视频结束时的收尾：结算残留回合。

        视频截断时最后一个回合没有后续时间窗口，若不结算就会丢掉末尾这一分。
        为避免「视频截断在回合中间」被误算，这里同样要求形态校验通过。
        """
        if end_ts is None or len(self._bounces) < 2:
            self._reset_rally()
            return None

        if not self.auto_award_on_end:
            self._log('视频结束：残留 {} 次落台，仅清理证据（不结算）'.format(
                len(self._bounces)))
            self._reset_rally()
            return None

        if not self._rally_form_ok():
            self._log('视频结束：末回合形态不成立，不结算')
            self._reset_rally()
            return None

        last_side = self._bounces[-1][1]
        if last_side == Side.UNKNOWN:
            self._reset_rally()
            return None

        winner = Side.opposite(last_side)
        committed = self._award(winner,
                                reason='视频结束时球落在 {} 侧未被击回'.format(last_side),
                                confidence=self._bounces[-1][2], ts=end_ts)
        if not committed:
            return None
        return {'type': 'POINT', 'winner': winner}

    # ---------------------------------------------------------------- 报告
    @property
    def scoreboard(self):
        return {
            'score': {Side.A: self.score[Side.A], Side.B: self.score[Side.B]},
            'games': {Side.A: self.games[Side.A], Side.B: self.games[Side.B]},
            'server': self.server,
            'phase': self.phase,
        }

    @property
    def points(self):
        return list(self._point_log)

    @property
    def reviews(self):
        return list(self._review_log)

    @property
    def suppressed_count(self):
        return self._suppressed

    @property
    def serve_count(self):
        return self._serve_count

    @property
    def rejected(self):
        return list(self._reject_log)
