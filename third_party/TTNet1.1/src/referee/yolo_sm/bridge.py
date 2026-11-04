# -*- coding: utf-8 -*-
"""桥接层：把 TTNet 的事件流喂进 YOLO 项目的共用计分状态机。

为什么需要桥接
--------------
YOLO 状态机（:mod:`referee.yolo_sm.state_machine`）的输入是
:class:`referee.yolo_sm.schemas.RefereeEvent`，事件类型只有 4 种：
``BOUNCE`` / ``RACKET_HIT`` / ``NET_TOUCH`` / ``OUT``。

TTNet 侧的事件是 :class:`referee.unified_types.Event`，类型有
``BOUNCE`` / ``NET_HIT`` / ``HIT`` / ``BALL_LOST``。

两边只差一层字段映射，计分规则完全不用动 —— 这正是方案文档
§1「三个模型只差适配层」的字面落实。

字段映射
--------
====================  ==========================  ====================================
TTNet                 YOLO RefereeEvent           说明
====================  ==========================  ====================================
``BOUNCE``            ``BOUNCE``                  side 原样透传（A/B/UNKNOWN）
``HIT``               ``RACKET_HIT``              轨迹方向反转推断的击球
``NET_HIT``           ``NET_TOUCH``               擦网
``BALL_LOST``         （不送）                     YOLO 状态机没有这个概念；回合结束
                                                  由 ``tick()`` 的时间判据负责
====================  ==========================  ====================================

``BALL_LOST`` 刻意不送：YOLO 状态机的时间判据（``rally_end_s``）本来就覆盖了
「球消失」这个情形，再塞一个 BALL_LOST 进去只会让时间线错乱。

``OUT`` 事件 TTNet 侧没有（TTNet 只回归 bounce / net 两个分数，没有出界输出），
因此 ``enable_out`` 规则在 TTNet 分支上永远不会触发 —— 这是模型能力差异，
不是实现缺失。TTNet 的出界由「落台链静默 + NO_RETURN」间接覆盖。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from referee.yolo_sm.schemas import EventType, RefereeEvent, Side
from referee.yolo_sm.state_machine import (
    Decision,
    DecisionKind,
    MatchRules,
    MatchState,
    TableTennisStateMachine,
)

# TTNet 事件类型 → YOLO 事件类型
_TTNET_TO_YOLO: Dict[str, str] = {
    'BOUNCE': EventType.BOUNCE.value,
    'HIT': EventType.RACKET_HIT.value,
    'NET_HIT': EventType.NET_TOUCH.value,
    # 幂等项：允许上游直接传入已经是 YOLO 命名的类型（例如 TrackNet 分支或
    # 二次转发的场景）。缺了这几条会让「已经是 RACKET_HIT 的事件」被静默丢弃，
    # 静默丢事件比报错危险得多 —— 比分会莫名其妙少判。
    EventType.BOUNCE.value: EventType.BOUNCE.value,
    EventType.RACKET_HIT.value: EventType.RACKET_HIT.value,
    EventType.NET_TOUCH.value: EventType.NET_TOUCH.value,
    EventType.OUT.value: EventType.OUT.value,
}

# 明确不支持的类型（有据可查地丢弃，而不是「不认识就扔」）
_DROPPED = {'BALL_LOST', 'SERVE'}

# 反向：用于 UI 文案
YOLO_EVENT_TEXT = {
    EventType.BOUNCE.value: '落台',
    EventType.RACKET_HIT.value: '击球',
    EventType.NET_TOUCH.value: '擦网',
    EventType.OUT.value: '出界',
}

_SIDE_OK = {Side.A.value, Side.B.value, Side.UNKNOWN.value}


def to_referee_event(raw: Any, seq: int) -> Optional[RefereeEvent]:
    """把一条 TTNet 事件（dict 或对象）转成 ``RefereeEvent``。

    :param raw: TTNet ``unified_types.Event`` 或它的 ``to_dict()`` 结果。
    :param seq: 该事件类型内的序号，用于生成稳定 event_id。
    :return: 转换成功返回事件；``BALL_LOST`` 等无法映射的类型返回 ``None``。

    时间戳、side 一律原样透传 —— 桥接层**不做任何猜测与修正**，
    「看到了什么」是上游的职责，「该给谁加分」才是状态机的职责。
    """
    d = raw if isinstance(raw, dict) else raw.to_dict()

    src_type = str(d.get('type', ''))
    yolo_type = _TTNET_TO_YOLO.get(src_type)
    if yolo_type is None:
        if src_type in _DROPPED:
            return None
        # 不认识的事件类型：既不能猜也不能静默扔掉，直接报错让人看见。
        raise ValueError(
            'unsupported event type {!r}; expected one of {} (dropped on purpose: {})'
            .format(src_type, sorted(_TTNET_TO_YOLO), sorted(_DROPPED)))

    side = str(d.get('side') or Side.UNKNOWN.value)
    if side not in _SIDE_OK:
        side = Side.UNKNOWN.value

    pos = d.get('position_px')
    if pos is not None:
        try:
            pos = (float(pos[0]), float(pos[1]))
        except (TypeError, IndexError, ValueError):
            pos = None

    conf = d.get('confidence')
    if conf is not None:
        conf = float(conf)
        # schemas 要求 confidence ∈ [0,1]；TTNet 的驼峰峰值理论上已在范围内，
        # 但保底 clamp 一次，避免一条越界数据把整条流水线打挂。
        conf = min(1.0, max(0.0, conf))

    sources = list(d.get('sources') or [])
    # schemas 只认 yolo / tracknet / ttnet，兼容上游可能塞进来的来源标签
    sources = [s for s in sources if s in ('yolo', 'tracknet', 'ttnet')] or ['ttnet']

    return RefereeEvent(
        event_id='{}_{:05d}'.format(yolo_type.lower(), seq),
        timestamp_s=float(d['timestamp_s']),
        type=yolo_type,
        side=side,
        position_px=pos,
        confidence=conf,
        sources=sources,
    )


def to_referee_events(raw_events: Iterable[Any]) -> List[RefereeEvent]:
    """批量转换，并按时间戳稳定排序（状态机要求时间单调递增）。"""
    counters: Dict[str, int] = {}
    out: List[RefereeEvent] = []
    for raw in raw_events:
        d = raw if isinstance(raw, dict) else raw.to_dict()
        src_type = str(d.get('type', ''))
        yolo_type = _TTNET_TO_YOLO.get(src_type)
        if yolo_type is None:
            if src_type in _DROPPED:
                continue
            raise ValueError(
                'unsupported event type {!r}; expected one of {} (dropped on purpose: {})'
                .format(src_type, sorted(_TTNET_TO_YOLO), sorted(_DROPPED)))
        counters[yolo_type] = counters.get(yolo_type, 0) + 1
        ev = to_referee_event(d, counters[yolo_type])
        if ev is not None:
            out.append(ev)
    # 同一时刻的多种事件要有确定顺序：先击球、再落台、后擦网
    order = {EventType.RACKET_HIT.value: 0, EventType.BOUNCE.value: 1,
             EventType.NET_TOUCH.value: 2, EventType.OUT.value: 3}
    out.sort(key=lambda e: (e.timestamp_s, order.get(e.type, 9)))
    return out


class YoloCoreScorer:
    """用 YOLO 项目的状态机给 TTNet 事件计分。

    与 TTNet 原状态机的接口差异
    --------------------------
    原状态机（``referee.score_state_machine``）是「喂事件 + tick」的流式接口，
    并且在 11 分时按累计模式处理。

    YOLO 状态机的接口同样是流式的（``update(ev)`` / ``tick(t)``），但它有一个
    **真正的局结束语义**：``game_over=True`` 之后 ``update()`` 一律返回空，
    后面的比赛不再计分。对一场完整的比赛这是对的；对 30~300 秒的测试片段，
    如果不处理就会看到「比分卡在 11:x 不动」。

    因此本类在检测到 GAME_OVER 时做**局间接力**：
    记录本局结果 → 用上一局胜者作为新局首发方新建状态机 → 继续计分。
    这样既保持 YOLO 状态机的规则原样，又让短片/连场比赛的比分能一直走下去。
    """

    def __init__(self, rules: Optional[MatchRules] = None, first_server: str = 'A',
                 verbose: bool = False, profile: str = 'ttnet'):
        """
        :param profile: 证据质量档位，见 :class:`referee.yolo_sm.ttnet_state_machine.Profile`。

            * ``'ttnet'``（默认）—— 用 :class:`TTNetStateMachine`，启用 TTNet 专属修正
              （击球不清空落台侧 / WAITING_SERVE 落台可进 RALLY / 放宽 NO_RETURN /
              抑制同回合重复判分）。这些修正只**追加**在父类之上。
            * ``'yolo'`` —— 直接用父类 :class:`TableTennisStateMachine`，
              行为与原 YOLO 分支逐条一致（有回归测试保证）。
        """
        if profile not in ('yolo', 'ttnet'):
            raise ValueError("profile must be 'yolo' or 'ttnet', got {!r}".format(profile))
        self.profile = profile
        if rules is None:
            if profile == 'ttnet':
                from referee.yolo_sm.ttnet_state_machine import TTNetRules
                rules = TTNetRules()
            else:
                rules = MatchRules()
        self._rules = rules
        self._first_server = first_server
        self.verbose = verbose

        # ---- 跨局累计账本（必须先于 _new_game 初始化）
        self.games: Dict[str, int] = {'A': 0, 'B': 0}
        # 跨局累计的赢球总数（只统计真正判给某方的分，LET / 死球不计）
        self.total_points: Dict[str, int] = {'A': 0, 'B': 0}
        self.all_decisions: List[Decision] = []
        self.game_log: List[Dict[str, Any]] = []
        self.rule_counts: Dict[str, int] = {}

        self._new_game(first_server)

    # ------------------------------------------------------------------ 内部
    def _new_game(self, first_server: str) -> None:
        if self.profile == 'ttnet':
            from referee.yolo_sm.ttnet_state_machine import (
                Profile, TTNetStateMachine,
            )
            self.machine = TTNetStateMachine(
                rules=self._rules, first_server=first_server, profile=Profile.TTNET)
        else:
            self.machine = TableTennisStateMachine(
                rules=self._rules, first_server=first_server)
        self._game_no = sum(self.games.values()) + 1

    def _absorb(self, decisions: Sequence[Decision]) -> List[Decision]:
        """登记判罚；遇到 GAME_OVER 就开新局。"""
        for d in decisions:
            self.all_decisions.append(d)
            if d.rule:
                self.rule_counts[d.rule] = self.rule_counts.get(d.rule, 0) + 1
            # 真正判给某方的分要记进跨局累计账本 —— 换局时 machine.points 会归零，
            # 只看 machine.points 会让「累计得分」在换局瞬间倒退。
            if d.kind == DecisionKind.POINT and d.winner in ('A', 'B'):
                self.total_points[d.winner] += 1
        if self.machine.game_over:
            winner = 'A' if self.machine.points['A'] > self.machine.points['B'] else 'B'
            self.games[winner] = self.games.get(winner, 0) + 1
            self.game_log.append({
                'game': self._game_no,
                'winner': winner,
                'score': (self.machine.points['A'], self.machine.points['B']),
                'timestamp_s': decisions[-1].timestamp_s if decisions else None,
            })
            if self.verbose:
                print('    [YOLO状态机] 第 {} 局结束：{} 胜 {}-{}，开新局'.format(
                    self._game_no, winner,
                    self.machine.points['A'], self.machine.points['B']))
            # 下一局由上一局胜者首发（ITTF 规定）
            self._first_server = winner
            self._new_game(winner)
        return list(decisions)

    # ------------------------------------------------------------------ 输入
    def update(self, event: RefereeEvent) -> List[Decision]:
        """喂入一条物理事件。"""
        out = self._absorb(self.machine.update(event))
        if self.verbose:
            for d in out:
                print('    [YOLO状态机] {:.3f}s {}'.format(d.timestamp_s, d.describe()))
        return out

    def tick(self, timestamp_s: float) -> List[Decision]:
        """时间推进（超时判据需要）。"""
        out = self._absorb(self.machine.tick(timestamp_s))
        if self.verbose:
            for d in out:
                print('    [YOLO状态机] {:.3f}s {}'.format(d.timestamp_s, d.describe()))
        return out

    def finalize(self, end_ts: Optional[float] = None) -> List[Decision]:
        """视频结束：把残留回合按超时判据结算掉（不伪造时间）。

        为什么必须**分段多次 tick**（这是一个真实的踩坑）
        ----------------------------------------------
        YOLO 状态机的 ``tick()`` **每次调用只推进一步**：

        * 第 1 步：``RALLY`` 静默超过 ``rally_end_s`` → 转入 ``RALLY_END_CANDIDATE``，
          记下 ``candidate_since``；
        * 第 2 步：候选期再静默 ``confirm_s`` → 才真正裁决。

        因此「一次性 tick 到末帧 + 1 秒」**永远判不出来** —— 它只完成了第 1 步，
        状态停在 ``RALLY_END_CANDIDATE``，末尾那一分就静默丢了。

        实测（test_2，最后落台 29.900s，视频 29.992s 结束）：
        * 单次 ``tick(30.99)`` → 无判罚，比分停在 A0:B1；
        * 分两次 ``tick(30.85)`` + ``tick(31.15)`` → 正确判出 A 得分，A1:B1。

        这里按「候选起点 + 消抖窗口」三个时间点各推进一次，保证每一步的阈值
        都真正过去。**时间点只推进到刚好越过阈值，绝不把结算结果倒灌到更早的帧**
        —— 上一版曾把终局比分覆盖到末帧前 2 秒，造成 2 秒的时序错位。
        """
        if end_ts is None:
            return []

        rules = self._rules
        out: List[Decision] = []
        # 第 1 步：越过 rally_end_s，进入回合结束候选
        out.extend(self.tick(end_ts + rules.rally_end_s + 0.01))
        # 第 2 步：越过 confirm_s，正式裁决
        out.extend(self.tick(end_ts + rules.rally_end_s + rules.confirm_s + 0.02))
        # 第 3 步：兜底再推一次，处理「裁决后又产生新候选」的极端情况
        out.extend(self.tick(end_ts + 2 * (rules.rally_end_s + rules.confirm_s) + 0.03))
        return out

    # ------------------------------------------------------------------ 查询
    @property
    def points(self) -> Tuple[int, int]:
        return (self.machine.points['A'], self.machine.points['B'])

    @property
    def server(self) -> str:
        return self.machine.server

    @property
    def state(self) -> str:
        return self.machine.state.value

    def scoreboard(self) -> Dict[str, Any]:
        """与 TTNet 原状态机同形的比分快照（供渲染层直接复用）。

        ``score_a`` / ``score_b`` 是**跨局累计**的赢球数：测试片段不存在「打完一局」，
        必须让比分单调上升；换局瞬间 ``machine.points`` 会归零，若直接透传就会
        看到比分倒退。``games_a`` / ``games_b`` 保留真实的局分。
        """
        return {
            'score_a': self.total_points['A'],
            'score_b': self.total_points['B'],
            'games_a': self.games['A'],
            'games_b': self.games['B'],
            'server': self.machine.server,
            'phase': self.machine.state.value,
            'game_no': self._game_no,
            'game_points': (self.machine.points['A'], self.machine.points['B']),
        }

    def describe(self) -> Dict[str, Any]:
        d = self.machine.describe()
        d['games_won'] = dict(self.games)
        d['rule_counts_total'] = dict(self.rule_counts)
        d['game_log'] = list(self.game_log)
        return d
