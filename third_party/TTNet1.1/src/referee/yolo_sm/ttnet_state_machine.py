# -*- coding: utf-8 -*-
"""TTNet 专属状态机：**继承** YOLO 共用状态机，只做增量扩展。

设计约束（用户明确要求）
------------------------
1. **不允许删除或修改适配 YOLO 的部分** —— 因此本模块用**继承 + 覆写**，
   `yolo_sm/state_machine.py` 与 `yolo_sm/schemas.py` **一个字节都不动**。
2. **必须同时兼容两者** —— 由 :class:`Profile` 开关控制：
   ``Profile.YOLO`` 时行为与父类**完全一致**（逐分可回归验证），
   ``Profile.TTNET`` 时启用下面三条 TTNet 专属修正。

为什么需要 TTNet 专属修正（实测依据）
------------------------------------
TTNet 与 YOLO 的**证据质量**根本不同，同一套判罚规则会系统性吃亏：

==========  ==========================  ==========================
维度        YOLO                        TTNet
==========  ==========================  ==========================
球可见率    ~83%（Recall@25px）         25%~48%
击球(HIT)   真实检测框                  靠「轨迹水平反转」推断，
                                        球撞台面同样反转 → 大量假阳性
发球        能看到球拍动作               看不到，常漏掉发球那一次落台
==========  ==========================  ==========================

实测这三处差异造成的具体损失（test_1 / test_2 / test_3）：

* **79/106、36/40、24/41 次落台之后 3 秒内都跟着一个 HIT** ——
  父类 `_on_racket_hit` 会**清空 `last_bounce_side`**，于是
  `NO_RETURN`（「球落在对方半台后消失」）失去证据，整回合退化成死球。
  在 YOLO 里这是对的（HIT 可信，清了应该重来）；在 TTNet 里这是错的
  （HIT 有 30%~50% 是球撞台面的假阳性）。
* **27/106、5/40、16/41 次落台发生在 `WAITING_SERVE`** —— 父类直接忽略。
  TTNet 常看不到发球那一拍击球，于是永远进不了 RALLY，这些落台白丢。
* 低可见率下 `last_hitter` 常年为 None，`NO_RETURN` 的硬前置
  （要求 hitter 与 bounce 都已知且不同侧）经常凑不齐。

三条 TTNet 专属修正
------------------
:attr:`TTNetRules.preserve_bounce_side_on_hit`
    击球**不清空** `last_bounce_side`。落台侧是**物理事实**，
    不会因为「有人挥了一下拍」而失效；被击球方则单独记录。

:attr:`TTNetRules.bounce_enters_rally_from_waiting`
    `WAITING_SERVE` 下出现落台时，允许直接进入 `RALLY` 并登记落台证据。
    （YOLO 有可靠发球击球，不需要这条；TTNet 需要。）

:attr:`TTNetRules.relax_no_return_evidence`
    `NO_RETURN` 的证据要求放宽：只要「最后一次落台在某一侧」且该侧
    **不是**最后击球方（或击球方未知），就判定对方未接到。
    即：击球方未知时不再直接放弃，而是用「球落在谁那边没被接回」
    这个更强的物理事实来判。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from referee.yolo_sm.schemas import EventType, RefereeEvent, Side
from referee.yolo_sm.state_machine import (
    Decision,
    DecisionKind,
    MatchRules,
    MatchState,
    TableTennisStateMachine,
)

__all__ = ['Profile', 'TTNetRules', 'TTNetStateMachine']


class Profile:
    """证据质量档位。切换它决定是否启用 TTNet 专属修正。"""

    YOLO = 'yolo'      # 与父类行为**完全一致**（默认，保证向后兼容）
    TTNET = 'ttnet'    # 启用三条 TTNet 专属修正


@dataclass(frozen=True)
class TTNetRules(MatchRules):
    """在父类 `MatchRules` 之上追加 TTNet 专属参数。

    继承 `MatchRules` 而不是另起一套 —— 这样父类的规则开关
    （`enable_serve_fault` 等）依然原样生效，只是多了几个 TTNet 选项。
    """

    # ---- 修正 1：击球不清空落台侧 ----
    preserve_bounce_side_on_hit: bool = True

    # ---- 修正 2：WAITING_SERVE 下落台可进入 RALLY ----
    bounce_enters_rally_from_waiting: bool = True

    # ---- 修正 3：放宽 NO_RETURN 的证据要求 ----
    relax_no_return_evidence: bool = True

    # ---- 修正 4：抹掉「同一回合内重复判分」 ----
    # TTNet 的落台常有拖尾重复（同一次落台被检出 2~3 次），
    # 会让父类的「同侧双弹」规则在同一个回合里连判两次分（实测 test_2 9.63s/10.03s）。
    # 这里要求两次判分之间至少间隔该秒数，否则第二次判分被抑制。
    # 注意：父类只有 `min_point_interval` 的概念在 legacy 分支里，
    # YOLO 状态机**没有**这个护栏（它的落台检测足够干净，不需要）。
    min_point_gap_s: float = 1.20

    # ---- 修正 3 的关键参数：落台后多久之内的击球才算「把球接回来了」----
    #
    # 球落在台面后弹起、飞到拍面再被打回，物理上约需 0.15~0.8 秒。
    # 超过这个窗口还出现的 HIT，基本都是「球已飞出画面、轨迹抖动被误判」。
    # 用「落台后窗口内无回击」判定该侧失分，比依赖 last_hitter 稳得多。
    return_window_s: float = 0.80

    # ---- 修正 3 的「结果验证」窗口：击球后多久之内必须出现对侧落台 ----
    # 这是「球真被打到对面了」的物理铁证，也是区分真假击球最可靠的一条。
    return_followup_s: float = 1.50




class TTNetStateMachine(TableTennisStateMachine):
    """TTNet 版计分状态机：继承共用状态机 + TTNet 专属修正。

    Parameters
    ----------
    rules:
        :class:`TTNetRules`（也接受父类的 ``MatchRules``）。
    first_server:
        首发球方。
    profile:
        ``Profile.TTNET`` 启用修正；``Profile.YOLO`` 时**行为等同父类**
        （用于回归验证 —— 保证 YOLO 分支不受影响）。

    Examples
    --------
    >>> sm = TTNetStateMachine(profile=Profile.TTNET, first_server='A')
    >>> sm.profile
    'ttnet'
    """

    def __init__(self, rules: Optional[MatchRules] = None,
                 first_server: str = 'A', profile: str = Profile.TTNET) -> None:
        if profile not in (Profile.YOLO, Profile.TTNET):
            raise ValueError('profile must be {!r} or {!r}, got {!r}'.format(
                Profile.YOLO, Profile.TTNET, profile))
        self.profile = profile
        # 只有 TTNET 档位才赋予 TTNetRules 的默认值；YOLO 档位沿用父类规则，
        # 这样 `profile=YOLO` 时与父类逐分一致（有回归测试保证）。
        rules = rules or (TTNetRules() if profile == Profile.TTNET else MatchRules())
        super().__init__(rules=rules, first_server=first_server)
        self._last_point_ts: Optional[float] = None
        # 父类没有「落台/击球时刻」，这里以实例属性补充（不改父类）。
        # 用单元素 list 而不是直接存 float：一次 setattr 就能更新，
        # 不需要在每次 reset 时重新绑定，也不会被 `_reset_rally` 清掉。
        self._last_hit_ts: List[Optional[float]] = [None]
        self._last_bounce_ts: List[Optional[float]] = [None]
        self._last_hit_side: List[Optional[str]] = [None]
        # 一次落台之后「下一次」落台的时刻与侧别，用于验证击球是否把球打到对面
        self._next_bounce_ts: List[Optional[float]] = [None]
        self._next_bounce_side: List[Optional[str]] = [None]

    # ------------------------------------------------------------------ 工具
    @property
    def _is_ttnet(self) -> bool:
        return self.profile == Profile.TTNET

    def _flag(self, name: str, default_on: bool = True) -> bool:
        """读 TTNet 专属开关；YOLO 档位一律视为关闭。"""
        if not self._is_ttnet:
            return False
        return bool(getattr(self.rules, name, default_on))

    # ------------------------------------------------------------------ 修正 4
    def _award(self, winner: str, t: float, rule: str, reason: str) -> List[Decision]:
        """覆写判分入口：加一道「同回合重复判分」抑制（仅 TTNet 档位）。"""
        gap = getattr(self.rules, 'min_point_gap_s', 0.0)
        if (self._flag('relax_no_return_evidence') and gap > 0
                and self._last_point_ts is not None
                and (t - self._last_point_ts) < gap):
            # 同一回合内的重复判分：不计分，但要把回合证据清掉重新布防，
            # 否则下一个真实落台会继续与旧证据配对，越判越多。
            self.dead_ball_count += 1
            decision = Decision(
                timestamp_s=t, kind=DecisionKind.DEAD_BALL,
                rule='DUPLICATE_POINT', score=self._score(),
                state=self.state.value,
                reason='距上次判分仅 {:.2f}s < {:.2f}s，判为落台拖尾重复'.format(
                    t - self._last_point_ts, gap),
            )
            self.decisions.append(decision)
            self._reset_rally(keep_server=True)
            return [decision]

        out = super()._award(winner, t, rule, reason)
        if any(d.kind == DecisionKind.POINT for d in out):
            self._last_point_ts = t
        return out

    # ------------------------------------------------------------------ 修正 1
    def _on_racket_hit(self, side: str, t: float,
                       prev_event_time: Optional[float]) -> List[Decision]:
        """覆写击球：TTNet 档位下**不清空** `last_bounce_side`。

        父类实现（YOLO 用）在击球时会把 `last_bounce_side` 置 None ——
        因为 YOLO 的 HIT 可信，新的回合确实开始了。

        但 TTNet 的 HIT 是「轨迹水平方向反转」推断出来的，球撞台面同样会反转：
        实测 test_1 有 79/106 次落台之后 3 秒内跟着一个 HIT，
        于是 `NO_RETURN` 所需的 `last_bounce_side` 被反复擦掉，整回合退化成死球。

        落台侧是**物理事实**，不会因为「有人挥了一下拍」而失效，
        因此 TTNet 档位保留它。

        另外记录 `last_hitter` 的**时刻**（`_last_hit_ts`），供修正 3 判断
        「这次击球是不是真的把球接回来了」。父类没有这个字段，
        这里以实例属性补充，不修改父类。
        """
        if not self._flag('preserve_bounce_side_on_hit'):
            return super()._on_racket_hit(side, t, prev_event_time)

        ctx = self._ctx

        if side == Side.UNKNOWN.value:
            # 与父类一致：UNKNOWN 不建立归属
            if self.state == MatchState.WAITING_SERVE:
                self.state = MatchState.RALLY
            ctx.last_hitter = None
            ctx.last_bounce_side = None
            return []

        if self.state == MatchState.WAITING_SERVE:
            gap = (t - prev_event_time) if prev_event_time is not None else float('inf')
            if gap >= self.rules.serve_min_gap_s:
                ctx.server = side
                ctx.serve_own_bounce_seen = False
                self.state = MatchState.SERVE_IN_PROGRESS
            else:
                self.state = MatchState.RALLY

        ctx.last_hitter = side
        ctx.last_bounce_side = ctx.last_bounce_side    # ★ 保留，不置 None
        self._last_hit_ts[0] = t
        self._last_hit_side[0] = side
        return []

    # ------------------------------------------------------------------ 修正 2
    def _on_bounce(self, side: str, t: float) -> List[Decision]:
        """覆写落台：TTNet 档位下 `WAITING_SERVE` 的落台可进入 RALLY。"""
        if side in (Side.A.value, Side.B.value):
            prev = self._last_bounce_ts[0]
            if prev is not None:
                # 上上次落台 → 这次落台，补齐「next」信息
                self._next_bounce_ts[0] = t
                self._next_bounce_side[0] = side
            self._last_bounce_ts[0] = t
        if (self._flag('bounce_enters_rally_from_waiting')
                and self.state == MatchState.WAITING_SERVE
                and side in (Side.A.value, Side.B.value)):
            # 看不到发球击球，但看到了落台 —— 这就是回合已经开始的正向证据。
            self.state = MatchState.RALLY
        return super()._on_bounce(side, t)

    # ------------------------------------------------------------------ 修正 3
    def _resolve_rally_end(self, t: float) -> List[Decision]:
        """覆写回合终了裁决：TTNet 档位下放宽 `NO_RETURN` 的证据要求。

        父类要求 `last_hitter` **与** `last_bounce_side` 都有值且不同侧。
        TTNet 有两个现实问题：

        1. 球可见率低，`last_hitter` 常年为 None；
        2. HIT 有 ~30% 的假阳性，且常常出现在**球已经飞出画面之后**，
           于是 `last_hitter` 被改成「根本没碰到球的那个人」。

        因此这里的关键判据不是「谁被标记为击球方」，而是
        **最后一次落台之后，球有没有被真正接回来**。
        接球的物理标志是「落台后短时间内的击球」：

        * 真实回击：落台后 0.15~0.8 秒（球从台面弹起、飞到拍面再打回）；
        * 假击球：往往在落台 1 秒以后（球早就飞走了，只是轨迹抖动被误判）。

        所以只要 **最后一次落台之后 `return_window_s` 内没有击球**，
        就说明这一侧没接回来 → 判对方得分。这条判据不依赖 `last_hitter`，
        天然免疫假阳性。
        """
        if not self._flag('relax_no_return_evidence'):
            return super()._resolve_rally_end(t)

        ctx = self._ctx
        rules = self.rules

        last_side = ctx.last_bounce_side
        if not (rules.enable_no_return and last_side in (Side.A.value, Side.B.value)):
            return super()._resolve_rally_end(t)

        # 球落回击球方自己半台 → 父类按 OWN_COURT 处理更合适，直接回落。
        if ctx.last_hitter is not None and ctx.last_hitter == last_side \
                and self._hit_returned_ball(last_side, rules):
            return super()._resolve_rally_end(t)

        # ★ 核心：球落在 last_side 之后，这一侧有没有「把球接回去」
        if self._hit_returned_ball(last_side, rules):
            return super()._resolve_rally_end(t)

        # ★★ 反向判据：如果**下一次落台已经出现且在对侧**，说明回合一直在继续，
        #     根本没有「球消失」这回事 —— 不该判分。
        #
        # 这条守的是一个实测踩过的坑（test_1 的 104s / test_3 的 26s）：
        # 落台 102.75s(B) → 下一拍击球 105.19s(B)，间隔 2.4s 超出 0.8s 回击窗口，
        # 于是被判「B 侧未接回」。可球明明是打着回来的 ——
        # 105.23s 又落在 B、105.62s 落在 A，回合完全在继续。
        # 病根是 TTNet 漏检了中间那一拍击球（球可见率只有 31%），
        # 「没看到回击」不等于「没回击」。
        nxt = self._next_bounce_ts[0]
        nxt_side = self._next_bounce_side[0]
        if nxt is not None and nxt > (self._last_bounce_ts[0] or 0) and (
                nxt_side is not None and nxt_side != last_side):
            return super()._resolve_rally_end(t)

        winner = rules.opponent(last_side)
        reason = '球落在 {} 侧后 {}s 内无有效回击，判定该侧未接回'.format(
            last_side, '%.1f' % float(getattr(rules, 'return_window_s', 0.80)))
        return self._award(winner, t, 'NO_RETURN', reason)

    def _hit_returned_ball(self, bounce_side: str, rules: MatchRules) -> bool:
        """判定「球落在 bounce_side 之后，这一侧有没有把球接回去」。

        三条判据（依次收紧，越往后越可靠）：

        1. **时间**：回击必须发生在落台后 ``return_window_s`` 之内。
           球从台面弹起、飞到拍面再打回，实测约 0.15~0.8 秒。
        2. **归属**：这次击球必须判给 **bounce_side 那一侧**。
        3. **结果（最强）**：击球之后，球必须**继续飞并对侧落台**。
           这是「接回去了」的物理铁证 —— 球被打回对方半台了。
           反之，击球之后球就消失，说明那个所谓的 HIT 只是轨迹抖动，
           球其实已经落地/飞出画面。

        第 3 条是本项目实测出来的关键区分器（`referee/` 下的诊断脚本）：

        ========  ==================  ==================
        视频      「真回击」样本        「假击球」样本
        ========  ==================  ==================
        test_1    70                  15
        test_2    33                  5
        test_3    16                  15
        ========  ==================  ==================

        即「落台后 0.8s 内出现 HIT」这件事本身**不足以**说明接球成功，
        必须再确认球确实被打到了对面。
        """
        window = float(getattr(rules, 'return_window_s', 0.80))
        follow = float(getattr(rules, 'return_followup_s', 1.50))

        bounce_ts = self._last_bounce_ts[0]
        hit_ts = self._last_hit_ts[0]
        hit_side = self._last_hit_side[0]

        if bounce_ts is None or hit_ts is None:
            return False
        # 判据 1：回击出现在落台之后、且足够及时
        if not (0 < (hit_ts - bounce_ts) <= window):
            return False
        # 判据 2：必须是落台那一侧挥的拍
        if hit_side != bounce_side:
            return False
        # 判据 3：击球之后球继续飞、并对侧落台
        nxt = self._next_bounce_ts[0]
        nxt_side = self._next_bounce_side[0]
        if nxt is None or nxt_side is None:
            return False
        if not (hit_ts < nxt <= hit_ts + follow):
            return False
        return nxt_side != bounce_side
