# -*- coding: utf-8 -*-
"""从 YOLO 项目（pingpong-smart-referee）移植的共用计分状态机。

来源
----
``https://github.com/V2tin19/pingpong-smart-referee``

* ``core/schemas.py``        → :mod:`referee.yolo_sm.schemas`
* ``core/state_machine.py``  → :mod:`referee.yolo_sm.state_machine`
* ``tests/test_state_machine.py`` → :mod:`referee.yolo_sm.test_state_machine`

移植原则
--------
**代码原样保留，只改导入路径。** 这样 YOLO 分支后续升级状态机规则时，
可以直接覆盖这两个文件，TTNet 侧一行都不用改——这正是方案文档
《TTNet、TrackNet、YOLO 共用计分状态机方案》的核心诉求：
「更换检测模型时，乒乓球规则不会变化」。

与 TTNet 侧的差异（为什么不能直接用原来那套）
---------------------------------------------
TTNet 原状态机（``referee/score_state_machine.py``）是「自下而上」长出来的：
它按 TTNet 的事件特性（bounce 只有位置、HIT 靠轨迹反转推断）堆了大量
经验护栏（可见性判据、颠球判据、钩子判据……）。

YOLO 项目这套是「按方案文档自上而下」写的：6 状态序列 + 5 条 ITTF 判罚规则，
每条规则独立开关，用 ``RefereeEvent`` 作为唯一入口。它更接近教科书定义，
也更容易解释「这分是哪条规则给的」。

本包提供的桥接
--------------
* :func:`referee.yolo_sm.bridge.ttnet_event_to_referee_event`
  —— 把 TTNet 的 ``unified_types.Event`` 转成 ``schemas.RefereeEvent``。
* :class:`referee.yolo_sm.bridge.YoloCoreScorer`
  —— 跑 YOLO 状态机，产出与 TTNet 原状态机同形的「帧号 → 比分快照」。
"""

from referee.yolo_sm.schemas import (
    BallObservation,
    EventScores,
    EventType,
    FrameObservation,
    RefereeEvent,
    Side,
    Source,
    make_event_id,
)
from referee.yolo_sm.state_machine import (
    Decision,
    DecisionKind,
    MatchRules,
    MatchState,
    TableTennisStateMachine,
)

__all__ = [
    'BallObservation',
    'EventScores',
    'EventType',
    'FrameObservation',
    'RefereeEvent',
    'Side',
    'Source',
    'make_event_id',
    'Decision',
    'DecisionKind',
    'MatchRules',
    'MatchState',
    'TableTennisStateMachine',
]
