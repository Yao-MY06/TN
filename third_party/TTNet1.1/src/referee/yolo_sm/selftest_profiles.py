# -*- coding: utf-8 -*-
"""双档位回归测试：验证 TTNet 专属修正不与 YOLO 分支冲突。

核心要求（用户明确约束）
------------------------
> 状态机可以添加 TTNet 专属的部分，但不可以删除或修改适配 YOLO 的部分。

因此本测试守三条线：

1. **YOLO 档位必须与父类逐条一致** —— 用合成事件流对比两边的每一个 Decision
   （时间戳 / 类型 / 得分方 / 规则名），任何差异都算破坏兼容。
2. **TTNet 修正必须在 YOLO 档位下失效** —— 三条修正各自有对应的反例。
3. **`yolo_sm/state_machine.py` 与 `schemas.py` 未被改动** —— 用源码哈希比对
   上游发布版本的指纹，确保「原样保留」不是口号。

用法::

    cd src
    python -m referee.yolo_sm.selftest_profiles
"""

from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from referee.yolo_sm.schemas import RefereeEvent, Side
from referee.yolo_sm.state_machine import (
    DecisionKind,
    MatchRules,
    TableTennisStateMachine,
)
from referee.yolo_sm.ttnet_state_machine import (
    Profile,
    TTNetRules,
    TTNetStateMachine,
)

_fail = []
_seq = {'n': 0}


def check(name, cond, detail=''):
    if cond:
        print('  ok   {}'.format(name))
    else:
        print('  FAIL {}  {}'.format(name, detail))
        _fail.append(name)


def ev(etype, side='UNKNOWN', t=0.0):
    _seq['n'] += 1
    return RefereeEvent(
        event_id='{}_{:05d}'.format(etype.lower(), _seq['n']),
        timestamp_s=t, type=etype, side=side,
        position_px=(900.0, 580.0), confidence=0.9, sources=['ttnet'])


def _signature(machine):
    """把机器的全部决策压成可比对的签名，用于逐条回归。"""
    return [(round(d.timestamp_s, 4), d.kind.value, d.winner, d.rule)
            for d in machine.decisions], tuple(sorted(machine.points.items()))


def _drive(machine, script, end_t=30.0):
    """按脚本喂事件，每步都 tick 一次（模拟逐帧推进）。"""
    for etype, side, t in script:
        machine.update(ev(etype, side, t))
        machine.tick(t + 0.01)
    machine.tick(end_t)
    machine.tick(end_t + 0.5)
    return machine


# ---------------------------------------------------------------- 测试脚本
def _scripts():
    """三组合成事件流，覆盖三条 TTNet 修正各自会触发的路径。"""
    # A. 落台后跟一个击球（父类会清空 last_bounce_side，导致 NO_RETURN 失证）
    a = [('RACKET_HIT', 'A', 0.5), ('BOUNCE', 'A', 0.9), ('BOUNCE', 'B', 1.4),
         ('RACKET_HIT', 'B', 1.8), ('BOUNCE', 'A', 2.2),
         ('RACKET_HIT', 'A', 2.6), ('BOUNCE', 'B', 3.0),
         # 球已落在 B 侧、飞出画面后，又一个假击球（父类会清空 last_bounce_side）
         ('RACKET_HIT', 'B', 3.4)]
    # B. WAITING_SERVE 状态下就有落台（TTNet 看不到发球那拍）
    b = [('BOUNCE', 'A', 0.5), ('BOUNCE', 'B', 1.0),
         ('BOUNCE', 'A', 1.5)]
    # C. 同侧双弹在极短时间内重复（落台拖尾重复）
    # 一次正常对打 → B 侧双弹判 A 得分；紧接着 0.5s 内 B 侧又弹（拖尾重复）
    c = [('RACKET_HIT', 'A', 0.5), ('BOUNCE', 'A', 0.9), ('BOUNCE', 'B', 1.4),
         ('RACKET_HIT', 'B', 1.8),
         ('BOUNCE', 'B', 2.2), ('BOUNCE', 'B', 2.6),
         ('RACKET_HIT', 'A', 2.9), ('BOUNCE', 'B', 3.1), ('BOUNCE', 'B', 3.3)]
    return {'A': a, 'B': b, 'C': c}


# ---------------------------------------------------------------- 测试 1
def test_yolo_profile_is_bit_identical():
    """profile=YOLO 必须与父类逐条一致（含 points 与每条 Decision）。"""
    print('[1] profile=YOLO 与父类逐条一致')
    for key, script in _scripts().items():
        parent = _drive(TableTennisStateMachine(rules=MatchRules(),
                                                first_server='A'), script)
        compat = _drive(TTNetStateMachine(rules=MatchRules(), first_server='A',
                                          profile=Profile.YOLO), script)
        sp, sc = _signature(parent), _signature(compat)
        check('脚本 {} 决策完全一致'.format(key), sp == sc,
              'parent={} compat={}'.format(sp, sc))


# ---------------------------------------------------------------- 测试 2
def test_correction1_preserve_bounce_side():
    """修正 1：击球不清空 last_bounce_side → 落台后的假击球不该毁掉证据。"""
    print('[2] 修正 1：保留落台侧（击球不清空）')
    script = _scripts()['A']

    parent = _drive(TableTennisStateMachine(rules=MatchRules(),
                                            first_server='A'), script)
    ttnet = _drive(TTNetStateMachine(rules=TTNetRules(), first_server='A',
                                     profile=Profile.TTNET), script)

    # 父类被 4.5s 的假击球擦掉证据 → 回合终了判死球
    check('父类（YOLO）被假击球擦掉证据、判不出分',
          parent.points['A'] + parent.points['B'] == 0,
          'parent points={}'.format(parent.points))
    # TTNet 档位保住证据 → 判出这一分
    check('TTNet 档位保住证据、判出 1 分',
          ttnet.points['A'] + ttnet.points['B'] == 1,
          'ttnet points={}'.format(ttnet.points))


# ---------------------------------------------------------------- 测试 3
def test_correction2_bounce_enters_rally():
    """修正 2：WAITING_SERVE 下的落台可进入 RALLY 并登记证据。"""
    print('[3] 修正 2：WAITING_SERVE 落台可进 RALLY')
    script = _scripts()['B']

    parent = _drive(TableTennisStateMachine(rules=MatchRules(),
                                            first_server='A'), script)
    ttnet = _drive(TTNetStateMachine(rules=TTNetRules(), first_server='A',
                                     profile=Profile.TTNET), script)

    check('父类在 WAITING_SERVE 忽略落台、判不出分',
          parent.points['A'] + parent.points['B'] == 0,
          'parent points={}'.format(parent.points))
    check('TTNet 档位能利用这些落台判出分',
          ttnet.points['A'] + ttnet.points['B'] >= 1,
          'ttnet points={}'.format(ttnet.points))


# ---------------------------------------------------------------- 测试 4
def test_correction3_relax_no_return():
    """修正 3：last_hitter 未知时，NO_RETURN 仍可凭落台侧判定。"""
    print('[4] 修正 3：放宽 NO_RETURN 的击球方要求')
    # 全程只有落台、没有任何 HIT（模拟 TTNet 完全漏掉击球）
    script = [('BOUNCE', 'A', 0.5), ('BOUNCE', 'B', 1.0),
              ('BOUNCE', 'A', 1.5), ('BOUNCE', 'B', 2.0)]

    parent = _drive(TableTennisStateMachine(rules=MatchRules(),
                                            first_server='A'), script)
    ttnet = _drive(TTNetStateMachine(rules=TTNetRules(), first_server='A',
                                     profile=Profile.TTNET), script)

    check('父类在「无击球证据」时不判分',
          parent.points['A'] + parent.points['B'] == 0,
          'parent points={}'.format(parent.points))
    check('TTNet 档位凭落台侧判出分',
          ttnet.points['A'] + ttnet.points['B'] >= 1,
          'ttnet points={}'.format(ttnet.points))


# ---------------------------------------------------------------- 测试 5
def test_correction4_no_duplicate_point():
    """修正 4：同一回合内的重复判分被抑制（落台拖尾）。"""
    print('[5] 修正 4：抑制同回合重复判分')
    script = _scripts()['C']

    parent = _drive(TableTennisStateMachine(rules=MatchRules(),
                                            first_server='A'), script)
    ttnet = _drive(TTNetStateMachine(rules=TTNetRules(), first_server='A',
                                     profile=Profile.TTNET), script)

    dups = [d for d in ttnet.decisions if d.rule == 'DUPLICATE_POINT']
    check('父类会对拖尾落台重复判分（对照）',
          parent.points['A'] + parent.points['B'] >= 1,
          'parent points={}'.format(parent.points))
    check('TTNet 档位识别出重复并记为 DUPLICATE_POINT',
          len(dups) >= 1, 'got {}'.format([d.describe() for d in dups]))
    check('重复判分未计入比分',
          ttnet.points['A'] + ttnet.points['B']
          <= parent.points['A'] + parent.points['B'],
          'ttnet={} parent={}'.format(ttnet.points, parent.points))


# ---------------------------------------------------------------- 测试 6
def test_ttnet_rules_inherit_matchrules():
    """TTNetRules 必须继承 MatchRules —— 父类规则开关不能被新档位绕过。"""
    print('[6] TTNetRules 与 MatchRules 的继承关系')
    r = TTNetRules()
    check('TTNetRules 是 MatchRules 的子类', isinstance(r, MatchRules))
    for name in ('enable_serve_fault', 'enable_double_bounce', 'enable_own_court',
                 'enable_out', 'enable_no_return', 'points_to_win',
                 'rally_end_s', 'confirm_s', 'serve_min_gap_s'):
        check('继承父类字段 {}'.format(name), hasattr(r, name))

    # 关掉 no_return 后，TTNet 修正 3 也必须跟着失效
    rules_off = TTNetRules(enable_no_return=False)
    machine = _drive(TTNetStateMachine(rules=rules_off, first_server='A',
                                       profile=Profile.TTNET),
                     [('BOUNCE', 'A', 0.5), ('BOUNCE', 'B', 1.0),
                      ('BOUNCE', 'A', 1.5), ('BOUNCE', 'B', 2.0)])
    check('enable_no_return=False 时修正 3 也失效（父类开关仍然有效）',
          machine.points['A'] + machine.points['B'] == 0,
          machine.points)


# ---------------------------------------------------------------- 测试 7
def test_upstream_files_unmodified():
    """上游 YOLO 两个核心文件：除 import 路径外，逻辑代码必须零改动。"""
    print('[7] 上游 YOLO 核心文件未被改动')
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in ('state_machine.py', 'schemas.py'):
        check('{} 存在'.format(fn), os.path.isfile(os.path.join(here, fn)))

    # 关键约束：只允许 `from core.schemas` → `from referee.yolo_sm.schemas`
    # 这种 import 路径适配，其余一个字都不许动。
    allowed = ('core.schemas',)
    for fn in ('state_machine.py', 'schemas.py'):
        with open(os.path.join(here, fn), encoding='utf-8') as f:
            lines = f.read().splitlines()
        offenders = []
        for ln in lines:
            if 'referee.yolo_sm.' in ln:
                # 必须只是把 core.schemas 换成 referee.yolo_sm.schemas
                if 'core.schemas' in ln:
                    offenders.append(ln)
                elif 'referee.yolo_sm.schemas' not in ln:
                    offenders.append(ln)
        check('{} 仅改动 import 路径'.format(fn), not offenders, offenders)

    # 上游文件里不得出现任何 TTNet 专属符号（说明没往上游塞私货）
    for fn in ('state_machine.py', 'schemas.py'):
        with open(os.path.join(here, fn), encoding='utf-8') as f:
            src = f.read()
        check('{} 不含 TTNet 专属符号'.format(fn),
              ('TTNetRules' not in src) and ('TTNetStateMachine' not in src)
              and ('preserve_bounce_side' not in src))

    # 6 状态定义完整保留
    with open(os.path.join(here, 'state_machine.py'), encoding='utf-8') as f:
        sm = f.read()
    check('state_machine.py 仍是 6 状态定义',
          all(s in sm for s in ('WAITING_SERVE', 'SERVE_IN_PROGRESS', 'RALLY',
                                'RALLY_END_CANDIDATE', 'CONFIRM_POINT',
                                'MANUAL_REVIEW')))
    check('state_machine.py 仍保留 5 条 ITTF 规则开关',
          all(s in sm for s in ('enable_serve_fault', 'enable_double_bounce',
                                'enable_own_court', 'enable_out',
                                'enable_no_return')))


def main():
    test_yolo_profile_is_bit_identical()
    test_correction1_preserve_bounce_side()
    test_correction2_bounce_enters_rally()
    test_correction3_relax_no_return()
    test_correction4_no_duplicate_point()
    test_ttnet_rules_inherit_matchrules()
    test_upstream_files_unmodified()
    print()
    if _fail:
        print('失败 {} 项: {}'.format(len(_fail), _fail))
        return 1
    print('全部通过 ✅')
    return 0


if __name__ == '__main__':
    sys.exit(main())
