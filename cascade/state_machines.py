"""计分状态机的选择。

* ``tn``（默认）：TN 仓库 ``tt/_unpacked/core`` 的共用状态机，与 run_mixed 完全一致。
* ``ttnet11-yolo`` / ``ttnet11-ttnet``：chagoup/TTNet1.1 的 ``TTNetStateMachine``，
  分别对应 ``Profile.YOLO``（行为与父类一致）和 ``Profile.TTNET``（三条 TTNet 专属修正：
  击球不清空落台侧、WAITING_SERVE 下的落台直接进 RALLY、NO_RETURN 证据放宽，另有同回合去重）。

TTNet1.1 的 ``yolo_sm/state_machine.py``、``schemas.py`` 与 TN 的 ``core/`` 逐字节相同（仅导入路径不同），
所以加载 ``ttnet_state_machine.py`` 时把它的 ``referee.yolo_sm.*`` 导入映射到 TN 的 ``core.*``：
事件类型、判罚对象都是同一套类，不复制、不修改 TTNet1.1 的任何文件。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

from ._paths import TTNET11_ROOT  # noqa: F401  确保 core 可导入

import core.schemas as _schemas  # noqa: E402
import core.state_machine as _sm  # noqa: E402

CHOICES = ("tn", "ttnet11-yolo", "ttnet11-ttnet")


def _load_ttnet11(root: str = TTNET11_ROOT):
    path = os.path.join(root, "src", "referee", "yolo_sm", "ttnet_state_machine.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 TTNet1.1 的状态机：{path}")
    name = "cascade_ttnet11_state_machine"
    if name in sys.modules:
        return sys.modules[name]
    for pkg in ("referee", "referee.yolo_sm"):
        sys.modules.setdefault(pkg, types.ModuleType(pkg))
    saved = {k: sys.modules.get(k) for k in ("referee.yolo_sm.schemas", "referee.yolo_sm.state_machine")}
    sys.modules["referee.yolo_sm.schemas"] = _schemas
    sys.modules["referee.yolo_sm.state_machine"] = _sm
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


def make_state_machine(kind: str = "tn"):
    if kind == "tn":
        return _sm.TableTennisStateMachine(_sm.MatchRules())
    if kind in ("ttnet11-yolo", "ttnet11-ttnet"):
        mod = _load_ttnet11()
        profile = mod.Profile.YOLO if kind.endswith("yolo") else mod.Profile.TTNET
        return mod.TTNetStateMachine(profile=profile)
    raise ValueError(f"未知状态机 {kind!r}，可选 {CHOICES}")
