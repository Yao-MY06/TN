"""计分状态机的选择。

* ``gd`` / ``gd-traj``：国栋的门控版（V2tin19/pingpong-smart-referee ``main``，11-04 ``73ea183``）。
  该仓库旧版 ``728ef76`` 的 ``core/`` 与 TN ``tt/_unpacked/core`` 逐字节相同；新版改了状态机与轨迹层，
  ``core/schemas.py`` 不变（加载时校验），因此事件类型、判罚对象与 TN 是同一套类。
  ``gd`` 只换状态机；``gd-traj`` 连同新版 ``TrajectoryEventProcessor``（关联速度硬截断、确认帧防劫持）一起换。
  按文件路径加载、以独立模块名注册，``from core.xxx`` 解析到 TN 的 core；不复制、不修改国栋仓库。

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
from typing import Dict, Optional

from ._paths import PPQ_ROOT, YOLO_ROOT  # noqa: F401  确保 core 可导入

import core.schemas as _schemas  # noqa: E402
import core.state_machine as _sm  # noqa: E402

CHOICES = ("tn", "ttnet11-yolo", "ttnet11-ttnet", "gd", "gd-traj")
try:
    from ._paths import TTNET11_ROOT
except ImportError:
    TTNET11_ROOT = os.path.join(PPQ_ROOT, "TTNet1.1")


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


GD_ROOT = YOLO_ROOT


def _norm_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n")


def _load_gd(filename: str, root: str = GD_ROOT):
    """按路径加载国栋仓库 core/ 下的一个文件，模块名 cascade_gd_<stem>。"""
    path = os.path.join(root, "core", filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到国栋的 {filename}：{path}（先在 ppq 下克隆/拉取 pingpong-smart-referee）")
    gd_schemas = os.path.join(root, "core", "schemas.py")
    if _norm_text(gd_schemas) != _norm_text(_schemas.__file__):
        raise RuntimeError("国栋仓库的 core/schemas.py 与 TN core 不一致，事件类型不能混用；先核对版本")
    name = "cascade_gd_" + os.path.splitext(filename)[0]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def trajectory_processor_class(kind: str = "tn"):
    """提交层用的轨迹层类：只有 gd-traj 换成国栋新版，其余用 TN core。"""
    if kind == "gd-traj":
        return _load_gd("trajectory_processor.py").TrajectoryEventProcessor
    from core.trajectory_processor import TrajectoryEventProcessor
    return TrajectoryEventProcessor


def gate_stats(sm) -> Optional[Dict[str, int]]:
    """门控版状态机的各门触发计数（其他状态机返回 None）。"""
    stats = getattr(sm, "_gate_stats", None)
    return dict(stats) if isinstance(stats, dict) else None


def make_state_machine(kind: str = "tn"):
    if kind == "tn":
        return _sm.TableTennisStateMachine(_sm.MatchRules())
    if kind in ("ttnet11-yolo", "ttnet11-ttnet"):
        mod = _load_ttnet11()
        profile = mod.Profile.YOLO if kind.endswith("yolo") else mod.Profile.TTNET
        return mod.TTNetStateMachine(profile=profile)
    if kind in ("gd", "gd-traj"):
        mod = _load_gd("state_machine.py")
        return mod.TableTennisStateMachine(mod.MatchRules())
    raise ValueError(f"未知状态机 {kind!r}，可选 {CHOICES}")
