"""把现有仓库的模块挂到 sys.path 上。

cascade 只引用、不复制、不修改任何现有文件：
  * tt/_unpacked/core      状态机交付件（与 pingpong-smart-referee/core 逐字节相同）
  * tt/run_mixed.py        混合事件源（MixedEventSource、出界推断、抖动合并……）
  * adapter/observation.py Observation / Event 的构造与校验
  * TrackNet/TrackNet/detect_bounce.py  离线落台规则（按文件路径加载，见下）
"""

from __future__ import annotations

import importlib.util
import os
import sys

TN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TT_DIR = os.path.join(TN_ROOT, "tt")
SM_ROOT = os.path.join(TT_DIR, "_unpacked")
ADAPTER_DIR = os.path.join(TN_ROOT, "adapter")
TRACKNET_DIR = os.path.join(TN_ROOT, "TrackNet", "TrackNet")
CALIB_QUADS = os.path.join(TN_ROOT, "calib", "table_quads.json")
EXAMPLES_DIR = os.path.join(TN_ROOT, "adapter", "examples")

# 三个仓库并排放在同一个父目录下（ppq/TN、ppq/TTNet、ppq/pingpong-smart-referee）
PPQ_ROOT = os.path.dirname(TN_ROOT)
TTNET_ROOT = os.path.join(PPQ_ROOT, "TTNet")
YOLO_ROOT = os.path.join(PPQ_ROOT, "pingpong-smart-referee")

for _p in (SM_ROOT, TT_DIR, ADAPTER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_detect_bounce():
    """按文件路径加载 detect_bounce.py。

    不把 TrackNet/TrackNet 整个目录挂进 sys.path：那里有 datasets.py、test.py、
    main.py 这类通用名字，挂进去容易把别的包顶掉。
    """
    name = "detect_bounce"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(TRACKNET_DIR, "detect_bounce.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 机位预设名 -> calib/table_quads.json 里的键（两者命名历史上不一致）
PRESET_TO_QUAD = {"test2": "test_2", "game4": "game_4"}
