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

# 同组仓库的位置。两种布局都要支持：
#   本机：ppq/TN、ppq/TTNet、ppq/pingpong-voice-referee …… 并排（TN 的上一级是项目根）；
#   原机：TN 仓库本身就克隆在 ppq/，其他仓库放在它里面（ppq/pingpong-voice-referee），
#         fusion/three-way 分支另带 TN/third_party/ 快照。
# 查找顺序：环境变量 PPQ_ROOT（或 TN 的上一级）→ TN 根目录 → TN/third_party。
PPQ_ROOT = os.environ.get("PPQ_ROOT") or os.path.dirname(TN_ROOT)
THIRD_PARTY = os.path.join(TN_ROOT, "third_party")


def _first_existing(*cands):
    for c in cands:
        if os.path.isdir(c):
            return c
    return cands[0]


def find_repo(*names: str) -> str:
    """按上面的顺序找第一个存在的同组仓库目录；都不存在时返回第一个候选（报错信息用）。"""
    cands = [os.path.join(base, n) for base in (PPQ_ROOT, TN_ROOT, THIRD_PARTY) for n in names]
    return _first_existing(*cands)


TTNET_ROOT = find_repo("TTNet", "TTNet1.1")
TTNET11_ROOT = find_repo("TTNet1.1")
YOLO_ROOT = find_repo("pingpong-smart-referee")
VOICE_ROOT = find_repo("pingpong-voice-referee")

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
