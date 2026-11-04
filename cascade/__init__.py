"""YOLO → TrackNet 提议 → TTNet 区间复核 → 状态机 的级联层。

只在外层包装，不修改 tt/、adapter/、TrackNet/ 以及另外两个仓库的任何现有文件。
"""

from . import _paths  # noqa: F401  挂好 sys.path
