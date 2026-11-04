"""``detect_bounce.detect()`` 的流式等价实现：只做"窗口内 y 局部极大"这一步。

离线版对第 i 个检出点的判定只依赖 ``[f_i - win, f_i + win]`` 帧内的检出，
因此只要收齐 ``f_i + win`` 帧，就能得到与离线完全相同的结论——前瞻 ``win``
帧（默认 12 帧，120 fps 下 0.1 s）。

阈值（落差、台内、最小间隔、侧别）不在这里做：提议层用宽松阈值保召回，
规则复核器用离线的严格阈值，两者共享同一批局部极大点，所以
"严格版结果 ⊆ 宽松版提议"在构造上成立。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class LocalMax:
    frame_id: int
    timestamp_s: float
    x: float
    y: float
    drop_before: float      # 与离线 drop_b 相同：ys[i] - min(窗口内之前的 y)
    drop_after: float       # 与离线 drop_a 相同
    vx_in: float            # 上一检出 -> 本点的水平速度（px/s，按 fps 换算，同离线 dx）
    vx_out: float           # 本点 -> 下一检出


class StreamingLocalMax:
    def __init__(self, win: int = 12, fps: float = 120.0) -> None:
        self.win = int(win)
        self.fps = float(fps)
        self._f: List[int] = []
        self._t: List[float] = []
        self._x: List[float] = []
        self._y: List[float] = []
        self._next = 0          # 下一个待判定的检出下标
        self._done = False

    # ------------------------------------------------------------------
    def push(self, frame_id: int, timestamp_s: float, xy: Optional[Tuple[float, float]]) -> List[LocalMax]:
        """喂一帧。``xy`` 为 None 表示本帧没有真实检出（漏检或轨迹补帧）。"""
        if xy is not None:
            self._f.append(int(frame_id))
            self._t.append(float(timestamp_s))
            self._x.append(float(xy[0]))
            self._y.append(float(xy[1]))
        out: List[LocalMax] = []
        while self._next < len(self._f) and self._f[self._next] + self.win <= frame_id:
            lm = self._evaluate(self._next)
            if lm is not None:
                out.append(lm)
            self._next += 1
        self._trim()
        return out

    def flush(self) -> List[LocalMax]:
        """流结束：剩余检出按已有数据判定（与离线在视频末尾的行为一致）。"""
        out: List[LocalMax] = []
        while self._next < len(self._f):
            lm = self._evaluate(self._next)
            if lm is not None:
                out.append(lm)
            self._next += 1
        self._done = True
        return out

    @property
    def pending_frame_ids(self) -> List[int]:
        return self._f[self._next:]

    # ------------------------------------------------------------------
    def _evaluate(self, i: int) -> Optional[LocalMax]:
        f, ys = self._f, self._y
        fi = f[i]
        lo = bisect_left(f, fi - self.win)
        hi = bisect_right(f, fi + self.win)
        # 必须严格是窗口内的最大值，且前后各至少一帧（与离线一致）
        if lo >= i or hi - 1 <= i:
            return None
        yi = ys[i]
        if yi != max(ys[lo:hi]):
            return None
        drop_b = yi - min(ys[lo:i])
        drop_a = yi - min(ys[i + 1:hi])
        vx_in = (self._x[i] - self._x[i - 1]) / ((f[i] - f[i - 1]) / self.fps)
        vx_out = (self._x[i + 1] - self._x[i]) / ((f[i + 1] - f[i]) / self.fps)
        return LocalMax(fi, self._t[i], self._x[i], yi, float(drop_b), float(drop_a),
                        float(vx_in), float(vx_out))

    def _trim(self) -> None:
        """丢掉不会再被任何窗口用到的旧检出，长流不涨内存。"""
        if self._next < 2048:
            return
        keep_from = bisect_left(self._f, self._f[self._next] - self.win) - 1
        keep_from = max(0, keep_from)
        if keep_from <= 0:
            return
        del self._f[:keep_from], self._t[:keep_from], self._x[:keep_from], self._y[:keep_from]
        self._next -= keep_from
