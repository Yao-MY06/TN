"""帧源与 TTNet 原始帧环形缓存。"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from threading import Lock
from typing import Iterator, Optional, Tuple

import cv2

from cascade.ttnet_engine import H_IN, W_IN, small_rgb


class Camera(ABC):
    """统一帧源接口：所有阶段共用采集端生成的 frame_id / timestamp_s。"""

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def frames(self) -> Iterator[Tuple[int, float, object]]:
        raise NotImplementedError

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()


class VideoFileCamera(Camera):
    """用视频文件模拟摄像头。

    ``pace="realtime"`` 按视频 fps 用墙钟吐帧，下游跟不上时由工作器丢帧；
    ``pace="all"`` 不做墙钟等待，靠有界队列/生成器消费形成背压。
    """

    def __init__(self, path: str, pace: str = "realtime", speed: float = 1.0) -> None:
        if pace not in ("realtime", "all"):
            raise ValueError("pace 必须是 realtime 或 all")
        if speed <= 0:
            raise ValueError("speed 必须 > 0")
        self.path = path
        self.pace = pace
        self.speed = float(speed)
        self.cap = None
        self.fps = 0.0
        self.n_frames = 0
        self.width = 0
        self.height = 0

    def start(self) -> None:
        if self.cap is not None:
            return
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.path}")
        self.cap = cap
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 120.0)
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def frames(self) -> Iterator[Tuple[int, float, object]]:
        if self.cap is None:
            self.start()
        frame_id = 0
        start = time.perf_counter()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            ts = frame_id / self.fps if self.fps else 0.0
            if self.pace == "realtime":
                target = start + ts / self.speed
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            yield frame_id, ts, frame
            frame_id += 1


class NullCamera(Camera):
    """只产生 frame_id / timestamp 的空帧源，给全回放模式和测试使用。"""

    def __init__(self, n_frames: int, fps: float = 120.0, pace: str = "all",
                 speed: float = 1.0) -> None:
        if n_frames < 0:
            raise ValueError("n_frames 必须 >= 0")
        if fps <= 0:
            raise ValueError("fps 必须 > 0")
        if pace not in ("realtime", "all"):
            raise ValueError("pace 必须是 realtime 或 all")
        self.n_frames = int(n_frames)
        self.fps = float(fps)
        self.pace = pace
        self.speed = float(speed)
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def frames(self) -> Iterator[Tuple[int, float, None]]:
        start = time.perf_counter()
        for frame_id in range(self.n_frames):
            ts = frame_id / self.fps
            if self.pace == "realtime":
                target = start + ts / self.speed
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            yield frame_id, ts, None


class FrameRing:
    """线程安全的最近 N 秒小图缓存；只存 TTNet 的 320x128 RGB 输入。"""

    def __init__(self, capacity_s: float = 3.0, fps: float = 120.0) -> None:
        if capacity_s <= 0:
            raise ValueError("capacity_s 必须 > 0")
        if fps <= 0:
            raise ValueError("fps 必须 > 0")
        self.capacity_s = float(capacity_s)
        self.fps = float(fps)
        self.capacity_frames = max(1, int(round(self.capacity_s * self.fps)))
        self._frames: "OrderedDict[int, object]" = OrderedDict()
        self._lock = Lock()
        self._min_ever = 0
        self.evicted_before_read = 0

    def push(self, frame_id: int, bgr) -> Optional[object]:
        if bgr is None:
            return None
        small = small_rgb(bgr)
        with self._lock:
            self._frames[int(frame_id)] = small
            while len(self._frames) > self.capacity_frames:
                old_id, _ = self._frames.popitem(last=False)
                self._min_ever = max(self._min_ever, int(old_id) + 1)
        return small

    def get(self, frame_id: int) -> Optional[object]:
        fid = int(frame_id)
        with self._lock:
            img = self._frames.get(fid)
            if img is None and fid < self._min_ever:
                self.evicted_before_read += 1
            return img

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def min_frame_id(self) -> Optional[int]:
        with self._lock:
            return next(iter(self._frames), None)

    @property
    def max_frame_id(self) -> Optional[int]:
        with self._lock:
            return next(reversed(self._frames), None) if self._frames else None

    def stats(self) -> dict:
        with self._lock:
            return {"capacity_s": self.capacity_s, "capacity_frames": self.capacity_frames,
                    "size": len(self._frames), "evicted_before_read": self.evicted_before_read}
