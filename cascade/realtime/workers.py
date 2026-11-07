"""实时层工作线程：YOLO / TrackNet 逐帧 worker 与 TTNet 区间 worker。"""

from __future__ import annotations

import copy
import os
import queue
import sys
import threading
import time
from collections import OrderedDict, deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from cascade._paths import YOLO_ROOT, load_detect_bounce
from cascade.fuse_obs import _invisible
from cascade.ttnet_reviewer import ReplayScores, TTNetReviewer


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    i = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[i]


class FrameWorker:
    """单模型逐帧 worker。

    ``pace="realtime"`` 时输入队列满会丢最旧帧，并为每个被丢帧立即输出
    不可见观测；``pace="all"`` 时阻塞提交，保证不丢帧。
    """

    def __init__(self, name: str, replay: Optional[List[Dict[str, Any]]] = None,
                 sleep_ms: float = 0.0, pace: str = "all", queue_size: int = 64,
                 source: Optional[str] = None, image_size: Tuple[int, int] = (1920, 1080),
                 predictor: Optional[Callable[[Any, int, float], Dict[str, Any]]] = None) -> None:
        self.name = name
        self.source = source or name
        self.sleep_s = max(0.0, float(sleep_ms) / 1000.0)
        self.pace = pace
        self.image_size = image_size
        self.predictor = predictor
        self.replay = {int(o["frame_id"]): o for o in (replay or [])} if replay is not None else None
        self.input: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, int(queue_size)))
        self.output: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._sentinel = object()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.processed = 0
        self.dropped = 0
        self.queue_depth_max = 0
        self.process_s: List[float] = []
        self.started_wall: Optional[float] = None
        self.finished_wall: Optional[float] = None
        self._closed = False
        # 每一帧（包括随后会被丢掉、不做推理的帧）在入队前都交给它；TrackNet 用它维护连续三帧
        self.frame_hook: Optional[Callable[[int, Any], None]] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self.started_wall = time.monotonic()
        self._thread = threading.Thread(target=self._run, name=f"{self.name}-worker", daemon=True)
        self._thread.start()

    def _missing(self, item: Tuple[int, float, Any]) -> Dict[str, Any]:
        fid, ts, _ = item
        return _invisible(int(fid), float(ts), self.image_size, self.source)

    def _emit(self, obs: Dict[str, Any]) -> None:
        self.output.put(obs)

    def submit(self, frame_id: int, timestamp_s: float, bgr) -> bool:
        item = (int(frame_id), float(timestamp_s), bgr)
        if self.frame_hook is not None and bgr is not None:
            self.frame_hook(int(frame_id), bgr)
        try:
            self.input.put_nowait(item)
        except queue.Full:
            if self.pace == "all":
                self.input.put(item)
            else:
                try:
                    old = self.input.get_nowait()
                except queue.Empty:
                    old = None
                if old is not None:
                    with self._lock:
                        self.dropped += 1
                    self._emit(self._missing(old))
                try:
                    self.input.put_nowait(item)
                except queue.Full:
                    with self._lock:
                        self.dropped += 1
                    self._emit(self._missing(item))
                    return False
        with self._lock:
            self.queue_depth_max = max(self.queue_depth_max, self.input.qsize())
        return True

    def _process(self, item: Tuple[int, float, Any]) -> Dict[str, Any]:
        fid, ts, bgr = item
        t0 = time.perf_counter()
        if self.replay is not None:
            found = self.replay.get(int(fid))
            obs = copy.deepcopy(found) if found is not None else self._missing(item)
        elif self.predictor is not None:
            obs = self.predictor(bgr, int(fid), float(ts))
        else:
            obs = self._missing(item)
        if self.sleep_s:
            time.sleep(self.sleep_s)
        self.process_s.append(time.perf_counter() - t0)
        return obs

    def _run(self) -> None:
        while True:
            item = self.input.get()
            if item is self._sentinel:
                break
            obs = self._process(item)
            with self._lock:
                self.processed += 1
            self._emit(obs)
        self.finished_wall = time.monotonic()

    def drain(self) -> List[Dict[str, Any]]:
        out = []
        while True:
            try:
                out.append(self.output.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        if self._closed:
            return
        self.input.put(self._sentinel)
        if self._thread is not None:
            self._thread.join()
        self._closed = True

    def stats(self) -> Dict[str, Any]:
        wall = ((self.finished_wall or time.monotonic()) - (self.started_wall or time.monotonic()))
        with self._lock:
            return {
                "processed": self.processed, "dropped": self.dropped,
                "fps": round(self.processed / wall, 3) if wall > 0 else None,
                "process_ms_p50": None if not self.process_s else round(_pct(self.process_s, 0.5) * 1000, 3),
                "process_ms_p95": None if not self.process_s else round(_pct(self.process_s, 0.95) * 1000, 3),
                "queue_depth_max": self.queue_depth_max,
            }


class YoloWorker(FrameWorker):
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "yolo")
        kwargs.setdefault("source", "yolo")
        super().__init__(**kwargs)


class TrackNetWorker(FrameWorker):
    """TrackNet worker；真模型模式支持最多 4 帧、最多等待 10 ms 的小批量。"""

    def __init__(self, batch_predictor: Optional[Callable[[List[Tuple[int, float, Any]]],
                                                           List[Dict[str, Any]]]] = None,
                 batch_size: int = 4, batch_wait_ms: float = 10.0, **kwargs):
        kwargs.setdefault("name", "tracknet")
        kwargs.setdefault("source", "tracknet")
        self.batch_predictor = batch_predictor
        self.batch_size = max(1, int(batch_size))
        self.batch_wait_s = max(0.0, float(batch_wait_ms) / 1000.0)
        super().__init__(**kwargs)
        self.frame_hook = getattr(batch_predictor, "observe", None)

    def _process_batch(self, batch: List[Tuple[int, float, Any]]) -> List[Dict[str, Any]]:
        if self.batch_predictor is not None:
            t0 = time.perf_counter()
            obs = self.batch_predictor(batch)
            if self.sleep_s:
                time.sleep(self.sleep_s)
            self.process_s.append(time.perf_counter() - t0)
            return obs
        out = [self._process(item) for item in batch]
        return out

    def _run(self) -> None:
        stop = False
        while not stop:
            first = self.input.get()
            if first is self._sentinel:
                break
            batch = [first]
            deadline = time.monotonic() + self.batch_wait_s
            while len(batch) < self.batch_size:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    item = self.input.get(timeout=timeout)
                except queue.Empty:
                    break
                if item is self._sentinel:
                    stop = True
                    break
                batch.append(item)
            for obs in self._process_batch(batch):
                self._emit(obs)
            with self._lock:
                self.processed += len(batch)
        self.finished_wall = time.monotonic()


class ConsecutiveFrames:
    """按帧号保存最近若干帧（已缩放），给 TrackNet 取连续三帧 (t, t-1, t-2)。

    实时节奏下 TrackNet 会跳过部分帧不推理，但它的输入必须仍是相机上的**连续**三帧，
    与训练和离线 run_video 一致；不能拿“最近处理过的三帧”（可能是 t, t-5, t-10）。
    每一帧都经 ``observe`` 缩放后存入（缩放很便宜），推理时按帧号取。线程安全。
    """

    def __init__(self, resize: Callable[[Any], Any], keep: int = 32) -> None:
        self._resize = resize
        self._keep = max(3, int(keep))
        self._frames: "OrderedDict[int, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def observe(self, frame_id: int, bgr) -> None:
        fid = int(frame_id)
        with self._lock:
            if fid in self._frames:
                return
        small = self._resize(bgr)
        with self._lock:
            self._frames[fid] = small
            while len(self._frames) > self._keep:
                self._frames.popitem(last=False)

    def window(self, frame_id: int):
        fid = int(frame_id)
        with self._lock:
            got = [self._frames.get(f) for f in (fid, fid - 1, fid - 2)]
        return None if any(g is None for g in got) else tuple(got)


class TTNetWorker:
    """TTNet 区间复核 worker；有界队列由主线程非阻塞提交。"""

    def __init__(self, reviewer: TTNetReviewer, queue_size: int = 8) -> None:
        self.reviewer = reviewer
        self.tasks: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, int(queue_size)))
        self.results: "queue.Queue[Tuple[str, Any, float]]" = queue.Queue()
        self._sentinel = object()
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self.queue_depth_max = 0
        self.processed = 0
        self.started_wall: Optional[float] = None
        self.finished_wall: Optional[float] = None
        self.process_s: List[float] = []

    def start(self) -> None:
        if self._thread is not None:
            return
        self.started_wall = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="ttnet-worker", daemon=True)
        self._thread.start()

    def submit(self, clip, detections: Dict[int, List[Tuple[float, float, str]]]) -> bool:
        try:
            self.tasks.put_nowait((clip, detections))
        except queue.Full:
            return False
        self.queue_depth_max = max(self.queue_depth_max, self.tasks.qsize())
        return True

    def _run(self) -> None:
        while True:
            task = self.tasks.get()
            if task is self._sentinel:
                break
            clip, detections = task
            self.reviewer.detections = detections
            t0 = time.perf_counter()
            result = self.reviewer.review(clip)
            elapsed = time.perf_counter() - t0
            self.process_s.append(elapsed)
            self.processed += 1
            self.results.put((clip.clip_id, result, time.monotonic()))
        self.finished_wall = time.monotonic()

    def drain(self) -> List[Tuple[str, Any, float]]:
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        if self._closed:
            return
        self.tasks.put(self._sentinel)
        if self._thread is not None:
            self._thread.join()
        self._closed = True

    @property
    def pending(self) -> int:
        return self.tasks.qsize()

    def stats(self) -> Dict[str, Any]:
        wall = ((self.finished_wall or time.monotonic()) - (self.started_wall or time.monotonic()))
        return {
            "processed": self.processed,
            "fps": round(self.processed / wall, 3) if wall > 0 else None,
            "process_ms_p50": None if not self.process_s else round(_pct(self.process_s, 0.5) * 1000, 3),
            "process_ms_p95": None if not self.process_s else round(_pct(self.process_s, 0.95) * 1000, 3),
            "queue_depth_max": self.queue_depth_max,
        }


def make_yolo_predictor(weights: str, calib, imgsz: int = 960, conf: float = 0.18,
                        device: Optional[str] = None):
    """构造真模型 YOLO 预测器（只有原机需要调用）。"""
    root = os.path.abspath(YOLO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from adapters.yolo_adapter import YOLOAdapter
    adapter = YOLOAdapter(weights, calibration=calib, imgsz=imgsz,
                          conf_threshold=conf, device=device, strict_time=True)

    def predict(frame, frame_id: int, timestamp_s: float) -> Dict[str, Any]:
        obs = adapter.predict_frame(frame, frame_id, timestamp_s)
        return obs.to_dict() if hasattr(obs, "to_dict") else obs
    return predict


def make_tracknet_batch_predictor(weights: str, device: Optional[str] = None,
                                  min_conf: float = 0.5, history_frames: int = 32):
    """构造流式 TrackNet 批预测器；三帧顺序与 run_video 一致。"""
    import cv2
    import torch
    from adapter.tracknet_adapter import (BallTrackerNet, MODEL_H, MODEL_W, SPACE_SCALE,
                                           _heatmap_peak, make_observation, postprocess)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = BallTrackerNet()
    model.load_state_dict(torch.load(weights, map_location=dev))
    model = model.to(dev).eval()
    # 要覆盖“入队到推理”的最大积压帧数，否则 t-1、t-2 已被淘汰
    history = ConsecutiveFrames(lambda bgr: cv2.resize(bgr, (MODEL_W, MODEL_H)), keep=history_frames)

    def predict_batch(batch: List[Tuple[int, float, Any]]) -> List[Dict[str, Any]]:
        pending, outputs = [], []
        for fid, ts, frame in batch:
            h, w = frame.shape[:2]
            history.observe(fid, frame)        # 没经过 frame_hook 的调用方也能用
            three = history.window(fid)        # (t, t-1, t-2)，与 run_video 一致；缺帧则 None
            if three is None:
                outputs.append(make_observation(fid, ts, "tracknet", (w, h)))
                continue
            stack = np.concatenate(three, axis=2).astype(np.float32) / 255.0
            pending.append((fid, ts, w, h, np.rollaxis(stack, 2, 0)))
        if pending:
            arr = np.stack([p[4] for p in pending])
            with torch.no_grad():
                logits = model(torch.from_numpy(arr).float().to(dev))
            maps = logits.argmax(dim=1).detach().cpu().numpy()
            for (fid, ts, w, h, _), cm in zip(pending, maps):
                x, y = postprocess(cm.copy(), height=MODEL_H, width=MODEL_W,
                                   min_radius=2, max_radius=12)
                if x is None:
                    outputs.append(make_observation(fid, ts, "tracknet", (w, h)))
                    continue
                conf = _heatmap_peak(cm, x, y)
                if min_conf > 0 and conf < min_conf:
                    outputs.append(make_observation(fid, ts, "tracknet", (w, h)))
                    continue
                sx, sy = w / (MODEL_W * SPACE_SCALE), h / (MODEL_H * SPACE_SCALE)
                outputs.append(make_observation(fid, ts, "tracknet", (w, h),
                                                center_px=(x * sx, y * sy), confidence=conf))
        outputs.sort(key=lambda o: int(o["frame_id"]))
        return outputs
    predict_batch.observe = history.observe
    return predict_batch


def make_replay_ttnet_reviewer(cache: str, poly, fps: float = 120.0, k: int = 4,
                               per_window_s: float = 0.0) -> TTNetReviewer:
    from cascade.ttnet_reviewer import TableGeometry
    scores = ReplayScores(cache, per_window_s=per_window_s)
    return TTNetReviewer(scores, {}, TableGeometry(poly), k=k, fps=fps)


def make_live_ttnet_reviewer(weights: str, ring, poly, fps: float = 120.0, k: int = 4,
                             ttnet_src: Optional[str] = None, engine: str = "auto",
                             device: str = "cuda:0") -> TTNetReviewer:
    from cascade._paths import TTNET_ROOT
    from cascade.ttnet_engine import make_engine
    from cascade.ttnet_reviewer import LiveScores, TableGeometry
    eng = make_engine(engine, weights, ttnet_src or os.path.join(TTNET_ROOT, "src"), device)
    scores = LiveScores(eng, ring.get, lambda fid: fid / fps)
    return TTNetReviewer(scores, {}, TableGeometry(poly), k=k, fps=fps)
