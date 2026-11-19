"""固定机位视频抽帧：开播前自动取若干清晰帧做标定。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class SampledFrame:
    frame_id: int
    timestamp_s: float
    sharpness: float
    image: np.ndarray


def evenly_spaced_indices(start: int, end: int, count: int) -> List[int]:
    """返回 [start, end) 内均匀分布的 count 个整数帧号。"""
    start, end = int(start), int(end)
    count = int(count)
    if count <= 0:
        return []
    if end <= start:
        return []
    if count == 1:
        return [start + (end - start) // 2]
    span = end - start
    return sorted({start + int(round(i * (span - 1) / (count - 1))) for i in range(count)})


def sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sample_video_frames(
    video: str,
    *,
    count: int = 8,
    start_s: float = 0.0,
    window_s: float = 10.0,
    candidate_factor: int = 3,
    min_sharpness: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], List[SampledFrame]]:
    """从视频的 [start_s, start_s+window_s) 抽 count 个最清晰帧。

    返回 (帧号字符串 → BGR 图像, 采样明细)。视频短于窗口时自动截到片尾。
    """
    if count <= 0:
        raise ValueError("count 必须 > 0")
    if start_s < 0 or window_s <= 0:
        raise ValueError("start_s 必须 >= 0，window_s 必须 > 0")
    if candidate_factor < 1:
        raise ValueError("candidate_factor 必须 >= 1")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"打不开视频：{video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise ValueError(f"视频没有可读帧：{video}")
    start = max(0, min(total - 1, int(round(float(start_s) * fps))))
    end = min(total, max(start + 1, int(round((float(start_s) + float(window_s)) * fps))))
    candidates = evenly_spaced_indices(start, end, min(end - start, max(count, count * candidate_factor)))
    found: List[SampledFrame] = []
    try:
        for fid in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, img = cap.read()
            if not ok or img is None:
                continue
            s = sharpness(img)
            if s >= float(min_sharpness):
                found.append(SampledFrame(fid, fid / fps, s, img))
    finally:
        cap.release()
    found.sort(key=lambda x: (-x.sharpness, x.frame_id))
    chosen = sorted(found[:count], key=lambda x: x.frame_id)
    frames = {f"f{x.frame_id:06d}": x.image for x in chosen}
    return frames, chosen