"""固定机位视频抽帧。

两种取法：
  * opening：从 start_s 起 window_s 秒内均匀取帧（开场标定，开播前用）；
  * spread ：在整段视频 2%～98% 之间均匀取帧（离线验收用：帧之间差别大，
    球员站位、遮挡各不相同，多帧抖动才有意义）。
不按清晰度挑帧：挑帧会让多帧统计偏乐观；清晰度只记录下来供排查。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class SampledFrame:
    frame_id: int
    timestamp_s: float
    sharpness: float
    image: np.ndarray = field(repr=False)


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    seek_mismatch: List[int] = field(default_factory=list)  # 定位后读到的不是目标帧的帧号
    unreadable: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "seek_mismatch": self.seek_mismatch,
            "unreadable": self.unreadable,
        }


def evenly_spaced_indices(start: int, end: int, count: int) -> List[int]:
    """返回 [start, end) 内均匀分布的至多 count 个整数帧号（去重、升序）。"""
    start, end, count = int(start), int(end), int(count)
    if count <= 0 or end <= start:
        return []
    if count == 1:
        return [start + (end - start) // 2]
    span = end - start
    return sorted({start + int(round(i * (span - 1) / (count - 1))) for i in range(count)})


def sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def open_video(path: str) -> Tuple[cv2.VideoCapture, VideoInfo]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"打不开视频：{path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if total <= 0:
        cap.release()
        raise ValueError(f"视频没有可读帧：{path}")
    if fps <= 0:
        fps = 30.0
    return cap, VideoInfo(path, w, h, fps, total)


def read_frames(cap: cv2.VideoCapture, info: VideoInfo, ids: Sequence[int]) -> List[SampledFrame]:
    """按帧号读帧。

    读完后按解码器报告的位置核对（应为目标帧号 + 1），对不上的记进 info.seek_mismatch（仍保留该帧）。
    这只能发现后端自己知道的偏差；个别编码的视频定位不准而后端不报，需要靠画回原图目检。
    """
    out: List[SampledFrame] = []
    for fid in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, img = cap.read()
        if not ok or img is None:
            info.unreadable.append(int(fid))
            continue
        pos = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        if pos != int(fid) + 1:
            info.seek_mismatch.append(int(fid))
        out.append(SampledFrame(int(fid), fid / info.fps, sharpness(img), img))
    return out


def sample_ids(info: VideoInfo, mode: str, count: int, start_s: float = 0.0, window_s: float = 10.0) -> List[int]:
    if count <= 0:
        raise ValueError("count 必须 > 0")
    total = info.frame_count
    if mode == "spread":
        start = int(total * 0.02)
        end = max(int(total * 0.98), start + 1)
    elif mode == "opening":
        if start_s < 0 or window_s <= 0:
            raise ValueError("start_s 必须 >= 0，window_s 必须 > 0")
        start = max(0, min(total - 1, int(round(start_s * info.fps))))
        end = min(total, max(start + 1, int(round((start_s + window_s) * info.fps))))
    else:
        raise ValueError(f"未知抽帧方式 {mode!r}（opening / spread）")
    return evenly_spaced_indices(start, end, min(count, end - start))


def sample_video_frames(
    video: str,
    *,
    mode: str = "opening",
    count: int = 24,
    start_s: float = 0.0,
    window_s: float = 10.0,
    extra_ids: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, np.ndarray], List[SampledFrame], VideoInfo, Dict[int, np.ndarray]]:
    """返回 (帧号字符串 → 图像, 采样明细, 视频信息, 额外帧号 → 图像)。

    extra_ids 只读出来供保存（比如核对真值用的帧），不参与标定。
    """
    cap, info = open_video(video)
    try:
        ids = sample_ids(info, mode, count, start_s, window_s)
        sampled = read_frames(cap, info, ids)
        extras = {x.frame_id: x.image for x in read_frames(cap, info, list(extra_ids or []))}
    finally:
        cap.release()
    frames = {f"f{x.frame_id:06d}": x.image for x in sampled}
    return frames, sampled, info, extras
