"""视频 → 逐帧球拍观测缓存（racket.v1 JSONL），原机用。

球拍模型来自朗然的仓库 ``pingpong-racket-detector``（按 ``_paths.find_repo`` 查找，不复制、不修改）。
离线把每段视频的每一帧（或按 ``--stride`` 抽帧）都跑一遍，写成缓存，后续击球证据的评估与
提交层接入都在本机读缓存完成——和 TTNet 的 ``ttnet_cache`` 同一思路。

侧别：球拍中心相对**本机位的网线 x**（来自 TN 的球台标定，不用球拍仓库默认的画面正中）判 A/B，
压线 ±2% 画面宽度判 UNKNOWN。A = 画面左半台，与 TN 的物理侧一致。

    python -m cascade.racket_cache --video <test_2.mp4> --preset test2 --out _data/racket/test_2.jsonl

输出：
* ``<out>``            每行一条 racket.v1 观测（visible=True 的检出，以及跟踪中丢失时的 visible=False）
* ``<out>.meta.json``  帧数、抽帧步长、阈值、输入尺寸、网线 x、权重 sha256、单帧耗时 P50/P95
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from ._paths import find_repo

RACKET_ROOT = find_repo("pingpong-racket-detector")
DEFAULT_WEIGHTS = os.path.join(RACKET_ROOT, "weights", "racket_v8_960.pt")


def ensure_racket_path(root: str = RACKET_ROOT) -> str:
    root = os.path.abspath(root)
    if not os.path.isdir(os.path.join(root, "racket")):
        raise FileNotFoundError(
            f"找不到球拍仓库：{root}（按 cascade/_paths.py 的顺序查找 pingpong-racket-detector，"
            "或设环境变量 PPQ_ROOT）")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def net_x_for(preset: str, quad: Optional[str]) -> float:
    from .run_realtime import _clip_geometry
    _, calib = _clip_geometry(preset, quad)
    return float(calib.net_x)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * len(s)))], 3)


def run(video: str, out: str, *, preset: str = "test2", quad: Optional[str] = None,
        weights: str = DEFAULT_WEIGHTS, imgsz: int = 960, conf: float = 0.25,
        stride: int = 1, max_frames: Optional[int] = None, device: str = "auto",
        session_id: str = "s01", detector: Any = None, net_x: Optional[float] = None,
        quiet: bool = False) -> Dict[str, Any]:
    """跑一段视频。``detector`` 仅测试用（注入假检测器）；``net_x`` 缺省时由标定给出。"""
    import cv2

    ensure_racket_path()
    from racket.adapter import AdapterConfig, RacketAdapter

    if stride < 1:
        raise ValueError("stride 必须 >= 1")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"打不开视频：{video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 120.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if net_x is None:
        net_x = net_x_for(preset, quad)

    weights_sha = None
    if detector is None:
        from racket.detectors import YoloRacketDetector
        from racket.detectors.base import DetectorConfig
        detector = YoloRacketDetector(weights, DetectorConfig(conf_thr=conf, imgsz=imgsz, device=device))
        weights_sha = _sha256(weights)
    model_version = os.path.splitext(os.path.basename(weights))[0]
    adapter = RacketAdapter(detector, AdapterConfig(
        session_id=session_id, conf_thr=conf, net_x_ratio=net_x / max(1, width),
        model_version=model_version))

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    infer_ms: List[float] = []
    frames_read = frames_processed = n_obs = n_visible = 0
    started = time.monotonic()
    try:
        with open(out, "w", encoding="utf-8") as f:
            frame_id = 0
            while True:
                if max_frames is not None and frame_id >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                frames_read += 1
                if frame_id % stride == 0:
                    t0 = time.perf_counter()
                    observations = adapter.process_frame(frame, frame_id, frame_id / fps, (width, height))
                    infer_ms.append((time.perf_counter() - t0) * 1000.0)
                    frames_processed += 1
                    for obs in observations:
                        f.write(json.dumps(obs.to_dict(), ensure_ascii=False) + "\n")
                        n_obs += 1
                        n_visible += int(bool(obs.visible))
                    if not quiet and frames_processed % 1200 == 0:
                        print(f"  {frame_id} 帧，检出 {n_visible}", flush=True)
                frame_id += 1
    finally:
        cap.release()
        adapter.close()

    meta = {
        "video": os.path.abspath(video), "fps": fps, "image_size": [width, height],
        "frames_read": frames_read, "frames_processed": frames_processed, "stride": stride,
        "preset": preset, "quad": quad, "net_x": net_x,
        "weights": os.path.abspath(weights), "weights_sha256": weights_sha,
        "model_version": model_version, "imgsz": imgsz, "conf_thr": conf, "device": device,
        "observations": n_obs, "visible": n_visible,
        "infer_ms_p50": _pct(infer_ms, 0.5), "infer_ms_p95": _pct(infer_ms, 0.95),
        "wall_s": round(time.monotonic() - started, 3),
        "adapter": adapter.summary(),
    }
    with open(out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if not quiet:
        print(json.dumps({k: meta[k] for k in ("frames_processed", "visible", "infer_ms_p50",
                                                "infer_ms_p95", "wall_s")}, ensure_ascii=False))
    return meta


def load_racket_obs(path: str, visible_only: bool = True) -> Dict[int, List[Dict[str, Any]]]:
    """读缓存：frame_id → 该帧的观测列表。"""
    out: Dict[int, List[Dict[str, Any]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if visible_only and not d.get("visible"):
                continue
            out.setdefault(int(d["frame_id"]), []).append(d)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="视频 → 逐帧球拍观测缓存（racket.v1 JSONL）")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="test2")
    ap.add_argument("--quad", default=None)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="缓存用低阈值保留更多检出，评估时再按置信度筛（模型部署阈值是 0.5）")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    run(a.video, a.out, preset=a.preset, quad=a.quad, weights=a.weights, imgsz=a.imgsz,
        conf=a.conf, stride=a.stride, max_frames=a.max_frames, device=a.device, quiet=a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
