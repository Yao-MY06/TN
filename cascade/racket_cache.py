"""视频 → 逐帧球拍观测缓存（racket.v1 JSONL），原机用。

球拍模型来自朗然的仓库 ``pingpong-racket-detector``（按 ``_paths.find_repo`` 查找，不复制、不修改）。
离线把每段视频的每一帧（或按 ``--stride`` 抽帧）都跑一遍，写成缓存，后续击球证据的评估与
提交层接入都在本机读缓存完成——和 TTNet 的 ``ttnet_cache`` 同一思路。

侧别：球拍中心相对**本机位的网线 x**（来自 TN 的球台标定，不用球拍仓库默认的画面正中）判 A/B，
压线 ±2% 画面宽度判 UNKNOWN。A = 画面左半台，与 TN 的物理侧一致。
多类别权重（如 v11：racket_A / racket_B / racket_unknown）的适配器会优先用类别名定侧别，
不回落到网线；缓存写出前一律按网线重算 ``side``，模型自己给的侧别另存为 ``model_side``。
已有缓存可以用 ``--reside`` 离线重算，不必重跑模型。

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


def geometric_side(record: Dict[str, Any], net_x: float, dead_zone_ratio: float = 0.02) -> str:
    """按网线 x 判侧别（与球拍仓库 side_from_x 同一规则：压线 ±dead_zone 判 UNKNOWN）。"""
    center = record.get("center_px")
    width = (record.get("image_size") or [0, 0])[0]
    if not center or not width:
        return record.get("side") or "UNKNOWN"
    cx = float(center[0])
    if abs(cx - float(net_x)) <= float(width) * dead_zone_ratio:
        return "UNKNOWN"
    return "A" if cx < float(net_x) else "B"


def apply_geometric_side(record: Dict[str, Any], net_x: float,
                         dead_zone_ratio: float = 0.02) -> Dict[str, Any]:
    out = dict(record)
    out.setdefault("model_side", record.get("side"))
    out["side"] = geometric_side(record, net_x, dead_zone_ratio)
    return out


def reside_file(src: str, dst: str, net_x: float, dead_zone_ratio: float = 0.02) -> Dict[str, Any]:
    """已有缓存按网线重算侧别，写到 dst；返回改动统计。"""
    n = changed = 0
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            d = json.loads(line)
            new = apply_geometric_side(d, net_x, dead_zone_ratio)
            n += 1
            changed += int(new["side"] != d.get("side"))
            fout.write(json.dumps(new, ensure_ascii=False) + "\n")
    stats = {"src": os.path.abspath(src), "net_x": net_x, "records": n, "side_changed": changed}
    meta_src = src + ".meta.json"
    meta = {}
    if os.path.isfile(meta_src):
        with open(meta_src, encoding="utf-8") as f:
            meta = json.load(f)
    meta["reside"] = stats
    meta["net_x"] = net_x
    with open(dst + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return stats


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
                        rec = apply_geometric_side(obs.to_dict(), net_x)
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
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
    ap.add_argument("--video", default=None)
    ap.add_argument("--reside", default=None,
                    help="不跑模型：把已有缓存按网线重算侧别，写到 --out（v11 等多类别权重的缓存用）")
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
    if a.reside:
        stats = reside_file(a.reside, a.out, net_x_for(a.preset, a.quad))
        print(json.dumps(stats, ensure_ascii=False))
        return 0
    if not a.video:
        ap.error("需要 --video（或用 --reside 只重算侧别）")
    run(a.video, a.out, preset=a.preset, quad=a.quad, weights=a.weights, imgsz=a.imgsz,
        conf=a.conf, stride=a.stride, max_frames=a.max_frames, device=a.device, quiet=a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
