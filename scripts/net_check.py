"""批量检查网顶白带检测。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tcalib.calibrate import calibrate_frames  # noqa: E402


def main() -> None:
    frames_root = Path(ROOT) / "frames"
    out_root = Path(ROOT) / "out" / "net_check"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"{'视频':<12} {'来源':<10} {'覆盖率':>8} {'偏差(px)':>10}")
    for folder in sorted(path for path in frames_root.iterdir() if path.is_dir()):
        paths = sorted(folder.glob("*.png"))
        frames = {path.stem: cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR) for path in paths}
        result = calibrate_frames(frames)
        net = result.net
        stats = net.get("net_top_stats", {})
        offset = stats.get("offset_px")
        print(f"{folder.name:<12} {net.get('net_top_source', '-'): <10} {stats.get('coverage', 0):>8.3f} "
              f"{offset if offset is not None else float('nan'):>10.2f}")
        image = frames[paths[0].stem].copy()
        projected = net.get("net_top_line_projected")
        detected = net.get("net_top_line")
        lines = [np.asarray(v, dtype=float) for v in (projected, detected) if v is not None]
        pts = np.vstack(lines)
        x0, y0 = np.floor(pts.min(axis=0) - [35, 35]).astype(int)
        x1, y1 = np.ceil(pts.max(axis=0) + [35, 35]).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, image.shape[1]), min(y1, image.shape[0])
        if projected is not None:
            a, b = np.round(projected).astype(int)
            cv2.line(image, tuple(a), tuple(b), (0, 255, 0), 2, cv2.LINE_AA)
        if detected is not None:
            a, b = np.round(detected).astype(int)
            cv2.line(image, tuple(a), tuple(b), (255, 0, 255), 2, cv2.LINE_AA)
        crop = image[y0:y1, x0:x1]
        zoom = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        cv2.imencode(".jpg", zoom)[1].tofile(out_root / f"{folder.name}.jpg")


if __name__ == "__main__":
    main()
