"""把一次标定的结果写成文件：result.json、frames.csv、overlay.jpg、corners.png。"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, Optional

import cv2
import numpy as np

from .calibrate import CalibResult
from .table import CORNER_NAMES
from .viz import corner_crops, draw_overlay

SIDE_NAMES = ("far", "right", "near", "left")


def write_frames_csv(path: str, res: CalibResult) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        head = ["frame", "ok", "reason", "used"]
        head += [f"{c}_{a}" for c in CORNER_NAMES for a in "xy"]
        head += [f"{s}_inlier" for s in SIDE_NAMES]
        wr.writerow(head)
        for k in res.frame_ids:
            r = res.frame_results[k]
            row = [k, int(r.ok), r.reason, int(k in res.used_ids)]
            row += [round(float(v), 2) for v in r.corners.reshape(-1)] if r.corners is not None else [""] * 8
            row += [round(r.sides[s].inlier_ratio, 3) if s in r.sides else "" for s in SIDE_NAMES]
            wr.writerow(row)


def write_outputs(out_dir: str, d: Dict, res: CalibResult, frames: Dict[str, np.ndarray]) -> Optional[str]:
    """写 result.json / frames.csv / overlay.jpg / corners.png。返回叠加图所用的帧号。"""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    write_frames_csv(os.path.join(out_dir, "frames.csv"), res)
    if not frames:
        return None
    # 叠加图取参与汇总的中间一帧；没有可用帧时取第一帧（会画出 NO RESULT）
    show_id = res.used_ids[len(res.used_ids) // 2] if res.used_ids else sorted(frames)[0]
    img = frames[show_id]
    cv2.imwrite(os.path.join(out_dir, "overlay.jpg"), draw_overlay(img, d), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if d.get("points"):
        cv2.imwrite(os.path.join(out_dir, "corners.png"), corner_crops(img, {"auto": d["points"]}))
    return show_id
