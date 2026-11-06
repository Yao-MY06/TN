"""§7 离线漏检统计：全片 TTNet 驼峰中，没有被任何复核片段覆盖的真值事件。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cascade.ttnet_reviewer import BOUNCE_HIGH, BOUNCE_LOW, DEFAULT_K, check_k, find_humps


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_truth(path: str) -> List[int]:
    with open(path, encoding="utf-8") as f:
        markup = {int(k): v for k, v in json.load(f).items()}
    return sorted(f for f, label in markup.items() if label == "bounce")


def evaluate(ttnet_cache: str, clips_path: str, events_markup: str, k: int = DEFAULT_K,
             fps: float = 120.0, high: float = BOUNCE_HIGH,
             low: float = BOUNCE_LOW, tol: int = 5) -> Dict[str, Any]:
    """找峰值落在所有片段之外的 TTNet 驼峰，并标出与真值落台匹配者。"""
    k = check_k(k)
    rows = _read_jsonl(ttnet_cache)
    scores: Dict[int, float] = {}
    for row in rows:
        if "window_end" not in row or "bounce" not in row:
            continue
        event_frame = int(row["window_end"]) - k
        scores[event_frame] = float(row["bounce"])
    frames = sorted(scores)
    humps = find_humps(frames, [scores[f] for f in frames], high=high, low=low)

    clips = _read_jsonl(clips_path)
    spans = [(int(c["start_frame_id"]), int(c["end_frame_id"])) for c in clips
             if c.get("start_frame_id") is not None and c.get("end_frame_id") is not None]
    truth = _load_truth(events_markup)

    def covered(frame: int) -> bool:
        return any(a <= frame <= b for a, b in spans)

    outside = [h for h in humps if not covered(int(h.peak_frame))]
    used_truth, rows_out = set(), []
    for hump in outside:
        peak_frame = int(hump.peak_frame)
        matches = [(abs(g - peak_frame), i) for i, g in enumerate(truth)
                   if i not in used_truth and abs(g - peak_frame) <= tol]
        matched = bool(matches)
        truth_frame = None
        if matches:
            _, i = min(matches)
            used_truth.add(i)
            truth_frame = truth[i]
        rows_out.append({
            "peak_frame": peak_frame,
            "peak_timestamp_s": round(peak_frame / float(fps), 6),
            "peak": round(float(hump.peak), 6),
            "start_frame": int(hump.start),
            "end_frame": int(hump.end),
            "truncated": bool(hump.truncated),
            "truth_matched": matched,
            "truth_frame": truth_frame,
        })

    return {
        "ttnet_cache": os.path.basename(ttnet_cache),
        "clips": os.path.basename(clips_path),
        "events_markup": os.path.basename(events_markup),
        "k": k,
        "high": float(high),
        "low": float(low),
        "tol": int(tol),
        "humps_total": len(humps),
        "humps_outside": len(outside),
        "outside_truth_matches": sum(row["truth_matched"] for row in rows_out),
        "outside_rows": rows_out,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="统计片段外、但 TTNet 能认出的真值落台")
    ap.add_argument("--ttnet-cache", required=True)
    ap.add_argument("--clips", required=True, help="一次 run_cascade 的 clips.jsonl")
    ap.add_argument("--events-markup", required=True)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--fps", type=float, default=120.0)
    ap.add_argument("--high", type=float, default=BOUNCE_HIGH)
    ap.add_argument("--low", type=float, default=BOUNCE_LOW)
    ap.add_argument("--tol", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    result = evaluate(a.ttnet_cache, a.clips, a.events_markup, a.k, a.fps, a.high, a.low, a.tol)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
