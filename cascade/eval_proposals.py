"""第 4 步：用 OpenTTGames 标注评估提议层与复核结果。

    python -m cascade.eval_proposals --run-dir outputs/cascade/test_2_rule \\
        --events-markup <test_2 的 events_markup.json> \\
        --jsonl adapter/examples/test_2_observations.jsonl

输出：
  * 提议召回：真值落台帧落在某个复核片段 [start_frame, end_frame] 内的比例（TTNet 只给片段内的帧打分）
  * 漏提议明细：每条没被覆盖的真值落台，附最近锚点距离、±12 帧内 TrackNet 真实检出数
  * 复核覆盖率、TTNet 窗口占比（取自 summary.json）
  * 确认落台 P / R / F1：±5 帧（TN 口径）与 ±12 帧（≈ TTNet 报告的 ±0.1 s），匹配算法同 detect_bounce.evaluate

只统计标注覆盖到的范围（帧号 < 最大标注帧 + 1）：OpenTTGames 标注不覆盖整段视频，
范围外的输出不算 FP（TTNet 报告 3.3 节踩过这个坑）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from bisect import bisect_left
from typing import Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def match(pred: List[int], gt: List[int], tol: int) -> Dict:
    """与 detect_bounce.evaluate 相同的贪心最近匹配。"""
    used, tp = set(), 0
    for f in sorted(pred):
        best, bd = None, 1e9
        for i, g in enumerate(gt):
            if i in used:
                continue
            d = abs(g - f)
            if d <= tol and d < bd:
                best, bd = i, d
        if best is not None:
            used.add(best)
            tp += 1
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gt) if gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": len(pred) - tp, "fn": len(gt) - tp, "n_pred": len(pred),
            "n_gt": len(gt), "p": round(p, 4), "r": round(r, 4), "f1": round(f1, 4)}


# OpenTTGames 测试集只标了部分区间：test_1 / test_4 / test_7 各有一段 47~170 s 的空档，
# 里面有真实回合但没有任何事件标注。空档里的检出既不能算对也不能算错，因此 P/R/F1
# 另给一份只在标注区间内统计的版本。相邻两条标注间隔超过 UNANNOTATED_GAP_S 视为空档；
# 正常回合之间的死球时间（test_2～test_6 最长约 14 s）不会被切开。
UNANNOTATED_GAP_S = 30.0
SPAN_PAD_S = 1.0


def annotated_spans(markup: Dict[int, str], fps: float = 120.0,
                    gap_s: float = UNANNOTATED_GAP_S, pad_s: float = SPAN_PAD_S) -> List[List[int]]:
    """按标注帧聚类出"有人标过"的区间（闭区间，前后各外扩 pad_s）。"""
    ks = sorted(markup)
    if not ks:
        return []
    gap, pad = gap_s * fps, int(round(pad_s * fps))
    spans = [[ks[0], ks[0]]]
    for k in ks[1:]:
        if k - spans[-1][1] <= gap:
            spans[-1][1] = k
        else:
            spans.append([k, k])
    return [[max(0, a - pad), b + pad] for a, b in spans]


def evaluate(run_dir: str, markup_path: str, jsonl: Optional[str] = None,
             fps: float = 120.0, tol: int = 5) -> Dict:
    with open(markup_path, encoding="utf-8") as f:
        markup = {int(k): v for k, v in json.load(f).items()}
    max_frame = max(markup) + 1
    gt_bounce = sorted(f for f, v in markup.items() if v == "bounce")
    gt_net = sorted(f for f, v in markup.items() if v == "net")

    clips = _read_jsonl(os.path.join(run_dir, "clips.jsonl"))
    props = _read_jsonl(os.path.join(run_dir, "proposals.jsonl"))
    reviews = _read_jsonl(os.path.join(run_dir, "reviews.jsonl"))
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    spans = sorted((c["start_frame_id"], c["end_frame_id"]) for c in clips)
    anchors = sorted(p["anchor_frame_id"] for p in props)

    def covered(fr: int) -> bool:
        return any(a <= fr <= b for a, b in spans)

    vis = None
    if jsonl:
        vis = sorted(o["frame_id"] for o in _read_jsonl(jsonl)
                     if o["ball"]["visible"] and not o["ball"].get("predicted", False))

    missed = []
    for g in gt_bounce:
        if covered(g):
            continue
        i = bisect_left(anchors, g)
        near = [abs(anchors[j] - g) for j in (i - 1, i) if 0 <= j < len(anchors)]
        row = {"frame": g, "t": round(g / fps, 3), "nearest_anchor_frames": min(near) if near else None}
        if vis is not None:
            row["tracknet_visible_pm12"] = bisect_left(vis, g + 13) - bisect_left(vis, g - 12)
        missed.append(row)

    confirmed = sorted(round(e["timestamp_s"] * fps) for r in reviews for e in r["events"]
                       if e["type"] == "BOUNCE")
    confirmed = [f for f in confirmed if f < max_frame]

    spans_gt = annotated_spans(markup, fps)
    confirmed_in = [f for f in confirmed if any(a <= f <= b for a, b in spans_gt)]

    n_cov = len(gt_bounce) - len(missed)
    out = {
        "run_dir": run_dir, "markup": os.path.basename(markup_path), "annotated_frames": max_frame,
        "gt_bounce": len(gt_bounce), "gt_net": len(gt_net),
        "proposal_recall_bounce": round(n_cov / len(gt_bounce), 4) if gt_bounce else None,
        "proposal_recall_net": round(sum(covered(g) for g in gt_net) / len(gt_net), 4) if gt_net else None,
        "anchor_recall_bounce_tol": round(sum(1 for g in gt_bounce
                                              if any(abs(a - g) <= tol for a in anchors)) / max(1, len(gt_bounce)), 4),
        "missed_bounces": missed,
        "coverage": summary["clips"]["coverage"],
        "ttnet_window_ratio": summary["clips"]["ttnet_window_ratio"],
        "confirmed_bounce_tol5": match(confirmed, gt_bounce, 5),
        "confirmed_bounce_tol12": match(confirmed, gt_bounce, 12),
        "annotated_spans": spans_gt,
        "confirmed_outside_markup": len(confirmed) - len(confirmed_in),
        "confirmed_bounce_tol5_in_markup": match(confirmed_in, gt_bounce, 5),
        "score": summary["score"], "decisions": summary["decisions"],
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="用 events_markup.json 评估级联提议与复核")
    ap.add_argument("--run-dir", required=True, help="run_cascade --out-dir 的目录")
    ap.add_argument("--events-markup", required=True)
    ap.add_argument("--jsonl", default=None, help="可选：TrackNet 观测，用来分析漏提议处的可见性")
    ap.add_argument("--fps", type=float, default=120.0)
    a = ap.parse_args(argv)
    r = evaluate(a.run_dir, a.events_markup, a.jsonl, a.fps)
    with open(os.path.join(a.run_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    c5, c12 = r["confirmed_bounce_tol5"], r["confirmed_bounce_tol12"]
    print(f"标注范围        : 前 {r['annotated_frames']} 帧，真值落台 {r['gt_bounce']}、过网 {r['gt_net']}")
    print(f"提议召回        : 落台 {r['proposal_recall_bounce']}  过网 {r['proposal_recall_net']}"
          f"  （锚点 ±5 帧召回 {r['anchor_recall_bounce_tol']}）")
    print(f"覆盖 / 窗口占比 : {r['coverage'] * 100:.1f}% / {r['ttnet_window_ratio'] * 100:.1f}%")
    print(f"确认落台 ±5 帧  : P {c5['p']:.3f}  R {c5['r']:.3f}  F1 {c5['f1']:.3f}  ({c5['tp']}/{c5['n_pred']}/{c5['n_gt']})")
    print(f"确认落台 ±12 帧 : P {c12['p']:.3f}  R {c12['r']:.3f}  F1 {c12['f1']:.3f}")
    ci = r["confirmed_bounce_tol5_in_markup"]
    print(f"仅标注区间 ±5 帧: P {ci['p']:.3f}  R {ci['r']:.3f}  F1 {ci['f1']:.3f}"
          f"  （{len(r['annotated_spans'])} 段；空档内检出 {r['confirmed_outside_markup']} 条不计）")
    if r["missed_bounces"]:
        print(f"漏提议 {len(r['missed_bounces'])} 条：")
        for m in r["missed_bounces"]:
            print(f"   - {m}")
    print(f"已写入 {os.path.join(a.run_dir, 'eval.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
