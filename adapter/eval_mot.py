"""用行业标准工具（py-motmetrics）评估球轨迹，不自己写指标。

输入两边都是 MOT Challenge 格式：
  * 预测：observation.write_mot() 导出的 <seq>.txt
  * 真值：本脚本从 OpenTTGames 的 ball_markup.json 现场转换

为什么值得这么做
----------------
自己算 Precision/Recall 时，"什么算命中"是自定义的（我们用 20px）；
换成 MOT 指标后，MOTA / IDF1 / MOTP 都是社区公认定义，
队友的 YOLO、学长的 TTNet 只要导出同一份 MOT txt 就能横向对比。

用法
----
python eval_mot.py --pred examples/test_2_mot.txt \
                   --gt-json E:/.../test_2_extracted/ball_markup.json \
                   --max-frames 3600 --max-iou 0.3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import motmetrics as mm
import numpy as np

# motmetrics 1.4.0 仍在用 numpy 1.x 的 np.asfarray（numpy 2.0 已移除）。
# 补一个等价实现，避免为了跑评估去降级 numpy（会和 torch 冲突）。
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # noqa: E731

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import observation as O  # noqa: E402


def load_mot(path):
    """返回 {frame(1-based): [(id, x, y, w, h, conf), ...]}。"""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = [float(v) for v in line.split(",")]
            out.setdefault(int(p[0]), []).append((int(p[1]), p[2], p[3], p[4], p[5], p[6]))
    return out


def gt_to_mot(ball_markup_path, max_frames=None, radius=23.0):
    """OpenTTGames ball_markup.json -> ({frame: rows}, {标注帧集合})。

    关键：只评估**被标注过的帧**（OpenTTGames 是事件中心标注，全片只有约 10%
    的帧有标注，其余帧球在不在画面里是未知的）。标注里 (-1,-1) 表示"这一帧
    明确没有球"，保留为空 GT —— 在这些帧上的检测应算误检。

    返回的 frame 是 1-based（MOT 约定）。
    """
    with open(ball_markup_path, "r", encoding="utf-8") as f:
        marks = json.load(f)

    rows_by_frame, annotated = {}, set()
    for k, v in marks.items():
        frame0 = int(k)
        if max_frames is not None and frame0 >= max_frames:
            continue
        frame1 = frame0 + 1
        annotated.add(frame1)
        rows_by_frame.setdefault(frame1, [])

        x, y = v.get("x"), v.get("y")
        if x is None or y is None:
            continue
        x, y = float(x), float(y)
        if x < 0 or y < 0:                      # (-1,-1) = 无球
            continue
        rows_by_frame[frame1].append(
            (1, x - radius, y - radius, 2 * radius, 2 * radius, 1.0))
    return rows_by_frame, annotated


def run(args):
    pred = load_mot(args.pred)
    gt, annotated = gt_to_mot(args.gt_json, args.max_frames, args.ball_radius)

    # 只在标注帧上评估：未标注帧的预测既不算对也不算错
    frames = sorted(annotated)
    print(f"evaluating on {len(frames)} annotated frames "
          f"(predictions in that range: {sum(1 for f in frames if f in pred)})")

    acc = mm.MOTAccumulator(auto_id=True)
    for f in frames:
        g = gt.get(f, [])
        p = pred.get(f, [])
        gt_ids = [r[0] for r in g]
        pr_ids = [r[0] for r in p]
        gt_boxes = np.array([[r[1], r[2], r[3], r[4]] for r in g], dtype=float).reshape(-1, 4)
        pr_boxes = np.array([[r[1], r[2], r[3], r[4]] for r in p], dtype=float).reshape(-1, 4)
        dists = mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=args.max_iou)
        acc.update(gt_ids, pr_ids, dists)

    mh = mm.metrics.create()
    names = ["num_frames", "num_objects", "num_predictions",
             "mota", "motp", "idf1", "precision", "recall",
             "num_misses", "num_false_positives"]
    summary = mh.compute(acc, metrics=names, name=args.name)

    print(f"\n=== {args.name} (max_iou={args.max_iou}) ===")
    for k in names:
        v = summary.loc[args.name, k]
        if isinstance(v, (int, np.integer)):
            print(f"  {k:<22} {int(v)}")
        elif isinstance(v, float):
            print(f"  {k:<22} {v:.4f}")
        else:
            print(f"  {k:<22} {v}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="MOT txt from observation.write_mot()")
    ap.add_argument("--gt-json", required=True, help="OpenTTGames ball_markup.json")
    ap.add_argument("--max-frames", type=int, default=None, help="only evaluate frames < N (0-based)")
    ap.add_argument("--max-iou", type=float, default=0.3,
                    help="IoU threshold for a match; 0.3 on a 46px box ≈ centre error 14px")
    ap.add_argument("--ball-radius", type=float, default=23.0,
                    help="GT box radius when the markup has only a centre point (px @1920)")
    ap.add_argument("--name", default="tracknet")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
