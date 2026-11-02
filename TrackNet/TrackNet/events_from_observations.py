"""从统一 Observation 生成 Event：闭合「适配器 -> 事件层 -> 状态机」链路。

detect_bounce.py 直接读 TrackNet 的私有 CSV，只适合单模型快速迭代。
本脚本改成读**统一格式**的 observation JSONL，事件层不再依赖任何模型的
内部文件布局 —— 换成 YOLO 或 TTNet 的适配器输出，这一步不用改代码。

默认只使用真实检出（visible=True 且 predicted=False）的帧，与 CSV 模式
的 detected==1 等价。插值补出来的位置是推断值，用它找落台峰值会把转折
抹平，需要对比时用 --include_predicted 显式打开。

用法：
  python events_from_observations.py --obs <test_2_observations.jsonl> \
      --game test_2 --event_jsonl <out.jsonl>
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "adapter"))

from observation import load_observations  # noqa: E402
from detect_bounce import (build_events, load_quad, side_report,  # noqa: E402
                           table_homography, detect, write_events)


def obs_to_arrays(observations, include_predicted=False):
    """Observation 列表 -> (x, y, det) 数组，下标为 frame_id。"""
    n = max(o["frame_id"] for o in observations) + 1
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    det = np.zeros(n, dtype=bool)
    for o in observations:
        ball = o["ball"]
        if not ball["visible"]:
            continue
        if ball.get("predicted", False) and not include_predicted:
            continue
        f = int(o["frame_id"])
        x[f] = float(ball["center_px"][0])
        y[f] = float(ball["center_px"][1])
        det[f] = True
    return x, y, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True, help="统一 Observation JSONL")
    ap.add_argument("--game", required=True)
    ap.add_argument("--event_jsonl", default=None)
    ap.add_argument("--out", default=None, help="落台检测明细 CSV")
    ap.add_argument("--fps", type=float, default=120.0)
    ap.add_argument("--source", default=None,
                    help="事件 source 字段，默认取观测里的 source")
    ap.add_argument("--win", type=int, default=12)
    ap.add_argument("--min_drop", type=float, default=15.0)
    ap.add_argument("--min_gap", type=int, default=15)
    ap.add_argument("--dead_zone", type=float, default=0.05)
    ap.add_argument("--include_predicted", action="store_true")
    ap.add_argument("--max_frame", type=int, default=None)
    args = ap.parse_args()

    obs = load_observations(args.obs)
    source = args.source or (obs[0]["source"] if obs else "tracknet")
    x, y, det = obs_to_arrays(obs, args.include_predicted)

    poly = load_quad(args.game)
    H = table_homography(poly)
    n_frames = args.max_frame or len(x)

    dets = detect(x, y, det, poly, args.win, args.min_drop,
                  margin=0.0, min_gap=args.min_gap, H=H, dead=args.dead_zone)
    dets = [d for d in dets if d[0] < n_frames]

    print(f"obs: {os.path.basename(args.obs)}  frames={len(obs)}  "
          f"可用帧={int(det.sum())} ({det.mean() * 100:.1f}%)  source={source}")
    print(f"检出落台 {len(dets)} 次")
    cnt, same, pairs = side_report(dets)
    print(f"半台归属  A={cnt['A']}  B={cnt['B']}  UNKNOWN={cnt['UNKNOWN']}"
          f"  |  相邻两次同侧 {same}/{pairs} ({same / pairs * 100:.1f}%)")

    if args.out and dets:
        import pandas as pd
        pd.DataFrame(dets, columns=["frame", "x", "y", "drop_before",
                                    "drop_after", "side"]).to_csv(args.out, index=False)
        print(f"写入 {args.out}")
    if args.event_jsonl:
        events = build_events(dets, args.fps, source=source)
        n, _ = write_events(args.event_jsonl, events)
        print(f"写入统一事件 {n} 条 -> {args.event_jsonl}")


if __name__ == "__main__":
    main()
