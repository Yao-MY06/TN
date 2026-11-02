"""把状态机交付件的 RACKET_HIT 导出成统一 Event JSONL。

用途：状态机的 `trajectory_processor` 已经能产 RACKET_HIT，而我这边的
`detect_racket_hit.py` 在 30.6% 的检出率下会漏掉一部分击球（击球瞬间球被
球员遮挡，正好是 TrackNet 漏检最集中的地方）。所以默认方案是：

    BOUNCE   <- 我方 `detect_bounce.py`（召回 0.857，高于状态机自带的 0.607）
    RACKET_HIT <- 状态机交付件的轨迹层（能做到 40 条，覆盖比我方完整）

两边合并成一条事件流再喂状态机，见 `detect_racket_hit.py --hit_jsonl`。

用法：
  python export_his_racket_hits.py \
      --obs ../../adapter/examples/test_2_observations.jsonl \
      --preset test2 --out ../../outputs/test_2_hits_from_sm.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/_unpacked")

from core.geometry import preset  # noqa: E402
from core.schemas import FrameObservation  # noqa: E402
from core.trajectory_processor import TrajectoryEventProcessor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--preset", default="test2")
    ap.add_argument("--source", default="tracknet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default="RACKET_HIT",
                    help="逗号分隔，取哪些事件；默认只要击球")
    args = ap.parse_args()

    want = {t.strip() for t in args.types.split(",") if t.strip()}
    proc = TrajectoryEventProcessor(preset(args.preset), source=args.source)

    kept, allc = [], {}
    with open(args.obs, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obs = FrameObservation.from_dict(json.loads(line))
            for ev in proc.update(obs):
                allc[ev.type] = allc.get(ev.type, 0) + 1
                if ev.type in want:
                    kept.append(ev.to_dict())

    with open(args.out, "w", encoding="utf-8") as f:
        for ev in kept:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    print(f"轨迹层共产出: {allc}")
    print(f"导出 {len(kept)} 条 {sorted(want)} -> {args.out}")


if __name__ == "__main__":
    main()
