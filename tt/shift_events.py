"""把整场事件文件裁剪到某个时间窗内，并把时间戳平移到片段起点。

用法：
  python shift_events.py --src ../outputs/game_4_bounce_events.jsonl \
      --out ../outputs/game_4_seg470_bounces.jsonl --start 470 --end 526

片段视频是用 ffmpeg -ss 470 -t 56 切出来的，帧率不变（120fps），
因此片段内时间 = 全场时间 - 470，帧号同理减去 470*120。
"""
import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--fps", type=float, default=120.0)
    args = ap.parse_args()

    rows = []
    with open(args.src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if args.start <= ev["timestamp_s"] < args.end:
                ev["timestamp_s"] = round(ev["timestamp_s"] - args.start, 6)
                rows.append(ev)

    for i, ev in enumerate(rows):
        ev["event_id"] = f"bounce_{i:05d}"

    with open(args.out, "w", encoding="utf-8") as f:
        for ev in rows:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    print(f"裁剪 {len(rows)} 条 -> {args.out}")
    if rows:
        print(f"  时间范围 {rows[0]['timestamp_s']:.3f} .. {rows[-1]['timestamp_s']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
