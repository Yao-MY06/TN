"""校验统一 JSONL 是否符合契约（队友自查用）。

自动识别两种文件：
  - Observation（有 frame_id / ball）：逐帧观测，适配器输出
  - Event（有 event_id / type）：事件层输出

用法：
    python check_observations.py observations.jsonl [--expect-frames 3600] [--fps 120]
    python check_observations.py events.jsonl [--fps 120]
"""

from __future__ import annotations

import argparse
import json

from observation import validate_event, validate_observation


def load(path):
    """返回 (mode, rows, errors)。mode 由第一条记录的类型决定。"""
    mode = None
    rows, errors = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {line_no}: invalid JSON ({e})")
                continue
            if mode is None:
                mode = "event" if "event_id" in obj else "observation"
            errs = validate_event(obj) if mode == "event" else validate_observation(obj)
            if errs:
                key = obj.get("event_id") or obj.get("frame_id")
                errors.append(f"line {line_no} ({key}): {errs}")
            rows.append(obj)
    return mode, rows, errors


def report_observations(path, obs_list, errors, args):
    frames = [o["frame_id"] for o in obs_list]
    gaps = [b - a for a, b in zip(frames, frames[1:]) if b - a != 1]
    sources = sorted({o["source"] for o in obs_list})
    visible = sum(1 for o in obs_list if o["ball"]["visible"])
    pred = sum(1 for o in obs_list if o["ball"].get("predicted", False))
    confs = [o["ball"]["confidence"] for o in obs_list if o["ball"]["confidence"] is not None]

    print(f"file          : {path}")
    print(f"type          : observation")
    print(f"observations  : {len(obs_list)}  (frames {min(frames)}..{max(frames)})")
    print(f"source(s)     : {sources}")
    print(f"visible       : {visible}/{len(obs_list)} ({visible / len(obs_list) * 100:.1f}%)")
    if pred:
        print(f"predicted     : {pred} (trajectory-filled, NOT model detections)")
    print(f"confidence    : {'n/a' if not confs else f'{len(confs)} frames, range {min(confs):.3f}..{max(confs):.3f}'}")
    if gaps:
        print(f"WARN          : {len(gaps)} non-consecutive frame steps (first: {gaps[:5]})")
    if args.expect_frames is not None and len(obs_list) != args.expect_frames:
        print(f"WARN          : expected {args.expect_frames} frames, got {len(obs_list)}")
    if args.fps:
        ts_err = max(abs(o["timestamp_s"] - o["frame_id"] / args.fps) for o in obs_list)
        print(f"timestamp     : max deviation from frame_id/fps = {ts_err:.6f}s")
    return errors


def report_events(path, events, errors, args):
    ts = [e["timestamp_s"] for e in events]
    types = {}
    sides = {}
    srcs = set()
    for e in events:
        types[e["type"]] = types.get(e["type"], 0) + 1
        sides[e["side"]] = sides.get(e["side"], 0) + 1
        srcs.update(e["sources"])
    ids = [e["event_id"] for e in events]
    dup = len(ids) - len(set(ids))
    confs = [e["confidence"] for e in events if e["confidence"] is not None]
    conf_txt = "n/a (none)" if not confs else (
        f"{len(confs)} events, range {min(confs):.3f}..{max(confs):.3f}")

    print(f"file          : {path}")
    print(f"type          : event")
    print(f"events        : {len(events)}  (t {min(ts):.3f}..{max(ts):.3f}s)")
    print(f"types         : {types}")
    print(f"side          : {sides}")
    print(f"sources       : {sorted(srcs)}")
    print(f"confidence    : {conf_txt}")
    if dup:
        print(f"WARN          : {dup} duplicated event_id")
    if len(ts) > 1:
        d = sorted(b - a for a, b in zip(ts, ts[1:]))
        print(f"interval      : min {d[0]:.3f}s  median {d[len(d) // 2]:.3f}s  max {d[-1]:.3f}s")
    if args.fps:
        ts_err = max(abs(e["timestamp_s"] - round(e["timestamp_s"] * args.fps) / args.fps)
                     for e in events)
        print(f"timestamp     : max deviation from integer frame = {ts_err:.6f}s")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--expect-frames", type=int, default=None,
                    help="期望的总帧数（仅 Observation；检查是否缺帧）")
    ap.add_argument("--fps", type=float, default=None,
                    help="给出时校验时间戳是否落在整帧上")
    args = ap.parse_args()

    mode, rows, errors = load(args.path)
    if not rows:
        print("FAIL: no records found")
        return

    if mode == "event":
        errors = report_events(args.path, rows, errors, args)
    else:
        errors = report_observations(args.path, rows, errors, args)

    if errors:
        print(f"\nFAIL: {len(errors)} schema error(s)")
        for e in errors[:10]:
            print("  " + e)
        if len(errors) > 10:
            print(f"  ... {len(errors) - 10} more")
    else:
        print("\nPASS: schema OK")


if __name__ == "__main__":
    main()
