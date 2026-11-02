"""从统一 Observation 生成 RACKET_HIT（击球）事件。

为什么要单独做这一步：状态机交付件（`乒乓球裁判状态机_交付件.zip`）的
`state_machine.py:317` 只允许 `RACKET_HIT` 把状态从 `WAITING_SERVE` 推进到
`RALLY`，`_on_bounce()` 仅在 `SERVE_IN_PROGRESS` / `RALLY` 下生效。实测只喂
BOUNCE 会卡在 `WAITING_SERVE`，判罚 0 次、比分停在 0:0。所以对状态机而言，
**RACKET_HIT 与 BOUNCE 必须成对出现**，缺一不可。

判据沿用状态机交付件 `core/trajectory_processor.py::_extract_events` 的定义：

  1. 水平速度 `vx` 反号 —— 球被球拍打回，水平运动方向掉头；
  2. 反号前后至少一侧的 `|vx|` 超过 `hit_vx_min`；
  3. 击球方归属用**被击出后的飞行方向**推断，而不是击球位置：
     `vx > 0`（球打向右）→ 击球方在左，记 `A`；`vx < 0` → 记 `B`。

第 3 条的理由（原文注释）：速度反转是在实际击球之后几帧才被检出的，那时球
已经越过球网，用位置判会把击球方判成对面一侧，进而让状态机把正常对打误判成
`OWN_COURT`。

与 `detect_bounce.py` 的分工：落台是「vx 不改向、vy 由下转上」，击球是「vx
掉头」，两者物理上互斥，因此同一时刻不应同时产出两种事件。

用法：
  # 只产出击球事件
  python detect_racket_hit.py --obs <test_2_observations.jsonl> --out hits.jsonl

  # 与已有落台事件合并成一条完整事件流（时间升序，event_id 重编）
  python detect_racket_hit.py --obs <obs.jsonl> --out hits.jsonl \
      --bounce_jsonl <test_2_bounce_events.jsonl> --merged_jsonl <full.jsonl>
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "adapter"))

from observation import load_observations, make_event, validate_event  # noqa: E402


def write_events(path, events):
    """写事件 JSONL，逐条过 validate_event；不合规直接抛错，不静默落盘。"""
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            errs = validate_event(ev)
            if errs:
                raise ValueError(f"invalid event {ev.get('event_id')}: {errs}")
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return len(events), 0


def obs_to_arrays(observations, include_predicted=False):
    """Observation 列表 -> (x, y, det) 数组，下标为 frame_id。

    默认剔除 `predicted=True` 的插值点，与状态机交付件的做法一致：
    `_extract_events()` 开头就断言「只有真实观测点才能产生事件」。
    """
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


def velocities(x, y, det, fps):
    """在检出序列上算前向速度（px/s）。

    只在实际检出点之间求差，所以能跨过漏检帧；返回值下标与 `fs` 对齐。
    用前向差分而非中心差分，是为了让「掉头」这件事落在具体某一帧上 ——
    中心差分会把反转前后各摊一半，反而抹平判据。
    """
    fs = np.where(det)[0]
    if len(fs) < 3:
        return fs, np.zeros(0), np.zeros(0)
    xs, ys = x[fs], y[fs]
    t = fs.astype(float) / fps
    dt = np.diff(t)
    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt
    return fs, vx, vy


def detect_hits(x, y, det, fps, hit_vx_min=800.0, cooldown_s=0.18,
                max_speed_px_s=8000.0, max_gap_s=0.25, max_step_s=0.10):
    """返回 [(frame, x, y, vx_before, vx_after, side), ...]。

    这里有三处与状态机交付件的原始判据不同，都是实测踩出来的：

    1. **断轨判据用速度上限，而不是单帧位移上限。** 最初按「相邻检出点位移
       ≤ 220 px」过滤，而 TrackNet 在 test_2 上检出率只有 30%，相邻检出点
       常跨 2~4 帧，球速 2000 px/s 时位移就有 300~700 px，结果把大部分真实
       击球当成跳点滤掉（只剩 13 条）。

    2. **速度门限取反号两侧的较小值，不是较大值。** 用 `max()` 时，落台瞬间
       的「假反号」也会通过：球在台面反弹后往往被遮挡数帧（实测 f582→f606
       跨 24 帧），把这段求平均得到的反向速度接近 0，但落台前的正向速度高达
       2880 px/s，`max()` 照样超过门限，于是每次落台都附带一个假击球。改用
       `min()` 后要求「去」和「回」都真有速度——乒乓球碰撞能量损失很小，
       这一条在物理上成立。

    3. **单段跨度不能太大（`max_step_s`）。** 跨 24 帧的平均速度不代表瞬间
       速度，跨得越久，「反号」越可能只是两段不相关轨迹的拼接。
    """
    fs, vx, vy = velocities(x, y, det, fps)
    if len(fs) < 3:
        return []

    xs, ys = x[fs], y[fs]
    t = fs.astype(float) / fps
    speed = np.hypot(vx, vy)
    step_s = np.diff(t)
    # 速度超物理上限，或时间跨度太大（中间整段丢失），都视为断轨
    broken = (speed > max_speed_px_s) | (step_s > max_gap_s)

    out = []
    last_t = -1e9
    for i in range(1, len(vx)):
        # i 对应 fs[i+1]：vx[i-1] 是进入该点的速度，vx[i] 是离开该点的速度
        if broken[i - 1] or broken[i]:
            continue
        if step_s[i - 1] > max_step_s or step_s[i] > max_step_s:
            continue
        if vx[i - 1] * vx[i] >= 0:
            continue
        if min(abs(vx[i - 1]), abs(vx[i])) < hit_vx_min:
            continue
        f = int(fs[i])
        ts = f / fps
        if ts - last_t < cooldown_s:
            continue
        last_t = ts
        # 击球方按被击出后的方向归属：打向右 → 击球方在左(A)
        side = "A" if vx[i] > 0 else "B"
        out.append((f, float(xs[i]), float(ys[i]),
                    float(vx[i - 1]), float(vx[i]), side))
    return out


def build_hit_events(hits, fps, source="tracknet"):
    events = []
    for i, (frame, px, py, _vb, _va, side) in enumerate(hits):
        events.append(make_event(
            event_id=f"racket_hit_{i:05d}",
            timestamp_s=frame / float(fps),
            event_type="RACKET_HIT",
            side=side,
            position_px=[px, py],
            confidence=None,
            sources=[source],
        ))
    return events


def merge_events(hit_events, bounce_events, fps=120.0, window_s=2 / 120.0,
                 priority="bounce"):
    """合并击球与落台，并消解「同一次接触被判成两种事件」。

    球拍击球会让球在竖直方向掉头，形态与落台完全一样，所以两套独立的判定
    模块**必然**会在接触瞬间各报一个事件。实测 test_2 有两处：

    | 时刻 | 我的落台 | 轨迹层击球 | 真值 |
    |---|---|---|---|
    | 18.43 s | BOUNCE B (1195,595) | RACKET_HIT B (1195,538) | **有落台** |
    | 26.47 s | BOUNCE B (1225,556) | RACKET_HIT B (1243,553) | **有落台** |

    两处 `events_markup.json` 都标了 bounce，说明**落台是真的，击球是假的**
    —— 球刚弹起就被判成「被球拍打回」。所以默认 `priority="bounce"`：
    窗口内同侧的击球让位，落台保留。

    反过来先试过（击球优先），代价是丢掉两个真阳性，落台 F1 从 0.774 掉到
    0.767 —— 而击球没有任何真值可以校准，没有理由让它压过有真值的落台。
    """
    if priority == "bounce":
        kept, dropped = [], 0
        for h in hit_events:
            dup = any(
                b["side"] == h["side"]
                and abs(b["timestamp_s"] - h["timestamp_s"]) <= window_s
                for b in bounce_events
            )
            if dup:
                dropped += 1
                continue
            kept.append(h)
        all_ev = list(bounce_events) + kept
    elif priority == "hit":
        kept, dropped = [], 0
        for b in bounce_events:
            dup = any(
                h["side"] == b["side"]
                and abs(h["timestamp_s"] - b["timestamp_s"]) <= window_s
                for h in hit_events
            )
            if dup:
                dropped += 1
                continue
            kept.append(b)
        all_ev = kept + list(hit_events)
    else:
        all_ev, dropped = list(bounce_events) + list(hit_events), 0

    all_ev.sort(key=lambda e: (e["timestamp_s"], e["type"]))
    counters = {}
    for e in all_ev:
        prefix = e["type"].lower()
        n = counters.get(prefix, 0)
        counters[prefix] = n + 1
        e["event_id"] = f"{prefix}_{n:05d}"
    return all_ev, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True, help="统一 Observation JSONL")
    ap.add_argument("--out", default=None, help="击球事件 JSONL")
    ap.add_argument("--bounce_jsonl", default=None, help="已有落台事件 JSONL")
    ap.add_argument("--hit_jsonl", default=None,
                    help="外部击球事件 JSONL（如状态机交付件的 trajectory_processor "
                         "输出）；给了就不再自行检测")
    ap.add_argument("--merged_jsonl", default=None,
                    help="与落台合并后的完整事件流 JSONL")
    ap.add_argument("--dedupe_window", type=float, default=2.0,
                    help="同侧接触去重窗口（帧）：落台与击球相距在此范围内视为同一接触")
    ap.add_argument("--priority", choices=("bounce", "hit", "none"), default="none",
                    help="同一接触的归属优先级；默认不去重（三种策略实测见文档第七章）")
    ap.add_argument("--fps", type=float, default=120.0)
    ap.add_argument("--source", default=None)
    ap.add_argument("--hit_vx_min", type=float, default=800.0,
                    help="水平速度下限 px/s（取反号两侧的较小值）")
    ap.add_argument("--cooldown", type=float, default=0.18,
                    help="相邻击球的最小间隔（秒），状态机默认 0.18")
    ap.add_argument("--max_speed", type=float, default=8000.0,
                    help="速度上限 px/s，超出视为断轨（状态机默认同为 8000）")
    ap.add_argument("--max_gap", type=float, default=0.25,
                    help="相邻检出点的最大时间间隔 s，超出视为断轨")
    ap.add_argument("--max_step", type=float, default=0.10,
                    help="参与反号判定的单段最大时长 s（跨太久的平均速度无意义）")
    ap.add_argument("--include_predicted", action="store_true")
    args = ap.parse_args()

    obs = load_observations(args.obs)
    source = args.source or (obs[0]["source"] if obs else "tracknet")
    x, y, det = obs_to_arrays(obs, args.include_predicted)

    hits = detect_hits(x, y, det, args.fps, args.hit_vx_min, args.cooldown,
                       args.max_speed, args.max_gap, args.max_step)
    print(f"obs: {os.path.basename(args.obs)}  frames={len(obs)}  "
          f"可用帧={int(det.sum())} ({det.mean() * 100:.1f}%)  source={source}")
    print(f"检出击球 {len(hits)} 次  "
          f"A={sum(1 for h in hits if h[5] == 'A')}  "
          f"B={sum(1 for h in hits if h[5] == 'B')}")

    hit_events = build_hit_events(hits, args.fps, source)
    if args.hit_jsonl:
        with open(args.hit_jsonl, encoding="utf-8") as f:
            hit_events = [json.loads(l) for l in f if l.strip()]
        print(f"改用外部击球事件 {len(hit_events)} 条 <- {args.hit_jsonl}")
    if args.out:
        n, bad = write_events(args.out, hit_events)
        print(f"写入击球事件 {n} 条 -> {args.out}" + (f"  （校验失败 {bad}）" if bad else ""))

    if args.bounce_jsonl:
        with open(args.bounce_jsonl, encoding="utf-8") as f:
            bounce_events = [json.loads(l) for l in f if l.strip()]
        merged, dropped = merge_events(hit_events, bounce_events, args.fps,
                                       args.dedupe_window / args.fps,
                                       args.priority)
        n_b = sum(1 for e in merged if e["type"] == "BOUNCE")
        n_h = sum(1 for e in merged if e["type"] == "RACKET_HIT")
        print(f"同侧接触去重（{args.priority}）：丢弃 {dropped} 条")
        print(f"合并：BOUNCE {n_b} + RACKET_HIT {n_h} = {len(merged)} 条")
        if args.merged_jsonl:
            n, bad = write_events(args.merged_jsonl, merged)
            print(f"写入完整事件流 {n} 条 -> {args.merged_jsonl}"
                  + (f"  （校验失败 {bad}）" if bad else ""))


if __name__ == "__main__":
    main()
