"""规则版落台（bounce）检测。

判据（三条同时成立）：
  1. **位置**：落点位于球台台面多边形内（相机近乎侧视，台面高度与"在台面上方"在图像上
     不重叠，因此这一条同时也是与「球拍击球」区分开的依据——击球点要么在台面之外，
     要么在台面上方）。
  2. **形状**：该帧是球心 y 的**局部最大**（图像坐标 y 向下，y 最大 = 球的最低点）。
     球触台瞬间竖直速度由向下转向上，在轨迹上就是一个下凸的转折点。
  3. **幅度**：转折前后各自至少下降了 min_drop 像素，排除静止或抖动的伪峰。

用「检测序列上的局部最大」而不是「逐帧差分速度」，是因为球检出率只有约 24%，
逐帧差分大量失效；在检测序列上比较则天然容忍缺失帧。

半台归属（side）：把落点用台面四角的单应性变换到俯视平面（2.74m x 1.525m），
再与球网（长边中点连线，俯视下 x = 1.37m）比较。透视变换只作用于**桌面平面**，
适用于已经判定为落台的那一帧的位置——正是方案文档第 5 节要求的顺序：
先由时间序列判落台时刻，再定位落点，而不是把空中球直接投到桌面。
A = far_left / near_left 那一侧（图像左侧半台），B = 另一侧。距离网线不足
dead_zone 米时记 UNKNOWN，避免把压线球硬判到某一侧。

用法：
  python detect_bounce.py --csv <track.csv> --game game_4 --events <events_markup.json>
  python detect_bounce.py --csv <track.csv> --game game_4 --events ... --sweep
  python detect_bounce.py --csv <track.csv> --game game_4 --event_jsonl out.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(_HERE, "..", "..", "calib", "table_quads.json")

# 统一事件接口：Event 结构与校验由 adapter/observation.py 提供
sys.path.insert(0, os.path.join(_HERE, "..", "..", "adapter"))
from observation import make_event, validate_event  # noqa: E402

# 乒乓球台标准尺寸（ITTF）
TABLE_LEN_M = 2.74
TABLE_WID_M = 1.525
# 球网位于长边中点，垂直于长边
NET_X_M = TABLE_LEN_M / 2.0


def load_quad(game):
    with open(CALIB_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if game not in data:
        raise SystemExit(f"未标定 {game}，请先在 {CALIB_PATH} 中补充四角")
    return np.array(data[game]["points"], dtype=float)


def _pip(pts, poly):
    """向量化 point-in-polygon（射线法）。"""
    inside = np.zeros(len(pts), dtype=bool)
    n = len(poly)
    px, py = pts[:, 0], pts[:, 1]
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > py) != (y2 > py))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2 - x1) * (py - y1) / (y2 - y1) + x1
        inside ^= cond & (px < xint)
    return inside


def shrink_poly(poly, margin):
    """按质心方向收缩多边形，用于排除贴边擦球。"""
    if margin <= 0:
        return poly
    c = poly.mean(axis=0)
    v = poly - c
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return c + v * np.clip((norm - margin) / norm, 0, None)


def table_homography(poly):
    """台面四角 -> 俯视平面（单位：米）。

    四角顺序约定 far_left, far_right, near_right, near_left（见 calib/table_quads.json）：
        far_left   -> (0, 0)
        far_right  -> (L, 0)
        near_right -> (L, W)
        near_left  -> (0, W)
    于是 x 沿长边、y 沿短边，球网在 x = L/2。
    """
    import cv2
    src = np.asarray(poly, dtype=np.float32)
    dst = np.array([[0.0, 0.0],
                    [TABLE_LEN_M, 0.0],
                    [TABLE_LEN_M, TABLE_WID_M],
                    [0.0, TABLE_WID_M]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def side_of(pt, H, dead=0.05):
    """落点 -> ("A"|"B"|"UNKNOWN", 俯视坐标 (x, y) 单位米)。

    A = far_left/near_left 一侧（图像左侧半台），B = 另一侧。
    与网线距离小于 dead 米时记 UNKNOWN。
    """
    import cv2
    p = cv2.perspectiveTransform(np.array([[pt]], dtype=np.float32), H)[0, 0]
    dx = float(p[0]) - NET_X_M
    if abs(dx) < dead:
        return "UNKNOWN", (float(p[0]), float(p[1]))
    return ("A" if dx < 0 else "B"), (float(p[0]), float(p[1]))


def load_track(csv_path):
    df = pd.read_csv(csv_path)
    n = int(df["frame"].max()) + 1
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    det = np.zeros(n, dtype=bool)
    f = df["frame"].to_numpy(dtype=int)
    ok = (df["detected"].to_numpy(dtype=int) == 1) \
        & df["x"].notna().to_numpy() & df["y"].notna().to_numpy()
    x[f[ok]] = df["x"].to_numpy()[ok]
    y[f[ok]] = df["y"].to_numpy()[ok]
    det[f[ok]] = True
    interp = np.zeros(n, dtype=bool)
    if "interp" in df.columns:
        interp[f[df["interp"].fillna(0).to_numpy(dtype=int) == 1]] = True
    return x, y, det, interp


def detect(x, y, det, poly, win=12, min_drop=15.0, margin=0.0, min_gap=15,
           H=None, dead=0.05, fps=120.0, hit_vx_min=0.0):
    """返回 [(frame, x, y, drop_before, drop_after, side), ...]

    `hit_vx_min` > 0 时启用「击球排除」：球拍击球会让球在竖直方向出现转折
    （球员把球往上打），形态上与落台完全一样，但**水平运动方向会掉头**——
    落台时球继续朝同一侧飞，击球时球被打回另一侧。

    这条守卫照搬状态机交付件 `trajectory_processor._is_bounce()` 的做法。
    不加它的实测后果：test_2 上 34 条落台里混进了 10 条击球点，喂给状态机后
    被判成 11 次 `OWN_COURT`（球落在击球方自己半台），比分完全失真。

    判据落在**检出序列**上，与「局部最大」用的是同一套索引，所以稀疏检出
    （test_2 只有 30.6%）不会额外引入失效。
    """
    fs = np.where(det)[0]
    if len(fs) < 3:
        return []
    if H is None:
        H = table_homography(poly)
    xs, ys = x[fs], y[fs]
    # 逐段水平速度（px/s），长度 len(fs)-1；dx[k] 是 fs[k] -> fs[k+1] 的速度
    dx = np.diff(xs) / (np.diff(fs).astype(float) / fps)

    keep = []
    for i in range(1, len(fs) - 1):
        # 击球排除：进点与出点的水平速度反号，说明球在这里掉头
        if hit_vx_min > 0 and dx[i - 1] * dx[i] < 0 \
                and max(abs(dx[i - 1]), abs(dx[i])) >= hit_vx_min:
            continue
        lo = np.searchsorted(fs, fs[i] - win, "left")
        hi = np.searchsorted(fs, fs[i] + win, "right")
        # 必须严格是窗口内的最大值，且至少各有一帧在前后
        if lo >= i or hi - 1 <= i:
            continue
        if ys[i] != ys[lo:hi].max():
            continue
        if (ys[lo:i].size == 0) or (ys[i + 1:hi].size == 0):
            continue
        drop_b = float(ys[i] - ys[lo:i].min())
        drop_a = float(ys[i] - ys[i + 1:hi].min())
        if drop_b < min_drop and drop_a < min_drop:
            continue
        keep.append((int(fs[i]), float(xs[i]), float(ys[i]), drop_b, drop_a))

    if not keep:
        return []
    pts = np.array([[k[1], k[2]] for k in keep])
    inside = _pip(pts, shrink_poly(poly, margin))

    out, last = [], -10 ** 9
    for i, k in enumerate(keep):
        if not inside[i] or k[0] - last < min_gap:
            continue
        last = k[0]
        side, _ = side_of((k[1], k[2]), H, dead)
        out.append(k + (side,))
    return out


def load_events(path):
    with open(path, encoding="utf-8") as f:
        ev = json.load(f)
    return {int(k): v for k, v in ev.items()}


def build_events(dets, fps, source="tracknet"):
    """落台检测结果 -> 统一 Event 列表（方案文档 7.4）。

    confidence 一律为 None：规则法没有概率输出，填一个凑出来的数会让人
    误以为那是可信度。转折幅度 drop_before / drop_after 留在检测 CSV 里
    供分析，不塞进事件契约。
    """
    events = []
    for i, d in enumerate(dets):
        frame, px, py, _db, _da, side = d
        events.append(make_event(
            event_id=f"bounce_{i:05d}",
            timestamp_s=frame / float(fps),
            event_type="BOUNCE",
            side=side,
            position_px=[px, py],
            confidence=None,
            sources=[source],
        ))
    return events


def write_events(path, events):
    bad = 0
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            errs = validate_event(ev)
            if errs:
                bad += 1
                raise ValueError(f"invalid event {ev.get('event_id')}: {errs}")
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return len(events), bad


def side_report(dets):
    """半台归属的分布与交替性。"""
    sides = [d[5] for d in dets]
    cnt = {s: sides.count(s) for s in ("A", "B", "UNKNOWN")}
    seq = [s for s in sides if s != "UNKNOWN"]
    same = sum(1 for a, b in zip(seq, seq[1:]) if a == b)
    pairs = max(1, len(seq) - 1)
    return cnt, same, pairs


def evaluate(dets, events, tol, max_frame=None):
    gt = sorted(f for f, v in events.items() if v == "bounce")
    if max_frame is not None:
        gt = [f for f in gt if f < max_frame]
    pred = sorted(d[0] for d in dets)
    if max_frame is not None:
        pred = [f for f in pred if f < max_frame]
    used, tp = set(), 0
    for f in pred:
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
    return dict(tp=tp, fp=len(pred) - tp, fn=len(gt) - tp,
                n_pred=len(pred), n_gt=len(gt), p=p, r=r, f1=f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--events", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--win", type=int, default=12)
    ap.add_argument("--min_drop", type=float, default=15.0)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--min_gap", type=int, default=15)
    ap.add_argument("--max_frame", type=int, default=None)
    ap.add_argument("--tol", type=int, default=8)
    ap.add_argument("--fps", type=float, default=120.0,
                    help="视频帧率，用于把 frame 换算成 timestamp_s（OpenTTGames 为 120）")
    ap.add_argument("--dead_zone", type=float, default=0.05,
                    help="距网线多少米内记 UNKNOWN，单位米")
    ap.add_argument("--event_jsonl", default=None,
                    help="按统一 Event 接口输出 JSONL（方案文档 7.4）")
    ap.add_argument("--hit_vx_min", type=float, default=0.0,
                    help="击球排除：水平速度反号且超过该值(px/s)时判为击球而非落台。"
                         "默认关闭——game_4 上启用会丢 19 条真落台（F1 0.942→0.919）")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    poly = load_quad(args.game)
    H = table_homography(poly)
    x, y, det, interp = load_track(args.csv)
    n_frames = args.max_frame or len(x)
    print(f"track: {os.path.basename(args.csv)}  frames={len(x)}  "
          f"detected={int(det.sum())} ({det.mean() * 100:.1f}%)")

    kwargs = dict(margin=args.margin, min_gap=args.min_gap,
                  H=H, dead=args.dead_zone,
                  fps=args.fps, hit_vx_min=args.hit_vx_min)
    if args.sweep:
        if not args.events:
            raise SystemExit("--sweep 需要 --events")
        events = load_events(args.events)
        print(f"{'win':>4}{'drop':>6}{'margin':>7}{'gap':>5} | {'B':>4} {'P':>6} {'R':>6} {'F1':>6}")
        best = None
        for win in (8, 12, 16, 20):
            for drop in (5, 10, 15, 20, 30, 40):
                for margin in (0, 5, 10, 20):
                    for gap in (10, 15, 25, 40):
                        d = detect(x, y, det, poly, win, drop, margin, gap,
                                   H=H, dead=args.dead_zone)
                        m = evaluate(d, events, args.tol, n_frames)
                        if best is None or m["f1"] > best[0]["f1"]:
                            best = (m, win, drop, margin, gap)
                        print(f"{win:>4}{drop:>6}{margin:>7}{gap:>5} | "
                              f"{m['n_pred']:>4} {m['p']:>6.3f} {m['r']:>6.3f} {m['f1']:>6.3f}")
        m, win, drop, margin, gap = best
        print(f"\n最佳: win={win} min_drop={drop} margin={margin} gap={gap} "
              f"-> P={m['p']:.3f} R={m['r']:.3f} F1={m['f1']:.3f} "
              f"(TP{m['tp']} FP{m['fp']} FN{m['fn']})")
        return

    dets = detect(x, y, det, poly, args.win, args.min_drop, **kwargs)
    dets = [d for d in dets if d[0] < n_frames]
    print(f"检出落台 {len(dets)} 次")

    cnt, same, pairs = side_report(dets)
    print(f"半台归属  A={cnt['A']}  B={cnt['B']}  UNKNOWN={cnt['UNKNOWN']}"
          f"  |  相邻两次同侧 {same}/{pairs} ({same / pairs * 100:.1f}%)")

    if args.out:
        pd.DataFrame(dets, columns=["frame", "x", "y", "drop_before",
                                    "drop_after", "side"]) \
            .to_csv(args.out, index=False)
        print(f"写入 {args.out}")
    if args.event_jsonl:
        events = build_events(dets, args.fps)
        n, _ = write_events(args.event_jsonl, events)
        print(f"写入统一事件 {n} 条 -> {args.event_jsonl}")
        if n:
            print(f"  时间范围 {dets[0][0] / args.fps:.3f}.."
                  f"{dets[-1][0] / args.fps:.3f}s (fps={args.fps})")
    if args.events:
        events = load_events(args.events)
        for tol in (2, 3, 5, 8, 10, 15, 20, 30):
            m = evaluate(dets, events, tol, n_frames)
            print(f"  tol=±{tol:>2}帧  TP{m['tp']:>4} FP{m['fp']:>3} FN{m['fn']:>4} | "
                  f"P={m['p']:.3f} R={m['r']:.3f} F1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
