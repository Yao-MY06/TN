"""用 Extended OpenTTGames 标注核对判罚与击球。

标注来源：https://github.com/moamal01/table_tennis_data （CC BY-NC-SA 4.0，arXiv:2512.19327）
``data/raw/game_data/test/test_N.json``：键为帧号，值为击球（如 ``right_forehand_loop …``）
或回合结束（如 ``left_out``）。**前缀是造成这一分结束的一方，即丢分方**；A = 左，B = 右。
它不含落台——落台真值仍要原版 OpenTTGames 的 ``events_markup.json``（见 eval_proposals.py）。

    python -m cascade.eval_extended --run-dir outputs/cascade/test_2_rule \\
        --game-json _data/gt/extended_openttgames/game_data/test/test_2.json

核对三件事：
  1. 每一分：标注的回合结束 → 我方第一条得分判罚（在该回合结束之后、下一次击球之前），得分方是否一致
  2. 击球：RACKET_HIT 与标注击球按帧匹配（±5 / ±12 帧），并核对左右侧
  3. 提议覆盖：标注击球、回合结束落在复核片段内的比例
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cascade.eval_proposals import _read_jsonl, match  # noqa: E402

ENDINGS = ("net", "not_hitting_ball", "winner", "double_bounce", "out", "miss_on_own_side")


def parse_game_json(path: str):
    with open(path, encoding="utf-8") as f:
        raw = {int(k): v for k, v in json.load(f).items()}
    strokes, endings = [], []
    for fr in sorted(raw):
        head = raw[fr].split()[0]
        side_word, _, rest = head.partition("_")
        side = "A" if side_word == "left" else "B" if side_word == "right" else None
        if rest in ENDINGS:
            endings.append({"frame": fr, "loser": side, "winner": {"A": "B", "B": "A"}.get(side),
                            "kind": rest, "label": raw[fr]})
        else:
            strokes.append({"frame": fr, "side": side, "label": raw[fr]})
    return strokes, endings


def evaluate(run_dir: str, game_json: str, fps: float = 120.0) -> Dict:
    strokes, endings = parse_game_json(game_json)
    decisions = _read_jsonl(os.path.join(run_dir, "decisions.jsonl"))
    events = _read_jsonl(os.path.join(run_dir, "events.jsonl"))
    clips = _read_jsonl(os.path.join(run_dir, "clips.jsonl"))
    spans = [(c["start_frame_id"], c["end_frame_id"]) for c in clips]

    # ---- 1. 每一分
    points = [d for d in decisions if d.get("kind") == "POINT"]
    stroke_frames = [s["frame"] for s in strokes]
    used, rows = set(), []
    for e in endings:
        t_end = e["frame"] / fps
        nxt = next((f for f in stroke_frames if f > e["frame"]), None)
        t_next = nxt / fps if nxt is not None else float("inf")
        cand = [i for i, d in enumerate(points)
                if i not in used and t_end - 0.5 <= d["virtual_t"] < t_next]
        row = {"gt_t": round(t_end, 3), "gt": e["label"], "gt_winner": e["winner"]}
        if cand:
            i = cand[0]
            used.add(i)
            d = points[i]
            row.update({"our_t": d["virtual_t"], "our_winner": d["winner"], "our_rule": d["rule"],
                        "delay_s": round(d["virtual_t"] - t_end, 3),
                        "correct": d["winner"] == e["winner"]})
        else:
            row.update({"our_t": None, "correct": False, "note": "未判分"})
        rows.append(row)
    spurious = [{"t": d["virtual_t"], "winner": d["winner"], "rule": d["rule"]}
                for i, d in enumerate(points) if i not in used]
    dead = [{"t": d["virtual_t"], "rule": d["rule"]} for d in decisions if d.get("kind") == "DEAD_BALL"]

    # ---- 2. 击球
    hits = [e for e in events if e["type"] == "RACKET_HIT"]
    hit_frames = [round(e["timestamp_s"] * fps) for e in hits]
    m5, m12 = match(hit_frames, stroke_frames, 5), match(hit_frames, stroke_frames, 12)
    # 轨迹层要等球重新出现、看到水平速度反向才报击球，时间戳系统性偏晚，
    # 对称容差会把它全算成错。改用非对称窗口 [-lag_lo, +lag_hi] 匹配，并报告滞后分布。
    lag_lo, lag_hi = 5, 25
    used_h, lags, side_ok = set(), [], 0
    for s in strokes:
        cand = [(f - s["frame"], i) for i, f in enumerate(hit_frames)
                if i not in used_h and -lag_lo <= f - s["frame"] <= lag_hi]
        if not cand:
            continue
        lag, i = min(cand, key=lambda x: abs(x[0]))
        used_h.add(i)
        lags.append(lag)
        side_ok += hits[i]["side"] == s["side"]
    tp = len(lags)
    lag_p = tp / len(hits) if hits else 0.0
    lag_r = tp / len(strokes) if strokes else 0.0
    lag_match = {"window_frames": [-lag_lo, lag_hi], "tp": tp, "p": round(lag_p, 4), "r": round(lag_r, 4),
                 "f1": round(2 * lag_p * lag_r / (lag_p + lag_r), 4) if lag_p + lag_r else 0.0,
                 "lag_median_frames": sorted(lags)[len(lags) // 2] if lags else None,
                 "lag_min_max_frames": [min(lags), max(lags)] if lags else None,
                 "side_agree": f"{side_ok}/{tp}"}

    # ---- 3. 提议覆盖
    def cov(frames: List[int]):
        return round(sum(any(a <= f <= b for a, b in spans) for f in frames) / len(frames), 4) if frames else None

    n_ok = sum(r["correct"] for r in rows)
    return {
        "run_dir": run_dir, "game_json": os.path.basename(game_json),
        "points": {"gt": len(endings), "correct": n_ok, "rows": rows,
                   "spurious_points": spurious, "dead_ball": dead},
        "hits": {"gt": len(strokes), "pred": len(hits), "tol5": m5, "tol12": m12, "lag_aware": lag_match},
        "proposal_coverage": {"strokes": cov(stroke_frames),
                              "rally_endings": cov([e["frame"] for e in endings])},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="用 Extended OpenTTGames 标注核对判罚与击球")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--game-json", required=True)
    ap.add_argument("--fps", type=float, default=120.0)
    a = ap.parse_args(argv)
    r = evaluate(a.run_dir, a.game_json, a.fps)
    with open(os.path.join(a.run_dir, "eval_extended.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    p, h = r["points"], r["hits"]
    print(f"每一分          : {p['correct']}/{p['gt']} 得分方正确")
    for row in p["rows"]:
        ours = (f"{row['our_t']:.3f}s {row['our_winner']} [{row['our_rule']}] 延迟 {row['delay_s']:+.3f}s"
                if row["our_t"] is not None else row.get("note", ""))
        print(f"   - {row['gt_t']:7.3f}s {row['gt']:<22} → {row['gt_winner']} | 我方 {ours} | {'✓' if row['correct'] else '✗'}")
    if p["spurious_points"]:
        print(f"多判的分        : {p['spurious_points']}")
    if p["dead_ball"]:
        print(f"死球（不计分）  : {p['dead_ball']}")
    print(f"击球 ±5 帧      : P {h['tol5']['p']:.3f}  R {h['tol5']['r']:.3f}  F1 {h['tol5']['f1']:.3f}  "
          f"(预测 {h['pred']} / 真值 {h['gt']})")
    print(f"击球 ±12 帧     : P {h['tol12']['p']:.3f}  R {h['tol12']['r']:.3f}  F1 {h['tol12']['f1']:.3f}")
    la = h["lag_aware"]
    fps_ms = 1000.0 / a.fps
    print(f"击球 滞后窗口   : 窗口 {la['window_frames']} 帧  P {la['p']:.3f}  R {la['r']:.3f}  F1 {la['f1']:.3f}  "
          f"侧别一致 {la['side_agree']}")
    if la["lag_median_frames"] is not None:
        print(f"击球 时间滞后   : 中位 {la['lag_median_frames']} 帧（{la['lag_median_frames'] * fps_ms:.0f} ms），"
              f"范围 {la['lag_min_max_frames']} 帧")
    pc = r["proposal_coverage"]
    print(f"提议覆盖        : 击球 {pc['strokes']}  回合结束 {pc['rally_endings']}")
    print(f"已写入 {os.path.join(a.run_dir, 'eval_extended.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
