"""球拍检测证据：给 RACKET_HIT 分类 NEAR/FAR/NONE/OTHER_SIDE。

参数只用 test_2 校准：窗口中心相对击球事件前移 0.05 s，半宽 0.15 s，
置信度阈值 0.50，框到球距离按球拍框对角线归一化，NEAR 阈值 3.0。
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .eval_extended import parse_game_json
from .racket_cache import load_racket_obs

DEFAULT_DT_S = 0.15
DEFAULT_OFFSET_S = -0.05
DEFAULT_CONF = 0.50
DEFAULT_DISTANCE_RATIO = 3.0
DEFAULT_STATIC_MIN_S = 2.0
DEFAULT_STATIC_MAX_GAP_S = 0.20
DEFAULT_STATIC_CENTER_BIN_PX = 32
DEFAULT_STATIC_SIZE_BIN_PX = 8



def nearest_ball_lookup(ball_by_frame: Dict[int, Tuple[float, float]],
                        max_delta_frames: int = 3) -> Callable[[int], Optional[Tuple[float, float]]]:
    def lookup(frame_id: int):
        for delta in range(0, max_delta_frames + 1):
            for fid in ((frame_id,) if delta == 0 else (frame_id - delta, frame_id + delta)):
                if fid in ball_by_frame:
                    return ball_by_frame[fid]
        return None
    return lookup

def _bin(value: float, size: float) -> int:
    return int(math.floor(float(value) / max(float(size), 1e-9)))


def filter_static_observations(
        obs_by_frame: Dict[int, List[Dict[str, Any]]], *,
        min_static_s: float = DEFAULT_STATIC_MIN_S,
        max_gap_s: float = DEFAULT_STATIC_MAX_GAP_S,
        center_bin_px: float = DEFAULT_STATIC_CENTER_BIN_PX,
        size_bin_px: float = DEFAULT_STATIC_SIZE_BIN_PX) -> Dict[str, Any]:
    """按“位置/框大小几乎不变且持续足够久”标记静止目标；不依赖 track_id。"""
    if min_static_s < 0 or max_gap_s < 0:
        raise ValueError("静态过滤的持续时间/间隔必须 >= 0 秒")
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    total = 0
    for rows in obs_by_frame.values():
        for obs in rows:
            if not obs.get("visible") or not obs.get("bbox_xyxy") or not obs.get("center_px"):
                continue
            total += 1
            cx, cy = (float(v) for v in obs["center_px"][:2])
            x1, y1, x2, y2 = (float(v) for v in obs["bbox_xyxy"][:4])
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            key = (obs.get("side", "UNKNOWN"), _bin(cx, center_bin_px), _bin(cy, center_bin_px),
                   _bin(w, size_bin_px), _bin(h, size_bin_px))
            groups.setdefault(key, []).append(obs)
    filtered = 0
    by_side: Dict[str, int] = {}
    clusters = 0
    for rows in groups.values():
        rows.sort(key=lambda o: float(o["timestamp_s"]))
        run: List[Dict[str, Any]] = []
        for obs in rows + [None]:
            if obs is not None and (not run or
                                    float(obs["timestamp_s"]) - float(run[-1]["timestamp_s"]) <= max_gap_s):
                run.append(obs)
                continue
            if run:
                duration = float(run[-1]["timestamp_s"]) - float(run[0]["timestamp_s"])
                if duration >= min_static_s:
                    clusters += 1
                    for item in run:
                        item["_static_filtered"] = True
                        side = str(item.get("side", "UNKNOWN"))
                        by_side[side] = by_side.get(side, 0) + 1
                    filtered += len(run)
            run = [obs] if obs is not None else []
    return {"visible": total, "filtered": filtered, "clusters": clusters,
            "by_side": by_side, "min_static_s": min_static_s,
            "max_gap_s": max_gap_s, "center_bin_px": center_bin_px,
            "size_bin_px": size_bin_px}


def point_to_bbox_ratio(point: Tuple[float, float], bbox: Iterable[float]) -> float:
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = (float(v) for v in bbox)
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    diag = math.hypot(x2 - x1, y2 - y1)
    return math.hypot(dx, dy) / max(diag, 1e-6)


class RacketEvidence:
    def __init__(self, obs_by_frame: Optional[Dict[int, List[Dict[str, Any]]]] = None, *,
                 path: Optional[str] = None, static_filter: bool = False,
                 dt_s: float = DEFAULT_DT_S, offset_s: float = DEFAULT_OFFSET_S,
                 conf: float = DEFAULT_CONF,
                 distance_ratio: float = DEFAULT_DISTANCE_RATIO,
                 static_min_s: float = DEFAULT_STATIC_MIN_S,
                 static_max_gap_s: float = DEFAULT_STATIC_MAX_GAP_S):
        if dt_s < 0:
            raise ValueError("dt_s 必须 >= 0")
        self.path = path
        self.missing = (obs_by_frame is None) and (path is None or not os.path.isfile(path))
        self.dt_s = float(dt_s)
        self.offset_s = float(offset_s)
        self.conf = float(conf)
        self.distance_ratio = float(distance_ratio)
        raw = dict(obs_by_frame) if obs_by_frame is not None else {}
        self.all_obs_by_frame = raw
        self.static_stats = filter_static_observations(
            raw, min_static_s=static_min_s, max_gap_s=static_max_gap_s
        ) if static_filter else {"visible": 0, "filtered": 0, "by_side": {}}
        self.static_filter = bool(static_filter)
        flat: List[Dict[str, Any]] = []
        for rows in raw.values():
            for obs in rows:
                if (obs.get("visible") and obs.get("side") in ("A", "B")
                        and not obs.get("_static_filtered")):
                    flat.append(obs)
        flat.sort(key=lambda o: float(o["timestamp_s"]))
        self.observations = flat
        self.times = [float(o["timestamp_s"]) for o in flat]

    @classmethod
    def from_cache(cls, path: Optional[str], **kwargs) -> "RacketEvidence":
        if path and os.path.isfile(path):
            return cls(load_racket_obs(path), path=path, **kwargs)
        return cls(None, path=path, **kwargs)

    def decision_time(self, hit_t: float) -> float:
        return float(hit_t) + self.offset_s + self.dt_s

    def has_same_side_candidates(self, hit_side: str, hit_t: float) -> bool:
        start = float(hit_t) + self.offset_s - self.dt_s
        end = float(hit_t) + self.offset_s + self.dt_s
        lo = bisect.bisect_left(self.times, start - 1e-9)
        hi = bisect.bisect_right(self.times, end + 1e-9)
        return any(obs.get("side") == hit_side
                   and float(obs.get("confidence") or 0.0) >= self.conf
                   for obs in self.observations[lo:hi])

    def classify(self, hit_side: str, hit_t: float,
                 hit_position: Optional[Iterable[float]],
                 ball_at_frame: Optional[Callable[[int], Optional[Tuple[float, float]]]] = None
                 ) -> Dict[str, Any]:
        start = float(hit_t) + self.offset_s - self.dt_s
        end = float(hit_t) + self.offset_s + self.dt_s
        lo = bisect.bisect_left(self.times, start - 1e-9)
        hi = bisect.bisect_right(self.times, end + 1e-9)
        same: List[Dict[str, Any]] = []
        other: List[Dict[str, Any]] = []
        for obs in self.observations[lo:hi]:
            if float(obs.get("confidence") or 0.0) < self.conf:
                continue
            (same if obs.get("side") == hit_side else other).append(obs)
        fallback = tuple(float(v) for v in (hit_position or (0.0, 0.0))[:2])
        best: Optional[Tuple[float, Dict[str, Any]]] = None
        for obs in same:
            point = None
            if ball_at_frame is not None:
                point = ball_at_frame(int(obs["frame_id"]))
            ratio = point_to_bbox_ratio(point or fallback, obs["bbox_xyxy"])
            if best is None or ratio < best[0]:
                best = (ratio, obs)
        if best is not None and best[0] <= self.distance_ratio:
            label = "NEAR"
        elif best is not None:
            label = "FAR"
        elif other:
            label = "OTHER_SIDE"
        else:
            label = "NONE"
        best_obs = best[1] if best else (other[0] if other else None)
        return {
            "label": label,
            "window_s": [round(start, 6), round(end, 6)],
            "decision_time_s": round(self.decision_time(hit_t), 6),
            "distance_ratio": round(best[0], 4) if best else None,
            "racket_frame_id": int(best_obs["frame_id"]) if best_obs else None,
            "racket_side": best_obs.get("side") if best_obs else None,
            "racket_confidence": best_obs.get("confidence") if best_obs else None,
            "missing_cache": self.missing,
            "static_filter": self.static_filter,
        }

    def describe(self) -> Dict[str, Any]:
        return {
            "path": self.path, "missing": self.missing,
            "dt_s": self.dt_s, "offset_s": self.offset_s, "conf": self.conf,
            "distance_ratio": self.distance_ratio, "static_filter": self.static_filter,
            "static_stats": self.static_stats,
        }
# ---------------------------------------------------------------------------
# 证据报告
# ---------------------------------------------------------------------------

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _game_jsons(manifest: str) -> Dict[str, str]:
    with open(manifest, encoding="utf-8") as f:
        doc = json.load(f)
    base = os.path.dirname(os.path.abspath(manifest))
    out = {}
    for entry in doc.get("videos", []):
        p = entry.get("extended")
        if p:
            out[str(entry["name"])] = p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))
    return out


def match_hit_truth(hits: List[Dict[str, Any]], strokes: List[Dict[str, Any]],
                    fps: float = 120.0) -> Dict[int, Dict[str, Any]]:
    """与 eval_extended 的 lag-aware 窗口一致，并要求击球侧别相同。"""
    used = set()
    truth: Dict[int, Dict[str, Any]] = {}
    for stroke in strokes:
        candidates = []
        for i, hit in enumerate(hits):
            if i in used or hit.get("side") != stroke.get("side"):
                continue
            lag = round(float(hit["timestamp_s"]) * fps) - int(stroke["frame"])
            if -5 <= lag <= 25:
                candidates.append((abs(lag), i, lag))
        if candidates:
            _, i, lag = min(candidates)
            used.add(i)
            truth[i] = {"stroke_frame": int(stroke["frame"]), "lag_frames": lag}
    return truth


def _ball_lookup(run_dir: str):
    rows = _read_jsonl(os.path.join(run_dir, "fused_obs.jsonl"))
    by_frame = {int(o["frame_id"]): tuple(float(v) for v in o["center_px"][:2])
                for o in rows if o.get("visible") and o.get("center_px")}
    return nearest_ball_lookup(by_frame)


def analyze_hits(run_dir: str, game_json: str, racket_path: str, *,
                 video: str, reviewer: str, fps: float = 120.0,
                 static_filter: bool = False,
                 dt_s: float = DEFAULT_DT_S, offset_s: float = DEFAULT_OFFSET_S,
                 conf: float = DEFAULT_CONF,
                 distance_ratio: float = DEFAULT_DISTANCE_RATIO) -> Dict[str, Any]:
    strokes, _ = parse_game_json(game_json)
    hits = [e for e in _read_jsonl(os.path.join(run_dir, "events.jsonl"))
            if e.get("type") == "RACKET_HIT"]
    truth = match_hit_truth(hits, strokes, fps)
    provider = RacketEvidence.from_cache(
        racket_path, static_filter=static_filter, dt_s=dt_s, offset_s=offset_s,
        conf=conf, distance_ratio=distance_ratio)
    lookup = _ball_lookup(run_dir)
    rows = []
    for i, hit in enumerate(hits):
        ev = provider.classify(hit.get("side"), hit["timestamp_s"], hit.get("position_px"), lookup)
        rows.append({
            "video": video, "reviewer": reviewer, "static_filter": bool(static_filter),
            "hit_t": round(float(hit["timestamp_s"]), 6), "hit_side": hit.get("side"),
            "is_true": i in truth, "label": ev["label"],
            "distance_ratio": ev["distance_ratio"],
            "racket_confidence": ev["racket_confidence"],
            "racket_frame_id": ev["racket_frame_id"], "racket_side": ev["racket_side"],
            "window_start_s": ev["window_s"][0], "window_end_s": ev["window_s"][1],
            "missing_cache": ev["missing_cache"],
        })
    return {"rows": rows, "provider": provider.describe(), "gt_strokes": len(strokes)}


def summarize_evidence(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    groups: Dict[Tuple[str, str, bool], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["video"], row["reviewer"], row["static_filter"]), []).append(row)
    for (video, reviewer, static_filter), items in sorted(groups.items()):
        for label in ("NEAR", "FAR", "NONE", "OTHER_SIDE"):
            cls = [r for r in items if r["label"] == label]
            t = sum(1 for r in cls if r["is_true"])
            f = len(cls) - t
            out.append({"video": video, "reviewer": reviewer,
                        "static_filter": "after" if static_filter else "before",
                        "label": label, "true_hits": t, "false_hits": f,
                        "precision": round(t / len(cls), 4) if cls else None,
                        "total_hits": len(items)})
    return out


def _aggregate_classes(rows: List[Dict[str, Any]], videos: set[str]) -> Dict[str, Dict[str, int]]:
    selected = [r for r in rows if r["video"] in videos and not r["static_filter"]]
    out = {}
    for label in ("NEAR", "FAR", "NONE", "OTHER_SIDE"):
        cls = [r for r in selected if r["label"] == label]
        t = sum(1 for r in cls if r["is_true"])
        out[label] = {"true": t, "false": len(cls) - t}
    return out


def _fmt_pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.1%})" if d else "0/0"


def _md_table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    return ["| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|"] +                 [
                "| " + " | ".join("-" if v is None else str(v) for v in row) + " |"
                for row in rows]


def tune_test2(manifest: str, runs_root: str, racket_dir: str, fps: float = 120.0) -> List[Dict[str, Any]]:
    game = _game_jsons(manifest)["test_2"]
    run_dir = os.path.join(runs_root, "test_2_rule")
    hits = [e for e in _read_jsonl(os.path.join(run_dir, "events.jsonl"))
            if e.get("type") == "RACKET_HIT"]
    strokes, _ = parse_game_json(game)
    truth = match_hit_truth(hits, strokes, fps)
    lookup = _ball_lookup(run_dir)
    raw = load_racket_obs(os.path.join(racket_dir, "test_2.jsonl"))
    results = []
    for dt in (0.08, 0.12, 0.15, 0.20):
        for offset in (-0.20, -0.15, -0.10, -0.05):
            for conf in (0.25, 0.35, 0.50):
                for ratio in (0.7, 1.0, 1.2, 1.5, 2.0, 3.0):
                    provider = RacketEvidence(raw, dt_s=dt, offset_s=offset, conf=conf,
                                              distance_ratio=ratio)
                    counts = {k: [0, 0] for k in ("NEAR", "FAR", "NONE", "OTHER_SIDE")}
                    for i, hit in enumerate(hits):
                        ev = provider.classify(hit.get("side"), hit["timestamp_s"],
                                               hit.get("position_px"), lookup)
                        counts[ev["label"]][0 if i in truth else 1] += 1
                    ft, ff = counts["FAR"]
                    results.append({"dt_s": dt, "offset_s": offset, "conf": conf,
                                    "distance_ratio": ratio, "far_true": ft,
                                    "far_false": ff, "gain": ff - ft,
                                    "counts": counts})
    results.sort(key=lambda r: (r["far_true"], -r["far_false"], r["distance_ratio"],
                                r["dt_s"], abs(r["offset_s"])))
    return results


def write_evidence_report(manifest: str, runs_root: str, racket_dir: str,
                          out_dir: str, fps: float = 120.0) -> Dict[str, str]:
    games = _game_jsons(manifest)
    all_rows = []
    for n in range(1, 8):
        video = f"test_{n}"
        for reviewer in ("rule", "ttnet"):
            run_dir = os.path.join(runs_root, f"{video}_{reviewer}")
            if not os.path.isfile(os.path.join(run_dir, "events.jsonl")):
                continue
            result = analyze_hits(
                run_dir, games[video], os.path.join(racket_dir, f"{video}.jsonl"),
                video=video, reviewer=reviewer, fps=fps)
            all_rows.extend(result["rows"])
    # test_7 另跑静止过滤打开的一版。
    t7 = []
    for reviewer in ("rule", "ttnet"):
        run_dir = os.path.join(runs_root, f"test_7_{reviewer}")
        if os.path.isfile(os.path.join(run_dir, "events.jsonl")):
            t7.extend(analyze_hits(
                run_dir, games["test_7"], os.path.join(racket_dir, "test_7.jsonl"),
                video="test_7", reviewer=reviewer, fps=fps, static_filter=True)["rows"])
    rows_with_static = all_rows + t7
    summary = summarize_evidence(rows_with_static)
    tune = tune_test2(manifest, runs_root, racket_dir, fps)
    selected = next(r for r in tune if r["dt_s"] == DEFAULT_DT_S and r["offset_s"] == DEFAULT_OFFSET_S
                    and r["conf"] == DEFAULT_CONF and r["distance_ratio"] == DEFAULT_DISTANCE_RATIO)

    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "evidence.md")
    csv_path = os.path.join(out_dir, "evidence.csv")
    hits_path = os.path.join(out_dir, "evidence_hits.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        fields = ["video", "reviewer", "static_filter", "label", "true_hits",
                  "false_hits", "precision", "total_hits"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    with open(hits_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]) if all_rows else [])
        if all_rows:
            writer.writeheader()
            writer.writerows(all_rows)

    lines = ["# 球拍证据分析", "",
             "## 固定参数（只在 test_2 上扫）", "",
             f"- Δt = {DEFAULT_DT_S:.2f} s",
             f"- 窗口中心偏移 = {DEFAULT_OFFSET_S:+.2f} s",
             f"- 球拍置信度阈值 c = {DEFAULT_CONF:.2f}",
             f"- NEAR 归一化距离阈值 = {DEFAULT_DISTANCE_RATIO:.1f} 倍框对角线",
             f"- 真击球匹配窗口与 `eval_extended` 相同：[-5, +25] 帧，且侧别相同",
             f"- 选中配置在 test_2：FAR 真={selected['far_true']}、假={selected['far_false']}；"
             f"NEAR 真={selected['counts']['NEAR'][0]}、假={selected['counts']['NEAR'][1]}。", ""]
    lines += ["### test_2 参数扫描（按 FAR 真击球损失最少、假击球移除最多排序）", ""]
    tune_rows = [[r["dt_s"], r["offset_s"], r["conf"], r["distance_ratio"],
                  r["far_true"], r["far_false"]] for r in tune[:12]]
    lines += _md_table(["Δt(s)", "偏移(s)", "c", "距离阈值", "FAR真", "FAR假"], tune_rows)
    lines += ["", "## 每段视频四类分布", ""]
    cls_rows = []
    for r in summary:
        if r["static_filter"] != "before":
            continue
        mark = "（球拍训练集）" if r["video"] == "test_4" else ""
        cls_rows.append([r["video"] + mark, r["reviewer"], r["label"], r["true_hits"],
                         r["false_hits"], "-" if r["precision"] is None else f"{r['precision']:.3f}"])
    lines += _md_table(["视频", "复核", "类别", "真击球", "假击球", "精确率"], cls_rows)
    lines += ["", "## 留出集合计", ""]
    holdout_all = {f"test_{n}" for n in (1, 3, 4, 5, 6, 7)}
    holdout_no4 = {f"test_{n}" for n in (1, 3, 5, 6, 7)}
    agg_rows = []
    for reviewer in ("rule", "ttnet"):
        rows_rev = [r for r in all_rows if r["reviewer"] == reviewer]
        for scope, vids in (("含 test_4", holdout_all), ("不含 test_4", holdout_no4)):
            co = _aggregate_classes(rows_rev, vids)
            agg_rows.append([reviewer, scope, co["NEAR"]["true"], co["NEAR"]["false"],
                             co["FAR"]["true"], co["FAR"]["false"],
                             co["NONE"]["true"], co["OTHER_SIDE"]["false"]])
    lines += _md_table(["复核", "留出集", "NEAR真", "NEAR假", "FAR真", "FAR假", "NONE真", "异侧假"], agg_rows)
    lines += ["", "结论：去掉 FAR 会丢掉多少真击球，直接看每个留出集里的 FAR真；FAR假是可移除的假击球上限。", ""]
    lines += ["## test_7 静止过滤", ""]
    t7_rows = []
    for sf in (False, True):
        rev_rows = [r for r in rows_with_static if r["video"] == "test_7"
                    and r["reviewer"] == "rule" and r["static_filter"] == sf]
        co = {k: {"true": 0, "false": 0} for k in ("NEAR", "FAR", "NONE", "OTHER_SIDE")}
        for r in rev_rows:
            co[r["label"]]["true" if r["is_true"] else "false"] += 1
        t7_rows.append(["过滤前" if not sf else "过滤后",
                        co["NEAR"]["true"], co["FAR"]["true"], co["FAR"]["false"],
                        co["NONE"]["true"], co["OTHER_SIDE"]["false"]])
    lines += _md_table(["版本", "NEAR真", "FAR真", "FAR假", "NONE真", "异侧假"], t7_rows)
    t7_static = RacketEvidence.from_cache(
        os.path.join(racket_dir, "test_7.jsonl"), static_filter=True).static_stats
    lines += ["", f"test_7 静止过滤：visible {t7_static.get('visible', 0)}，"
              f"过滤 {t7_static.get('filtered', 0)}，按侧别 {t7_static.get('by_side', {})}。",
              "静止 UNKNOWN 检出不参与 A/B 证据；过滤后 A/B 类证据变化见上表。", ""]
    with open(md_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    return {"markdown": md_path, "csv": csv_path, "hits_csv": hits_path,
            "tuning": tune}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="球拍检测证据分析（只读已有输出和缓存）")
    ap.add_argument("--manifest", default="cascade/batch_manifest.json")
    ap.add_argument("--runs-root", default="outputs/cascade_batch_sm/tn")
    ap.add_argument("--racket-dir", default="_data/racket")
    ap.add_argument("--out-dir", default="outputs/racket_eval")
    ap.add_argument("--fps", type=float, default=120.0)
    a = ap.parse_args(argv)
    paths = write_evidence_report(a.manifest, a.runs_root, a.racket_dir, a.out_dir, a.fps)
    print(f"证据报告：{paths['markdown']}")
    print(f"证据 CSV：{paths['csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())