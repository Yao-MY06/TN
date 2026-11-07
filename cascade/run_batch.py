"""批量运行级联与汇总表。

清单格式（JSON）::

    {"videos": [{
      "name": "test_2",
      "split": "tuning",                         # tuning / holdout
      "tracknet": "_data/obs/tracknet/test_2.jsonl",
      "yolo": "_data/obs/yolo/test_2.jsonl",    # 可选
      "ttnet_cache": "_data/ttnet/test_2.jsonl",# 可选；缺省时只跑 rule
      "events_markup": "_data/.../events_markup.json",
      "extended": "_data/.../game_data/test/test_2.json", # 可选
      "quad": "test_2",                          # 可选，默认由预设推出
      "preset": "test2",                         # 可选
      "fps": 120.0,
      "review_deadline_s": 0,                    # 可选，秒（旧键 *_frames 按 fps 换算）
      "review_latency_s": 0                       # 可选，模拟慢复核（秒）
    }]}

相对路径相对清单文件所在目录解析。输出目录下每个视频有 ``<name>_rule`` 与
``<name>_ttnet`` 两个 run 目录；顶层写 ``summary.md`` 和 ``summary.csv``。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cascade.eval_extended import evaluate as evaluate_extended
from cascade.eval_proposals import evaluate as evaluate_proposals
from cascade.eval_ttnet_outside import evaluate as evaluate_ttnet_outside
from cascade.run_cascade import run, write_outputs
from cascade.ttnet_reviewer import DEFAULT_K


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _resolve(base: str, value: Optional[str]) -> Optional[str]:
    if value is None or value == "":
        return None
    return value if os.path.isabs(value) else os.path.normpath(os.path.join(base, value))


def _default_preset(name: str) -> str:
    key = name.lower()
    if key in ("test_2", "test2"):
        return "test2"
    if key in ("test_4", "test4"):
        return "test4"
    return name


def _load_manifest(path: str) -> List[Dict[str, Any]]:
    raw = _read_json(path)
    entries = raw.get("videos") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise ValueError("清单必须包含非空 videos 列表")
    return entries


def _pct_from_counts(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "p": round(p, 4), "r": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def _reviewer_stats(eval_prop: Dict[str, Any], eval_ext: Optional[Dict[str, Any]],
                    summary: Dict[str, Any], outside: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # 主指标只在标注区间内统计（见 eval_proposals.annotated_spans）；全片口径留在 *_full。
    b = eval_prop.get("confirmed_bounce_tol5_in_markup") or eval_prop["confirmed_bounce_tol5"]
    full = eval_prop["confirmed_bounce_tol5"]
    points = (eval_ext or {}).get("points", {})
    return {
        "tp": b["tp"], "fp": b["fp"], "fn": b["fn"],
        "p": b["p"], "r": b["r"], "f1": b["f1"],
        "f1_full": full["f1"],
        "outside_markup": eval_prop.get("confirmed_outside_markup", 0),
        "points_correct": points.get("correct"),
        "points_total": points.get("gt"),
        "spurious_points": len(points.get("spurious_points", [])) if eval_ext else None,
        "manual_confirm_count": summary.get("manual_confirm_count", 0),
        "unreviewable_count": summary.get("unreviewable_count", 0),
        "timeout_count": summary.get("timeout_count", 0),
        "clips": summary.get("clips", {}).get("total", 0),
        "ttnet_outside_truth_misses": (outside or {}).get("outside_truth_matches"),
    }


def _run_entry(entry: Dict[str, Any], manifest_dir: str, out_dir: str) -> Dict[str, Any]:
    name = str(entry.get("name") or "").strip()
    if not name:
        raise ValueError("清单项缺少 name")
    preset = entry.get("preset") or _default_preset(name)
    quad = entry.get("quad")
    tracknet = _resolve(manifest_dir, entry.get("tracknet"))
    yolo = _resolve(manifest_dir, entry.get("yolo"))
    ttnet_cache = _resolve(manifest_dir, entry.get("ttnet_cache"))
    markup = _resolve(manifest_dir, entry.get("events_markup"))
    extended = _resolve(manifest_dir, entry.get("extended"))
    fps = float(entry.get("fps", 120.0))
    # tracknet = 按方案（TrackNet 连续轨迹，有 yolo 观测时 YOLO 补 TrackNet 丢球的帧）；
    # yolo = 试验：YOLO 主跟踪、TrackNet 只补 YOLO 丢球的帧（需要 yolo 观测）
    primary = str(entry.get("primary", "tracknet"))
    # 方案 §4.1：超时按时间；旧清单里的 *_frames 按本视频 fps 换算成秒。
    deadline = float(entry.get("review_deadline_s", entry.get("review_deadline_frames", 0) / fps))
    latency = float(entry.get("review_latency_s", entry.get("review_latency_frames", 0) / fps))
    split = str(entry.get("split") or ("tuning" if name in ("test_2", "test2") else "holdout"))

    if not tracknet or not os.path.isfile(tracknet):
        raise FileNotFoundError(f"{name}: TrackNet 观测不存在: {tracknet}")
    if not markup or not os.path.isfile(markup):
        raise FileNotFoundError(f"{name}: events_markup 不存在: {markup}")
    if yolo and not os.path.isfile(yolo):
        yolo = None
    if primary == "yolo" and not yolo:
        raise FileNotFoundError(f"{name}: primary=yolo 需要 YOLO 观测")
    if extended and not os.path.isfile(extended):
        extended = None
    if ttnet_cache and not os.path.isfile(ttnet_cache):
        ttnet_cache = None

    row: Dict[str, Any] = {
        "video": name, "split": split, "frames": None,
        "proposal_recall": None, "coverage": None, "ttnet_window_ratio": None,
        "rule": None, "ttnet": None, "notes": [],
    }
    if not ttnet_cache:
        row["notes"].append("缺 TTNet 缓存，未跑 ttnet 组")

    reviewers = [("rule", None)]
    if ttnet_cache:
        reviewers.append(("ttnet", ttnet_cache))

    for reviewer, cache in reviewers:
        run_dir = os.path.join(out_dir, f"{name}_{reviewer}")
        tt_args = None
        if reviewer == "ttnet":
            k = int(entry.get("ttnet_k", DEFAULT_K))
            tt_args = SimpleNamespace(
                ttnet_cache=cache, video=None, ttnet_weights=None, ttnet_src=None,
                ttnet_engine="auto", ttnet_device="cpu", ttnet_k=k)
        res = run(
            tracknet, preset=preset, quad=quad, reviewer=reviewer,
            yolo_jsonl=yolo, review_latency_s=latency,
            review_deadline_s=deadline,
            pre_ctx=(8 - int(entry.get("ttnet_k", DEFAULT_K))) if reviewer == "ttnet" else 0,
            post_ctx=int(entry.get("ttnet_k", DEFAULT_K)) if reviewer == "ttnet" else 0,
            fps=fps, ttnet_args=tt_args, primary=primary,
            yolo_recapture=bool(entry.get("yolo_recapture", True)),
            net_hit_hold=bool(entry.get("net_hit_hold", False)),
            net_trigger=bool(entry.get("net_trigger", False)),
        )
        write_outputs(run_dir, res)
        eval_prop = evaluate_proposals(run_dir, markup, tracknet, fps)
        _write_json(os.path.join(run_dir, "eval.json"), eval_prop)
        eval_ext = None
        if extended:
            eval_ext = evaluate_extended(run_dir, extended, fps)
            _write_json(os.path.join(run_dir, "eval_extended.json"), eval_ext)
        outside = None
        if reviewer == "ttnet":
            outside = evaluate_ttnet_outside(cache, os.path.join(run_dir, "clips.jsonl"),
                                             markup, k=int(entry.get("ttnet_k", DEFAULT_K)),
                                             fps=fps)
            _write_json(os.path.join(run_dir, "eval_ttnet_outside.json"), outside)

        if row["frames"] is None:
            row["frames"] = res["summary"]["frames"]
            row["proposal_recall"] = eval_prop["proposal_recall_bounce"]
            row["coverage"] = eval_prop["coverage"]
            row["ttnet_window_ratio"] = eval_prop["ttnet_window_ratio"]
        row[reviewer] = _reviewer_stats(eval_prop, eval_ext, res["summary"], outside)
        row[reviewer]["_gt_bounce"] = eval_prop["gt_bounce"]
        row[reviewer]["_proposal_hits"] = eval_prop["gt_bounce"] - len(eval_prop["missed_bounces"])
    return row


def _blank(v: Any, digits: Optional[int] = None) -> str:
    if v is None:
        return "-"
    if digits is not None and isinstance(v, (int, float)):
        return f"{v:.{digits}f}"
    return str(v)


def _metric_pair(row: Dict[str, Any], reviewer: str, key: str) -> str:
    stats = row.get(reviewer)
    return _blank(stats.get(key) if stats else None, 3)


def aggregate_holdout(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """把 split=holdout 的计数汇总后重算指标；不做各视频 F1 平均。"""
    selected = [r for r in rows if r["split"] == "holdout"]
    if not selected:
        return None
    frames = sum(r["frames"] or 0 for r in selected)
    total_clips = sum((r.get(reviewer) or {}).get("clips", 0)
                      for r in selected for reviewer in ("rule", "ttnet"))
    out: Dict[str, Any] = {
        "video": "__holdout_total__", "split": "holdout", "frames": frames,
        "proposal_recall": None, "coverage": None, "ttnet_window_ratio": None,
        "rule": None, "ttnet": None, "notes": [], "n_videos": len(selected),
    }
    if frames:
        out["coverage"] = round(sum((r["coverage"] or 0.0) * (r["frames"] or 0)
                                    for r in selected) / frames, 4)
        out["ttnet_window_ratio"] = round(
            sum((r["ttnet_window_ratio"] or 0.0) * (r["frames"] or 0)
                for r in selected) / frames, 4)

    for reviewer in ("rule", "ttnet"):
        stats = [(r.get(reviewer) or {}) for r in selected if r.get(reviewer)]
        if not stats:
            continue
        tp = sum(s["tp"] for s in stats)
        fp = sum(s["fp"] for s in stats)
        fn = sum(s["fn"] for s in stats)
        counts = _pct_from_counts(tp, fp, fn)
        points = [s for s in stats if s.get("points_total") is not None]
        points_correct = sum(s["points_correct"] for s in points) if points else None
        points_total = sum(s["points_total"] for s in points) if points else None
        outside = [s["ttnet_outside_truth_misses"] for s in stats
                   if s.get("ttnet_outside_truth_misses") is not None]
        clips = sum(s["clips"] for s in stats)
        out[reviewer] = {
            **counts,
            "points_correct": points_correct, "points_total": points_total,
            "spurious_points": sum(s["spurious_points"] for s in stats
                                   if s.get("spurious_points") is not None) if points else None,
            "manual_confirm_count": sum(s["manual_confirm_count"] for s in stats),
            "unreviewable_count": sum(s.get("unreviewable_count", 0) for s in stats),
            "timeout_count": sum(s["timeout_count"] for s in stats),
            "outside_markup": sum(s.get("outside_markup", 0) for s in stats),
            "clips": clips,
            "ttnet_outside_truth_misses": sum(outside) if outside else None,
            "_gt_bounce": sum(s["_gt_bounce"] for s in stats),
            "_proposal_hits": sum(s["_proposal_hits"] for s in stats),
        }
        if out[reviewer]["_gt_bounce"]:
            out["proposal_recall"] = round(out[reviewer]["_proposal_hits"]
                                           / out[reviewer]["_gt_bounce"], 4)

    # 分母用两个 reviewer 的片段总数；任一组缺失时仍以已有运行计数为准。
    n_clip_observations = total_clips
    if n_clip_observations:
        all_manual = sum((r.get(reviewer) or {}).get("manual_confirm_count", 0)
                         for r in selected for reviewer in ("rule", "ttnet"))
        all_timeout = sum((r.get(reviewer) or {}).get("timeout_count", 0)
                          for r in selected for reviewer in ("rule", "ttnet"))
        out["manual_confirm_ratio"] = round(all_manual / n_clip_observations, 4)
        out["timeout_ratio"] = round(all_timeout / n_clip_observations, 4)
    return out


def _display_row(row: Dict[str, Any]) -> Dict[str, Any]:
    def ratio(row_: Dict[str, Any], reviewer: str, key: str) -> Optional[float]:
        stats = row_.get(reviewer) or {}
        clips = stats.get("clips", 0)
        return round(stats.get(key, 0) / clips, 4) if clips else None

    return {
        "video": row["video"], "split": row["split"], "frames": row["frames"],
        "proposal_recall": row.get("proposal_recall"),
        "coverage": row.get("coverage"),
        "ttnet_window_ratio": row.get("ttnet_window_ratio"),
        "rule_p": _metric_pair(row, "rule", "p"),
        "rule_r": _metric_pair(row, "rule", "r"),
        "rule_f1": _metric_pair(row, "rule", "f1"),
        "ttnet_p": _metric_pair(row, "ttnet", "p"),
        "ttnet_r": _metric_pair(row, "ttnet", "r"),
        "ttnet_f1": _metric_pair(row, "ttnet", "f1"),
        "rule_points": (f"{(row.get('rule') or {}).get('points_correct')}/"
                        f"{(row.get('rule') or {}).get('points_total')}"
                        if row.get("rule") and (row.get("rule") or {}).get("points_total") is not None else "-"),
        "ttnet_points": (f"{(row.get('ttnet') or {}).get('points_correct')}/"
                         f"{(row.get('ttnet') or {}).get('points_total')}"
                         if row.get("ttnet") and (row.get("ttnet") or {}).get("points_total") is not None else "-"),
        "rule_outside_markup": (row.get("rule") or {}).get("outside_markup"),
        "ttnet_outside_markup": (row.get("ttnet") or {}).get("outside_markup"),
        "rule_spurious": (row.get("rule") or {}).get("spurious_points"),
        "ttnet_spurious": (row.get("ttnet") or {}).get("spurious_points"),
        "rule_manual_confirm_ratio": ratio(row, "rule", "manual_confirm_count"),
        "ttnet_manual_confirm_ratio": ratio(row, "ttnet", "manual_confirm_count"),
        "rule_unreviewable_count": (row.get("rule") or {}).get("unreviewable_count"),
        "ttnet_unreviewable_count": (row.get("ttnet") or {}).get("unreviewable_count"),
        "rule_timeout_ratio": ratio(row, "rule", "timeout_count"),
        "ttnet_timeout_ratio": ratio(row, "ttnet", "timeout_count"),
        "ttnet_outside_truth_misses": (row.get("ttnet") or {}).get("ttnet_outside_truth_misses"),
        "notes": "；".join(row.get("notes", [])),
    }


CSV_FIELDS = [
    "video", "split", "frames", "proposal_recall", "coverage", "ttnet_window_ratio",
    "rule_p", "rule_r", "rule_f1", "ttnet_p", "ttnet_r", "ttnet_f1",
    "rule_outside_markup", "ttnet_outside_markup",
    "rule_points", "ttnet_points", "rule_spurious", "ttnet_spurious",
    "rule_manual_confirm_ratio", "ttnet_manual_confirm_ratio",
    "rule_unreviewable_count", "ttnet_unreviewable_count",
    "rule_timeout_ratio", "ttnet_timeout_ratio", "ttnet_outside_truth_misses", "notes",
]


def _write_summary(out_dir: str, rows: List[Dict[str, Any]], total: Optional[Dict[str, Any]]) -> None:
    display = [_display_row(r) for r in rows]
    if total is not None:
        display.append(_display_row(total))

    with open(os.path.join(out_dir, "summary.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(display)

    headers = ["视频", "划分", "帧数", "提议召回", "覆盖", "TTNet窗口", "规则 P/R/F1",
               "TTNet P/R/F1", "空档检出(规则/TTNet)", "规则得分", "TTNet得分", "多判(规则/TTNet)",
               "人工确认(规则/TTNet)", "超时(规则/TTNet)", "片段外真值漏检", "备注"]
    lines = ["# 级联批量汇总", "",
             "| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for d in display:
        rule_prf = "/".join(_blank(d[k]) for k in ("rule_p", "rule_r", "rule_f1"))
        tt_prf = "/".join(_blank(d[k]) for k in ("ttnet_p", "ttnet_r", "ttnet_f1"))
        manual = "/".join(_blank(d[k], 3) for k in ("rule_manual_confirm_ratio",
                                                     "ttnet_manual_confirm_ratio"))
        timeout = "/".join(_blank(d[k], 3) for k in ("rule_timeout_ratio",
                                                      "ttnet_timeout_ratio"))
        spurious = "/".join(_blank(d[k]) for k in ("rule_spurious", "ttnet_spurious"))
        unlabeled = "/".join(_blank(d[k]) for k in ("rule_outside_markup", "ttnet_outside_markup"))
        vals = [
            d["video"], d["split"], _blank(d["frames"]),
            _blank(d["proposal_recall"], 4), _blank(d["coverage"], 4),
            _blank(d["ttnet_window_ratio"], 4), rule_prf, tt_prf, unlabeled, d["rule_points"],
            d["ttnet_points"], spurious, manual, timeout,
            _blank(d["ttnet_outside_truth_misses"]), d["notes"] or "-",
        ]
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    lines += ["", "说明：holdout 合计的 P/R/F1 按 tp/fp/fn 汇总后重算，不是各视频 F1 平均。",
              "P/R/F1 只在 events_markup 的标注区间内统计：相邻标注间隔超过 30 s 视为未标注空档"
              "（test_1 / test_4 / test_7 各有一段），空档内的检出不计入，条数见“空档检出”列。",
              "人工确认比例不含复核器无法判断的片段（规则复核遇到只有丢球触发的片段，"
              "原因码 no_reviewable_candidate），其条数见 summary.csv 的 *_unreviewable_count。", ""]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def run_batch(manifest: str, out_dir: str) -> Dict[str, Any]:
    manifest = os.path.abspath(manifest)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    manifest_dir = os.path.dirname(manifest)
    entries = _load_manifest(manifest)
    rows = [_run_entry(entry, manifest_dir, out_dir) for entry in entries]
    total = aggregate_holdout(rows)
    _write_summary(out_dir, rows, total)
    return {"rows": rows, "holdout_total": total,
            "summary_md": os.path.join(out_dir, "summary.md"),
            "summary_csv": os.path.join(out_dir, "summary.csv")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="批量运行 rule/ttnet 级联并生成汇总表")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default=os.path.join("outputs", "cascade_batch"))
    a = ap.parse_args(argv)
    result = run_batch(a.manifest, a.out_dir)
    print(f"汇总表：{result['summary_md']}")
    print(f"CSV   ：{result['summary_csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
