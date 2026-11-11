"""对比球拍证据 off/shadow/drop 的批量输出（只读已有结果）。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cascade.eval_extended import parse_game_json  # noqa: E402
from cascade.eval_proposals import _read_jsonl  # noqa: E402
from cascade.racket_evidence import match_hit_truth  # noqa: E402

RUN_RE = re.compile(r"^(?P<video>.+)_(?P<reviewer>rule|ttnet)$")
REVIEWERS = ("rule", "ttnet")
SCOPES = ("test_2", "holdout_with_test_4", "holdout_without_test_4")


def _json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _manifest_games(manifest: str) -> Dict[str, str]:
    with open(manifest, encoding="utf-8") as f:
        doc = json.load(f)
    base = os.path.dirname(os.path.abspath(manifest))
    return {str(e["name"]): (e["extended"] if os.path.isabs(e["extended"])
                             else os.path.normpath(os.path.join(base, e["extended"])))
            for e in doc.get("videos", []) if e.get("extended")}


def _load_runs(root: str, games: Dict[str, str], fps: float) -> List[Dict[str, Any]]:
    runs = []
    for name in os.listdir(root):
        m = RUN_RE.match(name)
        if not m:
            continue
        run_dir = os.path.join(root, name)
        if not os.path.isfile(os.path.join(run_dir, "eval_extended.json")):
            continue
        summary = _json(os.path.join(run_dir, "summary.json"))
        mode = str((summary.get("commit") or {}).get("racket_hit_hold") or
                   os.path.basename(os.path.normpath(root)))
        ev = _json(os.path.join(run_dir, "eval_extended.json"))
        points = ev.get("points") or {}
        hits = [e for e in _read_jsonl(os.path.join(run_dir, "events.jsonl"))
                if e.get("type") == "RACKET_HIT"]
        strokes, _ = parse_game_json(games[m.group("video")])
        matched = match_hit_truth(hits, strokes, fps)
        runs.append({
            "mode": mode, "video": m.group("video"), "reviewer": m.group("reviewer"),
            "run_dir": run_dir, "rows": points.get("rows") or [],
            "spurious": len(points.get("spurious_points") or []),
            "dead": len(points.get("dead_ball") or []),
            "hits": len(hits), "hit_tp": len(matched), "strokes": len(strokes),
            "summary": summary,
        })
    return runs


def _scope_selected(runs: List[Dict[str, Any]], mode: str, reviewer: str,
                    scope: str) -> List[Dict[str, Any]]:
    selected = [r for r in runs if r["mode"] == mode and r["reviewer"] == reviewer]
    if scope == "test_2":
        return [r for r in selected if r["video"] == "test_2"]
    if scope == "holdout_with_test_4":
        return [r for r in selected if r["video"] != "test_2"]
    return [r for r in selected if r["video"] not in ("test_2", "test_4")]


def _aggregate(runs: List[Dict[str, Any]], mode: str, reviewer: str,
               scope: str) -> Dict[str, Any]:
    selected = _scope_selected(runs, mode, reviewer, scope)
    rows = [row for run in selected for row in run["rows"]]
    out = {
        "mode": mode, "reviewer": reviewer, "scope": scope, "gt": len(rows),
        "correct": 0, "missed_out": 0, "missed_net": 0, "missed_other": 0,
        "wrong": 0, "spurious": sum(r["spurious"] for r in selected),
        "dead": sum(r["dead"] for r in selected),
        "hits": sum(r["hits"] for r in selected), "hit_tp": sum(r["hit_tp"] for r in selected),
        "strokes": sum(r["strokes"] for r in selected),
    }
    for row in rows:
        gt = str(row.get("gt") or "")
        if row.get("our_t") is None:
            if gt.endswith("_out"):
                out["missed_out"] += 1
            elif gt.endswith("_net"):
                out["missed_net"] += 1
            else:
                out["missed_other"] += 1
        elif row.get("correct"):
            out["correct"] += 1
        else:
            out["wrong"] += 1
    out["hit_precision"] = round(out["hit_tp"] / out["hits"], 4) if out["hits"] else None
    out["hit_recall"] = round(out["hit_tp"] / out["strokes"], 4) if out["strokes"] else None
    return out


def _fmt_row(row: Dict[str, Any]) -> str:
    if row.get("our_t") is None:
        return "未判分"
    return f"{row.get('our_winner')} [{row.get('our_rule')}] @ {float(row['our_t']):.3f}s"


def build_report(base_dir: str, compare_dirs: List[str], manifest: str,
                 fps: float = 120.0) -> Dict[str, Any]:
    games = _manifest_games(manifest)
    roots = [base_dir, *compare_dirs]
    loaded = [_load_runs(root, games, fps) for root in roots]
    modes = [items[0]["mode"] if items else os.path.basename(os.path.normpath(root))
             for root, items in zip(roots, loaded)]
    reviewers = [r for r in REVIEWERS if any(run["reviewer"] == r for items in loaded for run in items)]
    totals = []
    for mode, runs in zip(modes, loaded):
        for reviewer in reviewers:
            if not any(r["reviewer"] == reviewer for r in runs):
                continue
            for scope in SCOPES:
                totals.append(_aggregate(runs, mode, reviewer, scope))

    base_rows = {(run["video"], run["reviewer"], row.get("gt_t"), row.get("gt"), row.get("gt_winner")): row
                 for run in loaded[0] for row in run["rows"]}
    changes = []
    for mode, runs in zip(modes[1:], loaded[1:]):
        for run in runs:
            for row in run["rows"]:
                key = (run["video"], run["reviewer"], row.get("gt_t"), row.get("gt"),
                       row.get("gt_winner"))
                before = base_rows.get(key)
                if before is None:
                    continue
                old_ok, new_ok = bool(before.get("correct")), bool(row.get("correct"))
                change = ("right_to_wrong" if old_ok and not new_ok else
                          "wrong_to_right" if not old_ok and new_ok else None)
                if change:
                    changes.append({
                        "mode": mode, "video": run["video"], "reviewer": run["reviewer"],
                        "gt_t": row.get("gt_t"), "gt": row.get("gt"),
                        "gt_winner": row.get("gt_winner"), "before": _fmt_row(before),
                        "after": _fmt_row(row), "change": change,
                        "test4": run["video"] == "test_4",
                    })
    return {"modes": modes, "reviewers": reviewers, "totals": totals, "changes": changes}


def _scope_label(scope: str) -> str:
    return {"test_2": "test_2", "holdout_with_test_4": "留出含test_4",
            "holdout_without_test_4": "留出不含test_4"}[scope]


def _table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    def c(v):
        return "-" if v is None else str(v).replace("|", "\\|").replace("\n", " ")
    return ["| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|"] +                 [
                "| " + " | ".join(c(v) for v in row) + " |" for row in rows]


def write_report(base_dir: str, compare_dirs: List[str], manifest: str, out_dir: str,
                 fps: float = 120.0) -> Dict[str, str]:
    report = build_report(base_dir, compare_dirs, manifest, fps)
    os.makedirs(out_dir, exist_ok=True)
    md = os.path.join(out_dir, "compare.md")
    csv_path = os.path.join(out_dir, "compare.csv")
    lines = ["# 球拍证据模式对比", "",
             "`test_4` 是球拍模型训练集；留出集同时给出含/不含 test_4。", ""]
    rows = []
    for t in report["totals"]:
        rows.append([t["mode"], t["reviewer"], _scope_label(t["scope"]), t["gt"], t["correct"],
                     t["missed_out"], t["missed_net"], t["missed_other"], t["wrong"],
                     t["spurious"], t["dead"],
                     "-" if t["hit_precision"] is None else f"{t['hit_precision']:.3f}",
                     "-" if t["hit_recall"] is None else f"{t['hit_recall']:.3f}"])
    lines += ["## 合计表", ""]
    lines += _table(["模式", "复核", "范围", "真值分", "判对", "未判出界", "未判下网",
                     "未判其他", "判错", "多判", "死球", "击球精确率", "击球召回"], rows)
    lines += ["", "## 逐分改判", ""]
    for change, title in (("right_to_wrong", "原来对→现在错"),
                          ("wrong_to_right", "原来错→现在对")):
        lines += [f"### {title}", ""]
        items = [c for c in report["changes"] if c["change"] == change]
        table = [[c["mode"], c["video"] + ("（球拍训练集）" if c["test4"] else ""),
                  c["reviewer"], f"{float(c['gt_t']):.3f}s", c["gt"], c["gt_winner"],
                  c["before"], c["after"]] for c in items]
        lines += _table(["模式", "视频", "复核", "真值时间", "结束方式", "真值得分方",
                         "off 判罚", "新模式判罚"], table) if table else ["无。"]
        lines += [""]
    with open(md, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    fields = ["mode", "reviewer", "scope", "gt", "correct", "missed_out", "missed_net",
              "missed_other", "wrong", "spurious", "dead", "hits", "hit_tp", "strokes",
              "hit_precision", "hit_recall"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in report["totals"]:
            writer.writerow({k: row.get(k) for k in fields})
    return {"markdown": md, "csv": csv_path}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="对比球拍证据 off/shadow/drop 批量输出")
    ap.add_argument("--base", required=True)
    ap.add_argument("--compare", nargs="+", required=True)
    ap.add_argument("--manifest", default="cascade/batch_manifest.json")
    ap.add_argument("--out-dir", default="outputs/cascade_batch_racket")
    ap.add_argument("--fps", type=float, default=120.0)
    a = ap.parse_args(argv)
    paths = write_report(a.base, a.compare, a.manifest, a.out_dir, a.fps)
    print(f"对比报告：{paths['markdown']}")
    print(f"对比 CSV：{paths['csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())