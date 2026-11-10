"""对比多个计分状态机的批量输出。

只读 ``<video>_<reviewer>`` 目录中已有的 ``eval_extended.json``、``summary.json``、
``events.jsonl``、``decisions.jsonl`` 与 ``fused_obs.jsonl``，不重新运行级联。
门因归因会用这些已保存记录回放同一个状态机。用法（在 TN 根目录）::

    python -m cascade.compare_sm \
        --base outputs/cascade_batch_sm/tn \
        --compare outputs/cascade_batch_sm/gd outputs/cascade_batch_sm/gd-traj outputs/cascade_batch_sm/gd-nr \
        --out outputs/cascade_batch_sm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cascade.committer import unique_state_machine_event_id  # noqa: E402
from cascade.state_machines import make_state_machine  # noqa: E402
from core.schemas import RefereeEvent  # noqa: E402

RUN_RE = re.compile(r"^(?P<video>.+)_(?P<reviewer>rule|ttnet)$")
REVIEWERS = ("rule", "ttnet")

GATE_CAUSE_LABELS = {
    "EVENT_ID_DUPLICATE": "事件 ID 撞号去重",
    "NO_RETURN_DISABLED": "NO_RETURN 默认关闭",
    "SERVE_ADMISSION_INCOMPLETE": "发球准入未完成",
    "SERVE_PHASE_IGNORED": "发球阶段判罚被忽略",
    "SPLICE_GUARD": "同侧两弹夹击球",
    "NO_HIT_REQUIRED": "回合内须有击球",
    "COOLDOWN_IGNORED": "判分后冷却",
    "WAITING_SERVE": "未进入可判分回合",
    "OTHER": "其他/未达判分条件",
    "NO_TRACE": "没有可回放快照",
}


class _GateLog(dict):
    def __init__(self, initial: Dict[str, int]) -> None:
        super().__init__(initial)
        self.changes: List[Tuple[str, int]] = []

    def __setitem__(self, key: str, value: int) -> None:
        old = super().get(key, 0)
        if value != old:
            self.changes.append((key, value - old))
        super().__setitem__(key, value)


GATE_ORDER = (
    "serve_prep_resets",
    "serve_admission_violations",
    "serve_no_hit_blocked",
    "serve_phase_ignored",
    "splice_skips",
    "no_hit_skips",
    "cooldown_ignored_events",
    "duplicate_event_ids_ignored",
)


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _state_name(root: str, summary: Dict[str, Any]) -> str:
    name = summary.get("state_machine")
    return str(name) if name else os.path.basename(os.path.normpath(root))


def _discover_pairs(base_dir: str, compare_dirs: List[str]) -> List[Tuple[str, str]]:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"基准输出目录不存在: {base_dir}")
    pairs = set()
    for root in [base_dir, *compare_dirs]:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"对比输出目录不存在: {root}")
        for name in os.listdir(root):
            m = RUN_RE.match(name)
            if not m:
                continue
            path = os.path.join(root, name, "eval_extended.json")
            if os.path.isfile(path):
                pairs.add((m.group("video"), m.group("reviewer")))
    if not pairs:
        raise ValueError("没有找到 <video>_<reviewer>/eval_extended.json")
    return sorted(pairs, key=lambda p: (p[0], REVIEWERS.index(p[1])))


def _load_runs(root: str, pairs: Iterable[Tuple[str, str]]) -> Dict[str, Any]:
    name = os.path.basename(os.path.normpath(root))
    summary0 = _read_json(os.path.join(root, f"{next(iter(pairs))[0]}_{next(iter(pairs))[1]}", "summary.json")) if pairs else {}
    state = _state_name(root, summary0)
    runs: List[Dict[str, Any]] = []
    for video, reviewer in pairs:
        run_dir = os.path.join(root, f"{video}_{reviewer}")
        eval_path = os.path.join(run_dir, "eval_extended.json")
        if not os.path.isfile(eval_path):
            continue
        summary = _read_json(os.path.join(run_dir, "summary.json"))
        if summary.get("state_machine"):
            state = str(summary["state_machine"])
        points = _read_json(eval_path).get("points", {})
        runs.append({
            "state_machine": state,
            "video": video,
            "reviewer": reviewer,
            "root": root,
            "run_dir": run_dir,
            "rows": list(points.get("rows", [])),
            "spurious": int(summary["spurious_points"]) if isinstance(summary.get("spurious_points"), int)
                        else len(points.get("spurious_points", [])),
            "dead": int(summary["dead_ball"]) if isinstance(summary.get("dead_ball"), int)
                    else len(points.get("dead_ball", [])),
            "summary": summary,
            "events": _read_jsonl(os.path.join(run_dir, "events.jsonl")),
            "decisions": _read_jsonl(os.path.join(run_dir, "decisions.jsonl")),
            "fused_obs": _read_jsonl(os.path.join(run_dir, "fused_obs.jsonl")),
        })
    return {"root": root, "state_machine": state, "runs": runs}


def _is_out(gt: str) -> bool:
    return str(gt).endswith("_out")


def _is_net(gt: str) -> bool:
    return str(gt).endswith("_net")


def _blank(v: Any) -> str:
    return "-" if v is None else str(v)


def _fmt_row(row: Dict[str, Any]) -> str:
    if row.get("our_t") is None:
        return "未判分"
    return (f"{_blank(row.get('our_winner'))} [{_blank(row.get('our_rule'))}] "
            f"@ {float(row['our_t']):.3f}s")


def _aggregate(runs: List[Dict[str, Any]], state: str, reviewer: str, scope: str) -> Dict[str, Any]:
    selected = [r for r in runs if r["reviewer"] == reviewer]
    if scope == "tuning":
        selected = [r for r in selected if r["video"] == "test_2"]
    else:
        selected = [r for r in selected if r["video"] != "test_2"]
    rows = [row for run in selected for row in run["rows"]]
    out = {
        "state_machine": state, "reviewer": reviewer, "scope": scope,
        "gt": len(rows), "correct": 0, "missed_out": 0, "missed_net": 0,
        "missed_other": 0, "wrong": 0, "wrong_a": 0, "wrong_b": 0,
        "spurious": sum(r["spurious"] for r in selected),
        "dead": sum(r["dead"] for r in selected),
    }
    for row in rows:
        if row.get("our_t") is None:
            if _is_out(row.get("gt", "")):
                out["missed_out"] += 1
            elif _is_net(row.get("gt", "")):
                out["missed_net"] += 1
            else:
                out["missed_other"] += 1
            continue
        if row.get("correct"):
            out["correct"] += 1
        else:
            out["wrong"] += 1
            side = row.get("our_winner")
            if side == "A":
                out["wrong_a"] += 1
            elif side == "B":
                out["wrong_b"] += 1
    return out



def _decision_kind(decision: Any) -> str:
    kind = getattr(decision, "kind", "")
    return kind.value if hasattr(kind, "value") else str(kind)


def _ctx_snapshot(sm: Any) -> Dict[str, Any]:
    ctx = getattr(sm, "_ctx", None)
    return {
        "last_hitter": getattr(ctx, "last_hitter", None),
        "last_bounce_side": getattr(ctx, "last_bounce_side", None),
        "rally_has_hit": getattr(ctx, "rally_has_hit", False),
        "hit_since_last_bounce": getattr(ctx, "hit_since_last_bounce", False),
        "serve_own_bounce_seen": getattr(ctx, "serve_own_bounce_seen", False),
        "cooldown_until": getattr(sm, "_cooldown_until", None),
    }


def _replay_run(run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """用已保存的 events + fused_obs 回放同一个状态机，仅用于门因归因。"""
    frames = run.get("fused_obs") or []
    if not frames:
        return None
    sm = make_state_machine(run["state_machine"])
    gate_log = _GateLog(getattr(sm, "_gate_stats", {}))
    if hasattr(sm, "_gate_stats"):
        sm._gate_stats = gate_log
    events = run.get("events") or []
    event_i = 0
    id_counts: Dict[str, int] = {}
    id_used: set[str] = set()
    decisions: List[Dict[str, Any]] = []
    event_records: Dict[Tuple[str, float], Dict[str, Any]] = {}
    ticks: List[Dict[str, Any]] = []
    gate_trace: List[Dict[str, Any]] = []
    for frame in frames:
        t = float(frame["timestamp_s"])
        tick_before_state = sm.state.value
        tick_before_ctx = _ctx_snapshot(sm)
        gate_log.changes = []
        tick_decisions = sm.tick(t)
        if gate_log.changes:
            gate_trace.append({"t": t, "trigger": "tick", "state": sm.state.value,
                               "gates": list(gate_log.changes)})
        tick_rec = {
            "current_t": t, "trigger": "tick", "event_id": None, "event_t": None,
            "gate_delta": list(gate_log.changes), "before_state": tick_before_state,
            "before_ctx": tick_before_ctx, "after_state": sm.state.value,
        }
        ticks.append(tick_rec)
        for d in tick_decisions:
            decisions.append({
                "kind": _decision_kind(d), "rule": getattr(d, "rule", None),
                "winner": getattr(d, "winner", None), "current_t": t,
                "trigger": "tick", "event_id": None, "event_t": None,
                "gate_delta": list(gate_log.changes),
                "before_state": tick_before_state, "before_ctx": tick_before_ctx,
            })
        while event_i < len(events) and float(events[event_i].get("timestamp_s", 1e9)) <= t + 1e-9:
            e = events[event_i]
            event_i += 1
            event_t = float(e["timestamp_s"])
            before_state = sm.state.value
            before_ctx = _ctx_snapshot(sm)
            sm_id, _ = unique_state_machine_event_id(e["event_id"], id_counts, id_used)
            event = RefereeEvent(
                event_id=sm_id, timestamp_s=event_t, type=e["type"], side=e.get("side", "UNKNOWN"),
                position_px=tuple(e["position_px"]) if e.get("position_px") else None,
                confidence=e.get("confidence"))
            gate_log.changes = []
            produced = sm.update(event)
            if gate_log.changes:
                gate_trace.append({"t": event_t, "trigger": "event", "state": before_state,
                                   "event_id": e["event_id"], "gates": list(gate_log.changes)})
            rec = {
                "event_id": e["event_id"], "event_t": event_t, "event_type": e["type"],
                "event_side": e.get("side"), "before_state": before_state,
                "before_ctx": before_ctx, "gate_delta": list(gate_log.changes),
                "decisions": [(_decision_kind(d), getattr(d, "rule", None),
                               getattr(d, "winner", None)) for d in produced],
            }
            event_records[(e["event_id"], round(event_t, 6))] = rec
            for d in produced:
                decisions.append({
                    "kind": _decision_kind(d), "rule": getattr(d, "rule", None),
                    "winner": getattr(d, "winner", None), "current_t": t,
                    "trigger": "event", "event_id": e["event_id"], "event_t": event_t,
                    "gate_delta": list(gate_log.changes),
                    "before_state": before_state, "before_ctx": before_ctx,
                })
    return {"decisions": decisions, "events": event_records, "ticks": ticks, "gate_trace": gate_trace}


def _find_tn_point(replay: Optional[Dict[str, Any]], change: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if replay is None or change.get("tn_t") is None:
        return None
    candidates = [d for d in replay["decisions"]
                  if d.get("kind") == "POINT" and d.get("rule") == change.get("tn_rule")
                  and d.get("winner") == change.get("tn_winner")
                  and abs(float(d["current_t"]) - float(change["tn_t"])) <= 0.2]
    return min(candidates, key=lambda d: abs(float(d["current_t"]) - float(change["tn_t"]))) if candidates else None


def _classify_gate_cause(tn_decision: Optional[Dict[str, Any]],
                         replay_event: Optional[Dict[str, Any]],
                         no_return_enabled: bool = False) -> Tuple[str, str]:
    if tn_decision is None:
        return "NO_TRACE", GATE_CAUSE_LABELS["NO_TRACE"]
    if tn_decision.get("rule") == "NO_RETURN" and not no_return_enabled:
        return "NO_RETURN_DISABLED", GATE_CAUSE_LABELS["NO_RETURN_DISABLED"]
    if replay_event is None:
        return "OTHER", "TN 判罚事件在对比组回放中无对应记录"

    delta = list(replay_event.get("gate_delta", [])) + list(replay_event.get("recent_gate_delta", []))
    gates = {name for name, _ in delta}
    priority = (
        ("duplicate_event_ids_ignored", "EVENT_ID_DUPLICATE"),
        ("cooldown_ignored_events", "COOLDOWN_IGNORED"),
        ("serve_phase_ignored", "SERVE_PHASE_IGNORED"),
        ("splice_skips", "SPLICE_GUARD"),
        ("no_hit_skips", "NO_HIT_REQUIRED"),
        ("serve_admission_violations", "SERVE_ADMISSION_INCOMPLETE"),
        ("serve_no_hit_blocked", "SERVE_ADMISSION_INCOMPLETE"),
    )
    for gate, code in priority:
        if gate in gates:
            return code, GATE_CAUSE_LABELS[code]

    state = replay_event.get("before_state")
    ctx = replay_event.get("before_ctx") or {}
    if state == "SERVE_IN_PROGRESS":
        return "SERVE_ADMISSION_INCOMPLETE", GATE_CAUSE_LABELS["SERVE_ADMISSION_INCOMPLETE"]
    if state == "WAITING_SERVE":
        return "WAITING_SERVE", GATE_CAUSE_LABELS["WAITING_SERVE"]
    if state in ("RALLY", "RALLY_END_CANDIDATE") and not ctx.get("rally_has_hit"):
        return "NO_HIT_REQUIRED", GATE_CAUSE_LABELS["NO_HIT_REQUIRED"]
    if state == "RALLY" and ctx.get("last_hitter") is None:
        return "OTHER", "无击球方归属"
    return "OTHER", GATE_CAUSE_LABELS["OTHER"]


def _build_gate_attributions(changes: List[Dict[str, Any]],
                             roots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    runs = {(run["state_machine"], run["video"], run["reviewer"]): run
            for root in roots for run in root["runs"]}
    replays: Dict[Tuple[str, str, str], Optional[Dict[str, Any]]] = {}
    out: List[Dict[str, Any]] = []
    for change in changes:
        if change["change"] != "right_to_wrong":
            continue
        base_key = ("tn", change["video"], change["reviewer"])
        new_key = (change["state_machine"], change["video"], change["reviewer"])
        if base_key not in runs or new_key not in runs:
            continue
        if base_key not in replays:
            replays[base_key] = _replay_run(runs[base_key])
        if new_key not in replays:
            replays[new_key] = _replay_run(runs[new_key])
        tn_point = _find_tn_point(replays[base_key], change)
        replay_event = None
        if tn_point and tn_point.get("event_id") is not None and replays[new_key]:
            replay_event = replays[new_key]["events"].get(
                (tn_point["event_id"], round(float(tn_point["event_t"]), 6)))
        elif tn_point and replays[new_key]:
            ticks = replays[new_key].get("ticks") or []
            if ticks:
                tick = min(ticks, key=lambda x: abs(float(x["current_t"]) - float(tn_point["current_t"])))
                if abs(float(tick["current_t"]) - float(tn_point["current_t"])) <= 0.05:
                    replay_event = dict(tick)
                    recent = [g for g in replays[new_key].get("gate_trace", [])
                              if float(tn_point["current_t"]) - 8.0 <= float(g["t"])
                              <= float(tn_point["current_t"]) + 0.05]
                    if recent:
                        replay_event["recent_gate_delta"] = list(recent[-1]["gates"])
        code, label = _classify_gate_cause(
            tn_point, replay_event, no_return_enabled=change["state_machine"] == "gd-nr")
        out.append({**change, "cause_code": code, "cause": label})
    return out


def _diagnose_run(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = sorted(run["rows"], key=lambda r: float(r.get("gt_t", 0.0)))
    events = run["events"]
    decisions = run["decisions"]
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if row.get("our_t") is not None or not _is_out(row.get("gt", "")):
            continue
        gt_t = float(row["gt_t"])
        start = float(rows[i - 1]["gt_t"]) if i else 0.0
        end = gt_t + 2.0
        out_events = [e for e in events
                      if e.get("type") == "OUT" and start <= float(e.get("timestamp_s", -1)) <= end]
        near_decisions = [d for d in decisions
                          if start <= float(d.get("virtual_t", -1)) <= end]
        category = "OUT_EVENT_NOT_SCORED" if out_events else "NO_OUT_EVENT"
        out.append({
            "state_machine": run["state_machine"], "video": run["video"],
            "reviewer": run["reviewer"], "gt_t": round(gt_t, 3),
            "gt": row.get("gt"), "gt_winner": row.get("gt_winner"),
            "window_s": [round(start, 3), round(end, 3)],
            "out_events": out_events, "decisions": near_decisions,
            "category": category,
        })
    return out


def build_report(base_dir: str, compare_dirs: List[str]) -> Dict[str, Any]:
    pairs = _discover_pairs(base_dir, compare_dirs)
    roots = [_load_runs(base_dir, pairs)] + [_load_runs(d, pairs) for d in compare_dirs]
    base = roots[0]
    states = [r["state_machine"] for r in roots]
    reviewers = [rev for rev in REVIEWERS
                 if any(run["reviewer"] == rev for root in roots for run in root["runs"])]

    totals: List[Dict[str, Any]] = []
    for root in roots:
        for reviewer in reviewers:
            if not any(run["reviewer"] == reviewer for run in root["runs"]):
                continue
            for scope in ("tuning", "holdout"):
                totals.append(_aggregate(root["runs"], root["state_machine"], reviewer, scope))

    base_rows = {(run["video"], run["reviewer"], row.get("gt_t"), row.get("gt"), row.get("gt_winner")): row
                 for run in base["runs"] for row in run["rows"]}
    changes: List[Dict[str, Any]] = []
    for root in roots[1:]:
        for run in root["runs"]:
            for row in run["rows"]:
                key = (run["video"], run["reviewer"], row.get("gt_t"), row.get("gt"), row.get("gt_winner"))
                before = base_rows.get(key)
                if before is None:
                    continue
                old_ok, new_ok = bool(before.get("correct")), bool(row.get("correct"))
                change = ("right_to_wrong" if old_ok and not new_ok else
                          "wrong_to_right" if not old_ok and new_ok else None)
                if change:
                    changes.append({
                        "state_machine": root["state_machine"], "video": run["video"],
                        "reviewer": run["reviewer"], "gt_t": row.get("gt_t"),
                        "gt": row.get("gt"), "gt_winner": row.get("gt_winner"),
                        "tn": _fmt_row(before), "new": _fmt_row(row), "change": change,
                        "tn_t": before.get("our_t"), "tn_rule": before.get("our_rule"),
                        "tn_winner": before.get("our_winner"),
                    })

    gates: List[Dict[str, Any]] = []
    for root in roots[1:]:
        for run in root["runs"]:
            stats = run["summary"].get("sm_gates")
            if isinstance(stats, dict):
                gates.append({"state_machine": root["state_machine"], "video": run["video"],
                              "reviewer": run["reviewer"], "gates": dict(stats)})

    gate_attributions = _build_gate_attributions(changes, roots)
    diagnostics = [d for root in roots for run in root["runs"] for d in _diagnose_run(run)]
    diag_by: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for d in diagnostics:
        diag_by.setdefault((d["state_machine"], d["reviewer"], d["video"]), []).append(d)
    for total in totals:
        selected_diag = [d for d in diagnostics
                         if d["state_machine"] == total["state_machine"]
                         and d["reviewer"] == total["reviewer"]
                         and ((total["scope"] == "tuning") == (d["video"] == "test_2"))]
        total["no_out_event"] = sum(d["category"] == "NO_OUT_EVENT" for d in selected_diag)
        total["out_event_not_scored"] = sum(d["category"] == "OUT_EVENT_NOT_SCORED"
                                            for d in selected_diag)
    return {"base": base, "roots": roots, "states": states, "reviewers": reviewers,
            "totals": totals, "changes": changes, "gate_attributions": gate_attributions,
            "gates": gates, "out_diagnostics": diagnostics}


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(_md_cell(v) for v in row) + " |" for row in rows]
    return out


def _scope_label(scope: str) -> str:
    return "test_2" if scope == "tuning" else "留出集"


def _gate_text(gates: Dict[str, Any]) -> str:
    keys = list(GATE_ORDER) + sorted(k for k in gates if k not in GATE_ORDER)
    return "；".join(f"{k}={int(gates.get(k, 0) or 0)}" for k in keys)


def _decision_text(decisions: List[Dict[str, Any]]) -> str:
    if not decisions:
        return "无"
    return "；".join(
        f"{float(d.get('virtual_t', 0)):.3f}s {_blank(d.get('kind'))}/{_blank(d.get('rule'))}"
        + (f"/{d['winner']}" if d.get("winner") else "")
        for d in decisions)


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [f"# 状态机对比（基准：{report['base']['state_machine']}）", ""]
    lines += ["## 结论摘要", ""]
    holdout = {(t["state_machine"], t["reviewer"]): t for t in report["totals"]
               if t["scope"] == "holdout"}
    for state in report["states"]:
        for reviewer in report["reviewers"]:
            cur = holdout.get((state, reviewer))
            if cur is None:
                continue
            base = holdout.get((report["base"]["state_machine"], reviewer))
            delta = lambda key: (cur[key] - base[key]) if base else cur[key]  # noqa: E731
            wrong_changes = sum(c["state_machine"] == state and c["reviewer"] == reviewer
                                and c["change"] == "right_to_wrong"
                                and c["video"] != "test_2" for c in report["changes"])
            lines.append(
                f"- `{state}` / {reviewer} 留出集：判对 {cur['correct']}/{cur['gt']}，"
                f"出界漏判 {cur['missed_out']}（Δ{delta('missed_out'):+d}），"
                f"下网漏判 {cur['missed_net']}（Δ{delta('missed_net'):+d}），"
                f"判错 {cur['wrong']}（Δ{delta('wrong'):+d}），"
                f"原对→现错 {wrong_changes}，多判 {cur['spurious']}（Δ{delta('spurious'):+d}），"
                f"死球 {cur['dead']}（Δ{delta('dead'):+d}）。"
            )
    lines += [""]

    lines += ["## a. 合计表", "",
              "test_2 是调参视频，单列；留出集为 test_1、test_3～7 合计。", ""]
    rows = []
    for t in report["totals"]:
        rows.append([
            t["state_machine"], t["reviewer"], _scope_label(t["scope"]), t["gt"], t["correct"],
            t["missed_out"], t["missed_net"], t["missed_other"],
            f"{t['wrong']} (A{t['wrong_a']}/B{t['wrong_b']})", t["spurious"], t["dead"],
        ])
    lines += _table(["状态机", "复核", "范围", "真值分", "得分方判对",
                     "未判-出界", "未判-下网", "未判-其他", "判错方(A/B)",
                     "多判", "死球"], rows)
    lines += [""]

    lines += ["## b. 逐分改判表", "",
              "只列相对基准状态机发生对错变化的真值分。", ""]
    for change, title in (("right_to_wrong", "原来对→现在错"),
                          ("wrong_to_right", "原来错→现在对")):
        lines += [f"### {title}", ""]
        selected = [c for c in report["changes"] if c["change"] == change]
        rows = [[c["state_machine"], c["video"], c["reviewer"], f"{float(c['gt_t']):.3f}s",
                 c["gt"], c["gt_winner"], c["tn"], c["new"]] for c in selected]
        lines += _table(["状态机", "视频", "复核", "真值时间", "结束方式", "真值得分方",
                         "tn 判罚", "新状态机判罚"], rows) if rows else ["无。"]
        lines += [""]

    lines += ["## b2. 门因归因表", "",
              "只统计相对 tn 的“原来对→现在错”。依据已保存的 events.jsonl、fused_obs.jsonl 和 decisions.jsonl，"
              "用同一个状态机回放定位门因；不重新运行级联。", ""]
    attr_counts = Counter(
        (a["state_machine"], a["reviewer"], _scope_label("tuning" if a["video"] == "test_2" else "holdout"),
         a["cause"]) for a in report.get("gate_attributions", []))
    attr_rows = []
    for state in report["states"]:
        for reviewer in report["reviewers"]:
            for scope in ("test_2", "留出集"):
                for cause, count in sorted(
                        ((cause, n) for (s, r, sc, cause), n in attr_counts.items()
                         if s == state and r == reviewer and sc == scope),
                        key=lambda item: item[0]):
                    attr_rows.append([state, reviewer, scope, cause, count])
    lines += _table(["状态机", "复核", "范围", "门因", "条数"], attr_rows) if attr_rows else ["无改判分。"]
    lines += [""]
    detail_rows = [[a["state_machine"], a["video"], a["reviewer"], f"{float(a['gt_t']):.3f}s",
                    a["gt"], a["tn"], a["new"], a["cause"]]
                   for a in report.get("gate_attributions", [])]
    lines += _table(["状态机", "视频", "复核", "真值时间", "结束方式", "tn 判罚",
                     "新状态机判罚", "门因"], detail_rows) if detail_rows else ["无改判分。"]
    lines += [""]

    lines += ["## c. 门控计数", "",
              "gd、gd-traj、gd-nr 每道门的触发次数；0 也列出。", ""]
    rows = [[g["state_machine"], g["video"], g["reviewer"], _gate_text(g["gates"])]
            for g in report["gates"]]
    lines += _table(["状态机", "视频", "复核", "门控计数"], rows) if rows else ["无门控计数。"]
    lines += [""]

    lines += ["## d. 出界诊断", "",
              "窗口 = 上一个真值分结束到本分 gt_t + 2 s；只看该窗口内的 OUT 事件与判罚。", ""]
    summary_rows = []
    for state in report["states"]:
        for reviewer in report["reviewers"]:
            for scope in ("tuning", "holdout"):
                t = next((x for x in report["totals"]
                          if x["state_machine"] == state and x["reviewer"] == reviewer
                          and x["scope"] == scope), None)
                if t:
                    summary_rows.append([state, reviewer, _scope_label(scope), t["missed_out"],
                                         t["no_out_event"], t["out_event_not_scored"]])
    lines += _table(["状态机", "复核", "范围", "未判出界", "没有 OUT 事件",
                     "有 OUT 事件但未判/死球"], summary_rows)
    lines += [""]
    diag_rows = []
    for d in report["out_diagnostics"]:
        cat = "没有 OUT 事件" if d["category"] == "NO_OUT_EVENT" else "有 OUT 事件但未判/死球"
        ev_text = "；".join(f"{float(e['timestamp_s']):.3f}s/{_blank(e.get('side'))}"
                            for e in d["out_events"]) or "无"
        diag_rows.append([d["state_machine"], d["video"], d["reviewer"],
                          f"{d['gt_t']:.3f}s", d["gt"], d["gt_winner"], cat,
                          ev_text, _decision_text(d["decisions"])])
    lines += _table(["状态机", "视频", "复核", "真值时间", "结束方式", "真值得分方",
                     "分类", "窗口内 OUT 事件", "窗口内判罚"], diag_rows) if diag_rows else ["无。"]
    lines += [""]
    return "\n".join(lines)


def write_report(base_dir: str, compare_dirs: List[str], out_dir: str,
                 report: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    report = report or build_report(base_dir, compare_dirs)
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "compare.md")
    csv_path = os.path.join(out_dir, "compare.csv")
    with open(md_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(_render_markdown(report))

    fields = ["state_machine", "reviewer", "scope", "gt", "correct", "missed_out",
              "missed_net", "missed_other", "wrong", "wrong_a", "wrong_b",
              "spurious", "dead", "no_out_event", "out_event_not_scored"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in report["totals"]:
            writer.writerow({k: row.get(k) for k in fields})
    return {"markdown": md_path, "csv": csv_path}


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="对比多个状态机的批量级联输出（只读已有结果）")
    ap.add_argument("--base", required=True, help="基准目录，例如 outputs/cascade_batch_sm/tn")
    ap.add_argument("--compare", nargs="+", required=True,
                    help="一个或多个对比目录，例如 .../gd .../gd-traj")
    ap.add_argument("--out", default=None, help="输出目录，默认是基准目录的上一级")
    return ap


def main(argv=None) -> int:
    a = _build_parser().parse_args(argv)
    out_dir = a.out or os.path.dirname(os.path.abspath(a.base))
    paths = write_report(a.base, a.compare, out_dir)
    print(f"对比报告：{paths['markdown']}")
    print(f"对比 CSV：{paths['csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())