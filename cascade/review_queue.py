"""待人工确认清单：从一次级联运行结果提取需要人看的片段。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

MANUAL_STATUSES = frozenset(("uncertain", "error", "timeout"))
# 复核器本身没有能力判断的片段（如规则复核遇到只有丢球触发、没有局部极大候选的片段）。
# 它们进清单但标 category="unreviewable"，不计入人工确认比例——否则规则组的比例会被
# 这类片段撑大，和 TTNet 组没有可比性。
UNREVIEWABLE_REASONS = frozenset(("no_reviewable_candidate",))


def _reason_codes(reason: str) -> List[str]:
    return [part.strip() for part in str(reason or "").split(";") if part.strip()]


def build_review_queue(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按 clip 汇总 uncertain / error / timeout 片段。

    rejected 和 confirmed 不进清单。每条带 ``category``：``manual``（需要人看）或
    ``unreviewable``（复核器无法判断，原因码见 UNREVIEWABLE_REASONS）。TTNet 的驼峰从 reviewer.log 按 clip_id 关联；
    判罚取虚拟时间在 clip 前后 1 秒内的记录。
    """
    clips = {c.clip_id: c for c in res.get("clips", [])}
    reviewer = res.get("reviewer")
    ttnet_log = {row.get("clip_id"): row for row in getattr(reviewer, "log", [])}
    decisions = list(getattr(res.get("referee"), "decision_records", []))

    queue: List[Dict[str, Any]] = []
    for result in res.get("results", []):
        if result.status not in MANUAL_STATUSES:
            continue
        clip = clips.get(result.clip_id)
        start_s = clip.start_timestamp_s if clip is not None else None
        end_s = clip.end_timestamp_s if clip is not None else None
        near = []
        if start_s is not None and end_s is not None:
            near = [dict(d) for d in decisions
                    if start_s - 1.0 <= float(d.get("virtual_t", 0.0)) <= end_s + 1.0]

        tt = ttnet_log.get(result.clip_id, {})
        humps = list(tt.get("humps") or [])
        peak: Optional[Dict[str, Any]] = None
        if humps:
            best = max(humps, key=lambda h: float(h.get("peak", 0.0)))
            peak = {"frame": best.get("peak_frame"), "score": round(float(best.get("peak", 0.0)), 6)}

        codes = _reason_codes(result.reason)
        category = ("unreviewable" if result.status == "uncertain"
                    and codes and all(c in UNREVIEWABLE_REASONS for c in codes) else "manual")
        queue.append({
            "clip_id": result.clip_id,
            "category": category,
            "proposal_ids": list(result.proposal_ids),
            "status": result.status,
            "reason": result.reason,
            "reason_codes": _reason_codes(result.reason),
            "start_timestamp_s": round(start_s, 6) if start_s is not None else None,
            "end_timestamp_s": round(end_s, 6) if end_s is not None else None,
            "start_frame_id": clip.start_frame_id if clip is not None else None,
            "end_frame_id": clip.end_frame_id if clip is not None else None,
            "missing_context": bool(clip.missing_context) if clip is not None else False,
            "hump_peak": peak,
            "humps": humps,
            "nearby_decisions": near,
            "reviewer": result.reviewer,
        })
    queue.sort(key=lambda row: (row["start_frame_id"] is None, row["start_frame_id"] or 0,
                                row["clip_id"]))
    return queue
