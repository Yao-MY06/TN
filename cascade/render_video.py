"""把一次级联运行画成视频（事后渲染，不参与判罚，不影响实时计时）。

    python -m cascade.render_video --run-dir outputs/realtime/a_all --video <test_2.mp4> \
        --preset test2 --out outputs/realtime/a_all/cascade_hud.mp4

画面内容：
- 台面四角与球网线；
- 球位与尾迹，按来源着色（TrackNet 橙、YOLO 青）；
- 当前所处的提议区间与 TTNet 复核结果（confirmed / rejected / uncertain / error / timeout）；
- 确认的事件（落台、击球、出界等）与最近的落点；
- 左上角比分与最近判罚；底部全片时间轴（区间按复核状态着色，竖线为当前时刻，三角为判罚）。

球位来源（按优先级）：
1. 运行目录里的 ``fused_obs.jsonl``（新版 run_cascade / run_realtime 会写）——就是本次运行实际用的球位；
2. ``--tracknet-obs`` [+ ``--yolo-obs``]：离线重新融合，画面上会标注 "ball: offline obs"，
   实时节奏下的丢帧不会反映在球位上；
3. 都没有：不画球，只画区间、事件和判罚。

区间的复核状态是事后才知道的结果，画面在区间开始时就按最终状态着色，便于看。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from bisect import bisect_right
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

# BGR
C_TN = (40, 140, 255)       # TrackNet 橙
C_YOLO = (255, 220, 40)     # YOLO 青
C_OTHER = (200, 200, 200)
C_TABLE = (255, 255, 255)
C_NET = (180, 180, 180)
C_TEXT = (245, 245, 245)
C_DIM = (160, 160, 160)
C_BG = (25, 25, 25)
STATUS_COLOR = {
    "confirmed": (80, 220, 80),
    "rejected": (130, 130, 130),
    "uncertain": (40, 210, 255),
    "error": (60, 60, 230),
    "timeout": (200, 80, 230),
    "pending": (100, 100, 100),
}
EVENT_COLOR = {
    "BOUNCE": (80, 220, 80),
    "RACKET_HIT": (240, 240, 240),
    "OUT": (60, 60, 230),
    "NET_TOUCH": (200, 80, 230),
}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def source_color(src: Optional[str]) -> Tuple[int, int, int]:
    s = (src or "").lower()
    if "yolo" in s:
        return C_YOLO
    if "tracknet" in s:
        return C_TN
    return C_OTHER


def load_ball_track(run_dir: str, tracknet_obs: Optional[str], yolo_obs: Optional[str],
                    primary: str = "tracknet") -> Tuple[Dict[int, Dict[str, Any]], str]:
    """返回 (frame_id -> {t, xy|None, source}, 球位来源说明)。"""
    rows = read_jsonl(os.path.join(run_dir, "fused_obs.jsonl"))
    label = "ball: this run (fused)"
    if not rows and tracknet_obs:
        from cascade.fuse_obs import FuseParams, fuse
        tn = read_jsonl(tracknet_obs)
        if yolo_obs:
            fused, _ = fuse(read_jsonl(yolo_obs), tn, FuseParams(primary=primary))
        else:
            fused = tn
        rows = []
        for d in fused:
            b = d["ball"]
            rows.append({"frame_id": d["frame_id"], "timestamp_s": d["timestamp_s"],
                         "source": d.get("source"), "visible": b.get("visible"),
                         "center_px": b.get("center_px"), "predicted": b.get("predicted", False)})
        label = "ball: offline obs (not this run)"
    if not rows:
        return {}, "ball: not available"
    track = {}
    for r in rows:
        xy = None
        if r.get("visible") and not r.get("predicted") and r.get("center_px") is not None:
            xy = (float(r["center_px"][0]), float(r["center_px"][1]))
        track[int(r["frame_id"])] = {"t": float(r["timestamp_s"]), "xy": xy, "source": r.get("source")}
    return track, label


def load_clips(run_dir: str) -> List[Dict[str, Any]]:
    clips = read_jsonl(os.path.join(run_dir, "clips.jsonl"))
    status: Dict[str, Tuple[str, str]] = {}
    for r in read_jsonl(os.path.join(run_dir, "reviews.jsonl")):
        status[r["clip_id"]] = (r.get("status", "pending"), r.get("reason", ""))
    out = []
    for c in clips:
        st, why = status.get(c["clip_id"], ("pending", ""))
        out.append({"clip_id": c["clip_id"], "t0": float(c["start_timestamp_s"]),
                    "t1": float(c["end_timestamp_s"]), "status": st, "reason": why})
    out.sort(key=lambda c: c["t0"])
    return out


def load_decisions(run_dir: str) -> List[Dict[str, Any]]:
    out = []
    for d in read_jsonl(os.path.join(run_dir, "decisions.jsonl")):
        out.append({"t": float(d.get("virtual_t", 0.0)), "kind": d.get("kind", ""),
                    "winner": d.get("winner"), "rule": d.get("rule", "")})
    out.sort(key=lambda d: d["t"])
    return out


def decision_text(d: Dict[str, Any]) -> str:
    who = {"A": "P1(left)", "B": "P2(right)"}.get(d.get("winner") or "", "")
    kind = (d.get("kind") or "").replace("_", " ")
    return f"{kind} {who} [{d.get('rule', '')}]".replace("  ", " ")


def score_at(decisions: List[Dict[str, Any]], t: float) -> Tuple[int, int]:
    a = b = 0
    for d in decisions:
        if d["t"] > t:
            break
        if d["kind"] == "POINT":
            if d["winner"] == "A":
                a += 1
            elif d["winner"] == "B":
                b += 1
    return a, b


class Renderer:
    def __init__(self, width: int, height: int, duration_s: float, poly, net_x: Optional[float],
                 clips, events, decisions, ball_label: str, stats_line: str = "",
                 title: str = "", trail_s: float = 0.4, event_hold_s: float = 0.6,
                 decision_hold_s: float = 3.0, landing_history: int = 6) -> None:
        self.w, self.h = width, height
        self.s = width / 1920.0
        self.duration_s = max(duration_s, 1e-6)
        self.poly = poly
        self.net_x = net_x
        self.clips = clips
        self.clip_t0 = [c["t0"] for c in clips]
        self.events = sorted(events, key=lambda e: float(e["timestamp_s"]))
        self.decisions = decisions
        self.ball_label = ball_label
        self.stats_line = stats_line
        self.title = title
        self.trail_s = trail_s
        self.event_hold_s = event_hold_s
        self.decision_hold_s = decision_hold_s
        self.landing_history = landing_history
        self.trail: List[Tuple[float, Tuple[float, float], Tuple[int, int, int]]] = []

    def px(self, v: float) -> int:
        return max(1, int(round(v * self.s)))

    def put(self, img, text, org, scale, color, thick=1):
        import cv2
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale * self.s, (0, 0, 0),
                    self.px(thick + 2), cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale * self.s, color,
                    self.px(thick), cv2.LINE_AA)

    def active_clips(self, t: float) -> List[Dict[str, Any]]:
        i = bisect_right(self.clip_t0, t)
        return [c for c in self.clips[max(0, i - 8):i] if c["t0"] <= t <= c["t1"]]

    def render(self, frame, fid: int, t: float, ball: Optional[Dict[str, Any]]):
        import cv2
        import numpy as np
        img = frame
        # 台面与球网
        if self.poly is not None:
            pts = np.asarray(self.poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, C_TABLE, self.px(1), cv2.LINE_AA)
        if self.net_x is not None:
            x = int(round(self.net_x))
            cv2.line(img, (x, int(self.h * 0.25)), (x, int(self.h * 0.75)), C_NET, self.px(1), cv2.LINE_AA)

        # 尾迹与当前球位
        if ball is not None and ball.get("xy") is not None:
            self.trail.append((t, ball["xy"], source_color(ball.get("source"))))
        self.trail = [p for p in self.trail if t - p[0] <= self.trail_s]
        for (t0, a, _), (t1, b, col) in zip(self.trail, self.trail[1:]):
            if t1 - t0 > 0.05:
                continue
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, self.px(2), cv2.LINE_AA)
        if ball is not None and ball.get("xy") is not None:
            x, y = int(ball["xy"][0]), int(ball["xy"][1])
            col = source_color(ball.get("source"))
            cv2.circle(img, (x, y), self.px(14), col, self.px(2), cv2.LINE_AA)
            cv2.circle(img, (x, y), self.px(3), col, -1, cv2.LINE_AA)
            name = "YOLO" if col == C_YOLO else ("TrackNet" if col == C_TN else str(ball.get("source")))
            self.put(img, name, (x + self.px(18), y - self.px(14)), 0.5, col, 1)

        # 最近落点（只画已确认的 BOUNCE）
        landed = [e for e in self.events if e["type"] == "BOUNCE"
                  and float(e["timestamp_s"]) <= t and e.get("position_px")]
        for e in landed[-self.landing_history:]:
            x, y = int(e["position_px"][0]), int(e["position_px"][1])
            cv2.drawMarker(img, (x, y), EVENT_COLOR["BOUNCE"], cv2.MARKER_TILTED_CROSS,
                           self.px(14), self.px(2), cv2.LINE_AA)

        # 刚发生的事件
        ev_lines = []
        for e in self.events:
            et = float(e["timestamp_s"])
            if et > t:
                break
            if t - et > self.event_hold_s:
                continue
            col = EVENT_COLOR.get(e["type"], C_TEXT)
            if e.get("position_px"):
                x, y = int(e["position_px"][0]), int(e["position_px"][1])
                r = self.px(22 + 40 * (t - et) / self.event_hold_s)
                cv2.circle(img, (x, y), r, col, self.px(3), cv2.LINE_AA)
            src = "+".join(e.get("sources") or [])
            ev_lines.append((f"{e['type']} {e.get('side', '')}  ({src})", col))

        # 比分板
        x0, y0 = self.px(18), self.px(18)
        cv2.rectangle(img, (x0, y0), (x0 + self.px(560), y0 + self.px(150)), C_BG, -1)
        self.put(img, self.title or "CASCADE: YOLO -> TrackNet -> TTNet", (x0 + self.px(14), y0 + self.px(28)), 0.55, C_TEXT, 1)
        a, b = score_at(self.decisions, t)
        self.put(img, f"P1  {a} : {b}  P2", (x0 + self.px(14), y0 + self.px(80)), 1.3, C_TEXT, 3)
        last = [d for d in self.decisions if d["t"] <= t]
        if last and t - last[-1]["t"] <= self.decision_hold_s:
            self.put(img, decision_text(last[-1]), (x0 + self.px(14), y0 + self.px(120)), 0.6, (0, 235, 255), 2)
        self.put(img, f"frame {fid}  t={t:7.3f}s   {self.ball_label}",
                 (x0 + self.px(14), y0 + self.px(142)), 0.42, C_DIM, 1)

        # 当前区间与事件列表（右上）
        rx = self.w - self.px(560)
        ry = self.px(40)
        for c in self.active_clips(t):
            col = STATUS_COLOR.get(c["status"], C_TEXT)
            self.put(img, f"TTNet review {c['clip_id'].split(':')[-1]}: {c['status']}"
                          + (f" ({c['reason']})" if c["reason"] else ""), (rx, ry), 0.55, col, 2)
            ry += self.px(30)
        for text, col in ev_lines:
            self.put(img, text, (rx, ry), 0.6, col, 2)
            ry += self.px(30)

        # 图例与统计
        ly = self.h - self.px(70)
        self.put(img, "TrackNet", (self.px(18), ly), 0.5, C_TN, 1)
        self.put(img, "YOLO", (self.px(120), ly), 0.5, C_YOLO, 1)
        lx = self.px(190)
        for st in ("confirmed", "rejected", "uncertain", "error", "timeout"):
            self.put(img, st, (lx, ly), 0.45, STATUS_COLOR[st], 1)
            lx += self.px(110)
        if self.stats_line:
            self.put(img, self.stats_line, (lx + self.px(20), ly), 0.45, C_DIM, 1)

        # 时间轴
        bx0, bx1 = self.px(18), self.w - self.px(18)
        by0, by1 = self.h - self.px(42), self.h - self.px(22)
        cv2.rectangle(img, (bx0, by0), (bx1, by1), C_BG, -1)
        span = bx1 - bx0

        def tx(v: float) -> int:
            return bx0 + int(span * min(max(v / self.duration_s, 0.0), 1.0))

        for c in self.clips:
            if c["t0"] > self.duration_s:
                continue
            cv2.rectangle(img, (tx(c["t0"]), by0), (max(tx(c["t1"]), tx(c["t0"]) + 1), by1),
                          STATUS_COLOR.get(c["status"], C_TEXT), -1)
        for d in self.decisions:
            if d["t"] > self.duration_s:
                continue
            x = tx(d["t"])
            cv2.drawMarker(img, (x, by0 - self.px(6)), (0, 235, 255), cv2.MARKER_TRIANGLE_DOWN,
                           self.px(12), self.px(2))
        cx = tx(t)
        cv2.line(img, (cx, by0 - self.px(4)), (cx, by1 + self.px(4)), (255, 255, 255), self.px(2))
        return img


def stats_summary(run_dir: str) -> str:
    p = os.path.join(run_dir, "realtime_stats.json")
    if not os.path.isfile(p):
        return ""
    with open(p, encoding="utf-8") as f:
        s = json.load(f)
    parts = []
    pace = (s.get("camera") or {}).get("pace")
    if pace:
        parts.append(f"pace={pace}")
    tn = (s.get("stages") or {}).get("tracknet") or {}
    if tn.get("processed") is not None:
        tot = tn.get("processed", 0) + tn.get("dropped", 0)
        if tot:
            parts.append(f"TrackNet dropped {100.0 * tn.get('dropped', 0) / tot:.0f}%")
    lat = (s.get("end_to_end_latency_s") or {}).get("decision_p95")
    if lat is not None:
        parts.append(f"decision P95 {lat:.2f}s")
    return "  ".join(parts)


def geometry(preset: str, quad: Optional[str]):
    """返回 (台面四角, net_x)。取不到时返回 (None, None)，不影响其它内容。"""
    try:
        from cascade.run_realtime import _clip_geometry
        poly, calib = _clip_geometry(preset, quad)
        return poly, float(calib.net_x)
    except Exception as e:  # noqa: BLE001
        print(f"[提示] 取不到台面标定（{e}），不画台面与球网")
        return None, None


def render(run_dir: str, video: str, out: str, preset: str = "test2", quad: Optional[str] = None,
           tracknet_obs: Optional[str] = None, yolo_obs: Optional[str] = None,
           max_frames: Optional[int] = None, scale: float = 1.0, title: str = "",
           primary: str = "tracknet") -> Dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    track, label = load_ball_track(run_dir, tracknet_obs, yolo_obs, primary)
    clips = load_clips(run_dir)
    events = read_jsonl(os.path.join(run_dir, "events.jsonl"))
    decisions = load_decisions(run_dir)
    if not clips and not events and not decisions:
        raise SystemExit(f"{run_dir} 里没有 clips/events/decisions，确认是 run_cascade / run_realtime 的输出目录")

    n = n_video if n_video > 0 else (max(track) + 1 if track else 0)
    if track:
        n = min(n, max(track) + 1) if n else max(track) + 1
    if max_frames:
        n = min(n, int(max_frames)) if n else int(max_frames)
    poly, net_x = geometry(preset, quad)
    w, h = int(round(w0 * scale)), int(round(h0 * scale))
    if scale != 1.0 and poly is not None:
        poly = [[p[0] * scale, p[1] * scale] for p in poly]
        net_x = net_x * scale if net_x is not None else None
    if scale != 1.0:
        for e in events:
            if e.get("position_px"):
                e["position_px"] = [e["position_px"][0] * scale, e["position_px"][1] * scale]

    r = Renderer(w, h, n / fps if n else 1.0, poly, net_x, clips, events, decisions, label,
                 stats_summary(run_dir), title)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"无法写视频: {out}")
    written = 0
    try:
        for fid in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            b = track.get(fid)
            t = b["t"] if b else fid / fps
            if b and b.get("xy") is not None and scale != 1.0:
                b = {**b, "xy": (b["xy"][0] * scale, b["xy"][1] * scale)}
            writer.write(r.render(frame, fid, t, b))
            written += 1
            if written % 600 == 0:
                print(f"  已渲染 {written}/{n} 帧")
    finally:
        cap.release()
        writer.release()
    print(f"[完成] {out}  {written} 帧  {label}")
    return {"frames": written, "ball_label": label, "out": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="把 run_cascade / run_realtime 的输出目录画成带 HUD 的视频")
    ap.add_argument("--run-dir", required=True, help="run_cascade / run_realtime 的 --out-dir")
    ap.add_argument("--video", required=True, help="原视频（与运行时同一个）")
    ap.add_argument("--out", default=None, help="输出 mp4，默认 <run-dir>/cascade_hud.mp4")
    ap.add_argument("--preset", default="test2")
    ap.add_argument("--quad", default=None)
    ap.add_argument("--tracknet-obs", default=None,
                    help="运行目录没有 fused_obs.jsonl 时，用这份 TrackNet 观测离线画球位")
    ap.add_argument("--yolo-obs", default=None, help="同上，配合 --tracknet-obs 做离线融合")
    ap.add_argument("--primary", choices=["tracknet", "yolo"], default="tracknet")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--scale", type=float, default=1.0, help="输出缩放，如 0.5 得到 960x540")
    ap.add_argument("--title", default="")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.run_dir, "cascade_hud.mp4")
    render(a.run_dir, a.video, out, a.preset, a.quad, a.tracknet_obs, a.yolo_obs,
           a.max_frames, a.scale, a.title, a.primary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
