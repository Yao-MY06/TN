"""把一次 run_mixed.py 的事件流画成单文件 HTML 时间轴（双击即开）。

输入两样东西，都来自同一次运行：

  python run_mixed.py ... --events-out ev.jsonl --log run.log
  python build_timeline.py --events ev.jsonl --log run.log \
      --out ../outputs/xxx_timeline.html --title "game_4 470~526s"

时间轴泳道上：上 A（物理左半台）、下 B（物理右半台）。
红色实心圆 = 落台 BOUNCE，蓝色空心方块 = 击球 RACKET_HIT，
橙色三角 = 出界 OUT，红色虚线 = 判罚时刻。
"""

import argparse
import json
import os
import re

OUT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.normpath(os.path.join(OUT, "..", "outputs"))

# 判罚日志形如：
#   [帧 0639 |   5.325s] >> 判罚: P1(左) 得分 [OWN_COURT] 球落在击球方(B)自己半台
DEC_RE = re.compile(r"\[帧\s*(\d+)\s*\|\s*([0-9.]+)s\]\s*>>\s*判罚:\s*(.+)$")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_decisions(log_path, fps):
    """从运行日志里抽出判罚行，并按顺序累计比分。"""
    out = []
    score = {"A": 0, "B": 0}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = DEC_RE.search(line.strip())
            if not m:
                continue
            frame, t, text = int(m.group(1)), float(m.group(2)), m.group(3).strip()
            kind = "dead"
            if text.startswith("P1"):
                score["A"] += 1
                kind = "point"
            elif text.startswith("P2"):
                score["B"] += 1
                kind = "point"
            head, _, detail = text.partition("]")
            head = head.lstrip("P1P2()左 右")  # 只用于展示，原文保留在 detail 里
            out.append(
                {
                    "frame": frame,
                    "t": t,
                    "text": text,
                    "kind": kind,
                    "score": f'{score["A"]} : {score["B"]}',
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="run_mixed.py --events-out 写出的 JSONL")
    ap.add_argument("--log", required=True, help="run_mixed.py --log 写出的判罚日志")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="混合事件流 · 判分链路")
    ap.add_argument("--sub", default="")
    ap.add_argument("--fps", type=float, default=120.0)
    ap.add_argument("--tmax", type=float, default=None, help="时间轴总长（秒），默认取最后一条事件")
    args = ap.parse_args()

    events = load_jsonl(args.events)
    decisions = parse_decisions(args.log, args.fps)
    tmax = args.tmax or (max([e["timestamp_s"] for e in events] + [d["t"] for d in decisions]) + 1.0)

    bounces = [e for e in events if e["type"] == "BOUNCE"]
    hits = [e for e in events if e["type"] == "RACKET_HIT"]
    outs = [e for e in events if e["type"] == "OUT"]
    points = [d for d in decisions if d["kind"] == "point"]

    W, PAD_L, PAD_R = 1240, 96, 30
    LANE_A_Y, LANE_B_Y = 128, 208
    AXIS_Y = 282
    H = 356

    def x_of(t):
        return PAD_L + (t / tmax) * (W - PAD_L - PAD_R)

    def lane(side):
        return LANE_A_Y if side == "A" else LANE_B_Y

    svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="display:block">']

    for y, label, color in (
        (LANE_A_Y, "半台 A（物理左）", "#38bdf8"),
        (LANE_B_Y, "半台 B（物理右）", "#fbbf24"),
    ):
        svg.append(
            f'<rect x="{PAD_L}" y="{y-24}" width="{W-PAD_L-PAD_R}" height="48" rx="6" '
            f'fill="{color}" opacity="0.07"/>'
        )
        svg.append(
            f'<text x="{PAD_L-12}" y="{y+4}" text-anchor="end" font-size="12" '
            f'fill="#94a3b8">{label}</text>'
        )
        svg.append(
            f'<line x1="{PAD_L}" y1="{y}" x2="{W-PAD_R}" y2="{y}" stroke="{color}" '
            f'stroke-opacity="0.25" stroke-width="1"/>'
        )

    for i, d in enumerate(decisions, 1):
        x = x_of(d["t"])
        col = "#ef4444" if d["kind"] == "point" else "#64748b"
        svg.append(
            f'<line x1="{x:.1f}" y1="86" x2="{x:.1f}" y2="{AXIS_Y}" stroke="{col}" '
            f'stroke-width="1.4" stroke-dasharray="5 4" opacity="0.75"/>'
        )
        svg.append(f'<circle cx="{x:.1f}" cy="78" r="10" fill="{col}"/>')
        svg.append(
            f'<text x="{x:.1f}" y="82" text-anchor="middle" font-size="11" '
            f'font-weight="700" fill="#0b0e14">{i}</text>'
        )

    for h in hits:
        x = x_of(h["timestamp_s"])
        y = lane(h["side"])
        svg.append(
            f'<rect x="{x-4:.1f}" y="{y-4}" width="8" height="8" rx="1.5" fill="none" '
            f'stroke="#7dd3fc" stroke-width="1.6" opacity="0.95"/>'
        )

    for b in bounces:
        x = x_of(b["timestamp_s"])
        y = lane(b["side"])
        svg.append(
            f'<circle cx="{x:.1f}" cy="{y}" r="5.4" fill="#f87171" stroke="#1a1f2b" stroke-width="1.2"/>'
        )

    for o in outs:
        x = x_of(o["timestamp_s"])
        y = lane(o["side"])
        svg.append(
            f'<path d="M {x:.1f} {y-6} L {x+6:.1f} {y+4} L {x-6:.1f} {y+4} Z" '
            f'fill="#fb923c" stroke="#1a1f2b" stroke-width="1.2"/>'
        )

    svg.append(
        f'<line x1="{PAD_L}" y1="{AXIS_Y}" x2="{W-PAD_R}" y2="{AXIS_Y}" stroke="#334155" stroke-width="1"/>'
    )
    step = 5 if tmax <= 40 else 10 if tmax <= 120 else 30
    s = 0
    while s <= tmax:
        x = x_of(s)
        svg.append(f'<line x1="{x:.1f}" y1="{AXIS_Y}" x2="{x:.1f}" y2="{AXIS_Y+5}" stroke="#475569"/>')
        svg.append(
            f'<text x="{x:.1f}" y="{AXIS_Y+20}" text-anchor="middle" font-size="11" '
            f'fill="#64748b">{s}s</text>'
        )
        s += step

    lx, ly = PAD_L, AXIS_Y + 50
    svg.append(f'<circle cx="{lx+6}" cy="{ly}" r="5.4" fill="#f87171"/>')
    svg.append(f'<text x="{lx+18}" y="{ly+4}" font-size="12" fill="#94a3b8">落台 BOUNCE（我方规则层）</text>')
    svg.append(
        f'<rect x="{lx+232}" y="{ly-4}" width="8" height="8" rx="1.5" fill="none" '
        f'stroke="#7dd3fc" stroke-width="1.6"/>'
    )
    svg.append(f'<text x="{lx+248}" y="{ly+4}" font-size="12" fill="#94a3b8">击球 RACKET_HIT（状态机轨迹层）</text>')
    svg.append(
        f'<path d="M {lx+500} {ly-6} L {lx+506} {ly+4} L {lx+494} {ly+4} Z" fill="#fb923c"/>'
    )
    svg.append(f'<text x="{lx+514}" y="{ly+4}" font-size="12" fill="#94a3b8">出界 OUT</text>')
    svg.append(
        f'<line x1="{lx+610}" y1="{ly-5}" x2="{lx+610}" y2="{ly+5}" stroke="#ef4444" '
        f'stroke-width="1.4" stroke-dasharray="5 4"/>'
    )
    svg.append(f'<text x="{lx+622}" y="{ly+4}" font-size="12" fill="#94a3b8">判罚</text>')
    svg.append("</svg>")
    svg_s = "\n".join(svg)

    rows = "\n".join(
        f"""<tr>
      <td class="num">{i}</td>
      <td class="mono">{d['t']:.3f}s</td>
      <td class="mono dim">帧 {d['frame']}</td>
      <td><span class="pill {'pill-point' if d['kind']=='point' else 'pill-dead'}">{d['text'].split(']')[0] + ']' if ']' in d['text'] else d['text']}</span></td>
      <td class="dim">{d['text'].split(']', 1)[1].strip() if ']' in d['text'] else ''}</td>
      <td class="mono score">{d['score']}</td>
    </tr>"""
        for i, d in enumerate(decisions, 1)
    )

    CLASS = {"BOUNCE": "ev-bounce", "RACKET_HIT": "ev-hit", "OUT": "ev-out"}
    ev_rows = []
    for e in sorted(events, key=lambda e: e["timestamp_s"]):
        pos = e.get("position_px")
        pos_s = f"({pos[0]:.0f}, {pos[1]:.0f})" if pos else "—"
        ev_rows.append(
            f'<tr><td class="mono">{e["timestamp_s"]:.3f}s</td>'
            f'<td class="mono dim">帧 {int(round(e["timestamp_s"] * args.fps))}</td>'
            f'<td><span class="tag {CLASS.get(e["type"], "")}">{e["type"]}</span></td>'
            f'<td class="mono">{e["side"]}</td>'
            f'<td class="mono dim">{pos_s}</td></tr>'
        )

    final = decisions[-1]["score"] if decisions else "0 : 0"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{args.title}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 32px 28px 60px; background: #0b0e14; color: #e2e8f0;
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; line-height: 1.6; }}
  .wrap {{ max-width: 1280px; margin: 0 auto; }}
  h1 {{ font-size: 21px; margin: 0 0 6px; letter-spacing: .2px; }}
  .sub {{ color: #64748b; font-size: 13px; margin-bottom: 24px; }}
  .card {{ background: #11151d; border: 1px solid #1e2634; border-radius: 12px;
    padding: 18px 20px; margin-bottom: 18px; }}
  .card h2 {{ font-size: 14px; margin: 0 0 14px; color: #cbd5e1; font-weight: 600; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px; }}
  .stat {{ background: #11151d; border: 1px solid #1e2634; border-radius: 10px; padding: 14px 16px; }}
  .stat .k {{ font-size: 11px; color: #64748b; letter-spacing: .6px; }}
  .stat .v {{ font-size: 24px; font-weight: 700; margin-top: 3px; }}
  .stat .v small {{ font-size: 12px; font-weight: 400; color: #64748b; margin-left: 3px; }}
  .c-red {{ color: #f87171; }} .c-blue {{ color: #7dd3fc; }} .c-dim {{ color: #94a3b8; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; font-weight: 500; color: #64748b; font-size: 11.5px;
    letter-spacing: .5px; padding: 7px 10px; border-bottom: 1px solid #1e2634; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #151b25; vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  .mono {{ font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-variant-numeric: tabular-nums; }}
  .dim {{ color: #64748b; }}
  .num {{ color: #475569; width: 28px; }}
  .score {{ font-weight: 700; color: #e2e8f0; text-align: right; }}
  .pill {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 12px; font-weight: 500; }}
  .pill-point {{ background: rgba(239,68,68,.14); color: #fca5a5; border: 1px solid rgba(239,68,68,.28); }}
  .pill-dead  {{ background: rgba(100,116,139,.16); color: #94a3b8; border: 1px solid rgba(100,116,139,.28); }}
  .tag {{ font-size: 11px; padding: 1px 7px; border-radius: 4px; font-family: ui-monospace, monospace; }}
  .ev-bounce {{ background: rgba(248,113,113,.14); color: #fca5a5; }}
  .ev-hit {{ background: rgba(125,211,252,.12); color: #7dd3fc; }}
  .ev-out {{ background: rgba(251,146,60,.14); color: #fdba74; }}
  .tablewrap {{ max-height: 340px; overflow-y: auto; }}
  .tablewrap::-webkit-scrollbar {{ width: 8px; }}
  .tablewrap::-webkit-scrollbar-thumb {{ background: #1e2634; border-radius: 4px; }}
  .note {{ font-size: 12.5px; color: #64748b; line-height: 1.75; }}
  .note b {{ color: #94a3b8; font-weight: 600; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{args.title}</h1>
  <div class="sub">{args.sub}</div>

  <div class="stats">
    <div class="stat"><div class="k">落台 BOUNCE</div><div class="v c-red">{len(bounces)}<small>条</small></div></div>
    <div class="stat"><div class="k">击球 RACKET_HIT</div><div class="v c-blue">{len(hits)}<small>条</small></div></div>
    <div class="stat"><div class="k">判罚（其中得分 {len(points)} 次）</div><div class="v c-dim">{len(decisions)}<small>条</small></div></div>
    <div class="stat"><div class="k">最终比分</div><div class="v">P1 {final} P2</div></div>
  </div>

  <div class="card">
    <h2>事件时间轴（{tmax:.1f}s，出界 {len(outs)} 条）</h2>
    {svg_s}
  </div>

  <div class="card">
    <h2>判罚明细</h2>
    <table>
      <thead><tr><th>#</th><th>时间</th><th>视频帧</th><th>判罚</th><th>依据</th><th style="text-align:right">比分</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>全部事件（{len(events)} 条）</h2>
    <div class="tablewrap">
      <table>
        <thead><tr><th>时间</th><th>视频帧</th><th>类型</th><th>半台</th><th>图像坐标</th></tr></thead>
        <tbody>{''.join(ev_rows)}</tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h2>怎么读这张图</h2>
    <div class="note">
      <b>上下两条泳道</b>分别是球台的物理左半台（A）与右半台（B）。正常对打的落台应在两侧交替出现。<br>
      <b>红色实心圆</b>＝本项目的规则层落台；<b>蓝色空心方块</b>＝状态机包轨迹层给出的击球；
      <b>橙色三角</b>＝出界（球在台外失联，两套事件层都不覆盖，由外层补推断）；<b>红色虚线</b>＝判罚时刻。
    </div>
  </div>
</div>
</body>
</html>
"""
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"written: {args.out}")
    print(f"bounces={len(bounces)} hits={len(hits)} outs={len(outs)} decisions={len(decisions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
