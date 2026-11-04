"""从 OpenTTGames 的人工球位置标注（ball_markup）推断事件标注（events_markup）。

为什么需要：Extended OpenTTGames 的 test_2 事件文件里没有落台/过网，原版
``events_markup.json`` 又只在原机的 ``_data`` 里。但 OpenTTGames 的球位置标注是
**围绕事件**采的：每个事件对应一段 17 帧的连续标注 ``[e-4, e+12]``，两事件相距不足
17 帧时两段连在一起。于是：

* 事件帧 = 每段起点 + 4；段长 > 17 时再加 终点 - 12
* 类型：离球网 < 60 px 且该帧不是 ±4 帧内 y 最大（球的最低点）→ 过网，否则落台

在另外 6 段有完整标注的测试视频（test_1、3–7）上验证（``--validate``）：
370 个落台/过网事件**逐帧全部找回**；多推出 10 个，基本对应 ``empty_event``；
类型判对 359/364（98.6%），球网位置偏 ±30 px 时仍有 357–359。

所以这是**推断真值**，不是原始标注：拿到原版 events_markup.json 后应以原版为准。

    python -m cascade.infer_markup --ball <ball_markup.json> --net-x 929 --out <events_markup_inferred.json>
    python -m cascade.infer_markup --validate --ext-root _data/gt/extended_openttgames
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List, Tuple

SEG_LEN = 17          # 每个事件的标注段长度
EVENT_OFFSET = 4      # 事件帧在段内的位置
NET_PX = 60.0
YMAX_K = 4


def _segments(frames: List[int]) -> List[Tuple[int, int]]:
    out, s, p = [], None, None
    for f in frames:
        if s is None:
            s = p = f
        elif f == p + 1:
            p = f
        else:
            out.append((s, p))
            s = p = f
    if s is not None:
        out.append((s, p))
    return out


def load_ball(path: str) -> Dict[int, Tuple[float, float]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): (float(v["x"]), float(v["y"])) for k, v in raw.items()}


def event_frames(ball: Dict[int, Tuple[float, float]]) -> List[int]:
    ev = []
    for s, e in _segments(sorted(ball)):
        n = e - s + 1
        if n < SEG_LEN:
            continue
        ev.append(s + EVENT_OFFSET)
        if n > SEG_LEN:
            ev.append(e - (SEG_LEN - 1 - EVENT_OFFSET))
    return ev


def _is_ymax(ball, f: int) -> bool:
    ys = [ball[x][1] for x in range(f - YMAX_K, f + YMAX_K + 1) if x in ball and ball[x][1] >= 0]
    return f in ball and ball[f][1] >= 0 and len(ys) >= 5 and ball[f][1] >= max(ys)


def classify(ball, f: int, net_x: float) -> str:
    x = ball[f][0] if f in ball else -1
    if x < 0:
        return "bounce" if _is_ymax(ball, f) else "unknown"
    near = abs(x - net_x) < NET_PX
    return "net" if near and not _is_ymax(ball, f) else "bounce"


def infer(ball, net_x: float) -> Dict[int, str]:
    return {f: classify(ball, f, net_x) for f in event_frames(ball)}


def validate(ext_root: str) -> Dict:
    rows, tot = [], {"gt": 0, "found": 0, "extra": 0, "typed": 0, "typed_ok": 0}
    for v in ("test_1", "test_3", "test_4", "test_5", "test_6", "test_7"):
        ball = load_ball(os.path.join(ext_root, "ball_data", "test", f"{v}.json"))
        with open(os.path.join(ext_root, "game_data", "test", f"{v}.json"), encoding="utf-8") as f:
            game = {int(k): s.split()[0] for k, s in json.load(f).items()}
        gt = {f: lab for f, lab in game.items() if lab in ("bounce", "net")}
        net_x = statistics.median(ball[f][0] for f, lab in gt.items()
                                  if lab == "net" and f in ball and ball[f][0] >= 0)
        inf = infer(ball, net_x)
        found = [f for f in inf if f in gt]
        typed_ok = sum(inf[f] == gt[f] for f in found)
        rows.append({"video": v, "gt": len(gt), "found": len(found), "extra": len(inf) - len(found),
                     "type_acc": f"{typed_ok}/{len(found)}"})
        tot["gt"] += len(gt)
        tot["found"] += len(found)
        tot["extra"] += len(inf) - len(found)
        tot["typed"] += len(found)
        tot["typed_ok"] += typed_ok
    return {"videos": rows, "total": tot}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="从球位置标注推断落台/过网事件帧")
    ap.add_argument("--ball", help="ball_markup.json（OpenTTGames 格式）")
    ap.add_argument("--net-x", type=float, help="球网在图像中的 x（像素），如 test2 预设的 929")
    ap.add_argument("--out", help="输出 events_markup 格式 JSON")
    ap.add_argument("--validate", action="store_true", help="在 test_1、3–7 上验证推断规则")
    ap.add_argument("--ext-root", default="_data/gt/extended_openttgames")
    a = ap.parse_args(argv)
    if a.validate:
        r = validate(a.ext_root)
        for row in r["videos"]:
            print(row)
        t = r["total"]
        print(f"事件帧找回 {t['found']}/{t['gt']}，多推 {t['extra']}，类型正确 {t['typed_ok']}/{t['typed']}")
        return 0
    if not (a.ball and a.net_x is not None and a.out):
        ap.error("需要 --ball --net-x --out，或 --validate")
    inf = infer(load_ball(a.ball), a.net_x)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in sorted(inf.items())}, f, ensure_ascii=False, indent=1)
    c = {k: sum(1 for v in inf.values() if v == k) for k in ("bounce", "net", "unknown")}
    print(f"推断事件 {len(inf)} 个：{c} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
