"""把自动标定结果和人工四角比较。

用法：
  python scripts/evaluate.py                       # 默认读 ../TN/calib/table_quads.json 与 out/*/result.json
  python scripts/evaluate.py --truth my_truth.json # 换真值文件（比如 test_2 重新核定后）

判定（与计划一致）：
  每个角误差 <= 5 px 合格；5~8 px 需目检；> 8 px 不合格。
  抖动：每个角的多帧标准差 <= 2 px（test_6 有遮挡，放宽到 5 px）。
输出 out/eval.md、out/eval.json。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from tcalib.table import CORNER_NAMES  # noqa: E402

PASS_PX, REVIEW_PX = 5.0, 8.0
JITTER_PX = {"default": 2.0, "test_6": 5.0}


def grade(err: float) -> str:
    if err <= PASS_PX:
        return "pass"
    if err <= REVIEW_PX:
        return "review"
    return "fail"


def cascade_net_x(quad) -> float:
    """TN/cascade/calib.py 现有的经验取法，用来对比。"""
    (flx, _), (frx, _), (nrx, _), (nlx, _) = quad
    return ((flx + frx) / 2.0 + (nlx + nrx) / 2.0) / 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=os.path.join(ROOT, "..", "TN", "calib", "table_quads.json"))
    ap.add_argument("--out-root", default=os.path.join(ROOT, "out"))
    args = ap.parse_args()

    with open(args.truth, encoding="utf-8") as f:
        truth = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

    rows, summary = [], {}
    for key in sorted(truth):
        path = os.path.join(args.out_root, key, "result.json")
        gt = np.array(truth[key]["points"], dtype=float)
        if not os.path.exists(path):
            summary[key] = {"status": "missing"}
            rows.append(f"| {key} | {truth[key].get('method', '')} | — | — | — | — | 无结果 |")
            continue
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
        if res.get("points") is None:
            summary[key] = {"status": "no_points", "confidence": res.get("confidence")}
            rows.append(f"| {key} | {truth[key].get('method', '')} | — | — | — | {res.get('confidence')} | 检测失败 |")
            continue
        pts = np.array(res["points"], dtype=float)
        delta = pts - gt
        err = np.linalg.norm(delta, axis=1)
        grades = [grade(e) for e in err]
        std = res.get("per_corner_std_px") or [float("nan")] * 4
        jit_lim = JITTER_PX.get(key, JITTER_PX["default"])
        jitter_ok = all(s <= jit_lim for s in std)
        corner_ok = all(g == "pass" for g in grades)
        overall = "pass" if corner_ok and jitter_ok else ("fail" if "fail" in grades else "review")
        tc = res.get("table_calibration") or {}
        summary[key] = {
            "status": overall,
            "err_px": [round(float(e), 2) for e in err],
            "delta_px": [[round(float(a), 2), round(float(b), 2)] for a, b in delta],
            "grades": grades,
            "std_px": std,
            "jitter_ok": jitter_ok,
            "confidence": res.get("confidence"),
            "net_x_new": tc.get("net_x"),
            "net_x_cascade": round(cascade_net_x(gt), 1),
            "reproj_px": (res.get("camera") or {}).get("reproj_rms_px"),
            "f_px": (res.get("camera") or {}).get("f_px"),
        }
        err_s = " / ".join(f"{e:.1f}" for e in err)
        std_s = " / ".join(f"{s:.1f}" for s in std)
        rows.append(
            f"| {key} | {truth[key].get('method', '')} | {err_s} | {err.max():.1f} | {std_s} | "
            f"{res.get('confidence')} | {overall} |"
        )

    n_pass = sum(1 for v in summary.values() if v.get("status") == "pass")
    lines = [
        "# 自动标定 vs 人工四角",
        "",
        f"真值：`{os.path.relpath(args.truth, ROOT)}`。角点误差 ≤{PASS_PX:.0f}px 合格，"
        f"≤{REVIEW_PX:.0f}px 需目检；抖动上限 {JITTER_PX['default']:.0f}px（test_6 为 {JITTER_PX['test_6']:.0f}px）。",
        "",
        "注意：除 test_2（手标）外，真值本身也来自自动拟合，误差小只说明与旧方法一致。",
        "",
        f"误差顺序：{' / '.join(CORNER_NAMES)}",
        "",
        "| 视频 | 真值来源 | 角点误差 px | 最大 | 多帧抖动 px | 置信度 | 结论 |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
        f"合格 {n_pass} / {len(truth)}（目标：至少 8 段全部合格）",
    ]
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out_root, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
