"""批量标定多段视频，并把结果打包成一个小 zip（用于从原机传回）。

用法（原机，Windows PowerShell）：
  python scripts\\batch_videos.py --videos videos.txt --out-root out --mode spread --count 24 `
      --save-frame test_2:1200 --truth <TN>\\calib\\table_quads.json --zip results.zip

videos.txt 每行一段：名字<TAB或空格>视频路径（# 开头为注释），名字与 table_quads.json 的键一致，
例如：
  test_2   D:\\data\\OpenTTGames\\test_2.mp4

每段视频的结果写到 <out-root>/<名字>/（result.json、frames.csv、overlay.jpg、corners.png、frames/*.png）。
汇总写到 <out-root>/batch_summary.json；给了 --truth 时再跑 evaluate.py 生成 eval.md / eval.json。
单段失败不影响其它段，错误记在汇总里。只读视频和真值文件，不改任何其它文件。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from calibrate_video import run_video  # noqa: E402

# 打包时只带这些文件（不带抽出的全部帧，体积小）
PACK_EXT = (".json", ".csv", ".jpg", ".png", ".md", ".txt")


def read_list(path: str) -> List[Tuple[str, str]]:
    items = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise SystemExit(f"videos 列表格式不对：{line!r}")
            items.append((parts[0], parts[1].strip().strip('"')))
    return items


def parse_save(values: List[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for v in values:
        name, fid = v.split(":", 1)
        out.setdefault(name, []).append(int(fid))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="视频列表文件")
    ap.add_argument("--out-root", default="out")
    ap.add_argument("--mode", choices=("opening", "spread"), default="spread")
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--save-frame", action="append", default=[], help="名字:帧号，另存原图 PNG，可重复")
    ap.add_argument("--truth", default=None, help="table_quads.json（只读），给了就跑验收")
    ap.add_argument("--zip", default=None, help="把结果打包到这个 zip")
    args = ap.parse_args()

    items = read_list(args.videos)
    saves = parse_save(args.save_frame)
    os.makedirs(args.out_root, exist_ok=True)
    summary = {"mode": args.mode, "count": args.count, "python": sys.version.split()[0], "videos": {}}
    try:
        import cv2

        summary["opencv"] = cv2.__version__
    except Exception:  # noqa: BLE001
        pass
    for name, path in items:
        t0 = time.time()
        print(f"== {name}: {path}", flush=True)
        try:
            d = run_video(path, name, os.path.join(args.out_root, name), mode=args.mode, count=args.count,
                          save_frame_ids=saves.get(name, []))
            summary["videos"][name] = {
                "ok": d.get("points") is not None,
                "confidence": d.get("confidence"),
                "reasons": d.get("confidence_reasons"),
                "n_used": d.get("n_used"),
                "n_frames": d.get("n_frames"),
                "points": d.get("points"),
                "std_px": d.get("per_corner_std_px"),
                "seek_mismatch": d["video"]["seek_mismatch"],
                "runtime_s": d.get("runtime_s"),
            }
            print(f"   conf={d.get('confidence')} used {d.get('n_used')}/{d.get('n_frames')} "
                  f"std={d.get('per_corner_std_px')} {time.time() - t0:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001  一段失败不影响其它段
            summary["videos"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                       "traceback": traceback.format_exc()}
            print(f"   失败：{e}", flush=True)
    with open(os.path.join(args.out_root, "batch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.truth:
        r = subprocess.run([sys.executable, os.path.join(HERE, "evaluate.py"), "--truth", args.truth,
                            "--out-root", args.out_root])
        if r.returncode != 0:
            print("[warn] evaluate.py 失败", file=sys.stderr)

    if args.zip:
        n = 0
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(args.out_root):
                for fn in files:
                    if fn.lower().endswith(PACK_EXT):
                        p = os.path.join(root, fn)
                        z.write(p, os.path.relpath(p, os.path.dirname(os.path.abspath(args.out_root))))
                        n += 1
        print(f"打包 {n} 个文件 → {args.zip}（{os.path.getsize(args.zip) / 1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
