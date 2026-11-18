"""用国栋仓库里的 TableCalibration / TableHomography 读一遍我们的输出，确认能直接用。

只 import、不修改 pingpong-smart-referee。
用法：
  python scripts/check_compat.py out/test_2/result.json [--repo ../pingpong-smart-referee]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--repo", default=os.path.join(ROOT, "..", "pingpong-smart-referee"))
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo))
    sys.dont_write_bytecode = True  # 不在别人的仓库里留下 __pycache__
    from core.geometry import TableCalibration  # noqa: E402
    from core.homography import TableHomography  # noqa: E402

    ok_all = True
    for path in args.results:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        tc = d.get("table_calibration")
        if not tc or tc.get("roi_box") is None or tc.get("net_zone") is None:
            print(f"{path}: 缺少 TableCalibration 字段，跳过")
            ok_all = False
            continue
        calib = TableCalibration.from_dict(tc)
        homo = TableHomography(calib)
        back = TableCalibration.from_dict(calib.to_dict())
        net_cm = homo.calibration_net_x_cm()
        ok = abs(net_cm - 137.0) <= 1.0 and back == calib
        ok_all &= ok
        print(
            f"{tc['name']}: from_dict OK, TableHomography OK, "
            f"net_x={calib.net_x}px → {net_cm:.1f} cm (要求 137±1)  {'OK' if ok else 'NG'}"
        )
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
