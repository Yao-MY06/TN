#!/usr/bin/env bash
# 用法：bash run_demo.sh [视频路径] [deploy_pc.py demo 的其他参数]
# 相对路径以项目根目录为基准；默认使用资产包内 test_2.mp4。
set -euo pipefail
PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
VIDEO="${1:-dataset/test/videos/test_2.mp4}"
if [ "$#" -gt 0 ]; then shift; fi
exec "$PYTHON" "$SCRIPT_DIR/deploy_pc.py" demo --video "$VIDEO" "$@"
