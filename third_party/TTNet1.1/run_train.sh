#!/usr/bin/env bash
# 用法：bash run_train.sh [deploy_pc.py train 的参数]
# 依次执行三个阶段，每阶段 30 epochs；需要完整训练及测试数据。
set -euo pipefail
PY="${PY:-${PYTHON:-python}}"
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
exec "$PY" "$SCRIPT_DIR/deploy_pc.py" train "$@"
