#!/usr/bin/env bash
# End-to-end pipeline: train -> analyze -> serve, all in isolated Python.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="python3 -I tools/run_iso.py"

echo "==> [1/3] training four models on real data (credit-g, digits)"
$PY experiments/01_train_models.py

echo "==> [2/3] running quantization analysis (all schemes + search + traces)"
$PY experiments/02_run_analysis.py

echo "==> [3/3] starting dashboard at http://127.0.0.1:8765 (Ctrl-C to stop)"
exec $PY -m quant_tool.webapp.server 8765
