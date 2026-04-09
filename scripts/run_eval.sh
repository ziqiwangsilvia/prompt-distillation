#!/usr/bin/env bash
# Run evaluation with local batch inference.
# Usage: bash run_eval.sh [config/pipeline.yaml] [--metrics-only]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}"
cd "${SCRIPT_DIR}"

METRICS_ONLY=false
CONFIG="config/pipeline.yaml"
for arg in "$@"; do
    case "$arg" in
        --metrics-only) METRICS_ONLY=true ;;
        *) CONFIG="$arg" ;;
    esac
done

eval "$(python3 -c "
import yaml
with open('${CONFIG}') as f:
    c = yaml.safe_load(f)
print(f'TRAIN_GPU=\"{c.get(\"gpu\", {}).get(\"train\", 1)}\"')
")"

echo "=== Running evaluation ==="
if [ "${METRICS_ONLY}" = true ]; then
    python3 evaluation/eval.py --pipeline_config "${CONFIG}" --metrics_only true
else
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python3 evaluation/eval.py --pipeline_config "${CONFIG}"
fi
echo "=== Evaluation complete ==="
