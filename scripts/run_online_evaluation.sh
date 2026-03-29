#!/usr/bin/env bash
# Run the online evaluation pipeline (live data simulation)
set -euo pipefail
echo "[online_eval] Starting online evaluation..."
python -m src.trust_assessment.trust_calculator \
    --mode online \
    --log-stream "data/sample_logs/openstack_sample.log" \
    --dqn-model "models/dqn_best.pt" \
    --lstm-model "models/lstm_global.pth" \
    --output-dir "experiments/results" \
    --steps "${EVAL_STEPS:-1000}"
echo "[online_eval] Evaluation complete. Results in experiments/results/"
