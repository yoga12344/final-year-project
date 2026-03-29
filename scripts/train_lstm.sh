#!/usr/bin/env bash
# Train the LSTM Trust Assessment model
set -euo pipefail
echo "[train_lstm] Starting LSTM training..."
python -m src.trust_assessment.trust_calculator \
    --mode train \
    --epochs "${LSTM_EPOCHS:-50}" \
    --batch-size "${LSTM_BATCH:-64}" \
    --data-path "data/processed/train.csv" \
    --model-out "models/lstm_global.pth"
echo "[train_lstm] Done. Model saved to models/lstm_global.pth"
