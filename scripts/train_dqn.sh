#!/usr/bin/env bash
# Train the DQN agent for adaptive policy control
set -euo pipefail
echo "[train_dqn] Starting DQN training..."
python -m src.rl_agent.dqn_agent \
    --mode train \
    --episodes "${DQN_EPISODES:-5000}" \
    --log-interval 100 \
    --save-path "models/dqn_best.pt"
echo "[train_dqn] Done. Model saved to models/dqn_best.pt"
