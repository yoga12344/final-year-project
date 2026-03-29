#!/usr/bin/env bash
# Start the Flower Federated Learning server + default clients
set -euo pipefail
N_CLIENTS="${FL_NUM_CLIENTS:-3}"
ROUNDS="${FL_ROUNDS:-50}"
echo "[federated] Starting Flower server for $ROUNDS rounds..."
python -m src.federated_learning.flower_server \
    --rounds "$ROUNDS" \
    --min-clients "$N_CLIENTS" &
SERVER_PID=$!
sleep 5   # wait for server to be ready
echo "[federated] Launching $N_CLIENTS simulated clients..."
for i in $(seq 1 "$N_CLIENTS"); do
    python -m src.federated_learning.flower_client \
        --client-id "$i" \
        --data-split "data/processed/split_${i}.csv" &
done
wait $SERVER_PID
echo "[federated] Federated training complete."
