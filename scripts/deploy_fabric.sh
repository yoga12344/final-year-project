#!/usr/bin/env bash
# Deploy Hyperledger Fabric network
set -euo pipefail
echo "[deploy_fabric] Bootstrapping Fabric network..."
cd "$(dirname "$0")/../blockchain/fabric-network"
bash ../scripts/bootstrap_fabric.sh
echo "[deploy_fabric] Fabric network deployed successfully."
