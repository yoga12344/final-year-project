#!/usr/bin/env bash
# =============================================================================
# bootstrap_fabric.sh
# Bootstraps the Hyperledger Fabric test network for DR-TBAC-ZT++.
#
# Usage:
#   chmod +x blockchain/scripts/bootstrap_fabric.sh
#   cd blockchain/fabric-network && ../scripts/bootstrap_fabric.sh
# =============================================================================
set -euo pipefail

FABRIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../fabric-network" && pwd)"
CHANNEL_NAME="tbacztchannel"
CC_NAME="access_log"
CC_VERSION="1.0"
CC_PATH="../../src/blockchain/chaincode"

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GRN}[fabric-bootstrap]${NC} $*"; }
warn() { echo -e "${YLW}[WARN]${NC} $*"; }
die()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------
for cmd in docker docker-compose cryptogen configtxgen; do
    command -v "$cmd" &>/dev/null || die "$cmd not found – please install Hyperledger Fabric binaries."
done

cd "$FABRIC_DIR"

log "Step 1/6: Generating crypto material..."
cryptogen generate --config=./crypto-config.yaml --output=./crypto-config \
    || die "cryptogen failed"

log "Step 2/6: Generating genesis block..."
mkdir -p channel-artifacts
configtxgen -profile TBACZTGenesis \
    -channelID system-channel \
    -outputBlock ./channel-artifacts/genesis.block \
    -configPath . \
    || die "Genesis block generation failed"

log "Step 3/6: Generating channel transaction..."
configtxgen -profile TBACZTChannel \
    -channelID "$CHANNEL_NAME" \
    -outputCreateChannelTx ./channel-artifacts/channel.tx \
    -configPath . \
    || die "Channel tx generation failed"

log "Step 4/6: Generating anchor peer updates..."
for org in Org1MSP Org2MSP; do
    configtxgen -profile TBACZTChannel \
        -channelID "$CHANNEL_NAME" \
        -outputAnchorPeersUpdate "./channel-artifacts/${org}anchors.tx" \
        -asOrg "$org" \
        -configPath . \
        || die "Anchor peer update failed for $org"
done

log "Step 5/6: Starting Fabric network..."
docker-compose -f docker-compose-fabric.yml up -d \
    || die "docker-compose failed"

log "Waiting 10s for peers to start..."
sleep 10

log "Step 6/6: Creating channel and deploying chaincode..."
# Use CLI container for channel & chaincode operations
docker exec cli peer channel create \
    -o orderer:7050 \
    -c "$CHANNEL_NAME" \
    -f /opt/gopath/src/github.com/hyperledger/fabric/peer/channel-artifacts/channel.tx \
    --tls --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/tbaczt.com/orderers/orderer.tbaczt.com/msp/tlscacerts/tlsca.tbaczt.com-cert.pem

docker exec cli peer channel join \
    -b "${CHANNEL_NAME}.block"

# Package, install, approve and commit chaincode
docker exec cli peer lifecycle chaincode package "${CC_NAME}.tar.gz" \
    --path "/opt/gopath/src/github.com/chaincode" \
    --lang golang \
    --label "${CC_NAME}_${CC_VERSION}"

docker exec cli peer lifecycle chaincode install "${CC_NAME}.tar.gz"

log "Fabric bootstrap complete! Channel: ${CHANNEL_NAME}, Chaincode: ${CC_NAME} v${CC_VERSION}"
echo ""
log "Run 'docker-compose -f $FABRIC_DIR/docker-compose-fabric.yml ps' to check status."
