# DR-TBAC-ZT++ System Architecture

## Overview

DR-TBAC-ZT++ (Dynamic Rule Trust-Based Access Control with Zero Trust) is a
multi-component AI-driven security system for OpenStack cloud environments.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Request Layer                                │
│   OpenStack API  ─→  Kafka Topic (raw-logs)  ─→  Preprocessor      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                     Trust Assessment                                │
│   LSTM (BiLSTM)  +  Federated Learning (Flower)                    │
│   Output: trust_score ∈ [0, 1]                                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                   RL-Driven Policy Control                          │
│   State Vector: (trust_score, resource_type, time_feature, …)      │
│   DQN Agent → action_idx → PolicyRewriter                          │
│   policy.json updated dynamically                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                   Policy Decision Point (PDP)                       │
│   Trust Gate → Explicit DENY → Explicit PERMIT → Default            │
│   Output: AccessDecision (PERMIT|DENY) + reason                    │
└──────────────┬─────────────────────────────────┬───────────────────┘
               │                                 │
    ┌──────────▼──────────┐          ┌───────────▼──────────┐
    │   Decision Logger   │          │   Blockchain Audit   │
    │   CSV + Kafka       │          │   Hyperledger Fabric │
    └─────────────────────┘          └──────────────────────┘
```

## Components

### 1. Trust Assessment Module (`src/trust_assessment/`)
- **Preprocessor** – Normalises raw OpenStack logs into feature vectors
- **LSTM Model** – BiLSTM for sequential anomaly detection
- **Trust Calculator** – Combines model output → scalar trust score

### 2. Federated Learning (`src/federated_learning/`)
- **Flower Server** – Aggregates model updates using FedAvg / custom strategy
- **Flower Client** – Local training on node-partitioned data
- **Dataset Splitter** – IID / non-IID partition generation
- **Strategy** – Custom FedProx / FedAdam aggregation

### 3. RL Agent (`src/rl_agent/`)
- **Environment** – MDP wrapper around the PDP + trust state
- **DQN Agent** – Double-DQN with prioritised replay
- **Replay Buffer** – Uniform + Prioritised Experience Replay

### 4. Access Control (`src/access_control/`)
- **PDP Engine** – Zero-Trust policy evaluation (DENY-first)
- **Policy Rewriter** – Translates DQN actions to JSON mutations
- **Decision Logger** – Audit log (CSV + optional Kafka streaming)

### 5. Blockchain (`src/blockchain/`)
- **Fabric SDK** – Python wrapper for HLF transactions
- **Access Log Chaincode** – Go chaincode for immutable audit records
- **Models** – `BlockchainRecord` schema

## Data Flow
```
Raw Log → Preprocess → LSTM → trust_score
                                    │
                              State Vector
                                    │
                              DQN Agent
                                    │
                            Action → PolicyRewriter
                                    │
                              Updated policy.json
                                    │
                              PDP Evaluation
                                    │
                         PERMIT/DENY Decision
                              │           │
                      Decision Logger   Fabric Ledger
```

## Deployment Options

| Option | Components |
|--------|-----------|
| Local  | docker-compose.yml |
| Test Fabric | blockchain/fabric-network/docker-compose-fabric.yml |
| Production | k8s/*.yaml manifests |
| Enterprise | deployment/helm/ (Helm chart) |
