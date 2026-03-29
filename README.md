# DR-TBAC-ZT++ : Dynamic Rule-Based Trust & Access Control with Zero Trust

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![Flower](https://img.shields.io/badge/Flower-1.6%2B-purple.svg)](https://flower.dev)

**Final Year Project** - A comprehensive implementation of AI-driven access control for cloud security.

---

## 🔐 Overview

**DR-TBAC-ZT++** is a cutting-edge, production-ready framework for **dynamic, AI-driven access control** in cloud environments (OpenStack). It combines:

| Component | Technology | Objective |
|---|---|---|
| Trust Assessment | Bi-LSTM + SHAP | Continuous user trust scoring |
| Federated Learning | Flower (flwr) | Privacy-preserving model training across nodes |
| Dynamic Rule Engine | DQN (Deep Q-Network) | Autonomous policy adaptation |
| Access Control | PDP / OpenStack Keystone | Policy Decision Point enforcement |
| Audit Blockchain | Hyperledger Fabric | Immutable, tamper-proof access logs |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       OpenStack Cloud Environment                     │
│                                                                        │
│  User/Service ──► Keystone Auth ──► PDP Engine ──► Resource Access   │
│                         │                │                             │
│                    Trust Score      Policy Decision                    │
│                         │                │                             │
│               ┌─────────┘          ┌─────┘                            │
│          Bi-LSTM Model          DQN Agent                             │
│          (Trust Scoring)        (Rule Adaptation)                     │
│               │                      │                                 │
│          Flower FL           Replay Buffer                             │
│          (Federated)         (Experience)                              │
│                    └────────────────────────────► Hyperledger Fabric  │
│                                                   (Audit Blockchain)  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- CUDA 11.8+ (optional, for GPU training)
- Go 1.21+ (for chaincode compilation)

### 1. Clone & Setup Environment

```bash
git clone https://github.com/your-org/dr-tbac-zt-plus-plus.git
cd dr-tbac-zt-plus-plus

# Using conda (recommended)
conda env create -f environment.yml
conda activate dr-tbac-zt

# Or using pip
pip install -r requirements.txt
```

### 2. Configure

```bash
cp src/config.py.example src/config.py
# Edit src/config.py with your OpenStack credentials and Kafka brokers
```

### 3. Launch Full Stack (Docker Compose)

```bash
docker-compose up -d
```

### 4. Run Training Pipeline

```bash
# Step 1: Train the Bi-LSTM trust model
bash scripts/train_lstm.sh

# Step 2: Start federated learning across nodes
bash scripts/start_federated.sh

# Step 3: Train the DQN agent
bash scripts/train_dqn.sh

# Step 4: Deploy Hyperledger Fabric
bash scripts/deploy_fabric.sh

# Step 5: Run online evaluation
bash scripts/run_online_evaluation.sh
```

---

## 📂 Project Structure

```
final-year-project/
├── api/                     # FastAPI REST API for trust assessment
├── blockchain/              # Hyperledger Fabric network & chaincode
│   └── fabric-network/      # Fabric config, crypto, scripts
├── config/                  # Configuration files
├── data/                    # Datasets (raw, processed, sample logs)
├── demo/                    # Demo videos & screenshots
├── deployment/              # Docker & Helm deployment configs
├── docs/                    # Documentation (API, architecture, etc.)
├── experiments/             # Experiment results & notebooks
├── k8s/                     # Kubernetes manifests
├── logs/                    # Application logs
├── models/                  # Trained models (LSTM, DQN)
├── scripts/                 # Shell scripts for training & deployment
├── src/                     # Source code
│   ├── access_control/      # PDP engine & policy management
│   ├── blockchain/          # Fabric SDK integration
│   ├── explainability/      # SHAP explainers
│   ├── federated_learning/  # Flower client/server
│   ├── rl_agent/            # DQN agent & environment
│   ├── trust_assessment/    # LSTM model & preprocessor
│   └── utils/               # Shared utilities
├── tests/                   # Unit & integration tests
├── app.py                   # Main application entry
├── dashboard.py             # Streamlit dashboard
├── run_pipeline.py          # Pipeline runner
├── streamlit_app.py         # Streamlit app
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment
├── docker-compose.yml       # Docker Compose setup
└── README.md                # This file
```

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Trust Prediction Accuracy | 97.3% |
| False Positive Rate | 1.2% |
| Policy Decision Latency | < 50ms |
| Federated Convergence (5 nodes) | 12 rounds |
| Blockchain Throughput | 1,200 TPS |

> Results obtained on CICIDS-2018 + synthetic OpenStack logs dataset.

---

## 📖 Research Objectives

1. **Objective 1**: Continuous trust assessment using Bi-LSTM on behavioral features
2. **Objective 2**: Policy Decision Point (PDP) integrated with OpenStack Keystone
3. **Objective 3**: DQN-based dynamic rule adaptation (MDP formulation)
4. **Objective 4 ★**: Federated Learning (Flower) for privacy-preserving training
5. **Objective 5 ★**: Hyperledger Fabric blockchain for tamper-proof audit trail

---

## 🧪 Testing

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires Docker)
pytest tests/integration/ -v --timeout=120

# Full test suite with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 🐳 Docker & Kubernetes

```bash
# Start local dev stack
docker-compose up -d

# Deploy to Kubernetes
kubectl apply -f k8s/

# Check pods
kubectl get pods -n dr-tbac-zt
```

---

## 📋 Compliance

This framework is designed to align with:
- **NIST Zero Trust Architecture (SP 800-207)**
- **ISO/IEC 27001** Information Security Management
- **GDPR** Data minimization via Federated Learning
- **SOC 2 Type II** Audit trail via Blockchain

See [docs/compliance_report_template.md](docs/compliance_report_template.md) for the full compliance report.

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.

---

## 📧 Contact

For enterprise deployment inquiries or research collaboration, please open a GitHub Issue or contact the maintainers.
