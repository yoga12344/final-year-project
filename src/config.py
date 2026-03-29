"""
DR-TBAC-ZT++ | src/config.py
All hyperparameters, paths, and environment-resolved configuration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

# ── Project Root ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# LSTM / Trust Assessment
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LSTMConfig:
    input_size: int = 32          # Number of behavioral features
    hidden_size: int = 128        # LSTM hidden units
    num_layers: int = 2           # Stacked LSTM layers
    dropout: float = 0.3
    bidirectional: bool = True    # Bi-LSTM
    seq_len: int = 20             # Time-window (log lines)
    num_classes: int = 3          # LOW / MEDIUM / HIGH trust
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 50
    early_stopping_patience: int = 7
    weight_decay: float = 1e-4
    model_save_path: Path = ROOT_DIR / "models" / "lstm_global.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Federated Learning (Flower)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FederatedConfig:
    server_address: str = os.getenv("FLOWER_SERVER_ADDR", "0.0.0.0:9091")
    num_rounds: int = int(os.getenv("FLOWER_NUM_ROUNDS", "20"))
    min_available_clients: int = int(os.getenv("FLOWER_MIN_CLIENTS", "2"))
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    # FedProx proximal term (0.0 = FedAvg)
    proximal_mu: float = 0.1
    dp_noise_multiplier: float = 0.1   # Differential privacy noise
    dp_max_grad_norm: float = 1.0
    num_clients: int = 5
    data_dir: Path = ROOT_DIR / "data" / "processed"


# ─────────────────────────────────────────────────────────────────────────────
# Reinforcement Learning (DQN)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DQNConfig:
    state_dim: int = 16           # State vector size (trust + context features)
    action_dim: int = 4           # ALLOW / DENY / CHALLENGE / THROTTLE
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    lr: float = 3e-4
    gamma: float = 0.99           # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10_000   # Steps to decay epsilon
    buffer_size: int = 50_000
    batch_size: int = 128
    target_update_freq: int = 500  # Steps between target network updates
    train_freq: int = 4
    learning_starts: int = 1_000
    max_steps: int = 200_000
    model_save_path: Path = ROOT_DIR / "models" / "dqn_best.zip"
    # Reward shaping weights
    reward_security: float = 1.0
    reward_usability: float = 0.5
    reward_efficiency: float = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# OpenStack / Access Control
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class OpenStackConfig:
    auth_url: str = os.getenv("OS_AUTH_URL", "http://localhost:5000/v3")
    username: str = os.getenv("OS_USERNAME", "admin")
    password: str = os.getenv("OS_PASSWORD", "secret")
    project_name: str = os.getenv("OS_PROJECT_NAME", "admin")
    user_domain_id: str = os.getenv("OS_USER_DOMAIN_ID", "default")
    project_domain_id: str = os.getenv("OS_PROJECT_DOMAIN_ID", "default")
    policy_file: Path = ROOT_DIR / "src" / "access_control" / "policy.json"
    # Trust thresholds for access tiers
    trust_threshold_high: float = 0.75
    trust_threshold_medium: float = 0.45
    trust_threshold_low: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Kafka / Streaming
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class KafkaConfig:
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic_access_events: str = "access-events"
    topic_trust_scores: str = "trust-scores"
    topic_policy_updates: str = "policy-updates"
    topic_audit_logs: str = "audit-logs"
    consumer_group_id: str = "dr-tbac-group"
    auto_offset_reset: str = "earliest"


# ─────────────────────────────────────────────────────────────────────────────
# Blockchain / Hyperledger Fabric
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FabricConfig:
    network_profile: Path = (
        ROOT_DIR / "blockchain" / "fabric-network" / "connection-profile.json"
    )
    channel_name: str = "dr-tbac-channel"
    chaincode_name: str = "access-log"
    org_name: str = "Org1MSP"
    peer_endpoint: str = os.getenv("FABRIC_PEER_ENDPOINT", "grpc://localhost:7051")
    orderer_endpoint: str = os.getenv(
        "FABRIC_ORDERER_ENDPOINT", "grpc://localhost:7050"
    )
    tls_cert_path: Path = ROOT_DIR / "blockchain" / "fabric-network" / "tls" / "ca.crt"
    admin_cert_path: Path = (
        ROOT_DIR / "blockchain" / "fabric-network" / "admin" / "cert.pem"
    )
    admin_key_path: Path = (
        ROOT_DIR / "blockchain" / "fabric-network" / "admin" / "key.pem"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Logging & Paths
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LoggingConfig:
    level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: Path = ROOT_DIR / "logs"
    log_file: Path = ROOT_DIR / "logs" / "dr_tbac.log"
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5
    json_format: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Global Config Singleton
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    openstack: OpenStackConfig = field(default_factory=OpenStackConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    fabric: FabricConfig = field(default_factory=FabricConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Data paths
    raw_data_dir: Path = ROOT_DIR / "data" / "raw"
    processed_data_dir: Path = ROOT_DIR / "data" / "processed"
    sample_logs_dir: Path = ROOT_DIR / "data" / "sample_logs"
    results_dir: Path = ROOT_DIR / "experiments" / "results"

    # Feature columns used by the LSTM preprocessor
    behavioral_features: List[str] = field(
        default_factory=lambda: [
            "req_per_min", "failed_login_rate", "unique_resources_accessed",
            "avg_response_time_ms", "data_volume_mb", "hour_of_day",
            "day_of_week", "geo_anomaly_score", "ip_reputation_score",
            "mfa_enabled", "session_duration_min", "privilege_level",
            "resource_sensitivity", "api_call_diversity", "off_hours_access",
            "new_device", "vpn_usage", "abnormal_agent", "repeated_resource",
            "burst_factor", "lateral_movement_score", "data_exfil_indicator",
            "concurrent_sessions", "token_age_hours", "cert_validity_days",
            "country_risk_score", "prev_violation_count", "account_age_days",
            "role_match_score", "context_trust_delta", "env_risk_level",
            "compliance_score",
        ]
    )


# Module-level singleton
cfg = Config()
