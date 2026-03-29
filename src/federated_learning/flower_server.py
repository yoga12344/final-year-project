"""
DR-TBAC-ZT++ | src/federated_learning/flower_server.py
Flower federated server with FedProx strategy and model aggregation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import flwr as fl
from flwr.common import Metrics, NDArrays, Parameters, Scalar
from flwr.server.strategy import FedAvg
from src.config import cfg
from src.federated_learning.strategy import FedProxStrategy
from src.trust_assessment.lstm_model import BiLSTMTrustModel
from src.federated_learning.flower_client import set_parameters
from src.utils.logger import get_logger

log = get_logger(__name__)


def weighted_average_accuracy(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate accuracy metrics weighted by number of samples."""
    total_samples = sum(n for n, _ in metrics)
    weighted_acc = sum(n * m.get("accuracy", 0) for n, m in metrics) / max(total_samples, 1)
    return {"weighted_accuracy": weighted_acc}


class TrustFederatedServer:
    """
    Orchestrates Flower federated training with FedProx strategy.
    Handles global model initialization, client configuration, and
    checkpoint saving at each round.
    """

    def __init__(self, save_dir: Optional[Path] = None):
        self.fc = cfg.federated
        self.save_dir = save_dir or cfg.lstm.model_save_path.parent
        self.global_model = BiLSTMTrustModel()

    def _get_initial_parameters(self) -> Parameters:
        ndarrays = [
            val.cpu().numpy() for _, val in self.global_model.state_dict().items()
        ]
        return fl.common.ndarrays_to_parameters(ndarrays)

    def _save_global_model(self, round_num: int, ndarrays: NDArrays):
        set_parameters(self.global_model, ndarrays)
        path = self.save_dir / f"lstm_global_round_{round_num:03d}.pth"
        torch.save(
            {
                "model_state_dict": self.global_model.state_dict(),
                "round": round_num,
            },
            path,
        )
        # Also overwrite the canonical best model
        torch.save(
            {"model_state_dict": self.global_model.state_dict(), "round": round_num},
            cfg.lstm.model_save_path,
        )
        log.info(f"[FL Round {round_num}] Global model saved to {path}")

    def build_strategy(self) -> fl.server.strategy.Strategy:
        initial_params = self._get_initial_parameters()

        strategy = FedProxStrategy(
            proximal_mu=self.fc.proximal_mu,
            fraction_fit=self.fc.fraction_fit,
            fraction_evaluate=self.fc.fraction_evaluate,
            min_fit_clients=self.fc.min_fit_clients,
            min_evaluate_clients=self.fc.min_evaluate_clients,
            min_available_clients=self.fc.min_available_clients,
            evaluate_metrics_aggregation_fn=weighted_average_accuracy,
            initial_parameters=initial_params,
            on_fit_config_fn=self._on_fit_config,
            on_evaluate_config_fn=self._on_evaluate_config,
            fit_metrics_aggregation_fn=weighted_average_accuracy,
        )
        return strategy

    @staticmethod
    def _on_fit_config(server_round: int) -> Dict[str, Scalar]:
        """Send per-round configuration to clients."""
        return {
            "server_round": server_round,
            "local_epochs": 2 if server_round < 5 else 3,
            "learning_rate": 1e-3 if server_round < 10 else 5e-4,
        }

    @staticmethod
    def _on_evaluate_config(server_round: int) -> Dict[str, Scalar]:
        return {"server_round": server_round}

    def start(self):
        log.info(
            f"Starting Flower server at {self.fc.server_address} | "
            f"{self.fc.num_rounds} rounds | "
            f"min_clients={self.fc.min_available_clients}"
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Hook to save model after each round
        strategy = self.build_strategy()
        original_aggregate_fit = strategy.aggregate_fit

        def patched_aggregate_fit(server_round, results, failures):
            aggregated = original_aggregate_fit(server_round, results, failures)
            if aggregated is not None:
                params, _ = aggregated
                ndarrays = fl.common.parameters_to_ndarrays(params)
                self._save_global_model(server_round, ndarrays)
            return aggregated

        strategy.aggregate_fit = patched_aggregate_fit

        history = fl.server.start_server(
            server_address=self.fc.server_address,
            config=fl.server.ServerConfig(num_rounds=self.fc.num_rounds),
            strategy=strategy,
        )
        log.info(f"FL training complete. History: {history}")
        return history
