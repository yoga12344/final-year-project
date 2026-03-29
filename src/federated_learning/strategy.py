"""
DR-TBAC-ZT++ | src/federated_learning/strategy.py
Custom FedProx aggregation strategy extending Flower's FedAvg.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes, NDArrays, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from src.utils.logger import get_logger

log = get_logger(__name__)


class FedProxStrategy(fl.server.strategy.FedAvg):
    """
    FedProx with proximal term — improves convergence on heterogeneous (non-IID)
    client data by adding a proximity regularizer during local client training.

    The server-side aggregation remains weighted FedAvg; the proximal term
    is communicated to clients via fit config.
    """

    def __init__(self, proximal_mu: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.proximal_mu = proximal_mu
        log.info(f"FedProxStrategy initialized with mu={proximal_mu}")

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """Add proximal_mu to client fit config."""
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        fit_ins = fl.common.FitIns(
            parameters,
            {
                "proximal_mu": self.proximal_mu,
                "server_round": server_round,
                "local_epochs": 2 if server_round < 5 else 3,
                "learning_rate": max(1e-4, 1e-3 * (0.95 ** server_round)),
            },
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        if failures:
            log.warning(f"Round {server_round}: {len(failures)} client(s) failed")

        # Weighted average by number of training samples
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        weighted_params: List[NDArrays] = []

        for _, fit_res in results:
            weight = fit_res.num_examples / total_examples
            client_params = parameters_to_ndarrays(fit_res.parameters)
            weighted_params.append([w * weight for w in client_params])

        aggregated = [
            sum(layer_params[i] for layer_params in weighted_params)
            for i in range(len(weighted_params[0]))
        ]

        # Aggregate metrics
        metrics_agg: Dict[str, Scalar] = {}
        aggregated_metrics_raw = [
            (fit_res.num_examples, fit_res.metrics) for _, fit_res in results
        ]
        if self.fit_metrics_aggregation_fn:
            metrics_agg = self.fit_metrics_aggregation_fn(aggregated_metrics_raw)

        val_accs = [
            fit_res.metrics.get("val_acc", 0) for _, fit_res in results
        ]
        log.info(
            f"Round {server_round} | Aggregated {len(results)} clients | "
            f"mean_val_acc={np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}"
        )

        return ndarrays_to_parameters(aggregated), metrics_agg
