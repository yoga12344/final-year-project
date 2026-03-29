"""
DR-TBAC-ZT++ | src/federated_learning/flower_client.py
Flower federated learning client — wraps the BiLSTM model for FL participation.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import flwr as fl
from flwr.common import NDArrays, Scalar

from src.config import cfg
from src.trust_assessment.lstm_model import BiLSTMTrustModel, LSTMTrainer
from src.utils.logger import get_logger

log = get_logger(__name__)


def get_parameters(model: torch.nn.Module) -> NDArrays:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: NDArrays):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


class TrustModelClient(fl.client.NumPyClient):
    """
    Flower NumPyClient that trains the BiLSTM trust model locally
    and participates in federated aggregation.

    Supports Differential Privacy via gradient clipping + Gaussian noise
    injection before sharing updates.
    """

    def __init__(
        self,
        client_id: str,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        dp_noise_multiplier: float = None,
        dp_max_grad_norm: float = None,
    ):
        self.client_id = client_id
        self.train_data = train_data
        self.val_data = val_data
        self.dp_noise = dp_noise_multiplier or cfg.federated.dp_noise_multiplier
        self.dp_clip = dp_max_grad_norm or cfg.federated.dp_max_grad_norm

        self.model = BiLSTMTrustModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.trainer = LSTMTrainer(self.model)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        log.debug(f"Client {self.client_id}: get_parameters()")
        return get_parameters(self.model)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Local training on client data."""
        set_parameters(self.model, parameters)
        local_epochs = int(config.get("local_epochs", 2))
        learning_rate = float(config.get("learning_rate", cfg.lstm.learning_rate))

        # Temporarily override epochs for FL round
        original_epochs = cfg.lstm.num_epochs
        cfg.lstm.num_epochs = local_epochs
        self.trainer.optimizer.param_groups[0]["lr"] = learning_rate

        history = self.trainer.fit(self.train_data, self.val_data)
        cfg.lstm.num_epochs = original_epochs

        # Differential privacy: add Gaussian noise to gradients
        noisy_params = self._apply_dp_noise(get_parameters(self.model))

        n_samples = len(self.train_data[0])
        metrics: Dict[str, Scalar] = {
            "client_id": self.client_id,
            "train_loss": float(history["train_loss"][-1]),
            "val_loss": float(history["val_loss"][-1]),
            "val_acc": float(history["val_acc"][-1]),
        }
        log.info(
            f"Client {self.client_id} | Fit: n={n_samples} "
            f"val_acc={metrics['val_acc']:.4f}"
        )
        return noisy_params, n_samples, metrics

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Server-side evaluation dispatched to client."""
        set_parameters(self.model, parameters)
        from torch.utils.data import DataLoader, TensorDataset

        X_val, y_val = self.val_data
        loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long),
            ),
            batch_size=cfg.lstm.batch_size,
            shuffle=False,
        )
        val_loss, val_acc = self.trainer._evaluate(loader)
        n = len(X_val)
        log.info(f"Client {self.client_id} | Eval: loss={val_loss:.4f} acc={val_acc:.4f}")
        return float(val_loss), n, {"accuracy": float(val_acc), "client_id": self.client_id}

    def _apply_dp_noise(self, params: NDArrays) -> NDArrays:
        """Add calibrated Gaussian noise for differential privacy."""
        if self.dp_noise <= 0:
            return params
        noisy = []
        for p in params:
            # Clip
            norm = np.linalg.norm(p)
            if norm > self.dp_clip:
                p = p * (self.dp_clip / norm)
            # Add noise
            noise = np.random.normal(0, self.dp_noise * self.dp_clip, p.shape)
            noisy.append((p + noise).astype(p.dtype))
        return noisy
