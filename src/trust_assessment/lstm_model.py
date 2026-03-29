"""
BiLSTM model for anomaly detection on OpenStack logs.
======================================================
Input  : (batch, window_size, feature_dim=55)
Output : (batch, 1) sigmoid probability of anomaly
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------

class BiLSTMTrustModel(nn.Module):
    """
    Bidirectional LSTM for sequence-level anomaly classification.

    Architecture:
        BiLSTM (2 layers)  →  Attention pooling  →  FC  →  Sigmoid
    """

    def __init__(
        self,
        input_dim: int    = 55,    # 7 numeric + 48 one-hot event dims
        hidden_dim: int   = 128,
        num_layers: int   = 2,
        dropout: float    = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.num_dirs      = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * self.num_dirs

        # Self-attention over time steps
        self.attn_fc = nn.Linear(lstm_out_dim, 1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, 1)  — raw (pre-sigmoid) scores
        """
        # lstm_out: (batch, seq_len, hidden_dim * num_dirs)
        lstm_out, _ = self.lstm(x)

        # Attention weights over time
        attn_scores = self.attn_fc(lstm_out)          # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
        context = (attn_weights * lstm_out).sum(dim=1)    # (B, hidden*dirs)

        logits = self.classifier(context)              # (B, 1)
        return logits


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_epoch(
    model: BiLSTMTrustModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.float().unsqueeze(1).to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(
    model: BiLSTMTrustModel,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.float().unsqueeze(1).to(device)
        logits  = model(X_batch)
        loss    = criterion(logits, y_batch)
        total_loss += loss.item()
        preds   = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == y_batch.long()).sum().item()
        total   += y_batch.size(0)
    return total_loss / len(loader), correct / total


def train_model(
    model: BiLSTMTrustModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int          = 50,
    batch_size: int      = 256,
    lr: float            = 1e-3,
    weight_decay: float  = 1e-4,
    patience: int        = 10,
    model_path: str | Path = "models/lstm_global.pth",
    device_str: str      = "auto",
) -> dict:
    """Train BiLSTMTrustModel and return training history."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "auto" else device_str
    )
    logger.info("Training on device: %s", device)
    model = model.to(device)

    # Dataset / loaders
    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train.astype(np.int64))
    Xv = torch.from_numpy(X_val)
    yv = torch.from_numpy(y_val.astype(np.int64))

    train_ds = torch.utils.data.TensorDataset(Xt, yt)
    val_ds   = torch.utils.data.TensorDataset(Xv, yv)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_ld   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Class imbalance: compute positive weight
    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    stopper   = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_ld, optimizer, criterion, device)
        vl_loss, vl_acc = eval_epoch(model, val_ld, criterion, device)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), model_path)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%3d | tr_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
                epoch, epochs, tr_loss, vl_loss, vl_acc,
            )

        if stopper.step(vl_loss):
            logger.info("Early stopping at epoch %d", epoch)
            break

    logger.info("Best val loss: %.4f  → saved to %s", best_val_loss, model_path)
    return history


class LSTMTrainer:
    """
    Lightweight wrapper around BiLSTMTrustModel for Federated Learning clients.
    """
    def __init__(self, model: nn.Module, device_str: str = "auto"):
        self.model = model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            if device_str == "auto" else device_str
        )
        self.model.to(self.device)
        # Default LR; overridden by Flower client config
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

    def fit(self, train_data: tuple[np.ndarray, np.ndarray], val_data: tuple[np.ndarray, np.ndarray]) -> dict:
        from src.config import cfg
        
        epochs = cfg.lstm.num_epochs
        batch_size = cfg.lstm.batch_size
        X_train, y_train = train_data
        X_val, y_val = val_data

        Xt = torch.from_numpy(X_train)
        yt = torch.from_numpy(y_train.astype(np.int64))
        Xv = torch.from_numpy(X_val)
        yv = torch.from_numpy(y_val.astype(np.int64))

        train_ds = torch.utils.data.TensorDataset(Xt, yt)
        val_ds   = torch.utils.data.TensorDataset(Xv, yv)
        
        train_ld = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
        
        pos_count = float(y_train.sum())
        neg_count = float(len(y_train) - pos_count)
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        
        for epoch in range(1, epochs + 1):
            tr_loss = train_epoch(self.model, train_ld, self.optimizer, criterion, self.device)
            vl_loss, vl_acc = self._evaluate(val_ds, criterion=criterion, batch_size=batch_size)

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)
            history["val_acc"].append(vl_acc)

        return history

    def _evaluate(self, loader_or_dataset, criterion=None, batch_size=256) -> tuple[float, float]:
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()
            
        if isinstance(loader_or_dataset, torch.utils.data.DataLoader):
            loader = loader_or_dataset
        else:
            loader = torch.utils.data.DataLoader(loader_or_dataset, batch_size=batch_size, shuffle=False)
            
        return eval_epoch(self.model, loader, criterion, self.device)
