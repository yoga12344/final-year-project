"""
Trust Calculator
================
Combines LSTM anomaly probability with context signals to produce a
trust score ∈ [0, 1] — higher means MORE trusted / less anomalous.

trust_score = 1 - weighted_anomaly_probability
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import joblib

from src.trust_assessment.lstm_model import BiLSTMTrustModel
from src.trust_assessment.preprocessor import (
    OpenStackPreprocessor,
    _parse_content,
    _event_onehot,
)

logger = logging.getLogger(__name__)

# Re-export for other modules
FEATURE_DIM = OpenStackPreprocessor.FEATURE_DIM   # 55


class TrustCalculator:
    """
    Online trust scorer.

    Usage:
        calc = TrustCalculator(model_path="models/lstm_global.pth",
                               scaler_path="data/processed/scaler.pkl")
        score = calc.score_event(content="10.11.10.1 ...", event_id="E42")
    """

    def __init__(
        self,
        model_path: str | Path = "models/lstm_global.pth",
        scaler_path: str | Path = "data/processed/scaler.pkl",
        window_size: int = 20,
        input_dim: int   = FEATURE_DIM,
        hidden_dim: int  = 128,
        device_str: str  = "auto",
    ):
        self.window_size = window_size
        self._buffer: list[np.ndarray] = []   # rolling window of feature vectors

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            if device_str == "auto" else device_str
        )

        # Load model
        self.model = BiLSTMTrustModel(input_dim=input_dim, hidden_dim=hidden_dim)
        model_path = Path(model_path)
        if model_path.exists():
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            logger.info("Loaded LSTM model from %s", model_path)
        else:
            logger.warning("Model file not found at %s – using random weights", model_path)
        self.model.eval().to(self.device)

        # Load scaler
        scaler_path = Path(scaler_path)
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info("Loaded scaler from %s", scaler_path)
        else:
            logger.warning("Scaler not found at %s – numeric features unscaled", scaler_path)
            self.scaler = None

    # ------------------------------------------------------------------
    def _build_feature(self, content: str, event_id: str) -> np.ndarray:
        """Build a single 55-dim feature vector for one log line."""
        numeric = _parse_content(content)
        onehot  = _event_onehot(event_id)
        raw_num = np.array([
            numeric["http_method"],
            numeric["http_status"],
            numeric["response_time"],
            numeric["resp_len"],
            numeric["source_ip_oct3"],
            numeric["source_ip_oct4"],
            numeric["has_instance"],
        ], dtype=np.float32)
        if self.scaler is not None:
            raw_num = self.scaler.transform(raw_num.reshape(1, -1)).flatten()
        return np.concatenate([raw_num, onehot]).astype(np.float32)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _infer(self, seq: np.ndarray) -> float:
        """Run LSTM inference on a (window_size, feature_dim) array."""
        x = torch.from_numpy(seq).unsqueeze(0).to(self.device)  # (1, W, F)
        logit = self.model(x)                                    # (1, 1)
        prob  = torch.sigmoid(logit).item()
        return float(prob)

    # ------------------------------------------------------------------
    def score_event(self, content: str, event_id: str) -> dict:
        """
        Process one log event and return trust information.

        Returns:
            {
                "trust_score": float,        # 0 = untrusted, 1 = fully trusted
                "anomaly_prob": float,       # raw LSTM anomaly probability
                "latency_ms": float,
                "buffer_ready": bool,        # False until window is full
            }
        """
        t0  = time.perf_counter()
        vec = self._build_feature(content, event_id)
        self._buffer.append(vec)

        if len(self._buffer) < self.window_size:
            return {
                "trust_score":  1.0,
                "anomaly_prob": 0.0,
                "latency_ms":   (time.perf_counter() - t0) * 1000,
                "buffer_ready": False,
            }

        # Keep only the latest window
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

        seq          = np.stack(self._buffer, axis=0)  # (W, F)
        anomaly_prob = self._infer(seq)
        trust_score  = max(0.0, 1.0 - anomaly_prob)

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "trust_score":  round(trust_score, 4),
            "anomaly_prob": round(anomaly_prob, 4),
            "latency_ms":   round(latency_ms, 2),
            "buffer_ready": True,
        }

    # ------------------------------------------------------------------
    def score_batch(self, contents: list[str], event_ids: list[str]) -> list[dict]:
        """Score a batch of log events sequentially (maintains state)."""
        return [
            self.score_event(c, e)
            for c, e in zip(contents, event_ids)
        ]

    # ------------------------------------------------------------------
    def reset_buffer(self) -> None:
        """Clear the rolling window (call when switching monitored entity)."""
        self._buffer.clear()
