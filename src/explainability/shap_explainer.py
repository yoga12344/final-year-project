"""
SHAP Explainability for DR-TBAC-ZT++
======================================
Provides human-readable explanations for every Bi-LSTM access decision.

Uses SHAP DeepExplainer (for PyTorch) on the BiLSTMTrustModel.
Falls back to KernelExplainer (model-agnostic) if needed.

Feature names (55 total):
  [0] http_method       [1] http_status       [2] response_time
  [3] resp_len          [4] source_ip_oct3     [5] source_ip_oct4
  [6] has_instance      [7..54] EventId one-hot (E1..E48)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import shap

from src.trust_assessment.lstm_model import BiLSTMTrustModel
from src.trust_assessment.preprocessor import ALL_EVENT_IDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

NUMERIC_FEATURE_NAMES = [
    "http_method",
    "http_status",
    "response_time",
    "resp_len",
    "source_ip_oct3",
    "source_ip_oct4",
    "has_instance",
]

EVENT_FEATURE_NAMES = [f"event_{eid}" for eid in ALL_EVENT_IDS]   # 48

ALL_FEATURE_NAMES = NUMERIC_FEATURE_NAMES + EVENT_FEATURE_NAMES   # 55


# ---------------------------------------------------------------------------
# Wrapper to expose LSTM as a callable for SHAP
# ---------------------------------------------------------------------------

class _LSTMWrapper:
    """
    Wraps BiLSTMTrustModel to accept a 2-D numpy array
    (num_samples, window_size * feature_dim) and return
    anomaly probabilities — required by SHAP KernelExplainer.
    """
    def __init__(self, model: BiLSTMTrustModel, window_size: int, feature_dim: int):
        self.model       = model
        self.window_size = window_size
        self.feature_dim = feature_dim

    def __call__(self, X_flat: np.ndarray) -> np.ndarray:
        """
        Args:
            X_flat: (N, window_size * feature_dim)
        Returns:
            probs: (N, 2) — columns [prob_normal, prob_anomaly]
        """
        N = len(X_flat)
        X = X_flat.reshape(N, self.window_size, self.feature_dim).astype(np.float32)
        tensor = torch.from_numpy(X)
        with torch.no_grad():
            logits = self.model(tensor).squeeze(1)
            p_anomaly = torch.sigmoid(logits).numpy()
        p_normal = 1.0 - p_anomaly
        return np.stack([p_normal, p_anomaly], axis=1)


# ---------------------------------------------------------------------------
# Main explainer class
# ---------------------------------------------------------------------------

class SHAPExplainer:
    """
    Computes SHAP values for Bi-LSTM trust decisions.

    Usage:
        explainer = SHAPExplainer(model, background_X)
        result = explainer.explain(sequence)
        print(result["top_features"])
    """

    def __init__(
        self,
        model: BiLSTMTrustModel,
        background_X: np.ndarray,              # (n_bg, window, feature_dim)
        window_size: int = 20,
        feature_dim: int = 55,
        n_background: int = 100,
        use_kernel: bool = True,
    ):
        self.model       = model.eval()
        self.window_size = window_size
        self.feature_dim = feature_dim

        # Subsample background for speed
        idx = np.random.choice(len(background_X), min(n_background, len(background_X)), replace=False)
        bg   = background_X[idx]                         # (n_bg, W, F)
        bg_flat = bg.reshape(len(bg), -1)                # (n_bg, W*F) for KernelExplainer

        self._wrapper = _LSTMWrapper(model, window_size, feature_dim)

        logger.info("Fitting SHAP KernelExplainer on %d background samples …", len(bg_flat))
        # Use KernelExplainer (model-agnostic, works with any PyTorch model)
        self.explainer  = shap.KernelExplainer(self._wrapper, bg_flat, link="logit")
        logger.info("SHAP explainer ready.")

        # Feature names for display: we explain per-timestep averages
        self.feature_names = ALL_FEATURE_NAMES   # 55 names

    # ------------------------------------------------------------------
    def explain(
        self,
        sequence: np.ndarray,          # (window_size, feature_dim)
        nsamples: int = 200,
    ) -> dict:
        """
        Compute SHAP values for a single sequence.

        Returns:
            {
                "shap_values":    np.ndarray (feature_dim,) — mean |SHAP| across time
                "feature_names":  list[str]
                "top_features":   list[dict] — top 10 features by |SHAP| importance
                "anomaly_prob":   float
                "trust_score":    float
                "verdict":        "ANOMALY" | "NORMAL"
            }
        """
        assert sequence.shape == (self.window_size, self.feature_dim), \
            f"Expected shape ({self.window_size}, {self.feature_dim}), got {sequence.shape}"

        seq_flat = sequence.reshape(1, -1)   # (1, W*F)

        # SHAP values: shape (n_classes=2, 1, W*F)
        raw_shap = self.explainer.shap_values(seq_flat, nsamples=nsamples, silent=True)

        # Take anomaly class (index 1), shape (1, W*F)
        shap_flat = raw_shap[1][0]                              # (W*F,)
        shap_3d   = shap_flat.reshape(self.window_size, self.feature_dim)    # (W, F)
        # Mean absolute SHAP per feature across time
        shap_per_feature = shap_3d.mean(axis=0)                 # (F,)
        abs_shap         = np.abs(shap_per_feature)

        # Anomaly probability
        prob  = self._wrapper(seq_flat)[0, 1]
        trust = float(1.0 - prob)

        # Top-10 features
        top_idx = np.argsort(abs_shap)[::-1][:10]
        top_features = [
            {
                "rank":         int(r + 1),
                "feature":      self.feature_names[i],
                "shap_value":   float(shap_per_feature[i]),
                "abs_shap":     float(abs_shap[i]),
                "direction":    "↑ anomaly" if shap_per_feature[i] > 0 else "↓ anomaly",
            }
            for r, i in enumerate(top_idx)
        ]

        return {
            "shap_values":   shap_per_feature,
            "abs_shap":      abs_shap,
            "feature_names": self.feature_names,
            "top_features":  top_features,
            "anomaly_prob":  float(prob),
            "trust_score":   round(trust, 4),
            "verdict":       "ANOMALY" if prob >= 0.5 else "NORMAL",
        }

    # ------------------------------------------------------------------
    def explain_text(self, sequence: np.ndarray, nsamples: int = 200) -> str:
        """Return a human-readable text summary of the explanation."""
        result = self.explain(sequence, nsamples=nsamples)
        lines = [
            f"Trust Score  : {result['trust_score']:.4f}",
            f"Anomaly Prob : {result['anomaly_prob']:.4f}",
            f"Verdict      : {result['verdict']}",
            "",
            "Top Contributing Features:",
        ]
        for feat in result["top_features"]:
            lines.append(
                f"  {feat['rank']:2d}. {feat['feature']:<25s} "
                f"SHAP={feat['shap_value']:+.4f}  ({feat['direction']})"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    @staticmethod
    def save_bar_plot(result: dict, output_path: str = "experiments/results/shap_bar.png"):
        """Save a horizontal bar chart of top SHAP values."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top  = result["top_features"][:10]
        names  = [f["feature"] for f in top]
        values = [f["shap_value"] for f in top]
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(names[::-1], values[::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on anomaly probability)")
        ax.set_title(
            f"SHAP Explanation — {result['verdict']}  "
            f"(trust={result['trust_score']:.3f})"
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=120)
        plt.close()
        logger.info("Saved SHAP bar plot → %s", output_path)
        return output_path
