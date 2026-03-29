"""
Visualisation utilities for DR-TBAC-ZT++.

Generates publication-quality plots for:
- Trust score time series
- RL training curves (reward, loss, epsilon)
- Federated learning round accuracy
- Access decision distribution (PERMIT / DENY)
- ROC / PR curves (if sklearn installed)
"""

import os
from pathlib import Path
from typing import Sequence

import numpy as np

# Matplotlib with non-interactive Agg backend (safe for headless servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = os.getenv("RESULTS_DIR", "experiments/results")


def _savefig(fig: plt.Figure, filename: str, dpi: int = 150) -> str:
    """Save figure and return the absolute path."""
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Trust Score Visualisation
# ---------------------------------------------------------------------------

def plot_trust_scores(
    timestamps: Sequence,
    scores:     Sequence[float],
    threshold:  float = 0.5,
    title:      str   = "Trust Score Over Time",
    filename:   str   = "trust_scores.png",
) -> str:
    """
    Line chart of trust scores with a decision threshold line.

    Returns
    -------
    Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(timestamps, scores, linewidth=1.2, color="#4C72B0", label="Trust Score")
    ax.axhline(threshold, color="#DD4444", linestyle="--", linewidth=1, label=f"Threshold ({threshold})")
    ax.fill_between(timestamps, scores, threshold,
                    where=[s < threshold for s in scores],
                    alpha=0.2, color="#DD4444", label="Below Threshold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Trust Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, filename)


# ---------------------------------------------------------------------------
# RL Training Curves
# ---------------------------------------------------------------------------

def plot_rl_training(
    episodes:  Sequence[int],
    returns:   Sequence[float],
    losses:    Sequence[float] | None = None,
    epsilons:  Sequence[float] | None = None,
    filename:  str = "rl_training.png",
) -> str:
    """
    Multi-panel training dashboard for the DQN agent.

    Returns
    -------
    Path to the saved PNG.
    """
    n_panels = 1 + (losses is not None) + (epsilons is not None)
    fig = plt.figure(figsize=(12, 4 * n_panels))
    gs  = gridspec.GridSpec(n_panels, 1, hspace=0.45)
    axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]

    # Episode return
    ax = axes[0]
    ax.plot(episodes, returns, color="#4C72B0", linewidth=0.8, alpha=0.6, label="Return")
    window = min(50, len(returns))
    if window > 1:
        smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
        ax.plot(episodes[window - 1 :], smooth, color="#DD4444", linewidth=1.5, label=f"MA-{window}")
    ax.set_title("Episode Return", fontweight="bold")
    ax.set_ylabel("Return")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    panel = 1
    if losses is not None:
        ax = axes[panel]
        ax.plot(range(len(losses)), losses, color="#DD8800", linewidth=0.8)
        ax.set_title("Training Loss (Huber)", fontweight="bold")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        panel += 1

    if epsilons is not None:
        ax = axes[panel]
        ax.plot(range(len(epsilons)), epsilons, color="#229944", linewidth=1.0)
        ax.set_title("Epsilon (Exploration Rate)", fontweight="bold")
        ax.set_ylabel("ε")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Episode / Update Step")
    return _savefig(fig, filename)


# ---------------------------------------------------------------------------
# Federated Learning Accuracy
# ---------------------------------------------------------------------------

def plot_fl_rounds(
    rounds:        Sequence[int],
    global_acc:    Sequence[float],
    client_accs:   list[Sequence[float]] | None = None,
    filename:      str = "fl_accuracy.png",
) -> str:
    """
    Plot global and per-client accuracy across FL rounds.

    Returns
    -------
    Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    if client_accs:
        cmap = matplotlib.colormaps.get_cmap("tab10")
        for i, c_acc in enumerate(client_accs):
            ax.plot(rounds[:len(c_acc)], c_acc,
                    color=cmap(i % 10), linewidth=0.8, alpha=0.5,
                    label=f"Client {i+1}")

    ax.plot(rounds, global_acc, color="black", linewidth=2.0,
            linestyle="-", marker="o", markersize=4, label="Global")
    ax.set_title("Federated Learning – Round Accuracy", fontsize=13, fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=min(4, (len(client_accs) if client_accs else 0) + 1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, filename)


# ---------------------------------------------------------------------------
# Decision Distribution
# ---------------------------------------------------------------------------

def plot_decision_distribution(
    permit_count: int,
    deny_count:   int,
    title:        str = "Access Decision Distribution",
    filename:     str = "decision_distribution.png",
) -> str:
    """
    Pie + bar summary of PERMIT vs DENY decisions.

    Returns
    -------
    Path to the saved PNG.
    """
    total   = permit_count + deny_count or 1
    labels  = ["PERMIT", "DENY"]
    values  = [permit_count, deny_count]
    colours = ["#2ecc71", "#e74c3c"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        values, labels=labels, colors=colours,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontweight("bold")

    # Bar chart
    bars = ax2.bar(labels, values, color=colours, edgecolor="white", linewidth=1.5)
    ax2.set_ylabel("Count")
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + total * 0.01,
                 f"{val:,}", ha="center", va="bottom", fontsize=10)
    ax2.set_ylim(0, max(values) * 1.15)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return _savefig(fig, filename)


# ---------------------------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    filename: str = "roc_curve.png",
) -> str:
    """
    ROC curve with AUC annotation.  Requires scikit-learn.

    Returns
    -------
    Path to the saved PNG.
    """
    try:
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        raise RuntimeError("scikit-learn is required to plot ROC curves.")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, filename)
