"""Utils package – logging, metrics, and visualisation."""
from .logger import setup_logging, get_logger
from .metrics import (
    compute_classification_metrics,
    EpisodeTracker,
    aggregate_fl_metrics,
    ThroughputMeter,
)
from .visualizer import (
    plot_trust_scores,
    plot_rl_training,
    plot_fl_rounds,
    plot_decision_distribution,
    plot_roc_curve,
)

__all__ = [
    "setup_logging", "get_logger",
    "compute_classification_metrics", "EpisodeTracker",
    "aggregate_fl_metrics", "ThroughputMeter",
    "plot_trust_scores", "plot_rl_training", "plot_fl_rounds",
    "plot_decision_distribution", "plot_roc_curve",
]
