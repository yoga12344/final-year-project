"""
Unit tests for Metrics utilities.
"""

import numpy as np
import pytest

from src.utils.metrics import (
    compute_classification_metrics,
    EpisodeTracker,
    aggregate_fl_metrics,
    ThroughputMeter,
)


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y = [0, 1, 0, 1, 1]
        m = compute_classification_metrics(y, y)
        assert m["accuracy"]  == 1.0
        assert m["precision"] == 1.0
        assert m["recall"]    == 1.0
        assert m["f1"]        == 1.0
        assert m["fpr"]       == 0.0
        assert m["fnr"]       == 0.0

    def test_all_wrong(self):
        y_true = [1, 1, 1]
        y_pred = [0, 0, 0]
        m = compute_classification_metrics(y_true, y_pred)
        assert m["accuracy"]  == 0.0
        assert m["recall"]    == 0.0

    def test_confusion_matrix_shape(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]
        m = compute_classification_metrics(y_true, y_pred)
        assert len(m["confusion_matrix"])    == 2
        assert len(m["confusion_matrix"][0]) == 2


class TestEpisodeTracker:
    def test_record_and_summary(self):
        tracker = EpisodeTracker(window=10)
        for i in range(5):
            tracker.record(episode_return=float(i), episode_length=100, loss=0.01)
        s = tracker.summary()
        assert s["episode"] == 5
        assert s["mean_return"] == pytest.approx(2.0, abs=0.01)

    def test_window_overflow(self):
        tracker = EpisodeTracker(window=3)
        for i in range(10):
            tracker.record(float(i), 50)
        s = tracker.summary()
        assert s["episode"] == 10
        # Only last 3 episodes in window: 7, 8, 9
        assert s["mean_return"] == pytest.approx(8.0, abs=0.01)


class TestFLMetrics:
    def test_aggregate(self):
        clients = [
            {"num_examples": 100, "accuracy": 0.9, "loss": 0.1},
            {"num_examples": 200, "accuracy": 0.8, "loss": 0.2},
        ]
        agg = aggregate_fl_metrics(clients)
        assert agg["num_clients"] == 2
        assert agg["total_examples"] == 300
        # Weighted accuracy: (0.9*100 + 0.8*200)/300 = 250/300 ≈ 0.8333
        assert agg["global_accuracy"] == pytest.approx(0.8333, abs=0.001)

    def test_empty_clients(self):
        assert aggregate_fl_metrics([]) == {}


class TestThroughputMeter:
    def test_stats_keys(self):
        meter = ThroughputMeter()
        for _ in range(10):
            meter.record(0.01)
        s = meter.stats()
        assert "throughput_rps"  in s
        assert "mean_latency_ms" in s
        assert "p95_latency_ms"  in s
