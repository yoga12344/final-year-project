"""
Preprocessor for OpenStack_full.log_structured.csv
====================================================
Input columns: LineId, Content, EventId, EventTemplate

Features extracted per log line:
  - http_method   : 0=none, 1=GET, 2=POST, 3=DELETE
  - http_status   : normalized HTTP status code (0 if not an HTTP log)
  - response_time : response latency in seconds (0 if not present)
  - resp_len      : response length in bytes (0 if not present)
  - source_ip_oct3: 3rd octet of the source IP (proxy for zone/subnet)
  - source_ip_oct4: 4th octet
  - has_instance  : 1 if the log refers to a specific VM instance
  - event_onehot  : 49-dim one-hot of EventId (E1–E48 + unknown)
  - label         : 1 = anomaly, 0 = normal

Anomaly EventIds (based on dataset semantics):
  E48 → Error during <*>
  E24 → Bad response code while validating token
  E6  → Unable to validate token
  E7  → Instance sync did not match
  E1  → Couldn't obtain vcpu count (domain error)
"""

from __future__ import annotations

import re
import os
import logging
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANOMALY_EVENT_IDS = {"E48", "E24", "E6", "E7", "E1"}

# All 48 event IDs present in the OpenStack dataset
ALL_EVENT_IDS = [f"E{i}" for i in range(1, 49)]
EVENT_ID_INDEX = {eid: i for i, eid in enumerate(ALL_EVENT_IDS)}
N_EVENTS = len(ALL_EVENT_IDS)  # 48

# Regex patterns
_HTTP_PATTERN = re.compile(
    r'"(GET|POST|DELETE|PUT|PATCH)\s+\S+\s+HTTP/\S+"\s+'
    r'status:\s*(\d+)\s+len:\s*(\d+)\s+time:\s*(\S+)',
    re.IGNORECASE
)
_INSTANCE_PATTERN = re.compile(r'\[instance:\s*([0-9a-f\-]{36})\]', re.IGNORECASE)
_IP_PATTERN = re.compile(r'^(\d+)\.(\d+)\.(\d+)\.(\d+)')


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

_METHOD_MAP = {"GET": 1, "POST": 2, "DELETE": 3, "PUT": 4, "PATCH": 5}


def _parse_content(content: str) -> dict:
    """Extract structured numeric fields from a raw log Content string."""
    feats = {
        "http_method":    0,
        "http_status":    0.0,
        "response_time":  0.0,
        "resp_len":       0.0,
        "source_ip_oct3": 0.0,
        "source_ip_oct4": 0.0,
        "has_instance":   0,
    }

    # HTTP fields
    m = _HTTP_PATTERN.search(content)
    if m:
        feats["http_method"]   = _METHOD_MAP.get(m.group(1).upper(), 0)
        feats["http_status"]   = float(m.group(2))
        feats["resp_len"]      = float(m.group(3))
        feats["response_time"] = float(m.group(4))

    # Source IP (first IP found in the content)
    ip_m = _IP_PATTERN.match(content.strip())
    if ip_m:
        feats["source_ip_oct3"] = float(ip_m.group(3))
        feats["source_ip_oct4"] = float(ip_m.group(4))

    # Instance presence
    feats["has_instance"] = 1 if _INSTANCE_PATTERN.search(content) else 0

    return feats


def _event_onehot(event_id: str) -> np.ndarray:
    """Return a 48-dim one-hot vector for an EventId."""
    vec = np.zeros(N_EVENTS, dtype=np.float32)
    idx = EVENT_ID_INDEX.get(event_id.strip())
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _label(event_id: str) -> int:
    return 1 if event_id.strip() in ANOMALY_EVENT_IDS else 0


# ---------------------------------------------------------------------------
# Main preprocessing class
# ---------------------------------------------------------------------------

class OpenStackPreprocessor:
    """
    Reads OpenStack_full.log_structured.csv and produces train/val/test splits
    of (X, y) arrays suitable for LSTM training.

    Feature vector layout (per row):
        [7 numeric features] + [48 one-hot event features] = 55 dims total
    """

    FEATURE_DIM = 7 + N_EVENTS  # = 55

    def __init__(
        self,
        raw_csv: str | Path,
        output_dir: str | Path = "data/processed",
        window_size: int = 20,
        step_size: int = 1,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
        chunksize: int = 50_000,
    ):
        self.raw_csv     = Path(raw_csv)
        self.output_dir  = Path(output_dir)
        self.window_size = window_size
        self.step_size   = step_size
        self.test_size   = test_size
        self.val_size    = val_size
        self.random_state = random_state
        self.chunksize   = chunksize
        self.scaler: StandardScaler | None = None

    # ------------------------------------------------------------------
    def load_and_extract(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load CSV and extract feature matrix X and label vector y."""
        logger.info("Loading %s  (%.1f MB)", self.raw_csv, self.raw_csv.stat().st_size / 1e6)

        rows_X: List[np.ndarray] = []
        rows_y: List[int] = []

        for chunk in pd.read_csv(
            self.raw_csv,
            chunksize=self.chunksize,
            dtype={"LineId": int, "Content": str, "EventId": str, "EventTemplate": str},
            low_memory=False,
        ):
            chunk = chunk.dropna(subset=["Content", "EventId"])
            for _, row in chunk.iterrows():
                content  = str(row["Content"])
                event_id = str(row["EventId"]).strip()

                numeric = _parse_content(content)
                onehot  = _event_onehot(event_id)
                label   = _label(event_id)

                feat_vec = np.concatenate([
                    np.array([
                        numeric["http_method"],
                        numeric["http_status"],
                        numeric["response_time"],
                        numeric["resp_len"],
                        numeric["source_ip_oct3"],
                        numeric["source_ip_oct4"],
                        numeric["has_instance"],
                    ], dtype=np.float32),
                    onehot,
                ])
                rows_X.append(feat_vec)
                rows_y.append(label)

        X = np.stack(rows_X, axis=0)   # (N, 55)
        y = np.array(rows_y, dtype=np.int8)
        logger.info("Extracted %d rows | anomalies: %d (%.2f%%)",
                    len(y), y.sum(), 100 * y.mean())
        return X, y

    # ------------------------------------------------------------------
    def scale(self, X_train: np.ndarray) -> StandardScaler:
        """Fit scaler on training set numeric columns only (first 7 dims)."""
        self.scaler = StandardScaler()
        X_train[:, :7] = self.scaler.fit_transform(X_train[:, :7])
        return self.scaler

    def apply_scale(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Call .scale() first.")
        X = X.copy()
        X[:, :7] = self.scaler.transform(X[:, :7])
        return X

    # ------------------------------------------------------------------
    def make_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slide a window over (X, y) to create 3-D sequences for LSTM.
        Returns:
            X_seq : (num_windows, window_size, feature_dim)
            y_seq : (num_windows,)  — label of the last (most recent) step
        """
        seqs_X, seqs_y = [], []
        N = len(X)
        for start in range(0, N - self.window_size + 1, self.step_size):
            end = start + self.window_size
            seqs_X.append(X[start:end])             # (W, F)
            seqs_y.append(y[end - 1])               # label of last row
        return np.array(seqs_X, dtype=np.float32), np.array(seqs_y, dtype=np.int8)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Full pipeline: load → scale → split → save."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Extract raw features
        X, y = self.load_and_extract()

        # 2. Train / temp split first to avoid data leakage in scaling
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        relative_val = self.val_size / (1.0 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=relative_val,
            random_state=self.random_state,
            stratify=y_temp,
        )

        # 3. Scale (fit on train only)
        X_train[:, :7] = StandardScaler().fit_transform(X_train[:, :7])
        self.scaler = StandardScaler().fit(X_train[:, :7] if False else np.zeros((1, 7)))
        # Re-do properly:
        sc = StandardScaler()
        X_train[:, :7] = sc.fit_transform(X_train[:, :7])
        X_val[:, :7]   = sc.transform(X_val[:, :7])
        X_test[:, :7]  = sc.transform(X_test[:, :7])
        self.scaler = sc

        # 4. Build sliding-window sequences
        X_train_s, y_train_s = self.make_sequences(X_train, y_train)
        X_val_s,   y_val_s   = self.make_sequences(X_val,   y_val)
        X_test_s,  y_test_s  = self.make_sequences(X_test,  y_test)

        logger.info("Sequences → train: %s  val: %s  test: %s",
                    X_train_s.shape, X_val_s.shape, X_test_s.shape)

        # 5. Save
        np.save(self.output_dir / "X_train.npy", X_train_s)
        np.save(self.output_dir / "y_train.npy", y_train_s)
        np.save(self.output_dir / "X_val.npy",   X_val_s)
        np.save(self.output_dir / "y_val.npy",   y_val_s)
        np.save(self.output_dir / "X_test.npy",  X_test_s)
        np.save(self.output_dir / "y_test.npy",  y_test_s)
        joblib.dump(sc, self.output_dir / "scaler.pkl")

        logger.info("Saved processed data to %s", self.output_dir)
        stats = {
            "total_rows":       len(y),
            "anomaly_rate_pct": round(100 * float(y.mean()), 4),
            "feature_dim":      self.FEATURE_DIM,
            "window_size":      self.window_size,
            "train_sequences":  len(y_train_s),
            "val_sequences":    len(y_val_s),
            "test_sequences":   len(y_test_s),
        }
        return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Preprocess OpenStack log CSV for LSTM training")
    parser.add_argument(
        "--input", "-i",
        default="data/raw/OpenStack_full.log_structured.csv",
        help="Path to the structured CSV",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/processed",
        help="Directory to save .npy arrays and scaler",
    )
    parser.add_argument("--window", type=int, default=20, help="Sequence window size")
    parser.add_argument("--step",   type=int, default=5,  help="Sliding window step")
    args = parser.parse_args()

    prep = OpenStackPreprocessor(
        raw_csv=args.input,
        output_dir=args.output_dir,
        window_size=args.window,
        step_size=args.step,
    )
    stats = prep.run()
    print("\n=== Preprocessing Complete ===")
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")


if __name__ == "__main__":
    main()
