"""
DR-TBAC-ZT++ | src/federated_learning/dataset_splitter.py
Splits the preprocessed dataset across simulated FL clients (IID and non-IID).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

ClientData = Tuple[np.ndarray, np.ndarray]


class DatasetSplitter:
    """
    Creates per-client data partitions for federated simulation.
    Supports both IID and Dirichlet-based non-IID splits.
    """

    def __init__(self, num_clients: int = None, iid: bool = False, alpha: float = 0.5):
        """
        Args:
            num_clients: number of FL clients.
            iid:         True → equal random splits; False → Dirichlet non-IID.
            alpha:       Dirichlet concentration (lower = more skewed).
        """
        self.num_clients = num_clients or cfg.federated.num_clients
        self.iid = iid
        self.alpha = alpha

    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[int, ClientData]:
        if self.iid:
            return self._iid_split(X, y)
        return self._dirichlet_split(X, y)

    def _iid_split(self, X: np.ndarray, y: np.ndarray) -> Dict[int, ClientData]:
        idx = np.random.permutation(len(X))
        chunks = np.array_split(idx, self.num_clients)
        splits = {
            cid: (X[chunk], y[chunk]) for cid, chunk in enumerate(chunks)
        }
        self._log_stats(splits)
        return splits

    def _dirichlet_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[int, ClientData]:
        classes = np.unique(y)
        client_indices: Dict[int, List[int]] = {c: [] for c in range(self.num_clients)}

        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            proportions = (proportions * len(cls_idx)).astype(int)
            # Fix rounding
            proportions[-1] += len(cls_idx) - proportions.sum()
            np.random.shuffle(cls_idx)
            start = 0
            for cid, n in enumerate(proportions):
                client_indices[cid].extend(cls_idx[start : start + n].tolist())
                start += n

        splits = {}
        for cid, idx_list in client_indices.items():
            idx_arr = np.array(idx_list)
            np.random.shuffle(idx_arr)
            splits[cid] = (X[idx_arr], y[idx_arr])

        self._log_stats(splits)
        return splits

    def _log_stats(self, splits: Dict[int, ClientData]):
        for cid, (X_, y_) in splits.items():
            class_dist = {int(c): int((y_ == c).sum()) for c in np.unique(y_)}
            log.info(f"Client {cid}: {len(X_)} samples | class_dist={class_dist}")

    def save_splits(self, splits: Dict[int, ClientData], output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        for cid, (X_, y_) in splits.items():
            np.save(output_dir / f"client_{cid}_X.npy", X_)
            np.save(output_dir / f"client_{cid}_y.npy", y_)
        log.info(f"Saved {len(splits)} client splits to {output_dir}")

    @staticmethod
    def load_client_split(data_dir: Path, client_id: int) -> ClientData:
        X = np.load(data_dir / f"client_{client_id}_X.npy")
        y = np.load(data_dir / f"client_{client_id}_y.npy")
        log.info(f"Loaded client {client_id} data: X={X.shape} y={y.shape}")
        return X, y
