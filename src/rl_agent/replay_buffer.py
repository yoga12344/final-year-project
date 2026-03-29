"""
Replay Buffer implementations for the DQN agent.

Provides:
- ``ReplayBuffer``          – standard uniform Experience Replay
- ``PrioritizedReplayBuffer`` – Prioritised Experience Replay (PER)
"""

import random
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Transition container
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool


# ---------------------------------------------------------------------------
# Uniform Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-size circular buffer storing (s, a, r, s', done) transitions.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Add a transition to the buffer (oldest entry is evicted if full)."""
        self.buffer.append(
            Transition(
                state=np.array(state,      dtype=np.float32),
                action=action,
                reward=reward,
                next_state=np.array(next_state, dtype=np.float32),
                done=float(done),
            )
        )

    # ------------------------------------------------------------------
    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Uniformly sample a mini-batch.

        Returns
        -------
        A tuple (states, actions, rewards, next_states, dones) where each
        element is a numpy array of shape (batch_size, …).
        """
        batch = random.sample(self.buffer, batch_size)
        states      = np.stack([t.state      for t in batch])
        actions     = np.array([t.action     for t in batch], dtype=np.int64)
        rewards     = np.array([t.reward     for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones       = np.array([t.done       for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """True when the buffer has at least ``batch_size`` transitions."""
        return len(self) >= batch_size


# ---------------------------------------------------------------------------
# Prioritised Experience Replay (PER)
# ---------------------------------------------------------------------------

class SumTree:
    """Binary sum-tree for O(log n) priority sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree   = np.zeros(2 * capacity, dtype=np.float64)
        self.data   = np.empty(capacity, dtype=object)
        self.write  = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        idx  = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritised Experience Replay buffer.

    Transitions with higher TD-error are sampled more frequently,
    corrected via importance-sampling weights.

    Parameters
    ----------
    capacity : int   Maximum buffer size.
    alpha    : float Priority exponent (0 = uniform, 1 = full priority).
    beta     : float IS-weight exponent (annealed towards 1 during training).
    beta_increment : float  Annealing step per sample call.
    """

    def __init__(
        self,
        capacity:       int   = 100_000,
        alpha:          float = 0.6,
        beta:           float = 0.4,
        beta_increment: float = 1e-4,
        eps:            float = 1e-5,
    ):
        self.tree   = SumTree(capacity)
        self.alpha  = alpha
        self.beta   = beta
        self.beta_increment = beta_increment
        self.eps    = eps
        self.max_priority = 1.0

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        transition = Transition(
            state=np.array(state,      dtype=np.float32),
            action=action,
            reward=reward,
            next_state=np.array(next_state, dtype=np.float32),
            done=float(done),
        )
        self.tree.add(self.max_priority ** self.alpha, transition)

    def sample(self, batch_size: int):
        """
        Sample a prioritised mini-batch.

        Returns
        -------
        Tuple: (states, actions, rewards, next_states, dones, indices, is_weights)
        """
        batch, indices, priorities = [], [], []
        segment = self.tree.total() / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # IS weights
        total    = self.tree.total()
        n        = self.tree.n_entries
        probs    = np.array(priorities) / total
        is_w     = (n * probs) ** (-self.beta)
        is_w    /= is_w.max()

        states      = np.stack([t.state      for t in batch])
        actions     = np.array([t.action     for t in batch], dtype=np.int64)
        rewards     = np.array([t.reward     for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones       = np.array([t.done       for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices, is_w.astype(np.float32)

    def update_priorities(self, indices: list[int], td_errors: np.ndarray) -> None:
        """Update priorities after a learning step."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.eps) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.tree.n_entries
