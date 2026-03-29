"""
DQN Agent for Dynamic Access Control Rule Generation.

Implements a Deep Q-Network (DQN) with:
- Experience replay
- Target network (soft / hard updates)
- Epsilon-greedy exploration with decay
- Double-DQN support
"""

import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from .replay_buffer import ReplayBuffer
from src.config import cfg

# Used internally
MODEL_DIR = cfg.dqn.model_save_path.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# ---------------------------------------------------------------------------
# Neural Network Architecture
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Multi-layer fully-connected Q-network.

    Maps (state) → Q-values for every discrete action.
    """

    def __init__(
        self,
        state_dim: int = None,
        action_dim: int = None,
        hidden_dims: list = None,
    ):
        state_dim = state_dim or cfg.dqn.state_dim
        action_dim = action_dim or cfg.dqn.action_dim
        hidden_dims = hidden_dims or cfg.dqn.hidden_dims
        super().__init__()

        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Double-DQN agent for adaptive access-control policy updates.

    Actions correspond to policy rule adjustments defined in
    ``src/rl_agent/environment.py``.
    """

    def __init__(
        self,
        state_dim: int = None,
        action_dim: int = None,
        device: str = DEVICE,
    ):
        self.state_dim = state_dim or cfg.dqn.state_dim
        self.action_dim = action_dim or cfg.dqn.action_dim
        self.device = torch.device(device)

        # ---- networks -------------------------------------------------------
        self.online_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = copy.deepcopy(self.online_net).to(self.device)
        self.target_net.eval()

        # ---- optimiser / loss -----------------------------------------------
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=cfg.dqn.lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        # ---- replay buffer --------------------------------------------------
        self.buffer = ReplayBuffer(cfg.dqn.buffer_size)

        # ---- hyper-params ---------------------------------------------------
        self.gamma = cfg.dqn.gamma
        self.batch_size = cfg.dqn.batch_size
        self.tau = 0.005 # Default tau for DQN soft updates
        self.target_update_freq = cfg.dqn.target_update_freq

        # ---- exploration ----------------------------------------------------
        self.epsilon = cfg.dqn.epsilon_start
        self.eps_end = cfg.dqn.epsilon_end
        self.eps_decay = 0.995 # Epsilon decay factor


        # ---- counters -------------------------------------------------------
        self.step_count = 0
        self.update_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        ε-greedy action selection.

        Args:
            state:     Current environment state vector (1-D numpy array).
            eval_mode: If True, always pick the greedy action.

        Returns:
            Integer action index.
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push one transition into the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
        self.step_count += 1
        self._decay_epsilon()

    def update(self) -> float | None:
        """
        Sample a mini-batch and perform one gradient step.

        Returns:
            Loss value (float) or None if buffer is not yet full enough.
        """
        if len(self.buffer) < self.batch_size:
            return None

        # ---- sample ---------------------------------------------------------
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ---- current Q-values -----------------------------------------------
        current_q = self.online_net(states_t).gather(1, actions_t)

        # ---- Double-DQN target ----------------------------------------------
        with torch.no_grad():
            # Action selected by online net
            best_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
            # Value estimated by target net
            next_q = self.target_net(next_states_t).gather(1, best_actions)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        # ---- loss & update --------------------------------------------------
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.update_count += 1

        # ---- target network update ------------------------------------------
        if self.update_count % self.target_update_freq == 0:
            self._soft_update_target()

        return loss.item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> str:
        """Save model weights. Returns the file path used."""
        if path is None:
            Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
            path = os.path.join(MODEL_DIR, "dqn_best.pt")
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "epsilon":    self.epsilon,
                "step_count": self.step_count,
            },
            path,
        )
        return path

    def load(self, path: str) -> None:
        """Load model weights from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon    = checkpoint.get("epsilon",    self.eps_end)
        self.step_count = checkpoint.get("step_count", 0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decay_epsilon(self) -> None:
        """Exponential epsilon decay."""
        self.epsilon = max(
            self.eps_end, self.epsilon * self.eps_decay
        )

    def _soft_update_target(self) -> None:
        """Polyak-average: θ_target ← τ·θ_online + (1-τ)·θ_target."""
        for t_param, o_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            t_param.data.copy_(
                self.tau * o_param.data + (1.0 - self.tau) * t_param.data
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return a dict of current training statistics."""
        return {
            "epsilon":      round(self.epsilon, 4),
            "buffer_size":  len(self.buffer),
            "step_count":   self.step_count,
            "update_count": self.update_count,
        }
