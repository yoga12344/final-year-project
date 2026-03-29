"""
DR-TBAC-ZT++ | src/rl_agent/environment.py
MDP environment for DQN policy adaptation.
State: behavioral + trust features. Actions: ALLOW / DENY / CHALLENGE / THROTTLE.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


class AccessAction(IntEnum):
    ALLOW = 0
    DENY = 1
    CHALLENGE = 2   # MFA challenge
    THROTTLE = 3    # Rate-limit the session


@dataclass
class AccessRequest:
    user_id: str
    trust_index: float          # 0.0 – 1.0
    trust_level: str            # LOW / MEDIUM / HIGH
    resource_sensitivity: float # 0.0 – 1.0
    privilege_level: float      # 0.0 – 1.0
    geo_anomaly_score: float    # 0.0 – 1.0
    burst_factor: float         # 0.0 – 1.0
    off_hours_access: float     # 0 or 1
    concurrent_sessions: int
    prev_violations: int
    mfa_enabled: float          # 0 or 1
    lateral_movement_score: float
    data_exfil_indicator: float
    context_trust_delta: float  # Change in trust since last window
    env_risk_level: float       # 0..1

    def to_state_vector(self) -> np.ndarray:
        return np.array([
            self.trust_index,
            float(self.trust_level == "HIGH"),
            float(self.trust_level == "MEDIUM"),
            float(self.trust_level == "LOW"),
            self.resource_sensitivity,
            self.privilege_level,
            self.geo_anomaly_score,
            self.burst_factor,
            self.off_hours_access,
            min(self.concurrent_sessions / 10.0, 1.0),
            min(self.prev_violations / 5.0, 1.0),
            self.mfa_enabled,
            self.lateral_movement_score,
            self.data_exfil_indicator,
            self.context_trust_delta,
            self.env_risk_level,
        ], dtype=np.float32)


class TrustAccessEnvironment(gym.Env):
    """
    OpenAI Gymnasium-compatible environment for the DQN rule agent.

    Observation space:  16-dimensional state vector (AccessRequest features)
    Action space:       Discrete(4) — ALLOW, DENY, CHALLENGE, THROTTLE

    Reward:
        +1.5  ALLOW   for HIGH trust user → correct
        +0.5  CHALLENGE for MEDIUM trust → step-up auth encouraged
        -2.0  ALLOW   for LOW trust user → security violation
        -1.0  DENY    for HIGH trust user → usability penalty
        +1.0  DENY    for LOW trust → correct rejection
        -0.1  THROTTLE → mild penalty (latency cost)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, request_generator=None):
        super().__init__()
        dc = cfg.dqn
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(dc.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(dc.action_dim)
        self.request_generator = request_generator or self._default_generator
        self._current_request: Optional[AccessRequest] = None
        self._step_count = 0
        self._episode_rewards: list = []

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._current_request = self.request_generator()
        self._step_count = 0
        return self._current_request.to_state_vector(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self._current_request is not None
        req = self._current_request
        reward = self._compute_reward(AccessAction(action), req)
        self._episode_rewards.append(reward)

        # Generate next request
        self._current_request = self.request_generator()
        obs = self._current_request.to_state_vector()
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= 1000   # Episode horizon

        info = {
            "action": AccessAction(action).name,
            "trust_level": req.trust_level,
            "trust_index": req.trust_index,
            "reward": reward,
        }
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _compute_reward(action: AccessAction, req: AccessRequest) -> float:
        rw = cfg.dqn.reward_security
        ru = cfg.dqn.reward_usability

        if action == AccessAction.ALLOW:
            if req.trust_index >= cfg.openstack.trust_threshold_high:
                return +1.5 * ru
            elif req.trust_index >= cfg.openstack.trust_threshold_medium:
                return +0.2 * ru
            else:
                return -2.0 * rw   # Critical: allowing LOW trust user

        elif action == AccessAction.DENY:
            if req.trust_index < cfg.openstack.trust_threshold_medium:
                return +1.0 * rw   # Correct rejection
            elif req.trust_index >= cfg.openstack.trust_threshold_high:
                return -1.0 * ru   # Denying trustworthy user
            else:
                return +0.3 * rw

        elif action == AccessAction.CHALLENGE:
            # Encourage step-up auth for medium trust
            if cfg.openstack.trust_threshold_medium <= req.trust_index < cfg.openstack.trust_threshold_high:
                return +1.0 * rw
            elif not req.mfa_enabled:
                return +0.5 * rw
            else:
                return -0.2 * ru   # Annoying challenge for high-trust user

        elif action == AccessAction.THROTTLE:
            if req.burst_factor > 0.7 or req.data_exfil_indicator > 0.5:
                return +0.8 * rw
            return -0.1 * cfg.dqn.reward_efficiency

        return 0.0

    @staticmethod
    def _default_generator() -> AccessRequest:
        """Generate a synthetic access request for training."""
        rng = np.random.default_rng()
        trust_index = float(rng.beta(3, 2))  # skewed towards high trust
        if trust_index > 0.75:
            trust_level = "HIGH"
        elif trust_index > 0.45:
            trust_level = "MEDIUM"
        else:
            trust_level = "LOW"

        return AccessRequest(
            user_id=f"user_{rng.integers(1, 100)}",
            trust_index=trust_index,
            trust_level=trust_level,
            resource_sensitivity=float(rng.beta(2, 5)),
            privilege_level=float(rng.beta(2, 4)),
            geo_anomaly_score=float(rng.beta(1, 9)),
            burst_factor=float(rng.beta(1, 8)),
            off_hours_access=float(rng.integers(0, 2)),
            concurrent_sessions=int(rng.integers(1, 6)),
            prev_violations=int(rng.poisson(0.3)),
            mfa_enabled=float(rng.integers(0, 2)),
            lateral_movement_score=float(rng.beta(1, 8)),
            data_exfil_indicator=float(rng.beta(1, 20)),
            context_trust_delta=float(rng.uniform(-0.2, 0.2)),
            env_risk_level=float(rng.beta(2, 6)),
        )

    def render(self):
        if self._current_request:
            req = self._current_request
            log.info(
                f"[ENV] user={req.user_id} trust={req.trust_level}({req.trust_index:.2f}) "
                f"resource_sens={req.resource_sensitivity:.2f}"
            )
