"""
Unit tests for DQN Replay Buffer.
"""

import numpy as np
import pytest

from src.rl_agent.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


STATE_DIM = 16


def make_transition(val: float = 0.0):
    state      = np.full(STATE_DIM, val, dtype=np.float32)
    next_state = np.full(STATE_DIM, val + 0.1, dtype=np.float32)
    return state, 0, 1.0, next_state, False


class TestReplayBuffer:
    def test_push_and_length(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(*make_transition(i))
        assert len(buf) == 10

    def test_capacity_eviction(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push(*make_transition(i))
        assert len(buf) == 5

    def test_sample_shapes(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(*make_transition(i))
        states, actions, rewards, next_states, dones = buf.sample(16)
        assert states.shape      == (16, STATE_DIM)
        assert actions.shape     == (16,)
        assert rewards.shape     == (16,)
        assert next_states.shape == (16, STATE_DIM)
        assert dones.shape       == (16,)

    def test_is_ready(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(31):
            buf.push(*make_transition(i))
        assert buf.is_ready(32) is False
        buf.push(*make_transition(31))
        assert buf.is_ready(32) is True


class TestPrioritizedReplayBuffer:
    def test_push_and_sample(self):
        buf = PrioritizedReplayBuffer(capacity=200, alpha=0.6, beta=0.4)
        for i in range(100):
            buf.push(*make_transition(i))
        assert len(buf) == 100
        states, actions, rewards, next_states, dones, indices, is_w = buf.sample(16)
        assert states.shape == (16, STATE_DIM)
        assert len(indices) == 16
        assert is_w.shape == (16,)

    def test_priority_update(self):
        buf = PrioritizedReplayBuffer(capacity=200)
        for i in range(100):
            buf.push(*make_transition(i))
        _, _, _, _, _, indices, _ = buf.sample(16)
        td_errors = np.random.rand(16).astype(np.float32)
        buf.update_priorities(indices, td_errors)   # should not raise
