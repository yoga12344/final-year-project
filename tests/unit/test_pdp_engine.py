"""
Unit tests for the PDP Engine.
"""

import pytest
import json
import tempfile
import os

from src.access_control.pdp_engine import PDPEngine, AccessRequest, AccessDecision


@pytest.fixture
def policy_file(tmp_path):
    """Create a temporary policy JSON file."""
    policy = {
        "version": "1.0",
        "rules": [
            {
                "rule_id": "R001",
                "effect": "PERMIT",
                "subjects": ["user:alice"],
                "resources": ["vm:*"],
                "actions": ["read"],
                "conditions": {"min_trust": 0.5},
            },
            {
                "rule_id": "R002",
                "effect": "DENY",
                "subjects": ["user:mallory"],
                "resources": ["*"],
                "actions": ["*"],
                "conditions": {},
            },
        ],
    }
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(policy))
    return str(p)


@pytest.fixture
def pdp(policy_file):
    return PDPEngine(policy_file=policy_file)


class TestAccessRequest:
    def test_request_has_uuid(self):
        req = AccessRequest("user:alice", "vm:db", "read", trust_score=0.8)
        assert len(req.request_id) == 36  # UUID4

    def test_to_dict_keys(self):
        req = AccessRequest("user:alice", "vm:db", "read", trust_score=0.7)
        d = req.to_dict()
        assert "request_id" in d
        assert "trust_score" in d


class TestPDPEngine:
    def test_permit_rule_match(self, pdp):
        req = AccessRequest("user:alice", "vm:prod", "read", trust_score=0.9)
        decision = pdp.evaluate(req)
        assert decision.decision == AccessDecision.PERMIT

    def test_deny_rule_match(self, pdp):
        req = AccessRequest("user:mallory", "vm:prod", "read", trust_score=0.95)
        decision = pdp.evaluate(req)
        assert decision.decision == AccessDecision.DENY

    def test_trust_gate_blocks(self, pdp):
        req = AccessRequest("user:alice", "vm:prod", "read", trust_score=0.1)
        decision = pdp.evaluate(req)
        assert decision.decision == AccessDecision.DENY
        assert "Trust score" in decision.reason

    def test_default_deny_unknown_user(self, pdp):
        req = AccessRequest("user:unknown", "vm:prod", "write", trust_score=0.9)
        decision = pdp.evaluate(req)
        # No matching PERMIT rule → default DENY
        assert decision.decision == AccessDecision.DENY

    def test_trust_condition_in_permit_rule(self, pdp):
        # Trust score is above gate but below rule's min_trust
        req = AccessRequest("user:alice", "vm:prod", "read", trust_score=0.55)
        decision = pdp.evaluate(req)
        assert decision.decision == AccessDecision.PERMIT

    def test_trust_condition_below_rule_min_trust(self, pdp):
        req = AccessRequest("user:alice", "vm:prod", "read", trust_score=0.49)
        decision = pdp.evaluate(req)
        # Blocked by trust gate (< TRUST_THRESHOLD_HIGH from config)
        assert decision.decision == AccessDecision.DENY
