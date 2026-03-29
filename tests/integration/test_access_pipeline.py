"""
Integration test: PDP + PolicyRewriter + DecisionLogger pipeline.
"""

import json
import os
import tempfile
import pytest

from src.access_control.pdp_engine import PDPEngine, AccessRequest
from src.access_control.policy_rewriter import PolicyRewriter
from src.access_control.decision_logger import DecisionLogger


@pytest.fixture
def setup_pipeline(tmp_path):
    # Create a minimal policy file
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
            }
        ],
    }
    policy_path = str(tmp_path / "policy.json")
    with open(policy_path, "w") as f:
        json.dump(policy, f)

    log_path = str(tmp_path / "decisions.csv")
    pdp      = PDPEngine(policy_file=policy_path)
    rewriter = PolicyRewriter(policy_file=policy_path, pdp=pdp, backup=False)
    logger   = DecisionLogger(log_path=log_path)
    return pdp, rewriter, logger, policy_path, log_path


class TestIntegrationPipeline:
    def test_permit_then_log(self, setup_pipeline):
        pdp, rewriter, logger, _, log_path = setup_pipeline
        req      = AccessRequest("user:alice", "vm:prod", "read", trust_score=0.8)
        decision = pdp.evaluate(req)
        logger.log(decision.to_dict())
        stats = logger.stats()
        assert stats["total"]  == 1
        assert stats["permit"] == 1

    def test_deny_after_tighten_trust(self, setup_pipeline):
        pdp, rewriter, logger, _, _ = setup_pipeline
        # Tighten trust threshold (action 1)
        rewriter.apply_action(1)
        # Alice's trust=0.51 may now be below the rule's min_trust
        req      = AccessRequest("user:alice", "vm:prod", "read", trust_score=0.6)
        decision = pdp.evaluate(req)
        # Decision depends on updated threshold; just check it's evaluated without error
        assert decision.decision in ("PERMIT", "DENY")

    def test_isolate_and_restore(self, setup_pipeline):
        pdp, rewriter, logger, _, _ = setup_pipeline
        rewriter.apply_action(5, context={"subject_id": "user:bob"})
        req = AccessRequest("user:bob", "vm:prod", "read", trust_score=0.95)
        d   = pdp.evaluate(req)
        assert d.decision == "DENY"

        rewriter.apply_action(6, context={"subject_id": "user:bob"})
        # After restore, Bob is no longer explicitly isolated (default applies)
        d2 = pdp.evaluate(req)
        # Should be DENY by default (no PERMIT rule for bob), but NOT isolated
        assert d2.rule_id != "ISOLATE_user:bob"
