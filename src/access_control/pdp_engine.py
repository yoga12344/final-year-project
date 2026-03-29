"""
Policy Decision Point (PDP) Engine.

Evaluates access requests against policy rules enriched by trust score
and RL-agent decisions. Returns PERMIT / DENY decisions with audit metadata.
"""

import time
import json
import uuid
import logging
from typing import Any

from src.config import POLICY_FILE, PDP_DEFAULT_DECISION, TRUST_THRESHOLD_HIGH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class AccessRequest:
    """Structured access request."""

    def __init__(
        self,
        subject_id:   str,
        resource:     str,
        action:       str,
        context:      dict[str, Any] | None = None,
        trust_score:  float = 0.0,
    ):
        self.request_id  = str(uuid.uuid4())
        self.subject_id  = subject_id
        self.resource    = resource
        self.action      = action
        self.context     = context or {}
        self.trust_score = trust_score
        self.timestamp   = time.time()

    def to_dict(self) -> dict:
        return {
            "request_id":  self.request_id,
            "subject_id":  self.subject_id,
            "resource":    self.resource,
            "action":      self.action,
            "context":     self.context,
            "trust_score": round(self.trust_score, 4),
            "timestamp":   self.timestamp,
        }


class AccessDecision:
    """Decision returned by the PDP."""

    PERMIT = "PERMIT"
    DENY   = "DENY"

    def __init__(
        self,
        request:   AccessRequest,
        decision:  str,
        reason:    str = "",
        rule_id:   str = "",
    ):
        self.decision_id = str(uuid.uuid4())
        self.request     = request
        self.decision    = decision
        self.reason      = reason
        self.rule_id     = rule_id
        self.timestamp   = time.time()

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "request_id":  self.request.request_id,
            "subject_id":  self.request.subject_id,
            "resource":    self.request.resource,
            "action":      self.request.action,
            "trust_score": self.request.trust_score,
            "decision":    self.decision,
            "reason":      self.reason,
            "rule_id":     self.rule_id,
            "timestamp":   self.timestamp,
        }


# ---------------------------------------------------------------------------
# PDP Engine
# ---------------------------------------------------------------------------

class PDPEngine:
    """
    Zero-Trust Policy Decision Point (PDP).

    Evaluation order:
    1. Trust-score gate  (hard threshold).
    2. Explicit DENY rules (highest priority).
    3. Explicit PERMIT rules.
    4. Default decision (configurable, default = DENY).

    Policy file format (JSON)::

        {
          "version": "1.0",
          "rules": [
            {
              "rule_id": "R001",
              "effect": "PERMIT",
              "subjects": ["user:alice", "role:admin"],
              "resources": ["vm:*", "storage:bucket-A"],
              "actions": ["read", "list"],
              "conditions": {"min_trust": 0.6}
            }
          ]
        }
    """

    def __init__(self, policy_file: str = POLICY_FILE):
        self.policy_file = policy_file
        self._policy: dict = {}
        self._load_policy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, request: AccessRequest) -> AccessDecision:
        """
        Evaluate an access request.

        Returns:
            ``AccessDecision`` with PERMIT or DENY.
        """
        # 1. Trust-score gate
        if request.trust_score < TRUST_THRESHOLD_HIGH:
            return AccessDecision(
                request=request,
                decision=AccessDecision.DENY,
                reason=f"Trust score {request.trust_score:.3f} below threshold {TRUST_THRESHOLD_HIGH}",
                rule_id="TRUST_GATE",
            )

        rules = self._policy.get("rules", [])

        # 2. DENY rules first
        for rule in rules:
            if rule.get("effect", "").upper() == "DENY":
                if self._rule_matches(rule, request):
                    return AccessDecision(
                        request=request,
                        decision=AccessDecision.DENY,
                        reason="Explicit DENY rule matched",
                        rule_id=rule.get("rule_id", "UNKNOWN"),
                    )

        # 3. PERMIT rules
        for rule in rules:
            if rule.get("effect", "").upper() == "PERMIT":
                if self._rule_matches(rule, request):
                    return AccessDecision(
                        request=request,
                        decision=AccessDecision.PERMIT,
                        reason="PERMIT rule matched",
                        rule_id=rule.get("rule_id", "UNKNOWN"),
                    )

        # 4. Default
        return AccessDecision(
            request=request,
            decision=PDP_DEFAULT_DECISION,
            reason="No matching rule – default applied",
            rule_id="DEFAULT",
        )

    def reload_policy(self) -> None:
        """Hot-reload policy from disk (called by PolicyRewriter)."""
        self._load_policy()
        logger.info("PDP: policy reloaded from %s", self.policy_file)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_policy(self) -> None:
        try:
            with open(self.policy_file, "r", encoding="utf-8") as fh:
                self._policy = json.load(fh)
            logger.info("PDP: loaded %d rules", len(self._policy.get("rules", [])))
        except FileNotFoundError:
            logger.warning("PDP: policy file %s not found, using empty policy", self.policy_file)
            self._policy = {"version": "1.0", "rules": []}
        except json.JSONDecodeError as exc:
            logger.error("PDP: malformed policy file – %s", exc)
            self._policy = {"version": "1.0", "rules": []}

    def _rule_matches(self, rule: dict, request: AccessRequest) -> bool:
        """Check whether a given rule applies to the request."""
        # Subject check
        subjects = rule.get("subjects", ["*"])
        if not self._matches_any(request.subject_id, subjects):
            return False

        # Resource check
        resources = rule.get("resources", ["*"])
        if not self._matches_any(request.resource, resources):
            return False

        # Action check
        actions = rule.get("actions", ["*"])
        if not self._matches_any(request.action, actions):
            return False

        # Conditions
        conditions = rule.get("conditions", {})
        min_trust = conditions.get("min_trust", 0.0)
        if request.trust_score < float(min_trust):
            return False

        return True

    @staticmethod
    def _matches_any(value: str, patterns: list[str]) -> bool:
        """Match value against a list of exact values or '*' wildcards."""
        for p in patterns:
            if p == "*" or p == value:
                return True
            # Simple prefix wildcard  e.g. "vm:*"
            if p.endswith(":*") and value.startswith(p[:-1]):
                return True
        return False
