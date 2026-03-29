"""
Policy Rewriter – Updates policy.json dynamically based on DQN decisions.

The DQN agent returns an action index that maps to a policy mutation
(e.g., tighten trust threshold, add DENY rule, promote user to higher
privilege tier). This module translates those action codes into concrete
JSON policy changes and persists them to disk.
"""

import json
import shutil
import logging
import time
from pathlib import Path
from typing import Any

from src.config import POLICY_FILE, RL_ACTION_MAP

logger = logging.getLogger(__name__)


class PolicyRewriter:
    """
    Applies DQN action outputs as mutations to the live policy file.

    Each action index is mapped to a lambda that modifies the in-memory
    policy dict. After every write the PDP is asked to hot-reload.
    """

    def __init__(
        self,
        policy_file: str = POLICY_FILE,
        pdp=None,        # Optional reference to a live PDPEngine instance
        backup: bool = True,
    ):
        self.policy_file = Path(policy_file)
        self.pdp         = pdp
        self.backup      = backup

        self._policy: dict = {}
        self._load()

        # Build action dispatch table from config
        self._action_handlers: dict[int, callable] = self._build_dispatch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_action(self, action_idx: int, context: dict[str, Any] | None = None) -> dict:
        """
        Apply a RL action to the policy.

        Parameters
        ----------
        action_idx : int
            Integer action from the DQN agent.
        context    : dict, optional
            Extra information (e.g., subject_id, resource) used by
            context-sensitive actions.

        Returns
        -------
        dict with keys ``action_name``, ``applied``, ``timestamp``.
        """
        context = context or {}
        action_name = RL_ACTION_MAP.get(action_idx, f"unknown_action_{action_idx}")

        handler = self._action_handlers.get(action_idx)
        if handler is None:
            logger.warning("PolicyRewriter: unknown action %d, skipping", action_idx)
            return {"action_name": action_name, "applied": False, "timestamp": time.time()}

        try:
            handler(context)
            self._write()
            if self.pdp is not None:
                self.pdp.reload_policy()
            logger.info("PolicyRewriter: applied action %d (%s)", action_idx, action_name)
            return {"action_name": action_name, "applied": True, "timestamp": time.time()}
        except Exception as exc:
            logger.error("PolicyRewriter: error applying action %d – %s", action_idx, exc)
            return {"action_name": action_name, "applied": False, "timestamp": time.time()}

    def get_policy_snapshot(self) -> dict:
        """Return a deep copy of the current in-memory policy."""
        import copy
        return copy.deepcopy(self._policy)

    # ------------------------------------------------------------------
    # Private: action handlers
    # ------------------------------------------------------------------

    def _build_dispatch(self) -> dict[int, callable]:
        return {
            0: self._action_noop,
            1: self._action_tighten_trust,
            2: self._action_relax_trust,
            3: self._action_add_deny_rule,
            4: self._action_remove_deny_rule,
            5: self._action_isolate_subject,
            6: self._action_restore_subject,
        }

    def _action_noop(self, _ctx: dict) -> None:
        logger.debug("PolicyRewriter: NOOP – policy unchanged")

    def _action_tighten_trust(self, ctx: dict) -> None:
        """Raise the global min_trust threshold by 0.05 (max 0.95)."""
        for rule in self._policy.get("rules", []):
            conds = rule.setdefault("conditions", {})
            cur   = float(conds.get("min_trust", 0.5))
            conds["min_trust"] = min(round(cur + 0.05, 2), 0.95)

    def _action_relax_trust(self, ctx: dict) -> None:
        """Lower the global min_trust threshold by 0.05 (min 0.10)."""
        for rule in self._policy.get("rules", []):
            conds = rule.setdefault("conditions", {})
            cur   = float(conds.get("min_trust", 0.5))
            conds["min_trust"] = max(round(cur - 0.05, 2), 0.10)

    def _action_add_deny_rule(self, ctx: dict) -> None:
        """Insert an explicit DENY rule for a subject / resource pair."""
        subject  = ctx.get("subject_id", "*")
        resource = ctx.get("resource",   "*")
        new_rule = {
            "rule_id":   f"DENY_DQN_{int(time.time())}",
            "effect":    "DENY",
            "subjects":  [subject],
            "resources": [resource],
            "actions":   ["*"],
            "conditions": {},
        }
        self._policy.setdefault("rules", []).insert(0, new_rule)

    def _action_remove_deny_rule(self, ctx: dict) -> None:
        """Remove the most recently added DQN-injected DENY rule."""
        rules = self._policy.get("rules", [])
        for i in reversed(range(len(rules))):
            if rules[i].get("rule_id", "").startswith("DENY_DQN_"):
                rules.pop(i)
                return

    def _action_isolate_subject(self, ctx: dict) -> None:
        """Add a blanket DENY rule for a specific subject."""
        subject = ctx.get("subject_id", "unknown")
        new_rule = {
            "rule_id":   f"ISOLATE_{subject}_{int(time.time())}",
            "effect":    "DENY",
            "subjects":  [subject],
            "resources": ["*"],
            "actions":   ["*"],
            "conditions": {},
        }
        self._policy.setdefault("rules", []).insert(0, new_rule)

    def _action_restore_subject(self, ctx: dict) -> None:
        """Remove all ISOLATE rules for a specific subject."""
        subject = ctx.get("subject_id", "")
        if not subject:
            return
        self._policy["rules"] = [
            r for r in self._policy.get("rules", [])
            if not r.get("rule_id", "").startswith(f"ISOLATE_{subject}_")
        ]

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.policy_file.exists():
            logger.warning("PolicyRewriter: %s not found, starting with empty policy", self.policy_file)
            self._policy = {"version": "1.0", "rules": []}
            return
        with open(self.policy_file, "r", encoding="utf-8") as fh:
            self._policy = json.load(fh)

    def _write(self) -> None:
        if self.backup and self.policy_file.exists():
            backup_path = self.policy_file.with_suffix(".json.bak")
            shutil.copy2(self.policy_file, backup_path)

        self.policy_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.policy_file, "w", encoding="utf-8") as fh:
            json.dump(self._policy, fh, indent=2)
