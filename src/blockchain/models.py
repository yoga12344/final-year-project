"""
Blockchain record schema for access-log immutable audit trail.

Defines the ``BlockchainRecord`` dataclass that is serialised to JSON
before being written to the Hyperledger Fabric ledger.
"""

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BlockchainRecord:
    """
    Immutable audit record stored on the Fabric ledger.

    Fields
    ------
    record_id      : UUID v4 – primary key on the ledger.
    subject_id     : Identity of the requesting entity (user / service account).
    resource       : Target resource identifier (e.g. ``vm:prod-db-01``).
    action         : Performed / attempted action (``read``, ``write``, …).
    decision       : ``PERMIT`` or ``DENY``.
    trust_score    : Trust score at decision time (0.0 – 1.0).
    rule_id        : Policy rule that triggered the decision.
    reason         : Human-readable decision rationale.
    fl_round       : Federated Learning round number (if applicable).
    dqn_action     : DQN action index applied in this step (-1 = none).
    source_ip      : Source IP address of the request.
    context        : Arbitrary additional context (serialised from dict).
    timestamp      : Unix epoch float of the decision.
    tx_id          : Fabric transaction ID (filled after ledger write).
    """

    # --- Core fields -------------------------------------------------------
    record_id:   str = field(default_factory=lambda: str(uuid.uuid4()))
    subject_id:  str = ""
    resource:    str = ""
    action:      str = ""
    decision:    str = ""          # "PERMIT" | "DENY"
    trust_score: float = 0.0
    rule_id:     str = ""
    reason:      str = ""

    # --- Extended metadata -------------------------------------------------
    fl_round:    int  = -1         # -1 = not from FL round
    dqn_action:  int  = -1         # -1 = DQN not involved
    source_ip:   str  = ""
    context:     dict = field(default_factory=dict)

    # --- Timestamps / chain info -------------------------------------------
    timestamp:   float = field(default_factory=time.time)
    tx_id:       str   = ""        # Populated after Fabric submit

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        d = asdict(self)
        d["timestamp"] = round(d["timestamp"], 6)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "BlockchainRecord":
        """Deserialise from a plain dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_access_decision(
        cls,
        decision_dict: dict,
        fl_round:  int = -1,
        dqn_action: int = -1,
        source_ip: str = "",
        context:   dict | None = None,
    ) -> "BlockchainRecord":
        """
        Build a ``BlockchainRecord`` from an ``AccessDecision.to_dict()`` payload.

        Parameters
        ----------
        decision_dict : dict
            Output of ``AccessDecision.to_dict()``.
        fl_round      : int   Current FL training round (-1 if offline).
        dqn_action    : int   Most recent DQN action (-1 if not applied).
        source_ip     : str   Originating IP of the access request.
        context       : dict  Extra metadata to embed.
        """
        return cls(
            record_id=str(uuid.uuid4()),
            subject_id=decision_dict.get("subject_id", ""),
            resource=decision_dict.get("resource", ""),
            action=decision_dict.get("action", ""),
            decision=decision_dict.get("decision", "DENY"),
            trust_score=float(decision_dict.get("trust_score", 0.0)),
            rule_id=decision_dict.get("rule_id", ""),
            reason=decision_dict.get("reason", ""),
            fl_round=fl_round,
            dqn_action=dqn_action,
            source_ip=source_ip,
            context=context or {},
            timestamp=decision_dict.get("timestamp", time.time()),
        )


# ---------------------------------------------------------------------------
# Ledger query result helpers
# ---------------------------------------------------------------------------

@dataclass
class LedgerQueryResult:
    """Wrapper around paginated ledger query results."""

    records:     list[BlockchainRecord] = field(default_factory=list)
    total_count: int  = 0
    page:        int  = 1
    page_size:   int  = 50
    has_more:    bool = False

    def to_dict(self) -> dict:
        return {
            "records":     [r.to_dict() for r in self.records],
            "total_count": self.total_count,
            "page":        self.page,
            "page_size":   self.page_size,
            "has_more":    self.has_more,
        }
