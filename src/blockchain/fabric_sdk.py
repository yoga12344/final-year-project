"""
Hyperledger Fabric Python SDK Wrapper.

Provides high-level helpers for:
- Submitting access-log transactions to the Fabric network
- Querying the chain for historical records
- Managing connection profiles and identity wallets

Requires the ``hfc`` (Hyperledger Fabric Client) or ``fabric-sdk-py``
package. When the SDK is not installed the module provides a stub that
logs locally so unit tests can still run without a live Fabric network.
"""

import json
import logging
import time
import uuid
from typing import Any

from src.config import (
    FABRIC_CHANNEL,
    FABRIC_CHAINCODE,
    FABRIC_MSPID,
    FABRIC_PEER_ENDPOINT,
    FABRIC_ORDERER_ENDPOINT,
    FABRIC_ADMIN_CERT,
    FABRIC_ADMIN_KEY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SDK availability guard
# ---------------------------------------------------------------------------
try:
    from hfc.fabric import Client as HFClient
    from hfc.fabric.peer import create_peer
    from hfc.fabric.orderer import Orderer
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False
    logger.warning(
        "fabric-sdk-py not installed – FabricSDK running in STUB mode. "
        "Install with: pip install fabric-sdk-py"
    )


# ---------------------------------------------------------------------------
# Fabric SDK wrapper
# ---------------------------------------------------------------------------

class FabricSDK:
    """
    High-level wrapper around the Hyperledger Fabric Python SDK.

    All public methods are safe to call even when the SDK is not
    installed (they fall back to a local-log stub).
    """

    def __init__(self):
        self._client = None
        if _SDK_AVAILABLE:
            try:
                self._init_client()
            except Exception as exc:
                logger.warning("FabricSDK: client init failed – %s. Running in stub mode.", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_access_log(self, record: dict[str, Any]) -> dict:
        """
        Submit an access-log record to the Fabric ledger.

        Parameters
        ----------
        record : dict
            Should conform to the BlockchainRecord schema (see models.py).

        Returns
        -------
        dict with keys ``tx_id``, ``status``, ``timestamp``.
        """
        tx_id = str(uuid.uuid4()).replace("-", "")
        if self._client is not None:
            return self._invoke_chaincode(
                func="RecordAccess",
                args=[json.dumps(record)],
                tx_id=tx_id,
            )
        # Stub mode
        logger.info("[STUB] FabricSDK.submit_access_log: %s", json.dumps(record, indent=2))
        return {"tx_id": tx_id, "status": "STUB_OK", "timestamp": time.time()}

    def query_access_log(self, record_id: str) -> dict | None:
        """
        Query the ledger for a specific access record.

        Parameters
        ----------
        record_id : str  The UUID of the record to retrieve.

        Returns
        -------
        Record dict or None if not found.
        """
        if self._client is not None:
            return self._query_chaincode(func="GetRecord", args=[record_id])
        logger.info("[STUB] FabricSDK.query_access_log: id=%s", record_id)
        return None

    def query_history(self, subject_id: str, limit: int = 50) -> list[dict]:
        """
        Retrieve the access history for a given subject.

        Parameters
        ----------
        subject_id : str   Subject identifier.
        limit      : int   Maximum number of records to return.

        Returns
        -------
        List of record dicts sorted newest-first.
        """
        if self._client is not None:
            return self._query_chaincode(
                func="GetHistoryBySubject",
                args=[subject_id, str(limit)],
            ) or []
        logger.info("[STUB] FabricSDK.query_history: subject=%s limit=%d", subject_id, limit)
        return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        """Initialise the Fabric client with network configuration."""
        self._client = HFClient()
        self._client.new_channel(FABRIC_CHANNEL)
        peer = create_peer(endpoint=FABRIC_PEER_ENDPOINT)
        self._client.get_channel(FABRIC_CHANNEL).add_peer(peer)
        logger.info("FabricSDK: connected to peer %s on channel %s",
                    FABRIC_PEER_ENDPOINT, FABRIC_CHANNEL)

    def _invoke_chaincode(self, func: str, args: list[str], tx_id: str) -> dict:
        """Send a chaincode INVOKE transaction."""
        try:
            response = self._client.chaincode_invoke(
                requestor=self._get_requestor(),
                channel_name=FABRIC_CHANNEL,
                peers=[FABRIC_PEER_ENDPOINT],
                args=args,
                cc_name=FABRIC_CHAINCODE,
                fcn=func,
                wait_for_event=True,
            )
            return {"tx_id": tx_id, "status": "OK", "response": str(response),
                    "timestamp": time.time()}
        except Exception as exc:
            logger.error("FabricSDK._invoke_chaincode: %s", exc)
            return {"tx_id": tx_id, "status": "ERROR", "error": str(exc),
                    "timestamp": time.time()}

    def _query_chaincode(self, func: str, args: list[str]) -> Any:
        """Send a chaincode QUERY transaction."""
        try:
            response = self._client.chaincode_query(
                requestor=self._get_requestor(),
                channel_name=FABRIC_CHANNEL,
                peers=[FABRIC_PEER_ENDPOINT],
                args=args,
                cc_name=FABRIC_CHAINCODE,
                fcn=func,
            )
            return json.loads(response) if response else None
        except Exception as exc:
            logger.error("FabricSDK._query_chaincode: %s", exc)
            return None

    def _get_requestor(self):
        """Return a Fabric User identity for signing transactions."""
        from hfc.fabric import User
        user = User(name="Admin", org=FABRIC_MSPID, state_store=None)
        with open(FABRIC_ADMIN_CERT, "rb") as f:
            user.enrollment.cert = f.read()
        with open(FABRIC_ADMIN_KEY, "rb") as f:
            user.enrollment.private_key_pem = f.read()
        return user
