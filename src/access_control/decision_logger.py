"""
Decision Logger – persists access decisions to CSV + optional Kafka topic.

Every PERMIT / DENY decision produced by the PDP is timestamped and
stored locally. In a live environment the same record is streamed to
a Kafka topic so downstream services (SIEM, dashboards) can consume it
in real time.
"""

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from src.config import DECISION_LOG_PATH, KAFKA_BOOTSTRAP, KAFKA_DECISION_TOPIC

logger = logging.getLogger(__name__)


class DecisionLogger:
    """
    Writes access-control decisions to a CSV file.

    Optionally streams each record to a Kafka topic when a real
    ``confluent_kafka.Producer`` is available.

    Parameters
    ----------
    log_path     : str   Path to the CSV output file.
    kafka_topic  : str   Kafka topic for real-time streaming.
    enable_kafka : bool  Attempt to push to Kafka (requires broker).
    """

    FIELDNAMES = [
        "decision_id",
        "request_id",
        "subject_id",
        "resource",
        "action",
        "trust_score",
        "decision",
        "reason",
        "rule_id",
        "timestamp",
    ]

    def __init__(
        self,
        log_path:     str  = DECISION_LOG_PATH,
        kafka_topic:  str  = KAFKA_DECISION_TOPIC,
        enable_kafka: bool = False,
    ):
        self.log_path    = Path(log_path)
        self.kafka_topic = kafka_topic
        self._producer   = None

        self._init_csv()

        if enable_kafka:
            self._init_kafka()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, decision_dict: dict[str, Any]) -> None:
        """
        Persist a single access decision.

        Parameters
        ----------
        decision_dict : dict
            Output of ``AccessDecision.to_dict()``.
        """
        self._write_csv(decision_dict)
        if self._producer is not None:
            self._write_kafka(decision_dict)

    def log_batch(self, decisions: list[dict[str, Any]]) -> None:
        """Log a list of decisions."""
        for d in decisions:
            self.log(d)

    def get_recent(self, n: int = 100) -> list[dict]:
        """Read the last *n* rows from the CSV log (newest first)."""
        if not self.log_path.exists():
            return []
        try:
            with open(self.log_path, "r", newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            return list(reversed(rows[-n:]))
        except Exception as exc:
            logger.error("DecisionLogger.get_recent: %s", exc)
            return []

    def stats(self) -> dict:
        """Return aggregate PERMIT / DENY counts from the log file."""
        permit = deny = total = 0
        if not self.log_path.exists():
            return {"total": 0, "permit": 0, "deny": 0}
        try:
            with open(self.log_path, "r", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    total += 1
                    if row.get("decision") == "PERMIT":
                        permit += 1
                    else:
                        deny += 1
        except Exception as exc:
            logger.error("DecisionLogger.stats: %s", exc)
        return {"total": total, "permit": permit, "deny": deny}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_csv(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def _write_csv(self, record: dict) -> None:
        try:
            with open(self.log_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.FIELDNAMES, extrasaction="ignore")
                writer.writerow(record)
        except Exception as exc:
            logger.error("DecisionLogger._write_csv: %s", exc)

    def _init_kafka(self) -> None:
        try:
            from confluent_kafka import Producer
            self._producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})
            logger.info("DecisionLogger: Kafka producer connected to %s", KAFKA_BOOTSTRAP)
        except ImportError:
            logger.warning("DecisionLogger: confluent_kafka not installed – Kafka logging disabled")
        except Exception as exc:
            logger.warning("DecisionLogger: Kafka init failed – %s", exc)

    def _write_kafka(self, record: dict) -> None:
        try:
            payload = json.dumps(record).encode("utf-8")
            self._producer.produce(
                self.kafka_topic,
                value=payload,
                key=record.get("request_id", "").encode("utf-8"),
                callback=self._kafka_delivery_report,
            )
            self._producer.poll(0)
        except Exception as exc:
            logger.error("DecisionLogger._write_kafka: %s", exc)

    @staticmethod
    def _kafka_delivery_report(err, msg) -> None:
        if err:
            logger.error("DecisionLogger: Kafka delivery failed – %s", err)
