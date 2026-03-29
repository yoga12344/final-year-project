# DR-TBAC-ZT++ API Reference

## Trust Assessment API

### POST `/trust/score`
Compute trust score for an entity.

**Request**
```json
{
  "subject_id": "user:alice",
  "features":   [0.12, 0.88, 1.0, 0.0, ...],
  "context":    {"source_ip": "10.0.1.5", "resource": "vm:prod-db"}
}
```

**Response**
```json
{
  "subject_id":  "user:alice",
  "trust_score": 0.734,
  "label":       "TRUSTED",
  "latency_ms":  8.2
}
```

---

## Policy Decision API

### POST `/policy/evaluate`
Evaluate an access request against the current policy.

**Request**
```json
{
  "subject_id":  "user:alice",
  "resource":    "vm:prod-db",
  "action":      "read",
  "trust_score": 0.734,
  "context":     {"source_ip": "10.0.1.5"}
}
```

**Response**
```json
{
  "decision_id": "d5f9e...",
  "decision":    "PERMIT",
  "reason":      "PERMIT rule matched",
  "rule_id":     "R001",
  "timestamp":   1741935162.3
}
```

---

## Federated Learning API

### GET `/fl/status`
Returns current FL round information.

**Response**
```json
{
  "round":        42,
  "num_clients":  5,
  "global_loss":  0.0312,
  "global_acc":   0.961
}
```

### POST `/fl/trigger`
Trigger a new FL round (admin only).

---

## Blockchain Audit API

### POST `/ledger/record`
Write an access decision to the Fabric ledger.

**Request**
```json
{
  "record_id":   "b2d7a...",
  "subject_id":  "user:alice",
  "resource":    "vm:prod-db",
  "decision":    "PERMIT",
  "trust_score": 0.734
}
```

**Response**
```json
{
  "tx_id":     "abcdef123...",
  "status":    "OK",
  "timestamp": 1741935162.8
}
```

### GET `/ledger/history/{subject_id}`
Query the access history for a subject from the Fabric ledger.
