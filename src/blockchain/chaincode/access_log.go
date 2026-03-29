// SPDX-License-Identifier: Apache-2.0
// Access Log Chaincode for Hyperledger Fabric
// Stores immutable access-control decision records on the ledger.
//
// Transactions:
//   - RecordAccess(jsonRecord)         → tx_id
//   - GetRecord(recordId)              → JSON record
//   - GetHistoryBySubject(subjectId, limit) → JSON array
//   - GetAllRecords(pageSize, bookmark) → JSON paginated result

package main

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

// AccessRecord mirrors the Python BlockchainRecord schema.
type AccessRecord struct {
	RecordID   string                 `json:"record_id"`
	SubjectID  string                 `json:"subject_id"`
	Resource   string                 `json:"resource"`
	Action     string                 `json:"action"`
	Decision   string                 `json:"decision"` // "PERMIT" | "DENY"
	TrustScore float64                `json:"trust_score"`
	RuleID     string                 `json:"rule_id"`
	Reason     string                 `json:"reason"`
	FLRound    int                    `json:"fl_round"`
	DQNAction  int                    `json:"dqn_action"`
	SourceIP   string                 `json:"source_ip"`
	Context    map[string]interface{} `json:"context"`
	Timestamp  float64                `json:"timestamp"`
	TxID       string                 `json:"tx_id"`
}

// ---------------------------------------------------------------------------
// SmartContract definition
// ---------------------------------------------------------------------------

// AccessLogContract provides functions for managing access records.
type AccessLogContract struct {
	contractapi.Contract
}

// ---------------------------------------------------------------------------
// Transactions
// ---------------------------------------------------------------------------

// RecordAccess stores a new access record on the ledger.
// Args: jsonRecord – JSON-serialised AccessRecord.
func (c *AccessLogContract) RecordAccess(ctx contractapi.TransactionContextInterface, jsonRecord string) error {
	var record AccessRecord
	if err := json.Unmarshal([]byte(jsonRecord), &record); err != nil {
		return fmt.Errorf("failed to unmarshal record: %v", err)
	}

	if record.RecordID == "" {
		return fmt.Errorf("record_id is required")
	}

	// Attach the Fabric transaction ID
	record.TxID = ctx.GetStub().GetTxID()
	// Overwrite timestamp with ledger time for tamper-proof ordering
	ts, _ := ctx.GetStub().GetTxTimestamp()
	if ts != nil {
		record.Timestamp = float64(ts.Seconds) + float64(ts.Nanos)/1e9
	}

	encoded, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("failed to marshal record: %v", err)
	}

	return ctx.GetStub().PutState(record.RecordID, encoded)
}

// GetRecord retrieves a single access record by ID.
func (c *AccessLogContract) GetRecord(ctx contractapi.TransactionContextInterface, recordID string) (*AccessRecord, error) {
	data, err := ctx.GetStub().GetState(recordID)
	if err != nil {
		return nil, fmt.Errorf("GetState failed: %v", err)
	}
	if data == nil {
		return nil, fmt.Errorf("record %s not found", recordID)
	}

	var record AccessRecord
	if err := json.Unmarshal(data, &record); err != nil {
		return nil, fmt.Errorf("failed to unmarshal record: %v", err)
	}
	return &record, nil
}

// GetHistoryBySubject returns the access history for a given subject.
// Uses a rich query (requires CouchDB state database).
func (c *AccessLogContract) GetHistoryBySubject(
	ctx contractapi.TransactionContextInterface,
	subjectID string,
	limit int,
) ([]*AccessRecord, error) {

	queryString := fmt.Sprintf(
		`{"selector":{"subject_id":"%s"},"sort":[{"timestamp":"desc"}],"limit":%d}`,
		subjectID, limit,
	)

	iter, err := ctx.GetStub().GetQueryResult(queryString)
	if err != nil {
		return nil, fmt.Errorf("GetQueryResult failed: %v", err)
	}
	defer iter.Close()

	var records []*AccessRecord
	for iter.HasNext() {
		result, err := iter.Next()
		if err != nil {
			return nil, err
		}
		var record AccessRecord
		if err := json.Unmarshal(result.Value, &record); err != nil {
			return nil, err
		}
		records = append(records, &record)
	}
	return records, nil
}

// GetAllRecords returns all records with pagination support.
func (c *AccessLogContract) GetAllRecords(
	ctx contractapi.TransactionContextInterface,
	pageSize int32,
	bookmark string,
) (map[string]interface{}, error) {

	queryString := `{"selector":{},"sort":[{"timestamp":"desc"}]}`
	iter, meta, err := ctx.GetStub().GetQueryResultWithPagination(queryString, pageSize, bookmark)
	if err != nil {
		return nil, fmt.Errorf("paginated query failed: %v", err)
	}
	defer iter.Close()

	var records []*AccessRecord
	for iter.HasNext() {
		result, err := iter.Next()
		if err != nil {
			return nil, err
		}
		var record AccessRecord
		if err := json.Unmarshal(result.Value, &record); err != nil {
			return nil, err
		}
		records = append(records, &record)
	}

	return map[string]interface{}{
		"records":           records,
		"fetched_count":     len(records),
		"bookmark":          meta.Bookmark,
		"records_count":     meta.FetchedRecordsCount,
	}, nil
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	_ = time.Now() // keep import
	chaincode, err := contractapi.NewChaincode(new(AccessLogContract))
	if err != nil {
		panic(fmt.Sprintf("Error creating access-log chaincode: %v", err))
	}
	if err := chaincode.Start(); err != nil {
		panic(fmt.Sprintf("Error starting chaincode: %v", err))
	}
}
