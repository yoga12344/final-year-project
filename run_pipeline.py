"""
run_pipeline.py
===============
End-to-end pipeline: preprocess → train LSTM → evaluate → demonstrate trust scoring

Usage:
    # Step 1: Preprocess (only needs to run once)
    python run_pipeline.py --step preprocess

    # Step 2: Train LSTM
    python run_pipeline.py --step train --epochs 30

    # Step 3: Evaluate on test set
    python run_pipeline.py --step evaluate

    # Step 4: Live demo using sample log lines
    python run_pipeline.py --step demo

    # All steps at once
    python run_pipeline.py --step all --epochs 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("run_pipeline")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_PATH   = MODEL_DIR / "lstm_global.pth"
SCALER_PATH  = DATA_PROC / "scaler.pkl"
CSV_PATH     = DATA_RAW / "OpenStack_full.log_structured.csv"


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def step_preprocess(window: int = 20, step: int = 5):
    from src.trust_assessment.preprocessor import OpenStackPreprocessor
    logger.info("=== STEP: PREPROCESS ===")
    if not CSV_PATH.exists():
        logger.error(
            "CSV not found at %s\n"
            "Please copy your OpenStack_full.log_structured.csv to data/raw/",
            CSV_PATH,
        )
        sys.exit(1)

    prep  = OpenStackPreprocessor(
        raw_csv=CSV_PATH,
        output_dir=DATA_PROC,
        window_size=window,
        step_size=step,
    )
    stats = prep.run()
    logger.info("Preprocessing stats:")
    for k, v in stats.items():
        logger.info("  %-25s: %s", k, v)
    return stats


def step_train(epochs: int = 30, batch_size: int = 256):
    from src.trust_assessment.lstm_model import BiLSTMTrustModel, train_model

    logger.info("=== STEP: TRAIN ===")

    X_train = np.load(DATA_PROC / "X_train.npy")
    y_train = np.load(DATA_PROC / "y_train.npy")
    X_val   = np.load(DATA_PROC / "X_val.npy")
    y_val   = np.load(DATA_PROC / "y_val.npy")

    logger.info("Train: %s  Val: %s", X_train.shape, X_val.shape)
    logger.info("Anomaly rate — train: %.2f%%  val: %.2f%%",
                100 * y_train.mean(), 100 * y_val.mean())

    model   = BiLSTMTrustModel(input_dim=X_train.shape[2])
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        model_path=MODEL_PATH,
    )
    logger.info("Training complete. Best val_loss = %.4f", min(history["val_loss"]))
    return history


def step_evaluate():
    import torch
    from sklearn.metrics import (
        classification_report, roc_auc_score, confusion_matrix,
    )
    from src.trust_assessment.lstm_model import BiLSTMTrustModel

    logger.info("=== STEP: EVALUATE ===")

    X_test = np.load(DATA_PROC / "X_test.npy")
    y_test = np.load(DATA_PROC / "y_test.npy")

    model = BiLSTMTrustModel(input_dim=X_test.shape[2])
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        Xt     = torch.from_numpy(X_test)
        logits = model(Xt).squeeze(1)
        probs  = torch.sigmoid(logits).numpy()
        preds  = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    cm  = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=["Normal", "Anomaly"])

    logger.info("\nConfusion Matrix:\n%s", cm)
    logger.info("\nClassification Report:\n%s", report)
    logger.info("ROC-AUC: %.4f", auc)
    return {"auc": auc, "confusion_matrix": cm.tolist()}


def step_demo():
    from src.trust_assessment.trust_calculator import TrustCalculator

    logger.info("=== STEP: DEMO ===")

    calc = TrustCalculator(model_path=MODEL_PATH, scaler_path=SCALER_PATH)

    # Sample log lines from the real dataset
    demo_events = [
        # Normal HTTP GET
        ('10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1893 time: 0.2699',  "E42"),
        ('10.11.10.1 "POST /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers HTTP/1.1" status: 202 len: 733 time: 0.489',         "E41"),
        ('[instance: 3edec1e4-9678-4a3a-a21b-a145a4ee5e61] Creating image',                                                   "E37"),
        ('[instance: 3edec1e4-9678-4a3a-a21b-a145a4ee5e61] Instance spawned successfully.',                                   "E18"),
        ('10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" status: 200 len: 1910 time: 0.280',  "E42"),
        # ---- Anomaly sequence starts ----
        ('Error during fabric chain-code invoke',                                                                              "E48"),
        ('Bad response code while validating token: 401',                                                                     "E24"),
        ("Identity response: invalid",                                                                                        "E45"),
        ("Unable to validate token: Failed to fetch token data from identity server",                                         "E6"),
        ("The instance sync for host 'cp-1.slowvm1.tcloud-pg0' did not match. Re-created its InstanceList.",                 "E7"),
    ]

    print("\n{:<8} {:<70} {:<6} {:<12} {}".format(
        "Step", "Event (truncated)", "EventId", "TrustScore", "Latency(ms)"))
    print("-" * 120)
    for i, (content, event_id) in enumerate(demo_events, 1):
        result = calc.score_event(content, event_id)
        marker = " ⚠️  ANOMALY" if result["trust_score"] < 0.5 and result["buffer_ready"] else ""
        print("{:<8} {:<70} {:<6} {:<12.4f} {:.1f} ms{}".format(
            i,
            content[:68],
            event_id,
            result["trust_score"],
            result["latency_ms"],
            marker,
        ))


def step_federated(epochs: int = 5, num_clients: int = 3):
    import flwr as fl
    from src.federated_learning.flower_server import TrustFederatedServer
    from src.federated_learning.flower_client import TrustModelClient
    from src.federated_learning.dataset_splitter import DatasetSplitter

    logger.info("=== STEP: FEDERATED LEARNING ===")

    # 1. Split data
    logger.info("Loading preprocessed data for splitting...")
    X_train = np.load(DATA_PROC / "X_train.npy")
    y_train = np.load(DATA_PROC / "y_train.npy")
    X_val   = np.load(DATA_PROC / "X_val.npy")
    y_val   = np.load(DATA_PROC / "y_val.npy")

    splitter = DatasetSplitter(num_clients=num_clients, iid=True)
    splits_train = splitter.split(X_train, y_train)
    splits_val   = splitter.split(X_val, y_val)
    
    # 2. Setup Server
    server = TrustFederatedServer(save_dir=MODEL_DIR)
    strategy = server.build_strategy()

    # 3. Setup Client Factory
    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)
        return TrustModelClient(
            client_id=str(client_id),
            train_data=splits_train[client_id],
            val_data=splits_val[client_id],
            dp_noise_multiplier=0.01,
            dp_max_grad_norm=1.0,
        ).to_client()

    # 4. Start Simulation
    logger.info(f"Starting Flower Simulation with {num_clients} clients...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=epochs),
        strategy=strategy,
    )
    logger.info("Federated Training complete.")


def step_dqn(episodes: int = 500):
    from src.rl_agent.dqn_agent import DQNAgent
    from src.rl_agent.environment import TrustAccessEnvironment
    
    logger.info(f"=== STEP: DQN TRAINING ({episodes} episodes) ===")
    
    env = TrustAccessEnvironment()
    agent = DQNAgent()
    
    best_reward = -float('inf')
    
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state
            ep_reward += reward
            
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save()
            
        if (ep + 1) % 50 == 0:
            stats = agent.get_stats()
            logger.info(f"Episode {ep+1:4d} | Reward: {ep_reward:.1f} | Epsilon: {stats['epsilon']:.3f} | Buffer: {stats['buffer_size']}")

    logger.info(f"DQN Training complete. Best offline reward: {best_reward:.1f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DR-TBAC-ZT++ Pipeline")
    parser.add_argument(
        "--step",
        choices=["preprocess", "train", "evaluate", "demo", "federated", "dqn", "all"],
        default="all",
        help="Which step to run",
    )
    parser.add_argument("--epochs",     type=int, default=30,  help="Training epochs (or FL rounds)")
    parser.add_argument("--batch",      type=int, default=256, help="Batch size")
    parser.add_argument("--window",     type=int, default=20,  help="Sequence window size")
    parser.add_argument("--step-size",  type=int, default=5,   help="Sliding window step")
    parser.add_argument("--num-clients", type=int, default=3,  help="Number of FL clients")
    parser.add_argument("--dqn-episodes", type=int, default=500, help="Number of DQN episodes")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if args.step in ("preprocess", "all"):
        step_preprocess(window=args.window, step=args.step_size)

    if args.step in ("train", "all"):
        step_train(epochs=args.epochs, batch_size=args.batch)
        
    if args.step in ("federated", "all"):
        step_federated(epochs=5, num_clients=args.num_clients)  # Keep FL rounds small
        
    if args.step in ("dqn", "all"):
        step_dqn(episodes=args.dqn_episodes)

    if args.step in ("evaluate", "all"):
        step_evaluate()

    if args.step in ("demo", "all"):
        step_demo()

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
