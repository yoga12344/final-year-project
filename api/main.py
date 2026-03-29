"""
DR-TBAC-ZT++ | api/main.py
FastAPI Inference & Policy Engine Service
"""
import time
import uuid
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import contextlib
from prometheus_client import Histogram, Counter, make_asgi_app

from src.trust_assessment.trust_calculator import TrustCalculator
from src.rl_agent.dqn_agent import DQNAgent
from src.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

# Global model references
calc: TrustCalculator = None
dqn_agent: DQNAgent = None
explainer = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global calc, dqn_agent
    log.info("Starting up FastAPI Service. Loading Models...")
    try:
        from pathlib import Path
        model_path = str(cfg.lstm.model_save_path)
        scaler_path = str(cfg.processed_data_dir / "scaler.pkl")
        
        calc = TrustCalculator(model_path=model_path, scaler_path=scaler_path)
        log.info("✅ BiLSTM Trust Model loaded successfully.")
        
        dqn_agent = DQNAgent()
        dqn_path = cfg.dqn.model_save_path
        if Path(dqn_path).exists():
            dqn_agent.load(str(dqn_path))
            log.info("✅ DQN Agent weights loaded successfully.")
        else:
            log.warning("⚠️ DQN weights not found. Using randomly initialized threshold agent.")
            
        from src.explainability.shap_explainer import SHAPExplainer
        try:
            X_train = np.load(cfg.processed_data_dir / "X_train.npy")
            explainer = SHAPExplainer(model=calc.model, background_X=X_train, n_background=30)
            log.info("✅ SHAP Explainer loaded successfully.")
        except Exception as e:
            log.error(f"❌ Failed to load SHAP explainer: {e}")
            explainer = None

    except Exception as e:
        log.error(f"❌ Failed to load models during startup: {e}")
    yield
    log.info("Shutting down FastAPI service...")


# Prometheus Metrics
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Time spent processing access request")
DECISION_COUNTER = Counter("access_decisions_total", "Count of decisions", ["decision"])

app = FastAPI(
    title="DR-TBAC-ZT++ Inference API",
    description="Real-time Zero Trust Policy Engine API",
    version="1.0.0",
    lifespan=lifespan
)

# Expose metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)



class AccessLogEvent(BaseModel):
    user_id: str
    event_content: str
    event_id: str
    context: Dict[str, float] = {}  # Optional external context (e.g. Geo IP, Device Posture)


class DecisionResponse(BaseModel):
    request_id: str
    trust_score: float
    decision: str
    latency_ms: float
    explanation_available: bool
    top_features: List[Dict[str, Any]] = []

@app.get("/health")
def health_check():
    status = "healthy" if (calc is not None) else "degraded"
    return {"status": status, "models_loaded": calc is not None}


@app.post("/v1/score_access", response_model=DecisionResponse)
async def score_access(event: AccessLogEvent):
    start_time = time.time()
    req_id = str(uuid.uuid4())
    
    if calc is None:
        raise HTTPException(status_code=503, detail="Machine Learning Models are not currently loaded.")

    try:
        # Step 1: Push through BiLSTM / Maintain Sequence Buffer
        res = calc.score_event(event.event_content, event.event_id)
        
        trust_score = res["trust_score"]
        buffer_ready = res["buffer_ready"]
        
        # Default decision if buffer is filling up
        decision = "CHALLENGE"
        top_features = []
        
        if buffer_ready:
            # Step 2: Formulate state for DQN
            # In a real system, we would construct the 16D state from the DB. 
            # Here we provide a mock state embedding the real trust score
            state = np.zeros(cfg.dqn.state_dim, dtype=np.float32)
            state[0] = trust_score
            # Action 0=PERMIT, 1=DENY, 2=CHALLENGE, 3=THROTTLE
            action_idx = dqn_agent.select_action(state, eval_mode=True)
            action_map = {0: "PERMIT", 1: "DENY", 2: "CHALLENGE", 3: "THROTTLE"}
            decision = action_map.get(action_idx, "CHALLENGE")
            
            # Step 3: SHAP Explainability
            if explainer is not None:
                seq = np.stack(calc._buffer, axis=0)
                # nsamples=20 for fast real-time inference
                shap_res = explainer.explain(seq, nsamples=20)
                top_features = shap_res["top_features"]

        latency = (time.time() - start_time) * 1000
        
        # Step 4: Decision Audit Log
        import json
        import datetime
        audit_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "user_id": event.user_id,
            "event_id": event.event_id,
            "trust_score": round(trust_score, 4),
            "decision": decision,
            "latency_ms": round(latency, 2),
            "top_features": top_features[:3] # Log top 3 features
        }
        with open(cfg.data_dir / "decision_audit.jsonl", "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
        
        # Log to monitoring
        INFERENCE_LATENCY.observe(time.time() - start_time)
        DECISION_COUNTER.labels(decision=decision).inc()
        log.info(f"[Inference] ID:{req_id} | User:{event.user_id} | Trust:{trust_score:.4f} | Decision:{decision} | Latency:{latency:.2f}ms")

        return DecisionResponse(
            request_id=req_id,
            trust_score=trust_score,
            decision=decision,
            latency_ms=latency,
            explanation_available=buffer_ready,
            top_features=top_features
        )
        
    except Exception as e:
        log.error(f"Inference error on {req_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Inference Error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
