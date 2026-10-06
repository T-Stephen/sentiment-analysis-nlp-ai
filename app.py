"""
SENTINEL AI — Executive Sentiment Intelligence Platform
Version: v14.3.0-PROD
Production Hardened Core Application
"""

import csv
import datetime
import hashlib
import io
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from contextlib import closing
from typing import Any, Dict, List, Literal, Optional

import joblib
import nltk
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("sentinel_core")

# --- ENTERPRISE CONFIGURATION & FILE SYSTEM PATHS ---
STORE_RAW_TEXT = os.getenv("STORE_RAW_TEXT", "false").lower() in ("true", "1", "yes")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "sentinel.db")
BENCHMARK_FILE = os.path.join(BASE_DIR, "artifacts", "benchmark", "model_metrics.json")
LEGACY_BENCHMARK_FILE = os.path.join(BASE_DIR, "model_benchmarks.pkl")

# --- DATABASE LAYER ---
def init_db():
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        with conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    request_id TEXT,
                    text TEXT, 
                    sentiment TEXT, 
                    confidence REAL, 
                    engine TEXT, 
                    ai_score REAL, 
                    emotion TEXT,
                    user_feedback TEXT, 
                    latency_ms REAL, 
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_req_id ON predictions(request_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON predictions(sentiment)")

init_db()

# --- MODEL ARTIFACT MANAGEMENT ---
try:
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    loaded_models = {
        "naive_bayes": joblib.load(os.path.join(MODELS_DIR, "naive_bayes_model.pkl")),
        "logistic_regression": joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.pkl")),
        "linear_svm": joblib.load(os.path.join(MODELS_DIR, "linear_svm_model.pkl"))
    }
    logger.info("All model artifacts and TF-IDF vectorizer loaded successfully.")
except Exception as e:
    logger.critical(f"FATAL: Artifact loading failure: {e}", exc_info=True)
    raise

# Ensure NLTK data availability
nltk.download("stopwords", quiet=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
vocab_lookup = {idx: word for word, idx in vectorizer.vocabulary_.items()}

# Precompute linear weights cache for mathematical explainability
def extract_model_weight_vector(clf, engine_name: str) -> Optional[np.ndarray]:
    if hasattr(clf, "coef_"):
        return clf.coef_[0]
    elif hasattr(clf, "calibrated_classifiers_"):
        # CalibratedClassifierCV wrapping LinearSVC: average coefficients of base estimators
        return np.mean([c.estimator.coef_[0] for c in clf.calibrated_classifiers_], axis=0)
    elif hasattr(clf, "feature_log_prob_"):
        # MultinomialNB: log odds ratio (weight of evidence for positive class)
        return clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    return None

model_weight_vectors = {
    name: extract_model_weight_vector(clf, name)
    for name, clf in loaded_models.items()
}

# --- APPLICATION FACTORY ---
app = FastAPI(
    title="SENTINEL AI — Executive Sentiment Intelligence Platform",
    description="High-precision corporate natural language intelligence platform engineered for executive sentiment analysis.",
    version="14.3.0-PROD",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- SECURITY & OBSERVABILITY MIDDLEWARE ---
class SecurityAndObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID")
        if not req_id:
            req_id = f"SN-{uuid.uuid4().hex[:8].upper()}"
        request.state.request_id = req_id

        start_time = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(f"Unhandled exception during {request.method} {request.url.path}: {exc}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "request_id": req_id,
                    "message": "An unexpected error occurred while processing the request."
                },
                headers={"X-Request-ID": req_id}
            )

        duration = (time.perf_counter() - start_time) * 1000.0

        # Inject Security Headers
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
        response.headers["X-Response-Time-Ms"] = f"{duration:.2f}"
        
        # Controlled CSP
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://fonts.gstatic.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "script-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )
        return response

app.add_middleware(SecurityAndObservabilityMiddleware)

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)

# --- PYDANTIC SCHEMAS ---
VALID_MODELS = ("logistic_regression", "linear_svm", "naive_bayes")

class TextPayload(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input text to analyze (1 to 5000 characters).")
    model_choice: Optional[str] = Field("logistic_regression", description="Selected ML classifier.")

    @field_validator("text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Input text cannot be empty or pure whitespace.")
        return stripped

    @field_validator("model_choice")
    @classmethod
    def validate_engine(cls, v: Optional[str]) -> str:
        if not v:
            return "logistic_regression"
        cleaned = v.strip().lower()
        if cleaned not in VALID_MODELS:
            # Gracefully handle valid choices while preserving strict validation
            return "logistic_regression"
        return cleaned

class FeedbackPayload(BaseModel):
    prediction_id: int = Field(..., ge=1, description="Database ID of the target prediction.")
    is_correct: bool = Field(..., description="Confirmation flag.")

class AttributionNode(BaseModel):
    token: str
    weight: float

class ExplanationModel(BaseModel):
    positive_drivers: List[AttributionNode]
    negative_drivers: List[AttributionNode]
    methodology: str = "Linear feature attribution: TF-IDF value multiplied by model weight vector."

class PredictionResponse(BaseModel):
    request_id: str
    prediction_id: int
    sentiment: str
    ai_score: float
    confidence: float
    emotion: str
    explanation: ExplanationModel
    latency_ms: float
    tokens_count: int

class SingleModelConsensus(BaseModel):
    sentiment: str
    ai_score: float
    confidence: float
    emotion: str

class ConsensusResponse(BaseModel):
    request_id: str
    agreement_ratio: str
    consensus_sentiment: str
    disagreement_detected: bool
    models: Dict[str, SingleModelConsensus]
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    database_connected: bool
    models_loaded: Dict[str, bool]
    vectorizer_loaded: bool

# --- CORE NLP & HEURISTICS ---
def clean_and_tokenize(text: str):
    """
    Cleans raw text, strips HTML tags and non-alphabet characters, removes stopwords, and returns stems.
    Matches the training preprocessing pipeline.
    """
    cleaned = re.sub(r"<.*?>", " ", text)
    raw = re.findall(r"[a-zA-Z]+", cleaned.lower())
    return [(w, stemmer.stem(w)) for w in raw if w not in stop_words]

def analyze_affective_heuristics(text: str, score: float) -> str:
    lowered = text.lower()
    anger_terms = {"terrible", "worst", "awful", "horrible", "hate", "disgusting", "poor", "pathetic"}
    joy_terms = {"masterpiece", "brilliant", "exceptional", "loved", "superb", "stunning", "great", "flawless"}
    sad_terms = {"boring", "slow", "tired", "depressing", "disappointing", "waste", "dull"}
    words = set(re.findall(r"[a-zA-Z]+", lowered))
    
    if words & anger_terms or score <= -0.5:
        return "Anger / Frustration"
    if words & joy_terms or score >= 0.5:
        return "Joy / High Satisfaction"
    if words & sad_terms or (-0.5 < score < -0.1):
        return "Critical / Discontent"
    if score > 0.1:
        return "Optimism"
    return "Balanced Neutral"

def compute_linear_attribution(model_name: str, vec_matrix):
    weights = model_weight_vectors.get(model_name)
    if weights is None:
        weights = model_weight_vectors.get("logistic_regression")
    
    feature_weights = []
    if weights is not None:
        nonzero_indices = vec_matrix.nonzero()[1]
        for idx in nonzero_indices:
            weight = float(weights[idx] * vec_matrix[0, idx])
            feature_weights.append((vocab_lookup.get(idx, ""), weight))
        
        feature_weights.sort(key=lambda x: x[1], reverse=True)
        return {
            "positive_drivers": [{"token": t, "weight": round(w, 4)} for t, w in feature_weights if w > 0][:5],
            "negative_drivers": [{"token": t, "weight": round(w, 4)} for t, w in feature_weights if w < 0][:5],
            "methodology": "Linear feature attribution: TF-IDF value multiplied by model weight vector."
        }
    return {"positive_drivers": [], "negative_drivers": [], "methodology": "Unavailable"}

def run_single_inference(engine: str, stemmed_text: str, original_text: str):
    clf = loaded_models.get(engine, loaded_models["logistic_regression"])
    vec = vectorizer.transform([stemmed_text])
    
    probs = clf.predict_proba(vec)[0]
    prob_neg, prob_pos = float(probs[0]), float(probs[1])
    
    diff = prob_pos - prob_neg
    if abs(diff) < 0.10:
        sentiment = "Neutral"
    else:
        sentiment = "Positive" if prob_pos > prob_neg else "Negative"
        
    ai_score = round(diff, 4)
    confidence = round(max(prob_pos, prob_neg) * 100, 2)
    emotion = analyze_affective_heuristics(original_text, ai_score)
    return {
        "engine": engine,
        "sentiment": sentiment,
        "ai_score": ai_score,
        "confidence": confidence,
        "emotion": emotion,
        "vec": vec
    }

# --- API ENDPOINTS ---

@app.get("/health", response_model=HealthResponse)
@app.get("/api/v1/health", response_model=HealthResponse)
def get_health():
    db_connected = False
    try:
        with closing(sqlite3.connect(DB_PATH, timeout=3)) as conn:
            conn.execute("SELECT 1 FROM predictions LIMIT 1")
            db_connected = True
    except Exception as e:
        logger.warning(f"Healthcheck DB check degraded: {e}")

    models_status = {k: (v is not None) for k, v in loaded_models.items()}
    status = "healthy" if (db_connected and all(models_status.values())) else "degraded"
    
    return {
        "status": status,
        "version": "14.3.0-PROD",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "database_connected": db_connected,
        "models_loaded": models_status,
        "vectorizer_loaded": (vectorizer is not None)
    }

@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_v1(payload: TextPayload, request: Request):
    start_time = time.perf_counter()
    req_id = getattr(request.state, "request_id", f"SN-{uuid.uuid4().hex[:8].upper()}")
    
    engine = payload.model_choice.lower() if payload.model_choice else "logistic_regression"
    if engine not in loaded_models:
        engine = "logistic_regression"
        
    tokens = clean_and_tokenize(payload.text)
    stemmed_text = " ".join([s for _, s in tokens])
    
    result = run_single_inference(engine, stemmed_text, payload.text)
    latency = round((time.perf_counter() - start_time) * 1000, 2)
    explanation = compute_linear_attribution(engine, result["vec"])
    
    stored_text = payload.text if STORE_RAW_TEXT else f"[REDACTED] - SHA256:{hashlib.sha256(payload.text.encode()).hexdigest()}"
    
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO predictions 
                   (request_id, text, sentiment, confidence, engine, ai_score, emotion, latency_ms) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (req_id, stored_text, result["sentiment"], result["confidence"], engine, result["ai_score"], result["emotion"], latency)
            )
            pred_id = cursor.lastrowid
            
    return {
        "request_id": req_id,
        "prediction_id": pred_id,
        "sentiment": result["sentiment"],
        "ai_score": result["ai_score"],
        "confidence": result["confidence"],
        "emotion": result["emotion"],
        "explanation": explanation,
        "latency_ms": latency,
        "tokens_count": len(tokens)
    }

@app.post("/api/v1/consensus", response_model=ConsensusResponse)
def evaluate_consensus(payload: TextPayload, request: Request):
    start_time = time.perf_counter()
    req_id = getattr(request.state, "request_id", f"SN-{uuid.uuid4().hex[:8].upper()}")
    
    tokens = clean_and_tokenize(payload.text)
    stemmed_text = " ".join([s for _, s in tokens])
    
    model_results = {}
    sentiment_votes = {}
    
    for engine in VALID_MODELS:
        res = run_single_inference(engine, stemmed_text, payload.text)
        model_results[engine] = {
            "sentiment": res["sentiment"],
            "ai_score": res["ai_score"],
            "confidence": res["confidence"],
            "emotion": res["emotion"]
        }
        sentiment_votes[res["sentiment"]] = sentiment_votes.get(res["sentiment"], 0) + 1
        
    latency = round((time.perf_counter() - start_time) * 1000, 2)
    
    # Majority determination
    sorted_votes = sorted(sentiment_votes.items(), key=lambda x: x[1], reverse=True)
    majority_sentiment, majority_count = sorted_votes[0]
    total_models = len(VALID_MODELS)
    agreement_ratio = f"{majority_count} / {total_models} models agree"
    disagreement_detected = (majority_count < total_models)
    
    return {
        "request_id": req_id,
        "agreement_ratio": agreement_ratio,
        "consensus_sentiment": majority_sentiment,
        "disagreement_detected": disagreement_detected,
        "models": model_results,
        "latency_ms": latency
    }

@app.post("/api/v1/feedback")
def submit_feedback(payload: FeedbackPayload):
    feedback_str = "Confirmed" if payload.is_correct else "Flagged Incorrect"
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        with conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE predictions SET user_feedback = ? WHERE id = ?", (feedback_str, payload.prediction_id))
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail=f"Prediction ID {payload.prediction_id} not found.")
    return {"status": "Telemetry recorded.", "prediction_id": payload.prediction_id}

@app.get("/api/v1/analytics")
def get_analytics():
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        total = cursor.execute("SELECT COUNT(*) as cnt FROM predictions").fetchone()["cnt"]
        pos = cursor.execute("SELECT COUNT(*) as cnt FROM predictions WHERE sentiment = 'Positive'").fetchone()["cnt"]
        neg = cursor.execute("SELECT COUNT(*) as cnt FROM predictions WHERE sentiment = 'Negative'").fetchone()["cnt"]
        neu = cursor.execute("SELECT COUNT(*) as cnt FROM predictions WHERE sentiment = 'Neutral'").fetchone()["cnt"]
        avg_lat = cursor.execute("SELECT AVG(latency_ms) as lat FROM predictions").fetchone()["lat"] or 0.0
        
        # Percentiles & model breakdown
        latencies = [row["latency_ms"] for row in cursor.execute("SELECT latency_ms FROM predictions WHERE latency_ms IS NOT NULL ORDER BY latency_ms ASC").fetchall()]
        
        p95 = round(float(np.percentile(latencies, 95)), 2) if len(latencies) >= 5 else "Insufficient data"
        p99 = round(float(np.percentile(latencies, 99)), 2) if len(latencies) >= 10 else "Insufficient data"
        
        usage_rows = cursor.execute("SELECT engine, COUNT(*) as cnt FROM predictions GROUP BY engine").fetchall()
        model_distribution = {r["engine"]: r["cnt"] for r in usage_rows}
        
    return {
        "total_analyzed": total,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "avg_latency_ms": round(avg_lat, 2),
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "model_distribution": model_distribution
    }

@app.get("/api/v1/history")
def get_history(limit: int = 15):
    limit = max(1, min(limit, 100))
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.cursor().execute(
            """SELECT id, request_id, sentiment, confidence, engine, ai_score, emotion, user_feedback, latency_ms, timestamp 
               FROM predictions 
               ORDER BY id DESC LIMIT ?""",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

@app.get("/api/v1/models")
def get_model_registry():
    # Return verified benchmark data from file without fabrication
    if os.path.exists(BENCHMARK_FILE):
        try:
            with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read benchmark file {BENCHMARK_FILE}: {e}")
            
    if os.path.exists(LEGACY_BENCHMARK_FILE):
        try:
            data = joblib.load(LEGACY_BENCHMARK_FILE)
            return {"source": "legacy_pickle", "models": data}
        except Exception:
            pass
            
    return {
        "status": "active",
        "models": {k: {"name": k, "status": "loaded"} for k in loaded_models.keys()}
    }

@app.get("/api/v1/export/{format_type}")
def export_telemetry(format_type: str):
    format_clean = format_type.strip().lower()
    if format_clean not in ("csv", "json"):
        raise HTTPException(status_code=400, detail="Invalid format. Supported formats are 'csv' and 'json'.")

    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.cursor().execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
        data = [dict(r) for r in rows]

    if format_clean == "csv":
        output = io.StringIO()
        if data:
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=sentinel_telemetry.csv"}
        )
    return JSONResponse(content=data)

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(content=b"", status_code=204)

# --- FLAGSHIP LUXURY UI: OBSIDIAN INTELLIGENCE TERMINAL ---
@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SENTINEL AI — Executive Sentiment Intelligence Platform</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🛡️</text></svg>">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            color-scheme: dark;

            /* Surfaces: Precision Intelligence Console */
            --surface-void: #07080B;
            --surface-chassis: #0C0E14;
            --surface-plate: #12151D;
            --surface-well: #090B10;
            --surface-hover: #171B24;

            /* Hairlines & Borders */
            --border-subtle: rgba(255, 255, 255, 0.06);
            --border-hairline: rgba(255, 255, 255, 0.10);
            --border-focus: #3B82F6;

            /* Typography */
            --text-primary: #F8FAFC;
            --text-body: #CBD5E1;
            --text-muted: #8090A5;
            --text-disabled: #475569;

            /* Semantic Data Signals */
            --signal-positive: #10B981;
            --signal-positive-bg: rgba(16, 185, 129, 0.10);
            --signal-positive-border: rgba(16, 185, 129, 0.25);

            --signal-negative: #F43F5E;
            --signal-negative-bg: rgba(244, 63, 94, 0.10);
            --signal-negative-border: rgba(244, 63, 94, 0.25);

            --signal-neutral: #94A3B8;
            --signal-neutral-bg: rgba(148, 163, 184, 0.10);
            --signal-neutral-border: rgba(148, 163, 184, 0.25);

            --signal-divergence: #F59E0B;
            --signal-divergence-bg: rgba(245, 158, 11, 0.12);
            --signal-divergence-border: rgba(245, 158, 11, 0.30);

            /* Primary Action */
            --action-primary: #2563EB;
            --action-primary-hover: #1D4ED8;
            --action-primary-active: #1E40AF;
            --action-primary-text: #FFFFFF;

            /* Geometry System */
            --radius-outer: 12px;
            --radius-inner: 8px;
            --radius-control: 6px;
            --radius-pill: 9999px;

            /* Typography Stacks */
            --font-sans: 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            --font-mono: 'Geist Mono', 'JetBrains Mono', ui-monospace, Menlo, Consolas, monospace;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            background-color: var(--surface-void);
            color: var(--text-body);
            font-family: var(--font-sans);
            font-size: 13px;
            line-height: 1.5;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            -webkit-font-smoothing: antialiased;
        }

        /* Subtle Technical Ambient Layer */
        .ambient-layer {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(circle at 20% 12%, rgba(37, 99, 235, 0.03) 0%, transparent 50%),
                        radial-gradient(circle at 80% 88%, rgba(16, 185, 129, 0.02) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }

        /* SYSTEM HEADER */
        header {
            padding: 14px 36px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--surface-chassis);
            border-bottom: 1px solid var(--border-hairline);
            position: sticky;
            top: 0;
            z-index: 100;
            gap: 20px;
        }

        .brand-block {
            display: flex;
            align-items: center;
            gap: 20px;
            flex-wrap: wrap;
        }

        .brand-identity {
            display: flex;
            flex-direction: column;
        }

        .brand-name {
            font-family: var(--font-sans);
            font-size: 15px;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--text-primary);
            text-transform: uppercase;
            line-height: 1.2;
        }

        .brand-subtitle {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-muted);
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        /* OPERATIONAL METADATA STRIP */
        .system-meta-strip {
            display: flex;
            align-items: center;
            gap: 8px;
            border-left: 1px solid var(--border-subtle);
            padding-left: 18px;
            flex-wrap: wrap;
        }

        .meta-pill {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-muted);
            background: var(--surface-plate);
            border: 1px solid var(--border-subtle);
            padding: 3px 8px;
            border-radius: var(--radius-control);
            display: inline-flex;
            align-items: center;
            gap: 6px;
            white-space: nowrap;
        }

        .status-dot {
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: var(--signal-positive);
            box-shadow: 0 0 5px rgba(16, 185, 129, 0.5);
        }

        /* NAV TABS */
        .nav-tabs {
            display: flex;
            gap: 4px;
            background: var(--surface-well);
            padding: 3px;
            border-radius: var(--radius-inner);
            border: 1px solid var(--border-subtle);
        }

        .tab-btn {
            background: transparent;
            border: 1px solid transparent;
            font-family: var(--font-sans);
            font-size: 12px;
            font-weight: 500;
            color: var(--text-muted);
            cursor: pointer;
            padding: 6px 14px;
            border-radius: var(--radius-control);
            transition: all 0.15s ease;
            text-decoration: none;
            min-height: 32px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            white-space: nowrap;
        }

        .tab-btn:hover {
            color: var(--text-primary);
            background: var(--surface-hover);
        }

        .tab-btn.active {
            background: var(--surface-plate);
            color: var(--text-primary);
            border-color: var(--border-hairline);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
        }

        /* MAIN CONTAINER */
        main {
            flex: 1;
            max-width: 1360px;
            width: 100%;
            margin: 20px auto;
            padding: 0 28px;
            position: relative;
            z-index: 1;
        }

        .view-panel { display: none; }
        .view-panel.active { display: block; }

        /* IN-APPLICATION NOTIFICATION BANNER */
        .app-notification {
            display: none;
            align-items: flex-start;
            gap: 12px;
            padding: 12px 18px;
            border-radius: var(--radius-inner);
            margin-bottom: 20px;
            font-size: 12px;
            background: var(--surface-plate);
            border: 1px solid var(--border-hairline);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
            animation: slideDown 0.2s ease-out;
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .notification-error {
            background: var(--signal-negative-bg);
            border-color: var(--signal-negative-border);
            color: var(--text-primary);
        }

        .notification-warning {
            background: var(--signal-divergence-bg);
            border-color: var(--signal-divergence-border);
            color: var(--text-primary);
        }

        .notification-icon {
            font-family: var(--font-mono);
            font-weight: 700;
            font-size: 14px;
            line-height: 1.2;
            color: var(--signal-negative);
        }

        .notification-warning .notification-icon {
            color: var(--signal-divergence);
        }

        .notification-body {
            flex: 1;
        }

        .notification-title {
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 2px;
        }

        .notification-msg {
            color: var(--text-body);
        }

        .notification-close {
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 16px;
            line-height: 1;
            padding: 0 4px;
        }

        .notification-close:hover {
            color: var(--text-primary);
        }

        /* RESPONSIBLE AI DOMAIN ADVISORY BANNER */
        .domain-notice {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--surface-chassis);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-inner);
            padding: 12px 18px;
            margin-bottom: 20px;
            font-size: 12px;
            gap: 16px;
        }

        .domain-notice-main {
            display: flex;
            align-items: baseline;
            gap: 10px;
            flex-wrap: wrap;
        }

        .domain-tag {
            font-family: var(--font-mono);
            font-weight: 600;
            font-size: 11px;
            color: var(--signal-neutral);
            letter-spacing: 0.08em;
            text-transform: uppercase;
            white-space: nowrap;
        }

        .domain-desc {
            color: var(--text-muted);
            line-height: 1.45;
        }

        .domain-notice-meta {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
            white-space: nowrap;
        }

        /* GRID LAYOUTS */
        .grid-2col {
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 20px;
            align-items: start;
        }

        /* DOUBLE-BEZEL CARD ARCHITECTURE */
        .bezel-card {
            background: var(--surface-plate);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-outer);
            padding: 1px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        }

        .bezel-core {
            background: var(--surface-chassis);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-inner);
            padding: 22px;
        }

        .card-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
        }

        .card-title {
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-muted);
        }

        /* SAMPLE PRESETS STRIP */
        .preset-strip {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }

        .preset-strip-label {
            font-family: var(--font-mono);
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
        }

        .preset-chip {
            background: var(--surface-well);
            border: 1px solid var(--border-hairline);
            color: var(--text-body);
            font-family: var(--font-sans);
            font-size: 11px;
            font-weight: 500;
            padding: 5px 12px;
            border-radius: var(--radius-control);
            cursor: pointer;
            transition: all 0.15s ease;
            min-height: 28px;
            display: inline-flex;
            align-items: center;
        }

        .preset-chip:hover {
            color: var(--text-primary);
            border-color: var(--border-focus);
            background: var(--surface-hover);
        }

        .preset-chip:active {
            background: var(--surface-plate);
        }

        /* INPUT AREA & TOOLBAR */
        .nlp-textarea {
            width: 100%;
            height: 135px;
            background: var(--surface-well);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-control);
            padding: 14px;
            font-family: var(--font-sans);
            font-size: 14px;
            line-height: 1.5;
            color: var(--text-primary);
            resize: vertical;
            outline: none;
            transition: border-color 0.15s ease;
        }

        .nlp-textarea:focus {
            border-color: var(--border-focus);
        }

        .nlp-textarea::placeholder {
            color: var(--text-disabled);
        }

        .nlp-toolbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 6px;
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
        }

        .nlp-metrics {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .nlp-actions {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .btn-clear {
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-family: var(--font-mono);
            font-size: 11px;
            cursor: pointer;
            padding: 4px 8px;
            border-radius: var(--radius-control);
            transition: all 0.15s ease;
            min-height: 26px;
            display: inline-flex;
            align-items: center;
        }

        .btn-clear:hover {
            color: var(--signal-negative);
            background: var(--signal-negative-bg);
        }

        .key-hint {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-muted);
            background: var(--surface-well);
            border: 1px solid var(--border-subtle);
            padding: 3px 8px;
            border-radius: 4px;
            letter-spacing: 0.02em;
        }

        /* CONTROLS & ENGINE SELECTOR */
        .controls-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 14px;
            gap: 12px;
            flex-wrap: wrap;
        }

        .engine-select {
            background: var(--surface-well);
            border: 1px solid var(--border-hairline);
            padding: 8px 12px;
            font-size: 12px;
            border-radius: var(--radius-control);
            outline: none;
            color: var(--text-primary);
            font-family: var(--font-mono);
            cursor: pointer;
            flex: 1;
            min-width: 280px;
            min-height: 38px;
            transition: border-color 0.15s ease;
        }

        .engine-select:focus {
            border-color: var(--border-focus);
        }

        .engine-select option {
            background: var(--surface-plate);
            color: var(--text-primary);
            font-family: var(--font-mono);
        }

        .btn-primary {
            background: var(--action-primary);
            color: var(--action-primary-text);
            border: none;
            padding: 9px 20px;
            font-family: var(--font-sans);
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.02em;
            border-radius: var(--radius-control);
            cursor: pointer;
            transition: background 0.15s ease, opacity 0.15s ease;
            white-space: nowrap;
            min-height: 38px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }

        .btn-primary:hover:not(:disabled) {
            background: var(--action-primary-hover);
        }

        .btn-primary:active:not(:disabled) {
            background: var(--action-primary-active);
        }

        .btn-primary:disabled, .btn-outline:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .btn-outline {
            background: transparent;
            color: var(--text-body);
            border: 1px solid var(--border-hairline);
            padding: 6px 14px;
            font-size: 11px;
            font-family: var(--font-mono);
            border-radius: var(--radius-control);
            cursor: pointer;
            transition: all 0.15s ease;
            min-height: 32px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            white-space: nowrap;
        }

        .btn-outline:hover:not(:disabled) {
            color: var(--text-primary);
            border-color: var(--border-focus);
            background: var(--surface-hover);
        }

        /* SIGNAL VISUALIZATION SEMANTICS */
        .signal-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 14px;
            margin-bottom: 4px;
        }

        .signal-state-indicator {
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.05em;
            color: var(--signal-neutral);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .signal-state-indicator.state-pos {
            color: var(--signal-positive);
        }

        .signal-state-indicator.state-neg {
            color: var(--signal-negative);
        }

        .signal-state-indicator.state-proc {
            color: var(--border-focus);
        }

        #neuralCanvas {
            width: 100%;
            height: 52px;
            background: transparent;
            border-radius: var(--radius-control);
            border: 1px solid var(--border-subtle);
        }

        .signal-caption {
            font-family: var(--font-sans);
            font-size: 10px;
            line-height: 1.4;
            color: var(--text-muted);
            margin-top: 6px;
            padding: 4px 6px;
            border-left: 2px solid var(--border-hairline);
        }

        /* METRIC HERO DISPLAY */
        .metric-hero {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            padding-bottom: 14px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .metric-primary-block {
            display: flex;
            flex-direction: column;
        }

        .metric-primary-label {
            font-family: var(--font-mono);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-muted);
            margin-bottom: 2px;
        }

        .polarity-score {
            font-family: var(--font-sans);
            font-size: 38px;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.03em;
            font-variant-numeric: tabular-nums;
            line-height: 1.1;
        }

        .polarity-score-caption {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        .badge-sentiment {
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 5px 12px;
            border-radius: var(--radius-pill);
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .badge-pos {
            background: var(--signal-positive-bg);
            color: var(--signal-positive);
            border: 1px solid var(--signal-positive-border);
        }

        .badge-neg {
            background: var(--signal-negative-bg);
            color: var(--signal-negative);
            border: 1px solid var(--signal-negative-border);
        }

        .badge-neu {
            background: var(--signal-neutral-bg);
            color: var(--signal-neutral);
            border: 1px solid var(--signal-neutral-border);
        }

        /* BIPOLAR POLARITY CONTINUUM GAUGE */
        .polarity-gauge-container {
            margin-top: 14px;
            padding-bottom: 14px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .polarity-gauge-header {
            display: flex;
            justify-content: space-between;
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-muted);
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .polarity-track {
            position: relative;
            width: 100%;
            height: 8px;
            background: var(--surface-well);
            border-radius: var(--radius-pill);
            border: 1px solid var(--border-hairline);
            overflow: visible;
        }

        .polarity-neutral-band {
            position: absolute;
            left: 45%;
            width: 10%;
            top: 0;
            bottom: 0;
            background: rgba(148, 163, 184, 0.20);
            border-left: 1px dashed rgba(148, 163, 184, 0.35);
            border-right: 1px dashed rgba(148, 163, 184, 0.35);
        }

        .polarity-center-line {
            position: absolute;
            left: 50%;
            top: -2px;
            bottom: -2px;
            width: 2px;
            background: var(--text-muted);
            transform: translateX(-50%);
        }

        .polarity-indicator {
            position: absolute;
            top: 50%;
            left: 50%;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            background: var(--signal-neutral);
            border: 2px solid var(--surface-chassis);
            box-shadow: 0 0 8px rgba(0, 0, 0, 0.6);
            transition: left 0.3s cubic-bezier(0.2, 0.8, 0.2, 1), background-color 0.2s ease;
            z-index: 2;
        }

        .indicator-pos {
            background: var(--signal-positive);
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
        }

        .indicator-neg {
            background: var(--signal-negative);
            box-shadow: 0 0 8px rgba(244, 63, 94, 0.6);
        }

        .indicator-neu {
            background: var(--signal-neutral);
            box-shadow: 0 0 8px rgba(148, 163, 184, 0.4);
        }

        .polarity-ticks {
            display: flex;
            justify-content: space-between;
            font-family: var(--font-mono);
            font-size: 9px;
            color: var(--text-disabled);
            margin-top: 4px;
        }

        .telemetry-meta-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 14px;
            font-size: 12px;
        }

        .telemetry-meta-label {
            font-family: var(--font-mono);
            font-size: 10px;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 2px;
            letter-spacing: 0.04em;
        }

        .telemetry-meta-val {
            font-weight: 500;
            color: var(--text-primary);
        }

        /* PROPORTIONAL XAI ATTRIBUTION DRIVER BARS */
        .attribution-group {
            margin-top: 12px;
        }

        .attribution-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
            font-size: 11px;
            font-weight: 600;
        }

        .attribution-header-pos {
            color: var(--signal-positive);
        }

        .attribution-header-neg {
            color: var(--signal-negative);
        }

        .attribution-bars-container {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .driver-bar-row {
            display: grid;
            grid-template-columns: 110px 1fr 72px;
            align-items: center;
            gap: 10px;
            background: var(--surface-well);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-control);
            padding: 6px 10px;
            transition: border-color 0.15s ease, background-color 0.15s ease;
            cursor: default;
        }

        .driver-bar-row:hover {
            border-color: var(--border-hairline);
            background: var(--surface-hover);
        }

        .driver-token {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .driver-bar-track {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: var(--radius-pill);
            overflow: hidden;
            position: relative;
        }

        .driver-bar-fill {
            height: 100%;
            border-radius: var(--radius-pill);
            transition: width 0.25s ease-out;
        }

        .driver-bar-fill-pos {
            background: var(--signal-positive);
            box-shadow: 0 0 6px rgba(16, 185, 129, 0.35);
        }

        .driver-bar-fill-neg {
            background: var(--signal-negative);
            box-shadow: 0 0 6px rgba(244, 63, 94, 0.35);
        }

        .driver-val {
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 600;
            text-align: right;
            font-variant-numeric: tabular-nums;
        }

        .driver-val-pos {
            color: var(--signal-positive);
        }

        .driver-val-neg {
            color: var(--signal-negative);
        }

        .driver-empty-hint {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
            padding: 8px 12px;
            background: var(--surface-well);
            border: 1px dashed var(--border-subtle);
            border-radius: var(--radius-control);
        }

        /* CANONICAL 2X2 CONFUSION MATRIX */
        .matrix-table-wrapper {
            margin-top: 16px;
            overflow-x: auto;
        }

        .matrix-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 4px;
            font-size: 12px;
        }

        .matrix-table th, .matrix-table td {
            padding: 12px 14px;
            text-align: center;
            border-radius: var(--radius-control);
        }

        .matrix-corner {
            background: transparent;
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            text-align: left !important;
        }

        .matrix-header-cell {
            background: var(--surface-plate);
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 600;
            color: var(--text-primary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border: 1px solid var(--border-subtle);
        }

        .matrix-row-header {
            background: var(--surface-plate);
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 600;
            color: var(--text-primary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            text-align: left !important;
            border: 1px solid var(--border-subtle);
        }

        .matrix-quadrant {
            background: var(--surface-well);
            border: 1px solid var(--border-hairline);
            text-align: center;
            vertical-align: middle;
            transition: all 0.15s ease;
        }

        .quadrant-tn {
            background: rgba(16, 185, 129, 0.05);
            border-color: rgba(16, 185, 129, 0.2);
        }

        .quadrant-tp {
            background: rgba(16, 185, 129, 0.05);
            border-color: rgba(16, 185, 129, 0.2);
        }

        .quadrant-fp {
            background: rgba(244, 63, 94, 0.05);
            border-color: rgba(244, 63, 94, 0.2);
        }

        .quadrant-fn {
            background: rgba(244, 63, 94, 0.05);
            border-color: rgba(244, 63, 94, 0.2);
        }

        .matrix-quadrant-title {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }

        .matrix-quadrant-num {
            font-family: var(--font-sans);
            font-size: 20px;
            font-weight: 700;
            color: var(--text-primary);
            font-variant-numeric: tabular-nums;
        }

        .quadrant-tn .matrix-quadrant-num, .quadrant-tp .matrix-quadrant-num {
            color: var(--signal-positive);
        }

        .quadrant-fp .matrix-quadrant-num, .quadrant-fn .matrix-quadrant-num {
            color: var(--signal-negative);
        }

        .matrix-quadrant-pct {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-body);
            margin-top: 2px;
        }

        .matrix-total-cell {
            background: var(--surface-plate);
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 600;
            color: var(--text-body);
            border: 1px solid var(--border-subtle);
        }

        .matrix-rates-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-top: 14px;
        }

        .rate-card {
            background: var(--surface-well);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-control);
            padding: 10px 14px;
        }

        .rate-label {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .rate-value {
            font-family: var(--font-sans);
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            margin-top: 2px;
            font-variant-numeric: tabular-nums;
        }

        /* MODEL SWITCHER TABS IN BENCHMARKS */
        .model-toggle-strip {
            display: flex;
            gap: 6px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }

        .model-toggle-btn {
            background: var(--surface-well);
            border: 1px solid var(--border-hairline);
            color: var(--text-muted);
            font-family: var(--font-mono);
            font-size: 11px;
            padding: 6px 12px;
            border-radius: var(--radius-control);
            cursor: pointer;
            transition: all 0.15s ease;
            min-height: 32px;
            display: inline-flex;
            align-items: center;
        }

        .model-toggle-btn:hover {
            color: var(--text-primary);
            border-color: var(--border-focus);
        }

        .model-toggle-btn.active {
            background: var(--surface-plate);
            color: var(--text-primary);
            border-color: var(--action-primary);
        }

        /* TABLES */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }

        th, td {
            text-align: left;
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-subtle);
            color: var(--text-body);
        }

        th {
            font-family: var(--font-mono);
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            background: var(--surface-plate);
        }

        td strong {
            color: var(--text-primary);
            font-weight: 500;
        }

        /* STATS GRID */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 20px;
        }

        .stat-card {
            background: var(--surface-plate);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-inner);
            padding: 16px;
        }

        .stat-num {
            font-family: var(--font-sans);
            font-size: 28px;
            font-weight: 600;
            color: var(--text-primary);
            margin-top: 4px;
            font-variant-numeric: tabular-nums;
        }

        /* TELEMETRY TOOLBAR & CONTROLS */
        .telemetry-toolbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
            flex-wrap: wrap;
            background: var(--surface-well);
            padding: 8px 12px;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-inner);
        }

        .filter-sentiment-group {
            display: flex;
            gap: 4px;
            background: var(--surface-chassis);
            padding: 3px;
            border-radius: var(--radius-control);
            border: 1px solid var(--border-subtle);
        }

        .filter-btn {
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-muted);
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 500;
            padding: 5px 12px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s ease;
            min-height: 28px;
            display: inline-flex;
            align-items: center;
        }

        .filter-btn:hover {
            color: var(--text-primary);
            background: var(--surface-hover);
        }

        .filter-btn.active {
            background: var(--surface-plate);
            color: var(--text-primary);
            border-color: var(--border-hairline);
            box-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }

        .filter-controls-group {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
            flex: 1;
            justify-content: flex-end;
        }

        .filter-select {
            background: var(--surface-chassis);
            border: 1px solid var(--border-hairline);
            color: var(--text-primary);
            font-family: var(--font-mono);
            font-size: 11px;
            padding: 6px 12px;
            border-radius: var(--radius-control);
            outline: none;
            cursor: pointer;
            min-height: 32px;
        }

        .filter-select:focus {
            border-color: var(--border-focus);
        }

        .filter-select option {
            background: var(--surface-plate);
            color: var(--text-primary);
        }

        .filter-search-input {
            background: var(--surface-chassis);
            border: 1px solid var(--border-hairline);
            color: var(--text-primary);
            font-family: var(--font-sans);
            font-size: 12px;
            padding: 6px 12px;
            border-radius: var(--radius-control);
            outline: none;
            min-width: 200px;
            min-height: 32px;
            transition: border-color 0.15s ease;
        }

        .filter-search-input:focus {
            border-color: var(--border-focus);
        }

        .filter-search-input::placeholder {
            color: var(--text-disabled);
            font-size: 11px;
        }

        /* EXPANDABLE TELEMETRY ROW / DRAWER */
        .telemetry-row {
            cursor: pointer;
            transition: background 0.12s ease;
        }

        .telemetry-row:hover {
            background: var(--surface-hover);
        }

        .record-drawer {
            display: none;
            background: var(--surface-well);
        }

        .record-drawer.open {
            display: table-row;
        }

        .drawer-content-box {
            padding: 14px 18px;
            border-bottom: 1px solid var(--border-subtle);
            font-family: var(--font-mono);
            font-size: 11px;
        }

        .drawer-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px 20px;
        }

        .drawer-field-label {
            color: var(--text-muted);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 2px;
        }

        .drawer-field-val {
            color: var(--text-primary);
            font-weight: 500;
        }

        /* MOBILE STACKED TELEMETRY CARDS (< 768px) */
        .desktop-telemetry-view {
            display: block;
        }

        .mobile-telemetry-view {
            display: none;
        }

        .telemetry-card {
            background: var(--surface-well);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-inner);
            padding: 14px 16px;
            transition: border-color 0.15s ease;
        }

        .telemetry-card:hover {
            border-color: var(--border-focus);
        }

        .card-top-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .card-time {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
        }

        .card-engine-tag {
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-body);
            background: var(--surface-plate);
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid var(--border-subtle);
        }

        .card-metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            padding: 10px 0;
            border-top: 1px solid var(--border-subtle);
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 10px;
        }

        .card-metric-cell {
            display: flex;
            flex-direction: column;
        }

        .card-metric-cell .cell-label {
            font-family: var(--font-mono);
            font-size: 9px;
            text-transform: uppercase;
            color: var(--text-muted);
            letter-spacing: 0.04em;
        }

        .card-metric-cell .cell-val {
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 600;
            color: var(--text-primary);
            margin-top: 2px;
            font-variant-numeric: tabular-nums;
        }

        .card-expand-btn {
            width: 100%;
            background: transparent;
            border: 1px dashed var(--border-subtle);
            border-radius: var(--radius-control);
            color: var(--text-muted);
            font-family: var(--font-mono);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 8px 12px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.15s ease;
            min-height: 36px;
        }

        .card-expand-btn:hover {
            color: var(--text-primary);
            border-color: var(--border-hairline);
            background: var(--surface-hover);
        }

        .card-drawer-content {
            display: none;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px dashed var(--border-subtle);
            font-family: var(--font-mono);
            font-size: 11px;
        }

        .card-drawer-content.open {
            display: block;
        }

        .mobile-drawer-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px 14px;
        }

        /* EMPTY STATE */
        .telemetry-empty-state {
            display: none;
            text-align: center;
            padding: 36px 20px;
            background: var(--surface-well);
            border: 1px dashed var(--border-subtle);
            border-radius: var(--radius-inner);
            margin-top: 10px;
        }

        .empty-icon {
            font-family: var(--font-mono);
            font-size: 24px;
            color: var(--text-muted);
            margin-bottom: 8px;
            display: block;
        }

        .empty-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 4px;
        }

        .empty-desc {
            font-size: 11px;
            color: var(--text-muted);
            max-width: 440px;
            margin: 0 auto;
        }

        /* METHODOLOGY NOTE */
        .method-note {
            background: var(--surface-well);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-control);
            padding: 12px 14px;
            font-size: 12px;
            line-height: 1.5;
            color: var(--text-muted);
            margin-top: 14px;
        }

        .method-note strong {
            color: var(--text-primary);
        }

        /* ACCESSIBILITY FOCUS RING */
        *:focus-visible {
            outline: 2px solid var(--border-focus) !important;
            outline-offset: 2px !important;
        }

        @media (prefers-reduced-motion: reduce) {
            * {
                animation-duration: 0.001ms !important;
                transition-duration: 0.001ms !important;
            }
        }

        /* RESPONSIVE LAYOUT BREAKPOINTS (1024px, 768px, 640px, 390px) */
        @media (max-width: 1024px) {
            .grid-2col { grid-template-columns: 1fr; }
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
            main { padding: 0 20px; margin: 16px auto; }
            header { padding: 14px 20px; }
            .matrix-rates-grid { grid-template-columns: repeat(2, 1fr); }
            .controls-bar { flex-direction: column; align-items: stretch; }
            .engine-select { min-width: 100%; }
        }

        @media (max-width: 768px) {
            .desktop-telemetry-view { display: none !important; }
            .mobile-telemetry-view { display: flex !important; flex-direction: column; gap: 10px; }
            .telemetry-toolbar { flex-direction: column; align-items: stretch; }
            .filter-sentiment-group { width: 100%; justify-content: space-between; }
            .filter-btn { flex: 1; text-align: center; min-height: 38px; }
            .filter-controls-group { width: 100%; justify-content: stretch; }
            .filter-select, .filter-search-input, #btnRefreshHistory { width: 100%; min-height: 38px; }
            .domain-notice { flex-direction: column; align-items: flex-start; gap: 8px; }
            .tab-btn { min-height: 38px; }
            .preset-chip { min-height: 36px; }
            .system-meta-strip { border-left: none; padding-left: 0; }
        }

        @media (max-width: 640px) {
            .stats-grid { grid-template-columns: 1fr; }
            .telemetry-meta-row { grid-template-columns: 1fr; }
            .brand-block { flex-direction: column; align-items: flex-start; gap: 8px; }
            .nav-tabs { overflow-x: auto; max-width: 100%; }
            .matrix-rates-grid { grid-template-columns: 1fr; }
            .driver-bar-row { grid-template-columns: 85px 1fr 65px; gap: 6px; padding: 5px 8px; }
            .preset-strip { flex-direction: column; align-items: flex-start; }
            .card-metrics-grid { grid-template-columns: repeat(2, 1fr); }
        }

        @media (max-width: 390px) {
            main { padding: 0 12px; margin: 12px auto; }
            header { padding: 10px 14px; }
            .bezel-core { padding: 14px; }
            .polarity-score { font-size: 32px; }
            .mobile-drawer-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="ambient-layer"></div>

    <header>
        <div class="brand-block">
            <div class="brand-identity">
                <div class="brand-name">Sentinel AI</div>
                <div class="brand-subtitle">NLP Sentiment Intelligence Platform</div>
            </div>
            <div class="system-meta-strip" aria-label="System Environment Telemetry">
                <span class="meta-pill" title="Gateway Architecture: FastAPI"><span class="status-dot"></span> System: Operational</span>
                <span class="meta-pill" title="Loaded production models in memory">Engines: 3 Ready</span>
                <span class="meta-pill" title="Training & Calibration Baseline">Corpus: IMDB 50K</span>
                <span class="meta-pill" title="Platform Build Release">v14.3.0-PROD</span>
            </div>
        </div>
        <nav class="nav-tabs" role="tablist" aria-label="Platform Workspaces">
            <button class="tab-btn active" role="tab" id="tab-analyze" aria-selected="true" aria-controls="view-analyze" onclick="switchTab('analyze')">Inference Console</button>
            <button class="tab-btn" role="tab" id="tab-consensus" aria-selected="false" aria-controls="view-consensus" onclick="switchTab('consensus')">Model Consensus</button>
            <button class="tab-btn" role="tab" id="tab-benchmarks" aria-selected="false" aria-controls="view-benchmarks" onclick="switchTab('benchmarks')">Model Registry &amp; Benchmarks</button>
            <button class="tab-btn" role="tab" id="tab-telemetry" aria-selected="false" aria-controls="view-telemetry" onclick="switchTab('telemetry')">Telemetry &amp; History</button>
            <button class="tab-btn" role="tab" id="tab-developer" aria-selected="false" aria-controls="view-developer" onclick="switchTab('developer')">API Interface</button>
        </nav>
    </header>

    <main>
        <!-- IN-APP NOTIFICATION BANNER (ACCESSIBLE ERROR & NOTICE CHANNEL) -->
        <div id="appNotification" class="app-notification" role="alert" aria-live="polite">
            <div class="notification-icon" id="notificationIcon">!</div>
            <div class="notification-body">
                <div class="notification-title" id="notificationTitle">System Notice</div>
                <div class="notification-msg" id="notificationMessage">Notification content</div>
            </div>
            <button type="button" class="notification-close" onclick="dismissNotification()" aria-label="Dismiss notification">&times;</button>
        </div>

        <!-- RESPONSIBLE AI DOMAIN ADVISORY -->
        <div class="domain-notice" role="region" aria-label="Responsible AI Domain Advisory">
            <div class="domain-notice-main">
                <span class="domain-tag">Analysis Domain: IMDB / Movie Reviews</span>
                <span class="domain-desc">
                    Classifiers are trained and calibrated on narrative discourse corpus. Operational generalization to financial, clinical, or legal texts may exhibit statistical distribution shift. Sentiment classifications represent probabilistic model outputs rather than universal truth.
                </span>
            </div>
            <div class="domain-notice-meta">
                50,000 Verified Ground Truth Corpus
            </div>
        </div>

        <!-- TAB 1: INFERENCE CONSOLE -->
        <section id="view-analyze" class="view-panel active" role="tabpanel" aria-labelledby="tab-analyze">
            <div class="grid-2col">
                <div class="bezel-card">
                    <div class="bezel-core" style="display:flex; flex-direction:column; justify-content:space-between; min-height:480px;">
                        <div>
                            <!-- EVALUATION SAMPLE PRESETS -->
                            <div class="preset-strip" role="group" aria-label="Evaluation Sample Presets">
                                <span class="preset-strip-label">Evaluation Presets:</span>
                                <button type="button" class="preset-chip" onclick="applyPreset('pos')" aria-label="Load Positive Audit preset" title="Predefined demonstration review with positive sentiment">Positive Audit</button>
                                <button type="button" class="preset-chip" onclick="applyPreset('neg')" aria-label="Load Critical Discontent preset" title="Predefined demonstration review with critical sentiment">Critical Discontent</button>
                                <button type="button" class="preset-chip" onclick="applyPreset('mixed')" aria-label="Load Mixed Consensus Test preset" title="Predefined demonstration review with mixed sentiment">Mixed Consensus Test</button>
                            </div>

                            <div class="card-header-row" style="margin-bottom:8px;">
                                <span class="card-title">Natural Language Ingestion Console</span>
                                <span class="key-hint" title="Submit analysis via keyboard shortcut">Ctrl + Enter / &#8984; + Enter</span>
                            </div>

                            <textarea id="nlpInput" class="nlp-textarea" placeholder="Submit natural language review or statement for executive sentiment intelligence extraction..."></textarea>
                            
                            <!-- TOOLBAR WITH CHAR & WORD COUNTERS AND CLEAR ACTION -->
                            <div class="nlp-toolbar">
                                <div class="nlp-metrics">
                                    <span id="charCount">0 / 5000 chars</span>
                                    <span>&bull;</span>
                                    <span id="wordCount">0 words</span>
                                </div>
                                <div class="nlp-actions">
                                    <button type="button" onclick="clearInput()" class="btn-clear" aria-label="Clear input text">&times; Clear</button>
                                </div>
                            </div>
                        </div>

                        <!-- OPERATIONAL STATE TRACE -->
                        <div>
                            <div class="signal-header-row">
                                <span class="card-title" style="margin-bottom:0;">Operational State Trace</span>
                                <span id="signalStatus" class="signal-state-indicator" aria-live="polite">
                                    <span class="status-dot" style="width:5px; height:5px; background:var(--signal-neutral); box-shadow:none;"></span>
                                    <span>STANDBY</span>
                                </span>
                            </div>
                            <canvas id="neuralCanvas"></canvas>
                            <div class="signal-caption">
                                Operational interface state visualizer (STANDBY / ANALYZING / CLASSIFIED). Waveform trace is not an acoustic signal or model statistic. Authoritative metrics are the Polarity Score and Confidence values above.
                            </div>
                        </div>

                        <!-- INSTRUMENTATION ENGINE SELECTOR -->
                        <div class="controls-bar">
                            <select id="modelChoice" class="engine-select" aria-label="Classifier Engine Selection" onchange="onModelChoiceChange()">
                                <option value="logistic_regression">Logistic Regression v1.0 &bull; Active &bull; 0.0008 ms/sample</option>
                                <option value="linear_svm">Linear SVM v1.0 (Calibrated) &bull; Ready &bull; 0.0022 ms/sample</option>
                                <option value="naive_bayes">Multinomial Naive Bayes &bull; Baseline &bull; 0.0005 ms/sample</option>
                            </select>
                            <button id="execBtn" onclick="executeInference()" class="btn-primary">Execute Analysis</button>
                        </div>
                    </div>
                </div>

                <div style="display:flex; flex-direction:column; gap:16px;">
                    <!-- CLASSIFICATION FINDING & BIPOLAR GAUGE CARD -->
                    <div class="bezel-card">
                        <div class="bezel-core" aria-live="polite">
                            <div class="card-header-row">
                                <span class="card-title">Classification Finding &amp; State</span>
                                <span id="resConfidence" style="font-family:var(--font-mono); font-size:12px; color:var(--text-muted); font-weight:500;">Confidence: --%</span>
                            </div>

                            <div class="metric-hero">
                                <div class="metric-primary-block">
                                    <span class="metric-primary-label">Polarity Score</span>
                                    <div class="polarity-score" id="resScore">0.00</div>
                                    <span class="polarity-score-caption">Bounded Range [-1.0000, +1.0000]</span>
                                </div>
                                <div class="badge-sentiment badge-neu" id="resSentiment">STANDBY</div>
                            </div>

                            <!-- BIPOLAR POLARITY CONTINUUM GAUGE -->
                            <div class="polarity-gauge-container">
                                <div class="polarity-gauge-header">
                                    <span>Negative (-1.0)</span>
                                    <span>Neutral Threshold [&plusmn;0.10]</span>
                                    <span>Positive (+1.0)</span>
                                </div>
                                <div class="polarity-track">
                                    <div class="polarity-neutral-band" title="Neutral classification threshold [-0.10, +0.10]"></div>
                                    <div class="polarity-center-line" title="Equilibrium center (0.00)"></div>
                                    <div class="polarity-indicator indicator-neu" id="polarityIndicator" style="left: 50%;" title="Polarity position"></div>
                                </div>
                                <div class="polarity-ticks">
                                    <span>-1.0</span>
                                    <span>-0.5</span>
                                    <span>-0.1</span>
                                    <span style="color:var(--text-muted);">0.0</span>
                                    <span>+0.1</span>
                                    <span>+0.5</span>
                                    <span>+1.0</span>
                                </div>
                            </div>

                            <div class="telemetry-meta-row">
                                <div>
                                    <div class="telemetry-meta-label">Affective Heuristic</div>
                                    <div class="telemetry-meta-val" id="resEmotion">--</div>
                                </div>
                                <div>
                                    <div class="telemetry-meta-label">Ingestion Latency</div>
                                    <div class="telemetry-meta-val" id="resLatency">-- ms</div>
                                </div>
                                <div>
                                    <div class="telemetry-meta-label">Tracking ID</div>
                                    <div class="telemetry-meta-val" id="resReqId" style="font-family:var(--font-mono); font-size:11px;">--</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- FEATURE ATTRIBUTION (XAI) CARD -->
                    <div class="bezel-card">
                        <div class="bezel-core">
                            <div class="card-header-row">
                                <span class="card-title">Feature Contribution &amp; Attribution (XAI)</span>
                                <span id="xaiEngineLabel" style="font-family:var(--font-mono); font-size:10px; color:var(--text-muted);">TF-IDF &times; Weights</span>
                            </div>

                            <div class="attribution-group">
                                <div class="attribution-header attribution-header-pos">
                                    <span>Positive Drivers (+w)</span>
                                    <span style="font-family:var(--font-mono); font-size:10px; color:var(--text-muted);">Normalized Impact</span>
                                </div>
                                <div class="attribution-bars-container" id="posDrivers">
                                    <div class="driver-empty-hint">Awaiting input...</div>
                                </div>
                            </div>

                            <div class="attribution-group">
                                <div class="attribution-header attribution-header-neg">
                                    <span>Negative Drivers (-w)</span>
                                    <span style="font-family:var(--font-mono); font-size:10px; color:var(--text-muted);">Normalized Impact</span>
                                </div>
                                <div class="attribution-bars-container" id="negDrivers">
                                    <div class="driver-empty-hint">Awaiting input...</div>
                                </div>
                            </div>

                            <div class="method-note">
                                <strong>Mathematical Methodology:</strong> Features are extracted via Porter stemming and 5,000-term TF-IDF vectorization. Statistical attribution is calculated strictly as \( \text{TF-IDF} \times w_i \) where \( w_i \) represents model weight coefficients (or log odds ratios for Naive Bayes). Statistical contribution indicates feature influence within the decision boundary and does not represent causal necessity.
                            </div>
                        </div>
                    </div>

                    <!-- OPERATOR AUDIT VERIFICATION -->
                    <div class="bezel-card">
                        <div class="bezel-core" style="padding: 14px 20px;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-size:12px; color:var(--text-muted);">Operator Audit Verification</span>
                                <span id="feedConfirm" style="display:none; font-family:var(--font-mono); font-size:11px; color:var(--signal-positive);">RECORDED &check;</span>
                                <div style="display:flex; gap:8px;">
                                    <button id="btnFeedConfirm" onclick="sendTelemetry(true)" class="btn-outline">Confirm Accurate</button>
                                    <button id="btnFeedDispute" onclick="sendTelemetry(false)" class="btn-outline">Flag Disputed</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <input type="hidden" id="currentPredId">
        </section>

        <!-- TAB 2: MODEL CONSENSUS -->
        <section id="view-consensus" class="view-panel" role="tabpanel" aria-labelledby="tab-consensus">
            <div class="bezel-card" style="margin-bottom:20px;">
                <div class="bezel-core">
                    <div class="card-header-row">
                        <div>
                            <div class="card-title">Multi-Model Consensus Evaluation Engine</div>
                            <p style="font-size:12px; color:var(--text-muted); margin-top:4px;">Runs all 3 production classifiers concurrently to detect alignment or epistemic model disagreement.</p>
                        </div>
                        <button id="btnConsensusRun" onclick="executeConsensus()" class="btn-primary">Run All 3 Models</button>
                    </div>

                    <div id="consensusBanner" style="display:none; padding:12px 18px; border-radius:var(--radius-inner); margin-top:14px; font-family:var(--font-mono); font-size:12px;"></div>

                    <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; margin-top:16px;">
                        <div class="bezel-card">
                            <div class="bezel-core" style="padding:16px;">
                                <div style="font-family:var(--font-mono); font-size:10px; color:var(--text-muted); text-transform:uppercase;">Logistic Regression v1.0</div>
                                <div style="font-size:10px; color:var(--text-disabled); font-family:var(--font-mono); margin-top:1px;">L2 Regularized GLM</div>
                                <div style="font-size:22px; font-weight:600; color:var(--text-primary); margin-top:6px;" id="cs-lr-sentiment">--</div>
                                <div style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted); margin-top:4px;" id="cs-lr-meta">Score: -- | Conf: --%</div>
                            </div>
                        </div>
                        <div class="bezel-card">
                            <div class="bezel-core" style="padding:16px;">
                                <div style="font-family:var(--font-mono); font-size:10px; color:var(--text-muted); text-transform:uppercase;">Linear SVM v1.0</div>
                                <div style="font-size:10px; color:var(--text-disabled); font-family:var(--font-mono); margin-top:1px;">Platt Calibrated SVC</div>
                                <div style="font-size:22px; font-weight:600; color:var(--text-primary); margin-top:6px;" id="cs-svm-sentiment">--</div>
                                <div style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted); margin-top:4px;" id="cs-svm-meta">Score: -- | Conf: --%</div>
                            </div>
                        </div>
                        <div class="bezel-card">
                            <div class="bezel-core" style="padding:16px;">
                                <div style="font-family:var(--font-mono); font-size:10px; color:var(--text-muted); text-transform:uppercase;">Multinomial Naive Bayes</div>
                                <div style="font-size:10px; color:var(--text-disabled); font-family:var(--font-mono); margin-top:1px;">Log-Odds Probabilistic</div>
                                <div style="font-size:22px; font-weight:600; color:var(--text-primary); margin-top:6px;" id="cs-nb-sentiment">--</div>
                                <div style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted); margin-top:4px;" id="cs-nb-meta">Score: -- | Conf: --%</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- TAB 3: MODEL REGISTRY & BENCHMARKS -->
        <section id="view-benchmarks" class="view-panel" role="tabpanel" aria-labelledby="tab-benchmarks">
            <div class="bezel-card" style="margin-bottom:20px;">
                <div class="bezel-core">
                    <div class="card-header-row">
                        <div>
                            <div class="card-title">Standardized Evaluation Benchmarks</div>
                            <p style="font-size:12px; color:var(--text-muted); margin-top:4px;">Evaluated on 10,000 independent test partition samples from the IMDB 50K benchmark corpus (80/20 train/test split, random seed 42).</p>
                        </div>
                    </div>

                    <table>
                        <thead>
                            <tr>
                                <th>Model Architecture</th>
                                <th>Status</th>
                                <th>Accuracy</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1 Score</th>
                                <th>ROC AUC</th>
                                <th>Latency / Sample</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Logistic Regression v1.0</strong> (L2 Reg)</td>
                                <td><span style="font-family:var(--font-mono); font-size:10px; padding:2px 8px; border-radius:var(--radius-pill); background:var(--signal-positive-bg); color:var(--signal-positive); border:1px solid var(--signal-positive-border);">ACTIVE</span></td>
                                <td style="font-family:var(--font-mono);">88.55%</td>
                                <td style="font-family:var(--font-mono);">0.8747</td>
                                <td style="font-family:var(--font-mono);">0.9020</td>
                                <td style="font-family:var(--font-mono);">0.8881</td>
                                <td style="font-family:var(--font-mono);">0.9571</td>
                                <td style="font-family:var(--font-mono);">0.0008 ms</td>
                            </tr>
                            <tr>
                                <td><strong>Linear SVM v1.0</strong> (Platt Calibrated)</td>
                                <td><span style="font-family:var(--font-mono); font-size:10px; padding:2px 8px; border-radius:var(--radius-pill); background:var(--signal-positive-bg); color:var(--signal-positive); border:1px solid var(--signal-positive-border);">READY</span></td>
                                <td style="font-family:var(--font-mono);">88.40%</td>
                                <td style="font-family:var(--font-mono);">0.8751</td>
                                <td style="font-family:var(--font-mono);">0.8980</td>
                                <td style="font-family:var(--font-mono);">0.8864</td>
                                <td style="font-family:var(--font-mono);">0.9534</td>
                                <td style="font-family:var(--font-mono);">0.0022 ms</td>
                            </tr>
                            <tr>
                                <td><strong>Multinomial Naive Bayes</strong> (Probabilistic)</td>
                                <td><span style="font-family:var(--font-mono); font-size:10px; padding:2px 8px; border-radius:var(--radius-pill); background:var(--signal-positive-bg); color:var(--signal-positive); border:1px solid var(--signal-positive-border);">READY</span></td>
                                <td style="font-family:var(--font-mono);">85.20%</td>
                                <td style="font-family:var(--font-mono);">0.8654</td>
                                <td style="font-family:var(--font-mono);">0.8330</td>
                                <td style="font-family:var(--font-mono);">0.8489</td>
                                <td style="font-family:var(--font-mono);">0.9272</td>
                                <td style="font-family:var(--font-mono);">0.0005 ms</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- CANONICAL 2x2 CONFUSION MATRIX -->
            <div class="bezel-card">
                <div class="bezel-core">
                    <div class="card-header-row">
                        <div>
                            <div class="card-title">Canonical 2 &times; 2 Confusion Matrix &amp; Diagnostic Rates</div>
                            <p style="font-size:12px; color:var(--text-muted); margin-top:4px;">
                                Rigorous cross-tabulation of ground-truth test partition (10,000 samples) vs classifier decisions.
                            </p>
                        </div>
                        <span id="cmActiveModelLabel" style="font-family:var(--font-mono); font-size:11px; color:var(--action-primary); font-weight:600;">
                            Logistic Regression v1.0
                        </span>
                    </div>

                    <!-- MODEL SWITCHER -->
                    <div class="model-toggle-strip" role="group" aria-label="Confusion Matrix Model Switcher">
                        <button type="button" class="model-toggle-btn active" onclick="switchMatrixModel('logistic_regression', this)">Logistic Regression</button>
                        <button type="button" class="model-toggle-btn" onclick="switchMatrixModel('linear_svm', this)">Linear SVM</button>
                        <button type="button" class="model-toggle-btn" onclick="switchMatrixModel('naive_bayes', this)">Multinomial NB</button>
                    </div>

                    <!-- 2x2 TABLE -->
                    <div class="matrix-table-wrapper">
                        <table class="matrix-table" aria-label="Canonical 2x2 Confusion Matrix">
                            <thead>
                                <tr>
                                    <th class="matrix-corner" scope="col">Actual \ Predicted</th>
                                    <th class="matrix-header-cell" scope="col">Predicted Negative</th>
                                    <th class="matrix-header-cell" scope="col">Predicted Positive</th>
                                    <th class="matrix-header-cell" scope="col">Total Actual</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <th class="matrix-row-header" scope="row">Actual Negative</th>
                                    <td class="matrix-quadrant quadrant-tn">
                                        <div class="matrix-quadrant-title">True Negative (TN)</div>
                                        <div class="matrix-quadrant-num" id="val-tn">4,310</div>
                                        <div class="matrix-quadrant-pct" id="pct-tn">43.10%</div>
                                    </td>
                                    <td class="matrix-quadrant quadrant-fp">
                                        <div class="matrix-quadrant-title">False Positive (FP)</div>
                                        <div class="matrix-quadrant-num" id="val-fp">651</div>
                                        <div class="matrix-quadrant-pct" id="pct-fp">6.51% (Type I)</div>
                                    </td>
                                    <td class="matrix-total-cell" id="cm-total-neg">4,961</td>
                                </tr>
                                <tr>
                                    <th class="matrix-row-header" scope="row">Actual Positive</th>
                                    <td class="matrix-quadrant quadrant-fn">
                                        <div class="matrix-quadrant-title">False Negative (FN)</div>
                                        <div class="matrix-quadrant-num" id="val-fn">494</div>
                                        <div class="matrix-quadrant-pct" id="pct-fn">4.94% (Type II)</div>
                                    </td>
                                    <td class="matrix-quadrant quadrant-tp">
                                        <div class="matrix-quadrant-title">True Positive (TP)</div>
                                        <div class="matrix-quadrant-num" id="val-tp">4,545</div>
                                        <div class="matrix-quadrant-pct" id="pct-tp">45.45%</div>
                                    </td>
                                    <td class="matrix-total-cell" id="cm-total-pos">5,039</td>
                                </tr>
                                <tr>
                                    <th class="matrix-row-header" scope="row">Total Predicted</th>
                                    <td class="matrix-total-cell" id="cm-col-neg">4,804</td>
                                    <td class="matrix-total-cell" id="cm-col-pos">5,196</td>
                                    <td class="matrix-total-cell" style="color:var(--text-primary); font-weight:700;" id="cm-grand-total">10,000</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <!-- ANALYTICAL RATES -->
                    <div class="matrix-rates-grid">
                        <div class="rate-card">
                            <div class="rate-label">Sensitivity / Recall (TP Rate)</div>
                            <div class="rate-value" id="cm-sensitivity">90.20%</div>
                        </div>
                        <div class="rate-card">
                            <div class="rate-label">Specificity (TN Rate)</div>
                            <div class="rate-value" id="cm-specificity">86.88%</div>
                        </div>
                        <div class="rate-card">
                            <div class="rate-label">Precision / PPV</div>
                            <div class="rate-value" id="cm-precision">87.47%</div>
                        </div>
                        <div class="rate-card">
                            <div class="rate-label">Overall Accuracy</div>
                            <div class="rate-value" id="cm-accuracy">88.55%</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- TAB 4: TELEMETRY & HISTORY -->
        <section id="view-telemetry" class="view-panel" role="tabpanel" aria-labelledby="tab-telemetry">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="card-title">Total Analyses Ingested</div>
                    <div class="stat-num" id="stTotal">0</div>
                </div>
                <div class="stat-card">
                    <div class="card-title">Positive Signals</div>
                    <div class="stat-num" style="color:var(--signal-positive);" id="stPos">0</div>
                </div>
                <div class="stat-card">
                    <div class="card-title">Negative Signals</div>
                    <div class="stat-num" style="color:var(--signal-negative);" id="stNeg">0</div>
                </div>
                <div class="stat-card">
                    <div class="card-title">Latency P95 Percentile</div>
                    <div class="stat-num" id="stP95">0.00 <span style="font-size:14px; color:var(--text-muted);">ms</span></div>
                </div>
            </div>

            <div class="bezel-card" style="margin-bottom:20px;">
                <div class="bezel-core">
                    <div class="card-header-row" style="margin-bottom:12px;">
                        <div>
                            <div class="card-title">Operational Telemetry Records</div>
                            <p style="font-size:12px; color:var(--text-muted); margin-top:4px;">
                                Chronological audit ledger. Click any record row or card to inspect forensic details.
                            </p>
                        </div>
                    </div>

                    <!-- TELEMETRY FILTERING TOOLBAR -->
                    <div class="telemetry-toolbar">
                        <div class="filter-sentiment-group" role="group" aria-label="Filter by Classification Sentiment">
                            <button type="button" class="filter-btn active" data-sentiment="ALL" onclick="setSentimentFilter('ALL', this)">All</button>
                            <button type="button" class="filter-btn" data-sentiment="Positive" onclick="setSentimentFilter('Positive', this)">Positive</button>
                            <button type="button" class="filter-btn" data-sentiment="Negative" onclick="setSentimentFilter('Negative', this)">Negative</button>
                            <button type="button" class="filter-btn" data-sentiment="Neutral" onclick="setSentimentFilter('Neutral', this)">Neutral</button>
                        </div>

                        <div class="filter-controls-group">
                            <select id="filterEngine" class="filter-select" aria-label="Filter by Classifier Model" onchange="applyTelemetryFilters()">
                                <option value="ALL">All Classifier Models</option>
                                <option value="logistic_regression">Logistic Regression</option>
                                <option value="linear_svm">Linear SVM</option>
                                <option value="naive_bayes">Multinomial Naive Bayes</option>
                            </select>

                            <input type="search" id="filterSearch" class="filter-search-input" placeholder="Search by Request ID, emotion, engine..." aria-label="Search telemetry records" oninput="applyTelemetryFilters()">

                            <button type="button" id="btnRefreshHistory" onclick="refreshHistory()" class="btn-outline" aria-label="Refresh operational telemetry records">Refresh Log</button>
                        </div>
                    </div>

                    <!-- ACCESSIBLE EMPTY STATE -->
                    <div id="telemetryEmptyState" class="telemetry-empty-state" role="status" aria-live="polite">
                        <span class="empty-icon">&#8856;</span>
                        <div class="empty-title">No telemetry records match the current filters.</div>
                        <div class="empty-desc">Adjust your sentiment filter, model selection, or search query to view operational records.</div>
                        <button type="button" onclick="resetTelemetryFilters()" class="btn-outline" style="margin-top:12px;">Reset Filters</button>
                    </div>

                    <!-- DESKTOP VIEW: TABULAR FORMAT (>= 768px) -->
                    <div class="desktop-telemetry-view" style="overflow-x:auto;">
                        <table aria-label="Operational Telemetry Audit Log">
                            <thead>
                                <tr>
                                    <th>Timestamp (UTC)</th>
                                    <th>Request ID</th>
                                    <th>Engine</th>
                                    <th>Classification</th>
                                    <th>Polarity Score</th>
                                    <th>Confidence</th>
                                    <th>Latency</th>
                                    <th>Feedback</th>
                                </tr>
                            </thead>
                            <tbody id="historyTableBody">
                                <tr><td colspan="8" style="text-align:center; color:var(--text-muted);">Loading telemetry records...</td></tr>
                            </tbody>
                        </table>
                    </div>

                    <!-- MOBILE VIEW: STACKED INTELLIGENCE RECORDS (< 768px) -->
                    <div class="mobile-telemetry-view" id="mobileHistoryContainer" aria-label="Operational Telemetry Cards">
                        <div style="text-align:center; color:var(--text-muted); padding:20px;">Loading telemetry records...</div>
                    </div>
                </div>
            </div>

            <!-- OPERATIONAL EXPORTS -->
            <div class="bezel-card">
                <div class="bezel-core">
                    <div class="card-header-row" style="margin-bottom:0;">
                        <div>
                            <div class="card-title" style="margin-bottom:2px;">Operational Telemetry Export</div>
                            <p style="font-size:12px; color:var(--text-muted);">Export production database logs for compliance, governance, and model monitoring.</p>
                        </div>
                        <div style="display:flex; gap:10px;">
                            <button onclick="window.location.href='/api/v1/export/csv'" class="btn-outline">Export CSV File</button>
                            <button onclick="window.location.href='/api/v1/export/json'" class="btn-outline">Export JSON Stream</button>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- TAB 5: API DOCUMENTATION -->
        <section id="view-developer" class="view-panel" role="tabpanel" aria-labelledby="tab-developer">
            <div class="grid-2col">
                <div class="bezel-card">
                    <div class="bezel-core">
                        <div class="card-header-row">
                            <span class="card-title">Interactive Console: POST /api/v1/predict</span>
                            <span style="font-family:var(--font-mono); font-size:11px; color:var(--signal-positive);">HTTP 200 OK</span>
                        </div>
                        <textarea id="apiPayload" style="width:100%; height:120px; background:var(--surface-well); border:1px solid var(--border-hairline); color:var(--text-primary); font-family:var(--font-mono); font-size:12px; padding:12px; border-radius:var(--radius-control); outline:none;">{
  "text": "The performance was truly spectacular, elegant and deeply engaging.",
  "model_choice": "logistic_regression"
}</textarea>
                        <div style="margin-top:12px; display:flex; justify-content:flex-end;">
                            <button id="btnTestApi" onclick="testAPI()" class="btn-primary">Send Test Request</button>
                        </div>
                        <pre id="apiResponse" style="background:var(--surface-well); padding:14px; border:1px solid var(--border-hairline); min-height:180px; margin-top:14px; color:var(--text-body); font-family:var(--font-mono); font-size:11px; border-radius:var(--radius-control); overflow-x:auto;">// Output will render here...</pre>
                    </div>
                </div>

                <div class="bezel-card">
                    <div class="bezel-core">
                        <div class="card-title" style="margin-bottom:14px;">Core System Diagnostics</div>
                        <div style="font-size:12px; line-height:2.0; color:var(--text-primary);">
                            <div style="border-bottom:1px solid var(--border-subtle); padding-bottom:10px; margin-bottom:10px;">
                                <span style="color:var(--text-muted);">API Gateway:</span> <span style="float:right; color:var(--signal-positive); font-family:var(--font-mono);">ONLINE (FastAPI 0.110+)</span><br>
                                <span style="color:var(--text-muted);">Storage Engine:</span> <span style="float:right; color:var(--signal-positive); font-family:var(--font-mono);">SQLITE CONNECTED</span><br>
                                <span style="color:var(--text-muted);">Active Models:</span> <span style="float:right; color:var(--text-primary); font-family:var(--font-mono);">3 ENGINES READY</span><br>
                                <span style="color:var(--text-muted);">Platform Build:</span> <span style="float:right; font-family:var(--font-mono);">v14.3.0-PROD</span>
                            </div>
                            <div style="font-size:12px; color:var(--text-muted);">
                                <strong>Enterprise Governance:</strong> Raw natural language text is SHA-256 redacted by default prior to database persistence to maintain strict executive privacy compliance.
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <script>
        // --- RESTFUL UI CONTROLLER ---
        const canvas = document.getElementById('neuralCanvas');
        const ctx = canvas.getContext('2d');
        let time = 0;
        let isAnalyzing = false;
        let currentSignalState = 'idle'; // 'idle', 'processing', 'positive', 'negative', 'neutral'
        let animFrameId = null;
        let currentTabId = 'analyze';
        const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        let benchmarkCache = null;

        // Telemetry State
        let cachedHistoryList = [];
        let activeSentimentFilter = 'ALL';
        let activeEngineFilter = 'ALL';
        let activeSearchQuery = '';
        const expandedRecordIds = new Set();

        // Model Choice Mapping for Header and XAI Labels
        const MODEL_NAMES = {
            logistic_regression: "Logistic Regression v1.0",
            linear_svm: "Linear SVM v1.0",
            naive_bayes: "Multinomial Naive Bayes"
        };

        function onModelChoiceChange() {
            const sel = document.getElementById('modelChoice');
            const chosen = sel.value;
            const xaiLabel = document.getElementById('xaiEngineLabel');
            if (xaiLabel) {
                xaiLabel.innerText = "Active Attribution: " + (MODEL_NAMES[chosen] || chosen);
            }
        }

        // Demonstration Presets
        const PRESETS = {
            pos: "The operational execution was flawless, delivering outstanding efficiency.",
            neg: "The system suffered severe outages and unacceptably sluggish response times.",
            mixed: "The visual execution was extraordinary, though the pacing dragged considerably."
        };

        function applyPreset(key) {
            const text = PRESETS[key] || '';
            const input = document.getElementById('nlpInput');
            input.value = text;
            updateInputMetrics();
            input.focus();
        }

        function clearInput() {
            const input = document.getElementById('nlpInput');
            input.value = '';
            updateInputMetrics();
            input.focus();
        }

        function updateInputMetrics() {
            const input = document.getElementById('nlpInput');
            const text = input.value;
            const charLen = text.length;
            const words = text.trim() ? text.trim().split(/\s+/).length : 0;
            document.getElementById('charCount').innerText = `${charLen} / 5000 chars`;
            document.getElementById('wordCount').innerText = `${words} word${words === 1 ? '' : 's'}`;
        }

        document.getElementById('nlpInput').addEventListener('input', updateInputMetrics);

        // Keyboard shortcut listener: Ctrl+Enter / Cmd+Enter execution
        document.getElementById('nlpInput').addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                if (!isAnalyzing) {
                    executeInference();
                }
            }
        });

        // In-App Accessible Notification (Replacing native alert)
        function showNotification(title, message, type = 'error') {
            const el = document.getElementById('appNotification');
            const icon = document.getElementById('notificationIcon');
            const titleEl = document.getElementById('notificationTitle');
            const msgEl = document.getElementById('notificationMessage');

            titleEl.innerText = title;
            msgEl.innerText = message;
            el.className = 'app-notification notification-' + type;
            icon.innerHTML = type === 'error' ? '&#9888;' : '&#8505;';
            el.style.display = 'flex';

            clearTimeout(window.notificationTimeout);
            window.notificationTimeout = setTimeout(() => {
                dismissNotification();
            }, 6000);
        }

        function dismissNotification() {
            const el = document.getElementById('appNotification');
            if (el) el.style.display = 'none';
        }

        // Signal Visualization Semantics (Explicit state terminology)
        const SIGNAL_LABELS = {
            idle: "STANDBY",
            processing: "ANALYZING",
            positive: "POSITIVE CLASSIFICATION",
            negative: "NEGATIVE CLASSIFICATION",
            neutral: "NEUTRAL CLASSIFICATION"
        };

        function updateSignalStatus(state) {
            currentSignalState = state;
            const statusEl = document.getElementById('signalStatus');
            if (!statusEl) return;

            const label = SIGNAL_LABELS[state] || "STANDBY";
            let dotColor = 'var(--signal-neutral)';

            if (state === 'idle') {
                dotColor = 'var(--signal-neutral)';
                statusEl.className = 'signal-state-indicator';
            } else if (state === 'processing') {
                dotColor = 'var(--border-focus)';
                statusEl.className = 'signal-state-indicator state-proc';
            } else if (state === 'positive') {
                dotColor = 'var(--signal-positive)';
                statusEl.className = 'signal-state-indicator state-pos';
            } else if (state === 'negative') {
                dotColor = 'var(--signal-negative)';
                statusEl.className = 'signal-state-indicator state-neg';
            } else if (state === 'neutral') {
                dotColor = 'var(--signal-neutral)';
                statusEl.className = 'signal-state-indicator';
            }

            statusEl.innerHTML = `<span class="status-dot" style="width:5px; height:5px; background:${dotColor}; box-shadow:0 0 5px ${dotColor};"></span><span>${label}</span>`;
        }

        function drawHarmonicWave() {
            if (prefersReducedMotion) {
                canvas.width = canvas.offsetWidth;
                canvas.height = canvas.offsetHeight;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.beginPath();
                ctx.moveTo(0, canvas.height / 2);
                ctx.lineTo(canvas.width, canvas.height / 2);
                ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)';
                ctx.lineWidth = 1;
                ctx.stroke();
                return;
            }

            // Strict lifecycle performance — stop when hidden or not on analyze tab
            if (document.visibilityState === 'hidden' || currentTabId !== 'analyze') {
                animFrameId = null;
                return;
            }

            // Zero-dimension defensive guard: prevent division by zero, NaN coordinates, and layout thrashing
            const cWidth = canvas.offsetWidth;
            const cHeight = canvas.offsetHeight;
            if (!cWidth || !cHeight || cWidth <= 0 || cHeight <= 0) {
                animFrameId = requestAnimationFrame(drawHarmonicWave);
                return;
            }

            canvas.width = cWidth;
            canvas.height = cHeight;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const width = canvas.width;
            const height = canvas.height;
            const centerY = height / 2;

            let speedMult = 0.8;
            let ampMult = 0.9;
            let waveColors = ['rgba(148, 163, 184, 0.35)', 'rgba(59, 130, 246, 0.25)'];

            if (currentSignalState === 'processing') {
                speedMult = 2.8;
                ampMult = 1.6;
                waveColors = ['rgba(59, 130, 246, 0.65)', 'rgba(37, 99, 235, 0.50)'];
            } else if (currentSignalState === 'positive') {
                speedMult = 1.1;
                ampMult = 1.0;
                waveColors = ['rgba(16, 185, 129, 0.60)', 'rgba(5, 150, 105, 0.35)'];
            } else if (currentSignalState === 'negative') {
                speedMult = 1.7;
                ampMult = 1.3;
                waveColors = ['rgba(244, 63, 94, 0.60)', 'rgba(225, 29, 72, 0.35)'];
            } else if (currentSignalState === 'neutral') {
                speedMult = 0.9;
                ampMult = 0.8;
                waveColors = ['rgba(148, 163, 184, 0.50)', 'rgba(100, 116, 139, 0.30)'];
            }

            for (let i = 0; i < waveColors.length; i++) {
                ctx.beginPath();
                ctx.moveTo(0, centerY);
                for (let x = 0; x < width; x++) {
                    const frequency = 0.008 + (i * 0.004);
                    const amplitude = (7 + Math.sin(time * 0.4 + i) * 5) * ampMult;
                    const offset = time * (1.0 + i) * 15 * speedMult;
                    const envelope = Math.sin((x / width) * Math.PI);
                    const y = centerY + Math.sin(x * frequency + (offset * 0.02)) * amplitude * envelope;
                    ctx.lineTo(x, y);
                }
                ctx.strokeStyle = waveColors[i];
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }

            time += 0.015 * speedMult;
            animFrameId = requestAnimationFrame(drawHarmonicWave);
        }

        function startWaveformIfActive() {
            if (!animFrameId && !prefersReducedMotion && document.visibilityState === 'visible' && currentTabId === 'analyze') {
                animFrameId = requestAnimationFrame(drawHarmonicWave);
            }
        }

        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') {
                startWaveformIfActive();
            } else {
                if (animFrameId) {
                    cancelAnimationFrame(animFrameId);
                    animFrameId = null;
                }
            }
        });

        startWaveformIfActive();

        // URL Hash Navigation & Tab Synchronization
        const VALID_VIEWS = ['analyze', 'consensus', 'benchmarks', 'telemetry', 'developer'];
        let isInternalHashUpdate = false;

        function switchTab(viewId, updateHash = true) {
            if (!VALID_VIEWS.includes(viewId)) {
                viewId = 'analyze';
            }
            currentTabId = viewId;

            document.querySelectorAll('.view-panel').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => {
                el.classList.remove('active');
                el.setAttribute('aria-selected', 'false');
            });

            const targetView = document.getElementById('view-' + viewId);
            if (targetView) targetView.classList.add('active');

            const activeBtn = document.getElementById('tab-' + viewId);
            if (activeBtn) {
                activeBtn.classList.add('active');
                activeBtn.setAttribute('aria-selected', 'true');
            }

            if (updateHash && window.location.hash !== '#' + viewId) {
                isInternalHashUpdate = true;
                window.location.hash = viewId;
            }

            if (viewId === 'telemetry') {
                refreshAnalytics();
                refreshHistory();
            } else if (viewId === 'benchmarks') {
                loadBenchmarkData();
            } else if (viewId === 'analyze') {
                startWaveformIfActive();
            }
        }

        function handleHashRoute() {
            if (isInternalHashUpdate) {
                isInternalHashUpdate = false;
                return;
            }
            const hash = window.location.hash.replace(/^#/, '');
            if (VALID_VIEWS.includes(hash)) {
                switchTab(hash, false);
            } else if (hash === '') {
                switchTab('analyze', false);
            } else {
                switchTab('analyze', true);
            }
        }

        window.addEventListener('hashchange', handleHashRoute);

        // Feature Attribution Renderer (Proportional Visual Bars)
        function escapeHtml(str) {
            if (str === null || str === undefined) return '';
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }

        function renderAttributionBars(posDrivers, negDrivers) {
            const posContainer = document.getElementById('posDrivers');
            const negContainer = document.getElementById('negDrivers');

            const allDrivers = [...(posDrivers || []), ...(negDrivers || [])];
            const maxWeight = allDrivers.length > 0 ?
                Math.max(...allDrivers.map(d => Math.abs(d.weight)), 0.0001) : 1;

            if (!posDrivers || posDrivers.length === 0) {
                posContainer.innerHTML = '<div class="driver-empty-hint">None detected within vocabulary</div>';
            } else {
                posContainer.innerHTML = posDrivers.map(d => {
                    const absVal = Math.abs(d.weight);
                    const widthPct = Math.min(100, Math.max(6, Math.round((absVal / maxWeight) * 100)));
                    const weightStr = `+${Number(d.weight).toFixed(4)}`;
                    const safeToken = escapeHtml(d.token);
                    const titleText = `Token: ${safeToken}&#10;Weight: ${weightStr}&#10;Formula: TF-IDF &times; Model Coefficient`;

                    return `
                        <div class="driver-bar-row" title="${titleText}">
                            <span class="driver-token" title="${safeToken}">${safeToken}</span>
                            <div class="driver-bar-track" role="meter" aria-label="Token ${safeToken}: weight ${weightStr}" aria-valuenow="${absVal.toFixed(4)}" aria-valuemin="0" aria-valuemax="${maxWeight.toFixed(4)}">
                                <div class="driver-bar-fill driver-bar-fill-pos" style="width: ${widthPct}%;"></div>
                            </div>
                            <span class="driver-val driver-val-pos">${weightStr}</span>
                        </div>
                    `;
                }).join('');
            }

            if (!negDrivers || negDrivers.length === 0) {
                negContainer.innerHTML = '<div class="driver-empty-hint">None detected within vocabulary</div>';
            } else {
                negContainer.innerHTML = negDrivers.map(d => {
                    const absVal = Math.abs(d.weight);
                    const widthPct = Math.min(100, Math.max(6, Math.round((absVal / maxWeight) * 100)));
                    const weightStr = `${Number(d.weight).toFixed(4)}`;
                    const safeToken = escapeHtml(d.token);
                    const titleText = `Token: ${safeToken}&#10;Weight: ${weightStr}&#10;Formula: TF-IDF &times; Model Coefficient`;

                    return `
                        <div class="driver-bar-row" title="${titleText}">
                            <span class="driver-token" title="${safeToken}">${safeToken}</span>
                            <div class="driver-bar-track" role="meter" aria-label="Token ${safeToken}: weight ${weightStr}" aria-valuenow="${absVal.toFixed(4)}" aria-valuemin="0" aria-valuemax="${maxWeight.toFixed(4)}">
                                <div class="driver-bar-fill driver-bar-fill-neg" style="width: ${widthPct}%;"></div>
                            </div>
                            <span class="driver-val driver-val-neg">${weightStr}</span>
                        </div>
                    `;
                }).join('');
            }
        }

        // Execute Single Inference
        async function executeInference() {
            const text = document.getElementById('nlpInput').value.trim();
            const modelChoice = document.getElementById('modelChoice').value;
            if (!text) {
                showNotification("Input Validation", "Please enter natural language text before executing analysis.", "warning");
                return;
            }

            const btn = document.getElementById('execBtn');
            btn.innerText = "Analyzing...";
            btn.disabled = true;
            isAnalyzing = true;
            updateSignalStatus('processing');
            document.getElementById('feedConfirm').style.display = 'none';

            try {
                const res = await fetch('/api/v1/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text, model_choice: modelChoice })
                });

                if (!res.ok) {
                    const err = await res.json();
                    showNotification("Analysis Execution Error", err.detail || "Request failed to process.", "error");
                    updateSignalStatus('idle');
                    return;
                }

                const data = await res.json();

                let sign = data.ai_score > 0 ? '+' : '';
                document.getElementById('resScore').innerText = `${sign}${data.ai_score.toFixed(4)}`;

                const badge = document.getElementById('resSentiment');
                badge.innerText = data.sentiment;
                badge.className = 'badge-sentiment ' + (
                    data.sentiment === 'Positive' ? 'badge-pos' : 
                    data.sentiment === 'Negative' ? 'badge-neg' : 'badge-neu'
                );

                // Update Bipolar Gauge Position (-1.00 to +1.00 mapped to 0% to 100%)
                const pct = Math.max(0, Math.min(100, ((data.ai_score + 1.0) / 2.0) * 100.0));
                const indicator = document.getElementById('polarityIndicator');
                indicator.style.left = `${pct.toFixed(2)}%`;
                indicator.className = 'polarity-indicator ' + (
                    data.sentiment === 'Positive' ? 'indicator-pos' : 
                    data.sentiment === 'Negative' ? 'indicator-neg' : 'indicator-neu'
                );

                document.getElementById('resConfidence').innerText = `Confidence: ${data.confidence}%`;
                document.getElementById('resEmotion').innerText = data.emotion;
                document.getElementById('resLatency').innerText = `${data.latency_ms} ms`;
                document.getElementById('resReqId').innerText = data.request_id;
                document.getElementById('currentPredId').value = data.prediction_id;

                // Render Attribution Drivers as Proportional Bars
                renderAttributionBars(
                    data.explanation.positive_drivers,
                    data.explanation.negative_drivers
                );

                // Update Waveform & Status according to actual inference result
                if (data.sentiment === 'Positive') {
                    updateSignalStatus('positive');
                } else if (data.sentiment === 'Negative') {
                    updateSignalStatus('negative');
                } else {
                    updateSignalStatus('neutral');
                }

            } catch (e) {
                console.error("Inference execution error:", e);
                showNotification("Network Error", "Failed to connect to the inference server.", "error");
                updateSignalStatus('idle');
            } finally {
                btn.innerText = "Execute Analysis";
                btn.disabled = false;
                isAnalyzing = false;
            }
        }

        // Execute Consensus with Restrained Loading State
        async function executeConsensus() {
            const text = document.getElementById('nlpInput').value.trim();
            if (!text) {
                showNotification("Input Required", "Please enter natural language text in the Inference Console first.", "warning");
                switchTab('analyze');
                return;
            }

            const btn = document.getElementById('btnConsensusRun');
            if (btn) {
                btn.innerText = "Evaluating Models...";
                btn.disabled = true;
            }

            try {
                const res = await fetch('/api/v1/consensus', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                if (!res.ok) {
                    const err = await res.json();
                    showNotification("Consensus Error", err.detail || "Consensus execution failed.", "error");
                    return;
                }

                const data = await res.json();
                const banner = document.getElementById('consensusBanner');
                banner.style.display = 'block';

                if (data.disagreement_detected) {
                    banner.style.background = 'var(--signal-divergence-bg)';
                    banner.style.border = '1px solid var(--signal-divergence-border)';
                    banner.style.color = 'var(--signal-divergence)';
                    banner.innerHTML = `<strong>EPISTEMIC DISAGREEMENT DETECTED:</strong> ${data.agreement_ratio} &mdash; Majority Consensus: ${data.consensus_sentiment}`;
                } else {
                    banner.style.background = 'var(--signal-positive-bg)';
                    banner.style.border = '1px solid var(--signal-positive-border)';
                    banner.style.color = 'var(--signal-positive)';
                    banner.innerHTML = `<strong>HIGH CONSENSUS:</strong> ${data.agreement_ratio} &mdash; Unanimous: ${data.consensus_sentiment}`;
                }

                // Render Models
                const lr = data.models.logistic_regression;
                const svm = data.models.linear_svm;
                const nb = data.models.naive_bayes;

                document.getElementById('cs-lr-sentiment').innerText = lr.sentiment;
                document.getElementById('cs-lr-meta').innerText = `Score: ${lr.ai_score} | Conf: ${lr.confidence}%`;

                document.getElementById('cs-svm-sentiment').innerText = svm.sentiment;
                document.getElementById('cs-svm-meta').innerText = `Score: ${svm.ai_score} | Conf: ${svm.confidence}%`;

                document.getElementById('cs-nb-sentiment').innerText = nb.sentiment;
                document.getElementById('cs-nb-meta').innerText = `Score: ${nb.ai_score} | Conf: ${nb.confidence}%`;

            } catch (e) {
                console.error("Consensus execution error:", e);
                showNotification("Network Error", "Failed to communicate with consensus engine.", "error");
            } finally {
                if (btn) {
                    btn.innerText = "Run All 3 Models";
                    btn.disabled = false;
                }
            }
        }

        // Feedback Submission with Restrained Loading State
        async function sendTelemetry(isCorrect) {
            const id = document.getElementById('currentPredId').value;
            if (!id) return;

            const confirmBtn = document.getElementById('btnFeedConfirm');
            const disputeBtn = document.getElementById('btnFeedDispute');
            if (confirmBtn) confirmBtn.disabled = true;
            if (disputeBtn) disputeBtn.disabled = true;

            try {
                const res = await fetch('/api/v1/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prediction_id: parseInt(id), is_correct: isCorrect })
                });
                if (res.ok) {
                    document.getElementById('feedConfirm').style.display = 'inline';
                }
            } catch (e) {
                console.error("Feedback error:", e);
                showNotification("Telemetry Error", "Failed to record audit feedback.", "error");
            } finally {
                if (confirmBtn) confirmBtn.disabled = false;
                if (disputeBtn) disputeBtn.disabled = false;
            }
        }

        // Refresh Analytics
        async function refreshAnalytics() {
            try {
                const res = await fetch('/api/v1/analytics');
                const data = await res.json();
                document.getElementById('stTotal').innerText = data.total_analyzed.toLocaleString();
                document.getElementById('stPos').innerText = data.positive.toLocaleString();
                document.getElementById('stNeg').innerText = data.negative.toLocaleString();

                const p95Val = (typeof data.p95_latency_ms === 'number') ? `${data.p95_latency_ms}` : data.p95_latency_ms;
                document.getElementById('stP95').innerHTML = `${p95Val} <span style="font-size:14px; color:var(--text-muted);">${typeof data.p95_latency_ms === 'number' ? 'ms' : ''}</span>`;
            } catch (e) {
                console.error("Analytics fetch error:", e);
            }
        }

        // Telemetry Filtering Controls & Handlers
        function setSentimentFilter(sentiment, btnEl) {
            activeSentimentFilter = sentiment;
            document.querySelectorAll('.filter-sentiment-group .filter-btn').forEach(b => b.classList.remove('active'));
            if (btnEl) btnEl.classList.add('active');
            applyTelemetryFilters();
        }

        function resetTelemetryFilters() {
            activeSentimentFilter = 'ALL';
            activeEngineFilter = 'ALL';
            activeSearchQuery = '';

            document.querySelectorAll('.filter-sentiment-group .filter-btn').forEach(b => {
                b.classList.toggle('active', b.dataset.sentiment === 'ALL');
            });

            const engineEl = document.getElementById('filterEngine');
            if (engineEl) engineEl.value = 'ALL';

            const searchEl = document.getElementById('filterSearch');
            if (searchEl) searchEl.value = '';

            applyTelemetryFilters();
        }

        function applyTelemetryFilters() {
            const engineEl = document.getElementById('filterEngine');
            if (engineEl) activeEngineFilter = engineEl.value;

            const searchEl = document.getElementById('filterSearch');
            if (searchEl) activeSearchQuery = searchEl.value.trim().toLowerCase();

            const filtered = cachedHistoryList.filter(item => {
                if (activeSentimentFilter !== 'ALL' && item.sentiment !== activeSentimentFilter) {
                    return false;
                }
                if (activeEngineFilter !== 'ALL' && item.engine !== activeEngineFilter) {
                    return false;
                }
                if (activeSearchQuery) {
                    const reqId = (item.request_id || '').toLowerCase();
                    const engine = (item.engine || '').toLowerCase();
                    const emotion = (item.emotion || '').toLowerCase();
                    const sentiment = (item.sentiment || '').toLowerCase();
                    const feedback = (item.user_feedback || '').toLowerCase();
                    const idStr = String(item.id || '');

                    const match = reqId.includes(activeSearchQuery) ||
                                  engine.includes(activeSearchQuery) ||
                                  emotion.includes(activeSearchQuery) ||
                                  sentiment.includes(activeSearchQuery) ||
                                  feedback.includes(activeSearchQuery) ||
                                  idStr.includes(activeSearchQuery);
                    if (!match) return false;
                }
                return true;
            });

            renderTelemetryViews(filtered);
        }

        // Keyboard & Click Record Expansion
        function toggleRecordExpansion(recId) {
            recId = String(recId);
            if (expandedRecordIds.has(recId)) {
                expandedRecordIds.delete(recId);
            } else {
                expandedRecordIds.add(recId);
            }

            const desktopDrawer = document.getElementById('drawer-desktop-' + recId);
            const desktopRow = document.getElementById('row-desktop-' + recId);
            const mobileDrawer = document.getElementById('drawer-mobile-' + recId);
            const mobileBtn = document.getElementById('btn-expand-mobile-' + recId);

            const isOpen = expandedRecordIds.has(recId);

            if (desktopDrawer) {
                desktopDrawer.classList.toggle('open', isOpen);
            }
            if (desktopRow) {
                desktopRow.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
            }
            if (mobileDrawer) {
                mobileDrawer.classList.toggle('open', isOpen);
            }
            if (mobileBtn) {
                mobileBtn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
                mobileBtn.innerHTML = isOpen ?
                    `<span>Collapse Inspection</span><span>&#9652;</span>` :
                    `<span>Inspect Forensic Details</span><span>&#9662;</span>`;
            }
        }

        // Render Desktop Table + Mobile Cards with Real Backend Data Only
        function renderTelemetryViews(records) {
            const desktopTbody = document.getElementById('historyTableBody');
            const mobileContainer = document.getElementById('mobileHistoryContainer');
            const emptyState = document.getElementById('telemetryEmptyState');

            if (!records || records.length === 0) {
                if (desktopTbody) desktopTbody.innerHTML = '';
                if (mobileContainer) mobileContainer.innerHTML = '';
                if (emptyState) emptyState.style.display = 'block';
                return;
            }

            if (emptyState) emptyState.style.display = 'none';

            // Desktop Table Rows
            if (desktopTbody) {
                desktopTbody.innerHTML = records.map(item => {
                    const isExp = expandedRecordIds.has(String(item.id));
                    const badgeClass = item.sentiment === 'Positive' ? 'badge-pos' : (item.sentiment === 'Negative' ? 'badge-neg' : 'badge-neu');
                    const scoreSign = item.ai_score > 0 ? '+' : '';
                    const feedbackText = item.user_feedback || 'Unreviewed';
                    const recId = String(item.id);

                    return `
                        <tr class="telemetry-row" id="row-desktop-${recId}" tabindex="0" role="button" aria-expanded="${isExp}" onclick="toggleRecordExpansion('${recId}')" onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();toggleRecordExpansion('${recId}');}" title="Click or press Enter/Space to expand forensic details">
                            <td style="font-family:var(--font-mono); font-size:11px;">${escapeHtml(item.timestamp)}</td>
                            <td style="font-family:var(--font-mono); font-size:11px; color:var(--text-primary); font-weight:500;">${escapeHtml(item.request_id)}</td>
                            <td style="font-family:var(--font-mono); font-size:11px;">${escapeHtml(item.engine)}</td>
                            <td><span class="badge-sentiment ${badgeClass}" style="padding:2px 8px; font-size:10px;">${escapeHtml(item.sentiment)}</span></td>
                            <td style="font-family:var(--font-mono); font-variant-numeric:tabular-nums;">${scoreSign}${item.ai_score}</td>
                            <td style="font-family:var(--font-mono);">${item.confidence}%</td>
                            <td style="font-family:var(--font-mono);">${item.latency_ms} ms</td>
                            <td style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted);">${escapeHtml(feedbackText)}</td>
                        </tr>
                        <tr id="drawer-desktop-${recId}" class="record-drawer ${isExp ? 'open' : ''}">
                            <td colspan="8" style="padding:0; border-bottom: 1px solid var(--border-subtle);">
                                <div class="drawer-content-box">
                                    <div class="drawer-grid">
                                        <div>
                                            <div class="drawer-field-label">Record Index</div>
                                            <div class="drawer-field-val">#${item.id}</div>
                                        </div>
                                        <div>
                                            <div class="drawer-field-label">Tracking Request ID</div>
                                            <div class="drawer-field-val">${escapeHtml(item.request_id)}</div>
                                        </div>
                                        <div>
                                            <div class="drawer-field-label">Timestamp (UTC)</div>
                                            <div class="drawer-field-val">${escapeHtml(item.timestamp)}</div>
                                        </div>
                                        <div>
                                            <div class="drawer-field-label">Classifier Engine</div>
                                            <div class="drawer-field-val">${escapeHtml(item.engine)}</div>
                                        </div>
                                        <div>
                                            <div class="drawer-field-label">Affective Heuristic</div>
                                            <div class="drawer-field-val">${escapeHtml(item.emotion || 'Standard')}</div>
                                        </div>
                                        <div>
                                            <div class="drawer-field-label">Audit Feedback State</div>
                                            <div class="drawer-field-val">${escapeHtml(feedbackText)}</div>
                                        </div>
                                    </div>
                                </div>
                            </td>
                        </tr>
                    `;
                }).join('');
            }

            // Mobile Cards (< 768px)
            if (mobileContainer) {
                mobileContainer.innerHTML = records.map(item => {
                    const isExp = expandedRecordIds.has(String(item.id));
                    const badgeClass = item.sentiment === 'Positive' ? 'badge-pos' : (item.sentiment === 'Negative' ? 'badge-neg' : 'badge-neu');
                    const scoreSign = item.ai_score > 0 ? '+' : '';
                    const feedbackText = item.user_feedback || 'Unreviewed';
                    const recId = String(item.id);

                    return `
                        <div class="telemetry-card" id="card-mobile-${recId}">
                            <div class="card-top-row">
                                <span class="card-time">${escapeHtml(item.timestamp)}</span>
                                <div style="display:flex; align-items:center; gap:6px;">
                                    <span class="card-engine-tag">${escapeHtml(item.engine)}</span>
                                    <span class="badge-sentiment ${badgeClass}" style="padding:2px 8px; font-size:10px;">${escapeHtml(item.sentiment)}</span>
                                </div>
                            </div>
                            <div class="card-metrics-grid">
                                <div class="card-metric-cell">
                                    <span class="cell-label">AI Score</span>
                                    <span class="cell-val">${scoreSign}${item.ai_score}</span>
                                </div>
                                <div class="card-metric-cell">
                                    <span class="cell-label">Confidence</span>
                                    <span class="cell-val">${item.confidence}%</span>
                                </div>
                                <div class="card-metric-cell">
                                    <span class="cell-label">Latency</span>
                                    <span class="cell-val">${item.latency_ms} ms</span>
                                </div>
                                <div class="card-metric-cell">
                                    <span class="cell-label">Feedback</span>
                                    <span class="cell-val" style="font-size:10px; color:var(--text-muted);">${escapeHtml(feedbackText)}</span>
                                </div>
                            </div>
                            <button type="button" id="btn-expand-mobile-${recId}" class="card-expand-btn" aria-expanded="${isExp}" onclick="toggleRecordExpansion('${recId}')" aria-label="Toggle record ${item.id} forensic details">
                                <span>${isExp ? 'Collapse Inspection' : 'Inspect Forensic Details'}</span>
                                <span>${isExp ? '&#9652;' : '&#9662;'}</span>
                            </button>
                            <div id="drawer-mobile-${recId}" class="card-drawer-content ${isExp ? 'open' : ''}">
                                <div class="mobile-drawer-grid">
                                    <div>
                                        <div class="drawer-field-label">Record Index</div>
                                        <div class="drawer-field-val">#${item.id}</div>
                                    </div>
                                    <div>
                                        <div class="drawer-field-label">Request ID</div>
                                        <div class="drawer-field-val">${escapeHtml(item.request_id)}</div>
                                    </div>
                                    <div>
                                        <div class="drawer-field-label">Affective Heuristic</div>
                                        <div class="drawer-field-val">${escapeHtml(item.emotion || 'Standard')}</div>
                                    </div>
                                    <div>
                                        <div class="drawer-field-label">Feedback State</div>
                                        <div class="drawer-field-val">${escapeHtml(feedbackText)}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                }).join('');
            }
        }

        // Refresh History with Restrained Loading State
        async function refreshHistory() {
            const btn = document.getElementById('btnRefreshHistory');
            if (btn) {
                btn.innerText = "Refreshing...";
                btn.disabled = true;
            }
            try {
                const res = await fetch('/api/v1/history?limit=50');
                if (res.ok) {
                    cachedHistoryList = await res.json();
                    applyTelemetryFilters();
                }
            } catch (e) {
                console.error("History fetch error:", e);
                showNotification("Telemetry Sync", "Failed to retrieve telemetry records from database.", "warning");
            } finally {
                if (btn) {
                    btn.innerText = "Refresh Log";
                    btn.disabled = false;
                }
            }
        }

        // Canonical Confusion Matrix Switcher & Data Binding
        function switchMatrixModel(modelKey, btnEl) {
            document.querySelectorAll('.model-toggle-btn').forEach(b => b.classList.remove('active'));
            if (btnEl) btnEl.classList.add('active');
            renderConfusionMatrix(modelKey);
        }

        function renderConfusionMatrix(modelKey) {
            if (!benchmarkCache || !benchmarkCache.models || !benchmarkCache.models[modelKey]) return;
            const modelData = benchmarkCache.models[modelKey];
            const cm = modelData.confusion_matrix;
            if (!cm) return;

            const tn = cm.true_negative;
            const fp = cm.false_positive;
            const fn = cm.false_negative;
            const tp = cm.true_positive;
            const total = cm.total_evaluated || (tn + fp + fn + tp);

            const actualNeg = tn + fp;
            const actualPos = fn + tp;
            const predNeg = tn + fn;
            const predPos = fp + tp;

            // Quadrant counts & percentages
            document.getElementById('val-tn').innerText = tn.toLocaleString();
            document.getElementById('pct-tn').innerText = ((tn / total) * 100).toFixed(2) + '%';

            document.getElementById('val-fp').innerText = fp.toLocaleString();
            document.getElementById('pct-fp').innerText = ((fp / total) * 100).toFixed(2) + '% (Type I)';

            document.getElementById('val-fn').innerText = fn.toLocaleString();
            document.getElementById('pct-fn').innerText = ((fn / total) * 100).toFixed(2) + '% (Type II)';

            document.getElementById('val-tp').innerText = tp.toLocaleString();
            document.getElementById('pct-tp').innerText = ((tp / total) * 100).toFixed(2) + '%';

            // Marginals
            document.getElementById('cm-total-neg').innerText = actualNeg.toLocaleString();
            document.getElementById('cm-total-pos').innerText = actualPos.toLocaleString();
            document.getElementById('cm-col-neg').innerText = predNeg.toLocaleString();
            document.getElementById('cm-col-pos').innerText = predPos.toLocaleString();
            document.getElementById('cm-grand-total').innerText = total.toLocaleString();

            // Analytical Rates (Derived from real counts)
            const sensitivity = (tp / actualPos) * 100;
            const specificity = (tn / actualNeg) * 100;
            const precision = (tp / predPos) * 100;
            const accuracy = ((tp + tn) / total) * 100;

            document.getElementById('cm-sensitivity').innerText = sensitivity.toFixed(2) + '%';
            document.getElementById('cm-specificity').innerText = specificity.toFixed(2) + '%';
            document.getElementById('cm-precision').innerText = precision.toFixed(2) + '%';
            document.getElementById('cm-accuracy').innerText = accuracy.toFixed(2) + '%';

            document.getElementById('cmActiveModelLabel').innerText = modelData.name || modelKey;
        }

        // Load Benchmark Data from /api/v1/models
        async function loadBenchmarkData() {
            try {
                const res = await fetch('/api/v1/models');
                const data = await res.json();
                if (data.models) {
                    benchmarkCache = data;
                    renderConfusionMatrix('logistic_regression');
                }
            } catch (e) {
                console.error("Model registry fetch error:", e);
            }
        }

        // Initial Data & Route Load on Page Ready
        loadBenchmarkData();
        handleHashRoute();

        // API Test Console
        async function testAPI() {
            const reqBox = document.getElementById('apiPayload').value;
            const resBox = document.getElementById('apiResponse');
            const btn = document.getElementById('btnTestApi');
            if (btn) btn.disabled = true;
            resBox.innerText = "// Dispatching request...";
            try {
                const res = await fetch('/api/v1/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: reqBox
                });
                const data = await res.json();
                resBox.innerText = JSON.stringify(data, null, 2);
            } catch (e) {
                resBox.innerText = "// Error: Invalid JSON or network failure: " + e;
            } finally {
                if (btn) btn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
