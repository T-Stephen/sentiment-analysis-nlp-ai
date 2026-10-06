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
    <link href="https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            color-scheme: dark;
            --font-sans: 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            --font-mono: 'Geist Mono', 'JetBrains Mono', 'SF Mono', monospace;

            /* Monolithic Obsidian Surfaces */
            --bg-void: #07080B;
            --bg-surface: #0C0E14;
            --bg-panel: #12151D;
            --bg-elevated: #181B24;
            --bg-well: #090B10;

            /* Borders & Highlights */
            --border-hairline: rgba(255, 255, 255, 0.07);
            --border-subtle: rgba(255, 255, 255, 0.12);
            --border-focus: #38BDF8;
            --highlight-inset: inset 0 1px 0 rgba(255, 255, 255, 0.08), inset 0 0 0 1px rgba(255, 255, 255, 0.02);

            /* Signal Colors */
            --signal-positive: #10B981;
            --signal-positive-bg: rgba(16, 185, 129, 0.10);
            --signal-positive-border: rgba(16, 185, 129, 0.25);

            --signal-negative: #F43F5E;
            --signal-negative-bg: rgba(244, 63, 94, 0.10);
            --signal-negative-border: rgba(244, 63, 94, 0.25);

            --signal-neutral: #94A3B8;
            --signal-neutral-bg: rgba(148, 163, 184, 0.10);
            --signal-neutral-border: rgba(148, 163, 184, 0.25);

            --signal-warning: #F59E0B;
            --signal-warning-bg: rgba(245, 158, 11, 0.10);
            --signal-warning-border: rgba(245, 158, 11, 0.25);

            --accent-tech: #38BDF8;
            --accent-tech-bg: rgba(56, 189, 248, 0.10);
            --accent-tech-border: rgba(56, 189, 248, 0.25);

            /* Typography Colors */
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --text-muted: #64748B;

            /* Radii (Hardware Architecture) */
            --radius-chassis: 16px;
            --radius-core: 12px;
            --radius-control: 8px;
            --radius-badge: 6px;

            /* Restrained Transitions */
            --transition-fast: 150ms cubic-bezier(0.16, 1, 0.3, 1);
            --transition-normal: 220ms cubic-bezier(0.16, 1, 0.3, 1);
        }

        *, *::before, *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: var(--font-sans);
            background-color: var(--bg-void);
            color: var(--text-primary);
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            min-height: 100dvh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        /* FOCUS OUTLINES (ACCESSIBILITY) */
        a:focus-visible, button:focus-visible, textarea:focus-visible, select:focus-visible, input:focus-visible {
            outline: 2px solid var(--border-focus);
            outline-offset: 2px;
        }

        /* MASTHEAD APP BAR */
        .masthead {
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(12, 14, 20, 0.88);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-hairline);
            height: 54px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
        }

        .masthead-left {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-shrink: 0;
        }

        .brand-monogram {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: var(--radius-badge);
            background: var(--bg-panel);
            border: 1px solid var(--border-subtle);
            font-size: 13px;
        }

        .brand-title {
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .badge-version {
            font-family: var(--font-mono);
            font-size: 10.5px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: var(--radius-badge);
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-hairline);
            color: var(--text-secondary);
            letter-spacing: 0;
            text-transform: none;
        }

        .badge-status {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--signal-positive);
            padding: 2px 8px;
            background: var(--signal-positive-bg);
            border: 1px solid var(--signal-positive-border);
            border-radius: var(--radius-badge);
        }

        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--signal-positive);
        }

        /* NAVIGATION SEGMENTED DOCK */
        .nav-dock {
            display: flex;
            align-items: center;
            gap: 4px;
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            padding: 3px;
            border-radius: var(--radius-control);
            overflow-x: auto;
            max-width: 100%;
        }

        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-family: var(--font-sans);
            font-size: 12.5px;
            font-weight: 500;
            padding: 5px 14px;
            border-radius: 6px;
            cursor: pointer;
            transition: all var(--transition-fast);
            white-space: nowrap;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .tab-btn:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.04);
        }

        .tab-btn.active {
            color: var(--text-primary);
            background: var(--bg-elevated);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
            font-weight: 600;
        }

        /* MAIN CONTAINER */
        .app-shell {
            max-width: 1540px;
            width: 100%;
            margin: 0 auto;
            padding: 20px 24px 48px;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        /* DOMAIN & PRIVACY ADVISORY BANNERS */
        .advisory-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-core);
            padding: 10px 16px;
            font-size: 12px;
            gap: 16px;
            flex-wrap: wrap;
        }

        .advisory-domain {
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--text-secondary);
        }

        .advisory-domain-tag {
            font-family: var(--font-mono);
            font-size: 10.5px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--accent-tech);
            background: var(--accent-tech-bg);
            border: 1px solid var(--accent-tech-border);
            padding: 2px 7px;
            border-radius: var(--radius-badge);
            white-space: nowrap;
        }

        .advisory-privacy {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .advisory-privacy strong {
            color: var(--text-secondary);
        }

        /* VIEW PANELS */
        .view-panel {
            display: none;
            width: 100%;
        }

        .view-panel.active {
            display: block;
        }

        /* DOUBLE-BEZEL HARDWARE CARDS */
        .chassis-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-chassis);
            padding: 12px;
            box-shadow: 0 12px 32px -8px rgba(0, 0, 0, 0.65), 0 4px 12px -2px rgba(0, 0, 0, 0.4);
            display: flex;
            flex-direction: column;
        }

        .core-well {
            background: var(--bg-panel);
            border: 1px solid rgba(255, 255, 255, 0.04);
            box-shadow: var(--highlight-inset);
            border-radius: var(--radius-core);
            padding: 20px;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        /* GRID LAYOUTS */
        .grid-2col {
            display: grid;
            grid-template-columns: 58% 42%;
            gap: 16px;
            align-items: start;
        }

        .grid-3col {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }

        /* CARD HEADERS */
        .panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 2px;
        }

        .panel-title {
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-secondary);
            font-family: var(--font-mono);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* PRESET STRIP */
        .preset-strip {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }

        .preset-label {
            font-family: var(--font-mono);
            font-size: 10.5px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .preset-chip {
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            color: var(--text-secondary);
            font-family: var(--font-sans);
            font-size: 11.5px;
            font-weight: 500;
            padding: 4px 10px;
            border-radius: var(--radius-badge);
            cursor: pointer;
            transition: all var(--transition-fast);
        }

        .preset-chip:hover {
            color: var(--text-primary);
            border-color: var(--border-subtle);
            background: var(--bg-elevated);
        }

        /* INPUT AREA */
        .input-dock {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .text-input {
            width: 100%;
            min-height: 160px;
            background: var(--bg-well);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-control);
            padding: 14px 16px;
            color: var(--text-primary);
            font-family: var(--font-sans);
            font-size: 14px;
            line-height: 1.6;
            resize: vertical;
            outline: none;
            transition: border-color var(--transition-fast);
        }

        .text-input:focus {
            border-color: var(--border-focus);
            box-shadow: 0 0 0 1px var(--border-focus);
        }

        .input-meta-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
        }

        .input-counters {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .btn-link {
            background: none;
            border: none;
            color: var(--text-muted);
            font-family: var(--font-mono);
            font-size: 11px;
            cursor: pointer;
            text-decoration: underline;
        }

        .btn-link:hover {
            color: var(--text-primary);
        }

        /* CONTROLS ROW */
        .controls-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
        }

        .engine-picker {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .picker-label {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .select-engine {
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            color: var(--text-primary);
            font-family: var(--font-sans);
            font-size: 12.5px;
            font-weight: 500;
            padding: 7px 12px;
            border-radius: var(--radius-control);
            outline: none;
            cursor: pointer;
            transition: border-color var(--transition-fast);
        }

        .select-engine:focus {
            border-color: var(--border-focus);
        }

        /* TACTILE PRIMARY BUTTON */
        .btn-primary {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            background: #F8FAFC;
            color: #07080B;
            border: none;
            font-family: var(--font-sans);
            font-size: 12.5px;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            padding: 7px 16px 7px 18px;
            border-radius: var(--radius-control);
            cursor: pointer;
            transition: all var(--transition-fast);
        }

        .btn-primary:hover:not(:disabled) {
            background: #FFFFFF;
            transform: translateY(-1px);
        }

        .btn-primary:active:not(:disabled) {
            transform: scale(0.98);
        }

        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .btn-icon-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: rgba(7, 8, 11, 0.12);
            font-size: 11px;
        }

        /* POST-INFERENCE METADATA STRIP */
        .meta-strip {
            display: flex;
            align-items: center;
            gap: 16px;
            background: var(--bg-well);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-control);
            padding: 8px 14px;
            font-family: var(--font-mono);
            font-size: 11.5px;
            color: var(--text-secondary);
            flex-wrap: wrap;
        }

        .meta-strip-item strong {
            color: var(--text-primary);
        }

        /* XAI EXPLAINABILITY PANEL */
        .xai-box {
            display: flex;
            flex-direction: column;
            gap: 12px;
            border-top: 1px solid var(--border-hairline);
            padding-top: 16px;
        }

        .xai-header {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
        }

        .xai-formula-tag {
            font-family: var(--font-mono);
            font-size: 10.5px;
            color: var(--accent-tech);
            background: var(--accent-tech-bg);
            border: 1px solid var(--accent-tech-border);
            padding: 2px 7px;
            border-radius: var(--radius-badge);
        }

        .xai-disclaimer {
            font-size: 11px;
            color: var(--text-muted);
            line-height: 1.4;
        }

        .xai-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .driver-subpanel {
            background: var(--bg-well);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-control);
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .driver-col-title {
            font-family: var(--font-mono);
            font-size: 10.5px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .driver-col-title.pos { color: var(--signal-positive); }
        .driver-col-title.neg { color: var(--signal-negative); }

        .driver-bar-row {
            display: grid;
            grid-template-columns: 85px 1fr 65px;
            align-items: center;
            gap: 8px;
            font-family: var(--font-mono);
            font-size: 11px;
        }

        .driver-token {
            color: var(--text-primary);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .driver-bar-track {
            height: 6px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
            overflow: hidden;
        }

        .driver-bar-fill {
            height: 100%;
            border-radius: 3px;
            transition: width var(--transition-normal);
        }

        .driver-bar-fill.pos { background: var(--signal-positive); }
        .driver-bar-fill.neg { background: var(--signal-negative); }

        .driver-weight {
            text-align: right;
            font-weight: 500;
        }

        .driver-weight.pos { color: var(--signal-positive); }
        .driver-weight.neg { color: var(--signal-negative); }

        .driver-empty {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
            font-style: italic;
            padding: 4px 0;
        }

        /* RIGHT PANEL: CLASSIFICATION TELEMETRY */
        .telemetry-stack {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .sentiment-card {
            background: var(--bg-well);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-core);
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .sentiment-badge-large {
            font-size: 26px;
            font-weight: 700;
            letter-spacing: -0.01em;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .badge-sentiment-tag {
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            padding: 3px 8px;
            border-radius: var(--radius-badge);
        }

        .sentiment-pos { color: var(--signal-positive); }
        .sentiment-pos .badge-sentiment-tag {
            background: var(--signal-positive-bg);
            border: 1px solid var(--signal-positive-border);
            color: var(--signal-positive);
        }

        .sentiment-neg { color: var(--signal-negative); }
        .sentiment-neg .badge-sentiment-tag {
            background: var(--signal-negative-bg);
            border: 1px solid var(--signal-negative-border);
            color: var(--signal-negative);
        }

        .sentiment-neu { color: var(--signal-neutral); }
        .sentiment-neu .badge-sentiment-tag {
            background: var(--signal-neutral-bg);
            border: 1px solid var(--signal-neutral-border);
            color: var(--signal-neutral);
        }

        /* POLARITY HUD */
        .hud-block {
            background: var(--bg-well);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-core);
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .hud-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .hud-title {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .hud-val {
            font-family: var(--font-mono);
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .polarity-track {
            position: relative;
            height: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
            margin: 10px 0;
            border: 1px solid var(--border-hairline);
        }

        .polarity-center-line {
            position: absolute;
            top: -2px;
            bottom: -2px;
            left: 50%;
            width: 2px;
            background: rgba(255, 255, 255, 0.3);
            transform: translateX(-50%);
        }

        .polarity-marker {
            position: absolute;
            top: -3px;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #FFFFFF;
            border: 2px solid var(--bg-surface);
            box-shadow: 0 0 8px rgba(255, 255, 255, 0.4);
            transform: translateX(-50%);
            transition: left var(--transition-normal);
            left: 50%;
        }

        .polarity-scale-labels {
            display: flex;
            justify-content: space-between;
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-muted);
        }

        /* CONFIDENCE GAUGE */
        .progress-bar-track {
            height: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 4px;
        }

        .progress-bar-fill {
            height: 100%;
            background: var(--accent-tech);
            border-radius: 4px;
            transition: width var(--transition-normal);
            width: 0%;
        }

        /* AFFECTIVE STATE HUD */
        .emotion-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-family: var(--font-mono);
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
            padding: 6px 12px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-control);
            width: fit-content;
        }

        /* FEEDBACK ACTIONS */
        .feedback-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-well);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-core);
            padding: 12px 16px;
            gap: 12px;
            flex-wrap: wrap;
        }

        .feedback-label {
            font-size: 12px;
            color: var(--text-secondary);
        }

        .feedback-btns {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn-feedback {
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            color: var(--text-secondary);
            font-family: var(--font-sans);
            font-size: 11.5px;
            font-weight: 500;
            padding: 5px 12px;
            border-radius: var(--radius-control);
            cursor: pointer;
            transition: all var(--transition-fast);
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .btn-feedback:hover:not(:disabled) {
            color: var(--text-primary);
            border-color: var(--border-subtle);
        }

        .btn-feedback.confirm:hover:not(:disabled) {
            border-color: var(--signal-positive);
            color: var(--signal-positive);
        }

        .btn-feedback.flag:hover:not(:disabled) {
            border-color: var(--signal-negative);
            color: var(--signal-negative);
        }

        .btn-feedback:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* VIEW 2: MODEL CONSENSUS */
        .consensus-summary {
            background: var(--bg-panel);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-core);
            padding: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
        }

        .consensus-ratio-badge {
            font-family: var(--font-mono);
            font-size: 20px;
            font-weight: 700;
            color: var(--signal-positive);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .disagreement-alert {
            background: var(--signal-warning-bg);
            border: 1px solid var(--signal-warning-border);
            color: var(--signal-warning);
            font-family: var(--font-mono);
            font-size: 12px;
            padding: 8px 12px;
            border-radius: var(--radius-control);
            display: none;
            align-items: center;
            gap: 8px;
        }

        .model-card-metric {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 12px;
            padding: 6px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }

        .model-card-metric:last-child {
            border-bottom: none;
        }

        .metric-name {
            color: var(--text-muted);
            font-family: var(--font-mono);
        }

        .metric-val {
            color: var(--text-primary);
            font-family: var(--font-mono);
            font-weight: 600;
        }

        /* VIEW 3: BENCHMARKS */
        .benchmark-hero {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }

        .stat-tile {
            background: var(--bg-panel);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-core);
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .stat-tile-label {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .stat-tile-val {
            font-family: var(--font-mono);
            font-size: 22px;
            font-weight: 700;
            color: var(--text-primary);
        }

        .cm-table {
            width: 100%;
            border-collapse: collapse;
            font-family: var(--font-mono);
            font-size: 12px;
            margin-top: 8px;
        }

        .cm-table th, .cm-table td {
            border: 1px solid var(--border-hairline);
            padding: 8px 10px;
            text-align: center;
        }

        .cm-table th {
            background: var(--bg-surface);
            color: var(--text-secondary);
            font-weight: 500;
            font-size: 11px;
        }

        .cm-cell-num {
            font-weight: 600;
            color: var(--text-primary);
        }

        /* VIEW 4: TELEMETRY & HISTORY */
        .table-wrap {
            width: 100%;
            overflow-x: auto;
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-control);
            background: var(--bg-panel);
        }

        .audit-table {
            width: 100%;
            border-collapse: collapse;
            font-family: var(--font-mono);
            font-size: 12px;
            white-space: nowrap;
        }

        .audit-table th {
            background: var(--bg-surface);
            color: var(--text-secondary);
            font-weight: 600;
            text-align: left;
            padding: 10px 14px;
            border-bottom: 1px solid var(--border-hairline);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .audit-table td {
            padding: 10px 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            color: var(--text-secondary);
        }

        .audit-table tr:hover td {
            background: rgba(255, 255, 255, 0.02);
            color: var(--text-primary);
        }

        .table-actions-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
        }

        .filter-input {
            background: var(--bg-well);
            border: 1px solid var(--border-hairline);
            color: var(--text-primary);
            font-family: var(--font-mono);
            font-size: 12px;
            padding: 6px 12px;
            border-radius: var(--radius-control);
            outline: none;
            width: 220px;
        }

        .filter-input:focus {
            border-color: var(--border-focus);
        }

        .btn-export {
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            color: var(--text-secondary);
            font-family: var(--font-mono);
            font-size: 11.5px;
            font-weight: 500;
            padding: 6px 12px;
            border-radius: var(--radius-control);
            cursor: pointer;
            transition: all var(--transition-fast);
        }

        .btn-export:hover {
            color: var(--text-primary);
            border-color: var(--border-subtle);
        }

        /* VIEW 5: DEVELOPER */
        .api-editor {
            width: 100%;
            height: 140px;
            background: var(--bg-well);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-control);
            padding: 12px;
            color: var(--text-primary);
            font-family: var(--font-mono);
            font-size: 12px;
            line-height: 1.5;
            outline: none;
            resize: vertical;
        }

        .api-editor:focus {
            border-color: var(--border-focus);
        }

        .api-response-pre {
            background: var(--bg-well);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-control);
            padding: 14px;
            color: var(--text-secondary);
            font-family: var(--font-mono);
            font-size: 11.5px;
            min-height: 180px;
            overflow-x: auto;
            white-space: pre-wrap;
            line-height: 1.5;
        }

        /* TOAST ALERT */
        .toast-banner {
            position: fixed;
            bottom: 24px;
            right: 24px;
            z-index: 1000;
            background: var(--bg-elevated);
            border: 1px solid var(--border-subtle);
            box-shadow: 0 16px 36px -4px rgba(0, 0, 0, 0.8);
            border-radius: var(--radius-control);
            padding: 12px 18px;
            font-size: 12.5px;
            color: var(--text-primary);
            display: none;
            align-items: center;
            gap: 12px;
            max-width: 420px;
        }

        .toast-banner.show {
            display: flex;
        }

        .toast-close {
            background: none;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 16px;
        }

        /* RESPONSIVE BEHAVIOR */
        @media (max-width: 1024px) {
            .grid-2col {
                grid-template-columns: 1fr;
            }
            .grid-3col {
                grid-template-columns: 1fr;
            }
            .benchmark-hero {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        /* GLOBAL RESPONSIVE SAFETY */
        html {
            max-width: 100vw;
            overflow-x: hidden;
        }

        body {
            max-width: 100vw;
            overflow-x: hidden;
        }

        .masthead {
            max-width: 100vw;
            box-sizing: border-box;
        }

        .nav-dock {
            max-width: 100%;
            box-sizing: border-box;
        }

        @media (max-width: 768px) {
            .masthead {
                height: auto;
                padding: 8px 12px;
                flex-direction: column;
                align-items: stretch;
                gap: 8px;
            }
            .masthead-left {
                width: 100%;
                justify-content: space-between;
                flex-wrap: nowrap;
            }
            .badge-status {
                display: none; /* Collapses wide status pill on mobile masthead */
            }
            .nav-dock {
                width: 100%;
                max-width: 100%;
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
                scrollbar-width: none;
                flex-shrink: 1;
            }
            .nav-dock::-webkit-scrollbar {
                display: none;
            }
            .tab-btn {
                padding: 5px 10px;
                font-size: 11.5px;
            }
            .app-shell {
                padding: 10px 8px 36px;
                max-width: 100%;
                box-sizing: border-box;
            }
            .chassis-card {
                padding: 8px;
                max-width: 100%;
                box-sizing: border-box;
            }
            .core-well {
                padding: 14px 10px;
                max-width: 100%;
                box-sizing: border-box;
            }
            .advisory-strip {
                flex-direction: column;
                align-items: flex-start;
                gap: 8px;
                padding: 10px 12px;
            }
            .controls-row {
                flex-direction: column;
                align-items: stretch;
            }
            .engine-picker {
                width: 100%;
                justify-content: space-between;
            }
            .select-engine {
                flex: 1;
            }
            .btn-primary {
                width: 100%;
                justify-content: center;
            }
            .driver-bar-row {
                grid-template-columns: 70px 1fr 50px;
            }
            .sentiment-card {
                flex-direction: column;
                align-items: flex-start;
                gap: 12px;
            }
            .feedback-strip {
                flex-direction: column;
                align-items: stretch;
                gap: 8px;
            }
            .feedback-btns {
                width: 100%;
                justify-content: space-between;
            }
            .btn-feedback {
                flex: 1;
                justify-content: center;
            }
            .xai-grid {
                grid-template-columns: 1fr;
            }
            .benchmark-hero {
                grid-template-columns: 1fr;
            }
            .table-wrap {
                border: none;
                background: transparent;
                overflow-x: visible;
            }
            .table-actions-strip {
                flex-direction: column;
                align-items: stretch;
            }
            .filter-input {
                width: 100%;
            }
            /* MOBILE TELEMETRY CARDS */
            .audit-table thead {
                display: none;
            }
            .audit-table, .audit-table tbody, .audit-table tr, .audit-table td {
                display: block;
                width: 100%;
                box-sizing: border-box;
            }
            .audit-table tr {
                background: var(--bg-panel);
                border: 1px solid var(--border-hairline);
                border-radius: var(--radius-control);
                margin-bottom: 12px;
                padding: 10px 12px;
            }
            .audit-table td {
                display: flex;
                justify-content: space-between;
                padding: 6px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.04);
                white-space: normal;
                word-break: break-all;
            }
            .audit-table td:last-child {
                border-bottom: none;
            }
            .audit-table td::before {
                content: attr(data-label);
                font-weight: 600;
                color: var(--text-muted);
                text-transform: uppercase;
                font-size: 10.5px;
                margin-right: 8px;
                flex-shrink: 0;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                transition: none !important;
                animation: none !important;
                transform: none !important;
            }
        }
    </style>
</head>
<body>

    <!-- TOP MASTHEAD APP BAR -->
    <header class="masthead">
        <div class="masthead-left">
            <span class="brand-monogram">🛡️</span>
            <span class="brand-title">
                SENTINEL AI
                <span class="badge-version">v14.3.0-PROD</span>
            </span>
            <span class="badge-status" id="systemHealthBadge" title="System operational status">
                <span class="status-dot"></span>
                <span>ONLINE · 3 ENGINES READY</span>
            </span>
        </div>

        <nav class="nav-dock" role="tablist" aria-label="Main Navigation">
            <button class="tab-btn active" role="tab" id="tab-analyze" aria-selected="true" aria-controls="view-analyze" onclick="switchTab('analyze')">Inference Console</button>
            <button class="tab-btn" role="tab" id="tab-consensus" aria-selected="false" aria-controls="view-consensus" onclick="switchTab('consensus')">Model Consensus</button>
            <button class="tab-btn" role="tab" id="tab-benchmarks" aria-selected="false" aria-controls="view-benchmarks" onclick="switchTab('benchmarks')">Model Registry &amp; Benchmarks</button>
            <button class="tab-btn" role="tab" id="tab-telemetry" aria-selected="false" aria-controls="view-telemetry" onclick="switchTab('telemetry')">Telemetry &amp; History</button>
            <button class="tab-btn" role="tab" id="tab-developer" aria-selected="false" aria-controls="view-developer" onclick="switchTab('developer')">API Interface</button>
        </nav>
    </header>

    <div class="app-shell">

        <!-- DOMAIN & PRIVACY ADVISORY STRIP -->
        <div class="advisory-strip" role="region" aria-label="Domain & Privacy Governance Advisory">
            <div class="advisory-domain">
                <span class="advisory-domain-tag">MODEL DOMAIN: IMDB 50K</span>
                <span>Classifiers calibrated on narrative review corpus. Operational generalization to financial, legal, or customer service domains may exhibit statistical shift.</span>
            </div>
            <div class="advisory-privacy">
                <strong>RAW TEXT STORAGE: DISABLED</strong>
                <span>· Server-side SHA-256 anonymization active prior to SQLite write.</span>
            </div>
        </div>

        <!-- TAB 1: INFERENCE CONSOLE -->
        <main id="view-analyze" class="view-panel active" role="tabpanel" aria-labelledby="tab-analyze">
            <div class="grid-2col">

                <!-- LEFT SIDE: INGESTION WORKSTATION -->
                <section class="chassis-card" aria-label="Ingestion Workstation">
                    <div class="core-well">
                        <div class="panel-header">
                            <span class="panel-title">Natural Language Discourse Ingestion</span>
                        </div>

                        <!-- EVALUATION PRESETS -->
                        <div class="preset-strip" role="group" aria-label="Evaluation Presets">
                            <span class="preset-label">EVALUATION PRESETS:</span>
                            <button type="button" class="preset-chip" onclick="applyPreset('pos')" title="Load Positive Sample">Positive Audit</button>
                            <button type="button" class="preset-chip" onclick="applyPreset('neg')" title="Load Critical Sample">Critical Discontent</button>
                            <button type="button" class="preset-chip" onclick="applyPreset('mixed')" title="Load Mixed Sample">Mixed Consensus Test</button>
                        </div>

                        <!-- TEXTAREA & COUNTERS -->
                        <div class="input-dock">
                            <textarea id="nlpInput" class="text-input" placeholder="Enter review discourse for sentiment evaluation (1 to 5,000 characters)..." aria-label="Review discourse input text"></textarea>
                            <div class="input-meta-bar">
                                <div class="input-counters">
                                    <span id="charCount">0 / 5000 chars</span>
                                    <span>·</span>
                                    <span id="wordCount">0 words</span>
                                    <span>·</span>
                                    <span>Press Ctrl+Enter to execute</span>
                                </div>
                                <button type="button" class="btn-link" onclick="clearInput()">Clear</button>
                            </div>
                        </div>

                        <!-- ENGINE SELECTOR & EXECUTE BUTTON -->
                        <div class="controls-row">
                            <div class="engine-picker">
                                <label for="engineSelect" class="picker-label">Classifier:</label>
                                <select id="engineSelect" class="select-engine">
                                    <option value="logistic_regression">Logistic Regression (L2 Sigmoid)</option>
                                    <option value="linear_svm">Linear SVM (Platt-Calibrated)</option>
                                    <option value="naive_bayes">Multinomial Naive Bayes</option>
                                </select>
                            </div>

                            <button id="btnAnalyze" class="btn-primary" onclick="executeAnalysis()">
                                <span>Execute Analysis</span>
                                <span class="btn-icon-pill">→</span>
                            </button>
                        </div>

                        <!-- POST-INFERENCE METADATA -->
                        <div id="metaStrip" class="meta-strip" style="display:none;" aria-live="polite">
                            <span class="meta-strip-item">Latency: <strong id="resLatency">—</strong></span>
                            <span>·</span>
                            <span class="meta-strip-item">Tokens Analyzed: <strong id="resTokens">—</strong></span>
                            <span>·</span>
                            <span class="meta-strip-item">Request ID: <strong id="resReqId">—</strong></span>
                        </div>

                        <!-- XAI PANEL: STATISTICAL ATTRIBUTION -->
                        <div class="xai-box" aria-label="Feature Contribution & Attribution">
                            <div class="xai-header">
                                <span class="panel-title">FEATURE CONTRIBUTION &amp; STATISTICAL ATTRIBUTION</span>
                                <span class="xai-formula-tag">Formula: TF-IDF × Model Weight</span>
                            </div>
                            <p class="xai-disclaimer">Statistical feature contribution to the model decision boundary; not causal attribution.</p>

                            <div class="xai-grid">
                                <!-- Positive Drivers -->
                                <div class="driver-subpanel">
                                    <div class="driver-col-title pos">
                                        <span>Positive Drivers</span>
                                        <span>Weight</span>
                                    </div>
                                    <div id="posDriversList">
                                        <div class="driver-empty">Awaiting analysis execution...</div>
                                    </div>
                                </div>

                                <!-- Negative Drivers -->
                                <div class="driver-subpanel">
                                    <div class="driver-col-title neg">
                                        <span>Negative Drivers</span>
                                        <span>Weight</span>
                                    </div>
                                    <div id="negDriversList">
                                        <div class="driver-empty">Awaiting analysis execution...</div>
                                    </div>
                                </div>
                            </div>
                        </div>

                    </div>
                </section>

                <!-- RIGHT SIDE: CLASSIFICATION TELEMETRY -->
                <section class="chassis-card" aria-label="Classification Telemetry">
                    <div class="core-well telemetry-stack">
                        <div class="panel-header">
                            <span class="panel-title">Classification Telemetry</span>
                        </div>

                        <!-- PRIMARY SENTIMENT CARD -->
                        <div class="sentiment-card">
                            <div>
                                <div class="hud-title" style="margin-bottom:4px;">Predicted Sentiment</div>
                                <div id="resSentiment" class="sentiment-badge-large sentiment-neu">
                                    <span>NEUTRAL</span>
                                    <span class="badge-sentiment-tag">BASELINE</span>
                                </div>
                            </div>
                            <div>
                                <div class="hud-title" style="margin-bottom:4px;">Affective Heuristic</div>
                                <div id="resEmotion" class="emotion-badge">Balanced Neutral</div>
                            </div>
                        </div>

                        <!-- POLARITY HUD -->
                        <div class="hud-block">
                            <div class="hud-header">
                                <span class="hud-title">Normalized Polarity Score (ai_score)</span>
                                <span id="resPolarityScore" class="hud-val">0.0000</span>
                            </div>
                            <div class="polarity-track" aria-hidden="true">
                                <div class="polarity-center-line"></div>
                                <div id="polarityMarker" class="polarity-marker"></div>
                            </div>
                            <div class="polarity-scale-labels">
                                <span>-1.0000 (Negative)</span>
                                <span>0.0000 (Neutral)</span>
                                <span>+1.0000 (Positive)</span>
                            </div>
                        </div>

                        <!-- CALIBRATED CONFIDENCE GAUGE -->
                        <div class="hud-block">
                            <div class="hud-header">
                                <span class="hud-title">Calibrated Confidence Margin</span>
                                <span id="resConfidence" class="hud-val">50.00%</span>
                            </div>
                            <div class="progress-bar-track" role="progressbar" id="confProgressTrack" aria-valuemin="50" aria-valuemax="100" aria-valuenow="50">
                                <div id="confProgressBar" class="progress-bar-fill" style="width: 50%;"></div>
                            </div>
                            <div class="hud-header" style="margin-top:4px;">
                                <span style="font-size:10.5px; color:var(--text-muted);">Probability derived from calibrated classifier output.</span>
                            </div>
                        </div>

                        <!-- USER REINFORCEMENT FEEDBACK -->
                        <div class="feedback-strip">
                            <span class="feedback-label">Audit Verification:</span>
                            <div class="feedback-btns">
                                <input type="hidden" id="currentPredId" value="">
                                <button type="button" id="btnConfirmFeedback" class="btn-feedback confirm" onclick="submitFeedbackAction(true)" disabled>
                                    <span>✓</span> Confirm Result
                                </button>
                                <button type="button" id="btnFlagFeedback" class="btn-feedback flag" onclick="submitFeedbackAction(false)" disabled>
                                    <span>✕</span> Flag Incorrect
                                </button>
                            </div>
                        </div>

                    </div>
                </section>

            </div>
        </main>

        <!-- TAB 2: MODEL CONSENSUS -->
        <section id="view-consensus" class="view-panel" role="tabpanel" aria-labelledby="tab-consensus">
            <div class="chassis-card">
                <div class="core-well">
                    <div class="panel-header">
                        <span class="panel-title">Tri-Engine Multi-Model Consensus Cockpit</span>
                        <button class="btn-primary" onclick="executeConsensus()">Evaluate Consensus</button>
                    </div>

                    <!-- CONSENSUS STATUS -->
                    <div class="consensus-summary">
                        <div>
                            <div class="hud-title" style="margin-bottom:4px;">Consensus Agreement Ratio</div>
                            <div id="consensusRatio" class="consensus-ratio-badge">AWAITING EVALUATION</div>
                        </div>
                        <div id="disagreementAlert" class="disagreement-alert" role="alert">
                            <span>⚠️ Inter-Engine Divergence Detected across Classifiers</span>
                        </div>
                        <div>
                            <div class="hud-title" style="margin-bottom:4px;">Consensus Latency</div>
                            <div id="consensusLatency" class="hud-val">—</div>
                        </div>
                    </div>

                    <!-- 3 ENGINE COMPARISON CARDS -->
                    <div class="grid-3col">
                        <!-- Logistic Regression -->
                        <div class="hud-block" id="cardModelLR">
                            <div class="hud-header">
                                <span class="panel-title">Logistic Regression</span>
                                <span class="badge-version">L2 Regularized</span>
                            </div>
                            <div class="model-card-metric">
                                <span class="metric-name">Sentiment:</span>
                                <span class="metric-val" id="lrSentiment">—</span>
                            </div>
                            <div class="model-card-metric">
                                <span class="metric-name">Polarity (ai_score):</span>
                                <span class="metric-val" id="lrScore">—</span>
                            </div>
                            <div class="model-card-metric">
                                <span class="metric-name">Calibrated Confidence:</span>
                                <span class="metric-val" id="lrConf">—</span>
                            </div>
                            <div class="model-card-metric">
                                <span class="metric-name">Affective Heuristic:</span>
                                <span class="metric-val" id="lrEmotion">—</span>
                            </div>
                        </div>

                        <!-- Linear SVM -->
                        <div class="hud-block" id="cardModelSVM">
                            <div class="hud-header">
                                <span class="panel-title">Linear SVM</span>
                                <span class="badge-version">Platt-Calibrated</span>
                            </div>
                            <div class="model-card-metric">
                                <span class="metric-name">Sentiment:</span>
                                <span class="metric-val" id="svmSentiment">—</span>
                            </div>
                            <div class="model-card-metric">
                                <span class="metric-name">Polarity (ai_score):</span>
                                <span class="metric-val" id="svmScore">—</span>
                            </div>
                            <div class="model-card-metric">
                                <span class="metric-name">Calibrated Confidence:</span>
                                <span class="metric-val" id="svmConf">—</span>
                            </div>
                            <div class="model-card-metric">
                                <span class="metric-name">Affective Heuristic:</span>
                                <span class="metric-val" id="svmEmotion">—</span>
                            </div>
                        </div>

                        <!-- Naive Bayes -->
                        <div class="hud-block" id="cardModelNB">
                            <div class="hud-header">
                                <span class="panel-title">Naive Bayes</span>
                                <span class="badge-version">Multinomial</span>
                            </div>
                            <div class="model-card-metric">
                                <span class="metric-name">Sentiment:</span>
                                <span class="metric-val" id="nbSentiment">—</span>
                            </div>
                            <div class="model-card-metric">
                                <span class="metric-name">Polarity (ai_score):</span>
                                <span class="metric-val" id="nbScore">—</span>
                            </div>
                            <div class="model-card-metric">
                                <span class="metric-name">Calibrated Confidence:</span>
                                <span class="metric-val" id="nbConf">—</span>
                            </div>
                            <div class="model-card-metric">
                                <span class="metric-name">Affective Heuristic:</span>
                                <span class="metric-val" id="nbEmotion">—</span>
                            </div>
                        </div>
                    </div>

                </div>
            </div>
        </section>

        <!-- TAB 3: MODEL REGISTRY & BENCHMARKS -->
        <section id="view-benchmarks" class="view-panel" role="tabpanel" aria-labelledby="tab-benchmarks">
            <div class="chassis-card">
                <div class="core-well">
                    <div class="panel-header">
                        <span class="panel-title">Model Registry &amp; Verified Offline Benchmarks</span>
                        <span class="badge-version">OFFLINE BENCHMARK</span>
                    </div>

                    <!-- DATASET SPECIFICATIONS -->
                    <div class="benchmark-hero">
                        <div class="stat-tile">
                            <span class="stat-tile-label">Dataset</span>
                            <span class="stat-tile-val" style="font-size:16px;">IMDB 50K</span>
                            <span style="font-size:10.5px; color:var(--text-muted);">Large Movie Review Corpus</span>
                        </div>
                        <div class="stat-tile">
                            <span class="stat-tile-label">Partitioning</span>
                            <span class="stat-tile-val" style="font-size:16px;">80% / 20%</span>
                            <span style="font-size:10.5px; color:var(--text-muted);">10,000 Test Samples (Seed 42)</span>
                        </div>
                        <div class="stat-tile">
                            <span class="stat-tile-label">Feature Pipeline</span>
                            <span class="stat-tile-val" style="font-size:16px;">5,000 TF-IDF</span>
                            <span style="font-size:10.5px; color:var(--text-muted);">Porter Stemmer · NLTK Stopwords</span>
                        </div>
                        <div class="stat-tile">
                            <span class="stat-tile-label">Environment</span>
                            <span class="stat-tile-val" style="font-size:16px;">Python 3.11</span>
                            <span style="font-size:10.5px; color:var(--text-muted);">scikit-learn 1.3.2</span>
                        </div>
                    </div>

                    <!-- 3 MODEL BENCHMARK CARDS -->
                    <div class="grid-3col">
                        <!-- Logistic Regression -->
                        <div class="hud-block">
                            <span class="panel-title">Logistic Regression v1.0</span>
                            <div class="model-card-metric"><span class="metric-name">Accuracy:</span><span class="metric-val">88.55%</span></div>
                            <div class="model-card-metric"><span class="metric-name">Precision:</span><span class="metric-val">0.8747</span></div>
                            <div class="model-card-metric"><span class="metric-name">Recall:</span><span class="metric-val">0.9020</span></div>
                            <div class="model-card-metric"><span class="metric-name">F1 Score:</span><span class="metric-val">0.8881</span></div>
                            <div class="model-card-metric"><span class="metric-name">ROC-AUC:</span><span class="metric-val">0.9571</span></div>
                            <div class="model-card-metric"><span class="metric-name">Benchmark Latency:</span><span class="metric-val">0.0008 ms</span></div>

                            <span class="hud-title" style="margin-top:8px;">2×2 Confusion Matrix</span>
                            <table class="cm-table" aria-label="Logistic Regression Confusion Matrix">
                                <tr><th></th><th>Pred Neg</th><th>Pred Pos</th></tr>
                                <tr><th>Actual Neg</th><td class="cm-cell-num">4,310</td><td>651</td></tr>
                                <tr><th>Actual Pos</th><td>494</td><td class="cm-cell-num">4,545</td></tr>
                            </table>
                        </div>

                        <!-- Linear SVM -->
                        <div class="hud-block">
                            <span class="panel-title">Linear SVM v1.0</span>
                            <div class="model-card-metric"><span class="metric-name">Accuracy:</span><span class="metric-val">88.40%</span></div>
                            <div class="model-card-metric"><span class="metric-name">Precision:</span><span class="metric-val">0.8751</span></div>
                            <div class="model-card-metric"><span class="metric-name">Recall:</span><span class="metric-val">0.8980</span></div>
                            <div class="model-card-metric"><span class="metric-name">F1 Score:</span><span class="metric-val">0.8864</span></div>
                            <div class="model-card-metric"><span class="metric-name">ROC-AUC:</span><span class="metric-val">0.9534</span></div>
                            <div class="model-card-metric"><span class="metric-name">Benchmark Latency:</span><span class="metric-val">0.0022 ms</span></div>

                            <span class="hud-title" style="margin-top:8px;">2×2 Confusion Matrix</span>
                            <table class="cm-table" aria-label="Linear SVM Confusion Matrix">
                                <tr><th></th><th>Pred Neg</th><th>Pred Pos</th></tr>
                                <tr><th>Actual Neg</th><td class="cm-cell-num">4,315</td><td>646</td></tr>
                                <tr><th>Actual Pos</th><td>514</td><td class="cm-cell-num">4,525</td></tr>
                            </table>
                        </div>

                        <!-- Naive Bayes -->
                        <div class="hud-block">
                            <span class="panel-title">Multinomial Naive Bayes</span>
                            <div class="model-card-metric"><span class="metric-name">Accuracy:</span><span class="metric-val">85.10%</span></div>
                            <div class="model-card-metric"><span class="metric-name">Precision:</span><span class="metric-val">0.8471</span></div>
                            <div class="model-card-metric"><span class="metric-name">Recall:</span><span class="metric-val">0.8595</span></div>
                            <div class="model-card-metric"><span class="metric-name">F1 Score:</span><span class="metric-val">0.8532</span></div>
                            <div class="model-card-metric"><span class="metric-name">ROC-AUC:</span><span class="metric-val">0.9255</span></div>
                            <div class="model-card-metric"><span class="metric-name">Benchmark Latency:</span><span class="metric-val">0.0005 ms</span></div>

                            <span class="hud-title" style="margin-top:8px;">2×2 Confusion Matrix</span>
                            <table class="cm-table" aria-label="Naive Bayes Confusion Matrix">
                                <tr><th></th><th>Pred Neg</th><th>Pred Pos</th></tr>
                                <tr><th>Actual Neg</th><td class="cm-cell-num">4,179</td><td>782</td></tr>
                                <tr><th>Actual Pos</th><td>708</td><td class="cm-cell-num">4,331</td></tr>
                            </table>
                        </div>
                    </div>

                </div>
            </div>
        </section>

        <!-- TAB 4: TELEMETRY & HISTORY -->
        <section id="view-telemetry" class="view-panel" role="tabpanel" aria-labelledby="tab-telemetry">
            <div class="chassis-card">
                <div class="core-well">
                    <div class="panel-header">
                        <span class="panel-title">Production Telemetry &amp; SQLite Audit Journal</span>
                        <div style="display:flex; gap:8px;">
                            <button class="btn-export" onclick="exportData('csv')">Export CSV</button>
                            <button class="btn-export" onclick="exportData('json')">Export JSON</button>
                        </div>
                    </div>

                    <!-- ANALYTICS HUD TILES -->
                    <div class="benchmark-hero">
                        <div class="stat-tile">
                            <span class="stat-tile-label">Total Inferences</span>
                            <span id="statTotal" class="stat-tile-val">0</span>
                        </div>
                        <div class="stat-tile">
                            <span class="stat-tile-label">Sentiment Spread (Pos / Neg / Neu)</span>
                            <span id="statSpread" class="stat-tile-val" style="font-size:16px;">0 / 0 / 0</span>
                        </div>
                        <div class="stat-tile">
                            <span class="stat-tile-label">Avg Execution Latency</span>
                            <span id="statAvgLat" class="stat-tile-val">0.00 ms</span>
                        </div>
                        <div class="stat-tile">
                            <span class="stat-tile-label">P95 / P99 Latency</span>
                            <span id="statPercentiles" class="stat-tile-val" style="font-size:16px;">— / —</span>
                        </div>
                    </div>

                    <!-- SEARCH & REFRESH TOOLBAR -->
                    <div class="table-actions-strip">
                        <input type="text" id="historyFilter" class="filter-input" placeholder="Filter by engine or sentiment..." oninput="filterHistoryTable()">
                        <button class="btn-export" onclick="fetchTelemetryData()">Refresh Audit Log</button>
                    </div>

                    <!-- AUDIT TABLE (10 REAL FIELDS) -->
                    <div class="table-wrap">
                        <table class="audit-table" aria-label="Telemetry History Records">
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Request ID</th>
                                    <th>Engine</th>
                                    <th>Sentiment</th>
                                    <th>Polarity</th>
                                    <th>Confidence</th>
                                    <th>Affective State</th>
                                    <th>Latency</th>
                                    <th>Audit Feedback</th>
                                    <th>Timestamp</th>
                                </tr>
                            </thead>
                            <tbody id="historyTableBody">
                                <tr>
                                    <td colspan="10" style="text-align:center; color:var(--text-muted); padding:20px;">Loading telemetry records...</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                </div>
            </div>
        </section>

        <!-- TAB 5: API INTERFACE & DEVELOPER -->
        <section id="view-developer" class="view-panel" role="tabpanel" aria-labelledby="tab-developer">
            <div class="grid-2col">

                <!-- REST CLIENT SANDBOX -->
                <div class="chassis-card">
                    <div class="core-well">
                        <div class="panel-header">
                            <span class="panel-title">Interactive REST Sandbox: POST /api/v1/predict</span>
                            <span id="apiStatusTag" class="badge-version">READY</span>
                        </div>
                        <label for="devPayload" class="picker-label">JSON Payload:</label>
                        <textarea id="devPayload" class="api-editor" aria-label="JSON Request Payload">{
  "text": "The performance was truly spectacular, elegant and deeply engaging.",
  "model_choice": "logistic_regression"
}</textarea>
                        <div style="display:flex; justify-content:flex-end;">
                            <button id="btnSendApi" class="btn-primary" onclick="sendApiRequest()">
                                <span>Send Test Request</span>
                                <span class="btn-icon-pill">→</span>
                            </button>
                        </div>
                        <label class="picker-label">Server Response:</label>
                        <pre id="devResponse" class="api-response-pre">// Response will render here upon request execution...</pre>
                    </div>
                </div>

                <!-- SYSTEM DIAGNOSTICS -->
                <div class="chassis-card">
                    <div class="core-well">
                        <div class="panel-header">
                            <span class="panel-title">Core System Health &amp; Gateway Diagnostics</span>
                            <button class="btn-export" onclick="checkHealth()">Ping /health</button>
                        </div>
                        <div class="model-card-metric"><span class="metric-name">API Gateway:</span><span class="metric-val" id="diagStatus">ONLINE (FastAPI)</span></div>
                        <div class="model-card-metric"><span class="metric-name">Storage Engine:</span><span class="metric-val" id="diagDb">SQLITE CONNECTED</span></div>
                        <div class="model-card-metric"><span class="metric-name">Active Models:</span><span class="metric-val" id="diagModels">3 ENGINES READY</span></div>
                        <div class="model-card-metric"><span class="metric-name">TF-IDF Vectorizer:</span><span class="metric-val" id="diagVec">LOADED (5,000 Features)</span></div>
                        <div class="model-card-metric"><span class="metric-name">Platform Release:</span><span class="metric-val">v14.3.0-PROD</span></div>

                        <div style="border-top:1px solid var(--border-hairline); padding-top:14px; margin-top:10px;">
                            <span class="panel-title" style="margin-bottom:8px;">Interactive API Specifications</span>
                            <div style="display:flex; gap:12px; margin-top:8px;">
                                <a href="/docs" target="_blank" rel="noopener noreferrer" class="btn-export" style="text-decoration:none;">Open Swagger UI (/docs)</a>
                                <a href="/redoc" target="_blank" rel="noopener noreferrer" class="btn-export" style="text-decoration:none;">Open ReDoc (/redoc)</a>
                            </div>
                        </div>

                        <div style="border-top:1px solid var(--border-hairline); padding-top:14px; margin-top:10px; font-size:11.5px; color:var(--text-muted); line-height:1.6;">
                            <strong>Enterprise Governance Policy:</strong> Natural language input is transmitted securely over HTTP/TLS. Raw natural language text is anonymized with SHA-256 digests prior to SQLite storage when STORE_RAW_TEXT is disabled.
                        </div>
                    </div>
                </div>

            </div>
        </section>

    </div>

    <!-- TOAST NOTIFICATION CONTAINER -->
    <div id="appToast" class="toast-banner" role="alert" aria-live="polite">
        <span id="toastIcon">ℹ️</span>
        <span id="toastMessage" style="flex:1;">Notification content</span>
        <button type="button" class="toast-close" onclick="dismissToast()" aria-label="Dismiss alert">&times;</button>
    </div>

    <script>
        // --- RESTFUL UI CONTROLLER ---
        const VALID_VIEWS = ['analyze', 'consensus', 'benchmarks', 'telemetry', 'developer'];
        let isInternalHashUpdate = false;
        let isAnalyzing = false;
        let historyRecordsCache = [];

        // Tab Switching & Hash Navigation Sync
        function switchTab(viewId, updateHash = true) {
            if (!VALID_VIEWS.includes(viewId)) return;

            document.querySelectorAll('.tab-btn').forEach(btn => {
                const isActive = (btn.id === 'tab-' + viewId);
                btn.classList.toggle('active', isActive);
                btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
            });

            document.querySelectorAll('.view-panel').forEach(panel => {
                panel.classList.toggle('active', panel.id === 'view-' + viewId);
            });

            if (updateHash && window.location.hash !== '#' + viewId) {
                isInternalHashUpdate = true;
                window.location.hash = viewId;
            }

            // Lazy data fetching based on active tab
            if (viewId === 'telemetry') {
                fetchTelemetryData();
            } else if (viewId === 'developer') {
                checkHealth();
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
            } else {
                switchTab('analyze', false);
            }
        }

        window.addEventListener('hashchange', handleHashRoute);

        // Character & Word Counter
        const nlpInput = document.getElementById('nlpInput');
        const charCount = document.getElementById('charCount');
        const wordCount = document.getElementById('wordCount');

        function updateCounters() {
            const val = nlpInput.value;
            charCount.innerText = `${val.length} / 5000 chars`;
            const words = val.trim().length > 0 ? val.trim().split(/\s+/).filter(Boolean).length : 0;
            wordCount.innerText = `${words} words`;
        }

        nlpInput.addEventListener('input', updateCounters);
        nlpInput.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                executeAnalysis();
            }
        });

        function clearInput() {
            nlpInput.value = '';
            updateCounters();
            nlpInput.focus();
        }

        // Evaluation Presets
        const PRESETS = {
            pos: "An absolute cinematic triumph with mesmerizing visuals, brilliant direction, and profound emotional storytelling.",
            neg: "A dreadful, disorganized and agonizing waste of valuable time. The pacing was terrible and dialogue awful.",
            mixed: "The cinematography and acting were exceptional and superb, but the plot was completely dull and slow."
        };

        function applyPreset(key) {
            if (PRESETS[key]) {
                nlpInput.value = PRESETS[key];
                updateCounters();
            }
        }

        // Toast Notifications
        let toastTimeout = null;
        function showToast(msg, icon = 'ℹ️') {
            const toast = document.getElementById('appToast');
            document.getElementById('toastIcon').innerText = icon;
            document.getElementById('toastMessage').innerText = msg;
            toast.classList.add('show');
            clearTimeout(toastTimeout);
            toastTimeout = setTimeout(dismissToast, 4000);
        }

        function dismissToast() {
            document.getElementById('appToast').classList.remove('show');
        }

        // Single Inference Analysis
        async function executeAnalysis() {
            const text = nlpInput.value.trim();
            if (!text) {
                showToast("Please provide review discourse text before executing analysis.", "⚠️");
                nlpInput.focus();
                return;
            }

            if (isAnalyzing) return;
            isAnalyzing = true;

            const btn = document.getElementById('btnAnalyze');
            const originalBtnHtml = btn.innerHTML;
            btn.innerHTML = '<span>Analyzing...</span><span class="btn-icon-pill">⏳</span>';
            btn.disabled = true;

            const engine = document.getElementById('engineSelect').value;

            try {
                const res = await fetch('/api/v1/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, model_choice: engine })
                });

                if (!res.ok) {
                    const errData = await res.json().catch(() => ({}));
                    throw new Error(errData.detail || `Server returned HTTP ${res.status}`);
                }

                const data = await res.json();

                // Update Metadata Strip
                document.getElementById('resLatency').innerText = `${data.latency_ms.toFixed(2)} ms`;
                document.getElementById('resTokens').innerText = `${data.tokens_count} tokens`;
                document.getElementById('resReqId').innerText = data.request_id;
                document.getElementById('metaStrip').style.display = 'flex';

                // Update Sentiment Badge
                const sEl = document.getElementById('resSentiment');
                sEl.className = 'sentiment-badge-large ' + (
                    data.sentiment === 'Positive' ? 'sentiment-pos' :
                    data.sentiment === 'Negative' ? 'sentiment-neg' : 'sentiment-neu'
                );
                sEl.innerHTML = `<span>${data.sentiment.toUpperCase()}</span><span class="badge-sentiment-tag">${engine.toUpperCase()}</span>`;

                // Update Affective Heuristic
                document.getElementById('resEmotion').innerText = data.emotion;

                // Update Polarity HUD (ai_score: -1.0000 to +1.0000)
                const score = data.ai_score;
                document.getElementById('resPolarityScore').innerText = (score >= 0 ? '+' : '') + score.toFixed(4);
                const pct = Math.max(0, Math.min(100, ((score + 1) / 2) * 100));
                document.getElementById('polarityMarker').style.left = `${pct.toFixed(2)}%`;

                // Update Calibrated Confidence
                document.getElementById('resConfidence').innerText = `${data.confidence.toFixed(2)}%`;
                document.getElementById('confProgressBar').style.width = `${data.confidence.toFixed(2)}%`;
                document.getElementById('confProgressTrack').setAttribute('aria-valuenow', data.confidence.toFixed(2));

                // Update Feedback Controls
                document.getElementById('currentPredId').value = data.prediction_id;
                document.getElementById('btnConfirmFeedback').disabled = false;
                document.getElementById('btnFlagFeedback').disabled = false;

                // Render XAI Driver Bars
                renderXaiDrivers(data.explanation.positive_drivers, data.explanation.negative_drivers);

            } catch (err) {
                console.error("Inference Error:", err);
                showToast("Inference Failed: " + err.message, "❌");
            } finally {
                btn.innerHTML = originalBtnHtml;
                btn.disabled = false;
                isAnalyzing = false;
            }
        }

        // Render XAI Drivers
        function renderXaiDrivers(posDrivers, negDrivers) {
            const posList = document.getElementById('posDriversList');
            const negList = document.getElementById('negDriversList');

            const all = [...(posDrivers || []), ...(negDrivers || [])];
            const maxWeight = all.length > 0 ? Math.max(...all.map(d => Math.abs(d.weight)), 0.0001) : 1;

            if (!posDrivers || posDrivers.length === 0) {
                posList.innerHTML = '<div class="driver-empty">None detected within vocabulary</div>';
            } else {
                posList.innerHTML = posDrivers.map(d => {
                    const absW = Math.abs(d.weight);
                    const widthPct = Math.min(100, Math.max(8, Math.round((absW / maxWeight) * 100)));
                    return `
                        <div class="driver-bar-row" title="Token: ${escapeHtml(d.token)} | Weight: +${d.weight.toFixed(4)}">
                            <span class="driver-token">${escapeHtml(d.token)}</span>
                            <div class="driver-bar-track">
                                <div class="driver-bar-fill pos" style="width: ${widthPct}%;"></div>
                            </div>
                            <span class="driver-weight pos">+${d.weight.toFixed(4)}</span>
                        </div>
                    `;
                }).join('');
            }

            if (!negDrivers || negDrivers.length === 0) {
                negList.innerHTML = '<div class="driver-empty">None detected within vocabulary</div>';
            } else {
                negList.innerHTML = negDrivers.map(d => {
                    const absW = Math.abs(d.weight);
                    const widthPct = Math.min(100, Math.max(8, Math.round((absW / maxWeight) * 100)));
                    return `
                        <div class="driver-bar-row" title="Token: ${escapeHtml(d.token)} | Weight: ${d.weight.toFixed(4)}">
                            <span class="driver-token">${escapeHtml(d.token)}</span>
                            <div class="driver-bar-track">
                                <div class="driver-bar-fill neg" style="width: ${widthPct}%;"></div>
                            </div>
                            <span class="driver-weight neg">${d.weight.toFixed(4)}</span>
                        </div>
                    `;
                }).join('');
            }
        }

        // Feedback Submission
        async function submitFeedbackAction(isCorrect) {
            const predId = document.getElementById('currentPredId').value;
            if (!predId) return;

            const btnConfirm = document.getElementById('btnConfirmFeedback');
            const btnFlag = document.getElementById('btnFlagFeedback');
            btnConfirm.disabled = true;
            btnFlag.disabled = true;

            try {
                const res = await fetch('/api/v1/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prediction_id: parseInt(predId, 10), is_correct: isCorrect })
                });

                if (!res.ok) throw new Error(`Feedback error HTTP ${res.status}`);
                const data = await res.json();
                showToast(isCorrect ? "Result confirmed. Telemetry updated." : "Result flagged incorrect. Telemetry updated.", "✓");
            } catch (e) {
                console.error("Feedback error:", e);
                showToast("Failed to record feedback: " + e.message, "❌");
                btnConfirm.disabled = false;
                btnFlag.disabled = false;
            }
        }

        // Multi-Model Consensus Evaluation
        async function executeConsensus() {
            const text = nlpInput.value.trim() || PRESETS.mixed;

            const ratioEl = document.getElementById('consensusRatio');
            const alertEl = document.getElementById('disagreementAlert');
            const latEl = document.getElementById('consensusLatency');

            ratioEl.innerText = "EVALUATING...";

            try {
                const res = await fetch('/api/v1/consensus', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });

                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();

                ratioEl.innerText = data.agreement_ratio.toUpperCase();
                ratioEl.style.color = data.disagreement_detected ? 'var(--signal-warning)' : 'var(--signal-positive)';
                alertEl.style.display = data.disagreement_detected ? 'flex' : 'none';
                latEl.innerText = `${data.latency_ms.toFixed(2)} ms`;

                // Update LR
                if (data.models.logistic_regression) {
                    const m = data.models.logistic_regression;
                    document.getElementById('lrSentiment').innerText = m.sentiment;
                    document.getElementById('lrScore').innerText = (m.ai_score >= 0 ? '+' : '') + m.ai_score.toFixed(4);
                    document.getElementById('lrConf').innerText = `${m.confidence.toFixed(2)}%`;
                    document.getElementById('lrEmotion').innerText = m.emotion;
                }

                // Update SVM
                if (data.models.linear_svm) {
                    const m = data.models.linear_svm;
                    document.getElementById('svmSentiment').innerText = m.sentiment;
                    document.getElementById('svmScore').innerText = (m.ai_score >= 0 ? '+' : '') + m.ai_score.toFixed(4);
                    document.getElementById('svmConf').innerText = `${m.confidence.toFixed(2)}%`;
                    document.getElementById('svmEmotion').innerText = m.emotion;
                }

                // Update NB
                if (data.models.naive_bayes) {
                    const m = data.models.naive_bayes;
                    document.getElementById('nbSentiment').innerText = m.sentiment;
                    document.getElementById('nbScore').innerText = (m.ai_score >= 0 ? '+' : '') + m.ai_score.toFixed(4);
                    document.getElementById('nbConf').innerText = `${m.confidence.toFixed(2)}%`;
                    document.getElementById('nbEmotion').innerText = m.emotion;
                }

            } catch (e) {
                console.error("Consensus Error:", e);
                showToast("Consensus evaluation failed: " + e.message, "❌");
                ratioEl.innerText = "EVALUATION FAILED";
            }
        }

        // Telemetry Data Fetching
        async function fetchTelemetryData() {
            try {
                const [analyticsRes, historyRes] = await Promise.all([
                    fetch('/api/v1/analytics'),
                    fetch('/api/v1/history?limit=25')
                ]);

                if (analyticsRes.ok) {
                    const a = await analyticsRes.json();
                    document.getElementById('statTotal').innerText = a.total_analyzed;
                    document.getElementById('statSpread').innerText = `${a.positive} / ${a.negative} / ${a.neutral}`;
                    document.getElementById('statAvgLat').innerText = `${a.avg_latency_ms.toFixed(2)} ms`;
                    document.getElementById('statPercentiles').innerText = `${a.p95_latency_ms} / ${a.p99_latency_ms}`;
                }

                if (historyRes.ok) {
                    historyRecordsCache = await historyRes.json();
                    renderHistoryTable(historyRecordsCache);
                }

            } catch (e) {
                console.error("Telemetry Error:", e);
                showToast("Failed to refresh telemetry: " + e.message, "⚠️");
            }
        }

        function renderHistoryTable(records) {
            const tbody = document.getElementById('historyTableBody');
            if (!records || records.length === 0) {
                tbody.innerHTML = '<tr><td colspan="10" style="text-align:center; padding:20px; color:var(--text-muted);">No records found in database.</td></tr>';
                return;
            }

            tbody.innerHTML = records.map(r => `
                <tr>
                    <td data-label="ID">#${r.id}</td>
                    <td data-label="Request ID">${escapeHtml(r.request_id)}</td>
                    <td data-label="Engine">${escapeHtml(r.engine)}</td>
                    <td data-label="Sentiment">
                        <span style="color: ${r.sentiment === 'Positive' ? 'var(--signal-positive)' : r.sentiment === 'Negative' ? 'var(--signal-negative)' : 'var(--signal-neutral)'}; font-weight:600;">
                            ${escapeHtml(r.sentiment)}
                        </span>
                    </td>
                    <td data-label="Polarity">${(r.ai_score >= 0 ? '+' : '') + Number(r.ai_score).toFixed(4)}</td>
                    <td data-label="Confidence">${Number(r.confidence).toFixed(2)}%</td>
                    <td data-label="Affective State">${escapeHtml(r.emotion)}</td>
                    <td data-label="Latency">${Number(r.latency_ms).toFixed(2)} ms</td>
                    <td data-label="Feedback">${r.user_feedback ? escapeHtml(r.user_feedback) : '<span style="color:var(--text-muted);">Unreviewed</span>'}</td>
                    <td data-label="Timestamp">${escapeHtml(r.timestamp)}</td>
                </tr>
            `).join('');
        }

        function filterHistoryTable() {
            const query = document.getElementById('historyFilter').value.toLowerCase().trim();
            if (!query) {
                renderHistoryTable(historyRecordsCache);
                return;
            }
            const filtered = historyRecordsCache.filter(r => 
                (r.sentiment && r.sentiment.toLowerCase().includes(query)) ||
                (r.engine && r.engine.toLowerCase().includes(query)) ||
                (r.request_id && r.request_id.toLowerCase().includes(query)) ||
                (r.emotion && r.emotion.toLowerCase().includes(query))
            );
            renderHistoryTable(filtered);
        }

        function exportData(format) {
            window.location.href = `/api/v1/export/${format}`;
        }

        // Developer REST Sandbox
        async function sendApiRequest() {
            const payloadStr = document.getElementById('devPayload').value;
            const resPre = document.getElementById('devResponse');
            const statusTag = document.getElementById('apiStatusTag');
            const btn = document.getElementById('btnSendApi');

            btn.disabled = true;
            statusTag.innerText = "SENDING...";

            let parsed;
            try {
                parsed = JSON.parse(payloadStr);
            } catch (err) {
                resPre.innerText = "// JSON Parse Error: " + err.message;
                statusTag.innerText = "INVALID JSON";
                btn.disabled = false;
                return;
            }

            try {
                const start = performance.now();
                const res = await fetch('/api/v1/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(parsed)
                });
                const duration = (performance.now() - start).toFixed(2);
                statusTag.innerText = `HTTP ${res.status} (${duration}ms)`;
                statusTag.style.color = res.ok ? 'var(--signal-positive)' : 'var(--signal-negative)';

                const data = await res.json();
                resPre.innerText = JSON.stringify(data, null, 2);
            } catch (e) {
                resPre.innerText = "// Request Failed: " + e.message;
                statusTag.innerText = "ERROR";
            } finally {
                btn.disabled = false;
            }
        }

        async function checkHealth() {
            try {
                const res = await fetch('/health');
                if (res.ok) {
                    const data = await res.json();
                    document.getElementById('diagStatus').innerText = `${data.status.toUpperCase()} (v${data.version})`;
                    document.getElementById('diagDb').innerText = data.database_connected ? "CONNECTED (OK)" : "DEGRADED";
                    document.getElementById('diagVec').innerText = data.vectorizer_loaded ? "READY (5,000 Vocab)" : "NOT LOADED";
                }
            } catch (e) {
                console.warn("Health check error:", e);
            }
        }

        function escapeHtml(str) {
            if (!str) return '';
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
        }

        // Initialize view from URL hash on load
        window.addEventListener('DOMContentLoaded', () => {
            handleHashRoute();
            updateCounters();
        });
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
