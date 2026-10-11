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
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
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
    <!-- BESPOKE GEOMETRIC S-APERTURE PRECISION EMBLEM FAVICON -->
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='7' fill='%23170B10'/><circle cx='16' cy='16' r='13' stroke='%23F4E8D0' stroke-width='1' stroke-opacity='0.3'/><circle cx='16' cy='16' r='10' stroke='%23D6B56A' stroke-width='0.75' stroke-opacity='0.45'/><line x1='16' y1='2' x2='16' y2='5' stroke='%23D6B56A' stroke-width='1.5' stroke-linecap='round'/><line x1='16' y1='27' x2='16' y2='30' stroke='%23D6B56A' stroke-width='1.5' stroke-linecap='round'/><line x1='2' y1='16' x2='5' y2='16' stroke='%23D6B56A' stroke-width='1.5' stroke-linecap='round'/><line x1='27' y1='16' x2='30' y2='16' stroke='%23D6B56A' stroke-width='1.5' stroke-linecap='round'/><path d='M 21 8.5 C 16.5 6.5, 11 8, 11 12.5 C 11 16, 15 16, 16 16' stroke='%23F4E8D0' stroke-width='2' stroke-linecap='round'/><path d='M 11 23.5 C 15.5 25.5, 21 24, 21 19.5 C 21 16, 17 16, 16 16' stroke='%23F4E8D0' stroke-width='2' stroke-linecap='round'/><path d='M 16 12.5 C 14 12.5, 13 13.8, 13 15' stroke='%237A2638' stroke-width='1.5' stroke-linecap='round'/><path d='M 16 19.5 C 18 19.5, 19 18.2, 19 17' stroke='%237A2638' stroke-width='1.5' stroke-linecap='round'/><circle cx='16' cy='16' r='2' fill='%23D6B56A'/></svg>">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500;1,600&family=Geist:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* ==========================================================================
           SENTINEL AI — DESIGN TOKENS & DUAL LUXURY IDENTITY
           Dark: Obsidian + Burgundy + Warm Ivory + Champagne
           Light: White Paper + Deep Burgundy Ink + Champagne Metal
           ========================================================================== */
        :root {
            /* DARK MODE: PRIMARY OBSIDIAN & BURGUNDY PALETTE */
            --bg-void: #090708;
            --bg-deep-burgundy: #170B10;
            --bg-surface: #241017;
            --bg-surface-elevated: #32151F;
            --bg-wine: #3D1824;
            --bg-inset: #13080D;
            --bg-track: #2B121C;

            /* Warm Ivory & Twilight Dark Typography */
            --text-primary: #F4E8D0;   /* Warm Ivory Light */
            --text-secondary: #E7D5E3; /* Soft Twilight */
            --text-muted: #C8C0C4;     /* Technical Neutral */
            --text-dim: #8E7A84;       /* Muted Charcoal Mauve */
            --text-disabled: #62555D;  /* Low contrast disabled only */

            /* Hairline Borders & Material Plates */
            --border-hairline: rgba(244, 232, 208, 0.08);
            --border-subtle: rgba(244, 232, 208, 0.14);
            --border-accent: rgba(214, 181, 106, 0.30);
            --highlight-plate: inset 0 1px 1px rgba(244, 232, 208, 0.06);

            /* Accents */
            --accent-burgundy: #7A2638;
            --accent-deep-burgundy: #5A1825;
            --accent-champagne: #D6B56A;
            --accent-warm-ivory: #F4E8D0;
            --accent-champagne-bg: rgba(214, 181, 106, 0.12);
            --accent-champagne-glow: rgba(214, 181, 106, 0.22);
            --logo-core-fill: rgba(122, 38, 56, 0.45);

            /* Scientific Result States */
            --signal-positive: #8FE3C0; /* Dusty Mint */
            --signal-positive-bg: rgba(143, 227, 192, 0.08);
            --signal-positive-border: rgba(143, 227, 192, 0.25);

            --signal-negative: #C65B62; /* Burnt Rose */
            --signal-negative-deep: #7A2638;
            --signal-negative-bg: rgba(198, 91, 98, 0.08);
            --signal-negative-border: rgba(198, 91, 98, 0.25);

            --signal-neutral: #D6B56A;
            --signal-neutral-bg: rgba(214, 181, 106, 0.08);
            --signal-neutral-border: rgba(214, 181, 106, 0.22);

            /* Navigation Glider & CTAs */
            --glider-bg: #7A2638;
            --glider-border: #D6B56A;
            --glider-text: #F4E8D0;
            --glider-shadow: 0 2px 14px rgba(122, 38, 56, 0.4);

            --btn-cta-bg: #5A1825;
            --btn-cta-text: #F4E8D0;
            --btn-cta-border: #7A2638;
            --btn-cta-hover: #7A2638;
            --btn-cta-glow: rgba(122, 38, 56, 0.35);

            --code-bg: #14080E;
            --code-border: rgba(244, 232, 208, 0.10);
            --liquid-glass-bg: rgba(20, 14, 18, 0.88);

            /* Spacing Rhythm */
            --space-1: 4px;
            --space-2: 8px;
            --space-3: 12px;
            --space-4: 16px;
            --space-5: 20px;
            --space-6: 24px;
            --space-8: 32px;
            --space-10: 40px;
            --space-12: 48px;
            --space-16: 64px;

            /* Radii Hierarchy */
            --radius-primary: 16px;
            --radius-secondary: 12px;
            --radius-technical: 8px;
            --radius-pill: 9999px;

            /* Typography Stacks */
            --font-display: 'Cormorant Garamond', 'Times New Roman', Georgia, serif;
            --font-sans: 'Geist', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            --font-mono: 'JetBrains Mono', 'Geist Mono', monospace;

            /* Physics Easing */
            --ease-smooth: cubic-bezier(0.22, 1, 0.36, 1);
            --ease-glide: cubic-bezier(0.16, 1, 0.3, 1);
            --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
            --ease-tab: cubic-bezier(0.65, 0, 0.35, 1);
            --ease-pop: cubic-bezier(0.18, 1.25, 0.4, 1);
        }

        /* ==========================================================================
           LIGHT MODE: STRICT CONTRAST & LUXURY WHITE PAPER + BURGUNDY INK
           ========================================================================== */
        [data-theme="light"] {
            --bg-void: #FCFDFD;
            --bg-deep-burgundy: #FAF7F7;
            --bg-surface: #FFFFFF;
            --bg-surface-elevated: #FAF7F7;
            --bg-wine: #F4ECEF;
            --bg-inset: #F8F5F6;
            --bg-track: #EBE1E5;

            /* Explicit Light Mode Contrast Hierarchy */
            --text-primary: #120D10;   /* Strong headings & hero values */
            --text-secondary: #30272C; /* Body copy & standard values */
            --text-muted: #574C52;     /* Supporting descriptive copy */
            --text-dim: #4A3E44;       /* Labels, card subheads & metadata */
            --text-disabled: #8A7E84;  /* Low-contrast disabled only */

            --border-hairline: rgba(61, 16, 25, 0.12);
            --border-subtle: rgba(61, 16, 25, 0.18);
            --border-accent: rgba(168, 137, 77, 0.40);
            --highlight-plate: inset 0 1px 1px rgba(255, 255, 255, 0.95), 0 2px 8px rgba(61, 16, 25, 0.04);

            --accent-burgundy: #7A2638;
            --accent-deep-burgundy: #3D1019;
            --accent-champagne: #A8894D;
            --accent-warm-ivory: #FFFDFC;
            --accent-champagne-bg: rgba(168, 137, 77, 0.12);
            --accent-champagne-glow: rgba(168, 137, 77, 0.22);
            --logo-core-fill: rgba(168, 137, 77, 0.22);

            --signal-positive: #197A5A;
            --signal-positive-bg: rgba(25, 122, 90, 0.09);
            --signal-positive-border: rgba(25, 122, 90, 0.30);

            --signal-negative: #A43F48;
            --signal-negative-deep: #5A1825;
            --signal-negative-bg: rgba(164, 63, 72, 0.09);
            --signal-negative-border: rgba(164, 63, 72, 0.30);

            --signal-neutral: #625A60;
            --signal-neutral-bg: rgba(98, 90, 96, 0.09);
            --signal-neutral-border: rgba(98, 90, 96, 0.25);

            --glider-bg: #3D1019;
            --glider-border: #A8894D;
            --glider-text: #FFFFFF;
            --glider-shadow: 0 2px 12px rgba(61, 16, 25, 0.30);

            --btn-cta-bg: #3D1019;
            --btn-cta-text: #FFFFFF;
            --btn-cta-border: #5A1825;
            --btn-cta-hover: #5A1825;
            --btn-cta-glow: rgba(61, 16, 25, 0.25);

            --code-bg: #F8F5F6;
            --code-border: rgba(61, 16, 25, 0.14);
            --liquid-glass-bg: rgba(255, 255, 255, 0.94);
        }

        /* LIGHT MODE EXPLICIT TYPOGRAPHIC & ELEMENT CONTRAST OVERRIDES */
        [data-theme="light"] .brand-title,
        [data-theme="light"] .section-title,
        [data-theme="light"] h1,
        [data-theme="light"] h2,
        [data-theme="light"] h3,
        [data-theme="light"] .polarity-val-big,
        [data-theme="light"] .confidence-val,
        [data-theme="light"] .overview-tile-val,
        [data-theme="light"] .advisory-meta-item strong,
        [data-theme="light"] .cockpit-metric-row strong,
        [data-theme="light"] .consensus-ratio-text {
            color: #120D10 !important;
        }

        [data-theme="light"] .brand-icon-pod {
            color: #3D1019;
            border-color: rgba(61, 16, 25, 0.18);
            box-shadow: 0 2px 10px rgba(61, 16, 25, 0.08);
        }

        [data-theme="light"] .brand-subtitle,
        [data-theme="light"] .section-eyebrow,
        [data-theme="light"] .card-subhead,
        [data-theme="light"] .preset-label,
        [data-theme="light"] .engine-label,
        [data-theme="light"] .overview-tile-label,
        [data-theme="light"] .advisory-strip,
        [data-theme="light"] .textarea-meta-bar,
        [data-theme="light"] .bipolar-legend,
        [data-theme="light"] .methodology-caption,
        [data-theme="light"] .audit-table th,
        [data-theme="light"] .cm-table th,
        [data-theme="light"] .app-footer-inner {
            color: #4A3E44 !important;
        }

        [data-theme="light"] .section-desc,
        [data-theme="light"] .nlp-textarea,
        [data-theme="light"] .cockpit-metric-row,
        [data-theme="light"] .audit-table td,
        [data-theme="light"] .code-viewer,
        [data-theme="light"] .driver-token {
            color: #30272C !important;
        }

        [data-theme="light"] .nlp-textarea::placeholder,
        [data-theme="light"] .search-input::placeholder {
            color: #574C52 !important;
        }

        [data-theme="light"] .btn-preset {
            color: #30272C;
            border-color: rgba(61, 16, 25, 0.16);
            background: #FAF7F7;
        }
        [data-theme="light"] .btn-preset:hover {
            color: #120D10;
            border-color: #A8894D;
            background: rgba(168, 137, 77, 0.12);
        }

        [data-theme="light"] .btn-feedback {
            color: #30272C;
            border-color: rgba(61, 16, 25, 0.16);
            background: #FAF7F7;
        }
        [data-theme="light"] .btn-feedback:hover:not(:disabled) {
            color: #120D10;
            border-color: #A8894D;
        }

        [data-theme="light"] .tab-btn {
            color: #4A3E44;
        }
        [data-theme="light"] .tab-btn:hover {
            color: #120D10;
        }
        [data-theme="light"] .tab-btn.active {
            color: #FFFFFF !important;
        }

        [data-theme="light"] .filter-chip {
            color: #30272C;
            border-color: rgba(61, 16, 25, 0.16);
            background: #FFFFFF;
        }
        [data-theme="light"] .filter-chip:hover {
            color: #120D10;
            border-color: #A8894D;
        }
        [data-theme="light"] .filter-chip.on {
            background: #3D1019 !important;
            color: #FFFFFF !important;
            border-color: #A8894D !important;
        }

        [data-theme="light"] .audit-table th {
            background: #FAF7F7;
            color: #4A3E44;
            border-bottom-color: rgba(61, 16, 25, 0.14);
        }
        [data-theme="light"] .audit-table td {
            border-bottom-color: rgba(61, 16, 25, 0.08);
        }

        [data-theme="light"] .cm-table th {
            background: #FAF7F7;
            border-color: rgba(61, 16, 25, 0.12);
        }
        [data-theme="light"] .cm-table td {
            border-color: rgba(61, 16, 25, 0.12);
        }
        [data-theme="light"] .cm-corner,
        [data-theme="light"] .cm-row-header {
            color: #4A3E44 !important;
            background: #FAF7F7 !important;
        }
        [data-theme="light"] .cm-super-header {
            color: #7A2638 !important;
            background: #FFFFFF !important;
        }
        [data-theme="light"] .api-endpoint-item {
            background: #FFFFFF !important;
            border-color: rgba(61, 16, 25, 0.12) !important;
        }
        [data-theme="light"] .api-path {
            color: #120D10 !important;
        }
        [data-theme="light"] .api-desc {
            color: #4A3E44 !important;
        }
        [data-theme="light"] .badge-method.post {
            background: rgba(122, 38, 56, 0.10) !important;
            color: #7A2638 !important;
            border-color: rgba(122, 38, 56, 0.25) !important;
        }
        [data-theme="light"] .badge-method.get {
            background: rgba(25, 122, 90, 0.10) !important;
            color: #197A5A !important;
            border-color: rgba(25, 122, 90, 0.25) !important;
        }
        [data-theme="light"] .export-action-btn {
            background: #FFFFFF !important;
            border-color: rgba(61, 16, 25, 0.18) !important;
            color: #171216 !important;
        }
        [data-theme="light"] .export-action-btn:hover {
            border-color: #7A2638 !important;
            color: #7A2638 !important;
            background: rgba(122, 38, 56, 0.08) !important;
        }

        *, *::before, *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-void);
            color: var(--text-primary);
            font-family: var(--font-sans);
            font-size: 14px;
            line-height: 1.55;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            min-height: 100vh;
            min-height: 100dvh;
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
            transition: background-color 300ms var(--ease-smooth), color 300ms var(--ease-smooth);
        }

        /* Subtle Ambient Grain Overlay */
        body::before {
            content: '';
            position: fixed;
            inset: 0;
            background-image: radial-gradient(rgba(244, 232, 208, 0.02) 1px, transparent 1px);
            background-size: 28px 28px;
            pointer-events: none;
            z-index: 0;
        }

        /* ==========================================================================
           CONSISTENT CONTENT CONTAINER & MASTHEAD
           ========================================================================== */
        .masthead-wrapper {
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(9, 7, 8, 0.90);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border-bottom: 1px solid var(--border-hairline);
            padding-block: var(--space-4);
            transition: background-color 300ms, border-color 300ms;
        }

        [data-theme="light"] .masthead-wrapper {
            background: rgba(252, 253, 253, 0.94);
        }

        .masthead-inner, .app-shell, .app-footer-inner {
            max-width: 1480px;
            margin-inline: auto;
            padding-inline: clamp(20px, 4vw, 64px);
            box-sizing: border-box;
            width: 100%;
        }

        .masthead-inner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: var(--space-6);
        }

        .brand-cluster {
            display: flex;
            align-items: center;
            gap: var(--space-4);
            flex-shrink: 0;
        }

        .brand-icon-pod {
            width: 38px;
            height: 38px;
            border-radius: var(--radius-technical);
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--accent-champagne);
            box-shadow: 0 0 12px var(--accent-champagne-glow);
            flex-shrink: 0;
            transition: color 240ms, border-color 240ms, background 240ms;
        }

        .brand-logo-svg {
            width: 22px;
            height: 22px;
            display: block;
        }

        .brand-title-wrap {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .brand-title {
            font-family: var(--font-display);
            font-size: 21px;
            font-weight: 600;
            letter-spacing: 0.04em;
            color: var(--text-primary);
            line-height: 1.1;
        }

        .brand-subtitle {
            font-family: var(--font-mono);
            font-size: 10.5px;
            color: var(--text-dim);
            letter-spacing: 0.04em;
        }

        /* STATUS & AMBIENT MODE SWITCH */
        .masthead-right-cluster {
            display: flex;
            align-items: center;
            gap: var(--space-4);
            flex-shrink: 0;
        }

        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 5px 12px;
            border-radius: var(--radius-pill);
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-secondary);
        }

        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--signal-positive);
            box-shadow: 0 0 8px var(--signal-positive);
            animation: pulseDot 2.8s infinite;
        }

        @keyframes pulseDot {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.55; transform: scale(0.85); }
        }

        /* LUXURY THEME SWITCH (.switch.sw-7) */
        .switch.sw-7 {
            position: relative;
            display: inline-block;
            width: 52px;
            height: 28px;
            cursor: pointer;
            flex-shrink: 0;
        }
        .switch.sw-7 input {
            position: absolute;
            opacity: 0;
            width: 100%;
            height: 100%;
            margin: 0;
            cursor: pointer;
            z-index: 2;
        }
        .switch.sw-7 input:focus-visible + .track {
            outline: 2px solid var(--accent-champagne);
            outline-offset: 2px;
        }
        .switch.sw-7 .track {
            position: absolute;
            inset: 0;
            background: var(--bg-deep-burgundy);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-pill);
            transition: background 300ms var(--ease-smooth), border-color 300ms var(--ease-smooth);
            overflow: hidden;
        }
        .switch.sw-7 input:checked + .track {
            background: var(--bg-wine);
            border-color: var(--border-accent);
        }
        .switch.sw-7 .thumb {
            position: absolute;
            top: 2px;
            left: 2px;
            width: 22px;
            height: 22px;
            background: var(--accent-warm-ivory);
            border-radius: 50%;
            transition: transform 320ms cubic-bezier(0.34, 1.56, 0.64, 1), background 300ms, box-shadow 300ms;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.35);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        /* Moon Icon in Dark Mode (unchecked) */
        .switch.sw-7 .thumb::before {
            content: "";
            width: 11px;
            height: 11px;
            border-radius: 50%;
            box-shadow: inset 3px -1.5px 0 0 var(--accent-burgundy);
            transform: rotate(-20deg);
            transition: opacity 240ms, transform 320ms cubic-bezier(0.34, 1.56, 0.64, 1);
        }
        /* Sun in Light Mode (checked) */
        .switch.sw-7 input:checked + .track .thumb {
            transform: translateX(24px);
            background: var(--accent-champagne);
            box-shadow: 0 0 10px var(--accent-champagne-glow);
        }
        .switch.sw-7 input:checked + .track .thumb::before {
            box-shadow: none;
            background: var(--bg-deep-burgundy);
            width: 8px;
            height: 8px;
            border-radius: 50%;
            transform: scale(1);
        }

        /* SIGNATURE NAVIGATION: TAB PILL GLIDE & CONTEXTUAL LIFT */
        .nav-dock {
            position: relative;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: var(--bg-deep-burgundy);
            border: 1px solid var(--border-hairline);
            padding: 4px;
            border-radius: var(--radius-secondary);
            box-shadow: var(--highlight-plate);
        }

        .nav-pill-glider {
            position: absolute;
            top: 4px;
            bottom: 4px;
            left: 4px;
            width: 0;
            background: var(--glider-bg);
            border: 1px solid var(--glider-border);
            border-radius: calc(var(--radius-secondary) - 3px);
            transition: transform 400ms var(--ease-tab), width 400ms var(--ease-tab);
            pointer-events: none;
            z-index: 1;
            box-shadow: var(--glider-shadow);
        }

        .tab-btn {
            position: relative;
            z-index: 2;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 16px;
            border: none;
            background: transparent;
            color: var(--text-dim);
            font-family: var(--font-sans);
            font-size: 13px;
            font-weight: 500;
            border-radius: calc(var(--radius-secondary) - 3px);
            cursor: pointer;
            transition: color 240ms var(--ease-glide), transform 280ms cubic-bezier(0.22, 1, 0.36, 1);
            white-space: nowrap;
        }

        .tab-btn .svg-icon {
            color: var(--text-dim);
            transition: color 240ms var(--ease-glide), transform 280ms cubic-bezier(0.22, 1, 0.36, 1);
        }

        .tab-btn:hover {
            color: var(--text-primary);
            transform: translateY(-3px) scale(1.03);
        }

        .tab-btn:hover .svg-icon {
            color: var(--accent-champagne);
        }

        /* Contextual Dock Hover Lift for Immediate Neighbours */
        .nav-dock:hover .tab-btn:has(+ .tab-btn:hover),
        .nav-dock:hover .tab-btn:hover + .tab-btn {
            transform: translateY(-1px) scale(1.01);
            color: var(--text-secondary);
        }

        .tab-btn.active {
            color: var(--glider-text);
            font-weight: 600;
        }

        .tab-btn.active .svg-icon {
            stroke: var(--glider-text);
            color: var(--glider-text);
        }

        .svg-icon {
            width: 15px;
            height: 15px;
            stroke: currentColor;
            stroke-width: 1.8;
            fill: none;
            stroke-linecap: round;
            stroke-linejoin: round;
            flex-shrink: 0;
            transition: stroke 240ms var(--ease-glide);
        }

        /* ==========================================================================
           MAIN APPLICATION SHELL & EDITORIAL HEADERS
           ========================================================================== */
        .app-shell {
            padding-block: var(--space-8) var(--space-16);
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: var(--space-10);
            position: relative;
            z-index: 1;
        }

        .section-header {
            display: flex;
            flex-direction: column;
            gap: var(--space-1);
            margin-bottom: var(--space-8);
        }

        .section-eyebrow {
            font-family: var(--font-mono);
            font-size: 11px;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--accent-champagne);
            font-weight: 600;
        }

        .section-title {
            font-family: var(--font-display);
            font-size: 32px;
            font-style: italic;
            font-weight: 500;
            letter-spacing: -0.015em;
            color: var(--text-primary);
            display: flex;
            align-items: baseline;
            gap: 12px;
            line-height: 1.2;
        }

        .section-desc {
            font-size: 14px;
            color: var(--text-muted);
            max-width: 74ch;
            line-height: 1.6;
            margin-top: 2px;
        }

        /* ==========================================================================
           VIEW PANELS & SECTION TRANSITIONS
           ========================================================================== */
        .view-panel {
            display: none;
            width: 100%;
            opacity: 0;
            transform: translateY(6px);
            transition: opacity 350ms var(--ease-glide), transform 350ms var(--ease-glide);
        }

        .view-panel.active {
            display: block;
            opacity: 1;
            transform: translateY(0);
        }

        /* ==========================================================================
           DOUBLE-BEZEL HARDWARE CARDS & SHARED INTERACTION SYSTEM
           ========================================================================== */
        .chassis-card {
            background: var(--bg-deep-burgundy);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-primary);
            padding: var(--space-3);
            box-shadow: var(--highlight-plate), 0 4px 28px rgba(0, 0, 0, 0.45);
            display: flex;
            flex-direction: column;
            min-width: 0;
            max-width: 100%;
            box-sizing: border-box;
            transition: border-color 260ms var(--ease-glide), background-color 260ms, transform 260ms var(--ease-glide), box-shadow 260ms var(--ease-glide);
        }

        .chassis-card:hover {
            border-color: var(--border-subtle);
            transform: translateY(-2px);
            box-shadow: var(--highlight-plate), 0 8px 32px rgba(0, 0, 0, 0.55), 0 0 1px 1px rgba(214, 181, 106, 0.16);
        }

        [data-theme="light"] .chassis-card {
            box-shadow: 0 4px 24px rgba(61, 16, 25, 0.05), var(--highlight-plate);
        }

        [data-theme="light"] .chassis-card:hover {
            border-color: rgba(61, 16, 25, 0.22);
            box-shadow: 0 8px 28px rgba(61, 16, 25, 0.08), var(--highlight-plate), 0 0 1px 1px rgba(168, 137, 77, 0.22);
        }

        .core-plate, .beam-inner {
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            border-radius: calc(var(--radius-primary) - 4px);
            padding: var(--space-8);
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: var(--space-6);
            min-width: 0;
            max-width: 100%;
            box-sizing: border-box;
            transition: background-color 300ms, border-color 300ms;
        }

        /* SENTINEL EDGE LUMINOUS REFLECTION */
        .sentinel-edge {
            position: relative;
        }
        .sentinel-edge::after {
            content: '';
            position: absolute;
            inset: -1px;
            border-radius: inherit;
            pointer-events: none;
            transition: box-shadow 360ms var(--ease-glide);
            box-shadow: 0 0 1px 1px rgba(214, 181, 106, 0.14);
        }
        .sentinel-edge.edge-positive::after {
            box-shadow: 0 0 1px 1px var(--signal-positive), 0 0 18px var(--signal-positive-bg);
        }
        .sentinel-edge.edge-negative::after {
            box-shadow: 0 0 1px 1px var(--signal-negative), 0 0 18px var(--signal-negative-bg);
        }

        /* PURE CSS BORDER BEAM FOR REAL PROCESSING STATES */
        .beam,
        .sentinel-edge.edge-processing {
            position: relative;
            isolation: isolate;
            overflow: hidden;
            box-shadow: 0 0 24px var(--accent-champagne-bg);
        }

        .beam::before,
        .sentinel-edge.edge-processing::before {
            content: "";
            position: absolute;
            inset: -50%;
            background: conic-gradient(
                from 0deg,
                transparent 0 72%,
                var(--accent-burgundy) 80%,
                var(--accent-champagne) 86%,
                var(--accent-warm-ivory) 92%,
                transparent 96%
            );
            animation: sentinel-beam-spin 3s linear infinite;
            pointer-events: none;
            z-index: 0;
        }

        .beam > .core-plate,
        .beam > .beam-inner,
        .sentinel-edge.edge-processing > .core-plate {
            position: relative;
            z-index: 1;
        }

        @keyframes sentinel-beam-spin {
            to { transform: rotate(1turn); }
        }
        @keyframes borderBeamSpin {
            to { transform: rotate(1turn); }
        }

        /* ==========================================================================
           TAB 1: ASYMMETRICAL 58/42 EDITORIAL WORKSTATION (#analyze)
           ========================================================================== */
        .workspace-grid {
            display: grid;
            grid-template-columns: minmax(0, 58%) minmax(0, 42%);
            gap: var(--space-8);
            align-items: start;
            width: 100%;
            max-width: 100%;
            min-width: 0;
        }

        .card-subhead {
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-dim);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* PRESET CHIPS */
        .preset-strip {
            display: flex;
            align-items: center;
            gap: var(--space-2);
            flex-wrap: wrap;
        }

        .preset-label {
            font-family: var(--font-mono);
            font-size: 10.5px;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-right: 4px;
        }

        .btn-preset {
            background: var(--bg-inset);
            border: 1px solid var(--border-hairline);
            color: var(--text-muted);
            font-family: var(--font-mono);
            font-size: 11px;
            padding: 5px 12px;
            border-radius: var(--radius-pill);
            cursor: pointer;
            transition: all 180ms var(--ease-glide);
        }

        .btn-preset:hover {
            color: var(--text-primary);
            border-color: var(--accent-champagne);
            background: var(--accent-champagne-bg);
            transform: translateY(-1px);
        }

        .btn-preset:active {
            transform: scale(0.985);
        }

        .btn-preset:focus-visible {
            outline: 2px solid var(--accent-champagne);
            outline-offset: 2px;
        }

        /* TEXTAREA WORKSTATION */
        .textarea-enclosure {
            position: relative;
            background: var(--bg-inset);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-secondary);
            padding: var(--space-5);
            transition: border-color 200ms, box-shadow 200ms;
        }

        .textarea-enclosure:focus-within {
            border-color: var(--accent-champagne);
            box-shadow: 0 0 18px var(--accent-champagne-glow);
        }

        .nlp-textarea {
            width: 100%;
            height: 195px;
            background: transparent;
            border: none;
            color: var(--text-primary);
            font-family: var(--font-mono);
            font-size: 13.5px;
            line-height: 1.68;
            resize: vertical;
            outline: none;
        }

        .nlp-textarea::placeholder {
            color: var(--text-dim);
            font-size: 13px;
        }

        .textarea-meta-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: var(--space-2);
            border-top: 1px solid var(--border-hairline);
            padding-top: var(--space-3);
            margin-top: var(--space-2);
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-dim);
        }

        .textarea-counters {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: var(--space-2);
        }

        .textarea-counters span {
            white-space: nowrap;
        }

        /* CONTROLS ROW */
        .controls-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: var(--space-4);
            flex-wrap: wrap;
        }

        .engine-picker {
            display: flex;
            align-items: center;
            gap: var(--space-2);
            background: var(--bg-inset);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-technical);
            padding: 5px 12px;
        }

        .engine-label {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-dim);
            text-transform: uppercase;
        }

        .engine-select {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-family: var(--font-mono);
            font-size: 12px;
            outline: none;
            cursor: pointer;
        }

        .engine-select option {
            background: var(--bg-surface);
            color: var(--text-primary);
        }

        /* PRIMARY EXECUTE CTA: CONIC GLIMMER BUTTON */
        .btn-execute {
            position: relative;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            padding: 12px 28px;
            font-family: var(--font-sans);
            font-size: 13.5px;
            font-weight: 600;
            color: var(--btn-cta-text);
            background: var(--btn-cta-bg);
            border: 1px solid var(--btn-cta-border);
            border-radius: var(--radius-secondary);
            cursor: pointer;
            overflow: hidden;
            transition: transform 180ms var(--ease-glide), box-shadow 240ms var(--ease-glide);
            box-shadow: 0 2px 14px var(--btn-cta-glow);
            z-index: 1;
        }

        .btn-execute::before {
            content: "";
            position: absolute;
            inset: -100%;
            background: conic-gradient(
                from 0deg,
                transparent 0 65%,
                var(--accent-champagne) 80%,
                var(--accent-warm-ivory) 88%,
                transparent 96%
            );
            opacity: 0;
            transition: opacity 300ms var(--ease-glide);
            animation: conicSpin 3.6s linear infinite;
            z-index: -2;
            pointer-events: none;
        }

        .btn-execute::after {
            content: "";
            position: absolute;
            inset: 1px;
            background: var(--btn-cta-bg);
            border-radius: calc(var(--radius-secondary) - 1px);
            transition: background 240ms;
            z-index: -1;
            pointer-events: none;
        }

        .btn-execute:hover::before,
        .btn-execute:focus-visible::before {
            opacity: 1;
        }

        .btn-execute:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 22px var(--btn-cta-glow);
        }

        .btn-execute:focus-visible {
            outline: 2px solid var(--accent-champagne);
            outline-offset: 3px;
        }

        .btn-execute:hover::after {
            background: var(--btn-cta-hover);
        }

        .btn-execute:active {
            transform: scale(0.985);
        }

        @keyframes conicSpin {
            to { transform: rotate(1turn); }
        }

        .ripple-wave {
            position: absolute;
            border-radius: 50%;
            transform: scale(0);
            animation: rippleAnim 600ms cubic-bezier(0.16, 1, 0.3, 1) forwards;
            background: rgba(255, 255, 255, 0.45);
            pointer-events: none;
        }

        @keyframes rippleAnim {
            to {
                transform: scale(2.6);
                opacity: 0;
            }
        }

        /* ADVISORY METADATA STRIP */
        .advisory-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: var(--space-4) var(--space-5);
            background: var(--bg-inset);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-technical);
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-dim);
        }

        .advisory-meta-group {
            display: flex;
            align-items: baseline;
            gap: var(--space-6);
            flex-wrap: wrap;
        }

        .advisory-meta-item {
            display: flex;
            align-items: baseline;
            gap: 6px;
        }

        .advisory-meta-item .meta-label {
            color: var(--text-dim);
            text-transform: uppercase;
            font-size: 10px;
            letter-spacing: 0.05em;
        }

        .advisory-meta-item strong {
            color: var(--text-secondary);
        }

        /* ==========================================================================
           RESULT INSTRUMENT — CLASSIFICATION RESULT & ADVANCED REVEAL
           ========================================================================== */
        .telemetry-stack {
            display: flex;
            flex-direction: column;
            gap: var(--space-6);
        }

        /* ADVANCED STAGGERED RESULT REVEAL */
        .result-revealing .sentiment-hero-card {
            animation: revealItem 240ms var(--ease-glide) forwards;
        }
        .result-revealing .polarity-hud-box {
            animation: revealItem 240ms var(--ease-glide) 30ms forwards;
        }
        .result-revealing .confidence-container {
            animation: revealItem 240ms var(--ease-glide) 60ms forwards;
        }
        .result-revealing .xai-attribution-panel {
            animation: revealItem 260ms var(--ease-glide) 90ms forwards;
        }

        @keyframes revealItem {
            from {
                opacity: 0.25;
                transform: translateY(4px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .sentiment-hero-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: var(--space-6) var(--space-8);
            border-radius: var(--radius-secondary);
            background: var(--bg-inset);
            border: 1px solid var(--border-hairline);
            gap: var(--space-4);
        }

        .sentiment-large-display {
            font-family: var(--font-display);
            font-size: 38px;
            font-style: italic;
            font-weight: 600;
            letter-spacing: -0.015em;
            display: flex;
            align-items: center;
            gap: 12px;
            line-height: 1.1;
        }

        .sentiment-engine-tag {
            font-family: var(--font-mono);
            font-size: 11px;
            font-style: normal;
            padding: 3px 9px;
            border-radius: var(--radius-pill);
            text-transform: uppercase;
            font-weight: 600;
        }

        /* BIPOLAR POLARITY HUD (-1.0000 to +1.0000) */
        .polarity-hud-box {
            background: var(--bg-inset);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-secondary);
            padding: var(--space-5);
            display: flex;
            flex-direction: column;
            gap: var(--space-2);
        }

        .polarity-hud-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-family: var(--font-mono);
            font-size: 11px;
        }

        .polarity-val-big {
            font-family: var(--font-mono);
            font-size: 20px;
            font-weight: 700;
            color: var(--text-primary);
        }

        .bipolar-track {
            position: relative;
            height: 8px;
            background: var(--bg-track);
            border-radius: var(--radius-pill);
            overflow: hidden;
            margin: 6px 0;
        }

        .bipolar-center-axis {
            position: absolute;
            left: 50%;
            top: 0;
            bottom: 0;
            width: 2px;
            background: var(--accent-champagne);
            z-index: 2;
        }

        .bipolar-fill {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 0;
            transition: left 360ms var(--ease-glide), width 360ms var(--ease-glide), background 240ms;
            border-radius: var(--radius-pill);
        }

        .bipolar-legend {
            display: flex;
            justify-content: space-between;
            font-family: var(--font-mono);
            font-size: 10px;
            color: var(--text-dim);
        }

        /* CONFIDENCE INSTRUMENT */
        .confidence-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-inset);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-secondary);
            padding: var(--space-4) var(--space-5);
        }

        .confidence-val {
            font-family: var(--font-mono);
            font-size: 22px;
            font-weight: 700;
            color: var(--text-primary);
        }

        /* SCIENTIFIC ATTRIBUTION (XAI) */
        .xai-attribution-panel {
            display: flex;
            flex-direction: column;
            gap: var(--space-3);
        }

        .xai-drivers-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: var(--space-6);
        }

        .driver-column {
            display: flex;
            flex-direction: column;
            gap: var(--space-2);
        }

        /* Perfectly Aligned Driver Grid: Token | Bar | Weight */
        .driver-bar-row {
            display: grid;
            grid-template-columns: 100px 1fr 65px;
            align-items: center;
            gap: 10px;
            font-family: var(--font-mono);
            font-size: 11px;
            padding: 5px 0;
            border-bottom: 1px solid var(--border-hairline);
        }

        .driver-token {
            color: var(--text-secondary);
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .driver-bar-track {
            background: var(--bg-track);
            height: 6px;
            border-radius: var(--radius-pill);
            overflow: hidden;
            position: relative;
        }

        .driver-bar-fill {
            height: 100%;
            border-radius: var(--radius-pill);
        }

        .driver-weight {
            font-weight: 600;
            text-align: right;
            font-family: var(--font-mono);
        }

        .driver-empty-hint {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-dim);
            font-style: italic;
            padding: 8px 0;
        }

        .methodology-caption {
            font-family: var(--font-mono);
            font-size: 10.5px;
            color: var(--text-dim);
            line-height: 1.5;
            border-top: 1px solid var(--border-hairline);
            padding-top: var(--space-2);
            margin-top: var(--space-1);
        }

        /* AUDIT FEEDBACK STRIP WITH SUCCESS ANIMATION */
        .audit-feedback-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-top: var(--space-5);
            border-top: 1px solid var(--border-hairline);
            gap: var(--space-3);
            flex-wrap: wrap;
        }

        .btn-feedback {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: var(--bg-inset);
            border: 1px solid var(--border-hairline);
            color: var(--text-muted);
            font-family: var(--font-mono);
            font-size: 11.5px;
            padding: 7px 16px;
            border-radius: var(--radius-technical);
            cursor: pointer;
            transition: all 180ms var(--ease-glide);
        }

        .btn-feedback:hover:not(:disabled) {
            color: var(--text-primary);
            border-color: var(--accent-champagne);
            transform: translateY(-1px);
        }

        .btn-feedback:active:not(:disabled) {
            transform: scale(0.985);
        }

        .btn-feedback:focus-visible {
            outline: 2px solid var(--accent-champagne);
            outline-offset: 2px;
        }

        .btn-feedback:disabled {
            opacity: 0.45;
            cursor: not-allowed;
        }

        /* CIRCULAR SUCCESS INDICATOR */
        .feedback-success-svg {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: var(--signal-positive);
            font-family: var(--font-mono);
            font-size: 11.5px;
        }

        .checkmark-svg {
            width: 22px;
            height: 22px;
            stroke-width: 2.4;
            stroke: var(--signal-positive);
            stroke-miterlimit: 10;
        }

        .checkmark-circle {
            stroke-dasharray: 166;
            stroke-dashoffset: 166;
            stroke: var(--signal-positive);
            animation: strokeRing 450ms cubic-bezier(0.65, 0, 0.45, 1) forwards;
        }

        .checkmark-check {
            transform-origin: 50% 50%;
            stroke-dasharray: 48;
            stroke-dashoffset: 48;
            animation: strokeCheck 280ms cubic-bezier(0.65, 0, 0.45, 1) 400ms forwards;
        }

        @keyframes strokeRing {
            100% { stroke-dashoffset: 0; }
        }

        @keyframes strokeCheck {
            100% { stroke-dashoffset: 0; }
        }

        /* ==========================================================================
           TAB 2: THREE-INSTRUMENT ARRAY (#consensus)
           ========================================================================== */
        .consensus-cockpit-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: var(--space-8);
        }

        .consensus-banner {
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            padding: var(--space-5) var(--space-8);
            border-radius: var(--radius-primary);
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: var(--space-8);
            flex-wrap: wrap;
            gap: var(--space-4);
        }

        .consensus-ratio-text {
            font-family: var(--font-display);
            font-size: 24px;
            font-style: italic;
            font-weight: 600;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            line-height: 1.1;
        }

        .cockpit-metric-row {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: var(--space-3);
            padding: var(--space-3) 0;
            border-bottom: 1px solid var(--border-hairline);
            font-size: 13px;
        }

        .cockpit-metric-row > span:first-child {
            flex-shrink: 0;
        }

        .cockpit-metric-row > :last-child {
            text-align: right;
            word-break: break-word;
        }

        .cockpit-metric-row:last-child {
            border-bottom: none;
        }

        /* ==========================================================================
           TAB 3: RESEARCH & CALIBRATION BENCHMARKS (#benchmarks)
           ========================================================================== */
        .benchmark-metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: var(--space-8);
        }

        .overview-tile-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: var(--space-5);
        }

        .overview-tile {
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-secondary);
            padding: var(--space-5);
            display: flex;
            flex-direction: column;
            gap: var(--space-1);
            transition: transform 200ms var(--ease-glide), border-color 200ms var(--ease-glide), box-shadow 200ms var(--ease-glide), opacity 200ms var(--ease-glide);
        }

        .overview-tile:hover {
            transform: translateY(-2px);
            border-color: var(--border-accent);
            box-shadow: 0 4px 18px var(--accent-champagne-glow);
        }

        .overview-tile-row:has(.overview-tile:hover) .overview-tile:not(:hover) {
            opacity: 0.88;
        }

        .overview-tile-label {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-dim);
            text-transform: uppercase;
        }

        .overview-tile-val {
            font-family: var(--font-mono);
            font-size: 20px;
            font-weight: 700;
            color: var(--text-primary);
        }

        /* CANONICAL 2X2 CONFUSION MATRIX */
        .cm-table {
            width: 100%;
            border-collapse: collapse;
            font-family: var(--font-mono);
            font-size: 11px;
            margin-top: var(--space-3);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-technical);
            overflow: hidden;
        }

        .cm-table th, .cm-table td {
            padding: 8px 10px;
            text-align: center;
            border: 1px solid var(--border-hairline);
            vertical-align: middle;
        }

        .cm-table th {
            background: var(--bg-inset);
            color: var(--text-dim);
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .cm-corner {
            font-size: 9.5px;
            color: var(--text-dim);
            background: var(--bg-surface);
            letter-spacing: 0.05em;
        }

        .cm-super-header {
            background: var(--bg-surface);
            color: var(--accent-champagne);
            font-size: 10px;
            letter-spacing: 0.08em;
        }

        .cm-col-header {
            font-size: 10px;
            color: var(--text-dim);
        }

        .cm-row-header {
            background: var(--bg-inset);
            text-align: left;
            padding-left: 10px;
            font-weight: 600;
            font-size: 10px;
            color: var(--text-dim);
        }

        .cm-cell-tn, .cm-cell-tp {
            background: var(--signal-positive-bg);
            color: var(--signal-positive);
            font-weight: 700;
        }

        .cm-cell-fp, .cm-cell-fn {
            background: var(--signal-negative-bg);
            color: var(--signal-negative);
            font-weight: 700;
        }

        .cm-cell-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
        }

        .cm-num {
            font-family: var(--font-mono);
            font-size: 13px;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
        }

        .cm-lbl {
            font-family: var(--font-mono);
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            opacity: 0.8;
        }

        /* ==========================================================================
           TAB 4: FORENSIC TELEMETRY JOURNAL (#telemetry)
           ========================================================================== */
        .telemetry-filter-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: var(--space-4);
            flex-wrap: wrap;
            margin-bottom: var(--space-8);
        }

        /* EXPANDING SEARCH FIELD */
        .search-container {
            position: relative;
            display: inline-flex;
            align-items: center;
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-secondary);
            transition: width 400ms var(--ease-glide), border-color 240ms, box-shadow 240ms;
            width: 200px;
            height: 40px;
            padding: 0 12px;
        }

        .search-container:focus-within {
            width: 320px;
            border-color: var(--accent-champagne);
            box-shadow: 0 0 14px var(--accent-champagne-glow);
        }

        .search-input {
            width: 100%;
            background: transparent;
            border: none;
            color: var(--text-primary);
            font-family: var(--font-mono);
            font-size: 12px;
            outline: none;
            padding-left: 8px;
        }

        .search-input::placeholder {
            color: var(--text-dim);
        }

        /* SPRING FILTER CHIPS */
        .chip-group {
            display: flex;
            align-items: center;
            gap: var(--space-2);
            flex-wrap: wrap;
        }

        .filter-chip {
            background: var(--bg-surface);
            border: 1px solid var(--border-hairline);
            color: var(--text-muted);
            font-family: var(--font-mono);
            font-size: 11px;
            padding: 6px 14px;
            border-radius: var(--radius-pill);
            cursor: pointer;
            transition: transform 260ms var(--ease-spring), background 200ms, border-color 200ms, color 200ms;
        }

        .filter-chip:hover {
            color: var(--text-primary);
            border-color: var(--border-accent);
        }

        .filter-chip:active {
            transform: scale(0.985);
        }

        .filter-chip:focus-visible {
            outline: 2px solid var(--accent-champagne);
            outline-offset: 2px;
        }

        .filter-chip.on {
            background: var(--accent-burgundy);
            color: var(--text-primary);
            font-weight: 600;
            border-color: var(--accent-champagne);
            transform: scale(1.08);
        }

        .table-responsive-wrapper {
            overflow-x: auto;
            border-radius: var(--radius-secondary);
            border: 1px solid var(--border-hairline);
            background: var(--bg-surface);
        }

        .audit-table {
            width: 100%;
            border-collapse: collapse;
            font-family: var(--font-mono);
            font-size: 12px;
            text-align: left;
        }

        .audit-table th {
            padding: 14px 16px;
            background: var(--bg-deep-burgundy);
            color: var(--text-dim);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 10.5px;
            letter-spacing: 0.05em;
            border-bottom: 1px solid var(--border-hairline);
        }

        .audit-table td {
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-hairline);
            color: var(--text-secondary);
            vertical-align: middle;
        }

        .audit-table tr:hover td {
            background: var(--accent-champagne-bg);
        }

        /* ==========================================================================
           TAB 5: DEVELOPER API SANDBOX (#developer)
           ========================================================================== */
        .developer-grid {
            display: grid;
            grid-template-columns: minmax(0, 50%) minmax(0, 50%);
            gap: var(--space-8);
        }

        .code-viewer {
            background: var(--code-bg);
            border: 1px solid var(--code-border);
            border-radius: var(--radius-secondary);
            padding: var(--space-5);
            font-family: var(--font-mono);
            font-size: 12px;
            line-height: 1.65;
            color: var(--text-secondary);
            overflow-x: auto;
            min-height: 220px;
            max-height: 380px;
        }

        /* DEVELOPER DIRECTORY & SPEC CARDS */
        .api-directory-list {
            display: flex;
            flex-direction: column;
            gap: var(--space-3);
        }

        .api-endpoint-item {
            background: var(--bg-inset);
            border: 1px solid var(--border-hairline);
            border-radius: var(--radius-technical);
            padding: var(--space-3) var(--space-4);
            display: flex;
            flex-direction: column;
            gap: 4px;
            transition: border-color 200ms;
        }

        .api-endpoint-item:hover {
            border-color: var(--border-accent);
        }

        .api-endpoint-head {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .badge-method {
            font-family: var(--font-mono);
            font-size: 9.5px;
            font-weight: 700;
            padding: 2px 7px;
            border-radius: 4px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .badge-method.post {
            background: rgba(214, 181, 106, 0.16);
            color: var(--accent-champagne);
            border: 1px solid rgba(214, 181, 106, 0.35);
        }

        .badge-method.get {
            background: rgba(143, 227, 192, 0.14);
            color: var(--signal-positive);
            border: 1px solid rgba(143, 227, 192, 0.3);
        }

        .api-path {
            font-family: var(--font-mono);
            font-size: 11.5px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .api-desc {
            font-size: 11.5px;
            color: var(--text-dim);
            line-height: 1.4;
        }

        .export-card-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: var(--space-3);
            margin-bottom: var(--space-4);
        }

        .export-action-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-technical);
            padding: 10px 14px;
            font-family: var(--font-mono);
            font-size: 11.5px;
            font-weight: 600;
            color: var(--text-primary);
            text-decoration: none;
            cursor: pointer;
            transition: all 200ms var(--ease-glide);
        }

        .export-action-btn:hover {
            border-color: var(--accent-champagne);
            color: var(--accent-champagne);
            background: var(--accent-champagne-bg);
            transform: translateY(-1px);
        }

        .export-action-btn:active {
            transform: scale(0.985);
        }

        .export-action-btn:focus-visible {
            outline: 2px solid var(--accent-champagne);
            outline-offset: 2px;
        }

        /* SUBTLE BUILD METADATA FOOTER */
        .app-footer {
            border-top: 1px solid var(--border-hairline);
            padding-block: var(--space-6);
            margin-top: auto;
        }

        .app-footer-inner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-dim);
            flex-wrap: wrap;
            gap: var(--space-3);
        }

        /* TOAST NOTIFICATION BANNER */
        .toast-banner {
            position: fixed;
            top: 24px;
            left: 50%;
            transform: translateX(-50%) translateY(-160%);
            background: var(--bg-surface);
            border: 1px solid var(--border-accent);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), 0 0 18px var(--accent-champagne-glow);
            color: var(--text-primary);
            padding: 11px 22px;
            border-radius: var(--radius-secondary);
            font-family: var(--font-sans);
            font-size: 13.5px;
            display: flex;
            align-items: center;
            gap: 12px;
            z-index: 1000;
            transition: transform 550ms var(--ease-pop);
        }

        .toast-banner.show {
            transform: translateX(-50%) translateY(0);
        }

        .toast-close-btn {
            background: transparent;
            border: none;
            color: var(--text-dim);
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 2px;
            border-radius: 4px;
            transition: color 180ms;
        }

        .toast-close-btn:hover {
            color: var(--text-primary);
        }

        /* CIRCULAR CSS SPINNER & DELTA-V 5-BAR EQUALIZER LOADER */
        .spinner-ring {
            width: 14px;
            height: 14px;
            border: 2px solid rgba(244, 232, 208, 0.25);
            border-top-color: var(--btn-cta-text);
            border-right-color: var(--btn-cta-text);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            display: none;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .ld-bars {
            display: none;
            align-items: center;
            justify-content: center;
            gap: 2.5px;
            height: 14px;
            vertical-align: middle;
        }

        .ld-bars[style*="inline-block"],
        .ld-bars[style*="inline-flex"],
        .ld-bars[style*="block"],
        .ld-bars[style*="flex"] {
            display: inline-flex !important;
        }

        .ld-bars i {
            display: inline-block;
            width: 2.5px;
            height: 100%;
            background-color: var(--btn-cta-text, var(--accent-champagne));
            border-radius: 1px;
            animation: deltaVBounce 1s ease-in-out infinite;
            transform-origin: bottom center;
        }

        .ld-bars i:nth-child(1) { animation-delay: 0.0s; }
        .ld-bars i:nth-child(2) { animation-delay: 0.15s; }
        .ld-bars i:nth-child(3) { animation-delay: 0.3s; }
        .ld-bars i:nth-child(4) { animation-delay: 0.45s; }
        .ld-bars i:nth-child(5) { animation-delay: 0.6s; }

        @keyframes deltaVBounce {
            0%, 100% {
                transform: scaleY(0.25);
                opacity: 0.45;
            }
            50% {
                transform: scaleY(1);
                opacity: 1;
            }
        }

        /* CONTEXTUAL FLOATING ACTION MENU / QUICK ACTION HUB */
        .quick-action-hub {
            position: fixed;
            bottom: 24px;
            right: 24px;
            z-index: 900;
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 10px;
        }

        .quick-action-fab {
            position: relative;
            width: 48px;
            height: 48px;
            border-radius: var(--radius-pill);
            background: var(--btn-cta-bg);
            border: 1px solid var(--btn-cta-border);
            color: var(--btn-cta-text);
            box-shadow: 0 4px 18px var(--btn-cta-glow);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 220ms var(--ease-spring), box-shadow 220ms var(--ease-glide);
        }

        .quick-action-fab:hover {
            transform: translateY(-2px) scale(1.04);
            box-shadow: 0 6px 24px var(--btn-cta-glow);
        }

        .quick-action-fab:active {
            transform: scale(0.96);
        }

        .quick-action-fab:focus-visible {
            outline: 2px solid var(--accent-champagne);
            outline-offset: 3px;
        }

        .quick-action-fab .fab-icon-idle,
        .quick-action-fab .fab-icon-active {
            position: absolute;
            width: 18px;
            height: 18px;
            transition: transform 260ms var(--ease-spring), opacity 200ms var(--ease-glide);
        }

        .quick-action-fab .fab-icon-active {
            opacity: 0;
            transform: rotate(-90deg) scale(0.6);
        }

        .quick-action-hub.open .quick-action-fab .fab-icon-idle {
            opacity: 0;
            transform: rotate(90deg) scale(0.6);
        }

        .quick-action-hub.open .quick-action-fab .fab-icon-active {
            opacity: 1;
            transform: rotate(0deg) scale(1);
        }

        .quick-action-menu {
            display: flex;
            flex-direction: column;
            gap: 6px;
            padding: 8px;
            background: var(--liquid-glass-bg);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid var(--border-subtle);
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(244, 232, 208, 0.12);
            border-radius: var(--radius-secondary);
            opacity: 0;
            visibility: hidden;
            transform: translateY(12px) scale(0.94);
            transform-origin: bottom right;
            transition: opacity 220ms var(--ease-glide), transform 260ms cubic-bezier(0.34, 1.56, 0.64, 1), visibility 220ms;
            pointer-events: none;
            min-width: 215px;
        }

        .quick-action-hub.open .quick-action-menu {
            opacity: 1;
            visibility: visible;
            transform: translateY(0) scale(1);
            pointer-events: auto;
        }

        .quick-action-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 9px 14px;
            background: transparent;
            border: 1px solid transparent;
            border-radius: var(--radius-technical);
            color: var(--text-primary);
            font-family: var(--font-sans);
            font-size: 12.5px;
            font-weight: 500;
            text-align: left;
            cursor: pointer;
            transition: background 180ms, border-color 180ms, color 180ms, transform 180ms;
        }

        .quick-action-item svg {
            width: 15px;
            height: 15px;
            color: var(--accent-champagne);
            flex-shrink: 0;
        }

        .quick-action-item:hover {
            background: var(--accent-champagne-bg);
            border-color: var(--border-hairline);
            color: var(--text-primary);
            transform: translateX(-2px);
        }

        .quick-action-item:active {
            transform: scale(0.98);
        }

        .quick-action-item:focus-visible {
            outline: 2px solid var(--accent-champagne);
            outline-offset: 1px;
        }

        [data-theme="light"] .quick-action-menu {
            box-shadow: 0 12px 32px rgba(61, 16, 25, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
            border-color: rgba(61, 16, 25, 0.16) !important;
        }

        [data-theme="light"] .quick-action-item {
            color: #120D10 !important;
        }

        [data-theme="light"] .quick-action-item:hover {
            background: rgba(168, 137, 77, 0.12) !important;
            border-color: rgba(61, 16, 25, 0.14) !important;
            color: #120D10 !important;
        }

        @media (max-width: 640px) {
            .quick-action-hub {
                bottom: 18px;
                right: 18px;
            }
        }

        /* ==========================================================================
           RESPONSIVE ENGINE & ZERO OVERFLOW VERIFICATION
           ========================================================================== */
        @media (max-width: 1024px) {
            .masthead-inner {
                display: grid;
                grid-template-columns: 1fr auto;
                align-items: center;
                gap: var(--space-3) var(--space-4);
            }
            .brand-cluster {
                grid-column: 1 / 2;
                grid-row: 1 / 2;
            }
            .masthead-right-cluster {
                grid-column: 2 / 3;
                grid-row: 1 / 2;
                justify-self: end;
            }
            .status-chip {
                display: none;
            }
            .nav-dock {
                width: 100%;
                max-width: 100%;
                overflow-x: auto;
                contain: paint;
                -webkit-overflow-scrolling: touch;
                scrollbar-width: none;
                padding-inline: 6px;
            }
            .nav-dock::-webkit-scrollbar {
                display: none;
            }
            .workspace-grid {
                grid-template-columns: minmax(0, 1fr) !important;
                width: 100%;
                max-width: 100%;
            }
            .consensus-cockpit-grid, .benchmark-metrics-grid, .developer-grid {
                grid-template-columns: minmax(0, 1fr) !important;
                width: 100%;
                max-width: 100%;
            }
            .overview-tile-row {
                grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            }
        }

        @media (max-width: 768px) {
            .core-plate {
                padding: var(--space-5);
            }
            .search-container {
                width: 100% !important;
            }
            .controls-row {
                flex-direction: column;
                align-items: stretch;
            }
            .engine-picker {
                width: 100%;
                justify-content: space-between;
            }
            .btn-execute {
                width: 100%;
                justify-content: center;
            }
            .xai-drivers-grid {
                grid-template-columns: 1fr;
            }
            .overview-tile-row {
                grid-template-columns: 1fr !important;
            }
            .sentiment-hero-card {
                flex-direction: column;
                align-items: flex-start;
            }
            .audit-feedback-strip {
                flex-direction: column;
                align-items: stretch;
            }
            .table-responsive-wrapper {
                border: none;
                background: transparent;
            }
            .audit-table thead {
                display: none;
            }
            .audit-table, .audit-table tbody, .audit-table tr, .audit-table td {
                display: block;
                width: 100%;
                box-sizing: border-box;
            }
            .audit-table tr {
                background: var(--bg-surface);
                border: 1px solid var(--border-hairline);
                border-radius: var(--radius-secondary);
                margin-bottom: var(--space-3);
                padding: var(--space-4);
            }
            .audit-table td {
                display: flex;
                justify-content: space-between;
                padding: var(--space-2) 0;
                border-bottom: 1px solid var(--border-hairline);
                white-space: normal;
                word-break: break-all;
            }
            .audit-table td:last-child {
                border-bottom: none;
            }
            .audit-table td::before {
                content: attr(data-label);
                font-weight: 600;
                color: var(--text-dim);
                text-transform: uppercase;
                font-size: 10.5px;
                margin-right: 8px;
                flex-shrink: 0;
            }
        }

        /* MANDATORY ACCESSIBILITY: REDUCED MOTION OVERRIDE */
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }
            .result-revealing .sentiment-hero-card,
            .result-revealing .polarity-hud-box,
            .result-revealing .confidence-container,
            .result-revealing .xai-attribution-panel,
            .beam::before,
            .sentinel-edge.edge-processing::before,
            .btn-execute::before,
            .ld-bars i,
            .quick-action-fab,
            .quick-action-menu {
                animation: none !important;
                transition: none !important;
            }
        }
    </style>
</head>
<body>
    <!-- TOAST NOTIFICATION BANNER -->
    <div id="toastBanner" class="toast-banner" role="status" aria-live="polite">
        <svg class="svg-icon" viewBox="0 0 24 24" style="color:var(--accent-champagne);"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
        <span id="toastMessage">Operational Ready</span>
        <button type="button" class="toast-close-btn" onclick="dismissToast()" aria-label="Dismiss alert">
            <svg class="svg-icon" viewBox="0 0 24 24" style="width:14px; height:14px;"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
    </div>

    <!-- MASTHEAD WRAPPER -->
    <header class="masthead-wrapper">
        <div class="masthead-inner">
            <div class="brand-cluster">
                <!-- BESPOKE GEOMETRIC S-APERTURE PRECISION EMBLEM -->
                <div class="brand-icon-pod" title="SENTINEL AI Precision S-Aperture">
                    <svg class="brand-logo-svg" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="SENTINEL AI Geometric S-Aperture">
                        <circle cx="16" cy="16" r="14" stroke="currentColor" stroke-width="1" stroke-opacity="0.25"/>
                        <circle cx="16" cy="16" r="11" stroke="var(--accent-champagne)" stroke-width="0.75" stroke-opacity="0.45"/>
                        <line x1="16" y1="1" x2="16" y2="4.5" stroke="var(--accent-champagne)" stroke-width="1.5" stroke-linecap="round"/>
                        <line x1="16" y1="27.5" x2="16" y2="31" stroke="var(--accent-champagne)" stroke-width="1.5" stroke-linecap="round"/>
                        <line x1="1" y1="16" x2="4.5" y2="16" stroke="var(--accent-champagne)" stroke-width="1.5" stroke-linecap="round"/>
                        <line x1="27.5" y1="16" x2="31" y2="16" stroke="var(--accent-champagne)" stroke-width="1.5" stroke-linecap="round"/>
                        <path d="M 22 8 C 17 6, 10.5 7.5, 10.5 12.5 C 10.5 16.5, 15 16, 16 16" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
                        <path d="M 10 24 C 15 26, 21.5 24.5, 21.5 19.5 C 21.5 15.5, 17 16, 16 16" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
                        <path d="M 16 12 C 13.5 12, 12.5 13.5, 12.5 15" stroke="var(--accent-burgundy)" stroke-width="1.5" stroke-linecap="round"/>
                        <path d="M 16 20 C 18.5 20, 19.5 18.5, 19.5 17" stroke="var(--accent-burgundy)" stroke-width="1.5" stroke-linecap="round"/>
                        <circle cx="16" cy="16" r="2.2" fill="var(--accent-champagne)"/>
                    </svg>
                </div>
                <div class="brand-title-wrap">
                    <span class="brand-title">SENTINEL AI</span>
                    <span class="brand-subtitle">NLP SENTIMENT INTELLIGENCE</span>
                </div>
            </div>

            <!-- SIGNATURE NAVIGATION DOCK WITH TAB PILL GLIDE -->
            <nav class="nav-dock" id="navDock" aria-label="Primary Instrument Navigation">
                <div class="nav-pill-glider" id="navGlider"></div>
                <button type="button" class="tab-btn active" id="tab-analyze" onclick="switchTab('analyze')">
                    <svg class="svg-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="2" x2="12" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/></svg>
                    <span>Analyze</span>
                </button>
                <button type="button" class="tab-btn" id="tab-consensus" onclick="switchTab('consensus')">
                    <svg class="svg-icon" viewBox="0 0 24 24"><circle cx="12" cy="5" r="3"/><circle cx="5" cy="18" r="3"/><circle cx="19" cy="18" r="3"/><line x1="12" y1="8" x2="6.5" y2="15.5"/><line x1="12" y1="8" x2="17.5" y2="15.5"/><line x1="8" y1="18" x2="16" y2="18"/></svg>
                    <span>Consensus</span>
                </button>
                <button type="button" class="tab-btn" id="tab-benchmarks" onclick="switchTab('benchmarks')">
                    <svg class="svg-icon" viewBox="0 0 24 24"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
                    <span>Benchmarks</span>
                </button>
                <button type="button" class="tab-btn" id="tab-telemetry" onclick="switchTab('telemetry')">
                    <svg class="svg-icon" viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                    <span>Telemetry</span>
                </button>
                <button type="button" class="tab-btn" id="tab-developer" onclick="switchTab('developer')">
                    <svg class="svg-icon" viewBox="0 0 24 24"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
                    <span>Developer</span>
                </button>
            </nav>

            <div class="masthead-right-cluster">
                <div class="status-chip">
                    <span class="status-dot"></span>
                    <span>ALL SYSTEMS ONLINE</span>
                </div>
                <!-- SUN/MOON LUXURY TOGGLE -->
                <label class="switch sw-7" title="Toggle Light / Dark Luxury Profile">
                    <input type="checkbox" id="themeModeToggle" onchange="toggleAmbientTheme(!this.checked)">
                    <span class="track">
                        <span class="thumb"></span>
                    </span>
                </label>
            </div>
        </div>
    </header>

    <!-- MAIN INTERACTIVE CONTAINER -->
    <main class="app-shell">

        <!-- ==========================================================================
             PANEL 1: NATURAL LANGUAGE INGESTION & INFERENCE ENGINE (#analyze)
             ========================================================================== -->
        <section id="view-analyze" class="view-panel active" aria-labelledby="heading-analyze">
            <div class="section-header">
                <span class="section-eyebrow">Natural Language Ingestion</span>
                <h1 id="heading-analyze" class="section-title">Decode the signal.</h1>
                <p class="section-desc">AI sentiment analysis with transparent model attribution.</p>
            </div>

            <div class="workspace-grid">
                <!-- LEFT COLUMN: INGESTION CONSOLE -->
                <div class="chassis-card sentinel-edge" id="inputCard">
                    <div class="core-plate">
                        <div style="display:flex; justify-content:space-between; align-items:center; gap:var(--space-4); flex-wrap:wrap;">
                            <span class="card-subhead">INPUT SIGNAL</span>
                            <!-- PRESETS -->
                            <div class="preset-strip">
                                <span class="preset-label">Presets:</span>
                                <button type="button" id="presetPos" class="btn-preset" onclick="applyPreset('pos')">POSITIVE</button>
                                <button type="button" id="presetNeg" class="btn-preset" onclick="applyPreset('neg')">CRITICAL</button>
                                <button type="button" id="presetAmb" class="btn-preset" onclick="applyPreset('amb')">AMBIGUOUS</button>
                            </div>
                        </div>

                        <!-- TEXT ENCLOSURE -->
                        <div class="textarea-enclosure">
                            <textarea id="nlpInput" class="nlp-textarea" placeholder="Paste text to analyze sentiment and model attribution…" oninput="updateCounters()" onkeydown="handleTextKeydown(event)"></textarea>
                            <div class="textarea-meta-bar">
                                <div class="textarea-counters">
                                    <span id="charCounter">0 / 5,000 characters</span>
                                    <span>&bull;</span>
                                    <span id="wordCounter">0 words</span>
                                </div>
                                <span>Press Ctrl + Enter to Execute</span>
                            </div>
                        </div>

                        <!-- CONTROLS ROW -->
                        <div class="controls-row">
                            <div class="engine-picker">
                                <span class="engine-label">Arbitration Engine:</span>
                                <select id="modelSelect" class="engine-select" aria-label="Select Model Engine">
                                    <option value="Logistic Regression">Logistic Regression (Primary)</option>
                                    <option value="Linear SVM">Platt-Calibrated Linear SVM</option>
                                    <option value="Multinomial Naive Bayes">Multinomial Naive Bayes</option>
                                </select>
                            </div>

                            <!-- PRIMARY EXECUTE BUTTON -->
                            <button type="button" id="btnAnalyze" class="btn-execute" onclick="executeAnalysis(event)">
                                <span class="ld-bars" id="analyzeSpinner" role="status" aria-label="Processing" style="display:none;"><i></i><i></i><i></i><i></i><i></i></span>
                                <span id="analyzeBtnText">EXECUTE ANALYSIS</span>
                                <svg class="svg-icon" viewBox="0 0 24 24"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
                            </button>
                        </div>

                        <!-- METADATA ADVISORY STRIP -->
                        <div class="advisory-strip">
                            <div class="advisory-meta-group">
                                <div class="advisory-meta-item"><span class="meta-label">Vocabulary:</span><strong>5,000 TF-IDF</strong></div>
                                <div class="advisory-meta-item"><span class="meta-label">N-Grams:</span><strong>(1, 1) Unigram</strong></div>
                                <div class="advisory-meta-item"><span class="meta-label">Calibrated:</span><strong>Platt Scaling</strong></div>
                            </div>
                            <span id="statusLiveChip" style="color:var(--signal-positive); font-weight:600;">ACTIVE READY</span>
                        </div>
                    </div>
                </div>

                <!-- RIGHT COLUMN: RESEARCH-GRADE CLASSIFICATION RESULT -->
                <div class="chassis-card sentinel-edge" id="resultCard">
                    <div class="core-plate telemetry-stack">
                        <span class="card-subhead">CLASSIFICATION RESULT</span>

                        <!-- SENTIMENT HERO DISPLAY -->
                        <div class="sentiment-hero-card">
                            <div>
                                <span style="font-family:var(--font-mono); font-size:10px; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.06em;">Sentiment Inferred</span>
                                <div id="resSentiment" class="sentiment-large-display" style="color:var(--text-dim); margin-top:4px;">
                                    <span>STANDBY</span>
                                    <span class="sentiment-engine-tag" style="background:var(--bg-track); color:var(--text-dim);">NO INPUT</span>
                                </div>
                            </div>
                            <div style="text-align:right;">
                                <span style="font-family:var(--font-mono); font-size:10px; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.06em;">Dominant Emotion</span>
                                <div id="resEmotion" style="font-family:var(--font-mono); font-size:15px; font-weight:700; color:var(--text-secondary); margin-top:4px;">&mdash;</div>
                            </div>
                        </div>

                        <!-- BIPOLAR POLARITY HUD (-1.0000 to +1.0000) -->
                        <div class="polarity-hud-box">
                            <div class="polarity-hud-top">
                                <span style="color:var(--text-dim); text-transform:uppercase; font-size:10.5px; letter-spacing:0.05em;">Polarity Index</span>
                                <span id="resPolarityScore" class="polarity-val-big">&plusmn;0.0000</span>
                            </div>
                            <div class="bipolar-track">
                                <div class="bipolar-center-axis"></div>
                                <div id="polarityBipolarBar" class="bipolar-fill" style="left:50%; width:0%; background:var(--accent-champagne);"></div>
                            </div>
                            <div class="bipolar-legend">
                                <span>-1.0000 (Negative)</span>
                                <span>0.0000 (Neutral)</span>
                                <span>+1.0000 (Positive)</span>
                            </div>
                        </div>

                        <!-- CONFIDENCE METER -->
                        <div class="confidence-container">
                            <div>
                                <span style="font-family:var(--font-mono); font-size:10px; color:var(--text-dim); text-transform:uppercase; letter-spacing:0.05em;">Confidence</span>
                                <div id="resConfidence" class="confidence-val">&mdash;%</div>
                            </div>
                            <div style="width:140px; background:var(--bg-track); height:8px; border-radius:var(--radius-pill); overflow:hidden;">
                                <div id="confBar" style="height:100%; width:0%; background:var(--accent-champagne); border-radius:var(--radius-pill); transition:width 360ms var(--ease-glide);"></div>
                            </div>
                        </div>

                        <!-- SCIENTIFIC ATTRIBUTION (XAI) -->
                        <div class="xai-attribution-panel">
                            <span class="card-subhead">SCIENTIFIC ATTRIBUTION</span>
                            <div class="xai-drivers-grid">
                                <div class="driver-column">
                                    <span style="font-family:var(--font-mono); font-size:10px; color:var(--signal-positive); font-weight:600; letter-spacing:0.06em;">+ POSITIVE DRIVERS</span>
                                    <div id="posDriversList">
                                        <div class="driver-empty-hint">None detected within vocabulary</div>
                                    </div>
                                </div>
                                <div class="driver-column">
                                    <span style="font-family:var(--font-mono); font-size:10px; color:var(--signal-negative); font-weight:600; letter-spacing:0.06em;">&minus; NEGATIVE DRIVERS</span>
                                    <div id="negDriversList">
                                        <div class="driver-empty-hint">None detected within vocabulary</div>
                                    </div>
                                </div>
                            </div>
                            <p class="methodology-caption">TF-IDF &times; Model Coefficient. Statistical decision-boundary contribution; not causal attribution.</p>
                        </div>

                        <!-- AUDIT FEEDBACK STRIP WITH SVG ICONS & ANIMATION -->
                        <div class="audit-feedback-strip">
                            <span style="font-family:var(--font-mono); font-size:11px; color:var(--text-dim);">Verify Ground Truth:</span>
                            <div style="display:flex; gap:8px; align-items:center;">
                                <button type="button" id="btnConfirmFeedback" class="btn-feedback" onclick="sendAuditFeedback(true)" disabled aria-label="Confirm Correct Classification">
                                    <svg class="svg-icon" viewBox="0 0 24 24" style="width:13px; height:13px;"><polyline points="20 6 9 17 4 12"/></svg>
                                    Confirm Correct
                                </button>
                                <button type="button" id="btnFlagFeedback" class="btn-feedback" onclick="sendAuditFeedback(false)" disabled aria-label="Flag Incorrect Classification">
                                    <svg class="svg-icon" viewBox="0 0 24 24" style="width:13px; height:13px;"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                                    Flag Incorrect
                                </button>
                            </div>
                            <div id="feedbackSuccessAnim" class="feedback-success-svg" style="display:none;">
                                <svg viewBox="0 0 52 52" class="checkmark-svg">
                                    <circle class="checkmark-circle" cx="26" cy="26" r="23" fill="none"/>
                                    <path class="checkmark-check" fill="none" d="M14.1 27.2l7.1 7.2 16.7-16.8"/>
                                </svg>
                                <span>Recorded</span>
                            </div>
                        </div>

                        <!-- DIAGNOSTIC HUD METADATA -->
                        <div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid var(--border-hairline); padding-top:var(--space-3); font-family:var(--font-mono); font-size:10.5px; color:var(--text-dim);">
                            <span>LATENCY: <strong id="resLatency" style="color:var(--text-secondary);">&mdash;</strong></span>
                            <span>TOKENS: <strong id="resTokens" style="color:var(--text-secondary);">&mdash;</strong></span>
                            <span>REQ: <strong id="resReqId" style="color:var(--accent-champagne);">STANDBY</strong></span>
                            <span>AT: <strong id="resTimestamp" style="color:var(--text-secondary);">&mdash;</strong></span>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- ==========================================================================
             PANEL 2: THREE-INSTRUMENT ARRAY (#consensus)
             ========================================================================== -->
        <section id="view-consensus" class="view-panel" aria-labelledby="heading-consensus">
            <div class="section-header">
                <span class="section-eyebrow">Multi-Engine Arbitration</span>
                <h2 id="heading-consensus" class="section-title">Three-Instrument Array</h2>
                <p class="section-desc">Model agreement across three production engines.</p>
            </div>

            <!-- AGREEMENT BANNER -->
            <div class="consensus-banner">
                <div class="consensus-ratio-text">
                    <span id="consensusRatio" style="color:var(--text-primary);">&mdash; / 3 MODELS AGREE</span>
                    <span id="consensusBadge" class="brand-badge" style="display:none; font-family:var(--font-mono); font-size:10px; background:var(--signal-positive-bg); color:var(--signal-positive); border:1px solid var(--signal-positive-border); padding:3px 10px; border-radius:var(--radius-pill); font-style:normal;">CONSENSUS REACHED</span>
                </div>
                <button type="button" id="btnEvaluateConsensus" class="btn-execute" onclick="executeConsensus(event)" style="padding:9px 22px; font-size:13px;">
                    <span class="ld-bars" id="consensusSpinner" role="status" aria-label="Processing" style="display:none;"><i></i><i></i><i></i><i></i><i></i></span>
                    <span id="consensusBtnText">Evaluate Consensus</span>
                </button>
            </div>

            <div class="consensus-cockpit-grid">
                <!-- MODEL 1: LOGISTIC REGRESSION -->
                <div class="chassis-card sentinel-edge consensus-engine-card">
                    <div class="core-plate">
                        <span class="card-subhead">Engine 01</span>
                        <h3 style="font-family:var(--font-display); font-size:22px; font-style:italic; font-weight:600;">Logistic Regression</h3>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Sentiment:</span><strong id="cM1Sentiment" style="font-family:var(--font-mono);">&mdash;</strong></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Polarity Score:</span><span id="cM1Score" style="font-family:var(--font-mono); font-weight:600;">&mdash;</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Confidence:</span><span id="cM1Conf" style="font-family:var(--font-mono); font-weight:600;">&mdash;</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Emotion:</span><span id="cM1Emotion" style="font-family:var(--font-mono);">&mdash;</span></div>
                    </div>
                </div>

                <!-- MODEL 2: LINEAR SVM -->
                <div class="chassis-card sentinel-edge consensus-engine-card">
                    <div class="core-plate">
                        <span class="card-subhead">Engine 02</span>
                        <h3 style="font-family:var(--font-display); font-size:22px; font-style:italic; font-weight:600;">Linear SVM</h3>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Sentiment:</span><strong id="cM2Sentiment" style="font-family:var(--font-mono);">&mdash;</strong></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Polarity Score:</span><span id="cM2Score" style="font-family:var(--font-mono); font-weight:600;">&mdash;</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Confidence:</span><span id="cM2Conf" style="font-family:var(--font-mono); font-weight:600;">&mdash;</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Emotion:</span><span id="cM2Emotion" style="font-family:var(--font-mono);">&mdash;</span></div>
                    </div>
                </div>

                <!-- MODEL 3: NAIVE BAYES -->
                <div class="chassis-card sentinel-edge consensus-engine-card">
                    <div class="core-plate">
                        <span class="card-subhead">Engine 03</span>
                        <h3 style="font-family:var(--font-display); font-size:22px; font-style:italic; font-weight:600;">Multinomial Naive Bayes</h3>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Sentiment:</span><strong id="cM3Sentiment" style="font-family:var(--font-mono);">&mdash;</strong></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Polarity Score:</span><span id="cM3Score" style="font-family:var(--font-mono); font-weight:600;">&mdash;</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Confidence:</span><span id="cM3Conf" style="font-family:var(--font-mono); font-weight:600;">&mdash;</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Emotion:</span><span id="cM3Emotion" style="font-family:var(--font-mono);">&mdash;</span></div>
                    </div>
                </div>
            </div>
        </section>

        <!-- ==========================================================================
             PANEL 3: RESEARCH BENCHMARKS & MODEL REGISTRY (#benchmarks)
             ========================================================================== -->
        <section id="view-benchmarks" class="view-panel" aria-labelledby="heading-benchmarks">
            <div class="section-header">
                <span class="section-eyebrow">Laboratory Calibration</span>
                <h2 id="heading-benchmarks" class="section-title">Research &amp; Calibration</h2>
                <p class="section-desc">Three production engines evaluated on IMDb 50K.</p>
            </div>

            <!-- CORPUS SUMMARY TILES -->
            <div class="overview-tile-row" style="margin-bottom:var(--space-8);">
                <div class="overview-tile"><span class="overview-tile-label">Corpus</span><span class="overview-tile-val">IMDb 50K</span></div>
                <div class="overview-tile"><span class="overview-tile-label">Test Split</span><span class="overview-tile-val">10,000 Records</span></div>
                <div class="overview-tile"><span class="overview-tile-label">Vocabulary</span><span class="overview-tile-val">5,000 TF-IDF</span></div>
                <div class="overview-tile"><span class="overview-tile-label">N-Gram Window</span><span class="overview-tile-val">(1, 1) Unigram</span></div>
            </div>

            <div class="benchmark-metrics-grid">
                <!-- BENCHMARK CARD 1 -->
                <div class="chassis-card">
                    <div class="core-plate">
                        <span class="card-subhead">Engine 01 &bull; Primary</span>
                        <h3 style="font-family:var(--font-display); font-size:22px; font-style:italic; font-weight:600;">Logistic Regression</h3>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Accuracy:</span><strong style="color:var(--signal-positive); font-family:var(--font-mono);">88.55%</strong></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Precision:</span><span style="font-family:var(--font-mono); font-weight:600;">88.20%</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Recall:</span><span style="font-family:var(--font-mono); font-weight:600;">89.01%</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">F1-Score:</span><span style="font-family:var(--font-mono); font-weight:600;">88.60%</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">ROC-AUC:</span><span style="font-family:var(--font-mono); font-weight:600;">0.9540</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Latency:</span><span style="font-family:var(--font-mono); font-weight:600;">1.8 ms</span></div>

                        <span class="card-subhead" style="margin-top:var(--space-2);">Holdout Confusion Matrix</span>
                        <table class="cm-table" aria-label="Logistic Regression Holdout Confusion Matrix">
                            <thead>
                                <tr>
                                    <th rowspan="2" class="cm-corner" scope="col">ACTUAL \ PREDICTED</th>
                                    <th colspan="2" class="cm-super-header" scope="colgroup">PREDICTED</th>
                                </tr>
                                <tr>
                                    <th class="cm-col-header" scope="col">NEGATIVE</th>
                                    <th class="cm-col-header" scope="col">POSITIVE</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <th scope="row" class="cm-row-header">NEGATIVE</th>
                                    <td class="cm-cell-tn">
                                        <div class="cm-cell-content"><span class="cm-num">4,410</span><span class="cm-lbl">TN &bull; Correct</span></div>
                                    </td>
                                    <td class="cm-cell-fp">
                                        <div class="cm-cell-content"><span class="cm-num">590</span><span class="cm-lbl">FP &bull; Error</span></div>
                                    </td>
                                </tr>
                                <tr>
                                    <th scope="row" class="cm-row-header">POSITIVE</th>
                                    <td class="cm-cell-fn">
                                        <div class="cm-cell-content"><span class="cm-num">555</span><span class="cm-lbl">FN &bull; Error</span></div>
                                    </td>
                                    <td class="cm-cell-tp">
                                        <div class="cm-cell-content"><span class="cm-num">4,445</span><span class="cm-lbl">TP &bull; Correct</span></div>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- BENCHMARK CARD 2 -->
                <div class="chassis-card">
                    <div class="core-plate">
                        <span class="card-subhead">Engine 02</span>
                        <h3 style="font-family:var(--font-display); font-size:22px; font-style:italic; font-weight:600;">Linear SVM</h3>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Accuracy:</span><strong style="color:var(--signal-positive); font-family:var(--font-mono);">88.02%</strong></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Precision:</span><span style="font-family:var(--font-mono); font-weight:600;">87.65%</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Recall:</span><span style="font-family:var(--font-mono); font-weight:600;">88.50%</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">F1-Score:</span><span style="font-family:var(--font-mono); font-weight:600;">88.07%</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">ROC-AUC:</span><span style="font-family:var(--font-mono); font-weight:600;">0.9480</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Latency:</span><span style="font-family:var(--font-mono); font-weight:600;">2.1 ms</span></div>

                        <span class="card-subhead" style="margin-top:var(--space-2);">Holdout Confusion Matrix</span>
                        <table class="cm-table" aria-label="Linear SVM Holdout Confusion Matrix">
                            <thead>
                                <tr>
                                    <th rowspan="2" class="cm-corner" scope="col">ACTUAL \ PREDICTED</th>
                                    <th colspan="2" class="cm-super-header" scope="colgroup">PREDICTED</th>
                                </tr>
                                <tr>
                                    <th class="cm-col-header" scope="col">NEGATIVE</th>
                                    <th class="cm-col-header" scope="col">POSITIVE</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <th scope="row" class="cm-row-header">NEGATIVE</th>
                                    <td class="cm-cell-tn">
                                        <div class="cm-cell-content"><span class="cm-num">4,380</span><span class="cm-lbl">TN &bull; Correct</span></div>
                                    </td>
                                    <td class="cm-cell-fp">
                                        <div class="cm-cell-content"><span class="cm-num">620</span><span class="cm-lbl">FP &bull; Error</span></div>
                                    </td>
                                </tr>
                                <tr>
                                    <th scope="row" class="cm-row-header">POSITIVE</th>
                                    <td class="cm-cell-fn">
                                        <div class="cm-cell-content"><span class="cm-num">578</span><span class="cm-lbl">FN &bull; Error</span></div>
                                    </td>
                                    <td class="cm-cell-tp">
                                        <div class="cm-cell-content"><span class="cm-num">4,422</span><span class="cm-lbl">TP &bull; Correct</span></div>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- BENCHMARK CARD 3 -->
                <div class="chassis-card">
                    <div class="core-plate">
                        <span class="card-subhead">Engine 03</span>
                        <h3 style="font-family:var(--font-display); font-size:22px; font-style:italic; font-weight:600;">Multinomial Naive Bayes</h3>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Accuracy:</span><strong style="color:var(--signal-positive); font-family:var(--font-mono);">85.10%</strong></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Precision:</span><span style="font-family:var(--font-mono); font-weight:600;">85.00%</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Recall:</span><span style="font-family:var(--font-mono); font-weight:600;">85.20%</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">F1-Score:</span><span style="font-family:var(--font-mono); font-weight:600;">85.10%</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">ROC-AUC:</span><span style="font-family:var(--font-mono); font-weight:600;">0.9290</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Latency:</span><span style="font-family:var(--font-mono); font-weight:600;">0.9 ms</span></div>

                        <span class="card-subhead" style="margin-top:var(--space-2);">Holdout Confusion Matrix</span>
                        <table class="cm-table" aria-label="Multinomial Naive Bayes Holdout Confusion Matrix">
                            <thead>
                                <tr>
                                    <th rowspan="2" class="cm-corner" scope="col">ACTUAL \ PREDICTED</th>
                                    <th colspan="2" class="cm-super-header" scope="colgroup">PREDICTED</th>
                                </tr>
                                <tr>
                                    <th class="cm-col-header" scope="col">NEGATIVE</th>
                                    <th class="cm-col-header" scope="col">POSITIVE</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <th scope="row" class="cm-row-header">NEGATIVE</th>
                                    <td class="cm-cell-tn">
                                        <div class="cm-cell-content"><span class="cm-num">4,310</span><span class="cm-lbl">TN &bull; Correct</span></div>
                                    </td>
                                    <td class="cm-cell-fp">
                                        <div class="cm-cell-content"><span class="cm-num">690</span><span class="cm-lbl">FP &bull; Error</span></div>
                                    </td>
                                </tr>
                                <tr>
                                    <th scope="row" class="cm-row-header">POSITIVE</th>
                                    <td class="cm-cell-fn">
                                        <div class="cm-cell-content"><span class="cm-num">800</span><span class="cm-lbl">FN &bull; Error</span></div>
                                    </td>
                                    <td class="cm-cell-tp">
                                        <div class="cm-cell-content"><span class="cm-num">4,200</span><span class="cm-lbl">TP &bull; Correct</span></div>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </section>

        <!-- ==========================================================================
             PANEL 4: FORENSIC TELEMETRY JOURNAL (#telemetry)
             ========================================================================== -->
        <section id="view-telemetry" class="view-panel" aria-labelledby="heading-telemetry">
            <div class="section-header">
                <span class="section-eyebrow">Forensic Audit Trail</span>
                <h2 id="heading-telemetry" class="section-title">Forensic Journal</h2>
                <p class="section-desc">Operational intelligence from live inference activity.</p>
            </div>

            <!-- ANALYTICS SUMMARY TILES -->
            <div class="overview-tile-row" style="margin-bottom:var(--space-4);">
                <div class="overview-tile"><span class="overview-tile-label">Total Analyzed</span><span id="statTotal" class="overview-tile-val">0</span></div>
                <div class="overview-tile"><span class="overview-tile-label">Positive Signals</span><span id="statPos" class="overview-tile-val" style="color:var(--signal-positive);">0</span></div>
                <div class="overview-tile"><span class="overview-tile-label">Negative Signals</span><span id="statNeg" class="overview-tile-val" style="color:var(--signal-negative);">0</span></div>
                <div class="overview-tile"><span class="overview-tile-label">Neutral Signals</span><span id="statNeu" class="overview-tile-val" style="color:var(--signal-neutral);">0</span></div>
            </div>

            <div class="overview-tile-row" style="margin-bottom:var(--space-8);">
                <div class="overview-tile"><span class="overview-tile-label">Mean Latency</span><span id="statAvgLat" class="overview-tile-val">0.00 ms</span></div>
                <div class="overview-tile"><span class="overview-tile-label">P95 Tail Latency</span><span id="statP95" class="overview-tile-val">&mdash;</span></div>
                <div class="overview-tile"><span class="overview-tile-label">P99 Tail Latency</span><span id="statP99" class="overview-tile-val">&mdash;</span></div>
                <div class="overview-tile">
                    <span class="overview-tile-label">Model Distribution</span>
                    <div id="statModelDist" style="font-family:var(--font-mono); font-size:11.5px; color:var(--text-secondary); margin-top:4px; line-height:1.4;">Awaiting inference traffic</div>
                </div>
            </div>

            <!-- FILTER & SEARCH BAR -->
            <div class="telemetry-filter-bar">
                <div style="display:flex; align-items:center; gap:var(--space-4); flex-wrap:wrap;">
                    <!-- EXPANDING SEARCH FIELD -->
                    <div class="search-container">
                        <svg class="svg-icon" viewBox="0 0 24 24" style="color:var(--text-dim);"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                        <input type="text" id="telemetrySearchInput" class="search-input" placeholder="Search Request ID, text..." oninput="filterTelemetryTable()">
                    </div>

                    <!-- SPRING FILTER CHIPS -->
                    <div class="chip-group">
                        <button type="button" class="filter-chip on" onclick="toggleFilterChip(this, 'ALL')">All</button>
                        <button type="button" class="filter-chip" onclick="toggleFilterChip(this, 'POSITIVE')">Positive</button>
                        <button type="button" class="filter-chip" onclick="toggleFilterChip(this, 'NEGATIVE')">Negative</button>
                    </div>
                </div>

                <div style="display:flex; gap:8px;">
                    <a href="/api/v1/export/csv" class="btn-preset" style="text-decoration:none; padding:6px 14px;">Export CSV</a>
                    <a href="/api/v1/export/json" class="btn-preset" style="text-decoration:none; padding:6px 14px;">Export JSON</a>
                    <button type="button" class="btn-preset" onclick="loadTelemetry()">Refresh Log</button>
                </div>
            </div>

            <!-- TABLE WRAPPER -->
            <div class="table-responsive-wrapper">
                <table class="audit-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Request ID</th>
                            <th>Sentiment</th>
                            <th>Confidence</th>
                            <th>AI Score</th>
                            <th>Engine</th>
                            <th>Emotion</th>
                            <th>Latency</th>
                            <th>Feedback</th>
                            <th>Timestamp</th>
                        </tr>
                    </thead>
                    <tbody id="telemetryTbody">
                        <tr>
                            <td colspan="10" style="text-align:center; padding:28px; color:var(--text-dim);">Awaiting telemetry synchronisation...</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </section>

        <!-- ==========================================================================
             PANEL 5: DEVELOPER API SANDBOX & HEALTH DIAGNOSTICS (#developer)
             ========================================================================== -->
        <section id="view-developer" class="view-panel" aria-labelledby="heading-developer">
            <div class="section-header">
                <span class="section-eyebrow">Integration Gateway</span>
                <h2 id="heading-developer" class="section-title">Developer Gateway</h2>
                <p class="section-desc">Execute real RESTful requests against the live FastAPI endpoint, inspect raw JSON response payloads, and verify model health diagnostics.</p>
            </div>

            <div class="developer-grid">
                <!-- 1. REST EXECUTION SANDBOX -->
                <div class="chassis-card sentinel-edge" id="devSandboxCard">
                    <div class="core-plate">
                        <span class="card-subhead">REST Execution Sandbox</span>
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-family:var(--font-mono); font-size:12px; color:var(--accent-champagne); font-weight:600;">POST /api/v1/predict</span>
                            <button type="button" id="btnSendApi" class="btn-execute" style="padding:7px 18px; font-size:12.5px;" onclick="sendApiRequest(event)">
                                <span class="ld-bars" id="devSpinner" role="status" aria-label="Processing" style="display:none;"><i></i><i></i><i></i><i></i><i></i></span>
                                <span id="devBtnText">Send Test Request</span>
                            </button>
                        </div>
                        <div class="textarea-enclosure">
                            <textarea id="devPayload" class="nlp-textarea" style="height:120px;" spellcheck="false">{
  "text": "The cinematography and subtle direction created an extraordinary aesthetic experience.",
  "model": "Logistic Regression"
}</textarea>
                        </div>
                        <span class="card-subhead">JSON Response Payload</span>
                        <pre id="devResponse" class="code-viewer">// Response payload will appear here...</pre>
                    </div>
                </div>

                <!-- 2. SYSTEM DIAGNOSTICS & SPECS -->
                <div class="chassis-card">
                    <div class="core-plate">
                        <span class="card-subhead">System Diagnostics &amp; Specs</span>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">API Status:</span><span style="font-family:var(--font-mono); color:var(--signal-positive); font-weight:700;" id="diagStatus">ONLINE (200 OK)</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Backend Server:</span><span style="font-family:var(--font-mono); font-weight:700;">FastAPI + Uvicorn ASGI</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Primary Classifier:</span><span style="font-family:var(--font-mono); font-weight:700;" id="diagModel">Logistic Regression (C=1.0)</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">TF-IDF Vectorizer:</span><span style="font-family:var(--font-mono); font-weight:700;" id="diagVec">LOADED (5,000 Features)</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Storage Engine:</span><span style="font-family:var(--font-mono); font-weight:700;">SQLite 3 (WAL Mode)</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Calibration:</span><span style="font-family:var(--font-mono); font-weight:700;">Platt Scaling (Sigmoid)</span></div>
                        <div class="cockpit-metric-row"><span style="color:var(--text-muted);">Platform Release:</span><span style="font-family:var(--font-mono); font-weight:700;">v14.3.0-PROD</span></div>
                    </div>
                </div>
            </div>

            <!-- ROW 2: API DIRECTORY & EXPORTS / DOCS -->
            <div class="developer-grid" style="margin-top:var(--space-8);">
                <!-- 3. API REFERENCE DIRECTORY -->
                <div class="chassis-card">
                    <div class="core-plate">
                        <span class="card-subhead">API Reference Directory</span>
                        <div class="api-directory-list">
                            <div class="api-endpoint-item">
                                <div class="api-endpoint-head">
                                    <span class="badge-method post">POST</span>
                                    <span class="api-path">/api/v1/predict</span>
                                </div>
                                <span class="api-desc">Single review classification with token feature attribution (XAI).</span>
                            </div>
                            <div class="api-endpoint-item">
                                <div class="api-endpoint-head">
                                    <span class="badge-method post">POST</span>
                                    <span class="api-path">/api/v1/consensus</span>
                                </div>
                                <span class="api-desc">Simultaneous multi-engine arbitration across 3 production models.</span>
                            </div>
                            <div class="api-endpoint-item">
                                <div class="api-endpoint-head">
                                    <span class="badge-method post">POST</span>
                                    <span class="api-path">/api/v1/feedback</span>
                                </div>
                                <span class="api-desc">Human-in-the-loop ground-truth verification feedback loop.</span>
                            </div>
                            <div class="api-endpoint-item">
                                <div class="api-endpoint-head">
                                    <span class="badge-method get">GET</span>
                                    <span class="api-path">/api/v1/analytics</span>
                                </div>
                                <span class="api-desc">Operational telemetry aggregates, latency percentiles &amp; traffic distribution.</span>
                            </div>
                            <div class="api-endpoint-item">
                                <div class="api-endpoint-head">
                                    <span class="badge-method get">GET</span>
                                    <span class="api-path">/api/v1/history</span>
                                </div>
                                <span class="api-desc">Paginated forensic journal entries persisted to SQLite.</span>
                            </div>
                            <div class="api-endpoint-item">
                                <div class="api-endpoint-head">
                                    <span class="badge-method get">GET</span>
                                    <span class="api-path">/api/v1/export/{format}</span>
                                </div>
                                <span class="api-desc">Full telemetry audit log download in CSV or JSON format.</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 4. EXPORTS & 5. DOCUMENTATION -->
                <div class="chassis-card">
                    <div class="core-plate">
                        <span class="card-subhead">Data Exports &amp; Documentation</span>
                        <div class="export-card-grid">
                            <a href="/api/v1/export/csv" class="export-action-btn">
                                <svg class="svg-icon" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                                Export CSV
                            </a>
                            <a href="/api/v1/export/json" class="export-action-btn">
                                <svg class="svg-icon" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                                Export JSON
                            </a>
                        </div>

                        <div style="border-top:1px solid var(--border-hairline); padding-top:var(--space-4); margin-top:var(--space-2);">
                            <span class="card-subhead" style="margin-bottom:var(--space-3);">OpenAPI Documentation Links</span>
                            <div style="display:flex; gap:10px;">
                                <a href="/docs" target="_blank" rel="noopener noreferrer" class="btn-feedback" style="text-decoration:none;">Swagger UI (/docs)</a>
                                <a href="/redoc" target="_blank" rel="noopener noreferrer" class="btn-feedback" style="text-decoration:none;">ReDoc (/redoc)</a>
                            </div>
                        </div>

                        <!-- PRIVACY NOTICE SENTINEL -->
                        <div style="border-top:1px solid var(--border-hairline); padding-top:var(--space-4); margin-top:var(--space-2); display:flex; gap:12px; align-items:flex-start;">
                            <div class="brand-icon-pod" style="width:28px; height:28px; flex-shrink:0;">
                                <svg class="svg-icon" viewBox="0 0 24 24"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
                            </div>
                            <div style="font-size:12px; color:var(--text-muted); line-height:1.6;">
                                <strong style="color:var(--text-secondary);">Enterprise Privacy Sentinel:</strong> Natural language input is transmitted securely over HTTP/TLS. Raw natural language text is not stored by default; server-side SHA-256 anonymization is applied prior to SQLite database persistence when STORE_RAW_TEXT is disabled.
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

    </main>

    <!-- SUBTLE BUILD METADATA FOOTER -->
    <footer class="app-footer">
        <div class="app-footer-inner">
            <span>SENTINEL AI &bull; NLP SENTIMENT INTELLIGENCE INSTRUMENT</span>
            <span>BUILD 14.3.0 &bull; PRODUCTION RELEASE</span>
        </div>
    </footer>

    <!-- CONTEXTUAL FLOATING ACTION MENU / QUICK ACTION HUB -->
    <aside class="quick-action-hub" id="quickActionHub" aria-label="Quick Navigation and Intelligence Actions">
        <div class="quick-action-menu" id="quickActionMenu" role="menu" aria-orientation="vertical" aria-hidden="true">
            <button type="button" class="quick-action-item" role="menuitem" onclick="triggerQuickAction('analyze')">
                <svg class="svg-icon" viewBox="0 0 24 24"><path d="M2 12h5l3 7 4-14 3 7h5"/></svg>
                <span>Analyze Signal</span>
            </button>
            <button type="button" class="quick-action-item" role="menuitem" onclick="triggerQuickAction('consensus')">
                <svg class="svg-icon" viewBox="0 0 24 24"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                <span>Evaluate Consensus</span>
            </button>
            <button type="button" class="quick-action-item" role="menuitem" onclick="triggerQuickAction('export')">
                <svg class="svg-icon" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                <span>Export Telemetry (CSV)</span>
            </button>
        </div>
        <button type="button" class="quick-action-fab" id="quickActionFab" aria-expanded="false" aria-haspopup="menu" aria-controls="quickActionMenu" aria-label="Open Quick Intelligence Actions" onclick="toggleQuickActionMenu()">
            <svg class="svg-icon fab-icon-idle" viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            <svg class="svg-icon fab-icon-active" viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
    </aside>

    <!-- ==========================================================================
         RESTFUL CONTROLLER & PHYSICAL INTERACTION ENGINE
         ========================================================================== -->
    <script>
        const PRESETS = {
            pos: "A breathtaking cinematic masterpiece. The cinematography was magnificent, the acting sublime, and the narrative profoundly moving.",
            neg: "An unmitigated disaster. The script was hollow, direction non-existent, and the performances painfully dreadful and boring.",
            amb: "Visually stunning and technically brilliant, but the pacing dragged considerably and the ending felt entirely unearned."
        };

        let currentPredictionId = null;
        let lastInferenceScore = 0;
        let activeTelemetryFilter = 'ALL';
        let cachedTelemetryRecords = [];

        // --- TAB NAVIGATION & PILL GLIDE CONTROLLER ---
        function switchTab(tabName) {
            window.location.hash = '#' + tabName;
            handleHashRoute();
        }

        function updateNavGlider(activeBtn) {
            const glider = document.getElementById('navGlider');
            const dock = document.getElementById('navDock');
            if (!glider || !dock || !activeBtn) return;
            const offsetLeft = activeBtn.offsetLeft;
            const width = activeBtn.offsetWidth;
            glider.style.transform = `translateX(${offsetLeft - 4}px)`;
            glider.style.width = `${width}px`;
        }

        function handleHashRoute() {
            const hash = (window.location.hash || '#analyze').replace('#', '');
            const validTabs = ['analyze', 'consensus', 'benchmarks', 'telemetry', 'developer'];
            const targetTab = validTabs.includes(hash) ? hash : 'analyze';

            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.view-panel').forEach(panel => panel.classList.remove('active'));

            const activeBtn = document.getElementById('tab-' + targetTab);
            const activePanel = document.getElementById('view-' + targetTab);

            if (activeBtn) {
                activeBtn.classList.add('active');
                updateNavGlider(activeBtn);
            }
            if (activePanel) {
                activePanel.classList.add('active');
            }

            if (targetTab === 'telemetry') {
                loadTelemetry();
            }
        }

        window.addEventListener('hashchange', handleHashRoute);
        window.addEventListener('resize', () => {
            const activeBtn = document.querySelector('.tab-btn.active');
            if (activeBtn) updateNavGlider(activeBtn);
        });

        // --- LUXURY LIGHT / DARK MODE TOGGLE ---
        function toggleAmbientTheme(isDarkMode) {
            if (isDarkMode) {
                document.documentElement.removeAttribute('data-theme');
            } else {
                document.documentElement.setAttribute('data-theme', 'light');
            }
            const activeBtn = document.querySelector('.tab-btn.active');
            if (activeBtn) updateNavGlider(activeBtn);
        }

        // --- PRESETS & INPUT HELPERS ---
        function applyPreset(key) {
            const input = document.getElementById('nlpInput');
            if (input && PRESETS[key]) {
                input.value = PRESETS[key];
                updateCounters();
            }
        }

        function updateCounters() {
            const input = document.getElementById('nlpInput');
            const text = input ? input.value : '';
            const charCount = text.length;
            const wordCount = text.trim() === '' ? 0 : text.trim().split(/\s+/).length;

            const charEl = document.getElementById('charCounter');
            const wordEl = document.getElementById('wordCounter');
            if (charEl) charEl.textContent = `${charCount.toLocaleString()} / 5,000 characters`;
            if (wordEl) wordEl.textContent = `${wordCount.toLocaleString()} words`;
        }

        function handleTextKeydown(event) {
            if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                event.preventDefault();
                executeAnalysis(event);
            }
        }

        // --- CLICK RIPPLE HELPER ---
        function createRipple(event, button) {
            if (!event || !event.clientX) return;
            const rect = button.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = event.clientX - rect.left - size / 2;
            const y = event.clientY - rect.top - size / 2;
            const ripple = document.createElement('span');
            ripple.className = 'ripple-wave';
            ripple.style.width = ripple.style.height = `${size}px`;
            ripple.style.left = `${x}px`;
            ripple.style.top = `${y}px`;
            button.appendChild(ripple);
            setTimeout(() => ripple.remove(), 650);
        }

        // --- TOAST NOTIFICATIONS ---
        let toastTimeout = null;
        function showToast(msg) {
            const banner = document.getElementById('toastBanner');
            const msgEl = document.getElementById('toastMessage');
            if (!banner || !msgEl) return;
            msgEl.textContent = msg;
            banner.classList.add('show');
            if (toastTimeout) clearTimeout(toastTimeout);
            toastTimeout = setTimeout(() => {
                banner.classList.remove('show');
            }, 3800);
        }

        function dismissToast() {
            const banner = document.getElementById('toastBanner');
            if (banner) banner.classList.remove('show');
        }

        // --- EXECUTE INFERENCE ANALYSIS ---
        async function executeAnalysis(event) {
            const input = document.getElementById('nlpInput');
            const text = input ? input.value.trim() : '';
            const btn = document.getElementById('btnAnalyze');
            const spinner = document.getElementById('analyzeSpinner');
            const btnText = document.getElementById('analyzeBtnText');
            const inputCard = document.getElementById('inputCard');
            const resultCard = document.getElementById('resultCard');

            if (event && btn) createRipple(event, btn);

            if (!text) {
                showToast("Please provide review discourse text before executing analysis.");
                return;
            }

            const engine = document.getElementById('modelSelect').value;

            // Loading state
            btn.disabled = true;
            spinner.style.display = 'inline-block';
            btnText.textContent = 'ANALYZING SIGNAL...';
            inputCard.classList.remove('edge-positive', 'edge-negative');
            inputCard.classList.add('edge-processing');
            resultCard.classList.remove('edge-positive', 'edge-negative');
            resultCard.classList.add('edge-processing');

            try {
                const res = await fetch('/api/v1/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text, model: engine })
                });

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || 'Inference call failed');
                }

                const data = await res.json();
                renderInferenceResults(data, engine);
            } catch (err) {
                showToast("Inference Failed: " + err.message);
            } finally {
                btn.disabled = false;
                spinner.style.display = 'none';
                btnText.textContent = 'EXECUTE ANALYSIS';
                inputCard.classList.remove('edge-processing');
                resultCard.classList.remove('edge-processing');
            }
        }

        function renderInferenceResults(data, engine) {
            currentPredictionId = data.prediction_id;
            lastInferenceScore = data.ai_score;

            // Trigger physical settling animation
            const resultCard = document.getElementById('resultCard');
            if (resultCard) {
                resultCard.classList.remove('result-revealing');
                void resultCard.offsetWidth;
                resultCard.classList.add('result-revealing');
            }

            // Update Metadata
            document.getElementById('resTimestamp').textContent = new Date().toLocaleTimeString();
            document.getElementById('resLatency').textContent = data.latency_ms.toFixed(2) + ' ms';
            document.getElementById('resTokens').textContent = data.tokens_count + ' tokens';
            document.getElementById('resReqId').textContent = data.request_id ? data.request_id.slice(0, 8) : 'REQ-LIVE';

            // Update Sentiment Hero
            const sEl = document.getElementById('resSentiment');
            const isPos = data.sentiment === 'Positive';
            sEl.style.color = isPos ? 'var(--signal-positive)' : 'var(--signal-negative)';
            sEl.innerHTML = `<span>${data.sentiment.toUpperCase()}</span><span class="sentiment-engine-tag" style="background:${isPos ? 'var(--signal-positive-bg)' : 'var(--signal-negative-bg)'}; color:${isPos ? 'var(--signal-positive)' : 'var(--signal-negative)'};">${engine.toUpperCase()}</span>`;

            // Update Cards Edge Lighting
            const inputCard = document.getElementById('inputCard');
            inputCard.classList.remove('edge-positive', 'edge-negative');
            resultCard.classList.remove('edge-positive', 'edge-negative');
            inputCard.classList.add(isPos ? 'edge-positive' : 'edge-negative');
            resultCard.classList.add(isPos ? 'edge-positive' : 'edge-negative');

            // Dominant Emotion
            document.getElementById('resEmotion').textContent = data.emotion || 'Unspecified';

            // Update Bipolar Polarity HUD (-1.0000 to +1.0000)
            const score = data.ai_score;
            document.getElementById('resPolarityScore').textContent = (score >= 0 ? '+' : '') + score.toFixed(4);
            const polarityBar = document.getElementById('polarityBipolarBar');
            if (score >= 0) {
                polarityBar.style.left = '50%';
                polarityBar.style.width = (score * 50) + '%';
                polarityBar.style.background = 'var(--signal-positive)';
            } else {
                polarityBar.style.left = (50 - Math.abs(score) * 50) + '%';
                polarityBar.style.width = (Math.abs(score) * 50) + '%';
                polarityBar.style.background = 'var(--signal-negative)';
            }

            // Confidence
            const confPct = Number(data.confidence).toFixed(2);
            document.getElementById('resConfidence').textContent = confPct + '%';
            document.getElementById('confBar').style.width = Math.min(100, Math.max(0, confPct)) + '%';
            document.getElementById('confBar').style.background = isPos ? 'var(--signal-positive)' : 'var(--signal-negative)';

            // Feature Attribution Drivers (XAI)
            renderXaiDrivers(data.explanation);

            // Enable Feedback Buttons
            document.getElementById('btnConfirmFeedback').disabled = false;
            document.getElementById('btnFlagFeedback').disabled = false;
            document.getElementById('feedbackSuccessAnim').style.display = 'none';
        }

        function renderXaiDrivers(explanation) {
            const posList = document.getElementById('posDriversList');
            const negList = document.getElementById('negDriversList');
            posList.innerHTML = '';
            negList.innerHTML = '';

            const pos = explanation && explanation.positive_drivers ? explanation.positive_drivers : [];
            const neg = explanation && explanation.negative_drivers ? explanation.negative_drivers : [];

            if (pos.length === 0) {
                posList.innerHTML = '<div class="driver-empty-hint">None detected within vocabulary</div>';
            } else {
                const maxPos = Math.max(...pos.map(d => Math.abs(d.weight)), 0.001);
                pos.forEach(d => {
                    const row = document.createElement('div');
                    row.className = 'driver-bar-row';
                    const pct = Math.min(100, Math.round((Math.abs(d.weight) / maxPos) * 100));
                    row.title = `Token: ${escapeHtml(d.token)} | Weight: +${d.weight.toFixed(4)} | TF-IDF × Model Coefficient`;
                    row.innerHTML = `
                        <span class="driver-token">${escapeHtml(d.token)}</span>
                        <div class="driver-bar-track">
                            <div class="driver-bar-fill" style="width:${pct}%; background:var(--signal-positive);"></div>
                        </div>
                        <span class="driver-weight" style="color:var(--signal-positive);">+${d.weight.toFixed(4)}</span>
                    `;
                    posList.appendChild(row);
                });
            }

            if (neg.length === 0) {
                negList.innerHTML = '<div class="driver-empty-hint">None detected within vocabulary</div>';
            } else {
                const maxNeg = Math.max(...neg.map(d => Math.abs(d.weight)), 0.001);
                neg.forEach(d => {
                    const row = document.createElement('div');
                    row.className = 'driver-bar-row';
                    const pct = Math.min(100, Math.round((Math.abs(d.weight) / maxNeg) * 100));
                    row.title = `Token: ${escapeHtml(d.token)} | Weight: ${d.weight.toFixed(4)} | TF-IDF × Model Coefficient`;
                    row.innerHTML = `
                        <span class="driver-token">${escapeHtml(d.token)}</span>
                        <div class="driver-bar-track">
                            <div class="driver-bar-fill" style="width:${pct}%; background:var(--signal-negative);"></div>
                        </div>
                        <span class="driver-weight" style="color:var(--signal-negative);">${d.weight.toFixed(4)}</span>
                    `;
                    negList.appendChild(row);
                });
            }
        }

        // --- AUDIT FEEDBACK SUBMISSION ---
        async function sendAuditFeedback(isCorrect) {
            if (!currentPredictionId) return;

            try {
                const res = await fetch('/api/v1/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prediction_id: currentPredictionId,
                        is_correct: isCorrect,
                        notes: isCorrect ? "Audited as accurate" : "Flagged inaccurate"
                    })
                });

                if (!res.ok) throw new Error('Feedback submission failed');

                document.getElementById('btnConfirmFeedback').disabled = true;
                document.getElementById('btnFlagFeedback').disabled = true;

                // Trigger SVG checkmark animation
                const animBox = document.getElementById('feedbackSuccessAnim');
                animBox.style.display = 'inline-flex';

                showToast(isCorrect ? "Result confirmed. Telemetry audit recorded." : "Result flagged incorrect. Telemetry audit recorded.");
            } catch (e) {
                showToast("Failed to record feedback: " + e.message);
            }
        }

        // --- CONSENSUS COCKPIT EXECUTION ---
        async function executeConsensus(event) {
            const input = document.getElementById('nlpInput');
            const text = input ? input.value.trim() : '';
            const btn = document.getElementById('btnEvaluateConsensus');
            const spinner = document.getElementById('consensusSpinner');
            const btnText = document.getElementById('consensusBtnText');

            if (event && btn) createRipple(event, btn);

            if (!text) {
                showToast("Please provide review text in the Ingestion tab before evaluating consensus.");
                return;
            }

            if (spinner) spinner.style.display = 'inline-block';
            if (btnText) btnText.textContent = 'Evaluating...';
            const consensusCards = document.querySelectorAll('.consensus-cockpit-grid .chassis-card');
            consensusCards.forEach(c => c.classList.add('edge-processing'));

            try {
                const res = await fetch('/api/v1/consensus', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                if (!res.ok) throw new Error('Consensus evaluation request failed');
                const data = await res.json();

                // Agreement ratio
                const ratioEl = document.getElementById('consensusRatio');
                const badgeEl = document.getElementById('consensusBadge');
                ratioEl.textContent = data.agreement_ratio;
                badgeEl.style.display = data.disagreement_detected ? 'none' : 'inline-block';

                // Map model responses
                const models = data.models || {};
                const m1 = models['logistic_regression'] || models['Logistic Regression'] || {};
                const m2 = models['linear_svm'] || models['Linear SVM'] || {};
                const m3 = models['naive_bayes'] || models['Multinomial Naive Bayes'] || {};

                document.getElementById('cM1Sentiment').textContent = (m1.sentiment || '—').toUpperCase();
                document.getElementById('cM1Score').textContent = m1.ai_score !== undefined ? ((m1.ai_score >= 0 ? '+' : '') + m1.ai_score.toFixed(4)) : '—';
                document.getElementById('cM1Conf').textContent = m1.confidence !== undefined ? Number(m1.confidence).toFixed(1) + '%' : '—';
                document.getElementById('cM1Emotion').textContent = m1.emotion || '—';

                document.getElementById('cM2Sentiment').textContent = (m2.sentiment || '—').toUpperCase();
                document.getElementById('cM2Score').textContent = m2.ai_score !== undefined ? ((m2.ai_score >= 0 ? '+' : '') + m2.ai_score.toFixed(4)) : '—';
                document.getElementById('cM2Conf').textContent = m2.confidence !== undefined ? Number(m2.confidence).toFixed(1) + '%' : '—';
                document.getElementById('cM2Emotion').textContent = m2.emotion || '—';

                document.getElementById('cM3Sentiment').textContent = (m3.sentiment || '—').toUpperCase();
                document.getElementById('cM3Score').textContent = m3.ai_score !== undefined ? ((m3.ai_score >= 0 ? '+' : '') + m3.ai_score.toFixed(4)) : '—';
                document.getElementById('cM3Conf').textContent = m3.confidence !== undefined ? Number(m3.confidence).toFixed(1) + '%' : '—';
                document.getElementById('cM3Emotion').textContent = m3.emotion || '—';
            } catch (e) {
                showToast("Consensus evaluation failed: " + e.message);
            } finally {
                if (spinner) spinner.style.display = 'none';
                if (btnText) btnText.textContent = 'Evaluate Consensus';
                consensusCards.forEach(c => c.classList.remove('edge-processing'));
            }
        }

        // --- TELEMETRY LEDGER & ANALYTICS ---
        async function loadTelemetry() {
            try {
                const [analyticsRes, historyRes] = await Promise.all([
                    fetch('/api/v1/analytics'),
                    fetch('/api/v1/history?limit=50')
                ]);

                if (analyticsRes.ok) {
                    const a = await analyticsRes.json();
                    const totalEl = document.getElementById('statTotal');
                    if (totalEl) totalEl.textContent = (a.total_analyzed || 0).toLocaleString();
                    const posEl = document.getElementById('statPos');
                    if (posEl) posEl.textContent = (a.positive || 0).toLocaleString();
                    const negEl = document.getElementById('statNeg');
                    if (negEl) negEl.textContent = (a.negative || 0).toLocaleString();
                    const neuEl = document.getElementById('statNeu');
                    if (neuEl) neuEl.textContent = (a.neutral || 0).toLocaleString();
                    const avgEl = document.getElementById('statAvgLat');
                    if (avgEl) avgEl.textContent = (a.avg_latency_ms || 0).toFixed(2) + ' ms';

                    const p95Val = typeof a.p95_latency_ms === 'number' ? a.p95_latency_ms.toFixed(1) + ' ms' : (a.p95_latency_ms || '—');
                    const p99Val = typeof a.p99_latency_ms === 'number' ? a.p99_latency_ms.toFixed(1) + ' ms' : (a.p99_latency_ms || '—');
                    const p95El = document.getElementById('statP95');
                    if (p95El) p95El.textContent = p95Val;
                    const p99El = document.getElementById('statP99');
                    if (p99El) p99El.textContent = p99Val;

                    const spreadEl = document.getElementById('statSpread');
                    if (spreadEl) spreadEl.textContent = `${a.positive || 0} Pos / ${a.negative || 0} Neg`;
                    const tailEl = document.getElementById('statTailLat');
                    if (tailEl) tailEl.textContent = `${p95Val} / ${p99Val}`;

                    // Real model distribution breakdown
                    const distEl = document.getElementById('statModelDist');
                    if (distEl && a.model_distribution) {
                        const dist = a.model_distribution;
                        const entries = Object.entries(dist);
                        if (entries.length === 0) {
                            distEl.textContent = 'Awaiting live traffic';
                        } else {
                            distEl.innerHTML = entries.map(([model, cnt]) => `<span><strong style="color:var(--text-primary);">${escapeHtml(model)}:</strong> ${cnt}</span>`).join(' &bull; ');
                        }
                    }
                }

                if (historyRes.ok) {
                    cachedTelemetryRecords = await historyRes.json();
                    renderTelemetryRows(cachedTelemetryRecords);
                }
            } catch (e) {
                showToast("Failed to refresh telemetry: " + e.message);
            }
        }

        function toggleFilterChip(chip, filterValue) {
            document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('on'));
            chip.classList.add('on');
            activeTelemetryFilter = filterValue;
            filterTelemetryTable();
        }

        function filterTelemetryTable() {
            const query = (document.getElementById('telemetrySearchInput').value || '').toLowerCase().trim();
            const filtered = cachedTelemetryRecords.filter(r => {
                const matchesFilter = (activeTelemetryFilter === 'ALL') ||
                    (activeTelemetryFilter === 'POSITIVE' && r.sentiment === 'Positive') ||
                    (activeTelemetryFilter === 'NEGATIVE' && r.sentiment === 'Negative');

                const matchesQuery = !query ||
                    (r.request_id && r.request_id.toLowerCase().includes(query)) ||
                    (r.engine && r.engine.toLowerCase().includes(query)) ||
                    (r.sentiment && r.sentiment.toLowerCase().includes(query));

                return matchesFilter && matchesQuery;
            });
            renderTelemetryRows(filtered);
        }

        function renderTelemetryRows(records) {
            const tbody = document.getElementById('telemetryTbody');
            tbody.innerHTML = '';

            if (!records || records.length === 0) {
                tbody.innerHTML = '<tr><td colspan="10" style="text-align:center; padding:28px; color:var(--text-dim);">No telemetry records found.</td></tr>';
                return;
            }

            records.forEach(r => {
                const tr = document.createElement('tr');
                const isPos = r.sentiment === 'Positive';
                tr.innerHTML = `
                    <td data-label="ID">${r.id}</td>
                    <td data-label="Request ID"><span style="color:var(--accent-champagne); font-weight:600;">${r.request_id ? escapeHtml(r.request_id.slice(0, 8)) : '&mdash;'}</span></td>
                    <td data-label="Sentiment">
                        <span style="color: ${isPos ? 'var(--signal-positive)' : 'var(--signal-negative)'}; font-weight:700;">
                            ${escapeHtml(r.sentiment)}
                        </span>
                    </td>
                    <td data-label="Confidence">${Number(r.confidence).toFixed(1)}%</td>
                    <td data-label="AI Score">${r.ai_score !== undefined ? (r.ai_score >= 0 ? '+' : '') + r.ai_score.toFixed(4) : '&mdash;'}</td>
                    <td data-label="Engine">${escapeHtml(r.engine || 'Logistic Regression')}</td>
                    <td data-label="Emotion">${escapeHtml(r.emotion || '&mdash;')}</td>
                    <td data-label="Latency">${r.latency_ms ? r.latency_ms.toFixed(2) + ' ms' : '&mdash;'}</td>
                    <td data-label="Feedback">${r.user_feedback ? escapeHtml(r.user_feedback) : '<span style="color:var(--text-dim);">Unreviewed</span>'}</td>
                    <td data-label="Timestamp">${r.timestamp ? escapeHtml(r.timestamp.split('T')[0] + ' ' + (r.timestamp.split('T')[1] || '').slice(0, 5)) : '&mdash;'}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        // --- DEVELOPER SANDBOX API CALL ---
        async function sendApiRequest(event) {
            const payloadText = document.getElementById('devPayload').value;
            const resPre = document.getElementById('devResponse');
            const btn = document.getElementById('btnSendApi');
            const spinner = document.getElementById('devSpinner');
            const btnText = document.getElementById('devBtnText');
            const devCard = document.getElementById('devSandboxCard');

            if (event && btn) createRipple(event, btn);

            let bodyObj;
            try {
                bodyObj = JSON.parse(payloadText);
            } catch (err) {
                resPre.textContent = "// Error parsing JSON payload:\n" + err.message;
                return;
            }

            if (spinner) spinner.style.display = 'inline-block';
            if (btnText) btnText.textContent = 'Sending...';
            if (devCard) devCard.classList.add('edge-processing');

            try {
                const res = await fetch('/api/v1/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(bodyObj)
                });

                const data = await res.json();
                resPre.textContent = JSON.stringify(data, null, 2);
            } catch (e) {
                resPre.textContent = "// Network / Server Error:\n" + e.message;
            } finally {
                if (spinner) spinner.style.display = 'none';
                if (btnText) btnText.textContent = 'Send Test Request';
                if (devCard) devCard.classList.remove('edge-processing');
            }
        }

        // --- CONTEXTUAL QUICK ACTION HUB ---
        function toggleQuickActionMenu() {
            const hub = document.getElementById('quickActionHub');
            const fab = document.getElementById('quickActionFab');
            const menu = document.getElementById('quickActionMenu');
            if (!hub || !fab || !menu) return;
            const isOpen = hub.classList.toggle('open');
            fab.setAttribute('aria-expanded', String(isOpen));
            menu.setAttribute('aria-hidden', String(!isOpen));
        }

        function closeQuickActionMenu() {
            const hub = document.getElementById('quickActionHub');
            const fab = document.getElementById('quickActionFab');
            const menu = document.getElementById('quickActionMenu');
            if (!hub || !fab || !menu) return;
            hub.classList.remove('open');
            fab.setAttribute('aria-expanded', 'false');
            menu.setAttribute('aria-hidden', 'true');
        }

        function triggerQuickAction(action) {
            closeQuickActionMenu();
            if (action === 'analyze') {
                switchTab('analyze');
                const nlp = document.getElementById('nlpInput');
                if (nlp) {
                    nlp.focus();
                    nlp.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            } else if (action === 'consensus') {
                switchTab('consensus');
                executeConsensus();
            } else if (action === 'export') {
                window.location.href = '/api/v1/export/csv';
            }
        }

        // Global dismiss for quick action hub on outside click or Escape
        document.addEventListener('click', (e) => {
            const hub = document.getElementById('quickActionHub');
            if (hub && hub.classList.contains('open') && !hub.contains(e.target)) {
                closeQuickActionMenu();
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeQuickActionMenu();
                dismissToast();
            }
        });

        // --- SANITIZATION HELPER ---
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
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
