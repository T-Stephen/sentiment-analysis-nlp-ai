# SENTINEL AI — Comprehensive Architectural & Engineering Audit Report
**Version:** v14.3.0-PROD  
**Date:** 2026-10-02  
**Auditor Roles:** Principal Software Architect, Senior ML Engineer, MLOps Engineer, Product Designer, Security Reviewer  
**Audit Scope:** Full repository assessment prior to production hardening and luxury UI redesign.

---

## 1. Executive Summary
SENTINEL AI is an executive sentiment intelligence platform combining classical machine learning models (Logistic Regression, Linear Support Vector Machines, and Multinomial Naive Bayes) with a lightweight FastAPI backend, SQLite telemetry persistence, and a single-page HTML5/CSS3/Vanilla JavaScript interface.

The application is functional and performs core inference tasks. However, several critical gaps and vulnerabilities exist in deployment configuration, explainability attribution across non-logistic models, telemetry completeness, domain transparency, model consensus, security headers, and visual interface restraint.

This audit establishes the baseline facts and sets the implementation blueprint for production hardening.

---

## 2. Current Architecture
- **Backend Service:** FastAPI application (`app.py`), running under Python 3.11 with Starlette ASGI toolkit and Pydantic v2 data validation.
- **Frontend Architecture:** Server-rendered single-page interface embedded directly as a string in `app.py` (`serve_dashboard()` returning `HTMLResponse`). Uses Vanilla JavaScript (ES6+), native Fetch API, and HTML5 Canvas for wave signal animation.
- **Storage Layer:** Local SQLite database located at `data/sentinel.db`. Connection handling uses Python's `sqlite3` standard library with `contextlib.closing` and 10-second timeout.
- **ML Pipeline:** NLTK PorterStemmer and stop words removal, scikit-learn `TfidfVectorizer` (5,000 max features), and scikit-learn classifiers (`LogisticRegression`, `CalibratedClassifierCV(LinearSVC)`, `MultinomialNB`).
- **Serialization:** Joblib (`.pkl`) artifacts for vectorizer and models.
- **Testing:** Pytest with FastAPI `TestClient` in `tests/test_api.py`.
- **Containerization:** `Dockerfile` (Python 3.11-slim) and `docker-compose.yml`.

---

## 3. Current Functionality
1. **Natural Language Preprocessing:**
   - Cleans text via regex (`[a-zA-Z]+`), converts to lowercase.
   - Filters English stop words using NLTK corpus.
   - Reduces tokens to root stems via Porter Stemmer.
2. **Sentiment Prediction:**
   - Accepts text up to 5,000 characters and model selection (`logistic_regression`, `linear_svm`, `naive_bayes`).
   - Transforms stemmed text into a 5,000-dimensional sparse TF-IDF matrix.
   - Computes class probabilities using `.predict_proba()`.
   - Determines sentiment polarity: Positive if `prob_pos > prob_neg`, Negative if `prob_neg > prob_pos`, or Neutral if difference $< 0.10$.
   - Computes polarity score `ai_score = prob_pos - prob_neg` $\in [-1.0, 1.0]$.
   - Reports confidence percentage $\max(prob_{pos}, prob_{neg}) \times 100$.
3. **Affective Heuristics Engine:**
   - Keyword lexicon mapping (anger, joy, sad terms) and polarity thresholds mapping to affective categories ("Anger / Frustration", "Joy / High Satisfaction", "Critical / Discontent", "Optimism", "Balanced Neutral").
4. **Explainable AI (Feature Attribution):**
   - Extracts non-zero TF-IDF features from input and computes linear attribution (`weight * tfidf_value`).
   - **Bug / Limitation Identified:** Currently assumes `clf.coef_` exists directly on the classifier object. As a result, only `logistic_regression` produces drivers; `linear_svm` (`CalibratedClassifierCV`) and `naive_bayes` (`MultinomialNB`) return empty drivers because neither exposes `coef_` on the top-level estimator.
5. **Telemetry & Feedback Storage:**
   - Persists predictions to SQLite database table `predictions`.
   - Hashes or redacts raw text when `STORE_RAW_TEXT=False`.
   - Allows users to submit positive/negative confirmation on prediction outcomes.
   - Provides `/api/v1/analytics` aggregate query (total count, positive count, negative count, average latency).
   - Provides `/api/v1/export/csv` and `/api/v1/export/json` operational export endpoints.

---

## 4. Current API Routes
| Method | Path | Request Body | Response Model / Type | Description |
|---|---|---|---|---|
| `GET` | `/` | None | `HTMLResponse` | Serves embedded web console. |
| `POST` | `/api/v1/predict` | `TextPayload` (`text`, `model_choice`) | `PredictionResponse` | Executes NLP inference, emotion classification, attribution, and logs telemetry. |
| `POST` | `/api/v1/feedback` | `FeedbackPayload` (`prediction_id`, `is_correct`) | JSON object | Records user confirmation or rejection of prediction. |
| `GET` | `/api/v1/analytics` | None | JSON object | Computes aggregate counts and average latency. |
| `GET` | `/api/v1/export/{format_type}` | Path param: `csv` or `json` | File attachment or JSON | Dumps predictions table records. |

*Deficiencies in API:*
- Missing dedicated `/health` or `/api/v1/health` endpoint.
- Missing model metadata endpoint (`/api/v1/models`).
- Missing multi-model consensus endpoint (`/api/v1/consensus`).
- Missing analysis history endpoint (`/api/v1/history`).

---

## 5. Current Frontend Architecture
- Embedded single-page interface inside `app.py`.
- Four navigation views:
  1. `Inference` (`#view-analyze`): Input textarea, canvas animation, model dropdown, submit button, metric display, feature attribution chips, feedback buttons.
  2. `Benchmarks` (`#view-benchmarks`): Table of model accuracies and static 2x2 confusion matrix cards.
  3. `Telemetry` (`#view-telemetry`): 4 aggregate statistic cards and CSV/JSON export buttons.
  4. `API` (`#view-developer`): Interactive API request tester and system status card.
- **Styling:** CSS variables defining dark theme (`--void`, `--surface`, `--border`), accompanied by high-saturation neon accents (`--accent-cyan: #00E5FF`, `--accent-magenta: #D500F9`, `--accent-coral: #FF0055`).
- **Canvas Wave:** `drawFluidWave()` renders triple sinewaves on an HTML5 canvas using `requestAnimationFrame`.

---

## 6. Current ML Pipeline & Model Artifacts
- **Training Source:** `data/IMDB Dataset.csv` (50,000 binary sentiment reviews).
- **Preprocessing Pipeline:**
  - Tokenization & Regex cleaning (`[a-zA-Z]+`).
  - English stopwords removal (NLTK).
  - Stemming using PorterStemmer.
  - TF-IDF Vectorization with max 5,000 features.
- **Serialized Artifacts Verified:**
  - `models/tfidf_vectorizer.pkl`: `sklearn.feature_extraction.text.TfidfVectorizer` (Vocabulary size: 5,000).
  - `models/logistic_regression_model.pkl`: `sklearn.linear_model.LogisticRegression`.
  - `models/linear_svm_model.pkl`: `sklearn.calibration.CalibratedClassifierCV` wrapping `LinearSVC(dual='auto', random_state=42)`.
  - `models/naive_bayes_model.pkl`: `sklearn.naive_bayes.MultinomialNB`.
  - `sentiment_model.pkl`: Legacy `MultinomialNB` model in root.
  - `model_benchmarks.pkl`: Serialized Python dictionary containing real training benchmarks:
    - *Naive Bayes:* Accuracy 85.10%, Precision 0.8471, Recall 0.8595, F1 0.8532, Latency ~0.0005 ms/sample.
    - *Logistic Regression:* Accuracy 88.55%, Precision 0.8747, Recall 0.9020, F1 0.8881, Latency ~0.0003 ms/sample.
    - *Linear SVM:* Accuracy 88.40%, Precision 0.8751, Recall 0.8980, F1 0.8864, Latency ~0.0020 ms/sample.

---

## 7. Current Database Schema
Database: `data/sentinel.db`  
Table: `predictions`
```sql
CREATE TABLE IF NOT EXISTS predictions (
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
);
```

---

## 8. Current Tests
Location: `tests/test_api.py` (3 test functions):
1. `test_preprocessing()`: Verifies token extraction and stemming.
2. `test_api_prediction_endpoint()`: Verifies HTTP 200, schema keys (`ai_score`, `confidence`, `sentiment`, `latency_ms < 250.0`).
3. `test_api_analytics_endpoint()`: Verifies HTTP 200 and presence of `total_analyzed`.

---

## 9. Current Deployment Capability
- `Dockerfile`:
  - Based on `python:3.11-slim`.
  - Attempts `COPY requirements.txt .` and `pip install -r requirements.txt`.
  - **CRITICAL DEFECT:** `requirements.txt` is missing from the repository, meaning any Docker build will fail immediately.
- `docker-compose.yml`:
  - Maps port 8000:8000.
  - Mounts `./sentinel.db:/app/sentinel.db`.
  - **DEFECT:** Database is stored at `data/sentinel.db`, causing volume mismatch and lost state.

---

## 10. Current Security Risks
1. **Missing Requirements File:** Breaks deterministic builds and container deployment.
2. **Permissive CORS:** `allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]`.
3. **Missing Security Headers:** No `X-Content-Type-Options`, `X-Frame-Options`, `Content-Security-Policy`, or `Strict-Transport-Security`.
4. **Git Repository Uninitialized & No `.gitignore`:** Raw dataset (`IMDB Dataset.csv` at 66.2 MB), compiled bytecode (`__pycache__`), cache directories (`.pytest_cache`), and database files risk accidental tracking.
5. **No Request ID Propagation:** Request IDs are generated for prediction records but not injected into HTTP response headers (`X-Request-ID`).
6. **Error Leakage:** Unhandled exceptions can expose internal framework error structures.
7. **Input Validation:** Empty whitespace strings can trigger inference unnecessarily.

---

## 11. Current UI Weaknesses
1. **Aesthetic Misalignment:** Heavy neon magenta/cyan/coral glows, intense button animations, and pulsating effects evoke a gaming/cyberpunk theme rather than a restrained obsidian intelligence platform.
2. **Inaccurate Benchmark Display:** Hardcoded HTML table in `#view-benchmarks` displays fabricated figures (e.g. 89.85% for SVM, when the actual benchmark artifact records 88.40%). The confusion matrix numbers are also hardcoded static HTML.
3. **Missing Model Consensus:** No feature allowing side-by-side or aggregated multi-model consensus.
4. **Missing Domain Transparency:** No responsible AI notice informing executive users that models were trained exclusively on IMDB movie reviews and may drift on enterprise documents.
5. **Missing Analysis History View:** No interactive log of recent analyses.
6. **Accessibility & Responsiveness:** Missing ARIA labels, focus states, and viewport optimizations for mobile/tablet screens.

---

## 12. Recommended Changes
1. **Engineering Hardening:**
   - Create `requirements.txt` with locked, verified dependencies.
   - Fix `docker-compose.yml` volume binding to point to `./data/sentinel.db`.
   - Add structured `/health` and `/api/v1/health` endpoints verifying model readiness and database connectivity.
   - Enforce security headers middleware and configurable CORS.
   - Validate whitespace and empty payloads.
   - Propagate `X-Request-ID` in HTTP response headers.
2. **Model Registry & Real Benchmarks:**
   - Add `/api/v1/models` endpoint serving actual metadata and metrics from `model_benchmarks.pkl` without fabrication.
   - Generate reproducible JSON benchmark artifact at `artifacts/benchmark/model_metrics.json`.
3. **Attribution Engine Fix (XAI):**
   - Correctly extract weights from `CalibratedClassifierCV` (`calibrated_classifiers_`) and log odds ratios from `MultinomialNB` (`feature_log_prob_`) so all three models provide legitimate mathematical attribution.
4. **Model Consensus:**
   - Add `/api/v1/consensus` endpoint evaluating all available models on the input, computing agreement ratio and disagreement indicators.
5. **Domain Transparency:**
   - Explicitly badge domain as `IMDB / MOVIE REVIEWS` and display responsible AI boundaries.
6. **Telemetry & Recent History:**
   - Add P95 and P99 latency calculations to telemetry.
   - Add `/api/v1/history` endpoint to retrieve recent predictions securely without raw text leakage.
7. **Premium Obsidian Redesign:**
   - Redesign interface in deep obsidian (`#08080A`), warm charcoal (`#141416`), smoked espresso (`#0F0D0B`), aged ivory (`#EAE6DF`), antique brass (`#C5A059`), deep patina (`#3D6B58`), and controlled oxblood (`#7A2626`).
   - Implement typography with Inter and JetBrains Mono.
   - Subtle, non-distracting canvas neural signal that responds strictly to inference state.
8. **Testing & Documentation:**
   - Expand unit and integration test suite across preprocessing, prediction, models, XAI, feedback, analytics, export, validation, and error handling.
   - Create comprehensive `README.md`, `ARCHITECTURE.md`, `SECURITY.md`, and `CHANGELOG.md`.

---

## 13. File Change Control Plan
### Files that will be modified:
- `app.py`: Hardening, new endpoints (`/health`, `/api/v1/models`, `/api/v1/consensus`, `/api/v1/history`), XAI attribution fix, telemetry percentiles, security headers, luxury UI redesign.
- `docker-compose.yml`: Correct volume mount path (`./data/sentinel.db:/app/data/sentinel.db`).
- `Dockerfile`: Production refinement with clean asset handling and healthcheck.

### Files that will be created:
- `requirements.txt`: Clean, production-ready dependency list.
- `.gitignore`: Comprehensive git exclusion rules.
- `artifacts/benchmark/model_metrics.json`: Reproducible machine-readable benchmark data.
- `baseline/test_results.txt`: Baseline verification results.
- `baseline/api_samples.json`: Baseline API response captures.
- `baseline/environment.txt`: Baseline runtime environment details.
- `baseline/metrics.json`: Baseline metrics summary.
- `tests/test_preprocessing.py`: Preprocessing unit tests.
- `tests/test_models.py`: Model registry and loading verification tests.
- `tests/test_prediction.py`: Core prediction pipeline tests.
- `tests/test_xai.py`: Feature attribution tests across all three engines.
- `tests/test_feedback.py`: User feedback telemetry tests.
- `tests/test_analytics.py`: Telemetry and aggregate calculation tests.
- `tests/test_exports.py`: CSV and JSON export verification tests.
- `tests/test_validation.py`: Input limits and validation error tests.
- `tests/test_errors.py`: Graceful error recovery and edge cases.
- `SECURITY.md`: Production security review and configuration.
- `ARCHITECTURE.md`: Complete system architecture and data flow.
- `CHANGELOG.md`: Version evolution record.
- `README.md`: Executive platform documentation.
- `FINAL_AUDIT_REPORT.md`: Final completion audit.

### Files that will NOT be modified (Untouched Artifacts & Data):
- `models/linear_svm_model.pkl`: Production model weights.
- `models/logistic_regression_model.pkl`: Production model weights.
- `models/naive_bayes_model.pkl`: Production model weights.
- `models/tfidf_vectorizer.pkl`: Production feature vocabulary.
- `model_benchmarks.pkl`: Historical benchmark dataset.
- `sentiment_model.pkl`: Baseline model artifact.
- `data/IMDB Dataset.csv`: Source dataset (66.2 MB).
- `data/sentinel.db`: Production database records (preserved).
- `tests/test_api.py`: Baseline regression tests (preserved untouched).
