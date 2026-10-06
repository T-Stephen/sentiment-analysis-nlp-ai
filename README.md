# SENTINEL AI — NLP Sentiment Intelligence Platform

[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E.svg)](https://scikit-learn.org/)
[![Test Suite](https://img.shields.io/badge/Tests-34%2F34%20Passing-brightgreen.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Release-v14.3.0--PROD-informational.svg)](https://github.com/T-Stephen/sentiment-analysis-nlp-ai)

> **Academic Topic:** NLP Project: Sentiment Analysis (AI + Text Processing)  
> **Repository:** [sentiment-analysis-nlp-ai](https://github.com/T-Stephen/sentiment-analysis-nlp-ai)  
> **Target Release:** `v14.3.0-PROD`  
> **Platform Category:** Classical ML Sentiment Intelligence Platform / Defense & Scientific Analytical Terminal  

---

## 1. Project Overview & Product Description
**SENTINEL AI** is a research-grade, privacy-conscious natural language processing (NLP) sentiment intelligence platform. Engineered as a precision analytical instrument, it performs high-throughput sentiment classification, multi-model consensus verification, and mathematical feature explainability without external cloud API dependencies.

Built on classical machine learning foundations, SENTINEL AI demonstrates that well-regularized linear classifiers combined with domain-aligned feature engineering can deliver low-latency inference, full statistical transparency, and defensible auditability.

---

## 2. Problem Statement & Why Sentiment Analysis Matters
In enterprise governance, customer intelligence, and academic research, evaluating unstructured qualitative discourse at scale presents three major operational hurdles:
1. **The Black-Box Dilemma:** Deep learning transformers often obscure *why* a particular sentiment score was assigned, making automated decisions difficult to audit in regulated environments.
2. **Model Fragility & Overconfidence:** Individual models frequently exhibit single-point failure modes when confronted with ambiguous discourse or borderline lexical polarity.
3. **Privacy & Data Governance:** Sending proprietary executive memos or sensitive customer feedback to third-party proprietary LLM APIs introduces data exfiltration and regulatory compliance risks.

**SENTINEL AI addresses these challenges directly** by offering:
- Completely local, offline inference using scikit-learn classifiers.
- Exact mathematical explainability (XAI) derived from linear decision boundaries.
- Multi-model consensus resolution to detect and flag model divergence.
- Default SHA-256 raw-text redaction to guarantee zero unmasked text persistence.

---

## 3. Key Capabilities
- **Three-Engine Model Ensemble:** Logistic Regression (L2 Regularized), Linear SVM (Platt Calibrated), and Multinomial Naive Bayes.
- **Epistemic Model Consensus:** Concurrent multi-model inference determining agreement ratios and raising automated `MODEL DISAGREEMENT DETECTED` flags.
- **Mathematical Explainability (XAI):** Exact linear token attribution ($w_i \times \text{TF-IDF}_i$) rendering positive and negative proportional driver bars.
- **Calibrated Polarity & Confidence:** Normalized polarity delta ($[-1.0, +1.0]$) paired with empirical class probabilities ($[50.0\%, 100.0\%]$).
- **Affective Heuristics Engine:** Deterministic mapping of lexical polarity and sentiment to cognitive emotional states (*Joy / High Satisfaction*, *Optimism*, *Anger / Frustration*, *Critical / Discontent*, *Balanced Neutral*).
- **Zero-Dependency Executive Terminal:** Precision Intelligence Instrument interface styled in deep obsidian (`#07080B`), chassis plates (`#0C0E14`), subtle brass accents, and responsive dual-mode telemetry (desktop data table and mobile stacked intelligence cards with collapsible forensic drawers).
- **Privacy-by-Design Persistence:** Automatic SHA-256 text hashing prior to SQLite storage (`STORE_RAW_TEXT=False` default).
- **Telemetry & Operational Metrics:** Real-time throughput analytics, P95/P99 latency tracking, and human auditor feedback loops (`Confirmed` / `Flagged Incorrect`).
- **Standardized Data Portability:** Streaming in-memory RFC-4180 CSV and JSON export pipelines.

---

## 4. System Architecture
The application runs as a modular, asynchronous FastAPI microservice with embedded Scikit-Learn pipelines and SQLite telemetry:

```
[User Text Input]
       │
       ▼
[1. Input Validation (Pydantic v2 Boundaries, 1–5000 chars)]
       │
       ▼
[2. NLP Preprocessing Pipeline (HTML Stripping, Regex, Lowercasing, Stopwords, Stemming)]
       │
       ▼
[3. TF-IDF Vectorization (5,000 Dimensions, Sublinear TF)]
       │
       ▼
[4. ML Model Registry] ───► Evaluates: Logistic Regression | Linear SVM | Naive Bayes
       │
       ▼
[5. Prediction Engine] ───► Computes class probabilities P(pos) and P(neg)
       │
       ▼
[6. Polarity & Confidence] ───► Polarity = P(pos) - P(neg); Confidence = max(P) * 100%
       │
       ▼
[7. XAI Attribution Engine] ───► Computes TF-IDF(i) × w_i for positive & negative drivers
       │
       ▼
[8. Consensus Engine] ───► Evaluates 3-way agreement ratio & divergence flags
       │
       ▼
[9. Telemetry & History] ───► Logs SHA-256 redacted record, latency, and auditor state to SQLite
       │
       ▼
[10. API Gateway / Dashboard] ───► Injects security headers (X-Request-ID, CSP) & streams responses
```

---

## 5. NLP Preprocessing Pipeline
To guarantee exact parity between training and inference pipelines:
1. **HTML Stripping:** Removes markup tags (`<.*?>`) preserving core textual sentences.
2. **Regex Sanitation:** Filters non-alphabetical characters (`[^a-zA-Z]`), normalizing word separators.
3. **Case Normalization:** Converts all tokens to standard lowercase.
4. **Stopword Elimination:** Filters non-informative grammatical tokens using NLTK's English stopword corpus.
5. **Morphological Stemming:** Applies the algorithmic Porter Stemmer to map inflected forms back to their morphological roots (e.g., `spectacular` $\rightarrow$ `spectacular`, `moving` $\rightarrow$ `move`).

---

## 6. TF-IDF Vectorization Methodology
Feature extraction uses a pre-fitted `TfidfVectorizer` configured with:
- **Vocabulary Dimension:** Exactly 5,000 maximum informative features (`max_features=5000`).
- **N-gram Range:** Unigrams (`ngram_range=(1, 1)`).
- **Term Weighting:** Standard TF-IDF weighting with sublinear term-frequency scaling:
  $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \left( \log \frac{1 + N}{1 + \text{DF}(t)} + 1 \right)$$
- **Sparse Representation:** Vectorized inputs are projected as $1 \times 5,000$ sparse matrices (`scipy.sparse.csr_matrix`), minimizing memory overhead.

---

## 7. Machine Learning Models
SENTINEL AI houses three complementary supervised classifiers in its model registry:

1. **Logistic Regression v1.0 (Primary Production Engine):**
   - **Type:** Generalized Linear Model with L2 regularization ($C=1.0$, `max_iter=1000`).
   - **Characteristics:** Provides smooth, well-calibrated class posterior probabilities via sigmoid transformation.
2. **Linear Support Vector Machine v1.0 (High-Margin Engine):**
   - **Type:** Linear Support Vector Classifier wrapped with Platt's `CalibratedClassifierCV` (5-fold sigmoid calibration).
   - **Characteristics:** Combines maximum-margin hyperplane separation with calibrated probability distributions.
3. **Multinomial Naive Bayes (Probabilistic Baseline):**
   - **Type:** Classical generative multinomial probabilistic classifier with Laplace smoothing ($\alpha=1.0$).
   - **Characteristics:** Fast Bayesian baseline evaluating class conditional word likelihoods.

---

## 8. Empirical Model Benchmarks & Validation
Evaluated across **10,000 independent holdout test samples** from the IMDb Large Movie Review Dataset (80/20 train/test split, random seed 42):

| Model Engine | Architecture | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression v1.0** | GLM (L2 Regularized) | **88.55%** | **0.8747** | **0.9020** | **0.8881** | **0.9571** | **ACTIVE** |
| **Linear SVM v1.0** | SVM (Platt Calibrated) | **88.40%** | **0.8751** | **0.8980** | **0.8864** | **0.9534** | **READY** |
| **Multinomial Naive Bayes** | Probabilistic Bayes | **85.10%** | **0.8471** | **0.8595** | **0.8532** | **0.9255** | **BASELINE** |

### Verified Test Partition Confusion Matrix (Logistic Regression, N=10,000)
| | Predicted Negative | Predicted Positive | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Negative** | **TN = 4,310** (86.20%) | **FP = 651** (Type I Error) | 4,961 |
| **Actual Positive** | **FN = 494** (Type II Error) | **TP = 4,545** (90.20%) | 5,039 |
| **Total Predicted** | 4,804 | 5,196 | **10,000** |

---

## 9. Sentiment Prediction & Scoring Workflow
For any input discourse:
1. Model generates raw posterior probabilities: $P(\text{Negative})$ and $P(\text{Positive})$ where $P(\text{Pos}) + P(\text{Neg}) = 1.0$.
2. **Polarity Delta ($\text{AI Score}$):**
   $$\text{Polarity} = P(\text{Positive}) - P(\text{Negative}) \in [-1.0, +1.0]$$
3. **Confidence Score:**
   $$\text{Confidence} = \max(P(\text{Positive}), P(\text{Negative})) \times 100\% \in [50.0\%, 100.0\%]$$
4. **Classification Boundary:**
   - $\text{Polarity} \ge +0.10 \implies \text{Positive}$
   - $\text{Polarity} \le -0.10 \implies \text{Negative}$
   - $-0.10 < \text{Polarity} < +0.10 \implies \text{Neutral}$ (Indeterminate / Low Margin)
5. **Affective Heuristic State:** Maps polarity combined with affective lexicons to cognitive states (*Joy / High Satisfaction*, *Optimism*, *Anger / Frustration*, *Critical / Discontent*, or *Balanced Neutral*).

---

## 10. Explainable AI (XAI) Methodology
SENTINEL AI provides deterministic, exact feature attribution directly from model mathematics without stochastic approximations (such as LIME or sampling-based SHAP):

$$\text{Attribution}(i) = \text{TF-IDF}(i) \times w_i$$

Where feature weight $w_i$ is derived strictly according to engine architecture:
- **Logistic Regression:** $w_i = \text{coef\_}[0][i]$.
- **Linear SVM:** $w_i = \frac{1}{K} \sum_{k=1}^K \text{coef\_}^{(k)}[i]$ averaged across calibrated base estimators.
- **Multinomial Naive Bayes:** $w_i = \log P(x_i \mid \text{Pos}) - \log P(x_i \mid \text{Neg})$ (Weight of Evidence).

Tokens with positive contributions ($w_i > 0$) are categorized as **Positive Drivers**; tokens with negative contributions ($w_i < 0$) are categorized as **Negative Drivers**. The user interface visualizes these drivers using proportional horizontal attribution bars.

---

## 11. Multi-Model Consensus Resolution
The consensus endpoint (`POST /api/v1/consensus`) evaluates text across all three engines simultaneously:
- Computes individual classifications, AI scores, and confidences.
- Determines majority consensus sentiment.
- Generates an exact agreement ratio (e.g., `3 / 3 models agree` or `2 / 3 models agree`).
- **Epistemic Disagreement Flag:** If unanimity is not achieved, the platform automatically flags `disagreement_detected = true`, alerting analysts that the discourse lies near engine decision boundaries.

---

## 12. Privacy Architecture & Raw-Text Redaction
- **Default Privacy Mode (`STORE_RAW_TEXT=False`):** Incoming natural language text is hashed via cryptographic SHA-256 prior to persistence:
  $$\text{Stored Record} = \text{"[REDACTED] - SHA256:"} + \text{SHA256}(\text{Text})$$
- **History Isolation:** The `/api/v1/history` endpoint strictly excludes textual payload columns from database queries, preventing accidental text leakage in frontend session memory.
- **Export Sanitization:** Telemetry exports retain correlation identifiers (`request_id`) and hash digests while preventing plaintext data leakage.

---

## 13. Security Architecture & Threat Mitigation
- **Input Boundaries:** Pydantic v2 schemas reject payloads $>5,000$ characters and reject pure-whitespace inputs with HTTP 422.
- **SQL Injection Defense:** 100% of SQLite database operations utilize parameterized queries with `?` parameter substitution.
- **HTTP Security Headers:** Injected by [`SecurityAndObservabilityMiddleware`](file:///d:/Capstone_NLP_Sentiment/app.py#L120-L163) on every response:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `X-XSS-Protection: 1; mode=block`
  - `Referrer-Policy: strict-origin-when-cross-origin`
  - `Permissions-Policy: geolocation=(), camera=(), microphone=()`
  - `Content-Security-Policy`: Restricts scripts, styles, and fonts to `self` and Google Fonts.
  - `X-Request-ID`: Correlation tracing header assigned to every inbound request.
- **Path Traversal Protection:** Telemetry exports stream in-memory via `io.StringIO` without reading from or writing to the host filesystem.

---

## 14. API Specification

| Method | Endpoint | Description | Sample Status |
| :--- | :--- | :--- | :---: |
| `GET` | `/` | Analytical Terminal User Interface (HTML5/CSS3/Vanilla JS) | `200 OK` |
| `GET` | `/health` | Core system health check (DB, models, vectorizer) | `200 OK` |
| `GET` | `/api/v1/health` | Standardized versioned health endpoint | `200 OK` |
| `GET` | `/api/v1/models` | Verified benchmark metrics & model metadata | `200 OK` |
| `POST` | `/api/v1/predict` | Single-engine sentiment inference & XAI drivers | `200 OK` |
| `POST` | `/api/v1/consensus` | Multi-engine concurrent consensus & divergence analysis | `200 OK` |
| `POST` | `/api/v1/feedback` | Records auditor verification (`Confirmed` / `Flagged Incorrect`) | `200 OK` |
| `GET` | `/api/v1/analytics` | Telemetry overview, P95/P99 latencies, engine distribution | `200 OK` |
| `GET` | `/api/v1/history` | Chronological audit logs (Privacy-redacted by default) | `200 OK` |
| `GET` | `/api/v1/export/csv` | Streams database audit logs as RFC-4180 CSV attachment | `200 OK` |
| `GET` | `/api/v1/export/json`| Streams database audit logs as structured JSON dump | `200 OK` |
| `GET` | `/docs` | Interactive Swagger UI API documentation | `200 OK` |
| `GET` | `/redoc` | Interactive ReDoc API documentation | `200 OK` |

### Example cURL Requests

**Single Model Inference:**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The operational results showed magnificent resilience and decisive strategic execution.",
    "model_choice": "logistic_regression"
  }'
```

**Multi-Model Consensus:**
```bash
curl -X POST http://localhost:8000/api/v1/consensus \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The leadership demonstrated exceptional clarity and unprecedented focus."
  }'
```

---

## 15. Repository Directory Structure
```
sentiment-analysis-nlp-ai/
├── artifacts/
│   └── benchmark/
│       └── model_metrics.json       # Machine-readable verified benchmark metrics
├── baseline/                        # Historical pre-hardening captures & baselines
│   ├── api_samples.json
│   ├── environment.txt
│   ├── metrics.json
│   └── test_results.txt
├── models/                          # Serialized scikit-learn model binaries
│   ├── linear_svm_model.pkl         # Platt-Calibrated Linear SVM artifact
│   ├── logistic_regression_model.pkl# L2-Regularized Logistic Regression artifact
│   ├── naive_bayes_model.pkl        # Multinomial Naive Bayes artifact
│   └── tfidf_vectorizer.pkl         # 5,000-term fitted TF-IDF vectorizer
├── tests/                           # 34-test automated regression suite
│   ├── test_analytics.py            # Telemetry & latency tests
│   ├── test_api.py                  # Baseline regression tests
│   ├── test_errors.py               # Health & error-handling tests
│   ├── test_exports.py              # CSV/JSON streaming tests
│   ├── test_feedback.py             # Telemetry audit feedback tests
│   ├── test_models.py               # Model registry tests
│   ├── test_prediction.py           # Single-model & consensus tests
│   ├── test_preprocessing.py        # Cleaning & stemming tests
│   ├── test_validation.py           # Input boundaries & validators
│   └── test_xai.py                  # Linear attribution mathematical tests
├── .dockerignore                    # Excludes 66MB raw CSV, caches, and local databases
├── .gitignore                       # Standard production exclusions
├── app.py                           # Core FastAPI application & Analytical Terminal UI
├── ARCHITECTURE.md                  # Comprehensive architectural topology specification
├── AUDIT_REPORT.md                  # Phase 0 hardening audit documentation
├── benchmark_evaluation.py          # Reproducible test partition evaluation script
├── CHANGELOG.md                     # Semantic version history (v14.3.0-PROD)
├── docker-compose.yml               # Container orchestration configuration
├── Dockerfile                       # Multi-stage production container specification
├── FINAL_AUDIT_REPORT.md            # Pre-release audit sign-off report
├── LICENSE                          # MIT Open Source License with dataset attribution
├── pytest.ini                       # Test suite runner configuration
├── README.md                        # Master platform documentation
├── requirements.txt                 # Pinned, verified production dependencies
├── SECURITY.md                      # Threat modeling & security controls specification
└── train_benchmarks.py              # Offline model training & benchmark generation script
```

---

## 16. Installation & Local Setup

### Prerequisites
- Python 3.11+
- Git

```bash
# 1. Clone repository
git clone https://github.com/T-Stephen/sentiment-analysis-nlp-ai.git
cd sentiment-analysis-nlp-ai

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 3. Install production dependencies
pip install -r requirements.txt

# 4. Run automated test suite
pytest -q

# 5. Launch the application
uvicorn app:app --host 0.0.0.0 --port 8000
```
Access the analytical dashboard at `http://localhost:8000`.

---

## 17. Docker Container Deployment

The application is containerized with an optimized `python:3.11-slim` image and pre-cached NLTK corpora to eliminate runtime internet dependencies:

```bash
# Build and launch via Docker Compose
docker compose up --build -d

# Verify container health
curl -f http://localhost:8000/health

# View live application logs
docker compose logs -f

# Gracefully terminate service
docker compose down
```

---

## 18. Automated Testing & Verification
The platform includes an automated 34-test regression suite ensuring stability across preprocessing, inference, validation, and security:

```bash
# Run complete test suite quietly
pytest -q

# Run with verbose output and test profiling
pytest -v --durations=10
```
**Current Baseline:** `34 passed in ~5.0s (100% pass rate)`.

---

## 19. Dataset Information
- **Source:** IMDb Large Movie Review Dataset ([Maas et al., Stanford University](https://ai.stanford.edu/~amaas/data/sentiment/)).
- **Sample Count:** 50,000 highly polarized movie reviews (25,000 positive, 25,000 negative).
- **Split Configuration:** 80% training partition (40,000 samples) / 20% independent holdout test partition (10,000 samples).
- **Random Seed:** 42 (Reproducible).
- **Exclusion:** The 66.2 MB raw CSV (`data/IMDB Dataset.csv`) is intentionally excluded from Git tracking via `.gitignore`. Evaluation can be reproduced using `benchmark_evaluation.py`.

---

## 20. Limitations & Boundaries
- **Domain Specificity:** Evaluated primarily on narrative cinematic discourse. While lexical polarity translates broadly, distribution shifts may occur when applied directly to legal contracts, technical manuals, or financial SEC filings.
- **Language Boundary:** The cleaning pipeline filters non-English characters; multilingual discourse is not currently supported.
- **Complex Negation & Sarcasm:** Complex double negatives or deeply nuanced sarcasm in lengthy text may reduce classifier confidence.
- **Persistence Scalability:** SQLite is optimized for local workstations and single-container deployments; high-concurrency multi-worker deployments should migrate to PostgreSQL.

---

## 21. Responsible AI & Transparency
- **No Artificial General Intelligence (AGI) Claims:** SENTINEL AI is a statistical machine learning tool. It does not possess understanding, human emotion, or consciousness.
- **Non-Causal Feature Attribution:** XAI driver bars represent linear mathematical correlations ($w_i \times \text{TF-IDF}_i$) with respect to the decision boundary, not causal mechanisms.
- **Auditor Governance:** Telemetry supports explicit human auditor review (`/api/v1/feedback`) to encourage human-in-the-loop oversight.

---

## 22. User Interface Preview

```
┌────────────────────────────────────────────────────────────────────────┐
│ SENTINEL AI // PRECISION INTELLIGENCE TERMINAL            v14.3.0-PROD │
│ DOMAIN: IMDB CORPUS (50K) │ ENGINES: 3 LOADED │ SYSTEM HEALTH: 100%    │
├────────────────────────────────────────────────────────────────────────┤
│ INPUT DISCOURSE:                                                       │
│ [ The leadership demonstrated exceptional clarity and vision.        ] │
│                                                                        │
│ [ EXECUTE INFERENCE (LOGISTIC REGRESSION) ]   [ EVALUATE CONSENSUS ]   │
├────────────────────────────────────────────────────────────────────────┤
│ PREDICTION: POSITIVE (JOY / HIGH SATISFACTION)                         │
│ POLARITY: +0.8534  │ CONFIDENCE: 92.67%  │ LATENCY: 1.84 ms            │
│ CONSENSUS: 3 / 3 ENGINES AGREE [LR, SVM, NB]                           │
├────────────────────────────────────────────────────────────────────────┤
│ STATISTICAL XAI ATTRIBUTION DRIVERS:                                   │
│   exceptional  [+1.4821] ██████████████████████                        │
│   vision       [+0.9124] █████████████                                 │
│   clarity      [+0.7410] ██████████                                    │
├────────────────────────────────────────────────────────────────────────┤
│ TELEMETRY AUDIT LOG (P95: 1.82ms | P99: 2.14ms):                       │
│ #163 │ 2026-10-06 │ SN-8F2B1A99 │ LR │ POSITIVE │ 92.67% │ [CONFIRMED] │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 23. Future Roadmap
- **Domain Adaptation Modules:** Sub-models fine-tuned on financial discourse (Financial PhraseBank).
- **Sub-word Tokenization:** Optional BPE tokenization to handle out-of-vocabulary technical jargon.
- **Distributed Persistence:** Optional pluggable PostgreSQL / TimescaleDB telemetry backend.
- **CI/CD Integration:** Automated GitHub Actions workflow running `pytest` across Python 3.10, 3.11, and 3.12.

---

## 24. License
This repository is open-sourced under the **MIT License**. See the [LICENSE](LICENSE) file for complete details.

### Dataset & Dependency Attribution
- **IMDb Dataset:** Provided for research and education by Andrew L. Maas et al., Stanford University.
- **Third-Party Libraries:** Scikit-Learn (BSD-3-Clause), NLTK (Apache 2.0), FastAPI (MIT), Pydantic (MIT), Uvicorn (BSD-3-Clause).

---

## 25. Author
**T-Stephen**  
GitHub: [@T-Stephen](https://github.com/T-Stephen)  
Repository: [sentiment-analysis-nlp-ai](https://github.com/T-Stephen/sentiment-analysis-nlp-ai)
