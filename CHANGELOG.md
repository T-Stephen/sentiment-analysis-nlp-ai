# SENTINEL AI — Changelog

All notable changes to the SENTINEL AI platform are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to Semantic Versioning.

---

## [v14.3.0-PROD] — 2026-10-06

### Added
- **Precision Intelligence Analytical Terminal UI:** Production dashboard featuring deep obsidian surfaces (`#07080B`), chassis plates (`#0C0E14`), antique brass accents (`#C5A059`), and refined typography (*Geist* for primary labels, *Geist Mono* for telemetry and metrics).
- **Three-Engine Model Architecture:** Concurrent model registry housing L2-regularized Logistic Regression (88.55% accuracy, 0.8881 F1), Platt-Calibrated Linear SVM (88.40% accuracy, 0.8864 F1), and Multinomial Naive Bayes (85.10% accuracy, 0.8532 F1).
- **Multi-Model Consensus Engine:** Endpoint `POST /api/v1/consensus` evaluating all 3 classifiers concurrently, calculating exact agreement ratios, and raising automated `MODEL DISAGREEMENT DETECTED` flags when classifiers diverge.
- **Explainable AI (XAI) Attribution Driver Bars:** Exact linear token attribution ($w_i \times \text{TF-IDF}_i$) rendered as proportional positive and negative visual attribution bars for full decision transparency.
- **Responsive Telemetry & Forensic Drawers:** Dual-mode telemetry view featuring responsive data tables on desktop and stacked intelligence cards on mobile (<768px), with collapsible forensic drawers displaying verified model metadata.
- **Privacy-by-Design Architecture:** Automatic cryptographic SHA-256 raw-text redaction (`STORE_RAW_TEXT=False` default) before SQLite persistence, with text columns strictly excluded from history queries.
- **Telemetry Observability & Feedback:** Endpoint `GET /api/v1/analytics` computing throughput and empirical P95/P99 latency percentiles, accompanied by auditor verification endpoints (`POST /api/v1/feedback`).
- **Security & Observability Middleware:** Automatic request ID propagation (`X-Request-ID`), strict CSP, and security defense headers (`X-Content-Type-Options`, `X-Frame-Options`, `Permissions-Policy`, `Referrer-Policy`).
- **Data Export Pipelines:** Memory-streamed RFC-4180 CSV and formatted JSON export endpoints (`/api/v1/export/csv`, `/api/v1/export/json`).
- **Containerization & Deployment Configuration:** Production `python:3.11-slim` container with offline NLTK pre-caching, health check probing, context-optimized `.dockerignore`, and directory-mounted persistence in `docker-compose.yml`.
- **Automated Regression Suite:** 34 unit and integration tests across preprocessing, predictions, model registry, XAI, feedback, analytics, exports, boundary validation, and error propagation.
- **Pre-Release Verification & Audit Sign-Off:** Exhaustive 9-phase release readiness audit confirming zero hardcoded credentials, 13/13 API endpoints operating with 200 OK, and 100% test pass rate.

### Fixed
- **Docker Compose Volume Mount:** Corrected volume mapping in `docker-compose.yml` to directory mount `./data:/app/data`, preventing directory creation anomalies on clean host checkouts.
- **Build Context Bloat:** Created `.dockerignore` excluding 66.2 MB raw training CSV, local runtime databases, and development caches.
- **Input Boundary Validation:** Enforced strict Pydantic v2 boundary constraints (1 to 5,000 characters) and rejection of empty/pure-whitespace strings with HTTP 422.
- **Documentation Alignment:** Aligned all design system, typography, and latency specifications across documentation with verified production implementation.

---

## [v14.0.0] — 2026-09-13
- Baseline FastAPI service implementation.
- Joblib serialization for Logistic Regression, Linear SVM, and Naive Bayes models.
- Basic SQLite telemetry logging.
- Initial HTML dashboard with canvas animation.
