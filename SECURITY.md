# SENTINEL AI — Security & Threat Modeling Review

**Platform:** SENTINEL AI — NLP Sentiment Intelligence Platform  
**Target Release:** `v14.3.0-PROD`  
**Security Status:** Defensible Production Standard  
**Repository:** [sentiment-analysis-nlp-ai](https://github.com/T-Stephen/sentiment-analysis-nlp-ai)

---

## 1. Threat Landscape & Risk Posture
SENTINEL AI processes executive communications and user feedback via a FastAPI ASGI microservice backed by SQLite persistence. The primary threat vectors addressed in this architecture include:
1. **Denial of Service (DoS):** Input poisoning or memory exhaustion via oversized text payloads.
2. **Database Manipulation & Injection:** Malicious inputs attempting SQL injection.
3. **Data Leakage & Privacy Violations:** Accidental persistence or unauthorized export of confidential discourse.
4. **Clickjacking & Cross-Site Scripting (XSS):** Web UI manipulation via untrusted scripts or iframe embedding.
5. **Path Traversal:** File system exploration via crafted export parameters.
6. **Insecure Deserialization:** Arbitrary code execution via untrusted pickle loading.

---

## 2. Implemented Security Controls

### 2.1 Input Validation & Payload Safeguards
- **Strict Boundary Constraints:** Pydantic v2 schemas (`TextPayload`) enforce a hard maximum length of 5,000 characters per inference request (`min_length=1, max_length=5000`).
- **Whitespace Sanitation:** Custom field validators strip surrounding whitespace and reject empty or whitespace-only inputs with structured HTTP 422 Unprocessable Entity responses.
- **Model Key Validation:** Model selectors are validated against an allowlist (`logistic_regression`, `linear_svm`, `naive_bayes`). Unrecognized strings default safely to regularized Logistic Regression rather than failing or executing arbitrary code.

### 2.2 Privacy & Cryptographic Text Redaction
- **Default Privacy Mode (`STORE_RAW_TEXT=False`):** Incoming natural language text is hashed via cryptographic SHA-256 prior to database persistence:
  $$\text{Stored Digest} = \text{"[REDACTED] - SHA256:"} + \text{SHA256}(\text{Text})$$
- **History Isolation:** The `/api/v1/history` endpoint explicitly omits raw text fields from its SQL query (`SELECT id, request_id, sentiment, confidence, engine, ai_score, emotion, user_feedback, latency_ms, timestamp`), guaranteeing zero textual data leakage into browser session memory.
- **Export Redaction:** Operational telemetry exports (`/api/v1/export/csv` and `/api/v1/export/json`) export the redacted hash digest unless an operator explicitly overrides the environment with `STORE_RAW_TEXT=true`.

### 2.3 SQL Injection Prevention
- **Parameterized SQL Queries:** 100% of SQLite database interactions utilize parameterized queries with `?` parameter substitution across `sqlite3` cursors. Zero string formatting, concatenation, or interpolation is used.
- **Connection Isolation:** Database connections are isolated using `contextlib.closing`, parameterized timeouts (10 seconds), and explicit commit blocks.

### 2.4 Correlation & Request Tracing
- **Request ID Propagation (`X-Request-ID`):** Outbound responses automatically receive a unique correlation identifier (`SN-<8-hex>`) via middleware, or inherit the client-provided header for distributed tracing.

### 2.5 HTTP Defense-in-Depth Headers
Injected by `SecurityAndObservabilityMiddleware` on every HTTP response:
- `X-Content-Type-Options: nosniff`: Prevents MIME-sniffing exploits.
- `X-Frame-Options: DENY`: Prohibits iframe embedding, eliminating clickjacking risks.
- `X-XSS-Protection: 1; mode=block`: Activates browser-side cross-site scripting filters.
- `Referrer-Policy: strict-origin-when-cross-origin`: Restricts referrer data leakage.
- `Permissions-Policy: geolocation=(), camera=(), microphone=()`: Blocks browser sensory hardware access.
- `Content-Security-Policy`: Restricts scripts, styles, and fonts strictly to `self` and Google Fonts (`fonts.googleapis.com`, `fonts.gstatic.com`).

### 2.6 Path Traversal & Export Validation
- **Format Allowlisting:** The telemetry export route `/api/v1/export/{format_type}` only accepts exact string matches for `csv` and `json`. Any other input immediately returns HTTP 400 Bad Request.
- **Memory-Safe Streaming:** Telemetry exports are generated entirely in-memory using `io.StringIO` and streamed over HTTP; no files are written to or read from arbitrary host filesystem locations.

### 2.7 Artifact Deserialization Integrity
- **Restricted Model Loading:** Model deserialization (`joblib.load`) is strictly confined to pre-packaged model binaries residing within the internal `models/` directory during startup.
- **No Dynamic User Deserialization:** The service does not accept or deserialize user-uploaded pickles or model files at runtime.

### 2.8 Secret Handling & Repository Cleanliness
- **Zero Hardcoded Secrets:** Automated credential scanning confirms zero hardcoded API keys, bearer tokens, passwords, or credentials exist in the codebase.
- **Environment Isolation:** Local `.env` files, certificates (`*.pem`, `*.key`), and secrets are strictly ignored via `.gitignore` and `.dockerignore`.

---

## 3. Deployment & Operational Considerations

### 3.1 Reverse Proxy TLS & HSTS
- SENTINEL AI is designed to run behind a corporate reverse proxy (e.g., NGINX, Cloudflare, Traefik, or AWS ALB) terminating TLS/HTTPS.
- HTTP Strict Transport Security (HSTS) should be enforced at the reverse proxy layer (`Strict-Transport-Security: max-age=31536000; includeSubDomains; preload`).

### 3.2 Docker Security
- The production container builds from `python:3.11-slim`, minimizing the attack surface.
- `.dockerignore` excludes raw datasets (66.2 MB CSV), local databases, `.git`, and test caches.
- **Recommended Container Hardening:** For hardened enterprise Kubernetes or ECS clusters, specify a non-root `USER` directive in `Dockerfile`:
  ```dockerfile
  RUN useradd -u 1000 -m sentinel
  USER sentinel
  ```

### 3.3 Database Concurrency
- SQLite utilizes file-level database locks. For high-volume multi-worker production fleets (>10 concurrent ASGI workers), persistence should be migrated to a dedicated PostgreSQL or MySQL instance.
