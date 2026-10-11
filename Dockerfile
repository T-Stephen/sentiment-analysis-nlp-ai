FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK stopwords to eliminate external runtime network dependencies
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')"

COPY . .

# Ensure data directory exists and assign ownership to unprivileged user
RUN useradd -u 1000 -m -s /bin/bash sentinel \
    && mkdir -p /app/data \
    && chown -R sentinel:sentinel /app

USER sentinel

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]