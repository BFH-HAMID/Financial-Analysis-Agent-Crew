# ---------------------------------------------------------------------------
# Financial Analysis Agent Crew — production image
# Build:  docker build -t financial-crew .
# Run:    docker run -p 8501:8501 --env-file .env financial-crew
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MPLBACKEND=Agg

WORKDIR /app

# System libraries: fonts for matplotlib/reportlab, curl for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        fonts-dejavu-core \
        libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so code changes do not invalidate the layer cache.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Optional: uncomment to enable Plotly's native PNG export (adds ~400 MB because
# of the bundled Chromium). Charts fall back to matplotlib without it.
# RUN pip install --no-cache-dir kaleido && plotly_get_chrome -y

COPY . .

# Non-root user; the app writes reports/logs into these directories.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p reports/output/charts reports/output/pdfs logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
