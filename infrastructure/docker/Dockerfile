# ── Multi-stage Docker build ───────────────────────────────────────────────
# Stage 1: Builder — install dependencies into a clean virtualenv
# Stage 2: Runtime — minimal Python 3.11 slim image with only what's needed
#
# Design decisions:
#   - Multi-stage keeps the final image small (~250MB vs 1GB+ for full Python)
#   - pip install --no-cache-dir avoids bloating the image with package caches
#   - Virtualenv allows the same image to be used for dev (with dev deps) and prod

FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies required by some packages (e.g. libmagic for file detection)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Runtime ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Create non-root user for security
RUN groupadd -r app && useradd -r -g app -d /app -s /bin/false app

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY app/ ./app/
COPY pyproject.toml .

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Run as non-root user
USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
