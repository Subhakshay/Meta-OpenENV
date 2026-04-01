FROM python:3.11-slim

# Metadata
LABEL maintainer="CustomerSupportEnv"
LABEL description="OpenEnv-compliant customer support triage environment"

# Create non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY environment.py .
COPY main.py .
COPY inference.py .
COPY openenv.yaml .

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# HuggingFace Spaces runs on port 7860
EXPOSE 7860

# Health check — the automated ping must return 200
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
