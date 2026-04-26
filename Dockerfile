FROM python:3.11-slim

LABEL maintainer="CustomerSupportEnv"
LABEL description="OpenEnv-compliant customer support triage environment"

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY environment.py .
COPY main.py .
COPY db.py .
COPY inference.py .
COPY attacker.py .
COPY rewards.py .
COPY world_state.py .
COPY policy.py .
COPY drift_scheduler.py .
COPY generate_training_plots.py .
COPY openenv.yaml .
COPY pyproject.toml .
COPY uv.lock .

# Copy server entry point
RUN mkdir -p server
COPY server/app.py server/

# Copy dashboard + result images
COPY static/ static/
COPY results/ results/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
