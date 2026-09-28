FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

RUN useradd --create-home --uid 1000 user

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user app ./app
COPY --chown=user:user src ./src
COPY --chown=user:user models ./models

USER user
EXPOSE 7860

# Hugging Face Docker Spaces expect 7860. Render injects PORT (default 10000).
# Cloud Run injects PORT (8080). An unset PORT falls back to 7860.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
