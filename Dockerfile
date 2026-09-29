FROM python:3.11-slim

WORKDIR /app

# Default = lite (BM25 only, ~100-200 MB RAM). For the full pipeline build with:
#   --build-arg REQUIREMENTS_FILE=requirements.txt   and run with LITE_MODE=false
ARG REQUIREMENTS_FILE=requirements-lite.txt
COPY requirements.txt requirements-lite.txt requirements-fastembed.txt ./
RUN pip install --no-cache-dir -r ${REQUIREMENTS_FILE}

COPY . .

EXPOSE 8000

# Render injects $PORT; fall back to 8000 locally.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
