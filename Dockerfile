# UdaPlay Streamlit frontend, for Google Cloud Run.
#
# Build and deploy:
#   gcloud run deploy udaplay --source . --region <region> \
#     --memory 2Gi --set-env-vars OPENAI_BASE_URL=https://openai.vocareum.com/v1 \
#     --set-secrets OPENAI_API_KEY=udaplay-openai-key:latest
#
# See the "Deploying to Cloud Run" section of README.md for the full walkthrough.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# faiss-cpu links against OpenMP, which python:*-slim does not ship. Without
# libgomp1 the image builds fine and then dies at `import faiss` with
# "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source: this layer stays cached across code edits.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Only what the running app needs. Tests, evals and docs are deliberately
# excluded -- see .dockerignore.
COPY app.py ./
COPY udaplay/ ./udaplay/
COPY data/ ./data/

# Cloud Run backs the container filesystem with memory and mounts the image
# read-only in places, so all writes go to /tmp. Both variables must be real
# environment variables, not .env entries: they are read at import time,
# before load_env() runs.
ENV UDAPLAY_INDEX_DIR=/tmp/faiss_index_udaplay \
    UDAPLAY_LOG_DIR=/tmp/logs \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Don't run as root. /tmp stays writable for any user.
RUN useradd --create-home --uid 1000 appuser
USER appuser

# Cloud Run injects PORT (8080 by default) and ignores EXPOSE; the default here
# is only so the image behaves the same under a plain `docker run`.
ENV PORT=8080
EXPOSE 8080

# Shell form so ${PORT} expands, with `exec` so Streamlit becomes PID 1 and
# receives Cloud Run's SIGTERM directly instead of it being swallowed by sh.
# --server.address=0.0.0.0 is required: Streamlit otherwise binds localhost and
# Cloud Run's health check cannot reach it.
CMD exec streamlit run app.py \
      --server.port=${PORT} \
      --server.address=0.0.0.0
