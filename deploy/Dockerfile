# FinGraph Intelligence System — API Container
#
# Build context: rbi_intel/ (set by docker-compose.yml context: ..)
# Copies the full rbi_intel/ directory into /app.
#
# NOTE: Ollama is NOT containerised — it runs as a systemd service on the host
# for GPU passthrough. The container reaches it via OLLAMA_HOST (see ENV below).
#
# Build + run (from rbi_intel/):
#   docker compose -f deploy/docker-compose.yml --profile prod up api --build
#
# Local one-off (from rbi_intel/):
#   docker build -f deploy/Dockerfile -t rbi_intel_api .
#   docker run -p 8000:8000 --env-file .env rbi_intel_api

FROM python:3.11-slim

# System packages:
#   curl          — Docker healthcheck (curl -f http://localhost:8000/health)
#   build-essential, libpq-dev — psycopg2 native build (binary wheel covers most cases,
#                                but keep these for edge-case platform mismatches)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying source so layer is cached
# unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project (build context is rbi_intel/)
COPY . .

# Apply ChromaDB 0.5.5 compatibility patches.
# In Docker, site-packages live under /usr/local (not ./venv), so we override
# the VENV variable that apply_patches.sh uses to locate chromadb.
RUN VENV=/usr/local bash patches/apply_patches.sh

# Silence Python stdout buffering so logs appear immediately in `docker logs`
ENV PYTHONUNBUFFERED=1

# Ollama runs on the host; Docker Desktop (Mac/Win) and most Linux Docker setups
# expose the host at host.docker.internal. Override this env var on EC2 if Ollama
# is on a different address (e.g., Ollama on the same EC2 host → 172.17.0.1).
ENV OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8000

# 2 workers matches the EC2 t3.medium memory budget (see docker-compose.yml comments).
# Increase only if you move to a larger instance.
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
