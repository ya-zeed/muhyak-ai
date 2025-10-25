FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build + runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ make \
    libgl1 libglib2.0-0 libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---- Pre-download InsightFace models into the image ----
# (uses default cache: /root/.insightface/models)
RUN mkdir -p /root/.insightface/models && \
    python - <<'PY'
from insightface.app import FaceAnalysis
a = FaceAnalysis(name="buffalo_l")
a.prepare(ctx_id=-1, det_size=(640,640))
print("✅ InsightFace model cached.")
PY

# Copy code last (better cache)
COPY . /app

EXPOSE 8000

# Preload ensures init happens once in master, then forks
CMD ["gunicorn","main:app","-w","1","-k","uvicorn.workers.UvicornWorker","--preload","-b","0.0.0.0:8000","--timeout","120"]
