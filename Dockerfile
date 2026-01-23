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

# Install Modal for serverless backend support
RUN pip install --no-cache-dir modal>=0.64,<1.0

# ---- Pre-download InsightFace models into the image ----
# Download both models so you can switch via INSIGHTFACE_MODEL env var
# buffalo_s: Fast model (~2-3x faster, good for most use cases)
# buffalo_l: Accurate model (slower, better for high-quality results)
RUN mkdir -p /root/.insightface/models && \
    python - <<'PY'
from insightface.app import FaceAnalysis
# Download buffalo_s (fast model - default)
a = FaceAnalysis(name="buffalo_s")
a.prepare(ctx_id=-1, det_size=(320,320))
print("✅ InsightFace buffalo_s model cached.")
# Download buffalo_l (accurate model - optional)
b = FaceAnalysis(name="buffalo_l")
b.prepare(ctx_id=-1, det_size=(640,640))
print("✅ InsightFace buffalo_l model cached.")
PY

# Copy code last (better cache)
COPY . /app

EXPOSE 8000

# Preload ensures init happens once in master, then forks
CMD ["gunicorn","main:app","-w","2","--threads", "2","-k","uvicorn.workers.UvicornWorker","--preload","-b","0.0.0.0:8000","--timeout","120"]
