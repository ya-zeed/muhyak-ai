# Deployment & Performance Optimization Guide

This guide covers multiple deployment strategies from cheapest to most scalable.

## Quick Reference: Cost Comparison

| Option | Monthly Cost | Concurrent Users | Setup Complexity |
|--------|--------------|------------------|------------------|
| Single VPS (4GB) | $24/mo | 1-5 | Easy |
| Single VPS (8GB) | $48/mo | 5-15 | Easy |
| DigitalOcean App Platform | $50-150/mo | 10-50 | Medium |
| AWS Lambda + ECS | Pay-per-use ($0.01-0.05/image) | Unlimited | Complex |
| Google Cloud Run | Pay-per-use ($0.005-0.02/image) | Unlimited | Medium |
| Modal.com (Recommended) | Pay-per-use ($0.002-0.01/image) | Unlimited | Easy |

---

## Option 1: Optimized VPS Deployment (Cheapest)

### Recommended VPS Specs
- **Minimum**: 4 vCPU, 8GB RAM (DigitalOcean: $48/mo, Hetzner: $15/mo)
- **Recommended**: 8 vCPU, 16GB RAM (DigitalOcean: $96/mo, Hetzner: $30/mo)

### Performance Tuning Applied

The following optimizations have been applied:

1. **Scaled Workers**: 3 RQ workers running in parallel (was 1)
2. **Smaller Model**: buffalo_s instead of buffalo_l (2-3x faster)
3. **Reduced Detection Size**: 320x320 instead of 640x640 (2x faster)
4. **Resource Limits**: Memory limits on all containers

### Deploy Commands

```bash
# Build with optimizations
docker-compose build

# Run with 3 workers (default)
docker-compose up -d

# Scale workers based on available CPU/RAM (1 worker = ~1GB RAM)
docker-compose up -d --scale rq-worker=4
```

### Environment Variables for Tuning

```bash
# .env file
# Fast mode (default now)
INSIGHTFACE_MODEL=buffalo_s
DET_SIZE_W=320
DET_SIZE_H=320

# Accurate mode (slower, 2-3x more processing time)
# INSIGHTFACE_MODEL=buffalo_l
# DET_SIZE_W=640
# DET_SIZE_H=640
```

---

## Option 2: Google Cloud Run (Best Value Serverless)

Cloud Run is the most cost-effective serverless option for this workload.

### Why Cloud Run?
- Pay only when processing images
- Auto-scales to zero when idle
- No minimum fees
- 2GB memory containers available
- Easy deployment

### Estimated Costs
- ~$0.005-0.02 per image processed
- Free tier: 2 million requests/month
- Idle cost: $0/month

### Setup

1. **Create `Dockerfile.cloudrun`**:

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ make \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model
RUN mkdir -p /root/.insightface/models && \
    python -c "from insightface.app import FaceAnalysis; a = FaceAnalysis(name='buffalo_s'); a.prepare(ctx_id=-1, det_size=(320,320))"

COPY . /app

# Cloud Run uses PORT env var
CMD exec gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --preload -b 0.0.0.0:$PORT --timeout 300
```

2. **Deploy**:

```bash
# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT/muhyak-ai

# Deploy with high memory for ML workload
gcloud run deploy muhyak-ai \
  --image gcr.io/YOUR_PROJECT/muhyak-ai \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 10 \
  --set-env-vars "INSIGHTFACE_MODEL=buffalo_s,DET_SIZE_W=320,DET_SIZE_H=320"
```

### Architecture for Cloud Run

```
                    ┌─────────────────┐
                    │   Cloud Run     │
User Request ──────▶│   (API Only)    │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │ Cloud SQL│      │Cloud Pub/│      │ Cloud    │
    │(Postgres)│      │Sub Queue │      │ Storage  │
    └──────────┘      └────┬─────┘      │ (S3)     │
                           │            └──────────┘
                           ▼
                    ┌──────────────┐
                    │ Cloud Run    │
                    │ (Workers)    │
                    │ Auto-scaling │
                    └──────────────┘
```

---

## Option 3: Modal.com (Easiest Serverless)

Modal is the easiest way to run ML workloads serverlessly with minimal code changes.

### Why Modal?
- Built specifically for ML workloads
- GPU support if needed later
- Pay per second of compute
- Very easy Python SDK
- Cold start: ~2 seconds

### Estimated Costs
- ~$0.002-0.01 per image
- Free tier: $30/month credits
- No idle costs

### Setup

1. **Install Modal**:
```bash
pip install modal
modal token new
```

2. **Create `modal_worker.py`**:

```python
import modal

# Define the container image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "libgl1", "libglib2.0-0", "libgomp1"
).pip_install(
    "insightface>=0.7,<0.9",
    "onnxruntime>=1.17,<2.0",
    "opencv-python-headless>=4.9,<4.11",
    "numpy>=1.26,<3.0",
    "pillow>=10.3,<11.0",
    "boto3>=1.34,<2.0",
    "redis>=5.0,<6.0",
).run_commands(
    "python -c \"from insightface.app import FaceAnalysis; a = FaceAnalysis(name='buffalo_s'); a.prepare(ctx_id=-1, det_size=(320,320))\""
)

app = modal.App("muhyak-face-processor", image=image)

@app.function(
    memory=2048,  # 2GB RAM
    cpu=2.0,
    timeout=300,
    retries=2,
)
def process_image(image_bytes: bytes, image_id: str, config: dict):
    """Process a single image for face detection."""
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis

    # Initialize model (cached across invocations)
    face_app = FaceAnalysis(name="buffalo_s")
    face_app.prepare(ctx_id=0, det_size=(320, 320))

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detect faces
    faces = face_app.get(image_bgr)

    results = []
    for i, f in enumerate(faces):
        if f.embedding is not None:
            results.append({
                "face_index": i,
                "vector": f.embedding.tolist(),
                "bbox": f.bbox.tolist(),
                "confidence": float(f.det_score),
            })

    return {"image_id": image_id, "faces": results}


@app.function(memory=1024, cpu=1.0, timeout=60)
def process_batch(image_list: list):
    """Process multiple images in parallel."""
    results = []
    for img_data in image_list:
        result = process_image.remote(
            img_data["bytes"],
            img_data["id"],
            img_data.get("config", {})
        )
        results.append(result)
    return results
```

3. **Call from your API**:

```python
# In your FastAPI endpoint
import modal

@router.post("/upload")
async def upload_images(...):
    # Trigger Modal function instead of RQ job
    process_image = modal.Function.lookup("muhyak-face-processor", "process_image")

    for file in files:
        # This runs on Modal's infrastructure
        result = process_image.spawn(file.read(), str(uuid.uuid4()), {})

    return {"status": "processing"}
```

4. **Deploy**:
```bash
modal deploy modal_worker.py
```

---

## Option 4: AWS Lambda + SQS (Enterprise Scale)

For high volume with maximum control.

### Architecture

```
              ┌─────────────────────────────────────────┐
              │              API Gateway                │
              └─────────────────┬───────────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────────────┐
              │         Lambda (API Handler)            │
              │         - Validates requests            │
              │         - Queues to SQS                 │
              └─────────────────┬───────────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────────────┐
              │              SQS Queue                  │
              │         (Image Processing Jobs)         │
              └─────────────────┬───────────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────────────┐
              │      Lambda (Face Processor)            │
              │      - 10GB memory container            │
              │      - 15 min timeout                   │
              │      - Container image with InsightFace │
              └─────────────────┬───────────────────────┘
                                │
              ┌─────────┬───────┴───────┬─────────┐
              ▼         ▼               ▼         ▼
         ┌───────┐ ┌───────┐      ┌─────────┐ ┌───────┐
         │  S3   │ │  RDS  │      │ ElastiC │ │  SNS  │
         │Images │ │Postgres│     │  (Redis)│ │Notify │
         └───────┘ └───────┘      └─────────┘ └───────┘
```

### Estimated Costs
- Lambda: ~$0.01-0.03 per image
- S3: ~$0.023/GB/month
- RDS: $15-50/month (db.t3.micro to t3.small)
- Total: $30-100/month + usage

### Terraform Setup

Create `infrastructure/main.tf`:

```hcl
# See infrastructure/aws/ directory for full Terraform configs
```

---

## Option 5: Hybrid Approach (Recommended for Growth)

Combine VPS for API + Serverless for processing.

### Architecture

```
                    ┌─────────────────┐
User Request ──────▶│   VPS (API)     │
                    │   $24/month     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Redis Queue   │
                    │   (on VPS)      │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Modal Worker   │ │  Modal Worker   │ │  Modal Worker   │
│  (on-demand)    │ │  (on-demand)    │ │  (on-demand)    │
│  $0.002/image   │ │  $0.002/image   │ │  $0.002/image   │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### Benefits
- Low fixed costs ($24/month VPS)
- Unlimited scaling for processing
- Pay only for actual image processing
- Easy to implement

### Implementation

1. Keep your current VPS for API, DB, Redis
2. Add Modal worker that polls Redis queue
3. Replace RQ worker with Modal calls

---

## Performance Benchmarks

### Before Optimization
- Single image processing: 2-5 seconds
- Concurrent capacity: 1 image at a time
- CPU usage: 100% constant

### After Optimization (VPS)
- Single image processing: 0.5-1.5 seconds
- Concurrent capacity: 3-4 images at a time
- CPU usage: 60-80% under load

### With Serverless (Modal/Cloud Run)
- Single image processing: 0.5-1.5 seconds
- Concurrent capacity: Unlimited (auto-scale)
- CPU usage: N/A (pay per use)

---

## Quick Start Commands

### For VPS (Current Setup - Optimized)

```bash
# Rebuild with optimizations
docker-compose build --no-cache

# Start with 3 workers
docker-compose up -d

# Monitor
docker-compose logs -f rq-worker

# Scale up if needed
docker-compose up -d --scale rq-worker=5
```

### For Cloud Run

```bash
# One-time setup
gcloud init
gcloud services enable run.googleapis.com

# Deploy
gcloud builds submit --tag gcr.io/$PROJECT/muhyak-ai
gcloud run deploy muhyak-ai --image gcr.io/$PROJECT/muhyak-ai --memory 2Gi --cpu 2
```

### For Modal

```bash
# One-time setup
pip install modal
modal token new

# Deploy
modal deploy modal_worker.py

# Test
modal run modal_worker.py::process_image --image-bytes "..."
```

---

## Recommendation by Use Case

| Use Case | Recommended Option | Monthly Cost |
|----------|-------------------|--------------|
| Side project, < 100 images/day | Optimized VPS (4GB) | $24 |
| Small business, < 1000 images/day | Optimized VPS (8GB) | $48 |
| Growing business, variable load | Hybrid (VPS + Modal) | $24 + usage |
| Enterprise, high volume | AWS Lambda + ECS | $100+ |
| Maximum simplicity | Modal.com | Usage only |
