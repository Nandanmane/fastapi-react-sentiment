https://github.com/Nandanmane/fastapi-react-sentiment/releases

# FastAPI React Sentiment Analysis — DistilBERT Full-Stack App 🚀

[![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github)](https://github.com/Nandanmane/fastapi-react-sentiment/releases)  
[![distilBERT](https://img.shields.io/badge/Model-DistilBERT-orange?logo=transformers)](https://huggingface.co/distilbert-base-uncased) [![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal?logo=fastapi)](https://fastapi.tiangolo.com) [![React](https://img.shields.io/badge/Frontend-React-blue?logo=react)](https://reactjs.org)  
[![Python](https://img.shields.io/badge/Lang-Python-3776AB?logo=python)](https://www.python.org) [![PyTorch](https://img.shields.io/badge/Runtime-PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org) [![HuggingFace](https://img.shields.io/badge/HF-Transformers-yellow?logo=huggingface)](https://huggingface.co)

![Architecture diagram](https://raw.githubusercontent.com/Nandanmane/fastapi-react-sentiment/main/docs/assets/architecture.png)

A professional full-stack sentiment analysis app. It uses DistilBERT for inference, a FastAPI backend for model serving, and a modular React frontend for interactive analysis. The app supports GPU acceleration, environment-aware config, and a clean component architecture.

Built for real-time text analysis, production readiness, and easy customization.

## Table of contents

- Features
- Tech stack
- Architecture
- Quick start
- Environment variables
- GPU and performance
- API reference
- Frontend structure
- Deployment
- Docker
- Testing
- Contributing
- License
- Releases

## Features

- Real-time sentiment scoring (positive, neutral, negative).
- Transformer inference powered by DistilBERT via Hugging Face Transformers.
- FastAPI REST API with async endpoints and pydantic schemas.
- Modular React frontend with component-driven UI.
- GPU support (CUDA) with fallback to CPU.
- Environment-aware config (development, staging, production).
- Health checks, logging, and metrics endpoints.
- Optional batch inference mode for throughput.
- Docker-ready images and example Kubernetes manifests.

## Tech stack

- Model: DistilBERT (Hugging Face transformers)
- Backend: FastAPI, uvicorn, pydantic, PyTorch
- Frontend: React, Vite, TypeScript (optional)
- Dev tooling: Docker, Poetry or pip, GitHub Actions
- Monitoring: Prometheus metrics endpoint (optional)

## Architecture

1. Client (React)
   - Sends text to /api/v1/analyze.
   - Shows score, label, and token-level visualization.
2. Backend (FastAPI)
   - Loads DistilBERT tokenizer and model.
   - Runs inference on GPU or CPU.
   - Returns structured JSON with scores and metadata.
3. Model layer
   - Single model instance per process.
   - Async request queue with batching option.
4. Infra
   - Optional reverse proxy (NGINX) and container orchestration.

## Quick start

Prerequisites
- Python 3.10+
- Node 16+
- GPU + CUDA if you plan to use GPU
- Git

Clone the repo
```bash
git clone https://github.com/Nandanmane/fastapi-react-sentiment.git
cd fastapi-react-sentiment
```

Backend setup (venv + pip)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Frontend setup
```bash
cd frontend
npm install
npm run dev
```

Start backend (development)
```bash
cd ../backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open the frontend at http://localhost:5173 and the API docs at http://localhost:8000/docs

## Environment variables

The app reads configuration from environment variables. Use .env files for each environment.

Common variables
- APP_ENV=development|staging|production
- MODEL_NAME=distilbert-base-uncased
- BATCH_ENABLED=true|false
- BATCH_SIZE=8
- CUDA_VISIBLE_DEVICES=0
- LOG_LEVEL=info
- HOST=0.0.0.0
- PORT=8000

Example .env
```env
APP_ENV=development
MODEL_NAME=distilbert-base-uncased
BATCH_ENABLED=false
BATCH_SIZE=8
LOG_LEVEL=debug
```

## GPU and performance

- The code checks torch.cuda.is_available() and moves the model to GPU when available.
- For high throughput, enable batching and set BATCH_SIZE according to GPU memory.
- Use torch.backends.cudnn.benchmark = True on fixed-size inputs to improve perf.
- Profile with torch.profiler or your GPU monitoring tools.

Performance tips
- Use half precision (fp16) with torch.cuda.amp for faster inference on supported GPUs.
- Use multiple worker processes in uvicorn/gunicorn and a shared model via TorchServe or gunicorn + uvicorn worker class for production.

## API reference

Base URL: /api/v1

POST /analyze
- Input: JSON { "text": "some sentence" }
- Output:
  {
    "label": "positive",
    "score": 0.9723,
    "scores": {"positive":0.9723,"neutral":0.015,"negative":0.012},
    "tokens": [{"token":"I","score":0.1}, ...],
    "metadata": { "model": "distilbert-base-uncased", "device": "cuda:0" }
  }

Example curl
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text":"I like this product. It works well."}'
```

GET /health
- Returns service status and model load status.

GET /metrics
- Exposes Prometheus metrics when enabled.

Swagger UI
- Visit /docs for interactive API docs (OpenAPI).

## Frontend structure

- src/
  - components/
    - SentimentForm.tsx
    - ResultCard.tsx
    - TokenViz.tsx
  - services/
    - api.ts (fetch wrapper)
  - pages/
    - Home.tsx
    - About.tsx
  - hooks/
    - useSentiment.ts
  - App.tsx

Patterns
- Keep API calls in services.
- Use hooks for state and side effects.
- Present token-level scores in a dedicated TokenViz component.

UI notes
- The frontend shows overall sentiment, probabilities, and token heatmap.
- Add additional visual features by extending TokenViz.

## Model and inference details

- Tokenizer: AutoTokenizer.from_pretrained(MODEL_NAME)
- Model: AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
- The backend wraps the model with a simple inference function that returns label and softmax scores.
- For token-level attribution, the project includes a simple integrated gradients utility. Extend or replace with Captum for more advanced explainability.

Example inference pseudocode
```py
inputs = tokenizer(text, return_tensors="pt", truncation=True)
with torch.no_grad():
    outputs = model(**inputs.to(device))
probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

## Deployment

Production server
- Use gunicorn + uvicorn workers or deploy via ASGI server of choice.
- Configure logging to write structured JSON to stdout for aggregation.
- Use a load balancer and autoscaling when needed.

Kubernetes
- Provide a Deployment and Service manifest in k8s/.
- Set resource limits and request GPU via device plugin when using CUDA.

CI/CD
- Include GitHub Actions for linting, tests, and build artifacts.
- Build Docker images in CI and push to a registry.

## Docker

Backend Dockerfile example
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run
```bash
docker build -t fastapi-sentiment:latest -f backend/Dockerfile .
docker run -p 8000:8000 fastapi-sentiment:latest
```

If you plan to use GPU, use nvidia runtime and a base image with CUDA libs.

## Testing

- Unit tests live in backend/tests and frontend/__tests__.
- Run backend tests
```bash
cd backend
pytest -q
```
- Run frontend tests
```bash
cd frontend
npm test
```

## Contributing

- Open an issue for new features or bugs.
- Fork the repo, create a feature branch, and submit a pull request.
- Keep commits small and focused.
- Add tests for new features.

## License

This repository uses the MIT License. See LICENSE file for details.

## Releases

Download the latest release and run the packaged installer or startup script. The release page contains artifacts for quick deployment. Download the release asset and execute the provided run script or installer to start a pre-built bundle.

Open releases: https://github.com/Nandanmane/fastapi-react-sentiment/releases

Common release files
- fastapi-react-sentiment-vX.Y.tar.gz — source bundle
- run.sh or start.sh — launcher script (execute after download)
- docker-image.tar — prebuilt image (load with docker load)

Example (after download)
```bash
tar -xzf fastapi-react-sentiment-v1.0.tar.gz
cd fastapi-react-sentiment
chmod +x run.sh
./run.sh
```

If a release link does not work, check the repository Releases section on GitHub for artifacts and instructions.

## Useful links and assets

- FastAPI docs: https://fastapi.tiangolo.com
- Hugging Face transformers: https://huggingface.co/docs/transformers
- PyTorch: https://pytorch.org
- React: https://reactjs.org

Logos and images used above
- FastAPI: https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png
- Hugging Face: https://huggingface.co/front/assets/huggingface_logo.svg
- PyTorch: https://pytorch.org/assets/images/pytorch-logo.png
- React: https://reactjs.org/logo-og.png

## Contact

Open issues or PRs in the repository to report bugs or propose changes.