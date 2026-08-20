# LeadGuard Deployment Guide

This repository is designed for easy, zero-infrastructure deployment. The reference architecture assumes deployment as a Hugging Face Space (Docker), but the container can run anywhere.

## Hugging Face Spaces (Free Tier)

We deploy LeadGuard using Hugging Face Spaces on their CPU Basic (Free) tier. The free tier gives us 2 vCPU and 16GB RAM, which is more than enough for our `<1ms` XGBoost inference latency.

### Multi-Process Container
Hugging Face Spaces exposes a single port (`7860`). Because we serve both a FastAPI backend and a Streamlit frontend, we use a single `Dockerfile` and a `start.sh` script to run both processes:
1. `uvicorn` runs FastAPI on port `8000` in the background.
2. `streamlit` runs on port `7860` in the foreground.
3. Streamlit connects to `http://localhost:8000` via the `API_URL` environment variable.

### Cold-Start Behavior
To save costs, HF Spaces puts instances to sleep after 48 hours of inactivity.
- **Wake Time:** Because our application is strictly offline and contains all models and data internally, cold-start wake time is fast (usually ~10-20 seconds). 
- **Offline Guarantee:** No API keys or remote fetches are required at runtime. The Docker image is 100% self-contained.

## Manual Deployment

You can run the container locally or on any server (e.g. AWS EC2, DigitalOcean, Azure):

```bash
docker build -t leadguard-demo .
# Map port 7860 to host port 80 (or any other)
docker run -p 80:7860 -d leadguard-demo
```

The Streamlit UI will be available at `http://localhost`, and the API will be accessible *internally* to Streamlit. (If you want external access to the API, you must expose port 8000 as well: `-p 8000:8000`).
