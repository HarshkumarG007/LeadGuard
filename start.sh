#!/bin/bash
set -e

echo "Starting FastAPI on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

echo "Waiting for FastAPI to become healthy..."
until curl -s http://localhost:8000/v1/health | grep -q '"status"'; do
  sleep 1
done
echo "FastAPI is up!"

echo "Starting Streamlit on port 7860 (Hugging Face Spaces default)..."
export API_URL="http://localhost:8000"
streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0

wait $FASTAPI_PID
