FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for building some python packages if needed, and curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements/package config
COPY pyproject.toml ./

# Install python dependencies
RUN pip install --no-cache-dir -e .

# Copy source code, models, and sample data
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY configs/ ./configs/
COPY data/sample/ ./data/sample/
COPY models/ ./models/
COPY data/fairness_reference.parquet ./data/
# Note: we assume models/ and data/fairness_reference.parquet exist and are populated by Phase 4/6.

# Ensure the app can find the leadguard package
ENV PYTHONPATH=/app/src

# Copy start script
COPY start.sh ./
RUN chmod +x start.sh

# HF Spaces exposes port 7860 by default for Docker spaces
EXPOSE 7860
# Internal FastAPI port
EXPOSE 8000

# Start both FastAPI and Streamlit
CMD ["./start.sh"]
