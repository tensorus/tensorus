FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for FAISS and HDF5
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data config

# Set environment variables
ENV TENSORUS_STORAGE_PATH=/app/data/tensor_db.h5
ENV TENSORUS_INDEX_PATH=/app/data/tensor_index.pkl
ENV TENSORUS_CONFIG_PATH=/app/config/db_config.json
ENV TENSORUS_USE_GPU=false
ENV PORT=5000
ENV DEBUG=false

# Expose the API port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"] 