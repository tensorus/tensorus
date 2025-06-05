FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (Node.js & npm for MCP server install)
RUN apt-get update && \
    apt-get install -y nodejs npm && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python and Node.js dependencies
RUN chmod +x ./setup.sh && ./setup.sh

# Expose FastAPI port
EXPOSE 8000

# Start the API server
CMD ["python", "-m", "uvicorn", "tensorus.api:app", "--host", "0.0.0.0", "--port", "8000"]
