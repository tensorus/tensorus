# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by Python packages
# Example: build-essential for some packages, libpq-dev for psycopg2 from source (though -binary avoids this)
# For psycopg2-binary, typically no extra system deps are needed for common platforms if using a compatible wheel.
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code into the container
# This structure assumes your main application code is within the 'tensorus' directory.
COPY ./tensorus ./tensorus
# If your app.py or main.py is at the root alongside Dockerfile, you'd copy it too:
# COPY app.py . # Or specific main file if it's at the root.
# Based on `CMD ["uvicorn", "tensorus.api.main:app"...]`, main:app is in tensorus/api/main.py.

# Set default environment variables for the application
# These can be overridden when running the container (e.g., via docker run -e or docker-compose.yml)
ENV TENSORUS_STORAGE_BACKEND="in_memory"
ENV TENSORUS_POSTGRES_HOST="db" # Default for docker-compose setup
ENV TENSORUS_POSTGRES_PORT="5432"
ENV TENSORUS_POSTGRES_USER="tensorus_user_dockerfile"
ENV TENSORUS_POSTGRES_PASSWORD="tensorus_pass_dockerfile"
ENV TENSORUS_POSTGRES_DB="tensorus_db_dockerfile"
ENV TENSORUS_POSTGRES_DSN=""
ENV TENSORUS_API_KEY_HEADER_NAME="X-API-KEY"
ENV TENSORUS_VALID_API_KEYS="" # Example: "key1,key2" - Must be set at runtime for security
ENV TENSORUS_AUTH_JWT_ENABLED="False" # Default JWT auth to disabled
ENV TENSORUS_AUTH_JWT_ISSUER=""
ENV TENSORUS_AUTH_JWT_AUDIENCE=""
ENV TENSORUS_AUTH_JWT_ALGORITHM="RS256"
ENV TENSORUS_AUTH_JWT_JWKS_URI=""
ENV TENSORUS_AUTH_DEV_MODE_ALLOW_DUMMY_JWT="False"

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# This assumes your FastAPI app instance is named 'app' in 'tensorus.api.main'
CMD ["uvicorn", "tensorus.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
