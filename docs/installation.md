# Installation and Configuration

## Installation

### Using Docker (Recommended for Local Development/Testing)

The easiest way to get Tensorus and its PostgreSQL backend running locally is by using Docker Compose:

1.  Ensure you have Docker and Docker Compose installed.
2.  Clone the repository (if applicable) or ensure you have the `docker-compose.yml` and application files.
3.  From the project root, run:
    ```bash
    docker-compose up --build
    ```
4.  The API will typically be available at `http://localhost:8000`.

### Manual Installation (From Source)

1.  Ensure you have Python 3.9+ installed.
2.  Clone the repository.
3.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5.  Set up the necessary environment variables (see Configuration below).
6.  Run the application using Uvicorn:
    ```bash
    uvicorn tensorus.api.main:app --host 0.0.0.0 --port 8000 --reload # for development
    ```

## Configuration

Tensorus is configured via environment variables. Key variables include:

*   `TENSORUS_STORAGE_BACKEND`: Specifies the storage backend.
    *   `in_memory` (default): Uses in-memory storage (data is not persisted).
    *   `postgres`: Uses PostgreSQL.
*   `TENSORUS_POSTGRES_HOST`: Hostname for PostgreSQL (e.g., `localhost` or `db` if using Docker Compose).
*   `TENSORUS_POSTGRES_PORT`: Port for PostgreSQL (e.g., `5432`).
*   `TENSORUS_POSTGRES_USER`: Username for PostgreSQL.
*   `TENSORUS_POSTGRES_PASSWORD`: Password for PostgreSQL.
*   `TENSORUS_POSTGRES_DB`: Database name for PostgreSQL.
*   `TENSORUS_POSTGRES_DSN`: Alternative DSN connection string for PostgreSQL.
*   `TENSORUS_VALID_API_KEYS`: Comma-separated list of valid API keys for write operations (e.g., `key1,key2,anotherkey`). If empty or not set, write operations requiring API keys will be inaccessible.
*   `TENSORUS_API_KEY_HEADER_NAME`: HTTP header name for the API key (default: `X-API-KEY`).

### JWT Authentication (Conceptual - For Future Use)
*   `TENSORUS_AUTH_JWT_ENABLED`: `True` or `False` (default `False`).
*   `TENSORUS_AUTH_JWT_ISSUER`: URL of the JWT issuer.
*   `TENSORUS_AUTH_JWT_AUDIENCE`: Expected audience for JWTs.
*   `TENSORUS_AUTH_JWT_ALGORITHM`: Algorithm (default `RS256`).
*   `TENSORUS_AUTH_JWT_JWKS_URI`: URI to fetch JWKS.
*   `TENSORUS_AUTH_DEV_MODE_ALLOW_DUMMY_JWT`: `True` to allow dummy JWTs for development if JWT auth is enabled (default `False`).

Refer to the `docker-compose.yml` for example environment variable settings when running with Docker. For manual setup, export these variables in your shell or use a `.env` file (if your setup supports it, though direct environment variables are primary).
