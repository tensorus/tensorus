# Tensorus API Authentication Guide

Tensorus implements industry-standard Bearer token authentication following patterns from OpenAI, Pinecone, and other major AI/ML services.

## Quick Start

### 1. Generate an API Key

```bash
# Generate a single API key
python generate_api_key.py

# Generate multiple keys
python generate_api_key.py --count 3

# Generate in environment variable format
python generate_api_key.py --format env
```

### 2. Configure Environment

```bash
# Set your API key
export TENSORUS_AUTH_ENABLED=true
export TENSORUS_API_KEYS=tsr_your_generated_key_here

# Or create a .env file
cp .env.example .env
# Edit .env with your API key
```

### 3. Use the API

```bash
# Using cURL
curl -H "Authorization: Bearer tsr_your_key" \
     http://localhost:7860/datasets

# Using Python requests
import requests
headers = {"Authorization": "Bearer tsr_your_key"}
response = requests.get("http://localhost:7860/datasets", headers=headers)
```

## Authentication Methods

### Bearer Token (Recommended)

Use the standard `Authorization: Bearer <token>` header:

```http
GET /datasets HTTP/1.1
Host: localhost:7860
Authorization: Bearer tsr_abc123...
```

### Legacy Header (Backward Compatibility)

The legacy `X-API-KEY` header is also supported:

```http
GET /datasets HTTP/1.1
Host: localhost:7860
X-API-KEY: tsr_abc123...
```

## API Key Format

Tensorus API keys follow this format:
- **Prefix**: `tsr_` (Tensorus identifier)
- **Length**: 48 characters (excluding prefix)
- **Characters**: URL-safe Base64 (a-z, A-Z, 0-9, -, _)
- **Example**: `tsr_abc123XYZ789-_def456GHI012jkl345MNO678pqr901STU234`

### Key Validation

```python
from tensorus.auth.key_generator import TensorusAPIKey

# Validate format
is_valid = TensorusAPIKey.validate_format("tsr_your_key")

# Mask for logging
masked = TensorusAPIKey.mask_key("tsr_your_key")  # "tsr_abc...234"
```

## Configuration Options

### Environment Variables

```bash
# Core authentication
TENSORUS_AUTH_ENABLED=true                     # Enable/disable auth
TENSORUS_API_KEYS=key1,key2,key3               # Comma-separated keys

# Legacy support
TENSORUS_VALID_API_KEYS=key1,key2              # Legacy format
TENSORUS_API_KEY_HEADER_NAME=Authorization     # Header name
```

### Multiple API Keys

Support multiple keys for different environments or users:

```bash
# Multiple keys (comma-separated)
TENSORUS_API_KEYS=tsr_prod_key...,tsr_dev_key...,tsr_test_key...
```

### Disable Authentication

For development or testing:

```bash
TENSORUS_AUTH_ENABLED=false
```

## Client Examples

### Python

```python
import httpx

# Async client
async with httpx.AsyncClient() as client:
    headers = {"Authorization": "Bearer tsr_your_key"}
    response = await client.get("http://localhost:7860/datasets", headers=headers)
    datasets = response.json()

# Sync client  
import requests
headers = {"Authorization": "Bearer tsr_your_key"}
response = requests.get("http://localhost:7860/datasets", headers=headers)
```

### JavaScript/Node.js

```javascript
// Using fetch
const response = await fetch('http://localhost:7860/datasets', {
  headers: {
    'Authorization': 'Bearer tsr_your_key',
    'Content-Type': 'application/json'
  }
});

// Using axios
const axios = require('axios');
const response = await axios.get('http://localhost:7860/datasets', {
  headers: { 'Authorization': 'Bearer tsr_your_key' }
});
```

### cURL

```bash
# GET request
curl -H "Authorization: Bearer tsr_your_key" \
     http://localhost:7860/datasets

# POST request
curl -X POST \
     -H "Authorization: Bearer tsr_your_key" \
     -H "Content-Type: application/json" \
     -d '{"name": "my_dataset"}' \
     http://localhost:7860/datasets/create
```

## MCP Server Integration

### Start with API Key

```bash
# Local backend
python -m tensorus.mcp_server \
    --transport streamable-http \
    --mcp-api-key tsr_your_key

# Remote backend
python -m tensorus.mcp_server \
    --transport streamable-http \
    --api-url https://your-tensorus-api.com \
    --mcp-api-key tsr_your_key
```

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "tensorus": {
      "command": "python",
      "args": [
        "-m", "tensorus.mcp_server",
        "--transport", "stdio",
        "--mcp-api-key", "tsr_your_key"
      ],
      "env": {
        "TENSORUS_API_BASE_URL": "https://your-tensorus-api.com"
      }
    }
  }
}
```

## Security Best Practices

### Key Management

1. **Generate Strong Keys**: Always use the provided generator
2. **Unique Keys**: Use different keys for different environments
3. **Rotation**: Regularly rotate API keys
4. **Storage**: Store keys securely, never commit to version control

### Environment Separation

```bash
# Development
TENSORUS_API_KEYS=tsr_dev_key_...

# Staging  
TENSORUS_API_KEYS=tsr_staging_key_...

# Production
TENSORUS_API_KEYS=tsr_prod_key_...
```

### Monitoring

API authentication events are logged for security monitoring:

```bash
# Check audit logs
tail -f tensorus_audit.log | grep API_AUTH
```

## Troubleshooting

### Common Issues

#### 401 Unauthorized

```json
{"detail": "Missing API key. Use 'Authorization: Bearer <api_key>' header."}
```

**Solution**: Add proper Authorization header

#### 401 Invalid API Key

```json
{"detail": "Invalid API key"}
```

**Solutions**:
- Check key format (should start with `tsr_`)
- Verify key is in `TENSORUS_API_KEYS`
- Ensure authentication is enabled

#### 503 Service Unavailable

```json
{"detail": "API authentication not configured"}
```

**Solution**: Set `TENSORUS_API_KEYS` environment variable

### Debug Authentication

```python
# Check configuration
from tensorus.config import settings
print("Auth enabled:", settings.AUTH_ENABLED)
print("Valid keys:", len(settings.valid_api_keys))

# Test key format
from tensorus.auth.key_generator import TensorusAPIKey
key = "tsr_your_key"
print("Valid format:", TensorusAPIKey.validate_format(key))
```

### Test Authentication

```bash
# Test with valid key
curl -H "Authorization: Bearer tsr_your_key" \
     http://localhost:7860/datasets

# Test without key (should fail)
curl http://localhost:7860/datasets
```

## Migration from Legacy

### Updating Existing Code

```python
# Before (legacy header)
headers = {"X-API-KEY": "your_key"}

# After (Bearer token)
headers = {"Authorization": "Bearer tsr_your_key"}
```

### Backward Compatibility

The system supports both methods during migration:

```python
# Both work simultaneously
headers1 = {"Authorization": "Bearer tsr_your_key"}      # New method
headers2 = {"X-API-KEY": "tsr_your_key"}               # Legacy method
```

## Production Deployment

### Secure Key Storage

```bash
# Use secure environment management
# HashiCorp Vault
vault kv put secret/tensorus api_key=tsr_prod_key_...

# AWS Secrets Manager
aws secretsmanager create-secret \
    --name tensorus/api-key \
    --secret-string tsr_prod_key_...

# Docker secrets
echo "tsr_prod_key_..." | docker secret create tensorus_api_key -
```

### Docker Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  tensorus:
    image: tensorus:latest
    environment:
      TENSORUS_AUTH_ENABLED: "true"
      TENSORUS_API_KEYS: "${TENSORUS_API_KEY}"
    secrets:
      - tensorus_api_key
```

### Health Checks

```bash
# Verify authentication is working
curl -f -H "Authorization: Bearer tsr_your_key" \
     http://localhost:7860/health || exit 1
```

This authentication system provides enterprise-grade security while maintaining the simplicity and developer-friendliness that Tensorus is known for.