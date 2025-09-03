# Tensorus Production Deployment Guide

## Overview

This guide covers deploying Tensorus in production environments with high availability, scalability, and security best practices. Choose the deployment option that best fits your infrastructure requirements and organizational policies.

## Deployment Architecture Options

### 1. Cloud-Native (Recommended)

**Kubernetes-based deployment** with auto-scaling and managed services integration.

```yaml
# tensorus-production.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: tensorus-prod

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tensorus-api
  namespace: tensorus-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tensorus-api
  template:
    metadata:
      labels:
        app: tensorus-api
    spec:
      containers:
      - name: tensorus
        image: tensorus/tensorus:v1.2.3
        ports:
        - containerPort: 8000
        env:
        - name: TENSORUS_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: tensorus-secrets
              key: database-url
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi" 
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 2. Docker Compose (Small-Medium Scale)

**Containerized deployment** for single-node or small cluster setups.

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  tensorus-api:
    image: tensorus/tensorus:v1.2.3
    ports:
      - "8000:8000"
    environment:
      - TENSORUS_ENV=production
      - DATABASE_URL=postgresql://user:pass@postgres:5432/tensorus
      - REDIS_URL=redis://redis:6379
      - STORAGE_BACKEND=s3
      - AWS_S3_BUCKET=tensorus-storage-prod
    volumes:
      - ./config/production.env:/app/.env
      - tensorus-data:/app/data
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=tensorus
      - POSTGRES_USER=tensorus
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - tensorus-api
    restart: unless-stopped

volumes:
  tensorus-data:
  postgres-data:
  redis-data:
```

### 3. Bare Metal (High Performance)

**Direct installation** on dedicated servers for maximum performance.

```bash
#!/bin/bash
# install-production.sh

# System preparation
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv postgresql redis-server nginx

# Create tensorus user
sudo useradd -m -s /bin/bash tensorus
sudo mkdir -p /opt/tensorus /var/log/tensorus /var/lib/tensorus
sudo chown -R tensorus:tensorus /opt/tensorus /var/log/tensorus /var/lib/tensorus

# Install Tensorus
sudo -u tensorus python3.10 -m venv /opt/tensorus/venv
sudo -u tensorus /opt/tensorus/venv/bin/pip install tensorus[production]==1.2.3

# Configure systemd service
sudo tee /etc/systemd/system/tensorus.service > /dev/null <<EOF
[Unit]
Description=Tensorus Tensor Database
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=tensorus
WorkingDirectory=/opt/tensorus
Environment=PATH=/opt/tensorus/venv/bin
ExecStart=/opt/tensorus/venv/bin/uvicorn tensorus.api:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable tensorus
sudo systemctl start tensorus
```

## Infrastructure Requirements

### Minimum Production Requirements

| Component | Specification |
|-----------|---------------|
| **CPU** | 8 cores (16 threads) |
| **Memory** | 32 GB RAM |
| **Storage** | 1 TB SSD (data) + 500 GB SSD (OS) |
| **Network** | 10 Gbps connection |
| **OS** | Ubuntu 22.04 LTS / RHEL 8+ / Amazon Linux 2 |

### Recommended High-Scale Configuration

| Component | Specification |
|-----------|---------------|
| **CPU** | 32 cores (64 threads) |
| **Memory** | 128 GB RAM |
| **Storage** | 10 TB NVMe SSD (data) + 1 TB SSD (OS) |
| **Network** | 25+ Gbps connection |
| **GPU** | NVIDIA A100/V100 (optional for acceleration) |

### Cloud Instance Recommendations

| Cloud Provider | Instance Type | Use Case |
|----------------|---------------|----------|
| **AWS** | c6i.8xlarge | CPU-optimized workloads |
| **AWS** | r6i.8xlarge | Memory-intensive operations |
| **AWS** | p4d.24xlarge | GPU-accelerated computing |
| **Azure** | F32s_v2 | CPU-optimized |
| **Azure** | E32s_v4 | Memory-optimized |
| **GCP** | c2-standard-60 | CPU-optimized |
| **GCP** | n2-highmem-32 | Memory-optimized |

## Configuration Management

### Production Configuration File

```yaml
# config/production.yml
environment: production

# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  worker_class: "uvicorn.workers.UvicornWorker"
  max_connections: 1000
  keepalive: 2

# Database Configuration  
database:
  url: "postgresql://tensorus:${DB_PASSWORD}@localhost:5432/tensorus_prod"
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600

# Redis Configuration
redis:
  url: "redis://localhost:6379/0"
  max_connections: 100
  socket_timeout: 5
  socket_connect_timeout: 5

# Storage Configuration
storage:
  backend: "s3"  # Options: local, s3, gcs, azure
  compression_default: "balanced"
  chunk_size_mb: 64
  cache_size_gb: 10
  
  # S3 Configuration
  s3:
    bucket: "tensorus-production-data"
    region: "us-west-2"
    endpoint_url: null  # Use for S3-compatible services
    
# Security Configuration
security:
  api_key_validation: true
  rate_limiting: true
  cors_enabled: false
  allowed_origins: []
  
# Monitoring Configuration
monitoring:
  metrics_enabled: true
  metrics_port: 9090
  health_check_enabled: true
  log_level: "INFO"
  
# Performance Configuration
performance:
  enable_gpu: false
  gpu_memory_fraction: 0.8
  max_tensor_size_gb: 10
  operation_timeout_seconds: 300
  async_job_timeout_hours: 24
```

### Environment Variables

```bash
# .env.production
TENSORUS_ENV=production
TENSORUS_CONFIG_FILE=/opt/tensorus/config/production.yml

# Database
DATABASE_URL=postgresql://tensorus:password@localhost:5432/tensorus_prod
DB_PASSWORD=your-secure-password

# Redis
REDIS_URL=redis://localhost:6379/0

# AWS (if using S3 storage)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-west-2

# Security
TENSORUS_SECRET_KEY=your-secret-key-here
TENSORUS_API_KEYS=key1,key2,key3

# Monitoring
PROMETHEUS_ENABLED=true
LOG_LEVEL=INFO
```

## Database Setup

### PostgreSQL Configuration

```sql
-- Create production database and user
CREATE DATABASE tensorus_prod;
CREATE USER tensorus WITH ENCRYPTED PASSWORD 'your-secure-password';
GRANT ALL PRIVILEGES ON DATABASE tensorus_prod TO tensorus;

-- Optimize PostgreSQL for Tensorus
ALTER SYSTEM SET shared_buffers = '8GB';
ALTER SYSTEM SET effective_cache_size = '24GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET min_wal_size = '2GB';
ALTER SYSTEM SET max_wal_size = '8GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET max_connections = 200;

SELECT pg_reload_conf();
```

### Database Migration

```bash
# Initialize database schema
tensorus-admin db upgrade --environment production

# Verify migration
tensorus-admin db status --environment production
```

## Load Balancer Configuration

### Nginx Configuration

```nginx
# /etc/nginx/sites-available/tensorus
upstream tensorus_backend {
    least_conn;
    server 127.0.0.1:8001 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8002 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8003 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8004 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.tensorus.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.tensorus.yourdomain.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/tensorus.crt;
    ssl_certificate_key /etc/ssl/private/tensorus.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    # Performance Settings
    client_max_body_size 100M;
    client_body_timeout 60s;
    client_header_timeout 60s;
    
    # Compression
    gzip on;
    gzip_types text/plain application/json application/x-javascript text/css application/xml;
    
    location / {
        proxy_pass http://tensorus_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
        
        # Buffering
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }
    
    # Health check endpoint (no auth required)
    location /health {
        proxy_pass http://tensorus_backend;
        access_log off;
    }
}
```

### HAProxy Alternative

```
# /etc/haproxy/haproxy.cfg
global
    daemon
    log 127.0.0.1:514 local0
    maxconn 4096
    
defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog
    
frontend tensorus_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/tensorus.pem
    redirect scheme https if !{ ssl_fc }
    
    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request reject if { sc_http_req_rate(0) gt 20 }
    
    default_backend tensorus_backend
    
backend tensorus_backend
    balance roundrobin
    option httpchk GET /health
    server api1 127.0.0.1:8001 check
    server api2 127.0.0.1:8002 check
    server api3 127.0.0.1:8003 check
    server api4 127.0.0.1:8004 check
```

## Security Hardening

### System Security

```bash
#!/bin/bash
# security-hardening.sh

# Firewall configuration
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# System updates
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y fail2ban unattended-upgrades

# Configure automatic security updates
echo 'Unattended-Upgrade::Automatic-Reboot "true";' | sudo tee -a /etc/apt/apt.conf.d/20auto-upgrades

# Disable unnecessary services
sudo systemctl disable apache2 2>/dev/null || true
sudo systemctl stop apache2 2>/dev/null || true

# Configure fail2ban for Tensorus
sudo tee /etc/fail2ban/jail.local > /dev/null <<EOF
[tensorus]
enabled = true
port = 8000
protocol = tcp
filter = tensorus
logpath = /var/log/tensorus/access.log
maxretry = 5
bantime = 3600
findtime = 600
EOF
```

### Application Security

```python
# tensorus/security/production.py
SECURITY_CONFIG = {
    # API Security
    'api_key_rotation_days': 90,
    'session_timeout_minutes': 30,
    'max_request_size_mb': 100,
    'rate_limit_per_minute': 1000,
    
    # Data Security
    'encryption_at_rest': True,
    'encryption_in_transit': True,
    'data_retention_days': 365,
    'audit_log_retention_days': 2555,  # 7 years
    
    # Network Security
    'allowed_ip_ranges': ['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'],
    'cors_origins': [],
    'csrf_protection': True,
    
    # Authentication
    'require_api_key': True,
    'api_key_min_length': 32,
    'failed_login_lockout_minutes': 15,
    'password_policy': {
        'min_length': 12,
        'require_uppercase': True,
        'require_lowercase': True,
        'require_numbers': True,
        'require_symbols': True
    }
}
```

## Monitoring & Observability

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'tensorus'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 5s
    metrics_path: /metrics
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']
      
  - job_name: 'nginx'
    static_configs:
      - targets: ['localhost:9113']

rule_files:
  - "tensorus_alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Tensorus Production Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(tensorus_http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(tensorus_http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(tensorus_http_requests_total{status=~\"4..|5..\"}[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      }
    ]
  }
}
```

### Custom Alerts

```yaml
# tensorus_alerts.yml
groups:
  - name: tensorus.rules
    rules:
      - alert: TensorusHighErrorRate
        expr: rate(tensorus_http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate in Tensorus API"
          description: "Error rate is {{ $value }} requests per second"
          
      - alert: TensorusHighResponseTime
        expr: histogram_quantile(0.95, rate(tensorus_http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time in Tensorus API"
          description: "95th percentile response time is {{ $value }} seconds"
          
      - alert: TensorusStorageSpaceHigh
        expr: tensorus_storage_usage_percent > 80
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Tensorus storage usage high"
          description: "Storage usage is {{ $value }}%"
```

## Backup & Recovery

### Automated Backup Script

```bash
#!/bin/bash
# backup-tensorus.sh

# Configuration
BACKUP_DIR="/opt/backups/tensorus"
S3_BUCKET="tensorus-backups-prod"
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Database backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_BACKUP="$BACKUP_DIR/tensorus_db_$TIMESTAMP.sql"

pg_dump -h localhost -U tensorus -d tensorus_prod > "$DB_BACKUP"
gzip "$DB_BACKUP"

# Storage backup (if using local storage)
STORAGE_BACKUP="$BACKUP_DIR/tensorus_storage_$TIMESTAMP.tar.gz"
tar -czf "$STORAGE_BACKUP" /var/lib/tensorus/storage/

# Configuration backup
CONFIG_BACKUP="$BACKUP_DIR/tensorus_config_$TIMESTAMP.tar.gz"
tar -czf "$CONFIG_BACKUP" /opt/tensorus/config/

# Upload to S3
aws s3 sync "$BACKUP_DIR" "s3://$S3_BUCKET/$(date +%Y/%m/%d)/"

# Cleanup old backups
find "$BACKUP_DIR" -name "*.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed: $TIMESTAMP"
```

### Recovery Procedures

```bash
#!/bin/bash
# restore-tensorus.sh

BACKUP_FILE="$1"
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop Tensorus
sudo systemctl stop tensorus

# Restore database
gunzip -c "$BACKUP_FILE" | psql -h localhost -U tensorus -d tensorus_prod

# Restore storage (if applicable)
# tar -xzf storage_backup.tar.gz -C /

# Start Tensorus
sudo systemctl start tensorus

# Verify restoration
sleep 10
curl -f http://localhost:8000/health || echo "Health check failed"

echo "Restoration completed"
```

## Scaling Strategies

### Horizontal Scaling

```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tensorus-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: tensorus-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Scaling

```bash
# Monitor resource usage
kubectl top nodes
kubectl top pods -n tensorus-prod

# Scale up resources
kubectl patch deployment tensorus-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"tensorus","resources":{"requests":{"memory":"8Gi","cpu":"4"},"limits":{"memory":"16Gi","cpu":"8"}}}]}}}}'
```

## Production Checklist

### Pre-Deployment

- [ ] Infrastructure provisioned and configured
- [ ] SSL certificates installed and valid
- [ ] Database configured with production settings
- [ ] Backup system tested and verified
- [ ] Monitoring and alerting configured
- [ ] Security hardening applied
- [ ] Load testing completed
- [ ] Disaster recovery plan documented

### Deployment

- [ ] Application deployed to production environment  
- [ ] Database migrations applied
- [ ] Configuration validated
- [ ] Health checks passing
- [ ] Load balancer configured
- [ ] Monitoring dashboards showing green metrics
- [ ] Backup verification completed

### Post-Deployment

- [ ] Smoke tests passed
- [ ] Performance benchmarks met
- [ ] Security scan completed
- [ ] Documentation updated
- [ ] Team training completed
- [ ] On-call procedures established
- [ ] Incident response plan activated

## Troubleshooting

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **High Memory Usage** | OOM kills, slow responses | Increase memory limits, optimize tensor operations |
| **Database Connections** | Connection timeouts | Increase connection pool size |
| **Storage Issues** | Write failures | Check disk space, permissions |
| **SSL Certificate** | HTTPS errors | Renew certificates, check configuration |

### Diagnostic Commands

```bash
# Check system resources
top
htop
iostat -x 1

# Check Tensorus logs
journalctl -u tensorus -f
tail -f /var/log/tensorus/app.log

# Database diagnostics
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity;"

# Network diagnostics
netstat -tulpn | grep 8000
ss -tulpn | grep 8000
```

---

**Need Production Support?**

Contact our enterprise support team:
- **Email**: production-support@tensorus.com
- **Phone**: 1-800-TENSORUS (24/7)
- **Slack**: #production-support (Enterprise customers)
- **Emergency**: emergency@tensorus.com