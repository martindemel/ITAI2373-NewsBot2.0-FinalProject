# NewsBot 2.0 Deployment Guide

## Production Deployment Guide for NewsBot Intelligence System

### Table of Contents
1. [Deployment Overview](#deployment-overview)
2. [System Requirements](#system-requirements)
3. [Local Development Setup](#local-development-setup)
4. [Production Deployment](#production-deployment)
5. [Cloud Deployment Options](#cloud-deployment-options)
6. [Configuration Management](#configuration-management)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Security Considerations](#security-considerations)
9. [Troubleshooting](#troubleshooting)
10. [Performance Optimization](#performance-optimization)

## Deployment Overview

NewsBot 2.0 is designed for flexible deployment across various environments:

- **Development**: Local machine with Jupyter notebooks
- **Production**: Linux server with web interface
- **Cloud**: AWS, Azure, or Google Cloud Platform
- **Docker**: Containerized deployment for scalability

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Production Environment                    │
├─────────────────────────────────────────────────────────────┤
│  Load Balancer (Nginx/Apache)                             │
├─────────────────────────────────────────────────────────────┤
│  NewsBot 2.0 Application Server                           │
│  ├─ Flask/FastAPI Web Interface                           │
│  ├─ Streamlit Dashboard                                    │
│  └─ Background Task Queue (Celery)                        │
├─────────────────────────────────────────────────────────────┤
│  Database Layer                                            │
│  ├─ Model Storage (File System/S3)                        │
│  ├─ Article Database (PostgreSQL/MongoDB)                 │
│  └─ Cache Layer (Redis)                                   │
├─────────────────────────────────────────────────────────────┤
│  External Services                                         │
│  ├─ OpenAI API                                            │
│  ├─ Translation APIs                                       │
│  └─ Monitoring (Prometheus/Grafana)                       │
└─────────────────────────────────────────────────────────────┘
```

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+, CentOS 8+, or macOS 10.15+
- **CPU**: 4 cores
- **Memory**: 8GB RAM
- **Storage**: 50GB SSD
- **Python**: 3.8+
- **Network**: Stable internet connection

### Recommended Production Requirements
- **OS**: Ubuntu 22.04 LTS
- **CPU**: 8+ cores
- **Memory**: 16GB+ RAM
- **Storage**: 100GB+ SSD
- **Python**: 3.10+
- **GPU**: Optional (NVIDIA with CUDA for ML acceleration)

## Local Development Setup

### Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/ITAI2373-NewsBot-Final.git
cd ITAI2373-NewsBot-Final

# Create virtual environment
python3 -m venv newsbot_env
source newsbot_env/bin/activate  # Linux/macOS
# newsbot_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download required NLP models
python -c "import spacy; spacy.cli.download('en_core_web_sm')"
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Initialize system
python newsbot_main.py --init

# Start development server
./smart_startup.sh
```

### Development Configuration
```bash
# Set environment variables
export FLASK_ENV=development
export FLASK_DEBUG=True
export NEWSBOT_LOG_LEVEL=DEBUG

# Optional: Set API keys for enhanced features
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_TRANSLATE_API_KEY="your-google-key"
```

## Production Deployment

### 1. Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+
sudo apt install python3.10 python3.10-venv python3.10-dev

# Install system dependencies
sudo apt install nginx postgresql redis-server supervisor
sudo apt install build-essential libpq-dev

# Create newsbot user
sudo useradd -m -s /bin/bash newsbot
sudo su - newsbot
```

### 2. Application Deployment

```bash
# Clone repository to production location
cd /opt
sudo git clone https://github.com/yourusername/ITAI2373-NewsBot-Final.git newsbot
sudo chown -R newsbot:newsbot /opt/newsbot
cd /opt/newsbot

# Create production virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install production dependencies
pip install -r requirements.txt
pip install gunicorn

# Install NLP models
python -c "import spacy; spacy.cli.download('en_core_web_sm')"
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create necessary directories
mkdir -p logs data/models data/results uploads
```

### 3. Database Setup

```bash
# PostgreSQL setup (optional for advanced features)
sudo -u postgres createuser newsbot
sudo -u postgres createdb newsbot_production -O newsbot

# Redis configuration
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

### 4. Configuration Management

Create production configuration file:

```bash
# /opt/newsbot/config/production.yaml
database:
  data_path: "/opt/newsbot/data/processed/newsbot_dataset.csv"
  models_path: "/opt/newsbot/data/models/"
  results_path: "/opt/newsbot/data/results/"

system:
  debug: false
  log_level: "INFO"
  max_workers: 8
  cache_enabled: true

web:
  host: "127.0.0.1"
  port: 8000
  secret_key: "your-production-secret-key-here"
  flask_debug: false

api:
  api_rate_limit: 100
  api_timeout: 60
```

### 5. Web Server Configuration

#### Nginx Configuration
```nginx
# /etc/nginx/sites-available/newsbot
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /opt/newsbot/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    client_max_body_size 50M;
}
```

#### Supervisor Configuration
```ini
# /etc/supervisor/conf.d/newsbot.conf
[program:newsbot-web]
command=/opt/newsbot/venv/bin/gunicorn -w 4 -b 127.0.0.1:8000 app:app
directory=/opt/newsbot
user=newsbot
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/opt/newsbot/logs/web.log

[program:newsbot-dashboard]
command=/opt/newsbot/venv/bin/streamlit run dashboard/newsbot_dashboard.py --server.port 8501 --server.address 127.0.0.1
directory=/opt/newsbot
user=newsbot
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/opt/newsbot/logs/dashboard.log
```

### 6. SSL/HTTPS Setup

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Cloud Deployment Options

### AWS Deployment

#### EC2 Instance
```bash
# Launch EC2 instance (t3.large recommended)
# Security Group: Allow ports 22, 80, 443

# Install dependencies
sudo yum update -y
sudo yum install python3 nginx git -y

# Follow standard deployment steps
```

#### ECS with Docker
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')"
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

EXPOSE 8000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
```

### Google Cloud Platform

#### App Engine Deployment
```yaml
# app.yaml
runtime: python310
env: standard

env_variables:
  FLASK_ENV: production
  NEWSBOT_LOG_LEVEL: INFO

automatic_scaling:
  min_instances: 1
  max_instances: 10

resources:
  cpu: 2
  memory_gb: 4
```

### Azure Deployment

#### Azure Container Instances
```bash
# Build and push Docker image
docker build -t newsbot:latest .
docker tag newsbot:latest your-registry.azurecr.io/newsbot:latest
docker push your-registry.azurecr.io/newsbot:latest

# Deploy to ACI
az container create \
  --resource-group newsbot-rg \
  --name newsbot-app \
  --image your-registry.azurecr.io/newsbot:latest \
  --ports 8000 \
  --memory 4 \
  --cpu 2
```

## Configuration Management

### Environment-Specific Configurations

```bash
# Development
export NEWSBOT_ENV=development
export FLASK_DEBUG=True

# Staging
export NEWSBOT_ENV=staging
export FLASK_DEBUG=False

# Production
export NEWSBOT_ENV=production
export FLASK_DEBUG=False
export NEWSBOT_SECRET_KEY="production-secret"
```

### API Key Management

```bash
# Use environment variables (recommended)
export OPENAI_API_KEY="your-openai-api-key-here"
export GOOGLE_TRANSLATE_API_KEY="..."

# Or use secure key management services
# AWS Secrets Manager, Azure Key Vault, etc.
```

## Monitoring and Maintenance

### Health Checks

```python
# health_check.py
import requests
import sys

def check_health():
    try:
        response = requests.get('http://localhost:8000/health')
        if response.status_code == 200:
            print("✅ Application is healthy")
            return 0
        else:
            print("❌ Application health check failed")
            return 1
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(check_health())
```

### Log Management

```bash
# Logrotate configuration
# /etc/logrotate.d/newsbot
/opt/newsbot/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 newsbot newsbot
    postrotate
        supervisorctl restart newsbot-web newsbot-dashboard
    endscript
}
```

### Backup Strategy

```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backup/newsbot"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup models and data
tar -czf "$BACKUP_DIR/models_$DATE.tar.gz" /opt/newsbot/data/models/
tar -czf "$BACKUP_DIR/results_$DATE.tar.gz" /opt/newsbot/data/results/

# Upload to cloud storage (optional)
aws s3 cp "$BACKUP_DIR/models_$DATE.tar.gz" s3://newsbot-backups/
```

## Security Considerations

### Application Security

1. **API Key Protection**: Use environment variables or secure vaults
2. **Input Validation**: Sanitize all user inputs
3. **Rate Limiting**: Implement API rate limiting
4. **HTTPS**: Always use SSL/TLS in production
5. **Access Control**: Implement proper authentication

### Server Security

```bash
# Firewall configuration
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw enable

# Fail2ban setup
sudo apt install fail2ban
sudo systemctl enable fail2ban
```

### Data Security

1. **Model Files**: Protect trained models from unauthorized access
2. **User Data**: Encrypt sensitive data at rest
3. **Audit Logs**: Maintain access and modification logs
4. **Regular Updates**: Keep dependencies updated

## Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Monitor memory usage
htop
free -h

# Adjust worker processes
# Reduce Gunicorn workers or optimize model loading
```

#### Performance Issues
```bash
# Profile application
pip install py-spy
py-spy top --pid $(pgrep -f gunicorn)

# Check database queries
# Enable SQL logging for analysis
```

#### Model Loading Errors
```bash
# Verify model files
ls -la /opt/newsbot/data/models/
python -c "import pickle; pickle.load(open('/opt/newsbot/data/models/best_classifier.pkl', 'rb'))"
```

### Log Analysis

```bash
# Monitor application logs
tail -f /opt/newsbot/logs/web.log

# Check for errors
grep -i error /opt/newsbot/logs/*.log

# Monitor Nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

## Performance Optimization

### Application Optimization

1. **Model Caching**: Load models once at startup
2. **Connection Pooling**: Use database connection pools
3. **Async Processing**: Use Celery for background tasks
4. **Response Caching**: Cache API responses when appropriate

### Infrastructure Optimization

1. **Load Balancing**: Use multiple application instances
2. **CDN**: Use CloudFlare or similar for static assets
3. **Database Optimization**: Tune PostgreSQL/MongoDB settings
4. **Monitoring**: Use Prometheus + Grafana for metrics

### Resource Monitoring

```bash
# System monitoring script
#!/bin/bash
echo "=== System Status ==="
date
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4"%"}'
echo "Memory Usage:"
free -h | grep ^Mem | awk '{print $3 "/" $2}'
echo "Disk Usage:"
df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 ")"}'
echo "NewsBot Processes:"
ps aux | grep -E "(newsbot|gunicorn|streamlit)" | grep -v grep
```

---

## Quick Reference Commands

### Production Management
```bash
# Start services
sudo supervisorctl start newsbot-web newsbot-dashboard

# Stop services
sudo supervisorctl stop newsbot-web newsbot-dashboard

# Restart services
sudo supervisorctl restart newsbot-web newsbot-dashboard

# Check status
sudo supervisorctl status

# View logs
sudo supervisorctl tail -f newsbot-web

# Update application
cd /opt/newsbot
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
sudo supervisorctl restart newsbot-web newsbot-dashboard
```

### Maintenance Tasks
```bash
# Check system health
python health_check.py

# Clean temporary files
find /opt/newsbot/uploads -type f -mtime +7 -delete

# Update NLP models
python -c "import spacy; spacy.cli.download('en_core_web_sm')"

# Backup data
./backup.sh
```

For additional support and detailed troubleshooting, refer to the [Technical Documentation](technical_documentation.md) and [User Guide](user_guide.md).