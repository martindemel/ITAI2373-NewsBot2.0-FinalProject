# NewsBot 2.0 API Reference

## Overview

NewsBot 2.0 provides a comprehensive RESTful API for programmatic access to all system functionality.

## Base URL
```
http://localhost:8080/api
```

## Core Endpoints

### Health Check
**GET /api/health** - System health status

### System Statistics  
**GET /api/stats** - System performance metrics

### Article Analysis
**POST /analyze** - Comprehensive article analysis

### Natural Language Queries
**POST /query** - Conversational interface

### Real-time Monitoring
**POST /api/realtime/start** - Start live monitoring

### Translation Services
**POST /translate** - Multi-language translation

## Example Usage

```python
import requests

# Analyze an article
response = requests.post('http://localhost:8080/api/analyze', json={
    "text": "Your article text here...",
    "include_sentiment": True,
    "include_entities": True
})

result = response.json()
print(f"Category: {result['classification']['predicted_category']}")
print(f"Sentiment: {result['sentiment']['overall_sentiment']}")
```

For complete API documentation, see the technical documentation.

