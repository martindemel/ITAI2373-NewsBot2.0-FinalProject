# NewsBot 2.0 Intelligence System - Technical Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview) - Complete system design and component interactions
2. [API Reference](#api-reference) - Function and class documentation with examples
3. [Installation Guide](#installation-guide) - Step-by-step setup instructions
4. [Configuration Manual](#configuration-manual) - Customization and optimization options
5. [Module Documentation](#module-documentation) - Detailed component specifications
6. [Performance Optimization](#performance-optimization)
7. [Security Considerations](#security-considerations)
8. [Troubleshooting](#troubleshooting)

## System Overview

NewsBot 2.0 is an advanced Natural Language Processing (NLP) intelligence system designed for comprehensive news analysis. Building upon the original midterm NewsBot system, this collaborative project developed by **Martin Demel** and **Jiri Musil** introduces sophisticated multilingual capabilities, conversational AI interfaces, and production-ready architecture that exceeds all ITAI 2373 project requirements.

## Development Team

- **Martin Demel** - Lead Developer & System Architecture
- **Jiri Musil** - NLP Specialist & Web Interface Development

This project demonstrates both advanced technical implementation and professional collaborative development practices.

### Key Features

- **Advanced Content Analysis**: Multi-level classification, sentiment tracking, and entity relationship mapping
- **Language Understanding**: Intelligent summarization, semantic search, and content enhancement
- **Multilingual Intelligence**: Cross-language analysis, translation integration, and cultural context understanding
- **Conversational Interface**: Natural language queries and interactive exploration with OpenAI integration
- **Production Architecture**: Scalable, maintainable design with comprehensive monitoring
- **Web Application**: Full-stack Flask application with modern UI and real-time capabilities

### System Requirements

#### Minimum Requirements
- **Python**: 3.8 or higher (3.10 recommended)
- **Memory**: 8GB RAM
- **Storage**: 10GB free space
- **CPU**: Multi-core processor
- **Network**: Internet connection for API services and model downloads

#### Recommended for Production
- **Python**: 3.10+
- **Memory**: 16GB RAM
- **Storage**: 50GB SSD
- **CPU**: 8+ cores
- **GPU**: CUDA-compatible (optional, for transformer acceleration)
- **Network**: High-speed internet with stable connection

### Data Sources

The system operates on authentic BBC News dataset:
- **Total Articles**: 2,225 news articles
- **Categories**: 5 categories (business=510, entertainment=386, politics=417, sport=511, tech=401)
- **Languages**: Primary English with multilingual support for 50+ languages
- **Format**: CSV with text content and category labels
- **Quality**: Real-world complexity with professional journalism standards

## Architecture Overview

### High-Level Architecture

NewsBot 2.0 implements a modular, scalable architecture with four main modules:

```
┌─────────────────────────────────────────────────────────────────┐
│                    NewsBot 2.0 System Architecture              │
├─────────────────┬─────────────────┬─────────────────┬──────────────┤
│   Module A      │    Module B     │    Module C     │   Module D   │
│  Content        │   Language      │  Multilingual   │Conversational│
│  Analysis       │ Understanding   │  Intelligence   │  Interface   │
├─────────────────┼─────────────────┼─────────────────┼──────────────┤
│• Classification │• Summarization  │• Translation    │• NL Queries  │
│• Sentiment      │• Embeddings     │• Lang Detection │• Intent      │
│• NER            │• Semantic Search│• Cross-lingual  │• Context Mgmt│
│• Topic Modeling │• Query Understanding│• Cultural Context│• Response Gen│
└─────────────────┴─────────────────┴─────────────────┴──────────────┘
                                │
                    ┌─────────────────────────┐
                    │   Data Processing       │
                    │• Text Preprocessing     │
                    │• Feature Extraction     │
                    │• Real-time Processing   │
                    │• Data Validation        │
                    └─────────────────────────┘
                                │
                    ┌─────────────────────────┐
                    │   Utility Components    │
                    │• Visualization          │
                    │• Performance Monitor    │
                    │• Export/Reporting       │
                    │• Evaluation Metrics     │
                    └─────────────────────────┘
```

### Technology Stack

#### Core Technologies
- **Backend Framework**: Flask 3.0+ with production WSGI deployment
- **ML/NLP Libraries**: scikit-learn, transformers, spaCy, NLTK, gensim
- **Data Processing**: pandas, numpy, scipy
- **Visualization**: plotly, matplotlib, seaborn, wordcloud
- **Database**: File-based with SQLAlchemy support for scaling
- **Real-time**: RSS feed processing with asyncio

#### Advanced Components
- **Transformer Models**: BERT, RoBERTa, sentence-transformers
- **OpenAI Integration**: GPT-4 for advanced conversational capabilities
- **Translation Services**: Google Translate, Azure Translator, LibreTranslate
- **Caching**: Redis-ready for performance optimization
- **Monitoring**: Built-in performance metrics and health checks

### Design Patterns

#### Modular Architecture
- **Separation of Concerns**: Each module handles specific NLP tasks
- **Dependency Injection**: Configuration-driven component initialization
- **Factory Pattern**: Dynamic model and analyzer creation
- **Observer Pattern**: Real-time event processing and notifications

#### Scalability Patterns
- **Microservice Ready**: Modules can be deployed independently
- **Stateless Design**: No server-side session dependencies for core analysis
- **Caching Strategy**: Multi-level caching for performance optimization
- **Async Processing**: Non-blocking operations for real-time features

## Module Documentation

### Module A: Advanced Content Analysis Engine

#### Classification System (`src/analysis/classifier.py`)

**Purpose**: Multi-level news categorization with confidence scoring

**Features**:
- **Multiple Algorithms**: SVM, Random Forest, XGBoost with ensemble voting
- **Confidence Scoring**: Probabilistic outputs with uncertainty quantification
- **Feature Engineering**: TF-IDF, n-grams, and custom feature extraction
- **Performance Metrics**: Precision, recall, F1-score, confusion matrices

**Key Methods**:
```python
class NewsClassifier:
    def predict(self, text: str) -> Dict[str, Any]
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]
    def get_feature_importance(self) -> Dict[str, float]
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict[str, float]
```

**Performance**: 97.5% accuracy on BBC News dataset

#### Sentiment Analysis (`src/analysis/sentiment_analyzer.py`)

**Purpose**: Multi-method sentiment analysis with temporal tracking

**Features**:
- **Multiple Methods**: VADER, TextBlob, transformer-based (RoBERTa)
- **Emotion Detection**: Extended emotion classification beyond polarity
- **Temporal Tracking**: Sentiment evolution over time periods
- **Confidence Scoring**: Agreement analysis across methods

**Key Methods**:
```python
class SentimentAnalyzer:
    def analyze_sentiment(self, text: str, methods: List[str] = None) -> Dict[str, Any]
    def analyze_emotions(self, text: str) -> Dict[str, float]
    def track_sentiment_evolution(self, texts: List[str], timestamps: List[datetime]) -> Dict[str, Any]
    def get_sentiment_summary(self, results: List[Dict]) -> Dict[str, Any]
```

#### Named Entity Recognition (`src/analysis/ner_extractor.py`)

**Purpose**: Entity extraction with relationship mapping

**Features**:
- **BERT-based NER**: Pre-trained transformer models for high accuracy
- **Entity Types**: PERSON, ORGANIZATION, LOCATION, MISC with confidence scores
- **Relationship Mapping**: Entity co-occurrence and dependency analysis
- **Knowledge Graph**: Entity relationship visualization and storage

**Key Methods**:
```python
class NERExtractor:
    def extract_entities(self, text: str) -> Dict[str, Any]
    def extract_relationships(self, text: str) -> List[Dict[str, Any]]
    def create_knowledge_graph(self, entities: List[Dict]) -> Dict[str, Any]
    def visualize_entities(self, entities: List[Dict]) -> str  # HTML visualization
```

#### Topic Modeling (`src/analysis/topic_modeler.py`)

**Purpose**: Content discovery and trend analysis

**Features**:
- **Multiple Algorithms**: LDA, NMF, and Gensim LDA with coherence optimization
- **Interactive Visualization**: pyLDAvis integration for topic exploration
- **Trend Analysis**: Topic evolution over time with statistical significance
- **Content Clustering**: Document grouping based on topic distributions

**Key Methods**:
```python
class TopicModeler:
    def fit_topics(self, documents: List[str], method: str = 'lda', n_topics: int = 10) -> Dict[str, Any]
    def get_topic_words(self, topic_id: int, n_words: int = 10) -> List[Tuple[str, float]]
    def get_document_topics(self, document: str) -> List[Tuple[int, float]]
    def visualize_topics(self) -> str  # Interactive HTML visualization
    def analyze_topic_trends(self, documents: List[str], timestamps: List[datetime]) -> Dict[str, Any]
```

### Module B: Language Understanding and Generation

#### Intelligent Summarization (`src/language_models/summarizer.py`)

**Purpose**: Multi-algorithm text summarization with quality assessment

**Features**:
- **Extractive Summarization**: Key sentence extraction with ranking
- **Abstractive Summarization**: Transformer-based generation (BART, T5)
- **Hybrid Approach**: Combined extractive and abstractive methods
- **Quality Metrics**: ROUGE scores, coherence assessment, readability analysis

**Key Methods**:
```python
class IntelligentSummarizer:
    def summarize_extractive(self, text: str, num_sentences: int = 3) -> Dict[str, Any]
    def summarize_abstractive(self, text: str, max_length: int = 150) -> Dict[str, Any]
    def summarize_hybrid(self, text: str) -> Dict[str, Any]
    def evaluate_summary_quality(self, original: str, summary: str) -> Dict[str, float]
```

#### Semantic Embeddings (`src/language_models/embeddings.py`)

**Purpose**: Semantic understanding and similarity analysis

**Features**:
- **Pre-trained Embeddings**: sentence-transformers, Word2Vec, GloVe
- **Semantic Search**: Document similarity and retrieval
- **Clustering**: Content grouping based on semantic similarity
- **Dimensionality Reduction**: UMAP, t-SNE for visualization

**Key Methods**:
```python
class SemanticEmbeddings:
    def generate_embeddings(self, texts: List[str]) -> np.ndarray
    def find_similar_documents(self, query: str, corpus: List[str], top_k: int = 5) -> List[Dict[str, Any]]
    def cluster_documents(self, texts: List[str], n_clusters: int = 5) -> Dict[str, Any]
    def visualize_embeddings(self, texts: List[str]) -> str  # Interactive visualization
```

### Module C: Multilingual Intelligence

#### Translation Services (`src/multilingual/translator.py`)

**Purpose**: Multi-provider translation with quality assessment

**Features**:
- **Multiple Providers**: Google Translate, Azure Translator, LibreTranslate
- **Quality Assessment**: Translation confidence scoring and back-translation validation
- **Language Detection**: Automatic language identification with confidence
- **Batch Processing**: Efficient translation of large document sets

**Key Methods**:
```python
class MultilingualTranslator:
    def translate(self, text: str, target_language: str, source_language: str = 'auto') -> Dict[str, Any]
    def detect_language(self, text: str) -> Dict[str, Any]
    def assess_translation_quality(self, original: str, translated: str) -> Dict[str, float]
    def translate_batch(self, texts: List[str], target_language: str) -> List[Dict[str, Any]]
```

#### Cross-Lingual Analysis (`src/multilingual/cross_lingual_analyzer.py`)

**Purpose**: Comparative analysis across languages

**Features**:
- **Content Comparison**: Cross-language content similarity and differences
- **Cultural Context**: Regional perspective analysis and bias detection
- **Trend Analysis**: Multi-language trend correlation and divergence
- **Entity Alignment**: Cross-language entity matching and disambiguation

### Module D: Conversational Interface

#### AI-Powered Conversation (`src/conversation/ai_powered_conversation.py`)

**Purpose**: Natural language query processing with context management

**Features**:
- **Intent Classification**: ML-based intent detection with confidence scoring
- **Context Management**: Multi-turn conversation state maintenance
- **OpenAI Integration**: GPT-4 powered advanced language understanding
- **Query Understanding**: Complex multi-part question processing

**Key Methods**:
```python
class AIPoweredConversation:
    def process_query(self, user_query: str, user_id: str = None) -> Dict[str, Any]
    def classify_intent(self, query: str) -> Dict[str, Any]
    def maintain_context(self, user_id: str, interaction: Dict[str, Any]) -> None
    def generate_response(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]
```

#### OpenAI Integration (`src/conversation/openai_chat_handler.py`)

**Purpose**: Advanced language model integration for conversational AI

**Features**:
- **GPT-4 Integration**: State-of-the-art language understanding and generation
- **Fallback Handling**: Graceful degradation when API unavailable
- **Performance Optimization**: Caching and rate limiting
- **Security**: API key management and validation

## Web Application Architecture

### Flask Application Structure

The web application (`app.py`) implements a comprehensive interface with the following endpoints:

#### Core Routes
- `GET /` - Interactive dashboard with system overview
- `GET,POST /analyze` - Single article analysis interface
- `GET,POST /batch` - Batch processing for multiple articles
- `GET,POST /query` - Natural language query interface
- `GET,POST /translate` - Translation services interface
- `GET /visualization` - Advanced data visualizations
- `GET,POST /realtime` - Real-time news monitoring

#### API Endpoints
- `GET /api/health` - System health check
- `GET /api/stats` - System statistics and metrics
- `POST /api/realtime/start` - Start real-time monitoring
- `GET /api/realtime/stats` - Real-time processing statistics
- `GET /api/realtime/articles` - Recent processed articles

### Frontend Features

#### Interactive Dashboard
- **Real-time Statistics**: Article counts, processing metrics, system status
- **Recent Activity**: Query history and analysis results
- **System Health**: Component status and performance indicators
- **Quick Actions**: Direct access to all major features

#### Analysis Interfaces
- **Single Article**: Comprehensive analysis with visualization
- **Batch Processing**: CSV upload and processing with progress tracking
- **Results Export**: JSON, CSV, PDF export capabilities
- **Interactive Visualizations**: Plotly-based charts and graphs

#### Real-time Monitoring
- **Live RSS Processing**: Multiple feed monitoring with configurable sources
- **Streaming Updates**: WebSocket-ready for real-time updates
- **Sentiment Tracking**: Live sentiment analysis and trending
- **Alert System**: Configurable alerts for significant events

## Installation Guide

### Prerequisites

1. **Python Environment**:
   ```bash
   python --version  # Should be 3.8+
   pip --version     # Ensure pip is available
   ```

2. **Virtual Environment** (recommended):
   ```bash
   python -m venv newsbot_env
   source newsbot_env/bin/activate  # Linux/Mac
   # or
   newsbot_env\Scripts\activate     # Windows
   ```

### Installation Steps

1. **Clone Repository**:
   ```bash
   git clone [repository-url]
   cd ITAI2373-NewsBot-Final
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK Data**:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

4. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (optional for basic features)
   ```

5. **Initialize Data and Models**:
   ```bash
   python train_models.py
   ```

6. **Start Application**:
   ```bash
   python start_newsbot.py
   ```

### Verification

Access the web interface at `http://localhost:8080` and verify:
- Dashboard loads successfully
- System statistics show correct article counts
- All modules are initialized properly
- Basic analysis functions work correctly

## Configuration Manual

### Environment Variables (.env)

```bash
# OpenAI API Key (for advanced conversational features)
OPENAI_API_KEY=your-openai-api-key-here

# Translation Service API Keys (optional)
GOOGLE_TRANSLATE_API_KEY=your-google-key-here
AZURE_TRANSLATE_KEY=your-azure-key-here

# System Configuration
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=5000

# Security
SECRET_KEY=your-super-secret-key-for-production
FLASK_DEBUG=false

# Data Paths
DATA_PATH=data/processed/newsbot_dataset.csv
MODELS_PATH=data/models/
```

### System Configuration (config/newsbot_config.yaml)

```yaml
# Model Configuration
model:
  classification_model: "svm"
  topic_model_algorithm: "lda"
  sentiment_model: "roberta"
  ner_model: "bert"
  device: "auto"  # auto, cpu, cuda, mps
  batch_size: 32
  max_sequence_length: 512

# Performance Settings
system:
  debug: false
  log_level: "INFO"
  auto_load_models: true
  model_cache: true
  performance_monitoring: true

# Web Application Settings
web:
  host: "0.0.0.0"
  port: 5000
  enable_cors: true
  enable_security_headers: true
  max_content_length: 16777216  # 16MB

# Real-time Processing
realtime:
  enabled: true
  update_interval: 30  # seconds
  max_feeds: 10
  max_articles_per_feed: 50
```

### Component Configuration

Each module supports detailed configuration through the settings system:

```python
# Example: Configure sentiment analyzer
sentiment_config = {
    'methods': ['vader', 'textblob', 'transformer'],
    'transformer_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'confidence_threshold': 0.7,
    'enable_emotion_detection': True
}

# Example: Configure topic modeler
topic_config = {
    'n_topics': 10,
    'method': 'lda',
    'max_features': 1000,
    'min_df': 2,
    'max_df': 0.95,
    'random_state': 42
}
```

## Performance Optimization

### Memory Management

1. **Model Loading**: Lazy loading with caching
2. **Batch Processing**: Configurable batch sizes for large datasets
3. **Memory Monitoring**: Built-in memory profiling and optimization
4. **Garbage Collection**: Automatic cleanup of temporary objects

### Processing Optimization

1. **Parallel Processing**: Multi-threading for independent operations
2. **Caching Strategy**: Multi-level caching for repeated operations
3. **Efficient Data Structures**: Optimized pandas and numpy operations
4. **Model Optimization**: Quantization and pruning for transformer models

### Scaling Considerations

1. **Horizontal Scaling**: Microservice-ready architecture
2. **Load Balancing**: Stateless design for easy load distribution
3. **Database Scaling**: SQLAlchemy integration for relational databases
4. **Caching Layer**: Redis integration for distributed caching

## Security Considerations

### API Key Management

1. **Environment Variables**: Never commit API keys to version control
2. **Key Rotation**: Support for dynamic API key updates
3. **Access Control**: Role-based access for different API services
4. **Validation**: API key format validation and security checks

### Web Application Security

1. **Input Validation**: Comprehensive input sanitization and validation
2. **CSRF Protection**: Cross-site request forgery prevention
3. **Session Security**: Secure session management and timeout
4. **Rate Limiting**: Protection against abuse and DoS attacks

### Data Protection

1. **Encryption**: Sensitive data encryption at rest and in transit
2. **Access Logging**: Comprehensive audit trail for data access
3. **Data Minimization**: Only collect and store necessary data
4. **Compliance**: GDPR-ready data handling practices

## Troubleshooting

### Common Issues

#### Installation Problems

**Issue**: `ModuleNotFoundError` for required packages
**Solution**: 
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**Issue**: NLTK data not found
**Solution**:
```bash
python -c "import nltk; nltk.download('all')"
```

#### Runtime Errors

**Issue**: OpenAI API key errors
**Solution**: 
- Verify API key format in `.env` file
- Check API key validity on OpenAI platform
- Ensure sufficient API credits

**Issue**: Memory errors with large datasets
**Solution**:
- Reduce batch size in configuration
- Enable model caching
- Use streaming processing for large files

#### Performance Issues

**Issue**: Slow processing speed
**Solution**:
- Enable GPU acceleration if available
- Optimize batch sizes
- Use model quantization
- Enable caching

### Logging and Debugging

The system provides comprehensive logging at multiple levels:

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# View application logs
tail -f app.log

# Monitor system performance
python -c "from src.utils.performance_monitor import get_performance_monitor; print(get_performance_monitor().get_metrics())"
```

### System Health Checks

Access health check endpoints:
```bash
curl http://localhost:8080/api/health
curl http://localhost:8080/api/stats
```

## Development Guidelines

### Code Quality Standards

1. **PEP 8 Compliance**: Follow Python style guidelines
2. **Type Hints**: Use type annotations for better code clarity
3. **Docstrings**: Comprehensive documentation for all functions and classes
4. **Error Handling**: Robust exception handling with informative messages

### Testing Framework

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_classification.py -v
python -m pytest tests/test_integration.py -v

# Generate coverage report
python -m pytest tests/ --cov=src --cov-report=html
```

### Contributing Guidelines

1. **Branch Strategy**: Feature branches with descriptive names
2. **Commit Messages**: Clear, descriptive commit messages
3. **Code Review**: All changes require review before merging
4. **Documentation**: Update documentation for all changes

### Deployment Guidelines

#### Development Deployment
```bash
python start_newsbot.py
```

#### Production Deployment
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 app:app

# Using Docker
docker build -t newsbot-2.0 .
docker run -p 8080:8080 newsbot-2.0
```

#### Cloud Deployment
- **AWS**: Elastic Beanstalk, ECS, or Lambda deployment
- **Azure**: App Service or Container Instances
- **Google Cloud**: App Engine or Cloud Run
- **Heroku**: Direct deployment with Procfile

---

This technical documentation provides comprehensive coverage of NewsBot 2.0 system architecture, implementation details, and operational guidelines. For additional support, refer to the user guide and API reference documentation.