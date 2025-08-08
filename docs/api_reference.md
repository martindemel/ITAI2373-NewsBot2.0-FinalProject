# NewsBot 2.0 API Reference

## Overview

This document provides comprehensive API reference for NewsBot 2.0 Intelligence System. The API is designed for programmatic access to all system capabilities including analysis, multilingual processing, and conversational interfaces.

## Table of Contents

1. [Core System API](#core-system-api)
2. [Data Processing API](#data-processing-api)
3. [Analysis Components API](#analysis-components-api)
4. [Language Models API](#language-models-api)
5. [Multilingual API](#multilingual-api)
6. [Conversational API](#conversational-api)
7. [Utilities API](#utilities-api)
8. [Configuration API](#configuration-api)
9. [Error Handling](#error-handling)
10. [Examples](#examples)

## Core System API

### NewsBot2System

Main system orchestration class providing unified access to all components.

#### Constructor

```python
NewsBot2System(config_path: Optional[str] = None)
```

**Parameters**:
- `config_path` (str, optional): Path to configuration file

**Example**:
```python
from newsbot_main import NewsBot2System

# Default configuration
system = NewsBot2System()

# Custom configuration
system = NewsBot2System('config/custom_config.yaml')
```

#### initialize_system

```python
initialize_system(load_models: bool = True, load_data: bool = True) -> Dict[str, Any]
```

Initialize the complete NewsBot 2.0 system.

**Parameters**:
- `load_models` (bool): Whether to load pre-trained models
- `load_data` (bool): Whether to load article database

**Returns**:
- Dictionary containing initialization results and status

**Example**:
```python
result = system.initialize_system(load_models=True, load_data=True)
print(f"Status: {result['status']}")
print(f"Components loaded: {len(result['components_initialized'])}")
```

#### process_natural_language_query

```python
process_natural_language_query(query: str, user_id: Optional[str] = None) -> Dict[str, Any]
```

Process natural language queries about news data.

**Parameters**:
- `query` (str): Natural language query
- `user_id` (str, optional): User identifier for personalization

**Returns**:
- Dictionary containing query results and response

**Example**:
```python
result = system.process_natural_language_query("Show me positive tech news")
print(result['response'])
```

#### analyze_articles

```python
analyze_articles(articles: List[Dict[str, Any]], 
                analysis_types: Optional[List[str]] = None) -> Dict[str, Any]
```

Perform comprehensive analysis on news articles.

**Parameters**:
- `articles` (List[Dict]): List of articles with 'text' and 'category' fields
- `analysis_types` (List[str], optional): Types of analysis to perform
  - Options: `['classification', 'sentiment', 'entities', 'topics', 'summary']`

**Returns**:
- Dictionary containing analysis results

**Example**:
```python
articles = [
    {'text': 'Apple announces new iPhone...', 'category': 'tech'},
    {'text': 'Stock markets rise today...', 'category': 'business'}
]

result = system.analyze_articles(
    articles, 
    analysis_types=['classification', 'sentiment', 'entities']
)
```

#### get_system_status

```python
get_system_status() -> Dict[str, Any]
```

Get current system status and statistics.

**Returns**:
- Dictionary containing system status information

**Example**:
```python
status = system.get_system_status()
print(f"System initialized: {status['system_initialized']}")
print(f"Components loaded: {status['components_loaded']}")
print(f"Uptime: {status['uptime_seconds']} seconds")
```

## Data Processing API

### TextPreprocessor

Advanced text preprocessing for news articles.

#### Constructor

```python
TextPreprocessor(config: Optional[Dict[str, Any]] = None)
```

#### preprocess_text

```python
preprocess_text(text: str, preserve_entities: bool = True) -> str
```

Preprocess single text document.

**Parameters**:
- `text` (str): Input text to preprocess
- `preserve_entities` (bool): Whether to preserve named entities

**Returns**:
- Preprocessed text string

**Example**:
```python
from src.data_processing.text_preprocessor import TextPreprocessor

preprocessor = TextPreprocessor()
cleaned_text = preprocessor.preprocess_text(
    "Apple Inc. reported STRONG earnings!!!", 
    preserve_entities=True
)
```

#### preprocess_batch

```python
preprocess_batch(texts: List[str]) -> List[str]
```

Preprocess multiple texts efficiently.

### FeatureExtractor

Extract features from preprocessed text.

#### extract_tfidf_features

```python
extract_tfidf_features(texts: List[str]) -> scipy.sparse.csr_matrix
```

Extract TF-IDF features from texts.

**Parameters**:
- `texts` (List[str]): List of preprocessed texts

**Returns**:
- Sparse matrix of TF-IDF features

#### extract_embeddings

```python
extract_embeddings(texts: List[str]) -> np.ndarray
```

Extract semantic embeddings using sentence transformers.

**Parameters**:
- `texts` (List[str]): List of texts to embed

**Returns**:
- NumPy array of embeddings

### DataValidator

Validate and clean news data.

#### validate_dataset

```python
validate_dataset(df: pd.DataFrame) -> Dict[str, Any]
```

Comprehensive dataset validation.

**Parameters**:
- `df` (DataFrame): Dataset to validate

**Returns**:
- Validation results with errors and warnings

## Analysis Components API

### AdvancedNewsClassifier

Multi-level news classification with confidence scoring.

#### train

```python
train(texts: List[str], labels: List[str]) -> Dict[str, Any]
```

Train classification model on provided data.

**Parameters**:
- `texts` (List[str]): Training texts
- `labels` (List[str]): Corresponding labels

**Returns**:
- Training results and performance metrics

#### predict_with_confidence

```python
predict_with_confidence(texts: List[str]) -> Dict[str, Any]
```

Predict categories with confidence scores.

**Parameters**:
- `texts` (List[str]): Texts to classify

**Returns**:
- Dictionary with predictions and confidence scores

**Example**:
```python
from src.analysis.classifier import AdvancedNewsClassifier

classifier = AdvancedNewsClassifier()
classifier.train(train_texts, train_labels)

result = classifier.predict_with_confidence([
    "Apple releases new MacBook with M3 chip"
])

print(f"Prediction: {result['predictions'][0]}")
print(f"Confidence: {result['confidence_scores'][0]:.3f}")
```

### AdvancedSentimentAnalyzer

Multi-method sentiment analysis with emotion detection.

#### analyze_sentiment

```python
analyze_sentiment(text: str) -> Dict[str, Any]
```

Comprehensive sentiment analysis using multiple methods.

**Parameters**:
- `text` (str): Text to analyze

**Returns**:
- Dictionary with sentiment results from all methods

**Example**:
```python
from src.analysis.sentiment_analyzer import AdvancedSentimentAnalyzer

analyzer = AdvancedSentimentAnalyzer()
result = analyzer.analyze_sentiment(
    "The company reported excellent quarterly results!"
)

print(f"VADER: {result['vader']['classification']}")
print(f"TextBlob: {result['textblob']['classification']}")
print(f"Aggregated: {result['aggregated']['classification']}")
```

#### analyze_batch_sentiment

```python
analyze_batch_sentiment(texts: List[str]) -> List[Dict[str, Any]]
```

Batch sentiment analysis for multiple texts.

#### detect_emotions

```python
detect_emotions(text: str) -> Dict[str, Any]
```

Detect emotions in text using transformer models.

### EntityRelationshipMapper

Advanced named entity recognition and relationship mapping.

#### extract_entities

```python
extract_entities(text: str) -> Dict[str, Any]
```

Extract entities using multiple NER methods.

**Parameters**:
- `text` (str): Text to process

**Returns**:
- Dictionary with entities from different methods

**Example**:
```python
from src.analysis.ner_extractor import EntityRelationshipMapper

ner = EntityRelationshipMapper()
result = ner.extract_entities(
    "Apple CEO Tim Cook met with President Biden in Washington."
)

entities = result['merged']['entities']
for entity in entities:
    print(f"{entity['text']} ({entity['label']})")
```

#### build_knowledge_graph

```python
build_knowledge_graph(texts: List[str]) -> Dict[str, Any]
```

Build knowledge graph from entity relationships.

### TopicModeler

Topic discovery and content clustering.

#### fit_transform

```python
fit_transform(documents: List[str]) -> Dict[str, Any]
```

Fit topic model and transform documents.

**Parameters**:
- `documents` (List[str]): List of documents for topic modeling

**Returns**:
- Topic modeling results

#### get_topic_words

```python
get_topic_words(topic_id: int, n_words: int = 10) -> List[str]
```

Get top words for a specific topic.

#### get_article_topics

```python
get_article_topics(text: str) -> Dict[str, Any]
```

Get topic distribution for a single article.

**Example**:
```python
from src.analysis.topic_modeler import TopicModeler

modeler = TopicModeler({'num_topics': 5})
modeler.fit_transform(documents)

# Get topic words
for i in range(5):
    words = modeler.get_topic_words(i, n_words=5)
    print(f"Topic {i}: {', '.join(words)}")

# Get article topics
article_topics = modeler.get_article_topics(sample_article)
print(f"Dominant topic: {article_topics['dominant_topic']}")
```

## Language Models API

### IntelligentSummarizer

Advanced text summarization with multiple methods.

#### summarize_article

```python
summarize_article(text: str, method: str = 'extractive', 
                 max_length: int = 150) -> Dict[str, Any]
```

Summarize news article using specified method.

**Parameters**:
- `text` (str): Article text to summarize
- `method` (str): Summarization method ('extractive', 'abstractive', 'hybrid')
- `max_length` (int): Maximum summary length

**Returns**:
- Dictionary with summary and quality metrics

**Example**:
```python
from src.language_models.summarizer import IntelligentSummarizer

summarizer = IntelligentSummarizer()
result = summarizer.summarize_article(
    long_article_text, 
    method='hybrid',
    max_length=100
)

print(f"Summary: {result['hybrid_summary']}")
print(f"Compression ratio: {result['compression_ratio']:.2f}")
```

#### assess_summary_quality

```python
assess_summary_quality(original: str, summary: str) -> Dict[str, Any]
```

Evaluate summary quality using multiple metrics.

### EmbeddingGenerator

Semantic embeddings and similarity search.

#### generate_embeddings

```python
generate_embeddings(texts: List[str]) -> np.ndarray
```

Generate semantic embeddings for texts.

#### semantic_search

```python
semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]
```

Perform semantic search in indexed documents.

**Example**:
```python
from src.language_models.embeddings import EmbeddingGenerator

embedder = EmbeddingGenerator()
embedder.build_search_index(documents)

results = embedder.semantic_search(
    "artificial intelligence research", 
    top_k=3
)

for result in results:
    print(f"Score: {result['similarity_score']:.3f}")
    print(f"Text: {result['text'][:100]}...")
```

## Multilingual API

### LanguageDetector

Automatic language identification.

#### detect_language

```python
detect_language(text: str, confidence_threshold: float = 0.8) -> Dict[str, Any]
```

Detect language of input text.

**Parameters**:
- `text` (str): Text to analyze
- `confidence_threshold` (float): Minimum confidence threshold

**Returns**:
- Dictionary with detected language and confidence

**Example**:
```python
from src.multilingual.language_detector import LanguageDetector

detector = LanguageDetector()
result = detector.detect_language("Bonjour le monde")

print(f"Language: {result['language']}")
print(f"Confidence: {result['confidence']:.3f}")
```

#### detect_languages_batch

```python
detect_languages_batch(texts: List[str]) -> List[Dict[str, Any]]
```

Batch language detection for multiple texts.

### MultilingualTranslator

Translation services with quality assessment.

#### translate_text

```python
translate_text(text: str, source_lang: str, target_lang: str, 
               auto_detect: bool = False) -> Dict[str, Any]
```

Translate text between languages.

**Parameters**:
- `text` (str): Text to translate
- `source_lang` (str): Source language code
- `target_lang` (str): Target language code
- `auto_detect` (bool): Auto-detect source language

**Returns**:
- Dictionary with translated text and quality metrics

**Example**:
```python
from src.multilingual.translator import MultilingualTranslator

translator = MultilingualTranslator()
result = translator.translate_text(
    text="Hello world",
    source_lang="en",
    target_lang="es"
)

print(f"Translation: {result['translated_text']}")
print(f"Quality score: {result['quality_score']:.3f}")
```

#### translate_batch

```python
translate_batch(texts: List[str], source_lang: str, 
                target_lang: str) -> List[Dict[str, Any]]
```

Batch translation for multiple texts.

### CrossLingualAnalyzer

Cross-language analysis and comparison.

#### compare_sentiment_across_languages

```python
compare_sentiment_across_languages(multilingual_articles: List[Dict]) -> Dict[str, Any]
```

Compare sentiment analysis across different languages.

## Conversational API

### IntentClassifier

Natural language query intent classification.

#### classify_intent

```python
classify_intent(query: str) -> Dict[str, Any]
```

Classify user query intent.

**Example**:
```python
from src.conversation.intent_classifier import IntentClassifier

intent_classifier = IntentClassifier()
result = intent_classifier.classify_intent(
    "Show me positive technology news"
)

print(f"Intent: {result['intent']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### QueryProcessor

Complex natural language query processing.

#### process_query

```python
process_query(query: str, user_id: Optional[str] = None) -> Dict[str, Any]
```

Process complex natural language queries.

### ResponseGenerator

Natural language response generation.

#### generate_response

```python
generate_response(query_result: Dict[str, Any], 
                 user_context: Optional[Dict] = None) -> Dict[str, Any]
```

Generate natural language responses to queries.

## Utilities API

### VisualizationGenerator

Advanced visualization creation.

#### create_sentiment_dashboard

```python
create_sentiment_dashboard(sentiment_results: Dict[str, Any]) -> Dict[str, Any]
```

Create comprehensive sentiment analysis dashboard.

#### create_topic_visualization

```python
create_topic_visualization(topic_results: Dict[str, Any]) -> Dict[str, Any]
```

Create topic modeling visualizations.

### EvaluationFramework

Model evaluation and benchmarking.

#### evaluate_classification_performance

```python
evaluate_classification_performance(predictions: List[str], 
                                  true_labels: List[str]) -> Dict[str, Any]
```

Evaluate classification model performance.

### ExportManager

Advanced export capabilities.

#### export_analysis_results

```python
export_analysis_results(results: Dict[str, Any], export_format: str = 'json',
                        output_path: Optional[str] = None) -> Dict[str, Any]
```

Export analysis results in various formats.

**Supported formats**: json, csv, excel, pdf, html, xml, yaml

## Configuration API

### NewsBot2Config

Centralized configuration management.

#### Constructor

```python
NewsBot2Config(config_file: Optional[str] = None)
```

#### get

```python
get(key: str, default: Any = None) -> Any
```

Get configuration value using dot notation.

**Example**:
```python
from config.settings import NewsBot2Config

config = NewsBot2Config()
log_level = config.get('system.log_level', 'INFO')
classifier_config = config.get_component_config('classifier')
```

#### get_component_config

```python
get_component_config(component_name: str) -> Dict[str, Any]
```

Get configuration for specific component.

## Error Handling

### Exception Types

All API methods use standard Python exceptions with descriptive messages:

- `ValueError`: Invalid input parameters
- `FileNotFoundError`: Missing data files or models
- `ConnectionError`: Network or API connection issues
- `RuntimeError`: System or model execution errors

### Error Response Format

API methods return error information in results:

```python
{
    'status': 'error',
    'error': 'Error description',
    'error_type': 'ValueError',
    'timestamp': '2024-12-01T10:30:00'
}
```

### Handling Errors

```python
try:
    result = system.analyze_articles(articles)
    if 'error' in result:
        print(f"Analysis error: {result['error']}")
    else:
        # Process successful result
        pass
except Exception as e:
    print(f"System error: {e}")
```

## Examples

### Complete Analysis Pipeline

```python
from newsbot_main import NewsBot2System

# Initialize system
system = NewsBot2System()
init_result = system.initialize_system()

if init_result['status'] == 'completed':
    # Prepare articles
    articles = [
        {
            'text': 'Apple Inc. announced record quarterly earnings...',
            'category': 'tech'
        },
        {
            'text': 'The Federal Reserve maintained interest rates...',
            'category': 'business'
        }
    ]
    
    # Perform comprehensive analysis
    analysis_result = system.analyze_articles(
        articles,
        analysis_types=['classification', 'sentiment', 'entities', 'topics']
    )
    
    # Create visualizations
    dashboard = system.create_analysis_dashboard(analysis_result)
    
    # Export results
    export_result = system.export_analysis_results(
        analysis_result,
        export_format='excel',
        output_path='analysis_results.xlsx'
    )
    
    print(f"Analysis completed. Results exported to {export_result['output_path']}")
```

### Natural Language Interface

```python
# Initialize system
system = NewsBot2System()
system.initialize_system()

# Interactive query processing
queries = [
    "How many technology articles are there?",
    "What is the overall sentiment of business news?",
    "Find articles mentioning Apple or Microsoft",
    "Show me the top topics in sports news",
    "Summarize the most positive entertainment article"
]

for query in queries:
    result = system.process_natural_language_query(query)
    print(f"Q: {query}")
    print(f"A: {result.get('response', 'Error processing query')}")
    print("-" * 50)
```

### Multilingual Analysis

```python
from src.multilingual.language_detector import LanguageDetector
from src.multilingual.translator import MultilingualTranslator
from src.analysis.sentiment_analyzer import AdvancedSentimentAnalyzer

# Initialize components
detector = LanguageDetector()
translator = MultilingualTranslator()
sentiment_analyzer = AdvancedSentimentAnalyzer()

# Multilingual article
article = "La empresa Apple anunció ganancias récord este trimestre."

# Detect language
lang_result = detector.detect_language(article)
print(f"Detected language: {lang_result['language']}")

# Translate to English
translation = translator.translate_text(
    text=article,
    source_lang=lang_result['language'],
    target_lang='en'
)
print(f"Translation: {translation['translated_text']}")

# Analyze sentiment
sentiment = sentiment_analyzer.analyze_sentiment(translation['translated_text'])
print(f"Sentiment: {sentiment['aggregated']['classification']}")
```

---

*API Reference Version 2.0 - Last Updated: December 2024*