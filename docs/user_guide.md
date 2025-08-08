# NewsBot 2.0 User Guide
## Your Complete Guide to Advanced News Intelligence

## Welcome to NewsBot 2.0

NewsBot 2.0 is your comprehensive AI-powered news analysis platform, designed to transform how you understand, analyze, and extract actionable insights from news content. Built with cutting-edge Natural Language Processing technology, this system provides enterprise-grade news intelligence capabilities accessible through an intuitive web interface.

### What Makes NewsBot 2.0 Special?

- **Production-Ready AI**: Enterprise-grade system with 97.5% accuracy
- **Real-Time Analysis**: Instant processing of news content with live monitoring
- **Multilingual Support**: Analysis and translation across 50+ languages
- **Conversational Interface**: Natural language queries for easy interaction
- **Professional Results**: Export-ready reports and visualizations

## Table of Contents

1. [Getting Started](#getting-started)
2. [Quick Start Guide](#quick-start-guide)
3. [Feature Overview](#feature-overview)
4. [Step-by-Step Tutorials](#step-by-step-tutorials)
5. [Advanced Usage](#advanced-usage)
6. [FAQ](#faq)
7. [Troubleshooting](#troubleshooting)
8. [Tips and Best Practices](#tips-and-best-practices)

## Getting Started

### What is NewsBot 2.0?

NewsBot 2.0 is an intelligent news analysis system that can:
- **Classify news articles** into categories (business, entertainment, politics, sport, tech)
- **Analyze sentiment** and emotional tone of articles
- **Extract key entities** like people, organizations, and locations
- **Discover topics** and trends in news content
- **Summarize articles** into concise, readable summaries
- **Process multilingual content** with automatic translation
- **Answer questions** about your news data in natural language

### System Requirements

- **Computer**: Mac, Windows, or Linux
- **Memory**: At least 4GB RAM (8GB recommended)
- **Storage**: 5GB free space
- **Internet**: Required for some features (translation, latest models)

### First Time Setup

1. **Installation**: Follow the installation guide in the technical documentation
2. **Data Preparation**: The system comes with the BBC News dataset (2,225 articles)
3. **Configuration**: Basic configuration works out of the box

## Quick Start Guide

### Starting NewsBot 2.0

Open your terminal/command prompt and navigate to the NewsBot 2.0 directory:

```bash
cd ITAI2373-NewsBot-Final
python newsbot_main.py --init
```

This initializes the system and loads the BBC News dataset.

### Your First Analysis

Try analyzing some articles:

```bash
python newsbot_main.py --analyze data/processed/newsbot_dataset.csv --format json
```

### Your First Query

Ask NewsBot a question:

```bash
python newsbot_main.py --query "Show me positive technology news"
```

### Check System Status

```bash
python newsbot_main.py --status
```

## Feature Overview

### 1. Article Classification

**What it does**: Automatically categorizes news articles into topics
**Categories**: Business, Entertainment, Politics, Sport, Technology
**Accuracy**: 97.5% on the BBC News dataset

**Example Use Cases**:
- Organize large collections of news articles
- Filter news feeds by topic
- Track coverage distribution across categories

### 2. Sentiment Analysis

**What it does**: Determines the emotional tone of articles
**Methods**: Multiple AI approaches (VADER, TextBlob, Transformers)
**Output**: Positive, Negative, or Neutral with confidence scores

**Example Use Cases**:
- Monitor public sentiment about topics
- Track sentiment changes over time
- Identify controversial or polarizing content

### 3. Entity Recognition

**What it does**: Identifies people, organizations, locations, and other entities
**Capabilities**: Relationship mapping and knowledge graphs
**Output**: Structured entity data with confidence scores

**Example Use Cases**:
- Track mentions of specific people or companies
- Understand relationships between entities
- Create knowledge maps of news content

### 4. Topic Modeling

**What it does**: Discovers hidden themes and topics in news collections
**Methods**: Advanced algorithms (LDA, NMF)
**Output**: Topic words, document assignments, evolution tracking

**Example Use Cases**:
- Discover emerging trends
- Group related articles
- Track how topics evolve over time

### 5. Intelligent Summarization

**What it does**: Creates concise summaries of long articles
**Methods**: Extractive and abstractive summarization
**Output**: Key points and main ideas

**Example Use Cases**:
- Quickly understand long articles
- Create executive briefings
- Generate content previews

### 6. Multilingual Support

**What it does**: Processes content in multiple languages
**Capabilities**: Language detection, translation, cross-language analysis
**Languages**: 50+ languages supported

**Example Use Cases**:
- Analyze international news sources
- Compare coverage across different countries
- Access content in your preferred language

### 7. Conversational Interface

**What it does**: Answer questions about your news data in natural language
**Capabilities**: Complex queries, context understanding, personalized responses
**Examples**: "Find articles about Apple", "What's the sentiment of business news?"

## Step-by-Step Tutorials

### Tutorial 1: Analyzing Your First Article

**Goal**: Analyze a single news article for classification, sentiment, and entities

**Step 1**: Prepare your article
Create a CSV file with your article:
```csv
text,category
"Apple Inc. announced record quarterly earnings driven by strong iPhone sales.",tech
```

**Step 2**: Run the analysis
```bash
python newsbot_main.py --analyze your_article.csv --format json --output results.json
```

**Step 3**: Understand the results
The output will contain:
- **Classification**: Predicted category and confidence
- **Sentiment**: Emotional tone analysis
- **Entities**: People, organizations, locations mentioned

**Step 4**: Interpret the output
```json
{
  "results": {
    "classification": {
      "predictions": ["tech"],
      "confidence_scores": [0.95]
    },
    "sentiment": {
      "sentiment_distribution": {
        "positive": 0.7,
        "neutral": 0.3,
        "negative": 0.0
      }
    },
    "entities": {
      "entities_by_type": {
        "ORG": ["Apple Inc."],
        "PRODUCT": ["iPhone"]
      }
    }
  }
}
```

### Tutorial 2: Exploring the BBC News Dataset

**Goal**: Understand the structure and content of the included dataset

**Step 1**: Check dataset information
```bash
python newsbot_main.py --status
```

**Step 2**: Load dataset details
```python
import pandas as pd
df = pd.read_csv('data/processed/newsbot_dataset.csv')
print(f"Total articles: {len(df)}")
print(f"Categories: {df['category'].value_counts()}")
```

**Step 3**: Analyze the full dataset
```bash
python newsbot_main.py --analyze data/processed/newsbot_dataset.csv
```

**Step 4**: Explore results
The analysis will show:
- Overall sentiment distribution
- Entity frequency
- Topic discoveries
- Classification performance

### Tutorial 3: Using Natural Language Queries

**Goal**: Learn to ask questions about your news data

**Step 1**: Start with simple queries
```bash
python newsbot_main.py --query "How many articles are about technology?"
```

**Step 2**: Try sentiment queries
```bash
python newsbot_main.py --query "What is the sentiment of sports news?"
```

**Step 3**: Ask about entities
```bash
python newsbot_main.py --query "Find articles mentioning Apple"
```

**Step 4**: Complex analytical queries
```bash
python newsbot_main.py --query "Show me positive business news from the last month"
```

### Tutorial 4: Topic Discovery

**Goal**: Discover hidden topics in your news collection

**Step 1**: Run topic modeling
```python
from src.analysis.topic_modeler import TopicModeler

topic_modeler = TopicModeler({'num_topics': 5})
topic_modeler.fit_transform(article_texts)
```

**Step 2**: Explore discovered topics
```python
for topic_id in range(5):
    words = topic_modeler.get_topic_words(topic_id)
    print(f"Topic {topic_id}: {', '.join(words)}")
```

**Step 3**: Assign articles to topics
```python
article_topics = topic_modeler.get_article_topics(sample_article)
print(f"Dominant topic: {article_topics['dominant_topic']}")
```

### Tutorial 5: Multilingual Analysis

**Goal**: Process and analyze content in multiple languages

**Step 1**: Detect language
```python
from src.multilingual.language_detector import LanguageDetector

detector = LanguageDetector()
result = detector.detect_language("Bonjour, comment allez-vous?")
print(f"Detected language: {result['language']}")
```

**Step 2**: Translate content
```python
from src.multilingual.translator import MultilingualTranslator

translator = MultilingualTranslator()
translation = translator.translate_text(
    text="Hello world",
    source_lang="en",
    target_lang="es"
)
print(f"Translation: {translation['translated_text']}")
```

**Step 3**: Cross-language analysis
```python
from src.multilingual.cross_lingual_analyzer import CrossLingualAnalyzer

analyzer = CrossLingualAnalyzer()
comparison = analyzer.compare_sentiment_across_languages(multilingual_articles)
```

## Advanced Usage

### Batch Processing

For processing large numbers of articles:

```python
from newsbot_main import NewsBot2System

# Initialize system
newsbot = NewsBot2System()
newsbot.initialize_system()

# Process large batch
articles = load_articles_from_database()  # Your article loading function
results = newsbot.analyze_articles(articles, analysis_types=['classification', 'sentiment'])

# Export results
newsbot.export_analysis_results(results, export_format='excel', output_path='analysis_results.xlsx')
```

### Custom Configuration

Create a custom configuration file:

```yaml
# custom_config.yaml
system:
  log_level: "DEBUG"
  max_workers: 8

components:
  classifier:
    confidence_threshold: 0.8
  sentiment_analyzer:
    methods: ["vader", "transformer"]
  topic_modeler:
    num_topics: 15
```

Use it:
```bash
python newsbot_main.py --config custom_config.yaml --analyze articles.csv
```

### API Integration

Use NewsBot 2.0 programmatically:

```python
from newsbot_main import NewsBot2System

# Initialize
newsbot = NewsBot2System()
newsbot.initialize_system()

# Analyze single article
article = {
    'text': 'Your news article text here',
    'category': 'unknown'
}

result = newsbot.analyze_articles([article])

# Process natural language query
query_result = newsbot.process_natural_language_query(
    "What is the sentiment of this article?"
)

print(query_result)
```

### Performance Optimization

For better performance with large datasets:

1. **Enable multiprocessing**:
   ```yaml
   performance:
     enable_multiprocessing: true
     max_workers: 8
   ```

2. **Use batch processing**:
   ```python
   # Process in chunks
   chunk_size = 100
   for i in range(0, len(articles), chunk_size):
       chunk = articles[i:i+chunk_size]
       results = newsbot.analyze_articles(chunk)
   ```

3. **Cache results**:
   ```yaml
   caching:
     enable_redis: true
     cache_ttl_hours: 24
   ```

## FAQ

### General Questions

**Q: What types of news articles work best with NewsBot 2.0?**
A: NewsBot 2.0 is trained on BBC News data and works best with professional news articles in English. It supports 5 categories: business, entertainment, politics, sport, and technology.

**Q: Can I analyze articles in languages other than English?**
A: Yes! NewsBot 2.0 supports 50+ languages with automatic language detection and translation capabilities.

**Q: How accurate is the classification?**
A: The classification system achieves 97.5% accuracy on the BBC News dataset. Accuracy may vary with different types of content.

**Q: Can I add my own categories?**
A: Currently, the system uses the 5 BBC News categories. Custom categories require retraining the classification models.

### Technical Questions

**Q: How much memory does NewsBot 2.0 need?**
A: Minimum 4GB RAM, but 8GB is recommended for optimal performance. Large transformer models may require more memory.

**Q: Can I run NewsBot 2.0 on a server?**
A: Yes, NewsBot 2.0 is designed for both desktop and server deployment. See the technical documentation for server configuration.

**Q: Is internet connection required?**
A: Some features like translation services and downloading new models require internet. Core analysis can work offline once models are downloaded.

**Q: Can I integrate NewsBot 2.0 with my existing applications?**
A: Yes, NewsBot 2.0 provides a Python API for integration. See the advanced usage section for examples.

### Data Questions

**Q: What format should my news articles be in?**
A: Articles should be in CSV format with 'text' and 'category' columns. See the tutorials for examples.

**Q: Can I analyze historical news data?**
A: Yes, NewsBot 2.0 can analyze news articles from any time period. Some features like temporal sentiment tracking work better with timestamp information.

**Q: How do I handle very long articles?**
A: The system can process articles of any length. Very long articles (>10,000 words) may take longer to process.

## Troubleshooting

### Common Issues

**Issue**: System runs out of memory
**Solution**: 
- Reduce batch size in configuration
- Close other applications
- Use a machine with more RAM
- Enable memory management settings

**Issue**: Processing is very slow
**Solution**:
- Enable multiprocessing in configuration
- Use GPU acceleration if available
- Process articles in smaller batches
- Check internet connection for model downloads

**Issue**: Translation not working
**Solution**:
- Check internet connection
- Verify API keys are configured
- Try different translation services
- Check rate limits

**Issue**: Classification accuracy seems low
**Solution**:
- Ensure articles are in supported categories
- Check article quality and language
- Verify the model is properly loaded
- Consider retraining with domain-specific data

### Error Messages

**"Model not found"**
- Download required models: `python -c "import spacy; spacy.cli.download('en_core_web_sm')"`

**"API key not configured"**
- Set up API keys in `config/api_keys.txt` or environment variables

**"Insufficient memory"**
- Reduce batch size or use a machine with more RAM

**"Network connection error"**
- Check internet connection and proxy settings

### Getting Help

1. **Check the logs**: Look at `newsbot2.log` for detailed error information
2. **Review configuration**: Ensure all settings are correct
3. **Test with sample data**: Try the included BBC News dataset first
4. **Check system requirements**: Verify your system meets minimum requirements

## Tips and Best Practices

### For Best Results

1. **Start Small**: Begin with a small dataset to understand the system
2. **Quality Input**: Use well-written, professional news articles
3. **Regular Updates**: Keep models and dependencies updated
4. **Monitor Performance**: Use system status checks to monitor health
5. **Backup Data**: Keep backups of your analysis results

### Optimization Tips

1. **Batch Processing**: Process multiple articles together for efficiency
2. **Caching**: Enable caching for repeated analyses
3. **Resource Management**: Monitor memory and CPU usage
4. **Network Optimization**: Use local models when possible

### Analysis Best Practices

1. **Understand Confidence Scores**: Pay attention to confidence levels in results
2. **Cross-Validate Results**: Use multiple analysis methods for important decisions
3. **Consider Context**: Remember that AI analysis has limitations
4. **Iterative Improvement**: Refine your approach based on results

### Security Considerations

1. **Protect API Keys**: Never share or commit API keys to version control
2. **Validate Input**: Ensure input data is from trusted sources
3. **Regular Updates**: Keep the system updated for security patches
4. **Access Control**: Limit access to sensitive news data

---

## Conclusion

NewsBot 2.0 is a powerful tool for news analysis that can help you understand, organize, and extract insights from news content. Start with the tutorials, experiment with your own data, and gradually explore the advanced features.

For technical support, refer to the technical documentation or troubleshooting sections. Happy analyzing!

---

*User Guide Version 2.0 - Last Updated: December 2024*