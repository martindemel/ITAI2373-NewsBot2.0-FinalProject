#!/usr/bin/env python3
"""
NewsBot 2.0 Configuration Management
Centralized configuration for all system components
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import yaml

@dataclass
class DatabaseConfig:
    """Database configuration"""
    data_path: str = "data/processed/newsbot_dataset.csv"
    models_path: str = "data/models/"
    results_path: str = "data/results/"
    backup_path: str = "data/backups/"

@dataclass
class APIConfig:
    """API configuration for external services"""
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 1000
    openai_temperature: float = 0.1
    
    google_translate_api_key: Optional[str] = None
    azure_translate_key: Optional[str] = None
    
    huggingface_api_key: Optional[str] = None
    
    # Rate limiting
    api_rate_limit: int = 60  # requests per minute
    api_timeout: int = 30  # seconds

@dataclass
class ModelConfig:
    """ML/NLP model configuration"""
    # Classification models
    classification_model: str = "best_classifier.pkl"
    classification_threshold: float = 0.7
    
    # Sentiment analysis
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_batch_size: int = 32
    
    # Topic modeling
    topic_model_type: str = "lda"  # lda, nmf, or gensim
    num_topics: int = 10
    topic_coherence_threshold: float = 0.4
    
    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    similarity_threshold: float = 0.7
    
    # Summarization
    summarization_model: str = "facebook/bart-large-cnn"
    max_summary_length: int = 150
    min_summary_length: int = 50
    
    # NER (Named Entity Recognition)
    ner_model: str = "en_core_web_sm"
    ner_confidence_threshold: float = 0.8
    
    # Translation
    translation_model: str = "Helsinki-NLP/opus-mt-en-es"
    translation_confidence_threshold: float = 0.8

@dataclass
class RealTimeConfig:
    """Real-time processing configuration"""
    enabled: bool = True
    article_queue_size: int = 1000
    max_processed_articles: int = 10000
    
    # RSS feeds for live monitoring
    rss_feeds: list = None
    feed_check_interval: int = 30  # seconds
    request_timeout: int = 10  # seconds
    
    # Event detection thresholds
    sentiment_spike_threshold: float = 0.8
    trend_detection_window_hours: int = 1
    breaking_news_keyword_threshold: int = 2
    
    def __post_init__(self):
        if self.rss_feeds is None:
            self.rss_feeds = [
                'https://feeds.bbci.co.uk/news/rss.xml',
                'https://rss.cnn.com/rss/edition.rss',
                'https://feeds.reuters.com/reuters/topNews'
            ]

@dataclass
class SystemConfig:
    """System-wide configuration"""
    debug: bool = False
    log_level: str = "INFO"
    max_workers: int = 4
    cache_enabled: bool = True
    cache_size: int = 1000
    
    # Performance
    batch_size: int = 32
    max_text_length: int = 5000
    max_articles_per_query: int = 100
    
    # Security
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    session_timeout: int = 3600  # seconds
    
    # Features
    enable_multilingual: bool = True
    enable_topic_modeling: bool = True
    enable_advanced_analytics: bool = True
    enable_export: bool = True

@dataclass
class WebConfig:
    """Web application configuration"""
    host: str = "0.0.0.0"
    port: int = 5000
    secret_key: str = "newsbot-2-0-production-key"
    
    # Flask settings
    flask_debug: bool = False
    flask_testing: bool = False
    
    # Upload settings
    upload_folder: str = "uploads/"
    max_file_size: int = 16 * 1024 * 1024  # 16MB
    allowed_extensions: set = None
    
    # Session settings
    permanent_session_lifetime: int = 3600
    
    # Security headers
    enable_csrf: bool = True
    enable_security_headers: bool = True

class NewsBot2Config:
    """
    Central configuration manager for NewsBot 2.0
    Handles loading, validation, and management of all configuration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), 'newsbot_config.yaml')
        
        # Initialize default configurations
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.model = ModelConfig()
        self.realtime = RealTimeConfig()
        self.system = SystemConfig()
        self.web = WebConfig()
        
        # Load configuration from file and environment
        self._load_configuration()
        self._load_environment_variables()
        self._validate_configuration()
        
        # Setup logging
        self._setup_logging()
    
    def _load_configuration(self):
        """Load configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Update configurations from file
                if 'database' in config_data:
                    self._update_dataclass(self.database, config_data['database'])
                if 'api' in config_data:
                    self._update_dataclass(self.api, config_data['api'])
                if 'model' in config_data:
                    self._update_dataclass(self.model, config_data['model'])
                if 'system' in config_data:
                    self._update_dataclass(self.system, config_data['system'])
                if 'web' in config_data:
                    self._update_dataclass(self.web, config_data['web'])
                
                logging.info(f"Configuration loaded from {self.config_path}")
            else:
                logging.info("No configuration file found, using defaults")
                
        except Exception as e:
            logging.warning(f"Failed to load configuration file: {e}")
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        # API Keys
        self.api.openai_api_key = os.getenv('OPENAI_API_KEY', self.api.openai_api_key)
        self.api.google_translate_api_key = os.getenv('GOOGLE_TRANSLATE_API_KEY', self.api.google_translate_api_key)
        self.api.azure_translate_key = os.getenv('AZURE_TRANSLATE_KEY', self.api.azure_translate_key)
        self.api.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY', self.api.huggingface_api_key)
        
        # System settings
        self.system.debug = os.getenv('DEBUG', str(self.system.debug)).lower() == 'true'
        self.system.log_level = os.getenv('LOG_LEVEL', self.system.log_level)
        
        # Web settings
        self.web.host = os.getenv('HOST', self.web.host)
        self.web.port = int(os.getenv('PORT', self.web.port))
        self.web.secret_key = os.getenv('SECRET_KEY', self.web.secret_key)
        self.web.flask_debug = os.getenv('FLASK_DEBUG', str(self.web.flask_debug)).lower() == 'true'
        
        # Paths
        self.database.data_path = os.getenv('DATA_PATH', self.database.data_path)
        self.database.models_path = os.getenv('MODELS_PATH', self.database.models_path)
        
        logging.info("Environment variables loaded")
    
    def _validate_configuration(self):
        """Validate configuration values"""
        # Check required paths exist
        required_dirs = [
            self.database.models_path,
            self.database.results_path,
            self.web.upload_folder
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Validate API configuration
        if self.api.openai_api_key:
            if not self.api.openai_api_key.startswith('sk-'):
                logging.warning("OpenAI API key format may be invalid")
        
        # Validate model parameters
        if self.model.num_topics < 2:
            self.model.num_topics = 2
            logging.warning("Number of topics must be at least 2, setting to 2")
        
        if self.model.classification_threshold < 0 or self.model.classification_threshold > 1:
            self.model.classification_threshold = 0.7
            logging.warning("Classification threshold must be between 0 and 1, setting to 0.7")
        
        # Set upload allowed extensions
        if self.web.allowed_extensions is None:
            self.web.allowed_extensions = {'txt', 'csv', 'json', 'pdf'}
        
        logging.info("Configuration validation completed")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.system.log_level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/newsbot.log', mode='a') if os.path.exists('logs') else logging.NullHandler()
            ]
        )
    
    def _update_dataclass(self, dataclass_instance, update_dict):
        """Update dataclass instance with dictionary values"""
        for key, value in update_dict.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
    
    def save_configuration(self, file_path: Optional[str] = None):
        """Save current configuration to file"""
        save_path = file_path or self.config_path
        
        config_data = {
            'database': asdict(self.database),
            'api': {k: v for k, v in asdict(self.api).items() if not k.endswith('_key')},  # Don't save API keys
            'model': asdict(self.model),
            'realtime': asdict(self.realtime),
            'system': asdict(self.system),
            'web': {k: v for k, v in asdict(self.web).items() if k != 'secret_key'}  # Don't save secret key
        }
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            logging.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for specific model type"""
        base_config = {
            'models_path': self.database.models_path,
            'cache_enabled': self.system.cache_enabled,
            'batch_size': self.model.sentiment_batch_size,
            'max_text_length': self.system.max_text_length
        }
        
        if model_type == 'classification':
            return {
                **base_config,
                'model_path': os.path.join(self.database.models_path, self.model.classification_model),
                'threshold': self.model.classification_threshold
            }
        elif model_type == 'sentiment':
            return {
                **base_config,
                'model_name': self.model.sentiment_model,
                'batch_size': self.model.sentiment_batch_size
            }
        elif model_type == 'topic':
            return {
                **base_config,
                'model_type': self.model.topic_model_type,
                'num_topics': self.model.num_topics,
                'coherence_threshold': self.model.topic_coherence_threshold
            }
        elif model_type == 'embeddings':
            return {
                **base_config,
                'model_name': self.model.embedding_model,
                'dimension': self.model.embedding_dimension,
                'similarity_threshold': self.model.similarity_threshold
            }
        elif model_type == 'summarization':
            return {
                **base_config,
                'model_name': self.model.summarization_model,
                'max_length': self.model.max_summary_length,
                'min_length': self.model.min_summary_length
            }
        elif model_type == 'ner':
            return {
                **base_config,
                'model_name': self.model.ner_model,
                'confidence_threshold': self.model.ner_confidence_threshold
            }
        else:
            return base_config
    
    def get_api_config(self, service: str) -> Dict[str, Any]:
        """Get API configuration for specific service"""
        base_config = {
            'timeout': self.api.api_timeout,
            'rate_limit': self.api.api_rate_limit
        }
        
        if service == 'openai':
            return {
                **base_config,
                'api_key': self.api.openai_api_key,
                'model': self.api.openai_model,
                'max_tokens': self.api.openai_max_tokens,
                'temperature': self.api.openai_temperature
            }
        elif service == 'google_translate':
            return {
                **base_config,
                'api_key': self.api.google_translate_api_key
            }
        elif service == 'azure_translate':
            return {
                **base_config,
                'api_key': self.api.azure_translate_key
            }
        elif service == 'huggingface':
            return {
                **base_config,
                'api_key': self.api.huggingface_api_key
            }
        else:
            return base_config
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a specific feature is enabled"""
        feature_flags = {
            'multilingual': self.system.enable_multilingual,
            'topic_modeling': self.system.enable_topic_modeling,
            'advanced_analytics': self.system.enable_advanced_analytics,
            'export': self.system.enable_export,
            'rate_limiting': self.system.enable_rate_limiting,
            'openai': bool(self.api.openai_api_key),
            'google_translate': bool(self.api.google_translate_api_key)
        }
        
        return feature_flags.get(feature, False)
    
    def get_flask_config(self) -> Dict[str, Any]:
        """Get Flask application configuration"""
        return {
            'SECRET_KEY': self.web.secret_key,
            'DEBUG': self.web.flask_debug,
            'TESTING': self.web.flask_testing,
            'UPLOAD_FOLDER': self.web.upload_folder,
            'MAX_CONTENT_LENGTH': self.web.max_file_size,
            'PERMANENT_SESSION_LIFETIME': self.web.permanent_session_lifetime,
            'SESSION_TYPE': 'filesystem'
        }
    
    def get(self, key: str, default=None):
        """Dictionary-like get method for configuration access"""
        # Map common configuration keys to their actual attributes
        key_mapping = {
            'log_level': self.system.log_level,
            'debug': self.system.debug,
            'cache_enabled': self.system.cache_enabled,
            'max_text_length': self.system.max_text_length,
            'auto_load_models': True,  # Default behavior
            'data_path': self.database.data_path,
            'models_path': self.database.models_path,
            'port': self.web.port,
            'host': self.web.host,
            'workers': 1  # Default worker count
        }
        
        return key_mapping.get(key, default)
    
    def get_component_config(self, component_name: str) -> Dict[str, Any]:
        """Get configuration for a specific component"""
        # Return component-specific configuration
        base_config = {
            'debug': self.system.debug,
            'cache_enabled': self.system.cache_enabled,
            'data_path': self.database.data_path,
            'models_path': self.database.models_path,
            'max_text_length': self.system.max_text_length
        }
        
        if component_name == 'classifier':
            return {
                **base_config,
                'model_path': os.path.join(self.database.models_path, 'best_classifier.pkl'),
                'threshold': self.model.classification_threshold
            }
        elif component_name == 'sentiment':
            return {
                **base_config,
                'model_name': self.model.sentiment_model,
                'batch_size': self.model.sentiment_batch_size
            }
        elif component_name == 'topic_modeler':
            return {
                **base_config,
                'model_path': os.path.join(self.database.models_path, 'topic_model.pkl'),
                'num_topics': self.model.num_topics
            }
        elif component_name == 'realtime_processor':
            return {
                **base_config,
                'enabled': self.realtime.enabled,
                'article_queue_size': self.realtime.article_queue_size,
                'max_processed_articles': self.realtime.max_processed_articles,
                'rss_feeds': self.realtime.rss_feeds,
                'feed_check_interval': self.realtime.feed_check_interval,
                'request_timeout': self.realtime.request_timeout,
                'sentiment_spike_threshold': self.realtime.sentiment_spike_threshold,
                'trend_detection_window_hours': self.realtime.trend_detection_window_hours,
                'breaking_news_keyword_threshold': self.realtime.breaking_news_keyword_threshold
            }
        else:
            return base_config
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"""
NewsBot 2.0 Configuration:
- Database Path: {self.database.data_path}
- Models Path: {self.database.models_path}
- OpenAI Enabled: {bool(self.api.openai_api_key)}
- Debug Mode: {self.system.debug}
- Log Level: {self.system.log_level}
- Web Port: {self.web.port}
- Features: {', '.join([f for f in ['multilingual', 'topic_modeling', 'advanced_analytics'] if self.is_feature_enabled(f)])}
        """.strip()

# Global configuration instance
_global_config = None

def get_config(config_path: Optional[str] = None) -> NewsBot2Config:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = NewsBot2Config(config_path)
    return _global_config

def reload_config(config_path: Optional[str] = None):
    """Reload global configuration"""
    global _global_config
    _global_config = NewsBot2Config(config_path)
    return _global_config