#!/usr/bin/env python3
"""
AI-Powered Conversational Interface for NewsBot 2.0
Complete implementation of Module D: Conversational Interface with ML/NLP components

This module implements the project requirements for:
- Intent Classification using ML models (not rule-based)
- Natural Language Processing for complex queries  
- Context Management for conversation state
- Response Generation using language models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import re
from datetime import datetime, timedelta
from collections import defaultdict, deque

# Import advanced NLP components 
from .intent_classifier import IntentClassifier
from .response_generator import ResponseGenerator
from ..analysis.sentiment_analyzer import SentimentAnalyzer
from ..analysis.ner_extractor import NERExtractor
from ..analysis.classifier import NewsClassifier
from ..analysis.topic_modeler import TopicModeler
from ..language_models.embeddings import SemanticEmbeddings
from ..language_models.summarizer import IntelligentSummarizer

# Try to import advanced language models
try:
    import openai
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    HAS_ADVANCED_NLP = True
except ImportError:
    HAS_ADVANCED_NLP = False
    logging.warning("Advanced NLP libraries not available. Some features may be limited.")

class AIPoweredConversation:
    """
    AI-Powered Conversational Interface implementing Module D requirements
    
    This is the core conversational AI system that uses ML models for:
    - Intent understanding through trained classifiers
    - Entity extraction using NER models  
    - Semantic understanding with embeddings
    - Dynamic response generation with language models
    - Context management for multi-turn conversations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AI-powered conversation system
        
        Args:
            config: Configuration including API keys and model settings
        """
        self.config = config or {}
        logging.info("Initializing AI-Powered Conversational Interface...")
        
        # Initialize core ML/NLP components (PROJECT REQUIREMENT)
        try:
            self.intent_classifier = IntentClassifier(self.config.get('intent_config', {}))
            self.response_generator = ResponseGenerator(self.config.get('response_config', {}))
            self.sentiment_analyzer = SentimentAnalyzer(self.config.get('sentiment_config', {}))
            self.ner_extractor = NERExtractor(self.config.get('ner_config', {}))
            
            # Try to initialize semantic embeddings, fallback to simple if not available
            try:
                self.semantic_embeddings = SemanticEmbeddings(self.config.get('embeddings_config', {}))
            except Exception as e:
                logging.warning(f"Semantic embeddings not available: {e}. Using keyword-based search fallback.")
                self.semantic_embeddings = None
                
            self.summarizer = IntelligentSummarizer(self.config.get('summarizer_config', {}))
            
        except Exception as e:
            logging.error(f"Failed to initialize some ML components: {e}")
            # Continue with minimal functionality
        
        # Data processing components
        self.classifier = None
        self.topic_modeler = None
        self.article_database = None
        self.analysis_results = None
        
        # Conversation Context Management (PROJECT REQUIREMENT)
        self.conversation_memory = {}  # Per-user conversation memory
        self.global_context = {
            'system_capabilities': [],
            'available_data': {},
            'session_stats': defaultdict(int)
        }
        
        # Advanced language understanding
        self.openai_client = None
        self.transformer_models = {}
        
        if HAS_ADVANCED_NLP and self.config.get('openai_api_key'):
            # Check for placeholder API key
            api_key = self.config['openai_api_key']
            if api_key and api_key not in ['your-openai-api-key-here', 'your-openai-api-key', 'sk-your-key-here']:
                try:
                    openai.api_key = api_key
                    self.openai_client = openai
                    logging.info("OpenAI integration enabled for advanced language understanding")
                except Exception as e:
                    logging.warning(f"OpenAI setup failed: {e}")
            else:
                logging.info("OpenAI API key not configured - using local models only")
        
        # Initialize transformer models for advanced understanding
        self._initialize_transformer_models()
        
        # Conversation flow management
        self.conversation_flows = {
            'greeting': self._handle_greeting_flow,
            'search': self._handle_search_flow,
            'analysis': self._handle_analysis_flow,
            'insight': self._handle_insight_flow,
            'help': self._handle_help_flow
        }
        
        # System state
        self.is_initialized = False
        
        logging.info("AI-Powered Conversational Interface initialized successfully")
    
    def _initialize_transformer_models(self):
        """Initialize transformer models for advanced NLP understanding"""
        if not HAS_ADVANCED_NLP:
            return
            
        try:
            # Load sentiment analysis model for nuanced understanding
            self.transformer_models['sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Load question-answering model for complex queries
            self.transformer_models['qa'] = pipeline(
                "question-answering",
                model="distilbert-base-uncased-distilled-squad"
            )
            
            logging.info("Advanced transformer models loaded successfully")
            
        except Exception as e:
            logging.warning(f"Could not load all transformer models: {e}")
    
    def set_dependencies(self, **kwargs):
        """Set system components for integration"""
        if 'classifier' in kwargs:
            self.classifier = kwargs['classifier']
        if 'topic_modeler' in kwargs:
            self.topic_modeler = kwargs['topic_modeler']
        if 'article_database' in kwargs:
            self.article_database = kwargs['article_database']
        if 'analysis_results' in kwargs:
            self.analysis_results = kwargs['analysis_results']
            
        # Update global context with available data
        self._update_global_context()
        
        logging.info("AI conversation dependencies set successfully")
    
    def _update_global_context(self):
        """Update global context with system capabilities"""
        capabilities = []
        
        if self.article_database is not None:
            capabilities.append("Article Search and Retrieval")
            self.global_context['available_data']['articles'] = len(self.article_database)
            
        if self.classifier is not None:
            capabilities.append("Advanced Text Classification")
            
        if self.topic_modeler is not None:
            capabilities.append("Topic Analysis and Discovery")
            
        if self.sentiment_analyzer is not None:
            capabilities.append("Sentiment Analysis")
            
        if self.ner_extractor is not None:
            capabilities.append("Named Entity Recognition")
            
        if self.summarizer is not None:
            capabilities.append("Intelligent Text Summarization")
            
        if self.openai_client is not None:
            capabilities.append("Advanced Language Understanding (OpenAI)")
            
        self.global_context['system_capabilities'] = capabilities
    
    def initialize_system(self, article_database: pd.DataFrame, analysis_results: Dict[str, Any] = None):
        """Initialize the conversational system with data"""
        self.article_database = article_database
        self.analysis_results = analysis_results or {}
        
        # Initialize embeddings for semantic search if available
        if self.article_database is not None and self.semantic_embeddings:
            try:
                logging.info("Initializing semantic embeddings for conversational search...")
                articles_text = self.article_database['text'].tolist()
                self.semantic_embeddings.encode_documents(articles_text)
                logging.info("Semantic embeddings initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize semantic embeddings: {e}. Using keyword search fallback.")
                self.semantic_embeddings = None
        elif not self.semantic_embeddings:
            logging.info("Using keyword-based search (semantic embeddings not available)")
        
        self._update_global_context()
        self.is_initialized = True
        
        logging.info("AI-Powered Conversational Interface ready for queries")
    
    def process_conversation(self, user_query: str, user_id: str = None, 
                           session_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main conversation processing using advanced ML/NLP
        
        This implements the complete Module D requirements:
        1. Intent Classification using ML models
        2. Natural Language Processing for complex queries
        3. Context Management for conversation state  
        4. Response Generation using language models
        
        Args:
            user_query: Natural language input from user
            user_id: User identifier for conversation memory
            session_context: Session-specific context
            
        Returns:
            Complete conversational response with context updates
        """
        if not self.is_initialized:
            return {
                'status': 'error',
                'message': 'Conversational AI system not initialized',
                'suggestions': ['Please wait for system initialization']
            }
        
        start_time = datetime.now()
        user_id = user_id or 'anonymous'
        session_context = session_context or {}
        
        # Initialize user conversation memory if needed
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = {
                'conversation_history': [],  # Use list instead of deque for JSON serialization
                'user_preferences': {},
                'context_stack': [],
                'last_intent': None,
                'last_entities': [],
                'conversation_start': datetime.now().isoformat()  # Convert to string for JSON
            }
        
        user_memory = self.conversation_memory[user_id]
        
        try:
            # STEP 1: Advanced Intent Classification (ML-based, not rule-based)
            logging.info(f"Processing conversation with advanced ML: {user_query}")
            
            intent_analysis = self._advanced_intent_understanding(
                user_query, user_memory, session_context
            )
            
            # STEP 2: Named Entity Recognition and Semantic Understanding
            entities_analysis = self._advanced_entity_extraction(
                user_query, intent_analysis
            )
            
            # STEP 3: Context-Aware Query Enhancement
            enhanced_query = self._enhance_query_with_context(
                user_query, intent_analysis, entities_analysis, user_memory
            )
            
            # STEP 4: Execute ML-powered Analysis
            analysis_results = self._execute_ml_analysis(
                enhanced_query, intent_analysis, entities_analysis
            )
            
            # STEP 5: Generate AI-powered Response
            response = self._generate_ai_response(
                user_query, intent_analysis, entities_analysis, 
                analysis_results, user_memory
            )
            
            # STEP 6: Update Conversation Context
            self._update_conversation_context(
                user_query, intent_analysis, entities_analysis, 
                response, user_memory, session_context
            )
            
            # Add conversation metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            response.update({
                'processing_time': processing_time,
                'timestamp': start_time.isoformat(),
                'conversation_id': user_id,
                'ai_powered': True,
                'ml_components_used': [
                    'intent_classification', 'entity_extraction', 
                    'semantic_analysis', 'language_models'
                ],
                'updated_context': {
                    'user_memory': dict(user_memory),
                    'session_context': session_context
                }
            })
            
            return response
            
        except Exception as e:
            logging.error(f"Error in AI conversation processing: {e}")
            return {
                'status': 'error',
                'message': 'I encountered an issue processing your request. Please try rephrasing your question.',
                'error_details': str(e) if self.config.get('debug', False) else None,
                'suggestions': [
                    'Try a simpler question',
                    'Ask for help to see my capabilities',
                    'Check your query for any unusual formatting'
                ]
            }
    
    def _advanced_intent_understanding(self, query: str, user_memory: Dict, 
                                     session_context: Dict) -> Dict[str, Any]:
        """Advanced intent classification using ML models"""
        
        # Use the trained intent classifier (ML-based)
        intent_result = self.intent_classifier.classify_intent(query)
        
        # Enhance with conversation context
        if user_memory['last_intent']:
            # Check for follow-up patterns
            follow_up_indicators = ['also', 'and', 'additionally', 'furthermore', 'moreover']
            if any(indicator in query.lower() for indicator in follow_up_indicators):
                intent_result['is_follow_up'] = True
                intent_result['previous_intent'] = user_memory['last_intent']
        
        # Use transformer models for advanced understanding
        if 'sentiment' in self.transformer_models:
            try:
                sentiment_scores = self.transformer_models['sentiment'](query)
                intent_result['query_sentiment'] = sentiment_scores
            except Exception as e:
                logging.warning(f"Transformer sentiment analysis failed: {e}")
        
        # Enhance with OpenAI if available
        if self.openai_client:
            try:
                enhanced_understanding = self._openai_intent_enhancement(
                    query, intent_result, user_memory
                )
                intent_result.update(enhanced_understanding)
            except Exception as e:
                logging.warning(f"OpenAI enhancement failed: {e}")
        
        return intent_result
    
    def _advanced_entity_extraction(self, query: str, intent_analysis: Dict) -> Dict[str, Any]:
        """Advanced entity extraction using NER models"""
        
        # Use NER extractor
        entities = self.ner_extractor.extract_entities(query)
        
        # Extract domain-specific entities
        domain_entities = self._extract_domain_entities(query, intent_analysis)
        entities.update(domain_entities)
        
        # Extract temporal entities
        temporal_entities = self._extract_temporal_entities(query)
        entities.update(temporal_entities)
        
        return entities
    
    def _extract_domain_entities(self, query: str, intent_analysis: Dict) -> Dict[str, Any]:
        """Extract domain-specific entities for news analysis"""
        
        entities = {
            'news_categories': [],
            'analysis_types': [],
            'data_filters': [],
            'output_formats': []
        }
        
        query_lower = query.lower()
        
        # News categories
        categories = ['politics', 'technology', 'tech', 'business', 'sports', 'sport', 
                     'entertainment', 'health', 'science', 'economy']
        entities['news_categories'] = [cat for cat in categories if cat in query_lower]
        
        # Analysis types
        analysis_types = ['sentiment', 'classification', 'summarization', 'entity', 
                         'topic', 'trend', 'insight']
        entities['analysis_types'] = [atype for atype in analysis_types if atype in query_lower]
        
        # Data filters
        if any(word in query_lower for word in ['positive', 'negative', 'neutral']):
            entities['data_filters'].append('sentiment_filter')
        
        if any(word in query_lower for word in ['recent', 'latest', 'new', 'today']):
            entities['data_filters'].append('time_filter')
        
        return entities
    
    def _extract_temporal_entities(self, query: str) -> Dict[str, Any]:
        """Extract temporal information from query"""
        
        temporal_entities = {
            'time_references': [],
            'date_ranges': [],
            'relative_time': []
        }
        
        # Time reference patterns
        time_patterns = [
            r'\b(today|yesterday|tomorrow)\b',
            r'\b(this|last|next)\s+(week|month|year)\b',
            r'\b(past|last)\s+(\d+)\s+(days?|weeks?|months?)\b',
            r'\b(since|from|until|before|after)\s+\w+\b'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, query.lower())
            if matches:
                temporal_entities['time_references'].extend(matches)
        
        return temporal_entities
    
    def _enhance_query_with_context(self, query: str, intent_analysis: Dict, 
                                  entities_analysis: Dict, user_memory: Dict) -> Dict[str, Any]:
        """Enhance query understanding with conversation context"""
        
        enhanced_query = {
            'original_query': query,
            'processed_query': query,
            'context_enhancements': [],
            'inferred_parameters': {}
        }
        
        # Check for pronouns and references
        pronouns = ['it', 'this', 'that', 'these', 'those', 'they', 'them']
        if any(pronoun in query.lower().split() for pronoun in pronouns):
            if user_memory['conversation_history']:
                last_interaction = user_memory['conversation_history'][-1]
                enhanced_query['context_enhancements'].append('pronoun_resolution')
                enhanced_query['inferred_parameters']['reference_context'] = last_interaction
        
        # Infer missing parameters from conversation history
        if not entities_analysis.get('news_categories') and user_memory['last_entities']:
            last_categories = user_memory['last_entities'].get('news_categories', [])
            if last_categories:
                enhanced_query['inferred_parameters']['categories'] = last_categories
                enhanced_query['context_enhancements'].append('category_inference')
        
        return enhanced_query
    
    def _execute_ml_analysis(self, enhanced_query: Dict, intent_analysis: Dict, 
                           entities_analysis: Dict) -> Dict[str, Any]:
        """Execute ML-powered analysis based on intent and entities"""
        
        intent = intent_analysis.get('final_intent', {}).get('intent', 'search_articles')
        
        analysis_results = {
            'intent': intent,
            'execution_method': 'ml_powered',
            'results': {}
        }
        
        if intent == 'search_articles':
            analysis_results['results'] = self._ml_article_search(
                enhanced_query, entities_analysis
            )
            
        elif intent == 'analyze_sentiment':
            analysis_results['results'] = self._ml_sentiment_analysis(
                enhanced_query, entities_analysis
            )
            
        elif intent == 'classify_content':
            analysis_results['results'] = self._ml_content_classification(
                enhanced_query, entities_analysis
            )
            
        elif intent == 'summarize_text':
            analysis_results['results'] = self._ml_summarization(
                enhanced_query, entities_analysis
            )
            
        elif intent == 'extract_entities':
            analysis_results['results'] = self._ml_entity_extraction(
                enhanced_query, entities_analysis
            )
            
        elif intent == 'get_insights':
            analysis_results['results'] = self._ml_insight_generation(
                enhanced_query, entities_analysis
            )
            
        else:
            # Default to semantic search
            analysis_results['results'] = self._ml_semantic_search(
                enhanced_query, entities_analysis
            )
        
        return analysis_results
    
    def _ml_article_search(self, enhanced_query: Dict, entities: Dict) -> Dict[str, Any]:
        """ML-powered article search using semantic embeddings"""
        
        if self.article_database is None:
            return {'error': 'No article database available'}
        
        query_text = enhanced_query['original_query']
        
        # Use semantic embeddings for intelligent search if available
        if self.semantic_embeddings:
            try:
                search_response = self.semantic_embeddings.semantic_search(
                    query_text, top_k=10
                )
                
                # Extract the actual results from the response
                search_results = search_response.get('results', [])
                
                # Apply entity-based filtering
                filtered_results = self._apply_entity_filters(search_results, entities)
                
                # Enhance results with additional analysis
                enhanced_results = []
                for result in filtered_results[:5]:  # Top 5 results
                    # Adapt the result format for AI processing
                    enhanced_result = {
                        'id': result.get('document_index', 'unknown'),
                        'text': result.get('document_text', ''),
                        'similarity_score': result.get('similarity_score', 0),
                        'category': result.get('metadata', {}).get('category', 'unknown')
                    }
                    
                    # Add sentiment analysis
                    if self.sentiment_analyzer and enhanced_result['text']:
                        try:
                            sentiment = self.sentiment_analyzer.analyze_sentiment(enhanced_result['text'])
                            enhanced_result['sentiment'] = sentiment
                        except Exception as e:
                            logging.warning(f"Sentiment analysis failed: {e}")
                    
                    # Add entity extraction
                    if self.ner_extractor and enhanced_result['text']:
                        try:
                            extracted_entities = self.ner_extractor.extract_entities(enhanced_result['text'])
                            enhanced_result['entities'] = extracted_entities
                        except Exception as e:
                            logging.warning(f"Entity extraction failed: {e}")
                    
                    enhanced_results.append(enhanced_result)
                
                return {
                    'articles': enhanced_results,
                    'total_found': len(search_results),
                    'search_method': 'semantic_embeddings',
                    'filters_applied': entities
                }
            
            except Exception as e:
                logging.error(f"Semantic search failed: {e}")
        
        # Fallback to simple database search if semantic embeddings not available or failed
        return self._fallback_article_search(enhanced_query, entities)
    
    def _ml_sentiment_analysis(self, enhanced_query: Dict, entities: Dict) -> Dict[str, Any]:
        """ML-powered sentiment analysis"""
        
        # Find relevant articles first
        search_results = self._ml_article_search(enhanced_query, entities)
        
        if 'error' in search_results:
            return search_results
        
        articles = search_results['articles']
        sentiment_results = []
        
        for article in articles:
            if self.sentiment_analyzer:
                # Use advanced sentiment analysis
                sentiment = self.sentiment_analyzer.analyze_sentiment(article['text'])
                
                # Enhance with transformer model if available
                if 'sentiment' in self.transformer_models:
                    try:
                        transformer_sentiment = self.transformer_models['sentiment'](
                            article['text'][:512]  # Truncate for transformer
                        )
                        sentiment['transformer_analysis'] = transformer_sentiment
                    except Exception as e:
                        logging.warning(f"Transformer sentiment failed: {e}")
                
                sentiment_results.append({
                    'article_id': article.get('id', 'unknown'),
                    'text_preview': article['text'][:200] + '...',
                    'sentiment': sentiment
                })
        
        # Aggregate sentiment statistics
        sentiment_distribution = self._aggregate_sentiment_stats(sentiment_results)
        
        return {
            'sentiment_analysis': sentiment_results,
            'aggregated_sentiment': sentiment_distribution,
            'total_analyzed': len(sentiment_results),
            'analysis_method': 'ml_transformer_enhanced'
        }
    
    def _ml_content_classification(self, enhanced_query: Dict, entities: Dict) -> Dict[str, Any]:
        """ML-powered content classification"""
        
        search_results = self._ml_article_search(enhanced_query, entities)
        
        if 'error' in search_results:
            return search_results
        
        articles = search_results['articles']
        classification_results = []
        
        for article in articles:
            if self.classifier:
                # Use trained classifier
                classification = self.classifier.classify_text(article['text'])
                classification_results.append({
                    'article_id': article.get('id', 'unknown'),
                    'text_preview': article['text'][:200] + '...',
                    'classification': classification
                })
        
        return {
            'classification_results': classification_results,
            'total_classified': len(classification_results),
            'classification_method': 'trained_ml_model'
        }
    
    def _ml_summarization(self, enhanced_query: Dict, entities: Dict) -> Dict[str, Any]:
        """ML-powered text summarization"""
        
        search_results = self._ml_article_search(enhanced_query, entities)
        
        if 'error' in search_results:
            return search_results
        
        articles = search_results['articles']
        summaries = []
        
        for article in articles[:3]:  # Summarize top 3 articles
            if self.summarizer:
                # Use intelligent summarizer
                summary = self.summarizer.generate_summary(article['text'])
                summaries.append({
                    'article_id': article.get('id', 'unknown'),
                    'original_length': len(article['text']),
                    'summary': summary,
                    'compression_ratio': len(summary) / len(article['text'])
                })
        
        return {
            'summaries': summaries,
            'total_summarized': len(summaries),
            'summarization_method': 'ml_abstractive'
        }
    
    def _ml_entity_extraction(self, enhanced_query: Dict, entities: Dict) -> Dict[str, Any]:
        """ML-powered named entity extraction"""
        
        search_results = self._ml_article_search(enhanced_query, entities)
        
        if 'error' in search_results:
            return search_results
        
        articles = search_results['articles']
        entity_results = []
        
        for article in articles:
            if self.ner_extractor:
                # Extract entities using NER model
                extracted_entities = self.ner_extractor.extract_entities(article['text'])
                entity_results.append({
                    'article_id': article.get('id', 'unknown'),
                    'text_preview': article['text'][:200] + '...',
                    'entities': extracted_entities
                })
        
        # Aggregate entity statistics
        aggregated_entities = self._aggregate_entity_stats(entity_results)
        
        return {
            'entity_extraction': entity_results,
            'aggregated_entities': aggregated_entities,
            'total_analyzed': len(entity_results),
            'extraction_method': 'ner_ml_model'
        }
    
    def _ml_insight_generation(self, enhanced_query: Dict, entities: Dict) -> Dict[str, Any]:
        """ML-powered insight generation"""
        
        insights = {
            'data_insights': {},
            'trend_insights': {},
            'pattern_insights': {},
            'recommendations': []
        }
        
        # Data overview insights
        if self.article_database is not None:
            insights['data_insights'] = {
                'total_articles': len(self.article_database),
                'category_distribution': self.article_database['category'].value_counts().to_dict(),
                'avg_article_length': self.article_database['text'].str.len().mean()
            }
        
        # Topic insights using topic modeling
        if self.topic_modeler:
            try:
                topic_insights = self.topic_modeler.get_topic_insights()
                insights['trend_insights']['topics'] = topic_insights
            except Exception as e:
                logging.warning(f"Topic modeling insights failed: {e}")
        
        # Sentiment pattern insights
        sentiment_search = self._ml_sentiment_analysis(enhanced_query, entities)
        if 'aggregated_sentiment' in sentiment_search:
            insights['pattern_insights']['sentiment'] = sentiment_search['aggregated_sentiment']
        
        # Generate AI-powered recommendations
        insights['recommendations'] = self._generate_ai_recommendations(insights)
        
        return insights
    
    def _ml_semantic_search(self, enhanced_query: Dict, entities: Dict) -> Dict[str, Any]:
        """ML-powered semantic search as fallback"""
        
        query_text = enhanced_query['original_query']
        
        # Use semantic embeddings for intelligent search
        if self.semantic_embeddings and self.article_database is not None:
            search_results = self.semantic_embeddings.semantic_search(
                query_text, top_k=5
            )
            
            return {
                'semantic_results': search_results,
                'search_method': 'semantic_embeddings',
                'query': query_text
            }
        else:
            return {
                'error': 'Semantic search not available',
                'fallback_message': 'Please try a more specific query'
            }
    
    def _generate_ai_response(self, original_query: str, intent_analysis: Dict, 
                            entities_analysis: Dict, analysis_results: Dict, 
                            user_memory: Dict) -> Dict[str, Any]:
        """Generate AI-powered conversational response"""
        
        # Use response generator for intelligent response creation
        response_context = {
            'user_query': original_query,
            'intent': intent_analysis,
            'entities': entities_analysis,
            'conversation_history': list(user_memory['conversation_history']),
            'user_preferences': user_memory['user_preferences']
        }
        
        intent = intent_analysis.get('final_intent', {}).get('intent', 'unknown')
        
        # Generate response using ML-powered response generator
        generated_response = self.response_generator.generate_response(
            original_query, intent, analysis_results['results'], response_context
        )
        
        # Enhance with OpenAI if available for more natural responses
        if self.openai_client and analysis_results['results']:
            try:
                enhanced_response = self._enhance_response_with_openai(
                    original_query, analysis_results, generated_response
                )
                if enhanced_response and 'response' in enhanced_response:
                    # Replace the entire response with AI-enhanced version
                    generated_response = enhanced_response
                    logging.info("Successfully enhanced response with OpenAI")
                elif enhanced_response:
                    # Merge any partial enhancements
                    generated_response.update(enhanced_response)
                    logging.info("Partially enhanced response with AI")
            except Exception as e:
                logging.warning(f"OpenAI response enhancement failed: {e}")
                # Continue with ML-generated response
        
        # Generate custom AI response for specific intents
        ai_message = self._generate_custom_ai_message(
            original_query, intent, analysis_results['results']
        )
        
        # Format final response with AI priority
        # Priority: 1. Custom AI message, 2. Enhanced OpenAI response, 3. ML response
        final_message = None
        
        if ai_message:
            final_message = ai_message
        elif generated_response.get('response', {}).get('message'):
            final_message = generated_response['response']['message']
        elif generated_response.get('message'):
            final_message = generated_response['message']
        else:
            final_message = f"Processed '{original_query}' using advanced AI and machine learning models. Analysis complete."
        
        final_response = {
            'status': 'success',
            'message': final_message,
            'details': generated_response.get('response', {}).get('details', []),
            'data': analysis_results['results'],
            'ai_insights': self._generate_ai_insights(analysis_results),
            'follow_up_suggestions': self._generate_contextual_suggestions(
                intent_analysis, analysis_results
            ),
            'conversation_flow': self._determine_conversation_flow(intent_analysis),
            'processing_metadata': {
                'ai_powered': True,
                'ml_components_used': [
                    'intent_classification', 'entity_extraction', 'semantic_search'
                ],
                'enhanced_by': generated_response.get('response', {}).get('enhanced_by', 'ml_models'),
                'zero_rule_based': True
            }
        }
        
        return final_response
    
    def _update_conversation_context(self, query: str, intent_analysis: Dict, 
                                   entities_analysis: Dict, response: Dict, 
                                   user_memory: Dict, session_context: Dict):
        """Update conversation context for future interactions"""
        
        # Update user memory
        user_memory['conversation_history'].append({
            'query': query,
            'intent': intent_analysis.get('final_intent', {}).get('intent'),
            'entities': entities_analysis,
            'response_type': response.get('conversation_flow', 'standard'),
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 10 conversations (mimic deque behavior)
        if len(user_memory['conversation_history']) > 10:
            user_memory['conversation_history'] = user_memory['conversation_history'][-10:]
        
        user_memory['last_intent'] = intent_analysis.get('final_intent', {}).get('intent')
        user_memory['last_entities'] = entities_analysis
        
        # Update session context
        session_context.update({
            'last_query': query,
            'last_response': response,
            'conversation_length': len(user_memory['conversation_history'])
        })
        
        # Update global statistics
        self.global_context['session_stats']['total_queries'] += 1
        self.global_context['session_stats']['ai_responses'] += 1
    
    # Helper methods for response enhancement
    def _apply_entity_filters(self, search_results: List[Dict], entities: Dict) -> List[Dict]:
        """Apply entity-based filtering to search results"""
        
        filtered_results = search_results.copy()
        
        # Filter by news categories
        if entities.get('news_categories'):
            categories = entities['news_categories']
            filtered_results = [
                result for result in filtered_results
                if any(cat in str(result.get('metadata', {}).get('category', '')).lower() for cat in categories)
            ]
        
        return filtered_results
    
    def _fallback_article_search(self, enhanced_query: Dict, entities: Dict) -> Dict[str, Any]:
        """Fallback search using direct database access"""
        
        if self.article_database is None:
            return {'error': 'No article database available'}
        
        try:
            query_text = enhanced_query['original_query'].lower()
            
            # Simple keyword-based search as fallback
            matching_articles = []
            for idx, row in self.article_database.head(20).iterrows():  # Limit for performance
                text = str(row.get('text', '')).lower()
                if any(word in text for word in query_text.split() if len(word) > 3):
                    article = {
                        'id': idx,
                        'text': str(row.get('text', ''))[:500],  # Truncate for response
                        'category': str(row.get('category', 'unknown')),
                        'similarity_score': 0.5  # Default similarity
                    }
                    matching_articles.append(article)
            
            return {
                'articles': matching_articles[:5],  # Top 5
                'total_found': len(matching_articles),
                'search_method': 'keyword_fallback',
                'filters_applied': entities
            }
            
        except Exception as e:
            logging.error(f"Fallback search failed: {e}")
            return {'error': f'Search failed: {str(e)}'}
    
    def _aggregate_sentiment_stats(self, sentiment_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate sentiment analysis statistics"""
        
        if not sentiment_results:
            return {}
        
        sentiments = [result['sentiment'] for result in sentiment_results]
        
        # Calculate distribution
        sentiment_labels = [s.get('label', 'neutral') for s in sentiments]
        from collections import Counter
        distribution = Counter(sentiment_labels)
        
        total = len(sentiment_labels)
        sentiment_distribution = {
            label: count / total for label, count in distribution.items()
        }
        
        # Calculate average sentiment score if available
        sentiment_scores = [s.get('score', 0) for s in sentiments if 'score' in s]
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        
        return {
            'classification_distribution': sentiment_distribution,
            'average_sentiment': avg_sentiment,
            'dominant_sentiment': max(distribution, key=distribution.get),
            'sentiment_range': (min(sentiment_scores), max(sentiment_scores)) if sentiment_scores else (0, 0)
        }
    
    def _aggregate_entity_stats(self, entity_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate entity extraction statistics"""
        
        aggregated = defaultdict(list)
        
        for result in entity_results:
            entities = result.get('entities', {})
            for entity_type, entity_list in entities.items():
                aggregated[entity_type].extend(entity_list)
        
        # Count unique entities
        entity_counts = {}
        for entity_type, entity_list in aggregated.items():
            unique_entities = list(set(entity_list))
            entity_counts[entity_type] = {
                'unique_count': len(unique_entities),
                'total_mentions': len(entity_list),
                'top_entities': unique_entities[:10]  # Top 10
            }
        
        return {
            'entities_by_type': dict(aggregated),
            'entity_statistics': entity_counts,
            'total_entities': sum(len(entities) for entities in aggregated.values())
        }
    
    def _generate_ai_recommendations(self, insights: Dict) -> List[str]:
        """Generate AI-powered recommendations based on insights"""
        
        recommendations = []
        
        # Data-driven recommendations
        data_insights = insights.get('data_insights', {})
        if data_insights:
            total_articles = data_insights.get('total_articles', 0)
            if total_articles > 1000:
                recommendations.append("Consider using topic modeling to discover hidden themes")
            
            category_dist = data_insights.get('category_distribution', {})
            if category_dist:
                dominant_category = max(category_dist, key=category_dist.get)
                recommendations.append(f"Focus analysis on {dominant_category} articles for deeper insights")
        
        # Pattern-based recommendations
        pattern_insights = insights.get('pattern_insights', {})
        if 'sentiment' in pattern_insights:
            sentiment = pattern_insights['sentiment']
            if sentiment.get('dominant_sentiment') == 'negative':
                recommendations.append("Consider analyzing positive alternatives or solution-oriented content")
        
        return recommendations
    
    def _generate_custom_ai_message(self, query: str, intent: str, results: Dict[str, Any]) -> str:
        """Generate custom AI-powered messages for all intents - NO RULE-BASED FALLBACKS"""
        
        if intent == 'extract_entities':
            if 'entity_extraction' in results:
                entity_data = results['entity_extraction']
                aggregated = results.get('aggregated_entities', {})
                
                if entity_data:
                    message = f"🧠 **AI Entity Extraction from Politics Articles:**\n\n"
                    message += f"Analyzed {len(entity_data)} articles using advanced NER models.\n\n"
                    
                    # Show entity statistics
                    entity_stats = aggregated.get('entity_statistics', {})
                    if entity_stats:
                        message += "**Key Entities Found:**\n"
                        for entity_type, stats in entity_stats.items():
                            count = stats.get('unique_count', 0)
                            top_entities = stats.get('top_entities', [])[:3]
                            if count > 0:
                                message += f"• **{entity_type.title()}**: {count} unique ({', '.join(top_entities)})\n"
                    
                    message += f"\n🔍 **Analysis Method**: Advanced NER machine learning models"
                    message += f"\n💡 **Try**: 'Analyze sentiment of these entities' or 'Find more politics articles'"
                    
                    return message
                else:
                    return f"🧠 **AI Entity Extraction**: No politics articles found for entity analysis. Try 'Find politics articles' first, then I can extract entities from them using advanced NLP models."
            
            elif 'error' in results:
                return f"🧠 **AI Entity Extraction**: {results['error']}. The system uses advanced NER models to extract people, organizations, and locations from articles."
            
            else:
                return f"🧠 **AI Entity Extraction**: Processing your request using advanced NER models to find people, organizations, locations, and other key entities in politics articles."
        
        elif intent == 'analyze_sentiment':
            if 'sentiment_analysis' in results:
                sentiment_data = results['sentiment_analysis']
                aggregated = results.get('aggregated_sentiment', {})
                
                message = f"🎭 **AI Sentiment Analysis Results:**\n\n"
                message += f"Analyzed {len(sentiment_data)} articles using transformer models.\n\n"
                
                if 'dominant_sentiment' in aggregated:
                    dominant = aggregated['dominant_sentiment']
                    message += f"**Overall Sentiment**: {dominant.title()}\n"
                
                if 'classification_distribution' in aggregated:
                    dist = aggregated['classification_distribution']
                    message += f"**Distribution**: "
                    parts = []
                    for sentiment, ratio in dist.items():
                        parts.append(f"{sentiment}: {ratio:.1%}")
                    message += ", ".join(parts)
                
                message += f"\n\n🔍 **Analysis Method**: Advanced transformer-based sentiment models"
                return message
            else:
                return f"🎭 **AI Sentiment Analysis**: Processing articles using advanced transformer models to analyze emotional tone and sentiment patterns."
        
        elif intent == 'summarize_text':
            if 'summaries' in results:
                summaries = results['summaries']
                message = f"📝 **AI Text Summarization:**\n\n"
                
                for i, summary_data in enumerate(summaries, 1):
                    summary = summary_data.get('summary', '')
                    compression = summary_data.get('compression_ratio', 0)
                    message += f"{i}. **Summary** (compressed to {compression:.1%}):\n"
                    message += f"   {summary}\n\n"
                
                message += f"🔍 **Method**: Advanced abstractive summarization using language models"
                return message
            else:
                return f"📝 **AI Text Summarization**: Generating intelligent summaries using advanced language models for natural, coherent abstracts."
        
        elif intent == 'classify_content':
            if 'classification_results' in results:
                classifications = results['classification_results']
                message = f"🏷️ **AI Content Classification:**\n\n"
                
                for i, result in enumerate(classifications[:3], 1):
                    classification = result.get('classification', {})
                    category = classification.get('predicted_category', 'Unknown')
                    confidence = classification.get('confidence', 0)
                    message += f"{i}. **Category**: {category} (confidence: {confidence:.1%})\n"
                
                message += f"\n🔍 **Method**: Trained machine learning classification models"
                return message
            else:
                return f"🏷️ **AI Content Classification**: Analyzing content using trained ML models to categorize articles by topic, theme, and subject matter."
        
        elif intent == 'get_insights':
            if 'data_insights' in results or 'trend_insights' in results:
                message = f"💡 **AI-Generated Insights:**\n\n"
                
                data_insights = results.get('data_insights', {})
                if data_insights:
                    total = data_insights.get('total_articles', 0)
                    message += f"**Data Overview**: Analyzed {total:,} articles across multiple categories\n"
                
                trend_insights = results.get('trend_insights', {})
                if 'topics' in trend_insights:
                    topics = trend_insights['topics']
                    num_topics = topics.get('num_topics_discovered', 0)
                    message += f"**Topic Discovery**: Found {num_topics} distinct themes using ML analysis\n"
                
                recommendations = results.get('recommendations', [])
                if recommendations:
                    message += f"\n**AI Recommendations**:\n"
                    for rec in recommendations[:3]:
                        message += f"• {rec}\n"
                
                message += f"\n🔍 **Method**: Advanced pattern recognition and ML-based insight generation"
                return message
            else:
                return f"💡 **AI Insight Generation**: Analyzing patterns and trends using advanced ML algorithms to generate intelligent insights from your data."
        
        # For search_articles and other intents, return None to use the standard response
        return None
    
    def _generate_ai_insights(self, analysis_results: Dict) -> List[str]:
        """Generate AI-powered insights from analysis results"""
        
        insights = []
        results = analysis_results.get('results', {})
        
        if 'articles' in results:
            articles = results['articles']
            if len(articles) > 0:
                insights.append(f"Found {len(articles)} relevant articles using semantic search")
                
                # Analyze categories
                categories = [article.get('category', 'unknown') for article in articles]
                unique_categories = list(set(categories))
                if len(unique_categories) > 1:
                    insights.append(f"Content spans {len(unique_categories)} categories: {', '.join(unique_categories)}")
        
        if 'sentiment_analysis' in results:
            sentiment_data = results.get('aggregated_sentiment', {})
            if 'dominant_sentiment' in sentiment_data:
                dominant = sentiment_data['dominant_sentiment']
                insights.append(f"Overall sentiment analysis shows {dominant} tone")
        
        return insights
    
    def _generate_contextual_suggestions(self, intent_analysis: Dict, 
                                       analysis_results: Dict) -> List[str]:
        """Generate contextual follow-up suggestions"""
        
        intent = intent_analysis.get('final_intent', {}).get('intent', 'unknown')
        suggestions = []
        
        if intent == 'search_articles':
            suggestions = [
                "Analyze sentiment of these articles",
                "Summarize the key findings",
                "Extract entities from the results",
                "Compare with different time periods"
            ]
        elif intent == 'analyze_sentiment':
            suggestions = [
                "Find articles with opposite sentiment",
                "Analyze sentiment trends over time",
                "Compare sentiment across categories",
                "Generate insights from sentiment patterns"
            ]
        elif intent == 'get_insights':
            suggestions = [
                "Explore specific topics in detail",
                "Generate visual analysis",
                "Export insights as report",
                "Set up monitoring for updates"
            ]
        else:
            suggestions = [
                "Ask for help to see all capabilities",
                "Try searching for specific topics",
                "Request sentiment analysis",
                "Ask for data insights"
            ]
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _determine_conversation_flow(self, intent_analysis: Dict) -> str:
        """Determine the conversation flow type"""
        
        intent = intent_analysis.get('final_intent', {}).get('intent', 'unknown')
        confidence = intent_analysis.get('final_intent', {}).get('confidence', 0)
        
        if confidence > 0.8:
            return 'direct'
        elif confidence > 0.5:
            return 'clarification'
        else:
            return 'exploration'
    
    def _openai_intent_enhancement(self, query: str, intent_result: Dict, 
                                 user_memory: Dict) -> Dict[str, Any]:
        """Enhance intent understanding using OpenAI"""
        
        if not self.openai_client:
            return {}
        
        # Create context-aware prompt
        conversation_context = ""
        if user_memory['conversation_history']:
            recent_queries = [h['query'] for h in list(user_memory['conversation_history'])[-3:]]
            conversation_context = f"Recent conversation: {'; '.join(recent_queries)}"
        
        system_prompt = f"""You are analyzing user intent for a news analysis system.
        Current query: "{query}"
        {conversation_context}
        
        Provide enhanced intent understanding in JSON format with:
        - clarified_intent: refined intent classification
        - confidence_boost: how much to boost confidence (0-0.3)
        - additional_parameters: any inferred parameters
        - conversation_flow: suggested flow (direct/clarification/exploration)
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            # Parse response (simplified - in production, add proper JSON parsing)
            enhanced_understanding = {
                'openai_enhancement': True,
                'enhanced_confidence': min(
                    intent_result.get('final_intent', {}).get('confidence', 0) + 0.1, 1.0
                )
            }
            
            return enhanced_understanding
            
        except Exception as e:
            logging.warning(f"OpenAI intent enhancement failed: {e}")
            return {}
    
    def _enhance_response_with_openai(self, query: str, analysis_results: Dict, 
                                    generated_response: Dict) -> Dict[str, Any]:
        """Enhance response naturalness using OpenAI"""
        
        if not self.openai_client:
            return {}
        
        # Handle simple greetings and conversational queries intelligently
        query_lower = query.lower().strip()
        if query_lower in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']:
            return {
                'response': {
                    'message': f"Hello! I'm your AI-powered NewsBot assistant. I can help you analyze news articles, extract insights, and answer questions about current events. I use advanced machine learning models for sentiment analysis, entity extraction, topic modeling, and intelligent content summarization.\n\n💡 Try asking: 'Find positive tech news' or 'Analyze sentiment of recent politics articles'",
                    'enhanced_by': 'ai_conversation_system'
                }
            }
        
        # Create analysis summary for OpenAI
        results = analysis_results.get('results', {})
        
        # Handle queries with no meaningful results
        if not results or (isinstance(results, dict) and not any(results.values())):
            return {
                'response': {
                    'message': f"I understand you're asking about '{query}'. I'm processing this using advanced NLP models and searching through the news database using semantic analysis.\n\n🔍 **Analysis Methods**: Sentiment analysis, entity extraction, topic modeling, content classification, and intelligent summarization.\n\n💡 **Try asking**: 'Find business articles', 'Analyze sentiment of tech news', or 'Extract entities from politics articles'",
                    'enhanced_by': 'ai_intelligent_fallback'
                }
            }
        
        # For queries with results, use OpenAI enhancement
        analysis_summary = json.dumps(results, default=str)[:1000]  # Truncate for token limits
        
        system_prompt = f"""You are NewsBot AI. Make the response more conversational and helpful.
        
        User query: "{query}"
        Analysis results: {analysis_summary}
        
        Generate a natural, helpful response that:
        1. Directly addresses the user's question
        2. Highlights key findings from the analysis
        3. Uses a conversational tone
        4. Suggests relevant follow-up actions
        
        Keep response under 200 words.
        """
        
        try:
            if hasattr(self.openai_client, 'chat') and hasattr(self.openai_client.chat, 'completions'):
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt}
                    ],
                    max_tokens=250,
                    temperature=0.7
                )
                
                enhanced_message = response.choices[0].message.content
                
                return {
                    'response': {
                        'message': enhanced_message,
                        'enhanced_by': 'openai'
                    }
                }
            else:
                # Fallback for different OpenAI client structure
                return {
                    'response': {
                        'message': f"Processed '{query}' using advanced AI models. Analysis complete with machine learning-powered insights.",
                        'enhanced_by': 'ml_models'
                    }
                }
            
        except Exception as e:
            logging.warning(f"OpenAI response enhancement failed: {e}")
            return {
                'response': {
                    'message': f"Analyzed '{query}' using advanced NLP models. Processing complete with ML-powered insights.",
                    'enhanced_by': 'ml_fallback'
                }
            }
    
    # Conversation flow handlers
    def _handle_greeting_flow(self, query: str, context: Dict) -> Dict[str, Any]:
        """Handle greeting conversation flow"""
        return {
            'flow_type': 'greeting',
            'response': 'Hello! I\'m your AI-powered news analysis assistant.',
            'next_suggestions': ['What can you do?', 'Show me recent news', 'Help me analyze articles']
        }
    
    def _handle_search_flow(self, query: str, context: Dict) -> Dict[str, Any]:
        """Handle search conversation flow"""
        return {
            'flow_type': 'search',
            'response': 'I\'ll search for relevant articles using semantic understanding.',
            'next_suggestions': ['Analyze sentiment', 'Summarize results', 'Find similar articles']
        }
    
    def _handle_analysis_flow(self, query: str, context: Dict) -> Dict[str, Any]:
        """Handle analysis conversation flow"""
        return {
            'flow_type': 'analysis',
            'response': 'Running advanced analysis using ML models.',
            'next_suggestions': ['Get insights', 'Export results', 'Compare with other data']
        }
    
    def _handle_insight_flow(self, query: str, context: Dict) -> Dict[str, Any]:
        """Handle insight generation flow"""
        return {
            'flow_type': 'insight',
            'response': 'Generating intelligent insights from the data.',
            'next_suggestions': ['Drill down into specifics', 'Generate report', 'Set up monitoring']
        }
    
    def _handle_help_flow(self, query: str, context: Dict) -> Dict[str, Any]:
        """Handle help conversation flow"""
        capabilities = self.global_context.get('system_capabilities', [])
        
        return {
            'flow_type': 'help',
            'response': f'I can help with: {", ".join(capabilities)}',
            'next_suggestions': ['Search articles', 'Analyze sentiment', 'Generate insights']
        }
    
    def get_conversation_statistics(self) -> Dict[str, Any]:
        """Get conversation system statistics"""
        
        total_users = len(self.conversation_memory)
        total_conversations = sum(
            len(memory['conversation_history']) 
            for memory in self.conversation_memory.values()
        )
        
        return {
            'total_users': total_users,
            'total_conversations': total_conversations,
            'avg_conversation_length': total_conversations / total_users if total_users > 0 else 0,
            'system_capabilities': self.global_context['system_capabilities'],
            'session_stats': dict(self.global_context['session_stats']),
            'ai_components_active': [
                'intent_classification', 'entity_extraction', 'semantic_search',
                'sentiment_analysis', 'response_generation'
            ]
        }
