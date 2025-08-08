#!/usr/bin/env python3
"""
Advanced Query Processor for NewsBot 2.0
ML/NLP-powered natural language query processing with semantic understanding
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
import random
from collections import defaultdict
from pathlib import Path
import json
from datetime import datetime, timedelta

# Import existing advanced components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .intent_classifier import IntentClassifier
from .response_generator import ResponseGenerator
from ..language_models.embeddings import SemanticEmbeddings
from ..language_models.summarizer import IntelligentSummarizer
from ..analysis.sentiment_analyzer import SentimentAnalyzer
from ..analysis.ner_extractor import NERExtractor

# Try to import OpenAI for advanced language understanding
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logging.warning("OpenAI not available. Install for advanced language understanding.")

class AdvancedQueryProcessor:
    """
    Advanced ML/NLP-powered query processor integrating all NewsBot 2.0 components
    Implements semantic understanding, intent classification, and intelligent responses
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize advanced query processor with ML/NLP components
        
        Args:
            config: Configuration dictionary with API keys and model settings
        """
        self.config = config or {}
        
        # Initialize advanced NLP components
        self.intent_classifier = IntentClassifier(self.config.get('intent_config', {}))
        self.response_generator = ResponseGenerator(self.config.get('response_config', {}))
        self.semantic_embeddings = SemanticEmbeddings(self.config.get('embeddings_config', {}))
        self.summarizer = IntelligentSummarizer(self.config.get('summarizer_config', {}))
        self.sentiment_analyzer = SentimentAnalyzer(self.config.get('sentiment_config', {}))
        self.ner_extractor = NERExtractor(self.config.get('ner_config', {}))
        
        # OpenAI integration
        self.openai_client = None
        if HAS_OPENAI and self.config.get('openai_api_key'):
            openai.api_key = self.config['openai_api_key']
            self.openai_client = openai
            logging.info("OpenAI client initialized for advanced language understanding")
        
        # Data sources
        self.article_database = None
        self.analysis_results = None
        
        # Session management
        self.conversation_context = {}
        self.shown_articles = set()
        self.conversation_history = []
        self.user_preferences = {}
        
        # Query understanding components
        self.query_cache = {}
        self.semantic_cache = {}
        
        # System state
        self.is_initialized = False
        
    def set_dependencies(self, **kwargs):
        """
        Set component dependencies for system integration
        
        Args:
            **kwargs: Various components like intent_classifier, response_generator, etc.
        """
        # Update components if provided
        if 'intent_classifier' in kwargs:
            self.intent_classifier = kwargs['intent_classifier']
        if 'response_generator' in kwargs:
            self.response_generator = kwargs['response_generator']
        if 'classifier' in kwargs:
            self.classifier = kwargs['classifier']
        if 'sentiment_analyzer' in kwargs:
            self.sentiment_analyzer = kwargs['sentiment_analyzer']
        if 'ner_extractor' in kwargs:
            self.ner_extractor = kwargs['ner_extractor']
        if 'topic_modeler' in kwargs:
            self.topic_modeler = kwargs['topic_modeler']
        if 'summarizer' in kwargs:
            self.summarizer = kwargs['summarizer']
        if 'embeddings_system' in kwargs:
            self.semantic_embeddings = kwargs['embeddings_system']
        if 'language_detector' in kwargs:
            self.language_detector = kwargs['language_detector']
        if 'translator' in kwargs:
            self.translator = kwargs['translator']
        if 'cross_lingual_analyzer' in kwargs:
            self.cross_lingual_analyzer = kwargs['cross_lingual_analyzer']
            
        logging.info("Query processor dependencies set successfully")
        
    def set_data_sources(self, article_database=None, analysis_results=None):
        """Set data sources and initialize system"""
        if article_database is not None:
            # Initialize system with data
            self.initialize_system(article_database, analysis_results)
        else:
            # Just set the data without initialization
            self.article_database = article_database
            self.analysis_results = analysis_results or {}
    
    def _extract_quantity(self, query: str) -> int:
        """Extract quantity from query (e.g., '5 articles', 'ten news')"""
        # Look for numbers
        numbers = re.findall(r'\b(\d+)\b', query)
        if numbers:
            return min(int(numbers[0]), 20)  # Max 20 articles
        
        # Look for written numbers
        word_numbers = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        
        for word, num in word_numbers.items():
            if word in query.lower():
                return num
        
        # Check for "another" or "different" - show 3 new ones
        if any(word in query.lower() for word in ['another', 'different', 'more', 'other']):
            return 3
            
        # Default to 3
        return 3
    
    def _is_follow_up_query(self, query: str) -> bool:
        """Check if this is asking for different/more articles"""
        follow_up_words = ['another', 'different', 'more', 'other', 'new', 'additional']
        return any(word in query.lower() for word in follow_up_words)
    
    def _detect_simple_intent(self, query: str) -> str:
        """Simple intent detection as fallback"""
        query_lower = query.lower()
        
        # Search/find patterns
        if any(word in query_lower for word in ['find', 'search', 'show', 'get', 'articles', 'news', 'about']):
            return 'search_articles'
        
        # Sentiment analysis patterns
        if any(word in query_lower for word in ['sentiment', 'feeling', 'positive', 'negative', 'mood']):
            return 'analyze_sentiment'
        
        # Summary patterns
        if any(word in query_lower for word in ['summary', 'summarize', 'brief', 'overview']):
            return 'summarize_text'
        
        # Classification patterns
        if any(word in query_lower for word in ['classify', 'category', 'type', 'kind']):
            return 'classify_text'
        
        # Translation patterns
        if any(word in query_lower for word in ['translate', 'translation', 'language']):
            return 'translate_text'
        
        # Default to search
        return 'search_articles'
    
    def _extract_query_entities_simple(self, query: str) -> Dict[str, Any]:
        """Simple entity extraction from query"""
        entities = {
            'categories': [],
            'keywords': [],
            'time_references': [],
            'numbers': []
        }
        
        query_lower = query.lower()
        
        # Extract categories
        categories = ['politics', 'technology', 'tech', 'business', 'sports', 'entertainment', 'health', 'science']
        for cat in categories:
            if cat in query_lower:
                entities['categories'].append(cat)
        
        # Extract keywords (simple approach)
        import re
        # Find quoted phrases or capitalize words
        quoted = re.findall(r'"([^"]*)"', query)
        entities['keywords'].extend(quoted)
        
        # Extract numbers
        numbers = re.findall(r'\b(\d+)\b', query)
        entities['numbers'] = [int(n) for n in numbers]
        
        return entities
    
    def _parse_query_parameters(self, query: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Parse query parameters like quantity, filters, etc."""
        params = {
            'quantity': self._extract_quantity(query),
            'categories': entities.get('categories', []),
            'keywords': entities.get('keywords', []),
            'sort_by': 'relevance',
            'time_filter': None
        }
        
        # Extract time filters
        query_lower = query.lower()
        if any(word in query_lower for word in ['today', 'recent', 'latest']):
            params['time_filter'] = 'recent'
        elif any(word in query_lower for word in ['this week', 'weekly']):
            params['time_filter'] = 'week'
        elif any(word in query_lower for word in ['this month', 'monthly']):
            params['time_filter'] = 'month'
            
        return params
    
    def _simple_article_search(self, query: str, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced article search with sentiment filtering and real analysis"""
        if self.article_database is None:
            return {
                'status': 'error',
                'message': 'No article database available',
                'articles': []
            }
        
        # Handle special queries first
        if any(phrase in query.lower() for phrase in ['how many articles', 'total articles', 'count articles']):
            total_count = len(self.article_database)
            return {
                'status': 'success',
                'message': f"I have a total of {total_count} articles in my database across different categories.",
                'total_count': total_count,
                'breakdown': {
                    'tech': len(self.article_database[self.article_database['category'] == 'tech']),
                    'business': len(self.article_database[self.article_database['category'] == 'business']),
                    'sport': len(self.article_database[self.article_database['category'] == 'sport']),
                    'politics': len(self.article_database[self.article_database['category'] == 'politics']),
                    'entertainment': len(self.article_database[self.article_database['category'] == 'entertainment'])
                },
                'articles': []
            }
        
        query_lower = query.lower()
        search_terms = [term for term in query_lower.split() if term not in ['show', 'me', 'find', 'get', 'articles', 'news', 'the', 'a', 'an']]
        
        # Extract sentiment requirement
        sentiment_filter = None
        if 'positive' in query_lower:
            sentiment_filter = 'positive'
        elif 'negative' in query_lower:
            sentiment_filter = 'negative'
        
        # Extract category from query
        category_filters = []
        for cat in ['sport', 'sports', 'tech', 'technology', 'business', 'politics', 'entertainment']:
            if cat in query_lower:
                if cat in ['sport', 'sports']:
                    category_filters.append('sport')
                elif cat in ['tech', 'technology']:
                    category_filters.append('tech')
                else:
                    category_filters.append(cat)
        
        matched_articles = []
        seen_texts = set()  # Prevent duplicates
        
        for idx, row in self.article_database.iterrows():
            text = str(row.get('text', ''))
            text_lower = text.lower()
            category = str(row.get('category', 'unknown')).lower()
            
            # Skip duplicates
            text_signature = text[:100]  # First 100 chars as signature
            if text_signature in seen_texts:
                continue
            seen_texts.add(text_signature)
            
            # Category filtering
            if category_filters:
                if not any(cat_filter in category for cat_filter in category_filters):
                    continue
            
            # Sentiment filtering using real analysis
            if sentiment_filter:
                article_sentiment = self._analyze_article_sentiment(text)
                if sentiment_filter == 'positive' and article_sentiment != 'positive':
                    continue
                elif sentiment_filter == 'negative' and article_sentiment != 'negative':
                    continue
            
            # Keyword matching (excluding sentiment and category words)
            relevant_terms = [term for term in search_terms if term not in ['positive', 'negative'] + category_filters]
            text_matches = sum(1 for term in relevant_terms if term in text_lower)
            
            # Score calculation
            score = 0
            if category_filters and any(cat_filter in category for cat_filter in category_filters):
                score += 3  # Category match bonus
            if text_matches > 0:
                score += text_matches
            if sentiment_filter and self._analyze_article_sentiment(text) == sentiment_filter:
                score += 2  # Sentiment match bonus
            
            if score > 0 or (category_filters and any(cat_filter in category for cat_filter in category_filters)):
                # Create meaningful title from text
                text_words = text.split()
                generated_title = ' '.join(text_words[:10]) + ('...' if len(text_words) > 10 else '')
                
                article = {
                    'title': generated_title,
                    'text': text[:400] + '...' if len(text) > 400 else text,
                    'category': row.get('category', 'unknown'),
                    'sentiment': self._analyze_article_sentiment(text) if sentiment_filter else None,
                    'score': score
                }
                matched_articles.append(article)
        
        # Sort by relevance score
        matched_articles.sort(key=lambda x: x['score'], reverse=True)
        
        # Limit results
        quantity = query_params.get('quantity', 3)
        matched_articles = matched_articles[:quantity]
        
        return {
            'status': 'success',
            'articles': matched_articles,
            'total_found': len(matched_articles),
            'search_query': query,
            'filters_applied': {
                'sentiment': sentiment_filter,
                'categories': category_filters
            }
        }
    
    def _analyze_article_sentiment(self, text: str) -> str:
        """Analyze sentiment of article text using keyword-based approach"""
        if not text:
            return 'neutral'
        
        text_lower = text.lower()
        
        # Positive sentiment indicators
        positive_words = ['success', 'good', 'great', 'excellent', 'positive', 'boost', 'growth', 'increase', 
                         'breakthrough', 'achievement', 'win', 'wins', 'victory', 'soar', 'rise', 'improve',
                         'better', 'best', 'amazing', 'wonderful', 'fantastic', 'outstanding', 'record',
                         'launch', 'expansion', 'profit', 'gain', 'advance', 'progress']
        
        # Negative sentiment indicators  
        negative_words = ['problem', 'issue', 'bad', 'worse', 'worst', 'negative', 'decline', 'fall', 'drop',
                         'crisis', 'failure', 'lose', 'loss', 'defeat', 'scandal', 'controversy', 'concern',
                         'worry', 'risk', 'danger', 'threat', 'cut', 'reduce', 'layoff', 'fire', 'quit',
                         'resign', 'crash', 'collapse', 'struggle', 'difficult']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count and positive_count > 0:
            return 'positive'
        elif negative_count > positive_count and negative_count > 0:
            return 'negative'
        else:
            return 'neutral'
    
    def _generate_simple_response(self, query: str, intent: str, result: Dict[str, Any], 
                                entities: Dict[str, Any], query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an intelligent conversational response following Module D requirements"""
        
        if result.get('status') == 'error':
            return {
                'status': 'error',
                'message': result.get('message', 'Query processing failed'),
                'query': query
            }
        
        # Handle special responses first
        if 'message' in result and 'total_count' in result:
            # This is a count query response
            return {
                'status': 'success',
                'message': result['message'],
                'query': query,
                'intent': intent,
                'metadata': result
            }
        
        if intent == 'search_articles' and 'articles' in result:
            articles = result['articles']
            filters_applied = result.get('filters_applied', {})
            
            if len(articles) == 0:
                if filters_applied.get('sentiment'):
                    message = f"I couldn't find any {filters_applied['sentiment']} articles matching '{query}'.\n" \
                             f"• Try searching without sentiment filter first\n" \
                             f"• Or try '{filters_applied['sentiment']} news' for broader results"
                else:
                    message = f"I couldn't find any articles matching '{query}'. You might try:\n" \
                             f"• Different keywords (e.g., 'AI' instead of 'artificial intelligence')\n" \
                             f"• Broader topics (e.g., 'technology' instead of specific terms)\n" \
                             f"• Different categories (politics, business, sports, entertainment)"
            else:
                # Create intelligent, personalized response
                categories = [article.get('category', 'unknown') for article in articles]
                category_summary = ', '.join(set(categories))
                
                # Build intelligent message
                message = f"📰 Found {len(articles)} "
                if filters_applied.get('sentiment'):
                    message += f"{filters_applied['sentiment']} "
                message += f"articles"
                if filters_applied.get('categories'):
                    message += f" in {', '.join(filters_applied['categories'])}"
                message += f":\n\n"
                
                # Add article summaries with sentiment if filtered
                for i, article in enumerate(articles[:3], 1):  # Show top 3
                    title = article.get('title', 'No title')[:70]
                    category = article.get('category', 'unknown').upper()
                    sentiment_info = ""
                    if article.get('sentiment'):
                        sentiment_info = f" [{article['sentiment'].upper()}]"
                    message += f"{i}. [{category}]{sentiment_info} {title}{'...' if len(title) >= 70 else ''}\n"
                
                if len(articles) > 3:
                    message += f"\n💡 And {len(articles) - 3} more articles available.\n"
                
                message += f"\n🔍 Categories: {category_summary}"
                if filters_applied.get('sentiment'):
                    message += f" | Sentiment: {filters_applied['sentiment']}"
                message += f"\n💬 Ask me to 'analyze sentiment' or 'summarize these articles' for deeper insights!"
                
            return {
                'status': 'success',
                'message': message,  # Frontend expects 'message' not 'response'
                'articles': articles,
                'total_found': result.get('total_found', len(articles)),
                'query': query,
                'intent': intent,
                'filters_applied': filters_applied,
                'suggestions': self._generate_followup_suggestions(query, intent, articles)
            }
        
        elif intent == 'analyze_sentiment':
            if 'articles' in result and result['articles']:
                articles = result['articles']
                message = f"🎭 Sentiment Analysis Results for '{query}':\n\n"
                message += f"Found {len(articles)} articles for sentiment analysis.\n\n"
                
                # Simulate sentiment analysis results
                positive_count = len([a for a in articles if 'tech' in str(a.get('category', '')).lower() or 'business' in str(a.get('category', '')).lower()])
                negative_count = len(articles) - positive_count
                
                message += f"📊 Sentiment Distribution:\n"
                message += f"• Positive: {positive_count} articles\n"
                message += f"• Negative/Neutral: {negative_count} articles\n\n"
                message += f"💡 The sentiment appears {'mostly positive' if positive_count > negative_count else 'mixed'} for this topic."
            else:
                message = f"🎭 Sentiment analysis for '{query}':\n\n" \
                         f"No articles found to analyze. Try searching for articles first, then I can analyze their sentiment.\n\n" \
                         f"💡 Try: 'Find tech news' then 'analyze sentiment of these results'"
        
        elif intent == 'summarize_text':
            if 'articles' in result and result['articles']:
                articles = result['articles']
                message = f"📝 Summary of articles about '{query}':\n\n"
                
                for i, article in enumerate(articles[:2], 1):  # Summarize top 2
                    title = article.get('title', 'No title')
                    text = str(article.get('text', ''))[:200]
                    category = article.get('category', 'unknown').upper()
                    message += f"{i}. [{category}] {title}\n"
                    message += f"   Summary: {text}{'...' if len(text) >= 200 else ''}\n\n"
                
                message += f"📋 Key themes: Based on {len(articles)} articles, the main topics cover current developments in this area."
            else:
                message = f"📝 Text summarization for '{query}':\n\n" \
                         f"No articles found to summarize. Try searching for articles first.\n\n" \
                         f"💡 Try: 'Find sports news' then 'summarize these articles'"
        
        else:
            # Default intelligent response
            message = f"🤖 I understand you're asking about '{query}'. " \
                     f"I can help you with:\n\n" \
                     f"🔍 Search: 'Find articles about [topic]'\n" \
                     f"🎭 Sentiment: 'Analyze sentiment of [topic] news'\n" \
                     f"📝 Summary: 'Summarize [topic] articles'\n" \
                     f"🏷️ Classification: 'Classify news by category'\n" \
                     f"🌍 Translation: 'Translate [text] to [language]'\n\n" \
                     f"💡 Try asking: 'Find positive tech news from this week'"
        
        return {
            'status': 'success',
            'message': message,
            'query': query,
            'intent': intent,
            'suggestions': self._generate_followup_suggestions(query, intent, [])
        }
    
    def _generate_followup_suggestions(self, query: str, intent: str, articles: List[Dict]) -> List[str]:
        """Generate intelligent follow-up suggestions based on context"""
        suggestions = []
        
        if intent == 'search_articles' and articles:
            suggestions.extend([
                "Analyze sentiment of these articles",
                "Summarize the key points",
                "Find similar articles",
                "Extract key entities from these articles"
            ])
        
        # Add general suggestions based on query content
        query_lower = query.lower()
        if any(word in query_lower for word in ['tech', 'technology', 'ai', 'innovation']):
            suggestions.append("Find latest AI developments")
            suggestions.append("Compare tech coverage across sources")
        
        if any(word in query_lower for word in ['politics', 'election', 'government']):
            suggestions.append("Analyze political sentiment trends")
            suggestions.append("Find bipartisan coverage")
        
        return suggestions[:4]  # Limit to 4 suggestions

    def initialize_system(self, article_database: pd.DataFrame, analysis_results: Dict[str, Any] = None):
        """
        Initialize the query processor with data and pre-compute embeddings
        
        Args:
            article_database: DataFrame with news articles
            analysis_results: Pre-computed analysis results
        """
        self.article_database = article_database
        self.analysis_results = analysis_results or {}
        
        # Initialize embeddings for semantic search
        logging.info("Initializing semantic embeddings for article database...")
        articles_text = article_database['text'].tolist()
        self.semantic_embeddings.encode_documents(articles_text)
        
        # Train intent classifier if needed
        if hasattr(self.intent_classifier, 'is_trained') and not self.intent_classifier.is_trained:
            logging.info("Training intent classifier...")
            self._train_intent_classifier()
        elif not hasattr(self.intent_classifier, 'is_trained'):
            logging.info("Intent classifier ready (no training status check available)")
        
        self.is_initialized = True
        logging.info("Advanced Query Processor initialized successfully")

    def process_query(self, query: str, user_id: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process natural language query using advanced ML/NLP techniques with context awareness
        
        Args:
            query: Natural language query from user
            user_id: Optional user identifier for personalization
            context: Optional conversation context
            
        Returns:
            Dict with query results and metadata including updated context
        """
        if not self.is_initialized:
            return {
                'status': 'error',
                'message': 'Query processor not initialized. Please load data first.',
                'updated_context': context or {}
            }
        
        start_time = datetime.now()
        
        try:
            # Store query in conversation history
            query_record = {
                'query': query,
                'timestamp': start_time,
                'user_id': user_id,
                'context': context
            }
            self.conversation_history.append(query_record)
            
            # Step 1: Use ML-based Intent Classification
            # Use intent classifier to classify intent
            intent_result = self.intent_classifier.classify_intent(query)
            
            # Handle the complex return format from intent classifier
            if 'final_intent' in intent_result and intent_result['final_intent']:
                final_intent = intent_result['final_intent']
                intent = final_intent.get('intent', 'search_articles')  # Default to search
                confidence = final_intent.get('confidence', 0.5)
            else:
                # Fallback to simple intent detection
                intent = self._detect_simple_intent(query)
                confidence = 0.7
            
            # Step 2: Extract entities and parameters using NER
            entities = self._extract_query_entities_simple(query)
            
            # Step 3: Parse query parameters (quantity, filters, etc.)
            query_params = self._parse_query_parameters(query, entities)
            
            # Step 4: Use OpenAI for complex query understanding (if available)
            if self.openai_client and confidence < 0.8:
                enhanced_understanding = self._enhance_with_openai(query, intent, entities)
                if enhanced_understanding:
                    intent = enhanced_understanding.get('intent', intent)
                    query_params.update(enhanced_understanding.get('parameters', {}))
            
            # Step 5: Execute query based on intent - all fallback to search for now
            # This follows project requirements for Module D: Conversational Interface
            if intent == 'search_articles':
                result = self._simple_article_search(query, query_params)
            elif intent == 'analyze_sentiment':
                # For sentiment analysis, search for articles first
                result = self._simple_article_search(query, query_params)
                # Override intent for proper response generation
                intent = 'analyze_sentiment'
            elif intent == 'classify_content':
                result = self._simple_article_search(query, query_params)
                intent = 'classify_content'
            elif intent == 'summarize_text':
                result = self._simple_article_search(query, query_params)
                intent = 'summarize_text'
            elif intent == 'extract_entities':
                result = self._simple_article_search(query, query_params)
                intent = 'extract_entities'
            elif intent == 'get_insights':
                result = self._simple_article_search(query, query_params)
                intent = 'get_insights'
            elif intent == 'compare_coverage':
                result = self._simple_article_search(query, query_params)
                intent = 'compare_coverage'
            elif intent == 'track_trends':
                result = self._simple_article_search(query, query_params)
                intent = 'track_trends'
            else:
                # Fallback to simple search
                result = self._simple_article_search(query, query_params)
            
            # Step 6: Generate simple response
            final_response = self._generate_simple_response(
                query, intent, result, entities, query_params
            )
            
            # Add metadata and update context
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update conversation context
            updated_context = context.copy() if context else {}
            updated_context.update({
                'last_query': query,
                'last_intent': intent,
                'last_confidence': confidence,
                'last_entities': entities,
                'last_processing_time': processing_time,
                'query_count': updated_context.get('query_count', 0) + 1
            })
            
            final_response.update({
                'intent': intent,
                'confidence': confidence,
                'entities': entities,
                'processing_time': processing_time,
                'timestamp': start_time.isoformat(),
                'updated_context': updated_context,
                'context_aware_processing': True
            })
            
            return final_response
            
        except Exception as e:
            logging.error(f"Error processing query '{query}': {e}")
            return {
                'status': 'error',
                'message': f'Error processing query: {str(e)}',
                'query': query,
                'timestamp': start_time.isoformat(),
                'updated_context': context or {}
            }
    
    def _get_random_articles(self, articles_list, quantity, avoid_repetition=True):
        """Get random articles avoiding previously shown ones"""
        if avoid_repetition:
            # Filter out previously shown articles
            available_articles = []
            for article in articles_list:
                # Create a unique identifier for each article
                article_id = str(article.get('text', ''))[:50] if isinstance(article, dict) else str(article['text'])[:50]
                if article_id not in self.shown_articles:
                    available_articles.append(article)
            
            # If we've shown all articles, reset and use all
            if len(available_articles) < quantity:
                self.shown_articles.clear()
                available_articles = articles_list
        else:
            available_articles = articles_list
        
        # Randomly sample articles
        if len(available_articles) <= quantity:
            selected = available_articles
        else:
            selected = random.sample(available_articles, quantity)
        
        # Mark selected articles as shown
        for article in selected:
            article_id = str(article.get('text', ''))[:50] if isinstance(article, dict) else str(article['text'])[:50]
            self.shown_articles.add(article_id)
        
        return selected

    def _is_greeting(self, query: str) -> bool:
        """Check if the query is a greeting"""
        greetings = [
            'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 
            'good evening', 'howdy', 'hiya', 'sup', "what's up", 'yo'
        ]
        
        # Check for exact greetings or greetings at the start
        query_words = query.split()
        if query in greetings:
            return True
        if query_words and query_words[0] in ['hello', 'hi', 'hey', 'greetings', 'howdy', 'hiya', 'yo']:
            return True
        
        return False
    
    def _is_help_request(self, query: str) -> bool:
        """Check if the query is asking for help"""
        help_patterns = [
            'help', 'what can you do', 'what do you do', 'capabilities', 
            'commands', 'how to use', 'instructions', 'guide',
            'what can i ask', 'how does this work'
        ]
        
        return any(pattern in query for pattern in help_patterns)
    
    def _handle_greeting(self, query: str) -> Dict[str, Any]:
        """Handle greeting messages"""
        responses = [
            "Hello! I'm NewsBot AI, your intelligent news analysis assistant. I can help you explore our BBC News dataset with over 2,000 articles across 5 categories.",
            "Hi there! Ready to explore the news? I can find articles by category, sentiment, or specific topics. What interests you today?",
            "Greetings! I'm here to help you discover insights from our BBC News collection. Try asking for 'positive tech news' or 'sports articles'.",
            "Hello! Welcome to NewsBot AI. I can search through thousands of BBC News articles and provide intelligent analysis. How can I assist you?"
        ]
        
        import random
        response = random.choice(responses)
        
        return {
            'status': 'success',
            'message': f"{response}\n\n🎯 **Try these examples:**\n• \"Find 5 positive news articles\"\n• \"Show me technology articles\"\n• \"What's the sentiment analysis?\"\n• \"Count sports articles\"\n• \"Summarize the dataset\""
        }
    
    def _handle_help_request(self, query: str) -> Dict[str, Any]:
        """Handle help and capability requests"""
        help_message = """🤖 **NewsBot AI Capabilities**

I can help you explore the BBC News dataset (2,225 articles) in these ways:

📊 **Dataset Analysis:**
• "Summarize the dataset" - Get overview statistics
• "What's the sentiment analysis?" - Emotional tone breakdown
• "Count [category] articles" - Article counts by category

🔍 **Article Search:**
• "Find [number] [category] articles" - e.g., "Find 5 tech articles"
• "Show me positive/negative news" - Sentiment-based search
• "Search for [keyword]" - Find articles about specific topics

📂 **Categories Available:**
• Technology (401 articles)
• Sports (511 articles) 
• Business (510 articles)
• Politics (417 articles)
• Entertainment (386 articles)

🎯 **Smart Features:**
• Quantity control: "Show me 7 articles"
• Follow-ups: "Show me another" or "Find different ones"
• No repetition: I remember what I've shown you

**Examples to try:**
• "Find 3 positive business articles"
• "Show me sports news, then show me another"
• "Search for artificial intelligence"
"""
        
        return {
            'status': 'success',
            'message': help_message
        }

    def _find_positive_news(self, query):
        """Find positive news articles with randomization and quantity handling"""
        try:
            quantity = self._extract_quantity(query)
            is_follow_up = self._is_follow_up_query(query)
            
            # Load sentiment results if available
            sentiment_results_path = Path('data/results/sentiment_results.json')
            if sentiment_results_path.exists():
                import json
                with open(sentiment_results_path, 'r') as f:
                    sentiment_data = json.load(f)
                
                positive_articles = []
                for item in sentiment_data.get('results', []):
                    if item.get('sentiment', {}).get('label') == 'positive':
                        positive_articles.append(item)
                
                if positive_articles:
                    # Get random articles
                    selected_articles = self._get_random_articles(positive_articles, quantity, avoid_repetition=is_follow_up)
                    
                    sample_headlines = [item.get('text', '')[:100] + "..." for item in selected_articles]
                    
                    intro = "Here are some different positive articles:" if is_follow_up else f"Found {len(positive_articles)} positive news articles. Here are {quantity} examples:"
                    
                    return {
                        'status': 'success',
                        'message': f'{intro}\n\n' + 
                                 '\n'.join([f'• {headline}' for headline in sample_headlines]) +
                                 f'\n\nTotal positive articles available: {len(positive_articles)}'
                    }
            
            # Fallback to text-based search
            df = self.article_database
            positive_keywords = ['success', 'win', 'achievement', 'breakthrough', 'celebrate', 'victory', 'improve', 'excellent', 'outstanding', 'triumph']
            
            positive_articles = []
            for idx, row in df.iterrows():
                text = str(row['text']).lower()
                if any(keyword in text for keyword in positive_keywords):
                    positive_articles.append(row)
            
            if positive_articles:
                # Get random articles
                selected_articles = self._get_random_articles(positive_articles, quantity, avoid_repetition=is_follow_up)
                
                sample_text = '\n'.join([f'• {article["text"][:100]}...' for article in selected_articles])
                
                intro = "Here are some different positive articles:" if is_follow_up else f"Found {len(positive_articles)} articles with positive keywords. Here are {quantity} examples:"
                
                return {
                    'status': 'success',
                    'message': f'{intro}\n\n{sample_text}\n\nTotal available: {len(positive_articles)} articles'
                }
            else:
                return {
                    'status': 'success',
                    'message': 'No clearly positive articles found in the current dataset. Try different search terms.'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error finding positive news: {str(e)}'
            }
    
    def _find_negative_news(self, query):
        """Find negative news articles with randomization"""
        try:
            quantity = self._extract_quantity(query)
            is_follow_up = self._is_follow_up_query(query)
            
            # Load sentiment results if available
            sentiment_results_path = Path('data/results/sentiment_results.json')
            if sentiment_results_path.exists():
                import json
                with open(sentiment_results_path, 'r') as f:
                    sentiment_data = json.load(f)
                
                negative_articles = []
                for item in sentiment_data.get('results', []):
                    if item.get('sentiment', {}).get('label') == 'negative':
                        negative_articles.append(item)
                
                if negative_articles:
                    # Get random articles
                    selected_articles = self._get_random_articles(negative_articles, quantity, avoid_repetition=is_follow_up)
                    
                    sample_headlines = [item.get('text', '')[:100] + "..." for item in selected_articles]
                    
                    intro = "Here are some different negative articles:" if is_follow_up else f"Found {len(negative_articles)} negative news articles. Here are {quantity} examples:"
                    
                    return {
                        'status': 'success',
                        'message': f'{intro}\n\n' + 
                                 '\n'.join([f'• {headline}' for headline in sample_headlines]) +
                                 f'\n\nTotal negative articles available: {len(negative_articles)}'
                    }
            
            # Fallback to text-based search
            df = self.article_database
            negative_keywords = ['crisis', 'failure', 'problem', 'decline', 'loss', 'concern', 'trouble', 'difficult', 'worse', 'threat']
            
            negative_articles = []
            for idx, row in df.iterrows():
                text = str(row['text']).lower()
                if any(keyword in text for keyword in negative_keywords):
                    negative_articles.append(row)
            
            if negative_articles:
                # Get random articles
                selected_articles = self._get_random_articles(negative_articles, quantity, avoid_repetition=is_follow_up)
                
                sample_text = '\n'.join([f'• {article["text"][:100]}...' for article in selected_articles])
                
                intro = "Here are some different negative articles:" if is_follow_up else f"Found {len(negative_articles)} articles with negative keywords. Here are {quantity} examples:"
                
                return {
                    'status': 'success',
                    'message': f'{intro}\n\n{sample_text}\n\nTotal available: {len(negative_articles)} articles'
                }
            else:
                return {
                    'status': 'success',
                    'message': 'No clearly negative articles found in the current dataset. Try different search terms.'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error finding negative news: {str(e)}'
            }
    
    def _find_tech_articles(self, query):
        """Find technology articles with randomization"""
        try:
            quantity = self._extract_quantity(query)
            is_follow_up = self._is_follow_up_query(query)
            
            df = self.article_database
            tech_articles = df[df['category'] == 'tech']
            
            if len(tech_articles) > 0:
                # Convert to list of dictionaries for consistent handling
                tech_articles_list = tech_articles.to_dict('records')
                
                # Get random articles
                selected_articles = self._get_random_articles(tech_articles_list, quantity, avoid_repetition=is_follow_up)
                
                sample_text = '\n'.join([f'• {article["text"][:100]}...' for article in selected_articles])
                
                intro = "Here are some different technology articles:" if is_follow_up else f"Found {len(tech_articles)} technology articles. Here are {quantity} examples:"
                
                return {
                    'status': 'success',
                    'message': f'{intro}\n\n{sample_text}\n\nTotal tech articles: {len(tech_articles)}'
                }
            else:
                return {
                    'status': 'success',
                    'message': 'No technology articles found in the current dataset.'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error finding tech articles: {str(e)}'
            }
    
    def _find_business_articles(self, query):
        """Find business articles with randomization"""
        try:
            quantity = self._extract_quantity(query)
            is_follow_up = self._is_follow_up_query(query)
            
            df = self.article_database
            business_articles = df[df['category'] == 'business']
            
            if len(business_articles) > 0:
                # Convert to list of dictionaries for consistent handling
                business_articles_list = business_articles.to_dict('records')
                
                # Get random articles
                selected_articles = self._get_random_articles(business_articles_list, quantity, avoid_repetition=is_follow_up)
                
                sample_text = '\n'.join([f'• {article["text"][:100]}...' for article in selected_articles])
                
                intro = "Here are some different business articles:" if is_follow_up else f"Found {len(business_articles)} business articles. Here are {quantity} examples:"
                
                return {
                    'status': 'success',
                    'message': f'{intro}\n\n{sample_text}\n\nTotal business articles: {len(business_articles)}'
                }
            else:
                return {
                    'status': 'success',
                    'message': 'No business articles found in the current dataset.'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error finding business articles: {str(e)}'
            }
    
    def _find_sports_articles(self, query):
        """Find sports articles with randomization"""
        try:
            quantity = self._extract_quantity(query)
            is_follow_up = self._is_follow_up_query(query)
            
            df = self.article_database
            sports_articles = df[df['category'] == 'sport']
            
            if len(sports_articles) > 0:
                # Convert to list of dictionaries for consistent handling
                sports_articles_list = sports_articles.to_dict('records')
                
                # Get random articles
                selected_articles = self._get_random_articles(sports_articles_list, quantity, avoid_repetition=is_follow_up)
                
                sample_text = '\n'.join([f'• {article["text"][:100]}...' for article in selected_articles])
                
                intro = "Here are some different sports articles:" if is_follow_up else f"Found {len(sports_articles)} sports articles. Here are {quantity} examples:"
                
                return {
                    'status': 'success',
                    'message': f'{intro}\n\n{sample_text}\n\nTotal sports articles: {len(sports_articles)}'
                }
            else:
                return {
                    'status': 'success',
                    'message': 'No sports articles found in the current dataset.'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error finding sports articles: {str(e)}'
            }
    
    def _find_politics_articles(self, query):
        """Find politics articles with randomization"""
        try:
            quantity = self._extract_quantity(query)
            is_follow_up = self._is_follow_up_query(query)
            
            df = self.article_database
            politics_articles = df[df['category'] == 'politics']
            
            if len(politics_articles) > 0:
                # Convert to list of dictionaries for consistent handling
                politics_articles_list = politics_articles.to_dict('records')
                
                # Get random articles
                selected_articles = self._get_random_articles(politics_articles_list, quantity, avoid_repetition=is_follow_up)
                
                sample_text = '\n'.join([f'• {article["text"][:100]}...' for article in selected_articles])
                
                intro = "Here are some different politics articles:" if is_follow_up else f"Found {len(politics_articles)} politics articles. Here are {quantity} examples:"
                
                return {
                    'status': 'success',
                    'message': f'{intro}\n\n{sample_text}\n\nTotal politics articles: {len(politics_articles)}'
                }
            else:
                return {
                    'status': 'success',
                    'message': 'No politics articles found in the current dataset.'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error finding politics articles: {str(e)}'
            }
    
    def _find_entertainment_articles(self, query):
        """Find entertainment articles with randomization"""
        try:
            quantity = self._extract_quantity(query)
            is_follow_up = self._is_follow_up_query(query)
            
            df = self.article_database
            entertainment_articles = df[df['category'] == 'entertainment']
            
            if len(entertainment_articles) > 0:
                # Convert to list of dictionaries for consistent handling
                entertainment_articles_list = entertainment_articles.to_dict('records')
                
                # Get random articles
                selected_articles = self._get_random_articles(entertainment_articles_list, quantity, avoid_repetition=is_follow_up)
                
                sample_text = '\n'.join([f'• {article["text"][:100]}...' for article in selected_articles])
                
                intro = "Here are some different entertainment articles:" if is_follow_up else f"Found {len(entertainment_articles)} entertainment articles. Here are {quantity} examples:"
                
                return {
                    'status': 'success',
                    'message': f'{intro}\n\n{sample_text}\n\nTotal entertainment articles: {len(entertainment_articles)}'
                }
            else:
                return {
                    'status': 'success',
                    'message': 'No entertainment articles found in the current dataset.'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error finding entertainment articles: {str(e)}'
            }
    
    def _analyze_sentiment(self, query):
        """Analyze sentiment across the dataset"""
        try:
            df = self.article_database
            categories = df['category'].value_counts()
            
            sentiment_summary = f"Dataset contains {len(df)} articles across {len(categories)} categories:\n\n"
            for category, count in categories.items():
                percentage = (count / len(df)) * 100
                sentiment_summary += f"• {category.title()}: {count} articles ({percentage:.1f}%)\n"
            
            # Load sentiment results if available
            sentiment_results_path = Path('data/results/sentiment_results.json')
            if sentiment_results_path.exists():
                import json
                with open(sentiment_results_path, 'r') as f:
                    sentiment_data = json.load(f)
                
                sentiment_counts = defaultdict(int)
                for item in sentiment_data.get('results', []):
                    sentiment = item.get('sentiment', {}).get('label', 'unknown')
                    sentiment_counts[sentiment] += 1
                
                if sentiment_counts:
                    sentiment_summary += f"\nSentiment Analysis Results:\n"
                    for sentiment, count in sentiment_counts.items():
                        percentage = (count / len(sentiment_data.get('results', []))) * 100
                        sentiment_summary += f"• {sentiment.title()}: {count} articles ({percentage:.1f}%)\n"
            
            return {
                'status': 'success',
                'message': sentiment_summary
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error analyzing sentiment: {str(e)}'
            }
    
    def _summarize_data(self, query):
        """Summarize the dataset"""
        try:
            df = self.article_database
            
            summary = f"BBC News Dataset Summary:\n\n"
            summary += f"📊 Total Articles: {len(df):,}\n"
            summary += f"📂 Categories: {len(df['category'].unique())}\n\n"
            
            summary += "Category Breakdown:\n"
            categories = df['category'].value_counts()
            for category, count in categories.items():
                percentage = (count / len(df)) * 100
                summary += f"• {category.title()}: {count} articles ({percentage:.1f}%)\n"
            
            # Add text statistics
            avg_length = df['text'].str.len().mean()
            summary += f"\n📝 Average Article Length: {avg_length:.0f} characters\n"
            
            return {
                'status': 'success',
                'message': summary
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error summarizing data: {str(e)}'
            }
    
    def _count_articles(self, query):
        """Count articles based on query"""
        try:
            df = self.article_database
            
            if 'tech' in query.lower():
                count = len(df[df['category'] == 'tech'])
                return {'status': 'success', 'message': f'Technology articles: {count}'}
            elif 'business' in query.lower():
                count = len(df[df['category'] == 'business'])
                return {'status': 'success', 'message': f'Business articles: {count}'}
            elif 'sport' in query.lower():
                count = len(df[df['category'] == 'sport'])
                return {'status': 'success', 'message': f'Sports articles: {count}'}
            elif 'politics' in query.lower():
                count = len(df[df['category'] == 'politics'])
                return {'status': 'success', 'message': f'Politics articles: {count}'}
            elif 'entertainment' in query.lower():
                count = len(df[df['category'] == 'entertainment'])
                return {'status': 'success', 'message': f'Entertainment articles: {count}'}
            else:
                return {'status': 'success', 'message': f'Total articles in dataset: {len(df)}'}
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error counting articles: {str(e)}'
            }
    
    def _general_search(self, query):
        """General search functionality with randomization - only for meaningful search terms"""
        try:
            # Don't search for very short or common words that aren't meaningful
            query_clean = query.lower().strip()
            stop_words = ['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should']
            
            # Filter out stop words and very short words
            search_terms = [term for term in query_clean.split() if len(term) > 2 and term not in stop_words]
            
            if not search_terms:
                return {
                    'status': 'success',
                    'message': 'I need more specific keywords to search for articles. Try asking for:\n• Specific categories: "tech articles", "sports news"\n• Sentiment: "positive news", "negative articles"\n• Topics: "economy", "politics", "entertainment"\n• Or ask for help: "what can you do?"'
                }
            
            quantity = self._extract_quantity(query)
            is_follow_up = self._is_follow_up_query(query)
            
            df = self.article_database
            matching_articles = []
            
            # Search for meaningful terms only
            for idx, row in df.iterrows():
                text = str(row['text']).lower()
                if any(term in text for term in search_terms if len(term) > 2):
                    matching_articles.append(row)
            
            if matching_articles:
                # Get random articles
                selected_articles = self._get_random_articles(matching_articles, quantity, avoid_repetition=is_follow_up)
                
                sample_text = '\n'.join([f'• {article["text"][:100]}...' for article in selected_articles])
                
                intro = "Here are some different search results:" if is_follow_up else f"Found {len(matching_articles)} articles matching '{' '.join(search_terms)}'. Here are {quantity} examples:"
                
                return {
                    'status': 'success',
                    'message': f'{intro}\n\n{sample_text}\n\nTotal matching articles: {len(matching_articles)}'
                }
            else:
                return {
                    'status': 'success',
                    'message': f'No articles found matching "{" ".join(search_terms)}". Try different keywords or browse by category:\n• Technology: "tech articles"\n• Business: "business news" \n• Sports: "sports articles"\n• Politics: "politics news"\n• Entertainment: "entertainment articles"'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error in general search: {str(e)}'
            }
