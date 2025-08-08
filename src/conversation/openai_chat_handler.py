#!/usr/bin/env python3
"""
OpenAI-Powered Chat Handler for NewsBot 2.0
Direct integration with OpenAI for natural language conversations
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    import openai
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logging.warning("OpenAI library not available")

class OpenAIChatHandler:
    """
    Direct OpenAI-powered chat handler for NewsBot 2.0
    Provides natural language conversation capabilities with intelligent responses
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenAI chat handler
        
        Args:
            config: Configuration dictionary with API settings
        """
        self.config = config or {}
        
        # API configuration
        self.api_key = self.config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        self.model = self.config.get('model', 'gpt-4o')  # Using GPT-4o for better speed/performance balance
        self.max_completion_tokens = self.config.get('max_completion_tokens', 800)  # Reduced for faster responses
        self.temperature = self.config.get('temperature', 0.7)  # Optimized temperature
        
        # Initialize OpenAI client
        self.client = None
        self.is_available = False
        
        if not HAS_OPENAI:
            logging.warning("OpenAI library not available. Chat features will be limited.")
            self.is_available = False
            return
        
        # Check for placeholder or missing API key
        if not self.api_key or self.api_key in ['your-openai-api-key-here', 'your-openai-api-key', 'sk-your-key-here']:
            logging.warning("OpenAI API key not configured (using placeholder). Advanced chat features disabled.")
            logging.info("To enable OpenAI features, set a valid API key in .env file")
            self.is_available = False
            return
        
        # Validate API key format
        if not self.api_key.startswith('sk-'):
            logging.warning("OpenAI API key format may be invalid")
            
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.is_available = True
            logging.info(f"✅ OpenAI chat handler initialized successfully with model: {self.model}")
        except Exception as e:
            logging.warning(f"OpenAI initialization failed: {e}")
            logging.info("Continuing with limited chat functionality")
            self.is_available = False
        
        # Data sources
        self.article_database = None
        self.analysis_results = None
        
        # Conversation context
        self.conversation_history = []
        
        # Performance optimizations
        self.response_cache = {}  # Cache responses for similar queries
        self.article_cache = None  # Cache filtered articles
        
    def set_data_sources(self, article_database: pd.DataFrame = None, analysis_results: Dict[str, Any] = None):
        """Set data sources for context-aware responses"""
        self.article_database = article_database
        self.analysis_results = analysis_results or {}
        
        if article_database is not None:
            logging.info(f"✅ Chat handler loaded {len(article_database)} articles for context")
    
    def process_chat_query(self, user_query: str, user_id: str = None) -> Dict[str, Any]:
        """
        Process chat query using OpenAI with performance optimizations
        
        Args:
            user_query: User's natural language query
            user_id: Optional user identifier
            
        Returns:
            Dict with response and metadata
        """
        if not self.is_available:
            return {
                'response': "I'm having trouble connecting to my AI language model right now. Please try again or rephrase your question.",
                'error': 'openai_unavailable',
                'source': 'fallback',
                'query_type': 'error',
                'processing_time': 0.0,
                'recommendations': [
                    "Check your OpenAI API key configuration",
                    "Ensure you have a valid API key in your .env file", 
                    "Try asking a simpler question about the news articles"
                ]
            }
        
        # PERFORMANCE: Check cache first for similar queries
        query_key = user_query.lower().strip()
        if query_key in self.response_cache:
            cached_response = self.response_cache[query_key].copy()
            cached_response['source'] = 'cache'
            cached_response['cached'] = True
            return cached_response
        
        try:
            # Handle specific query types
            if 'tech' in user_query.lower() and 'articles' in user_query.lower():
                result = self._handle_tech_articles_query(user_query)
                # Cache the result
                self.response_cache[query_key] = result.copy()
                return result
            elif 'positive' in user_query.lower() and 'news' in user_query.lower():
                return self._handle_positive_news_query(user_query)
            elif 'hello' in user_query.lower() or 'hi' in user_query.lower():
                return self._handle_greeting(user_query)
            else:
                return self._handle_general_query(user_query)
                
        except Exception as e:
            logging.error(f"Error processing chat query: {e}")
            return {
                'status': 'error',
                'message': f"Sorry, I encountered an error processing your request: {str(e)}",
                'source': 'error'
            }
    
    def _handle_tech_articles_query(self, user_query: str) -> Dict[str, Any]:
        """Handle queries for tech articles"""
        if self.article_database is None:
            return self._generate_openai_response(
                user_query,
                "I don't have access to article data right now. Please load the dataset first."
            )
        
        # PERFORMANCE: Use cached tech articles if available
        if self.article_cache is None:
            # Filter tech articles (cache this for future use)
            tech_articles = self.article_database[
                self.article_database['category'].str.contains('tech', case=False, na=False)
            ]
            
            if tech_articles.empty:
                # Check for technology-related keywords in titles/content
                tech_keywords = ['technology', 'tech', 'computer', 'software', 'AI', 'digital', 'internet']
                tech_mask = self.article_database['text'].str.contains('|'.join(tech_keywords), case=False, na=False)
                tech_articles = self.article_database[tech_mask]
            
            # Cache the filtered articles
            self.article_cache = tech_articles
        else:
            tech_articles = self.article_cache
        
        if tech_articles.empty:
            return {
                'status': 'success',
                'message': "I couldn't find any technology articles in the current dataset. Try searching for specific tech topics or check if the dataset contains technology news.",
                'articles_found': 0,
                'source': 'openai_chat'
            }
        
        # Get a sample of tech articles
        sample_size = min(5, len(tech_articles))
        sample_articles = tech_articles.head(sample_size)
        
        # Create context for OpenAI
        articles_context = ""
        for i, (_, article) in enumerate(sample_articles.iterrows(), 1):
            # Handle different column names in the dataset
            title = article.get('title', article.get('text', 'No title'))[:100] + "..."
            description = article.get('description', article.get('text', 'No description'))[:200]
            category = article.get('category', 'Unknown')
            articles_context += f"{i}. [{category.upper()}] {title}\n   {description}...\n\n"
        
        prompt = f"""
        The user asked: "{user_query}"
        
        Here are {sample_size} technology articles from our database:
        
        {articles_context}
        
        Please provide a helpful response that:
        1. Acknowledges their request for tech articles
        2. Summarizes what we found
        3. Highlights key topics or trends from these articles
        4. Offers to help with more specific tech topics
        
        Be conversational and informative.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are NewsBot AI, a helpful news analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=self.max_completion_tokens,
                temperature=self.temperature
            )
            
            result = {
                'status': 'success',
                'message': response.choices[0].message.content,
                'articles_found': len(tech_articles),
                'sample_articles': sample_articles.to_dict('records'),
                'source': 'openai_chat',
                'query': user_query
            }
            
            # Cache the response for future use  
            query_key = user_query.lower().strip()
            self.response_cache[query_key] = result.copy()
            return result
            
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return {
                'status': 'success',
                'message': f"I found {len(tech_articles)} technology articles for you! Here are the top {sample_size}:\n\n" + 
                          "\n".join([f"• [{row['category'].upper()}] {row['text'][:100]}..." for _, row in sample_articles.iterrows()]),
                'articles_found': len(tech_articles),
                'sample_articles': sample_articles.to_dict('records'),
                'source': 'fallback'
            }
    
    def _handle_positive_news_query(self, user_query: str) -> Dict[str, Any]:
        """Handle queries for positive news"""
        if self.article_database is None:
            return self._generate_openai_response(
                user_query,
                "I don't have access to article data right now. Please load the dataset first."
            )
        
        # Try to find positive news using sentiment or keywords
        positive_keywords = ['success', 'achievement', 'breakthrough', 'positive', 'good news', 'victory', 'win']
        # Use 'text' column instead of 'title' since that's what the dataset has
        positive_mask = self.article_database['text'].str.contains('|'.join(positive_keywords), case=False, na=False)
        positive_articles = self.article_database[positive_mask]
        
        # If we have sentiment analysis results, use those
        if 'sentiment' in self.article_database.columns:
            sentiment_positive = self.article_database[
                self.article_database['sentiment'].str.contains('positive', case=False, na=False)
            ]
            positive_articles = pd.concat([positive_articles, sentiment_positive]).drop_duplicates()
        
        if positive_articles.empty:
            return {
                'status': 'success',
                'message': "I couldn't find specifically labeled positive news in the current dataset. However, I can help you analyze the sentiment of articles or search for specific uplifting topics. What kind of positive news are you interested in?",
                'articles_found': 0,
                'source': 'openai_chat'
            }
        
        sample_size = min(5, len(positive_articles))
        sample_articles = positive_articles.head(sample_size)
        
        return {
            'status': 'success',
            'message': f"I found {len(positive_articles)} potentially positive news articles! Here are the top {sample_size} that seem uplifting:\n\n" + 
                      "\n".join([f"• [{row['category'].upper()}] {row['text'][:100]}..." for _, row in sample_articles.iterrows()]),
            'articles_found': len(positive_articles),
            'sample_articles': sample_articles.to_dict('records'),
            'source': 'openai_chat'
        }
    
    def _handle_greeting(self, user_query: str) -> Dict[str, Any]:
        """Handle greeting messages"""
        greeting_response = """
Hello! I'm your AI-powered NewsBot assistant. I can help you with:

🔍 **Finding Articles**: "Find tech articles" or "Show me business news"
📊 **Analysis**: "Analyze sentiment" or "Extract entities from politics articles"  
📈 **Insights**: "What are the trending topics?" or "Summarize recent news"
🌍 **Multilingual**: "Translate this article" or "Detect language"

What would you like to explore today?
        """
        
        return {
            'status': 'success',
            'message': greeting_response.strip(),
            'source': 'openai_chat',
            'query': user_query
        }
    
    def _handle_general_query(self, user_query: str) -> Dict[str, Any]:
        """Handle general queries using OpenAI"""
        # Prepare context about available data
        data_context = "No article data is currently loaded."
        if self.article_database is not None:
            num_articles = len(self.article_database)
            categories = self.article_database.get('category', pd.Series([])).value_counts().head(5)
            data_context = f"I have access to {num_articles} articles"
            if not categories.empty:
                data_context += f" across categories like: {', '.join(categories.index.tolist())}"
        
        prompt = f"""
        User query: "{user_query}"
        
        Available data: {data_context}
        
        You are NewsBot AI, an intelligent news analysis assistant. Respond to the user's query in a helpful and conversational way. If they're asking for specific data analysis or articles that I should be able to provide from my database, let them know what I can do and guide them on how to phrase their request.
        
        Be concise but informative.
        """
        
        return self._generate_openai_response(user_query, prompt)
    
    def _generate_openai_response(self, user_query: str, context_or_prompt: str) -> Dict[str, Any]:
        """Generate response using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are NewsBot AI, a helpful and intelligent news analysis assistant."},
                    {"role": "user", "content": context_or_prompt}
                ],
                max_completion_tokens=self.max_completion_tokens,
                temperature=self.temperature
            )
            
            return {
                'status': 'success',
                'message': response.choices[0].message.content,
                'source': 'openai_chat',
                'query': user_query,
                'token_usage': response.usage.total_tokens if hasattr(response, 'usage') else None
            }
            
        except Exception as e:
            logging.error(f"OpenAI API error in general response: {e}")
            return {
                'status': 'error',
                'message': "I'm having trouble connecting to my AI language model right now. Please try again or rephrase your question.",
                'source': 'error',
                'query': user_query
            }

