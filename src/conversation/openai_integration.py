#!/usr/bin/env python3
"""
OpenAI Integration for NewsBot 2.0
Advanced language understanding and generation using OpenAI's GPT models
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import aiohttp

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logging.warning("OpenAI library not available. Install with: pip install openai")

class OpenAIIntegration:
    """
    Advanced OpenAI integration for NewsBot 2.0
    Provides intelligent query understanding, response generation, and content analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenAI integration
        
        Args:
            config: Configuration dictionary with API settings
        """
        self.config = config or {}
        
        # API configuration
        self.api_key = self.config.get('api_key') or os.getenv('OPENAI_API_KEY')
        self.model = self.config.get('model', 'gpt-5')
        self.max_tokens = self.config.get('max_tokens', 1000)
        self.temperature = self.config.get('temperature', 0.1)
        self.timeout = self.config.get('timeout', 30)
        
        # Rate limiting
        self.rate_limit = self.config.get('rate_limit', 60)  # requests per minute
        self.request_times = []
        
        # Initialize OpenAI client
        self.client = None
        self.is_available = False
        
        if HAS_OPENAI and self.api_key:
            # Check for placeholder API key
            if self.api_key not in ['your-openai-api-key-here', 'your-openai-api-key', 'sk-your-key-here']:
                try:
                    openai.api_key = self.api_key
                    self.client = openai
                    self.is_available = True
                    logging.info("OpenAI integration initialized successfully")
                except Exception as e:
                    logging.warning(f"Failed to initialize OpenAI client: {e}")
            else:
                logging.info("OpenAI API key not configured (placeholder detected) - integration disabled")
        else:
            logging.info("OpenAI integration not available - API key missing or library not installed")
        
        # Usage statistics
        self.usage_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_used': 0,
            'average_response_time': 0.0
        }
        
        # Cache for frequent queries
        self.response_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = time.time()
        
        # Remove old requests (older than 1 minute)
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # Check if we're under the rate limit
        if len(self.request_times) >= self.rate_limit:
            return False
        
        self.request_times.append(now)
        return True
    
    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key for request"""
        cache_data = {
            'prompt': prompt,
            'model': kwargs.get('model', self.model),
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens)
        }
        return str(hash(json.dumps(cache_data, sort_keys=True)))
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cached response is still valid"""
        return time.time() - timestamp < self.cache_ttl
    
    async def enhance_query_understanding(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhance query understanding using OpenAI's language models
        
        Args:
            query: User query to analyze
            context: Optional context information
            
        Returns:
            Enhanced query understanding results
        """
        if not self.is_available:
            return {'error': 'OpenAI not available'}
        
        if not self._check_rate_limit():
            return {'error': 'Rate limit exceeded'}
        
        try:
            # Create enhanced prompt for query understanding
            system_prompt = """You are NewsBot AI, an advanced news analysis assistant. 
Analyze the user's query and extract:
1. Primary intent (search, analyze, summarize, translate, etc.)
2. Key entities and topics
3. Specific parameters (quantity, time range, sentiment, etc.)
4. Context clues for better understanding

Respond in JSON format with structured data."""
            
            user_prompt = f"""
Query: "{query}"
Context: {json.dumps(context) if context else "None"}

Analyze this query and provide:
{{
    "intent": "primary intent (search_articles, analyze_sentiment, etc.)",
    "entities": {{"people": [], "organizations": [], "locations": [], "topics": []}},
    "parameters": {{"quantity": 5, "sentiment": null, "category": null, "time_range": null}},
    "search_terms": ["key", "search", "terms"],
    "confidence": 0.95,
    "reasoning": "explanation of analysis"
}}
"""
            
            start_time = time.time()
            
            response = await self._make_request(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.1
            )
            
            response_time = time.time() - start_time
            self._update_stats(True, response_time, response.get('usage', {}))
            
            # Parse JSON response
            content = response['choices'][0]['message']['content']
            try:
                result = json.loads(content)
                result['processing_time'] = response_time
                result['source'] = 'openai'
                return result
            except json.JSONDecodeError:
                return {
                    'error': 'Failed to parse OpenAI response',
                    'raw_response': content
                }
                
        except Exception as e:
            self._update_stats(False, 0, {})
            logging.error(f"OpenAI query enhancement failed: {e}")
            return {'error': f'OpenAI request failed: {str(e)}'}
    
    async def generate_intelligent_response(self, query: str, analysis_results: Dict[str, Any], 
                                          conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate intelligent natural language response based on analysis results
        
        Args:
            query: Original user query
            analysis_results: Results from news analysis
            conversation_history: Previous conversation context
            
        Returns:
            Generated response
        """
        if not self.is_available:
            return {'message': 'Advanced response generation not available'}
        
        if not self._check_rate_limit():
            return {'message': 'Rate limit exceeded, please try again later'}
        
        try:
            # Create context-aware prompt
            system_prompt = """You are NewsBot AI, a sophisticated news analysis assistant.
Generate helpful, informative, and conversational responses based on the query and analysis results.

Guidelines:
- Be conversational and helpful
- Highlight key insights from the analysis
- Use appropriate formatting (bullets, numbers, etc.)
- Mention specific data points when relevant
- Adapt tone to the user's query
- If no results found, suggest alternatives"""
            
            # Prepare analysis summary
            analysis_summary = self._prepare_analysis_summary(analysis_results)
            
            user_prompt = f"""
User Query: "{query}"

Analysis Results:
{analysis_summary}

Conversation History: {json.dumps(conversation_history[-3:] if conversation_history else [])}

Generate a helpful, informative response that:
1. Directly addresses the user's query
2. Summarizes key findings from the analysis
3. Provides specific examples when available
4. Suggests follow-up questions if appropriate
"""
            
            start_time = time.time()
            
            response = await self._make_request(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
                temperature=0.3
            )
            
            response_time = time.time() - start_time
            self._update_stats(True, response_time, response.get('usage', {}))
            
            return {
                'message': response['choices'][0]['message']['content'],
                'processing_time': response_time,
                'source': 'openai',
                'token_usage': response.get('usage', {})
            }
            
        except Exception as e:
            self._update_stats(False, 0, {})
            logging.error(f"OpenAI response generation failed: {e}")
            return {
                'message': 'I encountered an issue generating a response. Here are the raw analysis results.',
                'error': str(e)
            }
    
    async def analyze_news_bias(self, article_text: str) -> Dict[str, Any]:
        """
        Analyze potential bias in news articles using OpenAI
        
        Args:
            article_text: News article text to analyze
            
        Returns:
            Bias analysis results
        """
        if not self.is_available:
            return {'error': 'OpenAI not available for bias analysis'}
        
        try:
            system_prompt = """You are an expert media analyst specializing in bias detection.
Analyze the given news article for potential bias indicators:
- Language tone and word choice
- Source diversity and attribution
- Factual vs. opinion content
- Political or ideological leanings
- Missing perspectives or context

Provide objective, balanced analysis."""
            
            user_prompt = f"""
Analyze this news article for potential bias:

"{article_text[:2000]}..."

Provide analysis in JSON format:
{{
    "bias_score": 0.3,
    "bias_type": "slight political lean",
    "language_indicators": ["loaded terms", "emotional language"],
    "source_quality": "well-sourced",
    "factual_content": 0.8,
    "missing_perspectives": ["opposing viewpoints"],
    "overall_assessment": "explanation",
    "confidence": 0.85
}}
"""
            
            response = await self._make_request(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.2
            )
            
            content = response['choices'][0]['message']['content']
            return json.loads(content)
            
        except Exception as e:
            logging.error(f"Bias analysis failed: {e}")
            return {'error': f'Bias analysis failed: {str(e)}'}
    
    async def generate_article_summary(self, article_text: str, summary_type: str = 'balanced') -> str:
        """
        Generate intelligent article summary using OpenAI
        
        Args:
            article_text: Article text to summarize
            summary_type: Type of summary (brief, balanced, detailed)
            
        Returns:
            Generated summary
        """
        if not self.is_available:
            return "Advanced summarization not available"
        
        try:
            length_instructions = {
                'brief': 'in 1-2 sentences',
                'balanced': 'in 3-4 sentences',
                'detailed': 'in 5-6 sentences with key details'
            }
            
            system_prompt = f"""You are an expert news summarizer. 
Create clear, accurate summaries that capture the essential information and main points.
Summarize {length_instructions.get(summary_type, 'concisely')}."""
            
            user_prompt = f"""Summarize this news article {length_instructions.get(summary_type, 'concisely')}:

"{article_text}"

Focus on:
- Main event or development
- Key people/organizations involved
- Important details and context
- Implications or significance"""
            
            response = await self._make_request(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.2
            )
            
            return response['choices'][0]['message']['content']
            
        except Exception as e:
            logging.error(f"Summary generation failed: {e}")
            return "Summary generation failed"
    
    async def _make_request(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Make request to OpenAI API with caching and error handling"""
        
        # Generate cache key
        cache_key = self._get_cache_key(str(messages), **kwargs)
        
        # Check cache
        if cache_key in self.response_cache:
            cached_response, timestamp = self.response_cache[cache_key]
            if self._is_cache_valid(timestamp):
                return cached_response
        
        # Make API request
        request_params = {
            'model': kwargs.get('model', self.model),
            'messages': messages,
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
            'temperature': kwargs.get('temperature', self.temperature),
            'timeout': self.timeout
        }
        
        response = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: self.client.ChatCompletion.create(**request_params)
        )
        
        # Cache response
        self.response_cache[cache_key] = (response, time.time())
        
        return response
    
    def _prepare_analysis_summary(self, results: Dict[str, Any]) -> str:
        """Prepare analysis results summary for OpenAI"""
        summary_parts = []
        
        if results.get('status') == 'success':
            if 'articles' in results:
                articles = results['articles']
                summary_parts.append(f"Found {len(articles)} articles")
                
                if articles:
                    # Sample article info
                    sample = articles[0]
                    if isinstance(sample, dict):
                        summary_parts.append(f"Sample: {sample.get('text', '')[:100]}...")
            
            if 'sentiment_distribution' in results:
                dist = results['sentiment_distribution']
                summary_parts.append(f"Sentiment: {dict(dist)}")
            
            if 'classification' in results:
                cls = results['classification']
                summary_parts.append(f"Category: {cls.get('category', 'unknown')}")
            
            if 'total_found' in results:
                summary_parts.append(f"Total found: {results['total_found']}")
        
        return "\n".join(summary_parts) if summary_parts else "No specific results available"
    
    def _update_stats(self, success: bool, response_time: float, usage: Dict[str, Any]):
        """Update usage statistics"""
        self.usage_stats['total_requests'] += 1
        
        if success:
            self.usage_stats['successful_requests'] += 1
            
            # Update average response time
            current_avg = self.usage_stats['average_response_time']
            total_successful = self.usage_stats['successful_requests']
            self.usage_stats['average_response_time'] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
            
            # Update token usage
            if 'total_tokens' in usage:
                self.usage_stats['total_tokens_used'] += usage['total_tokens']
        else:
            self.usage_stats['failed_requests'] += 1
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        success_rate = (
            self.usage_stats['successful_requests'] / self.usage_stats['total_requests']
            if self.usage_stats['total_requests'] > 0 else 0
        )
        
        return {
            **self.usage_stats,
            'success_rate': success_rate,
            'is_available': self.is_available,
            'cache_size': len(self.response_cache)
        }
    
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        logging.info("OpenAI response cache cleared")

# Synchronous wrapper functions for backward compatibility
class OpenAISync:
    """Synchronous wrapper for OpenAI integration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.async_client = OpenAIIntegration(config)
    
    def enhance_query_understanding(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Synchronous wrapper for query understanding"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.async_client.enhance_query_understanding(query, context)
        )
    
    def generate_intelligent_response(self, query: str, analysis_results: Dict[str, Any], 
                                    conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Synchronous wrapper for response generation"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.async_client.generate_intelligent_response(query, analysis_results, conversation_history)
        )