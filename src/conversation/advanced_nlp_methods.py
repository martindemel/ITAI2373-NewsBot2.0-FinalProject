#!/usr/bin/env python3
"""
Advanced NLP Methods for Query Processor
ML/NLP methods that replace rule-based approaches
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
import json
from datetime import datetime, timedelta
from collections import defaultdict

class AdvancedNLPMethods:
    """Advanced NLP methods for intelligent query processing"""
    
    def _classify_intent_with_ml(self, query: str) -> Dict[str, Any]:
        """
        Use ML-based intent classification instead of keyword matching
        
        Args:
            query: User query
            
        Returns:
            Dict with intent and confidence score
        """
        try:
            # Use the advanced intent classifier
            intent, confidence = self.intent_classifier.classify_intent(query)
            
            return {
                'intent': intent,
                'confidence': confidence,
                'method': 'ml_classifier'
            }
        except Exception as e:
            logging.warning(f"ML intent classification failed, using fallback: {e}")
            # Fallback to simple pattern matching
            return self._fallback_intent_classification(query)

    def _extract_query_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract named entities and query parameters using NER
        
        Args:
            query: User query
            
        Returns:
            Dict with extracted entities by type
        """
        try:
            # Use advanced NER extractor
            entities = self.ner_extractor.extract_entities(query)
            
            # Also extract query-specific entities
            query_entities = {
                'dates': self._extract_temporal_entities(query),
                'numbers': self._extract_numerical_entities(query),
                'categories': self._extract_category_entities(query),
                'sentiment_words': self._extract_sentiment_entities(query)
            }
            
            # Merge entities
            if isinstance(entities, dict):
                entities.update(query_entities)
            else:
                entities = query_entities
                
            return entities
            
        except Exception as e:
            logging.warning(f"Entity extraction failed: {e}")
            return self._extract_basic_entities(query)

    def _parse_query_parameters(self, query: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse query parameters like quantity, filters, time ranges
        
        Args:
            query: User query
            entities: Extracted entities
            
        Returns:
            Dict with parsed parameters
        """
        params = {
            'quantity': self._extract_quantity(query),
            'sentiment_filter': self._extract_sentiment_filter(query),
            'category_filter': self._extract_category_filter(query, entities),
            'time_filter': self._extract_time_filter(query, entities),
            'similarity_threshold': 0.7,
            'diversify_results': self._is_follow_up_query(query)
        }
        
        return params

    def _enhance_with_openai(self, query: str, current_intent: str, entities: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Use OpenAI to enhance query understanding for complex queries
        
        Args:
            query: User query
            current_intent: Current detected intent
            entities: Extracted entities
            
        Returns:
            Enhanced understanding or None
        """
        if not self.openai_client:
            return None
            
        try:
            prompt = f"""
            Analyze this news query and extract the following information:
            Query: "{query}"
            Current intent: {current_intent}
            Extracted entities: {entities}
            
            Please provide:
            1. Refined intent (search_articles, analyze_sentiment, classify_content, summarize_text, extract_entities, get_insights, compare_coverage, track_trends)
            2. Key search terms
            3. Filters (category, sentiment, time period)
            4. Desired quantity/format of results
            
            Respond in JSON format:
            {{
                "intent": "...",
                "search_terms": [...],
                "filters": {{"category": "...", "sentiment": "...", "time": "..."}},
                "quantity": 5,
                "reasoning": "..."
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                'intent': result.get('intent', current_intent),
                'parameters': {
                    'search_terms': result.get('search_terms', []),
                    'filters': result.get('filters', {}),
                    'quantity': result.get('quantity', 3)
                },
                'reasoning': result.get('reasoning', ''),
                'method': 'openai_enhanced'
            }
            
        except Exception as e:
            logging.warning(f"OpenAI enhancement failed: {e}")
            return None

    def _semantic_article_search(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform semantic search using embeddings instead of keyword matching
        
        Args:
            query: User query
            params: Query parameters
            
        Returns:
            Search results with articles and metadata
        """
        try:
            # Use semantic embeddings for similarity search
            query_embedding = self.semantic_embeddings.encode_query(query)
            
            # Find semantically similar articles
            similar_articles = self.semantic_embeddings.find_similar_documents(
                query_embedding,
                top_k=params.get('quantity', 5) * 3,  # Get more for filtering
                threshold=params.get('similarity_threshold', 0.3)
            )
            
            # Apply filters
            filtered_articles = self._apply_filters(similar_articles, params)
            
            # Diversify results if it's a follow-up query
            if params.get('diversify_results', False):
                filtered_articles = self._diversify_results(filtered_articles)
            
            # Limit to requested quantity
            final_articles = filtered_articles[:params.get('quantity', 5)]
            
            return {
                'status': 'success',
                'articles': final_articles,
                'total_found': len(similar_articles),
                'total_after_filtering': len(filtered_articles),
                'search_method': 'semantic_embeddings',
                'query': query
            }
            
        except Exception as e:
            logging.error(f"Semantic search failed: {e}")
            return self._fallback_search(query, params)

    def _advanced_sentiment_analysis(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform advanced sentiment analysis using ML models
        
        Args:
            query: User query
            params: Query parameters
            
        Returns:
            Sentiment analysis results
        """
        try:
            # Use advanced sentiment analyzer
            if 'sentiment_filter' in params and params['sentiment_filter']:
                # Filter articles by sentiment first
                filtered_articles = self._filter_by_sentiment(params['sentiment_filter'])
                articles_to_analyze = filtered_articles[:100]  # Analyze top 100
            else:
                articles_to_analyze = self.article_database.sample(n=min(100, len(self.article_database)))
            
            # Perform sentiment analysis
            sentiment_results = []
            for _, article in articles_to_analyze.iterrows():
                sentiment = self.sentiment_analyzer.analyze_sentiment(article['text'])
                sentiment_results.append({
                    'article_id': article.name,
                    'text_preview': article['text'][:100] + "...",
                    'category': article.get('category', 'unknown'),
                    'sentiment': sentiment
                })
            
            # Generate summary statistics
            sentiments = [r['sentiment']['label'] for r in sentiment_results if 'label' in r['sentiment']]
            sentiment_distribution = pd.Series(sentiments).value_counts().to_dict()
            
            return {
                'status': 'success',
                'sentiment_results': sentiment_results[:params.get('quantity', 10)],
                'sentiment_distribution': sentiment_distribution,
                'total_analyzed': len(sentiment_results),
                'method': 'advanced_ml'
            }
            
        except Exception as e:
            logging.error(f"Advanced sentiment analysis failed: {e}")
            return {'status': 'error', 'message': f'Sentiment analysis failed: {e}'}

    def _intelligent_summarization(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate intelligent summaries using advanced language models
        
        Args:
            query: User query
            params: Query parameters
            
        Returns:
            Summarization results
        """
        try:
            # Determine what to summarize based on query
            if 'articles' in params:
                articles_to_summarize = params['articles']
            else:
                # Find relevant articles first
                search_result = self._semantic_article_search(query, params)
                articles_to_summarize = search_result.get('articles', [])
            
            if not articles_to_summarize:
                return {'status': 'error', 'message': 'No articles found to summarize'}
            
            summaries = []
            for article in articles_to_summarize[:params.get('quantity', 5)]:
                try:
                    if isinstance(article, dict):
                        text = article.get('text', '')
                    else:
                        text = article
                    
                    # Use intelligent summarizer
                    summary = self.summarizer.summarize_text(
                        text,
                        summary_type='balanced',
                        max_length=150
                    )
                    
                    summaries.append({
                        'original_text': text[:200] + "...",
                        'summary': summary,
                        'summary_length': len(summary.split()),
                        'compression_ratio': len(summary.split()) / len(text.split()) if text else 0
                    })
                    
                except Exception as e:
                    logging.warning(f"Failed to summarize article: {e}")
                    continue
            
            return {
                'status': 'success',
                'summaries': summaries,
                'total_summarized': len(summaries),
                'method': 'advanced_language_models'
            }
            
        except Exception as e:
            logging.error(f"Intelligent summarization failed: {e}")
            return {'status': 'error', 'message': f'Summarization failed: {e}'}

    def _generate_intelligent_response(self, query: str, intent: str, result: Dict[str, Any], 
                                     entities: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate intelligent natural language response using advanced NLP
        
        Args:
            query: Original user query
            intent: Detected intent
            result: Query execution result
            entities: Extracted entities
            params: Query parameters
            
        Returns:
            Formatted response with intelligent natural language
        """
        try:
            # Use the response generator for intelligent responses
            response = self.response_generator.generate_response(
                query=query,
                intent=intent,
                results=result,
                entities=entities,
                context=self.conversation_context,
                user_preferences=self.user_preferences
            )
            
            # Add NewsBot-specific enhancements
            if result.get('status') == 'success':
                if intent == 'search_articles' and 'articles' in result:
                    articles = result['articles']
                    if articles:
                        response['message'] = f"🔍 **Found {len(articles)} relevant articles using semantic search:**\n\n"
                        for i, article in enumerate(articles[:3], 1):
                            if isinstance(article, dict):
                                text_preview = article.get('text', '')[:100] + "..."
                                category = article.get('category', 'Unknown')
                                response['message'] += f"{i}. **[{category}]** {text_preview}\n"
                        
                        if len(articles) > 3:
                            response['message'] += f"\n*...and {len(articles) - 3} more articles*"
                            
                elif intent == 'analyze_sentiment' and 'sentiment_distribution' in result:
                    dist = result['sentiment_distribution']
                    response['message'] = f"📊 **Sentiment Analysis Results:**\n\n"
                    for sentiment, count in dist.items():
                        response['message'] += f"• {sentiment.title()}: {count} articles\n"
                        
                elif intent == 'summarize_text' and 'summaries' in result:
                    summaries = result['summaries']
                    response['message'] = f"📝 **Article Summaries ({len(summaries)} articles):**\n\n"
                    for i, summary in enumerate(summaries, 1):
                        response['message'] += f"{i}. {summary['summary']}\n\n"
            
            # Add metadata
            response.update({
                'status': result.get('status', 'success'),
                'processing_method': 'advanced_nlp',
                'intent': intent,
                'confidence': result.get('confidence', 0.8)
            })
            
            return response
            
        except Exception as e:
            logging.error(f"Response generation failed: {e}")
            return {
                'status': 'error',
                'message': f'Failed to generate response: {e}',
                'raw_result': result
            }

    # Helper methods for entity extraction and filtering
    
    def _extract_temporal_entities(self, query: str) -> List[str]:
        """Extract temporal expressions from query"""
        temporal_patterns = [
            r'\b(today|yesterday|tomorrow)\b',
            r'\b(this|last|next)\s+(week|month|year)\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b'
        ]
        
        entities = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, query.lower())
            entities.extend(matches)
        
        return entities

    def _extract_numerical_entities(self, query: str) -> List[str]:
        """Extract numbers and quantities from query"""
        patterns = [
            r'\b\d+\b',
            r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b'
        ]
        
        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, query.lower())
            entities.extend(matches)
        
        return entities

    def _extract_category_entities(self, query: str) -> List[str]:
        """Extract news categories from query"""
        categories = ['technology', 'tech', 'business', 'politics', 'sport', 'sports', 'entertainment']
        found_categories = []
        
        for category in categories:
            if category in query.lower():
                found_categories.append(category)
        
        return found_categories

    def _extract_sentiment_entities(self, query: str) -> List[str]:
        """Extract sentiment-related words from query"""
        sentiment_words = {
            'positive': ['positive', 'good', 'happy', 'excellent', 'great', 'successful'],
            'negative': ['negative', 'bad', 'sad', 'terrible', 'crisis', 'problem'],
            'neutral': ['neutral', 'objective', 'factual']
        }
        
        found_sentiments = []
        for sentiment_type, words in sentiment_words.items():
            for word in words:
                if word in query.lower():
                    found_sentiments.append(sentiment_type)
                    break
        
        return found_sentiments

    def _apply_filters(self, articles: List[Dict[str, Any]], params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to article list"""
        filtered = articles.copy()
        
        # Apply category filter
        if params.get('category_filter'):
            category = params['category_filter'].lower()
            filtered = [a for a in filtered if a.get('category', '').lower() == category]
        
        # Apply sentiment filter
        if params.get('sentiment_filter'):
            # This would require pre-computed sentiment scores
            pass
        
        # Apply time filter
        if params.get('time_filter'):
            # This would require date information in articles
            pass
        
        return filtered

    def _diversify_results(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Diversify results to avoid repetition"""
        # Simple diversification by ensuring different categories
        seen_categories = set()
        diversified = []
        
        for article in articles:
            category = article.get('category', 'unknown')
            if category not in seen_categories or len(diversified) < 3:
                diversified.append(article)
                seen_categories.add(category)
        
        return diversified

    def _fallback_search(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback search when semantic search fails"""
        # Simple keyword-based search as fallback
        search_terms = query.lower().split()
        matching_articles = []
        
        for _, article in self.article_database.iterrows():
            text = article['text'].lower()
            if any(term in text for term in search_terms):
                matching_articles.append({
                    'text': article['text'],
                    'category': article.get('category', 'unknown'),
                    'score': 0.5  # Default score for fallback
                })
        
        return {
            'status': 'success',
            'articles': matching_articles[:params.get('quantity', 5)],
            'total_found': len(matching_articles),
            'search_method': 'keyword_fallback',
            'query': query
        }

    def _fallback_intent_classification(self, query: str) -> Dict[str, Any]:
        """Fallback intent classification using simple patterns"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['find', 'search', 'show', 'get']):
            return {'intent': 'search_articles', 'confidence': 0.6, 'method': 'pattern_fallback'}
        elif any(word in query_lower for word in ['sentiment', 'emotion', 'feeling']):
            return {'intent': 'analyze_sentiment', 'confidence': 0.6, 'method': 'pattern_fallback'}
        elif any(word in query_lower for word in ['summarize', 'summary']):
            return {'intent': 'summarize_text', 'confidence': 0.6, 'method': 'pattern_fallback'}
        else:
            return {'intent': 'search_articles', 'confidence': 0.4, 'method': 'default_fallback'}