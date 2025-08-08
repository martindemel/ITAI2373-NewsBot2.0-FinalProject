#!/usr/bin/env python3
"""
Intent Classifier for NewsBot 2.0
Advanced intent detection for natural language queries about news content
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import re
import pickle
from collections import defaultdict, Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import nltk

# Try to import transformers for advanced intent classification
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("transformers not available for advanced intent classification.")

class IntentClassifier:
    """
    Advanced intent classification for natural language queries about news analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize intent classifier
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Define intent categories based on PDF requirements
        self.intent_categories = {
            'search_articles': {
                'description': 'Search for specific articles or content',
                'examples': [
                    'find articles about technology',
                    'search for news about climate change',
                    'show me articles containing covid',
                    'get news about artificial intelligence'
                ],
                'keywords': ['find', 'search', 'show', 'get', 'articles', 'news', 'about', 'containing']
            },
            'analyze_sentiment': {
                'description': 'Analyze sentiment or emotional tone',
                'examples': [
                    'what is the sentiment about elections',
                    'analyze emotions in sports news',
                    'show positive news this week',
                    'how negative is political coverage'
                ],
                'keywords': ['sentiment', 'emotion', 'positive', 'negative', 'feeling', 'mood', 'tone']
            },
            'classify_content': {
                'description': 'Classify or categorize content',
                'examples': [
                    'what category is this article',
                    'classify these news pieces',
                    'what type of news is this',
                    'categorize by topic'
                ],
                'keywords': ['classify', 'category', 'type', 'categorize', 'what kind', 'what type']
            },
            'summarize_text': {
                'description': 'Generate summaries of content',
                'examples': [
                    'summarize this article',
                    'give me a brief overview',
                    'what are the key points',
                    'create a summary of recent news'
                ],
                'keywords': ['summarize', 'summary', 'brief', 'overview', 'key points', 'main points']
            },
            'extract_entities': {
                'description': 'Extract named entities or key information',
                'examples': [
                    'who are the people mentioned',
                    'what companies are involved',
                    'extract all locations',
                    'find organizations in this text'
                ],
                'keywords': ['who', 'what companies', 'organizations', 'people', 'locations', 'extract', 'entities']
            },
            'get_insights': {
                'description': 'Get analytical insights and patterns',
                'examples': [
                    'what are the trends in tech news',
                    'give me insights about market coverage',
                    'analyze patterns in political reporting',
                    'what insights can you provide'
                ],
                'keywords': ['trends', 'insights', 'patterns', 'analysis', 'analytics', 'tell me about']
            },
            'compare_sources': {
                'description': 'Compare coverage across sources or languages',
                'examples': [
                    'compare coverage between english and spanish',
                    'how do different sources cover this topic',
                    'compare sentiment across languages',
                    'cross-language analysis'
                ],
                'keywords': ['compare', 'comparison', 'between', 'across', 'different', 'versus', 'cross']
            },
            'trend_analysis': {
                'description': 'Analyze trends over time',
                'examples': [
                    'show trends over the past month',
                    'how has coverage changed over time',
                    'track sentiment evolution',
                    'temporal analysis of topics'
                ],
                'keywords': ['trends', 'over time', 'evolution', 'changed', 'temporal', 'track', 'history']
            },
            'export_results': {
                'description': 'Export data or generate reports',
                'examples': [
                    'export these results to csv',
                    'generate a report',
                    'download the analysis',
                    'save this data'
                ],
                'keywords': ['export', 'download', 'save', 'generate', 'report', 'csv', 'file']
            },
            'get_help': {
                'description': 'Get help or information about capabilities',
                'examples': [
                    'what can you do',
                    'help me with analysis',
                    'how do I search for articles',
                    'what are your capabilities'
                ],
                'keywords': ['help', 'how', 'what can', 'capabilities', 'assistance', 'guide']
            }
        }
        
        # Initialize models
        self.sklearn_model = None
        self.tfidf_vectorizer = None
        self.transformer_classifier = None
        
        # Initialize transformer-based classifier if available
        if HAS_TRANSFORMERS:
            try:
                model_name = self.config.get('intent_model', 'microsoft/DialoGPT-medium')
                # Note: In production, you'd use a model specifically trained for intent classification
                # For now, we'll use a rule-based approach with ML backup
                logging.info("Transformer models available for intent classification")
            except Exception as e:
                logging.warning(f"Could not initialize transformer intent classifier: {e}")
        
        # Entity extraction patterns
        self.entity_patterns = {
            'time_expressions': [
                r'\b(today|yesterday|tomorrow|this week|last week|next week)\b',
                r'\b(this month|last month|next month|this year|last year)\b',
                r'\b(past \d+ days?|past \d+ weeks?|past \d+ months?)\b',
                r'\b(since \w+|until \w+|from \w+ to \w+)\b'
            ],
            'categories': [
                r'\b(politics|political|election|government)\b',
                r'\b(business|economy|economic|market|finance)\b',
                r'\b(technology|tech|ai|artificial intelligence|software)\b',
                r'\b(sports?|game|match|championship|olympics)\b',
                r'\b(entertainment|celebrity|movie|music|culture)\b'
            ],
            'sentiment_modifiers': [
                r'\b(positive|good|optimistic|upbeat|happy)\b',
                r'\b(negative|bad|pessimistic|sad|angry|critical)\b',
                r'\b(neutral|balanced|objective|mixed)\b'
            ],
            'languages': [
                r'\b(english|spanish|french|german|italian|portuguese|russian|chinese|japanese|korean|arabic|hindi)\b'
            ]
        }
        
        # Training data for sklearn model
        self.training_data = self._generate_training_data()
        
        # Train the model
        self._train_sklearn_model()
        
        # Statistics
        self.classification_stats = {
            'total_queries': 0,
            'intent_distribution': defaultdict(int),
            'confidence_distribution': defaultdict(int),
            'avg_confidence': 0.0
        }
    
    def _generate_training_data(self) -> List[Tuple[str, str]]:
        """Generate training data from intent examples"""
        training_data = []
        
        for intent, info in self.intent_categories.items():
            # Add examples
            for example in info['examples']:
                training_data.append((example, intent))
            
            # Generate variations
            keywords = info['keywords']
            for i in range(5):  # Generate 5 variations per intent
                # Simple template-based generation
                if intent == 'search_articles':
                    variations = [
                        f"find news about technology",
                        f"search articles containing elections",
                        f"show me stories about climate",
                        f"get articles on business",
                        f"look for news regarding sports"
                    ]
                elif intent == 'analyze_sentiment':
                    variations = [
                        f"what's the sentiment of political news",
                        f"analyze the mood in tech articles",
                        f"show positive coverage of healthcare",
                        f"how negative is economic reporting",
                        f"sentiment analysis of sports news"
                    ]
                elif intent == 'summarize_text':
                    variations = [
                        f"give me a summary of this article",
                        f"brief overview of tech news",
                        f"summarize recent political developments",
                        f"key points from business coverage",
                        f"main highlights from sports"
                    ]
                # Add more variations for other intents...
                else:
                    # Generic variations using keywords
                    if len(keywords) >= 2:
                        variation = f"{keywords[0]} {keywords[1]} about news"
                        variations = [variation]
                    else:
                        variations = info['examples'][:1]  # Fallback to first example
                
                if i < len(variations):
                    training_data.append((variations[i], intent))
        
        return training_data
    
    def _train_sklearn_model(self):
        """Train sklearn model for intent classification"""
        if not self.training_data:
            logging.warning("No training data available for intent classification")
            return
        
        # Prepare data
        texts = [item[0] for item in self.training_data]
        intents = [item[1] for item in self.training_data]
        
        # Vectorize text
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        X = self.tfidf_vectorizer.fit_transform(texts)
        
        # Train classifier
        self.sklearn_model = LogisticRegression(random_state=42, max_iter=1000)
        self.sklearn_model.fit(X, intents)
        
        # Evaluate on training data (for monitoring)
        y_pred = self.sklearn_model.predict(X)
        accuracy = accuracy_score(intents, y_pred)
        
        logging.info(f"Intent classifier trained with accuracy: {accuracy:.3f}")
    
    def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify user intent from natural language query
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary with intent classification results
        """
        if not query or len(query.strip()) < 3:
            return {'error': 'Query too short for intent classification'}
        
        query_lower = query.lower()
        
        result = {
            'query': query,
            'timestamp': datetime.now().isoformat()
        }
        
        # Rule-based classification (high precision)
        rule_based_result = self._classify_with_rules(query_lower)
        result['rule_based'] = rule_based_result
        
        # ML-based classification (broader coverage)
        if self.sklearn_model and self.tfidf_vectorizer:
            ml_result = self._classify_with_ml(query)
            result['ml_based'] = ml_result
        
        # Aggregate results
        aggregated = self._aggregate_intent_results(result)
        result['final_intent'] = aggregated
        
        # Extract entities and parameters
        entities = self._extract_query_entities(query)
        result['entities'] = entities
        
        # Generate query plan
        query_plan = self._generate_query_plan(aggregated, entities)
        result['query_plan'] = query_plan
        
        # Update statistics
        self._update_classification_stats(result)
        
        return result
    
    def _classify_with_rules(self, query: str) -> Dict[str, Any]:
        """Classify intent using rule-based approach"""
        intent_scores = {}
        
        for intent, info in self.intent_categories.items():
            score = 0
            matched_keywords = []
            
            # Check for keyword matches
            for keyword in info['keywords']:
                if keyword in query:
                    score += 1
                    matched_keywords.append(keyword)
            
            # Boost score for exact phrase matches
            for example in info['examples']:
                if example.lower() in query:
                    score += 3
            
            # Normalize score
            normalized_score = score / len(info['keywords']) if info['keywords'] else 0
            
            if normalized_score > 0:
                intent_scores[intent] = {
                    'score': normalized_score,
                    'matched_keywords': matched_keywords,
                    'confidence': min(normalized_score, 1.0)
                }
        
        # Find best intent
        if intent_scores:
            best_intent = max(intent_scores, key=lambda x: intent_scores[x]['score'])
            confidence = intent_scores[best_intent]['confidence']
            
            return {
                'intent': best_intent,
                'confidence': confidence,
                'all_scores': intent_scores,
                'method': 'rule_based'
            }
        else:
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'all_scores': {},
                'method': 'rule_based'
            }
    
    def _classify_with_ml(self, query: str) -> Dict[str, Any]:
        """Classify intent using machine learning model"""
        try:
            # Vectorize query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Predict intent
            predicted_intent = self.sklearn_model.predict(query_vector)[0]
            
            # Get probability scores
            probabilities = self.sklearn_model.predict_proba(query_vector)[0]
            class_names = self.sklearn_model.classes_
            
            # Create scores dictionary
            all_scores = {}
            for i, class_name in enumerate(class_names):
                all_scores[class_name] = {
                    'score': probabilities[i],
                    'confidence': probabilities[i]
                }
            
            return {
                'intent': predicted_intent,
                'confidence': max(probabilities),
                'all_scores': all_scores,
                'method': 'ml_based'
            }
            
        except Exception as e:
            logging.error(f"ML intent classification failed: {e}")
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'error': str(e),
                'method': 'ml_based'
            }
    
    def _aggregate_intent_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate intent classification results from multiple methods"""
        
        # Weights for different methods
        method_weights = {
            'rule_based': 0.7,  # Higher weight for rules (more precise)
            'ml_based': 0.3     # Lower weight for ML (broader but less precise)
        }
        
        intent_scores = defaultdict(float)
        
        # Aggregate scores
        for method, weight in method_weights.items():
            if method in results and 'all_scores' in results[method]:
                for intent, score_info in results[method]['all_scores'].items():
                    score = score_info.get('score', 0)
                    intent_scores[intent] += score * weight
        
        # If no ML results, use rule-based only
        if 'ml_based' not in results and 'rule_based' in results:
            rule_result = results['rule_based']
            if rule_result['intent'] != 'unknown':
                return {
                    'intent': rule_result['intent'],
                    'confidence': rule_result['confidence'],
                    'method': 'rule_based_only',
                    'all_scores': dict(intent_scores) if intent_scores else {}
                }
        
        # Find best aggregated intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[best_intent]
            
            return {
                'intent': best_intent,
                'confidence': min(confidence, 1.0),
                'method': 'aggregated',
                'all_scores': dict(intent_scores)
            }
        else:
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'method': 'aggregated',
                'all_scores': {}
            }
    
    def _extract_query_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities and parameters from query"""
        entities = {
            'time_expressions': [],
            'categories': [],
            'sentiment_modifiers': [],
            'languages': [],
            'keywords': [],
            'numbers': []
        }
        
        query_lower = query.lower()
        
        # Extract using regex patterns
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    entities[entity_type].extend(matches)
        
        # Extract keywords (simple approach)
        words = nltk.word_tokenize(query_lower)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        keywords = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
        entities['keywords'] = keywords[:10]  # Limit to top 10
        
        # Extract numbers
        numbers = re.findall(r'\b\d+\b', query)
        entities['numbers'] = [int(num) for num in numbers]
        
        # Remove duplicates
        for key in entities:
            if isinstance(entities[key], list):
                entities[key] = list(set(entities[key]))
        
        return entities
    
    def _generate_query_plan(self, intent_result: Dict[str, Any], entities: Dict[str, Any]) -> Dict[str, Any]:
        """Generate execution plan for the query"""
        
        intent = intent_result.get('intent', 'unknown')
        
        query_plan = {
            'intent': intent,
            'steps': [],
            'parameters': {},
            'estimated_complexity': 'low'
        }
        
        # Generate steps based on intent
        if intent == 'search_articles':
            query_plan['steps'] = [
                'parse_search_criteria',
                'filter_articles',
                'rank_results',
                'format_response'
            ]
            query_plan['parameters'] = {
                'keywords': entities.get('keywords', []),
                'categories': entities.get('categories', []),
                'time_range': entities.get('time_expressions', []),
                'max_results': 10
            }
            
        elif intent == 'analyze_sentiment':
            query_plan['steps'] = [
                'filter_articles',
                'analyze_sentiment',
                'aggregate_results',
                'format_sentiment_response'
            ]
            query_plan['parameters'] = {
                'sentiment_filter': entities.get('sentiment_modifiers', []),
                'categories': entities.get('categories', []),
                'time_range': entities.get('time_expressions', [])
            }
            query_plan['estimated_complexity'] = 'medium'
            
        elif intent == 'classify_content':
            query_plan['steps'] = [
                'load_classification_model',
                'classify_articles',
                'format_classification_response'
            ]
            query_plan['estimated_complexity'] = 'medium'
            
        elif intent == 'summarize_text':
            query_plan['steps'] = [
                'identify_content',
                'generate_summary',
                'format_summary_response'
            ]
            query_plan['estimated_complexity'] = 'high'
            
        elif intent == 'extract_entities':
            query_plan['steps'] = [
                'load_ner_model',
                'extract_entities',
                'format_entity_response'
            ]
            query_plan['estimated_complexity'] = 'medium'
            
        elif intent == 'get_insights':
            query_plan['steps'] = [
                'analyze_patterns',
                'generate_insights',
                'format_insights_response'
            ]
            query_plan['estimated_complexity'] = 'high'
            
        elif intent == 'compare_sources':
            query_plan['steps'] = [
                'identify_comparison_criteria',
                'perform_cross_lingual_analysis',
                'format_comparison_response'
            ]
            query_plan['parameters'] = {
                'languages': entities.get('languages', []),
                'comparison_type': 'coverage'
            }
            query_plan['estimated_complexity'] = 'high'
            
        elif intent == 'trend_analysis':
            query_plan['steps'] = [
                'load_temporal_data',
                'analyze_trends',
                'generate_visualizations',
                'format_trend_response'
            ]
            query_plan['parameters'] = {
                'time_range': entities.get('time_expressions', []),
                'categories': entities.get('categories', [])
            }
            query_plan['estimated_complexity'] = 'high'
            
        elif intent == 'export_results':
            query_plan['steps'] = [
                'prepare_data',
                'format_export',
                'generate_download_link'
            ]
            query_plan['parameters'] = {
                'format': 'csv',  # Default format
                'include_metadata': True
            }
            
        elif intent == 'get_help':
            query_plan['steps'] = [
                'generate_help_response'
            ]
        
        return query_plan
    
    def get_intent_suggestions(self, partial_query: str) -> List[Dict[str, Any]]:
        """
        Get intent suggestions for partial queries (for autocomplete)
        
        Args:
            partial_query: Partial user query
            
        Returns:
            List of suggested completions with intents
        """
        suggestions = []
        
        if len(partial_query) < 2:
            # Return popular intents for very short queries
            popular_intents = ['search_articles', 'analyze_sentiment', 'get_insights']
            for intent in popular_intents:
                info = self.intent_categories[intent]
                suggestions.append({
                    'intent': intent,
                    'description': info['description'],
                    'example': info['examples'][0],
                    'completion_score': 1.0
                })
            return suggestions
        
        partial_lower = partial_query.lower()
        
        # Find matching examples
        for intent, info in self.intent_categories.items():
            for example in info['examples']:
                if example.lower().startswith(partial_lower):
                    suggestions.append({
                        'intent': intent,
                        'description': info['description'],
                        'suggested_completion': example,
                        'completion_score': len(partial_query) / len(example)
                    })
        
        # Sort by completion score
        suggestions.sort(key=lambda x: x['completion_score'], reverse=True)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def validate_intent_coverage(self, test_queries: List[str]) -> Dict[str, Any]:
        """
        Validate intent classification coverage on test queries
        
        Args:
            test_queries: List of test queries
            
        Returns:
            Validation results
        """
        validation_results = {
            'total_queries': len(test_queries),
            'classified_queries': 0,
            'unknown_queries': 0,
            'intent_distribution': defaultdict(int),
            'avg_confidence': 0.0,
            'low_confidence_queries': []
        }
        
        confidences = []
        
        for query in test_queries:
            result = self.classify_intent(query)
            final_intent = result.get('final_intent', {})
            
            intent = final_intent.get('intent', 'unknown')
            confidence = final_intent.get('confidence', 0.0)
            
            validation_results['intent_distribution'][intent] += 1
            
            if intent != 'unknown':
                validation_results['classified_queries'] += 1
            else:
                validation_results['unknown_queries'] += 1
            
            confidences.append(confidence)
            
            if confidence < 0.5:
                validation_results['low_confidence_queries'].append({
                    'query': query,
                    'intent': intent,
                    'confidence': confidence
                })
        
        if confidences:
            validation_results['avg_confidence'] = np.mean(confidences)
            validation_results['confidence_std'] = np.std(confidences)
        
        validation_results['classification_rate'] = (
            validation_results['classified_queries'] / validation_results['total_queries']
            if validation_results['total_queries'] > 0 else 0
        )
        
        return validation_results
    
    def _update_classification_stats(self, result: Dict[str, Any]):
        """Update classification statistics"""
        self.classification_stats['total_queries'] += 1
        
        final_intent = result.get('final_intent', {})
        intent = final_intent.get('intent', 'unknown')
        confidence = final_intent.get('confidence', 0.0)
        
        self.classification_stats['intent_distribution'][intent] += 1
        
        # Update confidence average
        total = self.classification_stats['total_queries']
        current_avg = self.classification_stats['avg_confidence']
        self.classification_stats['avg_confidence'] = (
            (current_avg * (total - 1) + confidence) / total
        )
        
        # Confidence distribution
        confidence_bin = int(confidence * 10) / 10  # Round to nearest 0.1
        self.classification_stats['confidence_distribution'][confidence_bin] += 1
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get intent classification statistics"""
        stats = {
            'total_queries': self.classification_stats['total_queries'],
            'avg_confidence': self.classification_stats['avg_confidence'],
            'intent_distribution': dict(self.classification_stats['intent_distribution']),
            'confidence_distribution': dict(self.classification_stats['confidence_distribution'])
        }
        
        # Add percentages
        total = stats['total_queries']
        if total > 0:
            stats['intent_percentages'] = {
                intent: (count / total) * 100 
                for intent, count in stats['intent_distribution'].items()
            }
        
        return stats
    
    def get_supported_intents(self) -> Dict[str, Dict[str, Any]]:
        """Get information about supported intents"""
        return self.intent_categories.copy()
    
    def save_classifier(self, filepath: str):
        """Save intent classifier"""
        save_data = {
            'config': self.config,
            'intent_categories': self.intent_categories,
            'entity_patterns': self.entity_patterns,
            'sklearn_model': self.sklearn_model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'training_data': self.training_data,
            'classification_stats': dict(self.classification_stats),
            'save_timestamp': datetime.now().isoformat()
        }
        
        # Convert defaultdicts to regular dicts
        for key in ['intent_distribution', 'confidence_distribution']:
            if key in save_data['classification_stats']:
                save_data['classification_stats'][key] = dict(self.classification_stats[key])
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logging.info(f"Intent classifier saved to {filepath}")
    
    def load_classifier(self, filepath: str):
        """Load intent classifier"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.config = save_data['config']
        self.intent_categories = save_data['intent_categories']
        self.entity_patterns = save_data['entity_patterns']
        self.sklearn_model = save_data['sklearn_model']
        self.tfidf_vectorizer = save_data['tfidf_vectorizer']
        self.training_data = save_data['training_data']
        
        # Restore statistics
        stats = save_data['classification_stats']
        self.classification_stats['total_queries'] = stats.get('total_queries', 0)
        self.classification_stats['avg_confidence'] = stats.get('avg_confidence', 0.0)
        
        # Restore defaultdicts
        self.classification_stats['intent_distribution'] = defaultdict(
            int, stats.get('intent_distribution', {})
        )
        self.classification_stats['confidence_distribution'] = defaultdict(
            int, stats.get('confidence_distribution', {})
        )
        
        logging.info(f"Intent classifier loaded from {filepath}")