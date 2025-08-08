#!/usr/bin/env python3
"""
Advanced Bias Detection for NewsBot 2.0
Cutting-edge NLP techniques for detecting bias in news articles
Bonus Feature: Advanced Research Extensions (20 points)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("Transformers not available for advanced bias detection")

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    logging.warning("spaCy not available for linguistic analysis")

class AdvancedBiasDetector:
    """
    Advanced bias detection using multiple NLP techniques and transformer models
    Implements cutting-edge research in media bias detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize advanced bias detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize models
        self.sentiment_analyzer = None
        self.bias_classifier = None
        self.sentence_embedder = None
        self.nlp = None
        
        # Bias detection thresholds
        self.bias_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
        
        # Political bias keywords (simplified for demonstration)
        self.political_keywords = {
            'left_leaning': [
                'progressive', 'liberal', 'socialist', 'democratic', 'inclusive',
                'diversity', 'equality', 'climate change', 'social justice'
            ],
            'right_leaning': [
                'conservative', 'traditional', 'patriotic', 'security', 'business',
                'free market', 'constitutional', 'law and order', 'fiscal responsibility'
            ],
            'neutral': [
                'reported', 'stated', 'according to', 'sources say', 'officials confirm'
            ]
        }
        
        # Emotional bias words
        self.emotional_words = {
            'positive': [
                'excellent', 'outstanding', 'remarkable', 'extraordinary', 'brilliant',
                'successful', 'triumphant', 'victorious', 'breakthrough', 'innovative'
            ],
            'negative': [
                'terrible', 'awful', 'disastrous', 'catastrophic', 'horrible',
                'failed', 'defeated', 'crisis', 'scandal', 'controversy'
            ]
        }
        
        # Initialize components
        self._initialize_models()
        
        # Analysis statistics
        self.analysis_stats = {
            'total_analyzed': 0,
            'bias_detected': 0,
            'average_bias_score': 0.0,
            'bias_types_found': defaultdict(int)
        }
    
    def _initialize_models(self):
        """Initialize NLP models for bias detection"""
        
        if HAS_TRANSFORMERS:
            try:
                # Initialize sentiment analyzer
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                
                # Initialize sentence embedder for semantic analysis
                self.sentence_embedder = SentenceTransformer('all-MiniLM-L6-v2')
                
                logging.info("Transformer models initialized for bias detection")
                
            except Exception as e:
                logging.warning(f"Failed to initialize transformer models: {e}")
        
        if HAS_SPACY:
            try:
                # Load spaCy model for linguistic analysis
                self.nlp = spacy.load('en_core_web_sm')
                logging.info("spaCy model loaded for linguistic analysis")
            except OSError:
                logging.warning("spaCy English model not found")
    
    def detect_bias(self, article_text: str, article_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive bias detection analysis
        
        Args:
            article_text: News article text
            article_metadata: Optional metadata about the article
            
        Returns:
            Comprehensive bias analysis results
        """
        results = {
            'text_length': len(article_text),
            'word_count': len(article_text.split()),
            'bias_detected': False,
            'overall_bias_score': 0.0,
            'bias_confidence': 0.0,
            'bias_types': [],
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        try:
            # 1. Lexical bias analysis
            lexical_bias = self._analyze_lexical_bias(article_text)
            results['lexical_bias'] = lexical_bias
            
            # 2. Political bias detection
            political_bias = self._detect_political_bias(article_text)
            results['political_bias'] = political_bias
            
            # 3. Emotional bias analysis
            emotional_bias = self._analyze_emotional_bias(article_text)
            results['emotional_bias'] = emotional_bias
            
            # 4. Source attribution analysis
            source_bias = self._analyze_source_attribution(article_text)
            results['source_bias'] = source_bias
            
            # 5. Linguistic complexity analysis
            linguistic_bias = self._analyze_linguistic_complexity(article_text)
            results['linguistic_bias'] = linguistic_bias
            
            # 6. Semantic bias using embeddings
            if self.sentence_embedder:
                semantic_bias = self._analyze_semantic_bias(article_text)
                results['semantic_bias'] = semantic_bias
            
            # 7. Calculate overall bias score
            overall_score = self._calculate_overall_bias_score(results)
            results['overall_bias_score'] = overall_score
            results['bias_confidence'] = min(overall_score * 1.2, 1.0)  # Confidence based on score
            
            # 8. Determine bias level and types
            if overall_score > self.bias_thresholds['low']:
                results['bias_detected'] = True
                results['bias_level'] = self._get_bias_level(overall_score)
                results['bias_types'] = self._identify_bias_types(results)
            
            # 9. Generate bias explanation
            results['explanation'] = self._generate_bias_explanation(results)
            
            # Update statistics
            self._update_statistics(results)
            
            return results
            
        except Exception as e:
            logging.error(f"Bias detection failed: {e}")
            results['error'] = str(e)
            return results
    
    def _analyze_lexical_bias(self, text: str) -> Dict[str, Any]:
        """Analyze lexical bias through word choice and frequency"""
        
        words = text.lower().split()
        total_words = len(words)
        
        # Count biased words
        positive_count = sum(1 for word in words if word in self.emotional_words['positive'])
        negative_count = sum(1 for word in words if word in self.emotional_words['negative'])
        
        # Calculate bias ratios
        emotional_ratio = (positive_count + negative_count) / total_words if total_words > 0 else 0
        sentiment_skew = (positive_count - negative_count) / total_words if total_words > 0 else 0
        
        return {
            'emotional_word_ratio': emotional_ratio,
            'sentiment_skew': sentiment_skew,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'bias_score': abs(sentiment_skew) + emotional_ratio * 0.5
        }
    
    def _detect_political_bias(self, text: str) -> Dict[str, Any]:
        """Detect political bias through keyword analysis"""
        
        text_lower = text.lower()
        
        # Count political keywords
        left_count = sum(1 for keyword in self.political_keywords['left_leaning'] 
                        if keyword in text_lower)
        right_count = sum(1 for keyword in self.political_keywords['right_leaning'] 
                         if keyword in text_lower)
        neutral_count = sum(1 for keyword in self.political_keywords['neutral'] 
                           if keyword in text_lower)
        
        total_political = left_count + right_count + neutral_count
        
        # Calculate political leaning
        if total_political > 0:
            left_ratio = left_count / total_political
            right_ratio = right_count / total_political
            neutral_ratio = neutral_count / total_political
        else:
            left_ratio = right_ratio = neutral_ratio = 0
        
        # Determine political bias
        if left_ratio > right_ratio + 0.2:
            political_leaning = 'left'
            bias_strength = left_ratio - right_ratio
        elif right_ratio > left_ratio + 0.2:
            political_leaning = 'right'
            bias_strength = right_ratio - left_ratio
        else:
            political_leaning = 'neutral'
            bias_strength = neutral_ratio
        
        return {
            'political_leaning': political_leaning,
            'bias_strength': bias_strength,
            'left_keywords': left_count,
            'right_keywords': right_count,
            'neutral_keywords': neutral_count,
            'bias_score': abs(left_ratio - right_ratio) if total_political > 0 else 0
        }
    
    def _analyze_emotional_bias(self, text: str) -> Dict[str, Any]:
        """Analyze emotional bias using sentiment analysis"""
        
        emotional_bias = {
            'sentiment_scores': [],
            'emotional_variance': 0.0,
            'dominant_emotion': 'neutral',
            'bias_score': 0.0
        }
        
        if self.sentiment_analyzer:
            try:
                # Analyze sentiment of sentences
                sentences = text.split('.')
                sentiment_scores = []
                
                for sentence in sentences[:10]:  # Limit to first 10 sentences
                    if len(sentence.strip()) > 10:
                        results = self.sentiment_analyzer(sentence.strip())
                        if results and len(results[0]) > 0:
                            # Get the score for the top prediction
                            top_result = max(results[0], key=lambda x: x['score'])
                            sentiment_scores.append({
                                'label': top_result['label'],
                                'score': top_result['score']
                            })
                
                emotional_bias['sentiment_scores'] = sentiment_scores
                
                # Calculate emotional variance
                if sentiment_scores:
                    scores = [s['score'] for s in sentiment_scores]
                    emotional_bias['emotional_variance'] = np.var(scores)
                    
                    # Determine dominant emotion
                    emotion_counts = Counter([s['label'] for s in sentiment_scores])
                    emotional_bias['dominant_emotion'] = emotion_counts.most_common(1)[0][0]
                    
                    # Calculate bias score based on emotional inconsistency
                    emotional_bias['bias_score'] = min(emotional_bias['emotional_variance'] * 2, 1.0)
                    
            except Exception as e:
                logging.warning(f"Sentiment analysis failed: {e}")
        
        return emotional_bias
    
    def _analyze_source_attribution(self, text: str) -> Dict[str, Any]:
        """Analyze source attribution and credibility indicators"""
        
        # Patterns for source attribution
        source_patterns = [
            r'according to',
            r'sources? (say|said|tell|told)',
            r'officials? (say|said|confirm|confirmed)',
            r'experts? (say|said|believe|think)',
            r'(he|she|they) (said|stated|reported|claimed)',
            r'quoted|citing|referenced'
        ]
        
        # Count source attributions
        source_count = 0
        for pattern in source_patterns:
            source_count += len(re.findall(pattern, text.lower()))
        
        # Count opinion words vs. factual words
        opinion_words = ['think', 'believe', 'feel', 'opinion', 'view', 'perspective']
        fact_words = ['data', 'evidence', 'study', 'research', 'statistics', 'findings']
        
        opinion_count = sum(1 for word in opinion_words if word in text.lower())
        fact_count = sum(1 for word in fact_words if word in text.lower())
        
        word_count = len(text.split())
        
        return {
            'source_attribution_count': source_count,
            'source_density': source_count / word_count if word_count > 0 else 0,
            'opinion_words': opinion_count,
            'fact_words': fact_count,
            'fact_opinion_ratio': fact_count / max(opinion_count, 1),
            'bias_score': max(0, (opinion_count - fact_count) / word_count) if word_count > 0 else 0
        }
    
    def _analyze_linguistic_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic complexity and readability bias"""
        
        sentences = text.split('.')
        words = text.split()
        
        # Calculate basic metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Count complex words (> 6 characters)
        complex_words = [word for word in words if len(word) > 6]
        complex_word_ratio = len(complex_words) / len(words) if words else 0
        
        # Simple readability approximation (Flesch-like)
        readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * complex_word_ratio)
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'complex_word_ratio': complex_word_ratio,
            'readability_score': readability_score,
            'bias_score': min(complex_word_ratio, 0.5)  # Higher complexity can indicate bias
        }
    
    def _analyze_semantic_bias(self, text: str) -> Dict[str, Any]:
        """Analyze semantic bias using sentence embeddings"""
        
        if not self.sentence_embedder:
            return {'bias_score': 0.0, 'error': 'Sentence embedder not available'}
        
        try:
            # Split into sentences
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
            
            if len(sentences) < 2:
                return {'bias_score': 0.0, 'note': 'Insufficient sentences for analysis'}
            
            # Get embeddings for sentences
            embeddings = self.sentence_embedder.encode(sentences)
            
            # Calculate semantic consistency (lower variance = more consistent/potentially biased)
            semantic_variance = np.var(embeddings, axis=0).mean()
            
            # Calculate pairwise similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(embeddings)
            
            # Get average similarity (excluding diagonal)
            mask = np.ones_like(similarities, dtype=bool)
            np.fill_diagonal(mask, False)
            avg_similarity = similarities[mask].mean()
            
            # High similarity might indicate repetitive or biased content
            bias_score = min(avg_similarity, 1.0)
            
            return {
                'semantic_variance': semantic_variance,
                'average_similarity': avg_similarity,
                'bias_score': bias_score,
                'sentence_count': len(sentences)
            }
            
        except Exception as e:
            logging.warning(f"Semantic bias analysis failed: {e}")
            return {'bias_score': 0.0, 'error': str(e)}
    
    def _calculate_overall_bias_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall bias score from individual analyses"""
        
        bias_scores = []
        weights = []
        
        # Collect bias scores with weights
        if 'lexical_bias' in results:
            bias_scores.append(results['lexical_bias'].get('bias_score', 0))
            weights.append(0.25)
        
        if 'political_bias' in results:
            bias_scores.append(results['political_bias'].get('bias_score', 0))
            weights.append(0.3)
        
        if 'emotional_bias' in results:
            bias_scores.append(results['emotional_bias'].get('bias_score', 0))
            weights.append(0.2)
        
        if 'source_bias' in results:
            bias_scores.append(results['source_bias'].get('bias_score', 0))
            weights.append(0.15)
        
        if 'linguistic_bias' in results:
            bias_scores.append(results['linguistic_bias'].get('bias_score', 0))
            weights.append(0.1)
        
        # Calculate weighted average
        if bias_scores and weights:
            overall_score = np.average(bias_scores, weights=weights)
            return min(overall_score, 1.0)
        
        return 0.0
    
    def _get_bias_level(self, bias_score: float) -> str:
        """Determine bias level from score"""
        if bias_score >= self.bias_thresholds['high']:
            return 'high'
        elif bias_score >= self.bias_thresholds['medium']:
            return 'medium'
        elif bias_score >= self.bias_thresholds['low']:
            return 'low'
        else:
            return 'minimal'
    
    def _identify_bias_types(self, results: Dict[str, Any]) -> List[str]:
        """Identify specific types of bias detected"""
        bias_types = []
        
        # Check political bias
        if 'political_bias' in results:
            political = results['political_bias']
            if political.get('bias_score', 0) > 0.3:
                bias_types.append(f"political_{political.get('political_leaning', 'unknown')}")
        
        # Check emotional bias
        if 'emotional_bias' in results:
            emotional = results['emotional_bias']
            if emotional.get('bias_score', 0) > 0.3:
                bias_types.append(f"emotional_{emotional.get('dominant_emotion', 'unknown')}")
        
        # Check source bias
        if 'source_bias' in results:
            source = results['source_bias']
            if source.get('bias_score', 0) > 0.3:
                bias_types.append('poor_sourcing')
        
        # Check lexical bias
        if 'lexical_bias' in results:
            lexical = results['lexical_bias']
            if lexical.get('bias_score', 0) > 0.3:
                if lexical.get('sentiment_skew', 0) > 0.1:
                    bias_types.append('positive_lexical')
                elif lexical.get('sentiment_skew', 0) < -0.1:
                    bias_types.append('negative_lexical')
        
        return bias_types
    
    def _generate_bias_explanation(self, results: Dict[str, Any]) -> str:
        """Generate human-readable explanation of bias analysis"""
        
        bias_score = results.get('overall_bias_score', 0)
        bias_level = self._get_bias_level(bias_score)
        bias_types = results.get('bias_types', [])
        
        if bias_score < self.bias_thresholds['low']:
            return "The article appears to maintain objectivity with minimal detected bias."
        
        explanation = f"Detected {bias_level} level bias (score: {bias_score:.2f}). "
        
        if bias_types:
            explanation += f"Identified bias types: {', '.join(bias_types)}. "
        
        # Add specific insights
        insights = []
        
        if 'political_bias' in results:
            political = results['political_bias']
            if political.get('bias_score', 0) > 0.3:
                leaning = political.get('political_leaning', 'unknown')
                insights.append(f"Shows {leaning}-leaning political language")
        
        if 'emotional_bias' in results:
            emotional = results['emotional_bias']
            if emotional.get('bias_score', 0) > 0.3:
                insights.append("Contains emotionally charged language")
        
        if 'source_bias' in results:
            source = results['source_bias']
            if source.get('source_density', 0) < 0.02:
                insights.append("Limited source attribution")
        
        if insights:
            explanation += "Key issues: " + "; ".join(insights) + "."
        
        return explanation
    
    def _update_statistics(self, results: Dict[str, Any]):
        """Update analysis statistics"""
        self.analysis_stats['total_analyzed'] += 1
        
        if results.get('bias_detected', False):
            self.analysis_stats['bias_detected'] += 1
            
            for bias_type in results.get('bias_types', []):
                self.analysis_stats['bias_types_found'][bias_type] += 1
        
        # Update average bias score
        current_avg = self.analysis_stats['average_bias_score']
        total = self.analysis_stats['total_analyzed']
        new_score = results.get('overall_bias_score', 0)
        
        self.analysis_stats['average_bias_score'] = (
            (current_avg * (total - 1) + new_score) / total
        )
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analysis statistics"""
        stats = self.analysis_stats.copy()
        
        if stats['total_analyzed'] > 0:
            stats['bias_detection_rate'] = stats['bias_detected'] / stats['total_analyzed']
        else:
            stats['bias_detection_rate'] = 0.0
        
        return stats
    
    def analyze_dataset_bias(self, articles: List[str], 
                           metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze bias patterns across a dataset of articles"""
        
        if not articles:
            return {'error': 'No articles provided for analysis'}
        
        results = {
            'total_articles': len(articles),
            'analyzed_articles': 0,
            'bias_distribution': defaultdict(int),
            'average_bias_score': 0.0,
            'bias_types_summary': defaultdict(int),
            'highly_biased_articles': []
        }
        
        bias_scores = []
        
        for i, article in enumerate(articles):
            try:
                article_metadata = metadata[i] if metadata and i < len(metadata) else {}
                bias_result = self.detect_bias(article, article_metadata)
                
                if 'error' not in bias_result:
                    results['analyzed_articles'] += 1
                    
                    bias_score = bias_result.get('overall_bias_score', 0)
                    bias_scores.append(bias_score)
                    
                    bias_level = self._get_bias_level(bias_score)
                    results['bias_distribution'][bias_level] += 1
                    
                    # Collect bias types
                    for bias_type in bias_result.get('bias_types', []):
                        results['bias_types_summary'][bias_type] += 1
                    
                    # Track highly biased articles
                    if bias_score > self.bias_thresholds['high']:
                        results['highly_biased_articles'].append({
                            'index': i,
                            'bias_score': bias_score,
                            'bias_types': bias_result.get('bias_types', []),
                            'preview': article[:200] + "..."
                        })
                
            except Exception as e:
                logging.warning(f"Failed to analyze article {i}: {e}")
                continue
        
        # Calculate summary statistics
        if bias_scores:
            results['average_bias_score'] = np.mean(bias_scores)
            results['bias_score_std'] = np.std(bias_scores)
            results['bias_score_median'] = np.median(bias_scores)
        
        return results