#!/usr/bin/env python3
"""
Enhanced Sentiment Analyzer for NewsBot 2.0
Advanced sentiment analysis with temporal tracking and emotion detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import pickle
from collections import defaultdict

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import spacy

# Try to import transformers for advanced sentiment analysis
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("transformers not available. Install for advanced sentiment features.")

class SentimentAnalyzer:
    """
    Advanced sentiment analysis with temporal tracking and emotion detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sentiment analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize VADER sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logging.warning("spaCy model not found. Some features will not be available.")
            self.nlp = None
        
        # Initialize transformer-based sentiment models if available
        self.transformer_sentiment = None
        self.transformer_emotion = None
        
        if HAS_TRANSFORMERS:
            try:
                # Load sentiment analysis model
                sentiment_model = self.config.get('sentiment_model', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
                self.transformer_sentiment = pipeline(
                    "sentiment-analysis", 
                    model=sentiment_model,
                    return_all_scores=True
                )
                
                # Load emotion detection model
                emotion_model = self.config.get('emotion_model', 'j-hartmann/emotion-english-distilroberta-base')
                self.transformer_emotion = pipeline(
                    "text-classification",
                    model=emotion_model,
                    return_all_scores=True
                )
                
                logging.info("Transformer-based sentiment models loaded successfully")
                
            except Exception as e:
                logging.warning(f"Could not load transformer models: {e}")
                self.transformer_sentiment = None
                self.transformer_emotion = None
        
        # Sentiment mapping
        self.sentiment_mapping = {
            'negative': -1,
            'neutral': 0,
            'positive': 1
        }
        
        # Emotion categories
        self.emotion_categories = [
            'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation'
        ]
        
        # Temporal tracking storage
        self.sentiment_history = defaultdict(list)
        self.trend_cache = {}
    
    def analyze_sentiment(self, text: str, methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis using multiple methods
        
        Args:
            text: Input text
            methods: List of methods to use ['vader', 'textblob', 'transformer']
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if methods is None:
            methods = ['vader', 'textblob']
            if self.transformer_sentiment:
                methods.append('transformer')
        
        results = {
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'methods_used': methods
        }
        
        # VADER Sentiment Analysis
        if 'vader' in methods:
            vader_scores = self.vader_analyzer.polarity_scores(text)
            results['vader'] = {
                'compound': vader_scores['compound'],
                'positive': vader_scores['pos'],
                'neutral': vader_scores['neu'],
                'negative': vader_scores['neg'],
                'classification': self._classify_vader_sentiment(vader_scores['compound'])
            }
        
        # TextBlob Sentiment Analysis
        if 'textblob' in methods:
            blob = TextBlob(text)
            results['textblob'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'classification': self._classify_textblob_sentiment(blob.sentiment.polarity)
            }
        
        # Transformer-based Sentiment Analysis
        if 'transformer' in methods and self.transformer_sentiment:
            try:
                # Truncate text if too long for the model (max 512 tokens)
                max_length = 500  # Leave some buffer for special tokens
                if len(text.split()) > max_length:
                    text_truncated = ' '.join(text.split()[:max_length])
                else:
                    text_truncated = text
                
                # Additional safety: check if text is too long for the tokenizer
                try:
                    transformer_results = self.transformer_sentiment(text_truncated)
                except Exception as token_error:
                    # If still failing, try with even shorter text
                    if len(text_truncated.split()) > 300:
                        text_truncated = ' '.join(text_truncated.split()[:300])
                        transformer_results = self.transformer_sentiment(text_truncated)
                    else:
                        raise token_error
                
                # Process results
                sentiment_scores = {}
                for result in transformer_results[0]:
                    label = result['label'].lower()
                    score = result['score']
                    sentiment_scores[label] = score
                
                # Get dominant sentiment
                dominant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
                
                results['transformer'] = {
                    'scores': sentiment_scores,
                    'dominant_sentiment': dominant_sentiment,
                    'confidence': sentiment_scores[dominant_sentiment],
                    'classification': self._map_transformer_sentiment(dominant_sentiment),
                    'text_truncated': len(text.split()) > max_length
                }
                
            except Exception as e:
                logging.warning(f"Transformer sentiment analysis failed: {e}")
                results['transformer'] = {'error': str(e)}
        
        # Emotion Analysis
        if self.transformer_emotion:
            try:
                emotion_results = self.analyze_emotions(text)
                results['emotions'] = emotion_results
            except Exception as e:
                logging.warning(f"Emotion analysis failed: {e}")
                results['emotions'] = {'error': str(e)}
        
        # Aggregate sentiment
        results['aggregate'] = self._aggregate_sentiment_scores(results)
        
        # Confidence assessment
        results['confidence'] = self._assess_confidence(results)
        
        return results
    
    def analyze_emotions(self, text: str) -> Dict[str, Any]:
        """
        Analyze emotions in text using transformer model
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with emotion analysis results
        """
        if not self.transformer_emotion:
            return {'error': 'Emotion model not available'}
        
        try:
            # Truncate text if too long for the model (max 512 tokens)
            max_length = 500  # Leave some buffer for special tokens
            if len(text.split()) > max_length:
                text_truncated = ' '.join(text.split()[:max_length])
            else:
                text_truncated = text
                
            # Additional safety: check if text is too long for the tokenizer
            try:
                emotion_results = self.transformer_emotion(text_truncated)
            except Exception as token_error:
                # If still failing, try with even shorter text
                if len(text_truncated.split()) > 300:
                    text_truncated = ' '.join(text_truncated.split()[:300])
                    emotion_results = self.transformer_emotion(text_truncated)
                else:
                    raise token_error
            
            # Process results
            emotion_scores = {}
            for result in emotion_results[0]:
                emotion = result['label'].lower()
                score = result['score']
                emotion_scores[emotion] = score
            
            # Get dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            
            # Categorize emotions
            positive_emotions = ['joy', 'trust', 'anticipation', 'surprise']
            negative_emotions = ['sadness', 'anger', 'fear', 'disgust']
            
            positive_score = sum(emotion_scores.get(emotion, 0) for emotion in positive_emotions)
            negative_score = sum(emotion_scores.get(emotion, 0) for emotion in negative_emotions)
            
            return {
                'emotion_scores': emotion_scores,
                'dominant_emotion': dominant_emotion,
                'confidence': emotion_scores[dominant_emotion],
                'positive_emotions_score': positive_score,
                'negative_emotions_score': negative_score,
                'emotional_intensity': max(emotion_scores.values()),
                'text_truncated': len(text.split()) > max_length
            }
            
        except Exception as e:
            logging.error(f"Emotion analysis error: {e}")
            return {'error': str(e)}
    
    def analyze_batch_sentiment(self, texts: List[str], include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for multiple texts efficiently
        
        Args:
            texts: List of texts to analyze
            include_metadata: Whether to include detailed metadata
            
        Returns:
            List of sentiment analysis results
        """
        logging.info(f"Analyzing sentiment for {len(texts)} texts...")
        
        results = []
        
        for i, text in enumerate(texts):
            if i % 100 == 0 and i > 0:
                logging.info(f"Processed {i}/{len(texts)} texts")
            
            try:
                result = self.analyze_sentiment(text)
                
                if not include_metadata:
                    # Keep only essential information
                    simplified_result = {
                        'aggregate': result.get('aggregate', {}),
                        'vader': result.get('vader', {}).get('compound', 0),
                        'textblob': result.get('textblob', {}).get('polarity', 0),
                        'classification': result.get('aggregate', {}).get('classification', 'neutral')
                    }
                    results.append(simplified_result)
                else:
                    results.append(result)
                    
            except Exception as e:
                logging.warning(f"Error analyzing text {i}: {e}")
                results.append({'error': str(e)})
        
        return results
    
    def track_sentiment_over_time(self, articles: pd.DataFrame, 
                                 text_column: str = 'text',
                                 date_column: str = 'date',
                                 category_column: str = 'category') -> Dict[str, Any]:
        """
        Track sentiment evolution over time
        
        Args:
            articles: DataFrame with articles
            text_column: Name of text column
            date_column: Name of date column
            category_column: Name of category column
            
        Returns:
            Dictionary with temporal sentiment analysis
        """
        logging.info("Analyzing sentiment trends over time...")
        
        # Ensure date column is datetime
        if date_column in articles.columns:
            articles[date_column] = pd.to_datetime(articles[date_column])
        else:
            logging.warning(f"Date column '{date_column}' not found. Using index as time reference.")
            articles[date_column] = pd.date_range(start='2020-01-01', periods=len(articles), freq='D')
        
        # Analyze sentiment for all articles
        sentiment_results = self.analyze_batch_sentiment(articles[text_column].tolist(), include_metadata=False)
        
        # Add sentiment data to DataFrame
        sentiment_df = articles.copy()
        sentiment_df['sentiment_compound'] = [r.get('vader', 0) for r in sentiment_results]
        sentiment_df['sentiment_polarity'] = [r.get('textblob', 0) for r in sentiment_results]
        sentiment_df['sentiment_class'] = [r.get('classification', 'neutral') for r in sentiment_results]
        
        # Temporal analysis
        temporal_analysis = {}
        
        # Overall trend
        temporal_analysis['overall_trend'] = self._calculate_sentiment_trend(
            sentiment_df, date_column, 'sentiment_compound'
        )
        
        # Category-specific trends
        if category_column in sentiment_df.columns:
            category_trends = {}
            for category in sentiment_df[category_column].unique():
                category_data = sentiment_df[sentiment_df[category_column] == category]
                category_trends[category] = self._calculate_sentiment_trend(
                    category_data, date_column, 'sentiment_compound'
                )
            temporal_analysis['category_trends'] = category_trends
        
        # Weekly aggregations
        sentiment_df['week'] = sentiment_df[date_column].dt.to_period('W')
        weekly_sentiment = sentiment_df.groupby('week').agg({
            'sentiment_compound': ['mean', 'std', 'count'],
            'sentiment_polarity': ['mean', 'std'],
            'sentiment_class': lambda x: x.value_counts().to_dict()
        }).reset_index()
        
        weekly_sentiment.columns = ['week', 'compound_mean', 'compound_std', 'count', 
                                   'polarity_mean', 'polarity_std', 'class_distribution']
        
        temporal_analysis['weekly_aggregation'] = weekly_sentiment.to_dict('records')
        
        # Monthly aggregations
        sentiment_df['month'] = sentiment_df[date_column].dt.to_period('M')
        monthly_sentiment = sentiment_df.groupby('month').agg({
            'sentiment_compound': ['mean', 'std', 'count'],
            'sentiment_polarity': ['mean', 'std'],
            'sentiment_class': lambda x: x.value_counts().to_dict()
        }).reset_index()
        
        monthly_sentiment.columns = ['month', 'compound_mean', 'compound_std', 'count',
                                    'polarity_mean', 'polarity_std', 'class_distribution']
        
        temporal_analysis['monthly_aggregation'] = monthly_sentiment.to_dict('records')
        
        # Detect sentiment anomalies
        temporal_analysis['anomalies'] = self._detect_sentiment_anomalies(sentiment_df, date_column)
        
        # Store processed data
        temporal_analysis['processed_data'] = sentiment_df
        temporal_analysis['analysis_timestamp'] = datetime.now().isoformat()
        temporal_analysis['total_articles'] = len(sentiment_df)
        temporal_analysis['date_range'] = {
            'start': sentiment_df[date_column].min().isoformat(),
            'end': sentiment_df[date_column].max().isoformat()
        }
        
        return temporal_analysis
    
    def _calculate_sentiment_trend(self, df: pd.DataFrame, date_column: str, sentiment_column: str) -> Dict[str, Any]:
        """Calculate sentiment trend metrics"""
        
        # Sort by date
        df_sorted = df.sort_values(date_column)
        
        # Calculate rolling averages
        df_sorted['sentiment_ma7'] = df_sorted[sentiment_column].rolling(window=7, min_periods=1).mean()
        df_sorted['sentiment_ma30'] = df_sorted[sentiment_column].rolling(window=30, min_periods=1).mean()
        
        # Calculate trend slope (linear regression)
        from scipy import stats
        
        x = np.arange(len(df_sorted))
        y = df_sorted[sentiment_column].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determine trend direction
        if slope > 0.001:
            trend_direction = 'increasing'
        elif slope < -0.001:
            trend_direction = 'decreasing'
        else:
            trend_direction = 'stable'
        
        # Calculate volatility
        volatility = df_sorted[sentiment_column].std()
        
        return {
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'trend_direction': trend_direction,
            'volatility': volatility,
            'mean_sentiment': df_sorted[sentiment_column].mean(),
            'start_sentiment': df_sorted[sentiment_column].iloc[0],
            'end_sentiment': df_sorted[sentiment_column].iloc[-1],
            'max_sentiment': df_sorted[sentiment_column].max(),
            'min_sentiment': df_sorted[sentiment_column].min()
        }
    
    def _detect_sentiment_anomalies(self, df: pd.DataFrame, date_column: str) -> List[Dict[str, Any]]:
        """Detect unusual sentiment patterns"""
        
        anomalies = []
        
        # Calculate z-scores for sentiment
        sentiment_mean = df['sentiment_compound'].mean()
        sentiment_std = df['sentiment_compound'].std()
        df['sentiment_zscore'] = (df['sentiment_compound'] - sentiment_mean) / sentiment_std
        
        # Find extreme sentiment days (|z-score| > 2.5)
        extreme_days = df[abs(df['sentiment_zscore']) > 2.5]
        
        for _, row in extreme_days.iterrows():
            anomaly = {
                'date': row[date_column].isoformat(),
                'sentiment_score': row['sentiment_compound'],
                'z_score': row['sentiment_zscore'],
                'type': 'extremely_positive' if row['sentiment_zscore'] > 0 else 'extremely_negative',
                'text_sample': row.get('text', '')[:200] + '...' if len(row.get('text', '')) > 200 else row.get('text', '')
            }
            anomalies.append(anomaly)
        
        # Detect sudden sentiment shifts (large day-to-day changes)
        df_sorted = df.sort_values(date_column)
        df_sorted['sentiment_change'] = df_sorted['sentiment_compound'].diff()
        
        large_changes = df_sorted[abs(df_sorted['sentiment_change']) > 0.5]
        
        for _, row in large_changes.iterrows():
            anomaly = {
                'date': row[date_column].isoformat(),
                'sentiment_change': row['sentiment_change'],
                'type': 'sudden_shift',
                'direction': 'positive' if row['sentiment_change'] > 0 else 'negative'
            }
            anomalies.append(anomaly)
        
        return sorted(anomalies, key=lambda x: x['date'])
    
    def _classify_vader_sentiment(self, compound_score: float) -> str:
        """Classify VADER compound score into sentiment category"""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def _classify_textblob_sentiment(self, polarity: float) -> str:
        """Classify TextBlob polarity into sentiment category"""
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _map_transformer_sentiment(self, label: str) -> str:
        """Map transformer sentiment label to standard classification"""
        label_mapping = {
            'positive': 'positive',
            'negative': 'negative',
            'neutral': 'neutral',
            'label_1': 'negative',  # RoBERTa model labels
            'label_2': 'positive'
        }
        return label_mapping.get(label.lower(), 'neutral')
    
    def _aggregate_sentiment_scores(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate sentiment scores from multiple methods"""
        
        scores = []
        classifications = []
        weights = []
        
        # VADER (weight: 0.4)
        if 'vader' in results:
            scores.append(results['vader']['compound'])
            classifications.append(results['vader']['classification'])
            weights.append(0.4)
        
        # TextBlob (weight: 0.3)
        if 'textblob' in results:
            scores.append(results['textblob']['polarity'])
            classifications.append(results['textblob']['classification'])
            weights.append(0.3)
        
        # Transformer (weight: 0.3)
        if 'transformer' in results and 'error' not in results['transformer']:
            transformer_score = results['transformer']['scores'].get('positive', 0) - results['transformer']['scores'].get('negative', 0)
            scores.append(transformer_score)
            classifications.append(results['transformer']['classification'])
            weights.append(0.3)
        
        if not scores:
            return {'error': 'No valid sentiment scores'}
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Calculate weighted average
        weighted_score = np.average(scores, weights=weights)
        
        # Determine final classification by majority vote
        classification_counts = {}
        for i, classification in enumerate(classifications):
            if classification not in classification_counts:
                classification_counts[classification] = 0
            classification_counts[classification] += weights[i]
        
        final_classification = max(classification_counts, key=classification_counts.get)
        
        return {
            'weighted_score': weighted_score,
            'classification': final_classification,
            'confidence': max(classification_counts.values()),
            'individual_scores': scores,
            'individual_classifications': classifications,
            'weights_used': weights.tolist()
        }
    
    def _assess_confidence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess confidence in sentiment analysis results"""
        
        confidence_factors = []
        
        # Agreement between methods
        if 'aggregate' in results and 'individual_classifications' in results['aggregate']:
            classifications = results['aggregate']['individual_classifications']
            agreement_ratio = classifications.count(results['aggregate']['classification']) / len(classifications)
            confidence_factors.append(('method_agreement', agreement_ratio))
        
        # Text length (longer texts generally more reliable)
        text_length = len(results.get('text', ''))
        length_confidence = min(text_length / 500, 1.0)  # Cap at 500 characters
        confidence_factors.append(('text_length', length_confidence))
        
        # Transformer confidence (if available)
        if 'transformer' in results and 'confidence' in results['transformer']:
            confidence_factors.append(('transformer_confidence', results['transformer']['confidence']))
        
        # VADER confidence (based on compound score magnitude)
        if 'vader' in results:
            vader_confidence = abs(results['vader']['compound'])
            confidence_factors.append(('vader_magnitude', vader_confidence))
        
        # Calculate overall confidence
        if confidence_factors:
            overall_confidence = np.mean([factor[1] for factor in confidence_factors])
        else:
            overall_confidence = 0.5
        
        # Categorize confidence level
        if overall_confidence >= 0.8:
            confidence_level = 'high'
        elif overall_confidence >= 0.6:
            confidence_level = 'medium'
        elif overall_confidence >= 0.4:
            confidence_level = 'low'
        else:
            confidence_level = 'very_low'
        
        return {
            'overall_confidence': overall_confidence,
            'confidence_level': confidence_level,
            'confidence_factors': dict(confidence_factors)
        }
    
    def save_sentiment_model(self, filepath: str):
        """Save sentiment analyzer configuration"""
        model_data = {
            'config': self.config,
            'sentiment_mapping': self.sentiment_mapping,
            'emotion_categories': self.emotion_categories,
            'has_transformers': HAS_TRANSFORMERS,
            'save_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info(f"Sentiment analyzer saved to {filepath}")
    
    def load_sentiment_model(self, filepath: str):
        """Load sentiment analyzer configuration"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.sentiment_mapping = model_data['sentiment_mapping']
        self.emotion_categories = model_data['emotion_categories']
        
        logging.info(f"Sentiment analyzer loaded from {filepath}")
    
    def get_sentiment_statistics(self, sentiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from sentiment analysis results"""
        
        if not sentiment_results:
            return {'error': 'No sentiment results provided'}
        
        # Extract sentiment scores and classifications
        compound_scores = []
        polarity_scores = []
        classifications = []
        
        for result in sentiment_results:
            if 'aggregate' in result:
                compound_scores.append(result['aggregate'].get('weighted_score', 0))
            if 'vader' in result:
                compound_scores.append(result['vader'].get('compound', 0))
            if 'textblob' in result:
                polarity_scores.append(result['textblob'].get('polarity', 0))
            if 'aggregate' in result:
                classifications.append(result['aggregate'].get('classification', 'neutral'))
        
        statistics = {}
        
        # Score statistics
        if compound_scores:
            statistics['compound_scores'] = {
                'mean': np.mean(compound_scores),
                'std': np.std(compound_scores),
                'min': np.min(compound_scores),
                'max': np.max(compound_scores),
                'median': np.median(compound_scores)
            }
        
        if polarity_scores:
            statistics['polarity_scores'] = {
                'mean': np.mean(polarity_scores),
                'std': np.std(polarity_scores),
                'min': np.min(polarity_scores),
                'max': np.max(polarity_scores),
                'median': np.median(polarity_scores)
            }
        
        # Classification distribution
        if classifications:
            from collections import Counter
            class_counts = Counter(classifications)
            total = len(classifications)
            
            statistics['classification_distribution'] = {
                'counts': dict(class_counts),
                'percentages': {k: (v/total)*100 for k, v in class_counts.items()}
            }
        
        statistics['total_analyzed'] = len(sentiment_results)
        statistics['analysis_timestamp'] = datetime.now().isoformat()
        
        return statistics