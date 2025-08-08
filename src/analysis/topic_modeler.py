#!/usr/bin/env python3
"""
Topic Modeler for NewsBot 2.0
Advanced topic modeling with LDA, NMF, and trend analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import pickle
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import gensim for advanced topic modeling
try:
    import gensim
    from gensim import corpora, models
    from gensim.models import CoherenceModel, LdaModel
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False
    logging.warning("gensim not available. Install for advanced topic modeling features.")

# Try to import pyLDAvis for visualization
try:
    import pyLDAvis
    import pyLDAvis.gensim_models  # sklearn module no longer exists in newer versions
    HAS_PYLDAVIS = True
except ImportError:
    HAS_PYLDAVIS = False
    logging.warning("pyLDAvis not available. Install for topic visualization features.")

class TopicModeler:
    """
    Advanced topic modeling for discovering themes and trends in news content
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize topic modeler
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Model parameters
        self.n_topics = self.config.get('n_topics', 10)
        self.method = self.config.get('method', 'lda')
        self.random_state = self.config.get('random_state', 42)
        
        # Vectorizer parameters
        self.vectorizer_params = {
            'max_features': self.config.get('max_features', 1000),
            'min_df': self.config.get('min_df', 2),
            'max_df': self.config.get('max_df', 0.95),
            'ngram_range': self.config.get('ngram_range', (1, 2)),
            'stop_words': 'english'
        }
        
        # Model components
        self.vectorizer = None
        self.topic_model = None
        self.document_term_matrix = None
        self.feature_names = None
        
        # Results storage
        self.topic_distributions = None
        self.topic_words = None
        self.topic_coherence_scores = None
        self.training_results = {}
        
        # Temporal analysis - Enhanced for evolution tracking
        self.temporal_topics = {}
        self.trend_analysis = {}
        self.topic_evolution_history = []
        self.temporal_coherence_scores = {}
        self.topic_emergence_tracking = {}
        self.cross_temporal_correlations = {}
        
        # Evolution tracking parameters
        self.evolution_window_size = self.config.get('evolution_window_size', 30)  # days
        self.evolution_step_size = self.config.get('evolution_step_size', 7)      # days
        self.min_articles_per_window = self.config.get('min_articles_per_window', 10)
        
        # Advanced models (if gensim available)
        self.gensim_dictionary = None
        self.gensim_corpus = None
        self.gensim_model = None
        
        self.is_fitted = False
    
    def fit_topics(self, documents: List[str], method: Optional[str] = None, 
                   n_topics: Optional[int] = None) -> Dict[str, Any]:
        """
        Discover topics in document collection
        
        Args:
            documents: List of preprocessed documents
            method: Topic modeling method ('lda', 'nmf', 'gensim_lda')
            n_topics: Number of topics to discover
            
        Returns:
            Dictionary with topic modeling results
        """
        logging.info(f"Starting topic modeling with {len(documents)} documents...")
        
        if method is not None:
            self.method = method
        if n_topics is not None:
            self.n_topics = n_topics
        
        # Prepare documents
        processed_docs = self._preprocess_documents(documents)
        
        # Create document-term matrix
        if self.method in ['lda', 'nmf']:
            self._create_sklearn_matrix(processed_docs)
            results = self._fit_sklearn_model()
        elif self.method == 'gensim_lda' and HAS_GENSIM:
            self._create_gensim_corpus(processed_docs)
            results = self._fit_gensim_model()
        else:
            logging.warning(f"Method {self.method} not available, falling back to sklearn LDA")
            self.method = 'lda'
            self._create_sklearn_matrix(processed_docs)
            results = self._fit_sklearn_model()
        
        # Extract topic information
        self._extract_topic_information()
        
        # Calculate topic coherence
        self._calculate_topic_coherence(processed_docs)
        
        # Analyze topic quality
        quality_metrics = self._analyze_topic_quality()
        
        # Store training results
        self.training_results = {
            'method': self.method,
            'n_topics': self.n_topics,
            'n_documents': len(documents),
            'quality_metrics': quality_metrics,
            'coherence_scores': self.topic_coherence_scores,
            'training_timestamp': datetime.now().isoformat()
        }
        
        self.is_fitted = True
        logging.info("Topic modeling completed successfully!")
        
        return self.training_results
    
    def _preprocess_documents(self, documents: List[str]) -> List[str]:
        """Additional preprocessing for topic modeling"""
        processed = []
        
        for doc in documents:
            if isinstance(doc, str) and len(doc.strip()) > 0:
                # Remove very short documents
                if len(doc.split()) >= 5:
                    processed.append(doc)
        
        logging.info(f"Preprocessed {len(processed)} documents (filtered {len(documents) - len(processed)} short documents)")
        return processed
    
    def _create_sklearn_matrix(self, documents: List[str]):
        """Create document-term matrix using sklearn"""
        
        if self.method == 'lda':
            # Use CountVectorizer for LDA
            self.vectorizer = CountVectorizer(**self.vectorizer_params)
        else:
            # Use TfidfVectorizer for NMF
            self.vectorizer = TfidfVectorizer(**self.vectorizer_params)
        
        self.document_term_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        logging.info(f"Created document-term matrix: {self.document_term_matrix.shape}")
    
    def _create_gensim_corpus(self, documents: List[str]):
        """Create gensim corpus and dictionary"""
        if not HAS_GENSIM:
            raise ImportError("gensim not available")
        
        # Tokenize documents
        tokenized_docs = [doc.split() for doc in documents]
        
        # Create dictionary
        self.gensim_dictionary = corpora.Dictionary(tokenized_docs)
        
        # Filter extremes
        self.gensim_dictionary.filter_extremes(
            no_below=self.vectorizer_params['min_df'],
            no_above=self.vectorizer_params['max_df']
        )
        
        # Create corpus
        self.gensim_corpus = [self.gensim_dictionary.doc2bow(doc) for doc in tokenized_docs]
        
        logging.info(f"Created gensim corpus with {len(self.gensim_corpus)} documents and {len(self.gensim_dictionary)} terms")
    
    def _fit_sklearn_model(self) -> Dict[str, Any]:
        """Fit sklearn topic model"""
        
        if self.method == 'lda':
            # Latent Dirichlet Allocation
            self.topic_model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=self.random_state,
                learning_method='online',
                max_iter=20,
                n_jobs=-1
            )
            
        elif self.method == 'nmf':
            # Non-negative Matrix Factorization
            self.topic_model = NMF(
                n_components=self.n_topics,
                random_state=self.random_state,
                max_iter=200,
                alpha=0.1,
                l1_ratio=0.5
            )
        
        # Fit model
        self.topic_distributions = self.topic_model.fit_transform(self.document_term_matrix)
        
        # Calculate perplexity (for LDA)
        results = {'method': self.method, 'n_topics': self.n_topics}
        
        if hasattr(self.topic_model, 'perplexity'):
            perplexity = self.topic_model.perplexity(self.document_term_matrix)
            results['perplexity'] = perplexity
            logging.info(f"Model perplexity: {perplexity}")
        
        return results
    
    def track_topic_evolution(self, articles_with_dates: List[Dict[str, Any]], 
                             window_size: Optional[int] = None,
                             step_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Track topic evolution over time with advanced temporal analysis
        
        Args:
            articles_with_dates: List of dicts with 'text', 'date', and optionally 'category'
            window_size: Size of temporal window in days
            step_size: Step size between windows in days
            
        Returns:
            Comprehensive topic evolution analysis
        """
        logging.info(f"Starting topic evolution tracking for {len(articles_with_dates)} articles...")
        
        window_size = window_size or self.evolution_window_size
        step_size = step_size or self.evolution_step_size
        
        # Sort articles by date
        sorted_articles = sorted(articles_with_dates, key=lambda x: pd.to_datetime(x['date']))
        
        if len(sorted_articles) < self.min_articles_per_window:
            return {'error': f'Insufficient articles. Need at least {self.min_articles_per_window}'}
        
        # Create temporal windows
        temporal_windows = self._create_temporal_windows(sorted_articles, window_size, step_size)
        
        # Track topic evolution across windows
        evolution_results = {
            'temporal_windows': [],
            'topic_emergence': {},
            'topic_decline': {},
            'topic_stability': {},
            'cross_window_correlations': {},
            'trend_analysis': {},
            'volatility_metrics': {},
            'semantic_drift': {}
        }
        
        # Process each temporal window
        for window_idx, window_data in enumerate(temporal_windows):
            window_result = self._analyze_temporal_window(window_data, window_idx)
            evolution_results['temporal_windows'].append(window_result)
        
        # Calculate cross-window relationships
        evolution_results['cross_window_correlations'] = self._calculate_cross_window_correlations(
            evolution_results['temporal_windows']
        )
        
        # Identify emerging and declining topics
        emergence_decline = self._identify_emerging_declining_topics(evolution_results['temporal_windows'])
        evolution_results['topic_emergence'] = emergence_decline['emerging']
        evolution_results['topic_decline'] = emergence_decline['declining']
        evolution_results['topic_stability'] = emergence_decline['stable']
        
        # Calculate trend analysis
        evolution_results['trend_analysis'] = self._calculate_topic_trends(evolution_results['temporal_windows'])
        
        # Calculate volatility metrics
        evolution_results['volatility_metrics'] = self._calculate_topic_volatility(evolution_results['temporal_windows'])
        
        # Analyze semantic drift
        evolution_results['semantic_drift'] = self._analyze_semantic_drift(evolution_results['temporal_windows'])
        
        # Store evolution history
        self.topic_evolution_history.append({
            'timestamp': datetime.now().isoformat(),
            'window_size': window_size,
            'step_size': step_size,
            'total_articles': len(sorted_articles),
            'total_windows': len(temporal_windows),
            'results': evolution_results
        })
        
        logging.info(f"Topic evolution tracking completed. Analyzed {len(temporal_windows)} temporal windows.")
        
        return evolution_results
    
    def _create_temporal_windows(self, sorted_articles: List[Dict[str, Any]], 
                                window_size: int, step_size: int) -> List[Dict[str, Any]]:
        """Create sliding temporal windows for analysis"""
        windows = []
        
        start_date = pd.to_datetime(sorted_articles[0]['date'])
        end_date = pd.to_datetime(sorted_articles[-1]['date'])
        
        current_start = start_date
        window_idx = 0
        
        while current_start + timedelta(days=window_size) <= end_date:
            current_end = current_start + timedelta(days=window_size)
            
            # Filter articles within current window
            window_articles = [
                article for article in sorted_articles
                if current_start <= pd.to_datetime(article['date']) < current_end
            ]
            
            if len(window_articles) >= self.min_articles_per_window:
                windows.append({
                    'window_idx': window_idx,
                    'start_date': current_start.isoformat(),
                    'end_date': current_end.isoformat(),
                    'articles': window_articles,
                    'article_count': len(window_articles)
                })
                window_idx += 1
            
            current_start += timedelta(days=step_size)
        
        return windows
    
    def _analyze_temporal_window(self, window_data: Dict[str, Any], window_idx: int) -> Dict[str, Any]:
        """Analyze topics within a single temporal window"""
        articles = window_data['articles']
        texts = [article['text'] for article in articles]
        
        # Create a temporary topic model for this window
        temp_modeler = TopicModeler({
            'n_topics': self.n_topics,
            'method': self.method,
            'random_state': self.random_state + window_idx  # Ensure reproducibility
        })
        
        # Fit topic model for this window
        window_results = temp_modeler.fit_topics(texts, method=self.method, n_topics=self.n_topics)
        
        # Get topic distributions for all articles in window
        topic_distributions = []
        for text in texts:
            try:
                topic_dist = temp_modeler.get_article_topics(text, top_k=self.n_topics)
                topic_distributions.append(topic_dist)
            except Exception as e:
                logging.warning(f"Failed to get topics for article in window {window_idx}: {e}")
        
        # Calculate window-specific metrics
        window_metrics = {
            'window_idx': window_idx,
            'start_date': window_data['start_date'],
            'end_date': window_data['end_date'],
            'article_count': len(articles),
            'topic_model_results': window_results,
            'topic_distributions': topic_distributions,
            'average_topic_coherence': window_results.get('coherence_score', 0),
            'topic_diversity': self._calculate_topic_diversity(topic_distributions),
            'dominant_topics': self._get_dominant_topics(topic_distributions),
            'topic_concentration': self._calculate_topic_concentration(topic_distributions)
        }
        
        return window_metrics
    
    def _calculate_cross_window_correlations(self, temporal_windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate correlations between topics across temporal windows"""
        correlations = {}
        
        if len(temporal_windows) < 2:
            return correlations
        
        # Extract topic distributions for each window
        window_topic_dists = []
        for window in temporal_windows:
            if 'dominant_topics' in window:
                # Create a vector of average topic probabilities
                topic_probs = np.zeros(self.n_topics)
                for topic_info in window['dominant_topics']:
                    topic_id = topic_info.get('topic_id', 0)
                    prob = topic_info.get('average_probability', 0)
                    if topic_id < self.n_topics:
                        topic_probs[topic_id] = prob
                window_topic_dists.append(topic_probs)
        
        if len(window_topic_dists) >= 2:
            # Calculate correlation matrix between consecutive windows
            correlation_matrix = np.corrcoef(window_topic_dists)
            
            correlations = {
                'correlation_matrix': correlation_matrix.tolist(),
                'average_correlation': np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]),
                'consecutive_correlations': [
                    correlation_matrix[i][i+1] for i in range(len(correlation_matrix)-1)
                ]
            }
        
        return correlations
    
    def _identify_emerging_declining_topics(self, temporal_windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify topics that are emerging, declining, or stable over time"""
        if len(temporal_windows) < 3:
            return {'emerging': [], 'declining': [], 'stable': []}
        
        # Track topic probabilities over time
        topic_time_series = defaultdict(list)
        
        for window in temporal_windows:
            window_topic_probs = np.zeros(self.n_topics)
            if 'dominant_topics' in window:
                for topic_info in window['dominant_topics']:
                    topic_id = topic_info.get('topic_id', 0)
                    prob = topic_info.get('average_probability', 0)
                    if topic_id < self.n_topics:
                        window_topic_probs[topic_id] = prob
            
            for topic_id in range(self.n_topics):
                topic_time_series[topic_id].append(window_topic_probs[topic_id])
        
        emerging_topics = []
        declining_topics = []
        stable_topics = []
        
        for topic_id, probabilities in topic_time_series.items():
            if len(probabilities) >= 3:
                # Calculate trend using linear regression slope
                x = np.arange(len(probabilities))
                slope = np.polyfit(x, probabilities, 1)[0]
                
                # Calculate variability
                std_dev = np.std(probabilities)
                mean_prob = np.mean(probabilities)
                
                # Classification thresholds
                emergence_threshold = 0.01  # Positive slope threshold
                decline_threshold = -0.01   # Negative slope threshold
                stability_threshold = 0.02  # Low variability threshold
                
                if slope > emergence_threshold and mean_prob > 0.05:
                    emerging_topics.append({
                        'topic_id': topic_id,
                        'slope': float(slope),
                        'mean_probability': float(mean_prob),
                        'trend_strength': float(abs(slope) / (std_dev + 1e-6)),
                        'final_probability': float(probabilities[-1])
                    })
                elif slope < decline_threshold and mean_prob > 0.05:
                    declining_topics.append({
                        'topic_id': topic_id,
                        'slope': float(slope),
                        'mean_probability': float(mean_prob),
                        'trend_strength': float(abs(slope) / (std_dev + 1e-6)),
                        'final_probability': float(probabilities[-1])
                    })
                elif std_dev < stability_threshold and mean_prob > 0.05:
                    stable_topics.append({
                        'topic_id': topic_id,
                        'slope': float(slope),
                        'mean_probability': float(mean_prob),
                        'stability_score': float(1 / (std_dev + 1e-6)),
                        'consistency': float(1 - (std_dev / (mean_prob + 1e-6)))
                    })
        
        return {
            'emerging': sorted(emerging_topics, key=lambda x: x['trend_strength'], reverse=True),
            'declining': sorted(declining_topics, key=lambda x: x['trend_strength'], reverse=True),
            'stable': sorted(stable_topics, key=lambda x: x['stability_score'], reverse=True)
        }
    
    def _calculate_topic_trends(self, temporal_windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive trend analysis for topics"""
        trends = {}
        
        if len(temporal_windows) < 2:
            return trends
        
        # Extract timestamps and topic data
        timestamps = [pd.to_datetime(window['start_date']) for window in temporal_windows]
        
        # Calculate trend metrics
        trends = {
            'overall_diversity_trend': self._calculate_diversity_trend(temporal_windows),
            'topic_volatility_trend': self._calculate_volatility_trend(temporal_windows),
            'coherence_trend': self._calculate_coherence_trend(temporal_windows),
            'concentration_trend': self._calculate_concentration_trend(temporal_windows)
        }
        
        return trends
    
    def _calculate_topic_volatility(self, temporal_windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate volatility metrics for topics over time"""
        volatility_metrics = {}
        
        if len(temporal_windows) < 2:
            return volatility_metrics
        
        # Calculate various volatility measures
        topic_concentrations = [window.get('topic_concentration', 0) for window in temporal_windows]
        diversity_scores = [window.get('topic_diversity', 0) for window in temporal_windows]
        
        volatility_metrics = {
            'concentration_volatility': float(np.std(topic_concentrations)),
            'diversity_volatility': float(np.std(diversity_scores)),
            'overall_volatility': float(np.std(topic_concentrations) + np.std(diversity_scores)) / 2,
            'volatility_trend': float(np.polyfit(range(len(topic_concentrations)), topic_concentrations, 1)[0])
        }
        
        return volatility_metrics
    
    def _analyze_semantic_drift(self, temporal_windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how topic semantics drift over time"""
        semantic_drift = {}
        
        if len(temporal_windows) < 2:
            return semantic_drift
        
        # Track how topic words change over time
        topic_word_evolution = defaultdict(list)
        
        for window in temporal_windows:
            if 'topic_model_results' in window and 'topics' in window['topic_model_results']:
                topics = window['topic_model_results']['topics']
                for topic_id, topic_words in topics.items():
                    if isinstance(topic_words, dict) and 'words' in topic_words:
                        topic_word_evolution[topic_id].append(set(topic_words['words'][:10]))
        
        # Calculate semantic drift metrics
        drift_scores = {}
        for topic_id, word_sets in topic_word_evolution.items():
            if len(word_sets) >= 2:
                # Calculate Jaccard similarity between consecutive time periods
                similarities = []
                for i in range(len(word_sets) - 1):
                    intersection = len(word_sets[i] & word_sets[i+1])
                    union = len(word_sets[i] | word_sets[i+1])
                    similarity = intersection / union if union > 0 else 0
                    similarities.append(similarity)
                
                drift_scores[f'topic_{topic_id}'] = {
                    'average_similarity': float(np.mean(similarities)),
                    'drift_rate': float(1 - np.mean(similarities)),
                    'similarity_trend': float(np.polyfit(range(len(similarities)), similarities, 1)[0]),
                    'max_drift': float(1 - min(similarities)) if similarities else 0
                }
        
        semantic_drift = {
            'topic_drift_scores': drift_scores,
            'average_drift_rate': float(np.mean([score['drift_rate'] for score in drift_scores.values()])) if drift_scores else 0,
            'most_stable_topic': min(drift_scores.items(), key=lambda x: x[1]['drift_rate'])[0] if drift_scores else None,
            'most_volatile_topic': max(drift_scores.items(), key=lambda x: x[1]['drift_rate'])[0] if drift_scores else None
        }
        
        return semantic_drift
    
    def _calculate_topic_diversity(self, topic_distributions: List[Dict[str, Any]]) -> float:
        """Calculate topic diversity within a window"""
        if not topic_distributions:
            return 0.0
        
        topic_counts = Counter()
        for dist in topic_distributions:
            if 'top_topics' in dist:
                for topic in dist['top_topics']:
                    topic_counts[topic.get('topic_id', 0)] += 1
        
        # Calculate Shannon diversity
        total = sum(topic_counts.values())
        if total == 0:
            return 0.0
        
        diversity = -sum((count / total) * np.log(count / total) for count in topic_counts.values())
        return float(diversity)
    
    def _get_dominant_topics(self, topic_distributions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get dominant topics within a window"""
        topic_probs = defaultdict(list)
        
        for dist in topic_distributions:
            if 'top_topics' in dist:
                for topic in dist['top_topics']:
                    topic_id = topic.get('topic_id', 0)
                    prob = topic.get('probability', 0)
                    topic_probs[topic_id].append(prob)
        
        dominant_topics = []
        for topic_id, probs in topic_probs.items():
            dominant_topics.append({
                'topic_id': topic_id,
                'average_probability': float(np.mean(probs)),
                'max_probability': float(max(probs)),
                'frequency': len(probs),
                'consistency': float(1 - np.std(probs) / (np.mean(probs) + 1e-6))
            })
        
        return sorted(dominant_topics, key=lambda x: x['average_probability'], reverse=True)
    
    def _calculate_topic_concentration(self, topic_distributions: List[Dict[str, Any]]) -> float:
        """Calculate how concentrated topics are (Herfindahl index)"""
        topic_counts = Counter()
        total_articles = len(topic_distributions)
        
        for dist in topic_distributions:
            if 'top_topics' in dist and dist['top_topics']:
                # Use the most probable topic for each article
                top_topic = dist['top_topics'][0]
                topic_id = top_topic.get('topic_id', 0)
                topic_counts[topic_id] += 1
        
        if total_articles == 0:
            return 0.0
        
        # Calculate Herfindahl index (sum of squared market shares)
        herfindahl = sum((count / total_articles) ** 2 for count in topic_counts.values())
        return float(herfindahl)
    
    def _calculate_diversity_trend(self, temporal_windows: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate trend in topic diversity over time"""
        diversity_scores = [window.get('topic_diversity', 0) for window in temporal_windows]
        if len(diversity_scores) < 2:
            return {'slope': 0.0, 'trend': 'stable'}
        
        x = np.arange(len(diversity_scores))
        slope = float(np.polyfit(x, diversity_scores, 1)[0])
        
        return {
            'slope': slope,
            'trend': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable',
            'initial_diversity': float(diversity_scores[0]),
            'final_diversity': float(diversity_scores[-1])
        }
    
    def _calculate_volatility_trend(self, temporal_windows: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate trend in topic volatility over time"""
        concentrations = [window.get('topic_concentration', 0) for window in temporal_windows]
        if len(concentrations) < 2:
            return {'slope': 0.0, 'trend': 'stable'}
        
        # Calculate rolling volatility
        rolling_volatility = []
        window_size = min(3, len(concentrations))
        for i in range(window_size - 1, len(concentrations)):
            window_vals = concentrations[i-window_size+1:i+1]
            rolling_volatility.append(np.std(window_vals))
        
        if len(rolling_volatility) < 2:
            return {'slope': 0.0, 'trend': 'stable'}
        
        x = np.arange(len(rolling_volatility))
        slope = float(np.polyfit(x, rolling_volatility, 1)[0])
        
        return {
            'slope': slope,
            'trend': 'increasing' if slope > 0.001 else 'decreasing' if slope < -0.001 else 'stable',
            'average_volatility': float(np.mean(rolling_volatility))
        }
    
    def _calculate_coherence_trend(self, temporal_windows: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate trend in topic coherence over time"""
        coherence_scores = [window.get('average_topic_coherence', 0) for window in temporal_windows]
        if len(coherence_scores) < 2:
            return {'slope': 0.0, 'trend': 'stable'}
        
        x = np.arange(len(coherence_scores))
        slope = float(np.polyfit(x, coherence_scores, 1)[0])
        
        return {
            'slope': slope,
            'trend': 'improving' if slope > 0.001 else 'degrading' if slope < -0.001 else 'stable',
            'initial_coherence': float(coherence_scores[0]),
            'final_coherence': float(coherence_scores[-1])
        }
    
    def _calculate_concentration_trend(self, temporal_windows: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate trend in topic concentration over time"""
        concentrations = [window.get('topic_concentration', 0) for window in temporal_windows]
        if len(concentrations) < 2:
            return {'slope': 0.0, 'trend': 'stable'}
        
        x = np.arange(len(concentrations))
        slope = float(np.polyfit(x, concentrations, 1)[0])
        
        return {
            'slope': slope,
            'trend': 'concentrating' if slope > 0.01 else 'diversifying' if slope < -0.01 else 'stable',
            'initial_concentration': float(concentrations[0]),
            'final_concentration': float(concentrations[-1])
        }
    
    def _fit_gensim_model(self) -> Dict[str, Any]:
        """Fit gensim LDA model"""
        if not HAS_GENSIM:
            raise ImportError("gensim not available")
        
        # Train LDA model
        self.gensim_model = LdaModel(
            corpus=self.gensim_corpus,
            id2word=self.gensim_dictionary,
            num_topics=self.n_topics,
            random_state=self.random_state,
            passes=10,
            alpha='auto',
            per_word_topics=True,
            eval_every=1
        )
        
        # Get document topic distributions
        self.topic_distributions = []
        for doc_bow in self.gensim_corpus:
            doc_topics = self.gensim_model.get_document_topics(doc_bow, minimum_probability=0)
            topic_dist = np.zeros(self.n_topics)
            for topic_id, prob in doc_topics:
                topic_dist[topic_id] = prob
            self.topic_distributions.append(topic_dist)
        
        self.topic_distributions = np.array(self.topic_distributions)
        
        # Calculate perplexity
        perplexity = self.gensim_model.log_perplexity(self.gensim_corpus)
        
        return {
            'method': 'gensim_lda',
            'n_topics': self.n_topics,
            'perplexity': perplexity
        }
    
    def _extract_topic_information(self):
        """Extract topic words and information"""
        self.topic_words = {}
        
        if self.method in ['lda', 'nmf'] and self.topic_model:
            # sklearn models
            n_top_words = 15
            
            for topic_idx, topic in enumerate(self.topic_model.components_):
                top_words_idx = topic.argsort()[-n_top_words:][::-1]
                top_words = [self.feature_names[i] for i in top_words_idx]
                top_scores = [topic[i] for i in top_words_idx]
                
                self.topic_words[topic_idx] = {
                    'words': top_words,
                    'scores': top_scores,
                    'word_score_pairs': list(zip(top_words, top_scores))
                }
        
        elif self.method == 'gensim_lda' and self.gensim_model:
            # gensim model
            for topic_idx in range(self.n_topics):
                topic_words = self.gensim_model.show_topic(topic_idx, topn=15)
                
                words = [word for word, score in topic_words]
                scores = [score for word, score in topic_words]
                
                self.topic_words[topic_idx] = {
                    'words': words,
                    'scores': scores,
                    'word_score_pairs': topic_words
                }
    
    def _calculate_topic_coherence(self, documents: List[str]):
        """Calculate topic coherence scores"""
        self.topic_coherence_scores = {}
        
        try:
            if self.method == 'gensim_lda' and HAS_GENSIM and self.gensim_model:
                # Gensim coherence
                coherence_model = CoherenceModel(
                    model=self.gensim_model,
                    texts=[doc.split() for doc in documents],
                    dictionary=self.gensim_dictionary,
                    coherence='c_v'
                )
                
                coherence_score = coherence_model.get_coherence()
                self.topic_coherence_scores['c_v'] = coherence_score
                
                # Per-topic coherence
                per_topic_coherence = coherence_model.get_coherence_per_topic()
                self.topic_coherence_scores['per_topic'] = per_topic_coherence
                
            else:
                # Simple coherence for sklearn models
                self.topic_coherence_scores['simple'] = self._calculate_simple_coherence(documents)
                
        except Exception as e:
            logging.warning(f"Could not calculate coherence scores: {e}")
            self.topic_coherence_scores = {'error': str(e)}
    
    def _calculate_simple_coherence(self, documents: List[str]) -> float:
        """Calculate simple coherence measure for sklearn models"""
        if not self.topic_words:
            return 0.0
        
        # Simple co-occurrence based coherence
        total_coherence = 0
        
        for topic_idx in self.topic_words:
            topic_words = self.topic_words[topic_idx]['words'][:10]  # Top 10 words
            
            # Calculate pairwise co-occurrence
            coherence_sum = 0
            pair_count = 0
            
            for i, word1 in enumerate(topic_words):
                for j, word2 in enumerate(topic_words[i+1:], i+1):
                    # Count co-occurrence in documents
                    cooccur_count = 0
                    word1_count = 0
                    
                    for doc in documents:
                        if word1 in doc:
                            word1_count += 1
                            if word2 in doc:
                                cooccur_count += 1
                    
                    if word1_count > 0:
                        coherence_sum += cooccur_count / word1_count
                        pair_count += 1
            
            if pair_count > 0:
                total_coherence += coherence_sum / pair_count
        
        return total_coherence / len(self.topic_words) if self.topic_words else 0.0
    
    def _analyze_topic_quality(self) -> Dict[str, Any]:
        """Analyze overall topic quality"""
        quality_metrics = {}
        
        if self.topic_distributions is not None:
            # Topic diversity
            dominant_topics = np.argmax(self.topic_distributions, axis=1)
            topic_counts = Counter(dominant_topics)
            
            # Calculate topic balance (how evenly distributed topics are)
            topic_balance = len(topic_counts) / self.n_topics
            quality_metrics['topic_balance'] = topic_balance
            
            # Topic concentration (how concentrated document-topic distributions are)
            avg_concentration = np.mean(np.max(self.topic_distributions, axis=1))
            quality_metrics['avg_topic_concentration'] = avg_concentration
            
            # Topic separation (how distinct topics are)
            topic_similarity_matrix = np.corrcoef(self.topic_distributions.T)
            avg_topic_similarity = np.mean(topic_similarity_matrix[np.triu_indices_from(topic_similarity_matrix, k=1)])
            quality_metrics['avg_topic_similarity'] = avg_topic_similarity
            
            # Document coverage (how well documents are covered by topics)
            doc_topic_sums = np.sum(self.topic_distributions, axis=1)
            quality_metrics['avg_document_coverage'] = np.mean(doc_topic_sums)
        
        return quality_metrics
    
    def get_article_topics(self, article_text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Get topic distribution for a single article
        
        Args:
            article_text: Preprocessed article text
            top_k: Number of top topics to return
            
        Returns:
            Dictionary with topic information
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_topics() first.")
        
        result = {
            'text': article_text,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.method in ['lda', 'nmf'] and self.vectorizer:
            # Transform text using fitted vectorizer
            doc_term_vector = self.vectorizer.transform([article_text])
            
            # Get topic distribution
            topic_dist = self.topic_model.transform(doc_term_vector)[0]
            
        elif self.method == 'gensim_lda' and self.gensim_model:
            # Transform using gensim
            doc_bow = self.gensim_dictionary.doc2bow(article_text.split())
            topic_dist_tuples = self.gensim_model.get_document_topics(doc_bow, minimum_probability=0)
            
            topic_dist = np.zeros(self.n_topics)
            for topic_id, prob in topic_dist_tuples:
                topic_dist[topic_id] = prob
        
        else:
            raise ValueError("No fitted model available")
        
        # Get top topics
        top_topic_indices = np.argsort(topic_dist)[-top_k:][::-1]
        
        top_topics = []
        for idx in top_topic_indices:
            topic_info = {
                'topic_id': int(idx),
                'probability': float(topic_dist[idx]),
                'top_words': self.topic_words[idx]['words'][:5] if idx in self.topic_words else []
            }
            top_topics.append(topic_info)
        
        result['topic_distribution'] = topic_dist.tolist()
        result['top_topics'] = top_topics
        result['dominant_topic'] = int(np.argmax(topic_dist))
        result['dominant_topic_probability'] = float(np.max(topic_dist))
        
        return result
    
    def track_topic_trends(self, articles_with_dates: pd.DataFrame, 
                          text_column: str = 'text',
                          date_column: str = 'date') -> Dict[str, Any]:
        """
        Analyze how topics change over time
        
        Args:
            articles_with_dates: DataFrame with articles and dates
            text_column: Name of text column
            date_column: Name of date column
            
        Returns:
            Dictionary with temporal topic analysis
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_topics() first.")
        
        logging.info("Analyzing topic trends over time...")
        
        # Ensure date column is datetime
        articles_with_dates[date_column] = pd.to_datetime(articles_with_dates[date_column])
        
        # Get topic distributions for all articles
        topic_results = []
        for text in articles_with_dates[text_column]:
            topic_result = self.get_article_topics(text)
            topic_results.append(topic_result)
        
        # Add topic information to DataFrame
        df_with_topics = articles_with_dates.copy()
        df_with_topics['dominant_topic'] = [r['dominant_topic'] for r in topic_results]
        df_with_topics['dominant_topic_prob'] = [r['dominant_topic_probability'] for r in topic_results]
        
        # Temporal analysis
        temporal_analysis = {}
        
        # Monthly topic trends
        df_with_topics['month'] = df_with_topics[date_column].dt.to_period('M')
        monthly_topics = df_with_topics.groupby(['month', 'dominant_topic']).size().unstack(fill_value=0)
        
        # Normalize to get proportions
        monthly_topic_props = monthly_topics.div(monthly_topics.sum(axis=1), axis=0)
        
        temporal_analysis['monthly_trends'] = {
            'raw_counts': monthly_topics.to_dict(),
            'proportions': monthly_topic_props.to_dict()
        }
        
        # Weekly topic trends
        df_with_topics['week'] = df_with_topics[date_column].dt.to_period('W')
        weekly_topics = df_with_topics.groupby(['week', 'dominant_topic']).size().unstack(fill_value=0)
        weekly_topic_props = weekly_topics.div(weekly_topics.sum(axis=1), axis=0)
        
        temporal_analysis['weekly_trends'] = {
            'raw_counts': weekly_topics.to_dict(),
            'proportions': weekly_topic_props.to_dict()
        }
        
        # Topic evolution metrics
        temporal_analysis['topic_evolution'] = self._analyze_topic_evolution(monthly_topic_props)
        
        # Emerging and declining topics
        temporal_analysis['emerging_declining'] = self._find_emerging_declining_topics(monthly_topic_props)
        
        # Seasonal patterns
        temporal_analysis['seasonal_patterns'] = self._analyze_seasonal_patterns(df_with_topics, date_column)
        
        # Topic stability
        temporal_analysis['topic_stability'] = self._calculate_topic_stability(monthly_topic_props)
        
        temporal_analysis['analysis_timestamp'] = datetime.now().isoformat()
        temporal_analysis['date_range'] = {
            'start': df_with_topics[date_column].min().isoformat(),
            'end': df_with_topics[date_column].max().isoformat()
        }
        
        self.temporal_topics = temporal_analysis
        return temporal_analysis
    
    def _analyze_topic_evolution(self, monthly_props: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how topics evolve over time"""
        evolution_metrics = {}
        
        for topic_id in monthly_props.columns:
            topic_series = monthly_props[topic_id]
            
            # Calculate trend
            x = np.arange(len(topic_series))
            y = topic_series.values
            
            # Linear trend
            slope, intercept = np.polyfit(x, y, 1)
            
            # Volatility
            volatility = topic_series.std()
            
            # Growth rate
            if len(topic_series) > 1:
                growth_rate = (topic_series.iloc[-1] - topic_series.iloc[0]) / topic_series.iloc[0] if topic_series.iloc[0] > 0 else 0
            else:
                growth_rate = 0
            
            evolution_metrics[int(topic_id)] = {
                'trend_slope': slope,
                'volatility': volatility,
                'growth_rate': growth_rate,
                'mean_proportion': topic_series.mean(),
                'max_proportion': topic_series.max(),
                'min_proportion': topic_series.min()
            }
        
        return evolution_metrics
    
    def _find_emerging_declining_topics(self, monthly_props: pd.DataFrame) -> Dict[str, Any]:
        """Find emerging and declining topics"""
        
        if len(monthly_props) < 3:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Split into early and late periods
        mid_point = len(monthly_props) // 2
        early_period = monthly_props.iloc[:mid_point]
        late_period = monthly_props.iloc[mid_point:]
        
        early_means = early_period.mean()
        late_means = late_period.mean()
        
        # Calculate change
        topic_changes = late_means - early_means
        
        # Identify emerging (positive change) and declining (negative change) topics
        emerging_threshold = 0.05  # 5% increase
        declining_threshold = -0.05  # 5% decrease
        
        emerging_topics = []
        declining_topics = []
        
        for topic_id, change in topic_changes.items():
            if change > emerging_threshold:
                emerging_topics.append({
                    'topic_id': int(topic_id),
                    'change': change,
                    'early_mean': early_means[topic_id],
                    'late_mean': late_means[topic_id],
                    'top_words': self.topic_words[topic_id]['words'][:5] if topic_id in self.topic_words else []
                })
            elif change < declining_threshold:
                declining_topics.append({
                    'topic_id': int(topic_id),
                    'change': change,
                    'early_mean': early_means[topic_id],
                    'late_mean': late_means[topic_id],
                    'top_words': self.topic_words[topic_id]['words'][:5] if topic_id in self.topic_words else []
                })
        
        # Sort by magnitude of change
        emerging_topics.sort(key=lambda x: x['change'], reverse=True)
        declining_topics.sort(key=lambda x: x['change'])
        
        return {
            'emerging_topics': emerging_topics,
            'declining_topics': declining_topics,
            'stable_topics': len(topic_changes) - len(emerging_topics) - len(declining_topics)
        }
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame, date_column: str) -> Dict[str, Any]:
        """Analyze seasonal patterns in topics"""
        
        # Add time features
        df['month'] = df[date_column].dt.month
        df['quarter'] = df[date_column].dt.quarter
        df['weekday'] = df[date_column].dt.dayofweek
        
        seasonal_patterns = {}
        
        # Monthly patterns
        monthly_topics = df.groupby(['month', 'dominant_topic']).size().unstack(fill_value=0)
        monthly_props = monthly_topics.div(monthly_topics.sum(axis=1), axis=0)
        seasonal_patterns['monthly'] = monthly_props.to_dict()
        
        # Quarterly patterns
        quarterly_topics = df.groupby(['quarter', 'dominant_topic']).size().unstack(fill_value=0)
        quarterly_props = quarterly_topics.div(quarterly_topics.sum(axis=1), axis=0)
        seasonal_patterns['quarterly'] = quarterly_props.to_dict()
        
        # Weekday patterns
        weekday_topics = df.groupby(['weekday', 'dominant_topic']).size().unstack(fill_value=0)
        weekday_props = weekday_topics.div(weekday_topics.sum(axis=1), axis=0)
        seasonal_patterns['weekday'] = weekday_props.to_dict()
        
        return seasonal_patterns
    
    def _calculate_topic_stability(self, monthly_props: pd.DataFrame) -> Dict[str, Any]:
        """Calculate topic stability metrics"""
        
        stability_metrics = {}
        
        for topic_id in monthly_props.columns:
            topic_series = monthly_props[topic_id]
            
            # Coefficient of variation (stability measure)
            cv = topic_series.std() / topic_series.mean() if topic_series.mean() > 0 else np.inf
            
            # Consistency (how often topic appears with non-zero probability)
            consistency = (topic_series > 0).mean()
            
            # Persistence (how often topic maintains similar levels)
            if len(topic_series) > 1:
                changes = topic_series.diff().abs()
                persistence = 1 - changes.mean()
            else:
                persistence = 1.0
            
            stability_metrics[int(topic_id)] = {
                'coefficient_variation': cv,
                'consistency': consistency,
                'persistence': persistence,
                'stability_score': (consistency + persistence) / 2  # Combined metric
            }
        
        return stability_metrics
    
    def visualize_topics(self, save_path: Optional[str] = None) -> Optional[str]:
        """
        Create interactive topic visualization
        
        Args:
            save_path: Path to save visualization HTML
            
        Returns:
            HTML content or None if visualization not available
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_topics() first.")
        
        if not HAS_PYLDAVIS:
            logging.warning("pyLDAvis not available for topic visualization, using basic HTML fallback")
            # Create basic HTML visualization without pyLDAvis
            return self._create_basic_topic_visualization()
        
        try:
            if self.method == 'gensim_lda' and self.gensim_model and HAS_GENSIM:
                # Gensim visualization
                vis = pyLDAvis.gensim_models.prepare(
                    self.gensim_model, 
                    self.gensim_corpus, 
                    self.gensim_dictionary
                )
            
            elif self.method in ['lda', 'nmf'] and self.topic_model:
                # sklearn visualization - pyLDAvis.sklearn is deprecated in newer versions
                # Create a manual visualization for sklearn models
                try:
                    # For sklearn models, we need to create the data manually
                    doc_topic = self.topic_model.transform(self.document_term_matrix)
                    topic_term = self.topic_model.components_
                    doc_lengths = self.document_term_matrix.sum(axis=1).A1
                    vocab = self.vectorizer.get_feature_names_out()
                    term_frequency = self.document_term_matrix.sum(axis=0).A1
                    
                    # Create the pyLDAvis visualization data manually
                    vis = pyLDAvis.prepare(
                        topic_term_dists=topic_term,
                        doc_topic_dists=doc_topic,
                        doc_lengths=doc_lengths,
                        vocab=vocab,
                        term_frequency=term_frequency
                    )
                except Exception as e:
                    logging.warning(f"pyLDAvis visualization failed for sklearn model: {e}")
                    return self._create_basic_topic_visualization()
            else:
                logging.warning("No compatible model for visualization, using basic fallback")
                return self._create_basic_topic_visualization()
            
            if save_path:
                pyLDAvis.save_html(vis, save_path)
                logging.info(f"Topic visualization saved to {save_path}")
            
            return pyLDAvis.prepared_data_to_html(vis)
            
        except Exception as e:
            logging.error(f"Error creating topic visualization: {e}")
            logging.info("Falling back to basic topic visualization")
            return self._create_basic_topic_visualization()
    
    def _create_basic_topic_visualization(self) -> str:
        """Create basic HTML visualization when pyLDAvis is not available"""
        try:
            if not self.is_fitted:
                return "<html><body><h2>Topic Model Not Fitted</h2><p>Please fit the model first.</p></body></html>"
            
            html_content = """
            <html>
            <head>
                <title>NewsBot 2.0 Topic Visualization</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .topic { border: 1px solid #ccc; margin: 10px 0; padding: 10px; border-radius: 5px; }
                    .topic-title { font-weight: bold; font-size: 1.2em; color: #333; }
                    .topic-words { margin: 5px 0; }
                    .word { background: #f0f0f0; padding: 2px 5px; margin: 2px; border-radius: 3px; display: inline-block; }
                </style>
            </head>
            <body>
                <h1>NewsBot 2.0 Topic Analysis</h1>
                <p>Generated {} topics using {} method</p>
            """.format(self.n_topics, self.method.upper())
            
            # Add each topic
            for topic_id in range(self.n_topics):
                if topic_id in self.topic_words:
                    words = self.topic_words[topic_id][:10]  # Top 10 words
                    html_content += f"""
                    <div class="topic">
                        <div class="topic-title">Topic {topic_id + 1}</div>
                        <div class="topic-words">
                    """
                    for word, score in words:
                        html_content += f'<span class="word">{word} ({score:.3f})</span>'
                    html_content += """
                        </div>
                    </div>
                    """
            
            html_content += """
                <hr>
                <p><em>Basic visualization generated by NewsBot 2.0. Install pyLDAvis for interactive visualizations.</em></p>
            </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            logging.error(f"Error creating basic visualization: {e}")
            return f"<html><body><h2>Visualization Error</h2><p>Error: {str(e)}</p></body></html>"
    
    def get_topic_info(self, topic_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific topic"""
        if not self.is_fitted or topic_id not in self.topic_words:
            return {'error': f'Topic {topic_id} not found'}
        
        topic_info = {
            'topic_id': topic_id,
            'top_words': self.topic_words[topic_id]['words'],
            'word_scores': self.topic_words[topic_id]['scores'],
            'word_score_pairs': self.topic_words[topic_id]['word_score_pairs']
        }
        
        # Add documents most representative of this topic
        if self.topic_distributions is not None:
            topic_probs = self.topic_distributions[:, topic_id]
            top_doc_indices = np.argsort(topic_probs)[-5:][::-1]
            
            topic_info['representative_documents'] = {
                'indices': top_doc_indices.tolist(),
                'probabilities': topic_probs[top_doc_indices].tolist()
            }
        
        # Add coherence score if available
        if self.topic_coherence_scores and 'per_topic' in self.topic_coherence_scores:
            if topic_id < len(self.topic_coherence_scores['per_topic']):
                topic_info['coherence_score'] = self.topic_coherence_scores['per_topic'][topic_id]
        
        return topic_info
    
    def save_topic_model(self, filepath: str):
        """Save topic model and components"""
        if not self.is_fitted:
            raise ValueError("No fitted model to save")
        
        model_data = {
            'config': self.config,
            'method': self.method,
            'n_topics': self.n_topics,
            'vectorizer': self.vectorizer,
            'topic_model': self.topic_model,
            'topic_words': self.topic_words,
            'topic_distributions': self.topic_distributions,
            'topic_coherence_scores': self.topic_coherence_scores,
            'training_results': self.training_results,
            'temporal_topics': self.temporal_topics,
            'gensim_model': self.gensim_model if HAS_GENSIM else None,
            'gensim_dictionary': self.gensim_dictionary if HAS_GENSIM else None,
            'is_fitted': self.is_fitted,
            'save_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info(f"Topic model saved to {filepath}")
    
    def load_topic_model(self, filepath: str):
        """Load topic model and components"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.method = model_data['method']
        self.n_topics = model_data['n_topics']
        self.vectorizer = model_data['vectorizer']
        self.topic_model = model_data['topic_model']
        self.topic_words = model_data['topic_words']
        self.topic_distributions = model_data['topic_distributions']
        self.topic_coherence_scores = model_data['topic_coherence_scores']
        self.training_results = model_data['training_results']
        self.temporal_topics = model_data.get('temporal_topics', {})
        
        if HAS_GENSIM:
            self.gensim_model = model_data.get('gensim_model')
            self.gensim_dictionary = model_data.get('gensim_dictionary')
        
        self.is_fitted = model_data['is_fitted']
        
        logging.info(f"Topic model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of topic model"""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        summary = {
            'status': 'fitted',
            'method': self.method,
            'n_topics': self.n_topics,
            'training_results': self.training_results,
            'coherence_scores': self.topic_coherence_scores,
            'topic_words_sample': {}
        }
        
        # Add sample of topic words
        for topic_id in range(min(5, len(self.topic_words))):
            if topic_id in self.topic_words:
                summary['topic_words_sample'][topic_id] = self.topic_words[topic_id]['words'][:5]
        
        return summary