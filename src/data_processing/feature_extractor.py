#!/usr/bin/env python3
"""
Enhanced Feature Extractor for NewsBot 2.0
TF-IDF, embeddings, linguistic features, and custom feature engineering
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import Counter
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logging.warning("sentence-transformers not available. Install for embedding features.")

class FeatureExtractor:
    """
    Advanced feature extraction for news articles with multiple feature types
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature extractor
        
        Args:
            config: Configuration dictionary with feature extraction parameters
        """
        self.config = config or {}
        
        # TF-IDF parameters
        self.tfidf_params = self.config.get('tfidf', {
            'max_features': 5000,
            'min_df': 2,
            'max_df': 0.95,
            'ngram_range': (1, 2),
            'use_idf': True,
            'smooth_idf': True,
            'sublinear_tf': True
        })
        
        # Initialize components
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.lda_model = None
        self.nmf_model = None
        self.sentence_transformer = None
        
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logging.warning("spaCy model not found. Some features will not be available.")
            self.nlp = None
        
        # Initialize sentence transformer if available
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
                self.sentence_transformer = SentenceTransformer(model_name)
            except Exception as e:
                logging.warning(f"Could not load sentence transformer: {e}")
                self.sentence_transformer = None
    
    def extract_tfidf_features(self, documents: List[str], fit: bool = True) -> np.ndarray:
        """
        Extract TF-IDF features from documents
        
        Args:
            documents: List of preprocessed documents
            fit: Whether to fit the vectorizer (True for training, False for inference)
            
        Returns:
            TF-IDF feature matrix
        """
        if fit or self.tfidf_vectorizer is None:
            # Adjust parameters based on document count to avoid errors
            adjusted_params = self.tfidf_params.copy()
            num_docs = len(documents)
            
            # Ensure min_df doesn't exceed available documents
            if isinstance(adjusted_params.get('min_df'), int):
                adjusted_params['min_df'] = min(adjusted_params['min_df'], max(1, num_docs // 10))
            
            # Ensure max_df is reasonable
            if isinstance(adjusted_params.get('max_df'), float):
                adjusted_params['max_df'] = min(adjusted_params['max_df'], 0.95)
            
            # Reduce max_features if we have few documents
            if num_docs < 100:
                adjusted_params['max_features'] = min(adjusted_params.get('max_features', 5000), num_docs * 50)
            
            self.tfidf_vectorizer = TfidfVectorizer(**adjusted_params)
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        else:
            if self.tfidf_vectorizer is None:
                # Handle the case where vectorizer is not fitted for single document analysis
                logging.warning("TF-IDF vectorizer not fitted. Creating emergency vectorizer for single document.")
                
                # Create a minimal corpus for fitting
                extended_docs = list(documents)
                # Add some generic news content to create a minimal vocabulary
                extended_docs.extend([
                    "breaking news story report",
                    "latest update information",
                    "article content text",
                    "news development story"
                ])
                
                # Use basic parameters for emergency vectorizer
                emergency_params = {
                    'max_features': 1000,
                    'min_df': 1,
                    'max_df': 1.0,
                    'stop_words': 'english',
                    'lowercase': True,
                    'ngram_range': (1, 1)
                }
                
                self.tfidf_vectorizer = TfidfVectorizer(**emergency_params)
                self.tfidf_vectorizer.fit(extended_docs)
                
            tfidf_matrix = self.tfidf_vectorizer.transform(documents)
        
        return tfidf_matrix
    
    def extract_count_features(self, documents: List[str], fit: bool = True) -> np.ndarray:
        """
        Extract count-based features (for topic modeling)
        
        Args:
            documents: List of preprocessed documents
            fit: Whether to fit the vectorizer
            
        Returns:
            Count feature matrix
        """
        count_params = {
            'max_features': self.config.get('max_features', 1000),
            'min_df': self.config.get('min_df', 2),
            'max_df': self.config.get('max_df', 0.95),
            'stop_words': 'english'
        }
        
        if fit or self.count_vectorizer is None:
            # Adjust parameters based on document count to avoid errors
            num_docs = len(documents)
            
            # Ensure min_df doesn't exceed available documents
            if isinstance(count_params.get('min_df'), int):
                count_params['min_df'] = min(count_params['min_df'], max(1, num_docs // 10))
            
            # Reduce max_features if we have few documents
            if num_docs < 100:
                count_params['max_features'] = min(count_params.get('max_features', 1000), num_docs * 50)
            
            self.count_vectorizer = CountVectorizer(**count_params)
            count_matrix = self.count_vectorizer.fit_transform(documents)
        else:
            if self.count_vectorizer is None:
                raise ValueError("Count vectorizer not fitted. Call with fit=True first.")
            count_matrix = self.count_vectorizer.transform(documents)
        
        return count_matrix
    
    def extract_linguistic_features(self, documents: List[str]) -> pd.DataFrame:
        """
        Extract linguistic features from documents
        
        Args:
            documents: List of original (non-preprocessed) documents
            
        Returns:
            DataFrame with linguistic features
        """
        features = []
        
        for doc in documents:
            doc_features = {}
            
            # Basic text statistics
            doc_features['char_count'] = len(doc)
            doc_features['word_count'] = len(doc.split())
            doc_features['sentence_count'] = len(nltk.sent_tokenize(doc))
            doc_features['avg_word_length'] = np.mean([len(word) for word in doc.split()])
            doc_features['avg_sentence_length'] = doc_features['word_count'] / max(doc_features['sentence_count'], 1)
            
            # Readability scores
            try:
                doc_features['flesch_reading_ease'] = flesch_reading_ease(doc)
                doc_features['flesch_kincaid_grade'] = flesch_kincaid_grade(doc)
            except:
                doc_features['flesch_reading_ease'] = 0
                doc_features['flesch_kincaid_grade'] = 0
            
            # Punctuation and capitalization features
            doc_features['exclamation_count'] = doc.count('!')
            doc_features['question_count'] = doc.count('?')
            doc_features['uppercase_ratio'] = sum(1 for c in doc if c.isupper()) / max(len(doc), 1)
            doc_features['digit_ratio'] = sum(1 for c in doc if c.isdigit()) / max(len(doc), 1)
            
            # spaCy-based features
            if self.nlp:
                nlp_doc = self.nlp(doc)
                
                # POS tag distribution
                pos_counts = Counter([token.pos_ for token in nlp_doc])
                total_tokens = len(nlp_doc)
                
                doc_features['noun_ratio'] = pos_counts.get('NOUN', 0) / max(total_tokens, 1)
                doc_features['verb_ratio'] = pos_counts.get('VERB', 0) / max(total_tokens, 1)
                doc_features['adj_ratio'] = pos_counts.get('ADJ', 0) / max(total_tokens, 1)
                doc_features['adv_ratio'] = pos_counts.get('ADV', 0) / max(total_tokens, 1)
                
                # Named entity features
                entities = [ent.label_ for ent in nlp_doc.ents]
                entity_counts = Counter(entities)
                
                doc_features['person_count'] = entity_counts.get('PERSON', 0)
                doc_features['org_count'] = entity_counts.get('ORG', 0)
                doc_features['gpe_count'] = entity_counts.get('GPE', 0)  # Geopolitical entities
                doc_features['date_count'] = entity_counts.get('DATE', 0)
                doc_features['money_count'] = entity_counts.get('MONEY', 0)
                
                # Dependency features
                doc_features['unique_deps'] = len(set([token.dep_ for token in nlp_doc]))
                
            features.append(doc_features)
        
        return pd.DataFrame(features)
    
    def extract_semantic_features(self, documents: List[str]) -> Optional[np.ndarray]:
        """
        Extract semantic embeddings using sentence transformers
        
        Args:
            documents: List of documents
            
        Returns:
            Semantic embedding matrix or None if not available
        """
        if not self.sentence_transformer:
            logging.warning("Sentence transformer not available, using fallback method")
            # Fallback to TF-IDF based semantic features
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.decomposition import TruncatedSVD
                
                # Create TF-IDF features
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform(documents)
                
                # Reduce dimensionality to simulate embeddings
                svd = TruncatedSVD(n_components=384, random_state=42)
                embeddings = svd.fit_transform(tfidf_matrix)
                
                logging.info(f"Generated semantic features using TF-IDF + SVD: {embeddings.shape}")
                return embeddings
                
            except Exception as fallback_error:
                logging.error(f"Fallback semantic features failed: {fallback_error}")
                # Return basic statistical features as last resort
                features = []
                for doc in documents:
                    doc_features = [
                        len(doc.split()),  # word count
                        len(doc),  # character count
                        len(set(doc.lower().split())),  # unique words
                        doc.count('.'),  # sentence count approximation
                    ]
                    # Pad to 384 dimensions
                    doc_features.extend([0.0] * (384 - len(doc_features)))
                    features.append(doc_features)
                return np.array(features)
        
        try:
            embeddings = self.sentence_transformer.encode(documents, show_progress_bar=True)
            return embeddings
        except Exception as e:
            logging.error(f"Error extracting semantic features with sentence transformer: {e}")
            # Use fallback method
            return self._fallback_semantic_features(documents)
    
    def extract_topic_features(self, documents: List[str], n_topics: int = 10, 
                             method: str = 'lda', fit: bool = True) -> np.ndarray:
        """
        Extract topic modeling features
        
        Args:
            documents: List of preprocessed documents
            n_topics: Number of topics to extract
            method: Topic modeling method ('lda' or 'nmf')
            fit: Whether to fit the model
            
        Returns:
            Topic distribution matrix
        """
        # Get count features for topic modeling
        count_matrix = self.extract_count_features(documents, fit=fit)
        
        if method == 'lda':
            if fit or self.lda_model is None:
                self.lda_model = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    learning_method='online',
                    max_iter=20
                )
                topic_matrix = self.lda_model.fit_transform(count_matrix)
            else:
                topic_matrix = self.lda_model.transform(count_matrix)
                
        elif method == 'nmf':
            if fit or self.nmf_model is None:
                self.nmf_model = NMF(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=200
                )
                topic_matrix = self.nmf_model.fit_transform(count_matrix)
            else:
                topic_matrix = self.nmf_model.transform(count_matrix)
        else:
            raise ValueError(f"Unknown topic modeling method: {method}")
        
        return topic_matrix
    
    def extract_similarity_features(self, documents: List[str], reference_docs: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract similarity features based on TF-IDF cosine similarity
        
        Args:
            documents: List of documents to compute features for
            reference_docs: Reference documents for similarity computation
            
        Returns:
            Similarity feature matrix
        """
        if reference_docs is None:
            reference_docs = documents
        
        # Get TF-IDF features
        all_docs = reference_docs + documents
        tfidf_matrix = self.extract_tfidf_features(all_docs, fit=True)
        
        # Split matrices
        ref_matrix = tfidf_matrix[:len(reference_docs)]
        doc_matrix = tfidf_matrix[len(reference_docs):]
        
        # Compute similarities
        similarity_matrix = cosine_similarity(doc_matrix, ref_matrix)
        
        # Extract statistical features from similarities
        similarity_features = []
        for row in similarity_matrix:
            features = {
                'max_similarity': np.max(row),
                'mean_similarity': np.mean(row),
                'std_similarity': np.std(row),
                'median_similarity': np.median(row)
            }
            similarity_features.append(list(features.values()))
        
        return np.array(similarity_features)
    
    def extract_custom_news_features(self, documents: List[str]) -> pd.DataFrame:
        """
        Extract custom features specific to news articles
        
        Args:
            documents: List of original documents
            
        Returns:
            DataFrame with custom news features
        """
        features = []
        
        # Define news-specific keywords
        urgency_words = ['breaking', 'urgent', 'alert', 'emergency', 'crisis', 'immediate']
        opinion_words = ['believe', 'think', 'opinion', 'view', 'perspective', 'argue']
        factual_words = ['according', 'reported', 'stated', 'confirmed', 'data', 'statistics']
        time_words = ['today', 'yesterday', 'tomorrow', 'recently', 'now', 'current']
        
        for doc in documents:
            doc_lower = doc.lower()
            word_count = len(doc.split())
            
            doc_features = {}
            
            # News-specific feature counts
            doc_features['urgency_score'] = sum(doc_lower.count(word) for word in urgency_words)
            doc_features['opinion_score'] = sum(doc_lower.count(word) for word in opinion_words)
            doc_features['factual_score'] = sum(doc_lower.count(word) for word in factual_words)
            doc_features['time_score'] = sum(doc_lower.count(word) for word in time_words)
            
            # Normalize by word count
            doc_features['urgency_ratio'] = doc_features['urgency_score'] / max(word_count, 1)
            doc_features['opinion_ratio'] = doc_features['opinion_score'] / max(word_count, 1)
            doc_features['factual_ratio'] = doc_features['factual_score'] / max(word_count, 1)
            doc_features['time_ratio'] = doc_features['time_score'] / max(word_count, 1)
            
            # Quote detection
            doc_features['quote_count'] = doc.count('"') // 2  # Pairs of quotes
            doc_features['quote_ratio'] = doc_features['quote_count'] / max(word_count, 1)
            
            # Number and percentage detection
            doc_features['number_count'] = len([word for word in doc.split() if any(char.isdigit() for char in word)])
            doc_features['percentage_count'] = doc_lower.count('%') + doc_lower.count('percent')
            
            features.append(doc_features)
        
        return pd.DataFrame(features)
    
    def extract_all_features(self, documents: List[str], original_documents: Optional[List[str]] = None,
                           include_embeddings: bool = True, include_topics: bool = True) -> Dict[str, Any]:
        """
        Extract all available features from documents
        
        Args:
            documents: List of preprocessed documents
            original_documents: List of original documents (for linguistic features)
            include_embeddings: Whether to include semantic embeddings
            include_topics: Whether to include topic features
            
        Returns:
            Dictionary containing all feature matrices and metadata
        """
        if original_documents is None:
            original_documents = documents
        
        features = {}
        
        # TF-IDF features
        logging.info("Extracting TF-IDF features...")
        features['tfidf'] = self.extract_tfidf_features(documents)
        features['tfidf_feature_names'] = self.tfidf_vectorizer.get_feature_names_out()
        
        # Linguistic features
        logging.info("Extracting linguistic features...")
        features['linguistic'] = self.extract_linguistic_features(original_documents)
        
        # Custom news features
        logging.info("Extracting custom news features...")
        features['custom_news'] = self.extract_custom_news_features(original_documents)
        
        # Semantic embeddings
        if include_embeddings:
            logging.info("Extracting semantic embeddings...")
            embeddings = self.extract_semantic_features(original_documents)
            if embeddings is not None:
                features['embeddings'] = embeddings
        
        # Topic features
        if include_topics:
            logging.info("Extracting topic features...")
            features['topics_lda'] = self.extract_topic_features(documents, method='lda')
            features['topics_nmf'] = self.extract_topic_features(documents, method='nmf')
        
        # Similarity features
        logging.info("Extracting similarity features...")
        features['similarity'] = self.extract_similarity_features(documents)
        
        # Combine numerical features
        numerical_features = []
        feature_names = []
        
        # Add linguistic features
        ling_features = features['linguistic'].values
        numerical_features.append(ling_features)
        feature_names.extend([f"ling_{col}" for col in features['linguistic'].columns])
        
        # Add custom news features
        news_features = features['custom_news'].values
        numerical_features.append(news_features)
        feature_names.extend([f"news_{col}" for col in features['custom_news'].columns])
        
        # Add topic features
        if include_topics:
            numerical_features.append(features['topics_lda'])
            feature_names.extend([f"topic_lda_{i}" for i in range(features['topics_lda'].shape[1])])
            
            numerical_features.append(features['topics_nmf'])
            feature_names.extend([f"topic_nmf_{i}" for i in range(features['topics_nmf'].shape[1])])
        
        # Add similarity features
        numerical_features.append(features['similarity'])
        feature_names.extend(['max_sim', 'mean_sim', 'std_sim', 'median_sim'])
        
        # Add embeddings if available
        if include_embeddings and 'embeddings' in features:
            numerical_features.append(features['embeddings'])
            feature_names.extend([f"embed_{i}" for i in range(features['embeddings'].shape[1])])
        
        # Combine all numerical features
        features['combined_numerical'] = np.hstack(numerical_features)
        features['numerical_feature_names'] = feature_names
        
        return features
    
    def save_extractors(self, filepath: str):
        """Save trained feature extractors"""
        extractors = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'count_vectorizer': self.count_vectorizer,
            'lda_model': self.lda_model,
            'nmf_model': self.nmf_model,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(extractors, f)
        
        logging.info(f"Feature extractors saved to {filepath}")
    
    def load_extractors(self, filepath: str):
        """Load trained feature extractors"""
        with open(filepath, 'rb') as f:
            extractors = pickle.load(f)
        
        self.tfidf_vectorizer = extractors['tfidf_vectorizer']
        self.count_vectorizer = extractors['count_vectorizer']
        self.lda_model = extractors['lda_model']
        self.nmf_model = extractors['nmf_model']
        self.config = extractors['config']
        
        logging.info(f"Feature extractors loaded from {filepath}")
    
    def get_feature_importance(self, feature_matrix: np.ndarray, labels: List[str], 
                             feature_names: List[str], top_k: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance using simple statistical measures
        
        Args:
            feature_matrix: Feature matrix
            labels: Class labels
            feature_names: Names of features
            top_k: Number of top features to return per class
            
        Returns:
            Dictionary of top features per class
        """
        from sklearn.feature_selection import chi2
        
        try:
            # Calculate chi-squared statistics
            chi2_stats, _ = chi2(feature_matrix, labels)
            
            # Get top features
            top_indices = np.argsort(chi2_stats)[-top_k:][::-1]
            
            importance_dict = {}
            for label in set(labels):
                top_features = [(feature_names[i], chi2_stats[i]) for i in top_indices]
                importance_dict[label] = top_features
            
            return importance_dict
            
        except Exception as e:
            logging.error(f"Error calculating feature importance: {e}")
            # Return basic feature importance based on frequency
            try:
                from collections import Counter
                import re
                
                # Simple word frequency based importance
                importance_dict = {}
                for label in self.label_encoder.classes_:
                    # Get texts for this label
                    label_texts = [texts[i] for i, l in enumerate(labels) if l == label]
                    all_text = ' '.join(label_texts).lower()
                    
                    # Extract words and count frequency
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
                    word_freq = Counter(words)
                    
                    # Get top words as important features
                    top_words = word_freq.most_common(n_features)
                    importance_dict[label] = [(word, freq/len(words)) for word, freq in top_words]
                
                logging.info("Generated basic feature importance using word frequency")
                return importance_dict
                
            except Exception as fallback_error:
                logging.error(f"Fallback feature importance failed: {fallback_error}")
                # Return empty structure with explanation
                return {
                    'error': 'Feature importance calculation failed',
                    'fallback_attempted': True,
                    'message': 'Please check if model is properly trained'
                }