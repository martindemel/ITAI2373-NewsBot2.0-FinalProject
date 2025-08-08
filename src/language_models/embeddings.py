#!/usr/bin/env python3
"""
Semantic Embeddings for NewsBot 2.0
Advanced semantic understanding and similarity matching using embeddings
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logging.warning("sentence-transformers not available. Install for semantic embeddings.")

# Try to import faiss for efficient similarity search
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logging.warning("faiss not available. Install for efficient similarity search.")

class SemanticEmbeddings:
    """
    Advanced semantic embeddings for understanding and searching news content
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize semantic embeddings system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Model configuration
        self.model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.embedding_dim = None
        
        # Initialize sentence transformer
        self.sentence_transformer = None
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.sentence_transformer = SentenceTransformer(self.model_name)
                self.embedding_dim = self.sentence_transformer.get_sentence_embedding_dimension()
                logging.info(f"Loaded sentence transformer: {self.model_name} (dim: {self.embedding_dim})")
            except Exception as e:
                logging.error(f"Could not load sentence transformer: {e}")
                self.sentence_transformer = None
        
        # Storage for embeddings
        self.document_embeddings = None
        self.document_texts = None
        self.document_metadata = None
        
        # FAISS index for efficient search
        self.faiss_index = None
        
        # Clustering results
        self.clusters = None
        self.cluster_labels = None
        self.cluster_centers = None
        
        # Query cache for performance
        self.query_cache = {}
        self.cache_size_limit = 1000
        
        # Statistics
        self.embedding_stats = {
            'total_documents_embedded': 0,
            'total_queries_processed': 0,
            'avg_embedding_time': 0,
            'avg_search_time': 0
        }
    
    def encode_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None,
                        batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Convert documents to semantic embeddings
        
        Args:
            documents: List of documents to embed
            metadata: Optional metadata for each document
            batch_size: Batch size for encoding
            show_progress: Whether to show progress
            
        Returns:
            Document embedding matrix
        """
        if not self.sentence_transformer:
            raise ValueError("Sentence transformer not available. Install sentence-transformers.")
        
        logging.info(f"Encoding {len(documents)} documents to embeddings...")
        
        start_time = datetime.now()
        
        # Encode documents in batches
        embeddings = self.sentence_transformer.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Store embeddings and documents
        self.document_embeddings = embeddings
        self.document_texts = documents
        self.document_metadata = metadata or [{}] * len(documents)
        
        # Build FAISS index for efficient search
        if HAS_FAISS:
            self._build_faiss_index(embeddings)
        
        # Update statistics
        end_time = datetime.now()
        embedding_time = (end_time - start_time).total_seconds()
        
        self.embedding_stats['total_documents_embedded'] = len(documents)
        self.embedding_stats['avg_embedding_time'] = embedding_time / len(documents)
        
        logging.info(f"Document encoding completed in {embedding_time:.2f} seconds")
        
        return embeddings
    
    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index for efficient similarity search"""
        try:
            # Normalize embeddings for cosine similarity
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self.faiss_index.add(normalized_embeddings.astype('float32'))
            
            logging.info(f"Built FAISS index with {embeddings.shape[0]} vectors")
            
        except Exception as e:
            logging.warning(f"Could not build FAISS index: {e}")
            self.faiss_index = None
    
    def find_similar_articles(self, query_article: str, top_k: int = 5,
                            threshold: float = 0.0) -> Dict[str, Any]:
        """
        Find semantically similar articles
        
        Args:
            query_article: Article text to find similarities for
            top_k: Number of similar articles to return
            threshold: Minimum similarity threshold
            
        Returns:
            Dictionary with similar articles and scores
        """
        if self.document_embeddings is None:
            raise ValueError("No documents embedded. Call encode_documents() first.")
        
        start_time = datetime.now()
        
        # Check cache
        cache_key = f"{hash(query_article)}_{top_k}_{threshold}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Encode query
        query_embedding = self.sentence_transformer.encode([query_article])
        
        # Find similarities
        if self.faiss_index and HAS_FAISS:
            similarities, indices = self._search_faiss(query_embedding, top_k)
        else:
            similarities, indices = self._search_brute_force(query_embedding, top_k)
        
        # Filter by threshold
        valid_results = [(sim, idx) for sim, idx in zip(similarities, indices) if sim >= threshold]
        
        # Prepare results
        results = []
        for similarity, doc_idx in valid_results:
            result = {
                'document_index': int(doc_idx),
                'similarity_score': float(similarity),
                'document_text': self.document_texts[doc_idx],
                'metadata': self.document_metadata[doc_idx]
            }
            results.append(result)
        
        # Update statistics
        end_time = datetime.now()
        search_time = (end_time - start_time).total_seconds()
        self.embedding_stats['total_queries_processed'] += 1
        
        # Update average search time
        total_queries = self.embedding_stats['total_queries_processed']
        current_avg = self.embedding_stats['avg_search_time']
        self.embedding_stats['avg_search_time'] = (
            (current_avg * (total_queries - 1) + search_time) / total_queries
        )
        
        result_dict = {
            'query_article': query_article,
            'similar_articles': results,
            'search_time': search_time,
            'total_candidates': len(self.document_texts),
            'results_returned': len(results),
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache result
        if len(self.query_cache) < self.cache_size_limit:
            self.query_cache[cache_key] = result_dict
        
        return result_dict
    
    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS index"""
        # Normalize query embedding
        normalized_query = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        similarities, indices = self.faiss_index.search(normalized_query.astype('float32'), top_k)
        
        return similarities[0], indices[0]
    
    def _search_brute_force(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search using brute force cosine similarity"""
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        
        return top_similarities, top_indices
    
    def semantic_search(self, query_text: str, article_database: Optional[List[str]] = None,
                       top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search articles using natural language queries
        
        Args:
            query_text: Natural language search query
            article_database: Optional specific database to search
            top_k: Number of results to return
            filters: Optional filters for metadata
            
        Returns:
            Dictionary with search results
        """
        if article_database is not None:
            # Temporary encoding for provided database
            temp_embeddings = self.sentence_transformer.encode(article_database)
            search_embeddings = temp_embeddings
            search_texts = article_database
            search_metadata = [{}] * len(article_database)
        else:
            # Use stored embeddings
            if self.document_embeddings is None:
                raise ValueError("No documents embedded. Call encode_documents() or provide article_database.")
            search_embeddings = self.document_embeddings
            search_texts = self.document_texts
            search_metadata = self.document_metadata
        
        # Encode query
        query_embedding = self.sentence_transformer.encode([query_text])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, search_embeddings)[0]
        
        # Apply filters if provided
        if filters:
            filtered_indices = self._apply_filters(search_metadata, filters)
            # Set similarities to 0 for filtered out documents
            mask = np.zeros(len(similarities), dtype=bool)
            mask[filtered_indices] = True
            similarities = similarities * mask
        
        # Get top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include positive similarities
                result = {
                    'document_index': int(idx),
                    'similarity_score': float(similarities[idx]),
                    'document_text': search_texts[idx],
                    'metadata': search_metadata[idx],
                    'query_relevance': self._calculate_query_relevance(query_text, search_texts[idx])
                }
                results.append(result)
        
        return {
            'query': query_text,
            'results': results,
            'total_searched': len(search_texts),
            'filters_applied': filters,
            'timestamp': datetime.now().isoformat()
        }
    
    def _apply_filters(self, metadata_list: List[Dict[str, Any]], 
                      filters: Dict[str, Any]) -> List[int]:
        """Apply metadata filters and return valid indices"""
        valid_indices = []
        
        for i, metadata in enumerate(metadata_list):
            is_valid = True
            
            for filter_key, filter_value in filters.items():
                if filter_key in metadata:
                    if isinstance(filter_value, list):
                        # Multiple acceptable values
                        if metadata[filter_key] not in filter_value:
                            is_valid = False
                            break
                    else:
                        # Single value
                        if metadata[filter_key] != filter_value:
                            is_valid = False
                            break
                else:
                    # Missing required filter key
                    is_valid = False
                    break
            
            if is_valid:
                valid_indices.append(i)
        
        return valid_indices
    
    def _calculate_query_relevance(self, query: str, document: str) -> float:
        """Calculate additional query relevance beyond semantic similarity"""
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        # Keyword overlap
        overlap = len(query_words.intersection(doc_words))
        query_relevance = overlap / len(query_words) if query_words else 0
        
        return query_relevance
    
    def cluster_similar_content(self, n_clusters: Optional[int] = None,
                               method: str = 'kmeans') -> Dict[str, Any]:
        """
        Group articles by semantic similarity
        
        Args:
            n_clusters: Number of clusters (auto-determined if None)
            method: Clustering method ('kmeans', 'hierarchical')
            
        Returns:
            Dictionary with clustering results
        """
        if self.document_embeddings is None:
            raise ValueError("No documents embedded. Call encode_documents() first.")
        
        logging.info(f"Clustering {len(self.document_embeddings)} documents...")
        
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = self._determine_optimal_clusters()
        
        # Perform clustering
        if method == 'kmeans':
            clustering_result = self._cluster_kmeans(n_clusters)
        else:
            raise ValueError(f"Clustering method '{method}' not supported")
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(clustering_result['labels'])
        
        # Store results
        self.clusters = clustering_result
        self.cluster_labels = clustering_result['labels']
        self.cluster_centers = clustering_result.get('centers')
        
        return {
            'method': method,
            'n_clusters': n_clusters,
            'cluster_labels': clustering_result['labels'].tolist(),
            'cluster_centers': clustering_result.get('centers'),
            'cluster_analysis': cluster_analysis,
            'clustering_timestamp': datetime.now().isoformat()
        }
    
    def _determine_optimal_clusters(self, max_clusters: int = 20) -> int:
        """Determine optimal number of clusters using elbow method"""
        if len(self.document_embeddings) < 10:
            return min(3, len(self.document_embeddings))
        
        # Try different numbers of clusters
        max_k = min(max_clusters, len(self.document_embeddings) // 2)
        inertias = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.document_embeddings)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point (simplified)
        if len(inertias) < 3:
            return 3
        
        # Calculate rate of change
        rate_changes = []
        for i in range(1, len(inertias)):
            rate_change = inertias[i-1] - inertias[i]
            rate_changes.append(rate_change)
        
        # Find point where rate of change starts to level off
        if len(rate_changes) >= 2:
            for i in range(1, len(rate_changes)):
                if rate_changes[i] < rate_changes[i-1] * 0.5:  # Significant drop in improvement
                    return i + 2  # +2 because we started from k=2
        
        # Default fallback
        return min(8, len(self.document_embeddings) // 3)
    
    def _cluster_kmeans(self, n_clusters: int) -> Dict[str, Any]:
        """Perform K-means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.document_embeddings)
        
        return {
            'labels': cluster_labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_
        }
    
    def _analyze_clusters(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze clustering results"""
        analysis = {}
        
        # Cluster sizes
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels.astype(int), counts.astype(int)))
        analysis['cluster_sizes'] = cluster_sizes
        
        # Cluster statistics
        analysis['num_clusters'] = len(unique_labels)
        analysis['largest_cluster_size'] = int(counts.max())
        analysis['smallest_cluster_size'] = int(counts.min())
        analysis['avg_cluster_size'] = float(counts.mean())
        
        # Balance metric (how evenly distributed clusters are)
        ideal_size = len(cluster_labels) / len(unique_labels)
        balance_score = 1 - np.std(counts) / ideal_size
        analysis['cluster_balance'] = float(max(0, balance_score))
        
        # Cluster coherence (average intra-cluster similarity)
        if self.document_embeddings is not None:
            coherence_scores = []
            for label in unique_labels:
                cluster_mask = cluster_labels == label
                cluster_embeddings = self.document_embeddings[cluster_mask]
                
                if len(cluster_embeddings) > 1:
                    # Calculate pairwise similarities within cluster
                    similarities = cosine_similarity(cluster_embeddings)
                    # Get upper triangle (excluding diagonal)
                    upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
                    coherence = np.mean(upper_triangle)
                    coherence_scores.append(coherence)
            
            if coherence_scores:
                analysis['avg_cluster_coherence'] = float(np.mean(coherence_scores))
                analysis['coherence_scores'] = [float(score) for score in coherence_scores]
        
        return analysis
    
    def get_cluster_topics(self, cluster_id: int, top_words: int = 10) -> Dict[str, Any]:
        """
        Get representative topics/words for a cluster
        
        Args:
            cluster_id: ID of the cluster
            top_words: Number of top words to return
            
        Returns:
            Dictionary with cluster topic information
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering performed. Call cluster_similar_content() first.")
        
        # Get documents in this cluster
        cluster_mask = self.cluster_labels == cluster_id
        cluster_documents = [self.document_texts[i] for i in range(len(self.document_texts)) if cluster_mask[i]]
        
        if not cluster_documents:
            return {'error': f'Cluster {cluster_id} not found or empty'}
        
        # Extract keywords using TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=top_words * 2,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(cluster_documents)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(avg_scores)[-top_words:][::-1]
            
            cluster_topics = {
                'cluster_id': cluster_id,
                'num_documents': len(cluster_documents),
                'top_words': [feature_names[i] for i in top_indices],
                'word_scores': [float(avg_scores[i]) for i in top_indices],
                'sample_documents': cluster_documents[:3]  # First 3 documents as examples
            }
            
            return cluster_topics
            
        except Exception as e:
            logging.error(f"Error extracting cluster topics: {e}")
            return {
                'cluster_id': cluster_id,
                'num_documents': len(cluster_documents),
                'error': str(e),
                'sample_documents': cluster_documents[:3]
            }
    
    def visualize_embeddings(self, method: str = 'pca', n_components: int = 2,
                           include_clusters: bool = True) -> Dict[str, Any]:
        """
        Create 2D visualization of document embeddings
        
        Args:
            method: Dimensionality reduction method ('pca', 'tsne')
            n_components: Number of dimensions for visualization
            include_clusters: Whether to include cluster information
            
        Returns:
            Dictionary with visualization data
        """
        if self.document_embeddings is None:
            raise ValueError("No documents embedded. Call encode_documents() first.")
        
        logging.info(f"Creating {method.upper()} visualization...")
        
        # Perform dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(self.document_embeddings)-1))
        else:
            raise ValueError(f"Method '{method}' not supported. Use 'pca' or 'tsne'.")
        
        reduced_embeddings = reducer.fit_transform(self.document_embeddings)
        
        # Prepare visualization data
        viz_data = {
            'method': method,
            'coordinates': reduced_embeddings.tolist(),
            'document_texts': self.document_texts,
            'metadata': self.document_metadata
        }
        
        # Add cluster information if available
        if include_clusters and self.cluster_labels is not None:
            viz_data['cluster_labels'] = self.cluster_labels.tolist()
            viz_data['has_clusters'] = True
        else:
            viz_data['has_clusters'] = False
        
        # Add explained variance for PCA
        if method == 'pca':
            viz_data['explained_variance_ratio'] = reducer.explained_variance_ratio_.tolist()
            viz_data['total_explained_variance'] = float(reducer.explained_variance_ratio_.sum())
        
        return viz_data
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding system statistics"""
        stats = self.embedding_stats.copy()
        
        # Add model information
        stats['model_name'] = self.model_name
        stats['embedding_dimension'] = self.embedding_dim
        stats['has_faiss_index'] = self.faiss_index is not None
        stats['cache_size'] = len(self.query_cache)
        
        # Add data information
        if self.document_embeddings is not None:
            stats['documents_in_memory'] = len(self.document_embeddings)
            stats['embedding_matrix_shape'] = self.document_embeddings.shape
        
        # Add clustering information
        if self.cluster_labels is not None:
            stats['num_clusters'] = len(np.unique(self.cluster_labels))
            stats['clustering_available'] = True
        else:
            stats['clustering_available'] = False
        
        return stats
    
    def save_embeddings(self, filepath: str):
        """Save embeddings and system state"""
        save_data = {
            'config': self.config,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'document_embeddings': self.document_embeddings,
            'document_texts': self.document_texts,
            'document_metadata': self.document_metadata,
            'cluster_labels': self.cluster_labels,
            'cluster_centers': self.cluster_centers,
            'embedding_stats': self.embedding_stats,
            'save_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logging.info(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str):
        """Load embeddings and system state"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.config = save_data['config']
        self.model_name = save_data['model_name']
        self.embedding_dim = save_data['embedding_dim']
        self.document_embeddings = save_data['document_embeddings']
        self.document_texts = save_data['document_texts']
        self.document_metadata = save_data['document_metadata']
        self.cluster_labels = save_data.get('cluster_labels')
        self.cluster_centers = save_data.get('cluster_centers')
        self.embedding_stats = save_data['embedding_stats']
        
        # Rebuild FAISS index if embeddings are loaded
        if self.document_embeddings is not None and HAS_FAISS:
            self._build_faiss_index(self.document_embeddings)
        
        logging.info(f"Embeddings loaded from {filepath}")
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logging.info("Query cache cleared")