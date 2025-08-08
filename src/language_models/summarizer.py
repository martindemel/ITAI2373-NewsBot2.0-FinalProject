#!/usr/bin/env python3
"""
Intelligent Text Summarizer for NewsBot 2.0
Advanced text summarization with multiple strategies and quality control
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import pickle
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

import nltk
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Try to import transformers for advanced summarization
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("transformers not available. Install for advanced summarization features.")

# Try to import sentence transformers for extractive summarization
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logging.warning("sentence-transformers not available. Install for enhanced extractive summarization.")

class IntelligentSummarizer:
    """
    Advanced text summarization with multiple strategies and quality control
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize intelligent summarizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logging.warning("spaCy model not found. Some features will not be available.")
            self.nlp = None
        
        # Initialize transformer models if available
        self.abstractive_summarizer = None
        self.sentence_transformer = None
        
        if HAS_TRANSFORMERS:
            try:
                # Load abstractive summarization model
                model_name = self.config.get('summarization_model', 'facebook/bart-large-cnn')
                self.abstractive_summarizer = pipeline(
                    "summarization",
                    model=model_name,
                    tokenizer=model_name
                )
                logging.info(f"Loaded abstractive summarization model: {model_name}")
            except Exception as e:
                logging.warning(f"Could not load abstractive summarization model: {e}")
                self.abstractive_summarizer = None
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                # Load sentence transformer for extractive summarization
                model_name = self.config.get('sentence_model', 'all-MiniLM-L6-v2')
                self.sentence_transformer = SentenceTransformer(model_name)
                logging.info(f"Loaded sentence transformer: {model_name}")
            except Exception as e:
                logging.warning(f"Could not load sentence transformer: {e}")
                self.sentence_transformer = None
        
        # Summarization parameters
        self.summary_ratios = {
            'brief': 0.1,      # 10% of original length
            'balanced': 0.25,   # 25% of original length
            'detailed': 0.4     # 40% of original length
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_sentences': 2,
            'max_sentences': 10,
            'min_words': 20,
            'max_words': 200,
            'min_readability': 30,  # Flesch reading ease
            'max_readability': 90
        }
        
        # Statistics tracking
        self.summarization_stats = {
            'total_summaries': 0,
            'avg_compression_ratio': 0,
            'avg_quality_score': 0,
            'method_usage': defaultdict(int)
        }
    
    def summarize_article(self, article_text: str, summary_type: str = 'balanced',
                         method: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate high-quality article summary
        
        Args:
            article_text: Original article text
            summary_type: Type of summary ('brief', 'balanced', 'detailed')
            method: Summarization method ('extractive', 'abstractive', 'hybrid')
            
        Returns:
            Dictionary with summary and metadata
        """
        if not article_text or len(article_text.strip()) < 100:
            return {'error': 'Article too short for summarization'}
        
        # Determine method if not specified
        if method is None:
            if self.abstractive_summarizer:
                method = 'hybrid'
            elif self.sentence_transformer:
                method = 'extractive'
            else:
                method = 'simple_extractive'
        
        # Preprocess text
        preprocessed_text = self._preprocess_text(article_text)
        
        # Generate summary based on method
        if method == 'abstractive' and self.abstractive_summarizer:
            summary_result = self._generate_abstractive_summary(preprocessed_text, summary_type)
        elif method == 'extractive' and self.sentence_transformer:
            summary_result = self._generate_extractive_summary(preprocessed_text, summary_type)
        elif method == 'hybrid':
            summary_result = self._generate_hybrid_summary(preprocessed_text, summary_type)
        else:
            summary_result = self._generate_simple_extractive_summary(preprocessed_text, summary_type)
        
        # Post-process and validate summary
        summary_result = self._post_process_summary(summary_result, article_text)
        
        # Assess quality
        quality_assessment = self._assess_summary_quality(article_text, summary_result['summary'])
        summary_result['quality'] = quality_assessment
        
        # Update statistics
        self._update_stats(summary_result, method)
        
        summary_result.update({
            'original_length': len(article_text),
            'summary_length': len(summary_result['summary']),
            'compression_ratio': len(summary_result['summary']) / len(article_text),
            'method_used': method,
            'summary_type': summary_type,
            'timestamp': datetime.now().isoformat()
        })
        
        return summary_result
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for summarization"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short sentences (likely incomplete)
        sentences = nltk.sent_tokenize(text)
        filtered_sentences = [s for s in sentences if len(s.split()) >= 5]
        
        return ' '.join(filtered_sentences)
    
    def _generate_abstractive_summary(self, text: str, summary_type: str) -> Dict[str, Any]:
        """Generate abstractive summary using transformer model"""
        try:
            # Calculate target length
            target_ratio = self.summary_ratios[summary_type]
            word_count = len(text.split())
            max_length = max(50, int(word_count * target_ratio))
            min_length = max(20, int(max_length * 0.5))
            
            # Handle long texts by chunking
            max_input_length = 1024  # BART limit
            if len(text.split()) > max_input_length:
                chunks = self._chunk_text(text, max_input_length)
                chunk_summaries = []
                
                for chunk in chunks:
                    chunk_summary = self.abstractive_summarizer(
                        chunk,
                        max_length=min(150, max_length // len(chunks)),
                        min_length=min(30, min_length // len(chunks)),
                        do_sample=False,
                        truncation=True
                    )
                    chunk_summaries.append(chunk_summary[0]['summary_text'])
                
                # Combine chunk summaries
                combined_summary = ' '.join(chunk_summaries)
                
                # Summarize the combined summary if still too long
                if len(combined_summary.split()) > max_length:
                    final_summary = self.abstractive_summarizer(
                        combined_summary,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False,
                        truncation=True
                    )[0]['summary_text']
                else:
                    final_summary = combined_summary
            else:
                # Single pass summarization
                summary_result = self.abstractive_summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )
                final_summary = summary_result[0]['summary_text']
            
            return {
                'summary': final_summary,
                'method': 'abstractive',
                'success': True
            }
            
        except Exception as e:
            logging.error(f"Abstractive summarization failed: {e}")
            return {
                'summary': '',
                'method': 'abstractive',
                'success': False,
                'error': str(e)
            }
    
    def _generate_extractive_summary(self, text: str, summary_type: str) -> Dict[str, Any]:
        """Generate extractive summary using sentence embeddings"""
        try:
            # Split into sentences
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) <= 3:
                return {
                    'summary': text,
                    'method': 'extractive',
                    'success': True,
                    'note': 'Text too short for extractive summarization'
                }
            
            # Calculate number of sentences to extract
            target_ratio = self.summary_ratios[summary_type]
            num_sentences = max(2, int(len(sentences) * target_ratio))
            num_sentences = min(num_sentences, len(sentences) - 1)
            
            # Get sentence embeddings
            sentence_embeddings = self.sentence_transformer.encode(sentences)
            
            # Calculate sentence importance scores
            importance_scores = self._calculate_sentence_importance(
                sentences, sentence_embeddings, text
            )
            
            # Select top sentences
            top_sentence_indices = np.argsort(importance_scores)[-num_sentences:]
            top_sentence_indices = sorted(top_sentence_indices)  # Maintain order
            
            # Extract sentences
            summary_sentences = [sentences[i] for i in top_sentence_indices]
            summary = ' '.join(summary_sentences)
            
            return {
                'summary': summary,
                'method': 'extractive',
                'success': True,
                'selected_sentences': len(summary_sentences),
                'importance_scores': importance_scores.tolist()
            }
            
        except Exception as e:
            logging.error(f"Extractive summarization failed: {e}")
            return {
                'summary': '',
                'method': 'extractive',
                'success': False,
                'error': str(e)
            }
    
    def _generate_hybrid_summary(self, text: str, summary_type: str) -> Dict[str, Any]:
        """Generate hybrid summary combining extractive and abstractive methods"""
        # First, use extractive to reduce length
        extractive_result = self._generate_extractive_summary(text, 'detailed')
        
        if not extractive_result['success']:
            # Fall back to abstractive only
            return self._generate_abstractive_summary(text, summary_type)
        
        # Then use abstractive to refine
        extractive_summary = extractive_result['summary']
        
        # Only use abstractive if extractive summary is still long
        if len(extractive_summary.split()) > 100:
            abstractive_result = self._generate_abstractive_summary(extractive_summary, summary_type)
            
            if abstractive_result['success']:
                return {
                    'summary': abstractive_result['summary'],
                    'method': 'hybrid',
                    'success': True,
                    'extractive_step': extractive_result,
                    'abstractive_step': abstractive_result
                }
        
        # Return extractive summary if abstractive step fails or isn't needed
        return {
            'summary': extractive_summary,
            'method': 'hybrid',
            'success': True,
            'extractive_step': extractive_result,
            'note': 'Used extractive only in hybrid approach'
        }
    
    def _generate_simple_extractive_summary(self, text: str, summary_type: str) -> Dict[str, Any]:
        """Generate simple extractive summary using TF-IDF and position scoring"""
        try:
            sentences = nltk.sent_tokenize(text)
            
            if len(sentences) <= 3:
                return {
                    'summary': text,
                    'method': 'simple_extractive',
                    'success': True
                }
            
            # Calculate target number of sentences
            target_ratio = self.summary_ratios[summary_type]
            num_sentences = max(2, int(len(sentences) * target_ratio))
            
            # Score sentences
            sentence_scores = self._score_sentences_simple(sentences, text)
            
            # Select top sentences
            top_indices = sorted(np.argsort(sentence_scores)[-num_sentences:])
            summary_sentences = [sentences[i] for i in top_indices]
            
            return {
                'summary': ' '.join(summary_sentences),
                'method': 'simple_extractive',
                'success': True,
                'sentence_scores': sentence_scores.tolist()
            }
            
        except Exception as e:
            logging.error(f"Simple extractive summarization failed: {e}")
            return {
                'summary': text[:500] + '...' if len(text) > 500 else text,
                'method': 'simple_extractive',
                'success': False,
                'error': str(e)
            }
    
    def _calculate_sentence_importance(self, sentences: List[str], 
                                     embeddings: np.ndarray, text: str) -> np.ndarray:
        """Calculate sentence importance scores using multiple factors"""
        
        # Factor 1: Similarity to document centroid
        doc_centroid = np.mean(embeddings, axis=0)
        centroid_similarities = np.array([
            np.dot(embedding, doc_centroid) / (np.linalg.norm(embedding) * np.linalg.norm(doc_centroid))
            for embedding in embeddings
        ])
        
        # Factor 2: Position in document (earlier sentences more important)
        position_scores = np.array([
            1 - (i / len(sentences)) for i in range(len(sentences))
        ])
        
        # Factor 3: Sentence length (moderate length preferred)
        length_scores = np.array([
            1 - abs(len(sent.split()) - 15) / 30 for sent in sentences
        ])
        length_scores = np.clip(length_scores, 0, 1)
        
        # Factor 4: Named entity density
        entity_scores = np.array([
            self._get_entity_density(sent) for sent in sentences
        ])
        
        # Factor 5: Numerical content (news often contains important numbers)
        number_scores = np.array([
            self._get_number_density(sent) for sent in sentences
        ])
        
        # Combine factors with weights
        weights = {
            'centroid': 0.3,
            'position': 0.2,
            'length': 0.2,
            'entities': 0.2,
            'numbers': 0.1
        }
        
        combined_scores = (
            weights['centroid'] * centroid_similarities +
            weights['position'] * position_scores +
            weights['length'] * length_scores +
            weights['entities'] * entity_scores +
            weights['numbers'] * number_scores
        )
        
        return combined_scores
    
    def _score_sentences_simple(self, sentences: List[str], text: str) -> np.ndarray:
        """Simple sentence scoring using TF-IDF and heuristics"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Calculate TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Sentence scores based on TF-IDF sum
        tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Position scores (earlier sentences more important)
        position_scores = np.array([
            1 - (i / len(sentences)) for i in range(len(sentences))
        ])
        
        # Length scores (prefer moderate length)
        length_scores = np.array([
            min(len(sent.split()) / 20, 1) for sent in sentences
        ])
        
        # Combine scores
        combined_scores = 0.5 * tfidf_scores + 0.3 * position_scores + 0.2 * length_scores
        
        return combined_scores
    
    def _get_entity_density(self, sentence: str) -> float:
        """Calculate named entity density in sentence"""
        if not self.nlp:
            return 0.0
        
        doc = self.nlp(sentence)
        entity_count = len(doc.ents)
        word_count = len([token for token in doc if not token.is_space])
        
        return entity_count / max(word_count, 1)
    
    def _get_number_density(self, sentence: str) -> float:
        """Calculate numerical content density in sentence"""
        words = sentence.split()
        number_words = [word for word in words if any(char.isdigit() for char in word)]
        
        return len(number_words) / max(len(words), 1)
    
    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks of maximum length"""
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _post_process_summary(self, summary_result: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Post-process and validate summary"""
        if not summary_result['success']:
            return summary_result
        
        summary = summary_result['summary']
        
        # Clean up summary
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        # Ensure proper capitalization
        sentences = nltk.sent_tokenize(summary)
        capitalized_sentences = []
        
        for sentence in sentences:
            if sentence and not sentence[0].isupper():
                sentence = sentence[0].upper() + sentence[1:]
            capitalized_sentences.append(sentence)
        
        summary = ' '.join(capitalized_sentences)
        
        # Validate length constraints
        word_count = len(summary.split())
        if word_count < self.quality_thresholds['min_words']:
            # Summary too short, try to extend with more sentences
            if summary_result['method'] == 'extractive' and 'importance_scores' in summary_result:
                # Add one more sentence
                pass  # Implementation would extend summary
        
        summary_result['summary'] = summary
        return summary_result
    
    def _assess_summary_quality(self, original_text: str, summary: str) -> Dict[str, Any]:
        """Assess summary quality using multiple metrics"""
        quality_assessment = {}
        
        # Basic metrics
        original_words = len(original_text.split())
        summary_words = len(summary.split())
        compression_ratio = summary_words / original_words
        
        quality_assessment['compression_ratio'] = compression_ratio
        quality_assessment['word_count'] = summary_words
        quality_assessment['sentence_count'] = len(nltk.sent_tokenize(summary))
        
        # Readability
        try:
            readability_score = flesch_reading_ease(summary)
            quality_assessment['readability_score'] = readability_score
            quality_assessment['readability_level'] = self._classify_readability(readability_score)
        except:
            quality_assessment['readability_score'] = 50  # Default
            quality_assessment['readability_level'] = 'moderate'
        
        # Content preservation (simple keyword overlap)
        original_keywords = self._extract_keywords(original_text)
        summary_keywords = self._extract_keywords(summary)
        
        if original_keywords:
            keyword_overlap = len(set(original_keywords) & set(summary_keywords)) / len(set(original_keywords))
            quality_assessment['keyword_preservation'] = keyword_overlap
        else:
            quality_assessment['keyword_preservation'] = 0.0
        
        # Coherence (sentence transition smoothness)
        coherence_score = self._assess_coherence(summary)
        quality_assessment['coherence_score'] = coherence_score
        
        # Overall quality score
        quality_score = (
            0.3 * min(quality_assessment['keyword_preservation'], 1.0) +
            0.3 * min(quality_assessment['coherence_score'], 1.0) +
            0.2 * min(quality_assessment['readability_score'] / 100, 1.0) +
            0.2 * (1 if 0.1 <= compression_ratio <= 0.5 else 0.5)
        )
        
        quality_assessment['overall_score'] = quality_score
        quality_assessment['quality_grade'] = self._grade_quality(quality_score)
        
        return quality_assessment
    
    def _extract_keywords(self, text: str, top_k: int = 20) -> List[str]:
        """Extract keywords from text using TF-IDF"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Split into sentences for TF-IDF
            sentences = nltk.sent_tokenize(text)
            if len(sentences) < 2:
                return text.split()[:top_k]
            
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=top_k,
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(avg_scores)[-top_k:]
            
            return [feature_names[i] for i in top_indices]
            
        except:
            # Fallback to simple word frequency
            words = nltk.word_tokenize(text.lower())
            words = [w for w in words if w.isalpha() and len(w) > 3]
            word_freq = Counter(words)
            return [word for word, freq in word_freq.most_common(top_k)]
    
    def _assess_coherence(self, summary: str) -> float:
        """Assess summary coherence using simple heuristics"""
        sentences = nltk.sent_tokenize(summary)
        
        if len(sentences) <= 1:
            return 1.0
        
        # Check for smooth transitions between sentences
        transition_words = [
            'however', 'therefore', 'moreover', 'furthermore', 'additionally',
            'meanwhile', 'consequently', 'thus', 'hence', 'also', 'then'
        ]
        
        coherence_factors = []
        
        # Factor 1: Presence of transition words
        transition_count = sum(1 for sent in sentences 
                             for word in transition_words 
                             if word in sent.lower())
        transition_score = min(transition_count / len(sentences), 1.0)
        coherence_factors.append(transition_score)
        
        # Factor 2: Consistent tense usage
        if self.nlp:
            tenses = []
            for sent in sentences:
                doc = self.nlp(sent)
                sent_tenses = [token.tag_ for token in doc if token.tag_.startswith('VB')]
                tenses.extend(sent_tenses)
            
            if tenses:
                most_common_tense = Counter(tenses).most_common(1)[0][0]
                tense_consistency = tenses.count(most_common_tense) / len(tenses)
                coherence_factors.append(tense_consistency)
        
        # Factor 3: Sentence length variation (too much variation indicates poor flow)
        sentence_lengths = [len(sent.split()) for sent in sentences]
        length_std = np.std(sentence_lengths)
        length_mean = np.mean(sentence_lengths)
        length_cv = length_std / length_mean if length_mean > 0 else 1
        length_score = max(0, 1 - length_cv)  # Lower variation is better
        coherence_factors.append(length_score)
        
        return np.mean(coherence_factors) if coherence_factors else 0.5
    
    def _classify_readability(self, score: float) -> str:
        """Classify readability score into categories"""
        if score >= 90:
            return 'very_easy'
        elif score >= 80:
            return 'easy'
        elif score >= 70:
            return 'fairly_easy'
        elif score >= 60:
            return 'standard'
        elif score >= 50:
            return 'fairly_difficult'
        elif score >= 30:
            return 'difficult'
        else:
            return 'very_difficult'
    
    def _grade_quality(self, score: float) -> str:
        """Grade overall quality score"""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def summarize_multiple_articles(self, articles: List[str], focus_topic: Optional[str] = None,
                                  summary_type: str = 'balanced') -> Dict[str, Any]:
        """
        Create unified summary from multiple articles
        
        Args:
            articles: List of article texts
            focus_topic: Optional topic to focus on
            summary_type: Type of summary to generate
            
        Returns:
            Dictionary with multi-document summary
        """
        if not articles:
            return {'error': 'No articles provided'}
        
        if len(articles) == 1:
            return self.summarize_article(articles[0], summary_type)
        
        # Individual summaries
        individual_summaries = []
        for i, article in enumerate(articles):
            summary_result = self.summarize_article(article, 'brief')
            if summary_result.get('success', True) and 'summary' in summary_result:
                individual_summaries.append(summary_result['summary'])
        
        if not individual_summaries:
            return {'error': 'Could not generate any individual summaries'}
        
        # Combine summaries
        combined_text = ' '.join(individual_summaries)
        
        # Focus on topic if specified
        if focus_topic:
            combined_text = self._filter_by_topic(combined_text, focus_topic)
        
        # Generate final summary
        final_summary = self.summarize_article(combined_text, summary_type)
        
        final_summary.update({
            'type': 'multi_document',
            'source_articles': len(articles),
            'individual_summaries': individual_summaries,
            'focus_topic': focus_topic
        })
        
        return final_summary
    
    def _filter_by_topic(self, text: str, topic: str) -> str:
        """Filter text to focus on specific topic"""
        sentences = nltk.sent_tokenize(text)
        topic_words = topic.lower().split()
        
        # Find sentences that mention topic
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in topic_words):
                relevant_sentences.append(sentence)
        
        # If not enough relevant sentences, include some context
        if len(relevant_sentences) < 3:
            # Add sentences before and after relevant ones
            for i, sentence in enumerate(sentences):
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in topic_words):
                    # Add context sentences
                    start_idx = max(0, i - 1)
                    end_idx = min(len(sentences), i + 2)
                    for j in range(start_idx, end_idx):
                        if sentences[j] not in relevant_sentences:
                            relevant_sentences.append(sentences[j])
        
        return ' '.join(relevant_sentences) if relevant_sentences else text
    
    def generate_headlines(self, article_text: str, num_headlines: int = 3) -> Dict[str, Any]:
        """
        Generate compelling headlines for article
        
        Args:
            article_text: Article text
            num_headlines: Number of headlines to generate
            
        Returns:
            Dictionary with generated headlines
        """
        # Extract key information
        sentences = nltk.sent_tokenize(article_text)
        first_sentences = ' '.join(sentences[:3])  # Use first few sentences
        
        # Generate brief summary for headline generation
        brief_summary = self.summarize_article(first_sentences, 'brief')
        
        if not brief_summary.get('success', True):
            return {'error': 'Could not generate summary for headline creation'}
        
        summary_text = brief_summary['summary']
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(summary_text)
        
        # Generate different headline styles
        headlines = []
        
        # Style 1: Question headline
        if '?' not in summary_text:
            question_headline = self._create_question_headline(summary_text, key_phrases)
            if question_headline:
                headlines.append({
                    'headline': question_headline,
                    'style': 'question',
                    'engagement_score': 0.8
                })
        
        # Style 2: Number/statistic headline
        numbers = re.findall(r'\d+', summary_text)
        if numbers:
            number_headline = self._create_number_headline(summary_text, numbers)
            if number_headline:
                headlines.append({
                    'headline': number_headline,
                    'style': 'statistic',
                    'engagement_score': 0.9
                })
        
        # Style 3: Action/impact headline
        action_headline = self._create_action_headline(summary_text, key_phrases)
        if action_headline:
            headlines.append({
                'headline': action_headline,
                'style': 'action',
                'engagement_score': 0.7
            })
        
        # Style 4: Simple informative headline
        if len(headlines) < num_headlines:
            simple_headline = self._create_simple_headline(summary_text)
            headlines.append({
                'headline': simple_headline,
                'style': 'informative',
                'engagement_score': 0.6
            })
        
        # Return top headlines
        headlines.sort(key=lambda x: x['engagement_score'], reverse=True)
        
        return {
            'headlines': headlines[:num_headlines],
            'source_summary': summary_text,
            'key_phrases': key_phrases,
            'generation_timestamp': datetime.now().isoformat()
        }
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        if not self.nlp:
            # Simple fallback
            words = nltk.word_tokenize(text.lower())
            words = [w for w in words if w.isalpha() and len(w) > 3]
            return list(Counter(words).keys())[:10]
        
        doc = self.nlp(text)
        
        # Extract noun phrases
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Extract named entities
        entities = [ent.text for ent in doc.ents]
        
        # Combine and deduplicate
        key_phrases = list(set(noun_phrases + entities))
        
        # Filter by length and frequency
        filtered_phrases = [phrase for phrase in key_phrases 
                          if 2 <= len(phrase.split()) <= 4]
        
        return filtered_phrases[:10]
    
    def _create_question_headline(self, text: str, key_phrases: List[str]) -> Optional[str]:
        """Create question-style headline"""
        # Simple question templates
        if any(phrase.lower() in ['company', 'business', 'corporation'] for phrase in key_phrases):
            return f"What Does This Mean for {key_phrases[0] if key_phrases else 'Business'}?"
        
        if any(phrase.lower() in ['government', 'policy', 'law'] for phrase in key_phrases):
            return f"How Will This Impact {key_phrases[0] if key_phrases else 'Policy'}?"
        
        # Fallback to generic breaking news headline
        words = text.split()[:8]  # First 8 words
        if len(words) >= 3:
            return f"Breaking: {' '.join(words)}..."
        return "Breaking News Update"
    
    def _create_number_headline(self, text: str, numbers: List[str]) -> Optional[str]:
        """Create number/statistic headline"""
        if not numbers:
            # Fallback: Look for any numeric-related content
            words = text.split()[:10]
            for word in words:
                if any(char.isdigit() for char in word):
                    return f"Key Numbers: {' '.join(words[:6])}..."
            return "Statistical Report Released"
        
        first_sentence = nltk.sent_tokenize(text)[0]
        
        # Look for the first number and context
        for number in numbers:
            if number in first_sentence:
                # Extract context around the number
                words = first_sentence.split()
                number_index = None
                for i, word in enumerate(words):
                    if number in word:
                        number_index = i
                        break
                
                if number_index is not None:
                    start = max(0, number_index - 3)
                    end = min(len(words), number_index + 4)
                    context = ' '.join(words[start:end])
                    
                    return f"{number} {context.replace(number, '').strip()}"
        
        # Fallback: Create headline from first sentence
        first_sentence = nltk.sent_tokenize(text)[0]
        words = first_sentence.split()[:8]
        return f"Report: {' '.join(words)}..."
    
    def _create_action_headline(self, text: str, key_phrases: List[str]) -> Optional[str]:
        """Create action/impact headline"""
        # Look for action verbs
        action_verbs = ['announces', 'reveals', 'launches', 'introduces', 'expands', 'develops']
        
        first_sentence = nltk.sent_tokenize(text)[0]
        
        for verb in action_verbs:
            if verb in first_sentence.lower():
                # Build headline around the action
                if key_phrases:
                    return f"{key_phrases[0]} {verb.title()}s Major Initiative"
        
        # Fallback: Create generic action headline
        if key_phrases:
            return f"{key_phrases[0]} Makes Strategic Move"
        
        # Last resort: Use first few words
        words = text.split()[:6]
        return f"Update: {' '.join(words)}..."
    
    def _create_simple_headline(self, text: str) -> str:
        """Create simple informative headline"""
        first_sentence = nltk.sent_tokenize(text)[0]
        
        # Truncate to reasonable headline length
        words = first_sentence.split()
        if len(words) > 10:
            headline = ' '.join(words[:10]) + '...'
        else:
            headline = first_sentence
        
        # Remove ending punctuation
        headline = headline.rstrip('.')
        
        return headline
    
    def _update_stats(self, summary_result: Dict[str, Any], method: str):
        """Update summarization statistics"""
        self.summarization_stats['total_summaries'] += 1
        self.summarization_stats['method_usage'][method] += 1
        
        if 'compression_ratio' in summary_result:
            # Update running average
            total = self.summarization_stats['total_summaries']
            current_avg = self.summarization_stats['avg_compression_ratio']
            new_ratio = summary_result['compression_ratio']
            
            self.summarization_stats['avg_compression_ratio'] = (
                (current_avg * (total - 1) + new_ratio) / total
            )
        
        if 'quality' in summary_result and 'overall_score' in summary_result['quality']:
            # Update quality average
            total = self.summarization_stats['total_summaries']
            current_avg = self.summarization_stats['avg_quality_score']
            new_score = summary_result['quality']['overall_score']
            
            self.summarization_stats['avg_quality_score'] = (
                (current_avg * (total - 1) + new_score) / total
            )
    
    def get_summarization_stats(self) -> Dict[str, Any]:
        """Get summarization statistics"""
        stats = self.summarization_stats.copy()
        stats['method_usage'] = dict(stats['method_usage'])
        return stats
    
    def save_summarizer(self, filepath: str):
        """Save summarizer configuration"""
        config_data = {
            'config': self.config,
            'summary_ratios': self.summary_ratios,
            'quality_thresholds': self.quality_thresholds,
            'summarization_stats': self.summarization_stats,
            'save_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(config_data, f)
        
        logging.info(f"Summarizer saved to {filepath}")
    
    def load_summarizer(self, filepath: str):
        """Load summarizer configuration"""
        with open(filepath, 'rb') as f:
            config_data = pickle.load(f)
        
        self.config = config_data['config']
        self.summary_ratios = config_data['summary_ratios']
        self.quality_thresholds = config_data['quality_thresholds']
        self.summarization_stats = config_data['summarization_stats']
        
        logging.info(f"Summarizer loaded from {filepath}")