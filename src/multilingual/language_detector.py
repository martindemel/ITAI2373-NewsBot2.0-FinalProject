#!/usr/bin/env python3
"""
Language Detector for NewsBot 2.0
Automatic language identification with confidence scoring and regional variants
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from collections import Counter, defaultdict
import re
import pickle

# Core language detection
try:
    from langdetect import detect, detect_langs, DetectorFactory, LangDetectException
    HAS_LANGDETECT = True
    # Set seed for reproducible results
    DetectorFactory.seed = 0
except ImportError:
    HAS_LANGDETECT = False
    logging.warning("langdetect not available. Install for language detection features.")

# Advanced language detection using transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("transformers not available for advanced language detection.")

class LanguageDetector:
    """
    Advanced language detection with confidence scoring and cultural context
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize language detector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Supported languages with regional variants
        self.supported_languages = {
            'en': {
                'name': 'English',
                'regions': ['US', 'UK', 'AU', 'CA', 'IN', 'ZA'],
                'cultural_markers': ['dollar', 'pound', 'cricket', 'football', 'soccer', 'rugby']
            },
            'es': {
                'name': 'Spanish',
                'regions': ['ES', 'MX', 'AR', 'CO', 'PE', 'VE', 'CL'],
                'cultural_markers': ['peso', 'euro', 'futbol', 'siesta', 'tapas']
            },
            'fr': {
                'name': 'French',
                'regions': ['FR', 'CA', 'BE', 'CH', 'MA', 'SN'],
                'cultural_markers': ['euro', 'baguette', 'champagne', 'quebec']
            },
            'de': {
                'name': 'German',
                'regions': ['DE', 'AT', 'CH'],
                'cultural_markers': ['euro', 'oktoberfest', 'bundesliga', 'autobahn']
            },
            'it': {
                'name': 'Italian',
                'regions': ['IT', 'CH', 'SM', 'VA'],
                'cultural_markers': ['euro', 'pasta', 'pizza', 'vatican', 'serie_a']
            },
            'pt': {
                'name': 'Portuguese',
                'regions': ['PT', 'BR', 'AO', 'MZ'],
                'cultural_markers': ['euro', 'real', 'football', 'carnival', 'fado']
            },
            'ru': {
                'name': 'Russian',
                'regions': ['RU', 'BY', 'KZ', 'KG'],
                'cultural_markers': ['ruble', 'kremlin', 'vodka', 'moscow', 'putin']
            },
            'ja': {
                'name': 'Japanese',
                'regions': ['JP'],
                'cultural_markers': ['yen', 'tokyo', 'anime', 'sushi', 'emperor']
            },
            'ko': {
                'name': 'Korean',
                'regions': ['KR', 'KP'],
                'cultural_markers': ['won', 'seoul', 'kimchi', 'kpop', 'samsung']
            },
            'zh': {
                'name': 'Chinese',
                'regions': ['CN', 'TW', 'HK', 'SG'],
                'cultural_markers': ['yuan', 'beijing', 'shanghai', 'hongkong', 'taiwan']
            },
            'ar': {
                'name': 'Arabic',
                'regions': ['SA', 'EG', 'AE', 'MA', 'JO', 'LB'],
                'cultural_markers': ['riyal', 'dirham', 'mecca', 'ramadan', 'mosque']
            },
            'hi': {
                'name': 'Hindi',
                'regions': ['IN'],
                'cultural_markers': ['rupee', 'bollywood', 'delhi', 'mumbai', 'cricket']
            }
        }
        
        # Initialize advanced language detection model if available
        self.transformer_detector = None
        if HAS_TRANSFORMERS:
            try:
                model_name = self.config.get('language_model', 'papluca/xlm-roberta-base-language-detection')
                self.transformer_detector = pipeline(
                    "text-classification",
                    model=model_name,
                    return_all_scores=True
                )
                logging.info(f"Loaded transformer language detection model: {model_name}")
            except Exception as e:
                logging.warning(f"Could not load transformer language detection: {e}")
                self.transformer_detector = None
        
        # Cultural context patterns
        self.cultural_patterns = self._build_cultural_patterns()
        
        # Language statistics
        self.detection_stats = {
            'total_detections': 0,
            'language_distribution': defaultdict(int),
            'confidence_distribution': defaultdict(int),
            'regional_distribution': defaultdict(int),
            'avg_confidence': 0.0
        }
        
        # Cache for performance
        self.detection_cache = {}
        self.cache_size_limit = 10000
    
    def detect_language(self, text: str, include_regional: bool = True) -> Dict[str, Any]:
        """
        Detect language with confidence scoring and regional analysis
        
        Args:
            text: Input text
            include_regional: Whether to include regional variant detection
            
        Returns:
            Dictionary with language detection results
        """
        if not text or len(text.strip()) < 3:
            return {'error': 'Text too short for language detection'}
        
        # Check cache
        cache_key = f"{hash(text)}_{include_regional}"
        if cache_key in self.detection_cache:
            return self.detection_cache[cache_key]
        
        results = {
            'text': text[:100] + '...' if len(text) > 100 else text,
            'timestamp': datetime.now().isoformat()
        }
        
        # Primary detection using langdetect
        if HAS_LANGDETECT:
            langdetect_result = self._detect_with_langdetect(text)
            results['langdetect'] = langdetect_result
        
        # Advanced detection using transformers
        if self.transformer_detector:
            transformer_result = self._detect_with_transformer(text)
            results['transformer'] = transformer_result
        
        # Aggregate results
        aggregated = self._aggregate_detection_results(results)
        results['aggregated'] = aggregated
        
        # Regional variant detection
        if include_regional and aggregated.get('language'):
            regional_info = self._detect_regional_variant(text, aggregated['language'])
            results['regional'] = regional_info
        
        # Cultural context analysis
        cultural_context = self._analyze_cultural_context(text, aggregated.get('language'))
        results['cultural_context'] = cultural_context
        
        # Confidence assessment
        confidence_assessment = self._assess_detection_confidence(results)
        results['confidence_assessment'] = confidence_assessment
        
        # Update statistics
        self._update_detection_stats(results)
        
        # Cache result
        if len(self.detection_cache) < self.cache_size_limit:
            self.detection_cache[cache_key] = results
        
        return results
    
    def _detect_with_langdetect(self, text: str) -> Dict[str, Any]:
        """Detect language using langdetect library"""
        try:
            # Single language detection
            detected_lang = detect(text)
            
            # Multiple language probabilities
            lang_probs = detect_langs(text)
            
            # Convert to standardized format
            language_scores = {}
            for lang_prob in lang_probs:
                language_scores[lang_prob.lang] = lang_prob.prob
            
            return {
                'detected_language': detected_lang,
                'confidence': language_scores.get(detected_lang, 0.0),
                'all_languages': language_scores,
                'method': 'langdetect',
                'success': True
            }
            
        except LangDetectException as e:
            return {
                'detected_language': 'unknown',
                'confidence': 0.0,
                'error': str(e),
                'method': 'langdetect',
                'success': False
            }
        except Exception as e:
            logging.error(f"Langdetect error: {e}")
            return {
                'detected_language': 'unknown',
                'confidence': 0.0,
                'error': str(e),
                'method': 'langdetect',
                'success': False
            }
    
    def _detect_with_transformer(self, text: str) -> Dict[str, Any]:
        """Detect language using transformer model"""
        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            # Get language predictions
            predictions = self.transformer_detector(text)[0]
            
            # Convert to standardized format
            language_scores = {}
            detected_language = None
            max_score = 0
            
            for prediction in predictions:
                lang_code = prediction['label'].lower()
                score = prediction['score']
                language_scores[lang_code] = score
                
                if score > max_score:
                    max_score = score
                    detected_language = lang_code
            
            return {
                'detected_language': detected_language,
                'confidence': max_score,
                'all_languages': language_scores,
                'method': 'transformer',
                'success': True
            }
            
        except Exception as e:
            logging.error(f"Transformer language detection error: {e}")
            return {
                'detected_language': 'unknown',
                'confidence': 0.0,
                'error': str(e),
                'method': 'transformer',
                'success': False
            }
    
    def _aggregate_detection_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple detection methods"""
        aggregated = {
            'language': 'unknown',
            'confidence': 0.0,
            'methods_used': [],
            'agreement_score': 0.0
        }
        
        detected_languages = []
        confidences = []
        method_weights = {'langdetect': 0.6, 'transformer': 0.4}
        
        # Collect results from each method
        for method in ['langdetect', 'transformer']:
            if method in results and results[method].get('success'):
                detected_lang = results[method]['detected_language']
                confidence = results[method]['confidence']
                
                detected_languages.append(detected_lang)
                confidences.append(confidence)
                aggregated['methods_used'].append(method)
        
        if not detected_languages:
            return aggregated
        
        # Find consensus
        language_votes = Counter(detected_languages)
        most_common_lang = language_votes.most_common(1)[0][0]
        
        # Calculate weighted confidence
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for i, method in enumerate(aggregated['methods_used']):
            if detected_languages[i] == most_common_lang:
                weight = method_weights.get(method, 0.5)
                weighted_confidence += confidences[i] * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_confidence /= total_weight
        
        # Agreement score (how much methods agree)
        agreement_score = language_votes[most_common_lang] / len(detected_languages)
        
        aggregated.update({
            'language': most_common_lang,
            'confidence': weighted_confidence,
            'agreement_score': agreement_score,
            'language_votes': dict(language_votes)
        })
        
        return aggregated
    
    def _detect_regional_variant(self, text: str, language: str) -> Dict[str, Any]:
        """Detect regional variants of detected language"""
        if language not in self.supported_languages:
            return {'region': 'unknown', 'confidence': 0.0}
        
        text_lower = text.lower()
        lang_info = self.supported_languages[language]
        
        # Score each region based on cultural markers
        region_scores = {}
        
        for region in lang_info['regions']:
            score = 0.0
            
            # Check for region-specific cultural markers
            region_patterns = self.cultural_patterns.get(f"{language}_{region}", [])
            
            for pattern in region_patterns:
                if pattern in text_lower:
                    score += 1.0
            
            # Normalize by text length
            word_count = len(text.split())
            if word_count > 0:
                score = score / word_count * 100  # Per 100 words
            
            region_scores[region] = score
        
        # Find most likely region
        if region_scores:
            best_region = max(region_scores, key=region_scores.get)
            best_score = region_scores[best_region]
            
            # Only return if there's some evidence
            if best_score > 0:
                return {
                    'region': best_region,
                    'confidence': min(best_score, 1.0),  # Cap at 1.0
                    'all_regions': region_scores,
                    'full_language_code': f"{language}_{best_region}"
                }
        
        # Default to first region if no specific markers found
        default_region = lang_info['regions'][0]
        return {
            'region': default_region,
            'confidence': 0.1,  # Low confidence for default
            'all_regions': region_scores,
            'full_language_code': f"{language}_{default_region}"
        }
    
    def _analyze_cultural_context(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze cultural context and themes in text"""
        if not language or language not in self.supported_languages:
            return {'cultural_markers': [], 'context_score': 0.0}
        
        text_lower = text.lower()
        lang_info = self.supported_languages[language]
        
        # Find cultural markers
        found_markers = []
        for marker in lang_info['cultural_markers']:
            if marker in text_lower:
                found_markers.append(marker)
        
        # Additional context analysis
        context_themes = self._identify_context_themes(text_lower, language)
        
        # Calculate context richness score
        word_count = len(text.split())
        context_score = len(found_markers) / max(word_count / 100, 1)  # Per 100 words
        context_score = min(context_score, 1.0)  # Cap at 1.0
        
        return {
            'cultural_markers': found_markers,
            'context_themes': context_themes,
            'context_score': context_score,
            'language_family': self._get_language_family(language)
        }
    
    def _identify_context_themes(self, text: str, language: str) -> List[str]:
        """Identify cultural and contextual themes"""
        themes = []
        
        # Theme patterns for different languages
        theme_patterns = {
            'en': {
                'politics': ['government', 'election', 'president', 'parliament', 'congress'],
                'business': ['company', 'market', 'stock', 'economy', 'finance'],
                'sports': ['game', 'match', 'team', 'player', 'championship'],
                'technology': ['software', 'computer', 'internet', 'digital', 'tech']
            },
            'es': {
                'politics': ['gobierno', 'elección', 'presidente', 'parlamento', 'congreso'],
                'business': ['empresa', 'mercado', 'economía', 'finanzas', 'negocio'],
                'sports': ['partido', 'equipo', 'jugador', 'fútbol', 'deporte'],
                'technology': ['tecnología', 'computadora', 'internet', 'digital']
            },
            'fr': {
                'politics': ['gouvernement', 'élection', 'président', 'parlement'],
                'business': ['entreprise', 'marché', 'économie', 'finance'],
                'sports': ['match', 'équipe', 'joueur', 'football', 'sport'],
                'technology': ['technologie', 'ordinateur', 'internet', 'numérique']
            }
        }
        
        if language in theme_patterns:
            for theme, keywords in theme_patterns[language].items():
                if any(keyword in text for keyword in keywords):
                    themes.append(theme)
        
        return themes
    
    def _get_language_family(self, language: str) -> str:
        """Get language family for linguistic context"""
        language_families = {
            'en': 'Germanic',
            'de': 'Germanic',
            'es': 'Romance',
            'fr': 'Romance',
            'it': 'Romance',
            'pt': 'Romance',
            'ru': 'Slavic',
            'ja': 'Japonic',
            'ko': 'Koreanic',
            'zh': 'Sino-Tibetan',
            'ar': 'Semitic',
            'hi': 'Indo-Aryan'
        }
        return language_families.get(language, 'Unknown')
    
    def _build_cultural_patterns(self) -> Dict[str, List[str]]:
        """Build cultural patterns for regional detection"""
        patterns = {}
        
        # English variants
        patterns['en_US'] = ['dollar', 'america', 'usa', 'american', 'washington', 'congress', 'nfl', 'nba']
        patterns['en_UK'] = ['pound', 'britain', 'british', 'london', 'parliament', 'premier league', 'bbc']
        patterns['en_AU'] = ['australia', 'australian', 'sydney', 'melbourne', 'afl', 'cricket']
        patterns['en_CA'] = ['canada', 'canadian', 'toronto', 'vancouver', 'hockey', 'quebec']
        
        # Spanish variants
        patterns['es_ES'] = ['españa', 'español', 'madrid', 'barcelona', 'euro', 'la liga']
        patterns['es_MX'] = ['méxico', 'mexicano', 'peso', 'guadalajara', 'azteca']
        patterns['es_AR'] = ['argentina', 'argentino', 'buenos aires', 'peso', 'tango']
        
        # French variants
        patterns['fr_FR'] = ['france', 'français', 'paris', 'euro', 'marseille', 'ligue 1']
        patterns['fr_CA'] = ['canada', 'québec', 'montreal', 'canadian', 'hockey']
        
        # German variants
        patterns['de_DE'] = ['deutschland', 'deutsch', 'berlin', 'münchen', 'bundesliga']
        patterns['de_AT'] = ['österreich', 'wien', 'salzburg', 'austrian']
        patterns['de_CH'] = ['schweiz', 'zürich', 'geneva', 'swiss', 'franken']
        
        return patterns
    
    def _assess_detection_confidence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall confidence in language detection"""
        confidence_factors = []
        
        # Method agreement
        if 'aggregated' in results:
            agreement_score = results['aggregated'].get('agreement_score', 0)
            confidence_factors.append(('method_agreement', agreement_score))
            
            base_confidence = results['aggregated'].get('confidence', 0)
            confidence_factors.append(('base_confidence', base_confidence))
        
        # Text length factor (longer texts generally more reliable)
        text_length = len(results.get('text', ''))
        length_factor = min(text_length / 200, 1.0)  # Normalize to 200 chars
        confidence_factors.append(('text_length', length_factor))
        
        # Cultural context factor
        if 'cultural_context' in results:
            context_score = results['cultural_context'].get('context_score', 0)
            confidence_factors.append(('cultural_context', context_score))
        
        # Calculate overall confidence
        if confidence_factors:
            weights = [0.4, 0.3, 0.2, 0.1]  # Weights for different factors
            overall_confidence = sum(
                factor[1] * weight 
                for factor, weight in zip(confidence_factors, weights[:len(confidence_factors)])
            )
        else:
            overall_confidence = 0.0
        
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
    
    def detect_multiple_languages(self, text: str, threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect multiple languages in mixed-language text
        
        Args:
            text: Input text that may contain multiple languages
            threshold: Minimum confidence threshold for language detection
            
        Returns:
            Dictionary with multiple language detection results
        """
        # Split text into segments (sentences/paragraphs)
        segments = self._segment_text(text)
        
        segment_results = []
        language_distribution = defaultdict(float)
        
        for i, segment in enumerate(segments):
            if len(segment.strip()) < 10:  # Skip very short segments
                continue
                
            segment_result = self.detect_language(segment, include_regional=False)
            
            if 'aggregated' in segment_result:
                detected_lang = segment_result['aggregated']['language']
                confidence = segment_result['aggregated']['confidence']
                
                if confidence >= threshold:
                    segment_results.append({
                        'segment_index': i,
                        'segment_text': segment[:100] + '...' if len(segment) > 100 else segment,
                        'language': detected_lang,
                        'confidence': confidence,
                        'length': len(segment)
                    })
                    
                    # Weight by segment length
                    language_distribution[detected_lang] += len(segment) * confidence
        
        # Normalize language distribution
        total_weight = sum(language_distribution.values())
        if total_weight > 0:
            language_distribution = {
                lang: weight / total_weight 
                for lang, weight in language_distribution.items()
            }
        
        # Determine primary language
        primary_language = max(language_distribution, key=language_distribution.get) if language_distribution else 'unknown'
        
        return {
            'text_length': len(text),
            'num_segments': len(segments),
            'segments_analyzed': len(segment_results),
            'primary_language': primary_language,
            'language_distribution': dict(language_distribution),
            'segment_results': segment_results,
            'is_multilingual': len(language_distribution) > 1,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _segment_text(self, text: str) -> List[str]:
        """Segment text into analyzable chunks"""
        import nltk
        
        try:
            # Try sentence segmentation first
            sentences = nltk.sent_tokenize(text)
            if len(sentences) > 1:
                return sentences
        except:
            pass
        
        # Fallback to paragraph segmentation
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            return [p.strip() for p in paragraphs if p.strip()]
        
        # Fallback to fixed-size chunks
        chunk_size = 200
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        return chunks
    
    def analyze_language_trends(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze language trends across multiple articles
        
        Args:
            articles: List of articles with 'text' and optional 'date' fields
            
        Returns:
            Dictionary with language trend analysis
        """
        logging.info(f"Analyzing language trends for {len(articles)} articles...")
        
        language_stats = defaultdict(int)
        temporal_data = defaultdict(list)
        regional_stats = defaultdict(int)
        
        for i, article in enumerate(articles):
            if 'text' not in article:
                continue
                
            # Detect language
            detection_result = self.detect_language(article['text'])
            
            if 'aggregated' in detection_result and detection_result['aggregated']['language'] != 'unknown':
                lang = detection_result['aggregated']['language']
                confidence = detection_result['aggregated']['confidence']
                
                # Only count high-confidence detections
                if confidence > 0.5:
                    language_stats[lang] += 1
                    
                    # Regional information
                    if 'regional' in detection_result:
                        region_code = detection_result['regional'].get('full_language_code', lang)
                        regional_stats[region_code] += 1
                    
                    # Temporal data if available
                    if 'date' in article:
                        temporal_data[lang].append({
                            'date': article['date'],
                            'confidence': confidence
                        })
        
        # Calculate language diversity
        total_articles = sum(language_stats.values())
        language_distribution = {
            lang: count / total_articles 
            for lang, count in language_stats.items()
        } if total_articles > 0 else {}
        
        # Calculate diversity index (Shannon entropy)
        diversity_index = 0.0
        if language_distribution:
            diversity_index = -sum(
                prob * np.log2(prob) 
                for prob in language_distribution.values() 
                if prob > 0
            )
        
        return {
            'total_articles_analyzed': len(articles),
            'total_articles_with_detection': total_articles,
            'language_statistics': dict(language_stats),
            'language_distribution': language_distribution,
            'regional_statistics': dict(regional_stats),
            'language_diversity_index': diversity_index,
            'dominant_language': max(language_stats, key=language_stats.get) if language_stats else None,
            'num_languages_detected': len(language_stats),
            'temporal_data': dict(temporal_data),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _update_detection_stats(self, results: Dict[str, Any]):
        """Update detection statistics"""
        self.detection_stats['total_detections'] += 1
        
        if 'aggregated' in results:
            lang = results['aggregated']['language']
            confidence = results['aggregated']['confidence']
            
            self.detection_stats['language_distribution'][lang] += 1
            
            # Update confidence average
            total = self.detection_stats['total_detections']
            current_avg = self.detection_stats['avg_confidence']
            self.detection_stats['avg_confidence'] = (
                (current_avg * (total - 1) + confidence) / total
            )
            
            # Confidence distribution
            confidence_bin = int(confidence * 10) / 10  # Round to nearest 0.1
            self.detection_stats['confidence_distribution'][confidence_bin] += 1
            
            # Regional distribution
            if 'regional' in results:
                region_code = results['regional'].get('full_language_code', lang)
                self.detection_stats['regional_distribution'][region_code] += 1
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get language detection statistics"""
        stats = {
            'total_detections': self.detection_stats['total_detections'],
            'avg_confidence': self.detection_stats['avg_confidence'],
            'language_distribution': dict(self.detection_stats['language_distribution']),
            'confidence_distribution': dict(self.detection_stats['confidence_distribution']),
            'regional_distribution': dict(self.detection_stats['regional_distribution'])
        }
        
        # Add percentages
        total = stats['total_detections']
        if total > 0:
            stats['language_percentages'] = {
                lang: (count / total) * 100 
                for lang, count in stats['language_distribution'].items()
            }
        
        return stats
    
    def clear_cache(self):
        """Clear detection cache"""
        self.detection_cache.clear()
        logging.info("Language detection cache cleared")
    
    def save_detector(self, filepath: str):
        """Save language detector configuration"""
        save_data = {
            'config': self.config,
            'supported_languages': self.supported_languages,
            'cultural_patterns': self.cultural_patterns,
            'detection_stats': dict(self.detection_stats),
            'save_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logging.info(f"Language detector saved to {filepath}")
    
    def load_detector(self, filepath: str):
        """Load language detector configuration"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.config = save_data['config']
        self.supported_languages = save_data['supported_languages']
        self.cultural_patterns = save_data['cultural_patterns']
        
        # Convert defaultdict back to regular dict then to defaultdict
        stats_dict = save_data['detection_stats']
        for key in ['language_distribution', 'confidence_distribution', 'regional_distribution']:
            if key in stats_dict:
                self.detection_stats[key] = defaultdict(int, stats_dict[key])
        
        self.detection_stats['total_detections'] = stats_dict.get('total_detections', 0)
        self.detection_stats['avg_confidence'] = stats_dict.get('avg_confidence', 0.0)
        
        logging.info(f"Language detector loaded from {filepath}")
    
    def get_supported_languages(self) -> Dict[str, Dict[str, Any]]:
        """Get information about supported languages"""
        return self.supported_languages.copy()