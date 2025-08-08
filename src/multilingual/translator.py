#!/usr/bin/env python3
"""
Multilingual Translator for NewsBot 2.0
Advanced translation with quality assessment and cultural adaptation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from collections import defaultdict
import re
import pickle
import time
import os

# Core translation libraries
try:
    from deep_translator import GoogleTranslator, MicrosoftTranslator, LibreTranslator
    HAS_DEEP_TRANSLATOR = True
except ImportError:
    HAS_DEEP_TRANSLATOR = False
    logging.warning("deep-translator not available. Install for translation features.")

# Fallback translation
try:
    from googletrans import Translator as GoogleTranslatorFallback
    HAS_GOOGLETRANS = True
except ImportError:
    HAS_GOOGLETRANS = False
    logging.warning("googletrans not available as fallback translator.")

# Advanced translation using transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("transformers not available for advanced translation.")

class Translator:
    """
    Alias for MultilingualTranslator for backwards compatibility
    """
    def __init__(self, *args, **kwargs):
        # Create an instance of MultilingualTranslator and delegate to it
        self._translator = MultilingualTranslator(*args, **kwargs)
        
    def __getattr__(self, name):
        # Delegate all attribute access to the internal translator
        return getattr(self._translator, name)

class MultilingualTranslator:
    """
    Advanced multilingual translation with quality assessment and cultural adaptation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize multilingual translator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Translation services configuration
        self.services = {}
        self.service_priority = ['google', 'microsoft', 'libre', 'transformer']
        
        # Initialize translation services
        self._initialize_services()
        
        # Language pairs and quality thresholds
        self.supported_languages = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese',
            'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi',
            'tr': 'Turkish', 'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish',
            'no': 'Norwegian', 'da': 'Danish', 'fi': 'Finnish', 'he': 'Hebrew'
        }
        
        # Quality assessment thresholds
        self.quality_thresholds = {
            'min_length_ratio': 0.3,    # Translated text should be at least 30% of original
            'max_length_ratio': 3.0,    # And at most 300% of original
            'min_word_preservation': 0.1,  # At least 10% words should be preserved (names, etc.)
            'max_repetition_ratio': 0.3    # No more than 30% repetitive text
        }
        
        # Cultural adaptation patterns
        self.cultural_adaptations = self._build_cultural_adaptations()
        
        # Translation cache for performance
        self.translation_cache = {}
        self.cache_size_limit = 5000
        
        # Translation statistics
        self.translation_stats = {
            'total_translations': 0,
            'successful_translations': 0,
            'language_pair_counts': defaultdict(int),
            'service_usage': defaultdict(int),
            'avg_translation_time': 0.0,
            'quality_scores': []
        }
    
    def _initialize_services(self):
        """Initialize available translation services"""
        
        # Google Translator (deep-translator)
        if HAS_DEEP_TRANSLATOR:
            try:
                self.services['google'] = GoogleTranslator()
                logging.info("Google Translator (deep-translator) initialized")
            except Exception as e:
                logging.warning(f"Could not initialize Google Translator: {e}")
        
        # Microsoft Translator
        if HAS_DEEP_TRANSLATOR:
            try:
                # Microsoft requires API key, so this might fail
                api_key = self.config.get('microsoft_translator_key')
                if api_key:
                    self.services['microsoft'] = MicrosoftTranslator(api_key=api_key)
                    logging.info("Microsoft Translator initialized")
            except Exception as e:
                logging.warning(f"Could not initialize Microsoft Translator: {e}")
        
        # LibreTranslate (free, open source)
        if HAS_DEEP_TRANSLATOR:
            try:
                libre_url = self.config.get('libre_translate_url', 'https://libretranslate.de')
                libre_api_key = self.config.get('libre_api_key') or os.getenv('LIBRE_API_KEY')
                
                if libre_api_key:
                    self.services['libre'] = LibreTranslator(base_url=libre_url, api_key=libre_api_key)
                    logging.info("LibreTranslate initialized with API key")
                else:
                    # Try without API key for free instances
                    self.services['libre'] = LibreTranslator(base_url=libre_url)
                    logging.info("LibreTranslate initialized without API key")
            except Exception as e:
                logging.warning(f"Could not initialize LibreTranslate: {e}")
        
        # Fallback Google Translator
        if HAS_GOOGLETRANS and 'google' not in self.services:
            try:
                self.services['google_fallback'] = GoogleTranslatorFallback()
                logging.info("Google Translator (fallback) initialized")
            except Exception as e:
                logging.warning(f"Could not initialize Google Translator fallback: {e}")
        
        # Transformer-based translation
        if HAS_TRANSFORMERS:
            try:
                # Load a multilingual translation model that doesn't require sentencepiece
                model_name = self.config.get('translation_model', 'Helsinki-NLP/opus-mt-en-de')
                
                # Try to check if tokenizer requirements are met
                try:
                    import sentencepiece
                    # If sentencepiece is available, we can use the original model
                    model_name = self.config.get('translation_model', 'Helsinki-NLP/opus-mt-en-mul')
                except ImportError:
                    # Use a simpler model that doesn't require sentencepiece
                    model_name = 'Helsinki-NLP/opus-mt-en-de'
                    logging.info("Using simplified translation model due to missing sentencepiece")
                
                self.services['transformer'] = pipeline(
                    "translation",
                    model=model_name
                )
                logging.info(f"Transformer translator initialized: {model_name}")
            except Exception as e:
                logging.warning(f"Could not initialize transformer translator: {e}")
        
        if not self.services:
            logging.error("No translation services available!")
    
    def translate_text(self, text: str, source_lang: str, target_lang: str,
                      service: Optional[str] = None, quality_check: bool = True) -> Dict[str, Any]:
        """
        Translate text with quality assessment and cultural adaptation
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            service: Preferred translation service
            quality_check: Whether to perform quality assessment
            
        Returns:
            Dictionary with translation results and quality metrics
        """
        if not text or len(text.strip()) < 3:
            return {'error': 'Text too short for translation'}
        
        # Validate language codes
        if source_lang not in self.supported_languages:
            return {'error': f'Unsupported source language: {source_lang}'}
        
        if target_lang not in self.supported_languages:
            return {'error': f'Unsupported target language: {target_lang}'}
        
        if source_lang == target_lang:
            return {
                'original_text': text,
                'translated_text': text,
                'source_language': source_lang,
                'target_language': target_lang,
                'translation_service': 'none',
                'quality_score': 1.0,
                'notes': 'Same language, no translation needed'
            }
        
        # Check cache
        cache_key = f"{hash(text)}_{source_lang}_{target_lang}_{service}"
        if cache_key in self.translation_cache:
            cached_result = self.translation_cache[cache_key]
            cached_result['from_cache'] = True
            return cached_result
        
        start_time = time.time()
        
        # Determine service to use
        services_to_try = [service] if service else self.service_priority
        
        translation_result = None
        service_used = None
        
        # Try services in order
        for service_name in services_to_try:
            if service_name in self.services:
                try:
                    translation_result = self._translate_with_service(
                        text, source_lang, target_lang, service_name
                    )
                    service_used = service_name
                    break
                except Exception as e:
                    logging.warning(f"Translation failed with {service_name}: {e}")
                    continue
        
        if not translation_result:
            return {
                'error': 'All translation services failed',
                'original_text': text,
                'source_language': source_lang,
                'target_language': target_lang
            }
        
        # Post-process translation
        processed_translation = self._post_process_translation(
            translation_result, source_lang, target_lang
        )
        
        # Cultural adaptation
        culturally_adapted = self._apply_cultural_adaptation(
            processed_translation, source_lang, target_lang
        )
        
        # Quality assessment
        quality_metrics = {}
        if quality_check:
            quality_metrics = self._assess_translation_quality(
                text, culturally_adapted, source_lang, target_lang
            )
        
        # Prepare final result
        end_time = time.time()
        translation_time = end_time - start_time
        
        result = {
            'original_text': text,
            'translated_text': culturally_adapted,
            'source_language': source_lang,
            'target_language': target_lang,
            'translation_service': service_used,
            'translation_time': translation_time,
            'quality_metrics': quality_metrics,
            'timestamp': datetime.now().isoformat(),
            'from_cache': False
        }
        
        # Update statistics
        self._update_translation_stats(result)
        
        # Cache result
        if len(self.translation_cache) < self.cache_size_limit:
            self.translation_cache[cache_key] = result
        
        return result
    
    def _translate_with_service(self, text: str, source_lang: str, 
                               target_lang: str, service_name: str) -> str:
        """Translate text using specific service"""
        
        service = self.services[service_name]
        
        if service_name == 'google':
            # Deep translator Google
            service.source = source_lang
            service.target = target_lang
            return service.translate(text)
        
        elif service_name == 'microsoft':
            # Microsoft translator
            return service.translate(text, source=source_lang, target=target_lang)
        
        elif service_name == 'libre':
            # LibreTranslate
            return service.translate(text, source=source_lang, target=target_lang)
        
        elif service_name == 'google_fallback':
            # Fallback Google translator
            result = service.translate(text, src=source_lang, dest=target_lang)
            return result.text
        
        elif service_name == 'transformer':
            # Transformer-based translation
            # Note: This is simplified - real implementation would need language-specific models
            translation_text = service(text)[0]['translation_text']
            return translation_text
        
        else:
            raise ValueError(f"Unknown service: {service_name}")
    
    def _post_process_translation(self, translation: str, source_lang: str, target_lang: str) -> str:
        """Post-process translation to fix common issues"""
        
        # Fix spacing issues
        processed = re.sub(r'\s+', ' ', translation).strip()
        
        # Fix punctuation spacing
        processed = re.sub(r'\s+([.,!?;:])', r'\1', processed)
        processed = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1 \2', processed)
        
        # Fix quotation marks
        processed = re.sub(r'\s+"([^"]*?)"\s+', r' "\1" ', processed)
        
        # Language-specific post-processing
        if target_lang == 'es':
            # Spanish: Fix inverted question/exclamation marks
            processed = re.sub(r'¿\s+', '¿', processed)
            processed = re.sub(r'\s+\?', '?', processed)
            processed = re.sub(r'¡\s+', '¡', processed)
            processed = re.sub(r'\s+!', '!', processed)
        
        elif target_lang == 'fr':
            # French: Fix spacing around punctuation
            processed = re.sub(r'\s*:\s*', ' : ', processed)
            processed = re.sub(r'\s*;\s*', ' ; ', processed)
            processed = re.sub(r'\s*!\s*', ' ! ', processed)
            processed = re.sub(r'\s*\?\s*', ' ? ', processed)
        
        elif target_lang == 'de':
            # German: Capitalize nouns (simplified)
            words = processed.split()
            # This is a very basic implementation - real German capitalization is complex
            for i, word in enumerate(words):
                if len(word) > 3 and word.islower():
                    # Simple heuristic for potential nouns
                    if i > 0 and words[i-1].lower() in ['der', 'die', 'das', 'ein', 'eine']:
                        words[i] = word.capitalize()
            processed = ' '.join(words)
        
        return processed
    
    def _apply_cultural_adaptation(self, translation: str, source_lang: str, target_lang: str) -> str:
        """Apply cultural adaptations to translation"""
        
        adapted = translation
        
        # Get cultural adaptation patterns
        adaptation_key = f"{source_lang}_{target_lang}"
        if adaptation_key in self.cultural_adaptations:
            adaptations = self.cultural_adaptations[adaptation_key]
            
            for pattern, replacement in adaptations.items():
                adapted = re.sub(pattern, replacement, adapted, flags=re.IGNORECASE)
        
        # Currency adaptations
        adapted = self._adapt_currencies(adapted, source_lang, target_lang)
        
        # Date format adaptations
        adapted = self._adapt_date_formats(adapted, target_lang)
        
        # Unit conversions (basic)
        adapted = self._adapt_units(adapted, target_lang)
        
        return adapted
    
    def _adapt_currencies(self, text: str, source_lang: str, target_lang: str) -> str:
        """Adapt currency references"""
        
        currency_mappings = {
            'en_es': {'dollar': 'dólar', 'dollars': 'dólares'},
            'en_fr': {'dollar': 'dollar', 'dollars': 'dollars'},
            'en_de': {'dollar': 'Dollar', 'dollars': 'Dollar'},
            'es_en': {'peso': 'peso', 'pesos': 'pesos'},
            'fr_en': {'euro': 'euro', 'euros': 'euros'},
            'de_en': {'euro': 'euro', 'euros': 'euros'}
        }
        
        mapping_key = f"{source_lang}_{target_lang}"
        if mapping_key in currency_mappings:
            for source_curr, target_curr in currency_mappings[mapping_key].items():
                text = re.sub(rf'\b{source_curr}\b', target_curr, text, flags=re.IGNORECASE)
        
        return text
    
    def _adapt_date_formats(self, text: str, target_lang: str) -> str:
        """Adapt date formats for target language"""
        
        # This is a simplified implementation
        # Real date adaptation would be much more complex
        
        if target_lang in ['de', 'fr', 'es', 'it']:
            # European date format (DD/MM/YYYY)
            text = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\1.\2.\3', text)
        
        return text
    
    def _adapt_units(self, text: str, target_lang: str) -> str:
        """Adapt measurement units"""
        
        # Imperial to metric conversions for non-US languages
        if target_lang != 'en':
            # This is very simplified - real unit conversion would be complex
            text = re.sub(r'\bfeet\b', 'meters', text, flags=re.IGNORECASE)
            text = re.sub(r'\bmiles\b', 'kilometers', text, flags=re.IGNORECASE)
            text = re.sub(r'\bpounds\b', 'kilograms', text, flags=re.IGNORECASE)
        
        return text
    
    def _assess_translation_quality(self, original: str, translation: str,
                                   source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Assess translation quality using multiple metrics"""
        
        quality_metrics = {}
        
        # Length ratio check
        original_length = len(original)
        translation_length = len(translation)
        length_ratio = translation_length / original_length if original_length > 0 else 0
        
        quality_metrics['length_ratio'] = length_ratio
        quality_metrics['length_ratio_normal'] = (
            self.quality_thresholds['min_length_ratio'] <= length_ratio <= self.quality_thresholds['max_length_ratio']
        )
        
        # Word preservation check (for proper nouns, numbers, etc.)
        original_words = set(original.split())
        translation_words = set(translation.split())
        
        # Words that should typically be preserved (numbers, proper nouns)
        preserved_candidates = set()
        for word in original_words:
            if (word.isdigit() or 
                word[0].isupper() or 
                len(word) > 10 or  # Long words often proper nouns
                word.lower() in ['covid', 'nasa', 'eu', 'un', 'usa', 'uk']):
                preserved_candidates.add(word.lower())
        
        preserved_words = preserved_candidates.intersection(
            {word.lower() for word in translation_words}
        )
        
        word_preservation_ratio = (
            len(preserved_words) / len(preserved_candidates) 
            if preserved_candidates else 1.0
        )
        
        quality_metrics['word_preservation_ratio'] = word_preservation_ratio
        quality_metrics['preserved_words'] = list(preserved_words)
        
        # Repetition check
        translation_word_list = translation.split()
        if translation_word_list:
            word_counts = {}
            for word in translation_word_list:
                word_counts[word.lower()] = word_counts.get(word.lower(), 0) + 1
            
            repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
            repetition_ratio = repeated_words / len(translation_word_list)
            
            quality_metrics['repetition_ratio'] = repetition_ratio
            quality_metrics['repetition_normal'] = repetition_ratio <= self.quality_thresholds['max_repetition_ratio']
        
        # Character set check (make sure translation uses appropriate characters)
        quality_metrics['character_set_appropriate'] = self._check_character_set(translation, target_lang)
        
        # Overall quality score
        quality_factors = [
            quality_metrics.get('length_ratio_normal', False),
            word_preservation_ratio >= self.quality_thresholds['min_word_preservation'],
            quality_metrics.get('repetition_normal', False),
            quality_metrics.get('character_set_appropriate', False)
        ]
        
        quality_score = sum(quality_factors) / len(quality_factors)
        quality_metrics['overall_quality_score'] = quality_score
        
        # Quality grade
        if quality_score >= 0.8:
            quality_metrics['quality_grade'] = 'excellent'
        elif quality_score >= 0.6:
            quality_metrics['quality_grade'] = 'good'
        elif quality_score >= 0.4:
            quality_metrics['quality_grade'] = 'fair'
        else:
            quality_metrics['quality_grade'] = 'poor'
        
        return quality_metrics
    
    def _check_character_set(self, text: str, language: str) -> bool:
        """Check if text uses appropriate character set for language"""
        
        # Expected character sets for different languages
        language_chars = {
            'es': 'áéíóúüñ¿¡',
            'fr': 'àâäéèêëîïôöùûüÿç',
            'de': 'äöüß',
            'it': 'àèéìíîòóù',
            'pt': 'ãâáàçéêíóôõú',
            'ru': 'абвгдежзийклмнопрстуфхцчшщъыьэюя',
            'ja': 'ひらがなカタカナ',  # Simplified
            'ko': '한글',  # Simplified
            'zh': '中文',  # Simplified
            'ar': 'العربية'  # Simplified
        }
        
        if language not in language_chars:
            return True  # Unknown language, assume OK
        
        expected_chars = language_chars[language]
        
        # Check if text contains at least some expected characters
        # (This is a very basic check)
        text_lower = text.lower()
        
        if language in ['ru', 'ja', 'ko', 'zh', 'ar']:
            # For non-Latin scripts, text should contain script-specific characters
            return any(char in text for char in expected_chars)
        else:
            # For Latin-based scripts, at least some accented characters expected
            # But not required for short texts
            if len(text) < 50:
                return True
            return any(char in text_lower for char in expected_chars)
    
    def translate_multiple_texts(self, texts: List[str], source_lang: str, target_lang: str,
                               batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Translate multiple texts efficiently
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of translation results
        """
        logging.info(f"Translating {len(texts)} texts from {source_lang} to {target_lang}")
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            if i % 50 == 0:  # Progress update every 50 texts
                logging.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            batch_results = []
            for text in batch:
                try:
                    result = self.translate_text(text, source_lang, target_lang)
                    batch_results.append(result)
                except Exception as e:
                    logging.error(f"Translation failed for text: {str(e)}")
                    batch_results.append({
                        'error': str(e),
                        'original_text': text,
                        'source_language': source_lang,
                        'target_language': target_lang
                    })
            
            results.extend(batch_results)
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        return results
    
    def compare_translation_services(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """
        Compare translation quality across available services
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Dictionary with comparison results
        """
        comparison_results = {
            'original_text': text,
            'source_language': source_lang,
            'target_language': target_lang,
            'service_results': {},
            'comparison_timestamp': datetime.now().isoformat()
        }
        
        # Try each available service
        for service_name in self.services.keys():
            try:
                result = self.translate_text(
                    text, source_lang, target_lang, 
                    service=service_name, quality_check=True
                )
                
                if 'error' not in result:
                    comparison_results['service_results'][service_name] = {
                        'translated_text': result['translated_text'],
                        'quality_metrics': result.get('quality_metrics', {}),
                        'translation_time': result.get('translation_time', 0)
                    }
                else:
                    comparison_results['service_results'][service_name] = {
                        'error': result['error']
                    }
                    
            except Exception as e:
                comparison_results['service_results'][service_name] = {
                    'error': str(e)
                }
        
        # Rank services by quality
        service_rankings = []
        for service, result in comparison_results['service_results'].items():
            if 'quality_metrics' in result:
                quality_score = result['quality_metrics'].get('overall_quality_score', 0)
                service_rankings.append((service, quality_score))
        
        service_rankings.sort(key=lambda x: x[1], reverse=True)
        comparison_results['service_rankings'] = service_rankings
        
        if service_rankings:
            comparison_results['best_service'] = service_rankings[0][0]
            comparison_results['best_quality_score'] = service_rankings[0][1]
        
        return comparison_results
    
    def _build_cultural_adaptations(self) -> Dict[str, Dict[str, str]]:
        """Build cultural adaptation patterns"""
        
        adaptations = {}
        
        # English to Spanish
        adaptations['en_es'] = {
            r'\bfootball\b': 'fútbol americano',
            r'\bsoccer\b': 'fútbol',
            r'\bmom\b': 'mamá',
            r'\bdad\b': 'papá'
        }
        
        # English to French
        adaptations['en_fr'] = {
            r'\bweekend\b': 'fin de semaine',
            r'\bemail\b': 'courriel',
            r'\bparking\b': 'stationnement'
        }
        
        # English to German
        adaptations['en_de'] = {
            r'\bmobile phone\b': 'Handy',
            r'\bcell phone\b': 'Handy',
            r'\binternet\b': 'Internet'
        }
        
        return adaptations
    
    def _update_translation_stats(self, result: Dict[str, Any]):
        """Update translation statistics"""
        self.translation_stats['total_translations'] += 1
        
        if 'error' not in result:
            self.translation_stats['successful_translations'] += 1
            
            # Language pair
            lang_pair = f"{result['source_language']}_{result['target_language']}"
            self.translation_stats['language_pair_counts'][lang_pair] += 1
            
            # Service usage
            service = result.get('translation_service', 'unknown')
            self.translation_stats['service_usage'][service] += 1
            
            # Translation time
            trans_time = result.get('translation_time', 0)
            total = self.translation_stats['successful_translations']
            current_avg = self.translation_stats['avg_translation_time']
            self.translation_stats['avg_translation_time'] = (
                (current_avg * (total - 1) + trans_time) / total
            )
            
            # Quality score
            quality_score = result.get('quality_metrics', {}).get('overall_quality_score')
            if quality_score is not None:
                self.translation_stats['quality_scores'].append(quality_score)
    
    def get_translation_statistics(self) -> Dict[str, Any]:
        """Get translation statistics"""
        stats = {
            'total_translations': self.translation_stats['total_translations'],
            'successful_translations': self.translation_stats['successful_translations'],
            'success_rate': 0.0,
            'avg_translation_time': self.translation_stats['avg_translation_time'],
            'language_pair_counts': dict(self.translation_stats['language_pair_counts']),
            'service_usage': dict(self.translation_stats['service_usage']),
            'avg_quality_score': 0.0,
            'quality_distribution': {}
        }
        
        # Success rate
        if stats['total_translations'] > 0:
            stats['success_rate'] = stats['successful_translations'] / stats['total_translations']
        
        # Quality statistics
        quality_scores = self.translation_stats['quality_scores']
        if quality_scores:
            stats['avg_quality_score'] = np.mean(quality_scores)
            stats['quality_std'] = np.std(quality_scores)
            
            # Quality distribution
            quality_bins = ['poor', 'fair', 'good', 'excellent']
            quality_counts = [0, 0, 0, 0]
            
            for score in quality_scores:
                if score >= 0.8:
                    quality_counts[3] += 1  # excellent
                elif score >= 0.6:
                    quality_counts[2] += 1  # good
                elif score >= 0.4:
                    quality_counts[1] += 1  # fair
                else:
                    quality_counts[0] += 1  # poor
            
            stats['quality_distribution'] = dict(zip(quality_bins, quality_counts))
        
        return stats
    
    def get_supported_language_pairs(self) -> List[Dict[str, str]]:
        """Get list of supported language pairs"""
        pairs = []
        
        for source_code, source_name in self.supported_languages.items():
            for target_code, target_name in self.supported_languages.items():
                if source_code != target_code:
                    pairs.append({
                        'source_code': source_code,
                        'source_name': source_name,
                        'target_code': target_code,
                        'target_name': target_name,
                        'pair_code': f"{source_code}_{target_code}"
                    })
        
        return pairs
    
    def clear_cache(self):
        """Clear translation cache"""
        self.translation_cache.clear()
        logging.info("Translation cache cleared")
    
    def save_translator(self, filepath: str):
        """Save translator configuration and statistics"""
        save_data = {
            'config': self.config,
            'supported_languages': self.supported_languages,
            'quality_thresholds': self.quality_thresholds,
            'cultural_adaptations': self.cultural_adaptations,
            'translation_stats': dict(self.translation_stats),
            'service_priority': self.service_priority,
            'save_timestamp': datetime.now().isoformat()
        }
        
        # Convert defaultdict to regular dict for serialization
        save_data['translation_stats']['language_pair_counts'] = dict(self.translation_stats['language_pair_counts'])
        save_data['translation_stats']['service_usage'] = dict(self.translation_stats['service_usage'])
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logging.info(f"Translator saved to {filepath}")
    
    def load_translator(self, filepath: str):
        """Load translator configuration and statistics"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.config = save_data['config']
        self.supported_languages = save_data['supported_languages']
        self.quality_thresholds = save_data['quality_thresholds']
        self.cultural_adaptations = save_data['cultural_adaptations']
        self.service_priority = save_data['service_priority']
        
        # Restore statistics
        stats = save_data['translation_stats']
        self.translation_stats['total_translations'] = stats.get('total_translations', 0)
        self.translation_stats['successful_translations'] = stats.get('successful_translations', 0)
        self.translation_stats['avg_translation_time'] = stats.get('avg_translation_time', 0.0)
        self.translation_stats['quality_scores'] = stats.get('quality_scores', [])
        
        # Restore defaultdicts
        self.translation_stats['language_pair_counts'] = defaultdict(int, stats.get('language_pair_counts', {}))
        self.translation_stats['service_usage'] = defaultdict(int, stats.get('service_usage', {}))
        
        logging.info(f"Translator loaded from {filepath}")
    
    def get_available_services(self) -> List[str]:
        """Get list of available translation services"""
        return list(self.services.keys())