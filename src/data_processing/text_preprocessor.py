#!/usr/bin/env python3
"""
Enhanced Text Preprocessor for NewsBot 2.0
Advanced text cleaning, tokenization, and preprocessing with multilingual support
"""

import re
import string
import unicodedata
from typing import List, Dict, Optional, Tuple
import logging

import pandas as pd
import numpy as np
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

class TextPreprocessor:
    """
    Advanced text preprocessor with multilingual support and various preprocessing options
    """
    
    def __init__(self, config: Dict = None, language: str = 'english', use_spacy: bool = True):
        """
        Initialize the text preprocessor
        
        Args:
            config: Configuration dictionary (for compatibility with NewsBot2System)
            language: Primary language for processing
            use_spacy: Whether to use spaCy for advanced processing
        """
        # Handle config parameter for system integration
        if config and isinstance(config, dict):
            self.language = config.get('language', 'english')
            self.use_spacy = config.get('use_spacy', True)
        else:
            self.language = language
            self.use_spacy = use_spacy
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words(self.language))
        except OSError as e:
            logging.warning(f"Could not load stopwords for {self.language}, using english: {e}")
            self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model if requested
        if self.use_spacy:
            try:
                if self.language == 'english':
                    self.nlp = spacy.load('en_core_web_sm')
                else:
                    # Fallback to English model for unsupported languages
                    self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                logging.warning("spaCy model not found, falling back to NLTK only")
                self.use_spacy = False
                self.nlp = None
        
        # Domain-specific stop words for news
        self.news_stopwords = {
            'said', 'says', 'according', 'reported', 'reports', 'news', 'story',
            'article', 'yesterday', 'today', 'tomorrow', 'monday', 'tuesday',
            'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
        }
        
        # Update stop words
        self.stop_words.update(self.news_stopwords)
        
        # Preprocessing options
        self.preprocessing_steps = {
            'normalize_unicode': True,
            'remove_urls': True,
            'remove_emails': True,
            'remove_phone_numbers': True,
            'remove_extra_whitespace': True,
            'convert_to_lowercase': True,
            'remove_punctuation': True,
            'remove_numbers': False,
            'remove_stopwords': True,
            'lemmatize': True,
            'remove_short_words': True,
            'min_word_length': 2
        }
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        return unicodedata.normalize('NFKD', text)
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.sub('', text)
    
    def remove_emails(self, text: str) -> str:
        """Remove email addresses from text"""
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        return email_pattern.sub('', text)
    
    def remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers from text"""
        phone_pattern = re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        return phone_pattern.sub('', text)
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize spaces"""
        return re.sub(r'\s+', ' ', text).strip()
    
    def clean_text(self, text: str) -> str:
        """
        Clean text using configured preprocessing steps
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Apply preprocessing steps
        if self.preprocessing_steps['normalize_unicode']:
            text = self.normalize_unicode(text)
        
        if self.preprocessing_steps['remove_urls']:
            text = self.remove_urls(text)
        
        if self.preprocessing_steps['remove_emails']:
            text = self.remove_emails(text)
        
        if self.preprocessing_steps['remove_phone_numbers']:
            text = self.remove_phone_numbers(text)
        
        if self.preprocessing_steps['convert_to_lowercase']:
            text = text.lower()
        
        if self.preprocessing_steps['remove_extra_whitespace']:
            text = self.remove_extra_whitespace(text)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            tokens = [token.text for token in doc if not token.is_space]
        else:
            tokens = word_tokenize(text)
        
        return tokens
    
    def remove_punctuation_and_numbers(self, tokens: List[str]) -> List[str]:
        """Remove punctuation and optionally numbers from tokens"""
        filtered_tokens = []
        
        for token in tokens:
            # Remove punctuation
            if self.preprocessing_steps['remove_punctuation']:
                if token in string.punctuation:
                    continue
            
            # Remove numbers
            if self.preprocessing_steps['remove_numbers']:
                if token.isdigit():
                    continue
            
            # Remove tokens with mixed punctuation
            if re.match(r'^[^\w\s]+$', token):
                continue
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def remove_stopwords_and_short_words(self, tokens: List[str]) -> List[str]:
        """Remove stopwords and short words"""
        filtered_tokens = []
        
        for token in tokens:
            # Remove stopwords
            if self.preprocessing_steps['remove_stopwords']:
                if token.lower() in self.stop_words:
                    continue
            
            # Remove short words
            if self.preprocessing_steps['remove_short_words']:
                if len(token) < self.preprocessing_steps['min_word_length']:
                    continue
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        if not self.preprocessing_steps['lemmatize']:
            return tokens
        
        if self.use_spacy and self.nlp:
            # Use spaCy for lemmatization
            doc = self.nlp(' '.join(tokens))
            return [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
        else:
            # Use NLTK for lemmatization
            return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete text preprocessing pipeline
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text as string
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove punctuation and numbers
        tokens = self.remove_punctuation_and_numbers(tokens)
        
        # Remove stopwords and short words
        tokens = self.remove_stopwords_and_short_words(tokens)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_documents(self, documents: List[str], show_progress: bool = True) -> List[str]:
        """
        Preprocess multiple documents
        
        Args:
            documents: List of documents to preprocess
            show_progress: Whether to show processing progress
            
        Returns:
            List of preprocessed documents
        """
        preprocessed_docs = []
        
        total_docs = len(documents)
        for i, doc in enumerate(documents):
            if show_progress and i % 100 == 0:
                print(f"Processing document {i+1}/{total_docs}")
            
            preprocessed_doc = self.preprocess_text(doc)
            preprocessed_docs.append(preprocessed_doc)
        
        return preprocessed_docs
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            return sent_tokenize(text)
    
    def extract_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """Extract part-of-speech tags"""
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            return [(token.text, token.pos_) for token in doc]
        else:
            tokens = word_tokenize(text)
            return pos_tag(tokens)
    
    def extract_named_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text"""
        entities = []
        
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        else:
            # Fallback to NLTK
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            chunks = ne_chunk(pos_tags)
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity_text = ' '.join([token for token, pos in chunk.leaves()])
                    entities.append({
                        'text': entity_text,
                        'label': chunk.label(),
                        'start': 0,  # NLTK doesn't provide character positions
                        'end': 0
                    })
        
        return entities
    
    def get_text_statistics(self, text: str) -> Dict[str, int]:
        """Get basic text statistics"""
        sentences = self.extract_sentences(text)
        tokens = self.tokenize(text)
        
        return {
            'char_count': len(text),
            'word_count': len(tokens),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': len(tokens) / len(sentences) if sentences else 0,
            'unique_words': len(set(tokens))
        }
    
    def configure_preprocessing(self, **kwargs):
        """Configure preprocessing options"""
        for key, value in kwargs.items():
            if key in self.preprocessing_steps:
                self.preprocessing_steps[key] = value
            else:
                logging.warning(f"Unknown preprocessing option: {key}")
    
    def add_custom_stopwords(self, words: List[str]):
        """Add custom stopwords"""
        self.stop_words.update(words)
    
    def remove_custom_stopwords(self, words: List[str]):
        """Remove custom stopwords"""
        for word in words:
            self.stop_words.discard(word)