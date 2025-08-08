#!/usr/bin/env python3
"""
Test Suite for NewsBot 2.0 Text Preprocessing Components
Tests the text preprocessing, feature extraction, and data validation components
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.text_preprocessor import TextPreprocessor
from data_processing.feature_extractor import FeatureExtractor
from data_processing.data_validator import DataValidator

class TestTextPreprocessor(unittest.TestCase):
    """Test suite for TextPreprocessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = TextPreprocessor()
        
        # Real test data samples
        self.sample_texts = [
            "This is a test article about technology and AI. It contains multiple sentences!",
            "Business news today: stocks are rising. The market looks good for investors.",
            "Sports update: The team won 3-2 in a thrilling match yesterday evening.",
            "Entertainment news: New movie releases this weekend include action films.",
            "Political analysis: The recent policy changes affect economic growth significantly."
        ]
        
        self.sample_noisy_text = "THIS IS ALL CAPS!!! with 123 numbers and @#$% symbols"
        self.sample_multilingual = "Hello world. Hola mundo. Bonjour le monde."
    
    def test_clean_text_basic(self):
        """Test basic text cleaning functionality"""
        cleaned = self.preprocessor.clean_text(self.sample_noisy_text)
        
        # Should be lowercase
        self.assertTrue(cleaned.islower())
        
        # Should not contain excessive punctuation
        self.assertNotIn('!!!', cleaned)
        
        # Should be a string
        self.assertIsInstance(cleaned, str)
    
    def test_tokenize_text(self):
        """Test text tokenization"""
        text = self.sample_texts[0]
        tokens = self.preprocessor.tokenize_text(text)
        
        # Should return a list
        self.assertIsInstance(tokens, list)
        
        # Should contain expected tokens
        self.assertIn('technology', [t.lower() for t in tokens])
        self.assertIn('test', [t.lower() for t in tokens])
        
        # Should not be empty
        self.assertGreater(len(tokens), 0)
    
    def test_remove_stopwords(self):
        """Test stopword removal"""
        tokens = ['this', 'is', 'a', 'test', 'with', 'important', 'words']
        filtered = self.preprocessor.remove_stopwords(tokens)
        
        # Should remove common stopwords
        self.assertNotIn('this', filtered)
        self.assertNotIn('is', filtered)
        self.assertNotIn('a', filtered)
        
        # Should keep content words
        self.assertIn('test', filtered)
        self.assertIn('important', filtered)
        self.assertIn('words', filtered)
    
    def test_lemmatize_tokens(self):
        """Test token lemmatization"""
        tokens = ['running', 'children', 'better', 'cars']
        lemmatized = self.preprocessor.lemmatize_tokens(tokens)
        
        # Should lemmatize words
        self.assertIn('run', lemmatized)  # running -> run
        self.assertIn('child', lemmatized)  # children -> child
        
        # Should be same length or shorter
        self.assertLessEqual(len(lemmatized), len(tokens))
    
    def test_preprocess_text_full_pipeline(self):
        """Test complete preprocessing pipeline"""
        text = self.sample_texts[0]
        processed = self.preprocessor.preprocess_text(text)
        
        # Should return processed text
        self.assertIsInstance(processed, str)
        
        # Should be lowercase
        self.assertTrue(processed.islower())
        
        # Should not contain original punctuation
        self.assertNotIn('!', processed)
        
        # Should contain meaningful words
        self.assertIn('technology', processed)
        self.assertIn('test', processed)
    
    def test_batch_preprocessing(self):
        """Test preprocessing multiple texts"""
        processed_texts = self.preprocessor.preprocess_batch(self.sample_texts)
        
        # Should return same number of texts
        self.assertEqual(len(processed_texts), len(self.sample_texts))
        
        # All should be strings
        for text in processed_texts:
            self.assertIsInstance(text, str)
        
        # All should be processed (lowercase)
        for text in processed_texts:
            self.assertTrue(text.islower())
    
    def test_language_detection(self):
        """Test language detection functionality"""
        english_text = "This is an English text about news and politics."
        
        # Should detect English
        detected_lang = self.preprocessor.detect_language(english_text)
        self.assertEqual(detected_lang, 'en')
    
    def test_preserve_entities(self):
        """Test entity preservation during preprocessing"""
        text_with_entities = "Apple Inc. reported strong earnings. CEO Tim Cook spoke yesterday."
        processed = self.preprocessor.preprocess_text(text_with_entities, preserve_entities=True)
        
        # Should preserve important entities
        self.assertIn('apple', processed.lower())
        self.assertIn('cook', processed.lower())


class TestFeatureExtractor(unittest.TestCase):
    """Test suite for FeatureExtractor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.feature_extractor = FeatureExtractor()
        
        # Use real preprocessed texts
        self.sample_texts = [
            "technology ai artificial intelligence machine learning",
            "business market stock price economy financial",
            "sport football team match game victory",
            "entertainment movie film cinema actor actress",
            "politics government policy election candidate"
        ]
    
    def test_extract_tfidf_features(self):
        """Test TF-IDF feature extraction"""
        features = self.feature_extractor.extract_tfidf_features(self.sample_texts)
        
        # Should return sparse matrix
        self.assertIsNotNone(features)
        
        # Should have correct shape
        self.assertEqual(features.shape[0], len(self.sample_texts))
        
        # Should have features
        self.assertGreater(features.shape[1], 0)
    
    def test_extract_word_embeddings(self):
        """Test word embedding extraction"""
        embeddings = self.feature_extractor.extract_embeddings(self.sample_texts)
        
        # Should return numpy array
        self.assertIsInstance(embeddings, np.ndarray)
        
        # Should have correct shape
        self.assertEqual(embeddings.shape[0], len(self.sample_texts))
        
        # Should have embedding dimensions
        self.assertGreater(embeddings.shape[1], 0)
    
    def test_extract_linguistic_features(self):
        """Test linguistic feature extraction"""
        text = "This is a test sentence with multiple words and punctuation!"
        features = self.feature_extractor.extract_linguistic_features(text)
        
        # Should return dictionary
        self.assertIsInstance(features, dict)
        
        # Should contain expected features
        self.assertIn('sentence_count', features)
        self.assertIn('word_count', features)
        self.assertIn('avg_word_length', features)
        
        # Should have reasonable values
        self.assertGreater(features['word_count'], 0)
        self.assertGreater(features['avg_word_length'], 0)
    
    def test_feature_combination(self):
        """Test combining different feature types"""
        tfidf_features = self.feature_extractor.extract_tfidf_features(self.sample_texts)
        embeddings = self.feature_extractor.extract_embeddings(self.sample_texts)
        
        # Both should have same number of samples
        self.assertEqual(tfidf_features.shape[0], embeddings.shape[0])
        
        # Should be able to combine features
        combined = self.feature_extractor.combine_features(tfidf_features, embeddings)
        self.assertIsNotNone(combined)


class TestDataValidator(unittest.TestCase):
    """Test suite for DataValidator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = DataValidator()
        
        # Create test dataframes
        self.valid_data = pd.DataFrame({
            'text': [
                'This is a valid news article about technology.',
                'Another valid article about business and finance.',
                'Sports news article about football match results.'
            ],
            'category': ['tech', 'business', 'sport']
        })
        
        self.invalid_data = pd.DataFrame({
            'text': [
                'Valid article',
                '',  # Empty text
                'a',  # Too short
                None  # Null value
            ],
            'category': ['tech', 'business', 'sport', 'entertainment']
        })
    
    def test_validate_text_length(self):
        """Test text length validation"""
        # Valid text
        self.assertTrue(self.validator.validate_text_length('This is a valid article with sufficient length.'))
        
        # Too short
        self.assertFalse(self.validator.validate_text_length('short'))
        
        # Empty
        self.assertFalse(self.validator.validate_text_length(''))
        
        # None
        self.assertFalse(self.validator.validate_text_length(None))
    
    def test_validate_required_columns(self):
        """Test required column validation"""
        # Valid data with required columns
        self.assertTrue(self.validator.validate_required_columns(self.valid_data))
        
        # Data missing required columns
        invalid_df = pd.DataFrame({'wrong_column': ['test']})
        self.assertFalse(self.validator.validate_required_columns(invalid_df))
    
    def test_detect_duplicates(self):
        """Test duplicate detection"""
        # Data with duplicates
        duplicate_data = pd.DataFrame({
            'text': ['Same article', 'Same article', 'Different article'],
            'category': ['tech', 'tech', 'business']
        })
        
        duplicates = self.validator.detect_duplicates(duplicate_data)
        self.assertGreater(len(duplicates), 0)
        
        # Data without duplicates
        no_duplicates = self.validator.detect_duplicates(self.valid_data)
        self.assertEqual(len(no_duplicates), 0)
    
    def test_validate_encoding(self):
        """Test text encoding validation"""
        # Valid UTF-8 text
        self.assertTrue(self.validator.validate_encoding('Valid text with normal characters'))
        
        # Text with special characters
        self.assertTrue(self.validator.validate_encoding('Text with émojis and speciál chäracters'))
    
    def test_comprehensive_validation(self):
        """Test comprehensive data validation"""
        # Valid data
        result = self.validator.validate_dataset(self.valid_data)
        self.assertTrue(result['is_valid'])
        
        # Invalid data
        result = self.validator.validate_dataset(self.invalid_data)
        self.assertFalse(result['is_valid'])
        self.assertGreater(len(result['errors']), 0)
    
    def test_clean_invalid_data(self):
        """Test cleaning invalid data"""
        cleaned_data = self.validator.clean_dataset(self.invalid_data)
        
        # Should remove invalid rows
        self.assertLess(len(cleaned_data), len(self.invalid_data))
        
        # Should only contain valid data
        validation_result = self.validator.validate_dataset(cleaned_data)
        self.assertTrue(validation_result['is_valid'])


class TestIntegrationPreprocessing(unittest.TestCase):
    """Integration tests for preprocessing pipeline"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.validator = DataValidator()
        
        # Real-world sample data similar to BBC News dataset
        self.real_sample_data = pd.DataFrame({
            'text': [
                "Apple Inc. announced record quarterly earnings driven by strong iPhone sales and growth in services revenue. CEO Tim Cook highlighted the company's commitment to innovation and sustainability in the technology sector.",
                "The Federal Reserve decided to maintain interest rates at current levels following concerns about inflation and economic uncertainty. Market analysts expect continued volatility in financial markets.",
                "Manchester United secured a decisive 3-1 victory against their rivals in yesterday's Premier League match. The team's new signing scored twice in his debut performance.",
                "The latest Marvel superhero film broke box office records during its opening weekend, earning over $200 million globally. Critics praised the visual effects and storytelling.",
                "Congressional leaders announced a bipartisan agreement on infrastructure spending that will allocate billions for road, bridge, and broadband improvements across the nation."
            ],
            'category': ['tech', 'business', 'sport', 'entertainment', 'politics']
        })
    
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline with real data"""
        # Validate original data
        validation_result = self.validator.validate_dataset(self.real_sample_data)
        self.assertTrue(validation_result['is_valid'])
        
        # Preprocess texts
        texts = self.real_sample_data['text'].tolist()
        preprocessed_texts = [self.preprocessor.preprocess_text(text) for text in texts]
        
        # Extract features
        tfidf_features = self.feature_extractor.extract_tfidf_features(preprocessed_texts)
        embeddings = self.feature_extractor.extract_embeddings(texts)
        
        # Verify feature extraction
        self.assertEqual(tfidf_features.shape[0], len(texts))
        self.assertEqual(embeddings.shape[0], len(texts))
        
        # Features should be different for different categories
        self.assertNotEqual(tfidf_features[0].toarray().tolist(), tfidf_features[1].toarray().tolist())
    
    def test_multilingual_preprocessing(self):
        """Test preprocessing with multilingual content"""
        multilingual_data = pd.DataFrame({
            'text': [
                "Technology news in English about artificial intelligence",
                "Noticias de tecnología en español sobre inteligencia artificial",
                "Nouvelles technologiques en français sur l'intelligence artificielle"
            ],
            'category': ['tech', 'tech', 'tech']
        })
        
        # Should handle multilingual content
        texts = multilingual_data['text'].tolist()
        preprocessed_texts = [self.preprocessor.preprocess_text(text) for text in texts]
        
        # All should be processed successfully
        self.assertEqual(len(preprocessed_texts), len(texts))
        for text in preprocessed_texts:
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0)


if __name__ == '__main__':
    # Set up test discovery and execution
    unittest.main(verbosity=2)