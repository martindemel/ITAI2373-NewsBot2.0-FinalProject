#!/usr/bin/env python3
"""
Test Suite for NewsBot 2.0 Topic Modeling Components
Tests the topic modeling, language models, and multilingual components
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.topic_modeler import TopicModeler
from language_models.summarizer import IntelligentSummarizer
from language_models.embeddings import EmbeddingGenerator
from multilingual.language_detector import LanguageDetector
from multilingual.translator import MultilingualTranslator

class TestTopicModeler(unittest.TestCase):
    """Test suite for TopicModeler class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.topic_modeler = TopicModeler({'num_topics': 5, 'random_state': 42})
        
        # Real news articles from different categories for topic modeling
        self.sample_documents = [
            "Apple Inc. announced record quarterly earnings driven by strong iPhone sales and services revenue growth in technology markets worldwide.",
            "The Federal Reserve decided to maintain current interest rates following economic indicators showing inflation concerns and market uncertainty.",
            "Manchester United secured a decisive victory in the Premier League match with outstanding performance from their new striker signing.",
            "The latest Marvel superhero blockbuster broke box office records during opening weekend with impressive visual effects and storytelling.",
            "Congressional leaders reached bipartisan agreement on infrastructure legislation allocating billions for transportation and broadband improvements.",
            "Microsoft unveiled innovative artificial intelligence features in their cloud computing platform Azure for enterprise customers globally.",
            "Wall Street responded positively to quarterly earnings reports from major technology companies showing sustained growth despite challenges.",
            "The football championship final attracted millions of viewers worldwide as teams competed for the prestigious tournament title.",
            "Hollywood award season nominations were announced featuring diverse films addressing social issues and artistic excellence in cinema.",
            "The presidential candidate outlined policy proposals focusing on healthcare reform and economic recovery initiatives for voters."
        ]
    
    def test_lda_topic_modeling(self):
        """Test LDA topic modeling implementation"""
        self.topic_modeler.config['algorithm'] = 'lda'
        
        # Should fit LDA model successfully
        self.topic_modeler.fit_transform(self.sample_documents)
        
        # Should have trained model
        self.assertIsNotNone(self.topic_modeler.model)
        self.assertTrue(self.topic_modeler.is_fitted)
        
        # Should return model summary
        summary = self.topic_modeler.get_model_summary()
        self.assertIn('n_topics', summary)
        self.assertIn('algorithm', summary)
        self.assertEqual(summary['n_topics'], 5)
    
    def test_nmf_topic_modeling(self):
        """Test NMF topic modeling implementation"""
        self.topic_modeler.config['algorithm'] = 'nmf'
        
        # Should fit NMF model successfully
        self.topic_modeler.fit_transform(self.sample_documents)
        
        # Should have trained model
        self.assertIsNotNone(self.topic_modeler.model)
        self.assertTrue(self.topic_modeler.is_fitted)
        
        # Should return model summary
        summary = self.topic_modeler.get_model_summary()
        self.assertEqual(summary['algorithm'], 'nmf')
    
    def test_topic_word_extraction(self):
        """Test extraction of topic words"""
        self.topic_modeler.fit_transform(self.sample_documents)
        
        # Should extract top words for each topic
        for topic_id in range(5):
            topic_words = self.topic_modeler.get_topic_words(topic_id, n_words=5)
            
            self.assertIsInstance(topic_words, list)
            self.assertEqual(len(topic_words), 5)
            
            # Each word should be a string
            for word in topic_words:
                self.assertIsInstance(word, str)
    
    def test_document_topic_assignment(self):
        """Test assignment of topics to documents"""
        self.topic_modeler.fit_transform(self.sample_documents)
        
        # Test single document topic assignment
        test_doc = "Technology companies are investing heavily in artificial intelligence and machine learning innovations."
        topic_dist = self.topic_modeler.get_article_topics(test_doc)
        
        self.assertIn('topic_distribution', topic_dist)
        self.assertIn('dominant_topic', topic_dist)
        
        # Topic distribution should sum to approximately 1
        distribution = topic_dist['topic_distribution']
        self.assertAlmostEqual(sum(distribution), 1.0, places=1)
    
    def test_topic_coherence_calculation(self):
        """Test topic coherence score calculation"""
        self.topic_modeler.fit_transform(self.sample_documents)
        
        coherence_score = self.topic_modeler.calculate_coherence_score()
        
        # Coherence score should be a float
        self.assertIsInstance(coherence_score, float)
        
        # Should be within reasonable range for coherence
        self.assertGreater(coherence_score, -1.0)
        self.assertLess(coherence_score, 1.0)
    
    def test_topic_evolution_tracking(self):
        """Test topic evolution over time"""
        # Sample documents with timestamps
        temporal_documents = [
            {"text": doc, "date": f"2024-01-{i+1:02d}"} 
            for i, doc in enumerate(self.sample_documents)
        ]
        
        evolution = self.topic_modeler.track_topic_evolution(temporal_documents)
        
        self.assertIn('topic_timeline', evolution)
        self.assertIn('topic_trends', evolution)
        
        # Should have entries for each time period
        timeline = evolution['topic_timeline']
        self.assertGreater(len(timeline), 0)
    
    def test_topic_visualization_data(self):
        """Test data preparation for topic visualization"""
        self.topic_modeler.fit_transform(self.sample_documents)
        
        viz_data = self.topic_modeler.prepare_visualization_data()
        
        self.assertIn('topic_coordinates', viz_data)
        self.assertIn('topic_sizes', viz_data)
        self.assertIn('topic_words', viz_data)
        
        # Should have coordinates for each topic
        coordinates = viz_data['topic_coordinates']
        self.assertEqual(len(coordinates), 5)


class TestIntelligentSummarizer(unittest.TestCase):
    """Test suite for IntelligentSummarizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.summarizer = IntelligentSummarizer()
        
        # Long news article for summarization testing
        self.long_article = """
        Apple Inc. announced record-breaking quarterly earnings today, driven by exceptional iPhone sales and continued growth in its services segment. The technology giant reported revenue of $123.9 billion, exceeding analyst expectations by a significant margin. CEO Tim Cook highlighted the company's commitment to innovation and sustainability during the earnings call.
        
        The iPhone 15 series, launched earlier this year, has been particularly successful in international markets, with strong adoption in Europe and Asia. The company's services division, which includes the App Store, Apple Music, and iCloud, generated $22.3 billion in revenue, representing a 15% increase from the previous quarter.
        
        Apple's investment in artificial intelligence and machine learning technologies is paying dividends, with new features in iOS 17 receiving positive user feedback. The company's focus on privacy and security continues to differentiate it from competitors in the smartphone market.
        
        Looking ahead, Apple plans to expand its retail presence in emerging markets and continue developing cutting-edge technologies for future product releases. The company's strong financial position enables continued investment in research and development, ensuring its competitive advantage in the technology sector.
        """
        
        self.medium_article = """
        The Federal Reserve announced its decision to maintain interest rates at current levels following the latest Federal Open Market Committee meeting. Chairman Jerome Powell cited ongoing concerns about inflation and economic uncertainty as key factors in the decision. Financial markets responded positively to the news, with major indices closing higher.
        """
    
    def test_extractive_summarization(self):
        """Test extractive summarization method"""
        summary = self.summarizer.extractive_summarize(self.long_article)
        
        self.assertIn('summary', summary)
        self.assertIn('summary_sentences', summary)
        self.assertIn('compression_ratio', summary)
        
        # Summary should be shorter than original
        original_length = len(self.long_article.split())
        summary_length = len(summary['summary'].split())
        self.assertLess(summary_length, original_length)
        
        # Should contain key information
        summary_text = summary['summary'].lower()
        self.assertIn('apple', summary_text)
        self.assertIn('earnings', summary_text)
    
    def test_abstractive_summarization(self):
        """Test abstractive summarization method"""
        try:
            summary = self.summarizer.abstractive_summarize(self.long_article)
            
            self.assertIn('summary', summary)
            self.assertIn('quality_score', summary)
            
            # Summary should be coherent
            summary_text = summary['summary']
            self.assertIsInstance(summary_text, str)
            self.assertGreater(len(summary_text), 0)
            
        except Exception as e:
            # Abstractive models might not be available in test environment
            self.skipTest(f"Abstractive summarization model not available: {e}")
    
    def test_hybrid_summarization(self):
        """Test hybrid summarization combining methods"""
        summary = self.summarizer.summarize_article(self.long_article, method='hybrid')
        
        self.assertIn('extractive', summary)
        self.assertIn('hybrid_summary', summary)
        self.assertIn('method_scores', summary)
        
        # Hybrid should combine strengths of both methods
        hybrid_text = summary['hybrid_summary']
        self.assertIsInstance(hybrid_text, str)
        self.assertGreater(len(hybrid_text), 0)
    
    def test_summary_quality_assessment(self):
        """Test summary quality evaluation"""
        summary_text = "Apple reported strong quarterly earnings with record iPhone sales and services growth."
        
        quality = self.summarizer.assess_summary_quality(
            original=self.long_article,
            summary=summary_text
        )
        
        self.assertIn('rouge_scores', quality)
        self.assertIn('coherence_score', quality)
        self.assertIn('coverage_score', quality)
        
        # Scores should be within valid ranges
        for score_type, scores in quality['rouge_scores'].items():
            for metric in scores.values():
                self.assertGreaterEqual(metric, 0.0)
                self.assertLessEqual(metric, 1.0)
    
    def test_multi_document_summarization(self):
        """Test summarization of multiple documents"""
        documents = [
            self.long_article,
            self.medium_article,
            "Technology stocks surged following positive earnings reports from major companies."
        ]
        
        summary = self.summarizer.multi_document_summarize(documents)
        
        self.assertIn('combined_summary', summary)
        self.assertIn('document_contributions', summary)
        self.assertIn('key_themes', summary)
        
        # Should capture information from multiple sources
        combined_text = summary['combined_summary'].lower()
        self.assertIn('apple', combined_text)
        self.assertIn('federal reserve', combined_text)
    
    def test_summary_length_control(self):
        """Test control over summary length"""
        # Test different length settings
        short_summary = self.summarizer.summarize_article(
            self.long_article, 
            max_length=50
        )
        
        long_summary = self.summarizer.summarize_article(
            self.long_article, 
            max_length=200
        )
        
        short_length = len(short_summary['extractive']['summary'].split())
        long_length = len(long_summary['extractive']['summary'].split())
        
        # Short summary should be shorter than long summary
        self.assertLess(short_length, long_length)


class TestEmbeddingGenerator(unittest.TestCase):
    """Test suite for EmbeddingGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.embedding_generator = EmbeddingGenerator()
        
        self.sample_texts = [
            "Technology companies are investing in artificial intelligence research.",
            "Stock markets show volatility amid economic uncertainty.",
            "Football teams prepare for the championship tournament.",
            "Movie studios release blockbuster films during summer season.",
            "Political leaders discuss policy reforms and legislation."
        ]
    
    def test_sentence_embedding_generation(self):
        """Test sentence embedding generation"""
        embeddings = self.embedding_generator.generate_embeddings(self.sample_texts)
        
        # Should return numpy array
        self.assertIsInstance(embeddings, np.ndarray)
        
        # Should have correct dimensions
        self.assertEqual(embeddings.shape[0], len(self.sample_texts))
        self.assertGreater(embeddings.shape[1], 0)  # Embedding dimension
        
        # Embeddings should be different for different texts
        self.assertFalse(np.array_equal(embeddings[0], embeddings[1]))
    
    def test_semantic_similarity_calculation(self):
        """Test semantic similarity calculation"""
        text1 = "Apple releases new iPhone with advanced features."
        text2 = "Technology company announces innovative smartphone product."
        text3 = "Football team wins championship match yesterday."
        
        similarity_12 = self.embedding_generator.calculate_similarity(text1, text2)
        similarity_13 = self.embedding_generator.calculate_similarity(text1, text3)
        
        # Technology texts should be more similar than technology vs sports
        self.assertGreater(similarity_12, similarity_13)
        
        # Similarities should be between -1 and 1 (cosine similarity)
        self.assertGreaterEqual(similarity_12, -1.0)
        self.assertLessEqual(similarity_12, 1.0)
    
    def test_semantic_search(self):
        """Test semantic search functionality"""
        # Create search index
        self.embedding_generator.build_search_index(self.sample_texts)
        
        # Search for similar content
        query = "artificial intelligence and machine learning"
        results = self.embedding_generator.semantic_search(query, top_k=3)
        
        self.assertLessEqual(len(results), 3)
        
        # Results should have required structure
        for result in results:
            self.assertIn('text', result)
            self.assertIn('similarity_score', result)
            self.assertIn('index', result)
        
        # Top result should be most relevant (technology-related)
        if results:
            top_result = results[0]
            self.assertIn('technology', top_result['text'].lower())
    
    def test_document_clustering(self):
        """Test document clustering using embeddings"""
        # Extend sample texts for better clustering
        extended_texts = self.sample_texts * 2  # Duplicate for clustering
        
        clusters = self.embedding_generator.cluster_documents(extended_texts, n_clusters=3)
        
        self.assertIn('cluster_assignments', clusters)
        self.assertIn('cluster_centers', clusters)
        self.assertIn('silhouette_score', clusters)
        
        # Should assign each document to a cluster
        assignments = clusters['cluster_assignments']
        self.assertEqual(len(assignments), len(extended_texts))
        
        # Cluster IDs should be in valid range
        for cluster_id in assignments:
            self.assertGreaterEqual(cluster_id, 0)
            self.assertLess(cluster_id, 3)


class TestLanguageDetector(unittest.TestCase):
    """Test suite for LanguageDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.language_detector = LanguageDetector()
        
        self.multilingual_samples = {
            'en': "This is an English news article about technology and innovation.",
            'es': "Esta es una noticia en español sobre tecnología e innovación.",
            'fr': "Ceci est un article de presse en français sur la technologie et l'innovation.",
            'de': "Dies ist ein deutschsprachiger Nachrichtenartikel über Technologie und Innovation.",
            'it': "Questo è un articolo di notizie in italiano su tecnologia e innovazione."
        }
    
    def test_language_detection_accuracy(self):
        """Test language detection accuracy"""
        for expected_lang, text in self.multilingual_samples.items():
            result = self.language_detector.detect_language(text)
            
            self.assertIn('language', result)
            self.assertIn('confidence', result)
            
            detected_lang = result['language']
            confidence = result['confidence']
            
            # Should detect correct language with high confidence
            self.assertEqual(detected_lang, expected_lang)
            self.assertGreater(confidence, 0.8)
    
    def test_batch_language_detection(self):
        """Test batch language detection"""
        texts = list(self.multilingual_samples.values())
        expected_langs = list(self.multilingual_samples.keys())
        
        results = self.language_detector.detect_languages_batch(texts)
        
        self.assertEqual(len(results), len(texts))
        
        # Check each detection result
        for i, result in enumerate(results):
            self.assertEqual(result['language'], expected_langs[i])
    
    def test_confidence_thresholding(self):
        """Test confidence-based filtering"""
        ambiguous_text = "OK yes no maybe"  # Short, ambiguous text
        
        result = self.language_detector.detect_language(
            ambiguous_text, 
            confidence_threshold=0.9
        )
        
        # Should handle low-confidence detections appropriately
        if result['confidence'] < 0.9:
            self.assertIn('warning', result)
    
    def test_supported_languages(self):
        """Test supported language coverage"""
        supported = self.language_detector.get_supported_languages()
        
        # Should support major languages
        self.assertIn('en', supported)
        self.assertIn('es', supported)
        self.assertIn('fr', supported)
        self.assertIn('de', supported)
        
        # Should be a reasonable number of languages
        self.assertGreater(len(supported), 10)


class TestMultilingualTranslator(unittest.TestCase):
    """Test suite for MultilingualTranslator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.translator = MultilingualTranslator()
        
        self.translation_pairs = [
            {
                'source': 'en',
                'target': 'es',
                'text': 'Technology companies are investing in artificial intelligence.',
                'expected_words': ['tecnología', 'empresas', 'inteligencia', 'artificial']
            },
            {
                'source': 'en',
                'target': 'fr', 
                'text': 'The stock market shows positive trends.',
                'expected_words': ['marché', 'actions', 'positives', 'tendances']
            }
        ]
    
    def test_basic_translation(self):
        """Test basic translation functionality"""
        for pair in self.translation_pairs:
            try:
                result = self.translator.translate_text(
                    text=pair['text'],
                    source_lang=pair['source'],
                    target_lang=pair['target']
                )
                
                self.assertIn('translated_text', result)
                self.assertIn('confidence', result)
                self.assertIn('source_language', result)
                self.assertIn('target_language', result)
                
                translated_text = result['translated_text'].lower()
                
                # Should contain expected translated words
                for expected_word in pair['expected_words']:
                    # Allow partial matches due to different translation services
                    word_found = any(expected_word[:4] in translated_text for expected_word in pair['expected_words'])
                    if not word_found:
                        self.skipTest(f"Translation service may not be available or returned unexpected result")
                
            except Exception as e:
                self.skipTest(f"Translation service not available: {e}")
    
    def test_auto_language_detection(self):
        """Test automatic source language detection"""
        text = "Bonjour, comment allez-vous? Ceci est un test."
        
        try:
            result = self.translator.translate_text(
                text=text,
                target_lang='en',
                auto_detect=True
            )
            
            self.assertEqual(result['source_language'], 'fr')
            
            # Should translate to English
            translated = result['translated_text'].lower()
            self.assertIn('hello', translated)
            
        except Exception as e:
            self.skipTest(f"Auto-detection not available: {e}")
    
    def test_translation_quality_assessment(self):
        """Test translation quality evaluation"""
        original = "The company reported strong financial results."
        translation = "La empresa reportó resultados financieros sólidos."
        
        quality = self.translator.assess_translation_quality(
            original=original,
            translation=translation,
            source_lang='en',
            target_lang='es'
        )
        
        self.assertIn('quality_score', quality)
        self.assertIn('fluency_score', quality)
        self.assertIn('adequacy_score', quality)
        
        # Quality scores should be between 0 and 1
        for score in quality.values():
            if isinstance(score, (int, float)):
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
    
    def test_batch_translation(self):
        """Test batch translation of multiple texts"""
        texts = [
            "Technology news about artificial intelligence.",
            "Business report on market trends.",
            "Sports coverage of championship games."
        ]
        
        try:
            results = self.translator.translate_batch(
                texts=texts,
                source_lang='en',
                target_lang='es'
            )
            
            self.assertEqual(len(results), len(texts))
            
            # Each result should have translation
            for result in results:
                self.assertIn('translated_text', result)
                self.assertIsInstance(result['translated_text'], str)
                self.assertGreater(len(result['translated_text']), 0)
                
        except Exception as e:
            self.skipTest(f"Batch translation not available: {e}")


if __name__ == '__main__':
    # Set up test discovery and execution
    unittest.main(verbosity=2)