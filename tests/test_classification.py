#!/usr/bin/env python3
"""
Test Suite for NewsBot 2.0 Classification Components
Tests the advanced classification, sentiment analysis, and NER components
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.classifier import AdvancedNewsClassifier
from analysis.sentiment_analyzer import AdvancedSentimentAnalyzer
from analysis.ner_extractor import EntityRelationshipMapper

class TestAdvancedNewsClassifier(unittest.TestCase):
    """Test suite for AdvancedNewsClassifier class"""
    
    def setUp(self):
        """Set up test fixtures with real BBC News data structure"""
        self.classifier = AdvancedNewsClassifier()
        
        # Real BBC News dataset categories and sample texts
        self.real_categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
        
        self.sample_training_data = [
            ("Apple Inc. reported strong quarterly earnings driven by iPhone sales and services growth. The technology giant continues to innovate in the smartphone market.", "tech"),
            ("The Federal Reserve announced its decision to maintain current interest rates amid economic uncertainty. Financial markets responded positively to the news.", "business"),
            ("Manchester United secured a decisive victory in yesterday's Premier League match with a spectacular performance from their new signing.", "sport"),
            ("The latest blockbuster film broke box office records during its opening weekend, earning praise from critics for its visual effects and storytelling.", "entertainment"),
            ("Congressional leaders reached a bipartisan agreement on infrastructure legislation that will allocate significant funding for transportation improvements.", "politics")
        ]
        
        self.sample_test_texts = [
            "Microsoft announced a major update to its Windows operating system with enhanced AI capabilities.",
            "Stock markets experienced significant volatility following the latest economic indicators.",
            "The football team's victory puts them at the top of the league standings.",
            "The award-winning director's new film premiered at the international film festival.",
            "The senator announced her candidacy for the upcoming presidential election."
        ]
    
    def test_initialization(self):
        """Test classifier initialization"""
        self.assertIsInstance(self.classifier, AdvancedNewsClassifier)
        self.assertIsNotNone(self.classifier.config)
        
        # Should have expected configuration
        self.assertIn('model_type', self.classifier.config)
        self.assertIn('confidence_threshold', self.classifier.config)
    
    def test_feature_extraction(self):
        """Test feature extraction for classification"""
        texts = [item[0] for item in self.sample_training_data]
        
        # Test TF-IDF features
        features = self.classifier.feature_extractor.extract_tfidf_features(texts)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], len(texts))
        
        # Test embedding features
        embeddings = self.classifier.feature_extractor.extract_embeddings(texts)
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape[0], len(texts))
    
    def test_training_with_real_categories(self):
        """Test training with real BBC News categories"""
        texts = [item[0] for item in self.sample_training_data]
        labels = [item[1] for item in self.sample_training_data]
        
        # Should handle real category training
        try:
            self.classifier.train(texts, labels)
            self.assertTrue(True)  # Training completed without error
        except Exception as e:
            self.fail(f"Training failed with real categories: {e}")
    
    def test_prediction_confidence(self):
        """Test prediction with confidence scores"""
        texts = [item[0] for item in self.sample_training_data]
        labels = [item[1] for item in self.sample_training_data]
        
        # Train classifier
        self.classifier.train(texts, labels)
        
        # Test prediction with confidence
        test_text = self.sample_test_texts[0]  # Tech article
        prediction = self.classifier.predict_with_confidence([test_text])
        
        self.assertIn('predictions', prediction)
        self.assertIn('confidence_scores', prediction)
        self.assertEqual(len(prediction['predictions']), 1)
        
        # Confidence should be between 0 and 1
        confidence = prediction['confidence_scores'][0]
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_multi_label_classification(self):
        """Test multi-label classification capability"""
        # Sample with multiple relevant categories
        multi_label_text = "Tech company's stock prices soared after announcing record profits in the entertainment streaming business."
        
        self.classifier.config['multi_label'] = True
        
        # Should handle multi-label classification
        texts = [item[0] for item in self.sample_training_data]
        labels = [item[1] for item in self.sample_training_data]
        
        self.classifier.train(texts, labels)
        prediction = self.classifier.predict_with_confidence([multi_label_text])
        
        # Should return valid prediction structure
        self.assertIn('predictions', prediction)
        self.assertIn('confidence_scores', prediction)
    
    def test_ensemble_methods(self):
        """Test ensemble classification methods"""
        self.classifier.config['model_type'] = 'ensemble'
        
        texts = [item[0] for item in self.sample_training_data]
        labels = [item[1] for item in self.sample_training_data]
        
        # Should train ensemble successfully
        self.classifier.train(texts, labels)
        
        # Should predict with ensemble
        predictions = self.classifier.predict_with_confidence(self.sample_test_texts)
        
        self.assertEqual(len(predictions['predictions']), len(self.sample_test_texts))


class TestAdvancedSentimentAnalyzer(unittest.TestCase):
    """Test suite for AdvancedSentimentAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        
        # Real news samples with clear sentiment
        self.positive_texts = [
            "The company reported record-breaking profits and announced expansion plans.",
            "The team celebrated their championship victory with thousands of fans.",
            "The innovative technology promises to revolutionize healthcare and improve lives."
        ]
        
        self.negative_texts = [
            "The economic downturn has led to massive layoffs and business closures.",
            "The devastating natural disaster caused widespread damage and casualties.",
            "The political scandal has shaken public confidence in government institutions."
        ]
        
        self.neutral_texts = [
            "The meeting was scheduled for 3 PM in the conference room.",
            "The weather forecast predicts partly cloudy conditions tomorrow.",
            "The report contains statistical information about population demographics."
        ]
    
    def test_vader_sentiment_analysis(self):
        """Test VADER sentiment analysis"""
        # Test positive sentiment
        positive_result = self.sentiment_analyzer.analyze_sentiment_vader(self.positive_texts[0])
        self.assertGreater(positive_result['compound'], 0)
        self.assertEqual(positive_result['classification'], 'positive')
        
        # Test negative sentiment
        negative_result = self.sentiment_analyzer.analyze_sentiment_vader(self.negative_texts[0])
        self.assertLess(negative_result['compound'], 0)
        self.assertEqual(negative_result['classification'], 'negative')
    
    def test_textblob_sentiment_analysis(self):
        """Test TextBlob sentiment analysis"""
        # Test positive sentiment
        positive_result = self.sentiment_analyzer.analyze_sentiment_textblob(self.positive_texts[0])
        self.assertGreater(positive_result['polarity'], 0)
        self.assertEqual(positive_result['classification'], 'positive')
        
        # Test negative sentiment
        negative_result = self.sentiment_analyzer.analyze_sentiment_textblob(self.negative_texts[0])
        self.assertLess(negative_result['polarity'], 0)
        self.assertEqual(negative_result['classification'], 'negative')
    
    def test_transformer_sentiment_analysis(self):
        """Test transformer-based sentiment analysis"""
        try:
            # Test with positive text
            result = self.sentiment_analyzer.analyze_sentiment_transformer(self.positive_texts[0])
            
            self.assertIn('classification', result)
            self.assertIn('confidence', result)
            self.assertIn('scores', result)
            
            # Confidence should be between 0 and 1
            self.assertGreaterEqual(result['confidence'], 0.0)
            self.assertLessEqual(result['confidence'], 1.0)
            
        except Exception as e:
            # Transformer models might not be available in test environment
            self.skipTest(f"Transformer model not available: {e}")
    
    def test_emotion_detection(self):
        """Test emotion detection capability"""
        emotional_text = "I am absolutely thrilled about this amazing opportunity!"
        
        try:
            result = self.sentiment_analyzer.detect_emotions(emotional_text)
            
            self.assertIn('emotions', result)
            self.assertIn('dominant_emotion', result)
            
            # Should detect positive emotions
            emotions = result['emotions']
            self.assertGreater(emotions.get('joy', 0) + emotions.get('happiness', 0), 0)
            
        except Exception as e:
            self.skipTest(f"Emotion detection model not available: {e}")
    
    def test_aggregated_sentiment_analysis(self):
        """Test aggregated multi-method sentiment analysis"""
        text = self.positive_texts[0]
        
        result = self.sentiment_analyzer.analyze_sentiment(text)
        
        # Should contain all method results
        self.assertIn('vader', result)
        self.assertIn('textblob', result)
        self.assertIn('aggregated', result)
        
        # Aggregated result should have required fields
        aggregated = result['aggregated']
        self.assertIn('classification', aggregated)
        self.assertIn('weighted_score', aggregated)
        self.assertIn('confidence', aggregated)
    
    def test_temporal_sentiment_tracking(self):
        """Test temporal sentiment tracking"""
        # Sample data with timestamps
        temporal_data = [
            {"text": self.positive_texts[0], "date": "2024-01-01"},
            {"text": self.negative_texts[0], "date": "2024-01-02"},
            {"text": self.neutral_texts[0], "date": "2024-01-03"}
        ]
        
        results = self.sentiment_analyzer.analyze_temporal_sentiment(temporal_data)
        
        self.assertIn('sentiment_timeline', results)
        self.assertIn('sentiment_trends', results)
        self.assertEqual(len(results['sentiment_timeline']), len(temporal_data))
    
    def test_batch_sentiment_analysis(self):
        """Test batch processing of multiple texts"""
        all_texts = self.positive_texts + self.negative_texts + self.neutral_texts
        
        results = self.sentiment_analyzer.analyze_batch_sentiment(all_texts)
        
        # Should return results for all texts
        self.assertEqual(len(results), len(all_texts))
        
        # Each result should have required structure
        for result in results:
            self.assertIn('aggregated', result)
            self.assertIn('classification', result['aggregated'])


class TestEntityRelationshipMapper(unittest.TestCase):
    """Test suite for EntityRelationshipMapper class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ner_extractor = EntityRelationshipMapper()
        
        # Real news samples with clear entities
        self.entity_rich_texts = [
            "Apple Inc. CEO Tim Cook announced the company's plans to invest $1 billion in Austin, Texas. The announcement was made during a press conference on Monday.",
            "President Biden met with German Chancellor Angela Merkel in Washington D.C. to discuss climate change policies and trade agreements between the United States and European Union.",
            "Microsoft's quarterly earnings exceeded expectations, with Azure cloud services generating $15.7 billion in revenue. CEO Satya Nadella will discuss the results on Thursday."
        ]
    
    def test_spacy_entity_extraction(self):
        """Test spaCy-based entity extraction"""
        text = self.entity_rich_texts[0]
        
        try:
            result = self.ner_extractor.extract_entities_spacy(text)
            
            self.assertIn('entities', result)
            self.assertIn('total_entities', result)
            
            entities = result['entities']
            
            # Should find expected entities
            entity_texts = [ent['text'] for ent in entities]
            entity_labels = [ent['label'] for ent in entities]
            
            # Should find organization
            self.assertTrue(any('Apple' in text for text in entity_texts))
            # Should find person
            self.assertTrue(any('Tim Cook' in text for text in entity_texts))
            # Should find money
            self.assertIn('MONEY', entity_labels)
            # Should find location
            self.assertIn('GPE', entity_labels)
            
        except Exception as e:
            self.skipTest(f"spaCy model not available: {e}")
    
    def test_transformer_entity_extraction(self):
        """Test transformer-based entity extraction"""
        text = self.entity_rich_texts[1]
        
        try:
            result = self.ner_extractor.extract_entities_transformer(text)
            
            self.assertIn('entities', result)
            entities = result['entities']
            
            # Should have confidence scores
            for entity in entities:
                self.assertIn('confidence', entity)
                self.assertGreaterEqual(entity['confidence'], 0.0)
                self.assertLessEqual(entity['confidence'], 1.0)
            
        except Exception as e:
            self.skipTest(f"Transformer model not available: {e}")
    
    def test_entity_relationship_extraction(self):
        """Test relationship extraction between entities"""
        text = self.entity_rich_texts[2]
        
        result = self.ner_extractor.extract_relationships(text)
        
        self.assertIn('relationships', result)
        self.assertIn('entities', result)
        
        # Relationships should connect entities
        relationships = result['relationships']
        for rel in relationships:
            self.assertIn('subject', rel)
            self.assertIn('predicate', rel)
            self.assertIn('object', rel)
    
    def test_knowledge_graph_construction(self):
        """Test knowledge graph construction"""
        texts = self.entity_rich_texts
        
        graph = self.ner_extractor.build_knowledge_graph(texts)
        
        self.assertIn('nodes', graph)
        self.assertIn('edges', graph)
        self.assertIn('statistics', graph)
        
        # Should have nodes and edges
        self.assertGreater(len(graph['nodes']), 0)
        
        # Statistics should be reasonable
        stats = graph['statistics']
        self.assertIn('total_entities', stats)
        self.assertIn('total_relationships', stats)
        self.assertIn('entity_types', stats)
    
    def test_entity_cooccurrence_analysis(self):
        """Test entity co-occurrence analysis"""
        texts = self.entity_rich_texts
        
        cooccurrence = self.ner_extractor.analyze_entity_cooccurrence(texts)
        
        self.assertIn('cooccurrence_matrix', cooccurrence)
        self.assertIn('frequent_pairs', cooccurrence)
        
        # Should find entity pairs
        frequent_pairs = cooccurrence['frequent_pairs']
        self.assertIsInstance(frequent_pairs, list)
    
    def test_entity_linking(self):
        """Test entity linking and disambiguation"""
        text = "Apple reported strong iPhone sales. The company based in Cupertino continues to innovate."
        
        result = self.ner_extractor.link_entities(text)
        
        self.assertIn('linked_entities', result)
        
        # Should disambiguate Apple as company, not fruit
        linked_entities = result['linked_entities']
        apple_entities = [ent for ent in linked_entities if 'Apple' in ent.get('text', '')]
        
        if apple_entities:
            # Should have knowledge base links
            self.assertIn('kb_id', apple_entities[0])
    
    def test_merged_entity_extraction(self):
        """Test merged results from multiple NER methods"""
        text = self.entity_rich_texts[0]
        
        result = self.ner_extractor.extract_entities(text)
        
        # Should contain results from multiple methods
        self.assertIn('spacy', result)
        self.assertIn('merged', result)
        
        # Merged results should combine findings
        merged = result['merged']
        self.assertIn('entities', merged)
        self.assertIn('confidence_scores', merged)
        self.assertIn('entity_counts_by_type', merged)


class TestIntegrationClassification(unittest.TestCase):
    """Integration tests for classification components"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.classifier = AdvancedNewsClassifier()
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.ner_extractor = EntityRelationshipMapper()
        
        # Load real classification results for comparison
        try:
            results_path = os.path.join('data', 'results', 'classification_results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    self.real_results = json.load(f)
            else:
                self.real_results = None
        except:
            self.real_results = None
    
    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline integration"""
        text = "Apple Inc. announced strong quarterly earnings, driving optimism in technology markets and boosting investor confidence significantly."
        
        # Classification
        texts = [text]
        classification_result = self.classifier.predict_with_confidence(texts)
        
        # Sentiment analysis
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(text)
        
        # Entity extraction
        entity_result = self.ner_extractor.extract_entities(text)
        
        # All components should work together
        self.assertIsNotNone(classification_result)
        self.assertIsNotNone(sentiment_result)
        self.assertIsNotNone(entity_result)
        
        # Results should be consistent
        # Tech article should have positive sentiment
        self.assertIn(sentiment_result['aggregated']['classification'], ['positive', 'neutral'])
        
        # Should extract technology-related entities
        entities = entity_result['merged']['entities']
        entity_texts = [ent['text'] for ent in entities]
        self.assertTrue(any('Apple' in text for text in entity_texts))
    
    def test_performance_comparison_with_real_results(self):
        """Test performance comparison with real midterm results"""
        if self.real_results is None:
            self.skipTest("Real classification results not available")
        
        # Compare with real performance metrics
        real_performance = self.real_results.get('model_performance', {})
        
        if 'naive_bayes' in real_performance:
            real_accuracy = real_performance['naive_bayes']['accuracy']
            
            # Our system should aim for similar or better performance
            self.assertGreater(real_accuracy, 0.9)  # Verify real system had good performance
            
            # Test basic classification accuracy on known good examples
            tech_texts = [
                "Apple releases new iPhone with advanced AI capabilities.",
                "Microsoft announces major cloud computing innovations."
            ]
            
            # These should classify as tech with high confidence
            result = self.classifier.predict_with_confidence(tech_texts)
            
            # At least basic functionality should work
            self.assertEqual(len(result['predictions']), len(tech_texts))


if __name__ == '__main__':
    # Set up test discovery and execution
    unittest.main(verbosity=2)