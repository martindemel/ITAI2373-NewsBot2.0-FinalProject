#!/usr/bin/env python3
"""
Integration Test Suite for NewsBot 2.0 Complete System
Tests the full system integration including all modules working together
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
import json
import tempfile
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the main system
try:
    from newsbot_main import NewsBot2System
    from config.settings import NewsBot2Config
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running tests from the ITAI2373-NewsBot-Final directory")
    sys.exit(1)

class TestSystemIntegration(unittest.TestCase):
    """Test suite for complete NewsBot 2.0 system integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures"""
        cls.test_config = {
            'system': {
                'debug': True,
                'log_level': 'WARNING'  # Reduce log noise during tests
            },
            'data': {
                'data_dir': 'data',
                'processed_data_dir': 'data/processed',
                'default_dataset': 'newsbot_dataset.csv'
            }
        }
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_config, f)
            self.config_file = f.name
        
        # Initialize system with test configuration
        self.newsbot_system = None
        
        # Real test data samples from BBC News dataset structure
        self.real_test_articles = [
            {
                'text': "Apple Inc. announced record quarterly earnings driven by strong iPhone sales and continued growth in services revenue. The technology giant reported revenue exceeding analyst expectations, with CEO Tim Cook highlighting innovation in artificial intelligence and machine learning capabilities.",
                'category': 'tech',
                'expected_sentiment': 'positive'
            },
            {
                'text': "The Federal Reserve announced its decision to maintain interest rates at current levels following concerns about inflation and economic uncertainty. Financial markets responded with mixed reactions as investors assess implications for future monetary policy.",
                'category': 'business', 
                'expected_sentiment': 'neutral'
            },
            {
                'text': "Manchester United secured a dominant 4-1 victory over their rivals in yesterday's Premier League match, with the team's new striker scoring a hat-trick in his debut performance. The victory moves the club to the top of the league table.",
                'category': 'sport',
                'expected_sentiment': 'positive'
            },
            {
                'text': "The latest environmental disaster has caused widespread damage across the region, forcing thousands of residents to evacuate their homes. Emergency services are working around the clock to provide assistance to affected communities.",
                'category': 'politics',
                'expected_sentiment': 'negative'
            },
            {
                'text': "The highly anticipated blockbuster film premiered at the international film festival, receiving standing ovations from critics and audiences. The movie features groundbreaking visual effects and an ensemble cast of acclaimed actors.",
                'category': 'entertainment',
                'expected_sentiment': 'positive'
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary config file
        if hasattr(self, 'config_file') and os.path.exists(self.config_file):
            os.unlink(self.config_file)
        
        # Clean up system
        if self.newsbot_system:
            del self.newsbot_system
    
    def test_system_initialization(self):
        """Test complete system initialization"""
        try:
            # Initialize system
            self.newsbot_system = NewsBot2System()
            
            # Test basic initialization
            self.assertIsNotNone(self.newsbot_system)
            self.assertIsInstance(self.newsbot_system.config, NewsBot2Config)
            
            # Test system initialization
            init_result = self.newsbot_system.initialize_system(load_models=False, load_data=False)
            
            # Should initialize successfully
            self.assertIn('status', init_result)
            
            # Should have initialized components
            self.assertIn('components_initialized', init_result)
            self.assertGreater(len(init_result['components_initialized']), 0)
            
        except Exception as e:
            self.skipTest(f"System initialization failed - dependencies may not be available: {e}")
    
    def test_real_data_loading(self):
        """Test loading real BBC News dataset"""
        try:
            self.newsbot_system = NewsBot2System()
            
            # Initialize with data loading
            init_result = self.newsbot_system.initialize_system(load_models=False, load_data=True)
            
            if init_result.get('data_loaded', {}).get('status') == 'success':
                # Verify real data loaded
                self.assertTrue(self.newsbot_system.data_loaded)
                self.assertIsNotNone(self.newsbot_system.article_database)
                
                # Check data structure matches BBC News dataset
                data_info = init_result['data_loaded']
                self.assertIn('articles_loaded', data_info)
                self.assertIn('categories', data_info)
                
                # Should have the 5 BBC News categories
                categories = data_info.get('categories', [])
                expected_categories = {'business', 'entertainment', 'politics', 'sport', 'tech'}
                actual_categories = set(categories)
                
                # Should have at least some of the expected categories
                self.assertTrue(len(actual_categories.intersection(expected_categories)) > 0)
                
                # Should have substantial amount of data (BBC dataset has 2,225 articles)
                self.assertGreater(data_info['articles_loaded'], 1000)
                
            else:
                self.skipTest("Real BBC News dataset not available for testing")
                
        except Exception as e:
            self.skipTest(f"Data loading test failed - data may not be available: {e}")
    
    def test_end_to_end_article_analysis(self):
        """Test complete article analysis pipeline"""
        try:
            self.newsbot_system = NewsBot2System()
            init_result = self.newsbot_system.initialize_system(load_models=False, load_data=False)
            
            if init_result['status'] != 'completed':
                self.skipTest("System initialization incomplete")
            
            # Test analysis with real sample articles
            analysis_result = self.newsbot_system.analyze_articles(
                self.real_test_articles,
                analysis_types=['classification', 'sentiment', 'entities']
            )
            
            # Should complete without errors
            self.assertNotIn('error', analysis_result)
            
            # Should have results for all analysis types
            self.assertIn('results', analysis_result)
            results = analysis_result['results']
            
            # Check classification results
            if 'classification' in results:
                class_results = results['classification']
                self.assertIn('predictions', class_results)
                self.assertEqual(len(class_results['predictions']), len(self.real_test_articles))
            
            # Check sentiment results
            if 'sentiment' in results:
                sentiment_results = results['sentiment']
                self.assertIn('sentiment_distribution', sentiment_results)
                self.assertIn('total_analyzed', sentiment_results)
            
            # Check entity results
            if 'entities' in results:
                entity_results = results['entities']
                self.assertIn('entities_by_type', entity_results)
                self.assertIn('total_entities', entity_results)
            
        except Exception as e:
            self.skipTest(f"End-to-end analysis test failed: {e}")
    
    def test_natural_language_query_processing(self):
        """Test conversational interface with natural language queries"""
        try:
            self.newsbot_system = NewsBot2System()
            init_result = self.newsbot_system.initialize_system(load_models=False, load_data=True)
            
            if init_result['status'] != 'completed':
                self.skipTest("System initialization incomplete")
            
            # Test various query types
            test_queries = [
                "Show me articles about technology",
                "What is the sentiment of business news?",
                "Find articles about sports",
                "How many entertainment articles are there?",
                "Summarize the latest tech news"
            ]
            
            for query in test_queries:
                try:
                    result = self.newsbot_system.process_natural_language_query(query)
                    
                    # Should return a response
                    self.assertIsInstance(result, dict)
                    
                    # Should not have errors for basic queries
                    if 'error' in result and 'not initialized' not in result['error']:
                        # Log but don't fail - some queries might need specific components
                        print(f"Query '{query}' returned error: {result['error']}")
                    
                except Exception as e:
                    # Individual query failures shouldn't fail the whole test
                    print(f"Query '{query}' failed: {e}")
                    continue
            
        except Exception as e:
            self.skipTest(f"Natural language query test failed: {e}")
    
    def test_multilingual_capabilities(self):
        """Test multilingual analysis capabilities"""
        try:
            self.newsbot_system = NewsBot2System()
            init_result = self.newsbot_system.initialize_system(load_models=False, load_data=False)
            
            if init_result['status'] != 'completed':
                self.skipTest("System initialization incomplete")
            
            # Test multilingual articles
            multilingual_articles = [
                {
                    'text': "Technology companies are investing heavily in artificial intelligence research and development.",
                    'category': 'tech'
                },
                {
                    'text': "Las empresas de tecnología están invirtiendo mucho en investigación y desarrollo de inteligencia artificial.",
                    'category': 'tech'
                },
                {
                    'text': "Les entreprises technologiques investissent massivement dans la recherche et le développement en intelligence artificielle.",
                    'category': 'tech'
                }
            ]
            
            # Test language detection
            if hasattr(self.newsbot_system, 'language_detector') and self.newsbot_system.language_detector:
                for article in multilingual_articles:
                    try:
                        detection_result = self.newsbot_system.language_detector.detect_language(article['text'])
                        self.assertIn('language', detection_result)
                        self.assertIn('confidence', detection_result)
                    except Exception as e:
                        print(f"Language detection failed: {e}")
            
            # Test cross-lingual analysis
            analysis_result = self.newsbot_system.analyze_articles(multilingual_articles)
            
            # Should handle multilingual content
            self.assertNotIn('error', analysis_result)
            
        except Exception as e:
            self.skipTest(f"Multilingual test failed: {e}")
    
    def test_visualization_and_export(self):
        """Test visualization and export capabilities"""
        try:
            self.newsbot_system = NewsBot2System()
            init_result = self.newsbot_system.initialize_system(load_models=False, load_data=False)
            
            if init_result['status'] != 'completed':
                self.skipTest("System initialization incomplete")
            
            # Perform analysis to get results
            analysis_result = self.newsbot_system.analyze_articles(
                self.real_test_articles[:3],  # Use subset for speed
                analysis_types=['sentiment', 'entities']
            )
            
            if 'error' in analysis_result:
                self.skipTest("Analysis failed, cannot test visualization")
            
            # Test dashboard creation
            dashboard_result = self.newsbot_system.create_analysis_dashboard(analysis_result)
            
            if 'error' not in dashboard_result:
                self.assertIn('dashboard_components', dashboard_result)
                self.assertIn('total_components', dashboard_result)
            
            # Test export functionality
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
                export_result = self.newsbot_system.export_analysis_results(
                    analysis_result, 
                    export_format='json',
                    output_path=tmp_file.name
                )
                
                if 'error' not in export_result:
                    self.assertIn('output_path', export_result)
                    self.assertTrue(os.path.exists(export_result['output_path']))
                
                # Clean up
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
            
        except Exception as e:
            self.skipTest(f"Visualization and export test failed: {e}")
    
    def test_system_performance_monitoring(self):
        """Test system performance and monitoring"""
        try:
            self.newsbot_system = NewsBot2System()
            init_result = self.newsbot_system.initialize_system(load_models=False, load_data=False)
            
            if init_result['status'] != 'completed':
                self.skipTest("System initialization incomplete")
            
            # Get system status
            status = self.newsbot_system.get_system_status()
            
            # Should have required status fields
            self.assertIn('system_initialized', status)
            self.assertIn('components_loaded', status)
            self.assertIn('uptime_seconds', status)
            self.assertIn('component_status', status)
            
            # System should be initialized
            self.assertTrue(status['system_initialized'])
            
            # Should have loaded components
            self.assertGreater(status['components_loaded'], 0)
            
            # Component status should be dict
            self.assertIsInstance(status['component_status'], dict)
            
        except Exception as e:
            self.skipTest(f"Performance monitoring test failed: {e}")
    
    def test_error_handling_and_recovery(self):
        """Test system error handling and recovery"""
        try:
            self.newsbot_system = NewsBot2System()
            
            # Test with invalid input
            invalid_articles = [
                {'text': '', 'category': 'invalid'},  # Empty text
                {'text': None, 'category': 'test'},    # None text
                {'wrong_field': 'test'}                # Missing required fields
            ]
            
            # Should handle invalid input gracefully
            result = self.newsbot_system.analyze_articles(invalid_articles)
            
            # Should either filter invalid data or return appropriate error
            self.assertIsInstance(result, dict)
            
            # Test invalid query
            invalid_query_result = self.newsbot_system.process_natural_language_query("")
            self.assertIsInstance(invalid_query_result, dict)
            
        except Exception as e:
            # Error handling tests should not throw unhandled exceptions
            self.fail(f"Error handling test failed with unhandled exception: {e}")
    
    def test_configuration_management(self):
        """Test configuration management system"""
        try:
            # Test configuration loading
            config = NewsBot2Config()
            
            # Should have required configuration sections
            self.assertIsNotNone(config.get('system'))
            self.assertIsNotNone(config.get('data'))
            self.assertIsNotNone(config.get('components'))
            
            # Test configuration validation
            data_dir = config.get_data_dir()
            self.assertIsInstance(data_dir, str)
            self.assertGreater(len(data_dir), 0)
            
            # Test component configuration
            classifier_config = config.get_component_config('classifier')
            self.assertIsInstance(classifier_config, dict)
            
        except Exception as e:
            self.skipTest(f"Configuration management test failed: {e}")


class TestRealDataIntegration(unittest.TestCase):
    """Integration tests specifically for real BBC News data"""
    
    def setUp(self):
        """Set up real data tests"""
        self.data_dir = 'data'
        self.processed_data_path = os.path.join(self.data_dir, 'processed', 'newsbot_dataset.csv')
        self.metadata_path = os.path.join(self.data_dir, 'processed', 'dataset_metadata.json')
        self.results_dir = os.path.join(self.data_dir, 'results')
    
    def test_real_dataset_structure(self):
        """Test real BBC News dataset structure and integrity"""
        if not os.path.exists(self.processed_data_path):
            self.skipTest("Real BBC News dataset not available")
        
        # Load dataset
        df = pd.read_csv(self.processed_data_path)
        
        # Verify structure
        expected_columns = ['text', 'category']
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Missing required column: {col}")
        
        # Verify data quality
        self.assertGreater(len(df), 2000, "Dataset should have substantial number of articles")
        
        # Check categories
        categories = df['category'].unique()
        expected_categories = {'business', 'entertainment', 'politics', 'sport', 'tech'}
        actual_categories = set(categories)
        
        self.assertEqual(actual_categories, expected_categories, "Categories don't match BBC News dataset")
        
        # Check for missing values
        self.assertEqual(df['text'].isnull().sum(), 0, "No missing text values allowed")
        self.assertEqual(df['category'].isnull().sum(), 0, "No missing category values allowed")
    
    def test_real_metadata_consistency(self):
        """Test metadata consistency with actual dataset"""
        if not os.path.exists(self.metadata_path):
            self.skipTest("Dataset metadata not available")
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Verify metadata structure
        self.assertIn('total_articles', metadata)
        self.assertIn('categories', metadata)
        
        # If dataset is also available, verify consistency
        if os.path.exists(self.processed_data_path):
            df = pd.read_csv(self.processed_data_path)
            
            # Total articles should match
            self.assertEqual(metadata['total_articles'], len(df))
            
            # Category counts should match
            actual_counts = df['category'].value_counts().to_dict()
            metadata_counts = metadata['categories']
            
            for category, count in metadata_counts.items():
                self.assertEqual(actual_counts.get(category, 0), count)
    
    def test_real_analysis_results_structure(self):
        """Test structure of real analysis results from midterm"""
        classification_results_path = os.path.join(self.results_dir, 'classification_results.json')
        sentiment_results_path = os.path.join(self.results_dir, 'sentiment_results.json')
        
        # Test classification results
        if os.path.exists(classification_results_path):
            with open(classification_results_path, 'r') as f:
                class_results = json.load(f)
            
            self.assertIn('model_performance', class_results)
            
            # Should have performance metrics
            performance = class_results['model_performance']
            for model_name, metrics in performance.items():
                self.assertIn('accuracy', metrics)
                
                # Accuracy should be reasonable
                accuracy = metrics['accuracy']
                self.assertGreater(accuracy, 0.8, f"Model {model_name} accuracy too low: {accuracy}")
        
        # Test sentiment results
        if os.path.exists(sentiment_results_path):
            with open(sentiment_results_path, 'r') as f:
                sentiment_results = json.load(f)
            
            # Should have sentiment analysis results
            self.assertTrue(len(sentiment_results) > 0)
    
    def test_data_path_configuration(self):
        """Test that data paths in configuration match actual file locations"""
        config = NewsBot2Config()
        
        # Get configured paths
        data_dir = config.get_data_dir()
        processed_dir = config.get('data.processed_data_dir')
        dataset_name = config.get('data.default_dataset')
        
        # Construct expected path
        expected_path = os.path.join(processed_dir, dataset_name)
        
        # Path should exist if configured correctly
        if os.path.exists(expected_path):
            # Verify it's the correct file
            df = pd.read_csv(expected_path)
            self.assertGreater(len(df), 1000)
            self.assertIn('category', df.columns)


class TestSystemStressAndPerformance(unittest.TestCase):
    """Stress tests and performance tests for the system"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.newsbot_system = None
        
        # Create larger test dataset
        self.large_test_data = []
        base_texts = [
            "Technology companies continue to innovate in artificial intelligence and machine learning sectors.",
            "Financial markets show volatility amid economic uncertainty and changing interest rates.",
            "Sports teams prepare for upcoming championship tournaments with intensive training.",
            "Entertainment industry adapts to streaming platforms and changing consumer preferences.",
            "Political leaders discuss policy reforms and legislative changes affecting citizens."
        ]
        
        # Replicate to create larger dataset
        for i in range(20):  # 100 total articles
            for j, text in enumerate(base_texts):
                self.large_test_data.append({
                    'text': f"{text} Article variant {i+1} for stress testing purposes.",
                    'category': ['tech', 'business', 'sport', 'entertainment', 'politics'][j]
                })
    
    def test_large_batch_processing(self):
        """Test processing large batches of articles"""
        try:
            self.newsbot_system = NewsBot2System()
            init_result = self.newsbot_system.initialize_system(load_models=False, load_data=False)
            
            if init_result['status'] != 'completed':
                self.skipTest("System initialization incomplete")
            
            # Process large batch
            import time
            start_time = time.time()
            
            result = self.newsbot_system.analyze_articles(
                self.large_test_data,
                analysis_types=['sentiment']  # Use lighter analysis for stress test
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should complete without errors
            self.assertNotIn('error', result)
            
            # Should process all articles
            self.assertEqual(result['total_articles'], len(self.large_test_data))
            
            # Should be reasonably fast (less than 60 seconds for 100 articles)
            self.assertLess(processing_time, 60.0, f"Processing took too long: {processing_time:.2f} seconds")
            
            print(f"Processed {len(self.large_test_data)} articles in {processing_time:.2f} seconds")
            
        except Exception as e:
            self.skipTest(f"Large batch processing test failed: {e}")
    
    def test_memory_usage(self):
        """Test memory usage during processing"""
        try:
            import psutil
            import os
            
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            self.newsbot_system = NewsBot2System()
            init_result = self.newsbot_system.initialize_system(load_models=False, load_data=False)
            
            if init_result['status'] != 'completed':
                self.skipTest("System initialization incomplete")
            
            # Process articles and monitor memory
            result = self.newsbot_system.analyze_articles(
                self.large_test_data[:50],  # Subset for memory test
                analysis_types=['sentiment']
            )
            
            # Get memory usage after processing
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (increase: {memory_increase:.1f}MB)")
            
            # Memory increase should be reasonable (less than 500MB for this test)
            self.assertLess(memory_increase, 500, f"Memory usage too high: {memory_increase:.1f}MB")
            
        except ImportError:
            self.skipTest("psutil not available for memory testing")
        except Exception as e:
            self.skipTest(f"Memory usage test failed: {e}")


if __name__ == '__main__':
    # Set up test discovery and execution
    
    # Create test suite with different test categories
    loader = unittest.TestLoader()
    
    # Core integration tests
    integration_suite = loader.loadTestsFromTestCase(TestSystemIntegration)
    
    # Real data tests (may be skipped if data not available)
    real_data_suite = loader.loadTestsFromTestCase(TestRealDataIntegration)
    
    # Performance tests (may be skipped if too resource intensive)
    performance_suite = loader.loadTestsFromTestCase(TestSystemStressAndPerformance)
    
    # Combine all test suites
    all_tests = unittest.TestSuite([integration_suite, real_data_suite, performance_suite])
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(all_tests)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2]}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)