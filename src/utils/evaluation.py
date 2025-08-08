#!/usr/bin/env python3
"""
Evaluation Framework for NewsBot 2.0
Comprehensive evaluation and testing framework for all system components
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from datetime import datetime
import json
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, silhouette_score
)
from sklearn.model_selection import cross_val_score

class EvaluationFramework:
    """
    Comprehensive evaluation framework for testing all NewsBot 2.0 components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluation framework
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Evaluation metrics for different components
        self.component_metrics = {
            'classification': ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix'],
            'sentiment_analysis': ['correlation', 'mae', 'accuracy', 'agreement'],
            'topic_modeling': ['coherence', 'silhouette', 'perplexity'],
            'ner_extraction': ['entity_precision', 'entity_recall', 'entity_f1'],
            'summarization': ['rouge_scores', 'compression_ratio', 'readability'],
            'translation': ['bleu_score', 'quality_assessment', 'fluency'],
            'intent_classification': ['accuracy', 'precision', 'recall', 'coverage'],
            'query_processing': ['response_time', 'success_rate', 'user_satisfaction']
        }
        
        # Test datasets and benchmarks
        self.test_datasets = {}
        self.benchmark_results = {}
        
        # Evaluation history
        self.evaluation_history = []
        
        # Component references (injected)
        self.classifier = None
        self.sentiment_analyzer = None
        self.ner_extractor = None
        self.topic_modeler = None
        self.summarizer = None
        self.translator = None
        self.intent_classifier = None
        self.query_processor = None
        
        # Quality thresholds
        self.quality_thresholds = {
            'classification_accuracy': 0.85,
            'sentiment_correlation': 0.6,
            'ner_f1_score': 0.8,
            'topic_coherence': 0.4,
            'summarization_rouge': 0.3,
            'translation_bleu': 0.4,
            'intent_accuracy': 0.8,
            'query_success_rate': 0.9
        }
    
    def set_components(self, **components):
        """Set component references for evaluation"""
        for name, component in components.items():
            setattr(self, name, component)
    
    def run_comprehensive_evaluation(self, test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of all system components
        
        Args:
            test_data: Optional test data dictionary
            
        Returns:
            Comprehensive evaluation results
        """
        logging.info("Starting comprehensive system evaluation...")
        
        evaluation_results = {
            'evaluation_id': f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'component_results': {},
            'overall_assessment': {},
            'recommendations': []
        }
        
        # Evaluate each component
        components_to_test = [
            ('classification', self.classifier, self.evaluate_classification),
            ('sentiment_analysis', self.sentiment_analyzer, self.evaluate_sentiment_analysis),
            ('ner_extraction', self.ner_extractor, self.evaluate_ner_extraction),
            ('topic_modeling', self.topic_modeler, self.evaluate_topic_modeling),
            ('summarization', self.summarizer, self.evaluate_summarization),
            ('translation', self.translator, self.evaluate_translation),
            ('intent_classification', self.intent_classifier, self.evaluate_intent_classification),
            ('query_processing', self.query_processor, self.evaluate_query_processing)
        ]
        
        for component_name, component, evaluation_function in components_to_test:
            if component is not None:
                try:
                    logging.info(f"Evaluating {component_name}...")
                    component_data = test_data.get(component_name) if test_data else None
                    results = evaluation_function(component, component_data)
                    evaluation_results['component_results'][component_name] = results
                except Exception as e:
                    logging.error(f"Evaluation failed for {component_name}: {e}")
                    evaluation_results['component_results'][component_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            else:
                evaluation_results['component_results'][component_name] = {
                    'status': 'not_available',
                    'message': f'{component_name} component not loaded'
                }
        
        # Generate overall assessment
        overall_assessment = self._generate_overall_assessment(evaluation_results['component_results'])
        evaluation_results['overall_assessment'] = overall_assessment
        
        # Generate recommendations
        recommendations = self._generate_recommendations(evaluation_results['component_results'])
        evaluation_results['recommendations'] = recommendations
        
        # Store in history
        self.evaluation_history.append(evaluation_results)
        
        logging.info("Comprehensive evaluation completed")
        return evaluation_results
    
    def evaluate_classification(self, classifier, test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate classification component"""
        
        results = {
            'component': 'classification',
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'status': 'unknown'
        }
        
        try:
            if not classifier.is_trained:
                results['status'] = 'not_trained'
                results['message'] = 'Classifier not trained'
                return results
            
            # Use provided test data or generate synthetic test
            if test_data and 'X_test' in test_data and 'y_test' in test_data:
                X_test = test_data['X_test']
                y_test = test_data['y_test']
            else:
                # Generate synthetic test data
                logging.warning("No test data provided, using model self-evaluation")
                training_results = classifier.training_results
                if 'model_results' in training_results:
                    best_model = training_results['best_model']
                    model_results = training_results['model_results'][best_model]
                    
                    results['metrics'] = {
                        'accuracy': model_results['accuracy'],
                        'classification_report': model_results['classification_report'],
                        'cross_validation_mean': model_results.get('cv_mean', 0),
                        'cross_validation_std': model_results.get('cv_std', 0)
                    }
                    
                    results['status'] = 'completed'
                    results['quality_assessment'] = self._assess_classification_quality(results['metrics'])
                    return results
            
            # Perform predictions
            prediction_results = classifier.predict_with_confidence(X_test)
            y_pred = prediction_results['predictions']
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            results['metrics'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'confidence_scores': prediction_results['confidence_scores'].tolist(),
                'avg_confidence': np.mean(prediction_results['confidence_scores'])
            }
            
            results['status'] = 'completed'
            results['quality_assessment'] = self._assess_classification_quality(results['metrics'])
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def evaluate_sentiment_analysis(self, sentiment_analyzer, test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate sentiment analysis component"""
        
        results = {
            'component': 'sentiment_analysis',
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'status': 'unknown'
        }
        
        try:
            # Use provided test data or create synthetic test
            if test_data and 'texts' in test_data:
                test_texts = test_data['texts']
                true_sentiments = test_data.get('sentiments')
            else:
                # Create simple test cases
                test_texts = [
                    "This is wonderful news and I'm very happy about it!",
                    "This is terrible and disappointing news.",
                    "The weather is okay today, nothing special.",
                    "Amazing breakthrough in technology!",
                    "Economic crisis causing major problems."
                ]
                true_sentiments = ['positive', 'negative', 'neutral', 'positive', 'negative']
            
            # Analyze sentiments
            predicted_sentiments = []
            sentiment_scores = []
            
            for text in test_texts:
                sentiment_result = sentiment_analyzer.analyze_sentiment(text)
                
                if 'aggregated' in sentiment_result:
                    predicted_sentiments.append(sentiment_result['aggregated'].get('classification', 'neutral'))
                    sentiment_scores.append(sentiment_result['aggregated'].get('weighted_score', 0))
                else:
                    predicted_sentiments.append('neutral')
                    sentiment_scores.append(0)
            
            # Calculate metrics
            if true_sentiments:
                accuracy = accuracy_score(true_sentiments, predicted_sentiments)
                class_report = classification_report(true_sentiments, predicted_sentiments, output_dict=True)
                
                results['metrics']['accuracy'] = accuracy
                results['metrics']['classification_report'] = class_report
            
            results['metrics']['sentiment_scores'] = sentiment_scores
            results['metrics']['predicted_sentiments'] = predicted_sentiments
            results['metrics']['avg_sentiment_score'] = np.mean(sentiment_scores)
            results['metrics']['sentiment_std'] = np.std(sentiment_scores)
            
            # Test consistency
            consistency_results = self._test_sentiment_consistency(sentiment_analyzer)
            results['metrics']['consistency'] = consistency_results
            
            results['status'] = 'completed'
            results['quality_assessment'] = self._assess_sentiment_quality(results['metrics'])
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def evaluate_ner_extraction(self, ner_extractor, test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate NER extraction component"""
        
        results = {
            'component': 'ner_extraction',
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'status': 'unknown'
        }
        
        try:
            # Use provided test data or create synthetic test
            if test_data and 'texts' in test_data:
                test_texts = test_data['texts']
                true_entities = test_data.get('entities')
            else:
                # Create simple test cases
                test_texts = [
                    "Apple Inc. CEO Tim Cook visited New York last Tuesday.",
                    "Microsoft Corporation announced new products in Seattle.",
                    "President Biden met with European leaders in Washington D.C."
                ]
                # Expected entities would be provided in real evaluation
                true_entities = None
            
            # Extract entities
            extraction_results = []
            total_entities = 0
            entity_type_counts = defaultdict(int)
            
            for text in test_texts:
                extraction_result = ner_extractor.extract_entities(text)
                
                if 'merged' in extraction_result:
                    entities = extraction_result['merged'].get('entities', [])
                    extraction_results.append(entities)
                    total_entities += len(entities)
                    
                    for entity in entities:
                        entity_type_counts[entity.get('label', 'OTHER')] += 1
                else:
                    extraction_results.append([])
            
            results['metrics']['total_entities_extracted'] = total_entities
            results['metrics']['avg_entities_per_text'] = total_entities / len(test_texts) if test_texts else 0
            results['metrics']['entity_type_distribution'] = dict(entity_type_counts)
            results['metrics']['extraction_results'] = extraction_results
            
            # Test extraction consistency
            consistency_results = self._test_ner_consistency(ner_extractor)
            results['metrics']['consistency'] = consistency_results
            
            # Get component statistics
            ner_stats = ner_extractor.get_extraction_statistics()
            results['metrics']['component_stats'] = ner_stats
            
            results['status'] = 'completed'
            results['quality_assessment'] = self._assess_ner_quality(results['metrics'])
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def evaluate_topic_modeling(self, topic_modeler, test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate topic modeling component"""
        
        results = {
            'component': 'topic_modeling',
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'status': 'unknown'
        }
        
        try:
            if not topic_modeler.is_fitted:
                results['status'] = 'not_fitted'
                results['message'] = 'Topic modeler not fitted'
                return results
            
            # Get model summary
            model_summary = topic_modeler.get_model_summary()
            results['metrics']['model_summary'] = model_summary
            
            # Test topic assignment consistency
            if test_data and 'documents' in test_data:
                test_documents = test_data['documents']
            else:
                # Use sample documents
                test_documents = [
                    "Technology companies are investing heavily in artificial intelligence research.",
                    "The stock market showed significant gains in the technology sector.",
                    "Sports teams are preparing for the upcoming championship season."
                ]
            
            # Test topic assignments
            topic_assignments = []
            for doc in test_documents:
                try:
                    topic_result = topic_modeler.get_article_topics(doc)
                    topic_assignments.append(topic_result)
                except Exception as e:
                    logging.warning(f"Topic assignment failed: {e}")
                    topic_assignments.append({'error': str(e)})
            
            results['metrics']['topic_assignments'] = topic_assignments
            
            # Calculate topic quality metrics
            if hasattr(topic_modeler, 'topic_coherence_scores'):
                coherence_scores = topic_modeler.topic_coherence_scores
                if coherence_scores and isinstance(coherence_scores, dict):
                    if 'c_v' in coherence_scores:
                        results['metrics']['coherence_score'] = coherence_scores['c_v']
                    elif 'simple' in coherence_scores:
                        results['metrics']['coherence_score'] = coherence_scores['simple']
            
            # Test topic consistency
            consistency_results = self._test_topic_consistency(topic_modeler)
            results['metrics']['consistency'] = consistency_results
            
            results['status'] = 'completed'
            results['quality_assessment'] = self._assess_topic_modeling_quality(results['metrics'])
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def evaluate_summarization(self, summarizer, test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate summarization component"""
        
        results = {
            'component': 'summarization',
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'status': 'unknown'
        }
        
        try:
            # Use provided test data or create synthetic test
            if test_data and 'texts' in test_data:
                test_texts = test_data['texts']
            else:
                # Create test texts
                test_texts = [
                    "Artificial intelligence is revolutionizing many industries. Machine learning algorithms can now process vast amounts of data and identify patterns that humans might miss. This technology is being applied in healthcare, finance, transportation, and many other sectors. Companies are investing billions of dollars in AI research and development. The potential benefits include improved efficiency, better decision-making, and new innovative products and services.",
                    "Climate change is one of the most pressing issues of our time. Rising temperatures, changing weather patterns, and melting ice caps are all signs of a changing climate. Scientists agree that human activities, particularly the burning of fossil fuels, are the primary cause. Governments and organizations worldwide are working to reduce greenhouse gas emissions and transition to renewable energy sources."
                ]
            
            # Generate summaries
            summarization_results = []
            compression_ratios = []
            quality_scores = []
            
            for text in test_texts:
                try:
                    summary_result = summarizer.summarize_article(text, 'balanced')
                    summarization_results.append(summary_result)
                    
                    if 'compression_ratio' in summary_result:
                        compression_ratios.append(summary_result['compression_ratio'])
                    
                    if 'quality' in summary_result and 'overall_score' in summary_result['quality']:
                        quality_scores.append(summary_result['quality']['overall_score'])
                        
                except Exception as e:
                    logging.warning(f"Summarization failed: {e}")
                    summarization_results.append({'error': str(e)})
            
            results['metrics']['summarization_results'] = summarization_results
            results['metrics']['avg_compression_ratio'] = np.mean(compression_ratios) if compression_ratios else 0
            results['metrics']['avg_quality_score'] = np.mean(quality_scores) if quality_scores else 0
            results['metrics']['successful_summaries'] = len([r for r in summarization_results if 'error' not in r])
            
            # Test summarization consistency
            consistency_results = self._test_summarization_consistency(summarizer)
            results['metrics']['consistency'] = consistency_results
            
            # Get component statistics
            summarizer_stats = summarizer.get_summarization_stats()
            results['metrics']['component_stats'] = summarizer_stats
            
            results['status'] = 'completed'
            results['quality_assessment'] = self._assess_summarization_quality(results['metrics'])
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def evaluate_translation(self, translator, test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate translation component"""
        
        results = {
            'component': 'translation',
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'status': 'unknown'
        }
        
        try:
            # Use provided test data or create synthetic test
            if test_data and 'texts' in test_data:
                test_texts = test_data['texts']
                source_lang = test_data.get('source_lang', 'en')
                target_lang = test_data.get('target_lang', 'es')
            else:
                # Create simple test cases
                test_texts = [
                    "Hello, how are you today?",
                    "The weather is beautiful.",
                    "Technology is changing our world."
                ]
                source_lang = 'en'
                target_lang = 'es'
            
            # Perform translations
            translation_results = []
            quality_scores = []
            translation_times = []
            
            for text in test_texts:
                try:
                    translation_result = translator.translate_text(text, source_lang, target_lang)
                    translation_results.append(translation_result)
                    
                    if 'quality_metrics' in translation_result:
                        quality_metrics = translation_result['quality_metrics']
                        if 'overall_quality_score' in quality_metrics:
                            quality_scores.append(quality_metrics['overall_quality_score'])
                    
                    if 'translation_time' in translation_result:
                        translation_times.append(translation_result['translation_time'])
                        
                except Exception as e:
                    logging.warning(f"Translation failed: {e}")
                    translation_results.append({'error': str(e)})
            
            results['metrics']['translation_results'] = translation_results
            results['metrics']['avg_quality_score'] = np.mean(quality_scores) if quality_scores else 0
            results['metrics']['avg_translation_time'] = np.mean(translation_times) if translation_times else 0
            results['metrics']['successful_translations'] = len([r for r in translation_results if 'error' not in r])
            
            # Get component statistics
            translator_stats = translator.get_translation_statistics()
            results['metrics']['component_stats'] = translator_stats
            
            results['status'] = 'completed'
            results['quality_assessment'] = self._assess_translation_quality(results['metrics'])
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def evaluate_intent_classification(self, intent_classifier, test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate intent classification component"""
        
        results = {
            'component': 'intent_classification',
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'status': 'unknown'
        }
        
        try:
            # Use provided test data or create synthetic test
            if test_data and 'queries' in test_data:
                test_queries = test_data['queries']
                true_intents = test_data.get('intents')
            else:
                # Create test queries
                test_queries = [
                    "find articles about technology",
                    "what is the sentiment of political news",
                    "summarize this article",
                    "extract entities from business news",
                    "help me with the system",
                    "show me insights about the data",
                    "compare sources across languages"
                ]
                true_intents = [
                    'search_articles', 'analyze_sentiment', 'summarize_text',
                    'extract_entities', 'get_help', 'get_insights', 'compare_sources'
                ]
            
            # Classify intents
            predicted_intents = []
            confidence_scores = []
            classification_results = []
            
            for query in test_queries:
                try:
                    classification_result = intent_classifier.classify_intent(query)
                    classification_results.append(classification_result)
                    
                    final_intent = classification_result.get('final_intent', {})
                    predicted_intents.append(final_intent.get('intent', 'unknown'))
                    confidence_scores.append(final_intent.get('confidence', 0))
                    
                except Exception as e:
                    logging.warning(f"Intent classification failed: {e}")
                    predicted_intents.append('unknown')
                    confidence_scores.append(0)
                    classification_results.append({'error': str(e)})
            
            results['metrics']['classification_results'] = classification_results
            results['metrics']['predicted_intents'] = predicted_intents
            results['metrics']['avg_confidence'] = np.mean(confidence_scores) if confidence_scores else 0
            results['metrics']['unknown_intent_ratio'] = predicted_intents.count('unknown') / len(predicted_intents)
            
            # Calculate accuracy if true intents provided
            if true_intents and len(true_intents) == len(predicted_intents):
                accuracy = accuracy_score(true_intents, predicted_intents)
                results['metrics']['accuracy'] = accuracy
                
                class_report = classification_report(true_intents, predicted_intents, output_dict=True)
                results['metrics']['classification_report'] = class_report
            
            # Test coverage
            coverage_results = intent_classifier.validate_intent_coverage(test_queries)
            results['metrics']['coverage'] = coverage_results
            
            # Get component statistics
            classifier_stats = intent_classifier.get_classification_statistics()
            results['metrics']['component_stats'] = classifier_stats
            
            results['status'] = 'completed'
            results['quality_assessment'] = self._assess_intent_classification_quality(results['metrics'])
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def evaluate_query_processing(self, query_processor, test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate query processing component"""
        
        results = {
            'component': 'query_processing',
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'status': 'unknown'
        }
        
        try:
            # Use provided test data or create synthetic test
            if test_data and 'queries' in test_data:
                test_queries = test_data['queries']
            else:
                # Create test queries
                test_queries = [
                    "find articles about artificial intelligence",
                    "analyze sentiment of technology news",
                    "help me understand the system",
                    "what insights can you provide"
                ]
            
            # Process queries
            processing_results = []
            execution_times = []
            success_count = 0
            
            for query in test_queries:
                try:
                    start_time = datetime.now()
                    processing_result = query_processor.process_query(query)
                    end_time = datetime.now()
                    
                    execution_time = (end_time - start_time).total_seconds()
                    execution_times.append(execution_time)
                    
                    processing_results.append(processing_result)
                    
                    if processing_result.get('status') == 'completed':
                        success_count += 1
                        
                except Exception as e:
                    logging.warning(f"Query processing failed: {e}")
                    processing_results.append({'error': str(e), 'status': 'failed'})
                    execution_times.append(0)
            
            results['metrics']['processing_results'] = processing_results
            results['metrics']['success_rate'] = success_count / len(test_queries) if test_queries else 0
            results['metrics']['avg_execution_time'] = np.mean(execution_times) if execution_times else 0
            results['metrics']['successful_queries'] = success_count
            results['metrics']['total_queries'] = len(test_queries)
            
            # Get component statistics
            processor_stats = query_processor.get_query_statistics()
            results['metrics']['component_stats'] = processor_stats
            
            results['status'] = 'completed'
            results['quality_assessment'] = self._assess_query_processing_quality(results['metrics'])
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    # Quality assessment methods
    
    def _assess_classification_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess classification quality"""
        
        assessment = {'overall_grade': 'unknown', 'issues': [], 'strengths': []}
        
        accuracy = metrics.get('accuracy', 0)
        f1_score = metrics.get('f1_score', 0)
        avg_confidence = metrics.get('avg_confidence', 0)
        
        # Grade calculation
        if accuracy >= 0.9 and f1_score >= 0.9:
            assessment['overall_grade'] = 'excellent'
        elif accuracy >= 0.8 and f1_score >= 0.8:
            assessment['overall_grade'] = 'good'
        elif accuracy >= 0.7 and f1_score >= 0.7:
            assessment['overall_grade'] = 'fair'
        else:
            assessment['overall_grade'] = 'poor'
        
        # Identify issues and strengths
        if accuracy < self.quality_thresholds['classification_accuracy']:
            assessment['issues'].append(f"Accuracy ({accuracy:.3f}) below threshold ({self.quality_thresholds['classification_accuracy']})")
        else:
            assessment['strengths'].append(f"Accuracy ({accuracy:.3f}) meets requirements")
        
        if avg_confidence < 0.7:
            assessment['issues'].append(f"Low average confidence ({avg_confidence:.3f})")
        else:
            assessment['strengths'].append(f"Good confidence scores ({avg_confidence:.3f})")
        
        return assessment
    
    def _assess_sentiment_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess sentiment analysis quality"""
        
        assessment = {'overall_grade': 'unknown', 'issues': [], 'strengths': []}
        
        accuracy = metrics.get('accuracy', 0)
        consistency = metrics.get('consistency', {})
        
        # Grade based on accuracy and consistency
        if accuracy >= 0.8 and consistency.get('agreement_ratio', 0) >= 0.8:
            assessment['overall_grade'] = 'excellent'
        elif accuracy >= 0.7 and consistency.get('agreement_ratio', 0) >= 0.7:
            assessment['overall_grade'] = 'good'
        elif accuracy >= 0.6:
            assessment['overall_grade'] = 'fair'
        else:
            assessment['overall_grade'] = 'poor'
        
        if accuracy > 0 and accuracy < 0.7:
            assessment['issues'].append(f"Low sentiment accuracy ({accuracy:.3f})")
        elif accuracy >= 0.8:
            assessment['strengths'].append(f"High sentiment accuracy ({accuracy:.3f})")
        
        return assessment
    
    def _assess_ner_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess NER extraction quality"""
        
        assessment = {'overall_grade': 'unknown', 'issues': [], 'strengths': []}
        
        avg_entities = metrics.get('avg_entities_per_text', 0)
        consistency = metrics.get('consistency', {})
        
        # Grade based on entity extraction and consistency
        if avg_entities >= 3 and consistency.get('agreement_ratio', 0) >= 0.8:
            assessment['overall_grade'] = 'excellent'
        elif avg_entities >= 2 and consistency.get('agreement_ratio', 0) >= 0.7:
            assessment['overall_grade'] = 'good'
        elif avg_entities >= 1:
            assessment['overall_grade'] = 'fair'
        else:
            assessment['overall_grade'] = 'poor'
        
        if avg_entities < 1:
            assessment['issues'].append(f"Very few entities extracted ({avg_entities:.1f} per text)")
        else:
            assessment['strengths'].append(f"Good entity extraction rate ({avg_entities:.1f} per text)")
        
        return assessment
    
    def _assess_topic_modeling_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess topic modeling quality"""
        
        assessment = {'overall_grade': 'unknown', 'issues': [], 'strengths': []}
        
        coherence_score = metrics.get('coherence_score', 0)
        consistency = metrics.get('consistency', {})
        
        # Grade based on coherence and consistency
        if coherence_score >= 0.5 and consistency.get('stability', 0) >= 0.8:
            assessment['overall_grade'] = 'excellent'
        elif coherence_score >= 0.4 and consistency.get('stability', 0) >= 0.7:
            assessment['overall_grade'] = 'good'
        elif coherence_score >= 0.3:
            assessment['overall_grade'] = 'fair'
        else:
            assessment['overall_grade'] = 'poor'
        
        if coherence_score < self.quality_thresholds['topic_coherence']:
            assessment['issues'].append(f"Low topic coherence ({coherence_score:.3f})")
        else:
            assessment['strengths'].append(f"Good topic coherence ({coherence_score:.3f})")
        
        return assessment
    
    def _assess_summarization_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess summarization quality"""
        
        assessment = {'overall_grade': 'unknown', 'issues': [], 'strengths': []}
        
        avg_quality = metrics.get('avg_quality_score', 0)
        avg_compression = metrics.get('avg_compression_ratio', 0)
        success_rate = metrics.get('successful_summaries', 0) / max(len(metrics.get('summarization_results', [])), 1)
        
        # Grade based on quality, compression, and success rate
        if avg_quality >= 0.8 and 0.1 <= avg_compression <= 0.5 and success_rate >= 0.9:
            assessment['overall_grade'] = 'excellent'
        elif avg_quality >= 0.6 and 0.1 <= avg_compression <= 0.6 and success_rate >= 0.8:
            assessment['overall_grade'] = 'good'
        elif avg_quality >= 0.4 and success_rate >= 0.7:
            assessment['overall_grade'] = 'fair'
        else:
            assessment['overall_grade'] = 'poor'
        
        if success_rate < 0.8:
            assessment['issues'].append(f"Low summarization success rate ({success_rate:.1%})")
        else:
            assessment['strengths'].append(f"High summarization success rate ({success_rate:.1%})")
        
        return assessment
    
    def _assess_translation_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess translation quality"""
        
        assessment = {'overall_grade': 'unknown', 'issues': [], 'strengths': []}
        
        avg_quality = metrics.get('avg_quality_score', 0)
        success_rate = metrics.get('successful_translations', 0) / max(len(metrics.get('translation_results', [])), 1)
        
        # Grade based on quality and success rate
        if avg_quality >= 0.8 and success_rate >= 0.9:
            assessment['overall_grade'] = 'excellent'
        elif avg_quality >= 0.6 and success_rate >= 0.8:
            assessment['overall_grade'] = 'good'
        elif avg_quality >= 0.4 and success_rate >= 0.7:
            assessment['overall_grade'] = 'fair'
        else:
            assessment['overall_grade'] = 'poor'
        
        if success_rate < 0.8:
            assessment['issues'].append(f"Low translation success rate ({success_rate:.1%})")
        else:
            assessment['strengths'].append(f"High translation success rate ({success_rate:.1%})")
        
        return assessment
    
    def _assess_intent_classification_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess intent classification quality"""
        
        assessment = {'overall_grade': 'unknown', 'issues': [], 'strengths': []}
        
        accuracy = metrics.get('accuracy', 0)
        avg_confidence = metrics.get('avg_confidence', 0)
        unknown_ratio = metrics.get('unknown_intent_ratio', 1)
        
        # Grade based on accuracy, confidence, and coverage
        if accuracy >= 0.9 and avg_confidence >= 0.8 and unknown_ratio <= 0.1:
            assessment['overall_grade'] = 'excellent'
        elif accuracy >= 0.8 and avg_confidence >= 0.7 and unknown_ratio <= 0.2:
            assessment['overall_grade'] = 'good'
        elif accuracy >= 0.7 and unknown_ratio <= 0.3:
            assessment['overall_grade'] = 'fair'
        else:
            assessment['overall_grade'] = 'poor'
        
        if accuracy > 0 and accuracy < self.quality_thresholds['intent_accuracy']:
            assessment['issues'].append(f"Intent accuracy ({accuracy:.3f}) below threshold")
        elif accuracy >= 0.8:
            assessment['strengths'].append(f"High intent accuracy ({accuracy:.3f})")
        
        if unknown_ratio > 0.2:
            assessment['issues'].append(f"High unknown intent ratio ({unknown_ratio:.1%})")
        else:
            assessment['strengths'].append(f"Good intent coverage ({1-unknown_ratio:.1%})")
        
        return assessment
    
    def _assess_query_processing_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess query processing quality"""
        
        assessment = {'overall_grade': 'unknown', 'issues': [], 'strengths': []}
        
        success_rate = metrics.get('success_rate', 0)
        avg_execution_time = metrics.get('avg_execution_time', 0)
        
        # Grade based on success rate and performance
        if success_rate >= 0.95 and avg_execution_time <= 2.0:
            assessment['overall_grade'] = 'excellent'
        elif success_rate >= 0.9 and avg_execution_time <= 5.0:
            assessment['overall_grade'] = 'good'
        elif success_rate >= 0.8 and avg_execution_time <= 10.0:
            assessment['overall_grade'] = 'fair'
        else:
            assessment['overall_grade'] = 'poor'
        
        if success_rate < self.quality_thresholds['query_success_rate']:
            assessment['issues'].append(f"Query success rate ({success_rate:.1%}) below threshold")
        else:
            assessment['strengths'].append(f"High query success rate ({success_rate:.1%})")
        
        if avg_execution_time > 10.0:
            assessment['issues'].append(f"Slow query processing ({avg_execution_time:.1f}s)")
        elif avg_execution_time <= 2.0:
            assessment['strengths'].append(f"Fast query processing ({avg_execution_time:.1f}s)")
        
        return assessment
    
    # Consistency testing methods
    
    def _test_sentiment_consistency(self, sentiment_analyzer) -> Dict[str, Any]:
        """Test sentiment analysis consistency"""
        
        test_texts = [
            "This is absolutely wonderful and amazing!",
            "This is terrible and awful."
        ]
        
        results = []
        for text in test_texts:
            # Run analysis multiple times
            sentiments = []
            for _ in range(3):
                try:
                    result = sentiment_analyzer.analyze_sentiment(text)
                    if 'aggregated' in result:
                        sentiments.append(result['aggregated'].get('classification', 'neutral'))
                except:
                    sentiments.append('error')
            results.append(sentiments)
        
        # Calculate agreement
        agreement_ratios = []
        for sentiments in results:
            if sentiments:
                most_common = max(set(sentiments), key=sentiments.count)
                agreement_ratio = sentiments.count(most_common) / len(sentiments)
                agreement_ratios.append(agreement_ratio)
        
        return {
            'agreement_ratio': np.mean(agreement_ratios) if agreement_ratios else 0,
            'test_results': results
        }
    
    def _test_ner_consistency(self, ner_extractor) -> Dict[str, Any]:
        """Test NER extraction consistency"""
        
        test_text = "Apple Inc. CEO Tim Cook announced new products in California."
        
        # Run extraction multiple times
        entity_counts = []
        for _ in range(3):
            try:
                result = ner_extractor.extract_entities(test_text)
                if 'merged' in result:
                    entity_count = len(result['merged'].get('entities', []))
                    entity_counts.append(entity_count)
            except:
                entity_counts.append(0)
        
        # Calculate consistency
        if entity_counts:
            avg_count = np.mean(entity_counts)
            std_count = np.std(entity_counts)
            consistency = 1 - (std_count / max(avg_count, 1))
        else:
            consistency = 0
        
        return {
            'consistency_score': max(0, consistency),
            'entity_counts': entity_counts,
            'avg_entities': np.mean(entity_counts) if entity_counts else 0
        }
    
    def _test_topic_consistency(self, topic_modeler) -> Dict[str, Any]:
        """Test topic modeling consistency"""
        
        test_doc = "Technology companies are investing in artificial intelligence research and development."
        
        # Run topic assignment multiple times
        topic_assignments = []
        for _ in range(3):
            try:
                result = topic_modeler.get_article_topics(test_doc)
                if 'dominant_topic' in result:
                    topic_assignments.append(result['dominant_topic'])
            except:
                topic_assignments.append(-1)
        
        # Calculate stability
        if topic_assignments:
            most_common = max(set(topic_assignments), key=topic_assignments.count)
            stability = topic_assignments.count(most_common) / len(topic_assignments)
        else:
            stability = 0
        
        return {
            'stability': stability,
            'topic_assignments': topic_assignments
        }
    
    def _test_summarization_consistency(self, summarizer) -> Dict[str, Any]:
        """Test summarization consistency"""
        
        test_text = "Artificial intelligence is transforming industries worldwide. Machine learning algorithms can process vast amounts of data. Companies are investing heavily in AI research and development."
        
        # Run summarization multiple times
        summaries = []
        for _ in range(3):
            try:
                result = summarizer.summarize_article(test_text, 'brief')
                if 'summary' in result:
                    summaries.append(result['summary'])
            except:
                summaries.append("")
        
        # Calculate consistency (simplified)
        summary_lengths = [len(s) for s in summaries if s]
        if summary_lengths:
            avg_length = np.mean(summary_lengths)
            std_length = np.std(summary_lengths)
            consistency = 1 - (std_length / max(avg_length, 1))
        else:
            consistency = 0
        
        return {
            'consistency_score': max(0, consistency),
            'summary_lengths': summary_lengths,
            'summaries': summaries[:2]  # Show first 2 for comparison
        }
    
    def _generate_overall_assessment(self, component_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall system assessment"""
        
        overall_assessment = {
            'system_grade': 'unknown',
            'component_grades': {},
            'system_strengths': [],
            'system_issues': [],
            'recommendations': []
        }
        
        # Collect component grades
        grades = []
        for component, results in component_results.items():
            if results.get('status') == 'completed' and 'quality_assessment' in results:
                grade = results['quality_assessment'].get('overall_grade', 'unknown')
                overall_assessment['component_grades'][component] = grade
                
                # Convert grade to numeric for averaging
                grade_values = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1, 'unknown': 0}
                grades.append(grade_values.get(grade, 0))
        
        # Calculate overall grade
        if grades:
            avg_grade = np.mean(grades)
            if avg_grade >= 3.5:
                overall_assessment['system_grade'] = 'excellent'
            elif avg_grade >= 2.5:
                overall_assessment['system_grade'] = 'good'
            elif avg_grade >= 1.5:
                overall_assessment['system_grade'] = 'fair'
            else:
                overall_assessment['system_grade'] = 'poor'
        
        # Collect strengths and issues
        for component, results in component_results.items():
            if results.get('status') == 'completed' and 'quality_assessment' in results:
                quality_assessment = results['quality_assessment']
                
                for strength in quality_assessment.get('strengths', []):
                    overall_assessment['system_strengths'].append(f"{component}: {strength}")
                
                for issue in quality_assessment.get('issues', []):
                    overall_assessment['system_issues'].append(f"{component}: {issue}")
        
        return overall_assessment
    
    def _generate_recommendations(self, component_results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        for component, results in component_results.items():
            if results.get('status') == 'completed' and 'quality_assessment' in results:
                quality_assessment = results['quality_assessment']
                grade = quality_assessment.get('overall_grade', 'unknown')
                
                if grade in ['poor', 'fair']:
                    recommendations.append(f"Improve {component} component - currently rated as {grade}")
                
                # Specific recommendations based on issues
                for issue in quality_assessment.get('issues', []):
                    if 'accuracy' in issue.lower():
                        recommendations.append(f"Retrain {component} model with more data or better features")
                    elif 'confidence' in issue.lower():
                        recommendations.append(f"Improve confidence calibration for {component}")
                    elif 'success rate' in issue.lower():
                        recommendations.append(f"Enhance error handling and robustness for {component}")
            
            elif results.get('status') == 'not_available':
                recommendations.append(f"Enable {component} component for full system functionality")
            
            elif results.get('status') == 'failed':
                recommendations.append(f"Fix {component} component - evaluation failed")
        
        # Add general recommendations
        if len(recommendations) > 3:
            recommendations.append("Consider comprehensive system refactoring to address multiple issues")
        
        return recommendations
    
    def save_evaluation_results(self, filepath: str):
        """Save evaluation results to file"""
        
        save_data = {
            'config': self.config,
            'quality_thresholds': self.quality_thresholds,
            'evaluation_history': self.evaluation_history,
            'save_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logging.info(f"Evaluation results saved to {filepath}")
    
    def load_evaluation_results(self, filepath: str):
        """Load evaluation results from file"""
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.config = save_data['config']
        self.quality_thresholds = save_data['quality_thresholds']
        self.evaluation_history = save_data['evaluation_history']
        
        logging.info(f"Evaluation results loaded from {filepath}")
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations"""
        
        if not self.evaluation_history:
            return {'message': 'No evaluations performed yet'}
        
        latest_evaluation = self.evaluation_history[-1]
        
        summary = {
            'total_evaluations': len(self.evaluation_history),
            'latest_evaluation': {
                'timestamp': latest_evaluation['timestamp'],
                'overall_grade': latest_evaluation['overall_assessment'].get('system_grade', 'unknown'),
                'components_evaluated': len(latest_evaluation['component_results']),
                'recommendations_count': len(latest_evaluation['recommendations'])
            },
            'component_status': {}
        }
        
        # Component status from latest evaluation
        for component, results in latest_evaluation['component_results'].items():
            summary['component_status'][component] = {
                'status': results.get('status', 'unknown'),
                'grade': results.get('quality_assessment', {}).get('overall_grade', 'unknown')
            }
        
        return summary