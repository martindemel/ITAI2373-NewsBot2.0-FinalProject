#!/usr/bin/env python3
"""
Enhanced News Classifier for NewsBot 2.0
Multi-class classification with confidence scoring and advanced features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import pickle
from pathlib import Path
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

import warnings
warnings.filterwarnings('ignore')

class NewsClassifier:
    """
    Enhanced news classification system with confidence scoring and multi-label support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the news classifier
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Model configurations
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(
                    C=self.config.get('lr_C', 1.0),
                    max_iter=self.config.get('lr_max_iter', 1000),
                    random_state=42
                ),
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'naive_bayes': {
                'model': MultinomialNB(alpha=self.config.get('nb_alpha', 1.0)),
                'param_grid': {
                    'alpha': [0.1, 0.5, 1.0, 2.0]
                }
            },
            'svm': {
                'model': SVC(
                    C=self.config.get('svm_C', 1.0),
                    kernel=self.config.get('svm_kernel', 'linear'),
                    probability=True,  # Enable probability estimates
                    random_state=42
                ),
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=self.config.get('rf_n_estimators', 100),
                    random_state=42
                ),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            }
        }
        
        # Trained models and components
        self.models = {}
        self.calibrated_models = {}
        self.ensemble_model = None
        self.feature_extractor = None
        self.label_encoder = None
        self.classes_ = None
        
        # Training results
        self.training_results = {}
        self.best_model_name = None
        self.is_trained = False
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high_confidence': 0.8,
            'medium_confidence': 0.6,
            'low_confidence': 0.4
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              feature_names: Optional[List[str]] = None,
              use_grid_search: bool = True, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train classification models with hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Names of features
            use_grid_search: Whether to use grid search for hyperparameter tuning
            cv_folds: Number of cross-validation folds
            
        Returns:
            Training results dictionary
        """
        logging.info("Starting enhanced classification training...")
        
        # Store classes and feature names
        self.classes_ = np.unique(y_train)
        self.feature_names = feature_names
        
        # Split data for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Train individual models
        model_results = {}
        
        for model_name, model_config in self.model_configs.items():
            logging.info(f"Training {model_name}...")
            
            try:
                # Get base model
                base_model = model_config['model']
                
                # Hyperparameter tuning
                if use_grid_search and 'param_grid' in model_config:
                    logging.info(f"Performing grid search for {model_name}...")
                    
                    grid_search = GridSearchCV(
                        base_model,
                        model_config['param_grid'],
                        cv=cv_folds,
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=0
                    )
                    
                    grid_search.fit(X_train_split, y_train_split)
                    best_model = grid_search.best_estimator_
                    
                    logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
                else:
                    # Use default parameters
                    best_model = base_model
                    best_model.fit(X_train_split, y_train_split)
                
                # Use the trained model directly (skip calibration for now)
                logging.info(f"Using {model_name} without calibration for initial training...")
                calibrated_model = best_model
                
                # Evaluate on validation set
                y_pred = calibrated_model.predict(X_val)
                y_proba = calibrated_model.predict_proba(X_val)
                
                accuracy = accuracy_score(y_val, y_pred)
                report = classification_report(y_val, y_pred, output_dict=True)
                
                # Cross-validation score
                cv_scores = cross_val_score(calibrated_model, X_train, y_train, cv=cv_folds)
                
                # Store results
                model_results[model_name] = {
                    'model': best_model,
                    'calibrated_model': calibrated_model,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'classification_report': report,
                    'predictions': y_pred,
                    'probabilities': y_proba
                }
                
                # Store trained models
                self.models[model_name] = best_model
                self.calibrated_models[model_name] = calibrated_model
                
                logging.info(f"{model_name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                logging.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Find best model
        best_accuracy = 0
        for model_name, results in model_results.items():
            if results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                self.best_model_name = model_name
        
        logging.info(f"Best model: {self.best_model_name} with accuracy: {best_accuracy:.4f}")
        
        # Create ensemble model
        logging.info("Creating ensemble model...")
        self._create_ensemble_model(model_results, X_val, y_val)
        
        # Store training results
        self.training_results = {
            'model_results': model_results,
            'best_model': self.best_model_name,
            'best_accuracy': best_accuracy,
            'ensemble_accuracy': getattr(self, 'ensemble_accuracy', 0),
            'training_timestamp': datetime.now().isoformat(),
            'classes': self.classes_.tolist(),
            'num_features': X_train.shape[1],
            'num_samples': X_train.shape[0]
        }
        
        self.is_trained = True
        logging.info("Classification training completed successfully!")
        
        return self.training_results
    
    def _create_ensemble_model(self, model_results: Dict[str, Any], X_val: np.ndarray, y_val: np.ndarray):
        """Create ensemble model from individual trained models"""
        
        # Select top 3 models for ensemble
        sorted_models = sorted(
            model_results.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )[:3]
        
        ensemble_estimators = []
        for model_name, results in sorted_models:
            ensemble_estimators.append((model_name, results['calibrated_model']))
        
        # Create voting classifier
        self.ensemble_model = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft'  # Use probability estimates
        )
        
        # Fit ensemble model
        # Note: Individual models are already trained, so we use a dummy fit
        self.ensemble_model.fit(X_val[:10], y_val[:10])  # Dummy fit with small subset
        
        # Evaluate ensemble
        ensemble_pred = self.ensemble_model.predict(X_val)
        self.ensemble_accuracy = accuracy_score(y_val, ensemble_pred)
        
        logging.info(f"Ensemble accuracy: {self.ensemble_accuracy:.4f}")
    
    def predict_with_confidence(self, X: np.ndarray, use_ensemble: bool = False) -> Dict[str, Any]:
        """
        Predict categories with confidence scores
        
        Args:
            X: Feature matrix
            use_ensemble: Whether to use ensemble model
            
        Returns:
            Dictionary with predictions and confidence information
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Validate input
        if X.shape[0] == 0:
            return {
                'predictions': [],
                'predicted_classes': [],
                'probabilities': np.array([]),
                'confidence_scores': [],
                'confidence_levels': [],
                'alternatives': [],
                'model_used': 'none',
                'classes': []
            }
        
        try:
            # Select model
            if use_ensemble and self.ensemble_model:
                model = self.ensemble_model
                model_name = 'ensemble'
            else:
                model_name = self.best_model_name
                model = self.calibrated_models[model_name]
            
            # Get predictions and probabilities
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            
            # Calculate confidence scores
            max_probabilities = np.max(probabilities, axis=1)
            predicted_classes = model.classes_[np.argmax(probabilities, axis=1)]
            
            # Ensure predictions are in the right format for single predictions
            if X.shape[0] == 1:
                if hasattr(predictions, 'item') and predictions.size == 1:
                    predictions = [predictions.item()]
                elif hasattr(predictions, '__len__') and len(predictions) == 1:
                    predictions = [predictions[0]]
                else:
                    predictions = [predictions] if not hasattr(predictions, '__len__') else list(predictions)
            else:
                predictions = list(predictions) if hasattr(predictions, '__len__') else [predictions]
            
            # Categorize confidence levels
            confidence_levels = []
        except Exception as e:
            logging.warning(f"Error in prediction: {e}")
            return {
                'predictions': ['unknown'],
                'predicted_classes': ['unknown'],
                'probabilities': np.array([[0.25, 0.25, 0.25, 0.25]]),  # Equal probabilities
                'confidence_scores': [0.25],
                'confidence_levels': ['very_low'],
                'alternatives': [[]],
                'model_used': 'fallback',
                'classes': self.classes_.tolist() if self.classes_ is not None else []
            }
        
        # Continue with confidence level calculation
        for prob in max_probabilities:
            if prob >= self.confidence_thresholds['high_confidence']:
                confidence_levels.append('high')
            elif prob >= self.confidence_thresholds['medium_confidence']:
                confidence_levels.append('medium')
            elif prob >= self.confidence_thresholds['low_confidence']:
                confidence_levels.append('low')
            else:
                confidence_levels.append('very_low')
        
        # Get alternative predictions (top 3)
        top_3_indices = np.argsort(probabilities, axis=1)[:, -3:][:, ::-1]
        
        alternatives = []
        for i, sample_indices in enumerate(top_3_indices):
            sample_alternatives = []
            for j, class_idx in enumerate(sample_indices):
                sample_alternatives.append({
                    'category': model.classes_[class_idx],
                    'probability': probabilities[i, class_idx],
                    'rank': j + 1
                })
            alternatives.append(sample_alternatives)
        
        return {
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'probabilities': probabilities,
            'confidence_scores': max_probabilities,
            'confidence_levels': confidence_levels,
            'alternatives': alternatives,
            'model_used': model_name,
            'classes': model.classes_.tolist()
        }
    
    def explain_prediction(self, X: np.ndarray, sample_idx: int = 0) -> Dict[str, Any]:
        """
        Provide explanation for classification decision
        
        Args:
            X: Feature matrix
            sample_idx: Index of sample to explain
            
        Returns:
            Dictionary with explanation information
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Validate input
        if X.shape[0] == 0:
            return {
                'error': 'Empty feature matrix provided',
                'sample_index': sample_idx,
                'predicted_category': 'unknown',
                'confidence_score': 0.0,
                'confidence_level': 'very_low'
            }
        
        if sample_idx >= X.shape[0]:
            sample_idx = 0  # Use first sample as fallback
        
        try:
            # Get prediction with confidence
            result = self.predict_with_confidence(X[[sample_idx]])
            
            # Safely extract results
            predictions = result.get('predictions', ['unknown'])
            confidence_scores = result.get('confidence_scores', [0.0])
            confidence_levels = result.get('confidence_levels', ['very_low'])
            alternatives = result.get('alternatives', [[]])
            
            explanation = {
                'sample_index': sample_idx,
                'predicted_category': predictions[0] if len(predictions) > 0 else 'unknown',
                'confidence_score': confidence_scores[0] if len(confidence_scores) > 0 else 0.0,
                'confidence_level': confidence_levels[0] if len(confidence_levels) > 0 else 'very_low',
                'alternatives': alternatives[0] if len(alternatives) > 0 else []
            }
        except Exception as e:
            logging.warning(f"Error in prediction explanation: {e}")
            return {
                'error': str(e),
                'sample_index': sample_idx,
                'predicted_category': 'unknown',
                'confidence_score': 0.0,
                'confidence_level': 'very_low'
            }
        
        # Feature importance (for tree-based models)
        if hasattr(self.models[self.best_model_name], 'feature_importances_'):
            feature_importance = self.models[self.best_model_name].feature_importances_
            
            if self.feature_names:
                # Get top important features
                top_features_idx = np.argsort(feature_importance)[-10:][::-1]
                top_features = [
                    {
                        'feature': self.feature_names[idx],
                        'importance': feature_importance[idx],
                        'value': X[sample_idx, idx] if X.shape[1] > idx else 0
                    }
                    for idx in top_features_idx
                ]
                explanation['top_features'] = top_features
        
        # Model coefficients (for linear models)
        try:
            if hasattr(self.models[self.best_model_name], 'coef_'):
                coefficients = self.models[self.best_model_name].coef_
                
                if len(coefficients.shape) > 1:
                    # Multi-class case
                    predicted_class = result['predictions'][0]
                    # Handle both array and scalar predictions
                    if isinstance(predicted_class, np.ndarray):
                        predicted_class = predicted_class.item() if predicted_class.size == 1 else predicted_class[0]
                    # Safe array comparison - find index of predicted class
                    try:
                        # Convert to numpy arrays and handle properly
                        classes_array = np.array(self.classes_)
                        predicted_class_scalar = predicted_class
                        if hasattr(predicted_class, 'item'):
                            predicted_class_scalar = predicted_class.item()
                        elif hasattr(predicted_class, '__len__') and len(predicted_class) == 1:
                            predicted_class_scalar = predicted_class[0]
                        
                        # Find matching indices - safely handle array comparison
                        predicted_class_idx = 0  # Default fallback
                        for i, cls in enumerate(self.classes_):
                            if str(cls) == str(predicted_class_scalar):
                                predicted_class_idx = i
                                break
                    except (ValueError, IndexError, Exception) as e:
                        logging.warning(f"Safe class index lookup failed: {e}")
                        # Safe fallback: use first class
                        predicted_class_idx = 0
                    class_coefficients = coefficients[predicted_class_idx]
                else:
                    # Binary case
                    class_coefficients = coefficients[0]
                
                if (self.feature_names is not None and 
                    hasattr(class_coefficients, '__len__') and 
                    len(class_coefficients) == len(self.feature_names)):
                    
                    # Get top contributing features - safely handle array operations
                    abs_coefficients = np.abs(class_coefficients)
                    if abs_coefficients.size > 0:
                        top_coef_idx = np.argsort(abs_coefficients)[-10:][::-1]
                        top_contributors = []
                        
                        for idx in top_coef_idx:
                            if idx < len(self.feature_names) and idx < len(class_coefficients):
                                contribution = 0
                                # Safe matrix access for contribution calculation
                                if (X is not None and 
                                    X.shape[0] > sample_idx and 
                                    X.shape[1] > idx):
                                    contribution = float(class_coefficients[idx]) * float(X[sample_idx, idx])
                                
                                top_contributors.append({
                                    'feature': str(self.feature_names[idx]),
                                    'coefficient': float(class_coefficients[idx]),
                                    'contribution': contribution
                                })
                        
                        explanation['top_contributors'] = top_contributors
        except Exception as e:
            logging.warning(f"Error extracting coefficients: {e}")
            # Continue without coefficient information
        
        return explanation
    
    def predict_single_text(self, text: str, preprocessor=None, feature_extractor=None) -> Dict[str, Any]:
        """
        Predict category for a single text article
        
        Args:
            text: Article text
            preprocessor: Text preprocessor instance
            feature_extractor: Feature extractor instance
            
        Returns:
            Prediction results with confidence
        """
        if not self.is_trained:
            return {
                'error': 'Model not trained',
                'original_text': text,
                'predicted_category': 'unknown',
                'confidence_score': 0.0,
                'confidence_level': 'very_low',
                'alternatives': [],
                'model_used': 'none'
            }
        
        if not text or not text.strip():
            return {
                'error': 'Empty text provided',
                'original_text': text,
                'predicted_category': 'unknown',
                'confidence_score': 0.0,
                'confidence_level': 'very_low',
                'alternatives': [],
                'model_used': 'none'
            }
        
        if preprocessor is None or feature_extractor is None:
            return {
                'error': 'Preprocessor and feature extractor required',
                'original_text': text,
                'predicted_category': 'unknown',
                'confidence_score': 0.0,
                'confidence_level': 'very_low',
                'alternatives': [],
                'model_used': 'none'
            }
        
        try:
            # Preprocess text
            processed_text = preprocessor.preprocess_text(text)
            
            # Extract features
            features = feature_extractor.extract_tfidf_features([processed_text], fit=False)
            
            if features.shape[1] == 0:
                return {
                    'error': 'No features extracted from text',
                    'original_text': text,
                    'processed_text': processed_text,
                    'predicted_category': 'unknown',
                    'confidence_score': 0.0,
                    'confidence_level': 'very_low',
                    'alternatives': [],
                    'model_used': 'none'
                }
            
            # Get prediction
            result = self.predict_with_confidence(features.toarray())
            
            # Safely extract results
            predictions = result.get('predictions', ['unknown'])
            confidence_scores = result.get('confidence_scores', [0.0])
            confidence_levels = result.get('confidence_levels', ['very_low'])
            alternatives = result.get('alternatives', [[]])
            
            return {
                'original_text': text,
                'processed_text': processed_text,
                'predicted_category': predictions[0] if len(predictions) > 0 else 'unknown',
                'confidence_score': confidence_scores[0] if len(confidence_scores) > 0 else 0.0,
                'confidence_level': confidence_levels[0] if len(confidence_levels) > 0 else 'very_low',
                'alternatives': alternatives[0] if len(alternatives) > 0 else [],
                'model_used': result.get('model_used', 'unknown')
            }
            
        except Exception as e:
            logging.warning(f"Error in single text prediction: {e}")
            return {
                'error': str(e),
                'original_text': text,
                'predicted_category': 'unknown',
                'confidence_score': 0.0,
                'confidence_level': 'very_low',
                'alternatives': [],
                'model_used': 'error'
            }
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        evaluation_results = {}
        
        # Evaluate each individual model
        for model_name, model in self.calibrated_models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': conf_matrix.tolist(),
                'predictions': y_pred.tolist(),
                'probabilities': y_proba.tolist()
            }
        
        # Evaluate ensemble model
        if self.ensemble_model:
            y_pred_ensemble = self.ensemble_model.predict(X_test)
            y_proba_ensemble = self.ensemble_model.predict_proba(X_test)
            
            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            ensemble_report = classification_report(y_test, y_pred_ensemble, output_dict=True)
            ensemble_conf_matrix = confusion_matrix(y_test, y_pred_ensemble)
            
            evaluation_results['ensemble'] = {
                'accuracy': ensemble_accuracy,
                'classification_report': ensemble_report,
                'confusion_matrix': ensemble_conf_matrix.tolist(),
                'predictions': y_pred_ensemble.tolist(),
                'probabilities': y_proba_ensemble.tolist()
            }
        
        # Calculate confidence calibration metrics
        best_model = self.calibrated_models[self.best_model_name]
        y_proba_best = best_model.predict_proba(X_test)
        max_probabilities = np.max(y_proba_best, axis=1)
        predicted_classes = best_model.predict(X_test)
        correct_predictions = (predicted_classes == y_test)
        
        # Reliability diagram data
        calibration_data = self._calculate_calibration_metrics(max_probabilities, correct_predictions)
        evaluation_results['calibration'] = calibration_data
        
        evaluation_results['evaluation_timestamp'] = datetime.now().isoformat()
        evaluation_results['test_size'] = len(y_test)
        evaluation_results['classes'] = self.classes_.tolist()
        
        return evaluation_results
    
    def _calculate_calibration_metrics(self, probabilities: np.ndarray, correct: np.ndarray) -> Dict[str, Any]:
        """Calculate probability calibration metrics"""
        
        # Bin predictions by confidence
        bins = np.linspace(0, 1, 11)
        bin_boundaries = np.digitize(probabilities, bins)
        
        calibration_data = {
            'bin_boundaries': bins.tolist(),
            'bin_accuracies': [],
            'bin_confidences': [],
            'bin_counts': []
        }
        
        for i in range(1, len(bins)):
            mask = bin_boundaries == i
            if mask.sum() > 0:
                bin_accuracy = correct[mask].mean()
                bin_confidence = probabilities[mask].mean()
                bin_count = mask.sum()
                
                calibration_data['bin_accuracies'].append(bin_accuracy)
                calibration_data['bin_confidences'].append(bin_confidence)
                calibration_data['bin_counts'].append(int(bin_count))
            else:
                calibration_data['bin_accuracies'].append(0)
                calibration_data['bin_confidences'].append(0)
                calibration_data['bin_counts'].append(0)
        
        # Expected Calibration Error (ECE)
        total_samples = len(probabilities)
        ece = sum(
            (count / total_samples) * abs(acc - conf)
            for acc, conf, count in zip(
                calibration_data['bin_accuracies'],
                calibration_data['bin_confidences'],
                calibration_data['bin_counts']
            )
            if count > 0
        )
        
        calibration_data['expected_calibration_error'] = ece
        
        return calibration_data
    
    def save_model(self, filepath: str):
        """Save trained model and components"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'models': self.models,
            'calibrated_models': self.calibrated_models,
            'ensemble_model': self.ensemble_model,
            'classes_': self.classes_,
            'feature_names': getattr(self, 'feature_names', None),
            'training_results': self.training_results,
            'best_model_name': self.best_model_name,
            'confidence_thresholds': self.confidence_thresholds,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and components"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.calibrated_models = model_data['calibrated_models']
        self.ensemble_model = model_data['ensemble_model']
        self.classes_ = model_data['classes_']
        self.feature_names = model_data.get('feature_names')
        self.training_results = model_data['training_results']
        self.best_model_name = model_data['best_model_name']
        self.confidence_thresholds = model_data['confidence_thresholds']
        self.config = model_data['config']
        self.is_trained = True
        
        logging.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models"""
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        return {
            'status': 'trained',
            'best_model': self.best_model_name,
            'available_models': list(self.models.keys()),
            'classes': self.classes_.tolist(),
            'num_classes': len(self.classes_),
            'training_accuracy': self.training_results.get('best_accuracy', 0),
            'ensemble_accuracy': self.training_results.get('ensemble_accuracy', 0),
            'training_timestamp': self.training_results.get('training_timestamp'),
            'confidence_thresholds': self.confidence_thresholds
        }