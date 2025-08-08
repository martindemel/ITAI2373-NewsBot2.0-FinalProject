#!/usr/bin/env python3
"""
Transformer Fine-tuning for NewsBot 2.0
Advanced research extension for fine-tuning BERT/RoBERTa models
Bonus Feature: Advanced Research Extensions (20 points)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import torch
from torch.utils.data import DataLoader, Dataset
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, pipeline
    )
    from datasets import Dataset as HFDataset
    import torch.nn.functional as F
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("Transformers not available for fine-tuning")

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class NewsDataset(Dataset):
    """Custom dataset for news classification fine-tuning"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TransformerFineTuner:
    """
    Advanced transformer fine-tuning for news classification
    Implements cutting-edge techniques for domain adaptation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize transformer fine-tuner
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Model configuration
        self.model_name = self.config.get('model_name', 'bert-base-uncased')
        self.num_labels = self.config.get('num_labels', 5)  # BBC News categories
        self.max_length = self.config.get('max_length', 512)
        
        # Training configuration
        self.batch_size = self.config.get('batch_size', 16)
        self.learning_rate = self.config.get('learning_rate', 2e-5)
        self.num_epochs = self.config.get('num_epochs', 3)
        self.warmup_steps = self.config.get('warmup_steps', 500)
        self.weight_decay = self.config.get('weight_decay', 0.01)
        
        # Paths
        self.model_save_path = self.config.get('model_save_path', 'data/models/fine_tuned')
        self.results_path = self.config.get('results_path', 'data/results')
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.label_encoder = None
        
        # Training history
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'f1_scores': [],
            'learning_rates': []
        }
        
        # Create directories
        Path(self.model_save_path).mkdir(parents=True, exist_ok=True)
        Path(self.results_path).mkdir(parents=True, exist_ok=True)
        
        if HAS_TRANSFORMERS:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize tokenizer and model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )
            logging.info(f"Initialized {self.model_name} for fine-tuning")
        except Exception as e:
            logging.error(f"Failed to initialize model: {e}")
    
    def prepare_data(self, df: pd.DataFrame, text_column: str = 'text', 
                    label_column: str = 'category') -> Tuple[Dataset, Dataset, Dict[str, int]]:
        """
        Prepare data for fine-tuning
        
        Args:
            df: DataFrame with text and labels
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Training dataset, validation dataset, label mapping
        """
        if not HAS_TRANSFORMERS:
            raise ValueError("Transformers not available")
        
        # Prepare texts and labels
        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].tolist()
        
        # Create label encoding
        unique_labels = sorted(list(set(labels)))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        encoded_labels = [self.label_encoder[label] for label in labels]
        
        # Split data
        if HAS_SKLEARN:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
            )
        else:
            # Simple split without stratification
            split_idx = int(0.8 * len(texts))
            train_texts, val_texts = texts[:split_idx], texts[split_idx:]
            train_labels, val_labels = encoded_labels[:split_idx], encoded_labels[split_idx:]
        
        # Create datasets
        train_dataset = NewsDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = NewsDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        logging.info(f"Prepared data: {len(train_dataset)} training, {len(val_dataset)} validation samples")
        
        return train_dataset, val_dataset, self.label_encoder
    
    def fine_tune(self, train_dataset: Dataset, val_dataset: Dataset) -> Dict[str, Any]:
        """
        Fine-tune the transformer model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Training results and metrics
        """
        if not HAS_TRANSFORMERS:
            return {'error': 'Transformers not available'}
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.model_save_path,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            logging_dir=f'{self.results_path}/logs',
            logging_steps=100,
            evaluation_strategy='steps',
            eval_steps=500,
            save_strategy='steps',
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            report_to=None  # Disable wandb/tensorboard
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            tokenizer=self.tokenizer
        )
        
        # Start training
        logging.info("Starting fine-tuning...")
        start_time = datetime.now()
        
        try:
            train_result = self.trainer.train()
            
            # Save the model
            self.trainer.save_model(self.model_save_path)
            self.tokenizer.save_pretrained(self.model_save_path)
            
            # Evaluate on validation set
            eval_result = self.trainer.evaluate()
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare results
            results = {
                'training_loss': train_result.training_loss,
                'eval_loss': eval_result['eval_loss'],
                'eval_accuracy': eval_result.get('eval_accuracy', 0),
                'eval_f1': eval_result.get('eval_f1', 0),
                'training_time': training_time,
                'num_epochs': self.num_epochs,
                'model_name': self.model_name,
                'model_path': self.model_save_path,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save results
            self._save_training_results(results)
            
            logging.info(f"Fine-tuning completed in {training_time:.2f} seconds")
            logging.info(f"Final validation accuracy: {results['eval_accuracy']:.4f}")
            
            return results
            
        except Exception as e:
            logging.error(f"Fine-tuning failed: {e}")
            return {'error': f'Fine-tuning failed: {str(e)}'}
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        if not HAS_SKLEARN:
            return {}
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def evaluate_model(self, test_dataset: Dataset) -> Dict[str, Any]:
        """
        Comprehensive evaluation of fine-tuned model
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Detailed evaluation metrics
        """
        if not self.trainer:
            return {'error': 'Model not trained yet'}
        
        # Evaluate on test set
        eval_results = self.trainer.evaluate(test_dataset)
        
        # Get predictions for detailed analysis
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Calculate detailed metrics
        if HAS_SKLEARN:
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Per-class metrics
            label_names = list(self.label_encoder.keys())
            per_class_metrics = {}
            
            for i, label in enumerate(label_names):
                per_class_metrics[label] = {
                    'precision': precision[i] if i < len(precision) else 0,
                    'recall': recall[i] if i < len(recall) else 0,
                    'f1': f1[i] if i < len(f1) else 0,
                    'support': support[i] if i < len(support) else 0
                }
            
            results = {
                'overall_accuracy': accuracy,
                'weighted_f1': f1.mean(),
                'weighted_precision': precision.mean(),
                'weighted_recall': recall.mean(),
                'per_class_metrics': per_class_metrics,
                'confusion_matrix': cm.tolist(),
                'label_names': label_names
            }
        else:
            results = {
                'overall_accuracy': eval_results.get('eval_accuracy', 0),
                'note': 'Limited metrics due to missing sklearn'
            }
        
        return results
    
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Make predictions on new texts
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of prediction results
        """
        if not self.model or not self.tokenizer:
            return [{'error': 'Model not loaded'}] * len(texts)
        
        # Create pipeline for inference
        classifier = pipeline(
            'text-classification',
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True
        )
        
        results = []
        
        for text in texts:
            try:
                # Get predictions
                predictions = classifier(text[:self.max_length])
                
                # Convert predictions to readable format
                if self.label_encoder:
                    reverse_encoder = {v: k for k, v in self.label_encoder.items()}
                    
                    formatted_predictions = []
                    for pred in predictions:
                        label_id = int(pred['label'].split('_')[-1]) if '_' in pred['label'] else int(pred['label'])
                        label_name = reverse_encoder.get(label_id, f'LABEL_{label_id}')
                        
                        formatted_predictions.append({
                            'label': label_name,
                            'confidence': pred['score']
                        })
                    
                    # Sort by confidence
                    formatted_predictions.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    results.append({
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'predictions': formatted_predictions,
                        'top_prediction': formatted_predictions[0] if formatted_predictions else None
                    })
                else:
                    results.append({
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'raw_predictions': predictions
                    })
                    
            except Exception as e:
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'error': str(e)
                })
        
        return results
    
    def load_fine_tuned_model(self, model_path: str = None) -> bool:
        """
        Load a previously fine-tuned model
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            True if successful, False otherwise
        """
        if not HAS_TRANSFORMERS:
            logging.error("Transformers not available")
            return False
        
        model_path = model_path or self.model_save_path
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Load label encoder if available
            label_encoder_path = Path(model_path) / 'label_encoder.json'
            if label_encoder_path.exists():
                with open(label_encoder_path, 'r') as f:
                    self.label_encoder = json.load(f)
            
            logging.info(f"Loaded fine-tuned model from {model_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return False
    
    def _save_training_results(self, results: Dict[str, Any]):
        """Save training results and metadata"""
        
        # Save training results
        results_file = Path(self.results_path) / 'fine_tuning_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save label encoder
        if self.label_encoder:
            label_encoder_file = Path(self.model_save_path) / 'label_encoder.json'
            with open(label_encoder_file, 'w') as f:
                json.dump(self.label_encoder, f, indent=2)
        
        # Save model configuration
        config = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'training_date': datetime.now().isoformat()
        }
        
        config_file = Path(self.model_save_path) / 'training_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"Training results saved to {self.results_path}")
    
    def compare_with_baseline(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare fine-tuned model performance with baseline
        
        Args:
            baseline_results: Baseline model performance metrics
            
        Returns:
            Comparison results
        """
        if not hasattr(self, 'last_eval_results'):
            return {'error': 'No fine-tuned model results available'}
        
        comparison = {
            'baseline_accuracy': baseline_results.get('accuracy', 0),
            'fine_tuned_accuracy': self.last_eval_results.get('overall_accuracy', 0),
            'improvement': 0,
            'relative_improvement': 0
        }
        
        # Calculate improvements
        baseline_acc = comparison['baseline_accuracy']
        fine_tuned_acc = comparison['fine_tuned_accuracy']
        
        if baseline_acc > 0:
            comparison['improvement'] = fine_tuned_acc - baseline_acc
            comparison['relative_improvement'] = (fine_tuned_acc - baseline_acc) / baseline_acc
        
        # Detailed comparison
        if 'per_class_metrics' in baseline_results and 'per_class_metrics' in self.last_eval_results:
            comparison['per_class_comparison'] = {}
            
            for label in self.last_eval_results['per_class_metrics']:
                if label in baseline_results['per_class_metrics']:
                    baseline_f1 = baseline_results['per_class_metrics'][label].get('f1', 0)
                    fine_tuned_f1 = self.last_eval_results['per_class_metrics'][label].get('f1', 0)
                    
                    comparison['per_class_comparison'][label] = {
                        'baseline_f1': baseline_f1,
                        'fine_tuned_f1': fine_tuned_f1,
                        'improvement': fine_tuned_f1 - baseline_f1
                    }
        
        return comparison
    
    def get_model_insights(self) -> Dict[str, Any]:
        """
        Generate insights about the fine-tuned model
        
        Returns:
            Model insights and recommendations
        """
        insights = {
            'model_type': 'transformer_fine_tuned',
            'base_model': self.model_name,
            'parameters': {
                'num_labels': self.num_labels,
                'max_length': self.max_length,
                'learning_rate': self.learning_rate,
                'epochs': self.num_epochs
            },
            'recommendations': []
        }
        
        # Add performance-based recommendations
        if hasattr(self, 'last_eval_results'):
            accuracy = self.last_eval_results.get('overall_accuracy', 0)
            
            if accuracy < 0.8:
                insights['recommendations'].append(
                    "Consider increasing training epochs or adjusting learning rate"
                )
            
            if accuracy > 0.95:
                insights['recommendations'].append(
                    "High accuracy achieved - check for overfitting on validation set"
                )
            
            # Check class balance
            if 'per_class_metrics' in self.last_eval_results:
                f1_scores = [
                    metrics['f1'] for metrics in self.last_eval_results['per_class_metrics'].values()
                ]
                
                if max(f1_scores) - min(f1_scores) > 0.2:
                    insights['recommendations'].append(
                        "Class imbalance detected - consider data augmentation or class weighting"
                    )
        
        return insights