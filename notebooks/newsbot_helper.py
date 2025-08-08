#!/usr/bin/env python3
"""
NewsBot 2.0 Notebook Helper
Provides common functions for loading trained models and data
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

def load_trained_models():
    """Load the trained NewsBot models"""
    try:
        # Load the trained classifier
        with open('../data/models/best_classifier.pkl', 'rb') as f:
            trained_classifier = pickle.load(f)
        
        # Load training metadata
        with open('../data/models/training_metadata.json', 'r') as f:
            training_metadata = json.load(f)
        
        # Import preprocessing components
        sys.path.append('..')
        from src.data_processing.text_preprocessor import TextPreprocessor
        from src.data_processing.feature_extractor import FeatureExtractor
        from src.analysis.sentiment_analyzer import SentimentAnalyzer
        
        preprocessor = TextPreprocessor()
        feature_extractor = FeatureExtractor()
        sentiment_analyzer = SentimentAnalyzer()
        
        print(f"✅ Models loaded - Accuracy: {training_metadata.get('training_results', {}).get('best_accuracy', 0):.3f}")
        
        return {
            'trained_classifier': trained_classifier,
            'training_metadata': training_metadata,
            'preprocessor': preprocessor,
            'feature_extractor': feature_extractor,
            'sentiment_analyzer': sentiment_analyzer
        }
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None

def load_bbc_dataset():
    """Load the BBC News dataset"""
    try:
        df = pd.read_csv('../data/processed/newsbot_dataset.csv')
        print(f"✅ Dataset loaded: {len(df)} articles")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

# Auto-load when imported
print("📡 NewsBot 2.0 Notebook Helper Loaded")
print("Use: models = load_trained_models()")
print("Use: df = load_bbc_dataset()")
