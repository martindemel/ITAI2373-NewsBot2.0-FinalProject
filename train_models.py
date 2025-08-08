#!/usr/bin/env python3
"""
Train NewsBot 2.0 Models
This script trains actual machine learning models using the real BBC News dataset
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append('src')

# Import NewsBot components
from src.analysis.classifier import NewsClassifier
from src.data_processing.text_preprocessor import TextPreprocessor
from src.data_processing.feature_extractor import FeatureExtractor
from src.analysis.sentiment_analyzer import SentimentAnalyzer

def main():
    print("=== NewsBot 2.0 Model Training ===")
    print(f"Training started at: {datetime.now()}")
    
    # Load the dataset
    print("\n1. Loading BBC News Dataset...")
    try:
        df = pd.read_csv('data/processed/newsbot_dataset.csv')
        print(f"✅ Dataset loaded: {len(df)} articles")
        print(f"Categories: {df['category'].value_counts().to_dict()}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Initialize components
    print("\n2. Initializing NewsBot Components...")
    classifier = NewsClassifier()
    preprocessor = TextPreprocessor()
    feature_extractor = FeatureExtractor()
    sentiment_analyzer = SentimentAnalyzer()
    
    # Prepare the data
    print("\n3. Preprocessing Data...")
    
    # Preprocess text data
    print("   - Preprocessing text...")
    texts = df['text'].tolist()
    preprocessed_texts = []
    for text in texts:
        processed = preprocessor.preprocess_text(text)
        preprocessed_texts.append(processed)
    
    # Extract features
    print("   - Extracting features...")
    features_dict = feature_extractor.extract_all_features(preprocessed_texts)
    
    print(f"   - Available features: {list(features_dict.keys())}")
    
    # Use TF-IDF features as the main feature matrix
    if 'tfidf' in features_dict:
        X = features_dict['tfidf']
    elif 'combined_features' in features_dict:
        X = features_dict['combined_features']
    else:
        # Use the first available feature matrix
        feature_key = list(features_dict.keys())[0]
        X = features_dict[feature_key]
        print(f"   - Using feature type: {feature_key}")
    
    y = df['category'].values
    
    print(f"   - Feature matrix shape: {X.shape}")
    print(f"   - Labels shape: {y.shape}")
    
    # Train the classifier
    print("\n4. Training Classification Models...")
    try:
        # Get feature names for training
        feature_names = features_dict.get('tfidf_feature_names', None)
        classifier.train(X, y, feature_names, use_grid_search=False, cv_folds=2)
        print("✅ Classification models trained successfully")
        
        # Save the trained classifier using the proper save method
        model_path = Path('data/models/best_classifier.pkl')
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        classifier.save_model(str(model_path))
        print(f"✅ Model saved to {model_path}")
        
        # CRITICAL: Save the feature extractor with the trained vectorizer
        feature_extractor_path = Path('data/models/feature_extractor.pkl')
        feature_extractor_data = {
            'tfidf_vectorizer': feature_extractor.tfidf_vectorizer,
            'count_vectorizer': feature_extractor.count_vectorizer,
            'config': feature_extractor.config,
            'tfidf_params': feature_extractor.tfidf_params
        }
        
        with open(feature_extractor_path, 'wb') as f:
            pickle.dump(feature_extractor_data, f)
        print(f"✅ Feature extractor saved to {feature_extractor_path}")
        
        # Get training results
        results = classifier.training_results
        print(f"Best model: {results.get('best_model', 'Unknown')}")
        print(f"Best accuracy: {results.get('best_accuracy', 0):.3f}")
        
    except Exception as e:
        print(f"❌ Error training classifier: {e}")
        return
    
    # Test sentiment analyzer
    print("\n5. Testing Sentiment Models...")
    try:
        # Test sentiment analyzer with a few samples
        test_texts = df['text'].head(5).tolist()
        for i, text in enumerate(test_texts):
            sentiment = sentiment_analyzer.analyze_sentiment(text)
            print(f"   Sample {i+1}: {sentiment.get('label', 'unknown')} ({sentiment.get('score', 0):.3f})")
        print("✅ Sentiment analyzer tested successfully")
    except Exception as e:
        print(f"⚠️ Warning: Sentiment analyzer error: {e}")
    
    # Train topic model
    print("\n5.5. Training Topic Model...")
    try:
        from src.analysis.topic_modeler import TopicModeler
        
        # Initialize topic modeler
        topic_config = {'n_topics': 5, 'method': 'lda', 'random_state': 42}
        topic_modeler = TopicModeler(topic_config)
        
        # Use preprocessed texts for topic modeling
        preprocessed_texts = [preprocessor.preprocess_text(text) for text in df['text'].head(1000)]
        
        # Train topic model
        topic_modeler.fit_topics(preprocessed_texts, method='lda', n_topics=5)
        
        # Save topic model
        topic_model_path = Path('data/models/topic_model.pkl')
        topic_modeler.save_topic_model(str(topic_model_path))
        print(f"✅ Topic model trained and saved to {topic_model_path}")
        
    except Exception as e:
        print(f"⚠️ Warning: Topic model training error: {e}")
    
    # Save training metadata
    print("\n6. Saving Training Metadata...")
    # Create safe metadata - only include serializable information
    serializable_results = {}
    if hasattr(classifier, 'training_results') and classifier.training_results:
        # Only copy safe, serializable values
        safe_keys = ['best_model_name', 'best_accuracy', 'training_timestamp', 'classes', 'num_features', 'num_samples']
        for key in safe_keys:
            if key in classifier.training_results:
                value = classifier.training_results[key]
                if isinstance(value, (str, int, float, bool, type(None))):
                    serializable_results[key] = value
                elif isinstance(value, (list, tuple)):
                    serializable_results[key] = list(value)
                elif hasattr(value, 'tolist'):  # numpy arrays
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = str(value)
        
        # Add model name if available
        if hasattr(classifier, 'best_model_name'):
            serializable_results['best_model'] = classifier.best_model_name
    
    metadata = {
        'training_date': datetime.now().isoformat(),
        'dataset_size': len(df),
        'categories': df['category'].value_counts().to_dict(),
        'model_type': 'NewsBot 2.0 Advanced Classification',
        'features': 'TF-IDF + Custom Features',
        'validation_split': 0.2,
        'training_results': serializable_results
    }
    
    metadata_path = Path('data/models/training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Training metadata saved to {metadata_path}")
    print(f"\n🎉 Model training completed successfully!")
    print(f"Training finished at: {datetime.now()}")

if __name__ == "__main__":
    main()