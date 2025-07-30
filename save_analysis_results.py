#!/usr/bin/env python3
"""
NewsBot Intelligence System - Analysis Results Saver
Helper module to save analysis results for dashboard consumption.
Import this in the notebook to save results.
"""

import json
import pickle
from pathlib import Path
from datetime import datetime

def save_sentiment_results(sentiment_df, sentiment_insights):
    """Save sentiment analysis results for dashboard."""
    
    # Create outputs directory
    Path("outputs/analysis_results").mkdir(parents=True, exist_ok=True)
    
    # Convert insights to JSON-serializable format
    def make_json_serializable(obj):
        """Convert numpy/pandas objects to JSON-serializable types."""
        if hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, 'to_dict'):  # pandas objects
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        else:
            return obj
    
    # Safely convert insights
    safe_insights = make_json_serializable(sentiment_insights)
    
    # Prepare sentiment results for dashboard
    sentiment_results = {
        'sentiment_by_category': sentiment_df.groupby('category').agg({
            'vader_compound': 'mean',
            'textblob_polarity': 'mean', 
            'textblob_subjectivity': 'mean'
        }).to_dict(),
        'sentiment_distribution': sentiment_df['sentiment_class'].value_counts().to_dict(),
        'correlation_vader_textblob': float(sentiment_df[['vader_compound', 'textblob_polarity']].corr().iloc[0,1]),
        'total_articles': int(len(sentiment_df)),
        'insights': safe_insights,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save sentiment results with error handling
    try:
        with open('outputs/analysis_results/sentiment_results.json', 'w') as f:
            json.dump(sentiment_results, f, indent=2, ensure_ascii=False)
        print("Sentiment analysis results saved for dashboard")
    except Exception as e:
        print(f"Error saving sentiment results: {e}")
        # Save a minimal version if full save fails
        minimal_results = {
            'sentiment_by_category': sentiment_df.groupby('category').agg({
                'vader_compound': 'mean',
                'textblob_polarity': 'mean', 
                'textblob_subjectivity': 'mean'
            }).to_dict(),
            'sentiment_distribution': sentiment_df['sentiment_class'].value_counts().to_dict(),
            'correlation_vader_textblob': float(sentiment_df[['vader_compound', 'textblob_polarity']].corr().iloc[0,1]),
            'total_articles': int(len(sentiment_df)),
            'insights': {"status": "Insights data could not be serialized"},
            'timestamp': datetime.now().isoformat()
        }
        with open('outputs/analysis_results/sentiment_results.json', 'w') as f:
            json.dump(minimal_results, f, indent=2)
        print("Minimal sentiment results saved successfully")
    
    return sentiment_results

def save_classification_results(evaluation_results, best_model_name, ensemble_accuracy, y_test, classifier_system):
    """Save classification results for dashboard."""
    
    # Create outputs directory
    Path("outputs/analysis_results").mkdir(parents=True, exist_ok=True)
    Path("data/models").mkdir(parents=True, exist_ok=True)
    
    # Import here to avoid dependency issues
    from sklearn.metrics import confusion_matrix
    
    # Prepare classification results for dashboard
    conf_matrix = confusion_matrix(y_test, evaluation_results[best_model_name]['predictions'])
    
    classification_results = {
        'model_performance': {
            model_name: {
                'accuracy': float(results['accuracy']),
                'classification_report': results['classification_report']
            } for model_name, results in evaluation_results.items()
        },
        'best_model': best_model_name,
        'best_accuracy': float(evaluation_results[best_model_name]['accuracy']),
        'ensemble_accuracy': float(ensemble_accuracy),
        'confusion_matrix': conf_matrix.tolist(),
        'categories': sorted(list(set(y_test))),
        'test_size': int(len(y_test)),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save classification results
    with open('outputs/analysis_results/classification_results.json', 'w') as f:
        json.dump(classification_results, f, indent=2)
    
    # Save the best model
    with open('data/models/best_classifier.pkl', 'wb') as f:
        pickle.dump(classifier_system.pipelines[best_model_name], f)
    
    print("Classification results and best model saved for dashboard")
    return classification_results

def save_preprocessing_results(df_processed, tfidf_vectorizer=None):
    """Save preprocessing results for dashboard."""
    
    # Create outputs directory
    Path("outputs/analysis_results").mkdir(parents=True, exist_ok=True)
    Path("data/models").mkdir(parents=True, exist_ok=True)
    
    # Save processed dataset with analysis-ready features
    analysis_df = df_processed.copy()
    analysis_df.to_csv('outputs/analysis_results/processed_with_features.csv', index=False)
    
    # Save TF-IDF vectorizer if provided
    if tfidf_vectorizer is not None:
        with open('data/models/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
    
    print("Preprocessing results saved for dashboard")

def auto_save_all_results():
    """
    AUTO-SAVE FUNCTION for easy use in notebooks.
    Automatically detects and saves all available analysis results.
    """
    
    print("🔍 Auto-detecting analysis results to save...")
    
    # Get the calling frame to access variables
    import inspect
    frame = inspect.currentframe().f_back
    local_vars = frame.f_locals
    global_vars = frame.f_globals
    
    # Combine local and global variables
    all_vars = {**global_vars, **local_vars}
    
    saved_count = 0
    
    # Try to save sentiment results
    sentiment_vars = ['sentiment_df', 'sentiment_insights']
    if all(var in all_vars for var in sentiment_vars):
        try:
            save_sentiment_results(all_vars['sentiment_df'], all_vars['sentiment_insights'])
            saved_count += 1
        except Exception as e:
            print(f"Warning: Could not save sentiment results: {e}")
    else:
        print("Info: Sentiment analysis results not found - run sentiment analysis first")
    
    # Try to save classification results
    classification_vars = ['evaluation_results', 'best_model_name', 'ensemble_accuracy', 'y_test', 'classifier_system']
    if all(var in all_vars for var in classification_vars):
        try:
            save_classification_results(
                all_vars['evaluation_results'], 
                all_vars['best_model_name'], 
                all_vars['ensemble_accuracy'], 
                all_vars['y_test'], 
                all_vars['classifier_system']
            )
            saved_count += 1
        except Exception as e:
            print(f"Warning: Could not save classification results: {e}")
    else:
        print("Info: Classification results not found - run classification analysis first")
    
    # Try to save preprocessing results
    preprocessing_vars = ['df_processed']
    if 'df_processed' in all_vars:
        try:
            tfidf_vectorizer = all_vars.get('tfidf_vectorizer', None)
            save_preprocessing_results(all_vars['df_processed'], tfidf_vectorizer)
            saved_count += 1
        except Exception as e:
            print(f"Warning: Could not save preprocessing results: {e}")
    else:
        print("Info: Processed data not found - run preprocessing first")
    
    if saved_count > 0:
        print(f"\nSuccessfully saved {saved_count} analysis result(s)!")
        print("You can now launch the dashboard: python3 main.py")
    else:
        print("\nNo analysis results found to save")
        print("Run analysis sections in the notebook first")

def load_analysis_results():
    """Load all saved analysis results."""
    results = {}
    
    # Load sentiment results
    sentiment_path = Path("outputs/analysis_results/sentiment_results.json")
    if sentiment_path.exists():
        with open(sentiment_path, 'r') as f:
            results['sentiment'] = json.load(f)
    
    # Load classification results
    classification_path = Path("outputs/analysis_results/classification_results.json")
    if classification_path.exists():
        with open(classification_path, 'r') as f:
            results['classification'] = json.load(f)
    
    return results 