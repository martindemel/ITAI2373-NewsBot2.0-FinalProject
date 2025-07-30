#!/usr/bin/env python3
"""
NewsBot Intelligence System - Environment Setup Script
Automatically downloads and installs required models and data for local development.
"""

import subprocess
import sys
import os
import nltk
import spacy
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors gracefully."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}: {e}")
        print(f"Output: {e.output}")
        return False

def setup_spacy_models():
    """Download required spaCy models."""
    models = [
        ("en_core_web_sm", "English small model"),
        ("en_core_web_md", "English medium model for better vectors")
    ]
    
    for model, description in models:
        print(f"\nInstalling spaCy model: {model}")
        if run_command(f"python -m spacy download {model}", f"spaCy {description}"):
            # Verify installation
            try:
                nlp = spacy.load(model)
                print(f"{model} loaded successfully")
            except OSError:
                print(f"{model} installation may have failed")

def setup_nltk_data():
    """Download required NLTK data packages."""
    print("\nSetting up NLTK data...")
    
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    nltk_data = [
        'punkt',           # Sentence tokenizer
        'stopwords',       # Stop words
        'averaged_perceptron_tagger',  # POS tagger
        'wordnet',         # WordNet lemmatizer
        'vader_lexicon',   # VADER sentiment lexicon
        'omw-1.4'         # Open Multilingual Wordnet
    ]
    
    for data_package in nltk_data:
        try:
            nltk.download(data_package, quiet=True)
            print(f"Downloaded {data_package}")
        except Exception as e:
            print(f"Error downloading {data_package}: {e}")

def create_data_directories():
    """Create necessary data directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/models",
        "outputs/visualizations",
        "outputs/reports",
        "dashboard/assets"
    ]
    
    print("\nCreating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")

def verify_installation():
    """Verify that all components are properly installed."""
    print("\nVerifying installation...")
    
    # Test spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("This is a test sentence.")
        print("spaCy working correctly")
    except Exception as e:
        print(f"spaCy verification failed: {e}")
    
    # Test NLTK
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.sentiment import SentimentIntensityAnalyzer
        stopwords.words('english')
        sia = SentimentIntensityAnalyzer()
        print("NLTK working correctly")
    except Exception as e:
        print(f"NLTK verification failed: {e}")
    
    # Test other key libraries
    libraries = ['sklearn', 'pandas', 'matplotlib', 'seaborn', 'textblob', 'streamlit']
    for lib in libraries:
        try:
            __import__(lib)
            print(f"{lib} available")
        except ImportError:
            print(f"{lib} not available")

def main():
    """Main setup function."""
    print("NewsBot Intelligence System - Environment Setup")
    print("=" * 50)
    
    # Create directories
    create_data_directories()
    
    # Setup spaCy models
    setup_spacy_models()
    
    # Setup NLTK data
    setup_nltk_data()
    
    # Verify installation
    verify_installation()
    
    print("\n" + "=" * 50)
    print("Setup completed! Your NewsBot environment is ready.")
    print("\nNext steps:")
    print("1. Run: jupyter notebook NewsBot_Intelligence_System.ipynb")
    print("2. Or run: streamlit run dashboard/newsbot_dashboard.py")
    print("\nFor any issues, check the requirements.txt and ensure all dependencies are installed.")

if __name__ == "__main__":
    main() 