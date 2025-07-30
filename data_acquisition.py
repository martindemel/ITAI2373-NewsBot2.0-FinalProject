#!/usr/bin/env python3
"""
NewsBot Intelligence System - Data Acquisition Script
Downloads and prepares the BBC News Classification dataset for local analysis.
"""

import os
import pandas as pd
import requests
import zipfile
from pathlib import Path
import json
import sys

def download_bbc_dataset():
    """
    Download BBC News Classification dataset.
    Note: This uses a publicly available version of the BBC dataset.
    """
    print("Downloading BBC News Classification dataset...")
    
    # Create data directories
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # BBC News dataset URL (publicly available version)
    # This is a curated version that doesn't require Kaggle API
    dataset_url = "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"
    
    try:
        response = requests.get(dataset_url)
        response.raise_for_status()
        
        # Save the dataset
        dataset_path = data_dir / "bbc_news_raw.csv"
        with open(dataset_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Dataset downloaded successfully to {dataset_path}")
        return dataset_path
        
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        print("Please check your internet connection and try again.")
        print("The BBC News dataset is required for this analysis.")
        sys.exit(1)

def prepare_dataset(dataset_path):
    """Prepare and clean the dataset for analysis."""
    print("Preparing dataset for analysis...")
    
    # Load the dataset
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        sys.exit(1)
    
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} articles")
    
    # Basic data cleaning
    df = df.dropna(subset=['category', 'text'])
    df['text'] = df['text'].astype(str)
    df['category'] = df['category'].astype(str)
    
    # Convert category names to lowercase for consistency
    df['category'] = df['category'].str.lower().str.strip()
    
    # Filter out very short articles (less than 50 characters)
    df = df[df['text'].str.len() >= 50].copy()
    
    # Display basic statistics
    print(f"\nDataset Statistics:")
    print(f"Total articles after cleaning: {len(df)}")
    print(f"Categories: {sorted(df['category'].unique())}")
    
    category_counts = df['category'].value_counts()
    print(f"\nCategory distribution:")
    for category, count in category_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {category}: {count} articles ({percentage:.1f}%)")
    
    # Ensure we have at least 4 categories with sufficient articles
    min_articles = 50
    valid_categories = category_counts[category_counts >= min_articles].index.tolist()
    
    if len(valid_categories) < 4:
        print(f"Error: Need at least 4 categories with {min_articles}+ articles each.")
        print(f"Current valid categories: {valid_categories}")
        sys.exit(1)
    
    # Filter to valid categories only
    df_final = df[df['category'].isin(valid_categories)].copy()
    
    # Save processed dataset
    processed_path = Path("data/processed/newsbot_dataset.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(processed_path, index=False)
    
    # Save metadata
    metadata = {
        'total_articles': len(df_final),
        'categories': df_final['category'].value_counts().to_dict(),
        'average_article_length': df_final['text'].str.len().mean(),
        'dataset_source': str(dataset_path),
        'processing_date': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = Path("data/processed/dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset prepared successfully!")
    print(f"Processed dataset saved to: {processed_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Final dataset: {len(df_final)} articles across {len(valid_categories)} categories")
    
    return processed_path, metadata_path

def main():
    """Main data acquisition function."""
    print("NewsBot Intelligence System - Data Acquisition")
    print("=" * 50)
    
    # Download dataset
    dataset_path = download_bbc_dataset()
    
    # Prepare dataset
    processed_path, metadata_path = prepare_dataset(dataset_path)
    
    print("\n" + "=" * 50)
    print("Data acquisition completed successfully!")
    print(f"\nDataset ready for analysis:")
    print(f"Main dataset: {processed_path}")
    print(f"Metadata: {metadata_path}")
    print("\nNext step: Run the main NewsBot notebook for analysis.")

if __name__ == "__main__":
    main() 