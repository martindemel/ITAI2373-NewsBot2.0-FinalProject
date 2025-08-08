#!/usr/bin/env python3
"""
NewsBot 2.0 Intelligence System - Advanced Data Acquisition
Real data acquisition module using BBC News dataset
"""

import os
import pandas as pd
import requests
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

class DataAcquisition:
    """Advanced data acquisition system for NewsBot 2.0"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data acquisition system"""
        self.config = config or {}
        self.setup_logging()
        
        # Data directories
        self.raw_data_dir = Path(self.config.get('data.raw_data_dir', 'data/raw'))
        self.processed_data_dir = Path(self.config.get('data.processed_data_dir', 'data/processed'))
        self.models_dir = Path(self.config.get('data.models_dir', 'data/models'))
        
        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging for data acquisition"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DataAcquisition')
    
    def download_bbc_dataset(self) -> Path:
        """
        Download BBC News Classification dataset from reliable source.
        Uses the same dataset as the original midterm project.
        """
        self.logger.info("Downloading BBC News Classification dataset...")
        
        # BBC News dataset URL (publicly available version used in original project)
        dataset_url = "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv"
        
        try:
            response = requests.get(dataset_url, timeout=30)
            response.raise_for_status()
            
            # Save the dataset
            dataset_path = self.raw_data_dir / "bbc_news_raw.csv"
            with open(dataset_path, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"Dataset downloaded successfully to {dataset_path}")
            self.logger.info(f"Dataset size: {len(response.content)} bytes")
            return dataset_path
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download dataset: {e}")
            self.logger.error("Please check your internet connection and try again.")
            raise RuntimeError("BBC News dataset download failed")
        except Exception as e:
            self.logger.error(f"Unexpected error during download: {e}")
            raise
    
    def prepare_dataset(self, dataset_path: Path) -> Tuple[Path, Path]:
        """
        Prepare and clean the dataset for NewsBot 2.0 analysis.
        Enhanced version of the original preparation with additional features.
        """
        self.logger.info("Preparing dataset for NewsBot 2.0 analysis...")
        
        # Load the dataset
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        self.logger.info(f"Loaded dataset with {len(df)} articles")
        
        # Enhanced data cleaning for NewsBot 2.0
        initial_count = len(df)
        
        # Remove duplicates and null values
        df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
        df = df.dropna(subset=['category', 'text'])
        
        # Data type consistency
        df['text'] = df['text'].astype(str)
        df['category'] = df['category'].astype(str)
        
        # Enhanced text cleaning
        df['text'] = df['text'].str.strip()
        df['category'] = df['category'].str.lower().str.strip()
        
        # Filter quality articles (enhanced criteria)
        min_length = 50
        max_length = 10000  # Remove extremely long articles that might be corrupted
        df = df[
            (df['text'].str.len() >= min_length) & 
            (df['text'].str.len() <= max_length)
        ].copy()
        
        # Remove articles with mostly non-alphabetic characters
        df['alpha_ratio'] = df['text'].apply(lambda x: sum(c.isalpha() or c.isspace() for c in x) / len(x))
        df = df[df['alpha_ratio'] >= 0.7].copy()  # At least 70% alphabetic/space characters
        df.drop('alpha_ratio', axis=1, inplace=True)
        
        # Enhanced statistics
        cleaning_stats = {
            'initial_articles': initial_count,
            'after_deduplication': len(df),
            'articles_removed': initial_count - len(df),
            'removal_percentage': ((initial_count - len(df)) / initial_count) * 100
        }
        
        self.logger.info(f"Data cleaning results: {cleaning_stats}")
        
        # Category analysis
        category_counts = df['category'].value_counts()
        self.logger.info(f"Categories found: {sorted(df['category'].unique())}")
        
        print(f"\nCategory distribution:")
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {category}: {count} articles ({percentage:.1f}%)")
        
        # Ensure quality distribution (enhanced validation)
        min_articles = 50
        valid_categories = category_counts[category_counts >= min_articles].index.tolist()
        
        if len(valid_categories) < 4:
            raise ValueError(f"Need at least 4 categories with {min_articles}+ articles each. "
                           f"Current valid categories: {valid_categories}")
        
        # Filter to valid categories
        df_final = df[df['category'].isin(valid_categories)].copy()
        
        # Add enhanced metadata
        df_final['article_id'] = range(len(df_final))
        df_final['word_count'] = df_final['text'].str.split().str.len()
        df_final['char_count'] = df_final['text'].str.len()
        df_final['sentence_count'] = df_final['text'].str.count(r'[.!?]+') + 1
        
        # Save processed dataset
        processed_path = self.processed_data_dir / "newsbot_dataset.csv"
        df_final.to_csv(processed_path, index=False)
        
        # Enhanced metadata for NewsBot 2.0
        metadata = {
            'total_articles': len(df_final),
            'categories': df_final['category'].value_counts().to_dict(),
            'average_article_length': df_final['word_count'].mean(),
            'median_article_length': df_final['word_count'].median(),
            'dataset_source': str(dataset_path),
            'processing_date': datetime.now().isoformat(),
            'data_quality': {
                'min_word_count': df_final['word_count'].min(),
                'max_word_count': df_final['word_count'].max(),
                'avg_sentences': df_final['sentence_count'].mean(),
                'categories_count': len(valid_categories),
                'cleaning_applied': True
            },
            'newsbot_version': '2.0',
            'features_added': ['article_id', 'word_count', 'char_count', 'sentence_count']
        }
        
        metadata_path = self.processed_data_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save data quality report
        quality_report = {
            'cleaning_statistics': cleaning_stats,
            'final_statistics': {
                'total_articles': len(df_final),
                'categories': len(valid_categories),
                'avg_words_per_article': df_final['word_count'].mean(),
                'category_balance': {
                    'most_common': category_counts.index[0],
                    'least_common': category_counts.index[-1],
                    'balance_ratio': category_counts.min() / category_counts.max()
                }
            },
            'quality_metrics': {
                'deduplication_rate': (initial_count - len(df)) / initial_count,
                'category_distribution_cv': category_counts.std() / category_counts.mean(),
                'ready_for_ml': True
            }
        }
        
        quality_report_path = self.processed_data_dir / "data_quality_report.json"
        with open(quality_report_path, 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        self.logger.info(f"Dataset prepared successfully!")
        self.logger.info(f"Processed dataset: {processed_path} ({len(df_final)} articles)")
        self.logger.info(f"Metadata: {metadata_path}")
        self.logger.info(f"Quality report: {quality_report_path}")
        self.logger.info(f"Final dataset: {len(df_final)} articles across {len(valid_categories)} categories")
        
        return processed_path, metadata_path
    
    def verify_data_integrity(self, processed_path: Path, metadata_path: Path) -> Dict[str, Any]:
        """
        Verify the integrity of the processed dataset.
        Enhanced verification for NewsBot 2.0.
        """
        self.logger.info("Verifying data integrity...")
        
        verification_results = {
            'files_exist': False,
            'data_loadable': False,
            'metadata_consistent': False,
            'categories_valid': False,
            'ready_for_analysis': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check file existence
            if processed_path.exists() and metadata_path.exists():
                verification_results['files_exist'] = True
            else:
                verification_results['errors'].append("Required files missing")
                return verification_results
            
            # Try loading data
            df = pd.read_csv(processed_path)
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            verification_results['data_loadable'] = True
            
            # Verify metadata consistency
            if len(df) == metadata['total_articles']:
                verification_results['metadata_consistent'] = True
            else:
                verification_results['warnings'].append("Metadata count mismatch")
            
            # Verify categories
            expected_categories = set(metadata['categories'].keys())
            actual_categories = set(df['category'].unique())
            
            if expected_categories == actual_categories:
                verification_results['categories_valid'] = True
            else:
                verification_results['warnings'].append("Category mismatch")
            
            # Overall readiness check
            if (verification_results['files_exist'] and 
                verification_results['data_loadable'] and 
                verification_results['categories_valid'] and
                len(df) > 1000):  # Minimum viable dataset size
                
                verification_results['ready_for_analysis'] = True
            
            verification_results['dataset_info'] = {
                'articles': len(df),
                'categories': len(actual_categories),
                'average_length': df['word_count'].mean() if 'word_count' in df.columns else 'unknown'
            }
            
        except Exception as e:
            verification_results['errors'].append(f"Verification failed: {str(e)}")
        
        # Log results
        if verification_results['ready_for_analysis']:
            self.logger.info("✅ Data integrity verification passed")
        else:
            self.logger.warning("⚠️ Data integrity issues found")
            for error in verification_results['errors']:
                self.logger.error(f"Error: {error}")
            for warning in verification_results['warnings']:
                self.logger.warning(f"Warning: {warning}")
        
        return verification_results
    
    def acquire_and_prepare_data(self) -> Dict[str, Any]:
        """
        Complete data acquisition pipeline for NewsBot 2.0.
        Downloads, prepares, and verifies the BBC News dataset.
        """
        self.logger.info("Starting complete data acquisition pipeline...")
        
        results = {
            'status': 'started',
            'steps_completed': [],
            'files_created': [],
            'errors': []
        }
        
        try:
            # Step 1: Download dataset
            self.logger.info("Step 1: Downloading BBC News dataset...")
            dataset_path = self.download_bbc_dataset()
            results['steps_completed'].append('download')
            results['files_created'].append(str(dataset_path))
            
            # Step 2: Prepare dataset
            self.logger.info("Step 2: Preparing and cleaning dataset...")
            processed_path, metadata_path = self.prepare_dataset(dataset_path)
            results['steps_completed'].append('preparation')
            results['files_created'].extend([str(processed_path), str(metadata_path)])
            
            # Step 3: Verify integrity
            self.logger.info("Step 3: Verifying data integrity...")
            verification = self.verify_data_integrity(processed_path, metadata_path)
            results['steps_completed'].append('verification')
            results['verification'] = verification
            
            if verification['ready_for_analysis']:
                results['status'] = 'completed'
                self.logger.info("✅ Data acquisition pipeline completed successfully")
            else:
                results['status'] = 'completed_with_warnings'
                self.logger.warning("⚠️ Data acquisition completed but with warnings")
            
            results['final_paths'] = {
                'raw_data': str(dataset_path),
                'processed_data': str(processed_path),
                'metadata': str(metadata_path)
            }
            
        except Exception as e:
            results['status'] = 'failed'
            results['errors'].append(str(e))
            self.logger.error(f"Data acquisition pipeline failed: {e}")
            raise
        
        return results

def main():
    """Main function for standalone data acquisition"""
    print("NewsBot 2.0 Intelligence System - Data Acquisition")
    print("=" * 60)
    
    # Initialize data acquisition system
    data_acq = DataAcquisition()
    
    try:
        # Run complete pipeline
        results = data_acq.acquire_and_prepare_data()
        
        print("\n" + "=" * 60)
        print("Data Acquisition Results:")
        print(f"Status: {results['status']}")
        print(f"Steps completed: {', '.join(results['steps_completed'])}")
        print(f"Files created: {len(results['files_created'])}")
        
        if results['status'] == 'completed':
            print("\n✅ NewsBot 2.0 dataset ready for analysis!")
            print(f"Main dataset: {results['final_paths']['processed_data']}")
            print(f"Metadata: {results['final_paths']['metadata']}")
        
    except Exception as e:
        print(f"\n❌ Data acquisition failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()