#!/usr/bin/env python3
"""
Data Validator for NewsBot 2.0
Comprehensive data quality checks and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
import re
from datetime import datetime
from collections import Counter
import warnings

class DataValidator:
    """
    Comprehensive data validation for news article datasets
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data validator
        
        Args:
            config: Configuration dictionary with validation parameters
        """
        self.config = config or {}
        
        # Default validation thresholds
        self.min_article_length = self.config.get('min_article_length', 50)
        self.max_article_length = self.config.get('max_article_length', 50000)
        self.min_articles_per_category = self.config.get('min_articles_per_category', 10)
        self.max_missing_ratio = self.config.get('max_missing_ratio', 0.1)
        self.min_unique_ratio = self.config.get('min_unique_ratio', 0.8)
        
        # Validation results
        self.validation_results = {}
        self.issues_found = []
        self.warnings_found = []
    
    def validate_dataset(self, df: pd.DataFrame, text_column: str = 'text', 
                        category_column: str = 'category') -> Dict[str, Any]:
        """
        Perform comprehensive dataset validation
        
        Args:
            df: DataFrame to validate
            text_column: Name of the text column
            category_column: Name of the category column
            
        Returns:
            Dictionary with validation results
        """
        self.validation_results = {}
        self.issues_found = []
        self.warnings_found = []
        
        logging.info("Starting comprehensive dataset validation...")
        
        # Basic structure validation
        self._validate_structure(df, text_column, category_column)
        
        # Text content validation
        self._validate_text_content(df, text_column)
        
        # Category validation
        self._validate_categories(df, category_column)
        
        # Missing data validation
        self._validate_missing_data(df)
        
        # Duplicate validation
        self._validate_duplicates(df, text_column)
        
        # Distribution validation
        self._validate_distribution(df, category_column)
        
        # Quality metrics validation
        self._validate_quality_metrics(df, text_column)
        
        # Language validation
        self._validate_language_consistency(df, text_column)
        
        # Compile final results
        self.validation_results['total_issues'] = len(self.issues_found)
        self.validation_results['total_warnings'] = len(self.warnings_found)
        self.validation_results['issues'] = self.issues_found
        self.validation_results['warnings'] = self.warnings_found
        self.validation_results['is_valid'] = len(self.issues_found) == 0
        self.validation_results['validation_timestamp'] = datetime.now().isoformat()
        
        # Log summary
        if self.validation_results['is_valid']:
            logging.info("Dataset validation completed successfully!")
        else:
            logging.warning(f"Dataset validation found {len(self.issues_found)} issues and {len(self.warnings_found)} warnings")
        
        return self.validation_results
    
    def _validate_structure(self, df: pd.DataFrame, text_column: str, category_column: str):
        """Validate basic DataFrame structure"""
        logging.info("Validating dataset structure...")
        
        # Check if DataFrame is empty
        if df.empty:
            self.issues_found.append("Dataset is empty")
            return
        
        # Check required columns exist
        required_columns = [text_column, category_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.issues_found.append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if text_column in df.columns and not df[text_column].dtype == 'object':
            self.warnings_found.append(f"Text column '{text_column}' is not string type")
        
        if category_column in df.columns and not df[category_column].dtype == 'object':
            self.warnings_found.append(f"Category column '{category_column}' is not string type")
        
        # Store basic info
        self.validation_results['dataset_shape'] = df.shape
        self.validation_results['columns'] = list(df.columns)
        self.validation_results['data_types'] = df.dtypes.to_dict()
    
    def _validate_text_content(self, df: pd.DataFrame, text_column: str):
        """Validate text content quality"""
        logging.info("Validating text content...")
        
        if text_column not in df.columns:
            return
        
        text_series = df[text_column].dropna()
        
        # Check text lengths
        text_lengths = text_series.str.len()
        
        too_short = (text_lengths < self.min_article_length).sum()
        too_long = (text_lengths > self.max_article_length).sum()
        
        if too_short > 0:
            self.warnings_found.append(f"{too_short} articles are shorter than {self.min_article_length} characters")
        
        if too_long > 0:
            self.warnings_found.append(f"{too_long} articles are longer than {self.max_article_length} characters")
        
        # Check for empty or whitespace-only texts
        empty_texts = text_series.str.strip().str.len() == 0
        empty_count = empty_texts.sum()
        
        if empty_count > 0:
            self.issues_found.append(f"{empty_count} articles contain only whitespace or are empty")
        
        # Check for non-text content (excessive numbers, symbols)
        def check_text_quality(text):
            if pd.isna(text):
                return False
            
            # Check if text is mostly numbers
            digits_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
            if digits_ratio > 0.5:
                return False
            
            # Check if text is mostly symbols
            alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
            if alpha_ratio < 0.3:
                return False
            
            return True
        
        low_quality_texts = ~text_series.apply(check_text_quality)
        low_quality_count = low_quality_texts.sum()
        
        if low_quality_count > 0:
            self.warnings_found.append(f"{low_quality_count} articles may be low quality (excessive numbers/symbols)")
        
        # Store text statistics
        self.validation_results['text_stats'] = {
            'min_length': int(text_lengths.min()),
            'max_length': int(text_lengths.max()),
            'mean_length': float(text_lengths.mean()),
            'median_length': float(text_lengths.median()),
            'std_length': float(text_lengths.std()),
            'empty_count': int(empty_count),
            'too_short_count': int(too_short),
            'too_long_count': int(too_long),
            'low_quality_count': int(low_quality_count)
        }
    
    def _validate_categories(self, df: pd.DataFrame, category_column: str):
        """Validate category distribution and content"""
        logging.info("Validating categories...")
        
        if category_column not in df.columns:
            return
        
        categories = df[category_column].dropna()
        
        # Check category distribution
        category_counts = categories.value_counts()
        
        # Check minimum articles per category
        categories_below_min = category_counts[category_counts < self.min_articles_per_category]
        
        if len(categories_below_min) > 0:
            self.warnings_found.append(
                f"Categories with fewer than {self.min_articles_per_category} articles: "
                f"{categories_below_min.to_dict()}"
            )
        
        # Check for inconsistent category names (case, spacing)
        unique_categories = set(categories.unique())
        normalized_categories = set(cat.lower().strip() for cat in categories.unique())
        
        if len(unique_categories) != len(normalized_categories):
            self.warnings_found.append("Found categories that differ only in case or spacing")
        
        # Check for numeric categories
        numeric_categories = [cat for cat in categories.unique() if str(cat).isdigit()]
        if numeric_categories:
            self.warnings_found.append(f"Found numeric categories: {numeric_categories}")
        
        # Store category information
        self.validation_results['category_stats'] = {
            'unique_categories': list(categories.unique()),
            'category_counts': category_counts.to_dict(),
            'num_categories': len(categories.unique()),
            'most_common_category': category_counts.index[0],
            'least_common_category': category_counts.index[-1],
            'imbalance_ratio': float(category_counts.max() / category_counts.min())
        }
    
    def _validate_missing_data(self, df: pd.DataFrame):
        """Validate missing data patterns"""
        logging.info("Validating missing data...")
        
        missing_stats = {}
        
        for column in df.columns:
            missing_count = df[column].isna().sum()
            missing_ratio = missing_count / len(df)
            
            missing_stats[column] = {
                'missing_count': int(missing_count),
                'missing_ratio': float(missing_ratio)
            }
            
            if missing_ratio > self.max_missing_ratio:
                self.warnings_found.append(
                    f"Column '{column}' has {missing_ratio:.2%} missing values "
                    f"(threshold: {self.max_missing_ratio:.2%})"
                )
        
        self.validation_results['missing_data'] = missing_stats
    
    def _validate_duplicates(self, df: pd.DataFrame, text_column: str):
        """Validate duplicate content"""
        logging.info("Validating duplicates...")
        
        if text_column not in df.columns:
            return
        
        # Check exact duplicates
        exact_duplicates = df.duplicated(subset=[text_column]).sum()
        
        if exact_duplicates > 0:
            self.warnings_found.append(f"Found {exact_duplicates} exact duplicate articles")
        
        # Check near duplicates (first 100 characters)
        if not df[text_column].empty:
            text_prefixes = df[text_column].str[:100]
            near_duplicates = text_prefixes.duplicated().sum()
            
            if near_duplicates > exact_duplicates:
                potential_near_dups = near_duplicates - exact_duplicates
                if potential_near_dups > 0:
                    self.warnings_found.append(f"Found {potential_near_dups} potential near-duplicate articles")
        
        self.validation_results['duplicate_stats'] = {
            'exact_duplicates': int(exact_duplicates),
            'near_duplicates': int(near_duplicates) if not df[text_column].empty else 0
        }
    
    def _validate_distribution(self, df: pd.DataFrame, category_column: str):
        """Validate data distribution"""
        logging.info("Validating data distribution...")
        
        if category_column not in df.columns:
            return
        
        categories = df[category_column].dropna()
        category_counts = categories.value_counts()
        
        # Calculate distribution metrics
        mean_count = category_counts.mean()
        std_count = category_counts.std()
        cv = std_count / mean_count if mean_count > 0 else 0  # Coefficient of variation
        
        # Check for severe imbalance
        if cv > 1.0:
            self.warnings_found.append(
                f"Severe class imbalance detected (CV: {cv:.2f}). "
                "Consider balancing techniques."
            )
        elif cv > 0.5:
            self.warnings_found.append(
                f"Moderate class imbalance detected (CV: {cv:.2f})"
            )
        
        # Calculate Gini coefficient for imbalance
        sorted_counts = np.sort(category_counts.values)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        
        self.validation_results['distribution_stats'] = {
            'coefficient_variation': float(cv),
            'gini_coefficient': float(gini),
            'mean_articles_per_category': float(mean_count),
            'std_articles_per_category': float(std_count)
        }
    
    def _validate_quality_metrics(self, df: pd.DataFrame, text_column: str):
        """Validate text quality metrics"""
        logging.info("Validating quality metrics...")
        
        if text_column not in df.columns:
            return
        
        text_series = df[text_column].dropna()
        
        # Calculate readability and complexity metrics
        quality_metrics = {}
        
        # Word count distribution
        word_counts = text_series.str.split().str.len()
        quality_metrics['word_count'] = {
            'mean': float(word_counts.mean()),
            'std': float(word_counts.std()),
            'min': int(word_counts.min()),
            'max': int(word_counts.max()),
            'median': float(word_counts.median())
        }
        
        # Sentence count distribution
        sentence_counts = text_series.str.split('.').str.len()
        quality_metrics['sentence_count'] = {
            'mean': float(sentence_counts.mean()),
            'std': float(sentence_counts.std()),
            'min': int(sentence_counts.min()),
            'max': int(sentence_counts.max()),
            'median': float(sentence_counts.median())
        }
        
        # Check for articles with very few sentences
        few_sentences = (sentence_counts < 3).sum()
        if few_sentences > 0:
            self.warnings_found.append(f"{few_sentences} articles have fewer than 3 sentences")
        
        # Vocabulary diversity (unique words / total words)
        def calculate_diversity(text):
            if pd.isna(text):
                return 0
            words = text.lower().split()
            return len(set(words)) / max(len(words), 1)
        
        diversity_scores = text_series.apply(calculate_diversity)
        quality_metrics['vocabulary_diversity'] = {
            'mean': float(diversity_scores.mean()),
            'std': float(diversity_scores.std()),
            'min': float(diversity_scores.min()),
            'max': float(diversity_scores.max())
        }
        
        # Check for low diversity articles
        low_diversity = (diversity_scores < 0.3).sum()
        if low_diversity > 0:
            self.warnings_found.append(f"{low_diversity} articles have low vocabulary diversity (<0.3)")
        
        self.validation_results['quality_metrics'] = quality_metrics
    
    def _validate_language_consistency(self, df: pd.DataFrame, text_column: str):
        """Validate language consistency"""
        logging.info("Validating language consistency...")
        
        if text_column not in df.columns:
            return
        
        try:
            from langdetect import detect, DetectorFactory, LangDetectException
            DetectorFactory.seed = 0  # For reproducible results
            
            text_series = df[text_column].dropna()
            
            # Sample texts for language detection (to avoid processing all)
            sample_size = min(100, len(text_series))
            sample_texts = text_series.sample(n=sample_size, random_state=42)
            
            detected_languages = []
            
            for text in sample_texts:
                try:
                    if len(str(text).strip()) > 10:  # Only detect for non-trivial texts
                        lang = detect(str(text))
                        detected_languages.append(lang)
                except LangDetectException:
                    detected_languages.append('unknown')
            
            # Count language distribution
            lang_counter = Counter(detected_languages)
            
            # Check for mixed languages
            if len(lang_counter) > 1:
                self.warnings_found.append(
                    f"Multiple languages detected in sample: {dict(lang_counter)}"
                )
            
            # Check if primary language is English (expected for news dataset)
            primary_lang = lang_counter.most_common(1)[0][0] if lang_counter else 'unknown'
            
            if primary_lang != 'en':
                self.warnings_found.append(
                    f"Primary detected language is '{primary_lang}', expected 'en'"
                )
            
            self.validation_results['language_stats'] = {
                'detected_languages': dict(lang_counter),
                'primary_language': primary_lang,
                'sample_size': sample_size
            }
            
        except ImportError:
            self.warnings_found.append("langdetect not available for language validation")
            self.validation_results['language_stats'] = {'status': 'not_available'}
    
    def get_validation_report(self) -> str:
        """Generate a human-readable validation report"""
        if not self.validation_results:
            return "No validation results available. Run validate_dataset() first."
        
        report = []
        report.append("=" * 60)
        report.append("NEWSBOT 2.0 DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Validation Timestamp: {self.validation_results.get('validation_timestamp', 'N/A')}")
        report.append(f"Dataset Shape: {self.validation_results.get('dataset_shape', 'N/A')}")
        report.append("")
        
        # Summary
        report.append("VALIDATION SUMMARY")
        report.append("-" * 20)
        is_valid = self.validation_results.get('is_valid', False)
        status = "PASS" if is_valid else "FAIL"
        report.append(f"Overall Status: {status}")
        report.append(f"Issues Found: {self.validation_results.get('total_issues', 0)}")
        report.append(f"Warnings: {self.validation_results.get('total_warnings', 0)}")
        report.append("")
        
        # Issues
        if self.issues_found:
            report.append("CRITICAL ISSUES")
            report.append("-" * 15)
            for issue in self.issues_found:
                report.append(f"❌ {issue}")
            report.append("")
        
        # Warnings
        if self.warnings_found:
            report.append("WARNINGS")
            report.append("-" * 8)
            for warning in self.warnings_found:
                report.append(f"⚠️  {warning}")
            report.append("")
        
        # Statistics
        if 'text_stats' in self.validation_results:
            stats = self.validation_results['text_stats']
            report.append("TEXT STATISTICS")
            report.append("-" * 15)
            report.append(f"Average Length: {stats.get('mean_length', 0):.0f} characters")
            report.append(f"Length Range: {stats.get('min_length', 0)} - {stats.get('max_length', 0)}")
            report.append(f"Empty Articles: {stats.get('empty_count', 0)}")
            report.append("")
        
        if 'category_stats' in self.validation_results:
            stats = self.validation_results['category_stats']
            report.append("CATEGORY STATISTICS")
            report.append("-" * 18)
            report.append(f"Number of Categories: {stats.get('num_categories', 0)}")
            report.append(f"Imbalance Ratio: {stats.get('imbalance_ratio', 0):.2f}")
            report.append(f"Most Common: {stats.get('most_common_category', 'N/A')}")
            report.append(f"Least Common: {stats.get('least_common_category', 'N/A')}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 13)
        
        if self.issues_found:
            report.append("• Fix critical issues before proceeding with model training")
        
        if self.validation_results.get('distribution_stats', {}).get('coefficient_variation', 0) > 0.5:
            report.append("• Consider data balancing techniques for better model performance")
        
        if self.validation_results.get('duplicate_stats', {}).get('exact_duplicates', 0) > 0:
            report.append("• Remove duplicate articles to prevent data leakage")
        
        if not self.issues_found and not self.warnings_found:
            report.append("• Dataset appears to be of good quality and ready for analysis")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def fix_common_issues(self, df: pd.DataFrame, text_column: str = 'text', 
                         category_column: str = 'category') -> pd.DataFrame:
        """
        Automatically fix common data issues
        
        Args:
            df: DataFrame to fix
            text_column: Name of text column
            category_column: Name of category column
            
        Returns:
            Fixed DataFrame
        """
        logging.info("Attempting to fix common data issues...")
        
        df_fixed = df.copy()
        fixes_applied = []
        
        # Fix 1: Remove rows with missing text or category
        initial_count = len(df_fixed)
        df_fixed = df_fixed.dropna(subset=[text_column, category_column])
        
        if len(df_fixed) < initial_count:
            removed = initial_count - len(df_fixed)
            fixes_applied.append(f"Removed {removed} rows with missing text or category")
        
        # Fix 2: Remove duplicate articles
        initial_count = len(df_fixed)
        df_fixed = df_fixed.drop_duplicates(subset=[text_column])
        
        if len(df_fixed) < initial_count:
            removed = initial_count - len(df_fixed)
            fixes_applied.append(f"Removed {removed} duplicate articles")
        
        # Fix 3: Filter articles by length
        initial_count = len(df_fixed)
        text_lengths = df_fixed[text_column].str.len()
        df_fixed = df_fixed[
            (text_lengths >= self.min_article_length) & 
            (text_lengths <= self.max_article_length)
        ]
        
        if len(df_fixed) < initial_count:
            removed = initial_count - len(df_fixed)
            fixes_applied.append(f"Removed {removed} articles outside length limits")
        
        # Fix 4: Normalize category names
        df_fixed[category_column] = df_fixed[category_column].str.lower().str.strip()
        fixes_applied.append("Normalized category names to lowercase")
        
        # Fix 5: Remove categories with too few articles
        category_counts = df_fixed[category_column].value_counts()
        valid_categories = category_counts[category_counts >= self.min_articles_per_category].index
        
        initial_count = len(df_fixed)
        df_fixed = df_fixed[df_fixed[category_column].isin(valid_categories)]
        
        if len(df_fixed) < initial_count:
            removed = initial_count - len(df_fixed)
            fixes_applied.append(f"Removed {removed} articles from categories with too few samples")
        
        # Log fixes applied
        if fixes_applied:
            logging.info("Applied fixes:")
            for fix in fixes_applied:
                logging.info(f"  - {fix}")
        else:
            logging.info("No fixes were necessary")
        
        return df_fixed