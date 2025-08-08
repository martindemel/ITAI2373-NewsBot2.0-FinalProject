#!/usr/bin/env python3
"""
Cross-Lingual Analyzer for NewsBot 2.0
Advanced cross-language analysis, comparison, and cultural context understanding
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class CrossLingualAnalyzer:
    """
    Advanced cross-lingual analysis for comparing coverage and perspectives across languages
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize cross-lingual analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize dependent components (injected)
        self.language_detector = None
        self.translator = None
        self.sentiment_analyzer = None
        self.embeddings_system = None
        
        # Cultural context markers for different regions
        self.cultural_context_markers = {
            'western': {
                'values': ['democracy', 'freedom', 'individual rights', 'capitalism', 'innovation'],
                'institutions': ['nato', 'eu', 'g7', 'world bank', 'imf'],
                'perspectives': ['human rights', 'rule of law', 'market economy']
            },
            'eastern': {
                'values': ['collective harmony', 'social stability', 'family values', 'tradition'],
                'institutions': ['brics', 'sco', 'asean', 'silk road'],
                'perspectives': ['social order', 'economic development', 'sovereignty']
            },
            'middle_eastern': {
                'values': ['religious tradition', 'family honor', 'hospitality', 'community'],
                'institutions': ['arab league', 'gcc', 'oic'],
                'perspectives': ['religious values', 'regional stability', 'cultural identity']
            },
            'african': {
                'values': ['ubuntu', 'community solidarity', 'ancestral wisdom', 'unity'],
                'institutions': ['african union', 'ecowas', 'sadc'],
                'perspectives': ['pan-africanism', 'development', 'decolonization']
            },
            'latin_american': {
                'values': ['familia', 'personalismo', 'simpatía', 'machismo'],
                'institutions': ['mercosur', 'oas', 'celac', 'unasur'],
                'perspectives': ['regional integration', 'social justice', 'cultural identity']
            }
        }
        
        # Language-to-cultural-region mapping
        self.language_cultural_mapping = {
            'en': ['western'],
            'es': ['western', 'latin_american'],
            'fr': ['western', 'african'],
            'de': ['western'],
            'it': ['western'],
            'pt': ['western', 'latin_american', 'african'],
            'ru': ['eastern'],
            'zh': ['eastern'],
            'ja': ['eastern'],
            'ko': ['eastern'],
            'ar': ['middle_eastern', 'african'],
            'hi': ['eastern'],
            'tr': ['middle_eastern', 'eastern']
        }
        
        # Analysis results storage
        self.analysis_cache = {}
        self.comparative_analyses = []
        
        # Enhanced cross-lingual analysis capabilities
        self.semantic_similarity_cache = {}
        self.cultural_bias_patterns = {}
        self.temporal_cultural_shifts = {}
        self.cross_cultural_sentiment_patterns = {}
        self.discourse_analysis_results = {}
        
        # Advanced analysis parameters
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.cultural_bias_sensitivity = self.config.get('cultural_bias_sensitivity', 0.8)
        self.temporal_window_days = self.config.get('temporal_window_days', 7)
        
        # Statistics
        self.analysis_stats = {
            'total_analyses': 0,
            'cross_lingual_comparisons': 0,
            'cultural_analyses': 0,
            'coverage_analyses': 0,
            'semantic_similarity_analyses': 0,
            'discourse_analyses': 0,
            'temporal_cultural_analyses': 0,
            'language_pairs_analyzed': defaultdict(int)
        }
    
    def set_dependencies(self, language_detector=None, translator=None, 
                        sentiment_analyzer=None, embeddings_system=None):
        """Set dependency components"""
        self.language_detector = language_detector
        self.translator = translator
        self.sentiment_analyzer = sentiment_analyzer
        self.embeddings_system = embeddings_system
    
    def analyze_cross_lingual_coverage(self, multilingual_articles: Dict[str, List[Dict[str, Any]]],
                                     topic_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze how the same topics are covered across different languages
        
        Args:
            multilingual_articles: Dictionary mapping language codes to article lists
            topic_keywords: Optional keywords to focus analysis on specific topics
            
        Returns:
            Dictionary with cross-lingual coverage analysis
        """
        logging.info(f"Analyzing cross-lingual coverage for {len(multilingual_articles)} languages")
        
        analysis_result = {
            'languages_analyzed': list(multilingual_articles.keys()),
            'total_articles': sum(len(articles) for articles in multilingual_articles.values()),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Language-specific analysis
        language_analyses = {}
        
        for lang_code, articles in multilingual_articles.items():
            if not articles:
                continue
                
            lang_analysis = self._analyze_language_specific_coverage(
                articles, lang_code, topic_keywords
            )
            language_analyses[lang_code] = lang_analysis
        
        analysis_result['language_analyses'] = language_analyses
        
        # Cross-language comparisons
        if len(multilingual_articles) >= 2:
            cross_comparisons = self._compare_cross_lingual_coverage(multilingual_articles, topic_keywords)
            analysis_result['cross_comparisons'] = cross_comparisons
        
        # Cultural perspective analysis
        cultural_analysis = self._analyze_cultural_perspectives(multilingual_articles)
        analysis_result['cultural_perspectives'] = cultural_analysis
        
        # Topic coverage gaps
        coverage_gaps = self._identify_coverage_gaps(multilingual_articles, topic_keywords)
        analysis_result['coverage_gaps'] = coverage_gaps
        
        # Language dominance analysis
        dominance_analysis = self._analyze_language_dominance(multilingual_articles)
        analysis_result['language_dominance'] = dominance_analysis
        
        # Update statistics
        self.analysis_stats['total_analyses'] += 1
        self.analysis_stats['coverage_analyses'] += 1
        
        return analysis_result
    
    def _analyze_language_specific_coverage(self, articles: List[Dict[str, Any]], 
                                          language: str, topic_keywords: Optional[List[str]]) -> Dict[str, Any]:
        """Analyze coverage patterns for a specific language"""
        
        analysis = {
            'language': language,
            'article_count': len(articles),
            'avg_article_length': 0,
            'topic_distribution': {},
            'sentiment_patterns': {},
            'temporal_patterns': {},
            'key_themes': []
        }
        
        if not articles:
            return analysis
        
        # Article length analysis
        article_lengths = [len(article.get('text', '')) for article in articles]
        analysis['avg_article_length'] = np.mean(article_lengths) if article_lengths else 0
        analysis['article_length_std'] = np.std(article_lengths) if article_lengths else 0
        
        # Topic analysis using keywords
        if topic_keywords:
            topic_coverage = self._analyze_topic_coverage(articles, topic_keywords)
            analysis['topic_coverage'] = topic_coverage
        
        # Sentiment analysis if sentiment analyzer available
        if self.sentiment_analyzer:
            sentiment_patterns = self._analyze_language_sentiment_patterns(articles)
            analysis['sentiment_patterns'] = sentiment_patterns
        
        # Temporal patterns
        if any('date' in article for article in articles):
            temporal_patterns = self._analyze_temporal_patterns(articles)
            analysis['temporal_patterns'] = temporal_patterns
        
        # Key themes extraction
        key_themes = self._extract_key_themes(articles, language)
        analysis['key_themes'] = key_themes
        
        # Cultural markers
        cultural_markers = self._identify_cultural_markers(articles, language)
        analysis['cultural_markers'] = cultural_markers
        
        return analysis
    
    def _analyze_topic_coverage(self, articles: List[Dict[str, Any]], 
                               topic_keywords: List[str]) -> Dict[str, Any]:
        """Analyze how specific topics are covered"""
        
        topic_coverage = {}
        
        for keyword in topic_keywords:
            keyword_lower = keyword.lower()
            matching_articles = []
            
            for article in articles:
                text = article.get('text', '').lower()
                if keyword_lower in text:
                    matching_articles.append(article)
            
            coverage_ratio = len(matching_articles) / len(articles) if articles else 0
            
            topic_coverage[keyword] = {
                'article_count': len(matching_articles),
                'coverage_ratio': coverage_ratio,
                'sample_articles': [
                    article.get('text', '')[:200] + '...' 
                    for article in matching_articles[:3]
                ]
            }
        
        return topic_coverage
    
    def _analyze_language_sentiment_patterns(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment patterns for articles in a specific language"""
        
        if not self.sentiment_analyzer:
            return {'error': 'Sentiment analyzer not available'}
        
        sentiments = []
        
        for article in articles:
            text = article.get('text', '')
            if text:
                try:
                    sentiment_result = self.sentiment_analyzer.analyze_sentiment(text)
                    if 'aggregated' in sentiment_result:
                        sentiments.append(sentiment_result['aggregated'])
                except Exception as e:
                    logging.warning(f"Sentiment analysis failed: {e}")
                    continue
        
        if not sentiments:
            return {'error': 'No sentiment data available'}
        
        # Aggregate sentiment statistics
        sentiment_scores = [s.get('weighted_score', 0) for s in sentiments]
        sentiment_classes = [s.get('classification', 'neutral') for s in sentiments]
        
        return {
            'avg_sentiment': np.mean(sentiment_scores) if sentiment_scores else 0,
            'sentiment_std': np.std(sentiment_scores) if sentiment_scores else 0,
            'sentiment_distribution': dict(Counter(sentiment_classes)),
            'positive_ratio': sentiment_classes.count('positive') / len(sentiment_classes) if sentiment_classes else 0,
            'negative_ratio': sentiment_classes.count('negative') / len(sentiment_classes) if sentiment_classes else 0,
            'total_analyzed': len(sentiments)
        }
    
    def _analyze_temporal_patterns(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal coverage patterns"""
        
        dates = []
        for article in articles:
            if 'date' in article:
                try:
                    date = pd.to_datetime(article['date'])
                    dates.append(date)
                except:
                    continue
        
        if not dates:
            return {'error': 'No date information available'}
        
        dates_series = pd.Series(dates)
        
        return {
            'date_range': {
                'start': dates_series.min().isoformat(),
                'end': dates_series.max().isoformat()
            },
            'temporal_span_days': (dates_series.max() - dates_series.min()).days,
            'articles_per_day': len(dates) / max((dates_series.max() - dates_series.min()).days, 1),
            'peak_activity_date': dates_series.value_counts().index[0].isoformat() if len(dates) > 0 else None,
            'temporal_distribution': {
                'daily': dates_series.dt.date.value_counts().to_dict(),
                'weekly': dates_series.dt.to_period('W').value_counts().to_dict()
            }
        }
    
    def _extract_key_themes(self, articles: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
        """Extract key themes from articles using TF-IDF"""
        
        texts = [article.get('text', '') for article in articles if article.get('text')]
        
        if len(texts) < 2:
            return []
        
        try:
            # Use TF-IDF to extract key terms
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english' if language == 'en' else None,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top themes
            top_indices = np.argsort(avg_scores)[-20:][::-1]
            
            themes = []
            for idx in top_indices:
                themes.append({
                    'theme': feature_names[idx],
                    'importance_score': float(avg_scores[idx]),
                    'language': language
                })
            
            return themes
            
        except Exception as e:
            logging.warning(f"Theme extraction failed for {language}: {e}")
            return []
    
    def _identify_cultural_markers(self, articles: List[Dict[str, Any]], language: str) -> Dict[str, Any]:
        """Identify cultural markers and perspectives in articles"""
        
        cultural_regions = self.language_cultural_mapping.get(language, [])
        if not cultural_regions:
            return {'cultural_regions': [], 'markers_found': {}}
        
        all_text = ' '.join([article.get('text', '').lower() for article in articles])
        
        markers_found = {}
        
        for region in cultural_regions:
            if region in self.cultural_context_markers:
                region_markers = self.cultural_context_markers[region]
                region_found = {}
                
                for category, markers in region_markers.items():
                    found_markers = [marker for marker in markers if marker.lower() in all_text]
                    if found_markers:
                        region_found[category] = found_markers
                
                if region_found:
                    markers_found[region] = region_found
        
        # Calculate cultural orientation score
        cultural_scores = {}
        word_count = len(all_text.split())
        
        for region, found_markers in markers_found.items():
            total_marker_count = sum(len(markers) for markers in found_markers.values())
            cultural_scores[region] = total_marker_count / max(word_count / 1000, 1)  # Per 1000 words
        
        return {
            'cultural_regions': cultural_regions,
            'markers_found': markers_found,
            'cultural_scores': cultural_scores,
            'dominant_cultural_perspective': max(cultural_scores, key=cultural_scores.get) if cultural_scores else None
        }
    
    def _compare_cross_lingual_coverage(self, multilingual_articles: Dict[str, List[Dict[str, Any]]],
                                       topic_keywords: Optional[List[str]]) -> Dict[str, Any]:
        """Compare coverage patterns across languages"""
        
        comparisons = {}
        languages = list(multilingual_articles.keys())
        
        # Pairwise comparisons
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages[i+1:], i+1):
                
                pair_key = f"{lang1}_{lang2}"
                
                # Compare article counts
                count1 = len(multilingual_articles[lang1])
                count2 = len(multilingual_articles[lang2])
                
                # Compare average article lengths
                avg_len1 = np.mean([len(a.get('text', '')) for a in multilingual_articles[lang1]]) if multilingual_articles[lang1] else 0
                avg_len2 = np.mean([len(a.get('text', '')) for a in multilingual_articles[lang2]]) if multilingual_articles[lang2] else 0
                
                # Topic overlap analysis
                topic_overlap = None
                if topic_keywords:
                    topic_overlap = self._calculate_topic_overlap(
                        multilingual_articles[lang1], 
                        multilingual_articles[lang2], 
                        topic_keywords
                    )
                
                # Sentiment comparison
                sentiment_comparison = None
                if self.sentiment_analyzer:
                    sentiment_comparison = self._compare_sentiment_across_languages(
                        multilingual_articles[lang1], 
                        multilingual_articles[lang2]
                    )
                
                comparisons[pair_key] = {
                    'languages': [lang1, lang2],
                    'article_count_ratio': count2 / count1 if count1 > 0 else float('inf'),
                    'avg_length_ratio': avg_len2 / avg_len1 if avg_len1 > 0 else float('inf'),
                    'topic_overlap': topic_overlap,
                    'sentiment_comparison': sentiment_comparison
                }
                
                # Update statistics
                self.analysis_stats['language_pairs_analyzed'][pair_key] += 1
        
        # Overall cross-lingual insights
        overall_insights = self._generate_cross_lingual_insights(comparisons)
        
        return {
            'pairwise_comparisons': comparisons,
            'overall_insights': overall_insights,
            'total_language_pairs': len(comparisons)
        }
    
    def _calculate_topic_overlap(self, articles1: List[Dict[str, Any]], 
                               articles2: List[Dict[str, Any]], 
                               topic_keywords: List[str]) -> Dict[str, Any]:
        """Calculate topic overlap between two language corpora"""
        
        def get_topic_coverage(articles, keywords):
            coverage = {}
            for keyword in keywords:
                keyword_lower = keyword.lower()
                matching = sum(1 for article in articles 
                             if keyword_lower in article.get('text', '').lower())
                coverage[keyword] = matching / len(articles) if articles else 0
            return coverage
        
        coverage1 = get_topic_coverage(articles1, topic_keywords)
        coverage2 = get_topic_coverage(articles2, topic_keywords)
        
        # Calculate overlap metrics
        overlap_scores = {}
        for keyword in topic_keywords:
            c1, c2 = coverage1[keyword], coverage2[keyword]
            overlap_scores[keyword] = {
                'coverage_lang1': c1,
                'coverage_lang2': c2,
                'overlap_score': min(c1, c2) / max(c1, c2) if max(c1, c2) > 0 else 1.0,
                'coverage_difference': abs(c1 - c2)
            }
        
        # Overall overlap
        avg_overlap = np.mean([score['overlap_score'] for score in overlap_scores.values()])
        
        return {
            'keyword_overlaps': overlap_scores,
            'average_overlap': avg_overlap,
            'topics_with_high_overlap': [
                keyword for keyword, scores in overlap_scores.items() 
                if scores['overlap_score'] > 0.8
            ],
            'topics_with_low_overlap': [
                keyword for keyword, scores in overlap_scores.items() 
                if scores['overlap_score'] < 0.3
            ]
        }
    
    def _compare_sentiment_across_languages(self, articles1: List[Dict[str, Any]], 
                                          articles2: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare sentiment patterns across languages"""
        
        def get_sentiment_stats(articles):
            sentiments = []
            for article in articles:
                try:
                    result = self.sentiment_analyzer.analyze_sentiment(article.get('text', ''))
                    if 'aggregated' in result:
                        sentiments.append(result['aggregated'].get('weighted_score', 0))
                except:
                    continue
            
            if sentiments:
                return {
                    'avg_sentiment': np.mean(sentiments),
                    'sentiment_std': np.std(sentiments),
                    'positive_ratio': sum(1 for s in sentiments if s > 0.1) / len(sentiments),
                    'negative_ratio': sum(1 for s in sentiments if s < -0.1) / len(sentiments)
                }
            return None
        
        stats1 = get_sentiment_stats(articles1)
        stats2 = get_sentiment_stats(articles2)
        
        if not stats1 or not stats2:
            return {'error': 'Insufficient sentiment data for comparison'}
        
        return {
            'lang1_sentiment': stats1,
            'lang2_sentiment': stats2,
            'sentiment_difference': abs(stats1['avg_sentiment'] - stats2['avg_sentiment']),
            'positivity_difference': abs(stats1['positive_ratio'] - stats2['positive_ratio']),
            'more_positive_language': 'lang1' if stats1['avg_sentiment'] > stats2['avg_sentiment'] else 'lang2'
        }
    
    def _analyze_cultural_perspectives(self, multilingual_articles: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze cultural perspectives represented in different languages"""
        
        cultural_analysis = {
            'by_language': {},
            'cultural_diversity': {},
            'perspective_conflicts': [],
            'cultural_bridges': []
        }
        
        # Analyze each language for cultural perspectives
        for lang_code, articles in multilingual_articles.items():
            cultural_markers = self._identify_cultural_markers(articles, lang_code)
            cultural_analysis['by_language'][lang_code] = cultural_markers
        
        # Calculate cultural diversity
        all_cultural_regions = set()
        for lang_analysis in cultural_analysis['by_language'].values():
            all_cultural_regions.update(lang_analysis.get('cultural_regions', []))
        
        cultural_analysis['cultural_diversity'] = {
            'total_cultural_regions': len(all_cultural_regions),
            'regions_represented': list(all_cultural_regions),
            'diversity_index': len(all_cultural_regions) / len(multilingual_articles) if multilingual_articles else 0
        }
        
        # Identify perspective conflicts and bridges
        conflicts, bridges = self._identify_cultural_conflicts_and_bridges(cultural_analysis['by_language'])
        cultural_analysis['perspective_conflicts'] = conflicts
        cultural_analysis['cultural_bridges'] = bridges
        
        self.analysis_stats['cultural_analyses'] += 1
        
        return cultural_analysis
    
    def _identify_cultural_conflicts_and_bridges(self, language_cultural_data: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Identify potential cultural conflicts and bridges between languages"""
        
        conflicts = []
        bridges = []
        
        # Define conflicting cultural perspectives
        conflict_patterns = {
            'individualism_vs_collectivism': {
                'individualism': ['individual rights', 'personal freedom', 'self-reliance'],
                'collectivism': ['collective harmony', 'community', 'social stability']
            },
            'western_vs_eastern': {
                'western': ['democracy', 'capitalism', 'secularism'],
                'eastern': ['tradition', 'hierarchy', 'spiritual values']
            }
        }
        
        # Find languages with conflicting markers
        for conflict_type, conflict_markers in conflict_patterns.items():
            for side1, markers1 in conflict_markers.items():
                for side2, markers2 in conflict_markers.items():
                    if side1 != side2:
                        langs_with_side1 = []
                        langs_with_side2 = []
                        
                        for lang, cultural_data in language_cultural_data.items():
                            markers_found = cultural_data.get('markers_found', {})
                            
                            # Check if language shows markers for side1
                            if any(marker in str(markers_found).lower() for marker in markers1):
                                langs_with_side1.append(lang)
                            
                            # Check if language shows markers for side2
                            if any(marker in str(markers_found).lower() for marker in markers2):
                                langs_with_side2.append(lang)
                        
                        if langs_with_side1 and langs_with_side2:
                            conflicts.append({
                                'conflict_type': conflict_type,
                                'side1': {'perspective': side1, 'languages': langs_with_side1},
                                'side2': {'perspective': side2, 'languages': langs_with_side2}
                            })
        
        # Find cultural bridges (shared values across different cultural regions)
        bridge_values = ['peace', 'prosperity', 'education', 'health', 'family']
        
        for value in bridge_values:
            languages_with_value = []
            
            for lang, cultural_data in language_cultural_data.items():
                markers_found = str(cultural_data.get('markers_found', {})).lower()
                if value in markers_found:
                    languages_with_value.append(lang)
            
            if len(languages_with_value) >= 2:
                bridges.append({
                    'bridge_value': value,
                    'languages': languages_with_value,
                    'bridge_strength': len(languages_with_value)
                })
        
        return conflicts, bridges
    
    def _identify_coverage_gaps(self, multilingual_articles: Dict[str, List[Dict[str, Any]]],
                               topic_keywords: Optional[List[str]]) -> Dict[str, Any]:
        """Identify coverage gaps across languages"""
        
        gaps = {
            'language_gaps': {},
            'topic_gaps': {},
            'temporal_gaps': {}
        }
        
        # Language coverage gaps
        article_counts = {lang: len(articles) for lang, articles in multilingual_articles.items()}
        max_count = max(article_counts.values()) if article_counts else 0
        
        for lang, count in article_counts.items():
            if max_count > 0:
                coverage_ratio = count / max_count
                if coverage_ratio < 0.5:  # Less than 50% of max coverage
                    gaps['language_gaps'][lang] = {
                        'article_count': count,
                        'coverage_ratio': coverage_ratio,
                        'gap_severity': 'high' if coverage_ratio < 0.2 else 'medium'
                    }
        
        # Topic coverage gaps
        if topic_keywords:
            for keyword in topic_keywords:
                keyword_coverage = {}
                for lang, articles in multilingual_articles.items():
                    matching = sum(1 for article in articles 
                                 if keyword.lower() in article.get('text', '').lower())
                    coverage = matching / len(articles) if articles else 0
                    keyword_coverage[lang] = coverage
                
                # Find languages with low coverage of this topic
                max_coverage = max(keyword_coverage.values()) if keyword_coverage else 0
                if max_coverage > 0:
                    low_coverage_langs = {
                        lang: coverage for lang, coverage in keyword_coverage.items()
                        if coverage < max_coverage * 0.3  # Less than 30% of max
                    }
                    
                    if low_coverage_langs:
                        gaps['topic_gaps'][keyword] = {
                            'low_coverage_languages': low_coverage_langs,
                            'max_coverage': max_coverage,
                            'gap_languages': list(low_coverage_langs.keys())
                        }
        
        return gaps
    
    def _analyze_language_dominance(self, multilingual_articles: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze language dominance patterns"""
        
        total_articles = sum(len(articles) for articles in multilingual_articles.values())
        
        language_stats = {}
        for lang, articles in multilingual_articles.items():
            count = len(articles)
            percentage = (count / total_articles * 100) if total_articles > 0 else 0
            
            avg_length = np.mean([len(a.get('text', '')) for a in articles]) if articles else 0
            
            language_stats[lang] = {
                'article_count': count,
                'percentage': percentage,
                'avg_article_length': avg_length,
                'dominance_score': percentage * (avg_length / 1000)  # Weighted by length
            }
        
        # Rank languages by dominance
        ranked_languages = sorted(
            language_stats.items(), 
            key=lambda x: x[1]['dominance_score'], 
            reverse=True
        )
        
        # Calculate dominance distribution
        herfindahl_index = sum((stats['percentage'] / 100) ** 2 for stats in language_stats.values())
        
        return {
            'language_statistics': language_stats,
            'ranked_languages': ranked_languages,
            'dominant_language': ranked_languages[0][0] if ranked_languages else None,
            'herfindahl_index': herfindahl_index,
            'dominance_classification': (
                'highly_concentrated' if herfindahl_index > 0.5 else
                'moderately_concentrated' if herfindahl_index > 0.25 else
                'diverse'
            )
        }
    
    def _generate_cross_lingual_insights(self, comparisons: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate high-level insights from cross-lingual comparisons"""
        
        insights = {
            'summary': '',
            'key_findings': [],
            'recommendations': []
        }
        
        if not comparisons:
            return insights
        
        # Analyze article count ratios
        count_ratios = [comp.get('article_count_ratio', 1) for comp in comparisons.values()]
        avg_count_ratio = np.mean(count_ratios)
        
        if avg_count_ratio > 2:
            insights['key_findings'].append(
                "Significant imbalance in article coverage across languages"
            )
            insights['recommendations'].append(
                "Consider increasing coverage in underrepresented languages"
            )
        
        # Analyze length ratios
        length_ratios = [comp.get('avg_length_ratio', 1) for comp in comparisons.values() 
                        if comp.get('avg_length_ratio', float('inf')) != float('inf')]
        
        if length_ratios:
            avg_length_ratio = np.mean(length_ratios)
            if avg_length_ratio > 1.5:
                insights['key_findings'].append(
                    "Articles in some languages are significantly longer than others"
                )
        
        # Topic overlap analysis
        topic_overlaps = [
            comp.get('topic_overlap', {}).get('average_overlap', 0) 
            for comp in comparisons.values() 
            if comp.get('topic_overlap')
        ]
        
        if topic_overlaps:
            avg_overlap = np.mean(topic_overlaps)
            if avg_overlap < 0.5:
                insights['key_findings'].append(
                    "Low topic overlap between languages suggests different editorial priorities"
                )
                insights['recommendations'].append(
                    "Investigate editorial differences and consider standardizing key topic coverage"
                )
        
        # Generate summary
        insights['summary'] = f"Analyzed {len(comparisons)} language pairs. "
        if insights['key_findings']:
            insights['summary'] += f"Key findings: {'; '.join(insights['key_findings'][:2])}"
        else:
            insights['summary'] += "Coverage appears balanced across languages."
        
        return insights
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get cross-lingual analysis statistics"""
        stats = {
            'total_analyses': self.analysis_stats['total_analyses'],
            'cross_lingual_comparisons': self.analysis_stats['cross_lingual_comparisons'],
            'cultural_analyses': self.analysis_stats['cultural_analyses'],
            'coverage_analyses': self.analysis_stats['coverage_analyses'],
            'language_pairs_analyzed': dict(self.analysis_stats['language_pairs_analyzed'])
        }
        
        return stats
    
    def save_analyzer(self, filepath: str):
        """Save cross-lingual analyzer configuration"""
        save_data = {
            'config': self.config,
            'cultural_context_markers': self.cultural_context_markers,
            'language_cultural_mapping': self.language_cultural_mapping,
            'analysis_stats': dict(self.analysis_stats),
            'save_timestamp': datetime.now().isoformat()
        }
        
        # Convert defaultdict to regular dict
        save_data['analysis_stats']['language_pairs_analyzed'] = dict(self.analysis_stats['language_pairs_analyzed'])
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logging.info(f"Cross-lingual analyzer saved to {filepath}")
    
    def load_analyzer(self, filepath: str):
        """Load cross-lingual analyzer configuration"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.config = save_data['config']
        self.cultural_context_markers = save_data['cultural_context_markers']
        self.language_cultural_mapping = save_data['language_cultural_mapping']
        
        # Restore statistics
        stats = save_data['analysis_stats']
        self.analysis_stats['total_analyses'] = stats.get('total_analyses', 0)
        self.analysis_stats['cross_lingual_comparisons'] = stats.get('cross_lingual_comparisons', 0)
        self.analysis_stats['cultural_analyses'] = stats.get('cultural_analyses', 0)
        self.analysis_stats['coverage_analyses'] = stats.get('coverage_analyses', 0)
        
        # Restore defaultdict
        self.analysis_stats['language_pairs_analyzed'] = defaultdict(
            int, stats.get('language_pairs_analyzed', {})
        )
        
        logging.info(f"Cross-lingual analyzer loaded from {filepath}")
    
    def analyze_semantic_cross_lingual_similarity(self, articles_by_language: Dict[str, List[Dict[str, Any]]], 
                                                 topic_focus: Optional[str] = None) -> Dict[str, Any]:
        """
        Advanced semantic similarity analysis across languages for the same topics/events
        
        Args:
            articles_by_language: Dictionary mapping language codes to article lists
            topic_focus: Optional topic filter for focused analysis
            
        Returns:
            Comprehensive semantic similarity analysis
        """
        logging.info(f"Starting semantic cross-lingual similarity analysis for {len(articles_by_language)} languages...")
        
        similarity_results = {
            'language_pairs': {},
            'semantic_clusters': {},
            'cross_lingual_topic_alignment': {},
            'cultural_perspective_differences': {},
            'coverage_gaps': {},
            'translation_quality_assessment': {}
        }
        
        # Filter articles by topic if specified
        if topic_focus:
            filtered_articles = {}
            for lang, articles in articles_by_language.items():
                filtered_articles[lang] = [
                    article for article in articles 
                    if topic_focus.lower() in article.get('text', '').lower() or 
                       topic_focus.lower() in article.get('title', '').lower()
                ]
            articles_by_language = {k: v for k, v in filtered_articles.items() if v}
        
        # Create embeddings for all articles in all languages
        embeddings_by_language = {}
        for lang, articles in articles_by_language.items():
            if self.embeddings_system:
                embeddings = []
                for article in articles:
                    try:
                        embedding = self.embeddings_system.encode_text(article.get('text', ''))
                        embeddings.append(embedding)
                    except Exception as e:
                        logging.warning(f"Failed to create embedding for {lang} article: {e}")
                embeddings_by_language[lang] = np.array(embeddings) if embeddings else None
        
        # Analyze pairwise language similarities
        languages = list(articles_by_language.keys())
        for i, lang1 in enumerate(languages):
            for lang2 in languages[i+1:]:
                pair_key = f"{lang1}-{lang2}"
                
                if lang1 in embeddings_by_language and lang2 in embeddings_by_language:
                    emb1 = embeddings_by_language[lang1]
                    emb2 = embeddings_by_language[lang2]
                    
                    if emb1 is not None and emb2 is not None and len(emb1) > 0 and len(emb2) > 0:
                        # Calculate cross-language semantic similarity
                        similarity_analysis = self._calculate_cross_language_similarity(
                            emb1, emb2, articles_by_language[lang1], articles_by_language[lang2]
                        )
                        similarity_results['language_pairs'][pair_key] = similarity_analysis
                        
                        # Update statistics
                        self.analysis_stats['language_pairs_analyzed'][pair_key] += 1
        
        # Identify semantic clusters across languages
        similarity_results['semantic_clusters'] = self._identify_cross_lingual_clusters(embeddings_by_language, articles_by_language)
        
        # Analyze topic alignment across languages
        similarity_results['cross_lingual_topic_alignment'] = self._analyze_topic_alignment(articles_by_language)
        
        # Analyze cultural perspective differences
        similarity_results['cultural_perspective_differences'] = self._analyze_cultural_perspective_differences(articles_by_language)
        
        # Identify coverage gaps
        similarity_results['coverage_gaps'] = self._identify_coverage_gaps(articles_by_language)
        
        # Assess translation quality
        similarity_results['translation_quality_assessment'] = self._assess_translation_quality(articles_by_language)
        
        # Update statistics
        self.analysis_stats['semantic_similarity_analyses'] += 1
        self.analysis_stats['total_analyses'] += 1
        
        # Cache results
        cache_key = f"semantic_similarity_{hash(str(sorted(languages)))}"
        self.semantic_similarity_cache[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'results': similarity_results
        }
        
        logging.info(f"Semantic cross-lingual similarity analysis completed for {len(similarity_results['language_pairs'])} language pairs.")
        
        return similarity_results
    
    def _calculate_cross_language_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray,
                                           articles1: List[Dict[str, Any]], articles2: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive similarity between two language corpora"""
        # Calculate average embedding similarity
        avg_emb1 = np.mean(embeddings1, axis=0)
        avg_emb2 = np.mean(embeddings2, axis=0)
        avg_similarity = cosine_similarity([avg_emb1], [avg_emb2])[0][0]
        
        # Calculate pairwise similarities
        pairwise_similarities = cosine_similarity(embeddings1, embeddings2)
        max_similarities = np.max(pairwise_similarities, axis=1)
        
        # Find most similar article pairs
        similar_pairs = []
        for i, row in enumerate(pairwise_similarities):
            max_idx = np.argmax(row)
            if row[max_idx] > self.similarity_threshold:
                similar_pairs.append({
                    'article1_idx': i,
                    'article2_idx': max_idx,
                    'similarity': float(row[max_idx]),
                    'article1_snippet': articles1[i].get('text', '')[:200],
                    'article2_snippet': articles2[max_idx].get('text', '')[:200]
                })
        
        return {
            'average_semantic_similarity': float(avg_similarity),
            'max_similarity': float(np.max(pairwise_similarities)),
            'min_similarity': float(np.min(pairwise_similarities)),
            'median_similarity': float(np.median(pairwise_similarities)),
            'similarity_distribution': {
                'q25': float(np.percentile(pairwise_similarities, 25)),
                'q75': float(np.percentile(pairwise_similarities, 75)),
                'std': float(np.std(pairwise_similarities))
            },
            'high_similarity_pairs': sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)[:10],
            'coverage_overlap': len(similar_pairs) / len(articles1) if articles1 else 0
        }
    
    def _identify_cross_lingual_clusters(self, embeddings_by_language: Dict[str, np.ndarray],
                                       articles_by_language: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Identify semantic clusters that span multiple languages"""
        # Combine all embeddings
        all_embeddings = []
        embedding_sources = []
        
        for lang, embeddings in embeddings_by_language.items():
            if embeddings is not None:
                all_embeddings.extend(embeddings)
                embedding_sources.extend([(lang, i) for i in range(len(embeddings))])
        
        if len(all_embeddings) < 2:
            return {'clusters': [], 'cross_lingual_clusters': 0}
        
        # Perform clustering
        n_clusters = min(10, len(all_embeddings) // 5)  # Adaptive cluster count
        if n_clusters < 2:
            return {'clusters': [], 'cross_lingual_clusters': 0}
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(all_embeddings)
        
        # Analyze clusters for cross-lingual content
        clusters_info = []
        cross_lingual_count = 0
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_sources = [embedding_sources[i] for i in cluster_indices]
            
            # Group by language
            languages_in_cluster = defaultdict(list)
            for lang, article_idx in cluster_sources:
                languages_in_cluster[lang].append(article_idx)
            
            if len(languages_in_cluster) > 1:  # Cross-lingual cluster
                cross_lingual_count += 1
                
                cluster_articles = {}
                for lang, indices in languages_in_cluster.items():
                    cluster_articles[lang] = [articles_by_language[lang][i] for i in indices]
                
                clusters_info.append({
                    'cluster_id': cluster_id,
                    'languages': list(languages_in_cluster.keys()),
                    'language_distribution': {lang: len(indices) for lang, indices in languages_in_cluster.items()},
                    'articles_by_language': cluster_articles,
                    'cluster_size': len(cluster_indices),
                    'cross_lingual': True
                })
        
        return {
            'clusters': clusters_info,
            'total_clusters': n_clusters,
            'cross_lingual_clusters': cross_lingual_count,
            'cross_lingual_ratio': cross_lingual_count / n_clusters if n_clusters > 0 else 0
        }
    
    def clear_cache(self):
        """Clear analysis cache"""
        self.analysis_cache.clear()
        logging.info("Cross-lingual analysis cache cleared")