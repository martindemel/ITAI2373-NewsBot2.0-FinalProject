#!/usr/bin/env python3
"""
Response Generator for NewsBot 2.0
Advanced natural language response generation for query results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import json
import re
from collections import defaultdict
import pickle

class ResponseGenerator:
    """
    Advanced response generator for creating natural language responses to query results
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize response generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Response templates for different intents
        self.response_templates = {
            'search_articles': {
                'intro_templates': [
                    "I found {count} articles {criteria}.",
                    "Here are {count} articles {criteria}.",
                    "Your search returned {count} articles {criteria}."
                ],
                'no_results': "I couldn't find any articles matching your criteria. Try broader search terms.",
                'summary_templates': [
                    "The articles cover topics including {topics}.",
                    "Key themes include {topics}.",
                    "The results span topics such as {topics}."
                ]
            },
            'analyze_sentiment': {
                'intro_templates': [
                    "Sentiment analysis of {count} articles shows {sentiment_summary}.",
                    "The emotional tone analysis reveals {sentiment_summary}.",
                    "Based on {count} articles, the sentiment is {sentiment_summary}."
                ],
                'detailed_templates': [
                    "{positive_pct}% positive, {negative_pct}% negative, {neutral_pct}% neutral.",
                    "Overall sentiment score: {avg_sentiment:.2f} (range: -1 to 1)."
                ]
            },
            'classify_content': {
                'intro_templates': [
                    "Content classification analysis completed.",
                    "I've categorized the content as follows:",
                    "Classification results:"
                ],
                'confidence_templates': [
                    "High confidence ({confidence:.1%}): {category}",
                    "Likely category ({confidence:.1%}): {category}",
                    "Uncertain classification ({confidence:.1%}): {category}"
                ]
            },
            'summarize_text': {
                'intro_templates': [
                    "Here's a summary of the content:",
                    "Summary (compressed to {compression:.1%} of original):",
                    "Key highlights:"
                ],
                'quality_templates': [
                    "Summary quality: {quality_grade}",
                    "Compression ratio: {compression:.1%}",
                    "Generated using {method} method"
                ]
            },
            'extract_entities': {
                'intro_templates': [
                    "I found {total_entities} entities in the content:",
                    "Entity extraction identified {total_entities} key entities:",
                    "Here are the key entities I discovered:"
                ],
                'entity_templates': [
                    "People: {people}",
                    "Organizations: {organizations}",
                    "Locations: {locations}",
                    "Other entities: {other}"
                ]
            },
            'get_insights': {
                'intro_templates': [
                    "Here are the key insights from your data:",
                    "Analysis reveals several important patterns:",
                    "Key findings from the data analysis:"
                ],
                'insight_templates': [
                    "Data overview: {data_summary}",
                    "Trending topics: {trends}",
                    "Sentiment patterns: {sentiment}"
                ]
            },
            'compare_sources': {
                'intro_templates': [
                    "Cross-language comparison analysis:",
                    "Comparative analysis across {language_count} languages:",
                    "Source comparison results:"
                ],
                'comparison_templates': [
                    "Coverage varies significantly across languages",
                    "Similar sentiment patterns detected",
                    "Different editorial priorities identified"
                ]
            },
            'trend_analysis': {
                'intro_templates': [
                    "Trend analysis over {time_period} shows:",
                    "Temporal patterns in the data reveal:",
                    "Time-based analysis indicates:"
                ],
                'trend_templates': [
                    "Increasing trend in {category}",
                    "Declining coverage of {category}",
                    "Stable patterns for {category}"
                ]
            },
            'export_results': {
                'intro_templates': [
                    "Export completed successfully.",
                    "Your data has been prepared for download.",
                    "Export file generated:"
                ],
                'export_templates': [
                    "Format: {format}",
                    "Size: {size} bytes",
                    "Download: {url}"
                ]
            },
            'get_help': {
                'intro_templates': [
                    "I'm NewsBot 2.0, here to help with news analysis.",
                    "Welcome to NewsBot 2.0! Here's what I can do:",
                    "NewsBot 2.0 capabilities:"
                ],
                'capability_templates': [
                    "• {name}: {description}",
                    "✓ {name} - {description}"
                ]
            }
        }
        
        # Response personalization templates
        self.personalization_templates = {
            'greeting': [
                "Hello! ",
                "Hi there! ",
                "Welcome back! ",
                ""
            ],
            'confidence_modifiers': {
                'high': ["I'm confident that", "The analysis clearly shows", "Results indicate"],
                'medium': ["It appears that", "The data suggests", "Analysis indicates"],
                'low': ["It's possible that", "There are indications that", "Preliminary analysis suggests"]
            },
            'follow_up_suggestions': {
                'search_articles': [
                    "Would you like me to analyze the sentiment of these articles?",
                    "I can summarize these articles if you'd like.",
                    "Would you like to see entities mentioned in these articles?"
                ],
                'analyze_sentiment': [
                    "Would you like to see which articles are most positive/negative?",
                    "I can analyze sentiment trends over time if you have date information.",
                    "Would you like me to compare sentiment across different categories?"
                ],
                'classify_content': [
                    "Would you like me to explain why I classified it this way?",
                    "I can analyze the sentiment of this classified content.",
                    "Would you like to see similar articles in this category?"
                ]
            }
        }
        
        # Response formatting options
        self.formatting_options = {
            'detailed': {
                'include_metadata': True,
                'include_confidence': True,
                'include_suggestions': True,
                'max_items_shown': 10
            },
            'brief': {
                'include_metadata': False,
                'include_confidence': False,
                'include_suggestions': False,
                'max_items_shown': 3
            },
            'technical': {
                'include_metadata': True,
                'include_confidence': True,
                'include_suggestions': False,
                'max_items_shown': 20,
                'show_technical_details': True
            }
        }
        
        # Statistics
        self.generation_stats = {
            'total_responses': 0,
            'responses_by_intent': defaultdict(int),
            'avg_response_length': 0,
            'personalization_usage': defaultdict(int)
        }
    
    def generate_response(self, query: str, intent: str, execution_result: Dict[str, Any],
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate natural language response for query results
        
        Args:
            query: Original user query
            intent: Detected intent
            execution_result: Results from query execution
            context: Additional context information
            
        Returns:
            Dictionary with generated response and metadata
        """
        try:
            # Determine response format
            response_format = context.get('preferred_format', 'detailed')
            formatting = self.formatting_options.get(response_format, self.formatting_options['detailed'])
            
            # Generate response based on intent
            if intent in self.response_templates:
                response_content = self._generate_intent_specific_response(
                    intent, execution_result, context, formatting
                )
            else:
                response_content = self._generate_generic_response(execution_result, formatting)
            
            # Add personalization
            personalized_response = self._apply_personalization(
                response_content, context, query
            )
            
            # Format final response
            final_response = self._format_final_response(
                personalized_response, execution_result, context, formatting
            )
            
            # Update statistics
            self._update_generation_stats(intent, final_response)
            
            return {
                'response': final_response,
                'intent': intent,
                'format': response_format,
                'generation_timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            logging.error(f"Response generation failed: {e}")
            return {
                'response': self._generate_error_response(str(e)),
                'intent': intent,
                'error': str(e),
                'generation_timestamp': datetime.now().isoformat(),
                'success': False
            }
    
    def _generate_intent_specific_response(self, intent: str, execution_result: Dict[str, Any],
                                         context: Dict[str, Any], formatting: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response specific to the intent"""
        
        templates = self.response_templates[intent]
        
        if intent == 'search_articles':
            return self._generate_search_response(execution_result, templates, formatting)
        elif intent == 'analyze_sentiment':
            return self._generate_sentiment_response(execution_result, templates, formatting)
        elif intent == 'classify_content':
            return self._generate_classification_response(execution_result, templates, formatting)
        elif intent == 'summarize_text':
            return self._generate_summarization_response(execution_result, templates, formatting)
        elif intent == 'extract_entities':
            return self._generate_entity_response(execution_result, templates, formatting)
        elif intent == 'get_insights':
            return self._generate_insights_response(execution_result, templates, formatting)
        elif intent == 'compare_sources':
            return self._generate_comparison_response(execution_result, templates, formatting)
        elif intent == 'trend_analysis':
            return self._generate_trend_response(execution_result, templates, formatting)
        elif intent == 'export_results':
            return self._generate_export_response(execution_result, templates, formatting)
        elif intent == 'get_help':
            return self._generate_help_response(execution_result, templates, formatting)
        else:
            return self._generate_generic_response(execution_result, formatting)
    
    def _generate_search_response(self, result: Dict[str, Any], templates: Dict[str, Any],
                                formatting: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response for article search"""
        
        returned_count = result.get('returned_count', 0)
        total_found = result.get('total_found', 0)
        articles = result.get('articles', [])
        
        if returned_count == 0:
            return {
                'main_message': templates['no_results'],
                'details': [],
                'metadata': {}
            }
        
        # Generate intro message
        criteria_parts = []
        search_criteria = result.get('search_criteria', {})
        
        if search_criteria.get('keywords'):
            criteria_parts.append(f"containing '{', '.join(search_criteria['keywords'])}'")
        if search_criteria.get('categories'):
            criteria_parts.append(f"in {', '.join(search_criteria['categories'])}")
        
        criteria_text = " ".join(criteria_parts) if criteria_parts else "matching your search"
        
        intro_template = np.random.choice(templates['intro_templates'])
        main_message = intro_template.format(count=returned_count, criteria=criteria_text)
        
        # Generate details
        details = []
        
        if formatting['include_metadata'] and total_found > returned_count:
            details.append(f"Showing {returned_count} of {total_found} total results")
        
        # Show article previews
        max_shown = min(formatting['max_items_shown'], len(articles))
        
        for i, article in enumerate(articles[:max_shown]):
            article_info = {
                'title': article.get('title', f"Article {i+1}"),
                'category': article.get('category', 'Unknown'),
                'preview': article.get('text', '')[:150] + '...',
                'relevance_score': article.get('similarity_score', 'N/A')
            }
            
            if formatting['include_metadata']:
                details.append(f"**{article_info['title']}** ({article_info['category']})")
                details.append(article_info['preview'])
            else:
                details.append(article_info['preview'])
        
        # Extract topics from articles
        if len(articles) > 1:
            categories = [a.get('category', '') for a in articles if a.get('category')]
            unique_categories = list(set(categories))
            
            if unique_categories:
                summary_template = np.random.choice(templates['summary_templates'])
                topic_summary = summary_template.format(topics=', '.join(unique_categories[:3]))
                details.append(topic_summary)
        
        return {
            'main_message': main_message,
            'details': details,
            'metadata': {
                'returned_count': returned_count,
                'total_found': total_found,
                'ranking_method': result.get('ranking_method', 'unknown')
            }
        }
    
    def _generate_sentiment_response(self, result: Dict[str, Any], templates: Dict[str, Any],
                                   formatting: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response for sentiment analysis"""
        
        total_analyzed = result.get('total_analyzed', 0)
        aggregated = result.get('aggregated_sentiment', {})
        
        if total_analyzed == 0:
            return {
                'main_message': "No articles were available for sentiment analysis.",
                'details': [],
                'metadata': {}
            }
        
        # Generate sentiment summary
        sentiment_summary = self._create_sentiment_summary(aggregated)
        
        intro_template = np.random.choice(templates['intro_templates'])
        main_message = intro_template.format(
            count=total_analyzed,
            sentiment_summary=sentiment_summary
        )
        
        # Generate details
        details = []
        
        if 'classification_distribution' in aggregated:
            dist = aggregated['classification_distribution']
            
            positive_pct = int(dist.get('positive', 0) * 100)
            negative_pct = int(dist.get('negative', 0) * 100)
            neutral_pct = int(dist.get('neutral', 0) * 100)
            
            details.append(f"{positive_pct}% positive, {negative_pct}% negative, {neutral_pct}% neutral")
        
        if formatting['include_metadata'] and 'average_sentiment' in aggregated:
            avg_sentiment = aggregated['average_sentiment']
            details.append(f"Average sentiment score: {avg_sentiment:.3f} (range: -1 to +1)")
            
            if 'sentiment_range' in aggregated:
                min_sent, max_sent = aggregated['sentiment_range']
                details.append(f"Sentiment range: {min_sent:.3f} to {max_sent:.3f}")
        
        # Add dominant sentiment
        if 'dominant_sentiment' in aggregated:
            dominant = aggregated['dominant_sentiment']
            details.append(f"Overall tone: {dominant}")
        
        return {
            'main_message': main_message,
            'details': details,
            'metadata': {
                'total_analyzed': total_analyzed,
                'dominant_sentiment': aggregated.get('dominant_sentiment', 'neutral')
            }
        }
    
    def _generate_classification_response(self, result: Dict[str, Any], templates: Dict[str, Any],
                                        formatting: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response for content classification"""
        
        classification_results = result.get('classification_results', [])
        
        if not classification_results:
            return {
                'main_message': "No content was available for classification.",
                'details': [],
                'metadata': {}
            }
        
        intro_template = np.random.choice(templates['intro_templates'])
        main_message = intro_template
        
        details = []
        
        for i, class_result in enumerate(classification_results[:formatting['max_items_shown']]):
            category = class_result.get('predicted_category', 'Unknown')
            confidence = class_result.get('confidence', 0)
            
            # Choose template based on confidence
            if confidence >= 0.8:
                conf_template = templates['confidence_templates'][0]  # High confidence
            elif confidence >= 0.6:
                conf_template = templates['confidence_templates'][1]  # Medium confidence
            else:
                conf_template = templates['confidence_templates'][2]  # Low confidence
            
            detail = conf_template.format(confidence=confidence, category=category.title())
            details.append(detail)
            
            # Show alternatives if formatting allows
            if formatting['include_metadata'] and 'alternatives' in class_result:
                alternatives = class_result['alternatives'][:2]  # Top 2 alternatives
                alt_text = ", ".join([f"{alt['category']} ({alt['probability']:.1%})" for alt in alternatives])
                details.append(f"  Alternatives: {alt_text}")
        
        return {
            'main_message': main_message,
            'details': details,
            'metadata': {
                'total_classified': len(classification_results),
                'avg_confidence': np.mean([r.get('confidence', 0) for r in classification_results])
            }
        }
    
    def _generate_summarization_response(self, result: Dict[str, Any], templates: Dict[str, Any],
                                       formatting: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response for text summarization"""
        
        summary = result.get('summary', '')
        compression_ratio = result.get('compression_ratio', 0)
        quality_metrics = result.get('quality_metrics', {})
        
        if not summary:
            return {
                'main_message': "Unable to generate summary for the provided content.",
                'details': [],
                'metadata': {}
            }
        
        intro_template = np.random.choice(templates['intro_templates'])
        main_message = intro_template.format(compression=compression_ratio)
        
        details = [summary]
        
        if formatting['include_metadata']:
            method_used = result.get('method_used', 'unknown')
            details.append(f"Method: {method_used}")
            
            if 'quality_grade' in quality_metrics:
                quality_grade = quality_metrics['quality_grade']
                details.append(f"Quality: {quality_grade}")
            
            original_length = result.get('original_length', 0)
            summary_length = result.get('summary_length', 0)
            details.append(f"Reduced from {original_length} to {summary_length} characters")
        
        return {
            'main_message': main_message,
            'details': details,
            'metadata': {
                'compression_ratio': compression_ratio,
                'quality_grade': quality_metrics.get('quality_grade', 'unknown'),
                'method_used': result.get('method_used', 'unknown')
            }
        }
    
    def _generate_entity_response(self, result: Dict[str, Any], templates: Dict[str, Any],
                                formatting: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response for entity extraction"""
        
        entities_by_type = result.get('entities_by_type', {})
        total_entities = result.get('total_entities', 0)
        
        if total_entities == 0:
            return {
                'main_message': "No entities were found in the content.",
                'details': [],
                'metadata': {}
            }
        
        intro_template = np.random.choice(templates['intro_templates'])
        main_message = intro_template.format(total_entities=total_entities)
        
        details = []
        
        # Group entities by common types
        entity_groups = {
            'People': entities_by_type.get('PERSON', []),
            'Organizations': entities_by_type.get('ORG', []),
            'Locations': entities_by_type.get('GPE', []) + entities_by_type.get('LOC', []),
            'Dates': entities_by_type.get('DATE', []),
            'Money': entities_by_type.get('MONEY', [])
        }
        
        for group_name, entities in entity_groups.items():
            if entities:
                entity_texts = [e.get('text', '') for e in entities[:5]]  # Top 5
                unique_entities = list(set(entity_texts))
                
                if len(unique_entities) > 3:
                    entity_display = ', '.join(unique_entities[:3]) + f" and {len(unique_entities)-3} others"
                else:
                    entity_display = ', '.join(unique_entities)
                
                details.append(f"**{group_name}**: {entity_display}")
        
        # Add relationships if available
        relationships = result.get('relationships', [])
        if relationships and formatting['include_metadata']:
            rel_count = len(relationships)
            details.append(f"Found {rel_count} relationships between entities")
            
            # Show a few example relationships
            for rel in relationships[:2]:
                entity1 = rel.get('entity1', '')
                entity2 = rel.get('entity2', '')
                rel_type = rel.get('type', 'related to')
                details.append(f"  {entity1} {rel_type.lower().replace('_', ' ')} {entity2}")
        
        return {
            'main_message': main_message,
            'details': details,
            'metadata': {
                'total_entities': total_entities,
                'total_relationships': len(relationships),
                'entity_types_found': list(entities_by_type.keys())
            }
        }
    
    def _generate_insights_response(self, result: Dict[str, Any], templates: Dict[str, Any],
                                  formatting: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response for insights analysis"""
        
        intro_template = np.random.choice(templates['intro_templates'])
        main_message = intro_template
        
        details = []
        
        # Data insights
        data_insights = result.get('data_insights', {})
        if data_insights:
            total_articles = data_insights.get('total_articles', 0)
            details.append(f"**Data Overview**: Analyzed {total_articles:,} articles")
            
            if 'category_distribution' in data_insights:
                categories = list(data_insights['category_distribution'].keys())
                details.append(f"Categories covered: {', '.join(categories[:5])}")
        
        # Topic insights
        trend_insights = result.get('trend_insights', {})
        if 'topics' in trend_insights:
            topic_info = trend_insights['topics']
            num_topics = topic_info.get('num_topics_discovered', 0)
            details.append(f"**Topics**: Discovered {num_topics} main themes")
            
            top_topics = topic_info.get('top_topics', [])
            if top_topics:
                topic_names = [topic.get('description', 'Topic') for topic in top_topics[:3]]
                details.append(f"Key themes: {', '.join(topic_names)}")
        
        # Sentiment insights
        pattern_insights = result.get('pattern_insights', {})
        if 'sentiment' in pattern_insights:
            sentiment_info = pattern_insights['sentiment']
            overall_sentiment = sentiment_info.get('overall_sentiment', {})
            
            if overall_sentiment:
                pos_ratio = overall_sentiment.get('positive', 0)
                neg_ratio = overall_sentiment.get('negative', 0)
                details.append(f"**Sentiment**: {pos_ratio:.1%} positive, {neg_ratio:.1%} negative coverage")
        
        # Recommendations
        recommendations = result.get('recommendations', [])
        if recommendations and formatting['include_suggestions']:
            details.append("**Recommendations**:")
            for rec in recommendations[:3]:
                details.append(f"• {rec}")
        
        return {
            'main_message': main_message,
            'details': details,
            'metadata': {
                'insights_generated': len([k for k in result.keys() if 'insights' in k]),
                'recommendations_count': len(recommendations)
            }
        }
    
    def _generate_comparison_response(self, result: Dict[str, Any], templates: Dict[str, Any],
                                    formatting: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response for cross-lingual comparison"""
        
        languages_requested = result.get('languages_requested', [])
        status = result.get('status', 'unknown')
        
        if status == 'requires_multilingual_data':
            main_message = "Cross-lingual analysis requires articles in multiple languages."
            details = [
                "I can perform cross-language comparison when multilingual data is available.",
                "This includes comparing topic coverage, sentiment patterns, and cultural perspectives."
            ]
        else:
            intro_template = np.random.choice(templates['intro_templates'])
            main_message = intro_template.format(language_count=len(languages_requested))
            details = ["Comparison analysis completed."]
        
        capabilities = result.get('capabilities', [])
        if capabilities and formatting['include_metadata']:
            details.append("**Available capabilities**:")
            for capability in capabilities:
                details.append(f"• {capability}")
        
        return {
            'main_message': main_message,
            'details': details,
            'metadata': {
                'languages_requested': languages_requested,
                'status': status
            }
        }
    
    def _generate_trend_response(self, result: Dict[str, Any], templates: Dict[str, Any],
                               formatting: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response for trend analysis"""
        
        time_period = result.get('time_period', ['unspecified period'])
        articles_analyzed = result.get('articles_analyzed', 0)
        trend_analysis = result.get('trend_analysis', {})
        
        intro_template = np.random.choice(templates['intro_templates'])
        main_message = intro_template.format(time_period=', '.join(time_period))
        
        details = []
        
        if articles_analyzed > 0:
            details.append(f"Analyzed {articles_analyzed:,} articles")
        
        # Topic trends
        if 'topic_trends' in trend_analysis:
            details.append("**Topic Trends**: Topic evolution patterns identified")
        
        # Sentiment trends
        if 'sentiment_trends' in trend_analysis:
            details.append("**Sentiment Trends**: Emotional tone changes tracked over time")
        
        if not trend_analysis:
            details.append("Temporal analysis requires articles with date information.")
        
        return {
            'main_message': main_message,
            'details': details,
            'metadata': {
                'articles_analyzed': articles_analyzed,
                'time_period': time_period,
                'trends_found': len(trend_analysis)
            }
        }
    
    def _generate_export_response(self, result: Dict[str, Any], templates: Dict[str, Any],
                                formatting: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response for export results"""
        
        export_format = result.get('export_format', 'unknown')
        export_size = result.get('export_size', 0)
        download_url = result.get('download_url', '')
        
        intro_template = np.random.choice(templates['intro_templates'])
        main_message = intro_template
        
        details = [
            f"Format: {export_format.upper()}",
            f"Size: {self._format_file_size(export_size)}"
        ]
        
        if download_url:
            details.append(f"Download: {download_url}")
        
        data_summary = result.get('data_summary', {})
        if data_summary and formatting['include_metadata']:
            details.append("**Export Contents**:")
            for key, value in data_summary.items():
                details.append(f"• {key}: {value}")
        
        return {
            'main_message': main_message,
            'details': details,
            'metadata': {
                'export_format': export_format,
                'export_size': export_size,
                'download_url': download_url
            }
        }
    
    def _generate_help_response(self, result: Dict[str, Any], templates: Dict[str, Any],
                              formatting: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response for help queries"""
        
        intro_template = np.random.choice(templates['intro_templates'])
        main_message = intro_template
        
        details = []
        
        # Available capabilities
        capabilities = result.get('available_capabilities', [])
        if capabilities:
            details.append("**Available Capabilities**:")
            for capability in capabilities[:formatting['max_items_shown']]:
                name = capability.get('name', 'Unknown')
                description = capability.get('description', 'No description')
                details.append(f"• **{name}**: {description}")
        
        # Example queries
        example_queries = result.get('example_queries', [])
        if example_queries and formatting['include_suggestions']:
            details.append("**Example Queries**:")
            for example in example_queries[:5]:
                details.append(f"• \"{example.get('example', '')}\"")
        
        # Tips
        tips = result.get('tips', [])
        if tips:
            details.append("**Tips for Better Results**:")
            for tip in tips[:3]:
                details.append(f"• {tip}")
        
        return {
            'main_message': main_message,
            'details': details,
            'metadata': {
                'capabilities_count': len(capabilities),
                'examples_count': len(example_queries)
            }
        }
    
    def _generate_generic_response(self, result: Dict[str, Any], formatting: Dict[str, Any]) -> Dict[str, Any]:
        """Generate generic response for unknown intents"""
        
        status = result.get('execution_status', 'unknown')
        
        if status == 'failed':
            main_message = "I encountered an issue processing your request."
            details = [result.get('error', 'Unknown error occurred')]
        else:
            main_message = "I've processed your request."
            details = ["Results are available in the data section."]
        
        return {
            'main_message': main_message,
            'details': details,
            'metadata': {'status': status}
        }
    
    def _apply_personalization(self, response_content: Dict[str, Any], 
                             context: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Apply personalization to response"""
        
        personalized = response_content.copy()
        
        # Add greeting for first interaction
        conversation_history = context.get('conversation_history', [])
        if len(conversation_history) == 0:
            greeting = np.random.choice(self.personalization_templates['greeting'])
            if greeting:
                personalized['main_message'] = greeting + personalized['main_message']
        
        # Add confidence modifiers based on execution confidence
        if 'confidence_assessment' in context:
            confidence_level = context['confidence_assessment'].get('confidence_level', 'medium')
            confidence_modifiers = self.personalization_templates['confidence_modifiers'].get(confidence_level, [])
            
            if confidence_modifiers:
                modifier = np.random.choice(confidence_modifiers)
                personalized['confidence_note'] = f"{modifier} based on the analysis."
        
        return personalized
    
    def _format_final_response(self, response_content: Dict[str, Any], execution_result: Dict[str, Any],
                             context: Dict[str, Any], formatting: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final response structure"""
        
        final_response = {
            'message': response_content['main_message'],
            'details': response_content.get('details', []),
            'metadata': response_content.get('metadata', {}),
            'format': context.get('preferred_format', 'detailed')
        }
        
        # Add confidence note if available
        if 'confidence_note' in response_content:
            final_response['confidence_note'] = response_content['confidence_note']
        
        # Add follow-up suggestions
        if formatting.get('include_suggestions', False):
            intent = execution_result.get('intent', '')
            if intent in self.personalization_templates['follow_up_suggestions']:
                suggestions = self.personalization_templates['follow_up_suggestions'][intent]
                final_response['follow_up_suggestions'] = suggestions[:2]
        
        # Add technical details if requested
        if formatting.get('show_technical_details', False):
            final_response['technical_details'] = {
                'execution_status': execution_result.get('execution_status', 'unknown'),
                'processing_time': context.get('processing_time', 0),
                'data_sources_used': context.get('available_data', {})
            }
        
        return final_response
    
    def _create_sentiment_summary(self, aggregated_sentiment: Dict[str, Any]) -> str:
        """Create human-readable sentiment summary"""
        
        if 'dominant_sentiment' in aggregated_sentiment:
            dominant = aggregated_sentiment['dominant_sentiment']
            
            if 'classification_distribution' in aggregated_sentiment:
                dist = aggregated_sentiment['classification_distribution']
                dominant_pct = int(dist.get(dominant, 0) * 100)
                
                return f"predominantly {dominant} ({dominant_pct}%)"
            else:
                return f"predominantly {dominant}"
        
        elif 'average_sentiment' in aggregated_sentiment:
            avg_sentiment = aggregated_sentiment['average_sentiment']
            
            if avg_sentiment > 0.1:
                return "generally positive"
            elif avg_sentiment < -0.1:
                return "generally negative"
            else:
                return "mixed or neutral"
        
        else:
            return "mixed sentiment patterns"
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate error response"""
        
        return {
            'message': "I apologize, but I encountered an error while generating the response.",
            'details': [f"Error: {error_message}"],
            'metadata': {'error': True},
            'follow_up_suggestions': [
                "Please try rephrasing your question",
                "Ask for help to see available capabilities",
                "Try a simpler query to test the system"
            ]
        }
    
    def _update_generation_stats(self, intent: str, response: Dict[str, Any]):
        """Update response generation statistics"""
        
        self.generation_stats['total_responses'] += 1
        self.generation_stats['responses_by_intent'][intent] += 1
        
        # Calculate response length
        response_text = response.get('message', '') + ' '.join(response.get('details', []))
        response_length = len(response_text)
        
        # Update average length
        total = self.generation_stats['total_responses']
        current_avg = self.generation_stats['avg_response_length']
        self.generation_stats['avg_response_length'] = (
            (current_avg * (total - 1) + response_length) / total
        )
        
        # Track personalization usage
        if 'confidence_note' in response:
            self.generation_stats['personalization_usage']['confidence_notes'] += 1
        
        if 'follow_up_suggestions' in response:
            self.generation_stats['personalization_usage']['follow_up_suggestions'] += 1
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get response generation statistics"""
        
        stats = {
            'total_responses': self.generation_stats['total_responses'],
            'avg_response_length': self.generation_stats['avg_response_length'],
            'responses_by_intent': dict(self.generation_stats['responses_by_intent']),
            'personalization_usage': dict(self.generation_stats['personalization_usage'])
        }
        
        return stats
    
    def save_generator(self, filepath: str):
        """Save response generator configuration"""
        
        save_data = {
            'config': self.config,
            'response_templates': self.response_templates,
            'personalization_templates': self.personalization_templates,
            'formatting_options': self.formatting_options,
            'generation_stats': dict(self.generation_stats),
            'save_timestamp': datetime.now().isoformat()
        }
        
        # Convert defaultdicts to regular dicts
        save_data['generation_stats']['responses_by_intent'] = dict(self.generation_stats['responses_by_intent'])
        save_data['generation_stats']['personalization_usage'] = dict(self.generation_stats['personalization_usage'])
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logging.info(f"Response generator saved to {filepath}")
    
    def load_generator(self, filepath: str):
        """Load response generator configuration"""
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.config = save_data['config']
        self.response_templates = save_data['response_templates']
        self.personalization_templates = save_data['personalization_templates']
        self.formatting_options = save_data['formatting_options']
        
        # Restore statistics
        stats = save_data['generation_stats']
        self.generation_stats['total_responses'] = stats.get('total_responses', 0)
        self.generation_stats['avg_response_length'] = stats.get('avg_response_length', 0)
        
        # Restore defaultdicts
        self.generation_stats['responses_by_intent'] = defaultdict(
            int, stats.get('responses_by_intent', {})
        )
        self.generation_stats['personalization_usage'] = defaultdict(
            int, stats.get('personalization_usage', {})
        )
        
        logging.info(f"Response generator loaded from {filepath}")