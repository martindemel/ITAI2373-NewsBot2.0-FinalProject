"""
Analysis Module for NewsBot 2.0
Enhanced classification, sentiment analysis, NER, and topic modeling
"""

from .classifier import NewsClassifier
from .sentiment_analyzer import SentimentAnalyzer
from .ner_extractor import NERExtractor
from .topic_modeler import TopicModeler

__all__ = ['NewsClassifier', 'SentimentAnalyzer', 'NERExtractor', 'TopicModeler']