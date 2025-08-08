"""
Conversational Interface Module for NewsBot 2.0
Natural language query processing and interactive exploration
"""

from .query_processor import AdvancedQueryProcessor as QueryProcessor
from .intent_classifier import IntentClassifier
from .response_generator import ResponseGenerator

__all__ = ['QueryProcessor', 'IntentClassifier', 'ResponseGenerator']