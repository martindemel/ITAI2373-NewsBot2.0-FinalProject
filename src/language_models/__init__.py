"""
Language Models Module for NewsBot 2.0
Text summarization, generation, and semantic understanding
"""

from .summarizer import IntelligentSummarizer
from .embeddings import SemanticEmbeddings

__all__ = ['IntelligentSummarizer', 'SemanticEmbeddings']