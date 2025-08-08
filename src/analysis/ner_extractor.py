#!/usr/bin/env python3
"""
Enhanced Named Entity Recognition Extractor for NewsBot 2.0
Advanced NER with relationship extraction and knowledge graph construction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from collections import defaultdict, Counter
from datetime import datetime
import pickle
import re

import spacy
import nltk
from spacy import displacy

# Try to import networkx for knowledge graph construction
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logging.warning("networkx not available. Install for knowledge graph features.")

# Try to import transformers for advanced NER
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("transformers not available. Install for advanced NER features.")

class NERExtractor:
    """
    Advanced Named Entity Recognition with relationship extraction and knowledge graph construction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize NER extractor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Load spaCy model
        spacy_model = self.config.get('spacy_model', 'en_core_web_sm')
        try:
            self.nlp = spacy.load(spacy_model)
            logging.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logging.warning(f"spaCy model {spacy_model} not found. Trying alternative...")
            try:
                self.nlp = spacy.load('en_core_web_sm')
                logging.info("Loaded fallback spaCy model: en_core_web_sm")
            except OSError:
                raise RuntimeError("No spaCy model available. Please install: python -m spacy download en_core_web_sm")
        
        # Initialize transformer-based NER if available
        self.transformer_ner = None
        if HAS_TRANSFORMERS:
            try:
                ner_model = self.config.get('transformer_ner_model', 'dbmdz/bert-large-cased-finetuned-conll03-english')
                self.transformer_ner = pipeline(
                    "ner",
                    model=ner_model,
                    aggregation_strategy="simple"
                )
                logging.info(f"Loaded transformer NER model: {ner_model}")
            except Exception as e:
                logging.warning(f"Could not load transformer NER model: {e}")
                self.transformer_ner = None
        
        # Entity type mappings and hierarchies
        self.entity_types = {
            'PERSON': {'description': 'People, including fictional', 'color': '#ff9999'},
            'ORG': {'description': 'Companies, agencies, institutions', 'color': '#66b3ff'},
            'GPE': {'description': 'Countries, cities, states', 'color': '#99ff99'},
            'LOC': {'description': 'Non-GPE locations', 'color': '#ffcc99'},
            'DATE': {'description': 'Absolute or relative dates', 'color': '#ff99cc'},
            'TIME': {'description': 'Times smaller than a day', 'color': '#ccff99'},
            'MONEY': {'description': 'Monetary values', 'color': '#ffff99'},
            'PERCENT': {'description': 'Percentage values', 'color': '#ff9966'},
            'QUANTITY': {'description': 'Measurements, weights', 'color': '#9966ff'},
            'ORDINAL': {'description': 'First, second, etc.', 'color': '#66ffcc'},
            'CARDINAL': {'description': 'Numerals not covered by other types', 'color': '#ff66cc'}
        }
        
        # Relationship patterns (for rule-based relationship extraction)
        self.relationship_patterns = {
            'CEO_OF': [
                r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:the\s+)?CEO\s+of\s+([A-Z][a-zA-Z\s&]+)',
                r'CEO\s+(\w+(?:\s+\w+)*)\s+of\s+([A-Z][a-zA-Z\s&]+)',
                r'([A-Z][a-zA-Z\s&]+)\s+CEO\s+(\w+(?:\s+\w+)*)'
            ],
            'WORKS_FOR': [
                r'(\w+(?:\s+\w+)*)\s+works?\s+(?:for|at)\s+([A-Z][a-zA-Z\s&]+)',
                r'(\w+(?:\s+\w+)*)\s+(?:employee|staff|member)\s+(?:of|at)\s+([A-Z][a-zA-Z\s&]+)'
            ],
            'LOCATED_IN': [
                r'([A-Z][a-zA-Z\s&]+)\s+(?:in|located\s+in|based\s+in)\s+([A-Z][a-zA-Z\s,]+)',
                r'([A-Z][a-zA-Z\s&]+),?\s+([A-Z][a-zA-Z]+(?:,\s+[A-Z]{2})?)'  # Company, City pattern
            ],
            'ACQUIRED_BY': [
                r'([A-Z][a-zA-Z\s&]+)\s+(?:was\s+)?acquired\s+by\s+([A-Z][a-zA-Z\s&]+)',
                r'([A-Z][a-zA-Z\s&]+)\s+purchase[ds]\s+([A-Z][a-zA-Z\s&]+)'
            ],
            'PARTNERED_WITH': [
                r'([A-Z][a-zA-Z\s&]+)\s+partner(?:ed|ship)?\s+with\s+([A-Z][a-zA-Z\s&]+)',
                r'([A-Z][a-zA-Z\s&]+)\s+(?:and|&)\s+([A-Z][a-zA-Z\s&]+)\s+(?:announced|signed|agreed)'
            ]
        }
        
        # Knowledge graph
        self.knowledge_graph = None
        if HAS_NETWORKX:
            self.knowledge_graph = nx.DiGraph()
        
        # Entity cache for consistency
        self.entity_cache = defaultdict(set)
        self.relationship_cache = []
        
        # Statistics
        self.extraction_stats = {
            'total_texts_processed': 0,
            'total_entities_extracted': 0,
            'total_relationships_extracted': 0,
            'entity_type_counts': defaultdict(int),
            'relationship_type_counts': defaultdict(int)
        }
    
    def extract_entities(self, text: str, methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract named entities using multiple methods
        
        Args:
            text: Input text
            methods: List of methods to use ['spacy', 'transformer']
            
        Returns:
            Dictionary with extracted entities
        """
        if methods is None:
            methods = ['spacy']
            if self.transformer_ner:
                methods.append('transformer')
        
        results = {
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'methods_used': methods
        }
        
        # spaCy NER
        if 'spacy' in methods:
            spacy_entities = self._extract_spacy_entities(text)
            results['spacy'] = spacy_entities
        
        # Transformer NER
        if 'transformer' in methods and self.transformer_ner:
            transformer_entities = self._extract_transformer_entities(text)
            results['transformer'] = transformer_entities
        
        # Merge and deduplicate entities
        results['merged'] = self._merge_entities(results)
        
        # Extract relationships
        results['relationships'] = self.extract_relationships(text, results['merged']['entities'])
        
        # Update statistics
        self._update_stats(results)
        
        return results
    
    def _extract_spacy_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities using spaCy"""
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entity = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 1.0,  # spaCy doesn't provide confidence scores
                'description': self.entity_types.get(ent.label_, {}).get('description', 'Unknown')
            }
            entities.append(entity)
        
        # Additional processing for better entity resolution
        entities = self._resolve_entity_conflicts(entities)
        entities = self._normalize_entities(entities)
        
        return {
            'entities': entities,
            'entity_count': len(entities),
            'entity_types': list(set(ent['label'] for ent in entities)),
            'processing_time': 0  # Placeholder
        }
    
    def _extract_transformer_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities using transformer model"""
        try:
            # Split text into chunks if too long
            max_length = 512
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            
            all_entities = []
            offset = 0
            
            for chunk in chunks:
                chunk_entities = self.transformer_ner(chunk)
                
                # Process results
                for ent in chunk_entities:
                    entity = {
                        'text': ent['word'],
                        'label': ent['entity_group'],
                        'start': ent['start'] + offset,
                        'end': ent['end'] + offset,
                        'confidence': ent['score'],
                        'description': self._map_transformer_label(ent['entity_group'])
                    }
                    all_entities.append(entity)
                
                offset += len(chunk)
            
            # Post-process entities
            all_entities = self._resolve_entity_conflicts(all_entities)
            all_entities = self._normalize_entities(all_entities)
            
            return {
                'entities': all_entities,
                'entity_count': len(all_entities),
                'entity_types': list(set(ent['label'] for ent in all_entities)),
                'processing_time': 0  # Placeholder
            }
            
        except Exception as e:
            logging.error(f"Transformer NER extraction failed: {e}")
            return {
                'entities': [],
                'entity_count': 0,
                'entity_types': [],
                'error': str(e)
            }
    
    def _merge_entities(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Merge entities from different methods"""
        all_entities = []
        
        # Collect entities from all methods
        for method in ['spacy', 'transformer']:
            if method in results and 'entities' in results[method]:
                for entity in results[method]['entities']:
                    entity['source'] = method
                    all_entities.append(entity)
        
        # Remove duplicates based on text and position overlap
        merged_entities = []
        
        for entity in all_entities:
            is_duplicate = False
            
            for existing in merged_entities:
                # Check for overlap
                if (entity['text'].lower() == existing['text'].lower() or
                    self._entities_overlap(entity, existing)):
                    
                    # Keep entity with higher confidence
                    if entity['confidence'] > existing['confidence']:
                        merged_entities.remove(existing)
                        merged_entities.append(entity)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged_entities.append(entity)
        
        # Sort by position
        merged_entities.sort(key=lambda x: x['start'])
        
        return {
            'entities': merged_entities,
            'entity_count': len(merged_entities),
            'entity_types': list(set(ent['label'] for ent in merged_entities)),
            'sources_used': list(set(ent['source'] for ent in merged_entities))
        }
    
    def extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract relationships between entities
        
        Args:
            text: Input text
            entities: List of extracted entities
            
        Returns:
            Dictionary with extracted relationships
        """
        relationships = []
        
        # Rule-based relationship extraction
        rule_relationships = self._extract_rule_based_relationships(text)
        relationships.extend(rule_relationships)
        
        # Pattern-based relationship extraction using entities
        entity_relationships = self._extract_entity_relationships(text, entities)
        relationships.extend(entity_relationships)
        
        # Dependency parsing for relationships
        if self.nlp:
            dep_relationships = self._extract_dependency_relationships(text, entities)
            relationships.extend(dep_relationships)
        
        # Remove duplicates
        unique_relationships = self._deduplicate_relationships(relationships)
        
        return {
            'relationships': unique_relationships,
            'relationship_count': len(unique_relationships),
            'relationship_types': list(set(rel['type'] for rel in unique_relationships))
        }
    
    def _extract_rule_based_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships using predefined patterns"""
        relationships = []
        
        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    if len(match.groups()) >= 2:
                        entity1 = match.group(1).strip()
                        entity2 = match.group(2).strip()
                        
                        relationship = {
                            'type': rel_type,
                            'entity1': entity1,
                            'entity2': entity2,
                            'confidence': 0.8,  # Rule-based confidence
                            'source': 'rule_based',
                            'context': match.group(0),
                            'start': match.start(),
                            'end': match.end()
                        }
                        relationships.append(relationship)
        
        return relationships
    
    def _extract_entity_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships based on entity proximity and context"""
        relationships = []
        
        # Group entities by type
        persons = [e for e in entities if e['label'] == 'PERSON']
        orgs = [e for e in entities if e['label'] == 'ORG']
        locations = [e for e in entities if e['label'] in ['GPE', 'LOC']]
        
        # Look for person-organization relationships
        for person in persons:
            for org in orgs:
                # Check if they appear in the same sentence
                distance = abs(person['start'] - org['start'])
                if distance < 200:  # Within reasonable distance
                    
                    # Extract context
                    start = min(person['start'], org['start']) - 50
                    end = max(person['end'], org['end']) + 50
                    context = text[max(0, start):min(len(text), end)]
                    
                    # Determine relationship type based on context
                    rel_type = 'ASSOCIATED_WITH'
                    confidence = 0.6
                    
                    if any(keyword in context.lower() for keyword in ['ceo', 'chief', 'president', 'founder']):
                        rel_type = 'LEADS'
                        confidence = 0.8
                    elif any(keyword in context.lower() for keyword in ['employee', 'works', 'staff']):
                        rel_type = 'WORKS_FOR'
                        confidence = 0.7
                    
                    relationship = {
                        'type': rel_type,
                        'entity1': person['text'],
                        'entity2': org['text'],
                        'confidence': confidence,
                        'source': 'entity_proximity',
                        'context': context,
                        'distance': distance
                    }
                    relationships.append(relationship)
        
        # Look for organization-location relationships
        for org in orgs:
            for location in locations:
                distance = abs(org['start'] - location['start'])
                if distance < 100:
                    
                    start = min(org['start'], location['start']) - 30
                    end = max(org['end'], location['end']) + 30
                    context = text[max(0, start):min(len(text), end)]
                    
                    relationship = {
                        'type': 'LOCATED_IN',
                        'entity1': org['text'],
                        'entity2': location['text'],
                        'confidence': 0.7,
                        'source': 'entity_proximity',
                        'context': context,
                        'distance': distance
                    }
                    relationships.append(relationship)
        
        return relationships
    
    def _extract_dependency_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships using dependency parsing"""
        relationships = []
        
        doc = self.nlp(text)
        
        # Create entity span mapping
        entity_spans = {}
        for ent in entities:
            for token in doc:
                if token.idx >= ent['start'] and token.idx < ent['end']:
                    entity_spans[token.i] = ent
        
        # Look for specific dependency patterns
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                # Find related entities
                head = token.head
                
                if token.i in entity_spans and head.i in entity_spans:
                    entity1 = entity_spans[token.i]
                    entity2 = entity_spans[head.i]
                    
                    # Determine relationship type based on dependency and POS
                    rel_type = 'RELATED_TO'
                    confidence = 0.5
                    
                    if head.lemma_ in ['work', 'employ', 'hire']:
                        rel_type = 'WORKS_FOR'
                        confidence = 0.8
                    elif head.lemma_ in ['lead', 'head', 'manage']:
                        rel_type = 'LEADS'
                        confidence = 0.8
                    elif head.lemma_ in ['locate', 'base', 'situate']:
                        rel_type = 'LOCATED_IN'
                        confidence = 0.7
                    
                    relationship = {
                        'type': rel_type,
                        'entity1': entity1['text'],
                        'entity2': entity2['text'],
                        'confidence': confidence,
                        'source': 'dependency_parsing',
                        'dependency': token.dep_,
                        'head_lemma': head.lemma_
                    }
                    relationships.append(relationship)
        
        return relationships
    
    def build_knowledge_graph(self, relationship_data: List[Dict[str, Any]]) -> Optional[Any]:
        """
        Build knowledge graph from extracted relationships
        
        Args:
            relationship_data: List of relationships with entities
            
        Returns:
            NetworkX graph object or None if networkx not available
        """
        if not HAS_NETWORKX:
            logging.warning("NetworkX not available for knowledge graph construction")
            # Return a basic graph structure without NetworkX
            return {
                'graph': None,
                'nodes': [],
                'edges': [],
                'message': 'NetworkX not available for advanced graph features',
                'fallback_used': True
            }
        
        # Initialize or clear existing graph
        self.knowledge_graph = nx.DiGraph()
        
        # Add relationships to graph
        for rel_data in relationship_data:
            if 'relationships' in rel_data:
                for rel in rel_data['relationships']:
                    entity1 = rel['entity1']
                    entity2 = rel['entity2']
                    rel_type = rel['type']
                    confidence = rel['confidence']
                    
                    # Add nodes if they don't exist
                    if not self.knowledge_graph.has_node(entity1):
                        self.knowledge_graph.add_node(entity1, type='entity')
                    
                    if not self.knowledge_graph.has_node(entity2):
                        self.knowledge_graph.add_node(entity2, type='entity')
                    
                    # Add edge with relationship information
                    self.knowledge_graph.add_edge(
                        entity1, entity2,
                        relationship=rel_type,
                        confidence=confidence,
                        source=rel.get('source', 'unknown')
                    )
        
        return self.knowledge_graph
    
    def analyze_entity_network(self) -> Dict[str, Any]:
        """Analyze the entity knowledge graph"""
        if not self.knowledge_graph or not HAS_NETWORKX:
            return {'error': 'Knowledge graph not available'}
        
        analysis = {}
        
        # Basic graph statistics
        analysis['graph_stats'] = {
            'num_nodes': self.knowledge_graph.number_of_nodes(),
            'num_edges': self.knowledge_graph.number_of_edges(),
            'density': nx.density(self.knowledge_graph),
            'is_connected': nx.is_weakly_connected(self.knowledge_graph)
        }
        
        # Centrality measures
        if self.knowledge_graph.number_of_nodes() > 0:
            try:
                # Degree centrality
                degree_centrality = nx.degree_centrality(self.knowledge_graph)
                analysis['most_connected_entities'] = sorted(
                    degree_centrality.items(), key=lambda x: x[1], reverse=True
                )[:10]
                
                # PageRank
                pagerank = nx.pagerank(self.knowledge_graph)
                analysis['most_important_entities'] = sorted(
                    pagerank.items(), key=lambda x: x[1], reverse=True
                )[:10]
                
                # Betweenness centrality (for smaller graphs)
                if self.knowledge_graph.number_of_nodes() < 1000:
                    betweenness = nx.betweenness_centrality(self.knowledge_graph)
                    analysis['bridge_entities'] = sorted(
                        betweenness.items(), key=lambda x: x[1], reverse=True
                    )[:10]
                
            except Exception as e:
                logging.warning(f"Error calculating centrality measures: {e}")
        
        # Community detection
        try:
            # Convert to undirected for community detection
            undirected = self.knowledge_graph.to_undirected()
            communities = nx.algorithms.community.greedy_modularity_communities(undirected)
            
            analysis['communities'] = [list(community) for community in communities]
            analysis['num_communities'] = len(communities)
            
        except Exception as e:
            logging.warning(f"Error in community detection: {e}")
        
        # Relationship type analysis
        edge_types = defaultdict(int)
        for u, v, data in self.knowledge_graph.edges(data=True):
            edge_types[data.get('relationship', 'unknown')] += 1
        
        analysis['relationship_distribution'] = dict(edge_types)
        
        return analysis
    
    def find_entity_connections(self, entity1: str, entity2: str, max_path_length: int = 3) -> Dict[str, Any]:
        """
        Find connections between two entities in the knowledge graph
        
        Args:
            entity1: First entity
            entity2: Second entity
            max_path_length: Maximum path length to search
            
        Returns:
            Dictionary with connection information
        """
        if not self.knowledge_graph or not HAS_NETWORKX:
            return {'error': 'Knowledge graph not available'}
        
        if entity1 not in self.knowledge_graph or entity2 not in self.knowledge_graph:
            return {'error': 'One or both entities not found in knowledge graph'}
        
        connections = {}
        
        try:
            # Find shortest path
            if nx.has_path(self.knowledge_graph, entity1, entity2):
                shortest_path = nx.shortest_path(self.knowledge_graph, entity1, entity2)
                connections['shortest_path'] = shortest_path
                connections['path_length'] = len(shortest_path) - 1
                
                # Get relationship details along the path
                path_relationships = []
                for i in range(len(shortest_path) - 1):
                    edge_data = self.knowledge_graph[shortest_path[i]][shortest_path[i+1]]
                    path_relationships.append({
                        'from': shortest_path[i],
                        'to': shortest_path[i+1],
                        'relationship': edge_data.get('relationship', 'unknown'),
                        'confidence': edge_data.get('confidence', 0)
                    })
                connections['path_relationships'] = path_relationships
            
            # Find all paths up to max length
            try:
                all_paths = list(nx.all_simple_paths(
                    self.knowledge_graph, entity1, entity2, cutoff=max_path_length
                ))
                connections['all_paths'] = all_paths[:10]  # Limit to first 10 paths
                connections['num_paths'] = len(all_paths)
            except nx.NetworkXNoPath:
                connections['all_paths'] = []
                connections['num_paths'] = 0
            
            # Find common neighbors
            neighbors1 = set(self.knowledge_graph.neighbors(entity1))
            neighbors2 = set(self.knowledge_graph.neighbors(entity2))
            common_neighbors = neighbors1.intersection(neighbors2)
            connections['common_neighbors'] = list(common_neighbors)
            
        except Exception as e:
            connections['error'] = str(e)
        
        return connections
    
    def _resolve_entity_conflicts(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve conflicts between overlapping entities"""
        # Sort entities by start position
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        
        resolved_entities = []
        
        for entity in sorted_entities:
            # Check for overlap with existing entities
            overlaps = []
            for existing in resolved_entities:
                if self._entities_overlap(entity, existing):
                    overlaps.append(existing)
            
            if not overlaps:
                # No overlap, add entity
                resolved_entities.append(entity)
            else:
                # Handle overlap - keep entity with higher confidence or longer text
                best_entity = entity
                for overlap in overlaps:
                    if (overlap['confidence'] > best_entity['confidence'] or
                        (overlap['confidence'] == best_entity['confidence'] and 
                         len(overlap['text']) > len(best_entity['text']))):
                        best_entity = overlap
                
                # Remove overlapping entities and add best one
                for overlap in overlaps:
                    if overlap in resolved_entities:
                        resolved_entities.remove(overlap)
                
                if best_entity not in resolved_entities:
                    resolved_entities.append(best_entity)
        
        return resolved_entities
    
    def _normalize_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize entity text and types"""
        normalized_entities = []
        
        for entity in entities:
            normalized_entity = entity.copy()
            
            # Clean entity text
            normalized_entity['text'] = entity['text'].strip()
            
            # Normalize entity types
            label = entity['label'].upper()
            if label in ['PERSON', 'PER']:
                normalized_entity['label'] = 'PERSON'
            elif label in ['ORG', 'ORGANIZATION']:
                normalized_entity['label'] = 'ORG'
            elif label in ['LOC', 'LOCATION']:
                normalized_entity['label'] = 'LOC'
            elif label in ['GPE', 'GEOPOLITICAL']:
                normalized_entity['label'] = 'GPE'
            
            normalized_entities.append(normalized_entity)
        
        return normalized_entities
    
    def _entities_overlap(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> bool:
        """Check if two entities overlap in position"""
        return not (entity1['end'] <= entity2['start'] or entity2['end'] <= entity1['start'])
    
    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate relationships"""
        seen = set()
        unique_relationships = []
        
        for rel in relationships:
            # Create a key for deduplication
            key = (rel['entity1'].lower(), rel['entity2'].lower(), rel['type'])
            
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        return unique_relationships
    
    def _map_transformer_label(self, label: str) -> str:
        """Map transformer model labels to standard entity types"""
        label_mapping = {
            'PER': 'People and fictional characters',
            'ORG': 'Organizations and companies',
            'LOC': 'Locations and places',
            'MISC': 'Miscellaneous entities'
        }
        return label_mapping.get(label, 'Unknown entity type')
    
    def _update_stats(self, results: Dict[str, Any]):
        """Update extraction statistics"""
        self.extraction_stats['total_texts_processed'] += 1
        
        if 'merged' in results:
            entities = results['merged']['entities']
            self.extraction_stats['total_entities_extracted'] += len(entities)
            
            for entity in entities:
                self.extraction_stats['entity_type_counts'][entity['label']] += 1
        
        if 'relationships' in results:
            relationships = results['relationships']['relationships']
            self.extraction_stats['total_relationships_extracted'] += len(relationships)
            
            for rel in relationships:
                self.extraction_stats['relationship_type_counts'][rel['type']] += 1
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive extraction statistics"""
        stats = self.extraction_stats.copy()
        stats['entity_type_counts'] = dict(stats['entity_type_counts'])
        stats['relationship_type_counts'] = dict(stats['relationship_type_counts'])
        
        if stats['total_texts_processed'] > 0:
            stats['avg_entities_per_text'] = stats['total_entities_extracted'] / stats['total_texts_processed']
            stats['avg_relationships_per_text'] = stats['total_relationships_extracted'] / stats['total_texts_processed']
        
        return stats
    
    def save_ner_model(self, filepath: str):
        """Save NER extractor configuration and knowledge graph"""
        model_data = {
            'config': self.config,
            'entity_types': self.entity_types,
            'relationship_patterns': self.relationship_patterns,
            'extraction_stats': self.extraction_stats,
            'knowledge_graph': self.knowledge_graph if HAS_NETWORKX else None,
            'save_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info(f"NER extractor saved to {filepath}")
    
    def load_ner_model(self, filepath: str):
        """Load NER extractor configuration and knowledge graph"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.entity_types = model_data['entity_types']
        self.relationship_patterns = model_data['relationship_patterns']
        self.extraction_stats = model_data['extraction_stats']
        
        if HAS_NETWORKX and model_data['knowledge_graph']:
            self.knowledge_graph = model_data['knowledge_graph']
        
        logging.info(f"NER extractor loaded from {filepath}")
    
    def visualize_entities(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Generate HTML visualization of entities in text
        
        Args:
            text: Original text
            entities: List of extracted entities
            
        Returns:
            HTML string for visualization
        """
        if not self.nlp:
            return "spaCy not available for visualization"
        
        # Create spaCy doc
        doc = self.nlp(text)
        
        # Convert entities to spaCy format
        spacy_entities = []
        for ent in entities:
            spacy_entities.append((ent['start'], ent['end'], ent['label']))
        
        # Create spans
        spans = []
        for start, end, label in spacy_entities:
            span = doc.char_span(start, end, label=label)
            if span:
                spans.append(span)
        
        # Filter overlapping spans
        filtered_spans = spacy.util.filter_spans(spans)
        doc.ents = filtered_spans
        
        # Generate HTML
        html = displacy.render(doc, style="ent", jupyter=False)
        
        return html