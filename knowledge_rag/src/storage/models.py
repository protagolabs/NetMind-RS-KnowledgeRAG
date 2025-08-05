"""
Storage Models for Knowledge RAG System

Defines the data models for storing entities, relationships, episodes, and documents
in Neo4j graph database with full traceability back to source documents.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Import from knowledge extraction models
from ..knowledge_extraction.models import EntityType, RelationshipType


@dataclass
class StoredDocument:
    """Represents a document stored in the graph"""
    uuid: str
    file_path: str
    file_name: str
    file_hash: str
    title: str
    document_type: str  # 'markdown', 'pdf', 'docx', etc.
    created_at: datetime
    modified_at: datetime
    processed_at: datetime
    total_episodes: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    
    # Neo4j properties
    labels: List[str] = field(default_factory=lambda: ['Document'])
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StoredEpisode:
    """Represents an episode (chunk) stored in the graph"""
    uuid: str
    content: str
    episode_type: str
    sequence_number: int
    timestamp: datetime
    
    # Document linkage
    document_uuid: str
    document_path: str
    
    # Content metadata
    content_length: int
    header_level: Optional[int] = None
    header_text: Optional[str] = None
    
    # Processing metadata
    processed_at: datetime = field(default_factory=datetime.now)
    
    # Embedding
    embedding: Optional[List[float]] = None
    
    # Neo4j properties
    labels: List[str] = field(default_factory=lambda: ['Episode'])
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j node properties"""
        props = {
            'uuid': self.uuid,
            'content': self.content,
            'episode_type': self.episode_type,
            'sequence_number': self.sequence_number,
            'timestamp': self.timestamp,
            'document_uuid': self.document_uuid,
            'document_path': self.document_path,
            'content_length': self.content_length,
            'processed_at': self.processed_at
        }
        
        if self.header_level is not None:
            props['header_level'] = self.header_level
        if self.header_text:
            props['header_text'] = self.header_text
            
        props.update(self.properties)
        return props


@dataclass
class StoredEntity:
    """Represents an entity stored in the graph"""
    uuid: str
    name: str
    entity_type: str  # String representation of EntityType
    summary: str
    aliases: List[str]
    attributes: Dict[str, Any]
    confidence: float
    
    # Temporal information
    created_at: datetime
    first_mentioned_at: Optional[datetime]
    last_updated_at: datetime
    
    # Provenance
    source_episodes: List[str] = field(default_factory=list)  # Episode UUIDs
    source_documents: List[str] = field(default_factory=list)  # Document UUIDs
    
    # Embedding for semantic search
    embedding: Optional[List[float]] = None
    
    # Neo4j properties
    labels: List[str] = field(default_factory=lambda: ['Entity'])
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j node properties"""
        props = {
            'uuid': self.uuid,
            'name': self.name,
            'entity_type': self.entity_type,
            'summary': self.summary,
            'aliases': self.aliases,
            'confidence': self.confidence,
            'created_at': self.created_at,
            'last_updated_at': self.last_updated_at,
            'source_episodes': self.source_episodes,
            'source_documents': self.source_documents
        }
        
        if self.first_mentioned_at:
            props['first_mentioned_at'] = self.first_mentioned_at
            
        # Add attributes as individual properties
        props.update(self.attributes)
        props.update(self.properties)
        return props


@dataclass
class StoredRelationship:
    """Represents a relationship stored in the graph"""
    uuid: str
    source_entity_uuid: str
    target_entity_uuid: str
    relationship_type: str  # String representation of RelationshipType
    fact: str
    confidence: float
    
    # Bi-temporal tracking (inspired by Graphiti)
    created_at: datetime
    valid_at: Optional[datetime] = None      # When relation became true in real world
    invalid_at: Optional[datetime] = None    # When relation stopped being true
    expired_at: Optional[datetime] = None    # When marked as expired in system
    
    # Provenance
    source_episodes: List[str] = field(default_factory=list)  # Episode UUIDs
    source_documents: List[str] = field(default_factory=list)  # Document UUIDs
    
    # Additional attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Neo4j properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j relationship properties"""
        props = {
            'uuid': self.uuid,
            'relationship_type': self.relationship_type,
            'fact': self.fact,
            'confidence': self.confidence,
            'created_at': self.created_at,
            'source_episodes': self.source_episodes,
            'source_documents': self.source_documents
        }
        
        if self.valid_at:
            props['valid_at'] = self.valid_at
        if self.invalid_at:
            props['invalid_at'] = self.invalid_at
        if self.expired_at:
            props['expired_at'] = self.expired_at
            
        props.update(self.attributes)
        props.update(self.properties)
        return props


@dataclass
class SearchResult:
    """Represents a search result with traceability"""
    # Core content
    content: str
    score: float
    result_type: str  # 'entity', 'relationship', 'episode', 'document'
    
    # Source information
    source_uuid: str
    source_type: str
    
    # Document traceability
    document_uuid: str
    document_path: str
    document_name: str
    
    # Episode traceability (if applicable)
    episode_uuid: Optional[str] = None
    episode_sequence: Optional[int] = None
    
    # Entity information (if applicable)
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None
    
    # Relationship information (if applicable)
    relationship_fact: Optional[str] = None
    relationship_type: Optional[str] = None
    source_entity: Optional[str] = None
    target_entity: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_display_content(self) -> str:
        """Get content appropriate for display"""
        if self.result_type == 'entity':
            return f"{self.entity_name}: {self.content}"
        elif self.result_type == 'relationship':
            return f"{self.source_entity} → {self.target_entity}: {self.relationship_fact}"
        elif self.result_type == 'episode':
            return self.content[:500] + "..." if len(self.content) > 500 else self.content
        else:
            return self.content
    
    def get_source_context(self) -> str:
        """Get source context for display"""
        context = f"From: {self.document_name}"
        if self.episode_sequence is not None:
            context += f" (Section {self.episode_sequence})"
        return context


@dataclass 
class StorageStats:
    """Statistics about stored knowledge"""
    total_documents: int = 0
    total_episodes: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    total_embeddings: int = 0
    
    # Entity breakdown
    entities_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Relationship breakdown  
    relationships_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Storage sizes
    graph_size: Optional[str] = None
    vector_size: Optional[str] = None
    
    # Last updated
    last_updated: datetime = field(default_factory=datetime.now) 