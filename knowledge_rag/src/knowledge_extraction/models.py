"""
Data Models for Knowledge Extraction

Defines the core data structures for entities, relationships, documents, and extraction results.
Inspired by Graphiti's bi-temporal tracking and episodic processing.
"""

import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


class EntityType(Enum):
    """Standard entity types for classification"""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"  
    LOCATION = "LOCATION"
    CONCEPT = "CONCEPT"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    TECHNOLOGY = "TECHNOLOGY"
    DOCUMENT = "DOCUMENT"
    UNKNOWN = "UNKNOWN"
    
    @classmethod
    def from_string(cls, value: str) -> 'EntityType':
        """Convert string to EntityType, defaulting to UNKNOWN"""
        try:
            return cls(value.upper())
        except ValueError:
            return cls.UNKNOWN


class RelationshipType(Enum):
    """Standard relationship types"""
    # Structural relationships
    PART_OF = "PART_OF"
    CONTAINS = "CONTAINS"
    BELONGS_TO = "BELONGS_TO"
    
    # Social relationships
    WORKS_FOR = "WORKS_FOR"
    COLLABORATED_WITH = "COLLABORATED_WITH"
    REPORTS_TO = "REPORTS_TO"
    KNOWS = "KNOWS"
    
    # Temporal relationships
    PRECEDED_BY = "PRECEDED_BY"
    FOLLOWED_BY = "FOLLOWED_BY"
    FOLLOWS = "FOLLOWS"  # More natural than FOLLOWED_BY for some contexts
    CONCURRENT_WITH = "CONCURRENT_WITH"
    
    # Causal relationships
    CAUSED = "CAUSED"
    RESULTED_IN = "RESULTED_IN"
    INFLUENCED = "INFLUENCED"
    
    # Conceptual relationships
    SIMILAR_TO = "SIMILAR_TO"
    RELATED_TO = "RELATED_TO"
    OPPOSITE_OF = "OPPOSITE_OF"
    
    # Spatial relationships
    LOCATED_IN = "LOCATED_IN"
    NEAR = "NEAR"
    ADJACENT_TO = "ADJACENT_TO"
    
    # Document relationships
    MENTIONED_IN = "MENTIONED_IN"
    AUTHORED = "AUTHORED"
    REFERENCES = "REFERENCES"
    
    # Academic/Research relationships
    USES = "USES"
    USED_FOR = "USED_FOR"
    USED_WITH = "USED_WITH"
    USED_FOR_TRAINING = "USED_FOR_TRAINING"
    APPLIES = "APPLIES"
    APPLIED_TO = "APPLIED_TO"
    IMPLEMENTS = "IMPLEMENTS"
    BASED_ON = "BASED_ON"
    EXTENDS = "EXTENDS"
    IMPROVES_ON = "IMPROVES_ON"
    BUILDS_ON = "BUILDS_ON"
    
    # Performance/Comparison relationships
    OUTPERFORMS = "OUTPERFORMS"
    PERFORMS_BETTER_THAN = "PERFORMS_BETTER_THAN"
    ACHIEVES_HIGHER_SCORE_THAN = "ACHIEVES_HIGHER_SCORE_THAN"
    COMPARES_TO = "COMPARES_TO"
    PERFORMANCE_COMPARISON = "PERFORMANCE_COMPARISON"
    EVALUATED_AGAINST = "EVALUATED_AGAINST"
    BENCHMARKED_AGAINST = "BENCHMARKED_AGAINST"
    
    # Method/Technique relationships
    EMPLOYS = "EMPLOYS"
    UTILIZES = "UTILIZES"
    COMBINES_WITH = "COMBINES_WITH"
    INTEGRATED_WITH = "INTEGRATED_WITH"
    REQUIRES = "REQUIRES"
    DEPENDS_ON = "DEPENDS_ON"
    
    # Learning/Training relationships
    TRAINED_ON = "TRAINED_ON"
    LEARNS_FROM = "LEARNS_FROM"
    FINE_TUNED_ON = "FINE_TUNED_ON"
    TRAINED_WITH = "TRAINED_WITH"
    
    # Evaluation/Validation relationships
    EVALUATED_ON = "EVALUATED_ON"
    TESTED_ON = "TESTED_ON"
    VALIDATED_ON = "VALIDATED_ON"
    MEASURED_BY = "MEASURED_BY"
    
    # Generalization relationships
    GENERALIZES_TO = "GENERALIZES_TO"
    TRANSFERS_TO = "TRANSFERS_TO"
    APPLICABLE_TO = "APPLICABLE_TO"
    
    # Exception/Special case relationships
    EXCEPTION = "EXCEPTION"
    SPECIAL_CASE_OF = "SPECIAL_CASE_OF"
    VARIANT_OF = "VARIANT_OF"
    
    # Dataset relationships
    CONTAINS_DATA_FROM = "CONTAINS_DATA_FROM"
    SUBSET_OF = "SUBSET_OF"
    DERIVED_FROM = "DERIVED_FROM"
    
    # Generic
    UNKNOWN = "UNKNOWN"
    
    @classmethod
    def from_string(cls, value: str) -> 'RelationshipType':
        """Convert string to RelationshipType, defaulting to UNKNOWN"""
        try:
            return cls(value.upper())
        except ValueError:
            return cls.UNKNOWN


@dataclass
class Entity:
    """
    Represents an entity in the knowledge graph with bi-temporal tracking
    """
    # Core identifiers
    uuid: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    entity_type: EntityType = EntityType.UNKNOWN
    
    # Embeddings and similarity
    name_embedding: Optional[List[float]] = None
    
    # Content and metadata
    summary: str = ""
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Bi-temporal tracking (inspired by Graphiti)
    created_at: datetime = field(default_factory=datetime.utcnow)
    first_mentioned_at: Optional[datetime] = None
    last_updated_at: Optional[datetime] = None
    
    # Confidence and provenance
    confidence: float = 0.0
    source_episodes: List[str] = field(default_factory=list)
    source_documents: List[str] = field(default_factory=list)
    
    # Graph metadata
    group_id: str = "default"  # For partitioning/namespacing
    
    def __post_init__(self):
        """Post-initialization processing"""
        if not self.first_mentioned_at:
            self.first_mentioned_at = self.created_at
        if not self.last_updated_at:
            self.last_updated_at = self.created_at
    
    def generate_id(self) -> str:
        """Generate a deterministic ID based on name and type"""
        content = f"{self.name.lower()}:{self.entity_type.value}:{self.group_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def add_alias(self, alias: str):
        """Add an alias if not already present"""
        if alias not in self.aliases and alias != self.name:
            self.aliases.append(alias)
    
    def update_from_extraction(self, new_data: Dict[str, Any], episode_id: str, document_id: str):
        """Update entity with new extraction data"""
        # Update summary if provided
        if 'summary' in new_data and new_data['summary']:
            self.summary = new_data['summary']
        
        # Add new attributes
        if 'attributes' in new_data:
            self.attributes.update(new_data['attributes'])
        
        # Add aliases
        if 'aliases' in new_data:
            for alias in new_data['aliases']:
                self.add_alias(alias)
        
        # Update confidence (take max)
        if 'confidence' in new_data:
            self.confidence = max(self.confidence, new_data['confidence'])
        
        # Track provenance
        if episode_id not in self.source_episodes:
            self.source_episodes.append(episode_id)
        if document_id not in self.source_documents:
            self.source_documents.append(document_id)
        
        # Update timestamps
        self.last_updated_at = datetime.utcnow()


@dataclass
class Relationship:
    """
    Represents a relationship/edge in the knowledge graph with bi-temporal tracking
    """
    # Core identifiers
    uuid: str = field(default_factory=lambda: str(uuid4()))
    source_entity_id: str = ""
    target_entity_id: str = ""
    relationship_type: RelationshipType = RelationshipType.UNKNOWN
    
    # Content and description
    fact: str = ""  # Human-readable description of the relationship
    fact_embedding: Optional[List[float]] = None
    
    # Bi-temporal tracking (inspired by Graphiti)
    created_at: datetime = field(default_factory=datetime.utcnow)
    valid_at: Optional[datetime] = None  # When relationship became true in real world
    invalid_at: Optional[datetime] = None  # When relationship stopped being true in real world
    expired_at: Optional[datetime] = None  # When marked as expired in system
    
    # Confidence and provenance
    confidence: float = 0.0
    source_episodes: List[str] = field(default_factory=list)
    source_documents: List[str] = field(default_factory=list)
    
    # Additional attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Graph metadata
    group_id: str = "default"
    
    def generate_id(self) -> str:
        """Generate a deterministic ID based on entities and relationship type"""
        content = f"{self.source_entity_id}:{self.target_entity_id}:{self.relationship_type.value}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def is_expired(self) -> bool:
        """Check if relationship is expired"""
        return self.expired_at is not None
    
    def is_currently_valid(self, at_time: Optional[datetime] = None) -> bool:
        """Check if relationship is valid at given time (default: now)"""
        if at_time is None:
            at_time = datetime.utcnow()
        
        # Check system-level expiration
        if self.expired_at and self.expired_at <= at_time:
            return False
        
        # Check real-world validity
        if self.valid_at and self.valid_at > at_time:
            return False
        
        if self.invalid_at and self.invalid_at <= at_time:
            return False
        
        return True
    
    def expire(self):
        """Mark relationship as expired"""
        self.expired_at = datetime.utcnow()
    
    def update_temporal_info(self, valid_at: Optional[datetime] = None, invalid_at: Optional[datetime] = None):
        """Update temporal validity information"""
        if valid_at:
            self.valid_at = valid_at
        if invalid_at:
            self.invalid_at = invalid_at


@dataclass
class Document:
    """
    Represents a document layer that links episodes to their source document
    """
    # Core identifiers
    uuid: str = field(default_factory=lambda: str(uuid4()))
    file_path: str = ""
    file_name: str = ""
    file_hash: str = ""
    
    # Document metadata
    title: str = ""
    author: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: Optional[datetime] = None
    document_type: str = ""
    
    # Processing information
    processed_at: datetime = field(default_factory=datetime.utcnow)
    episode_ids: List[str] = field(default_factory=list)
    
    # Extracted knowledge summary
    entity_count: int = 0
    relationship_count: int = 0
    key_entities: List[str] = field(default_factory=list)
    key_relationships: List[str] = field(default_factory=list)
    
    # Content summary
    summary: str = ""
    topics: List[str] = field(default_factory=list)
    
    # Graph metadata
    group_id: str = "default"
    
    def add_episode(self, episode_id: str):
        """Add an episode ID if not already present"""
        if episode_id not in self.episode_ids:
            self.episode_ids.append(episode_id)
    
    def update_knowledge_stats(self, entities: List[Entity], relationships: List[Relationship]):
        """Update statistics based on extracted knowledge"""
        self.entity_count = len(entities)
        self.relationship_count = len(relationships)
        
        # Extract key entities (top 5 by confidence)
        sorted_entities = sorted(entities, key=lambda x: x.confidence, reverse=True)
        self.key_entities = [e.name for e in sorted_entities[:5]]
        
        # Extract key relationships (top 5 by confidence)
        sorted_relationships = sorted(relationships, key=lambda x: x.confidence, reverse=True)
        self.key_relationships = [r.fact for r in sorted_relationships[:5]]


@dataclass
class ExtractionResult:
    """
    Results from knowledge extraction process
    """
    # Extracted knowledge
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    
    # Processing metadata
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    # Statistics
    entities_extracted: int = 0
    relationships_extracted: int = 0
    entities_deduplicated: int = 0
    relationships_deduplicated: int = 0
    
    # Source information
    source_episode_id: str = ""
    source_document_id: str = ""
    
    def __post_init__(self):
        """Calculate statistics"""
        self.entities_extracted = len(self.entities)
        self.relationships_extracted = len(self.relationships)
    
    def add_entity(self, entity: Entity):
        """Add an entity to the results"""
        self.entities.append(entity)
        self.entities_extracted = len(self.entities)
    
    def add_relationship(self, relationship: Relationship):
        """Add a relationship to the results"""
        self.relationships.append(relationship)
        self.relationships_extracted = len(self.relationships)
    
    def set_deduplication_stats(self, entities_deduplicated: int, relationships_deduplicated: int):
        """Set deduplication statistics"""
        self.entities_deduplicated = entities_deduplicated
        self.relationships_deduplicated = relationships_deduplicated


# Type aliases for convenience
EntityDict = Dict[str, Entity]
RelationshipDict = Dict[str, Relationship]
DocumentDict = Dict[str, Document] 