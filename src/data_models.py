"""Data models for the knowledge graph system using Pydantic."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class EntityType(str, Enum):
    """Enumeration of entity types."""
    
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    PRODUCT = "product"
    OTHER = "other"


class RelationshipType(str, Enum):
    """Enumeration of relationship types."""
    
    WORKS_FOR = "works_for"
    LOCATED_IN = "located_in"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    CREATED_BY = "created_by"
    OCCURRED_AT = "occurred_at"
    MENTIONS = "mentions"
    OTHER = "other"


class Entity(BaseModel):
    """Entity schema for storing extracted entities."""
    
    entity_id: str
    entity_type: EntityType
    name: str
    description: str
    chunk_ids: List[str] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)


class Relationship(BaseModel):
    """Relationship schema for storing entity relationships."""
    
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationshipType
    description: str
    chunk_id: str
    document_id: str
    section_path: List[str] = Field(default_factory=list)
    temporal_info: Optional[datetime] = None
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)
    attributes: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Chunk schema for document chunks."""
    
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    section_path: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    image_paths: List[str] = Field(default_factory=list)
    tables: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class Document(BaseModel):
    """Document schema for storing document metadata."""
    
    document_id: str
    title: str
    source_path: str
    file_type: str
    total_chunks: int = Field(ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Community(BaseModel):
    """Community schema for graph communities."""
    
    community_id: str
    entity_ids: List[str] = Field(default_factory=list)
    summary: str = Field(default="")
    level: int = Field(default=0, ge=0)
    parent_community_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class SearchQuery(BaseModel):
    """Search query schema."""
    
    query_text: str
    search_type: str = Field(default="hybrid")
    filters: Dict[str, Any] = Field(default_factory=dict)
    temporal_filter: Optional[datetime] = None
    limit: int = Field(default=10, gt=0)
    include_communities: bool = Field(default=True)
    include_relationships: bool = Field(default=True)


class SearchResult(BaseModel):
    """Search result schema."""
    
    entity_id: Optional[str] = None
    chunk_id: Optional[str] = None
    community_id: Optional[str] = None
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    content: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_type: str = Field(default="chunk")