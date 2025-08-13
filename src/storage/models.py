"""
Models for paper extraction storage system.

This module defines the data models for storing extracted paper information
in both Neo4j graph database and vector databases.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class EntityType(str, Enum):
    """Types of entities that can be extracted from papers."""
    
    # Academic paper entities
    CONCEPT = "CONCEPT"
    METHODOLOGY = "METHODOLOGY"
    ALGORITHM = "ALGORITHM"
    MODEL = "MODEL"
    DATASET = "DATASET"
    METRIC = "METRIC"
    TECHNOLOGY = "TECHNOLOGY"
    FRAMEWORK = "FRAMEWORK"
    THEORY = "THEORY"
    HYPOTHESIS = "HYPOTHESIS"
    EXPERIMENT = "EXPERIMENT"
    RESULT = "RESULT"
    CONTRIBUTION = "CONTRIBUTION"
    LIMITATION = "LIMITATION"
    
    # Industry report entities
    COMPANY = "COMPANY"
    PRODUCT = "PRODUCT"
    MARKET = "MARKET"
    TREND = "TREND"
    COMPETITOR = "COMPETITOR"
    
    # General entities
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    DOCUMENT = "DOCUMENT"


class RelationshipType(str, Enum):
    """Types of relationships between entities."""
    
    # Academic relationships
    IMPROVES_UPON = "IMPROVES_UPON"
    EXTENDS = "EXTENDS"
    APPLIES = "APPLIES"
    EVALUATES_WITH = "EVALUATES_WITH"
    OUTPERFORMS = "OUTPERFORMS"
    BUILDS_ON = "BUILDS_ON"
    CONTRADICTS = "CONTRADICTS"
    VALIDATES = "VALIDATES"
    USES_DATASET = "USES_DATASET"
    IMPLEMENTS = "IMPLEMENTS"
    COMPARES_TO = "COMPARES_TO"
    INSPIRED_BY = "INSPIRED_BY"
    ADDRESSES_LIMITATION_OF = "ADDRESSES_LIMITATION_OF"
    ENABLES = "ENABLES"
    REQUIRES = "REQUIRES"
    
    # Industry relationships
    COMPETES_WITH = "COMPETES_WITH"
    PARTNERS_WITH = "PARTNERS_WITH"
    ACQUIRES = "ACQUIRES"
    
    # General relationships
    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"
    MENTIONS = "MENTIONS"
    REFERENCES = "REFERENCES"


class ExtractedEntity(BaseModel):
    """Entity extracted from a paper."""
    
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type from schema")
    description: str = Field(description="Entity description")
    confidence: float = Field(default=1.0, description="Extraction confidence score")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Entity attributes")
    context: Optional[str] = Field(None, description="Context where entity was found")
    chunk_id: Optional[str] = Field(None, description="ID of the chunk where entity was found")
    source_file: Optional[str] = Field(None, description="Source file path")
    paper_title: Optional[str] = Field(None, description="Title of the paper")
    created_at: datetime = Field(default_factory=datetime.now)


class ExtractedRelationship(BaseModel):
    """Relationship between entities extracted from a paper."""
    
    source_entity: str = Field(description="Source entity name")
    target_entity: str = Field(description="Target entity name")
    relationship_type: str = Field(description="Type of relationship")
    confidence: float = Field(default=1.0, description="Extraction confidence score")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")
    context: Optional[str] = Field(None, description="Context describing the relationship")
    chunk_id: Optional[str] = Field(None, description="ID of the chunk where relationship was found")
    source_file: Optional[str] = Field(None, description="Source file path")
    created_at: datetime = Field(default_factory=datetime.now)


class ExtractedPaper(BaseModel):
    """Complete extracted information from a paper."""
    
    file_path: str = Field(description="Path to the paper file")
    title: Optional[str] = Field(None, description="Paper title")
    document_type: str = Field(description="Type of document (e.g., academic_paper)")
    schema_used: str = Field(description="Schema used for extraction")
    entities: List[ExtractedEntity] = Field(default_factory=list, description="Extracted entities")
    relationships: List[ExtractedRelationship] = Field(default_factory=list, description="Extracted relationships")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Document chunks")


class SearchResult(BaseModel):
    """Result from searching the storage system."""
    
    content: str = Field(description="Content of the search result")
    score: float = Field(description="Relevance score")
    result_type: str = Field(description="Type of result (entity/relationship/chunk)")
    source_file: Optional[str] = Field(None, description="Source file path")
    entity_name: Optional[str] = Field(None, description="Entity name if applicable")
    entity_type: Optional[str] = Field(None, description="Entity type if applicable")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class StorageStats(BaseModel):
    """Statistics about the storage system."""
    
    total_papers: int = Field(default=0, description="Total number of papers stored")
    total_entities: int = Field(default=0, description="Total number of entities")
    total_relationships: int = Field(default=0, description="Total number of relationships")
    total_chunks: int = Field(default=0, description="Total number of chunks")
    entities_by_type: Dict[str, int] = Field(default_factory=dict, description="Entity count by type")
    relationships_by_type: Dict[str, int] = Field(default_factory=dict, description="Relationship count by type")
    papers_by_type: Dict[str, int] = Field(default_factory=dict, description="Paper count by document type")