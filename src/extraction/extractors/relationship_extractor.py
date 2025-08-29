"""
Relationship extractor module for extracting relationships between entities.

This module provides functionality to extract relationships between identified
entities using LLM and document schemas.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI library not installed. Install with: pip install openai")

# Import from parent modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from extraction.document_schemas import DocumentSchema

logger = logging.getLogger(__name__)


class ExtractedRelationship(BaseModel):
    """Extracted relationship with metadata."""
    
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    type: str = Field(description="Relationship type from schema")
    description: str = Field(default="", description="Description of the relationship")
    confidence: float = Field(default=0.5, description="Extraction confidence (0-1)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")
    evidence: str = Field(default="", description="Evidence supporting the relationship")
    context: str = Field(default="", description="Context where relationship was found")
    chunk_id: Optional[str] = Field(default=None, description="Source chunk ID")


class RelationshipExtractor:
    """
    Extracts relationships between entities using LLM and document schemas.
    
    Analyzes text to identify how entities are connected, including
    relationship types, confidence scores, and supporting evidence.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        max_entities_for_context: int = 20
    ):
        """
        Initialize the relationship extractor.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for extraction
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
            max_entities_for_context: Maximum entities to include in prompt
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library required. Install with: pip install openai")
        
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_entities_for_context = max_entities_for_context
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_relationships(
        self,
        content: str,
        entities: List[Dict[str, Any]],
        schema: DocumentSchema,
        chunk_id: Optional[str] = None
    ) -> Tuple[List[ExtractedRelationship], Dict[str, int]]:
        """
        Extract relationships between entities from text.
        
        Args:
            content: Text content to analyze
            entities: List of entities found in the text
            schema: Document schema with relationship types
            chunk_id: Optional chunk identifier
            
        Returns:
            Tuple of (List of ExtractedRelationship objects, usage statistics)
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized. Provide API key.")
        
        if not entities:
            return [], {}
        
        rel_schema = schema.relationship_schema
        
        # Create entity list for context (limit to avoid token overflow)
        entity_list = [
            f"- {e.get('name', e.get('name', ''))} ({e.get('type', 'UNKNOWN')})"
            for e in entities[:self.max_entities_for_context]
        ]
        
        prompt = f"""You are an expert at identifying relationships in {schema.description}.

ENTITIES FOUND IN TEXT:
{chr(10).join(entity_list)}

RELATIONSHIP TYPES TO EXTRACT:
{', '.join(rel_schema.relationship_types)}

PRIORITY RELATIONSHIPS (focus on these):
{', '.join(rel_schema.priority_relationships)}

EXTRACTION HINTS:
{json.dumps(rel_schema.extraction_hints, indent=2)}

TEXT TO ANALYZE:
{content}

INSTRUCTIONS:
1. Extract ALL relationships between the identified entities
2. Focus on explicit connections mentioned in the text
3. Include confidence scores and evidence
4. For technical documents, identify:
   - Dependencies (X requires/uses/depends on Y)
   - Comparisons (X outperforms/improves upon Y)
   - Compositions (X contains/includes/consists of Y)
   - Derivations (X is based on/extends/modifies Y)
   - Applications (X is applied to/used for Y)

Return a JSON array of relationships:
[
  {{
    "source": "source entity name",
    "target": "target entity name",
    "type": "RELATIONSHIP_TYPE",
    "description": "relationship description",
    "confidence": 0.9,
    "evidence": "specific text from content supporting this relationship"
  }}
]

Be precise and only extract relationships that are clearly stated or strongly implied."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise relationship extraction system."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Track usage for cost calculation
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            result = json.loads(response.choices[0].message.content)
            relationships_data = result.get("relationships", []) if "relationships" in result else result
            
            # Ensure it's a list
            if isinstance(relationships_data, dict):
                relationships_data = [relationships_data]
            
            # Convert to ExtractedRelationship objects
            relationships = []
            for rel_data in relationships_data:
                relationship = ExtractedRelationship(
                    source=rel_data.get("source", ""),
                    target=rel_data.get("target", ""),
                    type=rel_data.get("type", "RELATED_TO"),
                    description=rel_data.get("description", ""),
                    confidence=rel_data.get("confidence", 0.5),
                    properties=rel_data.get("properties", {}),
                    evidence=rel_data.get("evidence", ""),
                    context=rel_data.get("context", ""),
                    chunk_id=chunk_id
                )
                relationships.append(relationship)
            
            return relationships, usage
            
        except Exception as e:
            self.logger.error(f"Relationship extraction failed for chunk {chunk_id}: {e}")
            return [], {}
    
    def batch_extract_relationships(
        self,
        chunks: List[Dict[str, Any]],
        entities_per_chunk: List[List[Dict[str, Any]]],
        schema: DocumentSchema,
        show_progress: bool = True
    ) -> Tuple[List[ExtractedRelationship], List[Dict[str, int]]]:
        """
        Extract relationships from multiple chunks.
        
        Args:
            chunks: List of chunk dictionaries
            entities_per_chunk: List of entity lists, one per chunk
            schema: Document schema
            show_progress: Whether to log progress
            
        Returns:
            Tuple of (all relationships, list of usage statistics)
        """
        all_relationships = []
        all_usage = []
        total = len(chunks)
        
        # Ensure we have entities for each chunk
        if len(entities_per_chunk) != len(chunks):
            self.logger.warning(f"Mismatch: {len(chunks)} chunks but {len(entities_per_chunk)} entity lists")
            entities_per_chunk = entities_per_chunk[:len(chunks)] + [[]] * (len(chunks) - len(entities_per_chunk))
        
        for i, (chunk, entities) in enumerate(zip(chunks, entities_per_chunk)):
            if show_progress:
                self.logger.info(f"Extracting relationships from chunk {i+1}/{total}")
            
            relationships, usage = self.extract_relationships(
                content=chunk.get("content", ""),
                entities=entities,
                schema=schema,
                chunk_id=chunk.get("chunk_id", f"chunk_{i}")
            )
            
            all_relationships.extend(relationships)
            if usage:
                all_usage.append(usage)
        
        return all_relationships, all_usage
    
    def filter_relationships_by_confidence(
        self,
        relationships: List[ExtractedRelationship],
        min_confidence: float = 0.5
    ) -> List[ExtractedRelationship]:
        """
        Filter relationships by minimum confidence score.
        
        Args:
            relationships: List of extracted relationships
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of relationships
        """
        return [r for r in relationships if r.confidence >= min_confidence]
    
    def group_relationships_by_type(
        self,
        relationships: List[ExtractedRelationship]
    ) -> Dict[str, List[ExtractedRelationship]]:
        """
        Group relationships by their type.
        
        Args:
            relationships: List of extracted relationships
            
        Returns:
            Dictionary mapping relationship types to lists of relationships
        """
        grouped = {}
        for relationship in relationships:
            if relationship.type not in grouped:
                grouped[relationship.type] = []
            grouped[relationship.type].append(relationship)
        return grouped
    
    def find_entity_connections(
        self,
        entity_name: str,
        relationships: List[ExtractedRelationship]
    ) -> Dict[str, List[ExtractedRelationship]]:
        """
        Find all relationships connected to a specific entity.
        
        Args:
            entity_name: Name of the entity to search for
            relationships: List of all relationships
            
        Returns:
            Dictionary with 'incoming' and 'outgoing' relationship lists
        """
        entity_lower = entity_name.lower()
        connections = {
            "incoming": [],  # Where entity is the target
            "outgoing": []   # Where entity is the source
        }
        
        for rel in relationships:
            if rel.source.lower() == entity_lower:
                connections["outgoing"].append(rel)
            if rel.target.lower() == entity_lower:
                connections["incoming"].append(rel)
        
        return connections
    
    def get_relationship_statistics(
        self,
        relationships: List[ExtractedRelationship]
    ) -> Dict[str, Any]:
        """
        Calculate statistics about extracted relationships.
        
        Args:
            relationships: List of extracted relationships
            
        Returns:
            Dictionary with relationship statistics
        """
        type_counts = {}
        confidence_sum = 0
        unique_pairs = set()
        
        for rel in relationships:
            type_counts[rel.type] = type_counts.get(rel.type, 0) + 1
            confidence_sum += rel.confidence
            unique_pairs.add((rel.source, rel.target))
        
        return {
            "total_relationships": len(relationships),
            "relationship_types": type_counts,
            "average_confidence": confidence_sum / len(relationships) if relationships else 0,
            "unique_entity_pairs": len(unique_pairs)
        }