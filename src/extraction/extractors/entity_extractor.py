"""
Entity extractor module for extracting entities from text using LLM.

This module provides functionality to extract entities based on document schemas,
with support for different entity types and confidence scoring.
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


class ExtractedEntity(BaseModel):
    """Extracted entity with metadata."""
    
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type from schema")
    description: str = Field(description="Brief description of the entity")
    confidence: float = Field(default=0.5, description="Extraction confidence (0-1)")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Entity attributes")
    context: str = Field(default="", description="Analytical insight about entity's role")
    chunk_id: Optional[str] = Field(default=None, description="Source chunk ID")


class EntityExtractor:
    """
    Extracts entities from text using LLM and document schemas.
    
    Provides schema-driven entity extraction with confidence scoring
    and contextual insights about each entity's role.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        """
        Initialize the entity extractor.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for extraction
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library required. Install with: pip install openai")
        
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_entities(
        self,
        content: str,
        schema: DocumentSchema,
        chunk_id: Optional[str] = None
    ) -> Tuple[List[ExtractedEntity], Dict[str, int]]:
        """
        Extract entities from text using LLM and schema.
        
        Args:
            content: Text content to extract from
            schema: Document schema with entity types
            chunk_id: Optional chunk identifier
            
        Returns:
            Tuple of (List of ExtractedEntity objects, usage statistics)
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized. Provide API key.")
        
        entity_schema = schema.entity_schema
        
        # Create extraction prompt with improved context instructions
        prompt = f"""You are an expert knowledge extractor specializing in {schema.description}.

Extract entities from the following text according to this schema:

ENTITY TYPES TO EXTRACT:
{', '.join(entity_schema.entity_types)}

PRIORITY ENTITIES (focus on these):
{', '.join(entity_schema.priority_entities)}

EXTRACTION HINTS:
{json.dumps(entity_schema.extraction_hints, indent=2)}

TEXT TO ANALYZE:
{content}

INSTRUCTIONS:
1. Extract ALL relevant entities matching the specified types
2. Include confidence scores (0-1) for each entity
3. Extract key attributes when available
4. For the context field, provide analytical insights about the entity's role in this chunk

Return a JSON array of entities with this structure:
[
  {{
    "name": "entity name",
    "type": "ENTITY_TYPE",
    "description": "brief description of what the entity is",
    "confidence": 0.9,
    "attributes": {{"key": "value"}},
    "context": "analytical insight about this entity in the chunk"
  }}
]

IMPORTANT for the "context" field:
- DO NOT just quote text. Instead, explain the entity's role, purpose, or significance in this chunk
- Examples of good context across different domains:
  * TECHNOLOGY/TOOL: "Introduced as the primary framework for building the system, chosen for its scalability and 50% faster processing speed"
  * PERSON/ORGANIZATION: "Mentioned as the lead developer who contributed the core algorithm, affiliated with MIT"
  * PROCESS/METHOD: "Described as a three-stage pipeline that reduces processing time from hours to minutes"
  * PRODUCT/SERVICE: "Launched in Q2 2023 as a cloud-based solution targeting enterprise customers"
  * CONCEPT/THEORY: "Explained as the foundational principle underlying the new approach, contrasting with traditional methods"
  * LOCATION/FACILITY: "Identified as the manufacturing site where production increased by 30% after automation"
  * EVENT/MILESTONE: "Marked as the turning point when the company pivoted to AI-focused strategy"
  * STANDARD/REGULATION: "Referenced as the compliance requirement driving the system redesign"
- Include quantitative details if mentioned (percentages, metrics, dates, amounts)
- Mention relationships to other entities when relevant
- Explain what the chunk reveals about this entity (findings, comparisons, applications, impacts)
- Keep it concise but informative (1-2 sentences)

Focus on specific, technical, and factual information. Be comprehensive."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise knowledge extraction system."},
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
            entities_data = result.get("entities", []) if "entities" in result else result
            
            # Ensure it's a list
            if isinstance(entities_data, dict):
                entities_data = [entities_data]
            
            # Convert to ExtractedEntity objects
            entities = []
            for entity_data in entities_data:
                entity = ExtractedEntity(
                    name=entity_data.get("name", ""),
                    type=entity_data.get("type", "UNKNOWN"),
                    description=entity_data.get("description", ""),
                    confidence=entity_data.get("confidence", 0.5),
                    attributes=entity_data.get("attributes", {}),
                    context=entity_data.get("context", ""),
                    chunk_id=chunk_id
                )
                entities.append(entity)
            
            return entities, usage
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed for chunk {chunk_id}: {e}")
            return [], {}
    
    def batch_extract_entities(
        self,
        chunks: List[Dict[str, Any]],
        schema: DocumentSchema,
        show_progress: bool = True
    ) -> Tuple[List[ExtractedEntity], List[Dict[str, int]]]:
        """
        Extract entities from multiple chunks.
        
        Args:
            chunks: List of chunk dictionaries
            schema: Document schema
            show_progress: Whether to log progress
            
        Returns:
            Tuple of (all entities, list of usage statistics)
        """
        all_entities = []
        all_usage = []
        total = len(chunks)
        
        for i, chunk in enumerate(chunks):
            if show_progress:
                self.logger.info(f"Extracting entities from chunk {i+1}/{total}")
            
            entities, usage = self.extract_entities(
                content=chunk.get("content", ""),
                schema=schema,
                chunk_id=chunk.get("chunk_id", f"chunk_{i}")
            )
            
            all_entities.extend(entities)
            if usage:
                all_usage.append(usage)
        
        return all_entities, all_usage
    
    def filter_entities_by_confidence(
        self,
        entities: List[ExtractedEntity],
        min_confidence: float = 0.5
    ) -> List[ExtractedEntity]:
        """
        Filter entities by minimum confidence score.
        
        Args:
            entities: List of extracted entities
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of entities
        """
        return [e for e in entities if e.confidence >= min_confidence]
    
    def group_entities_by_type(
        self,
        entities: List[ExtractedEntity]
    ) -> Dict[str, List[ExtractedEntity]]:
        """
        Group entities by their type.
        
        Args:
            entities: List of extracted entities
            
        Returns:
            Dictionary mapping entity types to lists of entities
        """
        grouped = {}
        for entity in entities:
            if entity.type not in grouped:
                grouped[entity.type] = []
            grouped[entity.type].append(entity)
        return grouped
    
    def get_entity_statistics(
        self,
        entities: List[ExtractedEntity]
    ) -> Dict[str, Any]:
        """
        Calculate statistics about extracted entities.
        
        Args:
            entities: List of extracted entities
            
        Returns:
            Dictionary with entity statistics
        """
        type_counts = {}
        confidence_sum = 0
        
        for entity in entities:
            type_counts[entity.type] = type_counts.get(entity.type, 0) + 1
            confidence_sum += entity.confidence
        
        return {
            "total_entities": len(entities),
            "entity_types": type_counts,
            "average_confidence": confidence_sum / len(entities) if entities else 0,
            "unique_names": len(set(e.name for e in entities))
        }