"""
Relationship Extraction Prompt Template

Extracts relationships between entities with temporal information and confidence scoring.
Based on Graphiti's relationship extraction approach.
"""

from typing import Dict, Any, List
from ..models import RelationshipType


def RELATIONSHIP_EXTRACTION_PROMPT(context: Dict[str, Any]) -> str:
    """
    Generate relationship extraction prompt for a given context
    
    Args:
        context: Dictionary containing:
            - episode_content: The text content to extract relationships from
            - entities: List of extracted entities to find relationships between
            - relationship_types: List of valid relationship types
            - reference_time: ISO 8601 timestamp for temporal context
    """
    
    system_prompt = """You are an expert relationship extraction AI that identifies factual relationships between entities mentioned in text.

Your task is to:
1. Extract relationships between the provided entities that are explicitly stated or clearly implied in the text
2. Classify each relationship with the most appropriate type
3. Create human-readable fact descriptions
4. Determine temporal validity (when the relationship was/is true)
5. Provide confidence scores (0.0-1.0) for each relationship
6. Extract additional attributes when available

Focus on relationships that are:
- Factually stated or clearly implied in the text
- Meaningful for knowledge graph construction
- Specific and actionable
- Temporally bound when possible

Output your results as a valid JSON object."""

    entities_list = ""
    if 'entities' in context:
        entities_list = "\n".join([
            f"- {i+1}. {entity['name']} ({entity.get('entity_type', 'UNKNOWN')})"
            for i, entity in enumerate(context['entities'])
        ])

    # Get all relationship types from context (passed from extractor)
    # If not provided, get all types from the enum as fallback
    relationship_types = context.get('relationship_types', [rt.value for rt in RelationshipType])
    relationship_types_str = ', '.join(relationship_types)

    user_prompt = f"""
<EPISODE_CONTENT>
{context['episode_content']}
</EPISODE_CONTENT>

<ENTITIES>
{entities_list}
</ENTITIES>

<RELATIONSHIP_TYPES>
You MUST only use these exact relationship types - do not create new ones:
{relationship_types_str}

Total types available: {len(relationship_types)}
</RELATIONSHIP_TYPES>

<REFERENCE_TIME>
{context.get('reference_time', 'Not specified')}
</REFERENCE_TIME>

# TASK
Extract all factual relationships between the ENTITIES based on the EPISODE_CONTENT. For each relationship, provide:

1. **source_entity**: Name of the source entity (must match an entity from the list)
2. **target_entity**: Name of the target entity (must match an entity from the list)
3. **relationship_type**: One of the provided RELATIONSHIP_TYPES (exact match required)
4. **fact**: Human-readable description of the relationship (e.g., "John Smith works for OpenAI")
5. **confidence**: Float between 0.0-1.0 indicating extraction confidence
6. **valid_at**: ISO 8601 timestamp when relationship became true (if determinable)
7. **invalid_at**: ISO 8601 timestamp when relationship stopped being true (if determinable)
8. **attributes**: Additional properties of the relationship

# OUTPUT FORMAT
Return a JSON object with this structure:
```json
{{
  "relationships": [
    {{
      "source_entity": "Entity Name 1",
      "target_entity": "Entity Name 2", 
      "relationship_type": "RELATIONSHIP_TYPE",
      "fact": "Human readable description of the relationship",
      "confidence": 0.90,
      "valid_at": "2024-01-01T00:00:00Z",
      "invalid_at": null,
      "attributes": {{
        "context": "additional context",
        "strength": "strong/weak/moderate"
      }}
    }}
  ]
}}
```

# GUIDELINES
- Only extract relationships between entities from the provided ENTITIES list
- Each relationship must involve exactly two distinct entities
- The 'fact' should be a clear, human-readable statement
- Use temporal information from the text to set valid_at/invalid_at when possible
- Confidence should reflect how certain you are about the relationship
- Include directionality - consider if A->B is different from B->A
- For temporal relationships, be precise about dates/times mentioned
- Skip relationships that are too vague or uncertain
- Focus on factual relationships, not speculative connections
- CRITICAL: Only use relationship types from the provided list above
"""

    return f"System: {system_prompt}\n\nUser: {user_prompt}"


def RELATIONSHIP_EXTRACTION_STRUCTURED_PROMPT(context: Dict[str, Any]) -> list:
    """
    Generate structured messages for relationship extraction (for better API compatibility)
    
    Returns:
        List of message dictionaries for OpenAI API
    """
    
    system_message = {
        "role": "system",
        "content": """You are an expert relationship extraction AI. Extract factual relationships between entities with high precision.

Extract relationships that are:
- Explicitly stated or clearly implied in the text
- Between two distinct entities from the provided list
- Factually accurate and meaningful
- Temporally contextualized when possible

For each relationship, determine:
- Source and target entities (exact names from list)
- Most appropriate relationship type from the provided enum
- Clear, factual description
- Confidence score (0.0-1.0) 
- Temporal validity when determinable
- Additional contextual attributes

CRITICAL: Only use the exact relationship types provided in the list. Do not create new relationship types.

CRITICAL: You MUST respond with ONLY valid JSON in exactly this format - no other text, explanations, or formatting:

{
  "relationships": [
    {
      "source_entity": "Exact Entity Name",
      "target_entity": "Exact Entity Name", 
      "relationship_type": "TYPE_FROM_LIST",
      "fact": "Clear factual description",
      "confidence": 0.90,
      "valid_at": "2024-01-01T00:00:00Z",
      "invalid_at": null,
      "attributes": {}
    }
  ]
}

Return results as valid JSON only."""
    }
    
    entities_list = ""
    if 'entities' in context:
        entities_list = "\n".join([
            f"{i+1}. {entity['name']} ({entity.get('entity_type', 'UNKNOWN')})"
            for i, entity in enumerate(context['entities'])
        ])
    
    # Get all relationship types from context
    relationship_types = context.get('relationship_types', [rt.value for rt in RelationshipType])
    relationship_types_str = ', '.join(relationship_types)
    
    user_message = {
        "role": "user",
        "content": f"""
<CONTENT>
{context['episode_content']}
</CONTENT>

<ENTITIES>
{entities_list}
</ENTITIES>

<RELATIONSHIP_TYPES>
ONLY use these exact types ({len(relationship_types)} total):
{relationship_types_str}
</RELATIONSHIP_TYPES>

Extract all factual relationships between entities. Return ONLY the JSON response with no additional text, explanations, or formatting.

Requirements:
- Use exact entity names from the provided list
- Only extract relationships clearly supported by the text
- Include temporal information when available
- Focus on factual, meaningful connections
- MUST use only the relationship types listed above
- Return pure JSON only"""
    }
    
    return [system_message, user_message]


def create_relationship_extraction_schema() -> Dict[str, Any]:
    """
    Create JSON schema for structured output with relationship type enum
    
    Returns:
        JSON schema dictionary for OpenAI structured outputs
    """
    
    return {
        "type": "object",
        "properties": {
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_entity": {
                            "type": "string",
                            "description": "Name of the source entity (must match an entity from the provided list)"
                        },
                        "target_entity": {
                            "type": "string", 
                            "description": "Name of the target entity (must match an entity from the provided list)"
                        },
                        "relationship_type": {
                            "type": "string",
                            "enum": [rt.value for rt in RelationshipType if rt != RelationshipType.UNKNOWN],
                            "description": "Type of relationship - must be one of the predefined types"
                        },
                        "fact": {
                            "type": "string",
                            "description": "Human-readable description of the relationship"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence score between 0.0 and 1.0"
                        },
                        "valid_at": {
                            "type": ["string", "null"],
                            "description": "ISO 8601 timestamp when relationship became true"
                        },
                        "invalid_at": {
                            "type": ["string", "null"],
                            "description": "ISO 8601 timestamp when relationship stopped being true"
                        },
                        "attributes": {
                            "type": "object",
                            "description": "Additional properties of the relationship",
                            "additionalProperties": True
                        }
                    },
                    "required": ["source_entity", "target_entity", "relationship_type", "fact", "confidence"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["relationships"],
        "additionalProperties": False
    } 