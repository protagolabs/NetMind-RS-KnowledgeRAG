"""
Entity Extraction Prompt Template

Extracts entities from text episodes with classification and confidence scoring.
Based on Graphiti's entity extraction approach.
"""

from typing import Dict, Any


def ENTITY_EXTRACTION_PROMPT(context: Dict[str, Any]) -> str:
    """
    Generate entity extraction prompt for a given context
    
    Args:
        context: Dictionary containing:
            - episode_content: The text content to extract entities from
            - entity_types: List of valid entity types
            - reference_time: ISO 8601 timestamp for temporal context
            - previous_episodes: Optional list of related episodes for context
    """
    
    system_prompt = """You are an expert entity extraction AI that identifies and classifies entities from text content. 

Your task is to:
1. Extract all significant entities (people, organizations, locations, concepts, etc.) mentioned in the text
2. Classify each entity with the most appropriate type
3. Provide a confidence score (0.0-1.0) for each extraction
4. Include aliases or alternative names if mentioned
5. Extract attributes and properties when available

Focus on entities that are:
- Explicitly mentioned in the text
- Central to the content's meaning
- Useful for building a knowledge graph
- Distinct and meaningful (avoid generic terms unless contextually important)

Output your results as a valid JSON object."""

    user_prompt = f"""
<EPISODE_CONTENT>
{context['episode_content']}
</EPISODE_CONTENT>

<ENTITY_TYPES>
{', '.join(context.get('entity_types', ['PERSON', 'ORGANIZATION', 'LOCATION', 'CONCEPT', 'EVENT', 'PRODUCT', 'TECHNOLOGY', 'DOCUMENT']))}
</ENTITY_TYPES>

<REFERENCE_TIME>
{context.get('reference_time', 'Not specified')}
</REFERENCE_TIME>

{f'''<PREVIOUS_EPISODES>
{context.get('previous_episodes', [])}
</PREVIOUS_EPISODES>''' if context.get('previous_episodes') else ''}

# TASK
Extract all significant entities from the EPISODE_CONTENT. For each entity, provide:

1. **name**: The primary name of the entity as mentioned in the text
2. **entity_type**: One of the provided ENTITY_TYPES (or closest match)
3. **confidence**: Float between 0.0-1.0 indicating extraction confidence
4. **aliases**: List of alternative names/references found in text (if any)
5. **summary**: Brief description of the entity (1-2 sentences)
6. **attributes**: Key-value pairs of additional properties mentioned

# OUTPUT FORMAT
Return a JSON object with this structure:
```json
{{
  "entities": [
    {{
      "name": "Entity Name",
      "entity_type": "ENTITY_TYPE",
      "confidence": 0.95,
      "aliases": ["Alternative Name", "Nickname"],
      "summary": "Brief description of what this entity is",
      "attributes": {{
        "key1": "value1",
        "key2": "value2"
      }}
    }}
  ]
}}
```

# GUIDELINES
- Be thorough but precise - extract all meaningful entities
- Use the exact entity type names provided, defaulting to CONCEPT if unclear
- Confidence should reflect how certain you are about the entity and its classification
- Include pronouns resolution (e.g., if "he" refers to "John Smith", extract "John Smith")
- For organizations, include any mentioned details (location, industry, size, etc.)
- For concepts, focus on abstract ideas, methodologies, theories, etc.
- For events, include temporal information in attributes when available
- Avoid overly generic terms unless they're contextually significant
"""

    return f"System: {system_prompt}\n\nUser: {user_prompt}"


# Alternative more structured prompt for better JSON parsing
def ENTITY_EXTRACTION_STRUCTURED_PROMPT(context: Dict[str, Any]) -> list:
    """
    Generate structured messages for entity extraction (for better API compatibility)
    
    Returns:
        List of message dictionaries for OpenAI API
    """
    
    system_message = {
        "role": "system", 
        "content": """You are an expert entity extraction AI. Extract meaningful entities from text with high precision.

Your task:
1. Identify important entities (people, organizations, concepts, etc.)
2. Classify each entity with the most appropriate type
3. Provide clear summaries and confidence scores
4. Extract aliases and relevant attributes

Focus on entities that are:
- Central to the content's meaning
- Factually mentioned or discussed
- Important for knowledge graph construction
- Clearly identifiable and specific

CRITICAL: You MUST respond with ONLY valid JSON in exactly this format - no other text, explanations, or formatting:

{
  "entities": [
    {
      "name": "Entity Name",
      "entity_type": "TYPE_FROM_LIST",
      "summary": "Brief description",
      "aliases": ["Alternative Name"],
      "confidence": 0.95,
      "attributes": {"key": "value"}
    }
  ]
}"""
    }
    
    entity_types_list = ', '.join(context.get('entity_types', [
        'PERSON', 'ORGANIZATION', 'LOCATION', 'CONCEPT', 'EVENT', 'PRODUCT', 'TECHNOLOGY'
    ]))
    
    user_message = {
        "role": "user",
        "content": f"""
<CONTENT>
{context['episode_content']}
</CONTENT>

<ENTITY_TYPES>
Available types: {entity_types_list}
</ENTITY_TYPES>

Extract entities from the content above. Return ONLY the JSON response with no additional text or explanations.

Requirements:
- Extract 3-10 most important entities
- Use exact entity type names from the list above
- Provide meaningful summaries (10-50 words)
- Set confidence scores based on how clearly the entity is mentioned
- Include aliases only if explicitly mentioned in text
- Focus on factual, verifiable entities"""
    }
    
    return [system_message, user_message] 