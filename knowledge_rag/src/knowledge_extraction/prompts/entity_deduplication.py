"""
Entity Deduplication Prompt Template

Identifies duplicate entities and merges them with conflict resolution.
Based on Graphiti's deduplication approach.
"""

from typing import Dict, Any, List


def ENTITY_DEDUPLICATION_PROMPT(context: Dict[str, Any]) -> str:
    """
    Generate entity deduplication prompt for a given context
    
    Args:
        context: Dictionary containing:
            - entities: List of entities to deduplicate
            - similarity_threshold: Threshold for considering entities duplicates
    """
    
    system_prompt = """You are an expert entity deduplication AI that identifies duplicate entities and determines how to merge them.

Your task is to:
1. Identify entities that likely refer to the same real-world entity
2. Group duplicates together 
3. Determine the best primary name for each group
4. Merge attributes and aliases appropriately
5. Resolve conflicts between duplicate entities

Consider entities duplicates if they:
- Have identical or very similar names (accounting for variations, abbreviations, etc.)
- Refer to the same real-world entity based on context and attributes
- Have compatible entity types and attributes
- Share similar aliases or references

Output your results as a valid JSON object."""

    entities_text = ""
    if 'entities' in context:
        entities_text = "\n".join([
            f"Entity {i+1}:\n"
            f"  Name: {entity.get('name', 'N/A')}\n"
            f"  Type: {entity.get('entity_type', 'UNKNOWN')}\n"
            f"  Aliases: {entity.get('aliases', [])}\n"
            f"  Summary: {entity.get('summary', 'N/A')}\n"
            f"  Attributes: {entity.get('attributes', {})}\n"
            for i, entity in enumerate(context['entities'])
        ])

    user_prompt = f"""
<ENTITIES_TO_DEDUPLICATE>
{entities_text}
</ENTITIES_TO_DEDUPLICATE>

<SIMILARITY_THRESHOLD>
{context.get('similarity_threshold', 0.8)}
</SIMILARITY_THRESHOLD>

# TASK
Analyze the entities and identify groups that refer to the same real-world entity. For each duplicate group, provide:

1. **duplicate_group**: List of entity indices (1-based) that are duplicates
2. **primary_name**: The best name to use for the merged entity
3. **merged_entity_type**: The most appropriate entity type for the group
4. **merged_aliases**: Combined list of all unique aliases
5. **merged_summary**: Best summary or combination of summaries
6. **merged_attributes**: Combined attributes with conflict resolution
7. **confidence**: Float between 0.0-1.0 indicating confidence in the deduplication

# OUTPUT FORMAT
Return a JSON object with this structure:
```json
{{
  "duplicate_groups": [
    {{
      "duplicate_group": [1, 3, 5],
      "primary_name": "Best Entity Name",
      "merged_entity_type": "ENTITY_TYPE",
      "merged_aliases": ["alias1", "alias2", "alias3"],
      "merged_summary": "Combined or best summary",
      "merged_attributes": {{
        "key1": "resolved_value1",
        "key2": "resolved_value2"
      }},
      "confidence": 0.95
    }}
  ],
  "unique_entities": [2, 4, 6]
}}
```

# GUIDELINES
- Be conservative - only group entities you're confident refer to the same real-world entity
- Consider variations in names (abbreviations, nicknames, formal vs informal)
- Entity types should be compatible (e.g., PERSON and PERSON, but not PERSON and ORGANIZATION)
- When merging attributes, prefer more specific/detailed information
- For conflicting attributes, include both values with appropriate keys
- The confidence score should reflect how certain you are about the grouping
- List entities that have no duplicates in the "unique_entities" array
- Use 1-based indexing to match the entity numbering above
"""

    return f"System: {system_prompt}\n\nUser: {user_prompt}"


def ENTITY_DEDUPLICATION_STRUCTURED_PROMPT(context: Dict[str, Any]) -> list:
    """
    Generate structured messages for entity deduplication (for better API compatibility)
    
    Returns:
        List of message dictionaries for OpenAI API
    """
    
    system_message = {
        "role": "system",
        "content": """You are an expert entity deduplication AI. Identify entities that refer to the same real-world entity and merge them appropriately.

Consider entities duplicates when they:
- Have similar names (accounting for variations, abbreviations)
- Refer to the same real-world entity based on context
- Have compatible types and attributes
- Share aliases or contextual information

For duplicate groups:
- Choose the best primary name
- Merge all unique aliases
- Combine summaries intelligently  
- Resolve attribute conflicts appropriately
- Provide confidence scores

Be conservative - only group entities you're highly confident about.

Return results as valid JSON only."""
    }
    
    entities_text = ""
    if 'entities' in context:
        entities_text = "\n".join([
            f"{i+1}. {entity.get('name', 'N/A')} ({entity.get('entity_type', 'UNKNOWN')})\n"
            f"   Aliases: {entity.get('aliases', [])}\n"
            f"   Summary: {entity.get('summary', 'N/A')[:100]}...\n"
            for i, entity in enumerate(context['entities'])
        ])
    
    user_message = {
        "role": "user",
        "content": f"""
<ENTITIES>
{entities_text}
</ENTITIES>

Identify duplicate entities and merge them. Return JSON in this exact format:

{{
  "duplicate_groups": [
    {{
      "duplicate_group": [1, 3],
      "primary_name": "Best Name",
      "merged_entity_type": "TYPE",
      "merged_aliases": ["all", "unique", "aliases"],
      "merged_summary": "Combined summary",
      "merged_attributes": {{"key": "value"}},
      "confidence": 0.95
    }}
  ],
  "unique_entities": [2, 4, 5]
}}

Requirements:
- Use 1-based indexing for entity references
- Only group entities that clearly refer to the same real-world entity
- Resolve conflicts intelligently when merging
- Include all entities in either duplicate_groups or unique_entities"""
    }
    
    return [system_message, user_message] 