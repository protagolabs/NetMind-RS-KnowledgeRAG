"""
Relationship Deduplication Prompt Template

Identifies duplicate relationships and merges them with conflict resolution.
Based on Graphiti's relationship deduplication approach.
"""

from typing import Dict, Any, List


def RELATIONSHIP_DEDUPLICATION_PROMPT(context: Dict[str, Any]) -> str:
    """
    Generate relationship deduplication prompt for a given context
    
    Args:
        context: Dictionary containing:
            - relationships: List of relationships to deduplicate
            - similarity_threshold: Threshold for considering relationships duplicates
    """
    
    system_prompt = """You are an expert relationship deduplication AI that identifies duplicate relationships and determines how to merge them.

Your task is to:
1. Identify relationships that represent the same factual connection between entities
2. Group duplicates together
3. Determine the best fact description for each group
4. Merge temporal information appropriately
5. Resolve conflicts between duplicate relationships

Consider relationships duplicates if they:
- Connect the same source and target entities
- Represent the same type of relationship
- Describe the same factual connection (even if worded differently)
- Have compatible temporal information

Handle temporal conflicts by:
- Using the most specific temporal information available
- Extending validity periods when appropriate
- Marking conflicts when temporal information is contradictory

Output your results as a valid JSON object."""

    relationships_text = ""
    if 'relationships' in context:
        relationships_text = "\n".join([
            f"Relationship {i+1}:\n"
            f"  Source: {rel.get('source_entity', 'N/A')}\n"
            f"  Target: {rel.get('target_entity', 'N/A')}\n"
            f"  Type: {rel.get('relationship_type', 'UNKNOWN')}\n"
            f"  Fact: {rel.get('fact', 'N/A')}\n"
            f"  Valid At: {rel.get('valid_at', 'N/A')}\n"
            f"  Invalid At: {rel.get('invalid_at', 'N/A')}\n"
            f"  Confidence: {rel.get('confidence', 'N/A')}\n"
            f"  Attributes: {rel.get('attributes', {})}\n"
            for i, rel in enumerate(context['relationships'])
        ])

    user_prompt = f"""
<RELATIONSHIPS_TO_DEDUPLICATE>
{relationships_text}
</RELATIONSHIPS_TO_DEDUPLICATE>

<SIMILARITY_THRESHOLD>
{context.get('similarity_threshold', 0.8)}
</SIMILARITY_THRESHOLD>

# TASK
Analyze the relationships and identify groups that represent the same factual connection. For each duplicate group, provide:

1. **duplicate_group**: List of relationship indices (1-based) that are duplicates
2. **merged_source_entity**: Source entity name (should be consistent across duplicates)
3. **merged_target_entity**: Target entity name (should be consistent across duplicates)
4. **merged_relationship_type**: The relationship type (should be consistent or most appropriate)
5. **merged_fact**: The best fact description or combination of facts
6. **merged_valid_at**: Best valid_at timestamp (earliest or most specific)
7. **merged_invalid_at**: Best invalid_at timestamp (latest or most specific)
8. **merged_confidence**: Highest confidence score from the group
9. **merged_attributes**: Combined attributes with conflict resolution
10. **confidence**: Float between 0.0-1.0 indicating confidence in the deduplication

# OUTPUT FORMAT
Return a JSON object with this structure:
```json
{{
  "duplicate_groups": [
    {{
      "duplicate_group": [1, 3, 5],
      "merged_source_entity": "Entity Name 1",
      "merged_target_entity": "Entity Name 2",
      "merged_relationship_type": "RELATIONSHIP_TYPE",
      "merged_fact": "Best or combined fact description",
      "merged_valid_at": "2024-01-01T00:00:00Z",
      "merged_invalid_at": null,
      "merged_confidence": 0.95,
      "merged_attributes": {{
        "key1": "resolved_value1",
        "key2": "resolved_value2"
      }},
      "confidence": 0.90
    }}
  ],
  "unique_relationships": [2, 4, 6]
}}
```

# GUIDELINES
- Only group relationships that clearly represent the same factual connection
- Source and target entities must match exactly for relationships to be duplicates
- Relationship types should be identical or very compatible
- When merging facts, prefer more detailed or specific descriptions
- For temporal information:
  - Use earliest valid_at if multiple valid timestamps exist
  - Use latest invalid_at if multiple invalid timestamps exist
  - Mark as conflicting if temporal information is contradictory
- Take the highest confidence score from the duplicate group
- For conflicting attributes, include both values with appropriate keys
- List relationships that have no duplicates in the "unique_relationships" array
- Use 1-based indexing to match the relationship numbering above
"""

    return f"System: {system_prompt}\n\nUser: {user_prompt}"


def RELATIONSHIP_DEDUPLICATION_STRUCTURED_PROMPT(context: Dict[str, Any]) -> list:
    """
    Generate structured messages for relationship deduplication (for better API compatibility)
    
    Returns:
        List of message dictionaries for OpenAI API
    """
    
    system_message = {
        "role": "system",
        "content": """You are an expert relationship deduplication AI. Identify relationships that represent the same factual connection and merge them appropriately.

Consider relationships duplicates when they:
- Connect identical source and target entities
- Have the same or compatible relationship types
- Describe the same factual connection (despite different wording)
- Have compatible temporal information

For duplicate groups:
- Use the best fact description
- Merge temporal information intelligently
- Take the highest confidence score
- Resolve attribute conflicts appropriately
- Handle temporal conflicts by extending validity periods when logical

Be conservative - only group relationships that clearly represent the same fact.

Return results as valid JSON only."""
    }
    
    relationships_text = ""
    if 'relationships' in context:
        relationships_text = "\n".join([
            f"{i+1}. {rel.get('source_entity', 'N/A')} -> {rel.get('target_entity', 'N/A')}\n"
            f"    Type: {rel.get('relationship_type', 'UNKNOWN')}\n"
            f"    Fact: {rel.get('fact', 'N/A')[:100]}...\n"
            f"    Temporal: {rel.get('valid_at', 'N/A')} to {rel.get('invalid_at', 'ongoing')}\n"
            for i, rel in enumerate(context['relationships'])
        ])
    
    user_message = {
        "role": "user",
        "content": f"""
<RELATIONSHIPS>
{relationships_text}
</RELATIONSHIPS>

Identify duplicate relationships and merge them. Return JSON in this exact format:

{{
  "duplicate_groups": [
    {{
      "duplicate_group": [1, 3],
      "merged_source_entity": "Source Name",
      "merged_target_entity": "Target Name", 
      "merged_relationship_type": "TYPE",
      "merged_fact": "Best fact description",
      "merged_valid_at": "2024-01-01T00:00:00Z",
      "merged_invalid_at": null,
      "merged_confidence": 0.95,
      "merged_attributes": {{"key": "value"}},
      "confidence": 0.90
    }}
  ],
  "unique_relationships": [2, 4, 5]
}}

Requirements:
- Use 1-based indexing for relationship references
- Only group relationships that clearly represent the same fact
- Entities must match exactly for relationships to be considered duplicates
- Merge temporal information logically
- Include all relationships in either duplicate_groups or unique_relationships"""
    }
    
    return [system_message, user_message] 