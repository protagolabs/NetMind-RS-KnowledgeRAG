"""
Temporal Extraction Prompt Template

Extracts temporal information from relationships and facts.
Determines when relationships became valid and when they ended.
"""

from typing import Dict, Any, List


def TEMPORAL_EXTRACTION_PROMPT(context: Dict[str, Any]) -> str:
    """
    Generate temporal extraction prompt for a given context
    
    Args:
        context: Dictionary containing:
            - episode_content: The text content containing temporal information
            - relationships: List of relationships to extract temporal info for
            - reference_time: ISO 8601 timestamp for temporal context
    """
    
    system_prompt = """You are an expert temporal information extraction AI that identifies when relationships and facts became true, changed, or ended.

Your task is to:
1. Analyze the text for temporal information related to each relationship
2. Determine when relationships became valid (valid_at timestamp)
3. Identify when relationships ended or became invalid (invalid_at timestamp)
4. Handle relative time expressions (e.g., "last year", "since 2020", "until recently")
5. Resolve temporal ambiguities using context and reference time

Focus on temporal expressions that indicate:
- Start times: "began in", "started", "since", "from", "as of"
- End times: "ended", "stopped", "until", "ceased", "no longer"
- Duration: "for X years", "during", "throughout"
- Change events: "became", "turned into", "replaced", "succeeded"

Output your results as a valid JSON object with ISO 8601 timestamps."""

    relationships_text = ""
    if 'relationships' in context:
        relationships_text = "\n".join([
            f"Relationship {i+1}:\n"
            f"  Source: {rel.get('source_entity', 'N/A')}\n"
            f"  Target: {rel.get('target_entity', 'N/A')}\n"
            f"  Type: {rel.get('relationship_type', 'UNKNOWN')}\n"
            f"  Fact: {rel.get('fact', 'N/A')}\n"
            for i, rel in enumerate(context['relationships'])
        ])

    user_prompt = f"""
<EPISODE_CONTENT>
{context['episode_content']}
</EPISODE_CONTENT>

<RELATIONSHIPS>
{relationships_text}
</RELATIONSHIPS>

<REFERENCE_TIME>
{context.get('reference_time', 'Not specified')} (Current time context for resolving relative dates)
</REFERENCE_TIME>

# TASK
For each relationship, extract temporal information from the EPISODE_CONTENT. Provide:

1. **relationship_index**: Index of the relationship (1-based)
2. **valid_at**: ISO 8601 timestamp when the relationship became true (if determinable)
3. **invalid_at**: ISO 8601 timestamp when the relationship stopped being true (if determinable)
4. **temporal_expressions**: List of text phrases that provided temporal information
5. **confidence**: Float between 0.0-1.0 indicating confidence in the temporal extraction
6. **notes**: Any relevant notes about the temporal reasoning

# OUTPUT FORMAT
Return a JSON object with this structure:
```json
{{
  "temporal_extractions": [
    {{
      "relationship_index": 1,
      "valid_at": "2024-01-01T00:00:00Z",
      "invalid_at": null,
      "temporal_expressions": ["began in January 2024", "as of 2024"],
      "confidence": 0.85,
      "notes": "Start date explicitly mentioned, no end date indicated"
    }},
    {{
      "relationship_index": 2,
      "valid_at": "2020-06-01T00:00:00Z",
      "invalid_at": "2023-12-31T23:59:59Z",
      "temporal_expressions": ["from June 2020", "until the end of 2023"],
      "confidence": 0.95,
      "notes": "Both start and end dates clearly specified"
    }}
  ]
}}
```

# TEMPORAL REASONING GUIDELINES

**Absolute Dates**:
- "January 2024" → "2024-01-01T00:00:00Z"
- "March 15, 2023" → "2023-03-15T00:00:00Z"
- "Q2 2022" → "2022-04-01T00:00:00Z"

**Relative Dates** (using reference_time):
- "last year" → Calculate based on reference_time
- "since 2020" → "2020-01-01T00:00:00Z" (valid_at)
- "recently" → Use reference_time or recent past
- "currently" → Ongoing (no invalid_at)

**Duration Expressions**:
– "for 5 years" → Calculate end date from start date
- "during 2022" → "2022-01-01T00:00:00Z" to "2022-12-31T23:59:59Z"
- "throughout the pandemic" → Use known pandemic timeframe

**Ambiguous Cases**:
- If temporal information is vague, use null values
- If only approximate timing is available, use best estimate with lower confidence
- If conflicting temporal information exists, note in "notes" field

**Special Cases**:
- "until now" → No invalid_at (ongoing)
- "former" → Set invalid_at before reference_time
- "future" → Set valid_at after reference_time
- "always" → No valid_at or invalid_at (eternal)

# GUIDELINES
- Be conservative - only extract temporal information you're confident about
- Use null values when temporal information is not available or unclear
- Include confidence scores reflecting certainty in temporal extraction
- Note any assumptions or reasoning in the notes field
- Use ISO 8601 format for all timestamps (UTC timezone)
- For partial dates, use the earliest possible time (e.g., "2024" → "2024-01-01T00:00:00Z")
"""

    return f"System: {system_prompt}\n\nUser: {user_prompt}"


def TEMPORAL_EXTRACTION_STRUCTURED_PROMPT(context: Dict[str, Any]) -> list:
    """
    Generate structured messages for temporal extraction (for better API compatibility)
    
    Returns:
        List of message dictionaries for OpenAI API
    """
    
    system_message = {
        "role": "system",
        "content": """You are an expert temporal information extraction AI. Extract when relationships became valid and when they ended based on text content.

Identify temporal expressions that indicate:
- Start times: "began", "started", "since", "from", "as of"
- End times: "ended", "stopped", "until", "ceased"
- Duration: "for X time", "during", "throughout"
- Change events: "became", "replaced", "succeeded"

Handle relative time expressions using the provided reference time.
Use ISO 8601 format for all timestamps.
Be conservative - only extract temporal information you're confident about.

Return results as valid JSON only."""
    }
    
    relationships_text = ""
    if 'relationships' in context:
        relationships_text = "\n".join([
            f"{i+1}. {rel.get('source_entity', 'N/A')} {rel.get('relationship_type', 'UNKNOWN')} {rel.get('target_entity', 'N/A')}\n"
            f"    Fact: {rel.get('fact', 'N/A')[:150]}...\n"
            for i, rel in enumerate(context['relationships'])
        ])
    
    user_message = {
        "role": "user",
        "content": f"""
<CONTENT>
{context['episode_content']}
</CONTENT>

<RELATIONSHIPS>
{relationships_text}
</RELATIONSHIPS>

<REFERENCE_TIME>
{context.get('reference_time', 'Not specified')}
</REFERENCE_TIME>

Extract temporal information for each relationship. Return JSON in this exact format:

{{
  "temporal_extractions": [
    {{
      "relationship_index": 1,
      "valid_at": "2024-01-01T00:00:00Z",
      "invalid_at": null,
      "temporal_expressions": ["phrase1", "phrase2"],
      "confidence": 0.85,
      "notes": "reasoning"
    }}
  ]
}}

Requirements:
- Use 1-based indexing for relationships
- Use ISO 8601 format for timestamps (UTC)
- Include null for unknown/unavailable temporal information
- Provide confidence scores and reasoning notes
- Extract all relevant temporal expressions from the text"""
    }
    
    return [system_message, user_message] 