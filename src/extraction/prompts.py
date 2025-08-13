"""LLM prompts for extraction and processing tasks."""


ENTITY_EXTRACTION_PROMPT = """Extract named entities from the following text. Focus on identifying:
- People (PERSON)
- Organizations (ORGANIZATION)
- Locations (LOCATION)
- Concepts/Ideas (CONCEPT)
- Events (EVENT)
- Products (PRODUCT)

For each entity, provide:
1. Name
2. Type
3. Brief description
4. Any relevant attributes

Text: {text}

Return the entities in JSON format:
[
    {
        "name": "entity name",
        "type": "entity type",
        "description": "brief description",
        "attributes": {}
    }
]
"""


RELATIONSHIP_EXTRACTION_PROMPT = """Given the following text and entities, identify relationships between them.

Text: {text}

Entities: {entities}

For each relationship, provide:
1. Source entity
2. Target entity
3. Relationship type
4. Description of the relationship
5. Any temporal information if mentioned

Return relationships in JSON format:
[
    {
        "source": "entity name",
        "target": "entity name",
        "type": "relationship type",
        "description": "relationship description",
        "temporal_info": "date/time if mentioned"
    }
]
"""


ENTITY_DEDUPLICATION_PROMPT = """Compare these two entities and determine if they refer to the same real-world entity:

Entity 1:
- Name: {entity1_name}
- Type: {entity1_type}
- Description: {entity1_description}
- Context: {entity1_context}

Entity 2:
- Name: {entity2_name}
- Type: {entity2_type}
- Description: {entity2_description}
- Context: {entity2_context}

Consider:
1. Name similarity (including aliases)
2. Type compatibility
3. Contextual overlap
4. Attribute matching

Return JSON:
{
    "same_entity": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation"
}
"""


TEMPORAL_EXTRACTION_PROMPT = """Extract temporal information from the following text and section context.

Text: {text}

Section information:
- Section title: {section_title}
- Parent sections: {parent_sections}
- Document metadata: {document_metadata}

Look for:
1. Explicit dates/times
2. Relative temporal references (e.g., "last year", "recently")
3. Sequential indicators (e.g., "before", "after", "during")
4. Contextual temporal cues

Return JSON:
{
    "temporal_info": "extracted date/time or null",
    "temporal_type": "explicit/relative/contextual",
    "confidence": 0.0-1.0
}
"""


RELATIONSHIP_CONFLICT_RESOLUTION_PROMPT = """Resolve conflict between relationships:

Existing relationship:
- Source: {existing_source}
- Target: {existing_target}
- Type: {existing_type}
- Description: {existing_description}
- Temporal: {existing_temporal}
- Document: {existing_document}

New relationship:
- Source: {new_source}
- Target: {new_target}
- Type: {new_type}
- Description: {new_description}
- Temporal: {new_temporal}
- Document: {new_document}

Determine the best action:
1. MERGE - Combine information from both
2. REPLACE - New relationship supersedes old
3. KEEP_BOTH - Both are valid in different contexts
4. DISCARD - New relationship is invalid

Return JSON:
{
    "action": "MERGE/REPLACE/KEEP_BOTH/DISCARD",
    "reasoning": "explanation",
    "merged_description": "if action is MERGE"
}
"""


COMMUNITY_SUMMARY_PROMPT = """Generate a concise summary for this community of related entities:

Entities in community:
{entities}

Key relationships:
{relationships}

Create a summary that:
1. Identifies the main theme/topic of the community
2. Highlights key entities and their roles
3. Describes important relationships
4. Is concise (2-3 sentences)

Summary:
"""