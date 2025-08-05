"""
Knowledge Extraction Prompts

Contains all prompt templates for different extraction tasks.
Each module provides both regular and structured prompt versions.
"""

from .entity_extraction import ENTITY_EXTRACTION_STRUCTURED_PROMPT
from .relationship_extraction import RELATIONSHIP_EXTRACTION_STRUCTURED_PROMPT, create_relationship_extraction_schema
from .entity_deduplication import ENTITY_DEDUPLICATION_STRUCTURED_PROMPT
from .relationship_deduplication import RELATIONSHIP_DEDUPLICATION_STRUCTURED_PROMPT
from .temporal_extraction import TEMPORAL_EXTRACTION_STRUCTURED_PROMPT

__all__ = [
    'ENTITY_EXTRACTION_STRUCTURED_PROMPT',
    'RELATIONSHIP_EXTRACTION_STRUCTURED_PROMPT',
    'ENTITY_DEDUPLICATION_STRUCTURED_PROMPT', 
    'RELATIONSHIP_DEDUPLICATION_STRUCTURED_PROMPT',
    'TEMPORAL_EXTRACTION_STRUCTURED_PROMPT',
    'create_relationship_extraction_schema'
] 