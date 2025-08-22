"""
Modular extraction components for knowledge extraction pipeline.

This package provides separated, reusable components for document processing
and knowledge extraction, making the pipeline more maintainable and testable.
"""

from .document_loader import DocumentLoader, DocumentData, ChunkData
from .chunk_summarizer import ChunkSummarizer, ChunkSummary
from .entity_extractor import EntityExtractor, ExtractedEntity
from .relationship_extractor import RelationshipExtractor, ExtractedRelationship
from .deduplicator import Deduplicator, DeduplicationStats
from .result_formatter import ResultFormatter, ExtractionStatistics, ExtractionMetadata
from .extraction_pipeline import ExtractionPipeline

__all__ = [
    # Document loading
    'DocumentLoader',
    'DocumentData', 
    'ChunkData',
    
    # Chunk summarization
    'ChunkSummarizer',
    'ChunkSummary',
    
    # Entity extraction
    'EntityExtractor',
    'ExtractedEntity',
    
    # Relationship extraction
    'RelationshipExtractor',
    'ExtractedRelationship',
    
    # Deduplication
    'Deduplicator',
    'DeduplicationStats',
    
    # Result formatting
    'ResultFormatter',
    'ExtractionStatistics',
    'ExtractionMetadata',
    
    # Main pipeline
    'ExtractionPipeline',
]