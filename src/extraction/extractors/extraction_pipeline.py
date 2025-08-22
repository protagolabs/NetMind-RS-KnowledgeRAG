"""
Extraction pipeline that orchestrates all modular components.

This module provides the main pipeline that coordinates document loading,
summarization, entity extraction, relationship extraction, deduplication,
and result formatting.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Import all modular components
from .document_loader import DocumentLoader, DocumentData
from .chunk_summarizer import ChunkSummarizer
from .entity_extractor import EntityExtractor
from .relationship_extractor import RelationshipExtractor
from .deduplicator import Deduplicator
from .result_formatter import ResultFormatter, ExtractionMetadata

# Import from parent modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from extraction.document_classifier import DocumentClassifier
from extraction.document_schemas import DocumentType, get_schema

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    Main orchestrator that coordinates all extraction components.
    
    This pipeline manages the complete extraction workflow from document
    loading through result formatting, using modular components for each step.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        use_summarization: bool = True,
        use_deduplication: bool = True,
        keep_all_contexts: bool = False,
        max_chunks: Optional[int] = None
    ):
        """
        Initialize the extraction pipeline with all components.
        
        Args:
            api_key: OpenAI API key
            model: LLM model to use
            temperature: Temperature for generation
            use_summarization: Whether to summarize chunks
            use_deduplication: Whether to deduplicate results
            keep_all_contexts: Whether to keep all contexts in deduplication
            max_chunks: Maximum number of chunks to process
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.use_summarization = use_summarization
        self.use_deduplication = use_deduplication
        self.max_chunks = max_chunks
        
        # Initialize components
        self.loader = DocumentLoader()
        self.classifier = DocumentClassifier()
        
        # Initialize API-dependent components only if API key provided
        if api_key:
            self.summarizer = ChunkSummarizer(
                api_key=api_key,
                model=model,
                temperature=temperature
            ) if use_summarization else None
            
            self.entity_extractor = EntityExtractor(
                api_key=api_key,
                model=model,
                temperature=temperature
            )
            
            self.relationship_extractor = RelationshipExtractor(
                api_key=api_key,
                model=model,
                temperature=temperature
            )
        else:
            self.summarizer = None
            self.entity_extractor = None
            self.relationship_extractor = None
        
        self.deduplicator = Deduplicator(
            keep_all_contexts=keep_all_contexts
        ) if use_deduplication else None
        
        self.formatter = ResultFormatter()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_usage = []  # Track API usage
    
    def extract_from_document(
        self,
        document_path: Union[str, Path, Any],
        override_type: Optional[DocumentType] = None,
        save_results: bool = True,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Extract knowledge from a document using all pipeline components.
        
        Args:
            document_path: Path to document or parsed document object
            override_type: Optional document type override
            save_results: Whether to save results to file
            output_path: Optional output path (auto-generated if not provided)
            
        Returns:
            Dictionary with extraction results
        """
        start_time = time.time()
        self.last_usage = []
        
        # Step 1: Load document
        self.logger.info("Loading document...")
        doc_data = self.loader.load_document(document_path)
        
        # Step 2: Classify document and get schema
        self.logger.info("Classifying document...")
        if override_type:
            doc_type = override_type
            schema = get_schema(doc_type)
        else:
            # Use first chunk or content for classification
            sample_text = doc_data.chunks[0].content if doc_data.chunks else doc_data.content[:5000]
            doc_type, _, schema = self.classifier.classify(sample_text, doc_data.doc_name)
        
        self.logger.info(f"Processing as {doc_type} with {schema.document_type} schema")
        
        # Limit chunks if specified
        chunks_to_process = doc_data.chunks
        if self.max_chunks and len(chunks_to_process) > self.max_chunks:
            chunks_to_process = chunks_to_process[:self.max_chunks]
            self.logger.info(f"Limiting to {self.max_chunks} chunks")
        
        # Convert chunks to dict format for processing
        chunk_dicts = [
            {
                "content": chunk.content,
                "chunk_id": chunk.chunk_id,
                "section_title": chunk.section_title,
                "references": chunk.references
            }
            for chunk in chunks_to_process
        ]
        
        # Step 3: Summarize chunks (optional)
        summaries = []
        if self.use_summarization and self.summarizer:
            self.logger.info("Summarizing chunks...")
            summary_objects = self.summarizer.batch_summarize_chunks(chunk_dicts)
            summaries = [s.model_dump() for s in summary_objects]
            
            # Track usage
            for s in summary_objects:
                if s.usage:
                    self.last_usage.append(s.usage)
        
        # Step 4: Extract entities
        self.logger.info("Extracting entities...")
        if not self.entity_extractor:
            raise ValueError("Entity extractor not initialized. Provide API key.")
        
        entity_objects, entity_usage = self.entity_extractor.batch_extract_entities(
            chunk_dicts, schema
        )
        entities = [e.model_dump() for e in entity_objects]
        self.last_usage.extend(entity_usage)
        
        # Step 5: Extract relationships
        self.logger.info("Extracting relationships...")
        if not self.relationship_extractor:
            raise ValueError("Relationship extractor not initialized. Provide API key.")
        
        # Group entities by chunk for relationship extraction
        entities_per_chunk = []
        for chunk_dict in chunk_dicts:
            chunk_entities = [e for e in entities if e.get("chunk_id") == chunk_dict["chunk_id"]]
            entities_per_chunk.append(chunk_entities)
        
        rel_objects, rel_usage = self.relationship_extractor.batch_extract_relationships(
            chunk_dicts, entities_per_chunk, schema
        )
        relationships = [r.model_dump() for r in rel_objects]
        self.last_usage.extend(rel_usage)
        
        # Step 6: Deduplicate (optional)
        if self.use_deduplication and self.deduplicator:
            self.logger.info("Deduplicating entities and relationships...")
            entities, entity_stats = self.deduplicator.deduplicate_entities(entities)
            relationships, rel_stats = self.deduplicator.deduplicate_relationships(relationships)
            
            self.logger.info(f"Deduplication: {entity_stats.duplicates_removed} duplicate entities, "
                           f"{rel_stats.duplicates_removed} duplicate relationships removed")
        
        # Step 7: Prepare results
        self.logger.info("Formatting results...")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Calculate total API usage
        total_usage = self._calculate_total_usage()
        
        # Create metadata
        metadata = ExtractionMetadata(
            model=self.model,
            document_type=doc_type.value if hasattr(doc_type, 'value') else str(doc_type),
            schema_used=schema.document_type,
            processing_time=processing_time,
            api_calls=len(self.last_usage),
            total_tokens=total_usage.get("total_tokens", 0)
        ).model_dump()
        
        # Format results
        results = self.formatter.prepare_extraction_results(
            entities=entities,
            relationships=relationships,
            summaries=summaries,
            metadata=metadata
        )
        
        # Step 8: Save results (optional)
        if save_results:
            if not output_path:
                # Auto-generate output path
                if isinstance(document_path, (str, Path)):
                    base_path = Path(document_path).with_suffix("")
                    output_path = Path(f"{base_path}.extraction.json")
                else:
                    output_path = Path("extraction_results.json")
            
            self.formatter.save_results(results, output_path)
            results["output_file"] = str(output_path)
        
        # Add summary statistics
        results["summary"] = {
            "entities_extracted": len(entities),
            "relationships_extracted": len(relationships),
            "chunks_processed": len(chunks_to_process),
            "processing_time": f"{processing_time:.2f}s",
            "total_api_calls": len(self.last_usage),
            "total_tokens": total_usage.get("total_tokens", 0)
        }
        
        self.logger.info(f"Extraction complete: {len(entities)} entities, "
                        f"{len(relationships)} relationships in {processing_time:.2f}s")
        
        return results
    
    def _calculate_total_usage(self) -> Dict[str, int]:
        """Calculate total API usage from all components."""
        total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        for usage in self.last_usage:
            total["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total["completion_tokens"] += usage.get("completion_tokens", 0)
            total["total_tokens"] += usage.get("total_tokens", 0)
        
        return total
    
    def get_cost_estimate(self) -> float:
        """
        Estimate API costs based on usage.
        
        Returns:
            Estimated cost in USD
        """
        usage = self._calculate_total_usage()
        
        # Rough cost estimates (update based on current pricing)
        cost_per_1k = {
            "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
            "gpt-4": {"prompt": 0.03, "completion": 0.06}
        }
        
        model_costs = cost_per_1k.get(self.model, cost_per_1k["gpt-3.5-turbo"])
        
        prompt_cost = (usage["prompt_tokens"] / 1000) * model_costs["prompt"]
        completion_cost = (usage["completion_tokens"] / 1000) * model_costs["completion"]
        
        return prompt_cost + completion_cost