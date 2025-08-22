"""
Result formatter module for preparing and saving extraction results.

This module handles formatting extraction results into standardized formats
and saving them to various output formats.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExtractionStatistics(BaseModel):
    """Statistics about the extraction results."""
    
    total_entities: int = Field(default=0, description="Total number of entities")
    total_relationships: int = Field(default=0, description="Total number of relationships")
    entity_types: Dict[str, int] = Field(default_factory=dict, description="Count by entity type")
    relationship_types: Dict[str, int] = Field(default_factory=dict, description="Count by relationship type")
    chunks_processed: int = Field(default=0, description="Number of chunks processed")
    chunks_summarized: int = Field(default=0, description="Number of chunks summarized")
    average_confidence: float = Field(default=0.0, description="Average confidence score")
    unique_entity_names: int = Field(default=0, description="Number of unique entity names")


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""
    
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    model: str = Field(default="gpt-3.5-turbo", description="LLM model used")
    document_type: str = Field(default="unknown", description="Type of document processed")
    schema_used: str = Field(default="generic", description="Schema used for extraction")
    processing_time: Optional[float] = Field(default=None, description="Total processing time in seconds")
    api_calls: int = Field(default=0, description="Number of API calls made")
    total_tokens: int = Field(default=0, description="Total tokens used")


class ResultFormatter:
    """
    Formats and saves extraction results in various formats.
    
    Handles preparation of extraction results, statistics calculation,
    and saving to different output formats (JSON, CSV, etc.).
    """
    
    def __init__(
        self,
        include_statistics: bool = True,
        include_metadata: bool = True,
        pretty_print: bool = True
    ):
        """
        Initialize the result formatter.
        
        Args:
            include_statistics: Whether to include statistics in results
            include_metadata: Whether to include metadata in results
            pretty_print: Whether to format JSON output for readability
        """
        self.include_statistics = include_statistics
        self.include_metadata = include_metadata
        self.pretty_print = pretty_print
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def prepare_extraction_results(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        summaries: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare extraction results in standardized format.
        
        Args:
            entities: List of extracted entities
            relationships: List of extracted relationships
            summaries: List of chunk summaries
            metadata: Optional metadata about extraction
            
        Returns:
            Dictionary with formatted extraction results
        """
        results = {
            "entities": entities,
            "relationships": relationships,
            "chunk_summaries": summaries
        }
        
        # Add statistics if enabled
        if self.include_statistics:
            results["statistics"] = self.calculate_statistics(
                entities, relationships, summaries
            ).model_dump()
        
        # Add metadata if enabled
        if self.include_metadata:
            if metadata:
                results["extraction_metadata"] = metadata
            else:
                results["extraction_metadata"] = ExtractionMetadata().model_dump()
        
        return results
    
    def calculate_statistics(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        summaries: Optional[List[Dict[str, Any]]] = None
    ) -> ExtractionStatistics:
        """
        Calculate statistics about extraction results.
        
        Args:
            entities: List of extracted entities
            relationships: List of extracted relationships
            summaries: Optional list of chunk summaries
            
        Returns:
            ExtractionStatistics object
        """
        stats = ExtractionStatistics()
        
        # Entity statistics
        stats.total_entities = len(entities)
        entity_names = set()
        confidence_sum = 0
        
        for entity in entities:
            entity_type = entity.get("type", "UNKNOWN")
            stats.entity_types[entity_type] = stats.entity_types.get(entity_type, 0) + 1
            entity_names.add(entity.get("name", "").lower())
            confidence_sum += entity.get("confidence", 0)
        
        stats.unique_entity_names = len(entity_names)
        
        # Relationship statistics
        stats.total_relationships = len(relationships)
        
        for rel in relationships:
            rel_type = rel.get("type", "UNKNOWN")
            stats.relationship_types[rel_type] = stats.relationship_types.get(rel_type, 0) + 1
            confidence_sum += rel.get("confidence", 0)
        
        # Average confidence
        total_items = stats.total_entities + stats.total_relationships
        if total_items > 0:
            stats.average_confidence = confidence_sum / total_items
        
        # Chunk statistics
        if summaries:
            stats.chunks_summarized = len(summaries)
            # Assume all summarized chunks were processed
            stats.chunks_processed = len(summaries)
        
        return stats
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
        format: str = "json"
    ) -> Path:
        """
        Save extraction results to file.
        
        Args:
            results: Extraction results dictionary
            output_path: Path for output file
            format: Output format ('json', 'jsonl', etc.)
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            self._save_as_json(results, output_path)
        elif format == "jsonl":
            self._save_as_jsonl(results, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def _save_as_json(self, results: Dict[str, Any], output_path: Path):
        """Save results as JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            if self.pretty_print:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            else:
                json.dump(results, f, ensure_ascii=False, default=str)
    
    def _save_as_jsonl(self, results: Dict[str, Any], output_path: Path):
        """Save results as JSONL file (one item per line)."""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write entities
            for entity in results.get("entities", []):
                f.write(json.dumps({"type": "entity", **entity}, ensure_ascii=False) + "\n")
            
            # Write relationships
            for rel in results.get("relationships", []):
                f.write(json.dumps({"type": "relationship", **rel}, ensure_ascii=False) + "\n")
            
            # Write summaries
            for summary in results.get("chunk_summaries", []):
                f.write(json.dumps({"type": "summary", **summary}, ensure_ascii=False) + "\n")
    
    def format_for_storage(
        self,
        results: Dict[str, Any],
        paper_path: str
    ) -> Dict[str, Any]:
        """
        Format results for storage in database.
        
        Args:
            results: Extraction results
            paper_path: Path to source paper
            
        Returns:
            Formatted dictionary for storage
        """
        return {
            "file_path": paper_path,
            "entities": results.get("entities", []),
            "relationships": results.get("relationships", []),
            "chunks": self._format_chunks_for_storage(
                results.get("chunk_summaries", [])
            ),
            "metadata": results.get("extraction_metadata", {}),
            "statistics": results.get("statistics", {})
        }
    
    def _format_chunks_for_storage(
        self,
        summaries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format chunk summaries for storage.
        
        Args:
            summaries: List of chunk summaries
            
        Returns:
            Formatted chunks for storage
        """
        chunks = []
        for i, summary in enumerate(summaries):
            chunk = {
                "chunk_id": summary.get("chunk_id", f"chunk_{i}"),
                "summary": summary.get("summary", ""),
                "section_title": summary.get("section_title", ""),
                "references": summary.get("references", []),
                "sequence": i
            }
            chunks.append(chunk)
        return chunks
    
    def create_summary_report(
        self,
        results: Dict[str, Any]
    ) -> str:
        """
        Create a human-readable summary report.
        
        Args:
            results: Extraction results
            
        Returns:
            Formatted summary report string
        """
        report = []
        report.append("=" * 60)
        report.append("EXTRACTION RESULTS SUMMARY")
        report.append("=" * 60)
        
        # Statistics
        if "statistics" in results:
            stats = results["statistics"]
            report.append(f"\nTotal Entities: {stats.get('total_entities', 0)}")
            report.append(f"Total Relationships: {stats.get('total_relationships', 0)}")
            report.append(f"Chunks Processed: {stats.get('chunks_processed', 0)}")
            report.append(f"Average Confidence: {stats.get('average_confidence', 0):.2f}")
            
            # Entity types
            if stats.get("entity_types"):
                report.append("\nEntity Types:")
                for entity_type, count in stats["entity_types"].items():
                    report.append(f"  • {entity_type}: {count}")
            
            # Relationship types
            if stats.get("relationship_types"):
                report.append("\nRelationship Types:")
                for rel_type, count in stats["relationship_types"].items():
                    report.append(f"  • {rel_type}: {count}")
        
        # Metadata
        if "extraction_metadata" in results:
            meta = results["extraction_metadata"]
            report.append(f"\nExtraction Details:")
            report.append(f"  Model: {meta.get('model', 'unknown')}")
            report.append(f"  Document Type: {meta.get('document_type', 'unknown')}")
            report.append(f"  Timestamp: {meta.get('timestamp', 'unknown')}")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)