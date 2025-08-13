"""
Schema-based knowledge extraction system.

Simple one that not use LLm, only for testing.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

from extraction.document_schemas import DocumentType, DocumentSchema, get_schema
from extraction.document_classifier import DocumentClassifier
from parsing.enhanced_document_parser import EnhancedParsedDocument, ContentChunk


logger = logging.getLogger(__name__)


class SchemaBasedExtractor:
    """
    Enhanced extractor that uses document-type-specific schemas.
    
    This extractor:
    1. Classifies documents to determine their type
    2. Applies appropriate extraction schemas
    3. Focuses on domain-relevant entities and relationships
    4. Integrates with existing knowledge extraction pipeline
    """
    
    def __init__(
        self,
        use_llm_classification: bool = False,
        llm_model: str = "gpt-3.5-turbo",
        confidence_threshold: float = 0.3
    ):
        """
        Initialize the schema-based extractor.
        
        Args:
            use_llm_classification: Whether to use LLM for document classification
            llm_model: LLM model for extraction
            confidence_threshold: Minimum confidence for classification
        """
        self.classifier = DocumentClassifier(use_llm=use_llm_classification)
        self.llm_model = llm_model
        self.confidence_threshold = confidence_threshold
        
        # Track extraction statistics
        self.stats = {
            "documents_processed": 0,
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "schemas_used": {}
        }
    
    def extract_from_document(
        self,
        parsed_doc: EnhancedParsedDocument,
        override_type: Optional[DocumentType] = None
    ) -> Dict[str, Any]:
        """
        Extract knowledge from a parsed document using appropriate schema.
        
        Args:
            parsed_doc: Enhanced parsed document with chunks
            override_type: Optional document type override
            
        Returns:
            Dictionary containing extracted entities and relationships
        """
        # Get document content for classification
        content = parsed_doc.content
        filename = parsed_doc.metadata.file_name
        
        # Classify document or use override
        if override_type:
            doc_type = override_type
            confidence = 1.0
            schema = get_schema(doc_type)
            logger.info(f"Using override document type: {doc_type}")
        else:
            doc_type, confidence, schema = self.classifier.classify(
                content, filename
            )
            logger.info(f"Classified as {doc_type} with confidence {confidence:.2f}")
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            logger.warning(f"Low confidence ({confidence:.2f}), using general schema")
            schema = get_schema(DocumentType.GENERAL)
        
        # Update statistics
        self.stats["documents_processed"] += 1
        self.stats["schemas_used"][doc_type] = self.stats["schemas_used"].get(doc_type, 0) + 1
        
        # Extract entities and relationships using schema
        entities = self._extract_entities_with_schema(parsed_doc, schema)
        relationships = self._extract_relationships_with_schema(parsed_doc, schema, entities)
        
        # Handle special chunks (tables, images)
        special_entities = self._extract_from_special_chunks(parsed_doc, schema)
        entities.extend(special_entities)
        
        # Update statistics
        self.stats["entities_extracted"] += len(entities)
        self.stats["relationships_extracted"] += len(relationships)
        
        return {
            "document_type": doc_type,
            "classification_confidence": confidence,
            "schema_used": schema.document_type,
            "entities": entities,
            "relationships": relationships,
            "extraction_metadata": {
                "timestamp": datetime.now().isoformat(),
                "chunks_processed": len(parsed_doc.text_chunks),
                "tables_processed": len(parsed_doc.table_chunks),
                "focus_areas": schema.extraction_focus
            }
        }
    
    def _extract_entities_with_schema(
        self,
        parsed_doc: EnhancedParsedDocument,
        schema: DocumentSchema
    ) -> List[Dict[str, Any]]:
        """
        Extract entities according to schema specifications.
        
        Args:
            parsed_doc: Parsed document
            schema: Document schema with entity types
            
        Returns:
            List of extracted entities
        """
        entities = []
        entity_schema = schema.entity_schema
        
        # Process each text chunk
        for chunk in parsed_doc.text_chunks:
            chunk_entities = self._extract_entities_from_chunk(
                chunk,
                entity_schema.entity_types,
                entity_schema.priority_entities,
                entity_schema.extraction_hints
            )
            entities.extend(chunk_entities)
        
        # Deduplicate entities
        entities = self._deduplicate_entities(entities)
        
        # Add custom attributes based on schema
        for entity in entities:
            entity_type = entity.get("type")
            if entity_type in entity_schema.custom_attributes:
                entity["custom_attributes"] = entity_schema.custom_attributes[entity_type]
        
        return entities
    
    def _extract_entities_from_chunk(
        self,
        chunk: ContentChunk,
        entity_types: List[str],
        priority_types: List[str],
        hints: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from a single chunk.
        
        Args:
            chunk: Text chunk
            entity_types: Types of entities to extract
            priority_types: High-priority entity types
            hints: Extraction hints
            
        Returns:
            List of entities from chunk
        """
        # This is a simplified version - in production would use LLM
        entities = []
        content = chunk.content.lower()
        
        # Example extraction logic for academic papers
        if "METHODOLOGY" in entity_types:
            # Look for methodology indicators
            method_patterns = [
                r"we propose ([\w\s]+)",
                r"our approach ([\w\s]+)",
                r"we introduce ([\w\s]+)",
                r"novel ([\w\s]+) method",
            ]
            for pattern in method_patterns:
                import re
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        "name": match.strip(),
                        "type": "METHODOLOGY",
                        "chunk_id": chunk.id,
                        "confidence": 0.8,
                        "is_priority": "METHODOLOGY" in priority_types
                    })
        
        if "MODEL" in entity_types:
            # Look for model names (e.g., "Transformer", "BERT", etc.)
            model_patterns = [
                r"\b(transformer|attention|encoder|decoder)\b",
                r"\b([A-Z]{2,}(?:\-[A-Z]+)*)\b",  # Acronyms like BERT, GPT
            ]
            for pattern in model_patterns:
                import re
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if len(match) > 2:  # Filter out very short matches
                        entities.append({
                            "name": match,
                            "type": "MODEL",
                            "chunk_id": chunk.id,
                            "confidence": 0.7,
                            "is_priority": "MODEL" in priority_types
                        })
        
        return entities
    
    def _extract_relationships_with_schema(
        self,
        parsed_doc: EnhancedParsedDocument,
        schema: DocumentSchema,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships according to schema specifications.
        
        Args:
            parsed_doc: Parsed document
            schema: Document schema with relationship types
            entities: Extracted entities
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        rel_schema = schema.relationship_schema
        
        # Create entity lookup
        entity_lookup = {e["name"].lower(): e for e in entities}
        
        # Process chunks for relationships
        for chunk in parsed_doc.text_chunks:
            chunk_rels = self._extract_relationships_from_chunk(
                chunk,
                entity_lookup,
                rel_schema.relationship_types,
                rel_schema.priority_relationships,
                rel_schema.extraction_hints
            )
            relationships.extend(chunk_rels)
        
        # Apply temporal and causal focus if specified
        if rel_schema.temporal_focus:
            relationships = self._enhance_temporal_relationships(relationships)
        
        if rel_schema.causal_focus:
            relationships = self._enhance_causal_relationships(relationships)
        
        return relationships
    
    def _extract_relationships_from_chunk(
        self,
        chunk: ContentChunk,
        entity_lookup: Dict[str, Any],
        rel_types: List[str],
        priority_types: List[str],
        hints: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships from a chunk.
        
        Args:
            chunk: Text chunk
            entity_lookup: Dictionary of entities
            rel_types: Relationship types to extract
            priority_types: Priority relationship types
            hints: Extraction hints
            
        Returns:
            List of relationships
        """
        relationships = []
        content = chunk.content.lower()
        
        # Example extraction for academic papers
        if "IMPROVES_UPON" in rel_types:
            patterns = [
                r"([\w\s]+) improves upon ([\w\s]+)",
                r"([\w\s]+) outperforms ([\w\s]+)",
                r"([\w\s]+) achieves better .* than ([\w\s]+)",
            ]
            
            for pattern in patterns:
                import re
                matches = re.findall(pattern, content, re.IGNORECASE)
                for source, target in matches:
                    source = source.strip()
                    target = target.strip()
                    
                    # Check if entities exist
                    if source in entity_lookup or target in entity_lookup:
                        relationships.append({
                            "source": source,
                            "target": target,
                            "type": "IMPROVES_UPON",
                            "chunk_id": chunk.id,
                            "confidence": 0.75,
                            "is_priority": "IMPROVES_UPON" in priority_types
                        })
        
        return relationships
    
    def _extract_from_special_chunks(
        self,
        parsed_doc: EnhancedParsedDocument,
        schema: DocumentSchema
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from tables and images.
        
        Args:
            parsed_doc: Parsed document with special chunks
            schema: Document schema
            
        Returns:
            List of entities from special chunks
        """
        entities = []
        
        # Extract from tables
        for table_chunk in parsed_doc.table_chunks:
            # Tables often contain results, metrics, comparisons
            if schema.document_type == DocumentType.ACADEMIC_PAPER:
                # Look for performance metrics in tables
                entities.append({
                    "name": f"Table_{table_chunk.id}",
                    "type": "RESULT",
                    "source": "table",
                    "chunk_id": table_chunk.id,
                    "content_summary": table_chunk.caption or "Performance results",
                    "metadata": {
                        "rows": table_chunk.rows,
                        "columns": table_chunk.columns
                    }
                })
            elif schema.document_type == DocumentType.FINANCIAL_REPORT:
                # Financial tables contain metrics
                entities.append({
                    "name": f"Financial_Table_{table_chunk.id}",
                    "type": "FINANCIAL_METRIC",
                    "source": "table",
                    "chunk_id": table_chunk.id,
                    "content_summary": table_chunk.caption or "Financial data"
                })
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate entities.
        
        Args:
            entities: List of entities
            
        Returns:
            Deduplicated list
        """
        seen = set()
        unique = []
        
        for entity in entities:
            key = (entity["name"].lower(), entity["type"])
            if key not in seen:
                seen.add(key)
                unique.append(entity)
            else:
                # Merge confidence scores
                for existing in unique:
                    if (existing["name"].lower(), existing["type"]) == key:
                        existing["confidence"] = max(
                            existing.get("confidence", 0),
                            entity.get("confidence", 0)
                        )
                        break
        
        return unique
    
    def _enhance_temporal_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enhance relationships with temporal information.
        
        Args:
            relationships: List of relationships
            
        Returns:
            Enhanced relationships
        """
        # Add temporal markers where applicable
        for rel in relationships:
            # This would use more sophisticated temporal extraction
            rel["temporal_marker"] = "current"  # Placeholder
        
        return relationships
    
    def _enhance_causal_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enhance relationships with causal information.
        
        Args:
            relationships: List of relationships
            
        Returns:
            Enhanced relationships
        """
        # Add causal strength where applicable
        causal_types = ["IMPROVES_UPON", "ENABLES", "CAUSES", "RESULTS_IN"]
        
        for rel in relationships:
            if rel["type"] in causal_types:
                rel["causal_strength"] = "strong"  # Placeholder
        
        return relationships
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get extraction statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()


def extract_with_schema(
    document_path: Path,
    document_type: Optional[DocumentType] = None
) -> Dict[str, Any]:
    """
    Helper function to extract knowledge using schema.
    
    Args:
        document_path: Path to document
        document_type: Optional document type override
        
    Returns:
        Extraction results
    """
    # Load parsed document
    if document_path.suffix == ".json":
        with open(document_path, 'r') as f:
            doc_data = json.load(f)
            
        # Convert to EnhancedParsedDocument
        # This is simplified - would need proper conversion
        from enhanced_document_parser import DocumentMetadata
        
        parsed_doc = EnhancedParsedDocument(
            content=doc_data.get("content", ""),
            text_chunks=[],  # Would convert from doc_data
            table_chunks=[],
            image_chunks=[],
            metadata=DocumentMetadata(
                file_name=document_path.name,
                file_path=str(document_path),
                file_size=0,
                parse_time=datetime.now().isoformat()
            ),
            sections=[],
            chunk_index={}
        )
    else:
        # Parse document first
        from enhanced_document_parser import parse_pdf_enhanced
        parsed_doc = parse_pdf_enhanced(document_path)
    
    # Extract with schema
    extractor = SchemaBasedExtractor()
    results = extractor.extract_from_document(parsed_doc, document_type)
    
    return results