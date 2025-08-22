"""
LLM-based schema extraction system.

This module provides LLM-powered extraction using document-type-specific schemas
for comprehensive and accurate knowledge extraction.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("python-dotenv not installed. Install with: pip install python-dotenv")

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI library not installed. Install with: pip install openai")

from extraction.document_schemas import DocumentType, DocumentSchema, get_schema
from extraction.document_classifier import DocumentClassifier
from parsing.enhanced_document_parser import EnhancedParsedDocument, ContentChunk

# Load environment variables from .env file
if DOTENV_AVAILABLE:
    load_dotenv()

logger = logging.getLogger(__name__)


class LLMSchemaExtractor:
    """
    LLM-powered extractor using document-type-specific schemas.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        """
        Initialize the LLM-based extractor.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY from .env or environment)
            model: Model to use for extraction (default: gpt-3.5-turbo)
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library required. Install with: pip install openai")
        
        # Try to get API key from: 1) parameter, 2) .env file, 3) environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY in .env file or environment")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.classifier = DocumentClassifier()
    
    def summarize_chunk_with_llm(
        self,
        content: str,
        chunk_id: Optional[str] = None,
        section_title: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Create a concise summary of a chunk's content.
        
        Args:
            content: Text content to summarize
            chunk_id: Optional chunk identifier
            section_title: Optional section title for context
            
        Returns:
            Tuple of (summary string, usage dict)
        """
        # Build context-aware prompt
        context = f"This chunk is from section: '{section_title}'\n\n" if section_title else ""
        
        prompt = f"""You are an expert at summarizing academic and technical content.

{context}TEXT TO SUMMARIZE:
{content[:4000]}

INSTRUCTIONS:
Create a concise, informative summary (2-3 sentences) that captures:
1. The main topic or concept being discussed
2. Key points, methods, or findings mentioned
3. Any important relationships or comparisons made

The summary should be self-contained and help someone quickly understand what this chunk discusses without reading the full text.

Focus on technical content and avoid generic descriptions."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise technical content summarizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=150  # Keep summaries concise
            )
            
            # Track usage for cost calculation
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            summary = response.choices[0].message.content.strip()
            
            return summary, usage
            
        except Exception as e:
            logger.error(f"Chunk summarization failed: {e}")
            return "Summary generation failed", {}
    
    def extract_entities_with_llm(
        self,
        content: str,
        schema: DocumentSchema,
        chunk_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract entities using LLM based on schema.
        
        Args:
            content: Text content to extract from
            schema: Document schema with entity types
            chunk_id: Optional chunk identifier
            
        Returns:
            Tuple of (List of extracted entities, usage dict)
        """
        entity_schema = schema.entity_schema
        
        # Create extraction prompt
        prompt = f"""You are an expert knowledge extractor specializing in {schema.description}.

Extract entities from the following text according to this schema:

ENTITY TYPES TO EXTRACT:
{', '.join(entity_schema.entity_types)}

PRIORITY ENTITIES (focus on these):
{', '.join(entity_schema.priority_entities)}

EXTRACTION HINTS:
{json.dumps(entity_schema.extraction_hints, indent=2)}

TEXT TO ANALYZE:
{content} 

INSTRUCTIONS:
1. Extract ALL relevant entities matching the specified types
2. Include confidence scores (0-1) for each entity
3. Extract key attributes when available
4. For the context field, provide analytical insights about the entity's role in this chunk

Return a JSON array of entities with this structure:
[
  {{
    "name": "entity name",
    "type": "ENTITY_TYPE",
    "description": "brief description of what the entity is",
    "confidence": 0.9,
    "attributes": {{"key": "value"}},
    "context": "analytical insight about this entity in the chunk"
  }}
]

IMPORTANT for the "context" field:
- DO NOT just quote text. Instead, explain the entity's role, purpose, or significance in this chunk
- Examples of good context across different domains:
  * TECHNOLOGY/TOOL: "Introduced as the primary framework for building the system, chosen for its scalability and 50% faster processing speed"
  * PERSON/ORGANIZATION: "Mentioned as the lead developer who contributed the core algorithm, affiliated with MIT"
  * PROCESS/METHOD: "Described as a three-stage pipeline that reduces processing time from hours to minutes"
  * PRODUCT/SERVICE: "Launched in Q2 2023 as a cloud-based solution targeting enterprise customers"
  * CONCEPT/THEORY: "Explained as the foundational principle underlying the new approach, contrasting with traditional methods"
  * LOCATION/FACILITY: "Identified as the manufacturing site where production increased by 30% after automation"
  * EVENT/MILESTONE: "Marked as the turning point when the company pivoted to AI-focused strategy"
  * STANDARD/REGULATION: "Referenced as the compliance requirement driving the system redesign"
- Include quantitative details if mentioned (percentages, metrics, dates, amounts)
- Mention relationships to other entities when relevant
- Explain what the chunk reveals about this entity (findings, comparisons, applications, impacts)
- Keep it concise but informative (1-2 sentences)

Focus on specific, technical, and factual information. Be comprehensive."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise knowledge extraction system."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Track usage for cost calculation
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            result = json.loads(response.choices[0].message.content)
            entities = result.get("entities", []) if "entities" in result else result
            
            # Ensure it's a list
            if isinstance(entities, dict):
                entities = [entities]
            
            # Add chunk_id if provided
            if chunk_id:
                for entity in entities:
                    entity["chunk_id"] = chunk_id
            
            return entities, usage
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return [], {}
    
    def extract_relationships_with_llm(
        self,
        content: str,
        entities: List[Dict[str, Any]],
        schema: DocumentSchema,
        chunk_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract relationships using LLM based on schema.
        
        Args:
            content: Text content
            entities: Previously extracted entities
            schema: Document schema with relationship types
            chunk_id: Optional chunk identifier
            
        Returns:
            Tuple of (List of extracted relationships, usage dict)
        """
        rel_schema = schema.relationship_schema
        
        # Create entity list for context
        entity_list = [f"- {e['name']} ({e['type']})" for e in entities[:20]]
        
        prompt = f"""You are an expert at identifying relationships in {schema.description}.

ENTITIES FOUND IN TEXT:
{chr(10).join(entity_list)}

RELATIONSHIP TYPES TO EXTRACT:
{', '.join(rel_schema.relationship_types)}

PRIORITY RELATIONSHIPS (focus on these):
{', '.join(rel_schema.priority_relationships)}

EXTRACTION HINTS:
{json.dumps(rel_schema.extraction_hints, indent=2)}

TEXT TO ANALYZE:
{content[:3000]}

INSTRUCTIONS:
1. Extract ALL relationships between the identified entities
2. For academic papers, focus on:
   - How methods/models improve upon or extend previous work
   - Performance comparisons (X outperforms Y)
   - What datasets or techniques are used
   - Dependencies and requirements
3. Include confidence scores and evidence

Return a JSON array of relationships:
[
  {{
    "source": "source entity name",
    "target": "target entity name",
    "type": "RELATIONSHIP_TYPE",
    "description": "relationship description",
    "confidence": 0.9,
    "evidence": "quote from text supporting this relationship"
  }}
]

Be comprehensive and precise."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise relationship extraction system."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Track usage for cost calculation
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            result = json.loads(response.choices[0].message.content)
            relationships = result.get("relationships", []) if "relationships" in result else result
            
            # Ensure it's a list
            if isinstance(relationships, dict):
                relationships = [relationships]
            
            # Add chunk_id if provided
            if chunk_id:
                for rel in relationships:
                    rel["chunk_id"] = chunk_id
            
            return relationships, usage
            
        except Exception as e:
            logger.error(f"LLM relationship extraction failed: {e}")
            return [], {}
    
    def extract_from_document(
        self,
        document_path,
        override_type: Optional[DocumentType] = None,
    ) -> Dict[str, Any]:
        """
        Extract knowledge from document using LLM and schema.
        
        Args:
            document_path: Path to document, parsed JSON, or EnhancedParsedDocument object
            override_type: Optional document type override
            max_chunks: Maximum chunks to process
            
        Returns:
            Extraction results with output_file, entity_count, relationship_count
        """
        # Handle different input types
        if isinstance(document_path, EnhancedParsedDocument):
            # Direct EnhancedParsedDocument object
            parsed_doc = document_path
            # Access content directly from parsed_doc, not from metadata
            content = parsed_doc.content
            chunks = parsed_doc.text_chunks
            doc_name = getattr(parsed_doc.metadata, "file_name", "document")
        elif isinstance(document_path, (str, Path)):
            document_path = Path(document_path)
            doc_name = document_path.name
            if document_path.suffix == ".json":
                with open(document_path, 'r') as f:
                    data = json.load(f)
                content = data.get("content", "")
                chunks = data.get("text_chunks", [])
            else:
                # Would need proper document parsing
                with open(document_path, 'r') as f:
                    content = f.read()
                chunks = []
        else:
            raise TypeError(f"Unsupported document type: {type(document_path)}")
        
        # Classify document
        if override_type:
            doc_type = override_type
            schema = get_schema(doc_type)
        else:
            doc_type, _, schema = self.classifier.classify(
                content[:5000] if content else (chunks[0].content[:5000] if chunks else ""),
                filename=doc_name
            )
        
        logger.info(f"Processing as {doc_type} with {schema.document_type} schema")
        
        all_entities = []
        all_relationships = []
        chunk_summaries = []  # Store summaries for each chunk
        self.last_usage = []  # Track API usage for cost tracking
        
        # Process chunks or full content
        if chunks:
            # Process each chunk
            for i, chunk in enumerate(chunks):
                # Handle ContentChunk objects from EnhancedParsedDocument
                if hasattr(chunk, 'content'):
                    chunk_content = chunk.content
                    chunk_id = chunk.id if hasattr(chunk, 'id') else f"chunk_{i}"
                    section_title = chunk.section_title if hasattr(chunk, 'section_title') else None
                    chunk_references = chunk.references if hasattr(chunk, 'references') else []
                elif isinstance(chunk, dict):
                    chunk_content = chunk.get("content", "")
                    chunk_id = chunk.get("id", f"chunk_{i}")
                    section_title = chunk.get("section_title", None)
                    chunk_references = chunk.get("references", [])
                else:
                    chunk_content = str(chunk)
                    chunk_id = f"chunk_{i}"
                    section_title = None
                    chunk_references = []
                
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Summarize chunk first
                summary, summary_usage = self.summarize_chunk_with_llm(
                    chunk_content,
                    chunk_id,
                    section_title
                )
                chunk_summaries.append({
                    "chunk_id": chunk_id,
                    "section_title": section_title,
                    "summary": summary,
                    "references": chunk_references  # Include references from parsing
                })
                if summary_usage:
                    self.last_usage.append(summary_usage)
                
                # Extract entities from chunk
                entities, entity_usage = self.extract_entities_with_llm(
                    chunk_content,
                    schema,
                    chunk_id
                )
                all_entities.extend(entities)
                if entity_usage:
                    self.last_usage.append(entity_usage)
                
                # Extract relationships from chunk
                relationships, rel_usage = self.extract_relationships_with_llm(
                    chunk_content,
                    entities,
                    schema,
                    chunk_id
                )
                all_relationships.extend(relationships)
                if rel_usage:
                    self.last_usage.append(rel_usage)
        else:
            # Process full content in sections
            sections = content.split('\n\n')
            for i, section in enumerate(sections):
                if len(section) > 100:  # Skip very short sections
                    entities, entity_usage = self.extract_entities_with_llm(
                        section,
                        schema,
                        f"section_{i}"
                    )
                    all_entities.extend(entities)
                    if entity_usage:
                        self.last_usage.append(entity_usage)
                    
                    relationships, rel_usage = self.extract_relationships_with_llm(
                        section,
                        entities,
                        schema,
                        f"section_{i}"
                    )
                    all_relationships.extend(relationships)
                    if rel_usage:
                        self.last_usage.append(rel_usage)
        
        # Deduplicate
        all_entities = self._deduplicate_entities(all_entities)
        all_relationships = self._deduplicate_relationships(all_relationships)
        
        # Prepare extraction results
        extraction_results = {
            "document_type": doc_type.value if hasattr(doc_type, 'value') else str(doc_type),
            "schema_used": schema.document_type,
            "entities": all_entities,
            "relationships": all_relationships,
            "chunk_summaries": chunk_summaries,  # Add summaries to results
            "statistics": {
                "total_entities": len(all_entities),
                "total_relationships": len(all_relationships),
                "entity_types": self._count_types(all_entities),
                "relationship_types": self._count_types(all_relationships),
                "chunks_summarized": len(chunk_summaries)
            },
            "extraction_metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "chunks_processed": min(len(chunks) if chunks else len(content.split('\n\n')))
            }
        }
        
        # Save extraction results to file if document was from EnhancedParsedDocument
        if isinstance(document_path, EnhancedParsedDocument):
            # Create output filename based on document metadata
            base_name = getattr(parsed_doc.metadata, "file_name", "document").replace(".pdf", "").replace(".json", "")
            # Try to get the original file path from metadata
            if hasattr(parsed_doc.metadata, "file_path") and parsed_doc.metadata.file_path:
                # Save next to the original PDF file
                original_path = Path(parsed_doc.metadata.file_path)
                output_path = original_path.parent / f"{base_name}.llm_extraction.json"
            else:
                # Fallback to current directory
                output_path = Path(f"./{base_name}.llm_extraction.json")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Extraction results saved to: {output_path}")
        else:
            output_path = str(document_path).replace(".json", ".llm_extraction.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Return in the format expected by the pipeline
        return {
            "output_file": str(output_path),
            "entity_count": len(all_entities),
            "relationship_count": len(all_relationships),
            "document_type": doc_type.value if hasattr(doc_type, 'value') else str(doc_type),
            "extraction_results": extraction_results  # Include full results for backward compatibility
        }
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate entities by name and type."""
        seen = {}
        unique = []
        
        for entity in entities:
            key = (entity.get("name", "").lower(), entity.get("type", ""))
            if key not in seen:
                seen[key] = entity
                unique.append(entity)
            else:
                # Merge confidence and attributes
                existing = seen[key]
                existing["confidence"] = max(
                    existing.get("confidence", 0),
                    entity.get("confidence", 0)
                )
                # Merge attributes
                if "attributes" in entity:
                    existing.setdefault("attributes", {}).update(entity["attributes"])
        
        return unique
    
    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate relationships."""
        seen = set()
        unique = []
        
        for rel in relationships:
            key = (
                rel.get("source", "").lower(),
                rel.get("target", "").lower(),
                rel.get("type", "")
            )
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        
        return unique
    
    def _count_types(self, items: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count occurrences of each type."""
        counts = {}
        for item in items:
            item_type = item.get("type", "UNKNOWN")
            counts[item_type] = counts.get(item_type, 0) + 1
        return counts


