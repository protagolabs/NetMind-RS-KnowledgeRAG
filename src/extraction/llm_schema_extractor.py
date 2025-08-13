"""
LLM-based schema extraction system.

This module provides LLM-powered extraction using document-type-specific schemas
for comprehensive and accurate knowledge extraction.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
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
    
    def extract_entities_with_llm(
        self,
        content: str,
        schema: DocumentSchema,
        chunk_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract entities using LLM based on schema.
        
        Args:
            content: Text content to extract from
            schema: Document schema with entity types
            chunk_id: Optional chunk identifier
            
        Returns:
            List of extracted entities
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
{content[:3000]}  # Limit to 3000 chars for API

INSTRUCTIONS:
1. Extract ALL relevant entities matching the specified types
2. For academic papers, focus on methodologies, algorithms, models, and contributions
3. Include confidence scores (0-1) for each entity
4. Extract key attributes when available

Return a JSON array of entities with this structure:
[
  {{
    "name": "entity name",
    "type": "ENTITY_TYPE",
    "description": "brief description",
    "confidence": 0.9,
    "attributes": {{"key": "value"}},
    "context": "relevant quote from text"
  }}
]

Focus on technical contributions, not generic concepts. Be comprehensive."""

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
            
            result = json.loads(response.choices[0].message.content)
            entities = result.get("entities", []) if "entities" in result else result
            
            # Ensure it's a list
            if isinstance(entities, dict):
                entities = [entities]
            
            # Add chunk_id if provided
            if chunk_id:
                for entity in entities:
                    entity["chunk_id"] = chunk_id
            
            return entities
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []
    
    def extract_relationships_with_llm(
        self,
        content: str,
        entities: List[Dict[str, Any]],
        schema: DocumentSchema,
        chunk_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships using LLM based on schema.
        
        Args:
            content: Text content
            entities: Previously extracted entities
            schema: Document schema with relationship types
            chunk_id: Optional chunk identifier
            
        Returns:
            List of extracted relationships
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
            
            result = json.loads(response.choices[0].message.content)
            relationships = result.get("relationships", []) if "relationships" in result else result
            
            # Ensure it's a list
            if isinstance(relationships, dict):
                relationships = [relationships]
            
            # Add chunk_id if provided
            if chunk_id:
                for rel in relationships:
                    rel["chunk_id"] = chunk_id
            
            return relationships
            
        except Exception as e:
            logger.error(f"LLM relationship extraction failed: {e}")
            return []
    
    def extract_from_document(
        self,
        document_path: Path,
        override_type: Optional[DocumentType] = None,
        max_chunks: int = 10
    ) -> Dict[str, Any]:
        """
        Extract knowledge from document using LLM and schema.
        
        Args:
            document_path: Path to document or parsed JSON
            override_type: Optional document type override
            max_chunks: Maximum chunks to process
            
        Returns:
            Extraction results
        """
        # Load document
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
        
        # Classify document
        if override_type:
            doc_type = override_type
            schema = get_schema(doc_type)
        else:
            doc_type, _, schema = self.classifier.classify(
                content[:5000],
                filename=document_path.name
            )
        
        logger.info(f"Processing as {doc_type} with {schema.document_type} schema")
        
        all_entities = []
        all_relationships = []
        
        # Process chunks or full content
        if chunks:
            # Process each chunk
            for i, chunk in enumerate(chunks[:max_chunks]):
                chunk_content = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
                chunk_id = chunk.get("id", f"chunk_{i}") if isinstance(chunk, dict) else f"chunk_{i}"
                
                logger.info(f"Processing chunk {i+1}/{min(len(chunks), max_chunks)}")
                
                # Extract entities from chunk
                entities = self.extract_entities_with_llm(
                    chunk_content,
                    schema,
                    chunk_id
                )
                all_entities.extend(entities)
                
                # Extract relationships from chunk
                relationships = self.extract_relationships_with_llm(
                    chunk_content,
                    entities,
                    schema,
                    chunk_id
                )
                all_relationships.extend(relationships)
        else:
            # Process full content in sections
            sections = content.split('\n\n')
            for i, section in enumerate(sections[:max_chunks]):
                if len(section) > 100:  # Skip very short sections
                    entities = self.extract_entities_with_llm(
                        section,
                        schema,
                        f"section_{i}"
                    )
                    all_entities.extend(entities)
                    
                    relationships = self.extract_relationships_with_llm(
                        section,
                        entities,
                        schema,
                        f"section_{i}"
                    )
                    all_relationships.extend(relationships)
        
        # Deduplicate
        all_entities = self._deduplicate_entities(all_entities)
        all_relationships = self._deduplicate_relationships(all_relationships)
        
        return {
            "document_type": doc_type,
            "schema_used": schema.document_type,
            "entities": all_entities,
            "relationships": all_relationships,
            "statistics": {
                "total_entities": len(all_entities),
                "total_relationships": len(all_relationships),
                "entity_types": self._count_types(all_entities),
                "relationship_types": self._count_types(all_relationships)
            },
            "extraction_metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "chunks_processed": min(len(chunks) if chunks else len(sections), max_chunks)
            }
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


def test_llm_extraction(model: str = "gpt-3.5-turbo", max_chunks: int = 5):
    """
    Test LLM-based extraction on Attention paper.
    
    Args:
        model: OpenAI model to use (default: gpt-3.5-turbo)
                Options: gpt-3.5-turbo, gpt-4, gpt-4-turbo-preview
        max_chunks: Maximum number of chunks to process (default: 5)
    """
    
    print("\n" + "=" * 80)
    print("LLM-BASED SCHEMA EXTRACTION TEST")
    print("=" * 80)
    
    # Check for API key (will be loaded from .env by load_dotenv)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nError: OPENAI_API_KEY not found")
        print("Please set your OpenAI API key in one of these ways:")
        print("1. Create a .env file with: OPENAI_API_KEY=your-key-here")
        print("2. Set environment variable: export OPENAI_API_KEY='your-key-here'")
        return
    
    print(f"\nUsing model: {model}")
    print(f"Processing {max_chunks} chunks")
    
    # Initialize extractor with specified model
    extractor = LLMSchemaExtractor(
        api_key=api_key,
        model=model,
        temperature=0.1
    )
    
    # Path to document
    doc_path = Path("/home/administrator/projects/NetMind-RS-KnowledgeRAG/paper_sets/paper_set_1/docs/Attention Is All You Need.enhanced.json")
    
    if not doc_path.exists():
        print(f"Error: {doc_path} not found")
        return
    
    print(f"\nExtracting from: {doc_path.name}")
    print(f"Using {model} for comprehensive extraction...")
    print(f"Processing first {max_chunks} chunks for demonstration...")
    
    # Extract with LLM
    results = extractor.extract_from_document(
        doc_path,
        override_type=DocumentType.ACADEMIC_PAPER,
        max_chunks=max_chunks  # Use configurable limit
    )
    
    # Display results
    print(f"\n{'='*40}")
    print("EXTRACTION RESULTS")
    print(f"{'='*40}")
    
    print(f"\nDocument Type: {results['document_type']}")
    print(f"Schema Used: {results['schema_used']}")
    
    print(f"\nEntities Extracted: {results['statistics']['total_entities']}")
    print("Entity Types:")
    for entity_type, count in results['statistics']['entity_types'].items():
        print(f"  • {entity_type}: {count}")
    
    print(f"\nRelationships Extracted: {results['statistics']['total_relationships']}")
    print("Relationship Types:")
    for rel_type, count in results['statistics']['relationship_types'].items():
        print(f"  • {rel_type}: {count}")
    
    # Show sample entities
    print(f"\n{'='*40}")
    print("SAMPLE ENTITIES")
    print(f"{'='*40}")
    
    for entity in results['entities'][:10]:
        print(f"\n[{entity['type']}] {entity['name']}")
        if entity.get('description'):
            print(f"  Description: {entity['description']}")
        if entity.get('confidence'):
            print(f"  Confidence: {entity['confidence']:.2f}")
        if entity.get('attributes'):
            print(f"  Attributes: {entity['attributes']}")
    
    # Show sample relationships
    print(f"\n{'='*40}")
    print("SAMPLE RELATIONSHIPS")
    print(f"{'='*40}")
    
    for rel in results['relationships'][:10]:
        print(f"\n{rel['source']} --[{rel['type']}]--> {rel['target']}")
        if rel.get('description'):
            print(f"  {rel['description']}")
        if rel.get('evidence'):
            print(f"  Evidence: {rel['evidence'][:100]}...")
    
    # Save results
    output_path = doc_path.parent / f"{doc_path.stem}_llm_extraction.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*40}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*40}")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    model = "gpt-3.5-turbo"  # Default model
    max_chunks = 100  # Default chunks
    
    if len(sys.argv) > 1:
        model = sys.argv[1]
    if len(sys.argv) > 2:
        max_chunks = int(sys.argv[2])
    
    # Show usage if help requested
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python llm_schema_extractor.py [model] [max_chunks]")
        print("  model: OpenAI model name (default: gpt-3.5-turbo)")
        print("         Options: gpt-3.5-turbo, gpt-4, gpt-4-turbo-preview")
        print("  max_chunks: Number of chunks to process (default: 5)")
        print("\nExamples:")
        print("  python llm_schema_extractor.py")
        print("  python llm_schema_extractor.py gpt-4")
        print("  python llm_schema_extractor.py gpt-3.5-turbo 10")
        sys.exit(0)
    
    test_llm_extraction(model=model, max_chunks=max_chunks)