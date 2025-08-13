"""
Test script for schema-based extraction on the Attention paper.
"""

import json
from pathlib import Path
import sys
sys.path.append('..')
from extraction.document_schemas import DocumentType, ACADEMIC_PAPER_SCHEMA
from extraction.document_classifier import DocumentClassifier
from extraction.schema_based_extractor import SchemaBasedExtractor
from parsing.enhanced_document_parser import EnhancedParsedDocument, ContentChunk, DocumentMetadata


def test_attention_paper():
    """Test schema-based extraction on the Attention Is All You Need paper."""
    
    print("=" * 80)
    print("SCHEMA-BASED KNOWLEDGE EXTRACTION TEST")
    print("Document: Attention Is All You Need")
    print("=" * 80)
    
    # Load the parsed JSON file
    json_path = Path("/home/administrator/projects/NetMind-RS-KnowledgeRAG/paper_sets/paper_set_1/docs/Attention Is All You Need.enhanced.json")
    
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        print("Please run simple_document_parser.py first")
        return
    
    # Load parsed content
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Step 1: Document Classification
    print("\n1. DOCUMENT CLASSIFICATION")
    print("-" * 40)
    
    classifier = DocumentClassifier()
    content_sample = data['content'][:5000]  # Use first 5000 chars for classification
    
    doc_type, confidence, schema = classifier.classify(
        content_sample,
        filename="Attention Is All You Need.pdf"
    )
    
    print(f"Document Type: {doc_type}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Schema: {schema.document_type}")
    
    # Show extraction focus for academic papers
    if doc_type == DocumentType.ACADEMIC_PAPER:
        print("\nExtraction Focus Areas:")
        for focus in schema.extraction_focus:
            print(f"  • {focus}")
    
    # Step 2: Entity Types for Academic Papers
    print("\n2. ENTITY TYPES TO EXTRACT")
    print("-" * 40)
    
    print("Priority Entities:")
    for entity_type in ACADEMIC_PAPER_SCHEMA.entity_schema.priority_entities:
        print(f"  • {entity_type}")
    
    print("\nAll Entity Types:")
    for entity_type in ACADEMIC_PAPER_SCHEMA.entity_schema.entity_types:
        hint = ACADEMIC_PAPER_SCHEMA.entity_schema.extraction_hints.get(entity_type, "")
        if hint:
            print(f"  • {entity_type}: {hint}")
        else:
            print(f"  • {entity_type}")
    
    # Step 3: Relationship Types for Academic Papers
    print("\n3. RELATIONSHIP TYPES TO EXTRACT")
    print("-" * 40)
    
    print("Priority Relationships:")
    for rel_type in ACADEMIC_PAPER_SCHEMA.relationship_schema.priority_relationships:
        print(f"  • {rel_type}")
    
    print("\nAll Relationship Types:")
    for rel_type in ACADEMIC_PAPER_SCHEMA.relationship_schema.relationship_types[:10]:  # Show first 10
        hint = ACADEMIC_PAPER_SCHEMA.relationship_schema.extraction_hints.get(rel_type, "")
        if hint:
            print(f"  • {rel_type}: {hint}")
        else:
            print(f"  • {rel_type}")
    
    # Step 4: Sample Extraction from Content
    print("\n4. SAMPLE EXTRACTION FROM CONTENT")
    print("-" * 40)
    
    # Simulate extraction from the abstract
    abstract_start = data['content'].find("## Abstract")
    abstract_end = data['content'].find("## 1 Introduction")
    
    if abstract_start != -1 and abstract_end != -1:
        abstract = data['content'][abstract_start:abstract_end]
        print("\nExtracting from Abstract:")
        print("-" * 20)
        
        # Look for key concepts
        key_concepts = []
        
        # Find the main contribution (Transformer)
        if "transformer" in abstract.lower():
            key_concepts.append({
                "name": "Transformer",
                "type": "MODEL",
                "description": "Novel neural network architecture based solely on attention mechanisms",
                "attributes": {
                    "novelty": "First transduction model relying entirely on self-attention",
                    "advantage": "More parallelizable and requires less training time"
                }
            })
        
        # Find methodology
        if "attention mechanism" in abstract.lower():
            key_concepts.append({
                "name": "Self-Attention Mechanism",
                "type": "METHODOLOGY",
                "description": "Attention mechanism relating different positions of a sequence",
                "attributes": {
                    "approach_type": "Attention-based",
                    "novelty": "Replaces recurrence and convolutions entirely"
                }
            })
        
        # Find results
        import re
        bleu_scores = re.findall(r"(\d+\.?\d*)\s*BLEU", abstract)
        if bleu_scores:
            key_concepts.append({
                "name": "BLEU Score Results",
                "type": "RESULT",
                "description": f"Achieved {bleu_scores[0]} BLEU on translation task",
                "attributes": {
                    "metric_values": bleu_scores,
                    "improvement": "Over 2 BLEU improvement over existing best results"
                }
            })
        
        print(f"Found {len(key_concepts)} key concepts:")
        for concept in key_concepts:
            print(f"\n  [{concept['type']}] {concept['name']}")
            print(f"    Description: {concept['description']}")
            if concept.get('attributes'):
                print(f"    Attributes: {concept['attributes']}")
    
    # Step 5: Extract Relationships
    print("\n5. SAMPLE RELATIONSHIP EXTRACTION")
    print("-" * 40)
    
    relationships = []
    
    # Find "improves upon" relationships
    content_lower = data['content'].lower()
    
    if "transformer" in content_lower and "rnn" in content_lower:
        relationships.append({
            "source": "Transformer",
            "target": "RNN",
            "type": "IMPROVES_UPON",
            "description": "Transformer improves upon RNN architectures",
            "evidence": "Allows for significantly more parallelization"
        })
    
    if "multi-head attention" in content_lower:
        relationships.append({
            "source": "Multi-Head Attention",
            "target": "Single Attention",
            "type": "EXTENDS",
            "description": "Multi-head attention extends single attention mechanism",
            "evidence": "Allows attending to information from different representation subspaces"
        })
    
    if "bleu" in content_lower and "state-of-the-art" in content_lower:
        relationships.append({
            "source": "Transformer",
            "target": "Previous State-of-the-art",
            "type": "OUTPERFORMS",
            "description": "Transformer outperforms previous state-of-the-art models",
            "evidence": "Achieves new state-of-the-art BLEU scores"
        })
    
    print(f"Found {len(relationships)} relationships:")
    for rel in relationships:
        print(f"\n  {rel['source']} --[{rel['type']}]--> {rel['target']}")
        print(f"    {rel['description']}")
        print(f"    Evidence: {rel['evidence']}")
    
    # Step 6: Tables as Special Entities
    print("\n6. TABLES AS KNOWLEDGE SOURCES")
    print("-" * 40)
    
    # Count tables in the document
    table_count = len(re.findall(r'\|[^\n]+\|\n\|[-:\s|]+\|', data['content']))
    print(f"Found {table_count} tables in the document")
    
    if table_count > 0:
        print("\nTables would be extracted as:")
        print("  • RESULT entities (containing performance metrics)")
        print("  • Sources for OUTPERFORMS relationships")
        print("  • Evidence for COMPARES_TO relationships")
    
    # Step 7: Summary
    print("\n7. EXTRACTION SUMMARY")
    print("-" * 40)
    
    print(f"""
For this academic paper, the schema-based extraction would focus on:

1. Key Contributions:
   - The Transformer architecture (MODEL)
   - Self-attention mechanism (METHODOLOGY)
   - Multi-head attention (ALGORITHM)

2. Performance Results:
   - BLEU scores (RESULT)
   - Training time comparisons (METRIC)
   - Model parameters (METRIC)

3. Relationships:
   - Transformer IMPROVES_UPON RNN/CNN architectures
   - Transformer OUTPERFORMS previous models
   - Multi-head attention EXTENDS self-attention
   - Transformer USES_DATASET WMT 2014

4. Limitations and Future Work:
   - Local attention for large inputs (LIMITATION)
   - Extension to other modalities (FUTURE_WORK)

This focused extraction ensures we capture the most relevant academic
contributions rather than generic entities like person names or locations.
""")


if __name__ == "__main__":
    test_attention_paper()