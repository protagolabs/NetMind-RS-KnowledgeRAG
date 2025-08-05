# Knowledge Extraction System

This module provides comprehensive entity and relationship extraction from processed episodes, building temporal knowledge graphs with automatic deduplication and conflict resolution.

## 🎯 Key Features

- **Bi-temporal Tracking**: Tracks both when facts became true in the real world (`valid_at`/`invalid_at`) and when they were added to the system (`created_at`/`expired_at`)
- **Automatic Deduplication**: Intelligent merging of duplicate entities and relationships using LLM-based analysis
- **Temporal Information Extraction**: Extracts temporal validity from natural language text
- **Confidence Scoring**: All extractions include confidence scores for quality assessment
- **Full Provenance Tracking**: Complete audit trail linking knowledge back to source episodes and documents
- **Conflict Resolution**: Automatic resolution of conflicting information with temporal reasoning

## 🏗️ Architecture

### Data Models (`models.py`)

- **`Entity`**: Represents entities with types, aliases, attributes, and temporal tracking
- **`Relationship`**: Bi-temporal relationships between entities with fact descriptions
- **`Document`**: Document layer linking episodes to their source files
- **`ExtractionResult`**: Complete results from the extraction process

### Extraction Pipeline (`extractor.py`)

1. **Entity Extraction**: Identifies and classifies entities from episode content
2. **Relationship Extraction**: Finds factual relationships between extracted entities
3. **Temporal Processing**: Extracts when relationships became valid/invalid
4. **Entity Deduplication**: Merges duplicate entities with existing knowledge base
5. **Relationship Deduplication**: Consolidates duplicate relationships
6. **Provenance Updates**: Links all knowledge back to source episodes and documents

### Prompt Templates (`prompts/`)

Modular prompt templates for each extraction task:
- **Entity Extraction**: Identifies and classifies entities
- **Relationship Extraction**: Finds relationships with temporal information
- **Entity Deduplication**: Merges duplicate entities intelligently
- **Relationship Deduplication**: Consolidates duplicate relationships
- **Temporal Extraction**: Extracts detailed temporal information

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install openai python-dotenv

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Basic Usage

```python
from src.knowledge_extraction import KnowledgeExtractor
from src.knowledge_extraction.models import Document
from src.file_management.models import Episode

# Initialize extractor
extractor = KnowledgeExtractor(
    model_name="gpt-4",
    temperature=0.1
)

# Extract knowledge from an episode
result = extractor.extract_from_episode(
    episode=episode,
    document=document,
    existing_entities=previous_entities,
    existing_relationships=previous_relationships
)

if result.success:
    print(f"Extracted {result.entities_extracted} entities")
    print(f"Extracted {result.relationships_extracted} relationships")
    print(f"Deduplicated {result.entities_deduplicated} entities")
```

### 3. Run Tests

```bash
# Test the complete knowledge extraction pipeline
python test_knowledge_extraction.py
```

## 📊 Supported Types

### Entity Types
- **PERSON**: Individuals, names, people
- **ORGANIZATION**: Companies, institutions, groups
- **LOCATION**: Places, addresses, geographical entities
- **CONCEPT**: Abstract ideas, methodologies, theories
- **EVENT**: Meetings, conferences, incidents
- **PRODUCT**: Software, hardware, tools
- **TECHNOLOGY**: Technical systems, frameworks
- **DOCUMENT**: Files, papers, specifications

### Relationship Types
- **Structural**: `PART_OF`, `CONTAINS`, `BELONGS_TO`
- **Social**: `WORKS_FOR`, `COLLABORATED_WITH`, `REPORTS_TO`
- **Temporal**: `PRECEDED_BY`, `FOLLOWED_BY`, `CONCURRENT_WITH`
- **Causal**: `CAUSED`, `RESULTED_IN`, `INFLUENCED`
- **Conceptual**: `SIMILAR_TO`, `RELATED_TO`, `OPPOSITE_OF`
- **Spatial**: `LOCATED_IN`, `NEAR`, `ADJACENT_TO`
- **Document**: `MENTIONED_IN`, `AUTHORED`, `REFERENCES`

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.1
OPENAI_MAX_TOKENS=4000
```

### Extractor Options

```python
extractor = KnowledgeExtractor(
    openai_api_key="your-key",      # API key (or from env)
    model_name="gpt-4",             # OpenAI model to use
    max_retries=3,                  # API retry attempts
    temperature=0.1                 # Generation temperature
)
```

## 📈 Performance & Statistics

The extractor tracks comprehensive statistics:

```python
stats = extractor.get_stats()
print(f"Episodes processed: {stats['episodes_processed']}")
print(f"API calls made: {stats['api_calls_made']}")
print(f"Total processing time: {stats['total_processing_time']:.2f}s")
```

## 🎯 Integration with File Management

The knowledge extraction system integrates seamlessly with the file management system:

1. **Episodes**: File manager creates episodes from documents
2. **Documents**: Document layer tracks episode provenance
3. **Knowledge**: Extractor builds knowledge graphs from episodes
4. **Deduplication**: Maintains consistency across document processing

## 🚧 Future Enhancements

- **Graph Database Storage**: Neo4j integration for persistent storage
- **Vector Embeddings**: Semantic similarity for improved deduplication
- **Community Detection**: Hierarchical organization of entities
- **Conflict Resolution**: Advanced temporal reasoning for contradictions
- **Performance Optimization**: Batch processing and caching
- **Custom Types**: User-defined entity and relationship types

## 🤝 Integration Points

- **File Management**: Consumes episodes from document parsing
- **Graph Storage**: Will integrate with Neo4j for persistence
- **Vector Search**: Will use embeddings for semantic retrieval
- **API Layer**: Exposes extraction capabilities via REST API

This knowledge extraction system forms the core intelligence layer of the Knowledge RAG system, transforming raw document content into structured, temporal knowledge graphs ready for intelligent retrieval and reasoning. 