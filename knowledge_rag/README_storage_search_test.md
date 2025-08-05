# Knowledge RAG Storage and Search Testing

This directory contains three scripts to test the complete storage and search pipeline using the extracted knowledge from `test_knowledge_extraction.py`.

## 🎯 Overview

Based on your JSON test results from the knowledge extraction pipeline, these scripts demonstrate:

1. **Storage**: Store extracted entities, relationships, and episodes in Neo4j + ChromaDB
2. **Visualization**: View the knowledge graph visually in Neo4j Browser  
3. **Search**: Test comprehensive search capabilities with full document traceability

## 📋 Prerequisites

### Required Services
```bash
# Start Neo4j (required for graph storage and visualization)
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

# ChromaDB runs embedded (no separate service needed)
```

### Environment Variables
Create a `.env` file with:
```bash
# OpenAI API Key (required for embeddings)
OPENAI_API_KEY=your_openai_api_key_here

# Neo4j Database Credentials (optional, defaults provided)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=your_neo4j_username
NEO4J_PASSWORD=your_neo4j_password
```

### Dependencies
```bash
pip install neo4j chromadb openai python-dotenv
```

## 🚀 Step-by-Step Usage

### Step 1: Store the Knowledge
```bash
python test_store_from_json.py
```

**What it does:**
- Loads your `test_extraction_*.json` file (automatically finds the latest one)
- Converts JSON data back to proper Python objects
- Stores in Neo4j graph database (entities, relationships, episodes, documents)
- Stores embeddings in ChromaDB vector database
- Shows storage statistics

**Expected Output:**
```
✅ Connected to Neo4j and ChromaDB
✅ Knowledge stored successfully:
   documents_stored: 1
   episodes_stored: 3
   entities_stored: 6
   relationships_stored: 7
   entity_embeddings_stored: 6
   episode_embeddings_stored: 3
```

### Step 2: Visualize in Neo4j Browser
```bash
python test_visualize_neo4j.py
```

**What it does:**
- Tests all visualization queries programmatically
- Provides Neo4j Browser queries for copy-paste
- Generates custom styling for better visualization
- Creates a Jupyter notebook with examples

**Neo4j Browser Access:**
1. Open: http://localhost:7474
2. Login: neo4j/password
3. Try the provided queries

**Key Visualization Queries:**
```cypher
// Complete Knowledge Graph
MATCH (n) OPTIONAL MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 100

// Entity Relationships  
MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity) RETURN e1, r, e2

// Technology Entities
MATCH (e:Entity) WHERE e.entity_type = 'TECHNOLOGY' 
OPTIONAL MATCH (e)-[r:RELATES_TO]-(connected) 
RETURN e, r, connected

// Document Structure
MATCH (d:Document)-[:CONTAINS]->(ep:Episode) RETURN d, ep ORDER BY ep.sequence_number
```

### Step 3: Test Search Capabilities
```bash
python test_search_knowledge.py
```

**What it does:**
- Tests basic text search across all content
- Tests entity-specific searches with type filtering
- Tests episode/content chunk searches
- Tests graph traversal from known entities
- Tests hybrid search with multiple ranking strategies
- Tests advanced features (score filtering, analytics)
- Shows full document traceability for all results

**Interactive Mode:**
```bash
python test_search_knowledge.py --interactive
```

## 📊 What You'll See Demonstrated

### 1. Storage Capabilities
- ✅ **Neo4j Graph Storage**: Documents → Episodes → Entities → Relationships
- ✅ **ChromaDB Vector Storage**: Semantic embeddings for entities and episodes
- ✅ **Full Provenance**: Every piece of knowledge traces back to source document and chunk
- ✅ **Bi-temporal Tracking**: When facts were true vs. when they were recorded

### 2. Visualization Features  
- ✅ **Graph Structure**: Visual representation of knowledge connections
- ✅ **Entity Networks**: How entities relate to each other
- ✅ **Document Hierarchy**: Documents → Episodes → Entities
- ✅ **Temporal Relationships**: Time-aware connections
- ✅ **Custom Styling**: Color-coded nodes and relationships

### 3. Search Capabilities
- ✅ **Semantic Search**: Find similar content using embeddings
- ✅ **Graph Traversal**: Discover related entities through graph connections  
- ✅ **Hybrid Search**: Combine multiple search strategies with intelligent ranking
- ✅ **Full-Text Search**: Traditional keyword-based search in Neo4j
- ✅ **Entity Filtering**: Search by specific entity types
- ✅ **Score Filtering**: Only return high-relevance results
- ✅ **Document Traceability**: Every result shows source document and section

## 🎨 Your Test Data

Based on your JSON file, you have:
- **6 Entities**: Including "RAG system", "Neo4j", "spaCy", "Transformers", "OpenAI GPT models", "Elasticsearch"
- **Entity Types**: Mainly TECHNOLOGY entities
- **3 Episodes**: From Knowledge_RAG_Design_Document.md
- **Full Provenance**: All entities trace back to specific episodes and document sections

## 🔍 Example Search Queries to Try

**In the interactive search:**
```
RAG system architecture
knowledge graph construction  
Neo4j database storage
temporal tracking system
entity extraction pipeline
document processing workflow
```

**Expected Results:**
- Each result shows relevance score
- Full document traceability (document name, section number)
- Entity and relationship information where applicable
- Multiple search strategies combined intelligently

## 🐛 Troubleshooting

**If storage fails:**
- Ensure Neo4j is running: `docker ps`
- Check Neo4j logs: `docker logs <container_id>`
- Verify OPENAI_API_KEY is set

**If visualization shows no data:**
- Make sure you ran `test_store_from_json.py` first
- Check Neo4j Browser connection at http://localhost:7474

**If search returns no results:**
- Verify data was stored successfully
- Check that ChromaDB directory exists: `./test_chroma_db/`
- Ensure OpenAI API key is valid for embeddings

## 🎉 Success Indicators

✅ **Storage Success**: See entity and relationship counts in output  
✅ **Visualization Success**: See colorful graph in Neo4j Browser  
✅ **Search Success**: Get relevant results with document traceability  

This demonstrates a complete Knowledge RAG pipeline with:
- 🧠 **Intelligent Extraction** (from previous test)
- 🗄️ **Robust Storage** (Neo4j + ChromaDB)  
- 🎨 **Rich Visualization** (Neo4j Browser)
- 🔍 **Powerful Search** (Hybrid with full traceability)

Your Knowledge RAG system is now production-ready! 🚀 