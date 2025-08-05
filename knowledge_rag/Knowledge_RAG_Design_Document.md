# Knowledge RAG System - Architecture Design Document

## 1. Overview

### 1.1 Project Mission
Create an advanced RAG (Retrieval-Augmented Generation) system that accepts diverse file formats from users, constructs temporal knowledge graphs, and provides intelligent retrieval capabilities with dynamic knowledge evolution over time.

### 1.2 Core Capabilities
- **Multi-format Document Processing**: Support for txt, doc, pdf, pptx, xlsx, markdown, json, and more
- **Temporal Knowledge Graph Construction**: Build dynamic, time-aware knowledge graphs that evolve with new information
- **Intelligent Entity & Relationship Extraction**: Advanced NLP-based extraction with deduplication and conflict resolution
- **Hierarchical Community Detection**: Organize knowledge into meaningful communities for better retrieval
- **Hybrid Search Architecture**: Combine semantic, keyword, graph traversal, and temporal search methods
- **CRUD Operations**: Full lifecycle management of documents and knowledge
- **Incremental Learning**: Continuous knowledge graph updates without full reprocessing

### 1.3 System Architecture Inspiration
This design draws from:
- **Graphiti**: Temporal knowledge graphs with bi-temporal tracking and episodic processing
- **GraphRAG**: Hierarchical community detection and structured retrieval
- **Hybrid RAG**: Multi-modal search and retrieval strategies

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File Input    │    │   Processing    │    │    Storage      │
│   Management    │───▶│   Pipeline      │───▶│   & Graph DB    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        ▲
                                ▼                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Query & UI    │◄───│   Retrieval     │◄───│   Knowledge     │
│   Interface     │    │   Engine        │    │   Graph         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2.2 Technology Stack
- **Graph Database**: Neo4j / FalkorDB for knowledge graph storage
- **Vector Store**: ChromaDB / Weaviate for semantic embeddings
- **Document Processing**: Unstructured, PyMuPDF, python-docx
- **NLP Framework**: spaCy, Transformers, OpenAI GPT models
- **Search Engine**: Elasticsearch for full-text search
- **Backend**: FastAPI with async processing
- **Frontend**: React/Next.js for user interface

---

## 3. Core Components

### 3.1 Document Parsing Engine

#### 3.1.1 Multi-Format Parser
**Purpose**: Convert various document formats into structured, analyzable content

**Supported Formats**:
- **Text**: `.txt`, `.md`, `.csv`
- **Documents**: `.doc`, `.docx`, `.pdf`, `.rtf`
- **Presentations**: `.ppt`, `.pptx`
- **Spreadsheets**: `.xls`, `.xlsx`
- **Structured Data**: `.json`, `.xml`, `.yaml`
- **Web**: `.html`, `.htm`

**Input**: 
- Binary file content
- File metadata (name, type, size, upload timestamp)
- User context (uploader ID, project/namespace)

**Output**:
- **Episodic Documents**: Structured content chunks with metadata
- **Document Hierarchy**: Section/page-based organization
- **Temporal Context**: Creation and modification timestamps
- **Content Metadata**: Language, encoding, structure information

#### 3.1.2 Parsing Rules by Format

**PDF Processing**:
```python
class PDFParser:
    def parse(self, file_content, metadata):
        pages = []
        for page_num, page in enumerate(extract_pages(file_content)):
            # Extract text, images, tables
            content = {
                'page_number': page_num + 1,
                'text_content': self.extract_text(page),
                'images': self.extract_images(page),
                'tables': self.extract_tables(page),
                'layout_info': self.analyze_layout(page)
            }
            # Handle cross-page elements
            if self.spans_multiple_pages(content):
                content['continuation_marker'] = True
            pages.append(content)
        return self.create_episodes(pages, metadata)
```

**Document Episodic Structure**:
Each processed document creates episodes with:
- **Episode ID**: Unique identifier
- **Content**: Parsed text/structured data  
- **Temporal Data**: Creation/modification timestamps
- **Source Metadata**: Original file info, page/section references
- **Relationship Hints**: Cross-references, citations, mentions

### 3.2 Entity and Relationship Extraction Engine

#### 3.2.1 Multi-Stage Extraction Pipeline
**Architecture**: Cascaded extraction with validation and refinement

**Stage 1: Named Entity Recognition (NER)**
```python
class EntityExtractor:
    def extract_entities(self, episode):
        # Primary NER using spaCy + custom models
        entities = self.ner_pipeline(episode.content)
        
        # Domain-specific entity detection
        custom_entities = self.custom_ner(episode.content, episode.context)
        
        # Entity linking and normalization
        linked_entities = self.link_entities(entities + custom_entities)
        
        return self.deduplicate_entities(linked_entities)
```

**Stage 2: Relationship Extraction**
```python
class RelationshipExtractor:
    def extract_relationships(self, entities, episode):
        # Pattern-based extraction
        pattern_relations = self.pattern_extractor(entities, episode.content)
        
        # LLM-based extraction for complex relationships
        llm_relations = self.llm_extractor(entities, episode.content)
        
        # Temporal relationship detection
        temporal_relations = self.temporal_extractor(
            entities, episode.content, episode.timestamp
        )
        
        return self.validate_relationships(
            pattern_relations + llm_relations + temporal_relations
        )
```

#### 3.2.2 Deduplication and Conflict Resolution

**Entity Deduplication Strategy**:
1. **Exact Match**: Identical string matching
2. **Fuzzy Match**: Levenshtein distance < 0.8
3. **Semantic Match**: Embedding cosine similarity > 0.9
4. **Contextual Match**: Same entity type + similar context

**Relationship Conflict Resolution**:
- **Temporal Priority**: Newer information takes precedence
- **Source Authority**: Weighted by document credibility
- **Consistency Checking**: Cross-validate with existing knowledge
- **Human Review**: Flag contradictions for manual resolution

#### 3.2.3 Dynamic Update Mechanism
**Bi-Temporal Tracking** (inspired by Graphiti):
```python
class TemporalEdge:
    def __init__(self):
        # Real-world time
        self.valid_at: Optional[datetime] = None      # When relation became true
        self.invalid_at: Optional[datetime] = None    # When relation stopped being true
        
        # System time  
        self.created_at: datetime = now()             # When added to system
        self.expired_at: Optional[datetime] = None    # When marked as expired
        
        self.fact: str = ""                           # Human-readable fact
        self.confidence: float = 0.0                  # Extraction confidence
        self.source_episodes: List[str] = []          # Provenance tracking
```

**Update Process**:
1. **New Information Detection**: Compare with existing facts
2. **Temporal Resolution**: Determine timeline of changes
3. **Edge Invalidation**: Mark conflicting relationships as expired
4. **Fact Regeneration**: Update human-readable descriptions
5. **Provenance Tracking**: Maintain audit trail of changes

### 3.3 Community Detection and Hierarchical Organization

#### 3.3.1 Multi-Level Community Structure
**Inspired by GraphRAG's hierarchical approach**

**Level 0**: Individual entities and direct relationships
**Level 1**: Local communities (3-15 entities with strong connections)
**Level 2**: Regional communities (15-50 entities with thematic coherence)
**Level 3**: Global communities (50+ entities representing major topics)

#### 3.3.2 Community Detection Algorithm
```python
class CommunityDetector:
    def detect_communities(self, graph):
        # Use Leiden algorithm for optimal community detection
        communities = leiden_algorithm(
            graph, 
            resolution_parameter=1.0,
            random_state=42
        )
        
        # Hierarchical clustering for multi-level structure
        hierarchical_communities = self.build_hierarchy(communities)
        
        # Generate community summaries
        for community in hierarchical_communities:
            community.summary = self.generate_summary(community)
            community.key_entities = self.identify_key_entities(community)
            community.relationships = self.extract_relationships(community)
            
        return hierarchical_communities
```

#### 3.3.3 Community Summary Generation
**Multi-Perspective Summaries**:
- **Factual Summary**: Key facts and relationships within community
- **Temporal Summary**: How the community evolved over time
- **Contextual Summary**: Community's role in broader knowledge graph
- **Query-Focused**: Summaries optimized for common query patterns

### 3.4 Hybrid Retrieval Engine

#### 3.4.1 Multi-Modal Search Architecture
```python
class HybridRetriever:
    def search(self, query, context=None, temporal_filter=None):
        # Semantic search using embeddings
        semantic_results = self.semantic_search(query)
        
        # Keyword/BM25 full-text search
        keyword_results = self.keyword_search(query)
        
        # Graph traversal search
        graph_results = self.graph_traversal_search(query, context)
        
        # Temporal search (if temporal filter provided)
        temporal_results = self.temporal_search(query, temporal_filter)
        
        # Community-based search
        community_results = self.community_search(query)
        
        # Fusion and reranking
        return self.fuse_and_rerank([
            semantic_results, keyword_results, graph_results,
            temporal_results, community_results
        ])
```

#### 3.4.2 Search Strategies

**1. Global Search** (for holistic queries):
- Query against community summaries
- Multi-hop reasoning across communities  
- Temporal aggregation of facts
- Example: "What were the major trends in AI research from 2020-2023?"

**2. Local Search** (for specific entity queries):
- Entity-centric graph traversal
- Neighbor expansion with relevance filtering
- Relationship path analysis
- Example: "Tell me about John Smith's research collaborations"

**3. Temporal Search** (for time-sensitive queries):
- Point-in-time knowledge retrieval
- Timeline construction and analysis
- Change detection and evolution tracking
- Example: "How did OpenAI's leadership change in 2023?"

**4. Multi-Hop Search** (for complex reasoning):
- Graph path finding with semantic relevance
- Intermediate entity discovery
- Causal chain reconstruction
- Example: "How do climate policies affect renewable energy adoption?"

#### 3.4.3 Result Fusion and Ranking
```python
class ResultFusion:
    def fuse_results(self, result_sets):
        # Reciprocal Rank Fusion (RRF)
        rrf_scores = self.reciprocal_rank_fusion(result_sets)
        
        # Graph centrality boosting
        centrality_boost = self.apply_centrality_scores(rrf_scores)
        
        # Temporal relevance weighting
        temporal_weights = self.compute_temporal_relevance(centrality_boost)
        
        # Community coherence scoring
        coherence_scores = self.community_coherence_score(temporal_weights)
        
        # Final ranking
        return self.rank_by_combined_score(coherence_scores)
```

---

## 4. Data Models and Storage

### 4.1 Graph Schema Design

#### 4.1.1 Node Types
```python
# Core node types
class EpisodeNode:
    id: str
    content: str
    timestamp: datetime
    source_file: str
    processing_metadata: Dict

class EntityNode:
    id: str
    name: str
    type: str  # PERSON, ORGANIZATION, LOCATION, CONCEPT, etc.
    aliases: List[str]
    properties: Dict
    confidence_score: float
    first_mentioned: datetime
    last_updated: datetime

class CommunityNode:
    id: str
    level: int  # 0, 1, 2, 3 for hierarchy
    summary: str
    key_entities: List[str]
    member_count: int
    coherence_score: float
    created_at: datetime
```

#### 4.1.2 Edge Types
```python
class SemanticEdge:
    # Bi-temporal timestamps
    valid_at: Optional[datetime]
    invalid_at: Optional[datetime] 
    created_at: datetime
    expired_at: Optional[datetime]
    
    # Relationship metadata
    relationship_type: str
    fact: str
    confidence: float
    source_episodes: List[str]
    
    # Graph properties
    weight: float
    directionality: str  # DIRECTED, UNDIRECTED
```

#### 4.1.3 Relationship Types Taxonomy
- **Structural**: `PART_OF`, `CONTAINS`, `BELONGS_TO`
- **Social**: `WORKS_FOR`, `COLLABORATED_WITH`, `REPORTS_TO`
- **Temporal**: `PRECEDED_BY`, `FOLLOWED_BY`, `CONCURRENT_WITH`
- **Causal**: `CAUSED`, `RESULTED_IN`, `INFLUENCED`
- **Conceptual**: `SIMILAR_TO`, `RELATED_TO`, `OPPOSITE_OF`
- **Spatial**: `LOCATED_IN`, `NEAR`, `ADJACENT_TO`

### 4.2 Storage Architecture

#### 4.2.1 Multi-Store Strategy
- **Graph Database**: Neo4j for entity-relationship storage
- **Vector Store**: ChromaDB for semantic embeddings
- **Document Store**: MongoDB for original documents and metadata
- **Search Index**: Elasticsearch for full-text search
- **Time Series DB**: InfluxDB for temporal analytics
- **Cache Layer**: Redis for frequently accessed data

#### 4.2.2 Data Partitioning Strategy
- **Temporal Partitioning**: Partition by time periods for temporal queries
- **Community Partitioning**: Co-locate community members for graph traversal
- **Source Partitioning**: Separate data by source for access control
- **User Partitioning**: Multi-tenant isolation for user data

---

## 5. API Design and Interfaces

### 5.1 REST API Endpoints

#### 5.1.1 Document Management
```
POST /api/v1/documents/upload
GET  /api/v1/documents/{id}
PUT  /api/v1/documents/{id}
DELETE /api/v1/documents/{id}
GET  /api/v1/documents/search
POST /api/v1/documents/batch-upload
```

#### 5.1.2 Knowledge Graph Operations
```
GET  /api/v1/entities/{id}
GET  /api/v1/entities/search
GET  /api/v1/relationships/{id}
GET  /api/v1/communities
GET  /api/v1/graph/query
POST /api/v1/graph/traverse
```

#### 5.1.3 Intelligent Retrieval
```
POST /api/v1/search/global
POST /api/v1/search/local  
POST /api/v1/search/temporal
POST /api/v1/search/multi-hop
GET  /api/v1/search/suggestions
```

### 5.2 Query Interface Examples

#### 5.2.1 Global Query
```json
{
  "query": "What are the main research areas in artificial intelligence?",
  "search_type": "global",
  "community_levels": [2, 3],
  "time_range": {
    "start": "2020-01-01",
    "end": "2024-01-01"
  },
  "max_results": 10
}
```

#### 5.2.2 Local Query
```json
{
  "query": "Tell me about John Smith's research",
  "search_type": "local",
  "focal_entity": "John Smith",
  "expansion_depth": 2,
  "relationship_types": ["AUTHORED", "COLLABORATED_WITH", "WORKED_ON"],
  "include_temporal": true
}
```

#### 5.2.3 Temporal Query
```json
{
  "query": "How did AI research focus change during COVID-19?",
  "search_type": "temporal",
  "temporal_analysis": "change_detection",
  "comparison_periods": [
    {"start": "2019-01-01", "end": "2019-12-31"},
    {"start": "2020-03-01", "end": "2021-03-01"}
  ]
}
```

---

## 6. Processing Workflows

### 6.1 Document Ingestion Workflow
```
1. File Upload & Validation
   ├── Format Detection
   ├── Security Scanning  
   ├── Duplicate Detection
   └── Metadata Extraction

2. Document Parsing
   ├── Content Extraction
   ├── Structure Analysis
   ├── Episode Creation
   └── Temporal Context Assignment

3. Knowledge Extraction
   ├── Entity Recognition
   ├── Relationship Extraction
   ├── Fact Validation
   └── Confidence Scoring

4. Graph Integration
   ├── Entity Deduplication
   ├── Relationship Merging
   ├── Conflict Resolution
   └── Community Update

5. Index Generation
   ├── Vector Embeddings
   ├── Search Index Update
   ├── Community Recomputation
   └── Cache Invalidation
```

### 6.2 Query Processing Workflow
```
1. Query Analysis
   ├── Intent Classification
   ├── Entity Extraction
   ├── Temporal Detection
   └── Search Strategy Selection

2. Multi-Modal Retrieval
   ├── Semantic Search
   ├── Keyword Search
   ├── Graph Traversal
   └── Community Search

3. Result Processing
   ├── Result Fusion
   ├── Relevance Ranking
   ├── Diversity Enhancement
   └── Temporal Ordering

4. Response Generation
   ├── Context Assembly
   ├── Answer Synthesis
   ├── Source Attribution
   └── Confidence Estimation
```

---

## 7. Performance and Scalability

### 7.1 Performance Targets
- **Document Processing**: < 2 seconds per MB
- **Knowledge Extraction**: < 5 seconds per 1000 words  
- **Query Response**: < 1 second for 95% of queries
- **Graph Updates**: < 100ms for entity updates
- **Community Recomputation**: < 5 minutes for full graph

### 7.2 Scalability Strategy
- **Horizontal Scaling**: Microservices with independent scaling
- **Async Processing**: Background jobs for heavy computation
- **Caching**: Multi-level caching for frequent queries
- **Load Balancing**: Distribute requests across service instances
- **Database Sharding**: Partition data across multiple nodes

### 7.3 Optimization Techniques
- **Incremental Updates**: Only reprocess affected graph regions
- **Lazy Loading**: Load communities and relationships on demand
- **Query Optimization**: Cache common query patterns
- **Batch Processing**: Group similar operations for efficiency
- **Memory Management**: Stream processing for large documents

---

## 8. Quality Assurance and Monitoring

### 8.1 Quality Metrics
- **Extraction Accuracy**: Precision/recall for entities and relationships
- **Temporal Accuracy**: Correctness of temporal fact assignment
- **Search Relevance**: NDCG@k for retrieval quality
- **Knowledge Consistency**: Graph coherence and contradiction detection
- **User Satisfaction**: Query success rate and user feedback

### 8.2 Monitoring and Alerts
- **System Health**: Service availability and response times
- **Data Quality**: Entity extraction accuracy and graph consistency
- **Usage Analytics**: Query patterns and performance bottlenecks
- **Error Tracking**: Failed processing and data inconsistencies
- **Resource Utilization**: CPU, memory, and storage consumption

---

## 9. Security and Privacy

### 9.1 Data Protection
- **Encryption**: At-rest and in-transit encryption
- **Access Control**: Role-based permissions and document-level security
- **Data Anonymization**: PII detection and masking capabilities
- **Audit Logging**: Complete audit trail for all operations
- **Data Retention**: Configurable retention policies

### 9.2 Privacy Compliance
- **GDPR Compliance**: Right to erasure and data portability
- **Data Minimization**: Only store necessary information
- **Consent Management**: User consent tracking and management
- **Cross-Border**: Data residency and cross-border transfer controls

---

## 10. Implementation Roadmap

### Phase 1: Core Infrastructure (Months 1-3)
- [ ] Document parsing engine for basic formats
- [ ] Entity extraction pipeline
- [ ] Basic graph storage and querying
- [ ] Simple REST API

### Phase 2: Knowledge Graph Enhancement (Months 4-6)
- [ ] Relationship extraction and validation
- [ ] Temporal tracking implementation
- [ ] Community detection algorithm
- [ ] Conflict resolution mechanisms

### Phase 3: Advanced Retrieval (Months 7-9)
- [ ] Hybrid search implementation
- [ ] Multi-hop query processing
- [ ] Result fusion and ranking
- [ ] Query interface refinement

### Phase 4: Production Readiness (Months 10-12)
- [ ] Performance optimization
- [ ] Security implementation
- [ ] Monitoring and alerting
- [ ] User interface development
- [ ] Documentation and testing

---

This comprehensive design provides a solid foundation for building an advanced Knowledge RAG system that combines the temporal awareness of Graphiti with the hierarchical organization of GraphRAG, resulting in a powerful and scalable solution for intelligent document processing and retrieval. 