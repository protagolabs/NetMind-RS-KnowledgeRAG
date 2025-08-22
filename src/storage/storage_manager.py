"""
Storage Manager for Paper Extraction System.

Unified interface for storing and retrieving extracted paper knowledge
from both Neo4j graph database and ChromaDB vector database.
"""

import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .graph_store import PaperGraphStore
from .vector_store import PaperVectorStore
from .entity_consolidator import EntityConsolidator
from .models import (
    ExtractedEntity, ExtractedRelationship, ExtractedPaper,
    SearchResult, StorageStats
)

logger = logging.getLogger(__name__)


class PaperStorageManager:
    """Unified storage manager for extracted paper knowledge."""
    
    def __init__(
        self,
        # Neo4j configuration
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "gfll9999",
        neo4j_database: str = "neo4j",
        
        # ChromaDB configuration
        chroma_persist_dir: str = "./chroma_db_papers",
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the storage manager.
        
        Args:
            neo4j_uri: Neo4j URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Database name
            chroma_persist_dir: ChromaDB persistence directory
            openai_api_key: OpenAI API key for embeddings
            embedding_model: Embedding model to use
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize stores
        self.graph_store = PaperGraphStore(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database
        )
        
        self.vector_store = PaperVectorStore(
            persist_directory=chroma_persist_dir,
            openai_api_key=openai_api_key,
            embedding_model=embedding_model
        )
        
        # Store OpenAI API key for entity consolidator
        self.openai_api_key = openai_api_key
        self.entity_consolidator = None  # Will be initialized when needed
        
        self.logger.info("PaperStorageManager initialized")
    
    async def initialize(self):
        """Initialize connections to both stores."""
        try:
            # Connect to graph store
            await self.graph_store.connect()
            
            # Initialize vector store collections
            await self.vector_store.initialize_collections()
            
            self.logger.info("PaperStorageManager fully initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PaperStorageManager: {e}")
            raise
    
    async def close(self):
        """Close connections to both stores."""
        try:
            await self.graph_store.disconnect()
            await self.vector_store.close()
            self.logger.info("PaperStorageManager connections closed")
        except Exception as e:
            self.logger.error(f"Error closing PaperStorageManager: {e}")
    
    def load_json_extraction(self, json_path: str) -> ExtractedPaper:
        """
        Load extracted paper data from JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            ExtractedPaper object
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON data to ExtractedPaper
            entities = []
            for entity_data in data.get('entities', []):
                entity = ExtractedEntity(
                    name=entity_data.get('name', ''),
                    type=entity_data.get('type', ''),
                    description=entity_data.get('description', ''),
                    confidence=entity_data.get('confidence', 1.0),
                    attributes=entity_data.get('attributes', {}),
                    context=entity_data.get('context'),
                    chunk_id=entity_data.get('chunk_id'),
                    source_file=json_path
                )
                entities.append(entity)
            
            relationships = []
            for rel_data in data.get('relationships', []):
                rel = ExtractedRelationship(
                    source_entity=rel_data.get('source', ''),
                    target_entity=rel_data.get('target', ''),
                    relationship_type=rel_data.get('type', ''),
                    confidence=rel_data.get('confidence', 1.0),
                    properties=rel_data.get('properties', {}),
                    context=rel_data.get('context'),
                    chunk_id=rel_data.get('chunk_id'),
                    source_file=json_path
                )
                relationships.append(rel)
            
            # Extract paper metadata - handle both .llm_extraction.json and .enhanced_llm_extraction.json
            paper_path = json_path.replace('.enhanced_llm_extraction.json', '').replace('.llm_extraction.json', '')
            if not paper_path.endswith('.pdf'):
                paper_path += '.pdf'
            
            # Try to load chunks from the enhanced.json file if available
            chunks = data.get('chunks', [])
            if not chunks:
                # Try to load from corresponding enhanced.json file
                enhanced_json_path = json_path.replace('.enhanced_llm_extraction.json', '.enhanced.json').replace('.llm_extraction.json', '.enhanced.json')
                if Path(enhanced_json_path).exists():
                    try:
                        with open(enhanced_json_path, 'r', encoding='utf-8') as f:
                            enhanced_data = json.load(f)
                            chunks = enhanced_data.get('text_chunks', [])
                            self.logger.info(f"Loaded {len(chunks)} chunks from {enhanced_json_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not load chunks from {enhanced_json_path}: {e}")
            
            paper = ExtractedPaper(
                file_path=paper_path,
                title=data.get('title', Path(paper_path).stem),
                document_type=data.get('document_type', 'general'),
                schema_used=data.get('schema_used', 'general'),
                entities=entities,
                relationships=relationships,
                metadata=data.get('metadata', {}),
                chunks=chunks
            )
            
            return paper
            
        except Exception as e:
            self.logger.error(f"Failed to load JSON extraction from {json_path}: {e}")
            raise
    
    async def store_paper(
        self,
        paper: Optional[ExtractedPaper] = None,
        json_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store extracted paper in both graph and vector stores.
        
        Args:
            paper: ExtractedPaper object
            json_path: Path to JSON file to load
            
        Returns:
            Storage results
        """
        try:
            # Load from JSON if path provided
            if json_path and not paper:
                paper = self.load_json_extraction(json_path)
            
            if not paper:
                raise ValueError("No paper data provided")
            
            self.logger.info(f"Storing paper: {paper.file_path} with {len(paper.entities)} entities and {len(paper.relationships)} relationships")
            
            # Store in both databases concurrently
            graph_task = self.graph_store.store_paper(paper)
            vector_task = self.vector_store.store_paper_embeddings(paper)
            
            graph_result, vector_result = await asyncio.gather(graph_task, vector_task)
            
            results = {
                'file_path': paper.file_path,
                'graph_stored': graph_result,
                'embeddings_stored': vector_result,
                'entities_count': len(paper.entities),
                'relationships_count': len(paper.relationships),
                'chunks_count': len(paper.chunks)
            }
            
            self.logger.info(f"Paper stored successfully: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to store paper: {e}")
            raise
    
    async def search_hybrid(
        self,
        query: str,
        search_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> Dict[str, List[SearchResult]]:
        """
        Perform hybrid search across graph and vector stores.
        
        Args:
            query: Search query
            search_types: Types of search to perform
                         ['semantic_entities', 'semantic_chunks', 'semantic_papers',
                          'graph_entities', 'graph_relationships']
            limit: Maximum results per search type
            
        Returns:
            Dictionary with search results by type
        """
        if search_types is None:
            search_types = ['semantic_entities', 'semantic_chunks', 'graph_entities']
        
        results = {}
        
        try:
            # Execute searches in parallel
            search_tasks = []
            
            if 'semantic_entities' in search_types:
                search_tasks.append(('semantic_entities', 
                                    self.vector_store.search_entities_semantic(query, limit=limit)))
            
            if 'semantic_chunks' in search_types:
                search_tasks.append(('semantic_chunks',
                                    self.vector_store.search_chunks_semantic(query, limit=limit)))
            
            if 'semantic_papers' in search_types:
                search_tasks.append(('semantic_papers',
                                    self.vector_store.search_papers_semantic(query, limit=limit)))
            
            if 'graph_entities' in search_types:
                search_tasks.append(('graph_entities',
                                    self.graph_store.search_entities(query, limit=limit)))
            
            # Execute all searches concurrently
            if search_tasks:
                task_results = await asyncio.gather(*[task[1] for task in search_tasks])
                
                for i, (search_type, _) in enumerate(search_tasks):
                    results[search_type] = task_results[i]
            
            # Graph relationship search (if entity name provided)
            if 'graph_relationships' in search_types:
                # Extract potential entity names from query
                entity_names = self._extract_entity_names(query)
                if entity_names:
                    relationships = []
                    for entity_name in entity_names[:3]:  # Limit to first 3 entities
                        entity_rels = await self.graph_store.find_relationships(entity_name)
                        relationships.extend(entity_rels)
                    results['graph_relationships'] = relationships[:limit]
                else:
                    results['graph_relationships'] = []
            
            return results
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            return {}
    
    def _extract_entity_names(self, query: str) -> List[str]:
        """
        Extract potential entity names from query.
        
        Args:
            query: Search query
            
        Returns:
            List of potential entity names
        """
        # Simple implementation - extract capitalized words and quoted phrases
        import re
        
        entity_names = []
        
        # Find quoted phrases
        quoted = re.findall(r'"([^"]*)"', query)
        entity_names.extend(quoted)
        
        # Find capitalized words (potential entity names)
        words = query.replace('"', '').split()
        for word in words:
            clean_word = word.strip('.,!?;:()[]{}')
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                if clean_word not in entity_names:
                    entity_names.append(clean_word)
        
        return entity_names
    
    async def get_paper_knowledge_graph(self, file_path: str) -> Dict[str, Any]:
        """
        Get the complete knowledge graph for a specific paper.
        
        Args:
            file_path: Path to the paper file
            
        Returns:
            Knowledge graph data
        """
        try:
            graph_data = await self.graph_store.get_paper_graph(file_path)
            return graph_data
        except Exception as e:
            self.logger.error(f"Failed to get paper knowledge graph: {e}")
            return {}
    
    async def find_similar_papers(
        self,
        paper_path: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """
        Find papers similar to a given paper.
        
        Args:
            paper_path: Path to the paper
            limit: Maximum results
            
        Returns:
            List of similar papers
        """
        try:
            # Get the paper's title from graph
            graph_data = await self.graph_store.get_paper_graph(paper_path)
            
            # Search for similar papers using title or key entities
            if graph_data and graph_data.get('entities'):
                # Use top entities as query
                top_entities = graph_data['entities'][:3]
                query = ' '.join([e['name'] for e in top_entities])
                
                results = await self.vector_store.search_papers_semantic(query, limit=limit)
                
                # Filter out the original paper
                results = [r for r in results if r.source_file != paper_path]
                
                return results[:limit]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to find similar papers: {e}")
            return []
    
    async def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive storage statistics.
        
        Returns:
            Storage statistics
        """
        try:
            # Get graph statistics
            graph_stats = await self.graph_store.get_statistics()
            
            # Get vector statistics
            vector_stats = self.vector_store.get_collection_stats()
            
            return {
                'graph_database': {
                    'total_papers': graph_stats.total_papers,
                    'total_entities': graph_stats.total_entities,
                    'total_relationships': graph_stats.total_relationships,
                    'total_chunks': graph_stats.total_chunks,
                    'entities_by_type': graph_stats.entities_by_type,
                    'relationships_by_type': graph_stats.relationships_by_type,
                    'papers_by_type': graph_stats.papers_by_type
                },
                'vector_database': vector_stats,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage statistics: {e}")
            return {}
    
    async def clear_all_data(self) -> bool:
        """
        Clear all data from both stores.
        
        Returns:
            Success status
        """
        try:
            graph_cleared = await self.graph_store.clear_all_data()
            vector_cleared = self.vector_store.clear_all_embeddings()
            
            success = graph_cleared and vector_cleared
            
            if success:
                self.logger.info("All data cleared from storage")
            else:
                self.logger.warning("Some data may not have been cleared")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to clear all data: {e}")
            return False
    
    async def consolidate_entities_globally(
        self,
        dry_run: bool = False,
        entity_types: Optional[List[str]] = None,
        semantic_threshold: float = 0.85,
        levenshtein_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Consolidate entity variants across all documents using semantic similarity.
        
        This method identifies and merges duplicate entities that appear across
        different documents while preserving full context and occurrence information.
        
        Args:
            dry_run: If True, only identify duplicates without modifying data
            entity_types: Specific entity types to consolidate (None for all)
            semantic_threshold: Minimum cosine similarity for semantic matching (0-1)
            levenshtein_threshold: Minimum string similarity for name matching (0-1)
            
        Returns:
            Dictionary with consolidation results:
            - mode: 'dry_run' or 'executed'
            - entities_consolidated: Number of entity groups consolidated
            - consolidation_details: Details of consolidated entities
            - timestamp: When consolidation was performed
            
        Example:
            # Dry run to see what would be consolidated
            results = await storage.consolidate_entities_globally(dry_run=True)
            print(f"Would consolidate {results['entities_to_consolidate']} entities")
            
            # Actual consolidation
            results = await storage.consolidate_entities_globally()
            print(f"Consolidated {results['entities_consolidated']} entity groups")
        """
        try:
            # Initialize entity consolidator if not already done
            if self.entity_consolidator is None:
                # Import OpenAI client if available
                llm_client = None
                if self.openai_api_key:
                    try:
                        from openai import OpenAI
                        llm_client = OpenAI(api_key=self.openai_api_key)
                    except ImportError:
                        self.logger.warning("OpenAI not available, will use basic description merging")
                
                self.entity_consolidator = EntityConsolidator(
                    graph_store=self.graph_store,
                    vector_store=self.vector_store,
                    llm_client=llm_client,
                    semantic_threshold=semantic_threshold,
                    levenshtein_threshold=levenshtein_threshold
                )
            
            # Run consolidation
            self.logger.info(f"Starting entity consolidation (dry_run={dry_run})")
            results = await self.entity_consolidator.consolidate_entities_globally(
                dry_run=dry_run,
                entity_types=entity_types,
                exclude_consolidated=True
            )
            
            if dry_run:
                self.logger.info(
                    f"Dry run complete: Found {results['potential_groups']} groups "
                    f"with {results['entities_to_consolidate']} entities to consolidate"
                )
            else:
                self.logger.info(
                    f"Consolidation complete: Processed {results['entities_consolidated']} groups"
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to consolidate entities: {e}")
            raise
    
    async def get_entity_occurrences(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        Get all occurrences of an entity across documents.
        
        For consolidated entities, this returns all contexts where the entity
        (or its aliases) appeared across different documents.
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            List of occurrence dictionaries containing:
            - context: The specific context where entity appeared
            - chunk_id: ID of the chunk containing the context
            - paper_path: Path to the source document
            - paper_title: Title of the source document
            - confidence: Extraction confidence score
            - original_name: How the entity was originally named in that document
            - extraction_date: When it was extracted
            
        Example:
            occurrences = await storage.get_entity_occurrences("New York City")
            for occ in occurrences:
                print(f"{occ['paper_title']}: {occ['context']}")
        """
        try:
            if self.entity_consolidator is None:
                # Initialize consolidator for querying
                self.entity_consolidator = EntityConsolidator(
                    graph_store=self.graph_store,
                    vector_store=self.vector_store,
                    llm_client=None
                )
            
            return await self.entity_consolidator.get_entity_occurrences(entity_name)
            
        except Exception as e:
            self.logger.error(f"Failed to get entity occurrences: {e}")
            return []
    
    async def get_consolidation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about consolidated entities.
        
        Returns:
            Dictionary with consolidation statistics:
            - total_entities: Total number of entities
            - consolidated_entities: Number of consolidated entities
            - total_occurrences: Total occurrence nodes
            - avg_occurrences_per_entity: Average occurrences
            - entities_by_document_count: Distribution by document count
        """
        try:
            query = """
            MATCH (e:Entity)
            WITH count(e) as total_entities,
                 sum(CASE WHEN e.is_consolidated = true THEN 1 ELSE 0 END) as consolidated_entities
            
            OPTIONAL MATCH (e2:Entity {is_consolidated: true})-[:HAS_OCCURRENCE]->(o:Occurrence)
            WITH total_entities, consolidated_entities, count(o) as total_occurrences
            
            OPTIONAL MATCH (e3:Entity {is_consolidated: true})
            WITH total_entities, consolidated_entities, total_occurrences,
                 collect({name: e3.name, doc_count: e3.document_count}) as entity_docs
            
            RETURN total_entities, consolidated_entities, total_occurrences,
                   CASE WHEN consolidated_entities > 0 
                        THEN total_occurrences * 1.0 / consolidated_entities 
                        ELSE 0 END as avg_occurrences_per_entity,
                   entity_docs
            """
            
            async with self.graph_store.driver.session() as session:
                result = await session.run(query)
                record = await result.single()
                
                if record:
                    # Calculate distribution by document count
                    doc_count_dist = {}
                    for entity in record['entity_docs']:
                        if entity['doc_count']:
                            count = entity['doc_count']
                            doc_count_dist[count] = doc_count_dist.get(count, 0) + 1
                    
                    return {
                        'total_entities': record['total_entities'],
                        'consolidated_entities': record['consolidated_entities'],
                        'unconsolidated_entities': record['total_entities'] - record['consolidated_entities'],
                        'total_occurrences': record['total_occurrences'],
                        'avg_occurrences_per_entity': round(record['avg_occurrences_per_entity'], 2),
                        'entities_by_document_count': doc_count_dist,
                        'consolidation_percentage': round(
                            record['consolidated_entities'] * 100.0 / record['total_entities']
                            if record['total_entities'] > 0 else 0, 2
                        )
                    }
                
                return {
                    'total_entities': 0,
                    'consolidated_entities': 0,
                    'unconsolidated_entities': 0,
                    'total_occurrences': 0,
                    'avg_occurrences_per_entity': 0,
                    'entities_by_document_count': {},
                    'consolidation_percentage': 0
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get consolidation statistics: {e}")
            return {}