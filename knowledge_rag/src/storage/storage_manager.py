"""
Storage Manager for Knowledge RAG System

Unified interface for storing and retrieving knowledge from both
Neo4j graph database and ChromaDB vector database with full traceability.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .graph_store import GraphStore
from .vector_store import VectorStore
from .models import (
    StoredDocument, StoredEpisode, StoredEntity, StoredRelationship,
    SearchResult, StorageStats
)

# Import from extraction models
from ..knowledge_extraction.models import Entity, Relationship, Document
from ..file_management.models import Episode

logger = logging.getLogger(__name__)


class StorageManager:
    """Unified storage manager for Knowledge RAG system"""
    
    def __init__(
        self,
        # Neo4j configuration
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j", 
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        
        # ChromaDB configuration
        chroma_persist_dir: str = "./chroma_db",
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.logger = logging.getLogger(__name__)
        
        # Initialize stores
        self.graph_store = GraphStore(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database
        )
        
        self.vector_store = VectorStore(
            persist_directory=chroma_persist_dir,
            openai_api_key=openai_api_key,
            embedding_model=embedding_model
        )
        
        self.logger.info("StorageManager initialized")
    
    async def initialize(self):
        """Initialize connections to both stores"""
        try:
            # Connect to graph store
            await self.graph_store.connect()
            
            # Initialize vector store collections
            await self.vector_store.initialize_collections()
            
            self.logger.info("StorageManager fully initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize StorageManager: {e}")
            raise
    
    async def close(self):
        """Close connections to both stores"""
        try:
            await self.graph_store.disconnect()
            await self.vector_store.close()
            self.logger.info("StorageManager connections closed")
        except Exception as e:
            self.logger.error(f"Error closing StorageManager: {e}")
    
    # Conversion Utilities
    def _document_to_stored(self, document: Document) -> StoredDocument:
        """Convert extraction Document to StoredDocument"""
        return StoredDocument(
            uuid=document.uuid,
            file_path=document.file_path,
            file_name=document.file_name,
            file_hash=document.file_hash,
            title=document.title,
            document_type=document.document_type,
            created_at=document.created_at,
            modified_at=document.modified_at,
            processed_at=datetime.now(),
            total_episodes=len(document.episode_ids),
            total_entities=document.entity_count,
            total_relationships=document.relationship_count
        )
    
    def _episode_to_stored(self, episode: Episode, document: Document) -> StoredEpisode:
        """Convert file management Episode to StoredEpisode"""
        return StoredEpisode(
            uuid=episode.id,
            content=episode.content,
            episode_type=episode.episode_type,
            sequence_number=episode.sequence_number,
            timestamp=episode.timestamp,
            document_uuid=document.uuid,
            document_path=document.file_path,
            content_length=len(episode.content),
            header_level=episode.metadata.get('header_level'),
            header_text=episode.metadata.get('header_text')
        )
    
    def _entity_to_stored(self, entity: Entity) -> StoredEntity:
        """Convert extraction Entity to StoredEntity"""
        return StoredEntity(
            uuid=entity.uuid,
            name=entity.name,
            entity_type=entity.entity_type.value,
            summary=entity.summary,
            aliases=entity.aliases,
            attributes=entity.attributes,
            confidence=entity.confidence,
            created_at=entity.created_at,
            first_mentioned_at=entity.first_mentioned_at,
            last_updated_at=entity.last_updated_at,
            source_episodes=entity.source_episodes,
            source_documents=entity.source_documents
        )
    
    def _relationship_to_stored(self, relationship: Relationship) -> StoredRelationship:
        """Convert extraction Relationship to StoredRelationship"""
        return StoredRelationship(
            uuid=relationship.uuid,
            source_entity_uuid=relationship.source_entity_id,
            target_entity_uuid=relationship.target_entity_id,
            relationship_type=relationship.relationship_type.value,
            fact=relationship.fact,
            confidence=relationship.confidence,
            created_at=relationship.created_at,
            valid_at=relationship.valid_at,
            invalid_at=relationship.invalid_at,
            source_episodes=relationship.source_episodes,
            source_documents=relationship.source_documents,
            attributes=relationship.attributes
        )
    
    # Storage Operations
    async def store_knowledge_base(
        self,
        document: Document,
        episodes: List[Episode],
        entities: List[Entity], 
        relationships: List[Relationship]
    ) -> Dict[str, int]:
        """Store complete knowledge base in both graph and vector stores"""
        
        self.logger.info(f"Storing knowledge base: {len(episodes)} episodes, {len(entities)} entities, {len(relationships)} relationships")
        
        try:
            # Convert to storage models
            stored_document = self._document_to_stored(document)
            stored_episodes = [self._episode_to_stored(ep, document) for ep in episodes]
            stored_entities = [self._entity_to_stored(ent) for ent in entities]
            stored_relationships = [self._relationship_to_stored(rel) for rel in relationships]
            
            # Store in graph database
            graph_results = await self._store_in_graph(
                stored_document, stored_episodes, stored_entities, stored_relationships
            )
            
            # Store in vector database
            vector_results = await self._store_in_vector(
                stored_document, stored_episodes, stored_entities
            )
            
            results = {
                'documents_stored': 1 if graph_results['documents'] > 0 else 0,
                'episodes_stored': graph_results['episodes'],
                'entities_stored': graph_results['entities'],
                'relationships_stored': graph_results['relationships'],
                'entity_embeddings_stored': vector_results['entities'],
                'episode_embeddings_stored': vector_results['episodes']
            }
            
            self.logger.info(f"Knowledge base stored successfully: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to store knowledge base: {e}")
            raise
    
    async def _store_in_graph(
        self,
        document: StoredDocument,
        episodes: List[StoredEpisode],
        entities: List[StoredEntity],
        relationships: List[StoredRelationship]
    ) -> Dict[str, int]:
        """Store data in Neo4j graph database"""
        
        results = {'documents': 0, 'episodes': 0, 'entities': 0, 'relationships': 0}
        
        try:
            # Store document
            if await self.graph_store.store_document(document):
                results['documents'] = 1
            
            # Store episodes
            results['episodes'] = await self.graph_store.store_episodes(episodes)
            
            # Store entities  
            results['entities'] = await self.graph_store.store_entities(entities)
            
            # Store relationships
            results['relationships'] = await self.graph_store.store_relationships(relationships)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to store in graph database: {e}")
            raise
    
    async def _store_in_vector(
        self,
        document: StoredDocument,
        episodes: List[StoredEpisode],
        entities: List[StoredEntity]
    ) -> Dict[str, int]:
        """Store embeddings in ChromaDB vector database"""
        
        results = {'entities': 0, 'episodes': 0}
        
        try:
            # Prepare entity data for embeddings
            entity_data = []
            for entity in entities:
                entity_data.append({
                    'uuid': entity.uuid,
                    'name': entity.name,
                    'summary': entity.summary,
                    'entity_type': entity.entity_type,
                    'document_uuid': document.uuid,
                    'document_path': document.file_path,
                    'document_name': document.file_name,
                    'episode_uuids': entity.source_episodes
                })
            
            # Store entity embeddings
            results['entities'] = await self.vector_store.store_entity_embeddings(entity_data)
            
            # Prepare episode data for embeddings
            episode_data = []
            for episode in episodes:
                episode_data.append({
                    'uuid': episode.uuid,
                    'content': episode.content,
                    'episode_type': episode.episode_type,
                    'sequence_number': episode.sequence_number,
                    'header_text': episode.header_text,
                    'document_uuid': document.uuid,
                    'document_path': document.file_path,
                    'document_name': document.file_name
                })
            
            # Store episode embeddings
            results['episodes'] = await self.vector_store.store_episode_embeddings(episode_data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to store in vector database: {e}")
            raise
    
    # Combined Search Operations
    async def search_hybrid(
        self,
        query: str,
        search_types: List[str] = None,
        limit: int = 10
    ) -> Dict[str, List[SearchResult]]:
        """
        Perform hybrid search across graph and vector stores
        
        Args:
            query: Search query
            search_types: List of search types to perform 
                         ['semantic_entities', 'semantic_episodes', 'graph_entities', 'graph_episodes', 'graph_traversal']
            limit: Maximum results per search type
        
        Returns:
            Dictionary with search results by type
        """
        
        if search_types is None:
            search_types = ['semantic_entities', 'semantic_episodes', 'graph_entities', 'graph_episodes']
        
        results = {}
        
        try:
            # Execute searches in parallel
            search_tasks = []
            
            if 'semantic_entities' in search_types:
                search_tasks.append(('semantic_entities', self.vector_store.search_entities_semantic(query, limit)))
            
            if 'semantic_episodes' in search_types:
                search_tasks.append(('semantic_episodes', self.vector_store.search_episodes_semantic(query, limit)))
            
            if 'graph_entities' in search_types:
                search_tasks.append(('graph_entities', self.graph_store.search_entities_by_name(query, limit)))
            
            if 'graph_episodes' in search_types:
                search_tasks.append(('graph_episodes', self.graph_store.search_episodes_by_content(query, limit)))
            
            # Execute all searches concurrently
            if search_tasks:
                task_results = await asyncio.gather(*[task[1] for task in search_tasks])
                
                for i, (search_type, _) in enumerate(search_tasks):
                    results[search_type] = task_results[i]
            
            # Optional: Graph traversal search (requires entity extraction from query)
            if 'graph_traversal' in search_types:
                # Simple entity extraction from query - can be enhanced
                entity_names = self._extract_potential_entities(query)
                if entity_names:
                    results['graph_traversal'] = await self.graph_store.graph_traversal_search(
                        entity_names, hops=2, limit=limit
                    )
                else:
                    results['graph_traversal'] = []
            
            return results
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            return {}
    
    def _extract_potential_entities(self, query: str) -> List[str]:
        """Extract potential entity names from query (simple implementation)"""
        # This is a simple implementation - could be enhanced with NER
        words = query.split()
        
        # Look for capitalized words that might be entities
        potential_entities = []
        for word in words:
            # Remove punctuation and check if it starts with capital
            clean_word = word.strip('.,!?;:"()[]{}')
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                potential_entities.append(clean_word)
        
        return potential_entities
    
    # Statistics and Management
    async def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        try:
            # Get graph statistics
            graph_stats = await self.graph_store.get_storage_stats()
            
            # Get vector statistics
            vector_stats = self.vector_store.get_collection_stats()
            
            return {
                'graph_database': {
                    'total_documents': graph_stats.total_documents,
                    'total_episodes': graph_stats.total_episodes,
                    'total_entities': graph_stats.total_entities,
                    'total_relationships': graph_stats.total_relationships,
                    'entities_by_type': graph_stats.entities_by_type,
                    'relationships_by_type': graph_stats.relationships_by_type
                },
                'vector_database': vector_stats,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage statistics: {e}")
            return {}
    
    async def clear_all_data(self) -> bool:
        """Clear all data from both stores"""
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