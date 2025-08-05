"""
Vector Store for Knowledge RAG System

Handles storage and semantic search of embeddings using ChromaDB
with full document traceability for entities and episodes.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import uuid

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.api.models.Collection import Collection
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Collection = None

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

from .models import SearchResult

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB vector database for semantic search in Knowledge RAG system"""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install with: pip install openai")
        
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client for embeddings
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Collections for different data types
        self.entities_collection: Optional[Collection] = None
        self.episodes_collection: Optional[Collection] = None
        
        self.logger.info(f"VectorStore initialized with persist directory: {persist_directory}")
    
    async def initialize_collections(self):
        """Initialize ChromaDB collections"""
        try:
            # Entity embeddings collection
            self.entities_collection = self.client.get_or_create_collection(
                name="entities",
                metadata={"description": "Entity embeddings for semantic search"}
            )
            
            # Episode embeddings collection  
            self.episodes_collection = self.client.get_or_create_collection(
                name="episodes",
                metadata={"description": "Episode content embeddings for semantic search"}
            )
            
            self.logger.info("ChromaDB collections initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB collections: {e}")
            raise
    
    def _create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text.replace('\n', ' ')[:8000]  # Limit text length
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Failed to create embedding: {e}")
            return []
    
    def _create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts in batch"""
        try:
            # Clean and limit text length
            cleaned_texts = [text.replace('\n', ' ')[:8000] for text in texts]
            
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=cleaned_texts
            )
            
            return [item.embedding for item in response.data]
        except Exception as e:
            self.logger.error(f"Failed to create batch embeddings: {e}")
            return []
    
    # Entity Operations
    async def store_entity_embeddings(
        self, 
        entities: List[Dict[str, Any]]
    ) -> int:
        """
        Store entity embeddings in vector database
        
        Args:
            entities: List of entity dicts with keys:
                - uuid: Entity UUID
                - name: Entity name
                - summary: Entity summary
                - entity_type: Entity type
                - document_uuid: Source document UUID
                - document_path: Source document path
                - document_name: Source document name
                - episode_uuids: List of source episode UUIDs
        """
        if not entities:
            return 0
        
        try:
            # Prepare texts for embedding (name + summary)
            texts = []
            metadatas = []
            ids = []
            
            for entity in entities:
                # Create searchable text from name and summary
                text = f"{entity['name']}: {entity.get('summary', '')}"
                texts.append(text)
                
                # Prepare metadata for traceability
                metadata = {
                    'entity_uuid': entity['uuid'],
                    'entity_name': entity['name'],
                    'entity_type': entity.get('entity_type', ''),
                    'document_uuid': entity.get('document_uuid', ''),
                    'document_path': entity.get('document_path', ''),
                    'document_name': entity.get('document_name', ''),
                    'episode_uuids': json.dumps(entity.get('episode_uuids', [])),
                    'content_type': 'entity',
                    'created_at': datetime.now().isoformat()
                }
                metadatas.append(metadata)
                ids.append(entity['uuid'])
            
            # Create embeddings in batch
            embeddings = self._create_embeddings_batch(texts)
            
            if not embeddings:
                self.logger.error("Failed to create embeddings for entities")
                return 0
            
            # Store in ChromaDB
            self.entities_collection.upsert(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Stored {len(entities)} entity embeddings")
            return len(entities)
            
        except Exception as e:
            self.logger.error(f"Failed to store entity embeddings: {e}")
            return 0
    
    async def search_entities_semantic(
        self, 
        query: str, 
        limit: int = 10,
        entity_types: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Search entities using semantic similarity"""
        try:
            # Create query embedding
            query_embedding = self._create_embedding(query)
            if not query_embedding:
                return []
            
            # Prepare where filter for entity types if specified
            where_filter = None
            if entity_types:
                where_filter = {"entity_type": {"$in": entity_types}}
            
            # Search in ChromaDB
            results = self.entities_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_filter
            )
            
            # Convert to SearchResult objects
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (0-1, higher is better)
                    score = max(0.0, 1.0 - distance)
                    
                    search_result = SearchResult(
                        content=doc,
                        score=score,
                        result_type='entity',
                        source_uuid=metadata['entity_uuid'],
                        source_type='semantic_entity',
                        document_uuid=metadata['document_uuid'],
                        document_path=metadata['document_path'],
                        document_name=metadata['document_name'],
                        entity_name=metadata['entity_name'],
                        entity_type=metadata['entity_type'],
                        metadata={
                            'similarity_distance': distance,
                            'episode_uuids': json.loads(metadata.get('episode_uuids', '[]'))
                        }
                    )
                    search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Entity semantic search failed: {e}")
            return []
    
    # Episode Operations
    async def store_episode_embeddings(
        self,
        episodes: List[Dict[str, Any]]
    ) -> int:
        """
        Store episode embeddings in vector database
        
        Args:
            episodes: List of episode dicts with keys:
                - uuid: Episode UUID
                - content: Episode content
                - episode_type: Episode type
                - sequence_number: Sequence number
                - header_text: Header text (optional)
                - document_uuid: Source document UUID
                - document_path: Source document path
                - document_name: Source document name
        """
        if not episodes:
            return 0
        
        try:
            # Prepare texts for embedding
            texts = []
            metadatas = []
            ids = []
            
            for episode in episodes:
                # Use episode content for embedding
                content = episode.get('content', '')
                
                # Add header text if available for better context
                if episode.get('header_text'):
                    content = f"{episode['header_text']}\n\n{content}"
                
                texts.append(content)
                
                # Prepare metadata for traceability
                metadata = {
                    'episode_uuid': episode['uuid'],
                    'episode_type': episode.get('episode_type', ''),
                    'sequence_number': episode.get('sequence_number', 0),
                    'header_text': episode.get('header_text', ''),
                    'document_uuid': episode.get('document_uuid', ''),
                    'document_path': episode.get('document_path', ''),
                    'document_name': episode.get('document_name', ''),
                    'content_length': len(content),
                    'content_type': 'episode',
                    'created_at': datetime.now().isoformat()
                }
                metadatas.append(metadata)
                ids.append(episode['uuid'])
            
            # Create embeddings in batch
            embeddings = self._create_embeddings_batch(texts)
            
            if not embeddings:
                self.logger.error("Failed to create embeddings for episodes")
                return 0
            
            # Store in ChromaDB
            self.episodes_collection.upsert(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Stored {len(episodes)} episode embeddings")
            return len(episodes)
            
        except Exception as e:
            self.logger.error(f"Failed to store episode embeddings: {e}")
            return 0
    
    async def search_episodes_semantic(
        self,
        query: str,
        limit: int = 10,
        document_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """Search episodes using semantic similarity"""
        try:
            # Create query embedding
            query_embedding = self._create_embedding(query)
            if not query_embedding:
                return []
            
            # Prepare where filter for document if specified
            where_filter = None
            if document_filter:
                where_filter = {"document_uuid": document_filter}
            
            # Search in ChromaDB
            results = self.episodes_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_filter
            )
            
            # Convert to SearchResult objects
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (0-1, higher is better)
                    score = max(0.0, 1.0 - distance)
                    
                    search_result = SearchResult(
                        content=doc,
                        score=score,
                        result_type='episode',
                        source_uuid=metadata['episode_uuid'],
                        source_type='semantic_episode',
                        document_uuid=metadata['document_uuid'],
                        document_path=metadata['document_path'],
                        document_name=metadata['document_name'],
                        episode_uuid=metadata['episode_uuid'],
                        episode_sequence=metadata.get('sequence_number'),
                        metadata={
                            'similarity_distance': distance,
                            'episode_type': metadata.get('episode_type'),
                            'header_text': metadata.get('header_text'),
                            'content_length': metadata.get('content_length')
                        }
                    )
                    search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Episode semantic search failed: {e}")
            return []
    
    # Hybrid Operations
    async def search_hybrid_semantic(
        self,
        query: str,
        entity_limit: int = 5,
        episode_limit: int = 5
    ) -> Tuple[List[SearchResult], List[SearchResult]]:
        """Perform hybrid semantic search across entities and episodes"""
        try:
            # Search both entities and episodes in parallel
            entity_results = await self.search_entities_semantic(query, entity_limit)
            episode_results = await self.search_episodes_semantic(query, episode_limit)
            
            return entity_results, episode_results
            
        except Exception as e:
            self.logger.error(f"Hybrid semantic search failed: {e}")
            return [], []
    
    # Management Operations
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings"""
        try:
            stats = {
                'entities': {
                    'count': self.entities_collection.count(),
                    'collection_name': self.entities_collection.name
                },
                'episodes': {
                    'count': self.episodes_collection.count(),
                    'collection_name': self.episodes_collection.name
                },
                'embedding_model': self.embedding_model,
                'persist_directory': self.persist_directory
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def clear_all_embeddings(self) -> bool:
        """Clear all embeddings from vector store"""
        try:
            # Delete and recreate collections
            self.client.delete_collection("entities")
            self.client.delete_collection("episodes")
            
            # Recreate collections
            self.entities_collection = self.client.create_collection("entities")
            self.episodes_collection = self.client.create_collection("episodes")
            
            self.logger.info("Cleared all embeddings from vector store")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear embeddings: {e}")
            return False
    
    async def close(self):
        """Close vector store connections"""
        # ChromaDB doesn't require explicit closing
        self.logger.info("Vector store closed") 