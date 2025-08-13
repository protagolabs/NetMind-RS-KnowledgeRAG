"""
Vector Store for Paper Extraction Storage System.

Handles storage and semantic search of embeddings using ChromaDB
for entities and chunks extracted from papers.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import hashlib

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

from .models import ExtractedEntity, ExtractedPaper, SearchResult

logger = logging.getLogger(__name__)


class PaperVectorStore:
    """ChromaDB vector database for semantic search of paper extractions."""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db_papers",
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist ChromaDB
            openai_api_key: OpenAI API key for embeddings
            embedding_model: OpenAI embedding model to use
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install with: pip install openai")
        
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client for embeddings
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            # Assume API key is set in environment
            self.openai_client = OpenAI()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Collections for different data types
        self.entities_collection: Optional[Collection] = None
        self.chunks_collection: Optional[Collection] = None
        self.papers_collection: Optional[Collection] = None
        
        self.logger.info(f"PaperVectorStore initialized with persist directory: {persist_directory}")
    
    async def initialize_collections(self):
        """Initialize ChromaDB collections."""
        try:
            # Entity embeddings collection
            self.entities_collection = self.client.get_or_create_collection(
                name="paper_entities",
                metadata={"description": "Entity embeddings from extracted papers"}
            )
            
            # Chunk embeddings collection
            self.chunks_collection = self.client.get_or_create_collection(
                name="paper_chunks",
                metadata={"description": "Chunk embeddings from papers"}
            )
            
            # Paper summary embeddings
            self.papers_collection = self.client.get_or_create_collection(
                name="papers",
                metadata={"description": "Paper title and summary embeddings"}
            )
            
            self.logger.info("ChromaDB collections initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB collections: {e}")
            raise
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
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
        """
        Create embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
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
    
    def _generate_entity_id(self, entity: ExtractedEntity, paper_path: str) -> str:
        """
        Generate unique ID for an entity.
        
        Args:
            entity: Entity to generate ID for
            paper_path: Path to the paper
            
        Returns:
            Unique entity ID
        """
        id_string = f"{paper_path}:{entity.name}:{entity.type}"
        return hashlib.md5(id_string.encode()).hexdigest()
    
    async def store_paper_embeddings(self, paper: ExtractedPaper) -> Dict[str, int]:
        """
        Store embeddings for a complete extracted paper.
        
        Args:
            paper: Extracted paper data
            
        Returns:
            Dictionary with counts of stored items
        """
        counts = {
            'entities': 0,
            'chunks': 0,
            'paper': 0
        }
        
        try:
            # Store paper-level embedding
            if paper.title:
                paper_text = f"{paper.title}\n{paper.document_type}"
                paper_embedding = self._create_embedding(paper_text)
                
                if paper_embedding:
                    self.papers_collection.upsert(
                        embeddings=[paper_embedding],
                        documents=[paper_text],
                        metadatas=[{
                            'file_path': paper.file_path,
                            'title': paper.title,
                            'document_type': paper.document_type,
                            'schema_used': paper.schema_used,
                            'entity_count': len(paper.entities),
                            'relationship_count': len(paper.relationships),
                            'extraction_timestamp': paper.extraction_timestamp.isoformat()
                        }],
                        ids=[hashlib.md5(paper.file_path.encode()).hexdigest()]
                    )
                    counts['paper'] = 1
            
            # Store entity embeddings
            if paper.entities:
                entity_texts = []
                entity_metadatas = []
                entity_ids = []
                
                for entity in paper.entities:
                    # Create searchable text from entity
                    entity_text = f"{entity.name}: {entity.description}"
                    if entity.context:
                        entity_text += f"\nContext: {entity.context}"
                    
                    entity_texts.append(entity_text)
                    
                    # Prepare metadata
                    metadata = {
                        'entity_name': entity.name,
                        'entity_type': entity.type,
                        'description': entity.description,
                        'confidence': entity.confidence,
                        'attributes': json.dumps(entity.attributes),
                        'chunk_id': entity.chunk_id or '',
                        'paper_path': paper.file_path,
                        'paper_title': paper.title or '',
                        'document_type': paper.document_type,
                        'created_at': entity.created_at.isoformat()
                    }
                    entity_metadatas.append(metadata)
                    entity_ids.append(self._generate_entity_id(entity, paper.file_path))
                
                # Create embeddings in batch
                entity_embeddings = self._create_embeddings_batch(entity_texts)
                
                if entity_embeddings:
                    self.entities_collection.upsert(
                        embeddings=entity_embeddings,
                        documents=entity_texts,
                        metadatas=entity_metadatas,
                        ids=entity_ids
                    )
                    counts['entities'] = len(entity_embeddings)
            
            # Store chunk embeddings
            if paper.chunks:
                chunk_texts = []
                chunk_metadatas = []
                chunk_ids = []
                
                for i, chunk in enumerate(paper.chunks):
                    chunk_content = chunk.get('content', '')
                    if chunk_content:
                        chunk_texts.append(chunk_content)
                        
                        metadata = {
                            'chunk_id': chunk.get('chunk_id', f"chunk_{i}"),
                            'sequence': i,
                            'paper_path': paper.file_path,
                            'paper_title': paper.title or '',
                            'document_type': paper.document_type,
                            'content_length': len(chunk_content)
                        }
                        chunk_metadatas.append(metadata)
                        
                        chunk_id = f"{paper.file_path}:chunk_{i}"
                        chunk_ids.append(hashlib.md5(chunk_id.encode()).hexdigest())
                
                if chunk_texts:
                    chunk_embeddings = self._create_embeddings_batch(chunk_texts)
                    
                    if chunk_embeddings:
                        self.chunks_collection.upsert(
                            embeddings=chunk_embeddings,
                            documents=chunk_texts,
                            metadatas=chunk_metadatas,
                            ids=chunk_ids
                        )
                        counts['chunks'] = len(chunk_embeddings)
            
            self.logger.info(f"Stored embeddings for paper {paper.file_path}: {counts}")
            return counts
            
        except Exception as e:
            self.logger.error(f"Failed to store paper embeddings: {e}")
            return counts
    
    async def search_entities_semantic(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search entities using semantic similarity.
        
        Args:
            query: Search query
            entity_types: Filter by entity types
            limit: Maximum results
            
        Returns:
            List of search results
        """
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
                        source_file=metadata['paper_path'],
                        entity_name=metadata['entity_name'],
                        entity_type=metadata['entity_type'],
                        metadata={
                            'description': metadata['description'],
                            'confidence': metadata['confidence'],
                            'attributes': json.loads(metadata.get('attributes', '{}')),
                            'paper_title': metadata.get('paper_title'),
                            'chunk_id': metadata.get('chunk_id'),
                            'similarity_distance': distance
                        }
                    )
                    search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Entity semantic search failed: {e}")
            return []
    
    async def search_chunks_semantic(
        self,
        query: str,
        paper_filter: Optional[str] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search chunks using semantic similarity.
        
        Args:
            query: Search query
            paper_filter: Filter by paper path
            limit: Maximum results
            
        Returns:
            List of search results
        """
        try:
            # Create query embedding
            query_embedding = self._create_embedding(query)
            if not query_embedding:
                return []
            
            # Prepare where filter for paper if specified
            where_filter = None
            if paper_filter:
                where_filter = {"paper_path": paper_filter}
            
            # Search in ChromaDB
            results = self.chunks_collection.query(
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
                        result_type='chunk',
                        source_file=metadata['paper_path'],
                        metadata={
                            'chunk_id': metadata['chunk_id'],
                            'sequence': metadata.get('sequence'),
                            'paper_title': metadata.get('paper_title'),
                            'document_type': metadata.get('document_type'),
                            'similarity_distance': distance
                        }
                    )
                    search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Chunk semantic search failed: {e}")
            return []
    
    async def search_papers_semantic(
        self,
        query: str,
        document_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search papers using semantic similarity.
        
        Args:
            query: Search query
            document_types: Filter by document types
            limit: Maximum results
            
        Returns:
            List of search results
        """
        try:
            # Create query embedding
            query_embedding = self._create_embedding(query)
            if not query_embedding:
                return []
            
            # Prepare where filter for document types if specified
            where_filter = None
            if document_types:
                where_filter = {"document_type": {"$in": document_types}}
            
            # Search in ChromaDB
            results = self.papers_collection.query(
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
                        result_type='paper',
                        source_file=metadata['file_path'],
                        metadata={
                            'title': metadata.get('title'),
                            'document_type': metadata.get('document_type'),
                            'schema_used': metadata.get('schema_used'),
                            'entity_count': metadata.get('entity_count'),
                            'relationship_count': metadata.get('relationship_count'),
                            'similarity_distance': distance
                        }
                    )
                    search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Paper semantic search failed: {e}")
            return []
    
    async def search_hybrid(
        self,
        query: str,
        search_entities: bool = True,
        search_chunks: bool = True,
        search_papers: bool = True,
        limit_per_type: int = 5
    ) -> Dict[str, List[SearchResult]]:
        """
        Perform hybrid search across all collections.
        
        Args:
            query: Search query
            search_entities: Whether to search entities
            search_chunks: Whether to search chunks
            search_papers: Whether to search papers
            limit_per_type: Maximum results per type
            
        Returns:
            Dictionary with search results by type
        """
        results = {}
        
        try:
            if search_entities:
                results['entities'] = await self.search_entities_semantic(query, limit=limit_per_type)
            
            if search_chunks:
                results['chunks'] = await self.search_chunks_semantic(query, limit=limit_per_type)
            
            if search_papers:
                results['papers'] = await self.search_papers_semantic(query, limit=limit_per_type)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            return {}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored embeddings.
        
        Returns:
            Statistics dictionary
        """
        try:
            stats = {
                'entities': {
                    'count': self.entities_collection.count() if self.entities_collection else 0,
                    'collection_name': 'paper_entities'
                },
                'chunks': {
                    'count': self.chunks_collection.count() if self.chunks_collection else 0,
                    'collection_name': 'paper_chunks'
                },
                'papers': {
                    'count': self.papers_collection.count() if self.papers_collection else 0,
                    'collection_name': 'papers'
                },
                'embedding_model': self.embedding_model,
                'persist_directory': self.persist_directory
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def clear_all_embeddings(self) -> bool:
        """
        Clear all embeddings from vector store.
        
        Returns:
            Success status
        """
        try:
            # Delete and recreate collections
            self.client.delete_collection("paper_entities")
            self.client.delete_collection("paper_chunks")
            self.client.delete_collection("papers")
            
            # Recreate collections
            self.entities_collection = self.client.create_collection("paper_entities")
            self.chunks_collection = self.client.create_collection("paper_chunks")
            self.papers_collection = self.client.create_collection("papers")
            
            self.logger.info("Cleared all embeddings from vector store")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear embeddings: {e}")
            return False
    
    async def close(self):
        """Close vector store connections."""
        # ChromaDB doesn't require explicit closing
        self.logger.info("Vector store closed")