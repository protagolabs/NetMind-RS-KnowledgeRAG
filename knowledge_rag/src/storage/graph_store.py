"""
Neo4j Graph Store for Knowledge RAG System

Handles storage and retrieval of entities, relationships, episodes, and documents
in Neo4j graph database with full provenance tracking.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
    from neo4j.exceptions import ServiceUnavailable, ClientError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncGraphDatabase = None
    AsyncDriver = None
    AsyncSession = None

from .models import (
    StoredDocument, StoredEpisode, StoredEntity, StoredRelationship,
    SearchResult, StorageStats
)

logger = logging.getLogger(__name__)


class GraphStore:
    """Neo4j graph database storage for Knowledge RAG system"""
    
    def __init__(
        self, 
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available. Install with: pip install neo4j")
        
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional[AsyncDriver] = None
        self.logger = logging.getLogger(__name__)
        
    async def connect(self):
        """Initialize connection to Neo4j"""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            
            # Test connection
            async with self.driver.session(database=self.database) as session:
                await session.run("RETURN 1")
            
            self.logger.info(f"Connected to Neo4j at {self.uri}")
            
            # Create indexes and constraints
            await self._create_indexes_and_constraints()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def disconnect(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            self.logger.info("Disconnected from Neo4j")
    
    async def _create_indexes_and_constraints(self):
        """Create necessary indexes and constraints"""
        async with self.driver.session(database=self.database) as session:
            # Create constraints (also creates indexes)
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.uuid IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Episode) REQUIRE e.uuid IS UNIQUE", 
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.uuid IS UNIQUE",
                
                # Additional indexes for performance
                "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.file_hash)",
                "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.file_path)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Episode) ON (e.document_uuid)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Episode) ON (e.sequence_number)", 
                "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.entity_type)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.source_documents)",
                "CREATE FULLTEXT INDEX IF NOT EXISTS entityNames FOR (n:Entity) ON EACH [n.name, n.summary]",
                "CREATE FULLTEXT INDEX IF NOT EXISTS episodeContent FOR (e:Episode) ON EACH [e.content, e.header_text]"
            ]
            
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except ClientError as e:
                    if "already exists" not in str(e).lower():
                        self.logger.warning(f"Failed to create constraint/index: {e}")
        
        self.logger.info("Created Neo4j indexes and constraints")
    
    # Document Operations
    async def store_document(self, document: StoredDocument) -> bool:
        """Store a document in the graph"""
        async with self.driver.session(database=self.database) as session:
            try:
                query = """
                MERGE (d:Document {uuid: $uuid})
                SET d.file_path = $file_path,
                    d.file_name = $file_name,
                    d.file_hash = $file_hash,
                    d.title = $title,
                    d.document_type = $document_type,
                    d.created_at = $created_at,
                    d.modified_at = $modified_at,
                    d.processed_at = $processed_at,
                    d.total_episodes = $total_episodes,
                    d.total_entities = $total_entities,
                    d.total_relationships = $total_relationships
                RETURN d.uuid
                """
                
                result = await session.run(query, **{
                    'uuid': document.uuid,
                    'file_path': document.file_path,
                    'file_name': document.file_name,
                    'file_hash': document.file_hash,
                    'title': document.title,
                    'document_type': document.document_type,
                    'created_at': document.created_at,
                    'modified_at': document.modified_at,
                    'processed_at': document.processed_at,
                    'total_episodes': document.total_episodes,
                    'total_entities': document.total_entities,
                    'total_relationships': document.total_relationships
                })
                
                record = await result.single()
                return record is not None
                
            except Exception as e:
                self.logger.error(f"Failed to store document {document.uuid}: {e}")
                return False
    
    # Episode Operations  
    async def store_episodes(self, episodes: List[StoredEpisode]) -> int:
        """Store multiple episodes in the graph"""
        if not episodes:
            return 0
            
        async with self.driver.session(database=self.database) as session:
            stored_count = 0
            
            try:
                # Batch insert episodes
                query = """
                UNWIND $episodes AS ep
                MERGE (e:Episode {uuid: ep.uuid})
                SET e += ep.properties
                
                // Link to document
                WITH e, ep
                MATCH (d:Document {uuid: ep.document_uuid})
                MERGE (d)-[:CONTAINS]->(e)
                
                RETURN count(e) as count
                """
                
                episode_data = []
                for episode in episodes:
                    episode_data.append({
                        'uuid': episode.uuid,
                        'document_uuid': episode.document_uuid,
                        'properties': episode.to_neo4j_properties()
                    })
                
                result = await session.run(query, episodes=episode_data)
                record = await result.single()
                stored_count = record['count'] if record else 0
                
                self.logger.info(f"Stored {stored_count} episodes in Neo4j")
                return stored_count
                
            except Exception as e:
                self.logger.error(f"Failed to store episodes: {e}")
                return stored_count
    
    # Entity Operations
    async def store_entities(self, entities: List[StoredEntity]) -> int:
        """Store multiple entities in the graph"""
        if not entities:
            return 0
            
        async with self.driver.session(database=self.database) as session:
            stored_count = 0
            
            try:
                # Batch insert entities
                query = """
                UNWIND $entities AS ent
                MERGE (e:Entity {uuid: ent.uuid})
                SET e += ent.properties
                RETURN count(e) as count
                """
                
                entity_data = []
                for entity in entities:
                    entity_data.append({
                        'uuid': entity.uuid,
                        'properties': entity.to_neo4j_properties()
                    })
                
                result = await session.run(query, entities=entity_data)
                record = await result.single()
                stored_count = record['count'] if record else 0
                
                # Create entity-episode relationships
                await self._link_entities_to_episodes(session, entities)
                
                self.logger.info(f"Stored {stored_count} entities in Neo4j")
                return stored_count
                
            except Exception as e:
                self.logger.error(f"Failed to store entities: {e}")
                return stored_count
    
    async def _link_entities_to_episodes(self, session: AsyncSession, entities: List[StoredEntity]):
        """Create MENTIONED_IN relationships between entities and episodes"""
        try:
            query = """
            UNWIND $links AS link
            MATCH (ent:Entity {uuid: link.entity_uuid})
            MATCH (ep:Episode {uuid: link.episode_uuid})
            MERGE (ent)-[:MENTIONED_IN]->(ep)
            """
            
            links = []
            for entity in entities:
                for episode_uuid in entity.source_episodes:
                    links.append({
                        'entity_uuid': entity.uuid,
                        'episode_uuid': episode_uuid
                    })
            
            if links:
                await session.run(query, links=links)
                
        except Exception as e:
            self.logger.error(f"Failed to link entities to episodes: {e}")
    
    # Relationship Operations
    async def store_relationships(self, relationships: List[StoredRelationship]) -> int:
        """Store multiple relationships in the graph"""
        if not relationships:
            return 0
            
        async with self.driver.session(database=self.database) as session:
            stored_count = 0
            
            try:
                # Create relationships between entities
                query = """
                UNWIND $relationships AS rel
                MATCH (source:Entity {uuid: rel.source_uuid})
                MATCH (target:Entity {uuid: rel.target_uuid})
                MERGE (source)-[r:RELATES_TO {uuid: rel.uuid}]->(target)
                SET r += rel.properties
                RETURN count(r) as count
                """
                
                rel_data = []
                for rel in relationships:
                    rel_data.append({
                        'uuid': rel.uuid,
                        'source_uuid': rel.source_entity_uuid,
                        'target_uuid': rel.target_entity_uuid,
                        'properties': rel.to_neo4j_properties()
                    })
                
                result = await session.run(query, relationships=rel_data)
                record = await result.single()
                stored_count = record['count'] if record else 0
                
                self.logger.info(f"Stored {stored_count} relationships in Neo4j")
                return stored_count
                
            except Exception as e:
                self.logger.error(f"Failed to store relationships: {e}")
                return stored_count
    
    # Search Operations
    async def search_entities_by_name(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search entities by name using full-text search"""
        async with self.driver.session(database=self.database) as session:
            try:
                cypher_query = """
                CALL db.index.fulltext.queryNodes("entityNames", $query) 
                YIELD node, score
                MATCH (node)-[:MENTIONED_IN]->(ep:Episode)-[:CONTAINS]-(d:Document)
                RETURN node, score, ep, d
                ORDER BY score DESC
                LIMIT $limit
                """
                
                result = await session.run(cypher_query, query=query, limit=limit)
                
                search_results = []
                async for record in result:
                    entity = record['node']
                    episode = record['ep']
                    document = record['d']
                    score = record['score']
                    
                    search_result = SearchResult(
                        content=entity.get('summary', ''),
                        score=score,
                        result_type='entity',
                        source_uuid=entity['uuid'],
                        source_type='entity',
                        document_uuid=document['uuid'],
                        document_path=document['file_path'],
                        document_name=document['file_name'],
                        episode_uuid=episode['uuid'],
                        episode_sequence=episode.get('sequence_number'),
                        entity_name=entity['name'],
                        entity_type=entity['entity_type']
                    )
                    search_results.append(search_result)
                
                return search_results
                
            except Exception as e:
                self.logger.error(f"Entity search failed: {e}")
                return []
    
    async def search_episodes_by_content(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search episodes by content using full-text search"""
        async with self.driver.session(database=self.database) as session:
            try:
                cypher_query = """
                CALL db.index.fulltext.queryNodes("episodeContent", $query)
                YIELD node, score
                MATCH (d:Document)-[:CONTAINS]->(node)
                RETURN node, score, d
                ORDER BY score DESC
                LIMIT $limit
                """
                
                result = await session.run(cypher_query, query=query, limit=limit)
                
                search_results = []
                async for record in result:
                    episode = record['node']
                    document = record['d']
                    score = record['score']
                    
                    search_result = SearchResult(
                        content=episode['content'],
                        score=score,
                        result_type='episode',
                        source_uuid=episode['uuid'],
                        source_type='episode',
                        document_uuid=document['uuid'],
                        document_path=document['file_path'],
                        document_name=document['file_name'],
                        episode_uuid=episode['uuid'],
                        episode_sequence=episode.get('sequence_number')
                    )
                    search_results.append(search_result)
                
                return search_results
                
            except Exception as e:
                self.logger.error(f"Episode search failed: {e}")
                return []
    
    async def graph_traversal_search(
        self, 
        entity_names: List[str], 
        hops: int = 2, 
        limit: int = 10
    ) -> List[SearchResult]:
        """Search using graph traversal from given entities"""
        async with self.driver.session(database=self.database) as session:
            try:
                cypher_query = f"""
                MATCH (start:Entity)
                WHERE start.name IN $entity_names
                MATCH path = (start)-[:RELATES_TO*1..{hops}]-(related:Entity)
                MATCH (related)-[:MENTIONED_IN]->(ep:Episode)
                MATCH (d:Document)-[:CONTAINS]->(ep)
                WITH related, ep, d, length(path) as distance
                RETURN related, ep, d, distance
                ORDER BY distance ASC
                LIMIT $limit
                """
                
                result = await session.run(cypher_query, entity_names=entity_names, limit=limit)
                
                search_results = []
                async for record in result:
                    entity = record['related']
                    episode = record['ep']
                    document = record['d']
                    distance = record['distance']
                    
                    # Use inverse distance as score (closer = higher score)
                    score = 1.0 / (distance + 1)
                    
                    search_result = SearchResult(
                        content=entity.get('summary', ''),
                        score=score,
                        result_type='entity',
                        source_uuid=entity['uuid'],
                        source_type='entity_traversal',
                        document_uuid=document['uuid'],
                        document_path=document['file_path'],
                        document_name=document['file_name'],
                        episode_uuid=episode['uuid'],
                        episode_sequence=episode.get('sequence_number'),
                        entity_name=entity['name'],
                        entity_type=entity['entity_type'],
                        metadata={'distance': distance}
                    )
                    search_results.append(search_result)
                
                return search_results
                
            except Exception as e:
                self.logger.error(f"Graph traversal search failed: {e}")
                return []
    
    # Statistics and Management
    async def get_storage_stats(self) -> StorageStats:
        """Get statistics about stored knowledge"""
        async with self.driver.session(database=self.database) as session:
            try:
                # Get counts
                count_query = """
                MATCH (d:Document) WITH count(d) as docs
                MATCH (e:Episode) WITH docs, count(e) as episodes  
                MATCH (n:Entity) WITH docs, episodes, count(n) as entities
                MATCH ()-[r:RELATES_TO]->() WITH docs, episodes, entities, count(r) as relationships
                RETURN docs, episodes, entities, relationships
                """
                
                result = await session.run(count_query)
                record = await result.single()
                
                stats = StorageStats(
                    total_documents=record['docs'],
                    total_episodes=record['episodes'],
                    total_entities=record['entities'], 
                    total_relationships=record['relationships']
                )
                
                # Get entity type breakdown
                entity_type_query = """
                MATCH (n:Entity)
                RETURN n.entity_type as type, count(n) as count
                ORDER BY count DESC
                """
                
                result = await session.run(entity_type_query)
                async for record in result:
                    stats.entities_by_type[record['type']] = record['count']
                
                # Get relationship type breakdown
                rel_type_query = """
                MATCH ()-[r:RELATES_TO]->()
                RETURN r.relationship_type as type, count(r) as count
                ORDER BY count DESC
                """
                
                result = await session.run(rel_type_query)
                async for record in result:
                    stats.relationships_by_type[record['type']] = record['count']
                
                return stats
                
            except Exception as e:
                self.logger.error(f"Failed to get storage stats: {e}")
                return StorageStats()
    
    async def clear_all_data(self) -> bool:
        """Clear all data from the graph database"""
        async with self.driver.session(database=self.database) as session:
            try:
                await session.run("MATCH (n) DETACH DELETE n")
                self.logger.info("Cleared all data from Neo4j")
                return True
            except Exception as e:
                self.logger.error(f"Failed to clear data: {e}")
                return False 