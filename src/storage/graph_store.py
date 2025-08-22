"""
Neo4j Graph Store for Paper Extraction Storage System.

Handles storage and retrieval of extracted entities and relationships
from academic papers in Neo4j graph database.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

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
    ExtractedEntity, ExtractedRelationship, ExtractedPaper,
    SearchResult, StorageStats
)

logger = logging.getLogger(__name__)


class PaperGraphStore:
    """Neo4j graph database storage for extracted paper information."""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "gfll9999",
        database: str = "neo4j"
    ):
        """
        Initialize the graph store.
        
        Args:
            uri: Neo4j URI
            username: Neo4j username
            password: Neo4j password
            database: Database name
        """
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available. Install with: pip install neo4j")
        
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional[AsyncDriver] = None
        self.logger = logging.getLogger(__name__)
    
    async def connect(self):
        """Initialize connection to Neo4j."""
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
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            self.logger.info("Disconnected from Neo4j")
    
    async def _create_indexes_and_constraints(self):
        """Create necessary indexes and constraints."""
        async with self.driver.session(database=self.database) as session:
            constraints = [
                # Paper constraints
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.file_path IS UNIQUE",
                
                # Entity constraints
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type, e.paper_path) IS UNIQUE",
                
                # Indexes for performance
                "CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.document_type)",
                "CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.title)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id)",
                "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.paper_path)",
            ]
            
            # Create regular indexes first
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except ClientError as e:
                    if "already exists" not in str(e).lower():
                        self.logger.warning(f"Failed to create constraint/index: {e}")
            
            # Create full-text indexes with correct syntax
            fulltext_indexes = [
                ("entitySearch", "Entity", ["name", "description"]),
                ("paperSearch", "Paper", ["title", "file_path"]),
                ("chunkSearch", "Chunk", ["content"])
            ]
            
            for index_name, label, properties in fulltext_indexes:
                try:
                    # Check if index already exists
                    check_query = "SHOW INDEXES YIELD name WHERE name = $index_name RETURN count(*) as count"
                    result = await session.run(check_query, index_name=index_name)
                    record = await result.single()
                    
                    if record['count'] == 0:
                        # Create fulltext index
                        props_str = ", ".join([f"n.{prop}" for prop in properties])
                        create_query = f"CREATE FULLTEXT INDEX {index_name} FOR (n:{label}) ON EACH [{props_str}]"
                        await session.run(create_query)
                        self.logger.info(f"Created fulltext index: {index_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to create fulltext index {index_name}: {e}")
        
        self.logger.info("Created Neo4j indexes and constraints")
    
    async def store_paper(self, paper: ExtractedPaper) -> bool:
        """
        Store a complete extracted paper in the graph.
        
        Args:
            paper: Extracted paper data
            
        Returns:
            Success status
        """
        async with self.driver.session(database=self.database) as session:
            try:
                # Start transaction
                tx = await session.begin_transaction()
                
                # Create paper node
                paper_query = """
                MERGE (p:Paper {file_path: $file_path})
                SET p.title = $title,
                    p.document_type = $document_type,
                    p.schema_used = $schema_used,
                    p.extraction_timestamp = $extraction_timestamp,
                    p.metadata = $metadata
                RETURN p
                """
                
                await tx.run(paper_query, {
                    'file_path': paper.file_path,
                    'title': paper.title or paper.file_path.split('/')[-1],
                    'document_type': paper.document_type,
                    'schema_used': paper.schema_used,
                    'extraction_timestamp': paper.extraction_timestamp.isoformat(),
                    'metadata': json.dumps(paper.metadata)
                })
                
                # Store chunks if available
                if paper.chunks:
                    chunk_query = """
                    UNWIND $chunks AS chunk
                    MERGE (c:Chunk {chunk_id: chunk.chunk_id, paper_path: $paper_path})
                    SET c.content = chunk.content,
                        c.chunk_type = chunk.chunk_type,
                        c.word_count = chunk.word_count,
                        c.sequence = chunk.sequence,
                        c.summary = chunk.summary,
                        c.references = chunk.references,
                        c.section_title = chunk.section_title,
                        c.section_level = chunk.section_level
                    WITH c
                    MATCH (p:Paper {file_path: $paper_path})
                    MERGE (p)-[:HAS_CHUNK]->(c)
                    """
                    
                    chunks_data = [
                        {
                            'chunk_id': chunk.get('id', chunk.get('chunk_id', f"chunk_{i}")),
                            'content': chunk.get('content', ''),
                            'chunk_type': chunk.get('chunk_type', 'text'),
                            'word_count': chunk.get('word_count', 0),
                            'sequence': i,
                            'summary': chunk.get('summary', ''),
                            'references': json.dumps(chunk.get('references', [])) if chunk.get('references') else '[]',
                            'section_title': chunk.get('section_title', ''),
                            'section_level': chunk.get('section_level', 0)
                        }
                        for i, chunk in enumerate(paper.chunks)
                    ]
                    
                    await tx.run(chunk_query, {
                        'chunks': chunks_data,
                        'paper_path': paper.file_path
                    })
                
                # Store entities
                for entity in paper.entities:
                    entity_query = """
                    MERGE (e:Entity {name: $name, type: $type, paper_path: $paper_path})
                    SET e.description = $description,
                        e.confidence = $confidence,
                        e.attributes = $attributes,
                        e.context = $context,
                        e.chunk_id = $chunk_id,
                        e.created_at = $created_at
                    """
                    
                    await tx.run(entity_query, {
                        'name': entity.name,
                        'type': entity.type,
                        'description': entity.description,
                        'confidence': entity.confidence,
                        'attributes': json.dumps(entity.attributes),
                        'context': entity.context,
                        'chunk_id': entity.chunk_id,
                        'paper_path': paper.file_path,
                        'created_at': entity.created_at.isoformat()
                    })
                    
                    # Link entity to chunk if chunk_id exists
                    if entity.chunk_id:
                        chunk_link_query = """
                        MATCH (e:Entity {name: $name, type: $type, paper_path: $paper_path})
                        MATCH (c:Chunk {chunk_id: $chunk_id, paper_path: $paper_path})
                        MERGE (c)-[:CONTAINS_ENTITY]->(e)
                        """
                        
                        await tx.run(chunk_link_query, {
                            'name': entity.name,
                            'type': entity.type,
                            'chunk_id': entity.chunk_id,
                            'paper_path': paper.file_path
                        })
                    else:
                        # If no chunk_id, link directly to paper (fallback)
                        paper_link_query = """
                        MATCH (e:Entity {name: $name, type: $type, paper_path: $paper_path})
                        MATCH (p:Paper {file_path: $paper_path})
                        MERGE (p)-[:CONTAINS_ENTITY]->(e)
                        """
                        
                        await tx.run(paper_link_query, {
                            'name': entity.name,
                            'type': entity.type,
                            'paper_path': paper.file_path
                        })
                
                # Store relationships
                for rel in paper.relationships:
                    # Create relationship between entities
                    rel_query = """
                    MATCH (source:Entity {name: $source_name})
                    WHERE source.paper_path = $paper_path OR $source_name IN source.name
                    MATCH (target:Entity {name: $target_name})
                    WHERE target.paper_path = $paper_path OR $target_name IN target.name
                    MERGE (source)-[r:PAPER_RELATIONSHIP {type: $rel_type}]->(target)
                    SET r.confidence = $confidence,
                        r.properties = $properties,
                        r.context = $context,
                        r.chunk_id = $chunk_id,
                        r.paper_path = $paper_path,
                        r.created_at = $created_at
                    """
                    
                    await tx.run(rel_query, {
                        'source_name': rel.source_entity,
                        'target_name': rel.target_entity,
                        'rel_type': rel.relationship_type,
                        'confidence': rel.confidence,
                        'properties': json.dumps(rel.properties),
                        'context': rel.context,
                        'chunk_id': rel.chunk_id,
                        'paper_path': paper.file_path,
                        'created_at': rel.created_at.isoformat()
                    })
                
                # Commit transaction
                await tx.commit()
                
                self.logger.info(f"Stored paper: {paper.file_path} with {len(paper.entities)} entities and {len(paper.relationships)} relationships")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to store paper {paper.file_path}: {e}")
                return False
    
    async def search_entities(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search for entities using full-text search.
        
        Args:
            query: Search query
            entity_types: Filter by entity types
            limit: Maximum results
            
        Returns:
            List of search results
        """
        async with self.driver.session(database=self.database) as session:
            try:
                where_clause = ""
                if entity_types:
                    where_clause = f"WHERE node.type IN {entity_types}"
                
                cypher_query = f"""
                CALL db.index.fulltext.queryNodes("entitySearch", $search_text)
                YIELD node, score
                {where_clause}
                OPTIONAL MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(node)
                OPTIONAL MATCH (p:Paper)-[:HAS_CHUNK]->(c)
                OPTIONAL MATCH (p2:Paper)-[:CONTAINS_ENTITY]->(node)
                WITH node, score, COALESCE(p, p2) as paper
                WHERE paper IS NOT NULL
                RETURN node, score, paper
                ORDER BY score DESC
                LIMIT $limit
                """
                
                result = await session.run(cypher_query, search_text=query, limit=limit)
                
                search_results = []
                async for record in result:
                    entity = record['node']
                    paper = record['paper']
                    score = record['score']
                    
                    search_result = SearchResult(
                        content=entity.get('description', ''),
                        score=score,
                        result_type='entity',
                        source_file=paper['file_path'],
                        entity_name=entity['name'],
                        entity_type=entity['type'],
                        metadata={
                            'confidence': entity.get('confidence', 1.0),
                            'attributes': json.loads(entity.get('attributes', '{}')),
                            'context': entity.get('context'),
                            'paper_title': paper.get('title'),
                            'chunk_id': entity.get('chunk_id')
                        }
                    )
                    search_results.append(search_result)
                
                return search_results
                
            except Exception as e:
                self.logger.error(f"Entity search failed: {e}")
                return []
    
    async def find_relationships(
        self,
        entity_name: str,
        relationship_types: Optional[List[str]] = None,
        depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Find relationships connected to an entity.
        
        Args:
            entity_name: Name of the entity
            relationship_types: Filter by relationship types
            depth: Maximum traversal depth
            
        Returns:
            List of relationships with connected entities
        """
        async with self.driver.session(database=self.database) as session:
            try:
                type_filter = ""
                if relationship_types:
                    types_str = '|'.join(relationship_types)
                    type_filter = f"WHERE r.type IN {relationship_types}"
                
                cypher_query = f"""
                MATCH (start:Entity {{name: $entity_name}})
                MATCH path = (start)-[r:PAPER_RELATIONSHIP*1..{depth}]-(end:Entity)
                {type_filter}
                RETURN path, [rel in relationships(path) | rel.type] as rel_types,
                       [node in nodes(path) | {{name: node.name, type: node.type}}] as nodes
                LIMIT 50
                """
                
                result = await session.run(cypher_query, entity_name=entity_name)
                
                relationships = []
                async for record in result:
                    path_nodes = record['nodes']
                    rel_types = record['rel_types']
                    
                    relationships.append({
                        'path': path_nodes,
                        'relationship_types': rel_types,
                        'depth': len(rel_types)
                    })
                
                return relationships
                
            except Exception as e:
                self.logger.error(f"Relationship search failed: {e}")
                return []
    
    async def get_paper_chunks_with_entities(self, file_path: str) -> Dict[str, Any]:
        """
        Get all chunks for a paper and their associated entities.
        
        Args:
            file_path: Path to the paper file
            
        Returns:
            Dictionary with chunks and their entities
        """
        async with self.driver.session(database=self.database) as session:
            try:
                # Get all chunks for the paper with their entities
                query = """
                MATCH (p:Paper {file_path: $file_path})-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity)
                WITH c, COLLECT(DISTINCT {
                    name: e.name,
                    type: e.type,
                    description: e.description,
                    confidence: e.confidence
                }) as entities
                ORDER BY c.sequence
                RETURN c.chunk_id as chunk_id,
                       c.content as content,
                       c.chunk_type as chunk_type,
                       c.word_count as word_count,
                       c.sequence as sequence,
                       entities
                """
                
                result = await session.run(query, file_path=file_path)
                chunks = []
                
                async for record in result:
                    chunk_data = {
                        'chunk_id': record['chunk_id'],
                        'content': record['content'][:200] + '...' if len(record.get('content', '')) > 200 else record.get('content', ''),
                        'chunk_type': record.get('chunk_type', 'text'),
                        'word_count': record.get('word_count', 0),
                        'sequence': record['sequence'],
                        'entities': [e for e in record['entities'] if e.get('name')]  # Filter out empty entities
                    }
                    chunks.append(chunk_data)
                
                return {
                    'file_path': file_path,
                    'chunks': chunks,
                    'total_chunks': len(chunks),
                    'total_entities': sum(len(c['entities']) for c in chunks)
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get paper chunks with entities: {e}")
                return {}
    
    async def get_paper_graph(self, file_path: str) -> Dict[str, Any]:
        """
        Get the complete graph for a specific paper.
        
        Args:
            file_path: Path to the paper file
            
        Returns:
            Dictionary with entities and relationships
        """
        async with self.driver.session(database=self.database) as session:
            try:
                # Get paper and all its entities (through chunks or directly)
                entity_query = """
                MATCH (p:Paper {file_path: $file_path})
                OPTIONAL MATCH (p)-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e1:Entity)
                OPTIONAL MATCH (p)-[:CONTAINS_ENTITY]->(e2:Entity)
                WITH COLLECT(DISTINCT e1) + COLLECT(DISTINCT e2) as entities
                UNWIND entities as e
                RETURN DISTINCT e
                """
                
                entity_result = await session.run(entity_query, file_path=file_path)
                entities = []
                async for record in entity_result:
                    entity = record['e']
                    entities.append({
                        'name': entity['name'],
                        'type': entity['type'],
                        'description': entity.get('description'),
                        'attributes': json.loads(entity.get('attributes', '{}'))
                    })
                
                # Get all relationships between entities in this paper
                rel_query = """
                MATCH (p:Paper {file_path: $file_path})
                OPTIONAL MATCH (p)-[:HAS_CHUNK]->(c1:Chunk)-[:CONTAINS_ENTITY]->(e1:Entity)
                OPTIONAL MATCH (p)-[:HAS_CHUNK]->(c2:Chunk)-[:CONTAINS_ENTITY]->(e2:Entity)
                OPTIONAL MATCH (p)-[:CONTAINS_ENTITY]->(e3:Entity)
                OPTIONAL MATCH (p)-[:CONTAINS_ENTITY]->(e4:Entity)
                WITH COLLECT(DISTINCT e1) + COLLECT(DISTINCT e3) as source_entities,
                     COLLECT(DISTINCT e2) + COLLECT(DISTINCT e4) as target_entities
                UNWIND source_entities as se
                UNWIND target_entities as te
                MATCH (se)-[r:PAPER_RELATIONSHIP]->(te)
                WHERE r.paper_path = $file_path
                RETURN se.name as source, te.name as target, r
                """
                
                rel_result = await session.run(rel_query, file_path=file_path)
                relationships = []
                async for record in rel_result:
                    rel = record['r']
                    relationships.append({
                        'source': record['source'],
                        'target': record['target'],
                        'type': rel['type'],
                        'properties': json.loads(rel.get('properties', '{}')),
                        'context': rel.get('context')
                    })
                
                return {
                    'file_path': file_path,
                    'entities': entities,
                    'relationships': relationships,
                    'entity_count': len(entities),
                    'relationship_count': len(relationships)
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get paper graph: {e}")
                return {}
    
    async def get_statistics(self) -> StorageStats:
        """
        Get storage statistics.
        
        Returns:
            Storage statistics
        """
        async with self.driver.session(database=self.database) as session:
            try:
                # Get counts - count each type separately to avoid cartesian products
                count_query = """
                CALL {
                    MATCH (p:Paper) RETURN count(p) as papers
                }
                CALL {
                    MATCH (e:Entity) RETURN count(e) as entities
                }
                CALL {
                    MATCH ()-[r:PAPER_RELATIONSHIP]->() RETURN count(r) as relationships
                }
                CALL {
                    OPTIONAL MATCH (c:Chunk) RETURN count(c) as chunks
                }
                RETURN papers, entities, relationships, chunks
                """
                
                result = await session.run(count_query)
                record = await result.single()
                
                stats = StorageStats(
                    total_papers=record['papers'] if record else 0,
                    total_entities=record['entities'] if record else 0,
                    total_relationships=record['relationships'] if record else 0,
                    total_chunks=record['chunks'] if record else 0
                )
                
                # Get entity type breakdown
                entity_type_query = """
                MATCH (e:Entity)
                RETURN e.type as type, count(e) as count
                ORDER BY count DESC
                """
                
                result = await session.run(entity_type_query)
                async for record in result:
                    stats.entities_by_type[record['type']] = record['count']
                
                # Get relationship type breakdown
                rel_type_query = """
                MATCH ()-[r:PAPER_RELATIONSHIP]->()
                RETURN r.type as type, count(r) as count
                ORDER BY count DESC
                """
                
                result = await session.run(rel_type_query)
                async for record in result:
                    stats.relationships_by_type[record['type']] = record['count']
                
                # Get paper type breakdown
                paper_type_query = """
                MATCH (p:Paper)
                RETURN p.document_type as type, count(p) as count
                ORDER BY count DESC
                """
                
                result = await session.run(paper_type_query)
                async for record in result:
                    stats.papers_by_type[record['type']] = record['count']
                
                return stats
                
            except Exception as e:
                self.logger.error(f"Failed to get statistics: {e}")
                return StorageStats()
    
    async def clear_all_data(self) -> bool:
        """
        Clear all data from the graph database.
        
        Returns:
            Success status
        """
        async with self.driver.session(database=self.database) as session:
            try:
                await session.run("MATCH (n) DETACH DELETE n")
                self.logger.info("Cleared all data from Neo4j")
                return True
            except Exception as e:
                self.logger.error(f"Failed to clear data: {e}")
                return False