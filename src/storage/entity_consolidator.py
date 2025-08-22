"""
Global Entity Consolidator for Cross-Document Deduplication.

This module provides functionality to consolidate entity variants across multiple
documents using semantic similarity while preserving full context traceability.
"""

import logging
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict
import numpy as np
from rapidfuzz import fuzz
import asyncio

from .models import ExtractedEntity, SearchResult
from .graph_store import PaperGraphStore
from .vector_store import PaperVectorStore

logger = logging.getLogger(__name__)


class EntityConsolidator:
    """
    Consolidates entity variants across documents using semantic similarity.
    
    This class identifies and merges duplicate entities that appear across
    different documents while preserving full context and occurrence information.
    """
    
    def __init__(
        self,
        graph_store: PaperGraphStore,
        vector_store: PaperVectorStore,
        llm_client: Optional[Any] = None,
        semantic_threshold: float = 0.85,
        levenshtein_threshold: float = 0.8,
        batch_size: int = 100
    ):
        """
        Initialize the entity consolidator.
        
        Args:
            graph_store: Neo4j graph store instance
            vector_store: ChromaDB vector store instance
            llm_client: OpenAI client for description synthesis
            semantic_threshold: Minimum cosine similarity for semantic matching
            levenshtein_threshold: Minimum string similarity for name matching
            batch_size: Number of entities to process in each batch
        """
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.semantic_threshold = semantic_threshold
        self.levenshtein_threshold = levenshtein_threshold
        self.batch_size = batch_size
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def consolidate_entities_globally(
        self,
        dry_run: bool = False,
        entity_types: Optional[List[str]] = None,
        exclude_consolidated: bool = True
    ) -> Dict[str, Any]:
        """
        Consolidate entity variants across all documents.
        
        Args:
            dry_run: If True, only identify duplicates without modifying data
            entity_types: Specific entity types to consolidate (None for all)
            exclude_consolidated: Skip already consolidated entities
            
        Returns:
            Dictionary with consolidation results and statistics
        """
        self.logger.info("Starting global entity consolidation...")
        
        # Get all unique entities from Neo4j
        all_entities = await self._get_all_entities(entity_types, exclude_consolidated)
        self.logger.info(f"Found {len(all_entities)} entities to analyze")
        
        # Find semantic duplicate groups
        entity_groups = await self._find_semantic_duplicates(all_entities)
        self.logger.info(f"Identified {len(entity_groups)} potential duplicate groups")
        
        if dry_run:
            return {
                'mode': 'dry_run',
                'potential_groups': len(entity_groups),
                'entities_to_consolidate': sum(len(g) for g in entity_groups),
                'groups': [
                    {
                        'canonical': self._select_canonical_entity(group)['name'],
                        'variants': [e['name'] for e in group]
                    }
                    for group in entity_groups
                ],
                'timestamp': datetime.now().isoformat()
            }
        
        # Perform actual consolidation
        consolidated_count = 0
        failed_count = 0
        consolidation_details = []
        
        for group_idx, group in enumerate(entity_groups, 1):
            if len(group) < 2:
                continue
            
            self.logger.info(f"Processing group {group_idx}/{len(entity_groups)}: {len(group)} entities")
            
            try:
                result = await self._consolidate_entity_group(group)
                consolidated_count += 1
                consolidation_details.append(result)
            except Exception as e:
                self.logger.error(f"Failed to consolidate group {group_idx}: {e}")
                failed_count += 1
        
        # Update Neo4j indices for consolidated entities
        if consolidated_count > 0:
            await self._update_consolidation_indices()
        
        return {
            'mode': 'executed',
            'entities_consolidated': consolidated_count,
            'groups_failed': failed_count,
            'total_groups_processed': len(entity_groups),
            'consolidation_details': consolidation_details[:10],  # First 10 for summary
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_all_entities(
        self,
        entity_types: Optional[List[str]] = None,
        exclude_consolidated: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all entities from Neo4j.
        
        Args:
            entity_types: Filter by entity types
            exclude_consolidated: Exclude already consolidated entities
            
        Returns:
            List of entity dictionaries
        """
        query = """
        MATCH (e:Entity)
        WHERE ($entity_types IS NULL OR e.type IN $entity_types)
        AND ($exclude_consolidated = false OR e.is_consolidated IS NULL OR e.is_consolidated = false)
        RETURN e.name as name, e.type as type, e.description as description,
               e.paper_path as paper_path, e.context as context, e.chunk_id as chunk_id,
               e.confidence as confidence, e.created_at as created_at,
               e.attributes as attributes
        """
        
        async with self.graph_store.driver.session() as session:
            result = await session.run(
                query,
                entity_types=entity_types,
                exclude_consolidated=exclude_consolidated
            )
            entities = []
            async for record in result:
                entities.append(dict(record))
            return entities
    
    async def _find_semantic_duplicates(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Find groups of semantically similar entities.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            List of entity groups (each group contains similar entities)
        """
        groups = []
        processed = set()
        
        # Process entities in batches for efficiency
        for i in range(0, len(entities), self.batch_size):
            batch = entities[i:i + self.batch_size]
            
            for entity in batch:
                if entity['name'] in processed:
                    continue
                
                # Find similar entities using vector search
                similar_entities = await self._find_similar_entities(entity, entities)
                
                if len(similar_entities) > 1:  # Include the entity itself
                    group = [entity] + similar_entities
                    groups.append(group)
                    
                    # Mark all entities in group as processed
                    for e in group:
                        processed.add(e['name'])
        
        return groups
    
    async def _find_similar_entities(
        self,
        entity: Dict[str, Any],
        all_entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find entities similar to the given entity.
        
        Args:
            entity: Target entity
            all_entities: Pool of entities to search
            
        Returns:
            List of similar entities
        """
        similar = []
        
        # Use semantic search to find candidates
        search_query = f"{entity['name']}: {entity.get('description', '')}"
        semantic_results = await self.vector_store.search_entities_semantic(
            query=search_query,
            entity_types=[entity['type']] if entity.get('type') else None,
            limit=20
        )
        
        # Get entity names from semantic results
        semantic_names = {
            result.entity_name for result in semantic_results
            if result.score >= self.semantic_threshold
        }
        
        # Filter with Levenshtein distance
        for other_entity in all_entities:
            if other_entity['name'] == entity['name']:
                continue
            
            # Check if in semantic results
            if other_entity['name'] not in semantic_names:
                continue
            
            # Calculate string similarity
            name_similarity = fuzz.ratio(
                entity['name'].lower(),
                other_entity['name'].lower()
            ) / 100.0
            
            if name_similarity >= self.levenshtein_threshold:
                similar.append(other_entity)
        
        return similar
    
    def _select_canonical_entity(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the canonical entity from a group.
        
        Selection criteria:
        1. Highest confidence score
        2. Longest name (more descriptive)
        3. Most recent extraction
        
        Args:
            group: List of similar entities
            
        Returns:
            The selected canonical entity
        """
        return max(group, key=lambda e: (
            e.get('confidence', 0),
            len(e.get('name', '')),
            e.get('created_at', '')
        ))
    
    async def _consolidate_entity_group(
        self,
        group: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Consolidate a group of similar entities into one.
        
        Args:
            group: List of similar entities to consolidate
            
        Returns:
            Consolidation result dictionary
        """
        # Select canonical entity
        canonical = self._select_canonical_entity(group)
        
        # Collect all occurrences with full context
        all_occurrences = []
        all_descriptions = []
        all_paper_paths = set()
        aliases = set()
        
        for entity in group:
            # Track aliases (different names for same entity)
            if entity['name'] != canonical['name']:
                aliases.add(entity['name'])
            
            # Collect occurrence information
            occurrence = {
                'context': entity.get('context', ''),
                'chunk_id': entity.get('chunk_id', ''),
                'paper_path': entity.get('paper_path', ''),
                'confidence': entity.get('confidence', 0.5),
                'original_name': entity['name'],
                'extraction_date': entity.get('created_at', datetime.now().isoformat())
            }
            
            # Get paper title if available
            if entity.get('paper_path'):
                paper_info = await self._get_paper_info(entity['paper_path'])
                occurrence['paper_title'] = paper_info.get('title', '')
                all_paper_paths.add(entity['paper_path'])
            
            all_occurrences.append(occurrence)
            
            if entity.get('description'):
                all_descriptions.append(entity['description'])
        
        # Synthesize description using LLM (if available)
        if self.llm_client and all_descriptions:
            merged_description = await self._llm_synthesize_description(
                canonical['name'],
                canonical['type'],
                all_descriptions
            )
        else:
            # Fallback: concatenate unique descriptions
            unique_descriptions = list(set(all_descriptions))
            merged_description = " | ".join(unique_descriptions)
        
        # Create consolidated entity data
        consolidated_entity = {
            'name': canonical['name'],
            'type': canonical.get('type', 'UNKNOWN'),
            'aliases': list(aliases),
            'description': merged_description,
            'occurrences': all_occurrences,
            'occurrence_count': len(all_occurrences),
            'document_count': len(all_paper_paths),
            'is_consolidated': True,
            'consolidation_date': datetime.now().isoformat(),
            'consolidated_from': [e['name'] for e in group],
            'confidence': max(e.get('confidence', 0) for e in group)
        }
        
        # Update databases
        await self._update_consolidated_entity(consolidated_entity, group)
        
        return {
            'canonical_name': canonical['name'],
            'aliases': list(aliases),
            'occurrences': len(all_occurrences),
            'documents': len(all_paper_paths),
            'entities_merged': len(group)
        }
    
    async def _get_paper_info(self, paper_path: str) -> Dict[str, Any]:
        """
        Get paper information from Neo4j.
        
        Args:
            paper_path: Path to the paper
            
        Returns:
            Paper information dictionary
        """
        query = """
        MATCH (p:Paper {file_path: $paper_path})
        RETURN p.title as title, p.document_type as document_type
        """
        
        async with self.graph_store.driver.session() as session:
            result = await session.run(query, paper_path=paper_path)
            record = await result.single()
            if record:
                return dict(record)
            return {}
    
    async def _llm_synthesize_description(
        self,
        entity_name: str,
        entity_type: str,
        descriptions: List[str]
    ) -> str:
        """
        Use LLM to synthesize multiple descriptions into one comprehensive description.
        
        Args:
            entity_name: Name of the entity
            entity_type: Type of the entity
            descriptions: List of descriptions to synthesize
            
        Returns:
            Synthesized description
        """
        if not self.llm_client:
            return " | ".join(set(descriptions))
        
        # Prepare prompt
        descriptions_text = "\n".join([f"- {desc}" for desc in set(descriptions)])
        
        prompt = f"""
        Synthesize the following descriptions of {entity_type} entity "{entity_name}" into a single, 
        comprehensive description that captures all important information:
        
        {descriptions_text}
        
        Create a 2-3 sentence description that:
        1. Combines all unique information
        2. Resolves any contradictions by noting different perspectives
        3. Maintains factual accuracy
        
        Synthesized description:
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"LLM synthesis failed: {e}")
            return " | ".join(set(descriptions))
    
    async def _update_consolidated_entity(
        self,
        consolidated_entity: Dict[str, Any],
        original_entities: List[Dict[str, Any]]
    ):
        """
        Update both Neo4j and ChromaDB with the consolidated entity.
        
        Args:
            consolidated_entity: The new consolidated entity data
            original_entities: List of original entities being consolidated
        """
        async with self.graph_store.driver.session() as session:
            # Start a transaction
            tx = await session.begin_transaction()
            
            try:
                # 1. Create the consolidated entity node
                create_query = """
                MERGE (e:Entity {name: $name})
                SET e.type = $type,
                    e.description = $description,
                    e.aliases = $aliases,
                    e.is_consolidated = true,
                    e.occurrence_count = $occurrence_count,
                    e.document_count = $document_count,
                    e.consolidation_date = $consolidation_date,
                    e.confidence = $confidence,
                    e.paper_path = 'CONSOLIDATED'
                RETURN e
                """
                
                await tx.run(create_query, **{
                    'name': consolidated_entity['name'],
                    'type': consolidated_entity['type'],
                    'description': consolidated_entity['description'],
                    'aliases': consolidated_entity['aliases'],
                    'occurrence_count': consolidated_entity['occurrence_count'],
                    'document_count': consolidated_entity['document_count'],
                    'consolidation_date': consolidated_entity['consolidation_date'],
                    'confidence': consolidated_entity['confidence']
                })
                
                # 2. Create occurrence nodes for each context
                for occurrence in consolidated_entity['occurrences']:
                    occurrence_query = """
                    CREATE (o:Occurrence {
                        entity_name: $entity_name,
                        context: $context,
                        chunk_id: $chunk_id,
                        paper_path: $paper_path,
                        paper_title: $paper_title,
                        confidence: $confidence,
                        original_name: $original_name,
                        extraction_date: $extraction_date
                    })
                    WITH o
                    MATCH (e:Entity {name: $entity_name})
                    CREATE (e)-[:HAS_OCCURRENCE]->(o)
                    """
                    
                    await tx.run(occurrence_query, **{
                        'entity_name': consolidated_entity['name'],
                        'context': occurrence['context'],
                        'chunk_id': occurrence['chunk_id'],
                        'paper_path': occurrence['paper_path'],
                        'paper_title': occurrence.get('paper_title', ''),
                        'confidence': occurrence['confidence'],
                        'original_name': occurrence['original_name'],
                        'extraction_date': occurrence['extraction_date']
                    })
                
                # 3. Transfer relationships from original entities
                for original in original_entities:
                    if original['name'] == consolidated_entity['name']:
                        continue  # Skip if same as canonical
                    
                    # Transfer outgoing relationships
                    transfer_out_query = """
                    MATCH (old:Entity {name: $old_name})-[r:PAPER_RELATIONSHIP]->(target)
                    MATCH (new:Entity {name: $new_name})
                    MERGE (new)-[r2:PAPER_RELATIONSHIP {type: r.type}]->(target)
                    SET r2 += properties(r)
                    DELETE r
                    """
                    
                    await tx.run(transfer_out_query, 
                                old_name=original['name'],
                                new_name=consolidated_entity['name'])
                    
                    # Transfer incoming relationships
                    transfer_in_query = """
                    MATCH (source)-[r:PAPER_RELATIONSHIP]->(old:Entity {name: $old_name})
                    MATCH (new:Entity {name: $new_name})
                    MERGE (source)-[r2:PAPER_RELATIONSHIP {type: r.type}]->(new)
                    SET r2 += properties(r)
                    DELETE r
                    """
                    
                    await tx.run(transfer_in_query,
                                old_name=original['name'],
                                new_name=consolidated_entity['name'])
                    
                    # Delete the old entity
                    delete_query = """
                    MATCH (e:Entity {name: $name})
                    DETACH DELETE e
                    """
                    
                    await tx.run(delete_query, name=original['name'])
                
                # Commit the transaction
                await tx.commit()
                
                # 4. Update ChromaDB
                await self._update_vector_embeddings(consolidated_entity, original_entities)
                
            except Exception as e:
                await tx.rollback()
                raise e
    
    async def _update_vector_embeddings(
        self,
        consolidated_entity: Dict[str, Any],
        original_entities: List[Dict[str, Any]]
    ):
        """
        Update vector embeddings in ChromaDB.
        
        Args:
            consolidated_entity: The new consolidated entity
            original_entities: Original entities to remove
        """
        # Generate IDs for deletion
        old_ids = []
        for entity in original_entities:
            entity_id = hashlib.md5(
                f"{entity['name']}:{entity.get('type', '')}:{entity.get('paper_path', '')}".encode()
            ).hexdigest()
            old_ids.append(entity_id)
        
        # Delete old embeddings
        if old_ids:
            try:
                self.vector_store.entities_collection.delete(ids=old_ids)
                self.logger.info(f"Deleted {len(old_ids)} old entity embeddings")
            except Exception as e:
                self.logger.warning(f"Could not delete some old embeddings: {e}")
        
        # Create new embedding for consolidated entity
        entity_text = f"{consolidated_entity['name']}: {consolidated_entity['description']}"
        
        # Add aliases to the text for better searchability
        if consolidated_entity['aliases']:
            aliases_text = ", ".join(consolidated_entity['aliases'])
            entity_text += f" (also known as: {aliases_text})"
        
        new_embedding = self.vector_store._create_embedding(entity_text)
        
        if new_embedding:
            new_id = hashlib.md5(
                f"{consolidated_entity['name']}:CONSOLIDATED".encode()
            ).hexdigest()
            
            metadata = {
                'entity_name': consolidated_entity['name'],
                'entity_type': consolidated_entity['type'],
                'description': consolidated_entity['description'],
                'aliases': json.dumps(consolidated_entity['aliases']),
                'is_consolidated': True,
                'occurrence_count': consolidated_entity['occurrence_count'],
                'document_count': consolidated_entity['document_count'],
                'confidence': consolidated_entity['confidence'],
                'paper_path': 'CONSOLIDATED',
                'created_at': consolidated_entity['consolidation_date']
            }
            
            self.vector_store.entities_collection.upsert(
                embeddings=[new_embedding],
                documents=[entity_text],
                metadatas=[metadata],
                ids=[new_id]
            )
            
            self.logger.info(f"Created new embedding for consolidated entity: {consolidated_entity['name']}")
    
    async def _update_consolidation_indices(self):
        """
        Update Neo4j indices for better query performance on consolidated entities.
        """
        async with self.graph_store.driver.session() as session:
            indices = [
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.is_consolidated)",
                "CREATE INDEX IF NOT EXISTS FOR (o:Occurrence) ON (o.entity_name)",
                "CREATE INDEX IF NOT EXISTS FOR (o:Occurrence) ON (o.paper_path)",
                "CREATE INDEX IF NOT EXISTS FOR (o:Occurrence) ON (o.chunk_id)"
            ]
            
            for index_query in indices:
                try:
                    await session.run(index_query)
                except Exception as e:
                    self.logger.debug(f"Index might already exist: {e}")
    
    async def get_entity_occurrences(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        Get all occurrences of a consolidated entity.
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            List of occurrence dictionaries
        """
        query = """
        MATCH (e:Entity {name: $name})-[:HAS_OCCURRENCE]->(o:Occurrence)
        RETURN o.context as context, o.chunk_id as chunk_id, 
               o.paper_path as paper_path, o.paper_title as paper_title,
               o.confidence as confidence, o.original_name as original_name,
               o.extraction_date as extraction_date
        ORDER BY o.extraction_date DESC
        """
        
        async with self.graph_store.driver.session() as session:
            result = await session.run(query, name=entity_name)
            occurrences = []
            async for record in result:
                occurrences.append(dict(record))
            return occurrences
    
    async def undo_consolidation(self, entity_name: str) -> Dict[str, Any]:
        """
        Undo a consolidation and restore original entities.
        
        Args:
            entity_name: Name of the consolidated entity to undo
            
        Returns:
            Result dictionary with undo statistics
        """
        # This is a complex operation - implement if needed
        # Would need to store original entity data to properly restore
        raise NotImplementedError("Undo consolidation not yet implemented")