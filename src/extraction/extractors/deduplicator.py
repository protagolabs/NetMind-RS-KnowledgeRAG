"""
Deduplicator module for handling entity and relationship deduplication.

This module provides functionality to deduplicate extracted entities and
relationships, merging attributes and maintaining the highest confidence scores.
"""

import logging
from typing import List, Dict, Any, Set, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DeduplicationStats(BaseModel):
    """Statistics about the deduplication process."""
    
    original_count: int = Field(description="Original number of items")
    unique_count: int = Field(description="Number after deduplication")
    duplicates_removed: int = Field(description="Number of duplicates removed")
    merge_count: int = Field(description="Number of items merged")


class Deduplicator:
    """
    Handles deduplication of entities and relationships.
    
    Provides methods to identify and merge duplicate entities/relationships
    based on configurable matching criteria.
    """
    
    def __init__(
        self,
        case_sensitive: bool = False,
        merge_attributes: bool = True,
        keep_all_contexts: bool = False
    ):
        """
        Initialize the deduplicator.
        
        Args:
            case_sensitive: Whether to use case-sensitive matching
            merge_attributes: Whether to merge attributes from duplicates
            keep_all_contexts: Whether to keep all context strings
        """
        self.case_sensitive = case_sensitive
        self.merge_attributes = merge_attributes
        self.keep_all_contexts = keep_all_contexts
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def deduplicate_entities(
        self,
        entities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], DeduplicationStats]:
        """
        Deduplicate entities by name and type.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            Tuple of (deduplicated entities, statistics)
        """
        seen = {}
        unique = []
        original_count = len(entities)
        merge_count = 0
        
        for entity in entities:
            # Create deduplication key
            name = entity.get("name", "")
            if not self.case_sensitive:
                name = name.lower()
            key = (name, entity.get("type", ""))
            
            if key not in seen:
                # First occurrence - keep it
                seen[key] = entity.copy()
                unique.append(seen[key])
                
                # Initialize contexts list if keeping all
                if self.keep_all_contexts and "context" in seen[key]:
                    seen[key]["contexts"] = [seen[key]["context"]]
                    seen[key]["chunk_ids"] = [entity.get("chunk_id", "unknown")]
            else:
                # Duplicate found - merge
                merge_count += 1
                existing = seen[key]
                
                # Take highest confidence
                existing["confidence"] = max(
                    existing.get("confidence", 0),
                    entity.get("confidence", 0)
                )
                
                # Merge attributes if enabled
                if self.merge_attributes and "attributes" in entity:
                    existing.setdefault("attributes", {}).update(entity["attributes"])
                
                # Handle contexts
                if self.keep_all_contexts:
                    if "context" in entity and entity["context"]:
                        existing.setdefault("contexts", []).append(entity["context"])
                        existing.setdefault("chunk_ids", []).append(
                            entity.get("chunk_id", "unknown")
                        )
                else:
                    # Keep the context with highest confidence
                    if entity.get("confidence", 0) > existing.get("confidence", 0):
                        existing["context"] = entity.get("context", "")
                        existing["chunk_id"] = entity.get("chunk_id", "unknown")
                
                # Merge description if longer
                if len(entity.get("description", "")) > len(existing.get("description", "")):
                    existing["description"] = entity["description"]
        
        stats = DeduplicationStats(
            original_count=original_count,
            unique_count=len(unique),
            duplicates_removed=original_count - len(unique),
            merge_count=merge_count
        )
        
        self.logger.info(f"Entity deduplication: {original_count} -> {len(unique)} "
                        f"({stats.duplicates_removed} duplicates removed)")
        
        return unique, stats
    
    def deduplicate_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], DeduplicationStats]:
        """
        Deduplicate relationships by source, target, and type.
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            Tuple of (deduplicated relationships, statistics)
        """
        seen = set()
        unique = []
        original_count = len(relationships)
        merge_count = 0
        merged_rels = {}
        
        for rel in relationships:
            # Create deduplication key
            source = rel.get("source", "")
            target = rel.get("target", "")
            if not self.case_sensitive:
                source = source.lower()
                target = target.lower()
            key = (source, target, rel.get("type", ""))
            
            if key not in seen:
                # First occurrence
                seen.add(key)
                rel_copy = rel.copy()
                merged_rels[key] = rel_copy
                unique.append(rel_copy)
                
                # Initialize evidence list if keeping all contexts
                if self.keep_all_contexts and "evidence" in rel_copy:
                    rel_copy["all_evidence"] = [rel_copy["evidence"]]
                    rel_copy["chunk_ids"] = [rel.get("chunk_id", "unknown")]
            else:
                # Duplicate found - merge
                merge_count += 1
                existing = merged_rels[key]
                
                # Take highest confidence
                existing["confidence"] = max(
                    existing.get("confidence", 0),
                    rel.get("confidence", 0)
                )
                
                # Merge properties
                if "properties" in rel:
                    existing.setdefault("properties", {}).update(rel["properties"])
                
                # Handle evidence/context
                if self.keep_all_contexts:
                    if "evidence" in rel and rel["evidence"]:
                        existing.setdefault("all_evidence", []).append(rel["evidence"])
                        existing.setdefault("chunk_ids", []).append(
                            rel.get("chunk_id", "unknown")
                        )
                else:
                    # Keep evidence with highest confidence
                    if rel.get("confidence", 0) > existing.get("confidence", 0):
                        existing["evidence"] = rel.get("evidence", "")
                        existing["context"] = rel.get("context", "")
                        existing["chunk_id"] = rel.get("chunk_id", "unknown")
        
        stats = DeduplicationStats(
            original_count=original_count,
            unique_count=len(unique),
            duplicates_removed=original_count - len(unique),
            merge_count=merge_count
        )
        
        self.logger.info(f"Relationship deduplication: {original_count} -> {len(unique)} "
                        f"({stats.duplicates_removed} duplicates removed)")
        
        return unique, stats
    
    def find_similar_entities(
        self,
        entities: List[Dict[str, Any]],
        similarity_threshold: float = 0.8
    ) -> List[List[Dict[str, Any]]]:
        """
        Find groups of similar entities (fuzzy matching).
        
        Args:
            entities: List of entity dictionaries
            similarity_threshold: Similarity threshold (0-1)
            
        Returns:
            List of entity groups that might be duplicates
        """
        # Simple implementation - can be enhanced with fuzzy matching
        groups = []
        processed = set()
        
        for i, entity1 in enumerate(entities):
            if i in processed:
                continue
                
            group = [entity1]
            name1 = entity1.get("name", "").lower()
            type1 = entity1.get("type", "")
            
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if j in processed:
                    continue
                    
                name2 = entity2.get("name", "").lower()
                type2 = entity2.get("type", "")
                
                # Simple similarity check (can be enhanced)
                if type1 == type2:
                    # Check for common patterns
                    if (name1 in name2 or name2 in name1 or
                        name1.replace("-", "") == name2.replace("-", "") or
                        name1.replace("_", "") == name2.replace("_", "") or
                        name1.replace(" ", "") == name2.replace(" ", "")):
                        group.append(entity2)
                        processed.add(j)
            
            if len(group) > 1:
                groups.append(group)
                processed.add(i)
        
        return groups
    
    def merge_entity_group(
        self,
        entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge a group of similar entities into one.
        
        Args:
            entities: List of similar entities to merge
            
        Returns:
            Merged entity dictionary
        """
        if not entities:
            return {}
        
        # Start with the entity with highest confidence
        entities_sorted = sorted(entities, key=lambda x: x.get("confidence", 0), reverse=True)
        merged = entities_sorted[0].copy()
        
        # Merge attributes from all entities
        all_attributes = {}
        all_contexts = []
        all_chunk_ids = []
        
        for entity in entities:
            # Merge attributes
            if "attributes" in entity:
                all_attributes.update(entity["attributes"])
            
            # Collect contexts
            if "context" in entity and entity["context"]:
                all_contexts.append(entity["context"])
            
            # Collect chunk IDs
            if "chunk_id" in entity:
                all_chunk_ids.append(entity["chunk_id"])
        
        # Update merged entity
        if all_attributes:
            merged["attributes"] = all_attributes
        
        if self.keep_all_contexts:
            merged["contexts"] = all_contexts
            merged["chunk_ids"] = list(set(all_chunk_ids))  # Unique chunk IDs
        
        # Take the longest description
        longest_desc = max(entities, key=lambda x: len(x.get("description", "")))
        merged["description"] = longest_desc.get("description", "")
        
        return merged