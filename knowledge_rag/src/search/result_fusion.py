"""
Result Fusion for Knowledge RAG Search System

Combines and ranks results from multiple search strategies using
various fusion techniques like RRF, weighted combination, and graph centrality.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import math

from .search_models import SearchResults, RankingStrategy
from ..storage.models import SearchResult

logger = logging.getLogger(__name__)


class ResultFusion:
    """Fuses and ranks results from multiple search strategies"""
    
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.logger = logging.getLogger(__name__)
    
    def fuse_results(
        self,
        result_sets: Dict[str, List[SearchResult]],
        strategy: RankingStrategy = RankingStrategy.RRF,
        weights: Optional[Dict[str, float]] = None,
        limit: int = 10
    ) -> SearchResults:
        """
        Fuse multiple result sets using specified strategy
        
        Args:
            result_sets: Dictionary mapping search type to results
            strategy: Fusion strategy to use
            weights: Weights for each search type (for weighted fusion)
            limit: Maximum number of results to return
        
        Returns:
            Fused SearchResults
        """
        
        if not result_sets:
            return SearchResults()
        
        # Choose fusion method
        if strategy == RankingStrategy.RRF:
            fused_results = self._reciprocal_rank_fusion(result_sets, limit)
        elif strategy == RankingStrategy.WEIGHTED_COMBINATION:
            fused_results = self._weighted_combination(result_sets, weights or {}, limit)
        elif strategy == RankingStrategy.RELEVANCE_SCORE:
            fused_results = self._relevance_score_fusion(result_sets, limit)
        elif strategy == RankingStrategy.GRAPH_CENTRALITY:
            fused_results = self._graph_centrality_fusion(result_sets, limit)
        elif strategy == RankingStrategy.TEMPORAL_RECENCY:
            fused_results = self._temporal_recency_fusion(result_sets, limit)
        else:
            # Default to RRF
            fused_results = self._reciprocal_rank_fusion(result_sets, limit)
        
        # Set fusion metadata
        fused_results.fusion_method = strategy.value
        fused_results.search_strategy_used = list(result_sets.keys())
        
        return fused_results
    
    def _reciprocal_rank_fusion(
        self, 
        result_sets: Dict[str, List[SearchResult]], 
        limit: int
    ) -> SearchResults:
        """
        Reciprocal Rank Fusion (RRF) - combines rankings from multiple systems
        Score = sum(1 / (k + rank)) for each system where item appears
        """
        
        # Map result UUID to combined score and result object
        fusion_scores = defaultdict(float)
        result_objects = {}
        result_appearances = defaultdict(int)
        
        for search_type, results in result_sets.items():
            for rank, result in enumerate(results):
                result_uuid = result.source_uuid
                
                # RRF score: 1 / (k + rank + 1)
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                fusion_scores[result_uuid] += rrf_score
                
                # Keep track of the result object and how many systems found it
                if result_uuid not in result_objects:
                    result_objects[result_uuid] = result
                result_appearances[result_uuid] += 1
        
        # Sort by fusion score and create final results
        sorted_results = sorted(
            fusion_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:limit]
        
        fused = SearchResults()
        for result_uuid, fusion_score in sorted_results:
            result = result_objects[result_uuid]
            
            # Update result score to fusion score
            # Also include boost for appearing in multiple systems
            appearance_boost = 1.0 + (0.1 * (result_appearances[result_uuid] - 1))
            final_score = fusion_score * appearance_boost
            
            # Create new result with updated score
            fused_result = SearchResult(
                content=result.content,
                score=final_score,
                result_type=result.result_type,
                source_uuid=result.source_uuid,
                source_type=f"rrf_fusion_{result.source_type}",
                document_uuid=result.document_uuid,
                document_path=result.document_path,
                document_name=result.document_name,
                episode_uuid=result.episode_uuid,
                episode_sequence=result.episode_sequence,
                entity_name=result.entity_name,
                entity_type=result.entity_type,
                relationship_fact=result.relationship_fact,
                relationship_type=result.relationship_type,
                source_entity=result.source_entity,
                target_entity=result.target_entity,
                metadata={
                    **result.metadata,
                    'rrf_score': fusion_score,
                    'appearances': result_appearances[result_uuid],
                    'original_score': result.score
                }
            )
            
            fused.add_result(fused_result)
        
        return fused
    
    def _weighted_combination(
        self,
        result_sets: Dict[str, List[SearchResult]],
        weights: Dict[str, float],
        limit: int
    ) -> SearchResults:
        """
        Weighted combination - multiply scores by weights and combine
        """
        
        # Normalize weights
        total_weight = sum(weights.values()) if weights else len(result_sets)
        normalized_weights = {}
        for search_type in result_sets.keys():
            weight = weights.get(search_type, 1.0)
            normalized_weights[search_type] = weight / total_weight
        
        # Map result UUID to weighted score
        fusion_scores = defaultdict(float)
        result_objects = {}
        
        for search_type, results in result_sets.items():
            weight = normalized_weights[search_type]
            
            for result in results:
                result_uuid = result.source_uuid
                weighted_score = result.score * weight
                
                fusion_scores[result_uuid] += weighted_score
                if result_uuid not in result_objects:
                    result_objects[result_uuid] = result
        
        # Sort and create results
        sorted_results = sorted(
            fusion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        fused = SearchResults()
        for result_uuid, fusion_score in sorted_results:
            result = result_objects[result_uuid]
            
            fused_result = SearchResult(
                content=result.content,
                score=fusion_score,
                result_type=result.result_type,
                source_uuid=result.source_uuid,
                source_type=f"weighted_fusion_{result.source_type}",
                document_uuid=result.document_uuid,
                document_path=result.document_path,
                document_name=result.document_name,
                episode_uuid=result.episode_uuid,
                episode_sequence=result.episode_sequence,
                entity_name=result.entity_name,
                entity_type=result.entity_type,
                relationship_fact=result.relationship_fact,
                relationship_type=result.relationship_type,
                source_entity=result.source_entity,
                target_entity=result.target_entity,
                metadata={
                    **result.metadata,
                    'weighted_score': fusion_score,
                    'original_score': result.score
                }
            )
            
            fused.add_result(fused_result)
        
        return fused
    
    def _relevance_score_fusion(
        self,
        result_sets: Dict[str, List[SearchResult]],
        limit: int
    ) -> SearchResults:
        """
        Simple relevance score fusion - just combine all results and sort by score
        """
        
        all_results = []
        seen_uuids = set()
        
        # Combine all results, avoiding duplicates
        for search_type, results in result_sets.items():
            for result in results:
                if result.source_uuid not in seen_uuids:
                    all_results.append(result)
                    seen_uuids.add(result.source_uuid)
        
        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        fused = SearchResults()
        for result in all_results[:limit]:
            # Update source type to indicate fusion
            fused_result = SearchResult(
                content=result.content,
                score=result.score,
                result_type=result.result_type,
                source_uuid=result.source_uuid,
                source_type=f"relevance_fusion_{result.source_type}",
                document_uuid=result.document_uuid,
                document_path=result.document_path,
                document_name=result.document_name,
                episode_uuid=result.episode_uuid,
                episode_sequence=result.episode_sequence,
                entity_name=result.entity_name,
                entity_type=result.entity_type,
                relationship_fact=result.relationship_fact,
                relationship_type=result.relationship_type,
                source_entity=result.source_entity,
                target_entity=result.target_entity,
                metadata=result.metadata
            )
            fused.add_result(fused_result)
        
        return fused
    
    def _graph_centrality_fusion(
        self,
        result_sets: Dict[str, List[SearchResult]],
        limit: int
    ) -> SearchResults:
        """
        Graph centrality fusion - boost results that are entities with more connections
        """
        
        # This is a simplified version - would need actual graph analysis
        # For now, boost entity results and use degree as proxy for centrality
        
        all_results = []
        seen_uuids = set()
        
        for search_type, results in result_sets.items():
            for result in results:
                if result.source_uuid not in seen_uuids:
                    # Apply centrality boost for entities
                    centrality_boost = 1.0
                    if result.result_type == 'entity':
                        # Simple heuristic: boost based on number of appearances
                        centrality_boost = 1.2
                    
                    boosted_score = result.score * centrality_boost
                    
                    boosted_result = SearchResult(
                        content=result.content,
                        score=boosted_score,
                        result_type=result.result_type,
                        source_uuid=result.source_uuid,
                        source_type=f"centrality_fusion_{result.source_type}",
                        document_uuid=result.document_uuid,
                        document_path=result.document_path,
                        document_name=result.document_name,
                        episode_uuid=result.episode_uuid,
                        episode_sequence=result.episode_sequence,
                        entity_name=result.entity_name,
                        entity_type=result.entity_type,
                        relationship_fact=result.relationship_fact,
                        relationship_type=result.relationship_type,
                        source_entity=result.source_entity,
                        target_entity=result.target_entity,
                        metadata={
                            **result.metadata,
                            'centrality_boost': centrality_boost,
                            'original_score': result.score
                        }
                    )
                    
                    all_results.append(boosted_result)
                    seen_uuids.add(result.source_uuid)
        
        # Sort by boosted score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        fused = SearchResults()
        for result in all_results[:limit]:
            fused.add_result(result)
        
        return fused
    
    def _temporal_recency_fusion(
        self,
        result_sets: Dict[str, List[SearchResult]],
        limit: int
    ) -> SearchResults:
        """
        Temporal recency fusion - boost more recent results
        """
        
        all_results = []
        seen_uuids = set()
        
        for search_type, results in result_sets.items():
            for result in results:
                if result.source_uuid not in seen_uuids:
                    # Apply temporal boost (simplified - would need actual timestamps)
                    temporal_boost = 1.0
                    
                    # Boost if result has temporal information indicating recency
                    if 'created_at' in result.metadata:
                        # Simple recency boost - would need more sophisticated calculation
                        temporal_boost = 1.1
                    
                    boosted_score = result.score * temporal_boost
                    
                    boosted_result = SearchResult(
                        content=result.content,
                        score=boosted_score,
                        result_type=result.result_type,
                        source_uuid=result.source_uuid,
                        source_type=f"temporal_fusion_{result.source_type}",
                        document_uuid=result.document_uuid,
                        document_path=result.document_path,
                        document_name=result.document_name,
                        episode_uuid=result.episode_uuid,
                        episode_sequence=result.episode_sequence,
                        entity_name=result.entity_name,
                        entity_type=result.entity_type,
                        relationship_fact=result.relationship_fact,
                        relationship_type=result.relationship_type,
                        source_entity=result.source_entity,
                        target_entity=result.target_entity,
                        metadata={
                            **result.metadata,
                            'temporal_boost': temporal_boost,
                            'original_score': result.score
                        }
                    )
                    
                    all_results.append(boosted_result)
                    seen_uuids.add(result.source_uuid)
        
        # Sort by boosted score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        fused = SearchResults()
        for result in all_results[:limit]:
            fused.add_result(result)
        
        return fused
    
    def deduplicate_results(
        self, 
        results: List[SearchResult],
        similarity_threshold: float = 0.9
    ) -> List[SearchResult]:
        """
        Remove duplicate results based on content similarity
        """
        
        deduplicated = []
        seen_content = set()
        
        for result in results:
            # Simple deduplication based on content hash
            content_key = hash(result.content.lower().strip())
            
            if content_key not in seen_content:
                deduplicated.append(result)
                seen_content.add(content_key)
        
        return deduplicated
    
    def apply_diversity_boost(
        self,
        results: List[SearchResult],
        diversity_factor: float = 0.1
    ) -> List[SearchResult]:
        """
        Apply diversity boost to promote variety in results
        """
        
        if not results:
            return results
        
        # Track document and entity type diversity
        doc_counts = Counter()
        type_counts = Counter()
        
        boosted_results = []
        
        for result in results:
            # Count appearances
            doc_counts[result.document_uuid] += 1
            if result.entity_type:
                type_counts[result.entity_type] += 1
            
            # Apply diversity penalty (lower scores for overrepresented categories)
            doc_penalty = 1.0 - (diversity_factor * (doc_counts[result.document_uuid] - 1))
            type_penalty = 1.0 - (diversity_factor * (type_counts.get(result.entity_type, 1) - 1))
            
            diversity_multiplier = doc_penalty * type_penalty
            boosted_score = result.score * max(0.1, diversity_multiplier)  # Minimum 0.1 multiplier
            
            # Create new result with diversity-adjusted score
            diversity_result = SearchResult(
                content=result.content,
                score=boosted_score,
                result_type=result.result_type,
                source_uuid=result.source_uuid,
                source_type=f"diversity_{result.source_type}",
                document_uuid=result.document_uuid,
                document_path=result.document_path,
                document_name=result.document_name,
                episode_uuid=result.episode_uuid,
                episode_sequence=result.episode_sequence,
                entity_name=result.entity_name,
                entity_type=result.entity_type,
                relationship_fact=result.relationship_fact,
                relationship_type=result.relationship_type,
                source_entity=result.source_entity,
                target_entity=result.target_entity,
                metadata={
                    **result.metadata,
                    'diversity_multiplier': diversity_multiplier,
                    'original_score': result.score
                }
            )
            
            boosted_results.append(diversity_result)
        
        return sorted(boosted_results, key=lambda x: x.score, reverse=True) 