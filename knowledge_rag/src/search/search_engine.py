"""
Search Engine for Knowledge RAG System

Main search engine that orchestrates hybrid search across the knowledge base,
combining semantic search, graph traversal, and full-text search with
intelligent result fusion and ranking.
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional

from .search_models import SearchQuery, SearchConfig, SearchResults, SearchType, SearchAnalytics
from .result_fusion import ResultFusion
from ..storage.storage_manager import StorageManager

logger = logging.getLogger(__name__)


class SearchEngine:
    """Main search engine for Knowledge RAG system"""
    
    def __init__(
        self,
        storage_manager: StorageManager,
        config: SearchConfig = None
    ):
        self.storage_manager = storage_manager
        self.config = config or SearchConfig()
        self.result_fusion = ResultFusion(rrf_k=self.config.rrf_k)
        self.analytics = SearchAnalytics()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("SearchEngine initialized")
    
    async def search(self, query: SearchQuery) -> SearchResults:
        """
        Main search method that executes the query and returns results
        
        Args:
            query: SearchQuery object with all search parameters
            
        Returns:
            SearchResults with fused and ranked results
        """
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing search query: '{query.query_text}'")
            
            # Determine search types to execute
            search_types = self._determine_search_types(query)
            
            # Execute searches
            raw_results = await self._execute_searches(query, search_types)
            
            # Fuse and rank results
            fused_results = self._fuse_and_rank_results(query, raw_results)
            
            # Apply final processing
            final_results = self._post_process_results(query, fused_results)
            
            # Set timing and metadata
            search_time_ms = (time.time() - start_time) * 1000
            final_results.search_time_ms = search_time_ms
            final_results.query = query
            
            # Record analytics
            self.analytics.record_query(query, final_results, success=True)
            
            self.logger.info(
                f"Search completed: {final_results.total_results} results in {search_time_ms:.1f}ms"
            )
            
            return final_results
            
        except Exception as e:
            search_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Search failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Record failed query
            self.analytics.record_query(query, SearchResults(), success=False, error=error_msg)
            
            # Return empty results with error info
            error_results = SearchResults()
            error_results.search_time_ms = search_time_ms
            error_results.query = query
            error_results.warnings.append(error_msg)
            
            return error_results
    
    def _determine_search_types(self, query: SearchQuery) -> List[str]:
        """Determine which search types to execute based on query"""
        
        search_types = []
        
        for search_type in query.search_types:
            if search_type == SearchType.HYBRID:
                # Hybrid search includes multiple types
                search_types.extend([
                    'semantic_entities',
                    'semantic_episodes', 
                    'graph_entities',
                    'graph_episodes'
                ])
                
                # Add graph traversal if we can extract entities
                if query.traversal_entities or self._has_potential_entities(query.query_text):
                    search_types.append('graph_traversal')
                    
            elif search_type == SearchType.SEMANTIC_ENTITIES:
                search_types.append('semantic_entities')
            elif search_type == SearchType.SEMANTIC_EPISODES:
                search_types.append('semantic_episodes')
            elif search_type == SearchType.GRAPH_ENTITIES:
                search_types.append('graph_entities')
            elif search_type == SearchType.GRAPH_EPISODES:
                search_types.append('graph_episodes')
            elif search_type == SearchType.GRAPH_TRAVERSAL:
                search_types.append('graph_traversal')
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(search_types))
    
    def _has_potential_entities(self, query_text: str) -> bool:
        """Check if query text might contain entity names"""
        # Simple heuristic - look for capitalized words
        words = query_text.split()
        for word in words:
            clean_word = word.strip('.,!?;:"()[]{}')
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                return True
        return False
    
    async def _execute_searches(
        self, 
        query: SearchQuery, 
        search_types: List[str]
    ) -> Dict[str, List]:
        """Execute all specified search types"""
        
        if self.config.parallel_search:
            # Execute searches in parallel
            return await self._execute_searches_parallel(query, search_types)
        else:
            # Execute searches sequentially  
            return await self._execute_searches_sequential(query, search_types)
    
    async def _execute_searches_parallel(
        self,
        query: SearchQuery,
        search_types: List[str]
    ) -> Dict[str, List]:
        """Execute searches in parallel for better performance"""
        
        # Create search tasks
        search_tasks = {}
        
        if 'semantic_entities' in search_types:
            search_tasks['semantic_entities'] = self.storage_manager.vector_store.search_entities_semantic(
                query.query_text,
                limit=min(query.limit * 2, self.config.max_results_per_type),
                entity_types=query.entity_types
            )
        
        if 'semantic_episodes' in search_types:
            search_tasks['semantic_episodes'] = self.storage_manager.vector_store.search_episodes_semantic(
                query.query_text,
                limit=min(query.limit * 2, self.config.max_results_per_type),
                document_filter=query.document_filter
            )
        
        if 'graph_entities' in search_types:
            search_tasks['graph_entities'] = self.storage_manager.graph_store.search_entities_by_name(
                query.query_text,
                limit=min(query.limit * 2, self.config.max_results_per_type)
            )
        
        if 'graph_episodes' in search_types:
            search_tasks['graph_episodes'] = self.storage_manager.graph_store.search_episodes_by_content(
                query.query_text,
                limit=min(query.limit * 2, self.config.max_results_per_type)
            )
        
        if 'graph_traversal' in search_types:
            # Extract entity names for traversal
            entity_names = query.traversal_entities or self._extract_entity_names(query.query_text)
            if entity_names:
                search_tasks['graph_traversal'] = self.storage_manager.graph_store.graph_traversal_search(
                    entity_names,
                    hops=min(query.graph_hops, self.config.max_graph_hops),
                    limit=min(query.limit * 2, self.config.max_traversal_results)
                )
        
        # Execute all tasks
        if search_tasks:
            task_names = list(search_tasks.keys())
            task_results = await asyncio.gather(*search_tasks.values(), return_exceptions=True)
            
            # Process results, handling exceptions
            results = {}
            for i, task_name in enumerate(task_names):
                task_result = task_results[i]
                if isinstance(task_result, Exception):
                    self.logger.warning(f"Search task {task_name} failed: {task_result}")
                    results[task_name] = []
                else:
                    results[task_name] = task_result
            
            return results
        else:
            return {}
    
    async def _execute_searches_sequential(
        self,
        query: SearchQuery,
        search_types: List[str]
    ) -> Dict[str, List]:
        """Execute searches sequentially"""
        
        results = {}
        
        try:
            if 'semantic_entities' in search_types:
                results['semantic_entities'] = await self.storage_manager.vector_store.search_entities_semantic(
                    query.query_text,
                    limit=min(query.limit * 2, self.config.max_results_per_type),
                    entity_types=query.entity_types
                )
            
            if 'semantic_episodes' in search_types:
                results['semantic_episodes'] = await self.storage_manager.vector_store.search_episodes_semantic(
                    query.query_text,
                    limit=min(query.limit * 2, self.config.max_results_per_type),
                    document_filter=query.document_filter
                )
            
            if 'graph_entities' in search_types:
                results['graph_entities'] = await self.storage_manager.graph_store.search_entities_by_name(
                    query.query_text,
                    limit=min(query.limit * 2, self.config.max_results_per_type)
                )
            
            if 'graph_episodes' in search_types:
                results['graph_episodes'] = await self.storage_manager.graph_store.search_episodes_by_content(
                    query.query_text,
                    limit=min(query.limit * 2, self.config.max_results_per_type)
                )
            
            if 'graph_traversal' in search_types:
                entity_names = query.traversal_entities or self._extract_entity_names(query.query_text)
                if entity_names:
                    results['graph_traversal'] = await self.storage_manager.graph_store.graph_traversal_search(
                        entity_names,
                        hops=min(query.graph_hops, self.config.max_graph_hops),
                        limit=min(query.limit * 2, self.config.max_traversal_results)
                    )
            
        except Exception as e:
            self.logger.error(f"Sequential search execution failed: {e}")
            
        return results
    
    def _extract_entity_names(self, query_text: str) -> List[str]:
        """Extract potential entity names from query text"""
        # Simple extraction - could be enhanced with NER
        words = query_text.split()
        entity_names = []
        
        for word in words:
            clean_word = word.strip('.,!?;:"()[]{}')
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                entity_names.append(clean_word)
        
        return entity_names[:5]  # Limit to first 5 potential entities
    
    def _fuse_and_rank_results(
        self,
        query: SearchQuery,
        raw_results: Dict[str, List]
    ) -> SearchResults:
        """Fuse and rank results from different search types"""
        
        if not raw_results:
            return SearchResults()
        
        # Apply score filtering
        filtered_results = {}
        for search_type, results in raw_results.items():
            filtered_results[search_type] = [
                r for r in results if r.score >= query.min_score
            ]
        
        # Apply search type weights if available
        weights = {
            'semantic_entities': self.config.semantic_weight,
            'semantic_episodes': self.config.semantic_weight,
            'graph_entities': self.config.graph_weight,
            'graph_episodes': self.config.graph_weight,
            'graph_traversal': self.config.graph_weight
        }
        
        # Fuse results using specified strategy
        fused_results = self.result_fusion.fuse_results(
            filtered_results,
            strategy=query.ranking_strategy,
            weights=weights,
            limit=query.limit * 2  # Get more results for post-processing
        )
        
        return fused_results
    
    def _post_process_results(
        self,
        query: SearchQuery,
        results: SearchResults
    ) -> SearchResults:
        """Apply final post-processing to results"""
        
        # Apply deduplication if requested
        if query.deduplicate_results:
            results.results = self.result_fusion.deduplicate_results(results.results)
            results.total_results = len(results.results)
        
        # Apply confidence filtering
        if self.config.min_confidence_score > 0:
            results.results = [
                r for r in results.results 
                if r.score >= self.config.min_confidence_score
            ]
            results.total_results = len(results.results)
        
        # Apply diversity boost if enabled  
        if hasattr(self.config, 'apply_diversity_boost') and self.config.apply_diversity_boost:
            results.results = self.result_fusion.apply_diversity_boost(
                results.results,
                diversity_factor=0.1
            )
        
        # Trim to final limit
        results.results = results.results[:query.limit]
        results.total_results = len(results.results)
        
        # Apply confidence and recency boosts
        if self.config.boost_high_confidence or self.config.boost_recent_results:
            results.results = self._apply_quality_boosts(results.results)
        
        return results
    
    def _apply_quality_boosts(self, results: List) -> List:
        """Apply quality-based score boosts"""
        
        boosted_results = []
        
        for result in results:
            boost_factor = 1.0
            
            # Boost high confidence results
            if self.config.boost_high_confidence and result.score > 0.8:
                boost_factor *= 1.1
            
            # Boost recent results (simplified)
            if self.config.boost_recent_results:
                # This would need actual temporal analysis
                boost_factor *= 1.0
            
            # Apply boost
            boosted_score = result.score * boost_factor
            
            # Update result score
            result.score = boosted_score
            result.metadata['quality_boost'] = boost_factor
            
            boosted_results.append(result)
        
        return sorted(boosted_results, key=lambda x: x.score, reverse=True)
    
    # Convenience Methods
    async def search_text(
        self, 
        query_text: str, 
        limit: int = 10,
        search_types: List[SearchType] = None
    ) -> SearchResults:
        """Simple text search interface"""
        
        query = SearchQuery(
            query_text=query_text,
            search_types=search_types or [SearchType.HYBRID],
            limit=limit
        )
        
        return await self.search(query)
    
    async def search_entities(
        self,
        query_text: str,
        entity_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> SearchResults:
        """Search specifically for entities"""
        
        query = SearchQuery(
            query_text=query_text,
            search_types=[SearchType.SEMANTIC_ENTITIES, SearchType.GRAPH_ENTITIES],
            entity_types=entity_types,
            limit=limit
        )
        
        return await self.search(query)
    
    async def search_episodes(
        self,
        query_text: str,
        document_filter: Optional[str] = None,
        limit: int = 10
    ) -> SearchResults:
        """Search specifically for episodes/chunks"""
        
        query = SearchQuery(
            query_text=query_text,
            search_types=[SearchType.SEMANTIC_EPISODES, SearchType.GRAPH_EPISODES],
            document_filter=document_filter,
            limit=limit
        )
        
        return await self.search(query)
    
    async def graph_traversal_search(
        self,
        entity_names: List[str],
        hops: int = 2,
        limit: int = 10
    ) -> SearchResults:
        """Perform graph traversal search from specific entities"""
        
        query = SearchQuery(
            query_text=" ".join(entity_names),  # Placeholder
            search_types=[SearchType.GRAPH_TRAVERSAL],
            traversal_entities=entity_names,
            graph_hops=hops,
            limit=limit
        )
        
        return await self.search(query)
    
    # Analytics and Management
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics and performance metrics"""
        return self.analytics.get_performance_summary()
    
    def reset_analytics(self):
        """Reset search analytics"""
        self.analytics = SearchAnalytics()
    
    async def warm_up(self):
        """Warm up the search engine with test queries"""
        try:
            # Execute a few test queries to warm up the system
            test_queries = [
                "knowledge graph",
                "document processing", 
                "search system"
            ]
            
            for test_query in test_queries:
                await self.search_text(test_query, limit=1)
            
            self.logger.info("Search engine warmed up successfully")
            
        except Exception as e:
            self.logger.warning(f"Search engine warm-up failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on search capabilities"""
        
        health_status = {
            'status': 'healthy',
            'components': {},
            'performance': {},
            'warnings': []
        }
        
        try:
            # Test storage manager
            stats = await self.storage_manager.get_storage_statistics()
            health_status['components']['storage'] = 'healthy' if stats else 'degraded'
            
            # Test search performance with simple query
            start_time = time.time()
            test_results = await self.search_text("test", limit=1)
            search_time = (time.time() - start_time) * 1000
            
            health_status['performance']['test_search_ms'] = search_time
            
            if search_time > 5000:  # More than 5 seconds
                health_status['warnings'].append('Search performance degraded')
            
            # Check analytics
            analytics = self.get_search_analytics()
            if analytics['success_rate'] < 0.9:
                health_status['warnings'].append('Low search success rate')
            
            if health_status['warnings']:
                health_status['status'] = 'degraded'
                
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status 